/* @author: YueLin (neural variant)
 *
 * tracker_neural: identical to planning.cpp except the "target visible" control
 * is produced by a neural policy (ONNX) instead of the MPC. The lost/reacquire
 * path (mpc->visibility()==false), mapping, and all publishers are unchanged.
 */

#include <algorithm>
#include <chrono>
#include <functional>

#include <pcl_conversions/pcl_conversions.h>

#include "geometry_msgs/msg/quaternion.hpp"

#include "local_visualization.hpp"
#include "planning_neural.hpp"

namespace
{
    using SteadyClock = std::chrono::steady_clock;

    inline double elapsed_ms(SteadyClock::time_point start,
                             SteadyClock::time_point end)
    {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }
}

inline double quaternion2yaw(geometry_msgs::msg::Quaternion q)
{
    double x = q.x * q.x, y = q.y * q.y, z = q.z * q.z, w = q.w * q.w;
    double s = 2 * (q.w * q.y - q.x * q.z) / (x + y + z + w);
    if(std::fabs(s) >= 1 - 1e-6)
        return std::copysign(2 * std::atan2(q.y, q.x), s);
    return std::atan2(2 * (q.x * q.y + q.z * q.w), w + x - y - z);
}

Tracker::Tracker(std::string name): Node(name)
{
    /* Declare hyperparameters */
    range = declare_parameter<double>("tracker.range");
    double dt = declare_parameter<double>("planner.delta_t");
    control_dt = dt;
    tracker.z() = declare_parameter<double>("tracker.radius");
    declare_parameter<double>("tracker.fov");
    message.header.frame_id = declare_parameter<std::string>("map.name");

    declare_parameter<int>("bezier.m");
    declare_parameter<double>("bezier.tau");

    declare_parameter<double>("map.size");
    declare_parameter<double>("map.radius");
    declare_parameter<double>("map.resolution");

    tracker_max_vel = declare_parameter<double>("tracker.max_vel");
    tracker_max_acc = declare_parameter<double>("tracker.max_acc");
    declare_parameter<double>("tracker.max_alpha");
    declare_parameter<double>("tracker.max_omega");

    declare_parameter<int>("planner.kappa");
    declare_parameter<int>("planner.samples");
    declare_parameter<int>("planner.num_angles");
    declare_parameter<double>("planner.epsilon");
    observation_distance = declare_parameter<double>(
        "planner.observation_distance");

    declare_parameter<double>("planner.lambda_a");
    declare_parameter<double>("planner.lambda_d");
    declare_parameter<double>("planner.lambda_o");
    declare_parameter<double>("planner.lambda_c");
    declare_parameter<double>("planner.lambda_f");
    declare_parameter<double>("planner.lambda_s");

    declare_parameter<int>("lbfgs.past");
    declare_parameter<int>("lbfgs.memory");
    declare_parameter<double>("lbfgs.delta");
    declare_parameter<double>("lbfgs.wolfe");
    declare_parameter<double>("lbfgs.armijo");
    declare_parameter<double>("lbfgs.epsilon");
    declare_parameter<int>("lbfgs.iterations");
    declare_parameter<double>("lbfgs.max_step");

    /* Neural policy parameters */
    std::string onnx = declare_parameter<std::string>("neural.onnx_path", "");
    int onnx_threads = declare_parameter<int>("neural.onnx_threads", 4);
    double omega_sign = declare_parameter<double>("neural.omega_sign", 1.0);
    // Safety fallback: if the near-horizon of the neural trajectory enters an
    // obstacle, hand control to the classical MPC (following the A*-planned
    // trajectory) for a short lock window, then retry the neural policy.
    neural_safety_steps = declare_parameter<int>("neural.safety_steps", 10);
    neural_fallback_lock = declare_parameter<double>("neural.fallback_lock", 0.6);
    stationary_guard = declare_parameter<bool>(
        "neural.stationary_guard", true);
    stationary_guard_distance_tolerance = declare_parameter<double>(
        "neural.stationary_guard_distance_tolerance", 0.25);
    stationary_guard_bearing_tolerance = declare_parameter<double>(
        "neural.stationary_guard_bearing_tolerance", 0.15);
    stationary_guard_target_speed = declare_parameter<double>(
        "neural.stationary_guard_target_speed", 0.05);
    stationary_guard_reverse_speed = declare_parameter<double>(
        "neural.stationary_guard_reverse_speed", -0.10);
    stationary_guard_max_omega = declare_parameter<double>(
        "neural.stationary_guard_max_omega", 0.30);
    fov_guard = declare_parameter<bool>("neural.fov_guard", true);
    fov_guard_min_bearing = declare_parameter<double>(
        "neural.fov_guard_min_bearing", 0.08);
    fov_guard_kp = declare_parameter<double>(
        "neural.fov_guard_kp", 1.0);
    fov_guard_max_omega = declare_parameter<double>(
        "neural.fov_guard_max_omega", 0.8);
    range_guard = declare_parameter<bool>("neural.range_guard", true);
    range_guard_start = declare_parameter<double>(
        "neural.range_guard_start", 3.0);
    range_guard_release = declare_parameter<double>(
        "neural.range_guard_release", 2.75);
    range_guard_max_bearing = declare_parameter<double>(
        "neural.range_guard_max_bearing", 0.45);
    range_guard_min_speed = declare_parameter<double>(
        "neural.range_guard_min_speed", 2.0);
    range_guard_kp = declare_parameter<double>(
        "neural.range_guard_kp", 0.8);
    range_guard_horizon = declare_parameter<double>(
        "neural.range_guard_horizon", 0.6);
    range_guard_clearance = declare_parameter<double>(
        "neural.range_guard_clearance", 0.10);
    timing_log_interval = declare_parameter<double>("neural.timing_log_interval", 2.0);
    const std::string target_topic = declare_parameter<std::string>(
        "target.topic", "/target/detection");
    const bool direct_target = declare_parameter<bool>("target.direct", false);
    use_fov_detection = declare_parameter<bool>("target.use_fov_detection", true);
    truth_fallback_after = declare_parameter<double>(
        "target.truth_fallback_after", 3.0);
    if(truth_fallback_after <= 0.0)
        throw std::runtime_error("target.truth_fallback_after must be positive");
    neural_lock_until = rclcpp::Clock().now();
    last_detection_time = neural_lock_until;
    detection_lost_since = rclcpp::Time(0, 0, RCL_ROS_TIME);
    timing_last_log = neural_lock_until;

    /* Initialize states */
    tigger = 0;
    pointer = 0;
    target.setZero();
    trajectory.resize(5, 0);
    tracker.head(2).setZero();
    state = Eigen::VectorXd::Zero(5);

    /* Initialize map */
    map = std::make_unique<diff_track::Map>(
        get_parameter("map.size").as_double(),
        get_parameter("map.size").as_double(),
        get_parameter("map.resolution").as_double(),
        get_parameter("map.radius").as_double()
    );

    /* Initialize Bezier curve */
    bezier = std::make_unique<diff_track::Bezier>(
        get_parameter("bezier.m").as_int() * 2 + 1,
        get_parameter("bezier.tau").as_double()
    );

    /* Initialize hybrid A* planner */
    planner = std::make_unique<diff_track::HybridAStar>(
        map.get(), bezier.get(),
        get_parameter("planner.num_angles").as_int(),
        get_parameter("planner.samples").as_int(),
        range, tracker.z(), dt,
        get_parameter("tracker.max_vel").as_double(),
        get_parameter("tracker.max_acc").as_double(),
        observation_distance,
        get_parameter("planner.epsilon").as_double(),
        bezier->duration
    );

    /* Initialize model predictive controller */
    mpc = std::make_unique<diff_track::ModelPredictiveControl>(
        map.get(), bezier.get(),
        2 * tracker.z(), dt,
        observation_distance,
        get_parameter("planner.lambda_a").as_double(),
        get_parameter("planner.lambda_d").as_double(),
        get_parameter("planner.lambda_o").as_double(),
        get_parameter("planner.lambda_c").as_double(),
        get_parameter("planner.lambda_f").as_double(),
        get_parameter("planner.lambda_s").as_double(),
        get_parameter("tracker.max_vel").as_double(),
        get_parameter("tracker.max_acc").as_double(),
        get_parameter("tracker.max_omega").as_double(),
        get_parameter("tracker.max_alpha").as_double(),
        get_parameter("lbfgs.past").as_int(),
        get_parameter("lbfgs.memory").as_int(),
        get_parameter("planner.kappa").as_int(),
        get_parameter("lbfgs.epsilon").as_double(),
        get_parameter("lbfgs.max_step").as_double(),
        get_parameter("lbfgs.delta").as_double(),
        get_parameter("lbfgs.epsilon").as_double(),
        get_parameter("lbfgs.iterations").as_int(),
        get_parameter("lbfgs.wolfe").as_double(),
        get_parameter("lbfgs.armijo").as_double()
    );

    /* Initialize neural policy */
    if(onnx.empty())
        throw std::runtime_error(
            "neural.onnx_path parameter is required for tracker_neural");
    neural = std::make_unique<diff_track::NeuralPolicy>(
        onnx, dt,
        get_parameter("tracker.max_vel").as_double(),
        get_parameter("tracker.max_omega").as_double(),
        omega_sign,
        onnx_threads
    );
    RCLCPP_INFO(
        get_logger(), "Neural policy loaded: %s (ONNX intra-op threads=%d)",
        onnx.c_str(), onnx_threads);

    /* Create publishers */
    publisher = create_publisher<nav_msgs::msg::Path>("/tracker/plan", 1);
    cmd = create_publisher<geometry_msgs::msg::Twist>("/tracker/cmd_vel", 1);
    planning_time = create_publisher<std_msgs::msg::Float64>(
        "/tracker/planning_time_ms", 10);
    control_branch = create_publisher<std_msgs::msg::UInt8>(
        "/tracker/control_branch", 10);
    metrics_sample = create_publisher<std_msgs::msg::Float64MultiArray>(
        "/tracker/metrics_sample", 100);
    observation_status = create_publisher<std_msgs::msg::String>(
        "/tracker/target_observation_status", 10);
    mapper = create_publisher<nav_msgs::msg::OccupancyGrid>("/tracker/map", 1);
    scan_visual = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/tracker/scan_local", 1);

    /* Create localization subscriber */
    localization = create_subscription<nav_msgs::msg::Odometry>(
        "/tracker/odom", 1,
        [this](const nav_msgs::msg::Odometry::SharedPtr odom){
            state.x() = odom->pose.pose.position.x;
            state.y() = odom->pose.pose.position.y;
            state.z() = quaternion2yaw(odom->pose.pose.orientation);
            if(!trajectory.cols()) trajectory = state;
            tracker.head(2) = state.head(2);
            tigger |= 1;
        }
    );

    /* The rolling map is built only from the actual lidar cloud. */
    sensor = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/tracker/scan", 1,
        [this](const sensor_msgs::msg::PointCloud2::SharedPtr pc){
            if(!(tigger & 1)) return;
            pcl::fromROSMsg(*pc, cloud);
            map->update(tracker, target, cloud);
            mapper->publish(*map->message(
                message.header.frame_id, tracker.z()
            ));
            sensor_msgs::msg::PointCloud2 scan_message;
            diff_track::makeLocalVisualization(
                cloud, tracker.head<2>(), map->windowRadius(), pc->header,
                scan_message);
            scan_visual->publish(scan_message);
            tigger |= 2;
        }
    );

    /* Create target detection subscriber */
    detect = create_subscription<geometry_msgs::msg::PoseStamped>(
        target_topic, 1,
        [this, direct_target](const geometry_msgs::msg::PoseStamped::SharedPtr detection){
            tigger |= 4;
            const bool visible = direct_target || !use_fov_detection
                || detection->pose.orientation.w > 0;
            detection_visible = visible;
            if(visible)
            {
                const rclcpp::Time sample_time = rclcpp::Clock().now();
                const Eigen::Vector2d sample_position(
                    detection->pose.position.x,
                    detection->pose.position.y);
                if(have_detection_sample)
                {
                    const double sample_dt =
                        (sample_time - last_detection_time).seconds();
                    if(sample_dt > 1e-3)
                    {
                        detected_target_speed =
                            (sample_position - last_detection_position).norm()
                            / sample_dt;
                    }
                }
                else
                {
                    have_detection_sample = true;
                    detected_target_speed = 0.0;
                }
                last_detection_position = sample_position;
                last_detection_time = sample_time;
                target.x() = detection->pose.position.x;
                target.y() = detection->pose.position.y;
                target.z() = detection->pose.position.z * 1.5;
                detection_lost_since = rclcpp::Time(0, 0, RCL_ROS_TIME);
                if(truth_fallback_active && !direct_target)
                {
                    truth_fallback_active = false;
                    mpc->visibility(true);
                    std_msgs::msg::String status;
                    status.data = "fov_detection_recovered";
                    observation_status->publish(status);
                    RCLCPP_INFO(
                        get_logger(),
                        "FOV detection recovered; leaving temporary truth fallback");
                }
            }
            else
            {
                if(detection_lost_since.nanoseconds() == 0)
                    detection_lost_since = rclcpp::Clock().now();
                target.z() = -1;
                stationary_hold = false;
                stationary_mpc_active = false;
                range_guard_active = false;
                have_detection_sample = false;
                detected_target_speed = 0.0;
                if(mpc->visibility() && !truth_fallback_active)
                {
                    pointer = 0;
                    twist.linear.x = 0;
                    twist.angular.z = 0;
                    cmd->publish(twist);
                    mpc->visibility(false);
                    state.tail(2).setZero();
                    trajectory = planner->reacquire(false, state);
                    RCLCPP_INFO(
                        get_logger(),
                        "Target detection lost; switched immediately to MPC reacquisition");
                }
            }
        }
    );
    if(use_fov_detection && !direct_target && target_topic != "/target/ground_truth")
    {
        truth_detect = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/target/ground_truth", 1,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr truth){
                if(!truth_fallback_active)
                    return;
                tigger |= 4;
                target.x() = truth->pose.position.x;
                target.y() = truth->pose.position.y;
                target.z() = truth->pose.position.z * 1.5;
                /* Keep the raw FOV visibility flag false.  Truth fallback is
                 * only a control recovery path; metrics must still record the
                 * detector as lost and mark the trial terminally failed. */
                mpc->visibility(true);
            }
        );
    }
    RCLCPP_INFO(
        get_logger(), "Target input: %s (%s)", target_topic.c_str(),
        direct_target? "direct position":
        (use_fov_detection? "FOV detection with truth fallback": "direct position"));

    /* Create subscriber for trajectory planning */
    planning = create_subscription<nav_msgs::msg::Path>(
        "/target/bezier", 1,
        std::bind(&Tracker::callback, this, std::placeholders::_1)
    );

    /* Create timer for replanning */
    int ms = 1e3 * bezier->time;
    replan = create_wall_timer(std::chrono::milliseconds(ms), [this](){
        if((tigger & 7) == 7 &&
           pointer + 1 >= trajectory.cols() &&
           bezier->duration < (rclcpp::Clock().now() - time).seconds())
        {
            pointer = 0;
            twist.linear.x = 0;
            twist.angular.z = 0;
            cmd->publish(twist);
            mpc->visibility(false);
            state.tail(2).setZero();
            trajectory = planner->reacquire(false, state);
        }
    });

    /* Create controller */
    ms = 1e3 * dt;
    controller = create_wall_timer(
        std::chrono::milliseconds(ms),
        std::bind(&Tracker::ctrl, this)
    );
}

void Tracker::log_timing_if_needed(const rclcpp::Time& now)
{
    if(timing_log_interval <= 0.0 ||
       (now - timing_last_log).seconds() < timing_log_interval)
        return;

    auto avg = [](double sum, uint64_t n){
        return n? sum / static_cast<double>(n): 0.0;
    };

    RCLCPP_INFO(
        get_logger(),
        "耗时统计：最近 %.1f 秒平均值，单位 ms。\n"
        "  控制循环总耗时=%.3f（一次 ctrl() 从进入到发布完成，样本数=%llu）\n"
        "  神经分支：推理总耗时=%.3f，ONNX运行=%.3f，ESDF局部图裁剪=%.3f，近程安全检查=%.3f，发布控制/轨迹=%.3f（样本数=%llu）\n"
        "  兜底分支：传统MPC求解=%.3f，发布控制/轨迹=%.3f（样本数=%llu）",
        timing_log_interval,
        avg(timing_ctrl_ms, timing_ctrl_samples),
        static_cast<unsigned long long>(timing_ctrl_samples),
        avg(timing_infer_ms, timing_neural_samples),
        avg(timing_infer_ort_ms, timing_neural_samples),
        avg(timing_infer_esdf_ms, timing_neural_samples),
        avg(timing_safety_ms, timing_neural_samples),
        avg(timing_neural_publish_ms, timing_neural_samples),
        static_cast<unsigned long long>(timing_neural_samples),
        avg(timing_mpc_ms, timing_fallback_samples),
        avg(timing_fallback_publish_ms, timing_fallback_samples),
        static_cast<unsigned long long>(timing_fallback_samples)
    );

    timing_last_log = now;
    timing_ctrl_samples = 0;
    timing_neural_samples = 0;
    timing_fallback_samples = 0;
    timing_ctrl_ms = 0.0;
    timing_target_ms = 0.0;
    timing_infer_ms = 0.0;
    timing_infer_esdf_ms = 0.0;
    timing_infer_target_ms = 0.0;
    timing_infer_visibility_ms = 0.0;
    timing_infer_tensor_ms = 0.0;
    timing_infer_ort_ms = 0.0;
    timing_infer_output_ms = 0.0;
    timing_safety_ms = 0.0;
    timing_neural_publish_ms = 0.0;
    timing_mpc_ms = 0.0;
    timing_fallback_publish_ms = 0.0;
}

void Tracker::ctrl()
{
    if((tigger & 7) != 7)
        return;

    const auto ctrl_start = SteadyClock::now();

    /* Set time stamp */
    rclcpp::Time now = rclcpp::Clock().now();
    if(use_fov_detection && !truth_fallback_active
       && detection_lost_since.nanoseconds() != 0
       && (now - detection_lost_since).seconds() >= truth_fallback_after)
    {
        truth_fallback_active = true;
        mpc->visibility(true);
        neural_lock_until = now;
        std_msgs::msg::String status;
        status.data = "truth_fallback_started";
        observation_status->publish(status);
        RCLCPP_ERROR(
            get_logger(),
            "FOV detection lost for %.1fs; using /target/ground_truth until recovery; "
            "benchmark trial is irrecoverably failed",
            truth_fallback_after);
    }
    double clock = (now - time).seconds();
    mpc->clock(clock);
    message.header.stamp = now;

    /* Branch codes published for benchmark accounting:
     * 0 = neural policy, 1 = classical MPC/reacquisition, 2 = stationary hold. */
    auto finish_ctrl = [&](uint8_t branch){
        const double cycle_ms = elapsed_ms(ctrl_start, SteadyClock::now());
        timing_ctrl_ms += cycle_ms;
        timing_ctrl_samples++;
        std_msgs::msg::Float64 timing_message;
        timing_message.data = cycle_ms;
        planning_time->publish(timing_message);
        std_msgs::msg::UInt8 branch_message;
        branch_message.data = branch;
        control_branch->publish(branch_message);
        /* One atomic benchmark record per completed control cycle. Keeping
         * command, runtime, branch and raw detector visibility in one DDS
         * sample prevents independent topic discovery/order from producing
         * different sample counts or a one-cycle visibility phase error. */
        std_msgs::msg::Float64MultiArray metrics_message;
        metrics_message.data = {
            static_cast<double>(++metrics_sequence),
            twist.linear.x,
            twist.angular.z,
            cycle_ms,
            static_cast<double>(branch),
            detection_visible? 1.0: 0.0
        };
        metrics_sample->publish(metrics_message);
        log_timing_if_needed(now);
    };

    /* A stopped target outside the acceptable observation pose must first be
     * brought to the center by the reliable classical controller.  Letting
     * the raw neural command handle this state can make it reverse toward the
     * 4 m sensor boundary, repeatedly alternating visible/lost modes. */
    if(mpc->visibility() && stationary_guard && have_detection_sample)
    {
        const double detected_dx = target.x() - state.x();
        const double detected_dy = target.y() - state.y();
        const double detected_range = std::hypot(detected_dx, detected_dy);
        const double detected_bearing = std::atan2(
            std::sin(std::atan2(detected_dy, detected_dx) - state.z()),
            std::cos(std::atan2(detected_dy, detected_dx) - state.z()));
        const bool target_stationary =
            detected_target_speed <= stationary_guard_target_speed;
        const bool inside_hold_pose =
            std::fabs(detected_range - observation_distance)
                <= stationary_guard_distance_tolerance
            && std::fabs(detected_bearing)
                <= stationary_guard_bearing_tolerance;
        const bool needs_stationary_mpc =
            target_stationary && !inside_hold_pose;

        if(needs_stationary_mpc)
        {
            neural_lock_until = now + rclcpp::Duration::from_seconds(
                neural_fallback_lock);
            if(!stationary_mpc_active)
                RCLCPP_INFO(
                    get_logger(),
                    "Stationary target outside hold pose; using MPC to "
                    "converge (range=%.3f m, bearing=%.3f rad)",
                    detected_range, detected_bearing);
            stationary_mpc_active = true;
        }
        else if(stationary_mpc_active)
        {
            stationary_mpc_active = false;
            RCLCPP_INFO(
                get_logger(),
                "Stationary-target MPC convergence complete/released "
                "(speed=%.3f m/s, range=%.3f m, bearing=%.3f rad)",
                detected_target_speed, detected_range, detected_bearing);
        }
    }
    else
        stationary_mpc_active = false;

    if(mpc->visibility() && now >= neural_lock_until)
    {
        /* === Target visible: neural policy control === */
        // Current target from the same Bezier prediction the MPC would use:
        // target(t) = bezier->get(t), velocity = bezier->derivative(t).
        auto stage_start = SteadyClock::now();
        Eigen::Vector2d target_pos = truth_fallback_active
            ? target.head<2>() : bezier->get(clock);
        Eigen::Vector2d target_vel = truth_fallback_active
            ? Eigen::Vector2d::Zero() : bezier->derivative(clock);
        timing_target_ms += elapsed_ms(stage_start, SteadyClock::now());

        /* The hold geometry must use the latest camera detection, not the
         * fitted prediction.  During predictor warm-up (and briefly after a
         * stop) the Bezier origin can lag the physical target enough to miss
         * an otherwise valid stationary equilibrium. */
        const double dx = target.x() - state.x();
        const double dy = target.y() - state.y();
        const double target_range = std::hypot(dx, dy);
        const double target_bearing = std::atan2(
            std::sin(std::atan2(dy, dx) - state.z()),
            std::cos(std::atan2(dy, dx) - state.z()));
        /* Use consecutive camera detections for the stationary decision.  A
         * fitted Bezier curve can retain a small derivative for several
         * cycles after the physical target has stopped, which previously
         * made the controller leave and re-enter the deadband repeatedly. */
        const bool target_stationary = have_detection_sample
            && detected_target_speed <= stationary_guard_target_speed;
        const bool stationary_enter = stationary_guard
            && target_stationary
            && std::fabs(target_range - observation_distance)
                <= stationary_guard_distance_tolerance
            && std::fabs(target_bearing)
                <= stationary_guard_bearing_tolerance;
        const bool stationary_exit = !stationary_guard
            || !have_detection_sample
            || detected_target_speed > stationary_guard_target_speed * 2.0
            || std::fabs(target_range - observation_distance)
                > stationary_guard_distance_tolerance * 1.6
            || std::fabs(target_bearing)
                > stationary_guard_bearing_tolerance * 1.4;

        if(stationary_hold && stationary_exit)
        {
            stationary_hold = false;
            RCLCPP_INFO(
                get_logger(),
                "Stationary-target hold released "
                "(speed=%.3f m/s, range=%.3f m, bearing=%.3f rad)",
                detected_target_speed, target_range, target_bearing);
        }
        else if(!stationary_hold && stationary_enter)
        {
            stationary_hold = true;
            RCLCPP_INFO(
                get_logger(),
                "Stationary-target hold engaged "
                "(range=%.3f m, bearing=%.3f rad)",
                target_range, target_bearing);
        }

        if(stationary_hold)
        {
            /* Hysteretic deadband at the stationary tracking equilibrium.
             * Publishing an explicit zero prevents neural/MPC hand-offs from
             * producing repeated micro-adjustments after the target stops. */
            state[3] = twist.linear.x = 0;
            state[4] = twist.angular.z = 0;
            range_guard_active = false;
            cmd->publish(twist);
            message.poses.resize(1);
            message.poses[0].pose.position.x = state.x();
            message.poses[0].pose.position.y = state.y();
            message.poses[0].pose.orientation.z = std::sin(state.z() / 2);
            message.poses[0].pose.orientation.w = std::cos(state.z() / 2);
            publisher->publish(message);
            finish_ctrl(2);
            return;
        }

        Eigen::Vector2d vw;
        stage_start = SteadyClock::now();
        std::vector<Eigen::Vector3d> traj =
            neural->infer(state, target_pos, target_vel, map.get(), vw);
        timing_infer_ms += elapsed_ms(stage_start, SteadyClock::now());
        const auto& infer_timing = neural->last_timing();
        timing_infer_esdf_ms += infer_timing.esdf_ms;
        timing_infer_target_ms += infer_timing.target_ms;
        timing_infer_visibility_ms += infer_timing.visibility_ms;
        timing_infer_tensor_ms += infer_timing.tensor_ms;
        timing_infer_ort_ms += infer_timing.ort_ms;
        timing_infer_output_ms += infer_timing.output_ms;
        timing_neural_samples++;

        /* The Python deployment has this guard, but the original C++ port
         * published raw ONNX omega.  Correct only commands that turn away
         * from an already off-center target; all other neural commands pass
         * through unchanged. */
        if(fov_guard
           && std::fabs(target_bearing) > fov_guard_min_bearing
           && target_bearing * vw[1] < 0.0)
        {
            const double raw_omega = vw[1];
            vw[1] = std::clamp(
                fov_guard_kp * target_bearing,
                -fov_guard_max_omega,
                fov_guard_max_omega);
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "FOV guard corrected omega %.3f -> %.3f for bearing %.3f",
                raw_omega, vw[1], target_bearing);
        }

        /* A dense local map can make the learned policy yield to obstacles
         * long enough for the target to cross the 4 m detection boundary.
         * Restore closing speed while there is still one metre of range
         * margin, but only when the target is sufficiently centred.  Check
         * the command that will actually be published (not the ONNX
         * trajectory) against the rolling ESDF before accepting the boost.
         * Unknown cells remain traversable by the map's existing semantics. */
        if(!range_guard)
            range_guard_active = false;
        else if(target_range > range_guard_start)
            range_guard_active = true;
        else if(target_range < range_guard_release)
            range_guard_active = false;

        if(range_guard_active
           && std::fabs(target_bearing) <= range_guard_max_bearing)
        {
            const double desired_speed = std::clamp(
                std::max(range_guard_min_speed,
                         detected_target_speed
                         + range_guard_kp
                         * (target_range - observation_distance)),
                0.0, tracker_max_vel);
            const double acceleration_limited_speed = std::min(
                desired_speed,
                std::max(0.0, state[3]) + tracker_max_acc * control_dt);
            const double boosted_speed = std::max(
                vw[0], acceleration_limited_speed);

            if(boosted_speed > vw[0] + 1e-3)
            {
                Eigen::Vector3d guarded_pose = state.head<3>();
                const int guard_steps = std::max(
                    1, static_cast<int>(std::ceil(
                        range_guard_horizon / control_dt)));
                const double required_clearance =
                    tracker.z() + range_guard_clearance;
                bool boost_safe = true;
                for(int step = 0; step < guard_steps; ++step)
                {
                    const double mid_yaw =
                        guarded_pose.z() + 0.5 * vw[1] * control_dt;
                    guarded_pose.x() += boosted_speed * control_dt
                                      * std::cos(mid_yaw);
                    guarded_pose.y() += boosted_speed * control_dt
                                      * std::sin(mid_yaw);
                    guarded_pose.z() = normalize(
                        guarded_pose.z() + vw[1] * control_dt);
                    if(map->at(guarded_pose.head<2>())
                       < required_clearance)
                    {
                        boost_safe = false;
                        break;
                    }
                }

                if(boost_safe)
                {
                    const double raw_speed = vw[0];
                    vw[0] = boosted_speed;
                    RCLCPP_INFO_THROTTLE(
                        get_logger(), *get_clock(), 1000,
                        "Range guard boosted speed %.3f -> %.3f "
                        "(range=%.3f, bearing=%.3f)",
                        raw_speed, vw[0], target_range, target_bearing);
                }
                else
                    RCLCPP_WARN_THROTTLE(
                        get_logger(), *get_clock(), 1000,
                        "Range guard withheld speed boost: local ESDF arc "
                        "is below %.2f m clearance",
                        required_clearance);
            }
        }

        // Safety check: does the NEAR horizon (the part about to be executed)
        // drive into an obstacle? The neural policy imitates an expert that
        // grazes walls, so far-horizon low clearance is normal and ignored;
        // only a near-term hard collision (esdf < robot radius) triggers the
        // classical fallback. Checking the near horizon (not the whole 3s)
        // avoids demoting the neural policy on harmless wall-hugging tails.
        int check = std::min(neural_safety_steps + 1,
                             static_cast<int>(traj.size()));
        bool neural_unsafe = false;
        stage_start = SteadyClock::now();
        for(int p = 1; p < check; p++)
            if(map->at(traj[p].head(2)) < tracker.z())
            {
                neural_unsafe = true; break;
            }
        timing_safety_ms += elapsed_ms(stage_start, SteadyClock::now());

        /* At the nominal stationary equilibrium the correct command is near
         * zero. Reject a neural command that immediately reverses away from a
         * centered, stationary target or turns hard enough to push it out of
         * the camera FOV. This leaves moving-target and obstacle-avoidance
         * behavior untouched. */
        const bool neural_divergent = stationary_enter
            && (vw[0] < stationary_guard_reverse_speed
                || std::fabs(vw[1]) > stationary_guard_max_omega);
        neural_unsafe |= neural_divergent;

        if(!neural_unsafe)
        {
            stage_start = SteadyClock::now();
            /* Neural trajectory is safe near-term: execute it. */
            state[3] = twist.linear.x  = vw[0];
            state[4] = twist.angular.z = vw[1];
            cmd->publish(twist);

            int length = static_cast<int>(traj.size());
            for(int p = 1; p < length; p++)
                if(map->at(traj[p].head(2)) < tracker.z())
                {
                    length = p; break;
                }
            message.poses.resize(length);
            for(int p = 0; p < length; p++)
            {
                message.poses[p].pose.position.x = traj[p].x();
                message.poses[p].pose.position.y = traj[p].y();
                message.poses[p].pose.orientation.z = std::sin(traj[p].z() / 2);
                message.poses[p].pose.orientation.w = std::cos(traj[p].z() / 2);
            }
            publisher->publish(message);
            timing_neural_publish_ms += elapsed_ms(stage_start, SteadyClock::now());
            finish_ctrl(0);
            return;
        }

        /* Neural trajectory would hit an obstacle: engage the classical MPC
         * fallback for a short lock window (re-plan onto the A*-planned
         * trajectory), then retry the neural policy. Fall through below. */
        neural_lock_until = now + rclcpp::Duration::from_seconds(neural_fallback_lock);
        if(neural_divergent)
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "Stationary-target guard rejected divergent neural command "
                "(v=%.3f, omega=%.3f); falling back to classical MPC",
                vw[0], vw[1]);
        else
            RCLCPP_WARN(get_logger(),
                "Neural trajectory hits obstacle in near horizon; "
                "falling back to classical MPC for %.2fs", neural_fallback_lock);
    }

    /* === Neural fallback (visible but unsafe/locked) OR target lost ===
     * Classical MPC follows the A*-planned trajectory. During the neural lock
     * window mpc->visibility() is still true, so this re-plans the visible
     * tracking with the reliable planner; when the target is lost it follows
     * the reacquire trajectory. Both share the code below. */
    auto stage_start = SteadyClock::now();
    Eigen::MatrixXd control;
    int length = trajectory.cols();
    if(length > pointer + 1)
        control = mpc->control(state, trajectory.rightCols(length - pointer++));
    else
        control = mpc->control(state, state);

    /* Publish velocity */
    float d0 = map->at(control.col(0).head(2));
    float d1 = map->at(control.col(1).head(2));
    if(d1 <= 0 && d1 < d0)
    {
        control = mpc->control(state, state);
        d1 = map->at(control.col(1).head(2));
        if(d1 <= 0 && d1 < d0)
            control.col(1).tail(2).setZero();
    }
    timing_mpc_ms += elapsed_ms(stage_start, SteadyClock::now());
    timing_fallback_samples++;

    stage_start = SteadyClock::now();
    state[3] = twist.linear.x = control(3, 1);
    state[4] = twist.angular.z = control(4, 1);
    cmd->publish(twist);

    /* Publish trajectory */
    bool colussion = false;
    length = control.cols();
    for(int p = 1; p < length; p++)
        if((colussion = map->at(control.col(p).head(2)) < tracker.z()))
        {
            length = p; break;
        }
    message.poses.resize(length);
    for(int p = 0; p < length; p++)
    {
        message.poses[p].pose.position.x = control(0, p);
        message.poses[p].pose.position.y = control(1, p);
        message.poses[p].pose.orientation.z = std::sin(control(2, p) / 2);
        message.poses[p].pose.orientation.w = std::cos(control(2, p) / 2);
    }
    publisher->publish(message);

    /* Replanning */
    if(colussion && !mpc->visibility())
    {
        pointer = 0;
        twist.linear.x = 0;
        twist.angular.z = 0;
        cmd->publish(twist);
        state.tail(2).setZero();
        trajectory = planner->reacquire(true, state);
    }
    timing_fallback_publish_ms += elapsed_ms(stage_start, SteadyClock::now());
    finish_ctrl(1);
}

void Tracker::callback(const nav_msgs::msg::Path::SharedPtr b)
{
    if((tigger & 7) != 7)
        return;

    /* Update Bezier curve */
    int num = b->poses.size();
    time = rclcpp::Clock().now();
    if(num == 0)
    {
        stationary_hold = false;
        stationary_mpc_active = false;
        /* Predictor publishes an empty path as an explicit target-lost event.
         * Never re-enable the neural visible-target branch with stale Bezier
         * controls; stop first, then enter the existing MPC reacquisition. */
        if(mpc->visibility())
        {
            pointer = 0;
            twist.linear.x = 0;
            twist.angular.z = 0;
            cmd->publish(twist);
            mpc->visibility(false);
            state.tail(2).setZero();
            trajectory = planner->reacquire(false, state);
            RCLCPP_INFO(
                get_logger(),
                "Target prediction became empty; switched to MPC reacquisition");
        }
        return;
    }
    for(int p = 0; p < num; p++)
    {
        bezier->control(0, p) = b->poses[p].pose.position.x;
        bezier->control(1, p) = b->poses[p].pose.position.y;
    }

    /* Reset trajectory */
    if(pointer + 1 >= trajectory.cols())
    {
        pointer = 0;
        trajectory = state;
    }

    /* Path planning (kept: seeds reacquire trajectory / MPC fallback) */
    if(planner->plan(trajectory, state))
        pointer = 0;

    /* Set visibility -> switches ctrl() into the neural branch */
    mpc->visibility(true);
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Tracker>("tracker_neural"));
    rclcpp::shutdown();
    return 0;
}
