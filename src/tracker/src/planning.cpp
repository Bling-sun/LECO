/* @author: YueLin */

#include <chrono>
#include <functional>

#include <pcl_conversions/pcl_conversions.h>

#include "geometry_msgs/msg/quaternion.hpp"

#include "local_visualization.hpp"
#include "planning.hpp"

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
    tracker.z() = declare_parameter<double>("tracker.radius");
    declare_parameter<double>("tracker.fov");
    message.header.frame_id = declare_parameter<std::string>("map.name");

    declare_parameter<int>("bezier.m");
    declare_parameter<double>("bezier.tau");

    declare_parameter<double>("map.size");
    declare_parameter<double>("map.radius");
    declare_parameter<double>("map.resolution");

    declare_parameter<double>("tracker.max_vel");
    const double configured_max_acc =
        declare_parameter<double>("tracker.max_acc");
    dynamic_accel_by_visibility = declare_parameter<bool>(
        "tracker.dynamic_accel_by_visibility", false);
    visible_max_accel = declare_parameter<double>(
        "tracker.max_acc_visible", configured_max_acc);
    lost_max_accel = declare_parameter<double>(
        "tracker.max_acc_lost", configured_max_acc);
    declare_parameter<double>("tracker.max_alpha");
    declare_parameter<double>("tracker.max_omega");

    declare_parameter<int>("planner.kappa");
    declare_parameter<int>("planner.samples");
    declare_parameter<int>("planner.num_angles");
    declare_parameter<double>("planner.epsilon");
    declare_parameter<double>("planner.observation_distance");

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
    timing_log_interval = declare_parameter<double>("timing.log_interval", 2.0);
    const std::string target_topic = declare_parameter<std::string>(
        "target.topic", "/target/detection");
    const bool direct_target = declare_parameter<bool>("target.direct", false);
    use_fov_detection = declare_parameter<bool>("target.use_fov_detection", false);
    truth_fallback_after = declare_parameter<double>("target.truth_fallback_after", 3.0);
    timing_last_log = rclcpp::Clock().now();

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

        /* Hyperparameters */
        get_parameter("planner.num_angles").as_int(),
        get_parameter("planner.samples").as_int(),
        
        /* Robot settings */
        range, tracker.z(), dt,

        /* Motion constraints */
        get_parameter("tracker.max_vel").as_double(),
        get_parameter("tracker.max_acc").as_double(),

        /* Range of perception */
        get_parameter("planner.observation_distance").as_double(),
        get_parameter("planner.epsilon").as_double(),
        bezier->duration
    );

    /* Initialize model predictive controller */
    mpc = std::make_unique<diff_track::ModelPredictiveControl>(
        map.get(), bezier.get(),

        /* Robot settings */
        2 * tracker.z(), dt,
        get_parameter("planner.observation_distance").as_double(),

        /* Weights of cost terms */
        get_parameter("planner.lambda_a").as_double(),
        get_parameter("planner.lambda_d").as_double(),
        get_parameter("planner.lambda_o").as_double(),
        get_parameter("planner.lambda_c").as_double(),
        get_parameter("planner.lambda_f").as_double(),
        get_parameter("planner.lambda_s").as_double(),

        /* Motion constraints */
        get_parameter("tracker.max_vel").as_double(),
        get_parameter("tracker.max_acc").as_double(),
        get_parameter("tracker.max_omega").as_double(),
        get_parameter("tracker.max_alpha").as_double(),
        
        /* Hyperparameters for L-BFGS */
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

    /* Create publishers */
    publisher = create_publisher<nav_msgs::msg::Path>("/tracker/plan", 1);
    cmd = create_publisher<geometry_msgs::msg::Twist>("/tracker/cmd_vel", 1);
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
            const bool visible = direct_target || !use_fov_detection || detection->pose.orientation.w > 0;
            if(visible)
            {
                detection_lost_since = rclcpp::Time(0, 0, RCL_ROS_TIME);
                if(truth_fallback_active) {
                    truth_fallback_active = false;
                    std_msgs::msg::String status; status.data = "fov_detection_recovered";
                    observation_status->publish(status);
                }
                target.x() = detection->pose.position.x;
                target.y() = detection->pose.position.y;
                target.z() = detection->pose.position.z * 1.5;
            }
            else
            {
                if(detection_lost_since.nanoseconds() == 0) detection_lost_since = rclcpp::Clock().now();
                target.z() = -1;
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
                        "Target detection lost; switched immediately to MPC reacquisition");
                }
            }
        }
    );
    /* The traditional FOV workflow may feed this controller through the
       detection/truth mux.  Sample the raw detector independently so the
       benchmark visibility bit is never changed to visible by truth fallback. */
    raw_detection_metrics = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/target/detection", 10,
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr detection){
            const bool visible = detection->pose.orientation.w > 0;
            if(dynamic_accel_by_visibility && visible != raw_detection_visible)
            {
                const double max_accel =
                    visible ? visible_max_accel : lost_max_accel;
                mpc->maxAcceleration(max_accel);
                RCLCPP_INFO(
                    get_logger(),
                    "FOV %s: tracker maximum acceleration -> %.3f m/s^2",
                    visible ? "visible" : "lost", max_accel);
            }
            raw_detection_visible = visible;
        }
    );
    observation_status = create_publisher<std_msgs::msg::String>("/tracker/target_observation_status", 10);
    metrics_sample = create_publisher<std_msgs::msg::Float64MultiArray>("/tracker/metrics_sample", 100);
    if(use_fov_detection && !direct_target) {
        truth = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/target/ground_truth", 1,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg){ latest_truth = *msg; have_truth = true; });
    }
    RCLCPP_INFO(
        get_logger(), "Target input: %s (%s)", target_topic.c_str(),
        direct_target? "direct position": "visibility detection");
    if(dynamic_accel_by_visibility)
        RCLCPP_INFO(
            get_logger(),
            "Dynamic FOV acceleration enabled: visible %.3f m/s^2, lost %.3f m/s^2",
            visible_max_accel, lost_max_accel);

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
        "  传统控制循环总耗时=%.3f（一次 ctrl() 从进入到发布完成，样本数=%llu）\n"
        "  传统MPC求解=%.3f（mpc->control + 立即碰撞保护重算）\n"
        "  发布控制/轨迹=%.3f（cmd_vel + plan 发布 + 必要重规划）",
        timing_log_interval,
        avg(timing_ctrl_ms, timing_ctrl_samples),
        static_cast<unsigned long long>(timing_ctrl_samples),
        avg(timing_mpc_ms, timing_ctrl_samples),
        avg(timing_publish_ms, timing_ctrl_samples)
    );

    timing_last_log = now;
    timing_ctrl_samples = 0;
    timing_ctrl_ms = 0.0;
    timing_mpc_ms = 0.0;
    timing_publish_ms = 0.0;
}

void Tracker::ctrl()
{
    if (use_fov_detection && !truth_fallback_active && have_truth &&
        detection_lost_since.nanoseconds() != 0 &&
        (rclcpp::Clock().now() - detection_lost_since).seconds() >= truth_fallback_after) {
        truth_fallback_active = true;
        target.x() = latest_truth.pose.position.x;
        target.y() = latest_truth.pose.position.y;
        target.z() = latest_truth.pose.position.z * 1.5;
        std_msgs::msg::String status; status.data = "truth_fallback_started";
        observation_status->publish(status);
        RCLCPP_WARN(get_logger(), "FOV target lost; using ground truth until recovery");
    }
    if (truth_fallback_active && have_truth) {
        target.x() = latest_truth.pose.position.x;
        target.y() = latest_truth.pose.position.y;
        target.z() = latest_truth.pose.position.z * 1.5;
    }
    if((tigger & 7) != 7)
        return;

    const auto ctrl_start = SteadyClock::now();

    /* Set time stamp */
    rclcpp::Time now = rclcpp::Clock().now();
    mpc->clock((now - time).seconds());
    message.header.stamp = now;
    
    /* Model predictive control */
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
    timing_publish_ms += elapsed_ms(stage_start, SteadyClock::now());
    const double cycle_ms = elapsed_ms(ctrl_start, SteadyClock::now());
    timing_ctrl_ms += cycle_ms;
    timing_ctrl_samples++;
    std_msgs::msg::Float64MultiArray sample;
    sample.data = {
        static_cast<double>(++metrics_sequence),
        static_cast<double>(twist.linear.x),
        static_cast<double>(twist.angular.z),
        cycle_ms,
        1.0,
        raw_detection_visible ? 1.0 : 0.0
    };
    metrics_sample->publish(sample);
    log_timing_if_needed(now);
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
        /* An empty prediction is an explicit target-lost event. Do not mark
           the stale Bezier curve visible again. */
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

    /* Path planning */
    if(planner->plan(trajectory, state))
        pointer = 0;

    /* Set visibility of MPC */
    mpc->visibility(true);
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Tracker>("tracker"));
    rclcpp::shutdown();
    return 0;
}
