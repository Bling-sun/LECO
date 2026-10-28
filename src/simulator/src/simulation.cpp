/* @author: YueLin */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>

#include "simulation.hpp"

Simulator::Simulator(std::string name): Node(name)
{
    /* Declare hyperparameters */
    dt = 1. / declare_parameter<double>("hz");
    range = declare_parameter<double>("tracker.range");
    world = declare_parameter<std::string>("map.name");
    target = declare_parameter<std::string>("target.name");
    tracker = declare_parameter<std::string>("tracker.name");
    horizontal_angle = declare_parameter<double>("tracker.fov") * PI / 360;
    vertical_angle = declare_parameter<double>("tracker.vertical_fov") * PI / 360;
    camera_height = declare_parameter<double>("tracker.camera_height");
    lidar_range = declare_parameter<double>("lidar.range", 40.0);
    lidar_angular_resolution = declare_parameter<double>(
        "lidar.angular_resolution", 1.0) * PI / 180.0;
    if(lidar_range <= 0. || lidar_angular_resolution <= 0.)
        throw std::invalid_argument(
            "lidar.range and lidar.angular_resolution must be positive");
    vm = declare_parameter<double>("target.max_vel") / std::sqrt(2);

    declare_parameter<int>("map.seed");
    declare_parameter<int>("map.obstacles");
    declare_parameter<double>("map.width");
    declare_parameter<double>("map.height");
    declare_parameter<double>("map.resolution");
    declare_parameter<std::string>("map.style");
    const std::string map_file = declare_parameter<std::string>(
        "map.file", "");
    const std::string map_style = get_parameter("map.style").as_string();
    
    declare_parameter<double>("target.x");
    declare_parameter<double>("target.y");
    declare_parameter<double>("target.yaw");
    declare_parameter<double>("target.radius");
    declare_parameter<double>("target.max_acc");
    const double target_planning_clearance = declare_parameter<double>(
        "target.planning_clearance", 0.6);
    declare_parameter<double>("target.preferred_clearance", 0.8);
    declare_parameter<double>("target.wall_cost_weight", 4.0);
    external_target_control = declare_parameter<bool>(
        "target.external_control", false);
    const bool random_motion_enabled = declare_parameter<bool>(
        "target.random_motion_enabled", true);

    declare_parameter<double>("tracker.x");
    declare_parameter<double>("tracker.y");
    declare_parameter<double>("tracker.yaw");
    declare_parameter<double>("tracker.radius");
    const double map3d_min_z = declare_parameter<double>("map3d.min_z", 0.05);
    const double map3d_max_z = declare_parameter<double>("map3d.max_z", 1.50);
    const double map3d_resolution = std::max(0.01, declare_parameter<double>(
        "map3d.vertical_resolution", 0.10));
    const double min_obstacle_height = declare_parameter<double>(
        "map3d.min_obstacle_height", 0.80);
    const double max_obstacle_height = declare_parameter<double>(
        "map3d.max_obstacle_height", map3d_max_z);
    temporary_obstacle_enabled = declare_parameter<bool>(
        "temporary_obstacle.enabled", false);
    temporary_obstacle_auto_add = declare_parameter<bool>(
        "temporary_obstacle.auto_add", false);
    const auto temporary_obstacle_seed = declare_parameter<std::int64_t>(
        "temporary_obstacle.seed", -1);
    temporary_obstacle_interval = declare_parameter<double>(
        "temporary_obstacle.interval", 5.0);
    temporary_obstacle_min_size = declare_parameter<double>(
        "temporary_obstacle.min_size", 0.60);
    temporary_obstacle_max_size = declare_parameter<double>(
        "temporary_obstacle.max_size", 1.20);
    temporary_obstacle_height = declare_parameter<double>(
        "temporary_obstacle.height", 4.0);
    temporary_obstacle_placement_jitter = declare_parameter<double>(
        "temporary_obstacle.placement_jitter", 1.50);
    temporary_obstacle_minimum_spacing = declare_parameter<double>(
        "temporary_obstacle.minimum_spacing", 0.50);
    temporary_obstacle_linear_motion_threshold = declare_parameter<double>(
        "temporary_obstacle.linear_motion_threshold", 0.02);
    temporary_obstacle_stuck_timeout = declare_parameter<double>(
        "temporary_obstacle.stuck_timeout", 3.0);
    if(temporary_obstacle_interval <= 0.
       || temporary_obstacle_min_size <= 0.
       || temporary_obstacle_max_size < temporary_obstacle_min_size
       || temporary_obstacle_height <= 0.
       || temporary_obstacle_placement_jitter < 0.
       || temporary_obstacle_minimum_spacing < 0.
       || temporary_obstacle_linear_motion_threshold < 0.
       || temporary_obstacle_stuck_timeout <= 0.
       || temporary_obstacle_seed < -1)
        throw std::invalid_argument(
            "temporary obstacle seed/interval/size/height/jitter "
            "configuration is invalid");
    if(temporary_obstacle_seed >= 0)
    {
        temporary_obstacle_rng.seed(
            static_cast<std::mt19937_64::result_type>(temporary_obstacle_seed));
        RCLCPP_INFO(
            get_logger(), "Temporary obstacle random seed: %ld (fixed)",
            static_cast<long>(temporary_obstacle_seed));
    }
    else
    {
        std::random_device random_device;
        const auto seed_time = static_cast<unsigned long long>(
            std::chrono::high_resolution_clock::now()
                .time_since_epoch().count());
        std::seed_seq obstacle_seed{
            random_device(), random_device(),
            static_cast<unsigned int>(seed_time),
            static_cast<unsigned int>(seed_time >> 32)
        };
        temporary_obstacle_rng.seed(obstacle_seed);
        RCLCPP_INFO(
            get_logger(), "Temporary obstacle random seed: fresh for this run");
    }
    last_tracker_translation_at = std::chrono::steady_clock::now();
    const double requested_target_x = get_parameter("target.x").as_double();
    const double requested_target_y = get_parameter("target.y").as_double();
    const double requested_tracker_x = get_parameter("tracker.x").as_double();
    const double requested_tracker_y = get_parameter("tracker.y").as_double();
    const double target_radius = get_parameter("target.radius").as_double();
    const double tracker_radius = get_parameter("tracker.radius").as_double();
    if(target_radius <= 0. || tracker_radius <= 0.
       || target_planning_clearance <= 0.)
        throw std::invalid_argument(
            "tracker/target radius and target planning clearance must be "
            "positive");

    /* Initialize random map */
    map = std::make_unique<sim::RandomMap>(
        get_parameter("map.seed").as_int(),
        get_parameter("map.obstacles").as_int(),
        map_style,
        get_parameter("map.width").as_double(),
        get_parameter("map.height").as_double(),
        get_parameter("map.resolution").as_double(),
        tracker_radius,
        requested_tracker_x,
        requested_tracker_y,
        requested_target_x,
        requested_target_y,
        min_obstacle_height,
        max_obstacle_height,
        map_file
    );
    /* A robot center needs its clearance on both sides of a passage. Use the
       target planner's hard clearance as well as both physical radii, so a
       generated gap is usable by both actors rather than merely non-overlapping.
       minimum_spacing remains a user-configurable absolute lower bound. */
    const double required_center_clearance = std::max({
        tracker_radius, target_radius, target_planning_clearance
    });
    temporary_obstacle_passage_width = std::max(
        temporary_obstacle_minimum_spacing,
        2.0 * required_center_clearance);
    /* static_cost() is sampled on grid-cell centers. The legacy encode()
       convention can shift a query by up to one cell, while rasterizing the
       static footprint contributes another half-cell uncertainty. Two full
       cell diagonals are deliberately conservative for both effects. */
    temporary_obstacle_static_grid_margin =
        2.0 * std::sqrt(2.0) * map->resolution;
    if(temporary_obstacle_enabled)
        RCLCPP_INFO(
            get_logger(),
            "Temporary-obstacle passage width: %.2f m (configured minimum "
            "%.2f m, tracker diameter %.2f m, target diameter %.2f m, "
            "target planner width %.2f m; static-grid margin %.3f m)",
            temporary_obstacle_passage_width,
            temporary_obstacle_minimum_spacing,
            2.0 * tracker_radius, 2.0 * target_radius,
            2.0 * target_planning_clearance,
            temporary_obstacle_static_grid_margin);
    if(!map_file.empty())
        RCLCPP_INFO(get_logger(), "Loaded custom map: %s", map_file.c_str());

    const auto tracker_spawn = map->nearest_free(
        requested_tracker_x, requested_tracker_y, tracker_radius + 0.05);
    const auto target_spawn = map->nearest_free(
        requested_target_x, requested_target_y, target_radius + 0.05);
    if(std::hypot(tracker_spawn.first - requested_tracker_x,
                  tracker_spawn.second - requested_tracker_y) > 1e-6)
        RCLCPP_WARN(
            get_logger(),
            "Tracker spawn was occupied; relocated from (%.3f, %.3f) "
            "to nearest free position (%.3f, %.3f)",
            requested_tracker_x, requested_tracker_y,
            tracker_spawn.first, tracker_spawn.second);
    if(std::hypot(target_spawn.first - requested_target_x,
                  target_spawn.second - requested_target_y) > 1e-6)
        RCLCPP_WARN(
            get_logger(),
            "Target spawn was occupied; relocated from (%.3f, %.3f) "
            "to nearest free position (%.3f, %.3f)",
            requested_target_x, requested_target_y,
            target_spawn.first, target_spawn.second);

    /* Initialize simulation environment */
    env = std::make_unique<sim::Environment>(
        world, map.get(),
        target_spawn.first,
        target_spawn.second,
        get_parameter("target.yaw").as_double(),
        target_radius,
        tracker_spawn.first,
        tracker_spawn.second,
        get_parameter("tracker.yaw").as_double(),
        tracker_radius
    );

    /* Camera visualization/detection and MID-360 scan share the simulator
       object, but use independent FoV/range parameters. */
    sensor = std::make_unique<sim::Camera>(
        tracker, env->tracker, horizontal_angle, vertical_angle,
        range, camera_height, lidar_range, lidar_angular_resolution);

    /* Initialize path planner for the target */
    planner = std::make_unique<sim::Planner>(
        &env->target, map.get(), dt * 2, vm,
        get_parameter("target.max_acc").as_double(),
        target_planning_clearance,
        get_parameter("target.preferred_clearance").as_double(),
        get_parameter("target.wall_cost_weight").as_double()
    );

    /* Initialize messages */
    scan.header.frame_id = world;
    transform.header.frame_id = world;

    /* Create publishers and broadcaster */
    planning = create_publisher<nav_msgs::msg::Path>(
        "/target/plan", 1
    );
    trajectories[0] = create_publisher<nav_msgs::msg::Path>(
        "/target/trajectory", 1
    );
    trajectories[1] = create_publisher<nav_msgs::msg::Path>(
        "/tracker/trajectory", 1
    );
    odometry = create_publisher<nav_msgs::msg::Odometry>(
        "/tracker/odom", 1
    );
    detector = create_publisher<geometry_msgs::msg::PoseStamped>(
        "/target/detection", 1
    );
    target_ground_truth = create_publisher<geometry_msgs::msg::PoseStamped>(
        "/target/ground_truth", 1
    );
    fov = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/tracker/fov", 1);
    mapper = create_publisher<nav_msgs::msg::OccupancyGrid>(
        "/map", rclcpp::QoS(1).reliable().transient_local());
    broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(this);
    depth = create_publisher<sensor_msgs::msg::PointCloud2>("/tracker/scan", 1);
    map3d = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/tracker/map3d", rclcpp::QoS(1).reliable().transient_local());

    /* Build a parallel 3D representation. /map remains available for truth
       visualization; tracker obstacle perception comes from MID-360 scan. */
    nav_msgs::msg::OccupancyGrid map_message = *map->message(world);
    pcl::PointCloud<pcl::PointXYZRGB> cloud3d;
    const int width = static_cast<int>(map_message.info.width);
    const int height = static_cast<int>(map_message.info.height);
    const double resolution = map_message.info.resolution;
    const int boundary_cells = std::max(
        1, static_cast<int>(std::ceil(
            get_parameter("tracker.radius").as_double() / resolution)));
    const double obstacle_height_span = std::max(
        1e-6, max_obstacle_height - min_obstacle_height);

    auto add_colored_point = [
        &cloud3d, &map_style, min_obstacle_height, obstacle_height_span
    ](float px, float py, float pz, float object_height,
      bool top_surface, bool map_border)
    {
        double height_mix = (object_height - min_obstacle_height)
                          / obstacle_height_span;
        height_mix = std::max(0.0, std::min(1.0, height_mix));
        const double vertical_mix = object_height > 1e-6
            ? std::max(0.0, std::min(
                1.0, static_cast<double>(pz / object_height))) : 0.0;

        double red = 86.0, green = 112.0, blue = 126.0;
        if(map_border)
        {
            red = 54.0; green = 63.0; blue = 70.0;
        }
        else if(map_style == "forest" || map_style == "forest_dense")
        {
            if(top_surface)
            {
                red = 48.0 + 18.0 * height_mix;
                green = 112.0 + 34.0 * height_mix;
                blue = 65.0 + 13.0 * height_mix;
            }
            else
            {
                red = 91.0 + 28.0 * vertical_mix;
                green = 61.0 + 20.0 * vertical_mix;
                blue = 38.0 + 12.0 * vertical_mix;
            }
        }
        else if(map_style == "rocky")
        {
            const double light = 22.0 * vertical_mix + 18.0 * height_mix;
            red = 92.0 + light;
            green = 99.0 + light;
            blue = 103.0 + light;
        }
        else if(map_style == "offroad_mixed")
        {
            if(object_height > 1.35)
            {
                red = top_surface ? 62.0 : 104.0;
                green = top_surface ? 127.0 : 75.0;
                blue = top_surface ? 72.0 : 44.0;
            }
            else if(object_height < 0.75)
            {
                red = 148.0 + 18.0 * vertical_mix;
                green = 119.0 + 15.0 * vertical_mix;
                blue = 80.0 + 12.0 * vertical_mix;
            }
            else
            {
                red = 101.0 + 25.0 * vertical_mix;
                green = 108.0 + 25.0 * vertical_mix;
                blue = 105.0 + 25.0 * vertical_mix;
            }
        }
        else
        {
            const double light = 30.0 * vertical_mix + 16.0 * height_mix;
            red += light; green += light; blue += light;
        }

        pcl::PointXYZRGB point;
        point.x = px; point.y = py; point.z = pz;
        point.r = static_cast<std::uint8_t>(
            std::max(0.0, std::min(255.0, red)));
        point.g = static_cast<std::uint8_t>(
            std::max(0.0, std::min(255.0, green)));
        point.b = static_cast<std::uint8_t>(
            std::max(0.0, std::min(255.0, blue)));
        cloud3d.push_back(point);
    };

    for(int y = 0; y < height; ++y)
        for(int x = 0; x < width; ++x)
        {
            const int index = y * width + x;
            if(map_message.data[index] < 50)
                continue;
            bool boundary = false;
            constexpr int dx[4] = {-1, 1, 0, 0};
            constexpr int dy[4] = {0, 0, -1, 1};
            for(int k = 0; k < 4 && !boundary; ++k)
            {
                const int nx = x + dx[k], ny = y + dy[k];
                boundary = nx < 0 || nx >= width || ny < 0 || ny >= height
                        || map_message.data[ny * width + nx] < 50;
            }
            const float px = map_message.info.origin.position.x
                           + (static_cast<double>(x) + 0.5) * resolution;
            const float py = map_message.info.origin.position.y
                           + (static_cast<double>(y) + 0.5) * resolution;
            const float obstacle_height = map->obstacle_height(x, y);
            const bool map_border = x < boundary_cells || y < boundary_cells
                || x >= width - boundary_cells || y >= height - boundary_cells;
            if(boundary)
            {
                const int layers = std::max(
                    1, static_cast<int>(std::floor(
                        (obstacle_height - map3d_min_z) / map3d_resolution
                        + 0.5)) + 1);
                for(int layer = 0; layer < layers; ++layer)
                {
                    const float pz = std::min(
                        obstacle_height,
                        static_cast<float>(
                            map3d_min_z + layer * map3d_resolution));
                    add_colored_point(
                        px, py, pz, obstacle_height, false, map_border);
                }
            }
            /* Top surfaces make the static cloud useful to 3D viewers and
               depth renderers, while vertical samples above remain sparse. */
            if(!boundary || cloud3d.empty()
               || std::fabs(cloud3d.back().z - obstacle_height) > 1e-4)
                add_colored_point(
                    px, py, obstacle_height, obstacle_height,
                    true, map_border);
        }

    /* Tree crowns are elevated visual geometry. They must not be rasterized
       into /map or the ESDF because doing so would turn the full crown shadow
       into an impassable ground obstacle. Add sampled ellipsoid surfaces only
       to the RGB 3-D cloud; ground collision continues to use the trunk. */
    const double canopy_spacing = std::max(0.20, map3d_resolution * 2.0);
    const double two_pi = 2.0 * PI;
    for(const auto& canopy: map->canopy_visuals())
    {
        const double max_radius = std::max({
            canopy.radius_x, canopy.radius_y, canopy.radius_z
        });
        const int latitude_steps = std::max(
            8, static_cast<int>(std::ceil(
                PI * max_radius / canopy_spacing)));
        const double cosine = std::cos(canopy.yaw);
        const double sine = std::sin(canopy.yaw);
        for(int latitude = 0; latitude <= latitude_steps; ++latitude)
        {
            const double theta = -PI * 0.5
                + PI * static_cast<double>(latitude) / latitude_steps;
            const double ring_scale = std::cos(theta);
            const double vertical_scale = std::sin(theta);
            const int longitude_steps = std::fabs(ring_scale) < 1e-6
                ? 1
                : std::max(12, static_cast<int>(std::ceil(
                    two_pi * std::max(
                        canopy.radius_x, canopy.radius_y)
                    * std::fabs(ring_scale) / canopy_spacing)));
            const double shade = 0.84 + 0.18
                * (vertical_scale + 1.0) * 0.5;
            for(int longitude = 0;
                longitude < longitude_steps; ++longitude)
            {
                const double phi = two_pi
                    * static_cast<double>(longitude) / longitude_steps;
                const double local_x = canopy.radius_x
                    * ring_scale * std::cos(phi);
                const double local_y = canopy.radius_y
                    * ring_scale * std::sin(phi);
                pcl::PointXYZRGB point;
                point.x = static_cast<float>(
                    canopy.x + cosine * local_x - sine * local_y);
                point.y = static_cast<float>(
                    canopy.y + sine * local_x + cosine * local_y);
                point.z = static_cast<float>(
                    canopy.z + canopy.radius_z * vertical_scale);
                point.r = static_cast<std::uint8_t>(std::min(
                    255.0, shade * canopy.red));
                point.g = static_cast<std::uint8_t>(std::min(
                    255.0, shade * canopy.green));
                point.b = static_cast<std::uint8_t>(std::min(
                    255.0, shade * canopy.blue));
                cloud3d.push_back(point);
            }
        }
    }

    /* Arch spans are elevated geometry and therefore cannot be recovered by
       extruding the two occupied support footprints. Sample the curved band
       explicitly while leaving the opening free in /map and the ESDF. */
    const double arch_spacing = std::max(0.05, map3d_resolution);
    for(const auto& arch: map->arch_visuals())
    {
        const double radius_x = std::max(
            arch.thickness, (arch.width - arch.thickness) * 0.5);
        const double radius_z = std::max(
            arch.thickness,
            arch.height - arch.clearance - arch.thickness * 0.5);
        const int theta_steps = std::max(
            16, static_cast<int>(std::ceil(
                PI * std::max(radius_x, radius_z) / arch_spacing)));
        const int depth_steps = std::max(
            1, static_cast<int>(std::ceil(
                arch.depth / arch_spacing)));
        const int thickness_steps = std::max(
            1, static_cast<int>(std::ceil(
                arch.thickness / arch_spacing)));
        const double cosine = std::cos(arch.yaw);
        const double sine = std::sin(arch.yaw);
        const auto add_arch_point = [
            &add_colored_point, &arch, cosine, sine
        ](double local_x, double local_y, double z, bool top_surface)
        {
            add_colored_point(
                static_cast<float>(
                    arch.x + cosine * local_x - sine * local_y),
                static_cast<float>(
                    arch.y + sine * local_x + cosine * local_y),
                static_cast<float>(z),
                static_cast<float>(arch.height),
                top_surface, false);
        };

        for(int step = 0; step <= theta_steps; ++step)
        {
            const double theta = PI * step / theta_steps;
            const double center_x = radius_x * std::cos(theta);
            const double center_z = arch.clearance
                + radius_z * std::sin(theta);
            const double tangent_x = -radius_x * std::sin(theta);
            const double tangent_z = radius_z * std::cos(theta);
            const double tangent_length = std::hypot(
                tangent_x, tangent_z);
            const double normal_x = tangent_z / tangent_length;
            const double normal_z = -tangent_x / tangent_length;

            for(int depth = 0; depth <= depth_steps; ++depth)
            {
                const double local_y = -arch.depth * 0.5
                    + arch.depth * depth / depth_steps;
                add_arch_point(
                    center_x + normal_x * arch.thickness * 0.5,
                    local_y,
                    center_z + normal_z * arch.thickness * 0.5,
                    true);
                add_arch_point(
                    center_x - normal_x * arch.thickness * 0.5,
                    local_y,
                    center_z - normal_z * arch.thickness * 0.5,
                    false);
            }
            for(int layer = 0; layer <= thickness_steps; ++layer)
            {
                const double offset = -arch.thickness * 0.5
                    + arch.thickness * layer / thickness_steps;
                const double local_x = center_x + normal_x * offset;
                const double z = center_z + normal_z * offset;
                add_arch_point(local_x, -arch.depth * 0.5, z, false);
                add_arch_point(local_x, arch.depth * 0.5, z, false);
            }
        }
    }

    sensor_msgs::msg::PointCloud2 map3d_cloud;
    pcl::toROSMsg(cloud3d, map3d_cloud);
    map3d_cloud.header.frame_id = world;

    /* Both maps are static. Transient-local durability keeps this single
       publication available to RViz and any other late-joining subscriber. */
    const auto static_map_stamp = now();
    map_message.header.stamp = static_map_stamp;
    map3d_cloud.header.stamp = static_map_stamp;
    mapper->publish(map_message);
    map3d->publish(map3d_cloud);
    if(temporary_obstacle_enabled)
        static_map3d_cloud = map3d_cloud;

    /* Create subscribers */
    controller = create_subscription<geometry_msgs::msg::Twist>(
        "/tracker/cmd_vel", 1, 
        [this](const geometry_msgs::msg::Twist::SharedPtr cmd){
            env->tracker.control(*cmd);
        }
    );
    goal = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", 1, 
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr goal){
            if(external_target_control)
                return;
            target_goal = goal->pose;
            has_target_goal = true;
            planner->plan(target_goal);
        }
    );
    if(external_target_control)
    {
        external_target = create_subscription<geometry_msgs::msg::PoseStamped>(
            "/target/reference_pose", 10,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr pose){
                if(!pose->header.frame_id.empty()
                   && pose->header.frame_id != world)
                {
                    RCLCPP_WARN_THROTTLE(
                        get_logger(), *get_clock(), 2000,
                        "Ignored target reference in frame '%s'; expected '%s'",
                        pose->header.frame_id.c_str(), world.c_str());
                    return;
                }
                external_target_pose = pose->pose;
                have_external_target_pose = true;
            });
        RCLCPP_INFO(
            get_logger(),
            "External target control enabled on /target/reference_pose");
    }
    rnd = create_subscription<geometry_msgs::msg::PointStamped>(
        "/clicked_point", 1, 
        [this, random_motion_enabled](
            const geometry_msgs::msg::PointStamped::SharedPtr msg){
            if(!random_motion_enabled)
                return;
            moving = !moving;
            std::srand(msg->header.stamp.nanosec);
        }
    );
    if(temporary_obstacle_enabled)
        temporary_obstacle_click = create_subscription<
            geometry_msgs::msg::PointStamped>(
            "/temporary_obstacle", 10,
            [this](const geometry_msgs::msg::PointStamped::SharedPtr msg){
                if(!msg->header.frame_id.empty()
                   && msg->header.frame_id != world)
                {
                    RCLCPP_WARN(
                        get_logger(),
                        "Ignored temporary obstacle in frame '%s'; "
                        "expected '%s'",
                        msg->header.frame_id.c_str(), world.c_str());
                    return;
                }
                if(!std::isfinite(msg->point.x)
                   || !std::isfinite(msg->point.y))
                {
                    RCLCPP_WARN(
                        get_logger(),
                        "Ignored temporary obstacle with non-finite position");
                    return;
                }
                if(!tracker_has_recent_linear_motion)
                {
                    RCLCPP_WARN(
                        get_logger(),
                        "Ignored RViz obstacle request: tracker has no recent "
                        "effective linear motion");
                    return;
                }
                const double reference_yaw = std::atan2(
                    env->target.pose.y - env->tracker.pose.y,
                    env->target.pose.x - env->tracker.pose.x);
                const double tracker_target_distance = std::hypot(
                    env->target.pose.x - env->tracker.pose.x,
                    env->target.pose.y - env->tracker.pose.y);
                if(!add_temporary_obstacle(
                       make_random_temporary_obstacle(
                           msg->point.x, msg->point.y, reference_yaw,
                           tracker_target_distance),
                       "RViz click"))
                    RCLCPP_WARN(
                        get_logger(),
                        "Ignored RViz obstacle request: placement is unsafe "
                        "or temporary obstacles are too dense");
            });

    /* Initialize timer */
    int ms = dt * 1e3;
    timer = create_wall_timer(
        std::chrono::milliseconds(ms),
        std::bind(&Simulator::step, this)
    );

    /* Detect target */
    ms = 1e3 / declare_parameter<double>("tracker.fps");
    detection = create_wall_timer(std::chrono::milliseconds(ms), [this](){
        detector->publish(*env->detect(
            horizontal_angle, vertical_angle, camera_height,
            range, get_logger()));
    });

    /* Random movement of the target */
    move = create_wall_timer(std::chrono::milliseconds(ms), [this](){
        if(moving && 1e-2 > std::fabs(env->target.vel.yaw) + hypot2(
            env->target.vel.x, env->target.vel.y
        ))
        {
            target_goal = map->goal();
            has_target_goal = true;
            planner->plan(target_goal);
        }
    });

    if(temporary_obstacle_enabled && temporary_obstacle_auto_add)
        automatic_obstacle_timer = create_wall_timer(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::duration<double>(temporary_obstacle_interval)),
            std::bind(&Simulator::add_automatic_temporary_obstacle, this));

    if(map_file.empty())
        RCLCPP_INFO(get_logger(), "Simulator ready");
    else
        RCLCPP_INFO(
            get_logger(), "Simulator ready with custom map: %s",
            map_file.c_str());

}

void Simulator::publish_temporary_obstacles()
{
    map->set_temporary_obstacles(temporary_obstacles);
    if(has_target_goal)
        planner->plan(target_goal);

    auto map_message = *map->message(world);
    map_message.header.stamp = now();
    mapper->publish(map_message);

    pcl::PointCloud<pcl::PointXYZRGB> dynamic_cloud;
    pcl::fromROSMsg(static_map3d_cloud, dynamic_cloud);
    const double spacing = std::max(0.05, map->resolution * 2.0);
    for(const auto& obstacle: temporary_obstacles)
    {
        std::uint8_t red = 255, green = 82, blue = 10;
        if(obstacle.shape == sim::TemporaryObstacleShape::Box)
        {
            red = 245; green = 170; blue = 30;
        }
        else if(obstacle.shape == sim::TemporaryObstacleShape::Wall)
        {
            red = 230; green = 45; blue = 85;
        }
        const auto add_point = [
            &dynamic_cloud, red, green, blue
        ](double x, double y, double z)
        {
            pcl::PointXYZRGB point;
            point.x = static_cast<float>(x);
            point.y = static_cast<float>(y);
            point.z = static_cast<float>(z);
            point.r = red;
            point.g = green;
            point.b = blue;
            dynamic_cloud.push_back(point);
        };

        const int layers = std::max(1, static_cast<int>(std::ceil(
            obstacle.height / spacing)));
        if(obstacle.shape == sim::TemporaryObstacleShape::Cylinder)
        {
            const double radius = obstacle.width * 0.5;
            const int angles = std::max(16, static_cast<int>(std::ceil(
                2.0 * PI * radius / spacing)));
            for(int layer = 0; layer <= layers; ++layer)
            {
                const double z = obstacle.height * layer / layers;
                for(int angle = 0; angle < angles; ++angle)
                {
                    const double theta = 2.0 * PI * angle / angles;
                    add_point(
                        obstacle.x + radius * std::cos(theta),
                        obstacle.y + radius * std::sin(theta), z);
                }
            }
            const int radial_steps = std::max(1, static_cast<int>(std::ceil(
                radius / spacing)));
            for(int radial = 0; radial <= radial_steps; ++radial)
            {
                const double ring_radius = radius * radial / radial_steps;
                const int ring_points = std::max(1,
                    static_cast<int>(std::ceil(
                        2.0 * PI * ring_radius / spacing)));
                for(int angle = 0; angle < ring_points; ++angle)
                {
                    const double theta = 2.0 * PI * angle / ring_points;
                    add_point(
                        obstacle.x + ring_radius * std::cos(theta),
                        obstacle.y + ring_radius * std::sin(theta),
                        obstacle.height);
                }
            }
        }
        else
        {
            const double cosine = std::cos(obstacle.yaw);
            const double sine = std::sin(obstacle.yaw);
            const auto add_local = [
                &add_point, &obstacle, cosine, sine
            ](double x, double y, double z)
            {
                add_point(
                    obstacle.x + cosine * x - sine * y,
                    obstacle.y + sine * x + cosine * y, z);
            };
            const int width_steps = std::max(1,
                static_cast<int>(std::ceil(obstacle.width / spacing)));
            const int depth_steps = std::max(1,
                static_cast<int>(std::ceil(obstacle.depth / spacing)));
            for(int layer = 0; layer <= layers; ++layer)
            {
                const double z = obstacle.height * layer / layers;
                for(int step = 0; step <= width_steps; ++step)
                {
                    const double x = -obstacle.width * 0.5
                        + obstacle.width * step / width_steps;
                    add_local(x, -obstacle.depth * 0.5, z);
                    add_local(x, obstacle.depth * 0.5, z);
                }
                for(int step = 0; step <= depth_steps; ++step)
                {
                    const double y = -obstacle.depth * 0.5
                        + obstacle.depth * step / depth_steps;
                    add_local(-obstacle.width * 0.5, y, z);
                    add_local(obstacle.width * 0.5, y, z);
                }
            }
            for(int x_step = 0; x_step <= width_steps; ++x_step)
                for(int y_step = 0; y_step <= depth_steps; ++y_step)
                    add_local(
                        -obstacle.width * 0.5
                            + obstacle.width * x_step / width_steps,
                        -obstacle.depth * 0.5
                            + obstacle.depth * y_step / depth_steps,
                        obstacle.height);
        }
    }
    sensor_msgs::msg::PointCloud2 dynamic_map3d;
    pcl::toROSMsg(dynamic_cloud, dynamic_map3d);
    dynamic_map3d.header.frame_id = world;
    dynamic_map3d.header.stamp = map_message.header.stamp;
    map3d->publish(dynamic_map3d);
}

sim::TemporaryObstacle Simulator::make_random_temporary_obstacle(
    double x, double y, double reference_yaw, double maximum_diameter)
{
    std::uniform_real_distribution<double> size_distribution(
        temporary_obstacle_min_size, temporary_obstacle_max_size);
    std::uniform_real_distribution<double> yaw_distribution(-PI, PI);
    std::uniform_int_distribution<int> shape_distribution(0, 2);

    sim::TemporaryObstacle obstacle;
    const int shape = shape_distribution(temporary_obstacle_rng);
    obstacle.shape = shape == 0
        ? sim::TemporaryObstacleShape::Cylinder
        : shape == 1 ? sim::TemporaryObstacleShape::Box
                     : sim::TemporaryObstacleShape::Wall;
    obstacle.x = x;
    obstacle.y = y;
    obstacle.height = temporary_obstacle_height;
    obstacle.yaw = reference_yaw
        + yaw_distribution(temporary_obstacle_rng);
    if(obstacle.shape == sim::TemporaryObstacleShape::Cylinder)
    {
        obstacle.width = size_distribution(temporary_obstacle_rng);
        obstacle.depth = obstacle.width;
    }
    else if(obstacle.shape == sim::TemporaryObstacleShape::Box)
    {
        obstacle.width = size_distribution(temporary_obstacle_rng);
        obstacle.depth = size_distribution(temporary_obstacle_rng);
    }
    else
    {
        std::uniform_real_distribution<double> wall_width_distribution(
            temporary_obstacle_min_size,
            temporary_obstacle_max_size * 1.6);
        std::uniform_real_distribution<double> wall_depth_distribution(
            temporary_obstacle_min_size * 0.25,
            temporary_obstacle_min_size * 0.55);
        obstacle.width = wall_width_distribution(temporary_obstacle_rng);
        obstacle.depth = wall_depth_distribution(temporary_obstacle_rng);
    }

    /* The circumscribed diameter is a stronger constraint than checking only
       width/depth: no rotated corner can exceed the tracker-target distance. */
    const double diameter = obstacle.bounding_radius() * 2.0;
    if(diameter > maximum_diameter && diameter > 1e-9)
    {
        const double scale = maximum_diameter / diameter;
        obstacle.width *= scale;
        obstacle.depth *= scale;
    }
    return obstacle;
}

bool Simulator::add_temporary_obstacle(
    const sim::TemporaryObstacle& obstacle, const char* source)
{
    const double tracker_target_distance = std::hypot(
        env->target.pose.x - env->tracker.pose.x,
        env->target.pose.y - env->tracker.pose.y);
    const double extent = obstacle.bounding_radius();
    if(extent <= 0. || extent * 2.0 > tracker_target_distance + 1e-9)
        return false;
    if(std::fabs(obstacle.x) + extent
           + temporary_obstacle_passage_width >= map->width * 0.5
       || std::fabs(obstacle.y) + extent
           + temporary_obstacle_passage_width >= map->height * 0.5)
        return false;
    constexpr double safety_margin = 0.10;
    const auto overlaps_robot = [
        &obstacle, extent, safety_margin
    ](const sim::Robot& robot){
        return std::hypot(
            obstacle.x - robot.pose.x,
            obstacle.y - robot.pose.y)
            <= extent + robot.radius + safety_margin;
    };
    if(overlaps_robot(env->tracker) || overlaps_robot(env->target))
        return false;
    /* Use the rotated shape's circumscribed radius, so even its farthest
       corner keeps a robot-passable corridor to every static wall. */
    if(map->static_cost(obstacle.x, obstacle.y)
       <= extent + temporary_obstacle_passage_width
           + temporary_obstacle_static_grid_margin)
        return false;
    double minimum_obstacle_clearance = std::min({
        map->static_cost(obstacle.x, obstacle.y) - extent
            - temporary_obstacle_static_grid_margin,
        map->width * 0.5 - std::fabs(obstacle.x) - extent,
        map->height * 0.5 - std::fabs(obstacle.y) - extent
    });
    for(const auto& existing: temporary_obstacles)
    {
        const double edge_clearance = sim::temporary_obstacle_clearance(
            obstacle, existing);
        if(edge_clearance <= temporary_obstacle_passage_width)
            return false;
        minimum_obstacle_clearance = std::min(
            minimum_obstacle_clearance, edge_clearance);
    }

    temporary_obstacles.push_back(obstacle);
    publish_temporary_obstacles();
    const char* shape =
        obstacle.shape == sim::TemporaryObstacleShape::Cylinder ? "cylinder"
        : obstacle.shape == sim::TemporaryObstacleShape::Box ? "box" : "wall";
    RCLCPP_INFO(
        get_logger(),
        "%s obstacle #%zu: %s at (%.2f, %.2f), size %.2f x %.2f x %.2f m; "
        "minimum clearance %.2f m (required %.2f m); "
        "tracker-target distance %.2f m; "
        "persists until process exit",
        source, temporary_obstacles.size(), shape,
        obstacle.x, obstacle.y, obstacle.width, obstacle.depth,
        obstacle.height, minimum_obstacle_clearance,
        temporary_obstacle_passage_width,
        tracker_target_distance);
    return true;
}

void Simulator::add_automatic_temporary_obstacle()
{
    if(!tracker_has_recent_linear_motion)
        return;

    const double dx = env->target.pose.x - env->tracker.pose.x;
    const double dy = env->target.pose.y - env->tracker.pose.y;
    const double distance = std::hypot(dx, dy);
    if(distance <= map->resolution * 2.0)
    {
        RCLCPP_WARN(
            get_logger(),
            "Skipped automatic obstacle: tracker-target distance %.2f m "
            "is below two map cells", distance);
        return;
    }

    std::uniform_real_distribution<double> interpolation(0.25, 0.75);
    const double jitter_limit = std::min(
        std::min(map->width, map->height) * 0.40,
        temporary_obstacle_placement_jitter);
    std::uniform_real_distribution<double> jitter(-jitter_limit, jitter_limit);
    const double normal_x = -dy / distance;
    const double normal_y = dx / distance;
    const double reference_yaw = std::atan2(dy, dx);
    for(int attempt = 0; attempt < 64; ++attempt)
    {
        const double t = interpolation(temporary_obstacle_rng);
        const double offset = jitter(temporary_obstacle_rng);
        const double x = env->tracker.pose.x + t * dx + offset * normal_x;
        const double y = env->tracker.pose.y + t * dy + offset * normal_y;
        const auto obstacle = make_random_temporary_obstacle(
            x, y, reference_yaw, distance);
        if(add_temporary_obstacle(obstacle, "Automatic"))
            return;
    }
    RCLCPP_WARN(
        get_logger(),
        "Skipped automatic obstacle after 64 attempts: no sufficiently "
        "sparse safe free space between tracker and target (distance %.2f m)",
        distance);
}

void Simulator::step()
{
    /* Control the target */
    planning->publish(*planner->message(world));
    if(external_target_control)
        env->target.control(sim::SE2(0., 0., 0.));
    else
        env->target.control(planner->control());

    /* Robot movement. Only measured translation qualifies for obstacle
       generation. Rotation in place never refreshes the stuck watchdog. */
    const sim::SE2 tracker_pose_before = env->tracker.pose;
    env->step(dt);
    if(external_target_control && have_external_target_pose)
    {
        env->target.pose.x = external_target_pose.position.x;
        env->target.pose.y = external_target_pose.position.y;
        env->target.pose.yaw = normalize(2.0 * std::atan2(
            external_target_pose.orientation.z,
            external_target_pose.orientation.w));
    }
    const double measured_linear_speed = std::hypot(
        env->tracker.pose.x - tracker_pose_before.x,
        env->tracker.pose.y - tracker_pose_before.y) / dt;
    const auto movement_now = std::chrono::steady_clock::now();
    if(measured_linear_speed > temporary_obstacle_linear_motion_threshold)
    {
        last_tracker_translation_at = movement_now;
        if(!tracker_has_recent_linear_motion)
        {
            tracker_has_recent_linear_motion = true;
            if(temporary_obstacle_enabled && temporary_obstacle_auto_add)
                RCLCPP_INFO(
                    get_logger(),
                    "Automatic obstacle generation resumed: tracker linear "
                    "speed %.3f m/s exceeds %.3f m/s",
                    measured_linear_speed,
                    temporary_obstacle_linear_motion_threshold);
        }
    }
    else if(tracker_has_recent_linear_motion)
    {
        const double stationary_duration =
            std::chrono::duration<double>(
                movement_now - last_tracker_translation_at).count();
        if(stationary_duration >= temporary_obstacle_stuck_timeout)
        {
            tracker_has_recent_linear_motion = false;
            if(temporary_obstacle_enabled && temporary_obstacle_auto_add)
                RCLCPP_INFO(
                    get_logger(),
                    "Automatic obstacle generation paused by stuck watchdog: "
                    "no effective linear motion for %.2f s (rotation ignored)",
                    stationary_duration);
        }
    }

    /* Publish the tracker's odometry */
    odometry->publish(env->tracker.odom(tracker, world));

    /* Publish simulator truth on a benchmark-only side channel. Detection
       keeps its original FOV/occlusion semantics and remains the only target
       input consumed by the tracking algorithms. */
    geometry_msgs::msg::PoseStamped target_truth;
    target_truth.header.stamp = now();
    target_truth.header.frame_id = world;
    target_truth.pose.position.x = env->target.pose.x;
    target_truth.pose.position.y = env->target.pose.y;
    target_truth.pose.position.z = env->target.radius;
    target_truth.pose.orientation.z = std::sin(env->target.pose.yaw / 2.0);
    target_truth.pose.orientation.w = std::cos(env->target.pose.yaw / 2.0);
    target_ground_truth->publish(target_truth);

    /* Boardcast transforms */
    transform.header.stamp = now();
    env->target.transform(target, broadcaster.get(), transform);
    env->tracker.transform(tracker, broadcaster.get(), transform);
    
    /* Print infomation */
    RCLCPP_DEBUG(
        get_logger(), 
        "Target: %f, %f \nTracker: %f, %f, %f",
        env->target.pose.x, env->target.pose.y,
        env->tracker.pose.x, env->tracker.pose.y, env->tracker.pose.yaw
    );

    /* Publish trajectories */
    trajectories[0]->publish(*env->target.trajectory(world));
    trajectories[1]->publish(*env->tracker.trajectory(world));

    /* Perception */
    sensor->update(map.get(), env->tracker, env->target);
    pcl::toROSMsg(*sensor->depth(), scan);
    scan.header.frame_id = world;
    depth->publish(scan);

    /* Visualize FoV */
    visualization_msgs::msg::MarkerArray* view = sensor->view(env->lost());
    const auto stamp = now();
    for(auto& marker: view->markers)
        marker.header.stamp = stamp;
    fov->publish(*view);
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Simulator>("simulator"));
    rclcpp::shutdown();
    return 0;
}
