/* @author: YueLin */

#include <memory>
#include <random>
#include <chrono>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#include "simulator/env.hpp"
#include "simulator/camera.hpp"
#include "simulator/planner.hpp"

class Simulator: public rclcpp::Node
{
    private:
        /* Hyperparameters */
        double dt, horizontal_angle, vertical_angle, camera_height, range, vm;
        double lidar_range, lidar_angular_resolution;
        std::string world, tracker, target;

        /* Objects */
        std::unique_ptr<sim::Camera> sensor;
        std::unique_ptr<sim::RandomMap> map;
        std::unique_ptr<sim::Environment> env;
        std::unique_ptr<sim::Planner> planner;

        /* ROS */
        std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster;
        rclcpp::TimerBase::SharedPtr timer, detection, move;
        rclcpp::TimerBase::SharedPtr automatic_obstacle_timer;

        /* Publishers */
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr fov;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr planning;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry;
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr trajectories[2];
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr mapper;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr depth;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr map3d;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr detector;
        rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_ground_truth;

        /* Subscribers */
        rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr controller;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr goal;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr external_target;
        rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr rnd;
        rclcpp::Subscription<geometry_msgs::msg::PointStamped>::SharedPtr temporary_obstacle_click;

        /* Messages */
        bool moving = false;
        bool has_target_goal = false;
        bool external_target_control = false;
        bool have_external_target_pose = false;
        sensor_msgs::msg::PointCloud2 scan;
        sensor_msgs::msg::PointCloud2 static_map3d_cloud;
        geometry_msgs::msg::TransformStamped transform;
        geometry_msgs::msg::Pose target_goal;
        geometry_msgs::msg::Pose external_target_pose;

        std::vector<sim::TemporaryObstacle> temporary_obstacles;
        std::mt19937_64 temporary_obstacle_rng;
        double temporary_obstacle_interval = 5.0;
        double temporary_obstacle_min_size = 0.60;
        double temporary_obstacle_max_size = 1.20;
        double temporary_obstacle_height = 4.0;
        double temporary_obstacle_placement_jitter = 1.50;
        double temporary_obstacle_minimum_spacing = 0.50;
        double temporary_obstacle_passage_width = 0.50;
        double temporary_obstacle_static_grid_margin = 0.0;
        double temporary_obstacle_linear_motion_threshold = 0.02;
        double temporary_obstacle_stuck_timeout = 3.0;
        bool temporary_obstacle_enabled = false;
        bool temporary_obstacle_auto_add = false;
        bool tracker_has_recent_linear_motion = false;
        std::chrono::steady_clock::time_point last_tracker_translation_at;
    
    private:
        void step();
        void publish_temporary_obstacles();
        void add_automatic_temporary_obstacle();
        bool add_temporary_obstacle(const sim::TemporaryObstacle& obstacle,
                                    const char* source);
        sim::TemporaryObstacle make_random_temporary_obstacle(
            double x, double y, double reference_yaw,
            double maximum_diameter);

    public:
        Simulator(std::string);
};
