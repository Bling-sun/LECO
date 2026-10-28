/* @author: YueLin */

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"

#include "planner/mpc.hpp"
#include "planner/astar.hpp"

class Tracker: public rclcpp::Node
{
    private:
        /* Hyperparameters */
        double range;

        /* Runtime timing statistics */
        double timing_log_interval = 2.0;
        rclcpp::Time timing_last_log;
        uint64_t timing_ctrl_samples = 0;
        double timing_ctrl_ms = 0.0;
        double timing_mpc_ms = 0.0;
        double timing_publish_ms = 0.0;

        /* States */
        int pointer;
        short tigger;  // Bit: detection, local mapping, localization
        Eigen::VectorXd state;            // x, y, yaw, vel, omega
        Eigen::MatrixXd trajectory;       // (5, n)
        Eigen::Vector3d target, tracker;  // x, y, radius

        /* Objects */
        rclcpp::Time time;
        pcl::PointCloud<pcl::PointXYZ> cloud;
        std::unique_ptr<diff_track::Map> map;
        std::unique_ptr<diff_track::Bezier> bezier;
        std::unique_ptr<diff_track::HybridAStar> planner;
        std::unique_ptr<diff_track::ModelPredictiveControl> mpc;

        /* Timer */
        rclcpp::TimerBase::SharedPtr replan, controller;

        /* Publishers */
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher;
        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd;
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr mapper;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scan_visual;

        /* Subscriber */
        rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr planning;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr localization;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sensor;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr detect;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr raw_detection_metrics;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr truth;
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr observation_status;
        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr metrics_sample;
        uint64_t metrics_sequence = 0;
        bool use_fov_detection = false;
        bool raw_detection_visible = false;
        bool dynamic_accel_by_visibility = false;
        double visible_max_accel = 3.0;
        double lost_max_accel = 3.0;
        bool truth_fallback_active = false;
        bool have_truth = false;
        geometry_msgs::msg::PoseStamped latest_truth;
        rclcpp::Time detection_lost_since;
        double truth_fallback_after = 3.0;

        /* Message */
        nav_msgs::msg::Path message;
        geometry_msgs::msg::Twist twist;
    
    private:
        void ctrl();
        void callback(const nav_msgs::msg::Path::SharedPtr);
        void log_timing_if_needed(const rclcpp::Time& now);

    public:
        Tracker(std::string);
};
