/* @author: YueLin (neural variant)
 *
 * Copy of planning.hpp for the tracker_neural node. Identical to the original
 * except: when the target is visible (mpc->visibility()==true), the control is
 * produced by a neural policy (ONNX) instead of the MPC. The lost/reacquire
 * path, mapping, and all publishers are unchanged.
 */

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "nav_msgs/msg/path.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "std_msgs/msg/string.hpp"

#include "planner/mpc.hpp"
#include "planner/astar.hpp"
#include "planner/neural_policy.hpp"

class Tracker: public rclcpp::Node
{
    private:
        /* Hyperparameters */
        double range, observation_distance;
        double control_dt, tracker_max_vel, tracker_max_acc;

        /* Neural-policy safety fallback (v40b) */
        int neural_safety_steps;      // near-horizon steps checked for collision
        double neural_fallback_lock;  // seconds to stay on classical MPC after a hit
        rclcpp::Time neural_lock_until;

        /* Reject obviously divergent neural commands at the stationary
           tracking equilibrium and temporarily hand control to MPC. */
        bool stationary_guard;
        bool stationary_hold = false;
        bool stationary_mpc_active = false;
        bool have_detection_sample = false;
        bool detection_visible = false;
        bool use_fov_detection = true;
        bool truth_fallback_active = false;
        double truth_fallback_after = 3.0;
        rclcpp::Time detection_lost_since;
        double stationary_guard_distance_tolerance;
        double stationary_guard_bearing_tolerance;
        double stationary_guard_target_speed;
        double stationary_guard_reverse_speed;
        double stationary_guard_max_omega;
        double detected_target_speed = 0.0;
        Eigen::Vector2d last_detection_position = Eigen::Vector2d::Zero();
        rclcpp::Time last_detection_time;

        /* Keep a visible moving target away from the camera boundary. */
        bool fov_guard;
        double fov_guard_min_bearing;
        double fov_guard_kp;
        double fov_guard_max_omega;

        /* Recover closing speed before a visible target reaches the camera's
           range boundary. The boosted command is accepted only when its
           short executed arc is clear in the rolling local ESDF. */
        bool range_guard;
        bool range_guard_active = false;
        double range_guard_start;
        double range_guard_release;
        double range_guard_max_bearing;
        double range_guard_min_speed;
        double range_guard_kp;
        double range_guard_horizon;
        double range_guard_clearance;

        /* Runtime timing statistics */
        double timing_log_interval = 2.0;
        rclcpp::Time timing_last_log;
        uint64_t timing_ctrl_samples = 0;
        uint64_t metrics_sequence = 0;
        uint64_t timing_neural_samples = 0;
        uint64_t timing_fallback_samples = 0;
        double timing_ctrl_ms = 0.0;
        double timing_target_ms = 0.0;
        double timing_infer_ms = 0.0;
        double timing_infer_esdf_ms = 0.0;
        double timing_infer_target_ms = 0.0;
        double timing_infer_visibility_ms = 0.0;
        double timing_infer_tensor_ms = 0.0;
        double timing_infer_ort_ms = 0.0;
        double timing_infer_output_ms = 0.0;
        double timing_safety_ms = 0.0;
        double timing_neural_publish_ms = 0.0;
        double timing_mpc_ms = 0.0;
        double timing_fallback_publish_ms = 0.0;

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
        std::unique_ptr<diff_track::NeuralPolicy> neural;

        /* Timer */
        rclcpp::TimerBase::SharedPtr replan, controller;

        /* Publishers */
        rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr publisher;
        rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd;
        rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr planning_time;
        rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr control_branch;
        rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr metrics_sample;
        rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr mapper;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr scan_visual;
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr observation_status;

        /* Subscriber */
        rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr planning;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr localization;
        rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sensor;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr detect;
        rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr truth_detect;

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
