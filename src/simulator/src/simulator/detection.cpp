/* @author: YueLin */

#include "simulator/env.hpp"

namespace sim
{
    Environment::Environment(std::string name,
                             RandomMap* rm,
                             double target_x,
                             double target_y,
                             double target_z,
                             double target_r,
                             double tracker_x,
                             double tracker_y,
                             double tracker_z,
                             double tracker_r): 
        target(target_x, target_y, target_z, target_r),
        tracker(tracker_x, tracker_y, tracker_z, tracker_r), map(rm)
    {
        detection.header.frame_id = name;
    }
    
    geometry_msgs::msg::PoseStamped* Environment::detect(
        double horizontal_fov, double vertical_fov, double camera_height,
        double range, const rclcpp::Logger& logger)
    {
        /* Check whether the target intersects the 3D camera frustum. The
           target radius supplies a conservative angular extent vertically;
           horizontal semantics remain center-based as in the 2D simulator. */
        double distance = std::hypot(
            target.pose.x - tracker.pose.x, 
            target.pose.y - tracker.pose.y
        );
        double angle = std::fabs(normalize(std::atan2(
            target.pose.y - tracker.pose.y,
            target.pose.x - tracker.pose.x
        ) - tracker.pose.yaw));
        const double target_center_z = target.radius;
        const double height_delta = target_center_z - camera_height;
        const double elevation = std::atan2(height_delta, distance);
        const double distance_3d = std::hypot(distance, height_delta);
        const double angular_radius = distance_3d > target.radius
            ? std::asin(std::min(1.0, target.radius / distance_3d))
            : PI / 2;
        const bool vertical_capture =
            std::fabs(elevation) <= vertical_fov + angular_radius;
        bool capture = distance <= range && angle <= horizontal_fov
                    && vertical_capture;
        RCLCPP_DEBUG(
            logger,
            "Distance: %f\nHorizontal angle: %f\nElevation: %f",
            distance, angle, elevation);
        
        /* Check occlusion */
        if(capture)
        {
            cv::LineIterator line = map->line(
                map->encode(target.pose.x, target.pose.y),
                map->encode(tracker.pose.x, tracker.pose.y)
            );
            for(int _ = line.count; capture && _; --_)
                capture &= map->index(line++.pos()) > 0;
            RCLCPP_DEBUG(logger, "Occlusion: %d", !capture);
        }

        /* Construct message */
        if(capture)
        {
            detection.pose.orientation.w = 1;
            detection.pose.position.x = target.pose.x;
            detection.pose.position.y = target.pose.y;
            detection.pose.position.z = target.radius;
            RCLCPP_DEBUG(logger, "Target detected");
        }
        else
        {
            detection.pose.orientation.w = -1;
            RCLCPP_DEBUG(logger, "Target lost");
        }
        return &detection;
    }
}
