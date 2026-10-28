/* @author: YueLin */

#include <algorithm>
#include <cmath>
#include <vector>

#include "geometry_msgs/msg/point.hpp"
#include "visualization_msgs/msg/marker.hpp"

#include "simulator/camera.hpp"

namespace sim
{
    Camera::Camera(std::string robot, Robot& r,
                   double horizontal_alpha, double vertical_alpha,
                   double distance, double camera_height,
                   double lidar_distance, double lidar_resolution):
        horizontal_angle(horizontal_alpha), vertical_angle(vertical_alpha),
        range(distance), height(camera_height),
        lidar_range(lidar_distance), lidar_angular_resolution(lidar_resolution)
    {
        const double origin_x = r.radius * 2;

        geometry_msgs::msg::Point origin;
        origin.x = origin_x;
        origin.z = height;

        // Build a pinhole FOV capped by a constant radial range. A flat far
        // plane would make the corner rays longer than `range`, which does not
        // match the simulator's radial ray casting and visibility checks.
        constexpr int horizontal_segments = 12;
        constexpr int vertical_segments = 8;
        constexpr int row_size = horizontal_segments + 1;
        std::vector<geometry_msgs::msg::Point> cap;
        cap.reserve((horizontal_segments + 1) * (vertical_segments + 1));
        for(int v = 0; v <= vertical_segments; ++v)
        {
            const double vertical = -vertical_angle
                + 2 * vertical_angle * v / vertical_segments;
            for(int h = 0; h <= horizontal_segments; ++h)
            {
                const double horizontal = -horizontal_angle
                    + 2 * horizontal_angle * h / horizontal_segments;
                const double y = std::tan(horizontal);
                const double z = std::tan(vertical);
                const double scale = range / std::sqrt(1 + y * y + z * z);
                geometry_msgs::msg::Point point;
                point.x = origin_x + scale;
                point.y = scale * y;
                point.z = height + scale * z;
                cap.push_back(point);
            }
        }
        const auto at = [&cap](int h, int v) -> const geometry_msgs::msg::Point&
        {
            return cap[v * row_size + h];
        };

        for(int lost = 0; lost < 2; ++lost)
        {
            visualization_msgs::msg::Marker fill;
            fill.header.frame_id = robot;
            fill.ns = "d435i_fov_fill";
            fill.id = 0;
            fill.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
            fill.action = visualization_msgs::msg::Marker::ADD;
            fill.pose.orientation.w = 1.0;
            fill.scale.x = 1.0;
            fill.scale.y = 1.0;
            fill.scale.z = 1.0;
            fill.frame_locked = true;
            fill.color.r = lost ? 1.00f : 0.11f;
            fill.color.g = lost ? 0.30f : 0.44f;
            fill.color.b = lost ? 0.00f : 0.85f;
            fill.color.a = 0.20f;

            // Tessellated spherical cap at the configured radial range.
            for(int v = 0; v < vertical_segments; ++v)
                for(int h = 0; h < horizontal_segments; ++h)
                {
                    fill.points.insert(fill.points.end(), {
                        at(h, v), at(h + 1, v), at(h + 1, v + 1),
                        at(h, v), at(h + 1, v + 1), at(h, v + 1)
                    });
                }

            // Close the four angular sides back to the camera origin.
            for(int h = 0; h < horizontal_segments; ++h)
            {
                fill.points.insert(fill.points.end(), {
                    origin, at(h, 0), at(h + 1, 0),
                    origin, at(h + 1, vertical_segments),
                    at(h, vertical_segments)
                });
            }
            for(int v = 0; v < vertical_segments; ++v)
            {
                fill.points.insert(fill.points.end(), {
                    origin, at(0, v + 1), at(0, v),
                    origin, at(horizontal_segments, v),
                    at(horizontal_segments, v + 1)
                });
            }

            visualization_msgs::msg::Marker outline;
            outline.header.frame_id = robot;
            outline.ns = "d435i_fov_outline";
            outline.id = 1;
            outline.type = visualization_msgs::msg::Marker::LINE_LIST;
            outline.action = visualization_msgs::msg::Marker::ADD;
            outline.pose.orientation.w = 1.0;
            outline.scale.x = 0.07;
            outline.frame_locked = true;
            outline.color.r = lost ? 1.00f : 0.00f;
            outline.color.g = lost ? 0.22f : 0.72f;
            outline.color.b = lost ? 0.00f : 1.00f;
            outline.color.a = 0.90f;
            // Four corner rays.
            for(const auto& corner: {
                at(0, 0), at(horizontal_segments, 0),
                at(horizontal_segments, vertical_segments),
                at(0, vertical_segments)})
            {
                outline.points.push_back(origin);
                outline.points.push_back(corner);
            }

            // Outer boundary plus horizontal/vertical center arcs make the
            // 3D curvature and the 87 x 58 degree opening easy to read.
            for(int h = 0; h < horizontal_segments; ++h)
                for(const int v: {0, vertical_segments, vertical_segments / 2})
                {
                    outline.points.push_back(at(h, v));
                    outline.points.push_back(at(h + 1, v));
                }
            for(int v = 0; v < vertical_segments; ++v)
                for(const int h: {0, horizontal_segments, horizontal_segments / 2})
                {
                    outline.points.push_back(at(h, v));
                    outline.points.push_back(at(h, v + 1));
                }
            outline.points.push_back(origin);
            outline.points.push_back(
                at(horizontal_segments / 2, vertical_segments / 2));

            fov[lost].markers.push_back(fill);
            fov[lost].markers.push_back(outline);
        }

        gaussian = std::normal_distribution<double>(0, 0.01);
    }

    void Camera::update(RandomMap* map, Robot& robot, Robot& obstacle)
    {
        cloud.clear();
        const double r2 = obstacle.radius * obstacle.radius;
        const double w = map->width / 2, h = map->height / 2;
        // /tracker/scan models the MID-360 horizontal slice. Target detection
        // remains a separate limited-FOV camera operation in Environment.
        const int rays = std::max(
            1, static_cast<int>(std::ceil(2 * PI / lidar_angular_resolution)));
        for(int ray = 0; ray < rays; ++ray)
        {
            const double theta = -PI + 2 * PI * ray / rays;
            double d = map->resolution;
            double yaw = robot.pose.yaw + theta;
            double sin = std::sin(yaw), cos = std::cos(yaw);
            while(d <= lidar_range)
            {
                double x = robot.pose.x + d * cos;
                double y = robot.pose.y + d * sin;
                if(map->cost(x, y) <= 0 ||
                   std::fabs(x) >= w || std::fabs(y) >= h ||  
                   hypot2(x - obstacle.pose.x, y - obstacle.pose.y) <= r2)
                {
                    cloud.emplace_back(x + noise(), y + noise(), 0); break;
                }
                d += map->resolution;
            }
        }
    }
}
