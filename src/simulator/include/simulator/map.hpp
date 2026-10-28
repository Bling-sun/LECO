/* @author: YueLin */

#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "geometry_msgs/msg/pose.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "simulator/temporary_obstacle.hpp"

namespace sim
{
    struct CanopyVisual
    {
        double x, y, z;
        double radius_x, radius_y, radius_z;
        double yaw;
        std::uint8_t red, green, blue;
    };

    struct ArchVisual
    {
        double x, y;
        double width, depth, height;
        double thickness, clearance;
        double yaw;
    };

    class RandomMap
    {
        private:
            cv::Mat data;
            cv::Mat elevation;
            cv::Mat static_occupancy_;
            cv::Mat static_elevation_;
            cv::Mat static_distance_;
            cv::Size offset;
            nav_msgs::msg::OccupancyGrid msg;
            std::vector<CanopyVisual> canopy_visuals_;
            std::vector<ArchVisual> arch_visuals_;

            void rebuild_distance_field(const cv::Mat& occupancy);
        
        public:
            double width, height, resolution;
            
        public:
            RandomMap(int, int, const std::string&,
                      double, double, double, double, 
                      double, double, double, double,
                      double, double, const std::string& = "");

            cv::Point encode(double x, double y) const
            {
                return cv::Point(
                    std::max(std::min(
                        data.cols - 1., 
                        std::round(x / resolution) + offset.width
                    ), 0.), 
                    std::max(std::min(
                        data.rows - 1., 
                        std::round(y / resolution) + offset.height
                    ), 0.)
                );
            }

            int rows() const {return data.rows;}
            int cols() const {return data.cols;}
            float index(int x, int y) const {return data.at<float>(y, x);}
            float index(cv::Point point) const {return data.at<float>(point);}
            float cost(double x, double y) const {return index(encode(x, y));}
            float static_cost(double x, double y) const
            {
                const cv::Point point = encode(x, y);
                return static_distance_.empty()
                    ? data.at<float>(point)
                    : static_distance_.at<float>(point);
            }
            float obstacle_height(int x, int y) const
            {
                return elevation.at<float>(y, x);
            }
            float obstacle_height(double x, double y) const
            {
                const cv::Point point = encode(x, y);
                return obstacle_height(point.x, point.y);
            }

            std::pair<double, double> nearest_free(
                double x, double y, double clearance) const;

            void set_temporary_obstacles(
                const std::vector<TemporaryObstacle>& obstacles);

            const std::vector<CanopyVisual>& canopy_visuals() const
            {
                return canopy_visuals_;
            }

            const std::vector<ArchVisual>& arch_visuals() const
            {
                return arch_visuals_;
            }

            cv::LineIterator line(const cv::Point& p1, const cv::Point& p2) 
            const
            {
                return cv::LineIterator(data, p1, p2);
            }

            nav_msgs::msg::OccupancyGrid* message(const std::string& frame)
            {
                msg.header.frame_id = frame; return &msg;
            }

            geometry_msgs::msg::Pose goal() const;
    };
}
