/* @author: YueLin */

#pragma once

#include <cmath>

#include <Eigen/Eigen>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <opencv2/opencv.hpp>

#include "nav_msgs/msg/occupancy_grid.hpp"

const double PI = std::acos(-1);
const double deg2rad = PI / 180;

/* Clip rad to (-PI, PI] */
inline double normalize(double rad)
{
    if(rad > PI || rad <= -PI)
        return 2 * std::floor(0.5 - rad / (2 * PI)) * PI + rad;
    return rad;
}

namespace diff_track
{
    class Map
    {
        private:
            cv::Mat map[3], sdf[2];
            cv::Mat valid_mask;
            nav_msgs::msg::OccupancyGrid msg;
            Eigen::Vector2d center = Eigen::Vector2d::Zero();
            double valid_radius;

            void recenter(const Eigen::Vector2d&);
            void updateEsdf();
            void shift(cv::Mat&, int, int, uchar);
            void clearOutsideWindow();
            bool validCell(const cv::Point&) const;
        
        public:
            cv::Size offset;
            double resolution;
            double width, height;
        
        public:
            Map(double, double, double, double = -1.0);

            void update(nav_msgs::msg::OccupancyGrid::SharedPtr);

            void update(const Eigen::Vector3d&,
                        const Eigen::Vector3d&,
                        const pcl::PointCloud<pcl::PointXYZ>&);

            nav_msgs::msg::OccupancyGrid* message(const std::string&, double);

            cv::LineIterator line(const cv::Point& p1, const cv::Point& p2) 
            const
            {
                return cv::LineIterator(*map, p1, p2);
            }

            cv::Point decode(int code) const
            {
                cv::Point point;
                point.y = code / map->cols;
                point.x = code - map->cols * point.y;
                return point;
            }

            int encode(const cv::Point& point) const
            {
                return map->cols * point.y + point.x;
            }
            
            cv::Point encode(const Eigen::Vector2d& point) const
            {
                return cv::Point(
                    std::max(std::min(
                        map->cols - 1.,
                        std::round((point.x() - center.x()) / resolution)
                            + offset.width
                    ), 0.), 
                    std::max(std::min(
                        map->rows - 1.,
                        std::round((point.y() - center.y()) / resolution)
                            + offset.height
                    ), 0.)
                );
            }

            cv::Point2d grid(const Eigen::Vector2d& point) const
            {
                return cv::Point2d(
                    (point.x() - center.x()) / resolution + offset.width,
                    (point.y() - center.y()) / resolution + offset.height
                );
            }

            Eigen::Vector2d decode(const cv::Point& point) const
            {
                return Eigen::Vector2d(
                    center.x() + resolution * (point.x - offset.width),
                    center.y() + resolution * (point.y - offset.height)
                );
            }

            bool contains(const Eigen::Vector2d& point,
                          double margin = 0.) const
            {
                const double hx = width * 0.5 - margin;
                const double hy = height * 0.5 - margin;
                const bool inside_storage = hx >= 0. && hy >= 0.
                    && std::fabs(point.x() - center.x()) <= hx
                    && std::fabs(point.y() - center.y()) <= hy;
                if(!inside_storage || valid_radius <= 0.)
                    return inside_storage;
                const double radius = valid_radius - margin;
                return radius >= 0.
                    && (point - center).squaredNorm() <= radius * radius;
            }

            Eigen::Vector2d clamp(const Eigen::Vector2d& point,
                                  double margin = 0.) const
            {
                if(valid_radius > 0.)
                {
                    const double radius = std::max(0., valid_radius - margin);
                    const Eigen::Vector2d delta = point - center;
                    const double distance = delta.norm();
                    if(distance > radius && distance > 0.)
                        return center + delta * (radius / distance);
                    return point;
                }
                const double hx = std::max(0., width * 0.5 - margin);
                const double hy = std::max(0., height * 0.5 - margin);
                return Eigen::Vector2d(
                    std::max(center.x() - hx,
                             std::min(center.x() + hx, point.x())),
                    std::max(center.y() - hy,
                             std::min(center.y() + hy, point.y()))
                );
            }

            const Eigen::Vector2d& centerWorld() const
            {
                return center;
            }

            float at(const cv::Point& point) const
            {
                return validCell(point)? sdf->at<float>(point): 0.f;
            }

            float at(const Eigen::Vector2d& point) const
            {
                return contains(point)? sdf->at<float>(encode(point)): 0.f;
            }

            /* Read-only access to the full signed ESDF (meters, obstacle < 0).
               Used by the neural policy to warpAffine a robot-centered patch,
               matching the Python extract_local_esdf pipeline. */
            const cv::Mat& sdfMat() const
            {
                return sdf[0];
            }

            bool known(const Eigen::Vector2d& point) const
            {
                return contains(point)
                    && (map + 1)->at<uchar>(encode(point));
            }

            double windowRadius() const
            {
                return valid_radius;
            }

            float esdf(const Eigen::Vector2d& point, Eigen::Vector2d& gradient)
            const 
            {
                if(!contains(point))
                {
                    gradient.setZero(); return 0;
                }
                double x = (point.x() - center.x()) / resolution
                         + offset.width;
                double y = (point.y() - center.y()) / resolution
                         + offset.height;
                
                /* Linear interpolation */
                int x1 = x, y1 = y;
                int x2 = x1 + 1, y2 = y1 + 1;
                double u = x - x1, v = y - y1, u_ = x2 - x, v_ = y2 - y;
                
                /* Boundary check */
                if(std::min(x1, y1) < 0 || x2 >= sdf->cols || y2 >= sdf->rows)
                {
                    gradient.setZero(); return 0;
                }
                
                /* Linear interpolation for gradient */
                gradient.x() = v * sdf->at<float>(y2, x2)
                             - v * sdf->at<float>(y2, x1)
                             + v_ * sdf->at<float>(y1, x2)
                             - v_ * sdf->at<float>(y1, x1);
                gradient.y() = u * sdf->at<float>(y2, x2)
                             - u * sdf->at<float>(y1, x2)
                             + u_ * sdf->at<float>(y2, x1)
                             - u_ * sdf->at<float>(y1, x1);

                /* Linear interpolation for ESDF value */
                return u * v * sdf->at<float>(y2, x2)
                     + u * v_ * sdf->at<float>(y1, x2)
                     + u_ * v * sdf->at<float>(y2, x1)
                     + u_ * v_ * sdf->at<float>(y1, x1);
            }
    };
}
