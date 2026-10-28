/* @author: YueLin */

#include <stdexcept>

#include "planner/map.hpp"

namespace diff_track
{
    Map::Map(double w, double h, double r, double radius):
        valid_radius(radius), resolution(r), width(w), height(h)
    {
        offset.width = std::round(w / (r * 2));
        offset.height = std::round(h / (r * 2));
        for(int m = 0; m < 3; m++)
            map[m] = cv::Mat::zeros(offset * 2, CV_8UC1);
        map[0] = ~*map;

        if(valid_radius > std::min(w, h) * 0.5)
            throw std::invalid_argument(
                "circular map radius exceeds square storage");
        if(valid_radius > 0.)
        {
            valid_mask = cv::Mat::zeros(map->size(), CV_8UC1);
            for(int y = 0; y < valid_mask.rows; ++y)
                for(int x = 0; x < valid_mask.cols; ++x)
                    if(validCell(cv::Point(x, y)))
                        valid_mask.at<uchar>(y, x) = 0xFF;
            clearOutsideWindow();
        }

        /* Initialize message */
        msg.info.resolution = r;
        msg.info.width = map->cols;
        msg.info.height = map->rows;
        msg.info.origin.position.x = center.x() - w / 2;
        msg.info.origin.position.y = center.y() - h / 2;
        msg.data.resize(map->cols * map->rows);
    }

    bool Map::validCell(const cv::Point& point) const
    {
        if(point.x < 0 || point.y < 0
           || point.x >= map->cols || point.y >= map->rows)
            return false;
        if(valid_radius <= 0.)
            return true;
        const double x = (point.x - offset.width) * resolution;
        const double y = (point.y - offset.height) * resolution;
        return x * x + y * y <= valid_radius * valid_radius;
    }

    void Map::clearOutsideWindow()
    {
        if(valid_mask.empty())
            return;
        cv::Mat outside;
        cv::bitwise_not(valid_mask, outside);
        map[0].setTo(0xFF, outside);  // unknown remains traversable
        map[1].setTo(0, outside);     // but is never published as observed
        map[2].setTo(0, outside);
    }

    void Map::shift(cv::Mat& source, int dx, int dy, uchar fill)
    {
        cv::Mat shifted(source.size(), source.type(), cv::Scalar(fill));
        if(std::abs(dx) < source.cols && std::abs(dy) < source.rows)
        {
            const int sx = std::max(0, -dx);
            const int sy = std::max(0, -dy);
            const int tx = std::max(0, dx);
            const int ty = std::max(0, dy);
            const int cols = source.cols - std::abs(dx);
            const int rows = source.rows - std::abs(dy);
            source(cv::Rect(sx, sy, cols, rows)).copyTo(
                shifted(cv::Rect(tx, ty, cols, rows))
            );
        }
        source = shifted;
    }

    void Map::recenter(const Eigen::Vector2d& position)
    {
        const Eigen::Vector2d next =
            (position / resolution).array().round().matrix() * resolution;
        const int dx = std::lround((center.x() - next.x()) / resolution);
        const int dy = std::lround((center.y() - next.y()) / resolution);
        if(dx || dy)
        {
            // Unknown space is traversable: occupancy is initialized free,
            // while map[1] separately records whether a cell was observed.
            shift(map[0], dx, dy, 0xFF);
            shift(map[1], dx, dy, 0);
            center = next;
        }
        msg.info.origin.position.x = center.x() - width * 0.5;
        msg.info.origin.position.y = center.y() - height * 0.5;
        clearOutsideWindow();
    }

    void Map::updateEsdf()
    {
        if(cv::countNonZero(*map) == map->rows * map->cols)
        {
            // distanceTransform would normally allocate these outputs. The
            // all-free fast path can also be the very first scan (for example
            // in a large JSON map), so allocate explicitly before publishing.
            sdf[0] = cv::Mat(
                map->size(), CV_32F,
                cv::Scalar(static_cast<float>(std::hypot(width, height))));
            sdf[1] = cv::Mat::zeros(map->size(), CV_32F);
        }
        else
        {
            cv::distanceTransform(
                *map, *sdf, cv::DIST_L2, cv::DIST_MASK_PRECISE
            );
            cv::distanceTransform(
                ~*map, *(sdf + 1), cv::DIST_L2, cv::DIST_MASK_PRECISE
            );
            for(int y = 0; y < map->rows; y++)
                for(int x = 0; x < map->cols; x++)
                    if(!map->at<uchar>(y, x))
                        sdf[0].at<float>(y, x) = -sdf[1].at<float>(y, x);
            *sdf *= resolution;
        }

        /* Do not inject the circular window edge into the ESDF. It is a local
         * planning/query limit, not a physical wall. contains()/at() enforce
         * the window while the distance field represents real lidar obstacles
         * only, avoiding artificial repulsion near the 6 m boundary. */
    }

    void Map::update(nav_msgs::msg::OccupancyGrid::SharedPtr grid)
    {
        /* Import into the local canvas without assuming equal dimensions. */
        center.x() = grid->info.origin.position.x
                   + grid->info.width * grid->info.resolution * 0.5;
        center.y() = grid->info.origin.position.y
                   + grid->info.height * grid->info.resolution * 0.5;
        for(int y = 0; y < map->rows; y++)
        {
            for(int x = 0; x < map->cols; x++)
            {
                const Eigen::Vector2d world = decode(cv::Point(x, y));
                const int gx = std::lround(
                    (world.x() - grid->info.origin.position.x)
                    / grid->info.resolution);
                const int gy = std::lround(
                    (world.y() - grid->info.origin.position.y)
                    / grid->info.resolution);
                if(gx >= 0 && gy >= 0
                   && gx < static_cast<int>(grid->info.width)
                   && gy < static_cast<int>(grid->info.height))
                {
                    const int8_t value = grid->data[
                        gy * grid->info.width + gx];
                    map[0].at<uchar>(y, x) = value > 0? 0: 0xFF;
                    map[1].at<uchar>(y, x) = value >= 0? 0xFF: 0;
                }
                else
                {
                    map[0].at<uchar>(y, x) = 0xFF;
                    map[1].at<uchar>(y, x) = 0;
                }
            }
        }
        msg.info.origin.position.x = center.x() - width * 0.5;
        msg.info.origin.position.y = center.y() - height * 0.5;
        clearOutsideWindow();
        updateEsdf();
    }

    void Map::update(const Eigen::Vector3d& self,
                     const Eigen::Vector3d& filter, 
                     const pcl::PointCloud<pcl::PointXYZ>& scan)
    {
        recenter(self.head<2>());
        map[2] *= 0;
        const cv::Point robot = encode(self.head<2>());

        /* Use the actual lidar returns. Each point contributes exactly one
         * measured ray; no camera-FOV sector is synthesized here. */
        for(const pcl::PointXYZ& p: scan.points)
        {
            if(!std::isfinite(p.x) || !std::isfinite(p.y))
                continue;
            const Eigen::Vector2d hit(p.x, p.y);
            const bool hit_inside = contains(hit, resolution);
            const Eigen::Vector2d endpoint = clamp(hit, resolution);
            cv::LineIterator ray = line(robot, encode(endpoint));
            for(int n = ray.count; n; --n)
            {
                const cv::Point cell = ray++.pos();
                map[0].at<uchar>(cell) = 0xFF;
                map[1].at<uchar>(cell) = 0xFF;
            }
            if(hit_inside)
            {
                const cv::Point cell = encode(hit);
                cv::circle(map[2], cell, 2, 0xFF, -1);
                cv::circle(map[1], cell, 2, 0xFF, -1);
            }
        }
        map[0].setTo(0, map[2]);

        /* Filter */
        if(filter.z() > 0 && contains(filter.head<2>()))
        {
            const cv::Point point = encode(filter.head<2>());
            cv::circle(*map, point,
                       std::round(2 * filter.z() / resolution), 0xFF, -1);
        }

        clearOutsideWindow();
        updateEsdf();
    }

    nav_msgs::msg::OccupancyGrid* Map::message(const std::string& frame, 
                                               double radius)
    {
        msg.header.frame_id = frame;
        for(int y = 0; y < map->rows; y++)
        {
            int z = y * map->cols;
            for(int x = 0; x < map->cols; x++)
                if(validCell(cv::Point(x, y))
                   && map[1].at<uchar>(y, x))
                {
                    float distance = sdf->at<float>(y, x);
                    msg.data[x + z] = radius >= distance? (
                        0 < distance? (radius - distance) / radius * 100: 100
                    ): 0;
                }
                else
                    msg.data[x + z] = -1;
        }
        return &msg;
    }
}
