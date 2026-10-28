#pragma once

#include <cmath>

#include <Eigen/Eigen>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/header.hpp"

namespace diff_track
{
    inline void makeLocalVisualization(
        const pcl::PointCloud<pcl::PointXYZ>& scan,
        const Eigen::Vector2d& center,
        double radius,
        const std_msgs::msg::Header& header,
        sensor_msgs::msg::PointCloud2& scan_message)
    {
        /* Retain real MID-360 returns only. Free and unknown cells are
         * deliberately absent, so RViz does not render a filled grey disc. */
        pcl::PointCloud<pcl::PointXYZ> local_scan;
        local_scan.points.reserve(scan.size());
        const double radius2 = radius * radius;
        for(const auto& hit: scan)
            if(std::isfinite(hit.x) && std::isfinite(hit.y)
               && (Eigen::Vector2d(hit.x, hit.y) - center).squaredNorm()
                    <= radius2)
            {
                pcl::PointXYZ visible_hit = hit;
                visible_hit.z = 0.06f;
                local_scan.push_back(visible_hit);
            }
        pcl::toROSMsg(local_scan, scan_message);
        scan_message.header = header;
    }
}
