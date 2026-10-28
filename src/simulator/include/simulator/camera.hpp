/* @author: YueLin */

#include <random>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include "visualization_msgs/msg/marker_array.hpp"

#include "simulator/map.hpp"
#include "simulator/robot.hpp"

namespace sim
{
    class Camera
    {
        private:
            double horizontal_angle, vertical_angle, range, height;
            double lidar_range, lidar_angular_resolution;
            visualization_msgs::msg::MarkerArray fov[2];
            std::default_random_engine random;
            pcl::PointCloud<pcl::PointXYZ> cloud;
            std::normal_distribution<double> gaussian;
        
        public:
            Camera(std::string, Robot&, double, double, double, double,
                   double, double);
        
        public:
            void update(RandomMap*, Robot&, Robot&);
            visualization_msgs::msg::MarkerArray* view(bool lost)
            {
                return fov + static_cast<int>(lost);
            }
            pcl::PointCloud<pcl::PointXYZ>* depth() {return &cloud;}
        
        private:
            double noise() {return gaussian(random);}
    };
}
