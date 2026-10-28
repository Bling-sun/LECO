#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace sim
{
    enum class TemporaryObstacleShape
    {
        Cylinder,
        Box,
        Wall
    };

    struct TemporaryObstacle
    {
        TemporaryObstacleShape shape;
        double x, y;
        double width, depth, height;
        double yaw;

        double bounding_radius() const
        {
            return shape == TemporaryObstacleShape::Cylinder
                ? width * 0.5 : std::hypot(width, depth) * 0.5;
        }
    };

    namespace temporary_obstacle_geometry
    {
        struct Point
        {
            double x;
            double y;
        };

        inline Point subtract(const Point& lhs, const Point& rhs)
        {
            return {lhs.x - rhs.x, lhs.y - rhs.y};
        }

        inline double dot(const Point& lhs, const Point& rhs)
        {
            return lhs.x * rhs.x + lhs.y * rhs.y;
        }

        inline double cross(const Point& lhs, const Point& rhs)
        {
            return lhs.x * rhs.y - lhs.y * rhs.x;
        }

        inline std::array<Point, 4> rectangle_corners(
            const TemporaryObstacle& obstacle)
        {
            const double half_width = obstacle.width * 0.5;
            const double half_depth = obstacle.depth * 0.5;
            const double cosine = std::cos(obstacle.yaw);
            const double sine = std::sin(obstacle.yaw);
            const std::array<Point, 4> local{{
                {-half_width, -half_depth},
                { half_width, -half_depth},
                { half_width,  half_depth},
                {-half_width,  half_depth},
            }};
            std::array<Point, 4> world{};
            for(std::size_t index = 0; index < local.size(); ++index)
                world[index] = {
                    obstacle.x + cosine * local[index].x
                        - sine * local[index].y,
                    obstacle.y + sine * local[index].x
                        + cosine * local[index].y,
                };
            return world;
        }

        inline bool point_in_rectangle(
            const Point& point, const TemporaryObstacle& rectangle)
        {
            const double dx = point.x - rectangle.x;
            const double dy = point.y - rectangle.y;
            const double cosine = std::cos(rectangle.yaw);
            const double sine = std::sin(rectangle.yaw);
            const double local_x = cosine * dx + sine * dy;
            const double local_y = -sine * dx + cosine * dy;
            constexpr double epsilon = 1e-12;
            return std::fabs(local_x) <= rectangle.width * 0.5 + epsilon
                && std::fabs(local_y) <= rectangle.depth * 0.5 + epsilon;
        }

        inline double point_segment_distance(
            const Point& point, const Point& start, const Point& end)
        {
            const Point segment = subtract(end, start);
            const double length_squared = dot(segment, segment);
            if(length_squared <= 1e-24)
                return std::hypot(point.x - start.x, point.y - start.y);
            const double interpolation = std::clamp(
                dot(subtract(point, start), segment) / length_squared,
                0.0, 1.0);
            const Point closest{
                start.x + interpolation * segment.x,
                start.y + interpolation * segment.y,
            };
            return std::hypot(point.x - closest.x, point.y - closest.y);
        }

        inline bool segments_intersect(
            const Point& a, const Point& b, const Point& c, const Point& d)
        {
            const Point ab = subtract(b, a);
            const Point cd = subtract(d, c);
            const double denominator = cross(ab, cd);
            constexpr double epsilon = 1e-12;
            if(std::fabs(denominator) <= epsilon)
                return point_segment_distance(a, c, d) <= epsilon
                    || point_segment_distance(b, c, d) <= epsilon
                    || point_segment_distance(c, a, b) <= epsilon
                    || point_segment_distance(d, a, b) <= epsilon;
            const Point ac = subtract(c, a);
            const double t = cross(ac, cd) / denominator;
            const double u = cross(ac, ab) / denominator;
            return t >= -epsilon && t <= 1.0 + epsilon
                && u >= -epsilon && u <= 1.0 + epsilon;
        }

        inline double segment_distance(
            const Point& a, const Point& b, const Point& c, const Point& d)
        {
            if(segments_intersect(a, b, c, d))
                return 0.0;
            return std::min({
                point_segment_distance(a, c, d),
                point_segment_distance(b, c, d),
                point_segment_distance(c, a, b),
                point_segment_distance(d, a, b),
            });
        }

        inline double circle_rectangle_clearance(
            const TemporaryObstacle& circle,
            const TemporaryObstacle& rectangle)
        {
            const double dx = circle.x - rectangle.x;
            const double dy = circle.y - rectangle.y;
            const double cosine = std::cos(rectangle.yaw);
            const double sine = std::sin(rectangle.yaw);
            const double local_x = cosine * dx + sine * dy;
            const double local_y = -sine * dx + cosine * dy;
            const double outside_x = std::max(
                std::fabs(local_x) - rectangle.width * 0.5, 0.0);
            const double outside_y = std::max(
                std::fabs(local_y) - rectangle.depth * 0.5, 0.0);
            return std::hypot(outside_x, outside_y) - circle.width * 0.5;
        }

        inline double rectangle_clearance(
            const TemporaryObstacle& first,
            const TemporaryObstacle& second)
        {
            const auto first_corners = rectangle_corners(first);
            const auto second_corners = rectangle_corners(second);
            if(point_in_rectangle(first_corners[0], second)
               || point_in_rectangle(second_corners[0], first))
                return 0.0;

            double clearance = std::numeric_limits<double>::infinity();
            for(std::size_t first_edge = 0; first_edge < 4; ++first_edge)
                for(std::size_t second_edge = 0; second_edge < 4; ++second_edge)
                    clearance = std::min(clearance, segment_distance(
                        first_corners[first_edge],
                        first_corners[(first_edge + 1) % 4],
                        second_corners[second_edge],
                        second_corners[(second_edge + 1) % 4]));
            return clearance;
        }
    }

    /* Exact 2-D footprint edge clearance. A non-positive result means the
       footprints touch or overlap. Box and Wall share rotated rectangles. */
    inline double temporary_obstacle_clearance(
        const TemporaryObstacle& first, const TemporaryObstacle& second)
    {
        const bool first_is_circle =
            first.shape == TemporaryObstacleShape::Cylinder;
        const bool second_is_circle =
            second.shape == TemporaryObstacleShape::Cylinder;
        if(first_is_circle && second_is_circle)
            return std::hypot(first.x - second.x, first.y - second.y)
                - first.width * 0.5 - second.width * 0.5;
        if(first_is_circle)
            return temporary_obstacle_geometry::circle_rectangle_clearance(
                first, second);
        if(second_is_circle)
            return temporary_obstacle_geometry::circle_rectangle_clearance(
                second, first);
        return temporary_obstacle_geometry::rectangle_clearance(first, second);
    }
}
