#include <cmath>

#include <gtest/gtest.h>

#include "simulator/temporary_obstacle.hpp"

namespace
{
    using sim::TemporaryObstacle;
    using sim::TemporaryObstacleShape;

    TemporaryObstacle obstacle(
        TemporaryObstacleShape shape, double x, double y,
        double width, double depth, double yaw = 0.0)
    {
        return {shape, x, y, width, depth, 4.0, yaw};
    }
}

TEST(TemporaryObstacleGeometry, ComputesCircleClearance)
{
    const auto first = obstacle(
        TemporaryObstacleShape::Cylinder, 0.0, 0.0, 1.0, 1.0);
    const auto second = obstacle(
        TemporaryObstacleShape::Cylinder, 2.0, 0.0, 1.0, 1.0);

    EXPECT_NEAR(sim::temporary_obstacle_clearance(first, second), 1.0, 1e-12);
}

TEST(TemporaryObstacleGeometry, ComputesCircleToRotatedBoxClearance)
{
    constexpr double quarter_pi = 0.78539816339744830962;
    const double root_two = std::sqrt(2.0);
    const auto circle = obstacle(
        TemporaryObstacleShape::Cylinder,
        root_two + 1.0, 0.0, 1.0, 1.0);
    const auto box = obstacle(
        TemporaryObstacleShape::Box, 0.0, 0.0, 2.0, 2.0, quarter_pi);

    EXPECT_NEAR(sim::temporary_obstacle_clearance(circle, box), 0.5, 1e-12);
}

TEST(TemporaryObstacleGeometry, UsesFootprintsInsteadOfBoundingCircles)
{
    const auto first = obstacle(
        TemporaryObstacleShape::Wall, 0.0, 0.0, 4.0, 0.2);
    const auto second = obstacle(
        TemporaryObstacleShape::Wall, 0.0, 1.0, 4.0, 0.2);

    EXPECT_NEAR(sim::temporary_obstacle_clearance(first, second), 0.8, 1e-12);
    EXPECT_LT(
        std::hypot(first.x - second.x, first.y - second.y)
            - first.bounding_radius() - second.bounding_radius(),
        0.0);
}

TEST(TemporaryObstacleGeometry, RejectsCrossingRotatedWalls)
{
    constexpr double half_pi = 1.57079632679489661923;
    const auto first = obstacle(
        TemporaryObstacleShape::Wall, 0.0, 0.0, 4.0, 0.2);
    const auto second = obstacle(
        TemporaryObstacleShape::Wall, 0.0, 0.0, 4.0, 0.2, half_pi);

    EXPECT_DOUBLE_EQ(sim::temporary_obstacle_clearance(first, second), 0.0);
}

TEST(TemporaryObstacleGeometry, RejectsContainedRectangles)
{
    const auto outer = obstacle(
        TemporaryObstacleShape::Box, 0.0, 0.0, 4.0, 4.0, 0.25);
    const auto inner = obstacle(
        TemporaryObstacleShape::Box, 0.0, 0.0, 1.0, 1.0, -0.40);

    EXPECT_DOUBLE_EQ(sim::temporary_obstacle_clearance(outer, inner), 0.0);
}
