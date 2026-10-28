/* @author: YueLin */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "simulator/map.hpp"

namespace sim
{
    static int randint(int low, int high)
    {
        if(high <= low)
            return low;
        return low + std::rand() % (high - low + 1);
    }

    RandomMap::RandomMap(int seed, int obstacles, const std::string& raw_style,
                         double w, double h, double r, double radius, 
                         double x1, double y1, double x2, double y2,
                         double min_obstacle_height,
                         double max_obstacle_height,
                         const std::string& map_file)
    {
        nlohmann::json custom;
        const bool custom_map = !map_file.empty();
        if(custom_map)
        {
            std::ifstream stream(map_file);
            if(!stream)
                throw std::runtime_error(
                    "Cannot open custom map JSON: " + map_file);
            stream >> custom;
            const std::string format = custom.value(
                "format", std::string());
            if(format != "neural-tracker-map"
               && format != "neural-tracker-forest-map")
                throw std::runtime_error(
                    "Custom map must use neural-tracker-map or "
                    "neural-tracker-forest-map format");
            if(!custom.contains("map") || !custom["map"].is_object())
                throw std::runtime_error("Custom map is missing map metadata");
            w = custom["map"].value("width", 0.0);
            h = custom["map"].value("height", 0.0);
            r = custom["map"].value("resolution", 0.0);
            if(w <= 0. || h <= 0. || r <= 0.)
                throw std::runtime_error(
                    "Custom map width, height and resolution must be positive");
        }

        /* Initialize */
        width = w; height = h;
        resolution = r; r = 1 / r;
        cv::Size size(std::round(w * r), std::round(h * r));
        cv::Mat map = cv::Mat::ones(size, CV_8UC1) * 0xFF;
        cv::Mat custom_elevation = cv::Mat::zeros(size, CV_32F);
        offset = size / 2;

        /* Generate random map */
        std::srand(seed);
        std::string style = raw_style;
        if(style.empty() || style == "default")
            style = "blocks";
        if(style == "mixed")
        {
            const std::vector<std::string> styles = {
                "blocks", "sparse", "dense", "pillars", "mixed_shapes",
                "polygons_s", "l_shapes", "corner_maze", "loop_maze",
                "corridors", "rooms"
            };
            style = styles[std::abs(seed) % styles.size()];
        }

        const int scale = std::sqrt(size.area()) / std::max(1, int(std::sqrt(obstacles))) + 1;
        const int anchor_step = std::max(18, scale * 2 / 3);
        std::vector<cv::Point> anchors;
        for(int y = 0; y < size.height; y += anchor_step)
            for(int x = 0; x < size.width; x += anchor_step)
                anchors.emplace_back(x, y);

        auto next_anchor = [&anchors, &size, anchor_step]()
        {
            if(anchors.empty())
                return cv::Point(randint(0, size.width - 1), randint(0, size.height - 1));

            int idx = randint(0, int(anchors.size()) - 1);
            cv::Point point = anchors[idx];
            anchors[idx] = anchors.back();
            anchors.pop_back();

            int jitter = std::max(2, anchor_step / 4);
            point.x = std::max(0, std::min(size.width - 1, point.x + randint(-jitter, jitter)));
            point.y = std::max(0, std::min(size.height - 1, point.y + randint(-jitter, jitter)));
            return point;
        };

        auto rect = [&map, &size](int x, int y, int w, int h)
        {
            x = std::max(0, std::min(x, size.width - 1));
            y = std::max(0, std::min(y, size.height - 1));
            w = std::max(1, std::min(w, size.width - x));
            h = std::max(1, std::min(h, size.height - y));
            cv::rectangle(map, cv::Rect(x, y, w, h), 0, -1);
        };

        const int wall_step = std::max(18, scale * 2 / 3);
        const int wall_lane_margin = std::max(int(std::ceil(0.9 * r)), scale / 5);
        std::vector<int> horizontal_wall_lanes;
        std::vector<int> vertical_wall_lanes;
        for(int y = wall_lane_margin; y < size.height - wall_lane_margin; y += wall_step)
            horizontal_wall_lanes.push_back(y);
        for(int x = wall_lane_margin; x < size.width - wall_lane_margin; x += wall_step)
            vertical_wall_lanes.push_back(x);

        auto next_wall_lane = [wall_step](std::vector<int>& lanes, int low, int high)
        {
            if(high <= low)
                return low;
            if(lanes.empty())
                return randint(low, high);

            int idx = randint(0, int(lanes.size()) - 1);
            int lane = lanes[idx];
            lanes[idx] = lanes.back();
            lanes.pop_back();

            int jitter = std::max(2, wall_step / 6);
            return std::max(low, std::min(high, lane + randint(-jitter, jitter)));
        };

        auto wall_with_two_gaps = [&rect, &size, &horizontal_wall_lanes,
                                   &vertical_wall_lanes, &next_wall_lane,
                                   wall_lane_margin](bool horizontal)
        {
            const int thickness = randint(5, 9);
            if(horizontal)
            {
                const int margin = wall_lane_margin;
                const int gap = randint(size.width / 9, size.width / 6);
                int y = next_wall_lane(horizontal_wall_lanes, margin, size.height - margin);
                int g1 = randint(margin, std::max(margin + 1, size.width / 2 - gap));
                int g2 = randint(size.width / 2, std::max(size.width / 2 + 1, size.width - margin - gap));
                rect(margin, y, std::max(1, g1 - margin), thickness);
                rect(g1 + gap, y, std::max(1, g2 - g1 - gap), thickness);
                rect(g2 + gap, y, std::max(1, size.width - margin - g2 - gap), thickness);
            }
            else
            {
                const int margin = wall_lane_margin;
                const int gap = randint(size.height / 9, size.height / 6);
                int x = next_wall_lane(vertical_wall_lanes, margin, size.width - margin);
                int g1 = randint(margin, std::max(margin + 1, size.height / 2 - gap));
                int g2 = randint(size.height / 2, std::max(size.height / 2 + 1, size.height - margin - gap));
                rect(x, margin, thickness, std::max(1, g1 - margin));
                rect(x, g1 + gap, thickness, std::max(1, g2 - g1 - gap));
                rect(x, g2 + gap, thickness, std::max(1, size.height - margin - g2 - gap));
            }
        };

        auto place_mask = [&map](const cv::Mat& mask, int clearance)
        {
            cv::Mat occupied = map == 0;
            if(clearance > 0)
            {
                int kernel_size = clearance * 2 + 1;
                cv::Mat kernel = cv::getStructuringElement(
                    cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size)
                );
                cv::dilate(occupied, occupied, kernel);
            }

            cv::Mat overlap;
            cv::bitwise_and(mask, occupied, overlap);
            if(cv::countNonZero(overlap) > 0)
                return false;

            map.setTo(0, mask);
            return true;
        };

        auto random_polygon = [&place_mask, &size, &next_anchor](int min_r, int max_r,
                                                                 int vertices, int clearance)
        {
            const double pi = 3.14159265358979323846;
            cv::Point center = next_anchor();
            double phase = randint(0, 359) * pi / 180.0;
            std::vector<cv::Point> points;
            points.reserve(vertices);

            for(int i = 0; i < vertices; ++i)
            {
                double angle = phase + 2.0 * pi * i / vertices + randint(-18, 18) * pi / 180.0;
                int radius = randint(min_r, max_r);
                int px = std::max(0, std::min(size.width - 1, int(std::round(center.x + std::cos(angle) * radius))));
                int py = std::max(0, std::min(size.height - 1, int(std::round(center.y + std::sin(angle) * radius))));
                points.emplace_back(px, py);
            }

            std::vector<std::vector<cv::Point>> contours{points};
            cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
            cv::fillPoly(mask, contours, 0xFF);
            return place_mask(mask, clearance);
        };

        auto s_curve = [&place_mask, &size, &next_anchor](bool vertical, int clearance)
        {
            const double pi = 3.14159265358979323846;
            const int samples = 32;
            const int thickness = randint(4, 7);
            cv::Point center = next_anchor();
            int cx = center.x;
            int cy = center.y;
            int amplitude = randint(std::max(6, size.width / 16), std::max(7, size.width / 8));
            int span = randint(std::max(20, size.height / 4), std::max(21, size.height / 2));
            std::vector<cv::Point> points;
            points.reserve(samples);

            for(int i = 0; i < samples; ++i)
            {
                double t = double(i) / (samples - 1);
                double wave = std::sin(-pi + 2.0 * pi * t);
                int x = cx + int(std::round(amplitude * wave));
                int y = cy - span / 2 + int(std::round(span * t));
                if(!vertical)
                    std::swap(x, y);
                x = std::max(0, std::min(size.width - 1, x));
                y = std::max(0, std::min(size.height - 1, y));
                points.emplace_back(x, y);
            }

            std::vector<std::vector<cv::Point>> contours{points};
            cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
            cv::polylines(mask, contours, false, 0xFF, thickness);
            return place_mask(mask, clearance);
        };

        auto l_shape = [&place_mask, &size, &next_anchor](int min_len, int max_len,
                                                          int min_thick, int max_thick,
                                                          int clearance)
        {
            cv::Point center = next_anchor();
            int len1 = randint(min_len, max_len);
            int len2 = randint(min_len, max_len);
            int thickness = randint(min_thick, max_thick);
            int orientation = randint(0, 3);

            cv::Point corner = center;
            cv::Point p1 = corner;
            cv::Point p2 = corner;
            if(orientation == 0)
            {
                p1.x += len1;
                p2.y += len2;
            }
            else if(orientation == 1)
            {
                p1.x -= len1;
                p2.y += len2;
            }
            else if(orientation == 2)
            {
                p1.x -= len1;
                p2.y -= len2;
            }
            else
            {
                p1.x += len1;
                p2.y -= len2;
            }

            cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
            cv::line(mask, corner, p1, 0xFF, thickness);
            cv::line(mask, corner, p2, 0xFF, thickness);
            cv::circle(mask, corner, thickness / 2, 0xFF, -1);
            return place_mask(mask, clearance);
        };

        auto distributed_block = [&place_mask, &size, &next_anchor](int min_w, int max_w,
                                                                    int min_h, int max_h,
                                                                    int clearance)
        {
            int w = randint(min_w, max_w);
            int h = randint(min_h, max_h);
            cv::Point anchor = next_anchor();
            int x = std::max(0, std::min(size.width - 1, anchor.x - w / 2));
            int y = std::max(0, std::min(size.height - 1, anchor.y - h / 2));
            cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
            cv::rectangle(mask, cv::Rect(x, y, std::min(w, size.width - x), std::min(h, size.height - y)), 0xFF, -1);
            return place_mask(mask, clearance);
        };

        auto distributed_square = [&distributed_block](int min_s, int max_s, int clearance)
        {
            return distributed_block(min_s, max_s, min_s, max_s, clearance);
        };

        auto distributed_pillar = [&place_mask, &size, &next_anchor](int min_r, int max_r, int clearance)
        {
            int radius = randint(min_r, max_r);
            cv::Point center = next_anchor();
            cv::Mat mask = cv::Mat::zeros(size, CV_8UC1);
            cv::circle(mask, center, radius, 0xFF, -1);
            return place_mask(mask, clearance);
        };

        if(custom_map)
        {
            const double origin_x = -width * 0.5;
            const double origin_y = -height * 0.5;
            const double pi = 3.14159265358979323846;
            const auto& custom_obstacles = custom.value(
                "obstacles", nlohmann::json::array());
            if(!custom_obstacles.is_array())
                throw std::runtime_error(
                    "Custom map obstacles must be an array");

            for(const auto& obstacle: custom_obstacles)
            {
                std::string shape = obstacle.value(
                    "shape", std::string("wall"));
                const double center_x = obstacle.value("x", 0.0);
                const double center_y = obstacle.value("y", 0.0);
                const double angle = obstacle.value("angle", 0.0) * pi / 180.;
                const double cosine = std::cos(angle);
                const double sine = std::sin(angle);
                const double obstacle_width = obstacle.value("width", 0.0);
                const double obstacle_depth = obstacle.value("depth", 0.0);
                double obstacle_radius = obstacle.value("radius", 0.0);
                double obstacle_height = obstacle.value(
                    "height", max_obstacle_height);
                double thickness = obstacle.value("thickness", 0.0);

                /* Forest benchmark maps keep the physical cylinder under a
                   nested trunk object, while Map Studio tree obstacles store
                   radius/height directly. In both cases the crown is visual
                   metadata and is excluded from the ground-level 2-D map. */
                if(shape == "tree")
                {
                    if(!obstacle.value("collision", true))
                        continue;
                    if(obstacle.contains("trunk"))
                    {
                        if(!obstacle["trunk"].is_object())
                            throw std::runtime_error(
                                "Custom tree trunk metadata must be an object");
                        const auto& trunk = obstacle["trunk"];
                        if(!trunk.value("collision", true))
                            continue;
                        const std::string primitive = trunk.value(
                            "primitive", std::string("cylinder"));
                        if(primitive != "cylinder")
                            throw std::runtime_error(
                                "Custom tree trunk primitive must be cylinder");
                        obstacle_radius = trunk.value("radius", 0.0);
                        obstacle_height = trunk.value(
                            "height", max_obstacle_height);

                        if(obstacle.contains("canopy"))
                        {
                            if(!obstacle["canopy"].is_object())
                                throw std::runtime_error(
                                    "Custom tree canopy metadata must be an object");
                            const auto& canopy = obstacle["canopy"];
                            const double radius_x = canopy.value(
                                "radius_x", 0.0);
                            const double radius_y = canopy.value(
                                "radius_y", 0.0);
                            const double radius_z = canopy.value(
                                "radius_z", 0.0);
                            if(radius_x <= 0. || radius_y <= 0.
                               || radius_z <= 0.)
                                throw std::runtime_error(
                                    "Custom tree canopy radii must be positive");
                            canopy_visuals_.push_back({
                                center_x, center_y,
                                canopy.value(
                                    "center_z", obstacle_height),
                                radius_x, radius_y, radius_z, angle,
                                54, 122, 69
                            });
                        }
                    }
                    else
                    {
                        const double crown_radius = obstacle.value(
                            "crownRadius", 0.0);
                        if(crown_radius <= 0.)
                            throw std::runtime_error(
                                "Map Studio tree crownRadius must be positive");
                        const double full_height = obstacle_height;
                        const double trunk_height = std::max(
                            0.2, full_height * 0.56);
                        const double crown_height = std::max(
                            0.05, full_height - trunk_height);
                        const double lower_height = crown_height * 0.62;
                        const double upper_height = crown_height * 0.55;
                        canopy_visuals_.push_back({
                            center_x, center_y,
                            trunk_height + lower_height * 0.38,
                            crown_radius, crown_radius * 0.92,
                            lower_height * 0.5, angle,
                            46, 112, 62
                        });
                        canopy_visuals_.push_back({
                            center_x, center_y,
                            full_height - upper_height * 0.42,
                            crown_radius * 0.72,
                            crown_radius * 0.68,
                            upper_height * 0.5, angle,
                            79, 138, 88
                        });
                        obstacle_height = trunk_height;
                    }
                    shape = "cylinder";
                }

                const bool composite_shape =
                    shape == "l_shape" || shape == "e_shape"
                    || shape == "t_shape" || shape == "h_shape"
                    || shape == "s_shape" || shape == "f_shape";
                const bool arch_shape = shape == "arch";
                const bool segmented_shape = composite_shape || arch_shape;
                double arch_clearance = 0.;

                if(obstacle_height <= 0.)
                    throw std::runtime_error(
                        "Custom obstacle height must be positive");
                if(shape == "cylinder")
                {
                    if(obstacle_radius <= 0.)
                        throw std::runtime_error(
                            "Custom cylinder radius must be positive");
                }
                else
                {
                    if(obstacle_width <= 0. || obstacle_depth <= 0.)
                        throw std::runtime_error(
                            "Custom obstacle width and depth must be positive");
                    if(segmented_shape)
                        thickness = std::max(
                            resolution, std::min({
                                thickness, obstacle_width, obstacle_depth
                            }));
                    if(arch_shape)
                    {
                        thickness = std::min(
                            thickness, obstacle_width / 3.);
                        arch_clearance = obstacle.value(
                            "clearance", 0.0);
                        if(arch_clearance <= 0.
                           || arch_clearance >= obstacle_height)
                            throw std::runtime_error(
                                "Custom arch clearance must be positive and "
                                "below its height");
                    }
                    else if(shape == "frustum")
                    {
                        const double top_scale = obstacle.value(
                            "topScale", 0.0);
                        if(top_scale <= 0. || top_scale > 1.)
                            throw std::runtime_error(
                                "Custom frustum topScale must be in (0, 1]");
                    }
                    else if(!composite_shape
                            && shape != "wall" && shape != "box")
                        throw std::runtime_error(
                            "Unsupported custom obstacle shape: " + shape);
                }

                if(arch_shape)
                    arch_visuals_.push_back({
                        center_x, center_y,
                        obstacle_width, obstacle_depth, obstacle_height,
                        thickness, arch_clearance, angle
                    });

                /* Keep these component definitions identical to the sibling
                 * map_editor project's app/map-types.ts::localRects(). Each entry is
                 * [center_x, center_y, width, depth] in obstacle-local axes;
                 * the obstacle's angle is applied once below. */
                std::vector<std::array<double, 4>> local_rectangles;
                if(segmented_shape)
                {
                    const double left = -(obstacle_width - thickness) * 0.5;
                    const double right = (obstacle_width - thickness) * 0.5;
                    const double top = (obstacle_depth - thickness) * 0.5;
                    const double bottom = -(obstacle_depth - thickness) * 0.5;
                    if(shape == "l_shape")
                    {
                        local_rectangles.push_back({
                            0., bottom, obstacle_width, thickness});
                        local_rectangles.push_back({
                            left, 0., thickness, obstacle_depth});
                    }
                    else if(shape == "e_shape")
                    {
                        local_rectangles.push_back({
                            left, 0., thickness, obstacle_depth});
                        local_rectangles.push_back({
                            0., top, obstacle_width, thickness});
                        local_rectangles.push_back({
                            0., 0., obstacle_width, thickness});
                        local_rectangles.push_back({
                            0., bottom, obstacle_width, thickness});
                    }
                    else if(shape == "t_shape")
                    {
                        local_rectangles.push_back({
                            0., top, obstacle_width, thickness});
                        local_rectangles.push_back({
                            0., -thickness * 0.5, thickness,
                            std::max(thickness,
                                     obstacle_depth - thickness)});
                    }
                    else if(shape == "h_shape")
                    {
                        local_rectangles.push_back({
                            left, 0., thickness, obstacle_depth});
                        local_rectangles.push_back({
                            right, 0., thickness, obstacle_depth});
                        local_rectangles.push_back({
                            0., 0., obstacle_width, thickness});
                    }
                    else if(shape == "s_shape")
                    {
                        const double connector_depth = std::max(
                            thickness,
                            (obstacle_depth - 3. * thickness) * 0.5
                                + thickness);
                        local_rectangles.push_back({
                            0., top, obstacle_width, thickness});
                        local_rectangles.push_back({
                            0., 0., obstacle_width, thickness});
                        local_rectangles.push_back({
                            0., bottom, obstacle_width, thickness});
                        local_rectangles.push_back({
                            left, obstacle_depth * 0.25,
                            thickness, connector_depth});
                        local_rectangles.push_back({
                            right, -obstacle_depth * 0.25,
                            thickness, connector_depth});
                    }
                    else if(shape == "f_shape")
                    {
                        local_rectangles.push_back({
                            left, 0., thickness, obstacle_depth});
                        local_rectangles.push_back({
                            0., top, obstacle_width, thickness});
                        local_rectangles.push_back({
                            0., 0., obstacle_width, thickness});
                    }
                    else if(shape == "arch")
                    {
                        /* The overhead span does not block a ground-level 2-D
                           route. Rasterize the two supporting legs, matching
                           Map Studio's localRects() and MuJoCo export. */
                        local_rectangles.push_back({
                            left, 0., thickness, obstacle_depth});
                        local_rectangles.push_back({
                            right, 0., thickness, obstacle_depth});
                    }
                }

                const double extent = shape == "cylinder"
                    ? obstacle_radius
                    : std::hypot(obstacle_width * 0.5,
                                 obstacle_depth * 0.5);
                const int ix_min = std::max(0, static_cast<int>(std::floor(
                    (center_x - extent - origin_x) / resolution)));
                const int ix_max = std::min(size.width - 1,
                    static_cast<int>(std::ceil(
                        (center_x + extent - origin_x) / resolution)));
                const int iy_min = std::max(0, static_cast<int>(std::floor(
                    (center_y - extent - origin_y) / resolution)));
                const int iy_max = std::min(size.height - 1,
                    static_cast<int>(std::ceil(
                        (center_y + extent - origin_y) / resolution)));

                for(int y = iy_min; y <= iy_max; ++y)
                {
                    const double world_y = origin_y
                        + (static_cast<double>(y) + 0.5) * resolution;
                    for(int x = ix_min; x <= ix_max; ++x)
                    {
                        const double world_x = origin_x
                            + (static_cast<double>(x) + 0.5) * resolution;
                        const double dx = world_x - center_x;
                        const double dy = world_y - center_y;
                        const double local_x = cosine * dx + sine * dy;
                        const double local_y = -sine * dx + cosine * dy;
                        bool occupied = false;
                        if(shape == "cylinder")
                            occupied = local_x * local_x + local_y * local_y
                                     <= obstacle_radius * obstacle_radius;
                        else if(segmented_shape)
                        {
                            for(const auto& rectangle: local_rectangles)
                                if(std::fabs(local_x - rectangle[0])
                                        <= rectangle[2] * 0.5
                                   && std::fabs(local_y - rectangle[1])
                                        <= rectangle[3] * 0.5)
                                {
                                    occupied = true;
                                    break;
                                }
                        }
                        else
                            occupied =
                                std::fabs(local_x) <= obstacle_width * 0.5
                                && std::fabs(local_y) <= obstacle_depth * 0.5;

                        if(occupied)
                        {
                            map.at<uchar>(y, x) = 0;
                            const float occupied_height = static_cast<float>(
                                arch_shape ? arch_clearance
                                           : obstacle_height);
                            custom_elevation.at<float>(y, x) = std::max(
                                custom_elevation.at<float>(y, x),
                                occupied_height);
                        }
                    }
                }
            }

            const auto boundary_config = custom.value(
                "boundary", nlohmann::json::object());
            if(boundary_config.value("enabled", false))
            {
                const double thickness = boundary_config.value(
                    "thickness", radius);
                const float boundary_height = boundary_config.value(
                    "height", static_cast<float>(max_obstacle_height));
                if(thickness <= 0. || boundary_height <= 0.)
                    throw std::runtime_error(
                        "Custom boundary thickness and height must be positive");
                const int cells = std::max(
                    1, static_cast<int>(std::ceil(thickness / resolution)));
                for(int y = 0; y < size.height; ++y)
                    for(int x = 0; x < size.width; ++x)
                        if(x < cells || y < cells
                           || x >= size.width - cells
                           || y >= size.height - cells)
                        {
                            map.at<uchar>(y, x) = 0;
                            custom_elevation.at<float>(y, x) = boundary_height;
                        }
            }
        }
        else if(style == "forest" || style == "forest_dense")
        {
            /* Tree trunks use a stratified, jittered distribution. This gives
               the blue-noise-like spacing used by randomized off-road scenes
               without changing the binary occupancy semantics. */
            const int count = std::max(8, obstacles);
            const int min_radius = std::max(2, int(std::ceil(0.12 * r)));
            const int max_radius = std::max(min_radius + 1,
                int(std::ceil((style == "forest_dense" ? 0.32 : 0.42) * r)));
            const int clearance = std::max(2, int(std::ceil(
                (style == "forest_dense" ? 0.18 : 0.45) * r)));
            int placed = 0;
            for(int i = 0; i < count * 24 && placed < count; ++i)
                if(distributed_pillar(min_radius, max_radius, clearance))
                    ++placed;
        }
        else if(style == "rocky")
        {
            const int count = std::max(8, obstacles);
            const int min_radius = std::max(3, int(std::ceil(0.18 * r)));
            const int max_radius = std::max(min_radius + 1,
                int(std::ceil(0.85 * r)));
            const int clearance = std::max(2, int(std::ceil(0.22 * r)));
            int placed = 0;
            for(int i = 0; i < count * 24 && placed < count; ++i)
                if(random_polygon(min_radius, max_radius,
                                  randint(5, 9), clearance))
                    ++placed;
        }
        else if(style == "offroad_mixed")
        {
            const int count = std::max(10, obstacles);
            const int clearance = std::max(2, int(std::ceil(0.25 * r)));
            int placed = 0;
            for(int i = 0; i < count * 28 && placed < count; ++i)
            {
                bool ok = false;
                if(i % 5 < 3)
                    ok = distributed_pillar(
                        std::max(2, int(std::ceil(0.12 * r))),
                        std::max(3, int(std::ceil(0.38 * r))), clearance);
                else if(i % 5 == 3)
                    ok = random_polygon(
                        std::max(3, int(std::ceil(0.20 * r))),
                        std::max(5, int(std::ceil(0.75 * r))),
                        randint(5, 8), clearance);
                else
                    ok = distributed_block(
                        std::max(3, int(std::ceil(0.25 * r))),
                        std::max(5, int(std::ceil(0.90 * r))),
                        std::max(3, int(std::ceil(0.20 * r))),
                        std::max(5, int(std::ceil(0.65 * r))), clearance);
                if(ok)
                    ++placed;
            }
        }
        else if(style == "sparse")
        {
            int count = std::max(4, obstacles / 2);
            int placed = 0;
            for(int i = 0; i < count * 12 && placed < count; ++i)
                if(distributed_block(scale / 2, scale * 2,
                                     scale / 2, scale * 2,
                                     std::max(6, scale / 8)))
                    ++placed;
        }
        else if(style == "dense")
        {
            int count = std::max(obstacles, obstacles * 3 / 2);
            int placed = 0;
            for(int i = 0; i < count * 16 && placed < count; ++i)
                if(distributed_block(scale / 4, scale,
                                     scale / 4, scale,
                                     std::max(4, scale / 14)))
                    ++placed;
        }
        else if(style == "pillars")
        {
            int count = std::max(obstacles, obstacles * 3 / 2);
            int placed = 0;
            for(int i = 0; i < count * 16 && placed < count; ++i)
                if(distributed_pillar(std::max(3, scale / 8),
                                      std::max(5, scale / 3),
                                      std::max(5, scale / 14)))
                    ++placed;
        }
        else if(style == "mixed_shapes")
        {
            int count = std::max(6, obstacles);
            int placed = 0;
            for(int i = 0; i < count * 14 && placed < count; ++i)
            {
                bool ok = i % 2 == 0?
                    distributed_pillar(std::max(3, scale / 10),
                                       std::max(5, scale / 3),
                                       std::max(4, scale / 16)):
                    distributed_square(std::max(4, scale / 5),
                                       std::max(6, scale * 2 / 3),
                                       std::max(4, scale / 16));
                if(ok)
                    ++placed;
            }
        }
        else if(style == "polygons_s")
        {
            int curve_clearance = std::max(4, scale / 18);
            int polygon_clearance = std::max(3, scale / 24);

            int curves = std::max(3, obstacles / 8);
            int placed_curves = 0;
            for(int i = 0; i < curves * 16 && placed_curves < curves; ++i)
                if(s_curve(i % 2 == 0, curve_clearance))
                    ++placed_curves;

            int polygons = std::max(12, obstacles * 3 / 4);
            int placed_polygons = 0;
            for(int i = 0; i < polygons * 20 && placed_polygons < polygons; ++i)
                if(random_polygon(std::max(4, scale / 12),
                                  std::max(7, scale / 3),
                                  randint(5, 8),
                                  polygon_clearance))
                    ++placed_polygons;
        }
        else if(style == "l_shapes")
        {
            int count = std::max(14, obstacles * 5 / 4);
            int placed = 0;
            for(int i = 0; i < count * 14 && placed < count; ++i)
                if(l_shape(std::max(10, scale / 4),
                           std::max(16, scale * 3 / 4),
                           std::max(4, scale / 12),
                           std::max(7, scale / 8),
                           std::max(3, scale / 26)))
                    ++placed;

            int small_pillars = std::max(2, obstacles / 8);
            int placed_pillars = 0;
            for(int i = 0; i < small_pillars * 10 && placed_pillars < small_pillars; ++i)
                if(distributed_pillar(std::max(3, scale / 12),
                                      std::max(5, scale / 4),
                                      std::max(4, scale / 18)))
                    ++placed_pillars;
        }
        else if(style == "corner_maze")
        {
            int walls = std::max(10, obstacles * 2 / 3);
            for(int i = 0; i < walls; ++i)
                wall_with_two_gaps(i % 2 == 0);

            int corners = std::max(8, obstacles / 2);
            int placed = 0;
            for(int i = 0; i < corners * 14 && placed < corners; ++i)
                if(l_shape(std::max(10, scale / 5),
                           std::max(16, scale / 2),
                           std::max(4, scale / 14),
                           std::max(6, scale / 9),
                           std::max(3, scale / 28)))
                    ++placed;
        }
        else if(style == "loop_maze")
        {
            int walls = std::max(12, obstacles * 3 / 4);
            for(int i = 0; i < walls; ++i)
                wall_with_two_gaps(i % 2 == 0);
        }
        else if(style == "corridors")
        {
            int count = std::max(6, obstacles / 2);
            while(count--)
                wall_with_two_gaps(count % 2 == 0);
            int extras = std::max(2, obstacles / 8);
            int placed = 0;
            for(int i = 0; i < extras * 10 && placed < extras; ++i)
                if(distributed_block(scale / 3, scale,
                                     scale / 3, scale,
                                     std::max(4, scale / 18)))
                    ++placed;
        }
        else if(style == "rooms")
        {
            int walls = std::max(4, obstacles / 5);
            while(walls--)
                wall_with_two_gaps(walls % 2 == 0);
            int extras = std::max(2, obstacles / 10);
            int placed = 0;
            for(int i = 0; i < extras * 10 && placed < extras; ++i)
                if(distributed_pillar(std::max(3, scale / 10),
                                      std::max(4, scale / 5),
                                      std::max(4, scale / 18)))
                    ++placed;
        }
        else
        {
            int count = obstacles;
            int placed = 0;
            for(int i = 0; i < count * 14 && placed < count; ++i)
                if(distributed_block(std::max(1, scale / std::max(1, int(std::sqrt(obstacles)))),
                                     scale,
                                     std::max(1, scale / std::max(1, int(std::sqrt(obstacles)))),
                                     scale,
                                     std::max(5, scale / 10)))
                    ++placed;
        }

        if(!custom_map)
        {
            cv::rectangle(map, cv::Rect(cv::Point(
                std::round((std::min(x1, x2) - 0xa * radius) * r) + offset.width,
                std::round((std::min(y1, y2) - 0xa * radius) * r) + offset.height
            ), cv::Point(
                std::round((std::max(x1, x2) + 0xa * radius) * r) + offset.width,
                std::round((std::max(y1, y2) + 0xa * radius) * r) + offset.height
            )), 0xFF, -1);

            /* Close procedurally generated maps with robot-radius walls. */
            const int boundary = std::max(1, int(std::ceil(radius * r)));
            rect(0, 0, size.width, boundary);
            rect(0, size.height - boundary, size.width, boundary);
            rect(0, 0, boundary, size.height);
            rect(size.width - boundary, 0, boundary, size.height);
        }

        /* Give every connected obstacle a deterministic physical height.
           The 2D ESDF below remains unchanged and is still the sole map used
           by the original tracker. Heights are only consumed by 3D outputs. */
        min_obstacle_height = std::max(0.05, min_obstacle_height);
        max_obstacle_height = std::max(min_obstacle_height,
                                       max_obstacle_height);
        if(custom_map)
            elevation = custom_elevation;
        else
        {
            cv::Mat occupied = map == 0;
            cv::Mat labels;
            const int components = cv::connectedComponents(
                occupied, labels, 8);
            std::vector<float> component_heights(components, 0.0f);
            const double height_span =
                max_obstacle_height - min_obstacle_height;
            for(int label = 1; label < components; ++label)
            {
                const unsigned int hash =
                    static_cast<unsigned int>(seed) * 2654435761u
                    ^ static_cast<unsigned int>(label) * 2246822519u;
                const double unit =
                    (hash & 0x00ffffffu) / double(0x00ffffffu);
                component_heights[label] = static_cast<float>(
                    min_obstacle_height + height_span * unit);
            }
            elevation = cv::Mat::zeros(size, CV_32F);
            for(int y = 0; y < size.height; ++y)
                for(int x = 0; x < size.width; ++x)
                {
                    const int label = labels.at<int>(y, x);
                    if(label > 0)
                        elevation.at<float>(y, x) =
                            component_heights[label];
                }
        }

        rebuild_distance_field(map);
    }

    void RandomMap::rebuild_distance_field(const cv::Mat& occupancy)
    {
        /* Create ESDF. Occupied cells are zero in the OpenCV map and 100 in
           the ROS OccupancyGrid. */
        cv::Mat esdf;
        cv::distanceTransform(
            occupancy, data, cv::DIST_L2, cv::DIST_MASK_PRECISE
        );
        cv::distanceTransform(
            ~occupancy, esdf, cv::DIST_L2, cv::DIST_MASK_PRECISE
        );
        for(int y = 0; y < occupancy.rows; y++)
            for(int x = 0; x < occupancy.cols; x++)
                if(!occupancy.at<uchar>(y, x))
                    data.at<float>(y, x) = -esdf.at<float>(y, x);
        data *= resolution;

        /* Initialize/update message. */
        msg.info.width = occupancy.cols;
        msg.info.height = occupancy.rows;
        msg.data.resize(occupancy.total());
        msg.info.resolution = resolution;
        msg.info.origin.position.x = -width / 2;
        msg.info.origin.position.y = -height / 2;
        for(int y = 0; y < data.rows; y++)
        {
            int z = y * data.cols;
            for(int x = 0; x < data.cols; x++)
                msg.data[x + z] = data.at<float>(y, x) <= 0? 100: 0;
        }
    }

    void RandomMap::set_temporary_obstacles(
        const std::vector<TemporaryObstacle>& obstacles)
    {
        /* Lazily preserve the static layers so the original simulator path
           pays no extra full-map memory cost when this feature is disabled. */
        if(static_occupancy_.empty())
        {
            static_distance_ = data.clone();
            static_occupancy_ = cv::Mat::ones(data.size(), CV_8UC1) * 0xFF;
            for(int y = 0; y < data.rows; ++y)
                for(int x = 0; x < data.cols; ++x)
                    if(data.at<float>(y, x) <= 0.)
                        static_occupancy_.at<uchar>(y, x) = 0;
            static_elevation_ = elevation.clone();
        }
        cv::Mat occupancy = static_occupancy_.clone();
        elevation = static_elevation_.clone();

        for(const auto& obstacle: obstacles)
        {
            if(obstacle.width <= 0. || obstacle.depth <= 0.
               || obstacle.height <= 0.)
                continue;
            const cv::Point center = encode(obstacle.x, obstacle.y);
            const double extent = obstacle.bounding_radius();
            const int cells = std::max(
                1, static_cast<int>(std::ceil(extent / resolution)));
            const int x_min = std::max(0, center.x - cells);
            const int x_max = std::min(occupancy.cols - 1, center.x + cells);
            const int y_min = std::max(0, center.y - cells);
            const int y_max = std::min(occupancy.rows - 1, center.y + cells);
            const double cosine = std::cos(obstacle.yaw);
            const double sine = std::sin(obstacle.yaw);
            for(int y = y_min; y <= y_max; ++y)
                for(int x = x_min; x <= x_max; ++x)
                {
                    const double world_x =
                        (x - offset.width + 0.5) * resolution;
                    const double world_y =
                        (y - offset.height + 0.5) * resolution;
                    const double dx = world_x - obstacle.x;
                    const double dy = world_y - obstacle.y;
                    const double local_x = cosine * dx + sine * dy;
                    const double local_y = -sine * dx + cosine * dy;
                    const bool occupied =
                        obstacle.shape == TemporaryObstacleShape::Cylinder
                        ? local_x * local_x + local_y * local_y
                            <= obstacle.width * obstacle.width * 0.25
                        : std::fabs(local_x) <= obstacle.width * 0.5
                            && std::fabs(local_y) <= obstacle.depth * 0.5;
                    if(occupied)
                    {
                        occupancy.at<uchar>(y, x) = 0;
                        elevation.at<float>(y, x) = std::max(
                            elevation.at<float>(y, x),
                            static_cast<float>(obstacle.height));
                    }
                }
        }

        rebuild_distance_field(occupancy);
    }

    geometry_msgs::msg::Pose RandomMap::goal() const
    {
        /* Generate random point */
        cv::Point point;
        const float min_clearance = 0.3f;
        const int attempts = data.cols * data.rows;
        bool found = false;
        for(int i = 0; i < attempts; ++i)
        {
            point.x = std::rand() % data.cols;
            point.y = std::rand() % data.rows;
            if(data.at<float>(point) > min_clearance)
            {
                found = true;
                break;
            }
        }
        if(!found)
        {
            double best = -1e9;
            cv::Point best_point = point;
            for(int y = 0; y < data.rows; ++y)
                for(int x = 0; x < data.cols; ++x)
                {
                    double value = data.at<float>(y, x);
                    if(value > best)
                    {
                        best = value;
                        best_point = cv::Point(x, y);
                    }
                }
            point = best_point;
        }

        /* Convert to PoseStamped */
        geometry_msgs::msg::Pose pose;
        pose.position.x = (point.x - offset.width) * resolution;
        pose.position.y = (point.y - offset.height) * resolution;
        pose.orientation.x = 0;
        pose.orientation.y = 0;
        pose.orientation.z = 0;
        pose.orientation.w = 0;
        return pose;
    }

    std::pair<double, double> RandomMap::nearest_free(
        double x, double y, double clearance) const
    {
        clearance = std::max(0.0, clearance);
        const cv::Point requested = encode(x, y);
        if(data.at<float>(requested) > clearance)
            return {x, y};

        double best_distance = std::numeric_limits<double>::infinity();
        cv::Point best;
        bool found = false;
        for(int iy = 0; iy < data.rows; ++iy)
            for(int ix = 0; ix < data.cols; ++ix)
            {
                if(data.at<float>(iy, ix) <= clearance)
                    continue;
                const double dx = static_cast<double>(ix - requested.x);
                const double dy = static_cast<double>(iy - requested.y);
                const double distance = dx * dx + dy * dy;
                if(distance < best_distance)
                {
                    best_distance = distance;
                    best = cv::Point(ix, iy);
                    found = true;
                }
            }
        if(!found)
            throw std::runtime_error(
                "Custom map has no free spawn position with the requested clearance");
        return {
            (best.x - offset.width) * resolution,
            (best.y - offset.height) * resolution
        };
    }
}
