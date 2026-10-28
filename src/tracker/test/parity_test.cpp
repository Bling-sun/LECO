/* Offline parity test for tracker_neural.
 *
 * Reads the synthetic scenario dumped by parity_dump.py, rebuilds the Map ESDF
 * and the neural policy inputs in C++, runs ONNX inference, and compares esdf
 * patch / target_rel / visibility_ray / controls[0] against the Python dumps.
 *
 * Build target: parity_test (see CMakeLists). Run from src/tracker/test so the
 * relative dump paths resolve, or pass the dir as argv[1].
 */
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "planner/map.hpp"
#include "planner/neural_policy.hpp"

using diff_track::Map;
using diff_track::NeuralPolicy;

static std::vector<double> load_col(const std::string& path)
{
    std::ifstream f(path);
    std::vector<double> v;
    double x;
    while (f >> x) v.push_back(x);
    return v;
}

static double max_abs_diff(const std::vector<float>& a,
                           const std::vector<double>& b, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; ++i)
        m = std::max(m, std::fabs(static_cast<double>(a[i]) - b[i]));
    return m;
}

int main(int argc, char** argv)
{
    std::string dir = (argc > 1) ? argv[1] : ".";
    std::string onnx = (argc > 2) ? argv[2]
        : DEFAULT_ONNX_PATH;

    // --- read scenario.txt ---
    std::ifstream f(dir + "/scenario.txt");
    if (!f) { std::fprintf(stderr, "cannot open scenario.txt in %s\n", dir.c_str()); return 2; }
    int N; double res;
    f >> N >> res;
    double rx, ry, ryaw, rv, rw;
    f >> rx >> ry >> ryaw >> rv >> rw;
    double tx, ty, tvx, tvy;
    f >> tx >> ty >> tvx >> tvy;
    std::vector<int> occ(N * N);
    for (int i = 0; i < N * N; ++i) f >> occ[i];

    // --- build an OccupancyGrid and feed Map::update (real ESDF path) ---
    auto grid = std::make_shared<nav_msgs::msg::OccupancyGrid>();
    grid->info.width = N;
    grid->info.height = N;
    grid->info.resolution = res;
    grid->info.origin.position.x = -N * res * 0.5;
    grid->info.origin.position.y = -N * res * 0.5;
    grid->data.resize(N * N);
    for (int i = 0; i < N * N; ++i) grid->data[i] = occ[i] ? 100 : 0;

    Map map(N * res, N * res, res);
    map.update(grid);

    // --- run neural policy ---
    NeuralPolicy policy(onnx, 0.1, 2.0, 1.2, 1.0, 4);
    Eigen::VectorXd state(5);
    state << rx, ry, ryaw, rv, rw;
    Eigen::Vector2d tpos(tx, ty), tvel(tvx, tvy), cmd;
    auto traj = policy.infer(state, tpos, tvel, &map, cmd);

    // Recompute esdf patch / target_rel / visibility here with the same public
    // Map + formulas the policy uses, to localize any preprocessing mismatch.
    {
        const cv::Mat& sdf = map.sdfMat();
        const double cx = rx / res + map.offset.width;
        const double cy = ry / res + map.offset.height;
        const double half = 40.0;
        cv::Mat rot = cv::getRotationMatrix2D(
            cv::Point2f((float)cx, (float)cy), -ryaw * 180.0 / M_PI, 1.0);
        rot.at<double>(0, 2) += half - cx;
        rot.at<double>(1, 2) += half - cy;
        cv::Mat patch;
        cv::warpAffine(sdf, patch, rot, cv::Size(80, 80),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(-1.0));
        std::vector<float> pflat(80 * 80);
        for (int r = 0; r < 80; ++r)
            for (int c = 0; c < 80; ++c) pflat[r * 80 + c] = patch.at<float>(r, c);
        auto exp_esdf = load_col(dir + "/expect_esdf.txt");
        double d_esdf = max_abs_diff(pflat, exp_esdf, 80 * 80);
        std::printf("esdf patch   max|d|=%.3e (m)\n", d_esdf);

        auto exp_rel = load_col(dir + "/expect_rel.txt");
        double dx = tx - rx, dy = ty - ry;
        double rho = std::hypot(dx, dy);
        double t_rel = std::atan2(std::sin(std::atan2(dy, dx) - ryaw),
                                  std::cos(std::atan2(dy, dx) - ryaw));
        double sp = std::hypot(tvx, tvy);
        double t_v = sp > 1e-3 ? std::atan2(std::sin(std::atan2(tvy, tvx) - ryaw),
                                            std::cos(std::atan2(tvy, tvx) - ryaw)) : 0.0;
        std::vector<float> rel = {(float)rho, (float)std::sin(t_rel), (float)std::cos(t_rel),
                                  (float)std::sin(t_v), (float)std::cos(t_v), (float)sp};
        std::printf("target_rel   max|d|=%.3e\n", max_abs_diff(rel, exp_rel, 6));

        // visibility_ray with the neural_policy formula
        auto exp_vis = load_col(dir + "/expect_vis.txt");
        std::vector<float> vis(8);
        for (int i = 0; i < 8; ++i) {
            double a = (double)i / 7.0;
            double pxm = rx * (1 - a) + tx * a;
            double pym = ry * (1 - a) + ty * a;
            int px = (int)((pxm + map.offset.width * res) / res);
            int py = (int)((pym + map.offset.height * res) / res);
            vis[i] = (px >= 0 && px < sdf.cols && py >= 0 && py < sdf.rows)
                     ? sdf.at<float>(py, px) : -1.0f;
        }
        std::printf("visibility   max|d|=%.3e\n", max_abs_diff(vis, exp_vis, 8));
        std::printf("  C++ vis:");
        for (int i = 0; i < 8; ++i) std::printf(" %.4f", vis[i]);
        std::printf("\n");
    }

    // controls parity
    auto exp_ctrl = load_col(dir + "/expect_ctrl.txt");
    std::vector<float> got_ctrl = {static_cast<float>(cmd[0]), static_cast<float>(cmd[1])};
    // Python dump did NOT clip/omega_sign (omega_sign=1, within limits) so equal.
    double d_ctrl = max_abs_diff(got_ctrl, exp_ctrl, 2);

    std::printf("controls[0]  C++=(%.6f, %.6f)  py=(%.6f, %.6f)  max|d|=%.3e\n",
                cmd[0], cmd[1], exp_ctrl[0], exp_ctrl[1], d_ctrl);

    // Rolling-grid geometry: actual lidar returns are supplied in world
    // coordinates, while storage and published origin follow the robot.
    Map rolling(12.0, 12.0, 0.05, 6.0);
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.emplace_back(14.0f, -4.0f, 0.0f);
    cloud.emplace_back(10.0f, -4.0f, 0.0f);
    cloud.emplace_back(12.0f, -2.0f, 0.0f);
    cloud.emplace_back(12.0f, -6.0f, 0.0f);
    Eigen::Vector3d robot(12.0, -4.0, 0.0);
    Eigen::Vector3d no_filter(0.0, 0.0, -1.0);
    rolling.update(robot, no_filter, cloud);
    const auto* local_msg = rolling.message("world", 0.15);
    bool rolling_ok = std::fabs(rolling.centerWorld().x() - 12.0) < 1e-9
        && std::fabs(rolling.centerWorld().y() + 4.0) < 1e-9
        && std::fabs(local_msg->info.origin.position.x - 6.0) < 1e-9
        && std::fabs(local_msg->info.origin.position.y + 10.0) < 1e-9
        && local_msg->info.width == 240 && local_msg->info.height == 240
        && rolling.at(Eigen::Vector2d(14.0, -4.0)) < 0.0f
        && rolling.contains(Eigen::Vector2d(17.5, -4.0))
        && !rolling.contains(Eigen::Vector2d(18.1, -4.0))
        && !rolling.contains(Eigen::Vector2d(17.0, 1.0))
        && rolling.at(Eigen::Vector2d(17.0, 1.0)) == 0.0f
        && local_msg->data.front() == -1;

    // Move the robot by one metre. The obstacle remains at the same world
    // coordinate after the underlying local arrays shift by 20 cells.
    robot.x() = 13.0;
    cloud.clear();
    rolling.update(robot, no_filter, cloud);
    rolling_ok = rolling_ok
        && std::fabs(rolling.decode(rolling.encode(
            Eigen::Vector2d(14.0, -4.0))).x() - 14.0) < 1e-9
        && rolling.at(Eigen::Vector2d(14.0, -4.0)) < 0.0f;
    std::printf("rolling map   %s (center=(%.2f, %.2f))\n",
                rolling_ok ? "PASS" : "FAIL",
                rolling.centerWorld().x(), rolling.centerWorld().y());

    // Regression: a large custom world can have no obstacle inside the local
    // window. A far lidar return marks a known-free ray, and message() must be
    // able to read a valid all-free ESDF on the very first scan.
    Map free_window(12.0, 12.0, 0.05, 6.0);
    pcl::PointCloud<pcl::PointXYZ> far_cloud;
    far_cloud.emplace_back(35.0f, 0.0f, 0.0f);
    Eigen::Vector3d far_robot(-9.5, 0.0, 0.0);
    free_window.update(far_robot, no_filter, far_cloud);
    const auto* free_msg = free_window.message("world", 0.15);
    bool free_window_ok = !free_window.sdfMat().empty()
        && free_window.sdfMat().rows == 240
        && free_window.sdfMat().cols == 240
        && free_msg->data.size() == 57600
        && free_window.at(far_robot.head<2>()) > 5.9f
        && free_msg->data.front() == -1;
    std::printf("free window   %s (ESDF=%dx%d)\n",
                free_window_ok ? "PASS" : "FAIL",
                free_window.sdfMat().cols, free_window.sdfMat().rows);

    const double tol = 2e-3;  // patch bilinear + ONNX fp32 accumulation
    bool ok = (d_ctrl < tol) && rolling_ok && free_window_ok;
    std::printf("%s (tol=%.1e)\n", ok ? "PASS" : "FAIL", tol);
    return ok ? 0 : 1;
}
