/* @see include/planner/neural_policy.hpp */

#include <chrono>
#include <cmath>
#include <stdexcept>

#include "planner/neural_policy.hpp"

namespace diff_track
{
    namespace
    {
        using Clock = std::chrono::steady_clock;

        static inline double elapsed_ms(Clock::time_point start,
                                        Clock::time_point end)
        {
            return std::chrono::duration<double, std::milli>(end - start).count();
        }
    }

    static inline double wrap_angle(double a)
    {
        return std::atan2(std::sin(a), std::cos(a));
    }

    NeuralPolicy::NeuralPolicy(const std::string& onnx_path,
                               double dt, double max_vel, double max_omega,
                               double omega_sign, int intra_op_threads)
        : dt_(dt), max_vel_(max_vel), max_omega_(max_omega),
          omega_sign_(omega_sign),
          env_(ORT_LOGGING_LEVEL_WARNING, "neural_policy"),
          mem_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        if(intra_op_threads <= 0)
            throw std::invalid_argument(
                "neural.onnx_threads must be a positive integer");
        opts_.SetIntraOpNumThreads(intra_op_threads);
        opts_.SetInterOpNumThreads(1);
        opts_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = std::make_unique<Ort::Session>(env_, onnx_path.c_str(), opts_);

        // Fixed input/output names matching export_onnx.py.
        in_names_str_ = {"esdf", "robot_state", "target_rel",
                         "history", "visibility_ray", "init_pose"};
        out_names_str_ = {"controls", "positions"};
        for (auto& s : in_names_str_) in_names_.push_back(s.c_str());
        for (auto& s : out_names_str_) out_names_.push_back(s.c_str());
    }

    // esdf (1,1,80,80): warpAffine a robot-centered, -yaw-rotated patch out of
    // the full ESDF, exactly like Python extract_local_esdf (scale=1.0,
    // INTER_LINEAR, border -1). The tracker Map ESDF is 0.05 m/px, offset 200.
    void NeuralPolicy::build_esdf_patch(const Map* map, double x, double y,
                                        double yaw, std::vector<float>& out) const
    {
        const cv::Mat& sdf = map->sdfMat();  // CV_32F, full grid
        const cv::Point2d robot_grid = map->grid(Eigen::Vector2d(x, y));
        const double cx = robot_grid.x;
        const double cy = robot_grid.y;
        const double half = LOCAL_PX / 2.0;

        cv::Mat rot = cv::getRotationMatrix2D(
            cv::Point2f(static_cast<float>(cx), static_cast<float>(cy)),
            -yaw * 180.0 / PI, 1.0);
        rot.at<double>(0, 2) += half - cx;
        rot.at<double>(1, 2) += half - cy;

        cv::Mat patch;
        cv::warpAffine(sdf, patch, rot, cv::Size(LOCAL_PX, LOCAL_PX),
                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(-1.0));

        out.resize(LOCAL_PX * LOCAL_PX);
        for (int r = 0; r < LOCAL_PX; ++r)
            for (int c = 0; c < LOCAL_PX; ++c)
                out[r * LOCAL_PX + c] = patch.at<float>(r, c);
    }

    // target_rel (6,): [rho, sin(t_rel), cos(t_rel), sin(t_v), cos(t_v), speed]
    void NeuralPolicy::build_target_rel(const Eigen::VectorXd& state,
                                        const Eigen::Vector2d& target_pos,
                                        const Eigen::Vector2d& target_vel,
                                        std::array<float, 6>& out) const
    {
        const double dx = target_pos.x() - state[0];
        const double dy = target_pos.y() - state[1];
        const double yaw = state[2];
        const double rho = std::hypot(dx, dy);
        const double t_rel = wrap_angle(std::atan2(dy, dx) - yaw);
        const double speed = std::hypot(target_vel.x(), target_vel.y());
        const double t_v = (speed > 1e-3)
            ? wrap_angle(std::atan2(target_vel.y(), target_vel.x()) - yaw)
            : 0.0;
        out[0] = static_cast<float>(rho);
        out[1] = static_cast<float>(std::sin(t_rel));
        out[2] = static_cast<float>(std::cos(t_rel));
        out[3] = static_cast<float>(std::sin(t_v));
        out[4] = static_cast<float>(std::cos(t_v));
        out[5] = static_cast<float>(speed);
    }

    // visibility_ray (8,): ESDF sampled on robot->target line, out-of-map -1.
    // Samples are converted from world coordinates into the current rolling
    // grid. The map center changes as the robot moves; tensor coordinates do
    // not.
    void NeuralPolicy::build_visibility_ray(const Map* map,
                                            const Eigen::Vector2d& robot_xy,
                                            const Eigen::Vector2d& target_xy,
                                            std::array<float, VIS_SAMPLES>& out) const
    {
        const cv::Mat& sdf = map->sdfMat();
        const double res = map->resolution;
        const int w = sdf.cols, h = sdf.rows;
        for (int i = 0; i < VIS_SAMPLES; ++i)
        {
            const double a = (VIS_SAMPLES == 1) ? 0.0
                                                : static_cast<double>(i) / (VIS_SAMPLES - 1);
            const double px_m = robot_xy.x() * (1 - a) + target_xy.x() * a;
            const double py_m = robot_xy.y() * (1 - a) + target_xy.y() * a;
            const Eigen::Vector2d center = map->centerWorld();
            const int px = static_cast<int>(
                (px_m - center.x() + map->offset.width * res) / res);
            const int py = static_cast<int>(
                (py_m - center.y() + map->offset.height * res) / res);
            out[i] = (map->contains(Eigen::Vector2d(px_m, py_m))
                      && px >= 0 && px < w && py >= 0 && py < h)
                ? sdf.at<float>(py, px)
                : -1.0f;
        }
    }

    std::vector<Eigen::Vector3d> NeuralPolicy::infer(
        const Eigen::VectorXd& state,
        const Eigen::Vector2d& target_pos,
        const Eigen::Vector2d& target_vel,
        const Map* map,
        Eigen::Vector2d& cmd)
    {
        Timing timing;
        const auto total_start = Clock::now();
        const double x = state[0], y = state[1], yaw = state[2];

        // Build the six inputs.
        auto stage_start = Clock::now();
        std::vector<float> esdf;
        build_esdf_patch(map, x, y, yaw, esdf);
        auto stage_end = Clock::now();
        timing.esdf_ms = elapsed_ms(stage_start, stage_end);

        std::array<float, 2> robot_state{
            static_cast<float>(state[3]), static_cast<float>(state[4])};

        stage_start = Clock::now();
        std::array<float, 6> target_rel;
        build_target_rel(state, target_pos, target_vel, target_rel);
        stage_end = Clock::now();
        timing.target_ms = elapsed_ms(stage_start, stage_end);

        std::array<float, HISTORY_DIM> history{};  // zeros

        stage_start = Clock::now();
        std::array<float, VIS_SAMPLES> vis_ray;
        build_visibility_ray(map, Eigen::Vector2d(x, y), target_pos, vis_ray);
        stage_end = Clock::now();
        timing.visibility_ms = elapsed_ms(stage_start, stage_end);

        std::array<float, 3> init_pose{
            static_cast<float>(x), static_cast<float>(y), static_cast<float>(yaw)};

        const std::array<int64_t, 4> esdf_shape{1, 1, LOCAL_PX, LOCAL_PX};
        const std::array<int64_t, 2> rs_shape{1, 2};
        const std::array<int64_t, 2> tr_shape{1, 6};
        const std::array<int64_t, 2> hist_shape{1, HISTORY_DIM};
        const std::array<int64_t, 2> vis_shape{1, VIS_SAMPLES};
        const std::array<int64_t, 2> ip_shape{1, 3};

        stage_start = Clock::now();
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_, esdf.data(), esdf.size(), esdf_shape.data(), esdf_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_, robot_state.data(), robot_state.size(), rs_shape.data(), rs_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_, target_rel.data(), target_rel.size(), tr_shape.data(), tr_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_, history.data(), history.size(), hist_shape.data(), hist_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_, vis_ray.data(), vis_ray.size(), vis_shape.data(), vis_shape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            mem_, init_pose.data(), init_pose.size(), ip_shape.data(), ip_shape.size()));
        stage_end = Clock::now();
        timing.tensor_ms = elapsed_ms(stage_start, stage_end);

        stage_start = Clock::now();
        auto outputs = session_->Run(Ort::RunOptions{nullptr},
                                     in_names_.data(), inputs.data(), inputs.size(),
                                     out_names_.data(), out_names_.size());
        stage_end = Clock::now();
        timing.ort_ms = elapsed_ms(stage_start, stage_end);

        stage_start = Clock::now();
        // controls (1,30,2), positions (1,31,3)
        const float* controls = outputs[0].GetTensorData<float>();
        const float* positions = outputs[1].GetTensorData<float>();

        double v0 = controls[0];
        double w0 = omega_sign_ * controls[1];
        v0 = std::max(std::min(v0, max_vel_), -max_vel_);
        w0 = std::max(std::min(w0, max_omega_), -max_omega_);
        cmd = Eigen::Vector2d(v0, w0);

        std::vector<Eigen::Vector3d> traj;
        traj.reserve(HORIZON + 1);
        for (int t = 0; t <= HORIZON; ++t)
            traj.emplace_back(positions[t * 3 + 0],
                              positions[t * 3 + 1],
                              positions[t * 3 + 2]);
        stage_end = Clock::now();
        timing.output_ms = elapsed_ms(stage_start, stage_end);
        timing.total_ms = elapsed_ms(total_start, stage_end);
        last_timing_ = timing;
        return traj;
    }
}
