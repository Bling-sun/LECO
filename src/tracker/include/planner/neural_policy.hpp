/* Neural tracking policy: ONNX Runtime inference wrapper.
 *
 * Reproduces, bit-for-bit, the preprocessing of the Python deploy node
 * (neural_tracker/scripts/deploy_neural_tracker_ros.py) so the exported
 * policy.onnx produces the same controls in C++ as in PyTorch. Used by
 * tracker_neural (planning_neural.cpp) for the "target visible" branch;
 * the lost/reacquire path stays on the classical MPC.
 */
#pragma once

#include <array>
#include <string>
#include <vector>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include "planner/map.hpp"

namespace diff_track
{
    class NeuralPolicy
    {
        public:
            /* Model geometry / limits — must match config.py used at export. */
            static constexpr int LOCAL_PX = 80;
            static constexpr int HORIZON = 30;
            static constexpr int VIS_SAMPLES = 8;
            static constexpr int HISTORY_DIM = 8;

            struct Timing
            {
                double esdf_ms = 0.0;
                double target_ms = 0.0;
                double visibility_ms = 0.0;
                double tensor_ms = 0.0;
                double ort_ms = 0.0;
                double output_ms = 0.0;
                double total_ms = 0.0;
            };

            NeuralPolicy(const std::string& onnx_path,
                         double dt, double max_vel, double max_omega,
                         double omega_sign, int intra_op_threads);

            /* Run one inference. Inputs are in the map-centered world frame
             * (the tracker Map is already centered on the world origin).
             *   state       : [x, y, yaw, v, omega]
             *   target_pos  : predicted target world position (map-centered)
             *   target_vel  : predicted target world velocity
             *   map         : tracker perception map (provides the ESDF)
             * Returns the full integrated trajectory (HORIZON+1, 3) [x,y,yaw]
             * and fills `cmd` with the first-step [v, omega] to publish. */
            std::vector<Eigen::Vector3d> infer(const Eigen::VectorXd& state,
                                               const Eigen::Vector2d& target_pos,
                                               const Eigen::Vector2d& target_vel,
                                               const Map* map,
                                               Eigen::Vector2d& cmd);

            const Timing& last_timing() const { return last_timing_; }

        private:
            void build_esdf_patch(const Map* map, double x, double y, double yaw,
                                  std::vector<float>& out) const;
            void build_target_rel(const Eigen::VectorXd& state,
                                  const Eigen::Vector2d& target_pos,
                                  const Eigen::Vector2d& target_vel,
                                  std::array<float, 6>& out) const;
            void build_visibility_ray(const Map* map,
                                      const Eigen::Vector2d& robot_xy,
                                      const Eigen::Vector2d& target_xy,
                                      std::array<float, VIS_SAMPLES>& out) const;

            double dt_, max_vel_, max_omega_, omega_sign_;

            Ort::Env env_;
            Ort::SessionOptions opts_;
            std::unique_ptr<Ort::Session> session_;
            Ort::MemoryInfo mem_;

            std::vector<std::string> in_names_str_, out_names_str_;
            std::vector<const char*> in_names_, out_names_;
            Timing last_timing_;
    };
}
