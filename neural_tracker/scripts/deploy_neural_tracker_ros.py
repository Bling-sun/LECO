#!/usr/bin/env python3
"""
ROS2 deployment node for the neural tracker policy.

Typical closed-loop usage:
    ros2 launch simulator simulation.launch.py
    python3 scripts/deploy_neural_tracker_ros.py \
        --model models/v24_control_safety/best.pth \
        --cmd-topic /tracker/cmd_vel

The node subscribes to simulator topics and publishes a Twist command plus a
preview Path for RViz. It intentionally lives in scripts/ so it can be run
without rebuilding the ROS package.
"""
import argparse
import os
import sys
import time
from collections import deque


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def add_runtime_paths():
    """Make ROS system packages and the project venv visible in either Python."""
    pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        os.path.join(ROOT, ".venv", "lib", pyver, "site-packages"),
        f"/opt/ros/jazzy/lib/{pyver}/site-packages",
        f"/usr/local/lib/{pyver}/dist-packages",
        "/usr/lib/python3/dist-packages",
        f"/usr/lib/{pyver}/dist-packages",
    ]
    for path in reversed(candidates):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)


add_runtime_paths()

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from geometry_msgs.msg import PoseStamped as PathPoseStamped


from config import Config, restore_config_from_checkpoint  # noqa: E402
from data_generator import ScenarioGenerator  # noqa: E402
from network import TrackingPolicy  # noqa: E402


def quat_to_yaw(q):
    return float(np.arctan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z),
    ))


def yaw_to_quat(yaw):
    return 0.0, 0.0, float(np.sin(yaw * 0.5)), float(np.cos(yaw * 0.5))


def wrap_angle(angle):
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class TargetBezier:
    """Python mirror of diff_track::Bezier evaluation.

    The original tracker uses a separate `prediction` node that fits a Bezier
    curve to the target's observation history and publishes its control points
    on /target/bezier. That curve is parameterized over t in [-duration,
    duration]: the observation window (past) maps to [-duration, 0] and the
    FUTURE prediction to [0, duration]. Evaluating get(age) therefore yields the
    predicted target position `age` seconds after the last detection, which
    handles turning targets far better than the constant-velocity extrapolation
    the deploy node used before. Matches bezier.cpp get()/derivative().
    """

    def __init__(self, m, tau):
        self.n = 2 * m
        num = self.n + 1
        self.duration = tau * (num >> 1)
        self._fact = [1] * num
        for i in range(1, num):
            self._fact[i] = i * self._fact[i - 1]
        self.control = None  # (2, num), set from /target/bezier

    def set_control(self, control):
        self.control = np.asarray(control, dtype=np.float64).reshape(2, self.n + 1)

    def valid(self):
        return self.control is not None

    def _bernstein(self, t, i, m=None):
        if m is None:
            m = self.n
        if i < 0 or i > m:
            return 0.0
        d = self.duration
        return (self._fact[m] / (self._fact[i] * self._fact[m - i])
                * (d - t) ** (m - i) * (d + t) ** i / (2 * d) ** m)

    def get(self, t):
        t = max(min(t, self.duration), -self.duration)
        point = np.zeros(2, dtype=np.float64)
        for p in range(self.n + 1):
            point += self.control[:, p] * self._bernstein(t, p)
        return point

    def derivative(self, t):
        t = max(min(t, self.duration), -self.duration)
        point = np.zeros(2, dtype=np.float64)
        for p in range(self.n + 1):
            point += self.control[:, p] * (
                self._bernstein(t, p - 1, self.n - 1)
                - self._bernstein(t, p, self.n - 1)
            )
        return point * (self.n / (self.duration * 2))


def adapt_state_dict_for_target_dim(model, state_dict):
    key = "state_encoder.mlp.0.weight"
    model_state = model.state_dict()
    if key not in state_dict or key not in model_state:
        return state_dict
    old_weight = state_dict[key]
    new_weight = model_state[key]
    if old_weight.shape == new_weight.shape:
        return state_dict
    if old_weight.ndim != 2 or new_weight.ndim != 2 or old_weight.shape[0] != new_weight.shape[0]:
        return state_dict
    adapted = dict(state_dict)
    merged = new_weight.clone()
    cols = min(old_weight.shape[1], new_weight.shape[1])
    merged[:, :cols] = old_weight[:, :cols]
    if new_weight.shape[1] > old_weight.shape[1]:
        merged[:, old_weight.shape[1]:] = 0.0
    adapted[key] = merged
    return adapted


class NeuralTrackerNode(Node):
    def __init__(self, args):
        super().__init__("neural_tracker_policy")
        self.args = args
        model_path = args.model if os.path.isabs(args.model) else os.path.abspath(args.model)
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        self.cfg = restore_config_from_checkpoint(checkpoint, Config(), exclude={'DEVICE'})
        self.cfg.DEVICE = args.device
        if args.use_history is None:
            args.use_history = bool(getattr(self.cfg, 'USE_CONTROL_HISTORY', False))
        self.device = torch.device(args.device)
        self.scenario = ScenarioGenerator(self.cfg)

        self.model = self._load_model(model_path, checkpoint)
        self.last_odom = None
        self.last_target = None
        self.prev_target = None
        self.prev_target_time = None
        self.prev_target_vel = np.zeros(2, dtype=np.float32)
        # Bezier target-trajectory prediction from the original prediction node.
        self.target_bezier = TargetBezier(args.bezier_m, args.bezier_tau)
        self.bezier_time = None  # monotonic time of last /target/bezier update
        self.global_esdf = None
        self.occupancy = None
        self.map_center = np.zeros(2, dtype=np.float32)
        self.map_frame = args.frame_id
        self.last_map_shape = None
        self.history = deque(maxlen=self.cfg.HISTORY_DIM // 2)
        self.last_cmd = np.zeros(2, dtype=np.float32)
        self.last_log_time = 0.0
        self.last_lost_log_time = 0.0
        self.recovery_until = 0.0
        self.recovery_cmd = np.zeros(2, dtype=np.float32)
        self.search_turn_sign = 1.0
        self.launch_lost_active = False
        self.launch_lost_goal = None
        self.launch_lost_goal_index = 0
        self.launch_lost_last_goal_time = 0.0
        self.timing_last_log = time.monotonic()
        self.timing_samples = 0
        self.timing_step_ms = 0.0
        self.timing_target_ms = 0.0
        self.timing_input_ms = 0.0
        self.timing_forward_ms = 0.0
        self.timing_safety_ms = 0.0
        self.timing_post_ms = 0.0
        self.timing_publish_ms = 0.0

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Odometry, args.odom_topic, self._odom_cb, qos)
        self.create_subscription(PoseStamped, args.target_topic, self._target_cb, qos)
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, args.map_topic, self._map_cb, map_qos)
        if args.use_bezier_prediction:
            self.create_subscription(Path, args.bezier_topic, self._bezier_cb, qos)

        self.cmd_pub = self.create_publisher(Twist, args.cmd_topic, 10)
        self.path_pub = self.create_publisher(Path, args.path_topic, 10)
        self.raw_path_pub = self.create_publisher(Path, args.raw_path_topic, 10)

        self.timer = self.create_timer(1.0 / args.hz, self._step)
        self.get_logger().info(
            "Neural tracker ready: model=%s, cmd_topic=%s, path_topic=%s"
            % (args.model, args.cmd_topic, args.path_topic)
        )

    def _load_model(self, model_path, ckpt=None):
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)

        model = TrackingPolicy(self.cfg).to(self.device)
        ckpt = ckpt or torch.load(model_path, map_location=self.device, weights_only=False)
        state_dict = adapt_state_dict_for_target_dim(model, ckpt["model_state_dict"])
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        self.get_logger().info(
            "Loaded checkpoint epoch=%s best_val_loss=%s"
            % (ckpt.get("epoch", "N/A"), ckpt.get("best_val_loss", "N/A"))
        )
        return model

    def _record_timing(self, step_start, target_ms=0.0, input_ms=0.0,
                       forward_ms=0.0, safety_ms=0.0, post_ms=0.0,
                       publish_ms=0.0):
        self.timing_samples += 1
        self.timing_step_ms += (time.perf_counter() - step_start) * 1000.0
        self.timing_target_ms += target_ms
        self.timing_input_ms += input_ms
        self.timing_forward_ms += forward_ms
        self.timing_safety_ms += safety_ms
        self.timing_post_ms += post_ms
        self.timing_publish_ms += publish_ms
        self._log_timing_if_needed()

    def _log_timing_if_needed(self):
        interval = float(self.args.timing_interval)
        now = time.monotonic()
        if interval <= 0.0 or now - self.timing_last_log < interval:
            return

        n = max(self.timing_samples, 1)
        self.get_logger().info(
            "耗时统计：最近 %.1f 秒平均值，单位 ms。\n"
            "  纯网络控制循环总耗时=%.3f（一次 _step() 有效控制路径，样本数=%d）\n"
            "  目标预测/选择=%.3f，输入构造=%.3f，模型前向=%.3f，轨迹积分/安全检查=%.3f，后处理限速=%.3f，发布控制/轨迹=%.3f"
            % (
                interval,
                self.timing_step_ms / n,
                self.timing_samples,
                self.timing_target_ms / n,
                self.timing_input_ms / n,
                self.timing_forward_ms / n,
                self.timing_safety_ms / n,
                self.timing_post_ms / n,
                self.timing_publish_ms / n,
            )
        )
        self.timing_last_log = now
        self.timing_samples = 0
        self.timing_step_ms = 0.0
        self.timing_target_ms = 0.0
        self.timing_input_ms = 0.0
        self.timing_forward_ms = 0.0
        self.timing_safety_ms = 0.0
        self.timing_post_ms = 0.0
        self.timing_publish_ms = 0.0

    def _odom_cb(self, msg):
        pose = msg.pose.pose
        twist = msg.twist.twist
        yaw = quat_to_yaw(pose.orientation)
        world_vx = twist.linear.x
        world_vy = twist.linear.y
        body_v = np.cos(yaw) * world_vx + np.sin(yaw) * world_vy
        self.last_odom = np.array([
            pose.position.x,
            pose.position.y,
            yaw,
            body_v,
            twist.angular.z,
        ], dtype=np.float32)

    def _target_cb(self, msg):
        valid = msg.pose.orientation.w > 0.0
        now = time.monotonic()
        pos = np.array([msg.pose.position.x, msg.pose.position.y], dtype=np.float32)

        vel = np.zeros(2, dtype=np.float32)
        if valid and self.prev_target is not None and self.prev_target_time is not None:
            dt = max(now - self.prev_target_time, 1e-3)
            vel = (pos - self.prev_target) / dt

        self.last_target = {
            "pos": pos,
            "vel": vel,
            "valid": valid,
            "time": now,
        }
        if valid:
            self.prev_target = pos.copy()
            self.prev_target_time = now
            self.prev_target_vel = vel.copy()

    def _bezier_cb(self, msg):
        """Cache the target Bezier control points from the prediction node.

        prediction.cpp publishes n+1 control points while the target is
        tracked, and an empty Path when the target is lost (predictor reset).
        """
        num = self.target_bezier.n + 1
        if len(msg.poses) < num:
            # Empty/short message means the predictor was reset (target lost).
            self.target_bezier.control = None
            self.bezier_time = None
            return
        ctrl = np.zeros((2, num), dtype=np.float64)
        for p in range(num):
            ctrl[0, p] = msg.poses[p].pose.position.x
            ctrl[1, p] = msg.poses[p].pose.position.y
        self.target_bezier.set_control(ctrl)
        self.bezier_time = time.monotonic()

    def _map_cb(self, msg):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        raw = np.asarray(msg.data, dtype=np.int16).reshape(height, width)
        occupancy = np.where(raw >= 50, np.uint8(1), np.uint8(0))

        target_px = int(round(self.cfg.MAP_SIZE / self.cfg.MAP_RESOLUTION))
        if (occupancy.shape != (target_px, target_px)
                or not np.isclose(resolution, self.cfg.MAP_RESOLUTION)):
            raise ValueError(
                "Python deployment requires a %.1fx%.1f m map at %.3f m/cell; "
                "received %dx%d at %.3f m/cell. Use expert_collection.yaml "
                "or the C++/ONNX backend for larger maps."
                % (self.cfg.MAP_SIZE, self.cfg.MAP_SIZE, self.cfg.MAP_RESOLUTION,
                   width, height, resolution)
            )

        self.occupancy = occupancy
        self.global_esdf = self.scenario.compute_esdf(occupancy)
        self.map_center = np.array([
            msg.info.origin.position.x + width * resolution * 0.5,
            msg.info.origin.position.y + height * resolution * 0.5,
        ], dtype=np.float32)
        self.map_frame = msg.header.frame_id or self.args.frame_id
        shape_key = (width, height, round(resolution, 6))
        if self.last_map_shape != shape_key:
            self.get_logger().info(
                "Map received: %dx%d @ %.3fm, center=(%.2f, %.2f)"
                % (width, height, resolution, self.map_center[0], self.map_center[1])
            )
            self.last_map_shape = shape_key

    def _ready(self):
        return (
            self.last_odom is not None
            and self.last_target is not None
            and self.global_esdf is not None
        )

    def _zero_cmd(self):
        msg = Twist()
        self.cmd_pub.publish(msg)
        self.last_cmd[:] = 0.0

    def _publish_cmd(self, v, omega):
        cmd = np.array([v, omega], dtype=np.float32)
        cmd[0] = np.clip(cmd[0], -self.cfg.MAX_VEL, self.cfg.MAX_VEL)
        cmd[1] = np.clip(cmd[1], -self.cfg.MAX_OMEGA, self.cfg.MAX_OMEGA)
        msg = Twist()
        msg.linear.x = float(cmd[0])
        msg.angular.z = float(cmd[1])
        self.cmd_pub.publish(msg)
        self.last_cmd = cmd
        self.history.append(cmd.copy())
        return cmd

    def _target_bearing(self, target_pos):
        rel = target_pos - self.last_odom[:2]
        bearing = wrap_angle(np.arctan2(rel[1], rel[0]) - self.last_odom[2])
        return rel, bearing

    def _centered_odom_xy(self):
        return self.last_odom[:2] - self.map_center

    def _line_clear(self, start_xy, end_xy, margin):
        if self.global_esdf is None:
            return False
        steps = max(
            2,
            int(np.ceil(np.linalg.norm(end_xy - start_xy) / max(self.cfg.MAP_RESOLUTION, 1e-3))),
        )
        for alpha in np.linspace(0.0, 1.0, steps):
            point = start_xy * (1.0 - alpha) + end_xy * alpha
            if self._esdf_at(point) < margin:
                return False
        return True

    def _tracking_launch_observation_goals(self, target_centered, robot_centered):
        """Lightweight equivalent of HybridAStar::reacquire/proposal goals."""
        goals = []
        ob = self.cfg.OBSERVATION_DISTANCE
        safe_margin = max(self.cfg.ROBOT_RADIUS, self.args.safety_hard_stop_margin)
        rel = target_centered - robot_centered
        rel_norm = np.linalg.norm(rel)
        if rel_norm > 1e-3:
            base_angle = np.arctan2(rel[1], rel[0])
            base_xy = target_centered - ob * rel / rel_norm
            base_yaw = np.arctan2(target_centered[1] - base_xy[1], target_centered[0] - base_xy[0])
            if self._esdf_at(base_xy) > safe_margin:
                goals.append(np.array([base_xy[0], base_xy[1], base_yaw], dtype=np.float32))

            if not self._line_clear(base_xy, target_centered, 2.0 * self.cfg.ROBOT_RADIUS):
                for angle in np.arange(np.deg2rad(5.0), np.pi + 1e-6, np.deg2rad(5.0)):
                    for sign in (1.0, -1.0):
                        theta = base_angle + sign * angle
                        xy = target_centered + ob * np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
                        if (
                            self._esdf_at(xy) > safe_margin
                            and self._line_clear(xy, target_centered, 2.0 * self.cfg.ROBOT_RADIUS)
                        ):
                            yaw = np.arctan2(target_centered[1] - xy[1], target_centered[0] - xy[0])
                            goals.append(np.array([xy[0], xy[1], yaw], dtype=np.float32))
                            break
                    if len(goals) > 1:
                        break

        # Match the launch tracker's reacquire sweep over map cells after the
        # first proposal. Sorting by target distance approximates its Dijkstra
        # ordering while staying cheap enough for Python deployment.
        cell = max(self.args.tracking_launch_lost_cell, self.cfg.MAP_RESOLUTION)
        half = self.cfg.MAP_SIZE * 0.5
        xs = np.arange(-half + cell * 0.5, half, cell)
        ys = np.arange(-half + cell * 0.5, half, cell)
        cells = []
        for x in xs:
            for y in ys:
                xy = np.array([x, y], dtype=np.float32)
                if self._esdf_at(xy) <= safe_margin:
                    continue
                yaw = np.arctan2(target_centered[1] - y, target_centered[0] - x)
                score = np.linalg.norm(xy - target_centered)
                cells.append((score, np.array([x, y, yaw], dtype=np.float32)))
        cells.sort(key=lambda item: item[0])
        goals.extend(goal for _, goal in cells)
        return goals

    def _reset_tracking_launch_lost(self):
        self.launch_lost_active = False
        self.launch_lost_goal = None
        self.launch_lost_goal_index = 0
        self.launch_lost_last_goal_time = 0.0

    def _tracking_launch_lost_reacquire(self):
        if self.prev_target is None or self.prev_target_time is None:
            self._zero_cmd()
            return "tracking_launch_stop_no_last_target", np.zeros(2, dtype=np.float32), 0.0, float("nan")

        now = time.monotonic()
        age = now - self.prev_target_time
        if age > self.args.tracking_launch_lost_stale_sec:
            self._reset_tracking_launch_lost()
            self._zero_cmd()
            rel, bearing = self._target_bearing(self.prev_target)
            return "tracking_launch_stop_stale_target", rel, bearing, age

        if not self.launch_lost_active:
            # Mirror tracker.cpp: stop and clear velocity state before launching
            # reacquisition, so the first recovery command is not blended with
            # the last model command.
            self.last_cmd[:] = 0.0
            self.launch_lost_active = True
            self.launch_lost_goal = None
            self.launch_lost_goal_index = 0
            self.launch_lost_last_goal_time = 0.0

        # Use the Bezier prediction for the reacquire goal target when available
        # (turning targets), else fall back to constant-velocity extrapolation.
        if (
            self.args.use_bezier_prediction
            and self.target_bezier.valid()
            and self.bezier_time is not None
        ):
            target_world = self.target_bezier.get(
                min(age, self.target_bezier.duration)
            ).astype(np.float32)
        else:
            vel = self.prev_target_vel
            vel_norm = np.linalg.norm(vel)
            if vel_norm > self.args.predict_max_speed:
                vel = vel * (self.args.predict_max_speed / max(vel_norm, 1e-6))
            target_world = self.prev_target + vel * min(age, self.args.predict_horizon)
        target_centered = target_world - self.map_center
        robot_centered = self._centered_odom_xy()

        goals = self._tracking_launch_observation_goals(target_centered, robot_centered)
        if not goals:
            return self._search_target()

        goal_age = now - self.launch_lost_last_goal_time
        arrived = (
            self.launch_lost_goal is not None
            and np.linalg.norm(self.launch_lost_goal[:2] - robot_centered) < self.args.tracking_launch_lost_epsilon
        )
        if (
            self.launch_lost_goal is None
            or arrived
            or goal_age > self.args.tracking_launch_lost_goal_timeout
        ):
            idx = self.launch_lost_goal_index % len(goals)
            self.launch_lost_goal = goals[idx]
            self.launch_lost_goal_index += 1
            self.launch_lost_last_goal_time = now

        goal = self.launch_lost_goal
        rel_goal = goal[:2] - robot_centered
        bearing = wrap_angle(np.arctan2(rel_goal[1], rel_goal[0]) - self.last_odom[2])
        yaw_err = wrap_angle(goal[2] - self.last_odom[2])
        dist = np.linalg.norm(rel_goal)

        omega_ref = bearing if dist > self.args.tracking_launch_lost_epsilon else yaw_err
        omega = np.clip(
            self.args.omega_sign * self.args.tracking_launch_lost_kp * omega_ref,
            -self.args.reacquire_max_omega,
            self.args.reacquire_max_omega,
        )
        v = 0.0
        if dist > self.args.tracking_launch_lost_epsilon and abs(bearing) < self.args.reacquire_forward_bearing:
            v = self.args.reacquire_forward_v

        cmd, next_esdf, min_esdf = self._safe_lost_command(np.array([v, omega], dtype=np.float32))
        self._publish_cmd(cmd[0], cmd[1])
        if now - self.last_lost_log_time > self.args.log_interval:
            rel_target, target_bearing = self._target_bearing(target_world)
            self.get_logger().info(
                "Tracking-launch lost reacquire: age=%.1fs target_dist=%.2f target_bearing=%.2f "
                "goal=(%.2f, %.2f) goal_dist=%.2f bearing=%.2f cmd v=%.2f omega=%.2f "
                "next_esdf=%.2f min_esdf=%.2f"
                % (
                    age,
                    np.linalg.norm(rel_target),
                    target_bearing,
                    goal[0] + self.map_center[0],
                    goal[1] + self.map_center[1],
                    dist,
                    bearing,
                    cmd[0],
                    cmd[1],
                    next_esdf,
                    min_esdf,
                )
            )
            self.last_lost_log_time = now
        return "tracking_launch_reacquire", target_world - self.last_odom[:2], bearing, age

    def _safe_lost_command(self, cmd):
        init_pose = (
            self.last_odom[:3]
            - np.array([self.map_center[0], self.map_center[1], 0.0], dtype=np.float32)
        )
        controls = np.tile(cmd.reshape(1, 2), (self.cfg.HORIZON, 1))
        positions = self._integrate_controls(init_pose, controls)
        min_esdf = self._min_esdf(positions)
        next_esdf = self._esdf_at(positions[min(1, len(positions) - 1), :2])
        if self.args.safety_stop and (
            next_esdf < self.args.safety_hard_stop_margin
            or min_esdf < self.args.safety_hard_stop_margin
        ):
            safe_cmd = cmd.copy()
            safe_cmd[0] = min(0.0, safe_cmd[0])
            controls[:, :] = safe_cmd
            positions = self._integrate_controls(init_pose, controls)
            min_esdf = self._min_esdf(positions)
            next_esdf = self._esdf_at(positions[min(1, len(positions) - 1), :2])
            if next_esdf < self.args.safety_hard_stop_margin:
                safe_cmd[:] = 0.0
                safe_cmd[1] = np.sign(cmd[1]) * min(
                    abs(cmd[1]), self.args.reacquire_max_omega
                )
                controls[:, :] = safe_cmd
                positions = self._integrate_controls(init_pose, controls)
            cmd = safe_cmd
        self._publish_path(positions)
        return cmd, next_esdf, min_esdf

    def _reacquire_target(self):
        if self.prev_target is None or self.prev_target_time is None:
            self._zero_cmd()
            return "stop_no_last_target", np.zeros(2, dtype=np.float32), 0.0, float("nan")

        age = time.monotonic() - self.prev_target_time
        rel, bearing = self._target_bearing(self.prev_target)
        if age > self.args.reacquire_stale_sec:
            self._zero_cmd()
            return "stop_stale_target", rel, bearing, age

        omega = np.clip(
            self.args.omega_sign * self.args.reacquire_kp * bearing,
            -self.args.reacquire_max_omega,
            self.args.reacquire_max_omega,
        )
        v = 0.0
        if (
            self.args.reacquire_forward_v > 0.0
            and abs(bearing) < self.args.reacquire_forward_bearing
        ):
            v = self.args.reacquire_forward_v
        cmd, _, _ = self._safe_lost_command(np.array([v, omega], dtype=np.float32))
        self._publish_cmd(cmd[0], cmd[1])
        return "reacquire", rel, bearing, age

    def _search_target(self):
        if self.prev_target is not None:
            rel, bearing = self._target_bearing(self.prev_target)
            if abs(bearing) > self.args.search_bearing_deadband:
                self.search_turn_sign = np.sign(bearing)
        else:
            rel = np.zeros(2, dtype=np.float32)
            bearing = 0.0

        age = (
            time.monotonic() - self.prev_target_time
            if self.prev_target_time is not None else float("nan")
        )
        omega = self.args.omega_sign * self.search_turn_sign * abs(self.args.search_omega)
        if self.prev_target is not None and abs(bearing) > self.args.search_bearing_deadband:
            omega = np.clip(
                self.args.omega_sign * self.args.reacquire_kp * bearing,
                -self.args.reacquire_max_omega,
                self.args.reacquire_max_omega,
            )
        v = self.args.search_forward_v if abs(bearing) < self.args.search_forward_bearing else 0.0
        cmd, next_esdf, min_esdf = self._safe_lost_command(np.array([v, omega], dtype=np.float32))
        self._publish_cmd(cmd[0], cmd[1])
        if time.monotonic() - self.last_lost_log_time > self.args.log_interval:
            self.get_logger().info(
                "Lost-target search: age=%.1fs dist=%.2f bearing=%.2f cmd v=%.2f omega=%.2f next_esdf=%.2f min_esdf=%.2f"
                % (age, np.linalg.norm(rel), bearing, cmd[0], cmd[1], next_esdf, min_esdf)
            )
            self.last_lost_log_time = time.monotonic()
        return "search", rel, bearing, age

    def _history_array(self):
        values = list(self.history)
        while len(values) < self.cfg.HISTORY_DIM // 2:
            values.insert(0, np.zeros(2, dtype=np.float32))
        return np.asarray(values, dtype=np.float32).reshape(-1)

    def _visibility_ray(self, robot_xy, target_xy):
        samples = np.zeros(self.cfg.VISIBILITY_SAMPLES, dtype=np.float32)
        alphas = np.linspace(
            0.0, 1.0, self.cfg.VISIBILITY_SAMPLES + 2, dtype=np.float32
        )[1:-1]
        h, w = self.global_esdf.shape
        for i, alpha in enumerate(alphas):
            point = robot_xy * (1.0 - alpha) + target_xy * alpha
            px = int(np.floor((point[0] + self.cfg.MAP_SIZE * 0.5) / self.cfg.MAP_RESOLUTION))
            py = int(np.floor((point[1] + self.cfg.MAP_SIZE * 0.5) / self.cfg.MAP_RESOLUTION))
            if 0 <= px < w and 0 <= py < h:
                samples[i] = self.global_esdf[py, px]
            else:
                samples[i] = -1.0
        return samples

    def _prediction_target(self):
        if self.last_target is not None and self.last_target["valid"]:
            return self.last_target, "detected", 0.0
        if (
            self.args.lost_policy != "predict"
            or self.prev_target is None
            or self.prev_target_time is None
        ):
            return self.last_target, "lost", float("nan")

        age = time.monotonic() - self.prev_target_time
        if age > self.args.predict_stale_sec:
            return self.last_target, "stale", age

        # Prefer the Bezier prediction (matches the original tracker): the curve
        # is fit to the target's observation history and extrapolates turning
        # motion, unlike constant-velocity. Its parameter t runs [-duration,
        # duration] with the future in [0, duration]; get(age) is the predicted
        # position `age` seconds after the last detection.
        if (
            self.args.use_bezier_prediction
            and self.target_bezier.valid()
            and self.bezier_time is not None
        ):
            t = min(age, self.target_bezier.duration)
            pos = self.target_bezier.get(t).astype(np.float32)
            vel = self.target_bezier.derivative(t).astype(np.float32)
            vel_norm = np.linalg.norm(vel)
            if vel_norm > self.args.predict_max_speed:
                vel = vel * (self.args.predict_max_speed / max(vel_norm, 1e-6))
            return {
                "pos": pos,
                "vel": vel.astype(np.float32),
                "valid": True,
                "time": time.monotonic(),
                "predicted": True,
            }, "predicted_bezier", age

        vel_norm = np.linalg.norm(self.prev_target_vel)
        if vel_norm > self.args.predict_max_speed:
            vel = self.prev_target_vel * (self.args.predict_max_speed / max(vel_norm, 1e-6))
        else:
            vel = self.prev_target_vel

        pos = self.prev_target + vel * min(age, self.args.predict_horizon)
        return {
            "pos": pos.astype(np.float32),
            "vel": vel.astype(np.float32),
            "valid": True,
            "time": time.monotonic(),
            "predicted": True,
        }, "predicted", age

    def _build_inputs(self, target_override=None):
        odom = self.last_odom.copy()
        target = target_override if target_override is not None else self.last_target
        robot_state = odom.copy()
        robot_state[:2] -= self.map_center
        target_pos = target["pos"].copy() - self.map_center
        target_vel = target["vel"].copy()

        esdf_patch = self.scenario.extract_local_esdf(
            self.global_esdf, robot_state[0], robot_state[1], robot_state[2]
        )

        dx, dy = target_pos[0] - robot_state[0], target_pos[1] - robot_state[1]
        rho = np.hypot(dx, dy)
        theta_rel = wrap_angle(np.arctan2(dy, dx) - robot_state[2])
        speed = np.hypot(target_vel[0], target_vel[1])
        theta_v = wrap_angle(np.arctan2(target_vel[1], target_vel[0]) - robot_state[2]) if speed > 1e-3 else 0.0
        target_rel = np.array([
            rho, np.sin(theta_rel), np.cos(theta_rel),
            np.sin(theta_v), np.cos(theta_v), speed,
        ], dtype=np.float32)

        history = self._history_array() if self.args.use_history else np.zeros(self.cfg.HISTORY_DIM, dtype=np.float32)
        vis_ray = self._visibility_ray(robot_state[:2], target_pos)

        return {
            "esdf": torch.from_numpy(esdf_patch).unsqueeze(0).unsqueeze(0).float().to(self.device),
            "robot_state": torch.from_numpy(robot_state[3:]).unsqueeze(0).float().to(self.device),
            "target_rel": torch.from_numpy(target_rel).unsqueeze(0).float().to(self.device),
            "history": torch.from_numpy(history).unsqueeze(0).float().to(self.device),
            "visibility_ray": torch.from_numpy(vis_ray).unsqueeze(0).float().to(self.device),
            "init_pose": torch.from_numpy(robot_state[:3]).unsqueeze(0).float().to(self.device),
        }

    def _publish_path(self, positions):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        for row in positions:
            pose = PathPoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(row[0] + self.map_center[0])
            pose.pose.position.y = float(row[1] + self.map_center[1])
            qx, qy, qz, qw = yaw_to_quat(float(row[2]))
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            msg.poses.append(pose)
        self.path_pub.publish(msg)

    def _publish_raw_path(self, positions):
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.map_frame
        for row in positions:
            pose = PathPoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(row[0] + self.map_center[0])
            pose.pose.position.y = float(row[1] + self.map_center[1])
            qx, qy, qz, qw = yaw_to_quat(float(row[2]))
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            msg.poses.append(pose)
        self.raw_path_pub.publish(msg)

    def _integrate_controls(self, init_pose, controls):
        traj = np.zeros((len(controls) + 1, 3), dtype=np.float32)
        traj[0] = init_pose[:3]
        for i, (v, omega) in enumerate(controls):
            yaw = traj[i, 2]
            traj[i + 1, 0] = traj[i, 0] + self.cfg.DT * v * np.cos(yaw)
            traj[i + 1, 1] = traj[i, 1] + self.cfg.DT * v * np.sin(yaw)
            traj[i + 1, 2] = wrap_angle(traj[i, 2] + self.cfg.DT * omega)
        return traj

    def _min_esdf(self, positions):
        if self.global_esdf is None:
            return float("nan")
        h, w = self.global_esdf.shape
        values = []
        for xy in positions[:, :2]:
            px = int(np.floor((xy[0] + self.cfg.MAP_SIZE * 0.5) / self.cfg.MAP_RESOLUTION))
            py = int(np.floor((xy[1] + self.cfg.MAP_SIZE * 0.5) / self.cfg.MAP_RESOLUTION))
            if 0 <= px < w and 0 <= py < h:
                values.append(float(self.global_esdf[py, px]))
            else:
                values.append(-float("inf"))
        return min(values) if values else float("nan")

    def _esdf_at(self, xy):
        if self.global_esdf is None:
            return -float("inf")
        h, w = self.global_esdf.shape
        px = int(np.floor((xy[0] + self.cfg.MAP_SIZE * 0.5) / self.cfg.MAP_RESOLUTION))
        py = int(np.floor((xy[1] + self.cfg.MAP_SIZE * 0.5) / self.cfg.MAP_RESOLUTION))
        if 0 <= px < w and 0 <= py < h:
            return float(self.global_esdf[py, px])
        return -float("inf")

    def _clearance_gradient(self, xy):
        eps = self.cfg.MAP_RESOLUTION * 2.0
        gx = self._esdf_at(np.array([xy[0] + eps, xy[1]], dtype=np.float32)) - self._esdf_at(
            np.array([xy[0] - eps, xy[1]], dtype=np.float32)
        )
        gy = self._esdf_at(np.array([xy[0], xy[1] + eps], dtype=np.float32)) - self._esdf_at(
            np.array([xy[0], xy[1] - eps], dtype=np.float32)
        )
        return np.array([gx, gy], dtype=np.float32)

    def _recovery_turn(self):
        grad = self._clearance_gradient(self.last_odom[:2] - self.map_center)
        if np.linalg.norm(grad) < 1e-4:
            if self.prev_target is not None:
                _, bearing = self._target_bearing(self.prev_target)
                return np.sign(bearing) if abs(bearing) > 1e-3 else 1.0
            return 1.0

        grad_world_angle = np.arctan2(grad[1], grad[0])
        grad_bearing = wrap_angle(grad_world_angle - self.last_odom[2])
        return np.sign(grad_bearing) if abs(grad_bearing) > 1e-3 else 1.0

    def _publish_recovery_plan(self, cmd):
        controls = np.tile(cmd.reshape(1, 2), (self.cfg.HORIZON, 1))
        init_pose = (self.last_odom[:3] - np.array([self.map_center[0], self.map_center[1], 0.0], dtype=np.float32))
        self._publish_path(self._integrate_controls(init_pose, controls))

    def _start_recovery(self, next_esdf, model_min_esdf):
        turn = self._recovery_turn()
        cmd = np.array([
            -abs(self.args.recovery_back_v),
            turn * abs(self.args.recovery_omega),
        ], dtype=np.float32)
        self.recovery_cmd = cmd
        self.recovery_until = time.monotonic() + self.args.recovery_duration
        self._publish_cmd(cmd[0], cmd[1])
        self._publish_recovery_plan(cmd)
        self.get_logger().warn(
            "Safety recovery: next_esdf=%.2f model_min_esdf=%.2f cmd v=%.2f omega=%.2f"
            % (next_esdf, model_min_esdf, cmd[0], cmd[1])
        )

    def _step(self):
        if not self._ready():
            if time.monotonic() - self.last_log_time > 2.0:
                self.get_logger().info("Waiting for odom + target detection + map...")
                self.last_log_time = time.monotonic()
            return

        step_start = time.perf_counter()

        if time.monotonic() < self.recovery_until:
            self._publish_cmd(self.recovery_cmd[0], self.recovery_cmd[1])
            self._publish_recovery_plan(self.recovery_cmd)
            if time.monotonic() - self.last_log_time > self.args.log_interval:
                self.get_logger().info(
                    "mode=recovery cmd v=%.2f omega=%.2f current_esdf=%.2f"
                    % (
                        self.recovery_cmd[0],
                        self.recovery_cmd[1],
                        self._esdf_at(self.last_odom[:2] - self.map_center),
                    )
                )
                self.last_log_time = time.monotonic()
            return

        stage_start = time.perf_counter()
        target_for_model, target_mode, target_age = self._prediction_target()
        target_ms = (time.perf_counter() - stage_start) * 1000.0

        if not target_for_model or not target_for_model["valid"]:
            fallback = self.args.predict_fallback if self.args.lost_policy == "predict" else self.args.lost_policy
            if self.args.tracking_launch_lost_reacquire:
                mode, rel, bearing, age = self._tracking_launch_lost_reacquire()
            elif fallback == "reacquire":
                mode, rel, bearing, age = self._reacquire_target()
            elif fallback == "search":
                mode, rel, bearing, age = self._search_target()
            else:
                self._zero_cmd()
                rel, bearing, age = self._target_bearing(self.last_target["pos"])[0], 0.0, 0.0
                mode = "stop"
            if time.monotonic() - self.last_lost_log_time > self.args.log_interval:
                self.get_logger().info(
                    "Target lost/invalid, mode=%s dist=%.2f bearing=%.2f age=%.1fs"
                    % (mode, np.linalg.norm(rel), bearing, age)
                )
                self.last_lost_log_time = time.monotonic()
            return

        if target_mode == "detected":
            self._reset_tracking_launch_lost()

        stage_start = time.perf_counter()
        inputs = self._build_inputs(target_for_model)
        init_pose = inputs["init_pose"][0].detach().cpu().numpy()
        input_ms = (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs)
        forward_ms = (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        controls = outputs["controls"][0].detach().cpu().numpy()
        controls[:, 1] *= self.args.omega_sign
        raw_positions = outputs["positions"][0].detach().cpu().numpy()
        model_exec_positions = self._integrate_controls(init_pose, controls)
        v, omega = controls[0]

        next_esdf = self._esdf_at(model_exec_positions[min(1, len(model_exec_positions) - 1), :2])
        model_min_esdf = self._min_esdf(model_exec_positions)
        if self.args.safety_stop:
            next_xy = model_exec_positions[min(1, len(model_exec_positions) - 1), :2]
            next_esdf = self._esdf_at(next_xy)
            hard_stop_dist = self.args.safety_hard_stop_margin
            if next_esdf < hard_stop_dist:
                if self.args.recovery:
                    self._start_recovery(next_esdf, model_min_esdf)
                else:
                    self.get_logger().warn(
                        "Safety hard stop: next_esdf=%.2f hard_stop=%.2f model_min_esdf=%.2f"
                        % (next_esdf, hard_stop_dist, model_min_esdf)
                    )
                    self._zero_cmd()
                safety_ms = (time.perf_counter() - stage_start) * 1000.0
                self._record_timing(
                    step_start, target_ms, input_ms, forward_ms, safety_ms
                )
                return
        safety_ms = (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        alpha = float(self.args.cmd_alpha)
        cmd = alpha * np.array([v, omega], dtype=np.float32) + (1.0 - alpha) * self.last_cmd
        cmd[0] *= self.args.v_scale
        cmd[0] = np.clip(cmd[0], -self.cfg.MAX_VEL, self.cfg.MAX_VEL)
        cmd[1] = np.clip(cmd[1], -self.cfg.MAX_OMEGA, self.cfg.MAX_OMEGA)

        rel, bearing = self._target_bearing(target_for_model["pos"])
        dist_err = np.linalg.norm(rel) - self.cfg.OBSERVATION_DISTANCE
        target_speed = float(np.linalg.norm(target_for_model["vel"]))
        target_forward_speed = float(np.dot(
            target_for_model["vel"],
            np.array([np.cos(self.last_odom[2]), np.sin(self.last_odom[2])], dtype=np.float32)
        ))
        assisted_v = 0.0
        if (
            self.args.target_speed_assist
            and abs(bearing) < self.args.assist_max_bearing
            and dist_err > self.args.assist_deadband
        ):
            assisted_v = (
                self.args.assist_target_gain * target_speed
                + self.args.assist_dist_gain * dist_err
            )
            assisted_v = min(assisted_v, self.args.assist_max_v)
            cmd[0] = max(cmd[0], assisted_v)

        if (
            self.args.min_v_when_far > 0.0
            and dist_err > self.args.far_deadband
            and abs(bearing) < self.args.min_v_max_bearing
        ):
            cmd[0] = max(cmd[0], self.args.min_v_when_far)

        if self.args.obs_distance_guard and cmd[0] > 0.0:
            follow_feedforward = 0.0
            if self.args.obs_guard_target_feedforward:
                follow_feedforward = max(target_forward_speed, 0.0)
            max_forward_v = (
                follow_feedforward
                + self.args.obs_guard_kp * max(dist_err - self.args.obs_guard_deadband, 0.0)
            )
            max_forward_v = min(max_forward_v, self.args.obs_guard_max_v)
            if cmd[0] > max_forward_v:
                old_v = cmd[0]
                cmd[0] = max_forward_v
                self.get_logger().warn(
                    "Obs-distance guard clipped v %.3f -> %.3f dist_err=%.2f target_fwd=%.2f"
                    % (old_v, cmd[0], dist_err, target_forward_speed)
                )

        if (
            self.args.fov_guard
            and abs(bearing) > self.args.fov_guard_min_bearing
            and bearing * cmd[1] < 0.0
        ):
            guarded_omega = np.clip(
                self.args.fov_guard_kp * bearing,
                -self.args.fov_guard_max_omega,
                self.args.fov_guard_max_omega,
            )
            self.get_logger().warn(
                "FOV guard corrected omega %.3f -> %.3f for bearing %.3f"
                % (cmd[1], guarded_omega, bearing)
            )
            cmd[1] = guarded_omega

        exec_controls = controls.copy()
        exec_controls[0] = cmd
        exec_positions = self._integrate_controls(init_pose, exec_controls)
        exec_next_esdf = self._esdf_at(exec_positions[min(1, len(exec_positions) - 1), :2])
        exec_min_esdf = self._min_esdf(exec_positions)

        # Hard stop only on the EXECUTED next step, not the whole-trajectory min.
        # When following a wall-hugging target the far end of the 3s prediction
        # routinely dips low even though the immediate step is safe; gating the
        # hard stop on exec_min_esdf froze the robot every frame near walls
        # ("hesitation / can't get through"). The whole-trajectory clearance is
        # still handled by the softer slow-down below.
        if self.args.safety_stop and exec_next_esdf < self.args.safety_hard_stop_margin:
            if self.args.recovery:
                self._start_recovery(exec_next_esdf, exec_min_esdf)
            else:
                self.get_logger().warn(
                    "Safety final hard stop: exec_next_esdf=%.2f hard_stop=%.2f exec_min_esdf=%.2f model_min_esdf=%.2f"
                    % (
                        exec_next_esdf,
                        self.args.safety_hard_stop_margin,
                        exec_min_esdf,
                        model_min_esdf,
                    )
                )
                self._zero_cmd()
            post_ms = (time.perf_counter() - stage_start) * 1000.0
            self._record_timing(
                step_start, target_ms, input_ms, forward_ms, safety_ms, post_ms
            )
            return

        if self.args.safety_stop and exec_min_esdf < self.args.safety_slow_margin:
            denom = max(self.args.safety_slow_margin - self.args.safety_hard_stop_margin, 1e-3)
            scale = np.clip(
                (exec_min_esdf - self.args.safety_hard_stop_margin) / denom,
                self.args.safety_min_speed_scale,
                1.0,
            )
            if scale < 1.0:
                old_v = cmd[0]
                cmd[0] *= scale
                exec_controls[0] = cmd
                exec_positions = self._integrate_controls(init_pose, exec_controls)
                exec_next_esdf = self._esdf_at(exec_positions[min(1, len(exec_positions) - 1), :2])
                exec_min_esdf = self._min_esdf(exec_positions)
                self.get_logger().warn(
                    "Safety final slow: exec_min_esdf=%.2f scale=%.2f v %.3f -> %.3f"
                    % (exec_min_esdf, scale, old_v, cmd[0])
                )
        post_ms = (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        self._publish_cmd(cmd[0], cmd[1])
        self._publish_path(exec_positions)
        self._publish_raw_path(raw_positions)
        publish_ms = (time.perf_counter() - stage_start) * 1000.0
        self._record_timing(
            step_start, target_ms, input_ms, forward_ms, safety_ms, post_ms, publish_ms
        )

        if time.monotonic() - self.last_log_time > self.args.log_interval:
            target_centered = target_for_model["pos"] - self.map_center
            exec_final_dist = np.linalg.norm(exec_positions[-1, :2] - target_centered)
            raw_final_dist = np.linalg.norm(raw_positions[-1, :2] - target_centered)
            self.get_logger().info(
                "mode=%s age=%.1fs cmd v=%.3f omega=%.3f now_dist=%.2f bearing=%.2f "
                "target_speed=%.2f target_fwd=%.2f assist_v=%.2f exec_final=%.2f raw_final=%.2f "
                "next_esdf=%.2f model_min_esdf=%.2f exec_min_esdf=%.2f raw_min_esdf=%.2f"
                % (
                    target_mode, target_age, cmd[0], cmd[1], np.linalg.norm(rel), bearing,
                    target_speed, target_forward_speed, assisted_v,
                    exec_final_dist, raw_final_dist,
                    next_esdf, model_min_esdf,
                    self._min_esdf(exec_positions), self._min_esdf(raw_positions),
                )
            )
            self.last_log_time = time.monotonic()


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy neural tracker in ROS2 simulation.")
    parser.add_argument("--model", default="models/v24_control_safety/best.pth")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--hz", type=float, default=10.0)
    parser.add_argument("--timing-interval", type=float, default=2.0,
                        help="Seconds between average runtime timing logs; <=0 disables logs.")
    parser.add_argument("--cmd-topic", default="/neural_tracker/cmd_vel")
    parser.add_argument("--path-topic", default="/tracker/plan")
    parser.add_argument("--raw-path-topic", default="/neural_tracker/raw_plan")
    parser.add_argument("--odom-topic", default="/tracker/odom")
    parser.add_argument("--target-topic", default="/target/detection")
    parser.add_argument("--map-topic", default="/map")
    parser.add_argument("--frame-id", default="world")
    # Bezier target-trajectory prediction (from the original prediction node).
    # Subscribes to /target/bezier and evaluates the same curve as the C++
    # tracker to predict turning targets after detection loss. Enabled by
    # default; use --no-bezier-prediction to fall back to constant-velocity.
    parser.add_argument("--no-bezier-prediction", dest="use_bezier_prediction",
                        action="store_false",
                        help="Disable Bezier target prediction; use constant-velocity extrapolation.")
    parser.set_defaults(use_bezier_prediction=True)
    parser.add_argument("--bezier-topic", default="/target/bezier",
                        help="Topic of target Bezier control points from the prediction node.")
    parser.add_argument("--bezier-m", type=int, default=3,
                        help="Bezier m (must match prediction node predictor.yaml; n=2m).")
    parser.add_argument("--bezier-tau", type=float, default=0.5,
                        help="Bezier tau (must match prediction node predictor.yaml).")
    parser.add_argument("--cmd-alpha", type=float, default=1.0,
                        help="Low-pass coefficient for commands. 1.0 publishes raw model output.")
    parser.add_argument("--v-scale", type=float, default=1.0,
                        help="Scale model linear velocity at deployment time.")
    parser.add_argument("--min-v-when-far", type=float, default=0.0,
                        help="Minimum forward speed when target is farther than observation distance.")
    parser.add_argument("--far-deadband", type=float, default=0.2,
                        help="Distance margin above observation distance before min-v-when-far is applied.")
    parser.add_argument("--min-v-max-bearing", type=float, default=0.7,
                        help="Only apply min-v-when-far if target bearing magnitude is below this.")
    parser.add_argument("--target-speed-assist", action="store_true",
                        help="Use target speed and distance error to raise deployment linear speed.")
    parser.add_argument("--assist-target-gain", type=float, default=0.8,
                        help="Feed-forward gain on target forward speed.")
    parser.add_argument("--assist-dist-gain", type=float, default=0.8,
                        help="Proportional gain on distance above observation distance.")
    parser.add_argument("--assist-deadband", type=float, default=0.05,
                        help="Distance error deadband for target speed assist.")
    parser.add_argument("--assist-max-bearing", type=float, default=0.9,
                        help="Only apply target speed assist if target bearing magnitude is below this.")
    parser.add_argument("--assist-max-v", type=float, default=1.2,
                        help="Max assisted linear speed.")
    parser.add_argument("--no-obs-distance-guard", dest="obs_distance_guard", action="store_false",
                        help="Disable clipping forward speed near the desired observation distance.")
    parser.set_defaults(obs_distance_guard=True)
    parser.add_argument("--obs-guard-deadband", type=float, default=0.10,
                        help="Forward speed is clipped when distance error is below this margin.")
    parser.add_argument("--obs-guard-kp", type=float, default=0.8,
                        help="Max forward speed near target is kp * max(dist_err - deadband, 0).")
    parser.add_argument("--no-obs-guard-target-feedforward", dest="obs_guard_target_feedforward",
                        action="store_false",
                        help="Do not add target forward speed to obs-distance guard limit.")
    parser.set_defaults(obs_guard_target_feedforward=True)
    parser.add_argument("--obs-guard-max-v", type=float, default=2.0,
                        help="Max forward speed allowed by obs-distance guard after feed-forward.")
    parser.add_argument("--use-history", dest="use_history", action="store_true",
                        help="Feed previous commands (default follows checkpoint training config).")
    parser.add_argument("--no-history", dest="use_history", action="store_false",
                        help="Explicitly feed zero history.")
    parser.set_defaults(use_history=None)
    parser.add_argument("--safety-stop", action="store_true",
                        help="Publish zero command if the next predicted step is in collision.")
    parser.add_argument("--safety-hard-stop-margin", type=float, default=0.05,
                        help="Hard stop only if next-step ESDF is below this distance.")
    parser.add_argument("--safety-slow-margin", type=float, default=0.18,
                        help="Scale down speed if predicted trajectory min ESDF is below this. "
                             "Lowered from 0.25: near wall-hugging targets the 3s prediction "
                             "min ESDF is often <0.25, causing constant throttling/hesitation.")
    parser.add_argument("--safety-min-speed-scale", type=float, default=0.35,
                        help="Minimum speed scale used by safety slowdown.")
    parser.add_argument("--no-recovery", dest="recovery", action="store_false",
                        help="Disable backup-and-turn recovery on hard safety stop.")
    parser.set_defaults(recovery=True)
    parser.add_argument("--recovery-back-v", type=float, default=0.18,
                        help="Reverse speed used when stuck at an obstacle corner.")
    parser.add_argument("--recovery-omega", type=float, default=0.55,
                        help="Turn speed used during obstacle-corner recovery.")
    parser.add_argument("--recovery-duration", type=float, default=0.8,
                        help="Seconds to keep each recovery command.")
    parser.add_argument("--lost-policy", choices=["predict", "reacquire", "search", "stop"], default="predict",
                        help="What to do when target detection is invalid.")
    parser.add_argument("--predict-fallback", choices=["search", "reacquire", "stop"], default="search",
                        help="What to do after predict-stale-sec expires while lost-policy=predict.")
    parser.add_argument("--tracking-launch-lost-reacquire", action="store_true",
                        help="Use a Python equivalent of ros2 launch tracker tracking.launch.py lost-target "
                             "reacquisition: stop/reset command state, move toward a safe observation goal near "
                             "the last predicted target, then sweep safe map cells if needed.")
    parser.add_argument("--tracking-launch-lost-stale-sec", type=float, default=8.0,
                        help="Stop tracking-launch reacquire if the last valid target is older than this.")
    parser.add_argument("--tracking-launch-lost-cell", type=float, default=6.0,
                        help="Cell size for the map sweep used by tracking-launch lost reacquire.")
    parser.add_argument("--tracking-launch-lost-epsilon", type=float, default=0.25,
                        help="Goal arrival radius for tracking-launch lost reacquire.")
    parser.add_argument("--tracking-launch-lost-goal-timeout", type=float, default=3.0,
                        help="Seconds before switching to the next reacquire goal while target remains lost.")
    parser.add_argument("--tracking-launch-lost-kp", type=float, default=1.2,
                        help="Angular P gain used by tracking-launch lost reacquire.")
    parser.add_argument("--reacquire-kp", type=float, default=1.2,
                        help="Angular P gain toward the last valid target when detection is lost.")
    parser.add_argument("--reacquire-max-omega", type=float, default=0.6,
                        help="Max angular speed for lost-target reacquisition.")
    parser.add_argument("--reacquire-stale-sec", type=float, default=8.0,
                        help="Stop if the last valid target is older than this.")
    parser.add_argument("--reacquire-forward-v", type=float, default=0.12,
                        help="Small forward speed while reacquiring if facing the last target.")
    parser.add_argument("--reacquire-forward-bearing", type=float, default=0.45,
                        help="Only move forward during reacquire if target bearing is below this.")
    parser.add_argument("--search-omega", type=float, default=0.45,
                        help="Angular speed used for lost-target active search.")
    parser.add_argument("--search-forward-v", type=float, default=0.10,
                        help="Small forward speed during lost-target search when roughly facing last target.")
    parser.add_argument("--search-forward-bearing", type=float, default=0.65,
                        help="Only move forward during search if last target bearing is below this.")
    parser.add_argument("--search-bearing-deadband", type=float, default=0.12,
                        help="Bearing deadband before search direction is updated from last target.")
    parser.add_argument("--predict-horizon", type=float, default=1.5,
                        help="Max time horizon for lost-target constant-velocity prediction.")
    parser.add_argument("--predict-stale-sec", type=float, default=3.0,
                        help="Use predicted target for this many seconds after detection is lost.")
    parser.add_argument("--predict-max-speed", type=float, default=1.7,
                        help="Clamp predicted target speed.")
    parser.add_argument("--omega-sign", type=float, choices=[-1.0, 1.0], default=1.0,
                        help="Set to -1 if the deployed simulator turns opposite to model omega.")
    parser.add_argument("--no-fov-guard", dest="fov_guard", action="store_false",
                        help="Disable bearing-based correction that prevents turning away from target.")
    parser.set_defaults(fov_guard=True)
    parser.add_argument("--fov-guard-min-bearing", type=float, default=0.08,
                        help="Only correct omega if target bearing magnitude is above this.")
    parser.add_argument("--fov-guard-kp", type=float, default=1.0,
                        help="Angular P gain used by FOV guard.")
    parser.add_argument("--fov-guard-max-omega", type=float, default=0.8,
                        help="Max angular speed used by FOV guard.")
    parser.add_argument("--log-interval", type=float, default=1.0)
    args, unknown = parser.parse_known_args()
    if unknown:
        print("WARNING: ignored unknown arguments: %s" % " ".join(unknown), file=sys.stderr)
    return args


def main():
    args = parse_args()
    rclpy.init()
    node = NeuralTrackerNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            try:
                node._zero_cmd()
            except Exception as exc:
                node.get_logger().warn("Failed to publish final zero command: %s" % exc)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
