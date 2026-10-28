#!/usr/bin/env python3
"""Online benchmark metrics for neural_tracker simulation experiments.

Per-goal mode (the default) treats every replayed Goal as one experiment.  A
target leaving the FOV starts a loss timer: recovery within ``lost_timeout`` is
counted as recovery, while continuous loss for that duration fails only the
current Goal.  The route then continues so an 80-Goal replay yields 80 trials.
Ctrl-C always writes a final report and marks a running trial incomplete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.utilities import remove_ros_args
from std_msgs.msg import Float64, Float64MultiArray, String, UInt8


BRANCH_NAMES = {0: "neural", 1: "traditional", 2: "stationary_hold"}
METRICS_STATUS_FORMAT = "neural-tracker-metrics-status"
METRICS_STATUS_VERSION = 1


def file_provenance(path: Optional[Path]) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    return {"path": str(resolved), "sha256": digest, "size_bytes": resolved.stat().st_size}


def percentile(values: list[float], percent: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "samples": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "p50": percentile(values, 50.0),
        "p95": percentile(values, 95.0),
        "max": max(values),
    }


class NeuralTrackerMetrics(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(
            "traditional_tracker_metrics"
            if args.tracker_family == "traditional"
            else "neural_tracker_metrics"
        )
        self.args = args
        self.started_wall = datetime.now(timezone.utc)
        self.started_monotonic = time.monotonic()
        self.provenance = {
            "algorithm_label": args.algorithm_label,
            "map": file_provenance(args.map_file),
            "policy": file_provenance(args.policy_file),
        }

        self.command_count = 0
        self.smoothness_sum = 0.0
        self.smoothness_transition_count = 0
        self.smoothness_linear_sum = 0.0
        self.smoothness_angular_sum = 0.0
        self.normalized_smoothness_sum = 0.0
        self.dynamic_sum = 0.0
        self.dynamic_transition_count = 0
        self.dynamic_linear_sum = 0.0
        self.dynamic_angular_sum = 0.0
        self.normalized_dynamic_sum = 0.0
        self.previous_commands: list[tuple[float, float]] = []
        self.last_command_time: Optional[float] = None
        self.command_periods: list[float] = []
        self.command_rows: list[tuple[int, float, float, float]] = []

        self.visibility_count = 0
        self.invisible_count = 0
        self.visibility_state: Optional[bool] = None
        self.loss_events = 0
        self.recovery_events = 0
        self.lost_episode_open = False
        self.visibility_loss_started_at: Optional[float] = None
        self.loss_failure_latched = False
        self.visibility_rows: list[tuple[int, float, int]] = []
        self.target_fov_rows: list[tuple[Any, ...]] = []
        self.target_fov_missing_pose_samples = 0

        self.planning_times_ms: list[float] = []
        self.planning_times_by_branch: dict[str, list[float]] = {
            name: [] for name in BRANCH_NAMES.values()
        }
        self.planning_rows: list[tuple[int, float, float]] = []
        self.branch_counts: Counter[str] = Counter()
        self.branch_rows: list[tuple[int, float, str]] = []
        self.last_recorded_branch: Optional[str] = None
        self.branch_switch_rows: list[tuple[Any, ...]] = []
        self.timeline_rows: list[tuple[Any, ...]] = []

        self.latest_detection_visible: Optional[bool] = None
        self.first_metrics_sequence: Optional[int] = None
        self.last_metrics_sequence: Optional[int] = None
        self.missed_metrics_samples = 0
        self.duplicate_metrics_samples = 0
        self.out_of_order_metrics_samples = 0
        self.metrics_stream_received = 0
        self.active_first_metrics_sequence: Optional[int] = None
        self.active_last_metrics_sequence: Optional[int] = None
        self.active_missed_metrics_samples = 0
        self.active_duplicate_metrics_samples = 0
        self.active_out_of_order_metrics_samples = 0

        self.tracker_xy: Optional[tuple[float, float]] = None
        self.tracker_yaw: Optional[float] = None
        self.tracker_pose_time: Optional[float] = None
        self.target_xy: Optional[tuple[float, float]] = None
        self.target_yaw: Optional[float] = None
        self.target_pose_time: Optional[float] = None
        self.static_map: Optional[OccupancyGrid] = None

        self.total_trials_started = 0
        self.trials: list[dict[str, Any]] = []
        self.active_trial: Optional[dict[str, Any]] = None
        self.measurement_active = False
        self.route_armed: Optional[dict[str, Any]] = None
        self.route_end_pending = False
        self.route_end_metadata: Optional[dict[str, Any]] = None
        self.auto_stop_requested = False
        self.truth_fallback_seen = False
        self.truth_fallback_active = False
        self.truth_fallback_trial_failure = False
        self.startup_ready = False
        self.startup_failure: Optional[str] = None
        self.runtime_input_failure: Optional[str] = None
        self.runtime_input_bad_since: Optional[float] = None
        self.input_received = {
            "metrics": False,
            "odom": False,
            "target_truth": False,
            "detection": False,
            "map": False,
        }
        self.last_ready_status_time = 0.0

        normal_qos = QoSProfile(depth=100)
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.metrics_status_publisher = self.create_publisher(
            String, args.metrics_status_topic, normal_qos
        )
        if args.legacy_separate_topics:
            self.create_subscription(Twist, args.cmd_topic, self.on_command, normal_qos)
            self.create_subscription(
                PoseStamped, args.detection_topic, self.on_detection, normal_qos
            )
            self.create_subscription(
                Float64, args.planning_time_topic, self.on_planning_time, normal_qos
            )
            self.create_subscription(
                UInt8, args.branch_topic, self.on_branch, normal_qos
            )
        else:
            self.create_subscription(
                Float64MultiArray,
                args.metrics_topic,
                self.on_metrics_sample,
                normal_qos,
            )
            self.create_subscription(
                PoseStamped,
                args.detection_topic,
                self.on_detection_state,
                normal_qos,
            )
        self.create_subscription(Odometry, args.odom_topic, self.on_odom, normal_qos)
        self.create_subscription(
            PoseStamped, args.target_truth_topic, self.on_target_truth, normal_qos
        )
        self.create_subscription(PoseStamped, args.goal_topic, self.on_goal, normal_qos)
        self.create_subscription(
            String, args.route_status_topic, self.on_route_status, normal_qos
        )
        self.create_subscription(
            String,
            args.observation_status_topic,
            self.on_observation_status,
            normal_qos,
        )
        self.create_subscription(OccupancyGrid, args.map_topic, self.on_map, map_qos)

        self.create_timer(0.05, self.evaluate_trial)
        self.create_timer(0.1, self.maybe_auto_stop)
        self.create_timer(0.25, self.check_startup_ready)
        self.create_timer(args.report_interval, self.periodic_report)
        self.get_logger().info(
            "评测已开始：标签=%s，模式=%s，%s，u=[v(m/s), omega(rad/s)]，Δt=%.3f s\n"
            "输出目录：%s"
            % (
                args.run_label,
                args.trial_mode,
                "旧版独立 topic 采样" if args.legacy_separate_topics
                else "同一控制周期原子采样",
                args.control_dt,
                args.output.parent,
            )
        )

    def elapsed(self, now: Optional[float] = None) -> float:
        return (time.monotonic() if now is None else now) - self.started_monotonic

    def input_topics(self) -> dict[str, str]:
        return {
            "metrics": self.args.cmd_topic
            if self.args.legacy_separate_topics else self.args.metrics_topic,
            "odom": self.args.odom_topic,
            "target_truth": self.args.target_truth_topic,
            "detection": self.args.detection_topic,
            "map": self.args.map_topic,
        }

    def publish_metrics_status(
        self,
        event: str,
        status: Optional[str] = None,
        reason: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        # SIGINT may invalidate the rclpy context before spin() raises
        # KeyboardInterrupt.  Status publication is auxiliary and must never
        # prevent the recorder from flushing its report during shutdown.
        if not rclpy.ok(context=self.context):
            return
        payload: dict[str, Any] = {
            "format": METRICS_STATUS_FORMAT,
            "version": METRICS_STATUS_VERSION,
            "event": event,
            "run_label": self.args.run_label,
            "trial_mode": self.args.trial_mode,
        }
        if status is not None:
            payload["status"] = status
        if reason is not None:
            payload["reason"] = reason
        route = metadata or self.route_armed or {}
        if route.get("sequence_sha256"):
            payload["sequence_sha256"] = route["sequence_sha256"]
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        try:
            self.metrics_status_publisher.publish(message)
        except Exception:
            if not rclpy.ok(context=self.context):
                return
            raise

    def check_startup_ready(self) -> None:
        if self.startup_failure is not None or self.runtime_input_failure is not None:
            return
        topics = self.input_topics()
        publisher_counts = {
            name: len(self.get_publishers_info_by_topic(topic))
            for name, topic in topics.items()
        }
        all_received = all(self.input_received.values())
        counts_valid = all(count == 1 for count in publisher_counts.values())
        if all_received and counts_valid:
            self.runtime_input_bad_since = None
            if not self.startup_ready:
                self.startup_ready = True
                self.get_logger().info(
                    "评测输入就绪：metrics、tracker odom、target truth、detection、map "
                    "均为单一发布源且已收到有效消息"
                )
            now = time.monotonic()
            if now - self.last_ready_status_time >= 0.5:
                self.publish_metrics_status("ready", status="ready")
                self.last_ready_status_time = now
            return

        missing = [name for name, received in self.input_received.items() if not received]
        bad_counts = [
            "%s=%d" % item
            for item in sorted(publisher_counts.items())
            if item[1] != 1
        ]
        details = []
        if missing:
            details.append("no message: " + ",".join(missing))
        if bad_counts:
            details.append("publisher count != 1: " + ",".join(bad_counts))
        if any(count > 1 for count in publisher_counts.values()):
            details.append("duplicate simulator/tracker likely running")
        failure = "; ".join(details) or "benchmark inputs not ready"

        if self.startup_ready:
            # DDS graph discovery can briefly report an endpoint transition.
            # Debounce it, but a sustained loss of any required publisher is
            # a runtime infrastructure failure and the route is no longer a
            # valid independent experiment.
            now = time.monotonic()
            if self.runtime_input_bad_since is None:
                self.runtime_input_bad_since = now
                self.get_logger().warning(
                    "评测运行中输入源暂时异常，等待 2.0 s 确认：%s" % failure
                )
                return
            if now - self.runtime_input_bad_since < 2.0:
                return
            self.runtime_input_failure = failure
            self.get_logger().error("评测运行中输入中断：%s" % failure)
            self.publish_metrics_status(
                "runtime_failed", status="invalid", reason=failure
            )
            self.auto_stop_requested = True
            return

        if self.elapsed() < self.args.startup_timeout:
            return
        self.startup_failure = failure
        self.get_logger().error("评测启动校验失败：%s" % self.startup_failure)
        self.publish_metrics_status(
            "startup_failed", status="invalid", reason=self.startup_failure
        )
        self.auto_stop_requested = True

    def on_command(self, msg: Twist) -> None:
        self.input_received["metrics"] = True
        if not self.measurement_active:
            return
        now = time.monotonic()
        self.record_command((float(msg.linear.x), float(msg.angular.z)), now)
        if self.latest_detection_visible is not None:
            self.record_target_fov(self.latest_detection_visible, now)

    def record_command(self, command: tuple[float, float], now: float) -> None:
        self.command_count += 1
        self.command_rows.append(
            (self.command_count, self.elapsed(now), command[0], command[1])
        )

        if self.previous_commands:
            dv = command[0] - self.previous_commands[-1][0]
            dw = command[1] - self.previous_commands[-1][1]
            linear_term = dv * dv
            angular_term = dw * dw
            self.smoothness_linear_sum += linear_term
            self.smoothness_angular_sum += angular_term
            self.smoothness_sum += linear_term + angular_term
            self.normalized_smoothness_sum += (
                (dv / self.args.linear_control_scale) ** 2
                + (dw / self.args.angular_control_scale) ** 2
            )
            self.smoothness_transition_count += 1
        if len(self.previous_commands) >= 2:
            older = self.previous_commands[-2]
            previous = self.previous_commands[-1]
            inv_dt2 = 1.0 / (self.args.control_dt * self.args.control_dt)
            av = (command[0] - 2.0 * previous[0] + older[0]) * inv_dt2
            aw = (command[1] - 2.0 * previous[1] + older[1]) * inv_dt2
            linear_term = av * av
            angular_term = aw * aw
            self.dynamic_linear_sum += linear_term
            self.dynamic_angular_sum += angular_term
            self.dynamic_sum += linear_term + angular_term
            self.normalized_dynamic_sum += (
                (av / self.args.linear_control_scale) ** 2
                + (aw / self.args.angular_control_scale) ** 2
            )
            self.dynamic_transition_count += 1

        self.previous_commands.append(command)
        if len(self.previous_commands) > 2:
            self.previous_commands.pop(0)
        if self.last_command_time is not None:
            self.command_periods.append(now - self.last_command_time)
        self.last_command_time = now

    def on_detection(self, msg: PoseStamped) -> None:
        self.input_received["detection"] = True
        visible = msg.pose.orientation.w > 0.0
        self.latest_detection_visible = visible
        if self.measurement_active:
            self.record_visibility(visible, time.monotonic())

    def on_detection_state(self, msg: PoseStamped) -> None:
        # Detection can run on its own timer.  Save only its latest state here;
        # N_all/N_occ are sampled exactly once by the next controller cycle.
        self.input_received["detection"] = True
        visible = msg.pose.orientation.w > 0.0
        self.latest_detection_visible = visible
        if visible and not self.measurement_active:
            self.loss_failure_latched = False
            self.visibility_loss_started_at = None
            self.lost_episode_open = False

    def record_visibility(self, visible: bool, now: float) -> None:
        self.visibility_count += 1
        if not visible:
            self.invisible_count += 1
        self.visibility_rows.append(
            (self.visibility_count, self.elapsed(now), int(visible))
        )

        starts_new_loss = (
            not visible
            and not self.loss_failure_latched
            and (
                self.visibility_state is True
                or (
                    self.active_trial is not None
                    and self.active_trial.get("lost_since") is None
                    and not self.lost_episode_open
                )
            )
        )
        if starts_new_loss:
            self.loss_events += 1
            self.lost_episode_open = True
            self.visibility_loss_started_at = now
            self.loss_failure_latched = False
            if self.active_trial is not None:
                self.active_trial["lost_since"] = now
        elif (
            self.visibility_state is False
            and visible
            and self.lost_episode_open
        ):
            lost_since = self.visibility_loss_started_at
            recovered_in_time = (
                lost_since is None
                or now - float(lost_since) <= self.args.lost_timeout
            )
            if recovered_in_time:
                self.recovery_events += 1
            elif self.active_trial is not None:
                self.active_trial["loss_timeout_exceeded"] = True
            self.lost_episode_open = False
            self.visibility_loss_started_at = None
            self.loss_failure_latched = False
        elif visible and self.loss_failure_latched:
            # A timed-out loss is not a recovery, but seeing the target again
            # releases the latch so a later FOV exit can form a new event.
            self.loss_failure_latched = False
            self.visibility_loss_started_at = None
        self.visibility_state = visible

        if self.active_trial is not None:
            if visible:
                self.active_trial["lost_since"] = None
            elif (
                self.active_trial["lost_since"] is None
                and not self.loss_failure_latched
            ):
                self.active_trial["lost_since"] = now

    def record_target_fov(self, detector_visible: bool, now: float) -> None:
        pose_values = (
            self.tracker_xy,
            self.tracker_yaw,
            self.tracker_pose_time,
            self.target_xy,
            self.target_pose_time,
        )
        if any(value is None for value in pose_values):
            self.target_fov_missing_pose_samples += 1
            return
        tracker_age = now - float(self.tracker_pose_time)
        target_age = now - float(self.target_pose_time)
        if (
            tracker_age > self.args.pose_max_age
            or target_age > self.args.pose_max_age
        ):
            self.target_fov_missing_pose_samples += 1
            return

        tracker_x, tracker_y = self.tracker_xy
        target_x, target_y = self.target_xy
        yaw = float(self.tracker_yaw)
        dx = target_x - tracker_x
        dy = target_y - tracker_y
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
        target_range = math.hypot(forward, lateral)
        bearing = math.atan2(lateral, forward)
        inside_fov = (
            target_range <= self.args.sensor_range
            and abs(bearing) <= math.radians(self.args.sensor_fov_deg) * 0.5
        )
        self.target_fov_rows.append(
            (
                self.command_count,
                self.elapsed(now),
                tracker_x,
                tracker_y,
                yaw,
                target_x,
                target_y,
                forward,
                lateral,
                target_range,
                bearing,
                math.degrees(bearing),
                int(detector_visible),
                int(inside_fov),
                tracker_age,
                target_age,
                self.args.sensor_fov_deg,
                self.args.sensor_range,
            )
        )

    def on_planning_time(self, msg: Float64) -> None:
        if self.measurement_active:
            self.record_planning_time(float(msg.data), time.monotonic())

    def record_planning_time(
        self, value: float, now: float, branch_name: Optional[str] = None
    ) -> None:
        if not math.isfinite(value) or value < 0.0:
            self.get_logger().warning("忽略非法规划耗时样本：%r" % value)
            return
        self.planning_times_ms.append(value)
        self.planning_rows.append(
            (len(self.planning_times_ms), self.elapsed(now), value)
        )
        if branch_name is not None:
            self.planning_times_by_branch.setdefault(branch_name, []).append(value)

    def on_branch(self, msg: UInt8) -> None:
        if self.measurement_active:
            self.record_branch(int(msg.data), time.monotonic())

    def record_branch(self, branch: int, now: float) -> bool:
        name = BRANCH_NAMES.get(branch, "unknown_%d" % branch)
        previous = self.last_recorded_branch
        changed = previous is not None and previous != name
        self.branch_counts[name] += 1
        self.branch_rows.append(
            (len(self.branch_rows) + 1, self.elapsed(now), name)
        )
        if changed:
            self.branch_switch_rows.append((
                len(self.branch_switch_rows) + 1,
                self.command_count,
                self.active_last_metrics_sequence,
                self.active_trial["id"] if self.active_trial is not None else None,
                self.elapsed(now),
                previous,
                name,
            ))
        self.last_recorded_branch = name
        return changed

    def record_timeline(
        self,
        sequence: int,
        now: float,
        command: tuple[float, float],
        visible: bool,
        planning_time_ms: float,
        branch: int,
        branch_name: str,
        branch_changed: bool,
    ) -> None:
        tracker_x = self.tracker_xy[0] if self.tracker_xy is not None else None
        tracker_y = self.tracker_xy[1] if self.tracker_xy is not None else None
        target_x = self.target_xy[0] if self.target_xy is not None else None
        target_y = self.target_xy[1] if self.target_xy is not None else None
        tracker_age = (
            now - self.tracker_pose_time
            if self.tracker_pose_time is not None else None
        )
        target_age = (
            now - self.target_pose_time
            if self.target_pose_time is not None else None
        )
        self.timeline_rows.append((
            self.command_count,
            sequence,
            self.active_trial["id"] if self.active_trial is not None else None,
            self.elapsed(now),
            tracker_x,
            tracker_y,
            self.tracker_yaw,
            target_x,
            target_y,
            self.target_yaw,
            tracker_age,
            target_age,
            command[0],
            command[1],
            int(visible),
            planning_time_ms,
            branch,
            branch_name,
            int(branch_changed),
        ))

    def on_metrics_sample(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 6:
            self.get_logger().warning(
                "忽略格式错误的 /tracker/metrics_sample：需要6个字段，实际%d"
                % len(msg.data)
            )
            return
        if any(not math.isfinite(float(value)) for value in msg.data[:6]):
            self.get_logger().warning("忽略包含非有限值的 /tracker/metrics_sample")
            return
        self.input_received["metrics"] = True
        sequence = int(round(msg.data[0]))
        self.metrics_stream_received += 1
        if self.last_metrics_sequence is not None:
            if sequence == self.last_metrics_sequence:
                self.duplicate_metrics_samples += 1
                if self.measurement_active:
                    self.active_duplicate_metrics_samples += 1
                return
            if sequence < self.last_metrics_sequence:
                self.out_of_order_metrics_samples += 1
                if self.measurement_active:
                    self.active_out_of_order_metrics_samples += 1
                return
            if sequence > self.last_metrics_sequence + 1:
                self.missed_metrics_samples += sequence - self.last_metrics_sequence - 1
        else:
            self.first_metrics_sequence = sequence
        self.last_metrics_sequence = sequence

        if not self.measurement_active:
            return

        if self.active_first_metrics_sequence is None:
            self.active_first_metrics_sequence = sequence
        if (
            self.active_last_metrics_sequence is not None
            and sequence > self.active_last_metrics_sequence + 1
        ):
            self.active_missed_metrics_samples += (
                sequence - self.active_last_metrics_sequence - 1
            )
        self.active_last_metrics_sequence = sequence

        now = time.monotonic()
        # In direct-target control the tracker correctly reports its truth
        # input as available, but visibility metrics must still represent the
        # independent raw detector. Startup readiness guarantees that its
        # latest state is available before a trial can begin.
        controller_visible = msg.data[5] > 0.5
        visible = (
            bool(self.latest_detection_visible)
            if self.args.direct_target
            and self.latest_detection_visible is not None
            else controller_visible
        )
        branch = int(round(msg.data[4]))
        branch_name = BRANCH_NAMES.get(branch, "unknown_%d" % branch)
        # All four records below now share one controller-cycle sample.
        command = (float(msg.data[1]), float(msg.data[2]))
        planning_time_ms = float(msg.data[3])
        self.record_command(command, now)
        self.record_planning_time(planning_time_ms, now, branch_name)
        branch_changed = self.record_branch(branch, now)
        self.record_visibility(visible, now)
        self.record_target_fov(visible, now)
        self.record_timeline(
            sequence,
            now,
            command,
            visible,
            planning_time_ms,
            branch,
            branch_name,
            branch_changed,
        )

    def on_odom(self, msg: Odometry) -> None:
        self.input_received["odom"] = True
        self.tracker_xy = (
            float(msg.pose.pose.position.x),
            float(msg.pose.pose.position.y),
        )
        self.tracker_speed = math.hypot(float(msg.twist.twist.linear.x), float(msg.twist.twist.linear.y))
        q = msg.pose.pose.orientation
        self.tracker_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.tracker_pose_time = time.monotonic()

    def on_target_truth(self, msg: PoseStamped) -> None:
        self.input_received["target_truth"] = True
        self.target_xy = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
        )
        q = msg.pose.orientation
        self.target_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.target_pose_time = time.monotonic()

    def on_map(self, msg: OccupancyGrid) -> None:
        self.input_received["map"] = True
        self.static_map = msg
        self.get_logger().info(
            "已接收静态地图：%d x %d，分辨率 %.3f m/cell"
            % (msg.info.width, msg.info.height, msg.info.resolution)
        )

    def start_trial(self, msg: PoseStamped, now: float) -> None:
        self.truth_fallback_trial_failure = False
        # Each active segment has its own finite-difference history. Otherwise
        # an idle gap or simulator reset creates a fictitious command jump.
        self.previous_commands.clear()
        self.last_command_time = None
        self.active_first_metrics_sequence = None
        self.active_last_metrics_sequence = None
        self.active_missed_metrics_samples = 0
        self.active_duplicate_metrics_samples = 0
        self.active_out_of_order_metrics_samples = 0
        self.measurement_active = True
        self.total_trials_started += 1
        open_initial_loss = (
            self.latest_detection_visible is False
            and not self.lost_episode_open
            and not self.loss_failure_latched
        )
        initial_lost_since = (
            self.visibility_loss_started_at
            if self.lost_episode_open else (now if open_initial_loss else None)
        )
        self.active_trial = {
            "id": self.total_trials_started,
            "goal_x": float(msg.pose.position.x),
            "goal_y": float(msg.pose.position.y),
            "start_time": now,
            "lost_since": initial_lost_since,
            "loss_timeout_exceeded": False,
            "truth_fallback_used": self.truth_fallback_active,
            "far_since": None,
            "near_since": None,
            "max_tracking_distance_m": None,
            "start_visibility_count": self.visibility_count,
            "start_invisible_count": self.invisible_count,
            "start_loss_events": self.loss_events,
            "start_recovery_events": self.recovery_events,
            "start_branch_counts": dict(self.branch_counts),
            "start_command_count": self.command_count,
            "start_planning_count": len(self.planning_times_ms),
            "start_branch_count": len(self.branch_rows),
            "start_fov_count": len(self.target_fov_rows),
            "start_fov_missing": self.target_fov_missing_pose_samples,
            "goal_count": 1,
            "replay_finished": False,
            "route_metadata": dict(self.route_armed or {}),
        }
        if open_initial_loss:
            self.loss_events += 1
            self.lost_episode_open = True
            self.visibility_loss_started_at = now
        if self.route_end_pending:
            self.mark_route_replay_finished_if_complete()
        if self.args.trial_mode == "route":
            self.route_armed = None
        self.get_logger().info(
            "试次 %d 开始：模式=%s，目标点 (%.3f, %.3f)"
            % (
                self.total_trials_started,
                self.args.trial_mode,
                self.active_trial["goal_x"],
                self.active_trial["goal_y"],
            )
        )

    def on_goal(self, msg: PoseStamped) -> None:
        now = time.monotonic()
        if not self.startup_ready:
            self.get_logger().error("输入尚未通过就绪校验，拒绝启动试次")
            self.startup_failure = "goal_received_before_metrics_ready"
            self.publish_metrics_status(
                "startup_failed", status="invalid", reason=self.startup_failure
            )
            self.auto_stop_requested = True
            return
        if self.args.trial_mode == "per-goal":
            if self.active_trial is not None:
                route = self.active_trial.get("route_metadata", {})
                if route.get("timing") == "reached":
                    # In reached-timing benchmark mode a new Goal is published
                    # only after the target reached the previous waypoint.
                    if (
                        self.trial_loss_timed_out(self.active_trial, now)
                        or self.active_trial.get("truth_fallback_used", False)
                    ):
                        self.fail_active_trial_for_fov_timeout(now)
                    else:
                        self.finish_trial("success", "target_reached_goal", now)
                else:
                    self.finish_trial("failure", "preempted_by_new_goal", now)
            self.start_trial(msg, now)
            return

        if self.active_trial is None:
            self.start_trial(msg, now)
            return

        # Route mode: intermediate Goal messages redirect the target but do not
        # represent tracker failures. Only the final goal after a route-end
        # status can complete the route successfully.
        self.active_trial["goal_x"] = float(msg.pose.position.x)
        self.active_trial["goal_y"] = float(msg.pose.position.y)
        self.active_trial["goal_count"] += 1
        if self.route_end_pending:
            self.mark_route_replay_finished_if_complete()

    def mark_route_replay_finished_if_complete(self) -> None:
        if self.active_trial is None or not self.route_end_pending:
            return
        metadata = self.route_end_metadata or self.route_armed or {}
        expected = int(metadata.get("goal_count", 0))
        received = int(self.active_trial.get("goal_count", 0))
        if expected > 0 and received < expected:
            return
        self.active_trial["replay_finished"] = True
        self.active_trial["route_metadata"].update(metadata)
        self.route_end_pending = False
        self.route_end_metadata = None

    def on_route_status(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("忽略非法路线状态 JSON")
            return
        if (
            payload.get("format") != "neural-tracker-route-status"
            or payload.get("version") != 1
        ):
            return
        event = payload.get("event")
        now = time.monotonic()
        if event == "start":
            if not self.startup_ready:
                self.startup_failure = "route_started_before_metrics_ready"
                self.publish_metrics_status(
                    "startup_failed",
                    status="invalid",
                    reason=self.startup_failure,
                    metadata=payload,
                )
                self.auto_stop_requested = True
                return
            replay_goal_tolerance = float(payload.get("goal_tolerance_m", 0.0))
            if not math.isclose(
                replay_goal_tolerance,
                self.args.goal_tolerance,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                self.startup_failure = (
                    "goal_tolerance_mismatch(metrics=%.6f,replay=%.6f)"
                    % (self.args.goal_tolerance, replay_goal_tolerance)
                )
                self.publish_metrics_status(
                    "startup_failed",
                    status="invalid",
                    reason=self.startup_failure,
                    metadata=payload,
                )
                self.auto_stop_requested = True
                return
            if self.active_trial is not None:
                self.finish_trial("failure", "route_restarted", now)
            self.route_armed = dict(payload)
            self.route_end_pending = False
            self.route_end_metadata = None
            self.get_logger().info(
                "已接收路线开始：%d goals，sha256=%s"
                % (payload.get("goal_count", 0), payload.get("sequence_sha256", "?"))
            )
        elif event == "end":
            if self.args.trial_mode == "per-goal":
                if self.active_trial is not None:
                    # The replay publishes route end only after the final
                    # target waypoint has been reached.
                    if (
                        self.trial_loss_timed_out(self.active_trial, now)
                        or self.active_trial.get("truth_fallback_used", False)
                    ):
                        self.fail_active_trial_for_fov_timeout(now)
                    else:
                        self.finish_trial("success", "target_reached_goal", now)
                self.route_armed = None
                self.route_end_pending = False
                self.route_end_metadata = None
                if self.args.auto_stop:
                    self.auto_stop_requested = True
                return
            self.route_end_pending = True
            self.route_end_metadata = dict(payload)
            if self.active_trial is not None:
                self.mark_route_replay_finished_if_complete()
                self.evaluate_trial()
        elif event == "abort" and self.active_trial is not None:
            self.finish_trial("incomplete", "route_replay_aborted", now)

    def on_observation_status(self, msg: String) -> None:
        if msg.data == "fov_detection_recovered":
            self.truth_fallback_active = False
            return
        if msg.data != "truth_fallback_started":
            return
        self.truth_fallback_seen = True
        self.truth_fallback_active = True
        if self.active_trial is not None:
            self.truth_fallback_trial_failure = True
            self.active_trial["truth_fallback_used"] = True
            self.get_logger().warning(
                "target 已触发真值恢复；当前 Goal 按 FOV 连续丢失失败，路线继续"
            )
            self.fail_active_trial_for_fov_timeout(time.monotonic())

    def maybe_auto_stop(self) -> None:
        if self.auto_stop_requested and rclpy.ok():
            rclpy.shutdown()

    def tracker_in_collision(self) -> bool:
        if self.static_map is None or self.tracker_xy is None:
            return False
        grid = self.static_map
        resolution = float(grid.info.resolution)
        if resolution <= 0.0 or grid.info.width == 0 or grid.info.height == 0:
            return False

        origin = grid.info.origin
        q = origin.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        dx = self.tracker_xy[0] - origin.position.x
        dy = self.tracker_xy[1] - origin.position.y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        center_x = int(math.floor(local_x / resolution))
        center_y = int(math.floor(local_y / resolution))
        radius_cells = max(0, int(math.ceil(self.args.robot_radius / resolution)))

        for offset_y in range(-radius_cells, radius_cells + 1):
            for offset_x in range(-radius_cells, radius_cells + 1):
                if math.hypot(offset_x * resolution, offset_y * resolution) > (
                    self.args.robot_radius + 0.5 * resolution
                ):
                    continue
                cell_x = center_x + offset_x
                cell_y = center_y + offset_y
                if (
                    cell_x < 0
                    or cell_y < 0
                    or cell_x >= grid.info.width
                    or cell_y >= grid.info.height
                ):
                    return True
                value = grid.data[cell_y * grid.info.width + cell_x]
                if value >= self.args.occupied_threshold:
                    return True
        return False

    def trial_loss_timed_out(self, trial: dict[str, Any], now: float) -> bool:
        if getattr(self.args, "direct_target", False):
            return False
        lost_since = trial.get("lost_since")
        return bool(
            trial.get("loss_timeout_exceeded", False)
            or (
                lost_since is not None
                and now - float(lost_since) >= self.args.lost_timeout
            )
        )

    def fail_active_trial_for_fov_timeout(self, now: float) -> None:
        # Latch this continuous loss until the detector sees the target again,
        # so one FOV-loss episode contributes at most one failed Goal.
        self.loss_failure_latched = True
        self.lost_episode_open = False
        self.visibility_loss_started_at = None
        self.finish_trial("failure", "target_lost_over_3s", now)

    def evaluate_trial(self) -> None:
        if self.active_trial is None:
            return
        now = time.monotonic()
        trial = self.active_trial

        # A goal callback and the 20 Hz evaluation timer can run before the
        # first 10 Hz atomic controller sample. Never create a zero-sample
        # invalid trial from that scheduling window; sustained input loss is
        # handled separately by check_startup_ready().
        if getattr(self, "active_last_metrics_sequence", 0) is None:
            return

        if self.trial_loss_timed_out(trial, now):
            self.fail_active_trial_for_fov_timeout(now)
            return

        tracking_distance = None
        if self.tracker_xy is not None and self.target_xy is not None:
            tracking_distance = math.dist(self.tracker_xy, self.target_xy)
            old_max = trial["max_tracking_distance_m"]
            trial["max_tracking_distance_m"] = (
                tracking_distance if old_max is None else max(old_max, tracking_distance)
            )
            if tracking_distance > self.args.max_tracking_distance:
                if trial["far_since"] is None:
                    trial["far_since"] = now
            else:
                trial["far_since"] = None
            if tracking_distance < self.args.min_tracking_distance:
                trial["near_since"] = trial["near_since"] or now
            else:
                trial["near_since"] = None

        if (trial.get("near_since") is not None and
                now - trial["near_since"] >= self.args.distance_timeout):
            self.finish_trial("failure", "target_tracker_too_close", now)
            return

        if self.tracker_in_collision():
            self.finish_trial("failure", "tracker_collision_or_stuck", now)
            return
        timeout = (
            self.args.route_timeout
            if self.args.trial_mode == "route"
            else self.args.trial_timeout
        )
        if now - trial["start_time"] >= timeout:
            self.finish_trial("failure", "trial_timeout", now)
            return

        if self.target_xy is not None:
            goal_distance = math.hypot(
                self.target_xy[0] - trial["goal_x"],
                self.target_xy[1] - trial["goal_y"],
            )
            route_can_finish = (
                self.args.trial_mode == "per-goal"
                or trial["replay_finished"]
            )
            if (
                route_can_finish
                and goal_distance <= self.args.goal_tolerance
                and (
                    getattr(self.args, "direct_target", False)
                    or self.latest_detection_visible is True
                )
            ):
                if trial.get("truth_fallback_used", False):
                    self.finish_trial("failure", "target_lost_over_3s", now)
                else:
                    self.finish_trial("success", "target_reached_goal", now)

    def trial_validation_errors(
        self, trial: dict[str, Any], require_complete_route: bool = False
    ) -> list[str]:
        k = self.command_count - trial["start_command_count"]
        visibility = self.visibility_count - trial["start_visibility_count"]
        planning = len(self.planning_times_ms) - trial["start_planning_count"]
        branches = len(self.branch_rows) - trial["start_branch_count"]
        fov_valid = len(self.target_fov_rows) - trial["start_fov_count"]
        fov_missing = (
            self.target_fov_missing_pose_samples - trial["start_fov_missing"]
        )
        errors = []
        if k <= 0:
            errors.append("K=0")
        if not all(self.input_received.values()):
            errors.append("required_input_missing")
        if len({k, visibility, planning, branches}) != 1:
            errors.append(
                "sample_count_mismatch(K=%d,visibility=%d,planning=%d,branch=%d)"
                % (k, visibility, planning, branches)
            )
        if self.active_missed_metrics_samples:
            errors.append("active_missed=%d" % self.active_missed_metrics_samples)
        if self.active_duplicate_metrics_samples:
            errors.append(
                "active_duplicate=%d" % self.active_duplicate_metrics_samples
            )
        if self.active_out_of_order_metrics_samples:
            errors.append(
                "active_out_of_order=%d" % self.active_out_of_order_metrics_samples
            )
        fov_total = fov_valid + fov_missing
        if fov_total != k:
            errors.append("fov_sample_count=%d!=K=%d" % (fov_total, k))
        missing_rate = fov_missing / k if k > 0 else 1.0
        if missing_rate > self.args.max_fov_missing_rate:
            errors.append(
                "fov_missing_rate=%.6f>%.6f"
                % (missing_rate, self.args.max_fov_missing_rate)
            )
        expected_goals = int(trial.get("route_metadata", {}).get("goal_count", 0))
        if (
            require_complete_route
            and expected_goals > 0
            and trial.get("goal_count") != expected_goals
        ):
            errors.append(
                "goal_count=%d!=expected=%d"
                % (trial.get("goal_count", 0), expected_goals)
            )
        return errors

    def finish_trial(
        self,
        status: str,
        reason: str,
        now: float,
        *,
        notify_ros: bool = True,
    ) -> None:
        if self.active_trial is None:
            return
        trial = self.active_trial
        start_branches = Counter(trial["start_branch_counts"])
        trial_branches = self.branch_counts - start_branches
        visibility_samples = self.visibility_count - trial["start_visibility_count"]
        invisible_samples = self.invisible_count - trial["start_invisible_count"]
        validation_errors = self.trial_validation_errors(
            trial,
            require_complete_route=(
                status == "success" and self.args.trial_mode == "route"
            ),
        )
        observed_status = status
        observed_reason = reason
        if status != "incomplete" and validation_errors:
            status = "invalid"
            reason = "invalid_metrics"
        record = {
            "id": trial["id"],
            "goal": {"x": trial["goal_x"], "y": trial["goal_y"]},
            "status": status,
            "reason": reason,
            "observed_status": observed_status,
            "observed_reason": observed_reason,
            "valid": not validation_errors and status != "incomplete",
            "validation_errors": validation_errors,
            "duration_s": now - trial["start_time"],
            "max_tracking_distance_m": trial["max_tracking_distance_m"],
            "visibility_samples": visibility_samples,
            "invisible_samples": invisible_samples,
            "occlusion_rate_percent": (
                100.0 * invisible_samples / visibility_samples
                if visibility_samples > 0
                else None
            ),
            "loss_events": self.loss_events - trial["start_loss_events"],
            "recovery_events": (
                self.recovery_events - trial["start_recovery_events"]
            ),
            "truth_fallback_used": bool(trial.get("truth_fallback_used", False)),
            "branch_usage": self.reported_branch_summary(trial_branches),
            "goal_count": trial["goal_count"],
            "command_samples": self.command_count - trial["start_command_count"],
            "route_metadata": trial["route_metadata"],
        }
        self.trials.append(record)
        self.active_trial = None
        self.measurement_active = False
        self.route_end_pending = False
        self.route_end_metadata = None
        log_message = (
            "试次 %d 结束：%s，原因=%s，耗时=%.2f s"
            % (record["id"], status, reason, record["duration_s"])
        )
        # rclpy Jazzy caches severity by Python call site.  Calling a bound
        # logger selected through a variable from the same source line first
        # as warning and later as info raises "severity cannot be changed".
        # Keep success and failure on distinct, fixed call sites.
        if not notify_ros or not rclpy.ok(context=self.context):
            print(log_message, flush=True)
        elif status == "success":
            self.get_logger().info(log_message)
        else:
            self.get_logger().warning(log_message)
        self.write_report()
        if notify_ros:
            self.publish_metrics_status(
                "trial_end",
                status=status,
                reason=reason,
                metadata=trial.get("route_metadata", {}),
            )
        if self.args.auto_stop and self.args.trial_mode == "route":
            self.auto_stop_requested = True

    @staticmethod
    def branch_summary(counts: Counter[str]) -> dict[str, Any]:
        neural = counts.get("neural", 0)
        traditional = counts.get("traditional", 0)
        hold = counts.get("stationary_hold", 0)
        total = sum(counts.values())
        algorithm_total = neural + traditional
        return {
            "total_control_cycles": total,
            "counts": dict(sorted(counts.items())),
            "algorithm_cycles_excluding_hold": algorithm_total,
            "neural_share_percent_excluding_hold": (
                100.0 * neural / algorithm_total if algorithm_total else None
            ),
            "traditional_share_percent_excluding_hold": (
                100.0 * traditional / algorithm_total if algorithm_total else None
            ),
            "neural_share_percent_of_all_cycles": (
                100.0 * neural / total if total else None
            ),
            "traditional_share_percent_of_all_cycles": (
                100.0 * traditional / total if total else None
            ),
            "stationary_hold_share_percent_of_all_cycles": (
                100.0 * hold / total if total else None
            ),
        }

    def reported_branch_summary(self, counts: Counter[str]) -> dict[str, Any]:
        summary = self.branch_summary(counts)
        if self.args.tracker_family == "traditional":
            for name in tuple(summary):
                if "share_percent" in name:
                    summary.pop(name)
        return summary

    def build_report(self, final: bool = False) -> dict[str, Any]:
        successful = sum(t["status"] == "success" for t in self.trials)
        failed_trials = [t for t in self.trials if t["status"] == "failure"]
        invalid_trials = [t for t in self.trials if t["status"] == "invalid"]
        incomplete = sum(t["status"] == "incomplete" for t in self.trials)
        if self.active_trial is not None:
            incomplete += 1
        completed = successful + len(failed_trials)
        failure_reasons = Counter(t["reason"] for t in failed_trials)
        expected_goal_trials = max(
            (
                int(t.get("route_metadata", {}).get("goal_count", 0))
                for t in self.trials
            ),
            default=0,
        )
        per_goal_count_valid = (
            self.args.trial_mode != "per-goal"
            or (
                expected_goal_trials > 0
                and len(self.trials) == expected_goal_trials
                and self.active_trial is None
            )
        )
        k = self.command_count
        all_visibility = self.visibility_count
        occlusion_rate = (
            100.0 * self.invisible_count / all_visibility
            if all_visibility > 0
            else None
        )
        return {
            # Legacy independent topics cannot provide controller-cycle atomic
            # visibility/timing/branch samples, so keep them out of the current
            # atomic schema.
            "schema_version": 2 if self.args.legacy_separate_topics else 4,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_label": self.args.run_label,
            "run_directory": str(self.args.output.parent),
            "provenance": self.provenance,
            "final": final,
            "elapsed_s": self.elapsed(),
            "definitions": {
                "control_vector": "u_k = [v, omega] = [linear.x, angular.z]",
                "control_smoothness": "sum(||u_(k+1)-u_k||_2^2)/(K-1)",
                "dynamic_feasibility": (
                    "sum(||(u_(k+1)-2u_k+u_(k-1))/dt^2||_2^2)/(K-2)"
                ),
                "occlusion_rate": (
                    "100*N_occ/N_all; invisible means /target/detection "
                    "pose.orientation.w <= 0 (includes FOV/range/obstacle loss)"
                ),
                "loss_event": "target transitions from inside to outside FOV",
                "recovery_event": (
                    "target returns inside FOV within lost_timeout_s"
                ),
                "goal_trial": (
                    "one replayed Goal is one trial; "
                    + (
                        "raw FOV loss is observed but does not terminate a Goal"
                        if self.args.direct_target
                        else "continuous FOV loss for lost_timeout_s fails only that Goal"
                    )
                ),
                "planning_time": (
                    "wall-clock duration of one %s ctrl() cycle, "
                    "from entry through command/path publication"
                    % (
                        "tracker_neural"
                        if self.args.tracker_family == "neural"
                        else "traditional tracker"
                    )
                ),
                "units": {
                    "v": "m/s",
                    "omega": "rad/s",
                    "smoothness_linear_component": "(m/s)^2",
                    "smoothness_angular_component": "(rad/s)^2",
                    "smoothness_raw_sum": "mixed units; report components separately",
                    "dynamic_linear_component": "(m/s^3)^2 = m^2/s^6",
                    "dynamic_angular_component": "(rad/s^3)^2 = rad^2/s^6",
                    "dynamic_raw_sum": "mixed units; report components separately",
                    "normalized_smoothness": "dimensionless",
                    "normalized_dynamic_feasibility": "s^-4",
                    "planning_time": "ms",
                    "occlusion_rate": "percent",
                },
                "branch_codes": (
                    BRANCH_NAMES
                    if self.args.tracker_family == "neural"
                    else {1: "traditional"}
                ),
            },
            "configuration": {
                "tracker_family": self.args.tracker_family,
                "direct_target": self.args.direct_target,
                "control_dt_s": self.args.control_dt,
                "goal_tolerance_m": self.args.goal_tolerance,
                "trial_timeout_s": self.args.trial_timeout,
                "route_timeout_s": self.args.route_timeout,
                "trial_mode": self.args.trial_mode,
                "lost_timeout_s": self.args.lost_timeout,
                "max_tracking_distance_m": self.args.max_tracking_distance,
                "distance_timeout_s": self.args.distance_timeout,
                "robot_radius_m": self.args.robot_radius,
                "occupied_threshold": self.args.occupied_threshold,
                "linear_control_scale_mps": self.args.linear_control_scale,
                "angular_control_scale_radps": self.args.angular_control_scale,
                "sensor_fov_deg": self.args.sensor_fov_deg,
                "sensor_range_m": self.args.sensor_range,
                "pose_max_age_s": self.args.pose_max_age,
                "max_fov_missing_rate": self.args.max_fov_missing_rate,
                "startup_timeout_s": self.args.startup_timeout,
                "legacy_separate_topics": self.args.legacy_separate_topics,
            },
            "sampling": {
                "mode": (
                    "legacy_independent_topics"
                    if self.args.legacy_separate_topics
                    else "atomic_controller_cycle"
                ),
                "metrics_topic": self.args.metrics_topic,
                "first_sequence": self.first_metrics_sequence,
                "last_sequence": self.last_metrics_sequence,
                "received_cycles": self.metrics_stream_received,
                "stream_received_cycles": self.metrics_stream_received,
                "measured_cycles": k,
                "active_first_sequence": self.active_first_metrics_sequence,
                "active_last_sequence": self.active_last_metrics_sequence,
                "active_missed_sequence_samples": self.active_missed_metrics_samples,
                "active_duplicate_sequence_samples": (
                    self.active_duplicate_metrics_samples
                ),
                "active_out_of_order_sequence_samples": (
                    self.active_out_of_order_metrics_samples
                ),
                "missed_sequence_samples": self.missed_metrics_samples,
                "duplicate_sequence_samples": self.duplicate_metrics_samples,
                "out_of_order_sequence_samples": self.out_of_order_metrics_samples,
                "target_fov_pose_samples": len(self.target_fov_rows),
                "target_fov_missing_or_stale_pose_samples": (
                    self.target_fov_missing_pose_samples
                ),
                "timeline_samples": len(self.timeline_rows),
                "branch_switches": len(self.branch_switch_rows),
            },
            "target_fov": {
                "samples": len(self.target_fov_rows),
                "missing_or_stale_pose_samples": self.target_fov_missing_pose_samples,
                "coordinate_frame": (
                    "tracker body frame: x=forward, y=left; sampled once per "
                    "active controller cycle from latest ground-truth poses"
                ),
                "sensor_fov_deg": self.args.sensor_fov_deg,
                "sensor_range_m": self.args.sensor_range,
                "raw_csv_suffix": "_target_fov.csv",
            },
            "control": {
                "samples_K": k,
                "smoothness_transition_samples": self.smoothness_transition_count,
                "smoothness": (
                    self.smoothness_sum / self.smoothness_transition_count
                    if self.smoothness_transition_count else None
                ),
                "smoothness_components": {
                    "linear_v": (
                        self.smoothness_linear_sum / self.smoothness_transition_count
                        if self.smoothness_transition_count else None
                    ),
                    "angular_omega": (
                        self.smoothness_angular_sum / self.smoothness_transition_count
                        if self.smoothness_transition_count else None
                    ),
                },
                "normalized_smoothness": (
                    self.normalized_smoothness_sum / self.smoothness_transition_count
                    if self.smoothness_transition_count else None
                ),
                "dynamic_transition_samples": self.dynamic_transition_count,
                "dynamic_feasibility": (
                    self.dynamic_sum / self.dynamic_transition_count
                    if self.dynamic_transition_count else None
                ),
                "dynamic_feasibility_components": {
                    "linear_v": (
                        self.dynamic_linear_sum / self.dynamic_transition_count
                        if self.dynamic_transition_count else None
                    ),
                    "angular_omega": (
                        self.dynamic_angular_sum / self.dynamic_transition_count
                        if self.dynamic_transition_count else None
                    ),
                },
                "normalized_dynamic_feasibility": (
                    self.normalized_dynamic_sum / self.dynamic_transition_count
                    if self.dynamic_transition_count else None
                ),
                "observed_command_period_s": stats(self.command_periods),
            },
            "visibility": {
                "N_all": all_visibility,
                "N_occ": self.invisible_count,
                "occlusion_rate_percent": occlusion_rate,
                "target_loss_count": self.loss_events,
                "target_recovery_count": self.recovery_events,
                "unrecovered_loss_count": max(
                    0, self.loss_events - self.recovery_events
                ),
            },
            "planning_time_ms": stats(self.planning_times_ms),
            "planning_time_by_branch_ms": {
                name: stats(values)
                for name, values in sorted(self.planning_times_by_branch.items())
            },
            "planning_time_algorithm_excluding_hold_ms": stats(
                self.planning_times_by_branch.get("neural", [])
                + self.planning_times_by_branch.get("traditional", [])
            ),
            "branch_usage": self.reported_branch_summary(self.branch_counts),
            "success_rate": {
                "total_started": self.total_trials_started,
                "completed": completed,
                "successful": successful,
                "failed": len(failed_trials),
                "invalid": len(invalid_trials),
                "incomplete": incomplete,
                "success_rate_percent": (
                    100.0 * successful / completed if completed else None
                ),
                "failure_reasons": dict(sorted(failure_reasons.items())),
                "expected_goal_trials": expected_goal_trials,
            },
            "validation": {
                "valid": (
                    self.startup_failure is None
                    and self.runtime_input_failure is None
                    and bool(self.trials)
                    and not invalid_trials
                    and incomplete == 0
                    and per_goal_count_valid
                    and all(t.get("valid", False) for t in self.trials)
                ),
                "startup_ready": self.startup_ready,
                "startup_failure": self.startup_failure,
                "runtime_input_failure": self.runtime_input_failure,
                "required_inputs_received": dict(self.input_received),
                "invalid_trials": len(invalid_trials),
                "expected_goal_trials": expected_goal_trials,
                "goal_trial_count_valid": per_goal_count_valid,
                "max_fov_missing_rate": self.args.max_fov_missing_rate,
            },
            "observation": {
                "status_topic": self.args.observation_status_topic,
                "truth_fallback_seen": self.truth_fallback_seen,
                "truth_fallback_terminal_reason": (
                    "target_lost_over_3s"
                    if self.truth_fallback_seen else None
                ),
            },
            "trials": self.trials,
        }

    def periodic_report(self) -> None:
        self.write_report()
        k = self.command_count
        algorithm_total = (
            self.branch_counts.get("neural", 0)
            + self.branch_counts.get("traditional", 0)
        )
        neural_pct = (
            100.0 * self.branch_counts.get("neural", 0) / algorithm_total
            if algorithm_total
            else 0.0
        )
        smoothness = (
            self.smoothness_sum / self.smoothness_transition_count
            if self.smoothness_transition_count else float("nan")
        )
        normalized_smoothness = (
            self.normalized_smoothness_sum / self.smoothness_transition_count
            if self.smoothness_transition_count else float("nan")
        )
        feasibility = (
            self.dynamic_sum / self.dynamic_transition_count
            if self.dynamic_transition_count else float("nan")
        )
        normalized_feasibility = (
            self.normalized_dynamic_sum / self.dynamic_transition_count
            if self.dynamic_transition_count else float("nan")
        )
        occlusion_rate = (
            100.0 * self.invisible_count / self.visibility_count
            if self.visibility_count else float("nan")
        )
        planning_mean = (
            statistics.fmean(self.planning_times_ms)
            if self.planning_times_ms else float("nan")
        )
        branch_text = (
            "，神经分支占比=%.1f%%" % neural_pct
            if self.args.tracker_family == "neural"
            else "（分支占比不适用）"
        )
        self.get_logger().info(
            "实时评测：活动路线统一周期 K=%d，活动窗口丢包=%d\n"
            "  控制平滑性=%.6f（原式混合单位），归一化=%.6f（无量纲）\n"
            "  动力可行性=%.6f（原式混合单位），归一化=%.6f s^-4\n"
            "  遮挡率=%.2f%%，丢失/找回=%d/%d\n"
            "  控制周期平均耗时=%.3f ms%s"
            % (
                k,
                self.active_missed_metrics_samples,
                smoothness,
                normalized_smoothness,
                feasibility,
                normalized_feasibility,
                occlusion_rate,
                self.loss_events,
                self.recovery_events,
                planning_mean,
                branch_text,
            )
        )

    def write_report(self, final: bool = False) -> None:
        output = self.args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.build_report(final), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(output)

    def write_raw_csv(self) -> None:
        if self.args.raw_csv_prefix is None:
            return
        prefix = self.args.raw_csv_prefix
        prefix.parent.mkdir(parents=True, exist_ok=True)

        tables = [
            (
                prefix.with_name(prefix.name + "_commands.csv"),
                ("sample", "elapsed_s", "v_mps", "omega_radps"),
                self.command_rows,
            ),
            (
                prefix.with_name(prefix.name + "_visibility.csv"),
                ("sample", "elapsed_s", "visible"),
                self.visibility_rows,
            ),
            (
                prefix.with_name(prefix.name + "_planning_time.csv"),
                ("sample", "elapsed_s", "planning_time_ms"),
                self.planning_rows,
            ),
            (
                prefix.with_name(prefix.name + "_branches.csv"),
                ("sample", "elapsed_s", "branch"),
                self.branch_rows,
            ),
            (
                prefix.with_name(prefix.name + "_branch_switches.csv"),
                (
                    "switch",
                    "sample",
                    "metrics_sequence",
                    "trial_id",
                    "elapsed_s",
                    "from_branch",
                    "to_branch",
                ),
                self.branch_switch_rows,
            ),
            (
                prefix.with_name(prefix.name + "_timeline.csv"),
                (
                    "sample",
                    "metrics_sequence",
                    "trial_id",
                    "elapsed_s",
                    "tracker_x_m",
                    "tracker_y_m",
                    "tracker_yaw_rad",
                    "target_x_m",
                    "target_y_m",
                    "target_yaw_rad",
                    "tracker_pose_age_s",
                    "target_pose_age_s",
                    "v_mps",
                    "omega_radps",
                    "target_visible",
                    "planning_time_ms",
                    "branch_code",
                    "branch",
                    "branch_changed",
                ),
                self.timeline_rows,
            ),
            (
                prefix.with_name(prefix.name + "_target_fov.csv"),
                (
                    "sample",
                    "elapsed_s",
                    "tracker_x_m",
                    "tracker_y_m",
                    "tracker_yaw_rad",
                    "target_x_m",
                    "target_y_m",
                    "relative_forward_m",
                    "relative_left_m",
                    "range_m",
                    "bearing_rad",
                    "bearing_deg",
                    "detector_visible",
                    "inside_geometric_fov",
                    "tracker_pose_age_s",
                    "target_pose_age_s",
                    "sensor_fov_deg",
                    "sensor_range_m",
                ),
                self.target_fov_rows,
            ),
        ]
        for path, header, rows in tables:
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(rows)

        trials_path = prefix.with_name(prefix.name + "_trials.csv")
        with trials_path.open("w", encoding="utf-8", newline="") as stream:
            fields = (
                "id",
                "goal_x",
                "goal_y",
                "status",
                "reason",
                "valid",
                "validation_errors",
                "duration_s",
                "max_tracking_distance_m",
                "visibility_samples",
                "invisible_samples",
                "occlusion_rate_percent",
                "loss_events",
                "recovery_events",
                "truth_fallback_used",
                "goal_count",
                "command_samples",
                "sequence_file",
                "sequence_sha256",
            )
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for trial in self.trials:
                writer.writerow(
                    {
                        "id": trial["id"],
                        "goal_x": trial["goal"]["x"],
                        "goal_y": trial["goal"]["y"],
                        "status": trial["status"],
                        "reason": trial["reason"],
                        "valid": trial.get("valid", False),
                        "validation_errors": ";".join(
                            trial.get("validation_errors", [])
                        ),
                        "duration_s": trial["duration_s"],
                        "max_tracking_distance_m": trial[
                            "max_tracking_distance_m"
                        ],
                        "visibility_samples": trial["visibility_samples"],
                        "invisible_samples": trial["invisible_samples"],
                        "occlusion_rate_percent": trial[
                            "occlusion_rate_percent"
                        ],
                        "loss_events": trial["loss_events"],
                        "recovery_events": trial["recovery_events"],
                        "truth_fallback_used": trial.get(
                            "truth_fallback_used", False
                        ),
                        "goal_count": trial.get("goal_count", 1),
                        "command_samples": trial.get("command_samples", 0),
                        "sequence_file": trial.get("route_metadata", {}).get(
                            "sequence_file", ""
                        ),
                        "sequence_sha256": trial.get("route_metadata", {}).get(
                            "sequence_sha256", ""
                        ),
                    }
                )

    def close(self) -> None:
        if self.active_trial is not None:
            self.finish_trial(
                "incomplete",
                "recorder_stopped",
                time.monotonic(),
                notify_ros=False,
            )
        self.write_report(final=True)
        self.write_raw_csv()
        print("最终评测结果已写入 %s" % self.args.output, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure neural_tracker smoothness, feasibility, visibility, runtime and success"
    )
    local_now = datetime.now().astimezone()
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path; default: experiments/DATE/RUN_LABEL/metrics.json",
    )
    parser.add_argument(
        "--raw-csv-prefix",
        type=Path,
        default=None,
        help="raw CSV prefix; default: <run directory>/metrics",
    )
    parser.add_argument(
        "--no-raw-csv",
        action="store_true",
        help="do not save raw CSV files",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("experiments"),
        help="root used by automatic DATE/RUN_LABEL output",
    )
    parser.add_argument(
        "--run-label",
        default="run_%s" % local_now.strftime("%H%M%S"),
        help="label for this test run, e.g. maze2_run01",
    )
    parser.add_argument(
        "--algorithm-label", default="neural_tracker",
        help="algorithm/configuration name stored in the report",
    )
    parser.add_argument(
        "--tracker-family",
        choices=("neural", "traditional"),
        default="neural",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--map-file", type=Path,
        default=None,
        help="map JSON used by this run; path and SHA256 are stored",
    )
    parser.add_argument(
        "--policy-file", type=Path,
        default=None,
        help="ONNX policy used by this run; path and SHA256 are stored",
    )
    parser.add_argument(
        "--trial-mode", choices=("route", "per-goal"), default="per-goal",
        help=(
            "per-goal: every Goal is one trial; route: one complete replay is "
            "one trial (default: per-goal)"
        ),
    )
    parser.add_argument(
        "--direct-target",
        action="store_true",
        help=(
            "tracker is controlled from /target/ground_truth; keep measuring "
            "raw FOV visibility but do not terminate trials on FOV loss"
        ),
    )
    parser.add_argument(
        "--route-status-topic", default="/goal_pose_sequence/status",
        help="route lifecycle JSON topic published by benchmark replay",
    )
    parser.add_argument(
        "--route-timeout", type=float, default=900.0,
        help="whole-route timeout in seconds",
    )
    parser.add_argument(
        "--auto-stop", action="store_true",
        help="write the final report and exit when a route reaches a terminal state",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="allow replacing an existing metrics JSON/CSV run",
    )
    parser.add_argument("--control-dt", type=float, default=0.1)
    parser.add_argument(
        "--linear-control-scale",
        type=float,
        default=2.5,
        help="v normalization scale in m/s (default: tracker.max_vel=2.5)",
    )
    parser.add_argument(
        "--angular-control-scale",
        type=float,
        default=1.2,
        help="omega normalization scale in rad/s (default: tracker.max_omega=1.2)",
    )
    parser.add_argument(
        "--sensor-fov-deg", type=float, default=87.0,
        help="horizontal target-detection FOV in degrees",
    )
    parser.add_argument(
        "--sensor-range", type=float, default=4.0,
        help="target-detection range in meters",
    )
    parser.add_argument(
        "--pose-max-age", type=float, default=0.25,
        help="maximum tracker/target truth pose age accepted for FOV samples",
    )
    parser.add_argument(
        "--max-fov-missing-rate",
        type=float,
        default=0.01,
        help="maximum accepted missing/stale FOV pose fraction (default: 0.01)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=30.0,
        help="seconds allowed for required input and publisher-uniqueness checks",
    )
    parser.add_argument(
        "--goal-tolerance", type=float, default=2.0,
        help=(
            "target-to-goal success radius in meters; must match replay "
            "--goal-tolerance (default: 2.0)"
        ),
    )
    parser.add_argument("--trial-timeout", type=float, default=60.0)
    parser.add_argument(
        "--lost-timeout", type=float, default=3.0,
        help="fail the current Goal after continuous FOV loss (default: 3.0 s)",
    )
    parser.add_argument("--max-tracking-distance", type=float, default=4.0)
    parser.add_argument("--min-tracking-distance", type=float, default=0.5,
                        help="fail when tracker/target remain closer than this distance")
    parser.add_argument("--distance-timeout", type=float, default=2.0)
    parser.add_argument("--robot-radius", type=float, default=0.15)
    parser.add_argument("--occupied-threshold", type=int, default=50)
    parser.add_argument("--report-interval", type=float, default=5.0)
    parser.add_argument("--metrics-topic", default="/tracker/metrics_sample")
    parser.add_argument(
        "--legacy-separate-topics",
        action="store_true",
        help="use the old independent cmd/detection/timing/branch topics",
    )
    parser.add_argument("--cmd-topic", default="/tracker/cmd_vel")
    parser.add_argument("--detection-topic", default="/target/detection")
    parser.add_argument("--planning-time-topic", default="/tracker/planning_time_ms")
    parser.add_argument("--branch-topic", default="/tracker/control_branch")
    parser.add_argument("--odom-topic", default="/tracker/odom")
    parser.add_argument("--target-truth-topic", default="/target/ground_truth")
    parser.add_argument(
        "--observation-status-topic",
        default="/tracker/target_observation_status",
        help="tracker observation state; truth fallback is a terminal failure",
    )
    parser.add_argument("--goal-topic", default="/goal_pose")
    parser.add_argument("--map-topic", default="/map")
    parser.add_argument(
        "--metrics-status-topic",
        default="/experiment/metrics_status",
        help="readiness and terminal-state handshake used by goal replay",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    positive = (
        "control_dt",
        "linear_control_scale",
        "angular_control_scale",
        "sensor_fov_deg",
        "sensor_range",
        "pose_max_age",
        "startup_timeout",
        "goal_tolerance",
        "trial_timeout",
        "route_timeout",
        "lost_timeout",
        "max_tracking_distance",
        "min_tracking_distance",
        "distance_timeout",
        "robot_radius",
        "report_interval",
    )
    for name in positive:
        if getattr(args, name) <= 0.0:
            parser.error("--%s must be positive" % name.replace("_", "-"))
    if args.sensor_fov_deg >= 180.0:
        parser.error("--sensor-fov-deg must be smaller than 180 degrees")
    if not 0.0 <= args.max_fov_missing_rate <= 1.0:
        parser.error("--max-fov-missing-rate must be in [0, 1]")
    if not 0 <= args.occupied_threshold <= 100:
        parser.error("--occupied-threshold must be in [0, 100]")
    if args.no_raw_csv and args.raw_csv_prefix is not None:
        parser.error("--no-raw-csv cannot be combined with --raw-csv-prefix")
    for name in ("map_file", "policy_file"):
        path = getattr(args, name)
        if path is not None:
            resolved = path.expanduser().resolve()
            if not resolved.is_file():
                parser.error("--%s does not exist: %s" % (name.replace("_", "-"), path))
            setattr(args, name, resolved)


def safe_run_label(parser: argparse.ArgumentParser, label: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", label.strip(), flags=re.UNICODE)
    cleaned = cleaned.strip("._")
    if not cleaned or cleaned.lower() in {"summary", "all"}:
        parser.error("--run-label must be a non-empty test label other than summary/all")
    return cleaned


def main(argv: Optional[list[str]] = None) -> int:
    full_args = sys.argv if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(remove_ros_args(args=full_args)[1:])
    validate_args(parser, args)
    args.run_label = safe_run_label(parser, args.run_label)
    if args.output is None:
        day = datetime.now().astimezone().strftime("%Y-%m-%d")
        run_directory = (
            args.experiment_root.expanduser().resolve() / day / args.run_label
        )
        args.output = run_directory / "metrics.json"
    else:
        args.output = args.output.expanduser().resolve()
        run_directory = args.output.parent
    if args.output.exists() and not args.overwrite:
        parser.error(
            "%s already exists; choose a new --run-label/output or pass --overwrite"
            % args.output
        )
    if args.no_raw_csv:
        args.raw_csv_prefix = None
    elif args.raw_csv_prefix is not None:
        args.raw_csv_prefix = args.raw_csv_prefix.expanduser().resolve()
    else:
        args.raw_csv_prefix = run_directory / args.output.stem

    rclpy.init(args=full_args)
    node = NeuralTrackerMetrics(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
