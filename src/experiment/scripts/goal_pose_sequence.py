#!/usr/bin/env python3
"""Record and replay RViz 2D Goal Pose messages for repeatable experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import Float64MultiArray, String


FORMAT = "neural-tracker-goal-sequence"
VERSION = 1
DEFAULT_TOPIC = "/goal_pose"
DEFAULT_STATUS_TOPIC = "/goal_pose_sequence/status"
DEFAULT_METRICS_STATUS_TOPIC = "/experiment/metrics_status"
RECOVERABLE_FOV_FAILURE_REASONS = {
    "target_lost_over_3s",
    "target_lost_over_5s",
    # Accept reports produced by the previous installation so an overlay or a
    # stale metrics process cannot turn one failed Goal into a route abort.
    "target_outside_fov_over_3s",
}


def goal_qos() -> QoSProfile:
    return QoSProfile(
        depth=10,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(sin_yaw, cos_yaw)


def pose_to_record(
    message: PoseStamped, elapsed: float, index: int
) -> dict[str, Any]:
    position = message.pose.position
    orientation = message.pose.orientation
    return {
        "index": index,
        "time": round(elapsed, 9),
        "frame_id": message.header.frame_id or "world",
        "position": {
            "x": position.x,
            "y": position.y,
            "z": position.z,
        },
        "orientation": {
            "x": orientation.x,
            "y": orientation.y,
            "z": orientation.z,
            "w": orientation.w,
        },
        "yaw": yaw_from_quaternion(
            orientation.x, orientation.y, orientation.z, orientation.w
        ),
    }


def validate_goal(goal: dict[str, Any], index: int) -> None:
    try:
        position = goal["position"]
        orientation = goal["orientation"]
        values = [
            float(goal["time"]),
            float(position["x"]),
            float(position["y"]),
            float(position.get("z", 0.0)),
            float(orientation.get("x", 0.0)),
            float(orientation.get("y", 0.0)),
            float(orientation.get("z", 0.0)),
            float(orientation.get("w", 1.0)),
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid goal #{index}: {error}") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"goal #{index} contains a non-finite value")
    if values[0] < 0.0:
        raise ValueError(f"goal #{index} has a negative relative time")
    if sum(value * value for value in values[4:]) < 1e-12:
        raise ValueError(f"goal #{index} contains a zero-length quaternion")


def load_sequence(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"sequence file does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error
    if payload.get("format") != FORMAT or payload.get("version") != VERSION:
        raise ValueError(
            f"{path} is not a supported {FORMAT} version {VERSION} file"
        )
    goals = payload.get("goals")
    if not isinstance(goals, list) or not goals:
        raise ValueError(f"{path} contains no goals")
    previous_time = -1.0
    for index, goal in enumerate(goals, 1):
        validate_goal(goal, index)
        relative_time = float(goal["time"])
        if relative_time < previous_time:
            raise ValueError("goal times must be monotonically non-decreasing")
        previous_time = relative_time
    return payload


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


class GoalRecorder(Node):
    def __init__(self, topic: str, output: Path, run_label: str) -> None:
        super().__init__("goal_pose_recorder")
        self.output = output
        self.started: float | None = None
        self.payload: dict[str, Any] = {
            "format": FORMAT,
            "version": VERSION,
            "topic": topic,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "run_label": run_label,
            "run_directory": str(output.parent),
            "time_basis": "seconds since first received goal",
            "goals": [],
        }
        self.subscription = self.create_subscription(
            PoseStamped, topic, self._on_goal, goal_qos()
        )

    def _on_goal(self, message: PoseStamped) -> None:
        received = time.monotonic()
        if self.started is None:
            self.started = received
        index = len(self.payload["goals"]) + 1
        record = pose_to_record(message, received - self.started, index)
        self.payload["goals"].append(record)
        atomic_write(self.output, self.payload)
        self.get_logger().info(
            "Recorded goal %d: x=%.3f y=%.3f yaw=%.3f t=%.3fs"
            % (
                index,
                record["position"]["x"],
                record["position"]["y"],
                record["yaw"],
                record["time"],
            )
        )


def make_message(node: Node, goal: dict[str, Any]) -> PoseStamped:
    message = PoseStamped()
    message.header.stamp = node.get_clock().now().to_msg()
    message.header.frame_id = str(goal.get("frame_id") or "world")
    position = goal["position"]
    orientation = goal["orientation"]
    message.pose.position.x = float(position["x"])
    message.pose.position.y = float(position["y"])
    message.pose.position.z = float(position.get("z", 0.0))
    message.pose.orientation.x = float(orientation.get("x", 0.0))
    message.pose.orientation.y = float(orientation.get("y", 0.0))
    message.pose.orientation.z = float(orientation.get("z", 0.0))
    message.pose.orientation.w = float(orientation.get("w", 1.0))
    return message


def spin_sleep(node: Node, duration: float) -> bool:
    deadline = time.monotonic() + max(0.0, duration)
    while rclpy.ok() and time.monotonic() < deadline:
        remaining = max(0.0, deadline - time.monotonic())
        rclpy.spin_once(node, timeout_sec=min(0.05, remaining))
    return rclpy.ok()


def wait_for_target_goal(
    node: Node,
    target_state: dict[str, Any],
    goal: dict[str, Any],
    tolerance: float,
    timeout: float,
) -> bool:
    deadline = time.monotonic() + timeout
    goal_x = float(goal["position"]["x"])
    goal_y = float(goal["position"]["y"])
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        abort_reason = target_state.get("abort_reason")
        if abort_reason:
            raise RuntimeError(
                "metrics terminated the route while waiting for a waypoint: %s"
                % abort_reason
            )
        xy = target_state.get("xy")
        if xy is not None and math.hypot(xy[0] - goal_x, xy[1] - goal_y) <= tolerance:
            return True
    return False


def apply_metrics_status(
    target_state: dict[str, Any],
    status: dict[str, Any],
    source_hash: str,
) -> None:
    """Apply metrics handshakes without aborting on an isolated Goal failure."""
    event = status.get("event")
    if event == "ready":
        target_state["metrics_ready"] = True
        return
    if event not in {"trial_end", "startup_failed", "runtime_failed"}:
        return
    route_hash = status.get("sequence_sha256")
    if route_hash not in {None, source_hash}:
        return
    failure_reason = status.get("reason")
    recoverable_fov_failure = (
        event == "trial_end"
        and status.get("status") == "failure"
        and failure_reason in RECOVERABLE_FOV_FAILURE_REASONS
    )
    per_goal_failure = (
        event == "trial_end"
        and status.get("trial_mode") == "per-goal"
        and status.get("status") == "failure"
    )
    if recoverable_fov_failure:
        # FOV loss is a terminal result for only the current Goal.  Clear a
        # stale abort set by an older/duplicate status message, let truth guide
        # reacquisition, and keep the remaining route intact.
        target_state["abort_reason"] = None
        target_state["recovery_required"] = True
        return
    if per_goal_failure:
        return
    if status.get("status") != "success":
        target_state["abort_reason"] = str(
            status.get("reason") or status.get("status") or event
        )


def consume_recovery_notice(target_state: dict[str, Any]) -> bool:
    """Consume a per-Goal FOV failure without blocking the remaining route."""
    return bool(target_state.pop("recovery_required", False))


def publisher_count(node: Node, topic: str) -> int:
    return len(node.get_publishers_info_by_topic(topic))


def wait_for_benchmark_ready(
    node: Node,
    state: dict[str, Any],
    required_topics: tuple[str, ...],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while rclpy.ok() and time.monotonic() < deadline:
        duplicate_topics = {
            topic: publisher_count(node, topic)
            for topic in required_topics
            if publisher_count(node, topic) > 1
        }
        if duplicate_topics:
            raise RuntimeError(
                "benchmark requires exactly one publisher per input topic; "
                "duplicates found: %s"
                % ", ".join(
                    "%s=%d" % item for item in sorted(duplicate_topics.items())
                )
            )
        publishers_ready = all(
            publisher_count(node, topic) == 1 for topic in required_topics
        )
        messages_ready = all(state["messages"].get(topic, False) for topic in required_topics)
        if publishers_ready and messages_ready and state.get("metrics_ready", False):
            return
        rclpy.spin_once(node, timeout_sec=0.1)

    missing_publishers = [
        "%s(%d)" % (topic, publisher_count(node, topic))
        for topic in required_topics
        if publisher_count(node, topic) != 1
    ]
    missing_messages = [
        topic for topic in required_topics if not state["messages"].get(topic, False)
    ]
    details = []
    if missing_publishers:
        details.append("publisher count != 1: " + ", ".join(missing_publishers))
    if missing_messages:
        details.append("no message: " + ", ".join(missing_messages))
    if not state.get("metrics_ready", False):
        details.append("metrics readiness handshake missing")
    raise RuntimeError(
        "benchmark inputs were not ready within %.1fs (%s)"
        % (timeout, "; ".join(details))
    )


def wait_for_subscribers(
    node: Node, publisher: Any, timeout: float, minimum: int = 1
) -> bool:
    if timeout <= 0.0:
        return True
    deadline = time.monotonic() + timeout
    while rclpy.ok() and time.monotonic() < deadline:
        if publisher.get_subscription_count() >= minimum:
            return True
        rclpy.spin_once(node, timeout_sec=0.1)
    return publisher.get_subscription_count() >= minimum


def publish_status(node: Node, publisher: Any, payload: dict[str, Any]) -> None:
    message = String()
    message.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    publisher.publish(message)
    # Give reliable DDS a short opportunity to deliver the cross-topic route
    # boundary before the corresponding first/last Goal message.
    spin_sleep(node, 0.1)


def sanitize_run_label(label: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", label.strip(), flags=re.UNICODE)
    cleaned = cleaned.strip("._")
    if not cleaned or cleaned.lower() in {"summary", "all"}:
        raise ValueError("--run-label must be a non-empty label other than summary/all")
    return cleaned


def resolve_record_output(args: argparse.Namespace) -> tuple[Path, str, bool]:
    if getattr(args, "exact_output", False):
        output = args.file.expanduser().resolve()
        if output.exists():
            raise ValueError(f"recording output already exists: {output}")
        return output, output.stem, False
    # Default recording is a flat, monotonically numbered archive in the
    # workspace experiments directory: maze2_goal1.json, maze2_goal2.json, …
    if args.file == Path("experiments/goals.json") and args.run_label is None:
        root = args.experiment_root.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        index = 1
        while (root / f"maze2_goal{index}.json").exists():
            index += 1
        output = root / f"maze2_goal{index}.json"
        return output, f"maze2_goal{index}", True
    requested = args.file.expanduser().resolve()
    experiment_root = args.experiment_root.expanduser().resolve()
    is_legacy_flat_path = requested.parent == experiment_root
    redirect_to_run = is_legacy_flat_path or args.run_label is not None
    if not redirect_to_run:
        return requested, requested.parent.name, False

    local_now = datetime.now().astimezone()
    # Without an explicit run label, use the filename stem deterministically:
    # repeating the same command resolves to the same file and overwrites it.
    # Users who want separate test runs provide distinct --run-label values.
    base_label = sanitize_run_label(args.run_label or requested.stem)
    day_directory = experiment_root / local_now.strftime("%Y-%m-%d")
    label = base_label
    output = day_directory / label / requested.name
    return output, label, True


def resolve_timing(requested: str | None, benchmark: bool) -> str:
    timing = requested or ("reached" if benchmark else "recorded")
    if benchmark and timing != "reached":
        raise ValueError("--benchmark requires --timing reached")
    return timing


def run_record(args: argparse.Namespace) -> int:
    output, run_label, redirected = resolve_record_output(args)
    if redirected:
        print("按日期/次标签保存到：%s" % output)
    output_existed = output.exists()
    rclpy.init()
    node = GoalRecorder(args.topic, output, run_label)
    node.get_logger().info(
        f"Recording {args.topic} to {output}; click '2D Goal Pose', then Ctrl+C"
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        count = len(node.payload["goals"])
        if count:
            atomic_write(output, node.payload)
        node.destroy_node()
        rclpy.try_shutdown()
    if not count:
        suffix = " Existing file was left unchanged." if output_existed else ""
        print(f"No goal was received; no sequence file was written.{suffix}")
        return 1
    print(f"Saved {count} goals to {output}")
    return 0


def run_replay(args: argparse.Namespace) -> int:
    source = args.file.expanduser().resolve()
    payload = load_sequence(source)
    goals = payload["goals"]
    topic = args.topic or str(payload.get("topic") or DEFAULT_TOPIC)
    args.timing = resolve_timing(args.timing, args.benchmark)
    if args.speed <= 0.0:
        raise ValueError("--speed must be positive")
    if args.interval < 0.0 or args.start_delay < 0.0 or args.repeat_delay < 0.0:
        raise ValueError("delays and --interval must be non-negative")
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if args.required_goal_subscribers < 1:
        raise ValueError("--required-goal-subscribers must be at least 1")
    if (
        args.goal_tolerance <= 0.0
        or args.waypoint_timeout <= 0.0
        or args.recovery_timeout <= 0.0
    ):
        raise ValueError(
            "--goal-tolerance, --waypoint-timeout and --recovery-timeout "
            "must be positive"
        )
    if args.subscriber_timeout <= 0.0 or args.benchmark_ready_timeout <= 0.0:
        raise ValueError(
            "--subscriber-timeout and --benchmark-ready-timeout must be positive"
        )
    if args.benchmark and args.repeat != 1:
        raise ValueError("--benchmark requires --repeat 1; run each route separately")

    if args.dry_run:
        duration = (
            float(goals[-1]["time"]) / args.speed
            if args.timing == "recorded"
            else ((len(goals) - 1) * args.interval if args.timing == "fixed" else None)
        )
        print(
            f"{source}: {len(goals)} goals, topic={topic}, "
            f"timing={args.timing}, one-pass duration="
            + (f"{duration:.3f}s" if duration is not None else "target-dependent")
        )
        for index, goal in enumerate(goals, 1):
            delay = (
                float(goal["time"]) / args.speed
                if args.timing == "recorded"
                else ((index - 1) * args.interval if args.timing == "fixed" else None)
            )
            print(
                f"  {index:03d}  "
                + (f"t={delay:8.3f}s  " if delay is not None else "t=on-reached  ")
                + f"x={float(goal['position']['x']):9.3f}  "
                f"y={float(goal['position']['y']):9.3f}  "
                f"yaw={float(goal.get('yaw', 0.0)):8.3f}"
            )
        return 0

    rclpy.init()
    node = Node("goal_pose_replayer")
    publisher = node.create_publisher(PoseStamped, topic, goal_qos())
    status_publisher = node.create_publisher(String, args.status_topic, goal_qos())
    target_state: dict[str, Any] = {
        "xy": None,
        "abort_reason": None,
        "metrics_ready": False,
        "detection_visible": None,
        "recovery_required": False,
        "messages": {},
    }
    subscriptions = []

    def on_target(message: PoseStamped) -> None:
        target_state["xy"] = (
            float(message.pose.position.x),
            float(message.pose.position.y),
        )
        target_state["messages"][args.target_truth_topic] = True

    if args.timing == "reached" or args.benchmark:
        subscriptions.append(node.create_subscription(
            PoseStamped, args.target_truth_topic, on_target, goal_qos()
        ))

    if args.benchmark:
        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        def mark_received(topic_name: str):
            def callback(_message: Any) -> None:
                target_state["messages"][topic_name] = True
            return callback

        def on_detection(message: PoseStamped) -> None:
            target_state["messages"][args.detection_topic] = True
            target_state["detection_visible"] = (
                message.pose.orientation.w > 0.0
            )

        subscriptions.extend([
            node.create_subscription(
                Float64MultiArray, args.metrics_topic,
                mark_received(args.metrics_topic), goal_qos(),
            ),
            node.create_subscription(
                Odometry, args.odom_topic,
                mark_received(args.odom_topic), goal_qos(),
            ),
            node.create_subscription(
                PoseStamped, args.detection_topic,
                on_detection, goal_qos(),
            ),
            node.create_subscription(
                OccupancyGrid, args.map_topic,
                mark_received(args.map_topic), map_qos,
            ),
        ])

        def on_metrics_status(message: String) -> None:
            try:
                status = json.loads(message.data)
            except json.JSONDecodeError:
                return
            if (
                status.get("format") != "neural-tracker-metrics-status"
                or status.get("version") != 1
            ):
                return
            apply_metrics_status(target_state, status, source_hash)

        subscriptions.append(node.create_subscription(
            String, args.metrics_status_topic, on_metrics_status, goal_qos()
        ))
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    route_open = False
    try:
        required_goals = max(
            args.required_goal_subscribers, 2 if args.benchmark else 1
        )
        if not wait_for_subscribers(
            node, publisher, args.subscriber_timeout, required_goals
        ):
            raise RuntimeError(
                f"expected {required_goals} subscribers on {topic}, found "
                f"{publisher.get_subscription_count()} within "
                f"{args.subscriber_timeout:.1f}s"
            )
        if args.benchmark and publisher.get_subscription_count() != required_goals:
            raise RuntimeError(
                f"benchmark expected exactly {required_goals} subscribers on {topic}, "
                f"found {publisher.get_subscription_count()}; duplicate simulator or "
                "metrics nodes may be running"
            )
        if args.benchmark and not wait_for_subscribers(
            node, status_publisher, args.subscriber_timeout, 1
        ):
            raise RuntimeError(
                f"no benchmark subscriber discovered on {args.status_topic} "
                f"within {args.subscriber_timeout:.1f}s"
            )
        if args.benchmark:
            required_topics = (
                args.metrics_topic,
                args.odom_topic,
                args.target_truth_topic,
                args.detection_topic,
                args.map_topic,
            )
            wait_for_benchmark_ready(
                node, target_state, required_topics, args.benchmark_ready_timeout
            )
            node.get_logger().info(
                "Benchmark preflight passed: one publisher and live data on all "
                "required topics; metrics is ready"
            )
        node.get_logger().info(
            f"Replaying {len(goals)} goals from {source} on {topic}"
        )
        if not spin_sleep(node, args.start_delay):
            return 130

        for repetition in range(args.repeat):
            route_payload = {
                "format": "neural-tracker-route-status",
                "version": 1,
                "event": "start",
                "sequence_file": str(source),
                "sequence_sha256": source_hash,
                "goal_count": len(goals),
                "topic": topic,
                "timing": args.timing,
                "speed": args.speed,
                "interval_s": args.interval,
                "target_truth_topic": args.target_truth_topic,
                "goal_tolerance_m": args.goal_tolerance,
                "continuous_waypoint_advance": args.timing == "reached",
                "waypoint_timeout_s": args.waypoint_timeout,
                "repetition": repetition + 1,
                "repeat_total": args.repeat,
            }
            publish_status(node, status_publisher, route_payload)
            route_open = True
            pass_started = time.monotonic()
            for index, goal in enumerate(goals, 1):
                due = (
                    float(goal["time"]) / args.speed
                    if args.timing == "recorded"
                    else (
                        (index - 1) * args.interval
                        if args.timing == "fixed"
                        else time.monotonic() - pass_started
                    )
                )
                if not spin_sleep(node, due - (time.monotonic() - pass_started)):
                    return 130
                publisher.publish(make_message(node, goal))
                node.get_logger().info(
                    "Published goal %d/%d (repeat %d/%d): x=%.3f y=%.3f"
                    % (
                        index,
                        len(goals),
                        repetition + 1,
                        args.repeat,
                        float(goal["position"]["x"]),
                        float(goal["position"]["y"]),
                    )
                )
                rclpy.spin_once(node, timeout_sec=0.01)
                if args.timing == "reached" and not wait_for_target_goal(
                    node,
                    target_state,
                    goal,
                    args.goal_tolerance,
                    args.waypoint_timeout,
                ):
                    raise RuntimeError(
                        "target did not reach goal %d/%d within %.1fs"
                        % (index, len(goals), args.waypoint_timeout)
                    )
                if args.benchmark and consume_recovery_notice(target_state):
                    node.get_logger().warning(
                        "Goal %d/%d failed after sustained FOV loss; truth-assisted "
                        "tracking remains active until raw FOV recovery, continuing "
                        "the route" % (index, len(goals))
                    )
            route_payload["event"] = "end"
            publish_status(node, status_publisher, route_payload)
            route_open = False
            if (
                repetition + 1 < args.repeat
                and not spin_sleep(node, args.repeat_delay)
            ):
                return 130
        spin_sleep(node, 0.2)
    finally:
        if route_open and rclpy.ok():
            publish_status(
                node,
                status_publisher,
                {
                    "format": "neural-tracker-route-status",
                    "version": 1,
                    "event": "abort",
                    "sequence_file": str(source),
                    "sequence_sha256": source_hash,
                },
            )
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record and replay RViz 2D Goal Pose sequences."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser("record", help="record /goal_pose messages")
    record.add_argument(
        "file", nargs="?", type=Path, default=Path("experiments/goals.json"),
        help=(
            "output JSON; a flat experiments/FILE path is automatically "
            "redirected to experiments/DATE/RUN_LABEL/FILE"
        ),
    )
    record.add_argument("--topic", default=DEFAULT_TOPIC)
    record.add_argument(
        "--exact-output", action="store_true",
        help="use the given path without date/run redirection; refuse existing files",
    )
    record.add_argument(
        "--overwrite", action="store_true", help=argparse.SUPPRESS
    )
    record.add_argument(
        "--experiment-root", type=Path, default=Path("experiments")
    )
    record.add_argument(
        "--run-label", help="test-run label shared with neural_tracker_metrics"
    )
    record.set_defaults(function=run_record)

    replay = subparsers.add_parser("replay", help="replay a recorded sequence")
    replay.add_argument("file", type=Path, help="input JSON sequence")
    replay.add_argument("--topic", help="override the topic stored in the file")
    replay.add_argument(
        "--timing", choices=("recorded", "fixed", "reached"), default=None,
        help=(
            "recorded/fixed schedule, or publish the next goal when the target "
            "enters the waypoint radius; defaults to reached with --benchmark "
            "and recorded otherwise"
        ),
    )
    replay.add_argument("--speed", type=float, default=1.0,
                        help="recorded-timing speed multiplier")
    replay.add_argument("--interval", type=float, default=2.0,
                        help="seconds between goals with --timing fixed")
    replay.add_argument(
        "--target-truth-topic", default="/target/ground_truth",
        help="target pose used by --timing reached",
    )
    replay.add_argument(
        "--goal-tolerance", type=float, default=2.0,
        help=(
            "target-to-intermediate-waypoint advance radius in meters for "
            "--timing reached (default: 2.0)"
        ),
    )
    replay.add_argument(
        "--waypoint-timeout", type=float, default=120.0,
        help="maximum seconds to wait for each waypoint with --timing reached",
    )
    replay.add_argument(
        "--recovery-timeout", type=float, default=120.0,
        help=(
            "deprecated compatibility option; FOV recovery no longer blocks "
            "publication of subsequent Goals"
        ),
    )
    replay.add_argument("--start-delay", type=float, default=1.0)
    replay.add_argument("--subscriber-timeout", type=float, default=10.0)
    replay.add_argument("--benchmark-ready-timeout", type=float, default=30.0)
    replay.add_argument(
        "--benchmark", action="store_true",
        help=(
            "require both simulator and metrics subscribers; route metadata "
            "is published for every non-dry replay"
        ),
    )
    replay.add_argument(
        "--status-topic", default=DEFAULT_STATUS_TOPIC,
        help="route lifecycle topic consumed by neural_tracker_metrics",
    )
    replay.add_argument(
        "--metrics-status-topic", default=DEFAULT_METRICS_STATUS_TOPIC,
        help="metrics readiness and terminal-state topic",
    )
    replay.add_argument("--metrics-topic", default="/tracker/metrics_sample")
    replay.add_argument("--odom-topic", default="/tracker/odom")
    replay.add_argument("--detection-topic", default="/target/detection")
    replay.add_argument("--map-topic", default="/map")
    replay.add_argument(
        "--required-goal-subscribers", type=int, default=1,
        help="minimum /goal_pose subscribers; --benchmark enforces at least 2",
    )
    replay.add_argument("--repeat", type=int, default=1)
    replay.add_argument("--repeat-delay", type=float, default=2.0)
    replay.add_argument("--dry-run", action="store_true",
                        help="validate and print without publishing")
    replay.set_defaults(function=run_replay)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.function(args)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
