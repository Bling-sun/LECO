#!/usr/bin/env python3
"""Publish target goals that avoid obstacles, short hops, and U-turns."""

import argparse
from collections import deque
import math
import random
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


class SafeGoalPublisher(Node):
    def __init__(self, args):
        super().__init__("safe_target_goal_publisher")
        self.args = args
        self._validate_args()
        self.rng = random.Random(args.seed)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.map_msg = None
        self.target_xy = None
        self.last_goal = None
        self.last_goal_msg = None
        self.last_goal_issue_time = 0.0
        self.last_publish_time = 0.0
        self.last_stuck_replan_time = 0.0
        self.last_target_update_time = 0.0
        self.last_nonempty_plan_time = 0.0
        self.empty_plan_since = None
        self.prev_dir = None
        self.position_history = deque()

        self.pub = self.create_publisher(PoseStamped, "/goal_pose", 1)
        map_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, "/map", self._map_cb, map_qos)
        self.create_subscription(Path, "/target/plan", self._plan_cb, qos)
        self.create_subscription(Path, "/target/trajectory", self._target_cb, qos)
        self.timer = self.create_timer(args.period, self._tick)

    def _validate_args(self):
        required = (
            "seed",
            "period",
            "republish_period",
            "min_goal_distance",
            "goal_clearance",
            "advance_distance",
            "reached_distance",
            "min_forward_dot",
            "escape_min_forward_dot",
            "stuck_window",
            "stuck_distance",
            "stuck_replan_cooldown",
            "no_trajectory_timeout",
            "max_attempts",
            "frame_id",
        )
        for name in required:
            getattr(self.args, name)

    def _map_cb(self, msg):
        self.map_msg = msg

    def _target_cb(self, msg):
        if not msg.poses:
            return
        p = msg.poses[-1].pose.position
        self.target_xy = (float(p.x), float(p.y))
        now = time.monotonic()
        self.last_target_update_time = now
        self.position_history.append((now, self.target_xy[0], self.target_xy[1]))
        while self.position_history and now - self.position_history[0][0] > self.args.stuck_window:
            self.position_history.popleft()

    def _plan_cb(self, msg):
        now = time.monotonic()
        if msg.poses:
            self.last_nonempty_plan_time = now
            self.empty_plan_since = None
        elif self.empty_plan_since is None:
            self.empty_plan_since = now

    def ready(self):
        return self.map_msg is not None and self.target_xy is not None

    def _cell_to_world(self, ix, iy):
        info = self.map_msg.info
        return (
            info.origin.position.x + (ix + 0.5) * info.resolution,
            info.origin.position.y + (iy + 0.5) * info.resolution,
        )

    def _is_clear(self, ix, iy):
        info = self.map_msg.info
        radius = max(1, int(math.ceil(self.args.goal_clearance / info.resolution)))
        if ix < radius or iy < radius:
            return False
        if ix >= info.width - radius or iy >= info.height - radius:
            return False

        data = self.map_msg.data
        for y in range(iy - radius, iy + radius + 1):
            row = y * info.width
            for x in range(ix - radius, ix + radius + 1):
                if data[row + x] != 0:
                    return False
        return True

    @staticmethod
    def _norm(vec):
        length = math.hypot(vec[0], vec[1])
        if length < 1e-9:
            return None
        return (vec[0] / length, vec[1] / length)

    @staticmethod
    def _dot(a, b):
        return a[0] * b[0] + a[1] * b[1]

    def _sample_goal(self, min_forward_dot=None):
        info = self.map_msg.info
        cx, cy = self.target_xy
        attempts = self.args.max_attempts
        if min_forward_dot is None:
            min_forward_dot = self.args.min_forward_dot

        for _ in range(attempts):
            ix = self.rng.randrange(info.width)
            iy = self.rng.randrange(info.height)
            if not self._is_clear(ix, iy):
                continue

            gx, gy = self._cell_to_world(ix, iy)
            vec = (gx - cx, gy - cy)
            dist = math.hypot(vec[0], vec[1])
            if dist < self.args.min_goal_distance:
                continue

            direction = self._norm(vec)
            if direction is None:
                continue

            if self.prev_dir is not None:
                if self._dot(self.prev_dir, direction) < min_forward_dot:
                    continue

            return gx, gy, direction, dist

        raise RuntimeError(
            "Could not find a safe target goal. Try lowering "
            "--min-goal-distance or --goal-clearance."
        )

    def _publish_next_goal(self, reason, min_forward_dot=None):
        try:
            gx, gy, direction, dist = self._sample_goal(min_forward_dot)
        except RuntimeError:
            if min_forward_dot == self.args.escape_min_forward_dot:
                raise
            self.get_logger().warn(
                "Could not find a goal with the continuity constraint; "
                "relaxing direction constraint once."
            )
            gx, gy, direction, dist = self._sample_goal(self.args.escape_min_forward_dot)

        cx, cy = self.target_xy
        yaw = math.atan2(gy - cy, gx - cx)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        msg.pose.position.x = gx
        msg.pose.position.y = gy
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)

        self._publish_goal_msg(msg)
        self.last_goal_issue_time = self.last_publish_time
        self.last_goal = (gx, gy)
        self.last_goal_msg = msg
        self.prev_dir = direction
        self.get_logger().info(
            f"{reason}: goal=({gx:.2f}, {gy:.2f}), "
            f"from=({cx:.2f}, {cy:.2f}), dist={dist:.2f}m"
        )

    def _publish_goal_msg(self, msg):
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)
        self.last_publish_time = time.monotonic()

    def _is_stuck(self, remaining):
        if remaining <= self.args.advance_distance:
            return False
        if len(self.position_history) < 2:
            return False

        first = self.position_history[0]
        last = self.position_history[-1]
        if last[0] - first[0] < self.args.stuck_window * 0.8:
            return False

        moved = math.hypot(last[1] - first[1], last[2] - first[2])
        return moved < self.args.stuck_distance

    def _trajectory_replan_reason(self, now, remaining):
        if remaining <= self.args.advance_distance:
            return None
        if self.last_goal_issue_time <= 0:
            return None
        if now - self.last_goal_issue_time < self.args.no_trajectory_timeout:
            return None

        if (
            self.empty_plan_since is not None
            and now - self.empty_plan_since >= self.args.no_trajectory_timeout
        ):
            return "empty_plan_replan"

        if (
            self.last_nonempty_plan_time > 0
            and self.last_nonempty_plan_time < self.last_goal_issue_time
        ):
            return "stale_plan_replan"

        if (
            self.last_target_update_time > 0
            and now - self.last_target_update_time >= self.args.no_trajectory_timeout
        ):
            return "target_trajectory_timeout"

        return None

    def _tick(self):
        if not self.ready():
            return

        if self.last_goal is None:
            self._publish_next_goal("initial")
            return

        dist = math.hypot(
            self.target_xy[0] - self.last_goal[0],
            self.target_xy[1] - self.last_goal[1],
        )
        now = time.monotonic()
        replan_reason = self._trajectory_replan_reason(now, dist)
        if self._is_stuck(dist):
            replan_reason = replan_reason or "stuck_escape"

        if (
            replan_reason is not None
            and now - self.last_stuck_replan_time >= self.args.stuck_replan_cooldown
        ):
            self.last_stuck_replan_time = now
            self.position_history.clear()
            self._publish_next_goal(
                replan_reason,
                min_forward_dot=self.args.escape_min_forward_dot,
            )
        elif dist <= self.args.advance_distance:
            reason = "advance" if dist > self.args.reached_distance else "reached"
            self._publish_next_goal(reason)
        elif now - self.last_publish_time >= self.args.republish_period:
            self._publish_goal_msg(self.last_goal_msg)
            self.get_logger().info(
                f"republish: goal=({self.last_goal[0]:.2f}, {self.last_goal[1]:.2f}), "
                f"remaining={dist:.2f}m"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--period", type=float, default=1.0)
    parser.add_argument("--republish-period", type=float, default=2.0)
    parser.add_argument("--min-goal-distance", type=float, default=3.0)
    parser.add_argument("--goal-clearance", type=float, default=0.6)
    parser.add_argument("--advance-distance", type=float, default=1.5)
    parser.add_argument("--reached-distance", type=float, default=0.75)
    parser.add_argument("--min-forward-dot", type=float, default=0.25)
    parser.add_argument("--escape-min-forward-dot", type=float, default=-1.0)
    parser.add_argument("--stuck-window", type=float, default=3.0)
    parser.add_argument("--stuck-distance", type=float, default=0.25)
    parser.add_argument("--stuck-replan-cooldown", type=float, default=3.0)
    parser.add_argument("--no-trajectory-timeout", type=float, default=2.0)
    parser.add_argument("--max-attempts", type=int, default=20000)
    parser.add_argument("--frame-id", type=str, default="world")
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = SafeGoalPublisher(args)
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
