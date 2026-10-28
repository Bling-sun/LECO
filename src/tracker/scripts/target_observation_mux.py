#!/usr/bin/env python3
"""Switch from FOV detections to target truth after a sustained loss."""

from __future__ import annotations

from copy import deepcopy
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String


def detection_is_visible(message: PoseStamped) -> bool:
    point = message.pose.position
    return (
        message.pose.orientation.w > 0.0
        and math.isfinite(point.x)
        and math.isfinite(point.y)
    )


def fallback_due(lost_since, now: float, timeout: float) -> bool:
    return lost_since is not None and now - lost_since >= max(0.0, timeout)


class TargetObservationMux(Node):
    """Publish FOV detection normally and temporary truth after timeout."""

    def __init__(self):
        super().__init__("target_observation_mux")
        self.declare_parameter("detection_topic", "/target/detection")
        self.declare_parameter("truth_topic", "/target/ground_truth")
        self.declare_parameter("output_topic", "/target/fov_or_truth")
        self.declare_parameter("fallback_after", 3.0)
        self.declare_parameter("detection_stale_after", 1.0)
        self.declare_parameter("publish_rate", 20.0)

        self.fallback_after = max(
            0.0, float(self.get_parameter("fallback_after").value))
        self.detection_stale_after = max(
            0.1, float(self.get_parameter("detection_stale_after").value))
        publish_rate = max(
            1.0, float(self.get_parameter("publish_rate").value))

        self.latest_detection = None
        self.latest_truth = None
        self.detection_visible = False
        self.last_detection_received = None
        self.lost_since = None
        self.truth_fallback_active = False

        self.publisher = self.create_publisher(
            PoseStamped, str(self.get_parameter("output_topic").value), 10)
        self.status_publisher = self.create_publisher(
            String, "/tracker/target_observation_status", 10)
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("detection_topic").value),
            self._on_detection,
            10,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("truth_topic").value),
            self._on_truth,
            10,
        )
        self.create_timer(1.0 / publish_rate, self._publish_selected)
        self.get_logger().info(
            "Target observation mux: FOV detection -> truth after "
            f"{self.fallback_after:.1f}s continuous loss")

    def _on_detection(self, message: PoseStamped) -> None:
        now = time.monotonic()
        self.latest_detection = message
        self.last_detection_received = now
        visible = detection_is_visible(message)
        if visible:
            recovered = self.truth_fallback_active
            self.detection_visible = True
            self.lost_since = None
            self.truth_fallback_active = False
            if recovered:
                self._publish_status("fov_detection_recovered")
                self.get_logger().info(
                    "FOV detection recovered; leaving truth fallback")
        else:
            if self.detection_visible or self.lost_since is None:
                self.lost_since = now
            self.detection_visible = False

    def _on_truth(self, message: PoseStamped) -> None:
        self.latest_truth = message

    def _publish_status(self, value: str) -> None:
        status = String()
        status.data = value
        self.status_publisher.publish(status)

    def _publish_selected(self) -> None:
        now = time.monotonic()
        if (
            self.detection_visible
            and self.last_detection_received is not None
            and now - self.last_detection_received > self.detection_stale_after
        ):
            self.detection_visible = False
            self.lost_since = now
            self.get_logger().warn(
                "FOV detection stream became stale; starting loss timer")

        if (
            not self.detection_visible
            and fallback_due(
                self.lost_since, now, self.fallback_after)
            and not self.truth_fallback_active
        ):
            self.truth_fallback_active = True
            self._publish_status("truth_fallback_started")
            self.get_logger().warn(
                f"FOV target lost for {self.fallback_after:.1f}s; "
                "using /target/ground_truth until detection recovers")

        if self.detection_visible and self.latest_detection is not None:
            self.publisher.publish(self.latest_detection)
            return
        if self.truth_fallback_active and self.latest_truth is not None:
            truth = deepcopy(self.latest_truth)
            # Downstream FOV-aware consumers use orientation.w as validity.
            truth.pose.orientation.x = 0.0
            truth.pose.orientation.y = 0.0
            truth.pose.orientation.z = 0.0
            truth.pose.orientation.w = 1.0
            self.publisher.publish(truth)
            return
        if self.latest_detection is not None:
            lost = deepcopy(self.latest_detection)
            lost.pose.orientation.w = -1.0
            self.publisher.publish(lost)


def main(args=None):
    rclpy.init(args=args)
    node = TargetObservationMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
