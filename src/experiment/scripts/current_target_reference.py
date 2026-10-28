#!/usr/bin/env python3
"""Feed the current observed target position to Tracker without prediction."""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node


class CurrentTargetReference(Node):
    def __init__(self) -> None:
        super().__init__("current_target_reference")
        self.declare_parameter("target_topic", "/target/fov_or_truth")
        self.declare_parameter("control_points", 7)
        topic = str(self.get_parameter("target_topic").value)
        self.control_points = max(
            1, int(self.get_parameter("control_points").value)
        )
        self.publisher = self.create_publisher(Path, "/target/bezier", 1)
        self.subscription = self.create_subscription(
            PoseStamped, topic, self.on_target, 1
        )
        self.get_logger().info(
            "Prediction disabled: using current target position as a "
            f"{self.control_points}-point zero-order reference"
        )

    def on_target(self, message: PoseStamped) -> None:
        reference = Path()
        reference.header = message.header
        point = message.pose.position
        visible = (
            message.pose.orientation.w > 0.0
            and math.isfinite(point.x)
            and math.isfinite(point.y)
        )
        if visible:
            for _ in range(self.control_points):
                pose = PoseStamped()
                pose.header = message.header
                pose.pose.position.x = float(point.x)
                pose.pose.position.y = float(point.y)
                pose.pose.position.z = float(point.z)
                pose.pose.orientation.w = 1.0
                reference.poses.append(pose)
        self.publisher.publish(reference)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CurrentTargetReference()
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
