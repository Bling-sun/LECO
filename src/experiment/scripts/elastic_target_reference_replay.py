#!/usr/bin/env python3
"""Replay an Elastic target reference exactly on a ROS 2 PoseStamped topic."""

import argparse
import bisect
import json
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


def interpolate_angle(start: float, end: float, ratio: float) -> float:
    delta = math.atan2(math.sin(end - start), math.cos(end - start))
    return start + ratio * delta


class ReferenceReplay(Node):
    def __init__(self, samples: list[dict], topic: str) -> None:
        super().__init__("elastic_target_reference_replay")
        self.samples = samples
        self.times = [float(sample["t"]) for sample in samples]
        self.publisher = self.create_publisher(PoseStamped, topic, 10)

    def publish_at(self, elapsed: float) -> None:
        if elapsed <= self.times[0]:
            sample = self.samples[0]
        elif elapsed >= self.times[-1]:
            sample = self.samples[-1]
        else:
            upper = bisect.bisect_right(self.times, elapsed)
            first = self.samples[upper - 1]
            second = self.samples[upper]
            span = float(second["t"]) - float(first["t"])
            ratio = 0.0 if span <= 0.0 else (elapsed - float(first["t"])) / span
            sample = {
                key: float(first[key]) + ratio * (float(second[key]) - float(first[key]))
                for key in ("x", "y", "z")
            }
            sample["yaw"] = interpolate_angle(
                float(first["yaw"]), float(second["yaw"]), ratio
            )
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "world"
        message.pose.position.x = float(sample["x"])
        message.pose.position.y = float(sample["y"])
        message.pose.position.z = float(sample["z"])
        yaw = float(sample["yaw"])
        message.pose.orientation.z = math.sin(0.5 * yaw)
        message.pose.orientation.w = math.cos(0.5 * yaw)
        self.publisher.publish(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("--topic", default="/target/reference_pose")
    parser.add_argument("--start-delay", type=float, default=2.0)
    parser.add_argument("--publish-rate", type=float, default=100.0)
    args = parser.parse_args()
    if args.start_delay < 0.0 or args.publish_rate <= 0.0:
        parser.error("start delay must be non-negative and publish rate positive")
    with open(args.reference, encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("format") != "elastic-target-reference-v1":
        parser.error("reference must use elastic-target-reference-v1")
    samples = payload.get("samples", [])
    if len(samples) < 2:
        parser.error("reference must contain at least two samples")

    rclpy.init()
    node = ReferenceReplay(samples, args.topic)
    try:
        period = 1.0 / args.publish_rate
        while rclpy.ok() and node.publisher.get_subscription_count() < 1:
            node.publish_at(0.0)
            rclpy.spin_once(node, timeout_sec=0.05)
        delay_end = time.monotonic() + args.start_delay
        while rclpy.ok() and time.monotonic() < delay_end:
            node.publish_at(0.0)
            rclpy.spin_once(node, timeout_sec=min(period, 0.05))
        started = time.monotonic()
        duration = float(samples[-1]["t"])
        node.get_logger().info(
            f"Replaying exact Elastic target reference: {len(samples)} samples, "
            f"{duration:.3f} s"
        )
        while rclpy.ok():
            cycle = time.monotonic()
            elapsed = min(duration, cycle - started)
            node.publish_at(elapsed)
            rclpy.spin_once(node, timeout_sec=0.0)
            if elapsed >= duration:
                break
            time.sleep(max(0.0, period - (time.monotonic() - cycle)))
        node.get_logger().info("Elastic target reference complete; holding final pose")
        while rclpy.ok():
            node.publish_at(duration)
            rclpy.spin_once(node, timeout_sec=period)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
