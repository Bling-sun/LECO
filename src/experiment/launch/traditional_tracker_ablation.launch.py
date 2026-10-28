"""Experiment-only Traditional tracker using current position, not prediction."""

from os import path

import launch
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def argument(name, default):
    return launch.actions.DeclareLaunchArgument(name, default_value=default)


def value(name):
    return launch.substitutions.LaunchConfiguration(name)


def float_value(name):
    return ParameterValue(value(name), value_type=float)


def generate_launch_description() -> launch.LaunchDescription:
    experiment_share = get_package_share_directory("experiment")
    description = launch.LaunchDescription([
        argument("truth_fallback_after", "3.9"),
    ])

    description.add_action(Node(
        output="screen",
        name="traditional_target_observation_mux",
        package="tracker",
        executable="target_observation_mux",
        parameters=[{
            "fallback_after": float_value("truth_fallback_after"),
        }],
    ))

    description.add_action(Node(
        output="screen",
        name="current_target_reference",
        package="experiment",
        executable="current_target_reference",
        parameters=[{
            "target_topic": "/target/fov_or_truth",
            "control_points": 7,
        }],
    ))

    description.add_action(Node(
        output="screen",
        name="tracking",
        package="tracker",
        executable="tracker",
        parameters=[
            path.join(
                experiment_share,
                "config",
                "traditional_maze1_no_prediction.yaml",
            ),
            {
                "timing.log_interval": 0.0,
                "target.topic": "/target/fov_or_truth",
                "target.direct": False,
                "target.use_fov_detection": True,
                "target.truth_fallback_after": float_value(
                    "truth_fallback_after"
                ),
            },
        ],
    ))
    return description
