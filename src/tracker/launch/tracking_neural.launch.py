import launch
from os import path
from pathlib import Path

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

__author__ = "YueLin (neural variant)"

PACKAGE = get_package_share_directory(path.basename(
    path.dirname(path.dirname(__file__))
))

# Absolute path to the exported ONNX policy. Override at launch with
#   ros2 launch tracker tracking_neural.launch.py onnx_path:=/abs/path/policy.onnx
DEFAULT_ONNX = str(
    Path(PACKAGE).parents[3] / "neural_tracker/models/v41_directional_recovery/policy.onnx"
)


def generate_launch_description() -> launch.LaunchDescription:
    description = launch.LaunchDescription()

    onnx_arg = launch.actions.DeclareLaunchArgument(
        "onnx_path", default_value=DEFAULT_ONNX,
        description="Absolute path to the exported policy.onnx"
    )
    description.add_action(onnx_arg)
    description.add_action(launch.actions.DeclareLaunchArgument(
        "onnx_threads", default_value="4",
        description="ONNX Runtime CPU intra-op thread count"
    ))
    timing_arg = launch.actions.DeclareLaunchArgument(
        "timing_log_interval", default_value="2.0",
        description="Seconds between average runtime timing logs; <=0 disables logs"
    )
    description.add_action(timing_arg)
    description.add_action(launch.actions.DeclareLaunchArgument(
        "target_topic", default_value="/target/detection",
        description="PoseStamped target input; use /target/ground_truth for direct tests"
    ))
    description.add_action(launch.actions.DeclareLaunchArgument(
        "target_direct", default_value="false",
        description="Bypass target visibility flags and read positions directly"
    ))
    description.add_action(launch.actions.DeclareLaunchArgument(
        "use_fov_detection", default_value="true",
        description=(
            "Use FOV visibility; after 3 s lost, temporarily read "
            "/target/ground_truth until FOV recovery"
        )
    ))
    description.add_action(launch.actions.DeclareLaunchArgument(
        "truth_fallback_after", default_value="3.0",
        description="Seconds of continuous FOV loss before truth fallback"
    ))

    description.add_action(launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            path.join(path.dirname(__file__), "prediction.launch.py")
        ),
        launch_arguments={
            "target_topic": launch.substitutions.LaunchConfiguration("target_topic"),
            "target_direct": launch.substitutions.LaunchConfiguration("target_direct"),
        }.items(),
    ))

    description.add_action(Node(
        output="screen",
        name="tracking",
        package="tracker",
        executable="tracker_neural",
        parameters=[
            path.join(PACKAGE, "config", "planner.yaml"),
            {"neural.onnx_path": launch.substitutions.LaunchConfiguration("onnx_path"),
             "neural.onnx_threads": ParameterValue(
                 launch.substitutions.LaunchConfiguration("onnx_threads"),
                 value_type=int),
             "target.topic": launch.substitutions.LaunchConfiguration("target_topic"),
             "target.direct": ParameterValue(
                 launch.substitutions.LaunchConfiguration("target_direct"),
                 value_type=bool),
             "target.use_fov_detection": ParameterValue(
                 launch.substitutions.LaunchConfiguration("use_fov_detection"),
                 value_type=bool),
             "target.truth_fallback_after": launch.substitutions.LaunchConfiguration(
                 "truth_fallback_after"),
             "neural.omega_sign": 1.0,
             "neural.stationary_guard": True,
             "neural.stationary_guard_distance_tolerance": 0.35,
             "neural.stationary_guard_bearing_tolerance": 0.20,
             "neural.stationary_guard_target_speed": 0.05,
             "neural.stationary_guard_reverse_speed": -0.10,
             "neural.stationary_guard_max_omega": 0.30,
             "neural.fov_guard": True,
             "neural.fov_guard_min_bearing": 0.08,
             "neural.fov_guard_kp": 1.0,
             "neural.fov_guard_max_omega": 0.8,
             "neural.range_guard": True,
             "neural.range_guard_start": 3.0,
             "neural.range_guard_release": 2.75,
             "neural.range_guard_max_bearing": 0.45,
             "neural.range_guard_min_speed": 2.0,
             "neural.range_guard_kp": 0.8,
             "neural.range_guard_horizon": 0.6,
             "neural.range_guard_clearance": 0.10,
             "neural.timing_log_interval": ParameterValue(
                 launch.substitutions.LaunchConfiguration("timing_log_interval"),
                 value_type=float)},
        ]
    ))

    return description
