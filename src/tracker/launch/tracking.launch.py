import launch
from os import path

from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

__author__ = "YueLin"

PACKAGE = get_package_share_directory(path.basename(
    path.dirname(path.dirname(__file__))
))


def generate_launch_description() -> launch.LaunchDescription:
    description = launch.LaunchDescription()
    timing_arg = launch.actions.DeclareLaunchArgument(
        "timing_log_interval", default_value="2.0",
        description="Seconds between average runtime timing logs; <=0 disables logs"
    )
    description.add_action(timing_arg)
    description.add_action(launch.actions.DeclareLaunchArgument(
        "target_topic", default_value="/target/detection"
    ))
    description.add_action(launch.actions.DeclareLaunchArgument(
        "target_direct", default_value="false"
    ))
    description.add_action(launch.actions.DeclareLaunchArgument(
        "max_linear_speed", default_value="2.0",
        description="Traditional tracker maximum linear speed in m/s"
    ))
    description.add_action(launch.actions.DeclareLaunchArgument(
        "max_linear_accel", default_value="3.0",
        description="Traditional tracker maximum linear acceleration in m/s^2"
    ))
    description.add_action(launch.actions.DeclareLaunchArgument(
        "dynamic_accel_by_visibility", default_value="false",
        description="Switch maximum acceleration using raw FOV visibility"
    ))
    description.add_action(launch.actions.DeclareLaunchArgument(
        "visible_max_linear_accel", default_value="3.0",
        description="Maximum acceleration while the target is visible"
    ))
    description.add_action(launch.actions.DeclareLaunchArgument(
        "lost_max_linear_accel", default_value="3.0",
        description="Maximum acceleration while the target is outside FOV"
    ))
    description.add_action(launch.actions.DeclareLaunchArgument(
        "use_fov_detection", default_value="false",
        description="Use the shared FOV detection/truth fallback mux"
    ))
    description.add_action(launch.actions.DeclareLaunchArgument(
        "truth_fallback_after", default_value="3.0",
        description="Seconds of continuous FOV loss before truth fallback"
    ))

    description.add_action(Node(
        output="screen",
        name="traditional_target_observation_mux",
        package="tracker",
        executable="target_observation_mux",
        parameters=[{
            "fallback_after": ParameterValue(
                launch.substitutions.LaunchConfiguration(
                    "truth_fallback_after"), value_type=float),
        }],
        condition=IfCondition(
            launch.substitutions.LaunchConfiguration("use_fov_detection")),
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
        executable="tracker",
        parameters=[
            path.join(PACKAGE, "config", "planner.yaml"),
            {"timing.log_interval": ParameterValue(
                 launch.substitutions.LaunchConfiguration("timing_log_interval"),
                 value_type=float),
             "target.topic": launch.substitutions.LaunchConfiguration("target_topic"),
             "tracker.max_vel": ParameterValue(
                 launch.substitutions.LaunchConfiguration("max_linear_speed"),
                 value_type=float),
             "tracker.max_acc": ParameterValue(
                 launch.substitutions.LaunchConfiguration("max_linear_accel"),
                 value_type=float),
             "tracker.dynamic_accel_by_visibility": ParameterValue(
                 launch.substitutions.LaunchConfiguration(
                     "dynamic_accel_by_visibility"), value_type=bool),
             "tracker.max_acc_visible": ParameterValue(
                 launch.substitutions.LaunchConfiguration(
                     "visible_max_linear_accel"), value_type=float),
             "tracker.max_acc_lost": ParameterValue(
                 launch.substitutions.LaunchConfiguration(
                     "lost_max_linear_accel"), value_type=float),
             "target.direct": ParameterValue(
                 launch.substitutions.LaunchConfiguration("target_direct"),
                 value_type=bool)},
        ]
    ))

    return description
