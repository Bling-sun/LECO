import launch
from os import path

from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory

__author__ = "YueLin"

PACKAGE = get_package_share_directory(path.basename(
    path.dirname(path.dirname(__file__))
))


def generate_launch_description() -> launch.LaunchDescription:
    description = launch.LaunchDescription()

    description.add_action(launch.actions.DeclareLaunchArgument(
        "target_topic", default_value="/target/detection",
        description="PoseStamped topic used as predictor target input"
    ))
    description.add_action(launch.actions.DeclareLaunchArgument(
        "target_direct", default_value="false",
        description="Treat every target message as visible ground-truth position"
    ))
    
    description.add_action(Node(
        output="screen",
        name="prediction",
        package="tracker",
        executable="prediction",
        parameters=[
            path.join(PACKAGE, "config", "predictor.yaml"),
            {"target.topic": launch.substitutions.LaunchConfiguration("target_topic"),
             "target.direct": ParameterValue(
                 launch.substitutions.LaunchConfiguration("target_direct"),
                 value_type=bool)},
        ]
    ))

    return description
