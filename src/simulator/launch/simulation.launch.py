import os
import json
import launch

import xacro
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

__author__ = "YueLin"

PACKAGE = get_package_share_directory(os.path.basename(
    os.path.dirname(os.path.dirname(__file__))
))


def _simulation_node(context):
    environment = LaunchConfiguration("environment").perform(context).strip()
    config = LaunchConfiguration("config").perform(context).strip()
    if not config:
        config = os.path.join(
            PACKAGE,
            "config",
            "environments",
            f"{environment}.yaml",
        ) if environment else os.path.join(PACKAGE, "config", "simulator.yaml")
    if not os.path.isfile(config):
        raise FileNotFoundError(f"Simulator configuration not found: {config}")

    overrides = {}
    map_file = LaunchConfiguration("map.file").perform(context).strip()
    if map_file:
        map_file = os.path.abspath(os.path.expanduser(map_file))
        if not os.path.isfile(map_file):
            raise FileNotFoundError(f"Custom map JSON not found: {map_file}")
        with open(map_file, encoding="utf-8") as stream:
            payload = json.load(stream)
        map_format = payload.get("format")
        supported_formats = {
            "neural-tracker-map",
            "neural-tracker-forest-map",
        }
        if map_format not in supported_formats:
            raise ValueError(
                "custom map must use neural-tracker-map or "
                "neural-tracker-forest-map format"
            )
        map_config = payload.get("map")
        if not isinstance(map_config, dict):
            raise ValueError("custom map is missing map metadata")
        overrides.update({
            "map.file": map_file,
            "map.width": float(map_config["width"]),
            "map.height": float(map_config["height"]),
            "map.resolution": float(map_config["resolution"]),
        })
        obstacles = payload.get("obstacles", [])
        tree_only_map = (
            isinstance(obstacles, list)
            and bool(obstacles)
            and all(
                isinstance(obstacle, dict)
                and obstacle.get("shape") == "tree"
                for obstacle in obstacles
            )
        )
        if map_format == "neural-tracker-forest-map" or tree_only_map:
            overrides["map.style"] = "forest"
        for actor in ("tracker", "target"):
            pose = payload.get(actor)
            if isinstance(pose, dict):
                overrides[f"{actor}.x"] = float(pose["x"])
                overrides[f"{actor}.y"] = float(pose["y"])
                overrides[f"{actor}.yaw"] = float(pose.get("yaw", 0.0))
    typed_overrides = (
        ("map.seed", int),
        ("map.obstacles", int),
        ("map.style", str),
        ("map.width", float),
        ("map.height", float),
        ("temporary_obstacle.enabled", lambda value: value.lower() == "true"),
        ("temporary_obstacle.auto_add", lambda value: value.lower() == "true"),
        ("temporary_obstacle.seed", int),
        ("temporary_obstacle.interval", float),
        ("temporary_obstacle.min_size", float),
        ("temporary_obstacle.max_size", float),
        ("temporary_obstacle.height", float),
        ("temporary_obstacle.placement_jitter", float),
        ("temporary_obstacle.minimum_spacing", float),
        ("temporary_obstacle.linear_motion_threshold", float),
        ("temporary_obstacle.stuck_timeout", float),
        ("target.external_control", lambda value: value.lower() == "true"),
    )
    for name, converter in typed_overrides:
        value = LaunchConfiguration(name).perform(context).strip()
        if value:
            overrides[name] = converter(value)

    return [Node(
        output="screen",
        name="simulation",
        package="simulator",
        executable="simulation",
        parameters=[config, overrides],
    )]


def generate_launch_description() -> launch.LaunchDescription:
    description = launch.LaunchDescription()

    description.add_action(DeclareLaunchArgument(
        "environment",
        default_value="",
        description="Named config from config/environments (for example forest)",
    ))
    description.add_action(DeclareLaunchArgument(
        "config",
        default_value="",
        description="Absolute YAML path; takes precedence over environment",
    ))
    description.add_action(DeclareLaunchArgument(
        "map.file",
        default_value="",
        description=(
            "Absolute neural-tracker-map or neural-tracker-forest-map JSON path"
        ),
    ))
    description.add_action(DeclareLaunchArgument(
        "rviz",
        default_value="true",
        description="Start RViz with the simulator layout",
    ))
    description.add_action(DeclareLaunchArgument(
        "rviz_clicked_point_topic",
        default_value="/clicked_point",
        description="Topic used by the RViz Publish Point tool",
    ))
    for name in ("map.seed", "map.obstacles", "map.style",
                 "map.width", "map.height",
                 "temporary_obstacle.enabled", "temporary_obstacle.auto_add",
                 "temporary_obstacle.seed",
                 "temporary_obstacle.interval", "temporary_obstacle.min_size",
                 "temporary_obstacle.max_size",
                 "temporary_obstacle.height",
                 "temporary_obstacle.placement_jitter",
                 "temporary_obstacle.minimum_spacing",
                 "temporary_obstacle.linear_motion_threshold",
                 "temporary_obstacle.stuck_timeout"):
        description.add_action(DeclareLaunchArgument(name, default_value=""))
    description.add_action(DeclareLaunchArgument(
        "target.external_control",
        default_value="",
        description="Drive target from /target/reference_pose when true",
    ))

    for model in sorted(os.listdir(os.path.join(PACKAGE, "urdf"))):
        robot = xacro.parse(open(os.path.join(PACKAGE, "urdf", model)))
        xacro.process_doc(robot)
        description.add_action(Node(
            name="robot_state_publisher",
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=os.path.splitext(model)[0],
            parameters=[{"robot_description": robot.toxml()}]
        ))
    
    description.add_action(OpaqueFunction(function=_simulation_node))

    description.add_action(Node(
        name="rviz",
        package="rviz2",
        executable="rviz2",
        # The simulator belongs to the neural/traditional tracker workflow.
        # YOPO owns and launches its RViz configuration from Yopo-Tracker.
        arguments=["-d", os.path.join(PACKAGE, "config", "rviz_neural.rviz")],
        remappings=[
            ("/clicked_point", LaunchConfiguration(
                "rviz_clicked_point_topic")),
        ],
        condition=IfCondition(LaunchConfiguration("rviz")),
    ))

    return description
