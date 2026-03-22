from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    panda_nmpc_share = FindPackageShare("panda_nmpc")

    params_file = LaunchConfiguration("params_file")
    topic = LaunchConfiguration("topic")

    publisher_node = Node(
        package="panda_nmpc",
        executable="dynamic_sphere_obstacle_publisher",
        output="screen",
        parameters=[
            ParameterFile(params_file, allow_substs=True),
            {"topic": topic},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [panda_nmpc_share, "config", "dynamic_sphere_obstacle_publisher.yaml"]
                ),
            ),
            DeclareLaunchArgument("topic", default_value="/dynamic_sphere_obstacles"),
            publisher_node,
        ]
    )
