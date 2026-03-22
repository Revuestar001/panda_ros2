from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    panda_nmpc_share = FindPackageShare("panda_nmpc")
    params_file = LaunchConfiguration("params_file")

    node = Node(
        package="panda_nmpc",
        executable="obstacle_marker_visualizer_node",
        output="screen",
        parameters=[ParameterFile(params_file, allow_substs=True)],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [panda_nmpc_share, "config", "obstacle_marker_visualizer.yaml"]
                ),
            ),
            node,
        ]
    )
