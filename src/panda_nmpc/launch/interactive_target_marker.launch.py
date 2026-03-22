from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    panda_nmpc_share = FindPackageShare("panda_nmpc")

    params_file = LaunchConfiguration("params_file")
    rviz_config_file = LaunchConfiguration("rviz_config_file")
    start_rviz = LaunchConfiguration("start_rviz")

    marker_node = Node(
        package="panda_nmpc",
        executable="interactive_target_marker_node",
        output="screen",
        parameters=[ParameterFile(params_file, allow_substs=True)],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(start_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [panda_nmpc_share, "config", "interactive_target_marker.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "rviz_config_file",
                default_value=PathJoinSubstitution(
                    [panda_nmpc_share, "rviz", "interactive_target_marker.rviz"]
                ),
            ),
            DeclareLaunchArgument(
                "start_rviz",
                default_value="true",
            ),
            marker_node,
            rviz_node,
        ]
    )
