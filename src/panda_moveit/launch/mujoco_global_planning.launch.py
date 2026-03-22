from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config_share = FindPackageShare("panda_moveit_config")
    panda_moveit_share = FindPackageShare("panda_moveit")
    panda_nmpc_share = FindPackageShare("panda_nmpc")
    moveit_config = MoveItConfigsBuilder(
        "panda", package_name="panda_moveit_config"
    ).to_moveit_configs()

    launch_rviz = LaunchConfiguration("launch_rviz")
    launch_static_virtual_tf = LaunchConfiguration("launch_static_virtual_tf")
    global_planning_params_file = LaunchConfiguration("global_planning_params_file")
    launch_interactive_target_marker = LaunchConfiguration("launch_interactive_target_marker")
    interactive_target_marker_params_file = LaunchConfiguration("interactive_target_marker_params_file")
    launch_obstacle_marker_visualizer = LaunchConfiguration("launch_obstacle_marker_visualizer")
    obstacle_marker_visualizer_params_file = LaunchConfiguration("obstacle_marker_visualizer_params_file")

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([moveit_config_share, "launch", "mujoco_moveit.launch.py"])
        ),
        launch_arguments={
            "launch_rviz": launch_rviz,
            "launch_static_virtual_tf": launch_static_virtual_tf,
        }.items(),
    )

    global_planning_node = Node(
        package="panda_moveit",
        executable="global_planning",
        output="screen",
        parameters=[
            global_planning_params_file,
            moveit_config.to_dict(),
        ],
    )

    interactive_target_marker_node = Node(
        package="panda_nmpc",
        executable="interactive_target_marker_node",
        output="screen",
        parameters=[interactive_target_marker_params_file],
        condition=IfCondition(launch_interactive_target_marker),
    )

    obstacle_marker_visualizer_node = Node(
        package="panda_nmpc",
        executable="obstacle_marker_visualizer_node",
        output="screen",
        parameters=[obstacle_marker_visualizer_params_file],
        condition=IfCondition(launch_obstacle_marker_visualizer),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("launch_static_virtual_tf", default_value="true"),
            DeclareLaunchArgument("launch_interactive_target_marker", default_value="true"),
            DeclareLaunchArgument("launch_obstacle_marker_visualizer", default_value="true"),
            DeclareLaunchArgument(
                "global_planning_params_file",
                default_value=PathJoinSubstitution(
                    [panda_moveit_share, "config", "global_planning.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "interactive_target_marker_params_file",
                default_value=PathJoinSubstitution(
                    [panda_nmpc_share, "config", "interactive_target_marker.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "obstacle_marker_visualizer_params_file",
                default_value=PathJoinSubstitution(
                    [panda_nmpc_share, "config", "obstacle_marker_visualizer.yaml"]
                ),
            ),
            moveit_launch,
            global_planning_node,
            interactive_target_marker_node,
            obstacle_marker_visualizer_node,
        ]
    )
