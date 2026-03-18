from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    moveit_config_share = FindPackageShare("panda_moveit_config")

    launch_rviz = LaunchConfiguration("launch_rviz")
    launch_static_virtual_tf = LaunchConfiguration("launch_static_virtual_tf")
    plan_on_target_update = LaunchConfiguration("plan_on_target_update")
    planning_time = LaunchConfiguration("planning_time")
    num_planning_attempts = LaunchConfiguration("num_planning_attempts")
    goal_position_tolerance = LaunchConfiguration("goal_position_tolerance")
    goal_orientation_tolerance = LaunchConfiguration("goal_orientation_tolerance")
    planning_target_frame = LaunchConfiguration("planning_target_frame")
    launch_static_obstacles = LaunchConfiguration("launch_static_obstacles")

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
            {
                "use_sim_time": True,
                "plan_on_target_update": ParameterValue(plan_on_target_update, value_type=bool),
                "planning_time": ParameterValue(planning_time, value_type=float),
                "num_planning_attempts": ParameterValue(num_planning_attempts, value_type=int),
                "goal_position_tolerance": ParameterValue(goal_position_tolerance, value_type=float),
                "goal_orientation_tolerance": ParameterValue(goal_orientation_tolerance, value_type=float),
                "target_frame": ParameterValue(planning_target_frame, value_type=str),
            }
        ],
    )

    static_obstacles_node = Node(
        package="panda_moveit",
        executable="static_obstacles_publisher",
        output="screen",
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(launch_static_obstacles),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("launch_static_virtual_tf", default_value="true"),
            DeclareLaunchArgument("launch_static_obstacles", default_value="true"),
            DeclareLaunchArgument("plan_on_target_update", default_value="false"),
            DeclareLaunchArgument("planning_time", default_value="3.0"),
            DeclareLaunchArgument("num_planning_attempts", default_value="5"),
            DeclareLaunchArgument("goal_position_tolerance", default_value="0.005"),
            DeclareLaunchArgument("goal_orientation_tolerance", default_value="0.05"),
            DeclareLaunchArgument("planning_target_frame", default_value=""),
            moveit_launch,
            global_planning_node,
            static_obstacles_node,
        ]
    )
