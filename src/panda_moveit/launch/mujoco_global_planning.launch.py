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
    planner_id = LaunchConfiguration("planner_id")
    max_multi_plan_rounds = LaunchConfiguration("max_multi_plan_rounds")
    min_rounds_before_early_stop = LaunchConfiguration("min_rounds_before_early_stop")
    goal_position_tolerance = LaunchConfiguration("goal_position_tolerance")
    goal_orientation_tolerance = LaunchConfiguration("goal_orientation_tolerance")
    accepted_terminal_position_error = LaunchConfiguration("accepted_terminal_position_error")
    accepted_terminal_orientation_error = LaunchConfiguration("accepted_terminal_orientation_error")
    use_exact_ik_joint_targets = LaunchConfiguration("use_exact_ik_joint_targets")
    ik_seed_attempts = LaunchConfiguration("ik_seed_attempts")
    max_ik_solutions = LaunchConfiguration("max_ik_solutions")
    ik_timeout = LaunchConfiguration("ik_timeout")
    ik_min_solution_distance = LaunchConfiguration("ik_min_solution_distance")
    planning_retries_per_ik_solution = LaunchConfiguration("planning_retries_per_ik_solution")
    planning_target_frame = LaunchConfiguration("planning_target_frame")
    launch_static_obstacles = LaunchConfiguration("launch_static_obstacles")
    load_static_obstacles_from_scene = LaunchConfiguration("load_static_obstacles_from_scene")
    obstacle_config_path = LaunchConfiguration("obstacle_config_path")
    scene_xml_path = LaunchConfiguration("scene_xml_path")
    scene_world_frame = LaunchConfiguration("scene_world_frame")

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
                "planner_id": ParameterValue(planner_id, value_type=str),
                "max_multi_plan_rounds": ParameterValue(max_multi_plan_rounds, value_type=int),
                "min_rounds_before_early_stop": ParameterValue(min_rounds_before_early_stop, value_type=int),
                "goal_position_tolerance": ParameterValue(goal_position_tolerance, value_type=float),
                "goal_orientation_tolerance": ParameterValue(goal_orientation_tolerance, value_type=float),
                "accepted_terminal_position_error": ParameterValue(
                    accepted_terminal_position_error, value_type=float
                ),
                "accepted_terminal_orientation_error": ParameterValue(
                    accepted_terminal_orientation_error, value_type=float
                ),
                "use_exact_ik_joint_targets": ParameterValue(
                    use_exact_ik_joint_targets, value_type=bool
                ),
                "ik_seed_attempts": ParameterValue(ik_seed_attempts, value_type=int),
                "max_ik_solutions": ParameterValue(max_ik_solutions, value_type=int),
                "ik_timeout": ParameterValue(ik_timeout, value_type=float),
                "ik_min_solution_distance": ParameterValue(
                    ik_min_solution_distance, value_type=float
                ),
                "planning_retries_per_ik_solution": ParameterValue(
                    planning_retries_per_ik_solution, value_type=int
                ),
                "target_frame": ParameterValue(planning_target_frame, value_type=str),
                "load_static_obstacles_from_scene": ParameterValue(
                    load_static_obstacles_from_scene, value_type=bool
                ),
                "obstacle_config_path": ParameterValue(obstacle_config_path, value_type=str),
                "scene_xml_path": ParameterValue(scene_xml_path, value_type=str),
                "scene_world_frame": ParameterValue(scene_world_frame, value_type=str),
            }
        ],
    )

    static_obstacles_node = Node(
        package="panda_moveit",
        executable="static_obstacles_publisher",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "obstacle_config_path": ParameterValue(obstacle_config_path, value_type=str),
                "scene_xml_path": ParameterValue(scene_xml_path, value_type=str),
                "world_frame": ParameterValue(scene_world_frame, value_type=str),
            }
        ],
        condition=IfCondition(launch_static_obstacles),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("launch_static_virtual_tf", default_value="true"),
            DeclareLaunchArgument("launch_static_obstacles", default_value="false"),
            DeclareLaunchArgument("load_static_obstacles_from_scene", default_value="true"),
            DeclareLaunchArgument(
                "obstacle_config_path",
                default_value="/home/cyh/panda_ros2/model/franka_emika_panda/static_sphere_obstacles.xml",
            ),
            DeclareLaunchArgument(
                "scene_xml_path",
                default_value="/home/cyh/panda_ros2/model/franka_emika_panda/scene_tau_ros.xml",
            ),
            DeclareLaunchArgument("scene_world_frame", default_value="world"),
            DeclareLaunchArgument("plan_on_target_update", default_value="false"),
            DeclareLaunchArgument("planning_time", default_value="3.0"),
            DeclareLaunchArgument("num_planning_attempts", default_value="5"),
            DeclareLaunchArgument("planner_id", default_value="RRTConnectkConfigDefault"),
            DeclareLaunchArgument("max_multi_plan_rounds", default_value="40"),
            DeclareLaunchArgument("min_rounds_before_early_stop", default_value="3"),
            DeclareLaunchArgument("goal_position_tolerance", default_value="0.0005"),
            DeclareLaunchArgument("goal_orientation_tolerance", default_value="0.005"),
            DeclareLaunchArgument("accepted_terminal_position_error", default_value="0.0005"),
            DeclareLaunchArgument("accepted_terminal_orientation_error", default_value="0.005"),
            DeclareLaunchArgument("use_exact_ik_joint_targets", default_value="true"),
            DeclareLaunchArgument("ik_seed_attempts", default_value="80"),
            DeclareLaunchArgument("max_ik_solutions", default_value="12"),
            DeclareLaunchArgument("ik_timeout", default_value="0.05"),
            DeclareLaunchArgument("ik_min_solution_distance", default_value="0.1"),
            DeclareLaunchArgument("planning_retries_per_ik_solution", default_value="3"),
            DeclareLaunchArgument("planning_target_frame", default_value=""),
            moveit_launch,
            global_planning_node,
            static_obstacles_node,
        ]
    )
