from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue


def generate_launch_description():
    goal_frame = LaunchConfiguration("goal_frame")
    target_x = LaunchConfiguration("target_x")
    target_y = LaunchConfiguration("target_y")
    target_z = LaunchConfiguration("target_z")
    target_qx = LaunchConfiguration("target_qx")
    target_qy = LaunchConfiguration("target_qy")
    target_qz = LaunchConfiguration("target_qz")
    target_qw = LaunchConfiguration("target_qw")
    auto_plan = LaunchConfiguration("auto_plan")
    stop_after_success = LaunchConfiguration("stop_after_success")
    command_period_ms = LaunchConfiguration("command_period_ms")
    plan_delay_after_publish_s = LaunchConfiguration("plan_delay_after_publish_s")

    goal_commander_node = Node(
        package="panda_moveit",
        executable="goal_commander",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "target_frame": ParameterValue(goal_frame, value_type=str),
                "target_x": ParameterValue(target_x, value_type=float),
                "target_y": ParameterValue(target_y, value_type=float),
                "target_z": ParameterValue(target_z, value_type=float),
                "target_qx": ParameterValue(target_qx, value_type=float),
                "target_qy": ParameterValue(target_qy, value_type=float),
                "target_qz": ParameterValue(target_qz, value_type=float),
                "target_qw": ParameterValue(target_qw, value_type=float),
                "auto_plan": ParameterValue(auto_plan, value_type=bool),
                "stop_after_success": ParameterValue(stop_after_success, value_type=bool),
                "command_period_ms": ParameterValue(command_period_ms, value_type=int),
                "plan_delay_after_publish_s": ParameterValue(plan_delay_after_publish_s, value_type=float),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("goal_frame", default_value="world"),
            DeclareLaunchArgument("target_x", default_value="0.74"),
            DeclareLaunchArgument("target_y", default_value="0.0"),
            DeclareLaunchArgument("target_z", default_value="0.32"),
            # [修改] 默认姿态改成与当前 Panda 末端常用抓手朝向一致（绕 X 轴旋转 pi）。
            DeclareLaunchArgument("target_qx", default_value="1.0"),
            DeclareLaunchArgument("target_qy", default_value="0.0"),
            DeclareLaunchArgument("target_qz", default_value="0.0"),
            DeclareLaunchArgument("target_qw", default_value="0.0"),
            DeclareLaunchArgument("auto_plan", default_value="true"),
            DeclareLaunchArgument("stop_after_success", default_value="true"),
            DeclareLaunchArgument("command_period_ms", default_value="500"),
            DeclareLaunchArgument("plan_delay_after_publish_s", default_value="0.2"),
            goal_commander_node,
        ]
    )
