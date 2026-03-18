from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("panda_moveit_config")

    launch_rviz = LaunchConfiguration("launch_rviz")
    launch_static_virtual_tf = LaunchConfiguration("launch_static_virtual_tf")

    # This launch intentionally does not start fake ros2_control or a fake joint state source.
    # It assumes MuJoCo is already publishing /joint_states and running robot_state_publisher.
    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "move_group.launch.py"])
        )
    )

    moveit_rviz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "moveit_rviz.launch.py"])
        ),
        condition=IfCondition(launch_rviz),
    )

    static_virtual_joint_tfs_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "static_virtual_joint_tfs.launch.py"])
        ),
        condition=IfCondition(launch_static_virtual_tf),
    )

    moveit_scope = GroupAction(
        [
            # [修改] MuJoCo 侧 joint_states 使用仿真时钟，MoveIt 必须跟它共用 use_sim_time，
            # 否则 current_state_monitor 会把机器人状态当成“过期状态”。
            SetParameter(name="use_sim_time", value=True),
            move_group_launch,
            moveit_rviz_launch,
            static_virtual_joint_tfs_launch,
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "launch_rviz",
                default_value="true",
                description="Launch MoveIt RViz alongside move_group.",
            ),
            DeclareLaunchArgument(
                "launch_static_virtual_tf",
                default_value="true",
                description="Publish the SRDF virtual joint TF, e.g. world -> link0.",
            ),
            moveit_scope,
        ]
    )
