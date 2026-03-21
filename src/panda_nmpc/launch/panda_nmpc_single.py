from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    panda_nmpc_share = FindPackageShare("panda_nmpc")

    nmpc_params_file = LaunchConfiguration("nmpc_params_file")

    nmpc_node = Node(
        package="panda_nmpc",
        executable="nmpc_tau",
        output="screen",
        # 单节点模式下，所有运行时参数都以 YAML 为唯一入口，避免 launch 默认值覆盖调参结果。
        parameters=[ParameterFile(nmpc_params_file, allow_substs=True)],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "nmpc_params_file",
                default_value=PathJoinSubstitution([panda_nmpc_share, "config", "nmpc_tau.yaml"]),
            ),
            nmpc_node,
        ]
    )
