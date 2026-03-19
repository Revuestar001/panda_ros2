from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    panda_ros_share = FindPackageShare("panda_ros")
    panda_nmpc_share = FindPackageShare("panda_nmpc")

    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_global_trajectory = LaunchConfiguration("use_global_trajectory")
    urdf_path = LaunchConfiguration("urdf_path")
    ee_frame_name = LaunchConfiguration("ee_frame_name")
    joint_states_topic = LaunchConfiguration("joint_states_topic")
    joint_trajectory_topic = LaunchConfiguration("joint_trajectory_topic")
    nmpc_result_topic = LaunchConfiguration("nmpc_result_topic")

    target_x = LaunchConfiguration("target_x")
    target_y = LaunchConfiguration("target_y")
    target_z = LaunchConfiguration("target_z")
    target_qx = LaunchConfiguration("target_qx")
    target_qy = LaunchConfiguration("target_qy")
    target_qz = LaunchConfiguration("target_qz")
    target_qw = LaunchConfiguration("target_qw")

    sorr_pos_damping_ratio = LaunchConfiguration("sorr_pos_damping_ratio")
    sorr_pos_natural_frequency = LaunchConfiguration("sorr_pos_natural_frequency")
    sorr_pos_max_linear_velocity = LaunchConfiguration("sorr_pos_max_linear_velocity")
    sorr_pos_max_linear_acceleration = LaunchConfiguration("sorr_pos_max_linear_acceleration")

    sorr_rot_damping_ratio = LaunchConfiguration("sorr_rot_damping_ratio")
    sorr_rot_natural_frequency = LaunchConfiguration("sorr_rot_natural_frequency")
    sorr_rot_max_angular_velocity = LaunchConfiguration("sorr_rot_max_angular_velocity")
    sorr_rot_max_angular_acceleration = LaunchConfiguration("sorr_rot_max_angular_acceleration")

    nmpc_node = Node(
        package="panda_nmpc",
        executable="nmpc_tau",
        output="screen",
        parameters=[
            ParameterFile(params_file, allow_substs=True),
            {
                "use_sim_time": ParameterValue(use_sim_time, value_type=bool),
                "use_global_trajectory": ParameterValue(use_global_trajectory, value_type=bool),
                "urdf_path": ParameterValue(urdf_path, value_type=str),
                "ee_frame_name": ParameterValue(ee_frame_name, value_type=str),
                "joint_states_topic": ParameterValue(joint_states_topic, value_type=str),
                "joint_trajectory_topic": ParameterValue(joint_trajectory_topic, value_type=str),
                "nmpc_result_topic": ParameterValue(nmpc_result_topic, value_type=str),
                "target_x": ParameterValue(target_x, value_type=float),
                "target_y": ParameterValue(target_y, value_type=float),
                "target_z": ParameterValue(target_z, value_type=float),
                "target_qx": ParameterValue(target_qx, value_type=float),
                "target_qy": ParameterValue(target_qy, value_type=float),
                "target_qz": ParameterValue(target_qz, value_type=float),
                "target_qw": ParameterValue(target_qw, value_type=float),
                "sorr_pos_damping_ratio": ParameterValue(sorr_pos_damping_ratio, value_type=float),
                "sorr_pos_natural_frequency": ParameterValue(sorr_pos_natural_frequency, value_type=float),
                "sorr_pos_max_linear_velocity": ParameterValue(sorr_pos_max_linear_velocity, value_type=float),
                "sorr_pos_max_linear_acceleration": ParameterValue(
                    sorr_pos_max_linear_acceleration, value_type=float
                ),
                "sorr_rot_damping_ratio": ParameterValue(sorr_rot_damping_ratio, value_type=float),
                "sorr_rot_natural_frequency": ParameterValue(sorr_rot_natural_frequency, value_type=float),
                "sorr_rot_max_angular_velocity": ParameterValue(sorr_rot_max_angular_velocity, value_type=float),
                "sorr_rot_max_angular_acceleration": ParameterValue(
                    sorr_rot_max_angular_acceleration, value_type=float
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution([panda_nmpc_share, "config", "nmpc_tau.yaml"]),
            ),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("use_global_trajectory", default_value="false"),
            DeclareLaunchArgument(
                "urdf_path",
                default_value=PathJoinSubstitution([panda_ros_share, "model", "panda_tau_sim.urdf"]),
            ),
            DeclareLaunchArgument("ee_frame_name", default_value="ee_center_body"),
            DeclareLaunchArgument("joint_states_topic", default_value="/joint_states"),
            DeclareLaunchArgument("joint_trajectory_topic", default_value="/global_joint_trajectory"),
            DeclareLaunchArgument("nmpc_result_topic", default_value="/nmpc_result"),
            DeclareLaunchArgument("target_x", default_value="0.25"),
            DeclareLaunchArgument("target_y", default_value="0.3"),
            DeclareLaunchArgument("target_z", default_value="0.25"),
            DeclareLaunchArgument("target_qx", default_value="1.0"),
            DeclareLaunchArgument("target_qy", default_value="0.0"),
            DeclareLaunchArgument("target_qz", default_value="0.0"),
            DeclareLaunchArgument("target_qw", default_value="0.0"),
            DeclareLaunchArgument("sorr_pos_damping_ratio", default_value="1.0"),
            DeclareLaunchArgument("sorr_pos_natural_frequency", default_value="10.0"),
            DeclareLaunchArgument("sorr_pos_max_linear_velocity", default_value="1.0"),
            DeclareLaunchArgument("sorr_pos_max_linear_acceleration", default_value="10.0"),
            DeclareLaunchArgument("sorr_rot_damping_ratio", default_value="1.0"),
            DeclareLaunchArgument("sorr_rot_natural_frequency", default_value="10.0"),
            DeclareLaunchArgument("sorr_rot_max_angular_velocity", default_value="0.7845"),
            DeclareLaunchArgument("sorr_rot_max_angular_acceleration", default_value="0.7845"),
            nmpc_node,
        ]
    )
