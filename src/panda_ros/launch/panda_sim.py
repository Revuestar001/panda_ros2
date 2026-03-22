#!/usr/bin/env python3
#
# Copyright 2026 PAL Robotics S.L.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tutorial 3: PID Control

This tutorial demonstrates using PID controllers with MuJoCo simulation.
PID control enables velocity and effort control modes by using motor actuators
instead of position actuators in the MJCF model.

Key concepts:
- Using demo_resources/pid_control/test_robot_pid.xml with motor actuators
- Configuring PID gains in config/mujoco_pid.yaml
- Position and velocity command interfaces
- Motor actuators for torque-based control

Resources used:
- demo_resources/scenes/scene_pid.xml (scene with PID robot)
- demo_resources/pid_control/test_robot_pid.xml (robot with motor actuators)
- config/mujoco_pid.yaml (PID gain configuration)

Usage:
    ros2 launch mujoco_ros2_control_demos 03_pid_control.launch.py
    ros2 launch mujoco_ros2_control_demos 03_pid_control.launch.py headless:=true
"""

import os
import subprocess

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.conditions import IfCondition
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue, ParameterFile
from launch_ros.substitutions import FindPackageShare


def launch_setup(context, *args, **kwargs):
    pkg_share = FindPackageShare("panda_ros")

    obstacle_config_path = LaunchConfiguration("obstacle_config_path").perform(context)
    scene_xml_path = LaunchConfiguration("scene_xml_path").perform(context)
    dynamic_obstacle_topic = LaunchConfiguration("dynamic_obstacle_topic")
    target_pose_topic = LaunchConfiguration("target_pose_topic")
    start_target_pose_marker = LaunchConfiguration("start_target_pose_marker")
    start_target_pose_marker_rviz = LaunchConfiguration("start_target_pose_marker_rviz")
    target_pose_marker_params_file = LaunchConfiguration("target_pose_marker_params_file")
    target_pose_marker_rviz_config_file = LaunchConfiguration("target_pose_marker_rviz_config_file")
    start_obstacle_marker_visualizer = LaunchConfiguration("start_obstacle_marker_visualizer")
    obstacle_marker_visualizer_params_file = LaunchConfiguration("obstacle_marker_visualizer_params_file")
    dynamic_obstacle_publisher_params_file = LaunchConfiguration("dynamic_obstacle_publisher_params_file")
    target_pose_marker_params = ParameterFile(target_pose_marker_params_file, allow_substs=True)
    obstacle_marker_visualizer_params = ParameterFile(obstacle_marker_visualizer_params_file, allow_substs=True)
    sync_executable = os.path.join(
        get_package_prefix("panda_nmpc"),
        "lib",
        "panda_nmpc",
        "sync_static_sphere_scene.py",
    )
    subprocess.run(
        [
            sync_executable,
            "--obstacle-config",
            obstacle_config_path,
            "--scene-xml",
            scene_xml_path,
        ],
        check=True,
    )

    # Build robot description with PID control enabled
    # This uses demo_resources/scenes/scene_pid.xml which includes the PID robot model
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([pkg_share, "model", "panda_tau_sim.urdf"]),
            " headless:=",
            LaunchConfiguration("headless"),
            " use_mjcf_from_topic:=false",
            " use_transmissions:=false",
        ]
    )

    robot_description_str = robot_description_content.perform(context)
    robot_description = {"robot_description": ParameterValue(value=robot_description_str, value_type=str)}

    parameters_file = PathJoinSubstitution([pkg_share, "config", "pinoController.yaml"])

    nodes = []

    # Robot state publisher
    nodes.append(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="both",
            parameters=[robot_description, {"use_sim_time": True}],
        )
    )

    nodes.append(
        Node(
            package="panda_ros",
            executable="panda_test_node",
            output="both",
            on_exit=Shutdown(),
        )
    )

    # ros2_control node with MuJoCo - PID gains loaded from mujoco_pid.yaml
    nodes.append(
        Node(
            package="mujoco_ros2_control",
            executable="ros2_control_node",
            emulate_tty=True,
            output="both",
            parameters=[
                {"use_sim_time": True},
                ParameterFile(parameters_file),
                {
                    "mujoco_plugins.dynamic_sphere_visualization.type":
                        "mujoco_ros2_control_plugins/DynamicSphereVisualizationPlugin",
                    "mujoco_plugins.dynamic_sphere_visualization.topic": dynamic_obstacle_topic,
                    "mujoco_plugins.dynamic_sphere_visualization.max_spheres": 8,
                    "mujoco_plugins.dynamic_sphere_visualization.timeout_sec": 0.25,
                    "mujoco_plugins.dynamic_sphere_visualization.hidden_z": -5.0,
                    "mujoco_plugins.dynamic_sphere_visualization.body_name_prefix":
                        "dynamic_obstacle_mocap_",
                    "mujoco_plugins.dynamic_sphere_visualization.geom_name_prefix":
                        "dynamic_obstacle_geom_",
                },
            ],
            remappings=(
                [("~/robot_description", "/robot_description")] if os.environ.get("ROS_DISTRO") == "humble" else []
            ),
            on_exit=Shutdown(),
        )
    )

    nodes.append(
        Node(
            package="panda_nmpc",
            executable="interactive_target_marker_node",
            output="both",
            parameters=[
                target_pose_marker_params,
                {
                    "use_sim_time": True,
                    "target_pose_topic": target_pose_topic,
                },
            ],
            condition=IfCondition(start_target_pose_marker),
        )
    )

    nodes.append(
        Node(
            package="rviz2",
            executable="rviz2",
            output="both",
            arguments=["-d", target_pose_marker_rviz_config_file],
            parameters=[{"use_sim_time": True}],
            condition=IfCondition(start_target_pose_marker_rviz),
        )
    )

    nodes.append(
        Node(
            package="panda_nmpc",
            executable="obstacle_marker_visualizer_node",
            output="both",
            parameters=[
                obstacle_marker_visualizer_params,
                {
                    "use_sim_time": True,
                    "obstacle_config_path": obstacle_config_path,
                    "dynamic_obstacle_topic": dynamic_obstacle_topic,
                },
            ],
            condition=IfCondition(start_obstacle_marker_visualizer),
        )
    )

    nodes.append(
        Node(
            package="panda_nmpc",
            executable="dynamic_sphere_obstacle_publisher",
            output="both",
            parameters=[
                ParameterFile(dynamic_obstacle_publisher_params_file, allow_substs=True),
                {
                    "use_sim_time": True,
                    "topic": dynamic_obstacle_topic,
                },
            ],
            condition=IfCondition(LaunchConfiguration("start_dynamic_obstacle_publisher")),
        )
    )

    # Controller spawners
    # controllers_to_spawn = ["joint_state_broadcaster", "effort_controller", "gripper_controller"]
    controllers_to_spawn = ["joint_state_broadcaster", "effort_controller"]
    for controller in controllers_to_spawn:
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=[controller, "--param-file", parameters_file],
                output="both",
            )
        )

    return nodes


def generate_launch_description():
    headless = DeclareLaunchArgument(
        "headless",
        default_value="false",
        description="Run simulation without visualization window",
    )
    obstacle_config_path = DeclareLaunchArgument(
        "obstacle_config_path",
        default_value="/home/cyh/panda_ros2/model/franka_emika_panda/static_sphere_obstacles.xml",
        description="Canonical static sphere obstacle config shared by MuJoCo, MoveIt and acados",
    )
    dynamic_obstacle_topic = DeclareLaunchArgument(
        "dynamic_obstacle_topic",
        default_value="/dynamic_sphere_obstacles",
        description="Shared dynamic sphere topic used by NMPC and MuJoCo visualization.",
    )
    target_pose_topic = DeclareLaunchArgument(
        "target_pose_topic",
        default_value="/global_target_pose",
        description="Pose topic published from the RViz interactive target marker.",
    )
    start_target_pose_marker = DeclareLaunchArgument(
        "start_target_pose_marker",
        default_value="true",
        description="Start the RViz interactive target marker node.",
    )
    start_target_pose_marker_rviz = DeclareLaunchArgument(
        "start_target_pose_marker_rviz",
        default_value="true",
        description="Start RViz2 with the target marker config.",
    )
    target_pose_marker_params_file = DeclareLaunchArgument(
        "target_pose_marker_params_file",
        default_value=PathJoinSubstitution(
            [FindPackageShare("panda_nmpc"), "config", "interactive_target_marker.yaml"]
        ),
        description="Parameter file for the RViz interactive target marker node.",
    )
    target_pose_marker_rviz_config_file = DeclareLaunchArgument(
        "target_pose_marker_rviz_config_file",
        default_value=PathJoinSubstitution(
            [FindPackageShare("panda_nmpc"), "rviz", "interactive_target_marker.rviz"]
        ),
        description="RViz2 config file used for the target pose interactive marker.",
    )
    start_obstacle_marker_visualizer = DeclareLaunchArgument(
        "start_obstacle_marker_visualizer",
        default_value="true",
        description="Start the RViz obstacle marker visualizer node.",
    )
    obstacle_marker_visualizer_params_file = DeclareLaunchArgument(
        "obstacle_marker_visualizer_params_file",
        default_value=PathJoinSubstitution(
            [FindPackageShare("panda_nmpc"), "config", "obstacle_marker_visualizer.yaml"]
        ),
        description="Parameter file for the RViz obstacle marker visualizer node.",
    )
    start_dynamic_obstacle_publisher = DeclareLaunchArgument(
        "start_dynamic_obstacle_publisher",
        default_value="true",
        description="Start the demo dynamic sphere obstacle publisher together with MuJoCo simulation.",
    )
    dynamic_obstacle_publisher_params_file = DeclareLaunchArgument(
        "dynamic_obstacle_publisher_params_file",
        default_value=PathJoinSubstitution(
            [FindPackageShare("panda_nmpc"), "config", "dynamic_sphere_obstacle_publisher.yaml"]
        ),
        description="Parameter file for the demo dynamic sphere obstacle publisher.",
    )
    scene_xml_path = DeclareLaunchArgument(
        "scene_xml_path",
        default_value="/home/cyh/panda_ros2/model/franka_emika_panda/scene_tau_ros.xml",
        description="MuJoCo scene XML that will be synchronized before startup",
    )

    return LaunchDescription(
        [
            headless,
            obstacle_config_path,
            dynamic_obstacle_topic,
            target_pose_topic,
            start_target_pose_marker,
            start_target_pose_marker_rviz,
            target_pose_marker_params_file,
            target_pose_marker_rviz_config_file,
            start_obstacle_marker_visualizer,
            obstacle_marker_visualizer_params_file,
            start_dynamic_obstacle_publisher,
            dynamic_obstacle_publisher_params_file,
            scene_xml_path,
            OpaqueFunction(function=launch_setup),
        ]
    )
