#!/usr/bin/env python3
"""
Generate acados code for Panda task-space pose NMPC.

Problem definition kept unchanged:
- state: x = [q, dq]
- control: u = ddq_ref
- parameters: target pose, nominal posture, EE twist reference, static sphere obstacles
- costs: EE pose error, EE twist error, joint-position regularization, joint-velocity regularization,
  acceleration regularization
- constraints: joint position/velocity/acceleration hard bounds, plus soft static sphere avoidance
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


@dataclass
class PandaLimits:
    q_min: np.ndarray = field(
        default_factory=lambda: np.array(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=float
        )
    )
    q_max: np.ndarray = field(
        default_factory=lambda: np.array(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=float
        )
    )
    dq_max: np.ndarray = field(
        default_factory=lambda: np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100], dtype=float)
    )
    ddq_max: np.ndarray = field(
        default_factory=lambda: np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0], dtype=float)
    )


@dataclass
class WeightConfig:
    pos: np.ndarray = field(default_factory=lambda: np.array([2500.0, 2500.0, 2500.0], dtype=float))
    rot: np.ndarray = field(default_factory=lambda: np.array([100.0, 100.0, 100.0], dtype=float))
    ee_lin_vel: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0, 5.0], dtype=float))
    ee_ang_vel: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 2.0], dtype=float))
    q_reg: np.ndarray = field(default_factory=lambda: np.array([0.5] * 7, dtype=float))
    dq_reg: np.ndarray = field(default_factory=lambda: np.array([0.1] * 7, dtype=float))
    ddq_reg: np.ndarray = field(default_factory=lambda: np.array([0.05] * 7, dtype=float))

    pos_e: np.ndarray = field(default_factory=lambda: np.array([4000.0, 4000.0, 4000.0], dtype=float))
    rot_e: np.ndarray = field(default_factory=lambda: np.array([200.0, 200.0, 200.0], dtype=float))
    ee_lin_vel_e: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0, 10.0], dtype=float))
    ee_ang_vel_e: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 4.0], dtype=float))
    q_reg_e: np.ndarray = field(default_factory=lambda: np.array([1.0] * 7, dtype=float))
    dq_reg_e: np.ndarray = field(default_factory=lambda: np.array([0.2] * 7, dtype=float))


@dataclass
class SoftConstraintPenalty:
    slack_linear: float = 1.0e4
    slack_quadratic: float = 1.0e6


@dataclass
class ObstacleConstraintConfig:
    num_obstacles: int = 0
    safety_margin: float = 0.05
    stage_penalty: SoftConstraintPenalty = field(default_factory=SoftConstraintPenalty)
    terminal_penalty: SoftConstraintPenalty = field(default_factory=SoftConstraintPenalty)


@dataclass
class RobotSphere:
    frame_name: str
    offset_xyz: np.ndarray
    radius: float


@dataclass
class StaticSphereObstacle:
    name: str
    center: np.ndarray
    radius: float


@dataclass
class OcpConfig:
    dt: float = 0.02
    horizon_steps: int = 20
    solver_name: str = "panda_task_space_nmpc"
    json_file: str = "panda_task_space_nmpc.json"
    code_export_dir: str = "c_generated_code"
    nlp_solver_type: str = "SQP_RTI"
    qp_solver: str = "FULL_CONDENSING_HPIPM"
    hessian_approx: str = "GAUSS_NEWTON"
    integrator_type: str = "ERK"
    sim_method_num_stages: int = 4
    sim_method_num_steps: int = 1
    nlp_solver_max_iter: int = 50
    print_level: int = 0
    limits: PandaLimits = field(default_factory=PandaLimits)
    weights: WeightConfig = field(default_factory=WeightConfig)
    obstacle: ObstacleConstraintConfig = field(default_factory=ObstacleConstraintConfig)

    @property
    def nx(self) -> int:
        return 14

    @property
    def nu(self) -> int:
        return 7

    @property
    def base_np_stage(self) -> int:
        return 25

    @property
    def np_stage(self) -> int:
        return self.base_np_stage + 4 * self.obstacle.num_obstacles

    @property
    def ny(self) -> int:
        return 33

    @property
    def ny_e(self) -> int:
        return 26


def assert_fixed_base_7dof(model: pin.Model) -> None:
    if model.nq != 7 or model.nv != 7:
        raise ValueError(f"Expected fixed-base 7-DoF Panda, got nq={model.nq}, nv={model.nv}.")


def resolve_frame_id(model: pin.Model, frame_name: str) -> int:
    if model.existFrame(frame_name):
        return model.getFrameId(frame_name)
    names = [frame.name for frame in model.frames]
    raise ValueError(f"Frame '{frame_name}' not found in URDF. Example frames: {names[:20]}")


def vee_of_skew(matrix: ca.SX) -> ca.SX:
    return ca.vertcat(matrix[2, 1], matrix[0, 2], matrix[1, 0])


def parse_xyz_attribute(xyz_text: str | None) -> np.ndarray:
    if xyz_text is None:
        return np.zeros(3, dtype=float)
    values = [float(value) for value in xyz_text.strip().split()]
    if len(values) != 3:
        raise ValueError(f"Invalid xyz attribute: {xyz_text}")
    return np.array(values, dtype=float)


def load_robot_spheres_from_urdf(urdf_path: str) -> list[RobotSphere]:
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    spheres: list[RobotSphere] = []
    for link_elem in root.findall("link"):
        link_name = link_elem.get("name")
        if link_name is None:
            continue

        for visual_elem in link_elem.findall("visual"):
            geometry_elem = visual_elem.find("geometry")
            if geometry_elem is None:
                continue

            sphere_elem = geometry_elem.find("sphere")
            if sphere_elem is None:
                continue

            radius_text = sphere_elem.get("radius")
            if radius_text is None:
                continue

            origin_elem = visual_elem.find("origin")
            spheres.append(
                RobotSphere(
                    frame_name=link_name,
                    offset_xyz=parse_xyz_attribute(None if origin_elem is None else origin_elem.get("xyz")),
                    radius=float(radius_text),
                )
            )
    return spheres


def load_static_sphere_obstacles(scene_xml_path: str) -> list[StaticSphereObstacle]:
    tree = ET.parse(scene_xml_path)
    root = tree.getroot()

    obstacles: list[StaticSphereObstacle] = []
    if root.tag == "static_sphere_obstacles":
        for obstacle_elem in root.findall("obstacle"):
            name = obstacle_elem.get("name")
            pos = obstacle_elem.get("pos")
            radius = obstacle_elem.get("radius")
            if name is None or pos is None or radius is None:
                raise ValueError(f"Obstacle config entries must contain name/pos/radius: {scene_xml_path}")
            obstacles.append(
                StaticSphereObstacle(
                    name=name,
                    center=parse_xyz_attribute(pos),
                    radius=float(radius),
                )
            )
        return obstacles

    worldbody_elem = root.find("worldbody")
    if worldbody_elem is None:
        raise ValueError(f"Scene XML has no <worldbody>: {scene_xml_path}")

    for body_elem in worldbody_elem.findall("body"):
        geom_elem = body_elem.find("geom")
        if geom_elem is None or geom_elem.get("type") != "sphere":
            continue

        name = body_elem.get("name")
        pos = body_elem.get("pos")
        size = geom_elem.get("size")
        if name is None or pos is None or size is None:
            raise ValueError(f"Sphere obstacle entries must contain name/pos/size: {scene_xml_path}")

        obstacles.append(
            StaticSphereObstacle(
                name=name,
                center=parse_xyz_attribute(pos),
                radius=float(size.strip().split()[0]),
            )
        )
    return obstacles


def build_symbolic_model(
    pin_model: pin.Model,
    ee_frame_id: int,
    robot_spheres: list[RobotSphere],
    scene_obstacles: list[StaticSphereObstacle],
    cfg: OcpConfig,
) -> tuple[AcadosModel, int]:
    cmodel = cpin.Model(pin_model)
    cdata = cmodel.createData()

    q = ca.SX.sym("q", 7, 1)
    dq = ca.SX.sym("dq", 7, 1)
    x = ca.vertcat(q, dq)

    xdot = ca.SX.sym("xdot", cfg.nx, 1)
    ddq_ref = ca.SX.sym("ddq_ref", cfg.nu, 1)

    p = ca.SX.sym("p", cfg.np_stage, 1)
    target_pos = p[0:3]
    target_rot = ca.reshape(p[3:12], 3, 3)
    q_nom = p[12:19]
    ee_lin_vel_ref = p[19:22]
    ee_ang_vel_ref = p[22:25]

    dynamics_expl = ca.vertcat(dq, ddq_ref)
    dynamics_impl = xdot - dynamics_expl

    cpin.framesForwardKinematics(cmodel, cdata, q)
    ee_pos = cdata.oMf[ee_frame_id].translation
    ee_rot = cdata.oMf[ee_frame_id].rotation

    ee_jacobian = cpin.computeFrameJacobian(cmodel, cdata, q, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
    ee_lin_vel = ee_jacobian[0:3, :] @ dq
    ee_ang_vel = ee_jacobian[3:6, :] @ dq

    pos_err = ee_pos - target_pos
    rot_err = 0.5 * vee_of_skew(target_rot.T @ ee_rot - ee_rot.T @ target_rot)
    ee_lin_vel_err = ee_lin_vel - ee_lin_vel_ref
    ee_ang_vel_err = ee_ang_vel - ee_ang_vel_ref

    cost_y = ca.vertcat(
        pos_err,
        rot_err,
        ee_lin_vel_err,
        ee_ang_vel_err,
        q - q_nom,
        dq,
        ddq_ref,
    )
    cost_y_e = ca.vertcat(
        pos_err,
        rot_err,
        ee_lin_vel_err,
        ee_ang_vel_err,
        q - q_nom,
        dq,
    )

    obstacle_constraints: list[ca.SX] = []
    obstacle_offset = cfg.base_np_stage
    for sphere in robot_spheres:
        sphere_frame_id = resolve_frame_id(pin_model, sphere.frame_name)
        sphere_transform = cdata.oMf[sphere_frame_id]
        sphere_offset = ca.DM(np.asarray(sphere.offset_xyz, dtype=float)).reshape((3, 1))
        sphere_center = sphere_transform.translation + sphere_transform.rotation @ sphere_offset

        for obstacle_idx in range(cfg.obstacle.num_obstacles):
            param_offset = obstacle_offset + 4 * obstacle_idx
            obstacle_center = p[param_offset:param_offset + 3]
            obstacle_radius = p[param_offset + 3]
            min_distance = sphere.radius + obstacle_radius + cfg.obstacle.safety_margin
            obstacle_constraints.append(ca.sumsqr(sphere_center - obstacle_center) - min_distance**2)

    model = AcadosModel()
    model.name = cfg.solver_name
    model.x = x
    model.xdot = xdot
    model.u = ddq_ref
    model.p = p
    model.f_expl_expr = dynamics_expl
    model.f_impl_expr = dynamics_impl
    model.cost_y_expr = cost_y
    model.cost_y_expr_e = cost_y_e

    if obstacle_constraints:
        con_h = ca.vertcat(*obstacle_constraints)
        model.con_h_expr = con_h
        model.con_h_expr_e = con_h

    return model, len(obstacle_constraints)


def apply_cost_configuration(ocp: AcadosOcp, cfg: OcpConfig) -> None:
    weights = cfg.weights
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag(
        np.concatenate(
            [
                weights.pos,
                weights.rot,
                weights.ee_lin_vel,
                weights.ee_ang_vel,
                weights.q_reg,
                weights.dq_reg,
                weights.ddq_reg,
            ]
        )
    )
    ocp.cost.W_e = np.diag(
        np.concatenate(
            [
                weights.pos_e,
                weights.rot_e,
                weights.ee_lin_vel_e,
                weights.ee_ang_vel_e,
                weights.q_reg_e,
                weights.dq_reg_e,
            ]
        )
    )
    ocp.cost.yref = np.zeros(cfg.ny, dtype=float)
    ocp.cost.yref_e = np.zeros(cfg.ny_e, dtype=float)


def apply_box_constraints(ocp: AcadosOcp, cfg: OcpConfig) -> None:
    limits = cfg.limits
    state_min = np.concatenate([limits.q_min, -limits.dq_max])
    state_max = np.concatenate([limits.q_max, limits.dq_max])

    ocp.constraints.idxbx = np.arange(cfg.nx, dtype=int)
    ocp.constraints.lbx = state_min
    ocp.constraints.ubx = state_max
    ocp.constraints.idxbx_e = np.arange(cfg.nx, dtype=int)
    ocp.constraints.lbx_e = state_min
    ocp.constraints.ubx_e = state_max
    ocp.constraints.x0 = np.zeros(cfg.nx, dtype=float)

    ocp.constraints.idxbu = np.arange(cfg.nu, dtype=int)
    ocp.constraints.lbu = -limits.ddq_max
    ocp.constraints.ubu = limits.ddq_max


def apply_obstacle_constraints(ocp: AcadosOcp, nh: int, cfg: OcpConfig) -> None:
    if nh <= 0:
        return

    ocp.constraints.lh = np.zeros(nh, dtype=float)
    ocp.constraints.uh = 1.0e15 * np.ones(nh, dtype=float)
    ocp.constraints.lh_e = np.zeros(nh, dtype=float)
    ocp.constraints.uh_e = 1.0e15 * np.ones(nh, dtype=float)
    ocp.constraints.idxsh = np.arange(nh, dtype=int)
    ocp.constraints.idxsh_e = np.arange(nh, dtype=int)

    stage_penalty = cfg.obstacle.stage_penalty
    terminal_penalty = cfg.obstacle.terminal_penalty
    ocp.cost.Zl = stage_penalty.slack_quadratic * np.ones(nh, dtype=float)
    ocp.cost.Zu = stage_penalty.slack_quadratic * np.ones(nh, dtype=float)
    ocp.cost.zl = stage_penalty.slack_linear * np.ones(nh, dtype=float)
    ocp.cost.zu = stage_penalty.slack_linear * np.ones(nh, dtype=float)

    ocp.cost.Zl_e = terminal_penalty.slack_quadratic * np.ones(nh, dtype=float)
    ocp.cost.Zu_e = terminal_penalty.slack_quadratic * np.ones(nh, dtype=float)
    ocp.cost.zl_e = terminal_penalty.slack_linear * np.ones(nh, dtype=float)
    ocp.cost.zu_e = terminal_penalty.slack_linear * np.ones(nh, dtype=float)


def apply_solver_options(ocp: AcadosOcp, cfg: OcpConfig) -> None:
    ocp.solver_options.tf = cfg.horizon_steps * cfg.dt
    ocp.solver_options.qp_solver = cfg.qp_solver
    ocp.solver_options.hessian_approx = cfg.hessian_approx
    ocp.solver_options.integrator_type = cfg.integrator_type
    ocp.solver_options.nlp_solver_type = cfg.nlp_solver_type
    ocp.solver_options.sim_method_num_stages = cfg.sim_method_num_stages
    ocp.solver_options.sim_method_num_steps = cfg.sim_method_num_steps
    ocp.solver_options.nlp_solver_max_iter = cfg.nlp_solver_max_iter
    ocp.solver_options.print_level = cfg.print_level


def build_default_runtime_parameters(scene_obstacles: list[StaticSphereObstacle], cfg: OcpConfig) -> np.ndarray:
    parameter_values = np.zeros(cfg.np_stage, dtype=float)
    parameter_values[3:12] = np.eye(3, dtype=float).reshape(-1, order="F")
    parameter_values[19:25] = 0.0

    for obstacle_idx, obstacle in enumerate(scene_obstacles):
        offset = cfg.base_np_stage + 4 * obstacle_idx
        parameter_values[offset:offset + 3] = obstacle.center
        parameter_values[offset + 3] = obstacle.radius
    return parameter_values


def build_acados_ocp(
    urdf_path: str,
    obstacle_source_path: str,
    ee_frame_name: str,
    cfg: OcpConfig,
) -> tuple[AcadosOcp, list[RobotSphere], list[StaticSphereObstacle], int]:
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    if not os.path.exists(obstacle_source_path):
        raise FileNotFoundError(f"Obstacle config not found: {obstacle_source_path}")

    pin_model = pin.buildModelFromUrdf(urdf_path)
    assert_fixed_base_7dof(pin_model)
    ee_frame_id = resolve_frame_id(pin_model, ee_frame_name)

    robot_spheres = load_robot_spheres_from_urdf(urdf_path)
    scene_obstacles = load_static_sphere_obstacles(obstacle_source_path)
    if not robot_spheres:
        raise ValueError("No robot spheres found in URDF visual geometry.")
    if not scene_obstacles:
        raise ValueError("No static sphere obstacles found in obstacle config.")
    if cfg.obstacle.num_obstacles != len(scene_obstacles):
        raise ValueError(
            "cfg.obstacle.num_obstacles does not match parsed obstacle count: "
            f"{cfg.obstacle.num_obstacles} != {len(scene_obstacles)}"
        )

    model, nh = build_symbolic_model(pin_model, ee_frame_id, robot_spheres, scene_obstacles, cfg)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = cfg.horizon_steps
    ocp.parameter_values = build_default_runtime_parameters(scene_obstacles, cfg)

    apply_cost_configuration(ocp, cfg)
    apply_box_constraints(ocp, cfg)
    apply_obstacle_constraints(ocp, nh, cfg)
    apply_solver_options(ocp, cfg)

    ocp.code_export_directory = cfg.code_export_dir
    return ocp, robot_spheres, scene_obstacles, nh


def resolve_obstacle_source_path(scene_xml_path: str, obstacle_config_path: str) -> str:
    if obstacle_config_path and os.path.exists(obstacle_config_path):
        return obstacle_config_path
    return scene_xml_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Panda second-order NMPC acados C code")
    parser.add_argument("--urdf", type=str, required=True, help="Panda URDF path")
    parser.add_argument(
        "--scene-xml",
        type=str,
        default="/home/cyh/panda_ros2/model/franka_emika_panda/scene_tau_ros.xml",
        help="Scene XML path",
    )
    parser.add_argument(
        "--obstacle-config",
        type=str,
        default="/home/cyh/panda_ros2/model/franka_emika_panda/static_sphere_obstacles.xml",
        help="Static sphere obstacle config path",
    )
    parser.add_argument("--ee-frame", type=str, default="ee_center_body", help="End-effector frame name")
    parser.add_argument("--dt", type=float, default=0.02, help="Sampling time")
    parser.add_argument("--N", type=int, default=20, help="Prediction horizon steps")
    parser.add_argument("--solver-name", type=str, default="panda_task_space_nmpc", help="Solver name")
    parser.add_argument("--json-file", type=str, default="panda_task_space_nmpc.json", help="JSON file name")
    parser.add_argument("--code-export-dir", type=str, default="c_generated_code", help="Export directory")
    parser.add_argument("--safety-margin", type=float, default=0.05, help="Obstacle safety margin")
    args = parser.parse_args()

    obstacle_source_path = resolve_obstacle_source_path(args.scene_xml, args.obstacle_config)
    scene_obstacles = load_static_sphere_obstacles(obstacle_source_path)

    cfg = OcpConfig(
        dt=args.dt,
        horizon_steps=args.N,
        solver_name=args.solver_name,
        json_file=args.json_file,
        code_export_dir=args.code_export_dir,
        obstacle=ObstacleConstraintConfig(
            num_obstacles=len(scene_obstacles),
            safety_margin=args.safety_margin,
        ),
    )

    ocp, robot_spheres, loaded_obstacles, nh = build_acados_ocp(
        urdf_path=args.urdf,
        obstacle_source_path=obstacle_source_path,
        ee_frame_name=args.ee_frame,
        cfg=cfg,
    )

    print("Generating Panda second-order task-space NMPC acados code...")
    print(f"URDF: {args.urdf}")
    print(f"Obstacle source: {obstacle_source_path}")
    print(f"EE frame: {args.ee_frame}")
    print(f"dt: {cfg.dt}")
    print(f"N: {cfg.horizon_steps}")
    print(f"solver: {cfg.solver_name}")
    print(
        f"nx={cfg.nx}, nu={cfg.nu}, np={cfg.np_stage}, ny={cfg.ny}, ny_e={cfg.ny_e}, "
        f"nh={nh}, integrator={cfg.integrator_type}"
    )
    print(
        "Parameter layout: p_ref(3) + R_ref(9) + q_nom(7) + ee_lin_vel_ref(3) + ee_ang_vel_ref(3) "
        f"+ obstacle_xyzr(4) * {cfg.obstacle.num_obstacles}"
    )
    print(f"Robot spheres: {len(robot_spheres)}, static obstacles: {len(loaded_obstacles)}")

    AcadosOcpSolver(ocp, json_file=cfg.json_file)
    print("Generation complete.")


if __name__ == "__main__":
    main()
