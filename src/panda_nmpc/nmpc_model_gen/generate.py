#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线生成 Panda 任务空间位姿跟踪 NMPC 的 acados C 代码。

这版在你现有“位姿误差 + q_nom 正则 + 关节速度/加速度正则 + 静态球障碍软约束”的基础上，
只做两类必要增强：

1) 参考参数扩展为“可接 MoveIt2 的时变参考”
   - 仍然保持任务空间位姿为主任务；
   - q_nom 仍然只是“正则/引导”，不是硬跟踪；
   - 新增末端线速度/角速度参考，用于跟踪时变参考轨迹而不是只盯静态终点。

2) 代价中加入末端速度误差（线速度 + 角速度）
   - 对静态目标：v_ee_ref = 0, w_ee_ref = 0；
   - 对 MoveIt2 给出的全局轨迹：ROS2 节点可在每个 stage 将关节轨迹经 FK/Jacobian
     转成 p_ref_k, R_ref_k, v_ee_ref_k, w_ee_ref_k，并把 q_nom_k 一起传入。

模型定义
--------
状态:  x = [q, v]                                        in R^14
控制:  u = a_ref                                         in R^7
参数:  p = [p_ref(3), R_ref(9), q_nom(7),
            v_ee_ref(3), w_ee_ref(3),
            obs_0_center(3), obs_0_radius(1), ...]

连续时间动力学
------------
q_dot = v
v_dot = a_ref

代价函数
--------
stage:
    || p_ee(q) - p_ref ||^2_Wp
  + || e_R(q)          ||^2_WR
  + || v_ee(q,v) - v_ee_ref ||^2_Wv_ee
  + || w_ee(q,v) - w_ee_ref ||^2_Ww_ee
  + || q - q_nom       ||^2_Wq
  + || v               ||^2_Wv
  + || a_ref           ||^2_Wa

terminal:
    || p_ee(q) - p_ref ||^2_Wp_e
  + || e_R(q)          ||^2_WR_e
  + || v_ee(q,v) - v_ee_ref ||^2_Wv_ee_e
  + || w_ee(q,v) - w_ee_ref ||^2_Ww_ee_e
  + || q - q_nom       ||^2_Wq_e
  + || v               ||^2_Wv_e

姿态误差
--------
仍然采用经典几何控制里的 SO(3) 误差向量：
    e_R = 0.5 * vee(R_ref^T R - R^T R_ref)
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
    # 位置仍然是主导任务
    pos: np.ndarray = field(default_factory=lambda: np.array([2500.0, 2500.0, 2500.0], dtype=float))
    rot: np.ndarray = field(default_factory=lambda: np.array([100.0, 100.0, 100.0], dtype=float))

    # === 新增: 末端速度误差权重（默认给小权重，避免压过位姿主任务） ===
    ee_lin_vel: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0, 5.0], dtype=float))
    ee_ang_vel: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 2.0], dtype=float))

    q_reg: np.ndarray = field(default_factory=lambda: np.array([0.5] * 7, dtype=float))
    dq_reg: np.ndarray = field(default_factory=lambda: np.array([0.1] * 7, dtype=float))
    ddq_reg: np.ndarray = field(default_factory=lambda: np.array([0.05] * 7, dtype=float))

    pos_e: np.ndarray = field(default_factory=lambda: np.array([4000.0, 4000.0, 4000.0], dtype=float))
    rot_e: np.ndarray = field(default_factory=lambda: np.array([200.0, 200.0, 200.0], dtype=float))

    # === 新增: 终端速度误差权重（静态终点时通常对应 0 速度） ===
    ee_lin_vel_e: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0, 10.0], dtype=float))
    ee_ang_vel_e: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 4.0], dtype=float))

    q_reg_e: np.ndarray = field(default_factory=lambda: np.array([1.0] * 7, dtype=float))
    dq_reg_e: np.ndarray = field(default_factory=lambda: np.array([0.2] * 7, dtype=float))


@dataclass
class ObstacleSoftConstraintConfig:
    num_obstacles: int = 1
    safety_margin: float = 0.03
    slack_linear: float = 1.0e4
    slack_quadratic: float = 1.0e6


@dataclass
class RobotSphere:
    frame_name: str
    offset_xyz: np.ndarray
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
    obstacle: ObstacleSoftConstraintConfig = field(default_factory=ObstacleSoftConstraintConfig)

    @property
    def nx(self) -> int:
        return 14

    @property
    def nu(self) -> int:
        return 7

    @property
    def base_np_stage(self) -> int:
        # === 修改: 在原 p_ref(3) + R_ref(9) + q_nom(7) 基础上增加 v_ee_ref(3) + w_ee_ref(3) ===
        return 25

    @property
    def np_stage(self) -> int:
        return self.base_np_stage + 4 * self.obstacle.num_obstacles

    @property
    def ny(self) -> int:
        # pos_err(3) + rot_err(3) + ee_lin_vel_err(3) + ee_ang_vel_err(3)
        # + q_reg(7) + dq_reg(7) + ddq_reg(7)
        return 33

    @property
    def ny_e(self) -> int:
        # pos_err(3) + rot_err(3) + ee_lin_vel_err(3) + ee_ang_vel_err(3)
        # + q_reg(7) + dq_reg(7)
        return 26


def assert_fixed_base_7dof(model: pin.Model) -> None:
    if model.nq != 7 or model.nv != 7:
        raise ValueError(
            f"当前脚本假定是固定底座 7 轴 Panda 机械臂，但读取到 nq={model.nq}, nv={model.nv}。"
        )


def resolve_frame_id(model: pin.Model, frame_name: str) -> int:
    if model.existFrame(frame_name):
        return model.getFrameId(frame_name)
    frame_names = [f.name for f in model.frames]
    raise ValueError(f"URDF 中未找到 frame '{frame_name}'。可用 frame 示例: {frame_names[:20]}")


def vee_of_skew(M: ca.SX) -> ca.SX:
    return ca.vertcat(M[2, 1], M[0, 2], M[1, 0])


def _parse_xyz_attr(xyz_text: str | None) -> np.ndarray:
    if xyz_text is None:
        return np.zeros(3, dtype=float)
    vals = [float(v) for v in xyz_text.strip().split()]
    if len(vals) != 3:
        raise ValueError(f"非法 xyz 字段: {xyz_text}")
    return np.array(vals, dtype=float)


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
            offset_xyz = _parse_xyz_attr(None if origin_elem is None else origin_elem.get("xyz"))
            radius = float(radius_text)

            spheres.append(
                RobotSphere(
                    frame_name=link_name,
                    offset_xyz=offset_xyz,
                    radius=radius,
                )
            )

    return spheres


def build_acados_ocp(urdf_path: str, ee_frame_name: str, cfg: OcpConfig) -> AcadosOcp:
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"找不到 URDF 文件: {urdf_path}")

    pin_model = pin.buildModelFromUrdf(urdf_path)
    assert_fixed_base_7dof(pin_model)
    ee_frame_id = resolve_frame_id(pin_model, ee_frame_name)

    robot_spheres = load_robot_spheres_from_urdf(urdf_path)
    if len(robot_spheres) == 0:
        raise ValueError(
            "在 URDF 中没有解析到任何 <visual><geometry><sphere/></geometry></visual> 球体，"
            "无法构造机械臂避障约束。"
        )
    for sphere in robot_spheres:
        _ = resolve_frame_id(pin_model, sphere.frame_name)

    cmodel = cpin.Model(pin_model)
    cdata = cmodel.createData()

    q = ca.SX.sym("q", 7, 1)
    v = ca.SX.sym("v", 7, 1)
    x = ca.vertcat(q, v)

    xdot = ca.SX.sym("xdot", 14, 1)
    a_ref = ca.SX.sym("a_ref", 7, 1)

    p = ca.SX.sym("p", cfg.np_stage, 1)

    # === 参数布局 ===
    # p_ref(3), R_ref(9), q_nom(7), v_ee_ref(3), w_ee_ref(3), obstacles...
    p_ref = p[0:3]
    R_ref = ca.reshape(p[3:12], 3, 3)  # 列主序恢复
    q_nom = p[12:19]
    ee_lin_vel_ref = p[19:22]
    ee_ang_vel_ref = p[22:25]

    obstacle_params_offset = cfg.base_np_stage
    obstacle_list: list[tuple[ca.SX, ca.SX]] = []
    for obs_idx in range(cfg.obstacle.num_obstacles):
        off = obstacle_params_offset + 4 * obs_idx
        obs_center = p[off:off + 3]
        obs_radius = p[off + 3]
        obstacle_list.append((obs_center, obs_radius))

    f_expl = ca.vertcat(v, a_ref)
    f_impl = xdot - f_expl

    cpin.framesForwardKinematics(cmodel, cdata, q)
    p_ee = cdata.oMf[ee_frame_id].translation
    R_ee = cdata.oMf[ee_frame_id].rotation

    # === 新增: 末端几何速度（LOCAL_WORLD_ALIGNED 下，Pinocchio 空间速度顺序为 [linear; anguler]） ===
    J_ee_lwa = cpin.computeFrameJacobian(cmodel, cdata, q, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
    ee_lin_vel = J_ee_lwa[0:3, :] @ v
    ee_ang_vel = J_ee_lwa[3:6, :] @ v

    pos_err = p_ee - p_ref
    R_err_mat = R_ref.T @ R_ee - R_ee.T @ R_ref
    rot_err = 0.5 * vee_of_skew(R_err_mat)

    ee_lin_vel_err = ee_lin_vel - ee_lin_vel_ref
    ee_ang_vel_err = ee_ang_vel - ee_ang_vel_ref

    cost_y = ca.vertcat(
        pos_err,           # 3
        rot_err,           # 3
        ee_lin_vel_err,    # 3
        ee_ang_vel_err,    # 3
        q - q_nom,         # 7
        v,                 # 7
        a_ref,             # 7
    )

    cost_y_e = ca.vertcat(
        pos_err,           # 3
        rot_err,           # 3
        ee_lin_vel_err,    # 3
        ee_ang_vel_err,    # 3
        q - q_nom,         # 7
        v,                 # 7
    )

    model = AcadosModel()
    model.name = cfg.solver_name
    model.x = x
    model.xdot = xdot
    model.u = a_ref
    model.p = p
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.cost_y_expr = cost_y
    model.cost_y_expr_e = cost_y_e

    h_expr_list: list[ca.SX] = []
    for sphere in robot_spheres:
        sphere_frame_id = resolve_frame_id(pin_model, sphere.frame_name)
        T_w_link = cdata.oMf[sphere_frame_id]
        offset_local = ca.DM(np.asarray(sphere.offset_xyz, dtype=float)).reshape((3, 1))
        sphere_center_world = T_w_link.translation + T_w_link.rotation @ offset_local

        for obs_center, obs_radius in obstacle_list:
            min_allowed_dist = sphere.radius + obs_radius + cfg.obstacle.safety_margin
            h_ij = ca.sumsqr(sphere_center_world - obs_center) - min_allowed_dist**2
            h_expr_list.append(h_ij)

    if len(h_expr_list) > 0:
        con_h_expr = ca.vertcat(*h_expr_list)
        model.con_h_expr = con_h_expr
        model.con_h_expr_e = con_h_expr

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = cfg.horizon_steps
    ocp.solver_options.tf = cfg.horizon_steps * cfg.dt

    parameter_values = np.zeros(cfg.np_stage, dtype=float)
    # 默认 R_ref 为单位阵，避免未设置时出现无意义旋转参考
    parameter_values[3:12] = np.eye(3, dtype=float).reshape(-1, order="F")
    # 默认末端速度参考为 0，适用于静态目标
    parameter_values[19:25] = 0.0
    for obs_idx in range(cfg.obstacle.num_obstacles):
        off = cfg.base_np_stage + 4 * obs_idx
        parameter_values[off:off + 3] = np.array([1000.0, 1000.0, 1000.0], dtype=float)
        parameter_values[off + 3] = 0.0
    ocp.parameter_values = parameter_values

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    w = cfg.weights
    ocp.cost.W = np.diag(
        np.concatenate(
            [
                w.pos,
                w.rot,
                w.ee_lin_vel,
                w.ee_ang_vel,
                w.q_reg,
                w.dq_reg,
                w.ddq_reg,
            ]
        )
    )
    ocp.cost.W_e = np.diag(
        np.concatenate(
            [
                w.pos_e,
                w.rot_e,
                w.ee_lin_vel_e,
                w.ee_ang_vel_e,
                w.q_reg_e,
                w.dq_reg_e,
            ]
        )
    )
    ocp.cost.yref = np.zeros(cfg.ny, dtype=float)
    ocp.cost.yref_e = np.zeros(cfg.ny_e, dtype=float)

    lim = cfg.limits
    x_min = np.concatenate([lim.q_min, -lim.dq_max])
    x_max = np.concatenate([lim.q_max, lim.dq_max])

    ocp.constraints.idxbx = np.arange(cfg.nx, dtype=int)
    ocp.constraints.lbx = x_min
    ocp.constraints.ubx = x_max
    ocp.constraints.idxbx_e = np.arange(cfg.nx, dtype=int)
    ocp.constraints.lbx_e = x_min
    ocp.constraints.ubx_e = x_max
    ocp.constraints.x0 = np.zeros(cfg.nx, dtype=float)

    ocp.constraints.idxbu = np.arange(cfg.nu, dtype=int)
    ocp.constraints.lbu = -lim.ddq_max
    ocp.constraints.ubu = lim.ddq_max

    if len(h_expr_list) > 0:
        nh = len(h_expr_list)
        ocp.constraints.lh = np.zeros(nh, dtype=float)
        ocp.constraints.uh = 1.0e15 * np.ones(nh, dtype=float)
        ocp.constraints.lh_e = np.zeros(nh, dtype=float)
        ocp.constraints.uh_e = 1.0e15 * np.ones(nh, dtype=float)

        ocp.constraints.idxsh = np.arange(nh, dtype=int)
        ocp.constraints.idxsh_e = np.arange(nh, dtype=int)

        ocp.cost.Zl = cfg.obstacle.slack_quadratic * np.ones(nh, dtype=float)
        ocp.cost.Zu = cfg.obstacle.slack_quadratic * np.ones(nh, dtype=float)
        ocp.cost.zl = cfg.obstacle.slack_linear * np.ones(nh, dtype=float)
        ocp.cost.zu = cfg.obstacle.slack_linear * np.ones(nh, dtype=float)

        ocp.cost.Zl_e = cfg.obstacle.slack_quadratic * np.ones(nh, dtype=float)
        ocp.cost.Zu_e = cfg.obstacle.slack_quadratic * np.ones(nh, dtype=float)
        ocp.cost.zl_e = cfg.obstacle.slack_linear * np.ones(nh, dtype=float)
        ocp.cost.zu_e = cfg.obstacle.slack_linear * np.ones(nh, dtype=float)

    ocp.solver_options.qp_solver = cfg.qp_solver
    ocp.solver_options.hessian_approx = cfg.hessian_approx
    ocp.solver_options.integrator_type = cfg.integrator_type
    ocp.solver_options.nlp_solver_type = cfg.nlp_solver_type
    ocp.solver_options.sim_method_num_stages = cfg.sim_method_num_stages
    ocp.solver_options.sim_method_num_steps = cfg.sim_method_num_steps
    ocp.solver_options.nlp_solver_max_iter = cfg.nlp_solver_max_iter
    ocp.solver_options.print_level = cfg.print_level

    ocp.code_export_directory = cfg.code_export_dir
    return ocp


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Panda 任务空间位姿跟踪 NMPC 的 acados C 代码")
    parser.add_argument("--urdf", type=str, required=True, help="Panda URDF 路径")
    parser.add_argument("--ee-frame", type=str, default="ee_center_body", help="末端 frame 名")
    parser.add_argument("--dt", type=float, default=0.02, help="采样时间")
    parser.add_argument("--N", type=int, default=20, help="预测步数")
    parser.add_argument("--solver-name", type=str, default="panda_task_space_nmpc", help="solver 名称")
    parser.add_argument("--json-file", type=str, default="panda_task_space_nmpc.json", help="json 文件名")
    parser.add_argument("--code-export-dir", type=str, default="c_generated_code", help="C 代码导出目录")
    parser.add_argument("--num-obstacles", type=int, default=1, help="每个 stage 传入的球障碍数量")
    parser.add_argument("--safety-margin", type=float, default=0.01, help="机械臂球与障碍物球之间的额外安全裕度 [m]")
    args = parser.parse_args()

    cfg = OcpConfig(
        dt=args.dt,
        horizon_steps=args.N,
        solver_name=args.solver_name,
        json_file=args.json_file,
        code_export_dir=args.code_export_dir,
        obstacle=ObstacleSoftConstraintConfig(
            num_obstacles=args.num_obstacles,
            safety_margin=args.safety_margin,
        ),
    )

    robot_spheres = load_robot_spheres_from_urdf(args.urdf)
    ocp = build_acados_ocp(args.urdf, args.ee_frame, cfg)

    print("开始生成 Panda 任务空间位姿跟踪 NMPC 的 acados C 代码...")
    print(f"URDF: {args.urdf}")
    print(f"EE frame: {args.ee_frame}")
    print(f"solver name: {cfg.solver_name}")
    print(f"从 URDF 中读取到机械臂球包络数量: {len(robot_spheres)}")
    print(
        f"参数维度 np = {cfg.np_stage} = p_ref(3) + R_ref(9) + q_nom(7)"
        f" + v_ee_ref(3) + w_ee_ref(3)"
        f" + obs_center(3)/obs_radius(1) * {cfg.obstacle.num_obstacles}"
    )
    print(
        f"输出维度 ny = {cfg.ny} = pos(3) + rot(3) + ee_lin_vel(3) + ee_ang_vel(3)"
        f" + q_reg(7) + dq_reg(7) + ddq_reg(7)"
    )
    print(
        f"避障约束数量 nh = {len(robot_spheres) * cfg.obstacle.num_obstacles}"
        f" = robot_spheres({len(robot_spheres)}) * obstacles({cfg.obstacle.num_obstacles})"
    )

    AcadosOcpSolver(ocp, json_file=cfg.json_file)
    print("生成成功！")


if __name__ == "__main__":
    main()
