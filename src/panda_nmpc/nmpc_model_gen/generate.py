#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线生成 Panda 任务空间位姿跟踪 NMPC 的 acados C 代码。

在你现有版本基础上，只做两类必要增强，且尽量保持接口兼容：

1) 障碍项增强为：
   - 安全软约束（保留，改成真实距离形式）
   - 净空代价（在安全边界外的激活壳层就开始起作用）
   - approach-speed 约束（限制靠近障碍的相对速度，减少贴边抖动）

2) 冗余项增强为：
   - joint-limit barrier（在硬限位之内，再额外排斥靠近限位的姿态）
   - manipulability cost（使用末端平移雅可比的 log-det surrogate，降低奇异/坏构型风险）
   - 固定中性姿态偏好（不再依赖 q_nom 做冗余分配）

运行时接口说明
--------------
- 参考量仍然走参数向量 p：
  [p_ref(3), R_ref(9), q_nom(7), v_ee_ref(3), w_ee_ref(3), cost_params..., obstacles...]
- 其中 q_nom 参数为了兼容运行时代码而保留，但在本版冗余项里不再作为主正则目标使用。
- 所有 stage/terminal 代价权重、中性姿态以及若干代价形状参数都改成了 p 的一部分，
  便于在运行时通过上层节点直接调参，而不是重新生成 solver。
- 软约束的 slack 惩罚仍使用 acados 的 zl/zu/Zl/Zu 接口在运行时写入，因为它们不属于
  external cost 表达式本身。

相比原版的结构变化
------------------
- 动力学、状态、控制、关节限位、加速度限幅全部保留；
- 原 NONLINEAR_LS 代价改为 EXTERNAL cost，原因是新增的净空代价 / barrier / manipulability
  更适合直接写成标量代价；
- 障碍约束维度略有增加：stage 里是 [安全约束, approach-speed 约束]，terminal 里只保留安全约束。
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
    # 主任务：末端位置/姿态跟踪
    pos: np.ndarray = field(default_factory=lambda: np.array([500.0, 500.0, 500.0], dtype=float))
    rot: np.ndarray = field(default_factory=lambda: np.array([100.0, 100.0, 100.0], dtype=float))

    # 末端速度误差（保持小权重，避免压过位姿）
    ee_lin_vel: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0, 5.0], dtype=float))
    ee_ang_vel: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 2.0], dtype=float))

    # 关节速度/加速度正则
    dq_reg: np.ndarray = field(default_factory=lambda: np.array([0.1] * 7, dtype=float))
    ddq_reg: np.ndarray = field(default_factory=lambda: np.array([0.05] * 7, dtype=float))

    # 终端
    pos_e: np.ndarray = field(default_factory=lambda: np.array([750.0, 750.0, 750.0], dtype=float))
    rot_e: np.ndarray = field(default_factory=lambda: np.array([200.0, 200.0, 200.0], dtype=float))
    ee_lin_vel_e: np.ndarray = field(default_factory=lambda: np.array([10.0, 10.0, 10.0], dtype=float))
    ee_ang_vel_e: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 4.0], dtype=float))
    dq_reg_e: np.ndarray = field(default_factory=lambda: np.array([0.2] * 7, dtype=float))

    # === 新增：冗余相关项 ===
    neutral_q_reg: np.ndarray = field(default_factory=lambda: np.array([15] * 7, dtype=float))
    neutral_q_reg_e: np.ndarray = field(default_factory=lambda: np.array([30] * 7, dtype=float))
    joint_limit_barrier: float = 0.05
    joint_limit_barrier_e: float = 0.08
    manipulability: float = 5
    manipulability_e: float = 8

    # === 新增：净空代价 ===
    clearance: float = 180.0
    clearance_e: float = 260.0


@dataclass
class ObstacleSoftConstraintConfig:
    num_obstacles: int = 1
    safety_margin: float = 0.03

    # 安全软约束（更强）
    slack_linear: float = 1.0e4
    slack_quadratic: float = 1.0e6

    # === 新增：净空壳层 ===
    clearance_activation_margin: float = 0.06  # 在 d_safe 外再多 6 cm 就开始起作用
    clearance_softplus_gain: float = 40.0

    # === 新增：approach-speed 约束 ===
    approach_alpha: float = 8.0
    approach_slack_linear: float = 2.0e3
    approach_slack_quadratic: float = 2.0e5

    # 距离计算里给一个很小正数，避免 sqrt(0)
    distance_eps: float = 1.0e-9


@dataclass
class RedundancyConfig:
    # 常见 Panda ready pose / neutral pose
    q_neutral: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854], dtype=float)
    )
    # manipulability surrogate 正则，避免 det 过小导致数值过激
    manipulability_eps: float = 1.0e-6
    # joint-limit barrier 里的小偏置，防止 log(0)
    barrier_eps: float = 1.0e-4


@dataclass
class RobotSphere:
    frame_name: str
    offset_xyz: np.ndarray
    radius: float


@dataclass
class OcpConfig:
    dt: float = 0.02
    horizon_steps: int = 50
    solver_name: str = "panda_task_space_nmpc"
    json_file: str = "panda_task_space_nmpc.json"
    code_export_dir: str = "c_generated_code"
    nlp_solver_type: str = "SQP_RTI"
    qp_solver: str = "FULL_CONDENSING_HPIPM"
    hessian_approx: str = "EXACT"   # EXTERNAL cost 下更合适
    integrator_type: str = "ERK"
    sim_method_num_stages: int = 4
    sim_method_num_steps: int = 1
    nlp_solver_max_iter: int = 50
    print_level: int = 0
    limits: PandaLimits = field(default_factory=PandaLimits)
    weights: WeightConfig = field(default_factory=WeightConfig)
    obstacle: ObstacleSoftConstraintConfig = field(default_factory=ObstacleSoftConstraintConfig)
    redundancy: RedundancyConfig = field(default_factory=RedundancyConfig)

    @property
    def nx(self) -> int:
        return 14

    @property
    def nu(self) -> int:
        return 7

    @property
    def base_np_stage(self) -> int:
        # 参考量基础参数：
        # p_ref(3) + R_ref(9) + q_nom(7) + ee_lin_vel_ref(3) + ee_ang_vel_ref(3)
        return 25

    @property
    def cost_np_stage(self) -> int:
        # stage/terminal 权重、中性姿态与少量代价形状参数
        return 76

    @property
    def obstacle_params_offset(self) -> int:
        return self.base_np_stage + self.cost_np_stage

    @property
    def np_stage(self) -> int:
        return self.obstacle_params_offset + 4 * self.obstacle.num_obstacles


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


def softplus(z: ca.SX) -> ca.SX:
    return ca.log(1.0 + ca.exp(z))


def quad_diag(err: ca.SX, w: np.ndarray) -> ca.SX:
    if isinstance(w, np.ndarray):
        w = ca.DM(np.asarray(w, dtype=float))
    w_vec = ca.reshape(w, err.size1(), 1)
    # CasADi 3.7 没有 ca.multiply；这里直接写成 sum_i w_i * err_i^2。
    return ca.dot(w_vec, ca.power(err, 2))


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

    # 参数布局：
    #   基础参考项:
    #     p_ref(3), R_ref(9), q_nom(7), ee_lin_vel_ref(3), ee_ang_vel_ref(3)
    #   运行时代价参数:
    #     stage/terminal 权重、中性姿态、部分代价形状参数
    #   障碍物:
    #     obs_center(3) + obs_radius(1) * num_obstacles
    p_ref = p[0:3]
    R_ref = ca.reshape(p[3:12], 3, 3)  # 列主序恢复
    _q_nom_unused = p[12:19]           # 兼容保留，不再作为冗余主目标
    ee_lin_vel_ref = p[19:22]
    ee_ang_vel_ref = p[22:25]

    cost_params_offset = cfg.base_np_stage
    w_pos = p[cost_params_offset + 0:cost_params_offset + 3]
    w_rot = p[cost_params_offset + 3:cost_params_offset + 6]
    w_ee_lin_vel = p[cost_params_offset + 6:cost_params_offset + 9]
    w_ee_ang_vel = p[cost_params_offset + 9:cost_params_offset + 12]
    w_dq_reg = p[cost_params_offset + 12:cost_params_offset + 19]
    w_ddq_reg = p[cost_params_offset + 19:cost_params_offset + 26]
    w_neutral_q_reg = p[cost_params_offset + 26:cost_params_offset + 33]

    w_pos_e = p[cost_params_offset + 33:cost_params_offset + 36]
    w_rot_e = p[cost_params_offset + 36:cost_params_offset + 39]
    w_ee_lin_vel_e = p[cost_params_offset + 39:cost_params_offset + 42]
    w_ee_ang_vel_e = p[cost_params_offset + 42:cost_params_offset + 45]
    w_dq_reg_e = p[cost_params_offset + 45:cost_params_offset + 52]
    w_neutral_q_reg_e = p[cost_params_offset + 52:cost_params_offset + 59]

    q_neutral = p[cost_params_offset + 59:cost_params_offset + 66]
    w_joint_limit_barrier = p[cost_params_offset + 66]
    w_joint_limit_barrier_e = p[cost_params_offset + 67]
    w_manipulability = p[cost_params_offset + 68]
    w_manipulability_e = p[cost_params_offset + 69]
    w_clearance = p[cost_params_offset + 70]
    w_clearance_e = p[cost_params_offset + 71]
    barrier_eps = p[cost_params_offset + 72]
    manip_eps = p[cost_params_offset + 73]
    clearance_activation_margin = p[cost_params_offset + 74]
    clearance_softplus_gain = p[cost_params_offset + 75]

    obstacle_params_offset = cfg.obstacle_params_offset
    obstacle_list: list[tuple[ca.SX, ca.SX]] = []
    for obs_idx in range(cfg.obstacle.num_obstacles):
        off = obstacle_params_offset + 4 * obs_idx
        obs_center = p[off:off + 3]
        obs_radius = p[off + 3]
        obstacle_list.append((obs_center, obs_radius))

    # 动力学保持不变
    f_expl = ca.vertcat(v, a_ref)
    f_impl = xdot - f_expl

    cpin.framesForwardKinematics(cmodel, cdata, q)
    p_ee = cdata.oMf[ee_frame_id].translation
    R_ee = cdata.oMf[ee_frame_id].rotation

    # 末端几何速度（LOCAL_WORLD_ALIGNED）
    J_ee_lwa = cpin.computeFrameJacobian(cmodel, cdata, q, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
    J_ee_lin = J_ee_lwa[0:3, :]
    J_ee_ang = J_ee_lwa[3:6, :]
    ee_lin_vel = J_ee_lin @ v
    ee_ang_vel = J_ee_ang @ v

    # 基础 tracking residual
    pos_err = p_ee - p_ref
    R_err_mat = R_ref.T @ R_ee - R_ee.T @ R_ref
    rot_err = 0.5 * vee_of_skew(R_err_mat)
    ee_lin_vel_err = ee_lin_vel - ee_lin_vel_ref
    ee_ang_vel_err = ee_ang_vel - ee_ang_vel_ref

    # === 冗余项 ===
    q_neutral = ca.reshape(q_neutral, 7, 1)
    neutral_q_err = q - q_neutral

    lim = cfg.limits
    q_min_dm = ca.DM(lim.q_min).reshape((7, 1))
    q_max_dm = ca.DM(lim.q_max).reshape((7, 1))

    # joint-limit barrier：硬限位之内的连续排斥项
    d_low = q - q_min_dm
    d_up = q_max_dm - q
    joint_limit_barrier = -ca.sum1(ca.log(d_low + barrier_eps)) - ca.sum1(ca.log(d_up + barrier_eps))

    # manipulability surrogate：只用末端平移 Jacobian，降低表达式复杂度
    # mu_pos^2 = det(Jv Jv^T + eps I3)
    JJt_pos = J_ee_lin @ J_ee_lin.T + manip_eps * ca.SX.eye(3)
    manipulability_penalty = -ca.log(ca.det(JJt_pos))

    # === 障碍项 ===
    safe_h_expr_list: list[ca.SX] = []
    approach_h_expr_list: list[ca.SX] = []
    safe_h_expr_list_e: list[ca.SX] = []

    clearance_cost = 0
    clearance_cost_e = 0

    obs_cfg = cfg.obstacle
    d_eps = float(obs_cfg.distance_eps)
    for sphere in robot_spheres:
        sphere_frame_id = resolve_frame_id(pin_model, sphere.frame_name)
        T_w_link = cdata.oMf[sphere_frame_id]
        offset_local = ca.DM(np.asarray(sphere.offset_xyz, dtype=float)).reshape((3, 1))
        sphere_center_world = T_w_link.translation + T_w_link.rotation @ offset_local

        # 点速度 Jacobian：直接对几何表达式对 q 求导，避免额外手推空间速度映射
        J_sphere = ca.jacobian(sphere_center_world, q)
        sphere_vel = J_sphere @ v

        for obs_center, obs_radius in obstacle_list:
            d_safe = sphere.radius + obs_radius + obs_cfg.safety_margin
            d_act = d_safe + clearance_activation_margin

            diff = sphere_center_world - obs_center
            dist = ca.sqrt(ca.sumsqr(diff) + d_eps)

            # 1) 安全软约束：dist - d_safe >= -s
            h_safe = dist - d_safe
            safe_h_expr_list.append(h_safe)
            safe_h_expr_list_e.append(h_safe)

            # 2) 净空代价：进入激活壳层 d < d_act 时逐渐变大
            clearance_activation = softplus(clearance_softplus_gain * (d_act - dist)) / clearance_softplus_gain
            clearance_cost += w_clearance * clearance_activation * clearance_activation
            clearance_cost_e += w_clearance_e * clearance_activation * clearance_activation

            # 3) approach-speed 约束：d_dot + alpha (d - d_safe) >= -s_app
            #    静态球障碍下，d_dot 就是机器人球心沿距离方向的速度投影
            dist_dot = ca.dot(diff, sphere_vel) / dist
            h_approach = dist_dot + obs_cfg.approach_alpha * (dist - d_safe)
            approach_h_expr_list.append(h_approach)

    # === EXTERNAL COST：基础 tracking + 冗余 + 净空 ===
    stage_cost = (
        quad_diag(pos_err, w_pos)
        + quad_diag(rot_err, w_rot)
        + quad_diag(ee_lin_vel_err, w_ee_lin_vel)
        + quad_diag(ee_ang_vel_err, w_ee_ang_vel)
        + quad_diag(v, w_dq_reg)
        + quad_diag(a_ref, w_ddq_reg)
        + quad_diag(neutral_q_err, w_neutral_q_reg)
        + w_joint_limit_barrier * joint_limit_barrier
        + w_manipulability * manipulability_penalty
        + clearance_cost
    )

    terminal_cost = (
        quad_diag(pos_err, w_pos_e)
        + quad_diag(rot_err, w_rot_e)
        + quad_diag(ee_lin_vel_err, w_ee_lin_vel_e)
        + quad_diag(ee_ang_vel_err, w_ee_ang_vel_e)
        + quad_diag(v, w_dq_reg_e)
        + quad_diag(neutral_q_err, w_neutral_q_reg_e)
        + w_joint_limit_barrier_e * joint_limit_barrier
        + w_manipulability_e * manipulability_penalty
        + clearance_cost_e
    )

    model = AcadosModel()
    model.name = cfg.solver_name
    model.x = x
    model.xdot = xdot
    model.u = a_ref
    model.p = p
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.cost_expr_ext_cost = stage_cost
    model.cost_expr_ext_cost_e = terminal_cost

    # stage: 安全约束 + approach-speed 约束
    if len(safe_h_expr_list) > 0:
        model.con_h_expr = ca.vertcat(*(safe_h_expr_list + approach_h_expr_list))
        # terminal: 只保留安全约束，避免 terminal 再额外叠 approach-speed 导致过保守
        model.con_h_expr_e = ca.vertcat(*safe_h_expr_list_e)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = cfg.horizon_steps
    ocp.solver_options.tf = cfg.horizon_steps * cfg.dt

    parameter_values = np.zeros(cfg.np_stage, dtype=float)
    parameter_values[3:12] = np.eye(3, dtype=float).reshape(-1, order="F")
    parameter_values[19:25] = 0.0
    w = cfg.weights
    parameter_values[cost_params_offset + 0:cost_params_offset + 3] = w.pos
    parameter_values[cost_params_offset + 3:cost_params_offset + 6] = w.rot
    parameter_values[cost_params_offset + 6:cost_params_offset + 9] = w.ee_lin_vel
    parameter_values[cost_params_offset + 9:cost_params_offset + 12] = w.ee_ang_vel
    parameter_values[cost_params_offset + 12:cost_params_offset + 19] = w.dq_reg
    parameter_values[cost_params_offset + 19:cost_params_offset + 26] = w.ddq_reg
    parameter_values[cost_params_offset + 26:cost_params_offset + 33] = w.neutral_q_reg
    parameter_values[cost_params_offset + 33:cost_params_offset + 36] = w.pos_e
    parameter_values[cost_params_offset + 36:cost_params_offset + 39] = w.rot_e
    parameter_values[cost_params_offset + 39:cost_params_offset + 42] = w.ee_lin_vel_e
    parameter_values[cost_params_offset + 42:cost_params_offset + 45] = w.ee_ang_vel_e
    parameter_values[cost_params_offset + 45:cost_params_offset + 52] = w.dq_reg_e
    parameter_values[cost_params_offset + 52:cost_params_offset + 59] = w.neutral_q_reg_e
    parameter_values[cost_params_offset + 59:cost_params_offset + 66] = cfg.redundancy.q_neutral
    parameter_values[cost_params_offset + 66] = w.joint_limit_barrier
    parameter_values[cost_params_offset + 67] = w.joint_limit_barrier_e
    parameter_values[cost_params_offset + 68] = w.manipulability
    parameter_values[cost_params_offset + 69] = w.manipulability_e
    parameter_values[cost_params_offset + 70] = w.clearance
    parameter_values[cost_params_offset + 71] = w.clearance_e
    parameter_values[cost_params_offset + 72] = cfg.redundancy.barrier_eps
    parameter_values[cost_params_offset + 73] = cfg.redundancy.manipulability_eps
    parameter_values[cost_params_offset + 74] = obs_cfg.clearance_activation_margin
    parameter_values[cost_params_offset + 75] = obs_cfg.clearance_softplus_gain
    for obs_idx in range(cfg.obstacle.num_obstacles):
        off = cfg.obstacle_params_offset + 4 * obs_idx
        parameter_values[off:off + 3] = np.array([1000.0, 1000.0, 1000.0], dtype=float)
        parameter_values[off + 3] = 0.0
    ocp.parameter_values = parameter_values

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

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

    if len(safe_h_expr_list) > 0:
        n_safe = len(safe_h_expr_list)
        n_app = len(approach_h_expr_list)
        nh = n_safe + n_app
        nh_e = n_safe

        ocp.constraints.lh = np.zeros(nh, dtype=float)
        ocp.constraints.uh = 1.0e15 * np.ones(nh, dtype=float)
        ocp.constraints.lh_e = np.zeros(nh_e, dtype=float)
        ocp.constraints.uh_e = 1.0e15 * np.ones(nh_e, dtype=float)

        ocp.constraints.idxsh = np.arange(nh, dtype=int)
        ocp.constraints.idxsh_e = np.arange(nh_e, dtype=int)

        zl = np.concatenate(
            [
                obs_cfg.slack_linear * np.ones(n_safe, dtype=float),
                obs_cfg.approach_slack_linear * np.ones(n_app, dtype=float),
            ]
        )
        Zu = zl.copy()
        Zl = np.concatenate(
            [
                obs_cfg.slack_quadratic * np.ones(n_safe, dtype=float),
                obs_cfg.approach_slack_quadratic * np.ones(n_app, dtype=float),
            ]
        )
        Zuq = Zl.copy()

        ocp.cost.zl = zl
        ocp.cost.zu = Zu
        ocp.cost.Zl = Zl
        ocp.cost.Zu = Zuq

        ocp.cost.zl_e = obs_cfg.slack_linear * np.ones(nh_e, dtype=float)
        ocp.cost.zu_e = obs_cfg.slack_linear * np.ones(nh_e, dtype=float)
        ocp.cost.Zl_e = obs_cfg.slack_quadratic * np.ones(nh_e, dtype=float)
        ocp.cost.Zu_e = obs_cfg.slack_quadratic * np.ones(nh_e, dtype=float)

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
    parser = argparse.ArgumentParser(description="生成 Panda 任务空间位姿跟踪 NMPC 的 acados C 代码（增强净空与冗余项）")
    parser.add_argument("--urdf", type=str, required=True, help="Panda URDF 路径")
    parser.add_argument("--ee-frame", type=str, default="ee_center_body", help="末端 frame 名")
    parser.add_argument("--dt", type=float, default=0.02, help="采样时间")
    parser.add_argument("--N", type=int, default=50, help="预测步数")
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

    print("开始生成 Panda 任务空间位姿跟踪 NMPC 的 acados C 代码（增强版）...")
    print(f"URDF: {args.urdf}")
    print(f"EE frame: {args.ee_frame}")
    print(f"solver name: {cfg.solver_name}")
    print(f"从 URDF 中读取到机械臂球包络数量: {len(robot_spheres)}")
    print(
        f"参数维度 np = {cfg.np_stage} = p_ref(3) + R_ref(9) + q_nom(7)"
        f" + v_ee_ref(3) + w_ee_ref(3) + cost_params({cfg.cost_np_stage})"
        f" + obs_center(3)/obs_radius(1) * {cfg.obstacle.num_obstacles}"
    )
    print(
        f"stage 避障约束数量 nh = {len(robot_spheres) * cfg.obstacle.num_obstacles * 2}"
        f" = safe({len(robot_spheres) * cfg.obstacle.num_obstacles})"
        f" + approach({len(robot_spheres) * cfg.obstacle.num_obstacles})"
    )
    print(
        f"terminal 避障约束数量 nh_e = {len(robot_spheres) * cfg.obstacle.num_obstacles}"
        f" = safe only"
    )
    print(f"neutral posture = {cfg.redundancy.q_neutral.tolist()}")

    AcadosOcpSolver(ocp, json_file=cfg.json_file)
    print("生成成功！")


if __name__ == "__main__":
    main()
