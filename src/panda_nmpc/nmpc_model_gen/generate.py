#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线生成 Panda 任务空间位置+姿态跟踪 NMPC 的 acados C 代码。

基于当前已经跑通的“位置跟踪 + a_ref 输出 + 下游关节空间阻抗控制器”版本，
这一步只增加姿态误差代价，不改变执行层接口。

模型定义
--------
状态:  x = [q, v]              in R^14
控制:  u = a_ref               in R^7
参数:  p = [p_ref(3), R_ref(9), q_nom(7)] in R^19

连续时间动力学
------------
q_dot = v
v_dot = a_ref

代价函数
--------
stage:
    || p_ee(q) - p_ref ||^2_Wp
  + || e_R(q)          ||^2_WR
  + || q - q_nom       ||^2_Wq
  + || v               ||^2_Wv
  + || a_ref           ||^2_Wa

terminal:
    || p_ee(q) - p_ref ||^2_Wp_e
  + || e_R(q)          ||^2_WR_e
  + || q - q_nom       ||^2_Wq_e
  + || v               ||^2_Wv_e

姿态误差写法
----------
这里不用 log3，而用经典几何控制里的 SO(3) 误差向量：
    e_R = 0.5 * vee(R_ref^T R - R^T R_ref)
它在小角度附近很平滑，作为“先小步加姿态代价”的版本更稳一些。
"""

from __future__ import annotations

import argparse
import os
import xml.etree.ElementTree as ET  # === 修改: 用于从 URDF 自动解析球包络 ===
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
    # 位置继续主导
    pos: np.ndarray = field(default_factory=lambda: np.array([2500.0, 2500.0, 2500.0], dtype=float))
    # 姿态先给小权重，确认不发散后再逐步加大
    rot: np.ndarray = field(default_factory=lambda: np.array([100.0, 100.0, 100.0], dtype=float))
    q_reg: np.ndarray = field(default_factory=lambda: np.array([0.5] * 7, dtype=float))
    dq_reg: np.ndarray = field(default_factory=lambda: np.array([0.1] * 7, dtype=float))
    ddq_reg: np.ndarray = field(default_factory=lambda: np.array([0.05] * 7, dtype=float))

    pos_e: np.ndarray = field(default_factory=lambda: np.array([4000.0, 4000.0, 4000.0], dtype=float))
    rot_e: np.ndarray = field(default_factory=lambda: np.array([200.0, 200.0, 200.0], dtype=float))
    q_reg_e: np.ndarray = field(default_factory=lambda: np.array([1.0] * 7, dtype=float))
    dq_reg_e: np.ndarray = field(default_factory=lambda: np.array([0.2] * 7, dtype=float))


# === 修改开始: 障碍物软约束配置 ===
@dataclass
class ObstacleSoftConstraintConfig:
    num_obstacles: int = 1          # 每个 stage 通过参数传入的球障碍数量
    safety_margin: float = 0.03     # 额外安全裕度 [m]
    slack_linear: float = 1.0e4     # soft constraint 线性罚
    slack_quadratic: float = 1.0e6  # soft constraint 二次罚
# === 修改结束: 障碍物软约束配置 ===


# === 修改开始: 机械臂球包络数据结构 ===
@dataclass
class RobotSphere:
    frame_name: str
    offset_xyz: np.ndarray
    radius: float
# === 修改结束: 机械臂球包络数据结构 ===


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
    obstacle: ObstacleSoftConstraintConfig = field(default_factory=ObstacleSoftConstraintConfig)  # === 修改 ===

    @property
    def nx(self) -> int:
        return 14

    @property
    def nu(self) -> int:
        return 7

    @property
    def base_np_stage(self) -> int:
        return 19  # p_ref(3) + R_ref(9) + q_nom(7)

    @property
    def np_stage(self) -> int:
        # === 修改: 增加球障碍参数 [obs_center(3), obs_radius(1)] * num_obstacles ===
        return self.base_np_stage + 4 * self.obstacle.num_obstacles

    @property
    def ny(self) -> int:
        return 27  # pos_err(3) + rot_err(3) + q_reg(7) + dq_reg(7) + ddq_reg(7)

    @property
    def ny_e(self) -> int:
        return 20  # pos_err(3) + rot_err(3) + q_reg(7) + dq_reg(7)


def assert_fixed_base_7dof(model: pin.Model) -> None:
    if model.nq != 7 or model.nv != 7:
        raise ValueError(
            f"当前脚本假定是固定底座 7 轴 Panda 机械臂，但读取到 nq={model.nq}, nv={model.nv}。"
        )


def resolve_frame_id(model: pin.Model, ee_frame_name: str) -> int:
    if model.existFrame(ee_frame_name):
        return model.getFrameId(ee_frame_name)
    frame_names = [f.name for f in model.frames]
    raise ValueError(f"URDF 中未找到 frame '{ee_frame_name}'。可用 frame 示例: {frame_names[:20]}")


def vee_of_skew(M: ca.SX) -> ca.SX:
    """将 3x3 反对称矩阵映射到 R^3。"""
    return ca.vertcat(M[2, 1], M[0, 2], M[1, 0])


# === 修改开始: 从 URDF 读取所有 visual sphere，作为机械臂球包络 ===
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
# === 修改结束: 从 URDF 读取所有 visual sphere，作为机械臂球包络 ===


def build_acados_ocp(urdf_path: str, ee_frame_name: str, cfg: OcpConfig) -> AcadosOcp:
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"找不到 URDF 文件: {urdf_path}")

    pin_model = pin.buildModelFromUrdf(urdf_path)
    assert_fixed_base_7dof(pin_model)
    ee_frame_id = resolve_frame_id(pin_model, ee_frame_name)

    # === 修改开始: 读取 URDF 中定义的机械臂球包络 ===
    robot_spheres = load_robot_spheres_from_urdf(urdf_path)
    if len(robot_spheres) == 0:
        raise ValueError(
            "在 URDF 中没有解析到任何 <visual><geometry><sphere/></geometry></visual> 球体，"
            "无法构造机械臂避障约束。"
        )
    # 确保每个球挂载的 frame 在 Pinocchio 模型中都存在
    for sphere in robot_spheres:
        _ = resolve_frame_id(pin_model, sphere.frame_name)
    # === 修改结束: 读取 URDF 中定义的机械臂球包络 ===

    cmodel = cpin.Model(pin_model)
    cdata = cmodel.createData()

    q = ca.SX.sym("q", 7, 1)
    v = ca.SX.sym("v", 7, 1)
    x = ca.vertcat(q, v)

    xdot = ca.SX.sym("xdot", 14, 1)
    a_ref = ca.SX.sym("a_ref", 7, 1)

    p = ca.SX.sym("p", cfg.np_stage, 1)
    p_ref = p[0:3]
    R_ref = ca.reshape(p[3:12], 3, 3)  # 列主序恢复
    q_nom = p[12:19]

    # === 修改开始: 从参数中解析球障碍 ===
    obstacle_params_offset = cfg.base_np_stage
    obstacle_list: list[tuple[ca.SX, ca.SX]] = []
    for obs_idx in range(cfg.obstacle.num_obstacles):
        off = obstacle_params_offset + 4 * obs_idx
        obs_center = p[off:off + 3]
        obs_radius = p[off + 3]
        obstacle_list.append((obs_center, obs_radius))
    # === 修改结束: 从参数中解析球障碍 ===

    f_expl = ca.vertcat(v, a_ref)
    f_impl = xdot - f_expl

    cpin.framesForwardKinematics(cmodel, cdata, q)
    p_ee = cdata.oMf[ee_frame_id].translation
    R_ee = cdata.oMf[ee_frame_id].rotation

    pos_err = p_ee - p_ref

    # e_R = 0.5 * vee(R_ref^T R - R^T R_ref)
    R_err_mat = R_ref.T @ R_ee - R_ee.T @ R_ref
    rot_err = 0.5 * vee_of_skew(R_err_mat)

    cost_y = ca.vertcat(
        pos_err,    # 3
        rot_err,    # 3
        q - q_nom,  # 7
        v,          # 7
        a_ref,      # 7
    )

    cost_y_e = ca.vertcat(
        pos_err,    # 3
        rot_err,    # 3
        q - q_nom,  # 7
        v,          # 7
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

    # === 修改开始: 构造“机械臂球包络 vs 外部球障碍”的 soft 非线性约束 ===
    # 约束形式:
    #   h(q, p) = ||c_i(q) - o_j||^2 - (r_i + r_j + margin)^2 >= 0
    #
    # 其中:
    #   c_i(q): 机械臂第 i 个球心（由 URDF 里的球挂在 link 本体系上，随 FK 运动）
    #   o_j:    第 j 个障碍物球心（通过 stage 参数 p 在线传入）
    #   r_i:    机械臂球半径（来自 URDF）
    #   r_j:    障碍物球半径（通过 stage 参数 p 在线传入）
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
    # === 修改结束: 构造“机械臂球包络 vs 外部球障碍”的 soft 非线性约束 ===

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = cfg.horizon_steps
    ocp.solver_options.tf = cfg.horizon_steps * cfg.dt

    # === 修改开始: 给新增障碍物参数一个“默认远离机器人”的初值，避免未设置参数时等价于障碍物贴在原点 ===
    parameter_values = np.zeros(cfg.np_stage, dtype=float)
    for obs_idx in range(cfg.obstacle.num_obstacles):
        off = cfg.base_np_stage + 4 * obs_idx
        parameter_values[off:off + 3] = np.array([1000.0, 1000.0, 1000.0], dtype=float)
        parameter_values[off + 3] = 0.0
    ocp.parameter_values = parameter_values
    # === 修改结束 ===

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    w = cfg.weights
    ocp.cost.W = np.diag(np.concatenate([w.pos, w.rot, w.q_reg, w.dq_reg, w.ddq_reg]))
    ocp.cost.W_e = np.diag(np.concatenate([w.pos_e, w.rot_e, w.q_reg_e, w.dq_reg_e]))
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

    # === 修改开始: 设置 soft 非线性避障约束 ===
    if len(h_expr_list) > 0:
        nh = len(h_expr_list)
        ocp.constraints.lh = np.zeros(nh, dtype=float)
        ocp.constraints.uh = 1.0e15 * np.ones(nh, dtype=float)
        ocp.constraints.lh_e = np.zeros(nh, dtype=float)
        ocp.constraints.uh_e = 1.0e15 * np.ones(nh, dtype=float)

        # 将全部 h 约束设置为 soft constraints
        ocp.constraints.idxsh = np.arange(nh, dtype=int)
        ocp.constraints.idxsh_e = np.arange(nh, dtype=int)

        # slack 代价：0.5 * Z * s^2 + z * s
        ocp.cost.Zl = cfg.obstacle.slack_quadratic * np.ones(nh, dtype=float)
        ocp.cost.Zu = cfg.obstacle.slack_quadratic * np.ones(nh, dtype=float)
        ocp.cost.zl = cfg.obstacle.slack_linear * np.ones(nh, dtype=float)
        ocp.cost.zu = cfg.obstacle.slack_linear * np.ones(nh, dtype=float)

        ocp.cost.Zl_e = cfg.obstacle.slack_quadratic * np.ones(nh, dtype=float)
        ocp.cost.Zu_e = cfg.obstacle.slack_quadratic * np.ones(nh, dtype=float)
        ocp.cost.zl_e = cfg.obstacle.slack_linear * np.ones(nh, dtype=float)
        ocp.cost.zu_e = cfg.obstacle.slack_linear * np.ones(nh, dtype=float)
    # === 修改结束: 设置 soft 非线性避障约束 ===

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
    parser = argparse.ArgumentParser(description="生成 Panda 任务空间位置+姿态跟踪 NMPC 的 acados C 代码")
    parser.add_argument("--urdf", type=str, required=True, help="Panda URDF 路径")
    parser.add_argument("--ee-frame", type=str, default="ee_center_body", help="末端 frame 名")
    parser.add_argument("--dt", type=float, default=0.02, help="采样时间")
    parser.add_argument("--N", type=int, default=20, help="预测步数")
    parser.add_argument("--solver-name", type=str, default="panda_task_space_nmpc", help="solver 名称")
    parser.add_argument("--json-file", type=str, default="panda_task_space_nmpc.json", help="json 文件名")
    parser.add_argument("--code-export-dir", type=str, default="c_generated_code", help="C 代码导出目录")
    # === 修改开始: 新增障碍物相关命令行参数 ===
    parser.add_argument("--num-obstacles", type=int, default=1, help="每个 stage 传入的球障碍数量")
    parser.add_argument("--safety-margin", type=float, default=0.03, help="机械臂球与障碍物球之间的额外安全裕度 [m]")
    # === 修改结束: 新增障碍物相关命令行参数 ===
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

    # === 修改开始: 生成前读取并打印 URDF 球包络信息，便于确认约束规模 ===
    robot_spheres = load_robot_spheres_from_urdf(args.urdf)
    # === 修改结束 ===

    ocp = build_acados_ocp(args.urdf, args.ee_frame, cfg)
    print("开始生成 Panda 任务空间位置+姿态跟踪 NMPC 的 acados C 代码...")
    print(f"URDF: {args.urdf}")
    print(f"EE frame: {args.ee_frame}")
    print(f"solver name: {cfg.solver_name}")
    print(f"从 URDF 中读取到机械臂球包络数量: {len(robot_spheres)}")
    print(
        f"参数维度 np = {cfg.np_stage} = p_ref(3) + R_ref(9) + q_nom(7)"
        f" + obs_center(3)/obs_radius(1) * {cfg.obstacle.num_obstacles}"
    )
    print(
        f"避障约束数量 nh = {len(robot_spheres) * cfg.obstacle.num_obstacles}"
        f" = robot_spheres({len(robot_spheres)}) * obstacles({cfg.obstacle.num_obstacles})"
    )
    AcadosOcpSolver(ocp, json_file=cfg.json_file)
    print("生成成功！")


if __name__ == "__main__":
    main()