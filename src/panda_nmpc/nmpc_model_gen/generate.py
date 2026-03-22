#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为 Panda 7 自由度机械臂生成“单点末端位姿 NMPC”的 acados C 代码。

本版严格匹配以下工程决策：
1. 三阶积分器模型: x = [q, dq, ddq], u = jerk。
2. 先只实现单点位姿跟踪；接口与数据结构保留到轨迹/路径扩展位。
3. 不加入外部碰撞约束，但保留约束结构，后续可继续在 h / h_e 中扩展。
4. 当前导出的 hard-only 版本只保留真实物理边界；soft/path constraints 预留为后续扩展。
5. 使用连续时间动力学 + acados 标准显式积分器接口，避免 disc_dyn_expr 与现有 API 不匹配。
6. 主代价保持 NONLINEAR_LS + GAUSS_NEWTON，便于 SQP_RTI 实时求解。
"""

from __future__ import annotations

import argparse
import os
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
    jerk_max: np.ndarray = field(
        default_factory=lambda: np.array([7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0], dtype=float)
    )


@dataclass
class WeightConfig:
    pos: np.ndarray = field(default_factory=lambda: np.array([2500.0, 2500.0, 2500.0], dtype=float))
    rot: np.ndarray = field(default_factory=lambda: np.array([120.0, 120.0, 120.0], dtype=float))
    ee_lin_vel: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0, 5.0], dtype=float))
    ee_ang_vel: np.ndarray = field(default_factory=lambda: np.array([3.0, 3.0, 3.0], dtype=float))
    q_reg: np.ndarray = field(default_factory=lambda: np.array([0.2] * 7, dtype=float))
    dq_reg: np.ndarray = field(default_factory=lambda: np.array([0.05] * 7, dtype=float))
    ddq_reg: np.ndarray = field(default_factory=lambda: np.array([0.02] * 7, dtype=float))
    jerk_reg: np.ndarray = field(default_factory=lambda: np.array([1.0e-4] * 7, dtype=float))

    pos_e: np.ndarray = field(default_factory=lambda: np.array([5000.0, 5000.0, 5000.0], dtype=float))
    rot_e: np.ndarray = field(default_factory=lambda: np.array([240.0, 240.0, 240.0], dtype=float))
    ee_lin_vel_e: np.ndarray = field(default_factory=lambda: np.array([8.0, 8.0, 8.0], dtype=float))
    ee_ang_vel_e: np.ndarray = field(default_factory=lambda: np.array([5.0, 5.0, 5.0], dtype=float))
    q_reg_e: np.ndarray = field(default_factory=lambda: np.array([0.4] * 7, dtype=float))
    dq_reg_e: np.ndarray = field(default_factory=lambda: np.array([0.1] * 7, dtype=float))
    ddq_reg_e: np.ndarray = field(default_factory=lambda: np.array([0.05] * 7, dtype=float))


@dataclass
class SoftConstraintPenalty:
    slack_linear: float = 1.0e3
    slack_quadratic: float = 1.0e6


@dataclass
class OcpConfig:
    dt: float = 0.02
    horizon_steps: int = 20
    solver_name: str = "panda_task_space_nmpc"
    json_file: str = "panda_task_space_nmpc.json"
    code_export_dir: str = "c_generated_code"
    nlp_solver_type: str = "SQP_RTI"
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    hessian_approx: str = "GAUSS_NEWTON"
    integrator_type: str = "ERK"
    sim_method_num_stages: int = 4
    sim_method_num_steps: int = 1
    nlp_solver_max_iter: int = 50
    qp_solver_cond_N: int = 5
    print_level: int = 0
    levenberg_marquardt: float = 1.0e-6
    limits: PandaLimits = field(default_factory=PandaLimits)
    weights: WeightConfig = field(default_factory=WeightConfig)
    # 预留给未来 soft/path constraints；当前 hard-only 版本不使用。
    soft_penalty_stage: SoftConstraintPenalty = field(default_factory=SoftConstraintPenalty)
    soft_penalty_terminal: SoftConstraintPenalty = field(
        default_factory=lambda: SoftConstraintPenalty(slack_linear=2.0e3, slack_quadratic=2.0e6)
    )

    @property
    def nx(self) -> int:
        return 21

    @property
    def nu(self) -> int:
        return 7

    @property
    def np_stage(self) -> int:
        # p_ref(3) + R_ref(9) + q_nom(7) + dq_nom(7) + ddq_nom(7) + ee_lin_vel_ref(3) + ee_ang_vel_ref(3)
        return 39

    @property
    def ny(self) -> int:
        # pos + rot + ee_twist + q + dq + ddq + jerk
        return 40

    @property
    def ny_e(self) -> int:
        # pos + rot + ee_twist + q + dq + ddq
        return 33

    @property
    def nh(self) -> int:
        # 当前 hard-only 版本不生成一般性 h 约束；后续若要恢复 soft/path constraints，再显式扩展。
        return 0

    @property
    def nh_e(self) -> int:
        return 0


def assert_fixed_base_7dof(model: pin.Model) -> None:
    if model.nq != 7 or model.nv != 7:
        raise ValueError(f"当前脚本假定固定底座 7 轴 Panda，但读取到 nq={model.nq}, nv={model.nv}。")


def resolve_frame_id(model: pin.Model, frame_name: str) -> int:
    if model.existFrame(frame_name):
        return model.getFrameId(frame_name)
    names = [frame.name for frame in model.frames]
    raise ValueError(f"URDF 中未找到 frame '{frame_name}'，前几个 frame: {names[:20]}")


def vee_of_skew(M: ca.SX) -> ca.SX:
    return ca.vertcat(M[2, 1], M[0, 2], M[1, 0])


def build_acados_ocp(urdf_path: str, ee_frame_name: str, cfg: OcpConfig) -> AcadosOcp:
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"找不到 URDF 文件: {urdf_path}")

    pin_model = pin.buildModelFromUrdf(urdf_path)
    assert_fixed_base_7dof(pin_model)
    ee_frame_id = resolve_frame_id(pin_model, ee_frame_name)

    cmodel = cpin.Model(pin_model)
    cdata = cmodel.createData()

    q = ca.SX.sym("q", 7, 1)
    dq = ca.SX.sym("dq", 7, 1)
    ddq = ca.SX.sym("ddq", 7, 1)
    x = ca.vertcat(q, dq, ddq)

    xdot = ca.SX.sym("xdot", cfg.nx, 1)
    jerk = ca.SX.sym("jerk", 7, 1)

    p = ca.SX.sym("p", cfg.np_stage, 1)
    p_ref = p[0:3]
    R_ref = ca.reshape(p[3:12], 3, 3)
    q_nom = p[12:19]
    dq_nom = p[19:26]
    ddq_nom = p[26:33]
    ee_lin_vel_ref = p[33:36]
    ee_ang_vel_ref = p[36:39]

    f_expl = ca.vertcat(dq, ddq, jerk)
    f_impl = xdot - f_expl

    cpin.forwardKinematics(cmodel, cdata, q)
    cpin.updateFramePlacements(cmodel, cdata)
    p_ee = cdata.oMf[ee_frame_id].translation
    R_ee = cdata.oMf[ee_frame_id].rotation

    J_ee_lwa = cpin.computeFrameJacobian(cmodel, cdata, q, ee_frame_id, pin.LOCAL_WORLD_ALIGNED)
    ee_lin_vel = J_ee_lwa[0:3, :] @ dq
    ee_ang_vel = J_ee_lwa[3:6, :] @ dq

    pos_err = p_ee - p_ref
    R_err_mat = R_ref.T @ R_ee - R_ee.T @ R_ref
    rot_err = 0.5 * vee_of_skew(R_err_mat)
    ee_lin_vel_err = ee_lin_vel - ee_lin_vel_ref
    ee_ang_vel_err = ee_ang_vel - ee_ang_vel_ref

    cost_y = ca.vertcat(
        pos_err,
        rot_err,
        ee_lin_vel_err,
        ee_ang_vel_err,
        q - q_nom,
        dq - dq_nom,
        ddq - ddq_nom,
        jerk,
    )
    cost_y_e = ca.vertcat(
        pos_err,
        rot_err,
        ee_lin_vel_err,
        ee_ang_vel_err,
        q - q_nom,
        dq - dq_nom,
        ddq - ddq_nom,
    )

    model = AcadosModel()
    model.name = cfg.solver_name
    model.x = x
    model.xdot = xdot
    model.u = jerk
    model.p = p
    model.f_expl_expr = f_expl
    model.f_impl_expr = f_impl
    model.cost_y_expr = cost_y
    model.cost_y_expr_e = cost_y_e

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = cfg.horizon_steps
    ocp.solver_options.tf = cfg.horizon_steps * cfg.dt

    parameter_values = np.zeros(cfg.np_stage, dtype=float)
    parameter_values[3:12] = np.eye(3, dtype=float).reshape(-1, order="F")
    ocp.parameter_values = parameter_values

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"

    w = cfg.weights
    ocp.cost.W = np.diag(
        np.concatenate([
            w.pos,
            w.rot,
            w.ee_lin_vel,
            w.ee_ang_vel,
            w.q_reg,
            w.dq_reg,
            w.ddq_reg,
            w.jerk_reg,
        ])
    )
    ocp.cost.W_e = np.diag(
        np.concatenate([
            w.pos_e,
            w.rot_e,
            w.ee_lin_vel_e,
            w.ee_ang_vel_e,
            w.q_reg_e,
            w.dq_reg_e,
            w.ddq_reg_e,
        ])
    )
    ocp.cost.yref = np.zeros(cfg.ny, dtype=float)
    ocp.cost.yref_e = np.zeros(cfg.ny_e, dtype=float)

    lim = cfg.limits
    x_min = np.concatenate([lim.q_min, -lim.dq_max, -lim.ddq_max])
    x_max = np.concatenate([lim.q_max, lim.dq_max, lim.ddq_max])

    ocp.constraints.idxbx = np.arange(cfg.nx, dtype=int)
    ocp.constraints.lbx = x_min
    ocp.constraints.ubx = x_max
    ocp.constraints.idxbx_e = np.arange(cfg.nx, dtype=int)
    ocp.constraints.lbx_e = x_min
    ocp.constraints.ubx_e = x_max
    ocp.constraints.x0 = np.zeros(cfg.nx, dtype=float)

    ocp.constraints.idxbu = np.arange(cfg.nu, dtype=int)
    ocp.constraints.lbu = -lim.jerk_max
    ocp.constraints.ubu = lim.jerk_max

    # hard-only 版本中，q/dq/ddq 仅通过 lbx/ubx 生效，jerk 仅通过 lbu/ubu 生效。
    # 不重复生成 h/h_e，也不生成任何 slack 变量。

    ocp.solver_options.qp_solver = cfg.qp_solver
    ocp.solver_options.qp_solver_cond_N = min(cfg.qp_solver_cond_N, cfg.horizon_steps)
    ocp.solver_options.hessian_approx = cfg.hessian_approx
    ocp.solver_options.integrator_type = cfg.integrator_type
    ocp.solver_options.sim_method_num_stages = cfg.sim_method_num_stages
    ocp.solver_options.sim_method_num_steps = cfg.sim_method_num_steps
    ocp.solver_options.nlp_solver_type = cfg.nlp_solver_type
    ocp.solver_options.nlp_solver_max_iter = cfg.nlp_solver_max_iter
    ocp.solver_options.print_level = cfg.print_level
    ocp.solver_options.levenberg_marquardt = cfg.levenberg_marquardt

    ocp.code_export_directory = cfg.code_export_dir
    return ocp


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Panda 三阶积分器单点位姿 NMPC 的 acados C 代码")
    parser.add_argument("--urdf", type=str, required=True, help="Panda URDF 路径")
    parser.add_argument("--ee-frame", type=str, default="ee_center_body", help="末端 frame 名")
    parser.add_argument("--dt", type=float, default=0.02, help="采样时间")
    parser.add_argument("--N", type=int, default=20, help="预测步数")
    parser.add_argument("--solver-name", type=str, default="panda_task_space_nmpc", help="solver 名称")
    parser.add_argument("--json-file", type=str, default="panda_task_space_nmpc.json", help="json 文件名")
    parser.add_argument("--code-export-dir", type=str, default="c_generated_code", help="代码导出目录")
    args = parser.parse_args()

    cfg = OcpConfig(
        dt=args.dt,
        horizon_steps=args.N,
        solver_name=args.solver_name,
        json_file=args.json_file,
        code_export_dir=args.code_export_dir,
    )
    ocp = build_acados_ocp(args.urdf, args.ee_frame, cfg)

    print("开始生成 Panda 三阶积分器单点位姿 NMPC acados C 代码...")
    print(f"URDF: {args.urdf}")
    print(f"EE frame: {args.ee_frame}")
    print(f"dt: {cfg.dt}")
    print(f"N: {cfg.horizon_steps}")
    print(f"solver: {cfg.solver_name}")
    print(
        f"nx={cfg.nx}, nu={cfg.nu}, np={cfg.np_stage}, ny={cfg.ny}, ny_e={cfg.ny_e}, "
        f"nh={cfg.nh}, nh_e={cfg.nh_e}, integrator={cfg.integrator_type}"
    )

    AcadosOcpSolver(ocp, json_file=cfg.json_file)
    print("生成成功。")


if __name__ == "__main__":
    main()
