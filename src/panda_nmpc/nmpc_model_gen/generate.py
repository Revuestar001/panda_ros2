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


@dataclass
class OcpConfig:
    dt: float = 0.02
    horizon_steps: int = 20
    solver_name: str = "panda_task_space_nmpc"
    json_file: str = "panda_task_space_nmpc.json"
    code_export_dir: str = "c_generated_code"
    nlp_solver_type: str = "SQP"
    qp_solver: str = "FULL_CONDENSING_HPIPM"
    hessian_approx: str = "GAUSS_NEWTON"
    integrator_type: str = "ERK"
    sim_method_num_stages: int = 4
    sim_method_num_steps: int = 1
    nlp_solver_max_iter: int = 50
    print_level: int = 0
    limits: PandaLimits = field(default_factory=PandaLimits)
    weights: WeightConfig = field(default_factory=WeightConfig)

    @property
    def nx(self) -> int:
        return 14

    @property
    def nu(self) -> int:
        return 7

    @property
    def np_stage(self) -> int:
        return 19  # p_ref(3) + R_ref(9) + q_nom(7)

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


def build_acados_ocp(urdf_path: str, ee_frame_name: str, cfg: OcpConfig) -> AcadosOcp:
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"找不到 URDF 文件: {urdf_path}")

    pin_model = pin.buildModelFromUrdf(urdf_path)
    assert_fixed_base_7dof(pin_model)
    ee_frame_id = resolve_frame_id(pin_model, ee_frame_name)

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

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = cfg.horizon_steps
    ocp.solver_options.tf = cfg.horizon_steps * cfg.dt

    ocp.parameter_values = np.zeros(cfg.np_stage, dtype=float)

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
    args = parser.parse_args()

    cfg = OcpConfig(
        dt=args.dt,
        horizon_steps=args.N,
        solver_name=args.solver_name,
        json_file=args.json_file,
        code_export_dir=args.code_export_dir,
    )

    ocp = build_acados_ocp(args.urdf, args.ee_frame, cfg)
    print("开始生成 Panda 任务空间位置+姿态跟踪 NMPC 的 acados C 代码...")
    print(f"URDF: {args.urdf}")
    print(f"EE frame: {args.ee_frame}")
    print(f"solver name: {cfg.solver_name}")
    print(f"参数维度 np = {cfg.np_stage} = p_ref(3) + R_ref(9) + q_nom(7)")
    AcadosOcpSolver(ocp, json_file=cfg.json_file)
    print("生成成功！")


if __name__ == "__main__":
    main()
