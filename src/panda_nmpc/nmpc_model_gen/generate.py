#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 acados + Pinocchio CasADi 为 Panda 机械臂生成 NMPC C 代码的单文件脚本。

这个版本只做两件事：
1) 定义系统模型与最优控制问题（OCP）
2) 生成 acados 对应的 C 代码（可选再编译）

它**不包含**下面这些运行时逻辑：
- 不创建 Python 闭环 NMPC 控制节点
- 不执行在线求解
- 不做 ROS2 节点封装
- 不做参考轨迹下发与状态回读

因此，这个脚本适合作为“离线代码生成器”：
你在 Python 里描述模型和优化问题，然后导出 acados 的 C 工程；
后续你再在 ROS2 / C++ 节点里链接并调用生成出来的 solver。

控制目标
--------
- 任务: 笛卡尔空间末端位姿跟踪（位置 + 姿态）
- 机器人: Franka Panda 7 自由度机械臂（不考虑夹爪）
- 末端 frame: 默认使用 "ee_center_body"，可通过命令行修改

优化变量与模型
--------------
- 状态 x = [q, dq, ddq]   ∈ R^21
- 控制 u = dddq (joint jerk) ∈ R^7
- 连续时间模型:
      q_dot   = dq
      dq_dot  = ddq
      ddq_dot = jerk

重要说明
--------
1) 本 OCP 不直接把“力矩”作为控制输入，因此适合外部再接高频内环
   （joint position / velocity / impedance / inverse dynamics inner loop）。
2) 力矩 tau 通过 Pinocchio 的 RNEA(q, dq, ddq) 在线计算，并作为非线性路径约束加入 OCP。
3) 因此可以同时约束：
   - 关节角
   - 关节速度
   - 关节加速度
   - 力矩
   - jerk
4) 运行时的参考位姿和舒适姿态通过参数 p 传入，但本脚本只负责“定义参数结构”，
   不负责在线设置 p。

运行时参数定义
--------------
每个 shooting node 的参数为：
    p = [p_des(3), R_des_col_major(9), q_nom(7)] ∈ R^19
其中：
- p_des           : 末端期望位置
- R_des_col_major : 末端期望旋转矩阵（按列优先展开）
- q_nom           : 关节正则化参考（舒适姿态）

代价函数
--------
stage cost:
    || p_ee(q) - p_des ||^2_{W_pos}
  + || log3(R_ee(q) * R_des^T) ||^2_{W_rot}
  + || q   - q_nom ||^2_{W_q}
  + || dq          ||^2_{W_dq}
  + || ddq         ||^2_{W_ddq}
  + || jerk        ||^2_{W_jerk}

terminal cost:
    与 stage cost 类似，但没有 jerk 项。

注：这里采用 acados 的 NONLINEAR_LS 形式，把误差表达式直接写进 cost_y_expr，
运行时 yref / yref_e 通常设为 0；真实参考通过参数 p 传入。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Tuple

import casadi as ca
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


# ==============================
# 1. Panda 默认限制与 OCP 配置
# ==============================


@dataclass
class PandaLimits:
    """Panda 关节空间限制。

    默认值按 Franka Panda 官方限制填写，可按你的系统安全需求自行收紧。

    单位：
    - q    : rad
    - dq   : rad/s
    - ddq  : rad/s^2
    - dddq : rad/s^3
    - tau  : Nm
    """

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
    dddq_max: np.ndarray = field(
        default_factory=lambda: np.array([7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0], dtype=float)
    )
    tau_max: np.ndarray = field(
        default_factory=lambda: np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=float)
    )


@dataclass
class WeightConfig:
    """代价函数权重。"""

    pos: np.ndarray = field(default_factory=lambda: np.array([500.0, 500.0, 500.0], dtype=float))
    rot: np.ndarray = field(default_factory=lambda: np.array([200.0, 200.0, 200.0], dtype=float))
    q_reg: np.ndarray = field(default_factory=lambda: np.array([2.0] * 7, dtype=float))
    dq_reg: np.ndarray = field(default_factory=lambda: np.array([0.3] * 7, dtype=float))
    ddq_reg: np.ndarray = field(default_factory=lambda: np.array([0.03] * 7, dtype=float))
    jerk_reg: np.ndarray = field(default_factory=lambda: np.array([1.0e-4] * 7, dtype=float))

    pos_e: np.ndarray = field(default_factory=lambda: np.array([1200.0, 1200.0, 1200.0], dtype=float))
    rot_e: np.ndarray = field(default_factory=lambda: np.array([500.0, 500.0, 500.0], dtype=float))
    q_reg_e: np.ndarray = field(default_factory=lambda: np.array([3.0] * 7, dtype=float))
    dq_reg_e: np.ndarray = field(default_factory=lambda: np.array([0.4] * 7, dtype=float))
    ddq_reg_e: np.ndarray = field(default_factory=lambda: np.array([0.05] * 7, dtype=float))


@dataclass
class OcpConfig:
    """acados OCP 配置。"""

    dt: float = 0.02
    horizon_steps: int = 20

    # 代码导出配置
    solver_name: str = "panda_pose_jerk_nmpc"
    json_file: str = "acados_panda_pose_jerk_nmpc.json"
    code_export_dir: str = "c_generated_code_panda_pose_jerk_nmpc"

    # acados 求解器配置（用于生成 C solver）
    nlp_solver_type: str = "SQP"
    qp_solver: str = "PARTIAL_CONDENSING_HPIPM"
    hessian_approx: str = "GAUSS_NEWTON"
    integrator_type: str = "ERK"
    sim_method_num_stages: int = 4
    sim_method_num_steps: int = 1
    nlp_solver_max_iter: int = 50
    print_level: int = 0

    # 关节角安全裕度
    q_margin: float = 0.0

    weights: WeightConfig = field(default_factory=WeightConfig)
    limits: PandaLimits = field(default_factory=PandaLimits)

    @property
    def nx(self) -> int:
        return 21

    @property
    def nu(self) -> int:
        return 7

    @property
    def np_stage(self) -> int:
        return 19

    @property
    def ny(self) -> int:
        return 34

    @property
    def ny_e(self) -> int:
        return 27

    def stage_weight_matrix(self) -> np.ndarray:
        w = self.weights
        diag = np.concatenate([w.pos, w.rot, w.q_reg, w.dq_reg, w.ddq_reg, w.jerk_reg])
        return np.diag(diag)

    def terminal_weight_matrix(self) -> np.ndarray:
        w = self.weights
        diag = np.concatenate([w.pos_e, w.rot_e, w.q_reg_e, w.dq_reg_e, w.ddq_reg_e])
        return np.diag(diag)

    def state_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 x = [q, dq, ddq] 的上下界。"""
        lim = self.limits
        q_min = lim.q_min + self.q_margin
        q_max = lim.q_max - self.q_margin
        x_min = np.concatenate([q_min, -lim.dq_max, -lim.ddq_max])
        x_max = np.concatenate([q_max, lim.dq_max, lim.ddq_max])
        return x_min, x_max


# ==============================
# 2. 基础检查函数
# ==============================


def assert_fixed_base_7dof(model: pin.Model) -> None:
    """检查是否为固定底座 7 轴机械臂模型。"""
    if model.nq != 7 or model.nv != 7:
        raise ValueError(
            f"当前脚本假定为固定底座 7 轴 Panda，但读取到 nq={model.nq}, nv={model.nv}。\n"
            f"请确认 URDF 不包含夹爪自由度、浮动基座，或自行修改脚本维度。"
        )


def resolve_frame_id(model: pin.Model, ee_frame_name: str) -> int:
    """根据 frame 名称查找末端执行器 frame id。"""
    if model.existFrame(ee_frame_name):
        return model.getFrameId(ee_frame_name)

    frame_names = [f.name for f in model.frames]
    hint = [name for name in frame_names if any(k in name.lower() for k in ("ee", "tool", "hand"))][:20]
    raise ValueError(
        f"URDF 中未找到 frame '{ee_frame_name}'。\n"
        f"请通过 --ee-frame 指定正确的末端 frame 名称。\n"
        f"候选 frame: {hint}"
    )


# ==============================
# 3. 构建符号模型
# ==============================


def build_symbolic_problem(urdf_path: str, ee_frame_name: str, cfg: OcpConfig) -> dict:
    """构建 CasADi + Pinocchio 符号模型与 OCP 所需表达式。"""
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF 文件不存在: {urdf_path}")

    # 数值模型仅用于读取维度、frame 等元数据
    pin_model = pin.buildModelFromUrdf(urdf_path)
    assert_fixed_base_7dof(pin_model)
    ee_frame_id = resolve_frame_id(pin_model, ee_frame_name)

    # CasADi 版 Pinocchio 模型用于构建符号图
    cmodel = cpin.Model(pin_model)
    cdata = cmodel.createData()

    # 状态 / 控制 / 参数
    q = ca.SX.sym("q", 7, 1)
    dq = ca.SX.sym("dq", 7, 1)
    ddq = ca.SX.sym("ddq", 7, 1)
    x = ca.vertcat(q, dq, ddq)
    xdot = ca.SX.sym("xdot", 21, 1)
    jerk = ca.SX.sym("jerk", 7, 1)

    # p = [p_des(3), R_des(9, col-major), q_nom(7)]
    p = ca.SX.sym("p", cfg.np_stage, 1)
    p_des = p[0:3]
    R_des = ca.reshape(p[3:12], 3, 3)
    q_nom = p[12:19]

    # 连续时间模型
    f_expl = ca.vertcat(dq, ddq, jerk)
    f_impl = xdot - f_expl

    # 末端位姿
    cpin.framesForwardKinematics(cmodel, cdata, q)
    ee_pos = cdata.oMf[ee_frame_id].translation
    ee_rot = cdata.oMf[ee_frame_id].rotation

    # 位姿误差
    pos_err = ee_pos - p_des
    rot_err = cpin.log3(ee_rot @ R_des.T)

    # 力矩路径约束: tau(q, dq, ddq)
    tau = cpin.rnea(cmodel, cdata, q, dq, ddq)

    # NONLINEAR_LS 输出向量
    cost_y = ca.vertcat(
        pos_err,
        rot_err,
        q - q_nom,
        dq,
        ddq,
        jerk,
    )

    cost_y_e = ca.vertcat(
        pos_err,
        rot_err,
        q - q_nom,
        dq,
        ddq,
    )

    return {
        "pin_model": pin_model,
        "ee_frame_id": ee_frame_id,
        "x": x,
        "xdot": xdot,
        "u": jerk,
        "p": p,
        "f_expl": f_expl,
        "f_impl": f_impl,
        "cost_y": cost_y,
        "cost_y_e": cost_y_e,
        "h": tau,
        "h_e": tau,
    }


# ==============================
# 4. 构建 acados OCP
# ==============================


def build_acados_model(urdf_path: str, ee_frame_name: str, cfg: OcpConfig) -> AcadosModel:
    """根据 Panda 模型与任务定义生成 AcadosModel。"""
    sym = build_symbolic_problem(urdf_path, ee_frame_name, cfg)

    model = AcadosModel()
    model.name = cfg.solver_name
    model.x = sym["x"]
    model.xdot = sym["xdot"]
    model.u = sym["u"]
    model.p = sym["p"]
    model.f_expl_expr = sym["f_expl"]
    model.f_impl_expr = sym["f_impl"]
    model.cost_y_expr = sym["cost_y"]
    model.cost_y_expr_e = sym["cost_y_e"]
    # model.con_h_expr = sym["h"]
    # model.con_h_expr_e = sym["h_e"]
    return model



def build_acados_ocp(urdf_path: str, ee_frame_name: str, cfg: OcpConfig) -> AcadosOcp:
    """定义 acados OCP。"""
    model = build_acados_model(urdf_path, ee_frame_name, cfg)
    lim = cfg.limits

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = cfg.horizon_steps
    ocp.solver_options.tf = cfg.horizon_steps * cfg.dt

    # 参数占位：这里只为代码生成声明参数维度
    ocp.parameter_values = np.zeros(cfg.np_stage, dtype=float)

    # 代价函数
    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = cfg.stage_weight_matrix()
    ocp.cost.W_e = cfg.terminal_weight_matrix()
    ocp.cost.yref = np.zeros(cfg.ny, dtype=float)
    ocp.cost.yref_e = np.zeros(cfg.ny_e, dtype=float)

    # 状态 box 约束：q, dq, ddq
    x_min, x_max = cfg.state_bounds()
    ocp.constraints.idxbx = np.arange(cfg.nx, dtype=int)
    ocp.constraints.lbx = x_min
    ocp.constraints.ubx = x_max
    ocp.constraints.idxbx_e = np.arange(cfg.nx, dtype=int)
    ocp.constraints.lbx_e = x_min
    ocp.constraints.ubx_e = x_max

    # 初值只用于定义维度，占位即可
    ocp.constraints.x0 = np.zeros(cfg.nx, dtype=float)

    # 控制 box 约束：jerk
    ocp.constraints.idxbu = np.arange(cfg.nu, dtype=int)
    ocp.constraints.lbu = -lim.dddq_max
    ocp.constraints.ubu = lim.dddq_max

    # 非线性约束：tau = RNEA(q, dq, ddq)
    # ocp.constraints.lh = -lim.tau_max
    # ocp.constraints.uh = lim.tau_max
    # ocp.constraints.lh_e = -lim.tau_max
    # ocp.constraints.uh_e = lim.tau_max

    # 求解器选项
    ocp.solver_options.qp_solver = cfg.qp_solver
    ocp.solver_options.hessian_approx = cfg.hessian_approx
    ocp.solver_options.integrator_type = cfg.integrator_type
    ocp.solver_options.nlp_solver_type = cfg.nlp_solver_type
    ocp.solver_options.sim_method_num_stages = cfg.sim_method_num_stages
    ocp.solver_options.sim_method_num_steps = cfg.sim_method_num_steps
    ocp.solver_options.nlp_solver_max_iter = cfg.nlp_solver_max_iter
    ocp.solver_options.print_level = cfg.print_level

    # C 代码导出目录
    ocp.code_export_directory = cfg.code_export_dir

    return ocp


# ==============================
# 5. 只生成 C 代码（可选再编译）
# ==============================


def generate_c_code(urdf_path: str, ee_frame_name: str, cfg: OcpConfig, build: bool, verbose: bool) -> None:
    """只做 acados C 代码生成。

    注意：
    - 这里不返回 Python solver wrapper
    - 这里只是离线生成 C 工程，便于后续在 ROS2 / C++ 里集成
    """
    ocp = build_acados_ocp(urdf_path, ee_frame_name, cfg)

    os.makedirs(cfg.code_export_dir, exist_ok=True)

    # 仅生成代码，不创建 Python 运行时闭环逻辑
    AcadosOcpSolver.generate(
        ocp,
        json_file=cfg.json_file,
        simulink_opts=None,
        cmake_builder=None,
        verbose=verbose,
    )

    if build:
        AcadosOcpSolver.build(
            cfg.code_export_dir,
            with_cython=False,
            cmake_builder=None,
            verbose=verbose,
        )


# ==============================
# 6. 命令行接口
# ==============================


def make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="定义 Panda 的 NMPC OCP 并导出 acados C 代码（不包含在线 ROS2 节点）"
    )

    parser.add_argument(
        "--urdf",
        type=str,
        required=True,
        help="Panda 7 轴 URDF 路径（不含夹爪自由度）",
    )
    parser.add_argument(
        "--ee-frame",
        type=str,
        default="ee_center_body",
        help="末端执行器 frame 名称，默认 ee_center_body",
    )

    parser.add_argument("--dt", type=float, default=0.02, help="采样时间，默认 0.02 s")
    parser.add_argument("--N", type=int, default=20, help="预测步数，默认 20")

    parser.add_argument(
        "--solver-name",
        type=str,
        default="panda_task_space_nmpc",
        help="acados 模型 / solver 名称",
    )
    parser.add_argument(
        "--json-file",
        type=str,
        default="acados_panda_pose_jerk_nmpc.json",
        help="渲染 acados 模板时使用的 json 文件名",
    )
    parser.add_argument(
        "--code-export-dir",
        type=str,
        default="c_generated_code_panda_pose_jerk_nmpc",
        help="生成的 acados C 工程导出目录",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="若设置，则在代码生成后进一步编译 C solver",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="降低生成阶段输出信息",
    )

    return parser


# ==============================
# 7. main
# ==============================


def main() -> None:
    args = make_argparser().parse_args()

    cfg = OcpConfig(
        dt=args.dt,
        horizon_steps=args.N,
        solver_name=args.solver_name,
        json_file=args.json_file,
        code_export_dir=args.code_export_dir,
    )

    generate_c_code(
        urdf_path=args.urdf,
        ee_frame_name=args.ee_frame,
        cfg=cfg,
        build=args.build,
        verbose=not args.quiet,
    )

    print("=" * 80)
    print("acados C 代码导出完成")
    print(f"URDF            : {args.urdf}")
    print(f"EE frame        : {args.ee_frame}")
    print(f"dt              : {cfg.dt}")
    print(f"N               : {cfg.horizon_steps}")
    print(f"solver name     : {cfg.solver_name}")
    print(f"json file       : {cfg.json_file}")
    print(f"code export dir : {cfg.code_export_dir}")
    print(f"build solver    : {args.build}")
    print("=" * 80)
    print(
        "\n后续建议：\n"
        "- 在 ROS2 / C++ 节点中包含并链接导出的 acados C solver\n"
        "- 在线设置 x0、stage 参数 p、参考轨迹与 warm start\n"
        "- 读取第一步 jerk 或预测状态，再交给你的高频内环执行\n"
    )


if __name__ == "__main__":
    main()
