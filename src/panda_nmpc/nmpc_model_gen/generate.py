import os
import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

def create_panda_dynamics_model() -> AcadosModel:
    """使用 Pinocchio + CasADi 构建 Panda 机械臂任务空间动力学模型"""
    
    # 替换为你实际的 URDF 路径
    urdf_path = "/home/cyh/panda_ros2/model/franka_panda_urdf/robots/panda_arm_tau.urdf" 
    
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"找不到 URDF 文件: {urdf_path}")

    model = pin.buildModelFromUrdf(urdf_path)
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()

    # --- 1. 定义 CasADi 符号变量 ---
    nq = model.nq 
    nv = model.nv 
    nx = nq + nv  
    nu = nv       

    q = ca.SX.sym('q', nq)
    v = ca.SX.sym('v', nv)
    x = ca.vertcat(q, v)
    u = ca.SX.sym('u', nu)

    # --- 2. 动力学方程 ---
    v_dot = cpin.aba(cmodel, cdata, q, v, u)
    f_expl = ca.vertcat(v, v_dot)
    x_dot_sym = ca.SX.sym('x_dot_sym', nx)
    f_impl = x_dot_sym - f_expl

    # --- 3. 任务空间运动学 ---
    cpin.framesForwardKinematics(cmodel, cdata, q)
    ee_frame_name = "ee_center_body" 
    if cmodel.existFrame(ee_frame_name):
        ee_frame_id = cmodel.getFrameId(ee_frame_name)
    else:
        print(f"警告: URDF 中找不到 '{ee_frame_name}'，默认使用最后一个 Frame 作为末端")
        ee_frame_id = cmodel.nframes - 1

    p_ee = cdata.oMf[ee_frame_id].translation
    
    # --- 4. 定义非线性代价函数的残差项 y ---
    # 【修改点】：为了控制零空间，我们将关节角度 q 也加入了代价函数中
    # 优化目标：[末端位置(3), 关节角度(7), 关节速度(7), 关节力矩(7)]
    cost_y_expr = ca.vertcat(p_ee, q, v, u)
    # 终端代价残差：不需要力矩
    cost_y_expr_e = ca.vertcat(p_ee, q, v)

    # --- 5. 构建 AcadosModel ---
    acados_model = AcadosModel()
    acados_model.f_impl_expr = f_impl
    acados_model.f_expl_expr = f_expl
    acados_model.x = x
    acados_model.xdot = x_dot_sym
    acados_model.u = u
    acados_model.cost_y_expr = cost_y_expr
    acados_model.cost_y_expr_e = cost_y_expr_e
    acados_model.name = "panda_task_space_nmpc"

    return acados_model


def generate_c_code():
    model = create_panda_dynamics_model()
    ocp = AcadosOcp()
    ocp.model = model
    
    nq = model.u.size()[0]  # 7
    nv = model.u.size()[0]  # 7
    nx = model.x.size()[0]  # 14
    nu = model.u.size()[0]  # 7
    N_horizon = 20
    T_f = 1.0
    
    ocp.dims.N = N_horizon

    # --- 1. 定义代价函数 (NONLINEAR_LS) ---
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    # y 的维度：末端(3) + 角度(7) + 速度(7) + 力矩(7) = 24
    ny = 3 + nq + nv + nu 
    # y_e 的维度：末端(3) + 角度(7) + 速度(7) = 17
    ny_e = 3 + nq + nv

    # 【修改点】：设置权重矩阵，体现主任务和零空间任务的优先级
    Q_p = 2500.0 * np.ones(3)     # 1. 任务空间追踪权重 (绝对主导，非常大)
    Q_q = 0.5 * np.ones(nq)       # 2. 零空间姿态权重 (很小，仅用于拉扯冗余自由度，避免奇异)
    Q_v = 0.1 * np.ones(nv)       # 3. 速度阻尼权重 (让动作平滑)
    R_u = 0.01 * np.ones(nu)      # 4. 力矩惩罚权重
    
    ocp.cost.W = np.diag(np.concatenate([Q_p, Q_q, Q_v, R_u]))
    ocp.cost.W_e = np.diag(np.concatenate([Q_p, Q_q, Q_v]))

    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # --- 2. 定义约束 (Constraints) ---
    
    # 2.1 状态量约束 (State Constraints: x = [q, v])
    # Franka Panda 官方设定的关节位置极限 (弧度)
    q_min = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    # Franka Panda 官方设定的关节速度极限 (rad/s)
    v_max = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])
    v_min = -v_max

    # 拼接 q 和 v 的极限作为状态 x 的极限
    x_min = np.concatenate([q_min, v_min])
    x_max = np.concatenate([q_max, v_max])

    ocp.constraints.lbx = x_min
    ocp.constraints.ubx = x_max
    ocp.constraints.idxbx = np.arange(nx) # 对所有 14 个状态施加约束

    # 2.2 控制量约束 (Control Constraints: u = tau)
    tau_limit = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])
    ocp.constraints.lbu = -tau_limit
    ocp.constraints.ubu =  tau_limit
    ocp.constraints.idxbu = np.arange(nu)

    # 初始化起始状态 x0
    ocp.constraints.x0 = np.zeros(nx)

    # --- 3. 求解器选项 ---
    ocp.solver_options.tf = T_f
    ocp.solver_options.integrator_type = 'IRK'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.nlp_solver_max_iter = 1

    print("开始生成带有关节限位和零空间控制的 acados C++ 代码...")
    solver = AcadosOcpSolver(ocp, json_file='panda_acados_ocp.json')
    print("生成成功！C 代码保存在 'c_generated_code' 文件夹中。")

if __name__ == "__main__":
    generate_c_code()