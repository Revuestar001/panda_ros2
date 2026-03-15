#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstring> // 用于 memset

// 引入 acados 核心底层接口
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// 引入自动生成的 Panda 任务空间 NMPC 专属头文件
// (请确保头文件名与你生成的 c_generated_code 文件夹中的一致)
#include "c_generated_code/acados_solver_panda_task_space_nmpc.h"

int main() {
    std::cout << "========== 初始化 Panda 任务空间 NMPC 求解器 ==========" << std::endl;

    // 1. 创建求解器胶囊
    panda_task_space_nmpc_solver_capsule* capsule = panda_task_space_nmpc_acados_create_capsule();
    int status = panda_task_space_nmpc_acados_create(capsule);
    if (status) {
        std::cerr << "[错误] 创建求解器失败, 错误码: " << status << std::endl;
        return 1;
    }

    // 2. 获取核心指针 
    ocp_nlp_config* nlp_config = panda_task_space_nmpc_acados_get_nlp_config(capsule);
    ocp_nlp_dims* nlp_dims     = panda_task_space_nmpc_acados_get_nlp_dims(capsule);
    ocp_nlp_in* nlp_in         = panda_task_space_nmpc_acados_get_nlp_in(capsule);
    ocp_nlp_out* nlp_out       = panda_task_space_nmpc_acados_get_nlp_out(capsule);

    int N  = nlp_dims->N; // 预测步数 (Horizon = 20)
    
    // ================= 定义维度 (需与 Python 脚本完全一致) =================
    int dim_p = 3;  // 末端 XYZ
    int dim_q = 7;  // 关节角度
    int dim_v = 7;  // 关节速度
    int dim_u = 7;  // 关节力矩
    
    int nx = dim_q + dim_v;                        // 14
    int ny = dim_p + dim_q + dim_v + dim_u;        // 24 (预测步)
    int ny_e = dim_p + dim_q + dim_v;              // 17 (终端步)

    // ================= 初始化目标与状态 =================

    // 模拟当前机械臂的真实状态 (比如全部在 Home 位置，速度为 0)
    double current_x[14] = {0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785,  // q
                            0.0,  0.0,   0.0,  0.0,   0.0, 0.0,   0.0};   // v

    // --- 构建参考轨迹 y_ref ---
    double y_ref[24] = {0.0}; 
    double y_ref_e[17] = {0.0};

    // 1. 主任务：设定目标末端位置 (Target EE Position)
    // 注意：这里的参考点应该在机械臂可达空间内，这里随便写了一个点用于演示
    double target_p_ee[3] = {0.4, 0.2, 0.5}; 
    
    // 2. 零空间/次级任务：设定期望的舒适姿态 (Posture Regularization)
    // 通常设为机械臂的 Home 位，防止动作怪异
    double target_q_rest[7] = {0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785};

    // 组装 y_ref (对于每一步 i = 0 ... N-1)
    for (int i = 0; i < N; i++) {
        // [0..2]: 目标 XYZ
        std::memcpy(&y_ref[0], target_p_ee, 3 * sizeof(double));
        // [3..9]: 期望姿态 q
        std::memcpy(&y_ref[3], target_q_rest, 7 * sizeof(double));
        // [10..16]: 期望速度 v (全0，表示希望机械臂在目标点停下)
        // [17..23]: 期望力矩 u (全0，表示希望消耗最小力矩)
        
        // 将组装好的 y_ref 喂给求解器
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", y_ref);
    }

    // 组装 y_ref_e (对于终端步 i = N)
    std::memcpy(&y_ref_e[0], target_p_ee, 3 * sizeof(double));
    std::memcpy(&y_ref_e[3], target_q_rest, 7 * sizeof(double));
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", y_ref_e);

    // [热启动 Warm-start] 将初始猜测设为当前状态
    for (int i = 0; i <= N; i++) {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, nlp_in, i, "x", current_x);
    }

    std::cout << "========== 开始任务空间控制仿真 (模拟 ROS 2 Timer) ==========" << std::endl;
    std::cout << "目标: 引导末端到达 [X: " << target_p_ee[0] << ", Y: " << target_p_ee[1] << ", Z: " << target_p_ee[2] << "]" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << " Step |  输出力矩 tau1 (Nm) | 求解状态" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // ================= 模拟连续的 ROS 2 控制循环 =================
    int num_simulation_steps = 20;
    double optimal_u[7] = {0.0};

    for (int step = 0; step < num_simulation_steps; ++step) {
        
        // --- 模拟 ROS 2 步骤 1: 读取当前关节状态，设为起点 ---
        // 在实机中，这是从 joint_states_topic 收到的 q 和 v
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "lbx", current_x);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, nlp_out, 0, "ubx", current_x);

        // --- 模拟 ROS 2 步骤 2: 调用 NMPC 求解器 ---
        status = panda_task_space_nmpc_acados_solve(capsule);
        
        if (status != 0 && status != 4) { 
            std::cerr << "求解异常, 错误码: " << status << std::endl;
        }

        // --- 模拟 ROS 2 步骤 3: 提取控制量 (力矩) 并下发给电机 ---
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "u", optimal_u);

        std::cout << std::setw(4) << step << "  |  "
                  << std::fixed << std::setprecision(5) << std::setw(15) << optimal_u[0] << "   |  " 
                  << (status == 0 || status == 4 ? "OK" : "ERR") << std::endl;

        // --- 模拟状态推进 (从求解器内部提取下一步的预测状态，当做下个周期的真实状态) ---
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 1, "x", current_x);
    }

    std::cout << "========== 测试完成，释放内存 ==========" << std::endl;
    
    // 释放内存
    panda_task_space_nmpc_acados_free(capsule);
    panda_task_space_nmpc_acados_free_capsule(capsule);

    return 0;
}