#ifndef PANDA_NMPC_CONTROLLER_HPP_
#define PANDA_NMPC_CONTROLLER_HPP_

#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <Eigen/Dense>

// 引入 acados 核心底层接口
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// 引入自动生成的 Panda 任务空间 NMPC 专属头文件
#include "c_generated_code/acados_solver_panda_task_space_nmpc.h"

class PandaNMPCController {
public:
    PandaNMPCController() {
        std::cout << "[PandaNMPC] 正在初始化 Acados 求解器..." << std::endl;

        // 1. 创建求解器胶囊
        capsule_ = panda_task_space_nmpc_acados_create_capsule();
        int status = panda_task_space_nmpc_acados_create(capsule_);
        if (status) {
            std::cerr << "[PandaNMPC 错误] 创建求解器失败, 错误码: " << status << std::endl;
            throw std::runtime_error("Acados solver initialization failed");
        }

        // 2. 获取核心指针 
        nlp_config_ = panda_task_space_nmpc_acados_get_nlp_config(capsule_);
        nlp_dims_   = panda_task_space_nmpc_acados_get_nlp_dims(capsule_);
        nlp_in_     = panda_task_space_nmpc_acados_get_nlp_in(capsule_);
        nlp_out_    = panda_task_space_nmpc_acados_get_nlp_out(capsule_);

        N_ = nlp_dims_->N; // 预测步数 (通常为 20)

        // 3. 初始化零空间参考姿态 (Home 位)
        target_q_rest_ << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;

        std::cout << "[PandaNMPC] 求解器初始化成功!" << std::endl;
    }

    ~PandaNMPCController() {
        if (capsule_) {
            panda_task_space_nmpc_acados_free(capsule_);
            panda_task_space_nmpc_acados_free_capsule(capsule_);
            std::cout << "[PandaNMPC] 求解器内存已释放。" << std::endl;
        }
    }

    /**
     * @brief 核心求解函数
     * 注意：注释掉未使用的参数名 (如 target_vel_) 是标准 C++ 消除警告的做法
     */
    Eigen::VectorXd NMPCSolve(
        const Eigen::Vector3d& target_pos_,
        const Eigen::Vector<double, 6>& /*target_vel_*/,
        const Eigen::Vector<double, 6>& /*target_acc_*/,
        const Eigen::Matrix3d& /*target_rot_*/,
        const Eigen::VectorXd& jq_,
        const Eigen::VectorXd& jv_
    ) {
        // 1. 安全检查
        if (jq_.size() != 7 || jv_.size() != 7) {
            std::cerr << "[PandaNMPC 错误] 关节状态维度不匹配！必须为 7。" << std::endl;
            return Eigen::VectorXd::Zero(7);
        }

        // 2. 构建当前状态 current_x = [q, v]
        double current_x[14];
        Eigen::Map<Eigen::VectorXd>(current_x, 7) = jq_;
        Eigen::Map<Eigen::VectorXd>(current_x + 7, 7) = jv_;

        // 设置 NMPC 当前时刻的初始约束 (lbx = ubx = current_x)
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", current_x);
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", current_x);

        // 3. 构建参考轨迹 y_ref 和 y_ref_e
        double y_ref[24] = {0.0}; 
        double y_ref_e[17] = {0.0};

        // 【终极修复】：显式创建具名的 Map 对象，彻底终结编译器的语法妄想症
        Eigen::Map<Eigen::Vector3d> map_pos(y_ref);
        Eigen::Map<Eigen::Vector3d> map_pos_e(y_ref_e);
        map_pos = target_pos_;
        map_pos_e = target_pos_;

        Eigen::Map<Eigen::VectorXd> map_q(y_ref + 3, 7);
        Eigen::Map<Eigen::VectorXd> map_q_e(y_ref_e + 3, 7);
        map_q = target_q_rest_;
        map_q_e = target_q_rest_;

        // 将 y_ref 下发到求解器的预测域 (0 to N-1)
        for (int i = 0; i < N_; i++) {
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "yref", y_ref);
        }
        // 下发终端代价值 (Node N)
        ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, N_, "yref", y_ref_e);

        // 4. 调用求解器
        int status = panda_task_space_nmpc_acados_solve(capsule_);

        if (status != 0 && status != 4) { 
            std::cerr << "[PandaNMPC 警告] 求解异常, 状态码: " << status << std::endl;
        }

        // 5. 提取最优控制指令 u (关节力矩)
        double optimal_u[7] = {0.0};
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", optimal_u);

        // 返回 Eigen 类型的结果
        return Eigen::Map<Eigen::VectorXd>(optimal_u, 7);
    }

private:
    panda_task_space_nmpc_solver_capsule* capsule_;
    ocp_nlp_config* nlp_config_;
    ocp_nlp_dims* nlp_dims_;
    ocp_nlp_in* nlp_in_;
    ocp_nlp_out* nlp_out_;
    
    int N_;
    Eigen::Matrix<double, 7, 1> target_q_rest_; 
};

#endif // PANDA_NMPC_CONTROLLER_HPP_