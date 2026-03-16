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

// 引入自动生成的 Panda Advanced NMPC 专属头文件 (与 Python 生成的 name 保持一致)
#include "acados_solver_panda_task_space_nmpc.h"

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

// 定义 NMPC 的输出结构体，方便 ROS 2 节点调用和对接阻抗控制器
struct NMPCResult {
    Eigen::Matrix<double, 7, 1> q_ref;     // 下一步的参考关节角
    Eigen::Matrix<double, 7, 1> v_ref;     // 下一步的参考关节速度
    Eigen::Matrix<double, 7, 1> a_ref;     // 下一步的参考关节加速度
    Eigen::Matrix<double, 7, 1> jerk_cmd;  // 当前计算出的最优加加速度指令
    int status;                            // 求解器状态码
};

class PandaNMPCController {
public:
    PandaNMPCController(const std::string& urdf_path,
                    const std::string& ee_frame_name)
    {
        std::cout << "[Panda Advanced NMPC] 正在初始化 SOTA Acados 求解器..." << std::endl;

        capsule_ = panda_task_space_nmpc_acados_create_capsule();
        int status = panda_task_space_nmpc_acados_create(capsule_);
        if (status) {
            std::cerr << "[Panda Advanced NMPC 错误] 创建求解器失败, 错误码: " << status << std::endl;
            throw std::runtime_error("Acados solver initialization failed");
        }

        nlp_config_ = panda_task_space_nmpc_acados_get_nlp_config(capsule_);
        nlp_dims_   = panda_task_space_nmpc_acados_get_nlp_dims(capsule_);
        nlp_in_     = panda_task_space_nmpc_acados_get_nlp_in(capsule_);
        nlp_out_    = panda_task_space_nmpc_acados_get_nlp_out(capsule_);

        N_ = nlp_dims_->N;

        std::memset(y_ref_, 0, sizeof(y_ref_));
        std::memset(y_ref_e_, 0, sizeof(y_ref_e_));

        for (int i = 0; i < N_; i++) {
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, i, "yref", y_ref_);
        }
        ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, N_, "yref", y_ref_e_);

        // 用同一个 URDF / frame 初始化 Pinocchio
        pinocchio::urdf::buildModel(urdf_path, pin_model_);
        pin_data_ = pinocchio::Data(pin_model_);
        ee_frame_id_ = pin_model_.getFrameId(ee_frame_name);

        std::cout << "[Panda Advanced NMPC] 求解器初始化成功! 预测步数 N = " << N_ << std::endl;
    }

    ~PandaNMPCController() {
        if (capsule_) {
            panda_task_space_nmpc_acados_free(capsule_);
            panda_task_space_nmpc_acados_free_capsule(capsule_);
            std::cout << "[Panda Advanced NMPC] 求解器内存已安全释放。" << std::endl;
        }
    }

    /**
     * @brief 核心求解函数
     * 
     * @param target_pos 任务空间目标位置 (3D)
     * @param target_rot 任务空间目标姿态 (SO(3) 旋转矩阵)
     * @param obs_pos    动态障碍物当前位置 (3D)
     * @param current_q  当前关节位置 (7D)
     * @param current_v  当前关节速度 (7D)
     * @param current_a  当前关节加速度 (7D) - 若无传感器可传入上一周期的 a_ref
     * @return NMPCResult 包含下一步的参考状态 (q_ref, v_ref, a_ref) 和求解状态
     */
    NMPCResult NMPCSolve(
        const Eigen::Vector3d& target_pos,
        const Eigen::Matrix3d& target_rot,
        const Eigen::VectorXd& reference_q,
        const Eigen::VectorXd& current_q,
        const Eigen::VectorXd& current_v,
        const Eigen::VectorXd& current_a
    ) {
        // Eigen::Vector3d fk_pos;
        // Eigen::Matrix3d fk_rot;
        // computeCurrentEEPose(current_q, fk_pos, fk_rot);

        // std::cout << "FK current pos = " << fk_pos.transpose() << std::endl;
        // std::cout << "target pos     = " << target_pos.transpose() << std::endl;
        // std::cout << "pos err norm    = " << (fk_pos - target_pos).norm() << std::endl;

        // Eigen::Matrix3d R_err = fk_rot * target_rot.transpose();
        // double tr = std::max(-1.0, std::min(3.0, (R_err.trace() - 1.0) * 0.5));
        // double ang = std::acos(std::max(-1.0, std::min(1.0, tr)));
        // std::cout << "rot err angle   = " << ang << std::endl;
        Eigen::Vector3d fk_pos;
        Eigen::Matrix3d fk_rot;
        computeCurrentEEPose(current_q, fk_pos, fk_rot);

        // 临时测试：强制目标=当前位姿
        const Eigen::Vector3d& target_pos_used = fk_pos;
        const Eigen::Matrix3d& target_rot_used = fk_rot;

        NMPCResult result;
        result.status = -1;

        // 1. 严格的安全维度检查 (nx = 21)
        if (reference_q.size() != 7 || current_q.size() != 7 || current_v.size() != 7 || current_a.size() != 7) {
            std::cerr << "[Panda NMPC 错误] 关节状态维度不匹配！必须均为 7。" << std::endl;
            return result;
        }

        // 2. 构建当前状态 x0 = [q, v, a]
        double current_x[PANDA_TASK_SPACE_NMPC_NX] = {0.0};
        Eigen::Map<Eigen::Matrix<double, 7, 1>>(current_x, 7) = current_q;
        Eigen::Map<Eigen::Matrix<double, 7, 1>>(current_x + 7, 7)  = current_v;
        Eigen::Map<Eigen::Matrix<double, 7, 1>>(current_x + 14, 7) = current_a;

        // 将当前状态作为 NMPC 初始节点的硬约束 (Initial State Constraint)
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx", current_x);
        ocp_nlp_constraints_model_set(nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx", current_x);

        // 3. 构建并下发实时参数 (Parameters p)
        double p_data[PANDA_TASK_SPACE_NMPC_NP] = {0.0};
        Eigen::Map<Eigen::Vector3d> map_target_pos(p_data);
        map_target_pos = target_pos;
        Eigen::Map<Eigen::Matrix3d> map_target_rot(p_data + 3);
        map_target_rot = target_rot; 
        Eigen::Map<Eigen::Vector<double, 7>> map_q_nom(p_data + 12);
        map_q_nom = reference_q;

        // 将参数更新到所有的预测视野节点 (0 到 N)
        for (int i = 0; i <= N_; i++) {
            panda_task_space_nmpc_acados_update_params(capsule_, i, p_data, PANDA_TASK_SPACE_NMPC_NP);
        }

        resetWarmStart(current_q, current_v, current_a);

        static bool printed_once = false;
        if (!printed_once) {
            std::cout << "\n===== NMPC DEBUG ONCE =====" << std::endl;
            std::cout << "current_q = " << current_q.transpose() << std::endl;
            std::cout << "current_v = " << current_v.transpose() << std::endl;
            std::cout << "current_a = " << current_a.transpose() << std::endl;
            std::cout << "reference_q = " << reference_q.transpose() << std::endl;

            std::cout << "target_pos = "
                    << p_data[0] << ", "
                    << p_data[1] << ", "
                    << p_data[2] << std::endl;

            std::cout << "target_rot(col-major) = ";
            for (int i = 3; i < 12; ++i) std::cout << p_data[i] << " ";
            std::cout << std::endl;

            std::cout << "q_nom = ";
            for (int i = 12; i < 19; ++i) std::cout << p_data[i] << " ";
            std::cout << std::endl;

            std::cout << "===========================" << std::endl;
            printed_once = true;
        }

        // 4. 调用 Acados RTI 求解器进行一步极速优化
        result.status = panda_task_space_nmpc_acados_solve(capsule_);

        if (result.status != 0 && result.status != 4) { 
            // 提示: 状态码 4 通常表示达到最大迭代次数，在 RTI 模式下可以接受
            std::cerr << "[Panda NMPC 警告] 求解异常/次优解, 状态码: " << result.status << std::endl;
        }

        // 5. 提取最优输出 (Extract SOTA Output)
        // 【关键逻辑】: 我们要发给阻抗控制器的不是 node 0 的状态(因为那是当前状态)，
        // 而是 NMPC 预测出的下一步 node 1 的参考状态 (x_1 = [q_ref, v_ref, a_ref])
        double next_x[PANDA_TASK_SPACE_NMPC_NX] = {0.0};
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 1, "x", next_x);

        // 提取当前周期的最优控制量 (u_0 = jerk)
        double optimal_u[PANDA_TASK_SPACE_NMPC_NU] = {0.0};
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", optimal_u);

        // 6. 装填返回值
        result.q_ref    = Eigen::Map<Eigen::Matrix<double, 7, 1>>(&next_x[0]);
        result.v_ref    = Eigen::Map<Eigen::Matrix<double, 7, 1>>(&next_x[7]);
        result.a_ref    = Eigen::Map<Eigen::Matrix<double, 7, 1>>(&next_x[14]);
        result.jerk_cmd = Eigen::Map<Eigen::Matrix<double, 7, 1>>(&optimal_u[0]);

        return result;
    }

    void computeCurrentEEPose(const Eigen::Matrix<double, 7, 1>& q,
                          Eigen::Vector3d& pos,
                          Eigen::Matrix3d& rot)
    {
        pinocchio::forwardKinematics(pin_model_, pin_data_, q);
        pinocchio::updateFramePlacements(pin_model_, pin_data_);

        const auto& oMf = pin_data_.oMf[ee_frame_id_];
        pos = oMf.translation();
        rot = oMf.rotation();
    }

private:
    panda_task_space_nmpc_solver_capsule* capsule_;
    ocp_nlp_config* nlp_config_;
    ocp_nlp_dims* nlp_dims_;
    ocp_nlp_in* nlp_in_;
    ocp_nlp_out* nlp_out_;
    
    int N_;

    // 代价函数残差目标值预分配内存
    double y_ref_[PANDA_TASK_SPACE_NMPC_NY]; 
    double y_ref_e_[PANDA_TASK_SPACE_NMPC_NYN]; 

    pinocchio::Model pin_model_;
    pinocchio::Data pin_data_{pin_model_};
    pinocchio::FrameIndex ee_frame_id_;

    void resetWarmStart(const Eigen::Matrix<double, 7, 1>& q,
                    const Eigen::Matrix<double, 7, 1>& v,
                    const Eigen::Matrix<double, 7, 1>& a)
    {
        Eigen::Matrix<double, 7, 1> qk = q;
        Eigen::Matrix<double, 7, 1> vk = v;
        Eigen::Matrix<double, 7, 1> ak = a;

        const double dt = 0.02;   // 必须和生成器一致；如果你改了生成器 dt，这里也要一起改

        for (int k = 0; k <= N_; ++k) {
            double x_init[PANDA_TASK_SPACE_NMPC_NX] = {0.0};

            for (int i = 0; i < 7; ++i) {
                x_init[i]      = qk(i);
                x_init[i + 7]  = vk(i);
                x_init[i + 14] = ak(i);
            }

            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "x", x_init);

            if (k < N_) {
                // 先把控制初值设成 0 jerk
                double u_init[PANDA_TASK_SPACE_NMPC_NU] = {0.0};
                ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "u", u_init);

                // 用 jerk = 0 做一个简单前滚，保证 warm start 至少大体符合动力学
                qk = qk + dt * vk + 0.5 * dt * dt * ak;
                vk = vk + dt * ak;
                // ak 保持不变，因为 jerk = 0
            }
        }
    }
};

#endif // PANDA_ADVANCED_NMPC_CONTROLLER_HPP_