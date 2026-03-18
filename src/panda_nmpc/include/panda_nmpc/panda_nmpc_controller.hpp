#ifndef PANDA_NMPC_CONTROLLER_HPP_
#define PANDA_NMPC_CONTROLLER_HPP_

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados_solver_panda_task_space_nmpc.h"

struct NMPCResult {
    Eigen::Matrix<double, 7, 1> q_ref;
    Eigen::Matrix<double, 7, 1> v_ref;
    Eigen::Matrix<double, 7, 1> a_ref;
    Eigen::Matrix<double, 7, 1> jerk_cmd;
    int status = -1;
};

struct NMPCStageReference {
    Eigen::Matrix<double, 3, 1> target_pos;
    Eigen::Matrix<double, 3, 3> target_rot;
    Eigen::Matrix<double, 7, 1> q_nom;
    Eigen::Matrix<double, 3, 1> ee_lin_vel_ref;
    Eigen::Matrix<double, 3, 1> ee_ang_vel_ref;
};

class PandaNMPCController {
public:
    using Vec7 = Eigen::Matrix<double, 7, 1>;
    using Vec3 = Eigen::Matrix<double, 3, 1>;
    using Mat3 = Eigen::Matrix<double, 3, 3>;

    PandaNMPCController() {
        std::cout << "[Panda NMPC] 正在初始化 acados solver..." << std::endl;

        capsule_ = panda_task_space_nmpc_acados_create_capsule();
        if (!capsule_) {
            throw std::runtime_error("[Panda NMPC] create_capsule() 失败");
        }

        const int status = panda_task_space_nmpc_acados_create(capsule_);
        if (status != 0) {
            panda_task_space_nmpc_acados_free_capsule(capsule_);
            capsule_ = nullptr;
            throw std::runtime_error("[Panda NMPC] acados_create() 失败");
        }

        nlp_config_ = panda_task_space_nmpc_acados_get_nlp_config(capsule_);
        nlp_dims_   = panda_task_space_nmpc_acados_get_nlp_dims(capsule_);
        nlp_in_     = panda_task_space_nmpc_acados_get_nlp_in(capsule_);
        nlp_out_    = panda_task_space_nmpc_acados_get_nlp_out(capsule_);

        N_ = PANDA_TASK_SPACE_NMPC_N;
        y_ref_.fill(0.0);
        y_ref_e_.fill(0.0);

        for (int k = 0; k < N_; ++k) {
            ocp_nlp_cost_model_set(
                nlp_config_, nlp_dims_, nlp_in_, k, "yref",
                const_cast<double*>(y_ref_.data()));
        }
        ocp_nlp_cost_model_set(
            nlp_config_, nlp_dims_, nlp_in_, N_, "yref",
            const_cast<double*>(y_ref_e_.data()));

        std::cout << "[Panda NMPC] 初始化成功, N = " << N_ << std::endl;
    }

    ~PandaNMPCController() {
        if (capsule_) {
            panda_task_space_nmpc_acados_free(capsule_);
            panda_task_space_nmpc_acados_free_capsule(capsule_);
            capsule_ = nullptr;
        }
    }

    // [修改] 保留旧接口兼容静态目标；按 generate.py 的新模型，这里默认末端速度参考为 0。
    NMPCResult NMPCSolve(
        const Vec3& target_pos,
        const Mat3& target_rot,
        const Vec7& q_nom,
        const Vec7& current_q,
        const Vec7& current_v
    ) {
        return NMPCSolve(target_pos, target_rot, q_nom, Vec3::Zero(), Vec3::Zero(), current_q, current_v);
    }

    // [修改] 与 generate.py 当前参数布局一致，显式接收末端线速度/角速度参考。
    NMPCResult NMPCSolve(
        const Vec3& target_pos,
        const Mat3& target_rot,
        const Vec7& q_nom,
        const Vec3& ee_lin_vel_ref,
        const Vec3& ee_ang_vel_ref,
        const Vec7& current_q,
        const Vec7& current_v
    ) {
        NMPCResult result;
        result.q_ref = current_q;
        result.v_ref = Vec7::Zero();
        result.a_ref = Vec7::Zero();
        result.jerk_cmd = Vec7::Zero();
        result.status = -1;

        std::array<double, PANDA_TASK_SPACE_NMPC_NX> x0{};
        for (int i = 0; i < 7; ++i) {
            x0[i] = current_q(i);
            x0[i + 7] = current_v(i);
        }

        ocp_nlp_constraints_model_set(
            nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx",
            const_cast<double*>(x0.data()));
        ocp_nlp_constraints_model_set(
            nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx",
            const_cast<double*>(x0.data()));

        std::array<double, PANDA_TASK_SPACE_NMPC_NP> p_data{};
        p_data[0] = target_pos(0);
        p_data[1] = target_pos(1);
        p_data[2] = target_pos(2);

        int idx = 3;
        for (int col = 0; col < 3; ++col) {
            for (int row = 0; row < 3; ++row) {
                p_data[idx++] = target_rot(row, col);
            }
        }
        for (int i = 0; i < 7; ++i) {
            p_data[kParamOffsetQNom + i] = q_nom(i);
        }

        // [修改] generate.py 中 p[19:25] 已经改成末端速度参考，先按新布局写入。
        for (int i = 0; i < 3; ++i) {
            p_data[kParamOffsetEeLinVel + i] = ee_lin_vel_ref(i);
            p_data[kParamOffsetEeAngVel + i] = ee_ang_vel_ref(i);
        }

        // [修改] 障碍物参数起始偏移改到 25；未显式使用的障碍物先放到远处，相当于关闭。
        for (int i = 0; i < kNumRuntimeObstacles; ++i) {
            const int base = kParamOffsetObstacles + i * kObstacleParamSize;
            p_data[base + 0] = 1000.0;
            p_data[base + 1] = 1000.0;
            p_data[base + 2] = 1000.0;
            p_data[base + 3] = 0.0;
        }
        if (kNumRuntimeObstacles > 0) {
            const int base = kParamOffsetObstacles;
            p_data[base + 0] = 0.4;   // obs_x
            p_data[base + 1] = -0.1;  // obs_y
            p_data[base + 2] = 0.35;  // obs_z
            p_data[base + 3] = 0.1;   // obs_r
        }
        if (kNumRuntimeObstacles > 1) {
            const int base = kParamOffsetObstacles + kObstacleParamSize;
            p_data[base + 0] = 0.4;  // obs_x
            p_data[base + 1] = 0.1;  // obs_y
            p_data[base + 2] = 0.35; // obs_z
            p_data[base + 3] = 0.1;  // obs_r
        }

        for (int k = 0; k <= N_; ++k) {
            const int st = panda_task_space_nmpc_acados_update_params(
                capsule_, k, p_data.data(), PANDA_TASK_SPACE_NMPC_NP);
            if (st != 0) {
                result.status = st;
                return result;
            }
        }

        resetWarmStart(current_q, current_v);

        result.status = panda_task_space_nmpc_acados_solve(capsule_);
        if (result.status != 0) {
            // [修改] 求解失败后丢弃旧 warm start，下一次回退到保守初始化。
            has_warm_start_ = false;
            return result;
        }

        // [修改] 只有在成功拿到一轮解之后，下一次才使用真正的 warm start。
        has_warm_start_ = true;

        std::array<double, PANDA_TASK_SPACE_NMPC_NX> x1{};
        std::array<double, PANDA_TASK_SPACE_NMPC_NU> u0{};
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 1, "x", x1.data());
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", u0.data());

        for (int i = 0; i < 7; ++i) {
            result.q_ref(i) = x1[i];
            result.v_ref(i) = x1[i + 7];
            result.a_ref(i) = u0[i];
            result.jerk_cmd(i) = 0.0;
        }

        return result;
    }

    NMPCResult NMPCSolveTrajectory(
        const std::vector<NMPCStageReference>& stage_refs,
        const Vec7& current_q,
        const Vec7& current_v
    ) {
        NMPCResult result;
        result.q_ref = current_q;
        result.v_ref = Vec7::Zero();
        result.a_ref = Vec7::Zero();
        result.jerk_cmd = Vec7::Zero();
        result.status = -1;

        // [修改] 轨迹接口要求每个 stage 都有一份参考：k = 0..N，共 N+1 个。
        if (stage_refs.size() != static_cast<std::size_t>(N_ + 1)) {
            result.status = -100;
            return result;
        }

        std::array<double, PANDA_TASK_SPACE_NMPC_NX> x0{};
        for (int i = 0; i < 7; ++i) {
            x0[i] = current_q(i);
            x0[i + 7] = current_v(i);
        }

        ocp_nlp_constraints_model_set(
            nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx",
            const_cast<double*>(x0.data()));
        ocp_nlp_constraints_model_set(
            nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx",
            const_cast<double*>(x0.data()));

        std::array<double, PANDA_TASK_SPACE_NMPC_NP> p_data_per_stage{};

        // [修改] 障碍物参数起始偏移改到 25；未显式使用的障碍物先放到远处，相当于关闭。
        for (int i = 0; i < kNumRuntimeObstacles; ++i) {
            const int base = kParamOffsetObstacles + i * kObstacleParamSize;
            p_data_per_stage[base + 0] = 1000.0;
            p_data_per_stage[base + 1] = 1000.0;
            p_data_per_stage[base + 2] = 1000.0;
            p_data_per_stage[base + 3] = 0.0;
        }
        if (kNumRuntimeObstacles > 0) {
            const int base = kParamOffsetObstacles;
            p_data_per_stage[base + 0] = 0.4;   // obs_x
            p_data_per_stage[base + 1] = -0.1;  // obs_y
            p_data_per_stage[base + 2] = 0.35;  // obs_z
            p_data_per_stage[base + 3] = 0.1;   // obs_r
        }
        if (kNumRuntimeObstacles > 1) {
            const int base = kParamOffsetObstacles + kObstacleParamSize;
            p_data_per_stage[base + 0] = 0.4;  // obs_x
            p_data_per_stage[base + 1] = 0.1;  // obs_y
            p_data_per_stage[base + 2] = 0.35; // obs_z
            p_data_per_stage[base + 3] = 0.1;  // obs_r
        }

        for (int k = 0; k <= N_; ++k) {
            const auto& stage_ref = stage_refs[k];

            p_data_per_stage[0] = stage_ref.target_pos(0);
            p_data_per_stage[1] = stage_ref.target_pos(1);
            p_data_per_stage[2] = stage_ref.target_pos(2);

            int idx = 3;
            for (int col = 0; col < 3; ++col) {
                for (int row = 0; row < 3; ++row) {
                    p_data_per_stage[idx++] = stage_ref.target_rot(row, col);
                }
            }

            for (int i = 0; i < 7; ++i) {
                p_data_per_stage[kParamOffsetQNom + i] = stage_ref.q_nom(i);
            }

            for (int i = 0; i < 3; ++i) {
                p_data_per_stage[kParamOffsetEeLinVel + i] = stage_ref.ee_lin_vel_ref(i);
                p_data_per_stage[kParamOffsetEeAngVel + i] = stage_ref.ee_ang_vel_ref(i);
            }

            const int st = panda_task_space_nmpc_acados_update_params(
                capsule_, k, p_data_per_stage.data(), PANDA_TASK_SPACE_NMPC_NP);
            if (st != 0) {
                result.status = st;
                return result;
            }
        }

        resetWarmStart(current_q, current_v);

        result.status = panda_task_space_nmpc_acados_solve(capsule_);
        if (result.status != 0) {
            has_warm_start_ = false;
            return result;
        }

        has_warm_start_ = true;

        std::array<double, PANDA_TASK_SPACE_NMPC_NX> x1{};
        std::array<double, PANDA_TASK_SPACE_NMPC_NU> u0{};
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 1, "x", x1.data());
        ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, 0, "u", u0.data());

        for (int i = 0; i < 7; ++i) {
            result.q_ref(i) = x1[i];
            result.v_ref(i) = x1[i + 7];
            result.a_ref(i) = u0[i];
            result.jerk_cmd(i) = 0.0;
        }

        return result;

    }

    double getDt() { return this->dt_; }
    int getN() { return this->N_; }

private:
    // [修改] 这些偏移与当前 generate.py 的参数布局保持一致。
    static constexpr int kParamOffsetQNom = 12;
    static constexpr int kParamOffsetEeLinVel = 19;
    static constexpr int kParamOffsetEeAngVel = 22;
    static constexpr int kParamOffsetObstacles = 25;
    static constexpr int kObstacleParamSize = 4;
    static_assert(PANDA_TASK_SPACE_NMPC_NP >= kParamOffsetObstacles, "NP 小于 generate.py 的基础参数维度");
    static_assert(
        (PANDA_TASK_SPACE_NMPC_NP - kParamOffsetObstacles) % kObstacleParamSize == 0,
        "NP 与 generate.py 的障碍物参数布局不一致");
    static constexpr int kNumRuntimeObstacles = (PANDA_TASK_SPACE_NMPC_NP - kParamOffsetObstacles) / kObstacleParamSize;

    void resetWarmStart(const Vec7& q, const Vec7& v) {
        if (!has_warm_start_) {
            initializeWarmStart(q, v);
            return;
        }

        // [修改] 真正的 warm start：把上一轮成功解沿预测时域前移一格。
        std::array<double, PANDA_TASK_SPACE_NMPC_NX> x_init{};
        std::array<double, PANDA_TASK_SPACE_NMPC_NU> u_init{};

        for (int i = 0; i < 7; ++i) {
            x_init[i] = q(i);
            x_init[i + 7] = v(i);
        }
        ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, 0, "x", x_init.data());

        for (int k = 0; k < N_; ++k) {
            const int u_src_stage = (k + 1 < N_) ? (k + 1) : (N_ - 1);
            if (u_src_stage >= 0) {
                ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, u_src_stage, "u", u_init.data());
                ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "u", u_init.data());
            }
        }

        for (int k = 1; k <= N_; ++k) {
            const int x_src_stage = (k + 1 <= N_) ? (k + 1) : N_;
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, x_src_stage, "x", x_init.data());
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "x", x_init.data());
        }
    }

    void initializeWarmStart(const Vec7& q, const Vec7& v) {
        // [修改] 没有上一轮有效解时，只做保守初始化，避免每次都覆盖掉真实 warm start。
        std::array<double, PANDA_TASK_SPACE_NMPC_NX> x_init{};
        std::array<double, PANDA_TASK_SPACE_NMPC_NU> u_init{};

        for (int i = 0; i < 7; ++i) {
            x_init[i] = q(i);
            x_init[i + 7] = v(i);
        }

        for (int k = 0; k <= N_; ++k) {
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "x", x_init.data());
            if (k < N_) {
                u_init.fill(0.0);
                ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "u", u_init.data());
            }
        }
    }

    panda_task_space_nmpc_solver_capsule* capsule_ = nullptr;
    ocp_nlp_config* nlp_config_ = nullptr;
    ocp_nlp_dims*   nlp_dims_   = nullptr;
    ocp_nlp_in*     nlp_in_     = nullptr;
    ocp_nlp_out*    nlp_out_    = nullptr;
    int N_ = 0;
    double dt_ = 0.02;
    bool has_warm_start_ = false;

    std::array<double, PANDA_TASK_SPACE_NMPC_NY>  y_ref_{};
    std::array<double, PANDA_TASK_SPACE_NMPC_NYN> y_ref_e_{};
};

#endif  // PANDA_NMPC_CONTROLLER_HPP_
