#ifndef PANDA_NMPC_CONTROLLER_HPP_
#define PANDA_NMPC_CONTROLLER_HPP_

#include <array>
#include <algorithm>
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

struct NMPCRuntimeCostConfig {
    using Vec3 = Eigen::Matrix<double, 3, 1>;
    using Vec7 = Eigen::Matrix<double, 7, 1>;

    Vec3 pos = Vec3::Constant(500.0);
    Vec3 rot = Vec3::Constant(100.0);
    Vec3 ee_lin_vel = Vec3::Constant(5.0);
    Vec3 ee_ang_vel = Vec3::Constant(2.0);
    Vec7 dq_reg = Vec7::Constant(0.1);
    Vec7 ddq_reg = Vec7::Constant(0.05);
    Vec7 neutral_q_reg = Vec7::Constant(15.0);

    Vec3 pos_e = Vec3::Constant(750.0);
    Vec3 rot_e = Vec3::Constant(200.0);
    Vec3 ee_lin_vel_e = Vec3::Constant(10.0);
    Vec3 ee_ang_vel_e = Vec3::Constant(4.0);
    Vec7 dq_reg_e = Vec7::Constant(0.2);
    Vec7 neutral_q_reg_e = Vec7::Constant(30.0);

    Vec7 q_neutral = [] {
        Vec7 q;
        q << 0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854;
        return q;
    }();

    double joint_limit_barrier = 0.05;
    double joint_limit_barrier_e = 0.08;
    double manipulability = 5.0;
    double manipulability_e = 8.0;
    double clearance = 180.0;
    double clearance_e = 260.0;
    double barrier_eps = 1.0e-4;
    double manipulability_eps = 1.0e-6;
    double clearance_activation_margin = 0.06;
    double clearance_softplus_gain = 40.0;

    double safe_slack_linear = 1.0e4;
    double safe_slack_quadratic = 1.0e6;
    double approach_slack_linear = 2.0e3;
    double approach_slack_quadratic = 2.0e5;
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

        // generate.py 现在默认生成 EXTERNAL cost，此时 NY/NYN 为 0，不能再按旧版 LS/NLS 接口写 yref。
        if constexpr (PANDA_TASK_SPACE_NMPC_NY > 0) {
            for (int k = 0; k < N_; ++k) {
                ocp_nlp_cost_model_set(
                    nlp_config_, nlp_dims_, nlp_in_, k, "yref",
                    const_cast<double*>(y_ref_.data()));
            }
        }
        if constexpr (PANDA_TASK_SPACE_NMPC_NYN > 0) {
            ocp_nlp_cost_model_set(
                nlp_config_, nlp_dims_, nlp_in_, N_, "yref",
                const_cast<double*>(y_ref_e_.data()));
        }

        applyRuntimeSlackConfig();

        std::cout << "[Panda NMPC] 初始化成功, N = " << N_ << std::endl;
    }

    ~PandaNMPCController() {
        if (capsule_) {
            panda_task_space_nmpc_acados_free(capsule_);
            panda_task_space_nmpc_acados_free_capsule(capsule_);
            capsule_ = nullptr;
        }
    }

    void setRuntimeCostConfig(const NMPCRuntimeCostConfig& config) {
        runtime_cost_config_ = config;
        applyRuntimeSlackConfig();
    }

    const NMPCRuntimeCostConfig& getRuntimeCostConfig() const {
        return runtime_cost_config_;
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

        // runtime cost block 先写入 p，障碍物参数仍放在参数向量末尾。
        writeRuntimeCostParams(p_data);
        writeDefaultObstacleParams(p_data);

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

        writeRuntimeCostParams(p_data_per_stage);
        writeDefaultObstacleParams(p_data_per_stage);

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
    static constexpr int kParamOffsetCost = 25;
    static constexpr int kCostParamSize = 76;
    static constexpr int kParamOffsetObstacles = kParamOffsetCost + kCostParamSize;
    static constexpr int kObstacleParamSize = 4;
    static_assert(PANDA_TASK_SPACE_NMPC_NP >= kParamOffsetObstacles, "NP 小于 generate.py 的基础参数维度");
    static_assert(
        (PANDA_TASK_SPACE_NMPC_NP - kParamOffsetObstacles) % kObstacleParamSize == 0,
        "NP 与 generate.py 的障碍物参数布局不一致");
    static constexpr int kNumRuntimeObstacles = (PANDA_TASK_SPACE_NMPC_NP - kParamOffsetObstacles) / kObstacleParamSize;

    void writeRuntimeCostParams(std::array<double, PANDA_TASK_SPACE_NMPC_NP>& p_data) const {
        int idx = kParamOffsetCost;

        writeVecToParameterArray(p_data, idx, runtime_cost_config_.pos);
        writeVecToParameterArray(p_data, idx, runtime_cost_config_.rot);
        writeVecToParameterArray(p_data, idx, runtime_cost_config_.ee_lin_vel);
        writeVecToParameterArray(p_data, idx, runtime_cost_config_.ee_ang_vel);
        writeVecToParameterArray(p_data, idx, runtime_cost_config_.dq_reg);
        writeVecToParameterArray(p_data, idx, runtime_cost_config_.ddq_reg);
        writeVecToParameterArray(p_data, idx, runtime_cost_config_.neutral_q_reg);

        writeVecToParameterArray(p_data, idx, runtime_cost_config_.pos_e);
        writeVecToParameterArray(p_data, idx, runtime_cost_config_.rot_e);
        writeVecToParameterArray(p_data, idx, runtime_cost_config_.ee_lin_vel_e);
        writeVecToParameterArray(p_data, idx, runtime_cost_config_.ee_ang_vel_e);
        writeVecToParameterArray(p_data, idx, runtime_cost_config_.dq_reg_e);
        writeVecToParameterArray(p_data, idx, runtime_cost_config_.neutral_q_reg_e);

        writeVecToParameterArray(p_data, idx, runtime_cost_config_.q_neutral);

        p_data[idx++] = runtime_cost_config_.joint_limit_barrier;
        p_data[idx++] = runtime_cost_config_.joint_limit_barrier_e;
        p_data[idx++] = runtime_cost_config_.manipulability;
        p_data[idx++] = runtime_cost_config_.manipulability_e;
        p_data[idx++] = runtime_cost_config_.clearance;
        p_data[idx++] = runtime_cost_config_.clearance_e;
        p_data[idx++] = runtime_cost_config_.barrier_eps;
        p_data[idx++] = runtime_cost_config_.manipulability_eps;
        p_data[idx++] = runtime_cost_config_.clearance_activation_margin;
        p_data[idx++] = runtime_cost_config_.clearance_softplus_gain;
    }

    void writeDefaultObstacleParams(std::array<double, PANDA_TASK_SPACE_NMPC_NP>& p_data) const {
        for (int i = 0; i < kNumRuntimeObstacles; ++i) {
            const int base = kParamOffsetObstacles + i * kObstacleParamSize;
            p_data[base + 0] = 1000.0;
            p_data[base + 1] = 1000.0;
            p_data[base + 2] = 1000.0;
            p_data[base + 3] = 0.0;
        }
        if (kNumRuntimeObstacles > 0) {
            const int base = kParamOffsetObstacles;
            p_data[base + 0] = 0.4;
            p_data[base + 1] = -0.1;
            p_data[base + 2] = 0.35;
            p_data[base + 3] = 0.1;
        }
        if (kNumRuntimeObstacles > 1) {
            const int base = kParamOffsetObstacles + kObstacleParamSize;
            p_data[base + 0] = 0.4;
            p_data[base + 1] = 0.1;
            p_data[base + 2] = 0.35;
            p_data[base + 3] = 0.1;
        }
    }

    template <typename Derived>
    static void writeVecToParameterArray(
        std::array<double, PANDA_TASK_SPACE_NMPC_NP>& p_data,
        int& idx,
        const Eigen::MatrixBase<Derived>& vec
    ) {
        for (Eigen::Index i = 0; i < vec.size(); ++i) {
            p_data[static_cast<std::size_t>(idx++)] = vec(i);
        }
    }

    void applyRuntimeSlackConfig() {
        constexpr int kNumSafeSlack = PANDA_TASK_SPACE_NMPC_NSN;
        constexpr int kNumStageSlack = PANDA_TASK_SPACE_NMPC_NS;
        constexpr int kNumApproachSlack = kNumStageSlack - kNumSafeSlack;

        if constexpr (kNumStageSlack > 0) {
            std::array<double, kNumStageSlack> zl{};
            std::array<double, kNumStageSlack> zu{};
            std::array<double, kNumStageSlack> Zl{};
            std::array<double, kNumStageSlack> Zu{};

            for (int i = 0; i < kNumSafeSlack; ++i) {
                zl[static_cast<std::size_t>(i)] = runtime_cost_config_.safe_slack_linear;
                zu[static_cast<std::size_t>(i)] = runtime_cost_config_.safe_slack_linear;
                Zl[static_cast<std::size_t>(i)] = runtime_cost_config_.safe_slack_quadratic;
                Zu[static_cast<std::size_t>(i)] = runtime_cost_config_.safe_slack_quadratic;
            }
            for (int i = 0; i < kNumApproachSlack; ++i) {
                const int idx = kNumSafeSlack + i;
                zl[static_cast<std::size_t>(idx)] = runtime_cost_config_.approach_slack_linear;
                zu[static_cast<std::size_t>(idx)] = runtime_cost_config_.approach_slack_linear;
                Zl[static_cast<std::size_t>(idx)] = runtime_cost_config_.approach_slack_quadratic;
                Zu[static_cast<std::size_t>(idx)] = runtime_cost_config_.approach_slack_quadratic;
            }

            for (int k = 1; k < N_; ++k) {
                ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, k, "zl", zl.data());
                ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, k, "zu", zu.data());
                ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, k, "Zl", Zl.data());
                ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, k, "Zu", Zu.data());
            }
        }

        if constexpr (kNumSafeSlack > 0) {
            std::array<double, kNumSafeSlack> zl_e{};
            std::array<double, kNumSafeSlack> zu_e{};
            std::array<double, kNumSafeSlack> Zl_e{};
            std::array<double, kNumSafeSlack> Zu_e{};
            zl_e.fill(runtime_cost_config_.safe_slack_linear);
            zu_e.fill(runtime_cost_config_.safe_slack_linear);
            Zl_e.fill(runtime_cost_config_.safe_slack_quadratic);
            Zu_e.fill(runtime_cost_config_.safe_slack_quadratic);

            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, N_, "zl", zl_e.data());
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, N_, "zu", zu_e.data());
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, N_, "Zl", Zl_e.data());
            ocp_nlp_cost_model_set(nlp_config_, nlp_dims_, nlp_in_, N_, "Zu", Zu_e.data());
        }
    }

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
    NMPCRuntimeCostConfig runtime_cost_config_{};

    std::array<double, PANDA_TASK_SPACE_NMPC_NY>  y_ref_{};
    std::array<double, PANDA_TASK_SPACE_NMPC_NYN> y_ref_e_{};
};

#endif  // PANDA_NMPC_CONTROLLER_HPP_
