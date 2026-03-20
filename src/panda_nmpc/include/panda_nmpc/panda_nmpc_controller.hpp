#ifndef PANDA_NMPC_CONTROLLER_HPP_
#define PANDA_NMPC_CONTROLLER_HPP_

#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "acados_c/external_function_interface.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_panda_task_space_nmpc.h"

constexpr std::size_t kPandaNmpcBaseParamDim = 25;
constexpr std::size_t kPandaNmpcObstacleValuesPerObstacle = 4;
static_assert(
    PANDA_TASK_SPACE_NMPC_NP >= static_cast<int>(kPandaNmpcBaseParamDim),
    "NP 小于 generate.py 的基础参数维度");
constexpr std::size_t kPandaNmpcObstacleParamDim =
    static_cast<std::size_t>(PANDA_TASK_SPACE_NMPC_NP) - kPandaNmpcBaseParamDim;
static_assert(
    kPandaNmpcObstacleParamDim % kPandaNmpcObstacleValuesPerObstacle == 0,
    "NP 与 generate.py 的球障碍参数布局不一致");
using PandaNmpcObstacleParamBlock = std::array<double, kPandaNmpcObstacleParamDim>;

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
    // 与 generate.py 的参数布局一致：p[25:] = [obs_0_xyzr, obs_1_xyzr, ...]。
    PandaNmpcObstacleParamBlock obstacle_params{};
};

class PandaNMPCController {
public:
    using Vec7 = Eigen::Matrix<double, 7, 1>;
    using Vec3 = Eigen::Matrix<double, 3, 1>;
    using Mat3 = Eigen::Matrix<double, 3, 3>;
    using ParamVector = std::array<double, PANDA_TASK_SPACE_NMPC_NP>;
    using ObstacleParamBlock = PandaNmpcObstacleParamBlock;
    using StageWeights = std::array<double, PANDA_TASK_SPACE_NMPC_NY>;
    using TerminalWeights = std::array<double, PANDA_TASK_SPACE_NMPC_NYN>;

    static constexpr std::size_t kNumRuntimeObstacles =
        kPandaNmpcObstacleParamDim / kPandaNmpcObstacleValuesPerObstacle;

    struct SoftConstraintPenalty {
        double slack_linear{1.0e4};
        double slack_quadratic{1.0e6};
    };

    struct CostWeights {
        std::array<double, 3> pos{{2500.0, 2500.0, 2500.0}};
        std::array<double, 3> rot{{100.0, 100.0, 100.0}};
        std::array<double, 3> ee_lin_vel{{5.0, 5.0, 5.0}};
        std::array<double, 3> ee_ang_vel{{2.0, 2.0, 2.0}};
        std::array<double, 7> q_reg{{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}};
        std::array<double, 7> dq_reg{{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}};
        std::array<double, 7> ddq_reg{{0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05}};

        std::array<double, 3> pos_e{{4000.0, 4000.0, 4000.0}};
        std::array<double, 3> rot_e{{200.0, 200.0, 200.0}};
        std::array<double, 3> ee_lin_vel_e{{10.0, 10.0, 10.0}};
        std::array<double, 3> ee_ang_vel_e{{4.0, 4.0, 4.0}};
        std::array<double, 7> q_reg_e{{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};
        std::array<double, 7> dq_reg_e{{0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2}};

        SoftConstraintPenalty soft_constraint_stage{};
        SoftConstraintPenalty soft_constraint_terminal{};
    };

    static CostWeights defaultCostWeights() { return CostWeights{}; }

    static ObstacleParamBlock disabledObstacleParams() {
        ObstacleParamBlock params{};
        params.fill(0.0);
        for (std::size_t obstacle_idx = 0; obstacle_idx < kNumRuntimeObstacles; ++obstacle_idx) {
            const std::size_t offset = obstacle_idx * kPandaNmpcObstacleValuesPerObstacle;
            params[offset + 0] = 1000.0;
            params[offset + 1] = 1000.0;
            params[offset + 2] = 1000.0;
            params[offset + 3] = 0.0;
        }
        return params;
    }

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
        nlp_dims_ = panda_task_space_nmpc_acados_get_nlp_dims(capsule_);
        nlp_in_ = panda_task_space_nmpc_acados_get_nlp_in(capsule_);
        nlp_out_ = panda_task_space_nmpc_acados_get_nlp_out(capsule_);

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

        // generate.py 会给 solver 写入 bootstrap 权重；控制器侧再覆盖一次，
        // 让运行时 ROS 参数成为真正的唯一入口。
        setCostWeights(defaultCostWeights());

        std::cout << "[Panda NMPC] 初始化成功, N = " << N_ << std::endl;
    }

    ~PandaNMPCController() {
        if (capsule_) {
            panda_task_space_nmpc_acados_free(capsule_);
            panda_task_space_nmpc_acados_free_capsule(capsule_);
            capsule_ = nullptr;
        }
    }

    NMPCResult NMPCSolve(
        const Vec3& target_pos,
        const Mat3& target_rot,
        const Vec7& q_nom,
        const Vec7& current_q,
        const Vec7& current_v
    ) {
        return NMPCSolve(
            target_pos,
            target_rot,
            q_nom,
            Vec3::Zero(),
            Vec3::Zero(),
            disabledObstacleParams(),
            current_q,
            current_v);
    }

    NMPCResult NMPCSolve(
        const Vec3& target_pos,
        const Mat3& target_rot,
        const Vec7& q_nom,
        const Vec3& ee_lin_vel_ref,
        const Vec3& ee_ang_vel_ref,
        const Vec7& current_q,
        const Vec7& current_v
    ) {
        return NMPCSolve(
            target_pos,
            target_rot,
            q_nom,
            ee_lin_vel_ref,
            ee_ang_vel_ref,
            disabledObstacleParams(),
            current_q,
            current_v);
    }

    NMPCResult NMPCSolve(
        const Vec3& target_pos,
        const Mat3& target_rot,
        const Vec7& q_nom,
        const Vec3& ee_lin_vel_ref,
        const Vec3& ee_ang_vel_ref,
        const ObstacleParamBlock& obstacle_params,
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

        ParamVector p_data = buildStageParamVector(
            target_pos, target_rot, q_nom, ee_lin_vel_ref, ee_ang_vel_ref, obstacle_params);

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

        for (int k = 0; k <= N_; ++k) {
            const auto& stage_ref = stage_refs[static_cast<std::size_t>(k)];
            ParamVector p_data_per_stage = buildStageParamVector(
                stage_ref.target_pos,
                stage_ref.target_rot,
                stage_ref.q_nom,
                stage_ref.ee_lin_vel_ref,
                stage_ref.ee_ang_vel_ref,
                stage_ref.obstacle_params);

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

    double getDt() { return dt_; }
    int getN() { return N_; }

    void setCostWeights(const CostWeights& cost_weights) {
        cost_weights_ = cost_weights;
        applyCostWeights();
    }

    const CostWeights& getCostWeights() const { return cost_weights_; }

private:
    static constexpr std::size_t kParamOffsetQNom = 12;
    static constexpr std::size_t kParamOffsetEeLinVel = 19;
    static constexpr std::size_t kParamOffsetEeAngVel = 22;
    static constexpr std::size_t kParamOffsetObstacles = kPandaNmpcBaseParamDim;
    static_assert(PANDA_TASK_SPACE_NMPC_NY0 == PANDA_TASK_SPACE_NMPC_NY,
                  "Stage 0 and intermediate stage cost dimensions diverged.");

    template <std::size_t TargetDim, std::size_t BlockDim>
    static void appendWeights(
        std::array<double, TargetDim>& dst,
        std::size_t& offset,
        const std::array<double, BlockDim>& src) {
        for (double value : src) {
            dst[offset++] = value;
        }
    }

    template <std::size_t Dim>
    static std::array<double, Dim * Dim> makeDiagonalMatrix(const std::array<double, Dim>& diagonal) {
        std::array<double, Dim * Dim> matrix{};
        matrix.fill(0.0);
        for (std::size_t i = 0; i < Dim; ++i) {
            matrix[i + Dim * i] = diagonal[i];
        }
        return matrix;
    }

    static StageWeights buildStageWeightVector(const CostWeights& cost_weights) {
        StageWeights weights{};
        weights.fill(0.0);

        std::size_t offset = 0;
        appendWeights(weights, offset, cost_weights.pos);
        appendWeights(weights, offset, cost_weights.rot);
        appendWeights(weights, offset, cost_weights.ee_lin_vel);
        appendWeights(weights, offset, cost_weights.ee_ang_vel);
        appendWeights(weights, offset, cost_weights.q_reg);
        appendWeights(weights, offset, cost_weights.dq_reg);
        appendWeights(weights, offset, cost_weights.ddq_reg);
        return weights;
    }

    static TerminalWeights buildTerminalWeightVector(const CostWeights& cost_weights) {
        TerminalWeights weights{};
        weights.fill(0.0);

        std::size_t offset = 0;
        appendWeights(weights, offset, cost_weights.pos_e);
        appendWeights(weights, offset, cost_weights.rot_e);
        appendWeights(weights, offset, cost_weights.ee_lin_vel_e);
        appendWeights(weights, offset, cost_weights.ee_ang_vel_e);
        appendWeights(weights, offset, cost_weights.q_reg_e);
        appendWeights(weights, offset, cost_weights.dq_reg_e);
        return weights;
    }

    void applyCostWeights() {
        const StageWeights stage_weights = buildStageWeightVector(cost_weights_);
        const TerminalWeights terminal_weights = buildTerminalWeightVector(cost_weights_);
        const auto W0 = makeDiagonalMatrix(stage_weights);
        const auto W = makeDiagonalMatrix(stage_weights);
        const auto We = makeDiagonalMatrix(terminal_weights);

        ocp_nlp_cost_model_set(
            nlp_config_, nlp_dims_, nlp_in_, 0, "W",
            const_cast<double*>(W0.data()));
        for (int k = 1; k < N_; ++k) {
            ocp_nlp_cost_model_set(
                nlp_config_, nlp_dims_, nlp_in_, k, "W",
                const_cast<double*>(W.data()));
        }
        ocp_nlp_cost_model_set(
            nlp_config_, nlp_dims_, nlp_in_, N_, "W",
            const_cast<double*>(We.data()));

        if (PANDA_TASK_SPACE_NMPC_NS > 0) {
            std::array<double, PANDA_TASK_SPACE_NMPC_NS> Zl{};
            std::array<double, PANDA_TASK_SPACE_NMPC_NS> Zu{};
            std::array<double, PANDA_TASK_SPACE_NMPC_NS> zl{};
            std::array<double, PANDA_TASK_SPACE_NMPC_NS> zu{};
            Zl.fill(cost_weights_.soft_constraint_stage.slack_quadratic);
            Zu.fill(cost_weights_.soft_constraint_stage.slack_quadratic);
            zl.fill(cost_weights_.soft_constraint_stage.slack_linear);
            zu.fill(cost_weights_.soft_constraint_stage.slack_linear);

            for (int k = 1; k < N_; ++k) {
                ocp_nlp_cost_model_set(
                    nlp_config_, nlp_dims_, nlp_in_, k, "Zl",
                    const_cast<double*>(Zl.data()));
                ocp_nlp_cost_model_set(
                    nlp_config_, nlp_dims_, nlp_in_, k, "Zu",
                    const_cast<double*>(Zu.data()));
                ocp_nlp_cost_model_set(
                    nlp_config_, nlp_dims_, nlp_in_, k, "zl",
                    const_cast<double*>(zl.data()));
                ocp_nlp_cost_model_set(
                    nlp_config_, nlp_dims_, nlp_in_, k, "zu",
                    const_cast<double*>(zu.data()));
            }
        }

        if (PANDA_TASK_SPACE_NMPC_NSN > 0) {
            std::array<double, PANDA_TASK_SPACE_NMPC_NSN> Zl_e{};
            std::array<double, PANDA_TASK_SPACE_NMPC_NSN> Zu_e{};
            std::array<double, PANDA_TASK_SPACE_NMPC_NSN> zl_e{};
            std::array<double, PANDA_TASK_SPACE_NMPC_NSN> zu_e{};
            Zl_e.fill(cost_weights_.soft_constraint_terminal.slack_quadratic);
            Zu_e.fill(cost_weights_.soft_constraint_terminal.slack_quadratic);
            zl_e.fill(cost_weights_.soft_constraint_terminal.slack_linear);
            zu_e.fill(cost_weights_.soft_constraint_terminal.slack_linear);

            ocp_nlp_cost_model_set(
                nlp_config_, nlp_dims_, nlp_in_, N_, "Zl",
                const_cast<double*>(Zl_e.data()));
            ocp_nlp_cost_model_set(
                nlp_config_, nlp_dims_, nlp_in_, N_, "Zu",
                const_cast<double*>(Zu_e.data()));
            ocp_nlp_cost_model_set(
                nlp_config_, nlp_dims_, nlp_in_, N_, "zl",
                const_cast<double*>(zl_e.data()));
            ocp_nlp_cost_model_set(
                nlp_config_, nlp_dims_, nlp_in_, N_, "zu",
                const_cast<double*>(zu_e.data()));
        }
    }

    ParamVector buildStageParamVector(
        const Vec3& target_pos,
        const Mat3& target_rot,
        const Vec7& q_nom,
        const Vec3& ee_lin_vel_ref,
        const Vec3& ee_ang_vel_ref,
        const ObstacleParamBlock& obstacle_params
    ) const {
        ParamVector p_data{};
        p_data.fill(0.0);

        p_data[0] = target_pos(0);
        p_data[1] = target_pos(1);
        p_data[2] = target_pos(2);

        int idx = 3;
        for (int col = 0; col < 3; ++col) {
            for (int row = 0; row < 3; ++row) {
                p_data[static_cast<std::size_t>(idx++)] = target_rot(row, col);
            }
        }

        for (int i = 0; i < 7; ++i) {
            p_data[kParamOffsetQNom + static_cast<std::size_t>(i)] = q_nom(i);
        }
        for (int i = 0; i < 3; ++i) {
            p_data[kParamOffsetEeLinVel + static_cast<std::size_t>(i)] = ee_lin_vel_ref(i);
            p_data[kParamOffsetEeAngVel + static_cast<std::size_t>(i)] = ee_ang_vel_ref(i);
        }

        for (std::size_t i = 0; i < obstacle_params.size(); ++i) {
            p_data[kParamOffsetObstacles + i] = obstacle_params[i];
        }
        return p_data;
    }

    void resetWarmStart(const Vec7& q, const Vec7& v) {
        if (!has_warm_start_) {
            initializeWarmStart(q, v);
            return;
        }

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
    ocp_nlp_dims* nlp_dims_ = nullptr;
    ocp_nlp_in* nlp_in_ = nullptr;
    ocp_nlp_out* nlp_out_ = nullptr;
    int N_ = 0;
    double dt_ = 0.02;
    bool has_warm_start_ = false;

    std::array<double, PANDA_TASK_SPACE_NMPC_NY> y_ref_{};
    std::array<double, PANDA_TASK_SPACE_NMPC_NYN> y_ref_e_{};
    CostWeights cost_weights_{defaultCostWeights()};
};

#endif  // PANDA_NMPC_CONTROLLER_HPP_
