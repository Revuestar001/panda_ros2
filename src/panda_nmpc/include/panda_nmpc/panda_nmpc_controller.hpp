#ifndef PANDA_NMPC_CONTROLLER_HPP_
#define PANDA_NMPC_CONTROLLER_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "acados_c/external_function_interface.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_panda_task_space_nmpc.h"

struct NMPCResult {
    Eigen::Matrix<double, 7, 1> q_ref{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> v_ref{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> a_ref{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> jerk_cmd{Eigen::Matrix<double, 7, 1>::Zero()};
    int status{-1};
};

struct NMPCStageReference {
    Eigen::Matrix<double, 3, 1> target_pos{Eigen::Matrix<double, 3, 1>::Zero()};
    Eigen::Matrix<double, 3, 3> target_rot{Eigen::Matrix<double, 3, 3>::Identity()};
    Eigen::Matrix<double, 7, 1> q_nom{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> dq_nom{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> ddq_nom{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 3, 1> ee_lin_vel_ref{Eigen::Matrix<double, 3, 1>::Zero()};
    Eigen::Matrix<double, 3, 1> ee_ang_vel_ref{Eigen::Matrix<double, 3, 1>::Zero()};
};

class PandaNMPCController {
public:
    using Vec7 = Eigen::Matrix<double, 7, 1>;
    using Vec3 = Eigen::Matrix<double, 3, 1>;
    using Mat3 = Eigen::Matrix<double, 3, 3>;

    using ParamVector = std::array<double, PANDA_TASK_SPACE_NMPC_NP>;
    using StageWeights = std::array<double, PANDA_TASK_SPACE_NMPC_NY>;
    using TerminalWeights = std::array<double, PANDA_TASK_SPACE_NMPC_NYN>;

    enum class SoftLimitMode {
        kHardOnly = 0,
        kSoft = 1,
    };

    struct SoftConstraintPenalty {
        double slack_linear{1.0e3};
        double slack_quadratic{1.0e6};
    };

    struct CostWeights {
        std::array<double, 3> pos{{2500.0, 2500.0, 2500.0}};
        std::array<double, 3> rot{{120.0, 120.0, 120.0}};
        std::array<double, 3> ee_lin_vel{{5.0, 5.0, 5.0}};
        std::array<double, 3> ee_ang_vel{{3.0, 3.0, 3.0}};
        std::array<double, 7> q_reg{{0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2}};
        std::array<double, 7> dq_reg{{0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05}};
        std::array<double, 7> ddq_reg{{0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02}};
        std::array<double, 7> jerk_reg{{1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4, 1.0e-4}};

        std::array<double, 3> pos_e{{5000.0, 5000.0, 5000.0}};
        std::array<double, 3> rot_e{{240.0, 240.0, 240.0}};
        std::array<double, 3> ee_lin_vel_e{{8.0, 8.0, 8.0}};
        std::array<double, 3> ee_ang_vel_e{{5.0, 5.0, 5.0}};
        std::array<double, 7> q_reg_e{{0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4}};
        std::array<double, 7> dq_reg_e{{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}};
        std::array<double, 7> ddq_reg_e{{0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05}};
    };

    struct HardLimits {
        std::array<double, 7> q_lower{{-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973}};
        std::array<double, 7> q_upper{{2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973}};
        std::array<double, 7> dq_abs{{2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100}};
        std::array<double, 7> ddq_abs{{15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0}};
        std::array<double, 7> jerk_abs{{7500.0, 3750.0, 5000.0, 6250.0, 7500.0, 10000.0, 10000.0}};
    };

    struct SoftBounds {
        std::array<double, 7> q_lower{{-2.75, -1.65, -2.75, -2.95, -2.75, 0.05, -2.75}};
        std::array<double, 7> q_upper{{2.75, 1.65, 2.75, -0.12, 2.75, 3.55, 2.75}};
        std::array<double, 7> dq_abs{{1.8, 1.8, 1.8, 1.8, 2.2, 2.2, 2.2}};
        std::array<double, 7> ddq_abs{{12.0, 6.0, 8.0, 10.0, 12.0, 16.0, 16.0}};
        std::array<double, 7> jerk_abs{{5000.0, 2500.0, 3500.0, 4500.0, 5000.0, 7000.0, 7000.0}};
    };

    struct ConstraintConfig {
        HardLimits hard_limits{};
        SoftBounds soft_bounds{};
        SoftLimitMode q_mode{SoftLimitMode::kSoft};
        SoftLimitMode dq_mode{SoftLimitMode::kSoft};
        SoftLimitMode ddq_mode{SoftLimitMode::kHardOnly};
        SoftLimitMode jerk_mode{SoftLimitMode::kHardOnly};
        SoftConstraintPenalty stage_penalty{};
        SoftConstraintPenalty terminal_penalty{2.0e3, 2.0e6};
    };

    static CostWeights defaultCostWeights() { return CostWeights{}; }
    static ConstraintConfig defaultConstraintConfig() { return ConstraintConfig{}; }

    PandaNMPCController() {
        capsule_ = panda_task_space_nmpc_acados_create_capsule();
        if (!capsule_) {
            throw std::runtime_error("[Panda NMPC] create_capsule() failed.");
        }

        const int status = panda_task_space_nmpc_acados_create(capsule_);
        if (status != 0) {
            panda_task_space_nmpc_acados_free_capsule(capsule_);
            capsule_ = nullptr;
            throw std::runtime_error("[Panda NMPC] acados_create() failed.");
        }

        nlp_config_ = panda_task_space_nmpc_acados_get_nlp_config(capsule_);
        nlp_dims_ = panda_task_space_nmpc_acados_get_nlp_dims(capsule_);
        nlp_in_ = panda_task_space_nmpc_acados_get_nlp_in(capsule_);
        nlp_out_ = panda_task_space_nmpc_acados_get_nlp_out(capsule_);

        N_ = PANDA_TASK_SPACE_NMPC_N;
        for (int k = 0; k < N_; ++k) {
            ocp_nlp_cost_model_set(
                nlp_config_, nlp_dims_, nlp_in_, k, "yref",
                const_cast<double*>(y_ref_.data()));
        }
        ocp_nlp_cost_model_set(
            nlp_config_, nlp_dims_, nlp_in_, N_, "yref",
            const_cast<double*>(y_ref_e_.data()));

        setCostWeights(defaultCostWeights());
        setConstraintConfig(defaultConstraintConfig());
    }

    ~PandaNMPCController() {
        if (capsule_) {
            panda_task_space_nmpc_acados_free(capsule_);
            panda_task_space_nmpc_acados_free_capsule(capsule_);
            capsule_ = nullptr;
        }
    }

    void setCostWeights(const CostWeights& weights) {
        cost_weights_ = weights;
        applyCostWeights();
    }

    const CostWeights& getCostWeights() const { return cost_weights_; }

    void setConstraintConfig(const ConstraintConfig& config) {
        constraint_config_ = sanitizeConstraintConfig(config);
        applyHardLimits();
        // 当前 hard-only solver 不生成 h/slack；这里只下发 box hard bounds。
    }

    const ConstraintConfig& getConstraintConfig() const { return constraint_config_; }

    NMPCResult NMPCSolveSingleTarget(
        const Vec3& target_pos,
        const Mat3& target_rot,
        const Vec7& q_nom,
        const Vec7& dq_nom,
        const Vec7& ddq_nom,
        const Vec3& ee_lin_vel_ref,
        const Vec3& ee_ang_vel_ref,
        const Vec7& current_q,
        const Vec7& current_dq,
        const Vec7& current_ddq) {

        const std::size_t stage_count = static_cast<std::size_t>(N_ + 1);
        if (stage_refs_buffer_.size() != stage_count) {
            stage_refs_buffer_.resize(stage_count);
        }
        for (auto& stage_ref : stage_refs_buffer_) {
            stage_ref.target_pos = target_pos;
            stage_ref.target_rot = target_rot;
            stage_ref.q_nom = q_nom;
            stage_ref.dq_nom = dq_nom;
            stage_ref.ddq_nom = ddq_nom;
            stage_ref.ee_lin_vel_ref = ee_lin_vel_ref;
            stage_ref.ee_ang_vel_ref = ee_ang_vel_ref;
        }
        return NMPCSolveStageReferences(stage_refs_buffer_, current_q, current_dq, current_ddq);
    }

    NMPCResult NMPCSolveStageReferences(
        const std::vector<NMPCStageReference>& stage_refs,
        const Vec7& current_q,
        const Vec7& current_dq,
        const Vec7& current_ddq) {

        NMPCResult result;
        result.q_ref = current_q;
        result.v_ref = current_dq;
        result.a_ref = current_ddq;
        result.jerk_cmd.setZero();
        result.status = -1;

        if (stage_refs.size() != static_cast<std::size_t>(N_ + 1)) {
            result.status = -100;
            return result;
        }

        std::array<double, PANDA_TASK_SPACE_NMPC_NX> x0{};
        for (int i = 0; i < 7; ++i) {
            x0[static_cast<std::size_t>(i)] = current_q(i);
            x0[static_cast<std::size_t>(i + 7)] = current_dq(i);
            x0[static_cast<std::size_t>(i + 14)] = current_ddq(i);
        }

        ocp_nlp_constraints_model_set(
            nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx",
            const_cast<double*>(x0.data()));
        ocp_nlp_constraints_model_set(
            nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx",
            const_cast<double*>(x0.data()));

        for (int k = 0; k <= N_; ++k) {
            const auto& stage_ref = stage_refs[static_cast<std::size_t>(k)];
            const ParamVector p_data = buildStageParamVector(stage_ref);
            const int st = panda_task_space_nmpc_acados_update_params(
                capsule_, k, const_cast<double*>(p_data.data()), PANDA_TASK_SPACE_NMPC_NP);
            if (st != 0) {
                result.status = st;
                return result;
            }
        }

        resetWarmStart(current_q, current_dq, current_ddq);

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
            result.q_ref(i) = x1[static_cast<std::size_t>(i)];
            result.v_ref(i) = x1[static_cast<std::size_t>(i + 7)];
            result.a_ref(i) = x1[static_cast<std::size_t>(i + 14)];
            result.jerk_cmd(i) = u0[static_cast<std::size_t>(i)];
        }

        projectToHardLimits(result);
        return result;
    }

    double getDt() const { return dt_; }
    int getN() const { return N_; }

private:
    static constexpr double kDisabledBound = 1.0e15;
    static constexpr std::size_t kParamOffsetQNom = 12;
    static constexpr std::size_t kParamOffsetDQNom = 19;
    static constexpr std::size_t kParamOffsetDDQNom = 26;
    static constexpr std::size_t kParamOffsetEeLinVel = 33;
    static constexpr std::size_t kParamOffsetEeAngVel = 36;

    static constexpr std::size_t kStageSoftDim = 56;
    static constexpr std::size_t kTerminalSoftDim = 42;

    static_assert(PANDA_TASK_SPACE_NMPC_NY0 == PANDA_TASK_SPACE_NMPC_NY,
                  "Stage-0 and stage cost dimensions must match.");

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

    static StageWeights buildStageWeightVector(const CostWeights& weights) {
        StageWeights vec{};
        vec.fill(0.0);
        std::size_t offset = 0;
        appendWeights(vec, offset, weights.pos);
        appendWeights(vec, offset, weights.rot);
        appendWeights(vec, offset, weights.ee_lin_vel);
        appendWeights(vec, offset, weights.ee_ang_vel);
        appendWeights(vec, offset, weights.q_reg);
        appendWeights(vec, offset, weights.dq_reg);
        appendWeights(vec, offset, weights.ddq_reg);
        appendWeights(vec, offset, weights.jerk_reg);
        return vec;
    }

    static TerminalWeights buildTerminalWeightVector(const CostWeights& weights) {
        TerminalWeights vec{};
        vec.fill(0.0);
        std::size_t offset = 0;
        appendWeights(vec, offset, weights.pos_e);
        appendWeights(vec, offset, weights.rot_e);
        appendWeights(vec, offset, weights.ee_lin_vel_e);
        appendWeights(vec, offset, weights.ee_ang_vel_e);
        appendWeights(vec, offset, weights.q_reg_e);
        appendWeights(vec, offset, weights.dq_reg_e);
        appendWeights(vec, offset, weights.ddq_reg_e);
        return vec;
    }

    static ConstraintConfig sanitizeConstraintConfig(const ConstraintConfig& raw) {
        ConstraintConfig cfg = raw;
        for (std::size_t i = 0; i < 7; ++i) {
            cfg.soft_bounds.q_lower[i] = std::max(cfg.soft_bounds.q_lower[i], cfg.hard_limits.q_lower[i]);
            cfg.soft_bounds.q_upper[i] = std::min(cfg.soft_bounds.q_upper[i], cfg.hard_limits.q_upper[i]);
            if (cfg.soft_bounds.q_lower[i] > cfg.soft_bounds.q_upper[i]) {
                const double mid = 0.5 * (cfg.hard_limits.q_lower[i] + cfg.hard_limits.q_upper[i]);
                cfg.soft_bounds.q_lower[i] = mid;
                cfg.soft_bounds.q_upper[i] = mid;
            }
            cfg.soft_bounds.dq_abs[i] = std::min(std::abs(cfg.soft_bounds.dq_abs[i]), cfg.hard_limits.dq_abs[i]);
            cfg.soft_bounds.ddq_abs[i] = std::min(std::abs(cfg.soft_bounds.ddq_abs[i]), cfg.hard_limits.ddq_abs[i]);
            cfg.soft_bounds.jerk_abs[i] = std::min(std::abs(cfg.soft_bounds.jerk_abs[i]), cfg.hard_limits.jerk_abs[i]);
        }
        cfg.stage_penalty.slack_linear = std::max(0.0, cfg.stage_penalty.slack_linear);
        cfg.stage_penalty.slack_quadratic = std::max(0.0, cfg.stage_penalty.slack_quadratic);
        cfg.terminal_penalty.slack_linear = std::max(0.0, cfg.terminal_penalty.slack_linear);
        cfg.terminal_penalty.slack_quadratic = std::max(0.0, cfg.terminal_penalty.slack_quadratic);
        return cfg;
    }

    static void fillSoftBoundsForGroup(
        std::array<double, kStageSoftDim>& uh,
        std::array<double, kStageSoftDim>& zl,
        std::array<double, kStageSoftDim>& zu,
        std::array<double, kStageSoftDim>& Zl,
        std::array<double, kStageSoftDim>& Zu,
        std::size_t start_index,
        const std::array<double, 7>& upper,
        const std::array<double, 7>& lower,
        SoftLimitMode mode,
        const SoftConstraintPenalty& penalty) {

        for (std::size_t i = 0; i < 7; ++i) {
            const std::size_t idx_pos = start_index + i;
            const std::size_t idx_neg = start_index + 7 + i;
            if (mode == SoftLimitMode::kSoft) {
                uh[idx_pos] = upper[i];
                uh[idx_neg] = -lower[i];
                zl[idx_pos] = penalty.slack_linear;
                zl[idx_neg] = penalty.slack_linear;
                zu[idx_pos] = penalty.slack_linear;
                zu[idx_neg] = penalty.slack_linear;
                Zl[idx_pos] = penalty.slack_quadratic;
                Zl[idx_neg] = penalty.slack_quadratic;
                Zu[idx_pos] = penalty.slack_quadratic;
                Zu[idx_neg] = penalty.slack_quadratic;
            } else {
                uh[idx_pos] = kDisabledBound;
                uh[idx_neg] = kDisabledBound;
                zl[idx_pos] = 0.0;
                zl[idx_neg] = 0.0;
                zu[idx_pos] = 0.0;
                zu[idx_neg] = 0.0;
                Zl[idx_pos] = 0.0;
                Zl[idx_neg] = 0.0;
                Zu[idx_pos] = 0.0;
                Zu[idx_neg] = 0.0;
            }
        }
    }

    static void fillSoftBoundsForGroupTerminal(
        std::array<double, kTerminalSoftDim>& uh,
        std::array<double, kTerminalSoftDim>& zl,
        std::array<double, kTerminalSoftDim>& zu,
        std::array<double, kTerminalSoftDim>& Zl,
        std::array<double, kTerminalSoftDim>& Zu,
        std::size_t start_index,
        const std::array<double, 7>& upper,
        const std::array<double, 7>& lower,
        SoftLimitMode mode,
        const SoftConstraintPenalty& penalty) {

        for (std::size_t i = 0; i < 7; ++i) {
            const std::size_t idx_pos = start_index + i;
            const std::size_t idx_neg = start_index + 7 + i;
            if (mode == SoftLimitMode::kSoft) {
                uh[idx_pos] = upper[i];
                uh[idx_neg] = -lower[i];
                zl[idx_pos] = penalty.slack_linear;
                zl[idx_neg] = penalty.slack_linear;
                zu[idx_pos] = penalty.slack_linear;
                zu[idx_neg] = penalty.slack_linear;
                Zl[idx_pos] = penalty.slack_quadratic;
                Zl[idx_neg] = penalty.slack_quadratic;
                Zu[idx_pos] = penalty.slack_quadratic;
                Zu[idx_neg] = penalty.slack_quadratic;
            } else {
                uh[idx_pos] = kDisabledBound;
                uh[idx_neg] = kDisabledBound;
                zl[idx_pos] = 0.0;
                zl[idx_neg] = 0.0;
                zu[idx_pos] = 0.0;
                zu[idx_neg] = 0.0;
                Zl[idx_pos] = 0.0;
                Zl[idx_neg] = 0.0;
                Zu[idx_pos] = 0.0;
                Zu[idx_neg] = 0.0;
            }
        }
    }

    void applyCostWeights() {
        const StageWeights stage_weights = buildStageWeightVector(cost_weights_);
        const TerminalWeights terminal_weights = buildTerminalWeightVector(cost_weights_);
        const auto W = makeDiagonalMatrix(stage_weights);
        const auto We = makeDiagonalMatrix(terminal_weights);

        for (int k = 0; k < N_; ++k) {
            ocp_nlp_cost_model_set(
                nlp_config_, nlp_dims_, nlp_in_, k, "W",
                const_cast<double*>(W.data()));
        }
        ocp_nlp_cost_model_set(
            nlp_config_, nlp_dims_, nlp_in_, N_, "W",
            const_cast<double*>(We.data()));
    }

    void applyHardLimits() {
        std::array<double, PANDA_TASK_SPACE_NMPC_NX> lbx{};
        std::array<double, PANDA_TASK_SPACE_NMPC_NX> ubx{};
        std::array<double, PANDA_TASK_SPACE_NMPC_NU> lbu{};
        std::array<double, PANDA_TASK_SPACE_NMPC_NU> ubu{};

        for (int i = 0; i < 7; ++i) {
            lbx[static_cast<std::size_t>(i)] = constraint_config_.hard_limits.q_lower[static_cast<std::size_t>(i)];
            ubx[static_cast<std::size_t>(i)] = constraint_config_.hard_limits.q_upper[static_cast<std::size_t>(i)];
            lbx[static_cast<std::size_t>(i + 7)] = -constraint_config_.hard_limits.dq_abs[static_cast<std::size_t>(i)];
            ubx[static_cast<std::size_t>(i + 7)] = constraint_config_.hard_limits.dq_abs[static_cast<std::size_t>(i)];
            lbx[static_cast<std::size_t>(i + 14)] = -constraint_config_.hard_limits.ddq_abs[static_cast<std::size_t>(i)];
            ubx[static_cast<std::size_t>(i + 14)] = constraint_config_.hard_limits.ddq_abs[static_cast<std::size_t>(i)];
            lbu[static_cast<std::size_t>(i)] = -constraint_config_.hard_limits.jerk_abs[static_cast<std::size_t>(i)];
            ubu[static_cast<std::size_t>(i)] = constraint_config_.hard_limits.jerk_abs[static_cast<std::size_t>(i)];
        }

        for (int k = 1; k < N_; ++k) {
            ocp_nlp_constraints_model_set(
                nlp_config_, nlp_dims_, nlp_in_, nlp_out_, k, "lbx",
                const_cast<double*>(lbx.data()));
            ocp_nlp_constraints_model_set(
                nlp_config_, nlp_dims_, nlp_in_, nlp_out_, k, "ubx",
                const_cast<double*>(ubx.data()));
        }
        ocp_nlp_constraints_model_set(
            nlp_config_, nlp_dims_, nlp_in_, nlp_out_, N_, "lbx",
            const_cast<double*>(lbx.data()));
        ocp_nlp_constraints_model_set(
            nlp_config_, nlp_dims_, nlp_in_, nlp_out_, N_, "ubx",
            const_cast<double*>(ubx.data()));

        for (int k = 0; k < N_; ++k) {
            ocp_nlp_constraints_model_set(
                nlp_config_, nlp_dims_, nlp_in_, nlp_out_, k, "lbu",
                const_cast<double*>(lbu.data()));
            ocp_nlp_constraints_model_set(
                nlp_config_, nlp_dims_, nlp_in_, nlp_out_, k, "ubu",
                const_cast<double*>(ubu.data()));
        }
    }

    void applySoftConstraintBounds() {
        std::array<double, kStageSoftDim> lh{};
        std::array<double, kStageSoftDim> uh{};
        std::array<double, kStageSoftDim> zl{};
        std::array<double, kStageSoftDim> zu{};
        std::array<double, kStageSoftDim> Zl{};
        std::array<double, kStageSoftDim> Zu{};
        lh.fill(-kDisabledBound);
        uh.fill(kDisabledBound);
        zl.fill(0.0);
        zu.fill(0.0);
        Zl.fill(0.0);
        Zu.fill(0.0);

        fillSoftBoundsForGroup(
            uh, zl, zu, Zl, Zu, 0,
            constraint_config_.soft_bounds.q_upper,
            constraint_config_.soft_bounds.q_lower,
            constraint_config_.q_mode,
            constraint_config_.stage_penalty);

        const auto dq_lower = negateArray(constraint_config_.soft_bounds.dq_abs);
        fillSoftBoundsForGroup(
            uh, zl, zu, Zl, Zu, 14,
            constraint_config_.soft_bounds.dq_abs,
            dq_lower,
            constraint_config_.dq_mode,
            constraint_config_.stage_penalty);

        const auto ddq_lower = negateArray(constraint_config_.soft_bounds.ddq_abs);
        fillSoftBoundsForGroup(
            uh, zl, zu, Zl, Zu, 28,
            constraint_config_.soft_bounds.ddq_abs,
            ddq_lower,
            constraint_config_.ddq_mode,
            constraint_config_.stage_penalty);

        const auto jerk_lower = negateArray(constraint_config_.soft_bounds.jerk_abs);
        fillSoftBoundsForGroup(
            uh, zl, zu, Zl, Zu, 42,
            constraint_config_.soft_bounds.jerk_abs,
            jerk_lower,
            constraint_config_.jerk_mode,
            constraint_config_.stage_penalty);

        for (int k = 0; k < N_; ++k) {
            ocp_nlp_constraints_model_set(
                nlp_config_, nlp_dims_, nlp_in_, nlp_out_, k, "lh",
                const_cast<double*>(lh.data()));
            ocp_nlp_constraints_model_set(
                nlp_config_, nlp_dims_, nlp_in_, nlp_out_, k, "uh",
                const_cast<double*>(uh.data()));
            ocp_nlp_cost_model_set(
                nlp_config_, nlp_dims_, nlp_in_, k, "zl",
                const_cast<double*>(zl.data()));
            ocp_nlp_cost_model_set(
                nlp_config_, nlp_dims_, nlp_in_, k, "zu",
                const_cast<double*>(zu.data()));
            ocp_nlp_cost_model_set(
                nlp_config_, nlp_dims_, nlp_in_, k, "Zl",
                const_cast<double*>(Zl.data()));
            ocp_nlp_cost_model_set(
                nlp_config_, nlp_dims_, nlp_in_, k, "Zu",
                const_cast<double*>(Zu.data()));
        }
    }

    void applySoftConstraintPenalties() {
        std::array<double, kTerminalSoftDim> lh_e{};
        std::array<double, kTerminalSoftDim> uh_e{};
        std::array<double, kTerminalSoftDim> zl_e{};
        std::array<double, kTerminalSoftDim> zu_e{};
        std::array<double, kTerminalSoftDim> Zl_e{};
        std::array<double, kTerminalSoftDim> Zu_e{};
        lh_e.fill(-kDisabledBound);
        uh_e.fill(kDisabledBound);
        zl_e.fill(0.0);
        zu_e.fill(0.0);
        Zl_e.fill(0.0);
        Zu_e.fill(0.0);

        fillSoftBoundsForGroupTerminal(
            uh_e, zl_e, zu_e, Zl_e, Zu_e, 0,
            constraint_config_.soft_bounds.q_upper,
            constraint_config_.soft_bounds.q_lower,
            constraint_config_.q_mode,
            constraint_config_.terminal_penalty);

        const auto dq_lower = negateArray(constraint_config_.soft_bounds.dq_abs);
        fillSoftBoundsForGroupTerminal(
            uh_e, zl_e, zu_e, Zl_e, Zu_e, 14,
            constraint_config_.soft_bounds.dq_abs,
            dq_lower,
            constraint_config_.dq_mode,
            constraint_config_.terminal_penalty);

        const auto ddq_lower = negateArray(constraint_config_.soft_bounds.ddq_abs);
        fillSoftBoundsForGroupTerminal(
            uh_e, zl_e, zu_e, Zl_e, Zu_e, 28,
            constraint_config_.soft_bounds.ddq_abs,
            ddq_lower,
            constraint_config_.ddq_mode,
            constraint_config_.terminal_penalty);

        ocp_nlp_constraints_model_set(
            nlp_config_, nlp_dims_, nlp_in_, nlp_out_, N_, "lh",
            const_cast<double*>(lh_e.data()));
        ocp_nlp_constraints_model_set(
            nlp_config_, nlp_dims_, nlp_in_, nlp_out_, N_, "uh",
            const_cast<double*>(uh_e.data()));
        ocp_nlp_cost_model_set(
            nlp_config_, nlp_dims_, nlp_in_, N_, "zl",
            const_cast<double*>(zl_e.data()));
        ocp_nlp_cost_model_set(
            nlp_config_, nlp_dims_, nlp_in_, N_, "zu",
            const_cast<double*>(zu_e.data()));
        ocp_nlp_cost_model_set(
            nlp_config_, nlp_dims_, nlp_in_, N_, "Zl",
            const_cast<double*>(Zl_e.data()));
        ocp_nlp_cost_model_set(
            nlp_config_, nlp_dims_, nlp_in_, N_, "Zu",
            const_cast<double*>(Zu_e.data()));
    }

    static std::array<double, 7> negateArray(const std::array<double, 7>& values) {
        std::array<double, 7> out{};
        for (std::size_t i = 0; i < 7; ++i) {
            out[i] = -values[i];
        }
        return out;
    }

    ParamVector buildStageParamVector(const NMPCStageReference& ref) const {
        ParamVector p_data{};
        p_data.fill(0.0);

        p_data[0] = ref.target_pos(0);
        p_data[1] = ref.target_pos(1);
        p_data[2] = ref.target_pos(2);

        std::size_t idx = 3;
        for (int col = 0; col < 3; ++col) {
            for (int row = 0; row < 3; ++row) {
                p_data[idx++] = ref.target_rot(row, col);
            }
        }
        for (int i = 0; i < 7; ++i) {
            p_data[kParamOffsetQNom + static_cast<std::size_t>(i)] = ref.q_nom(i);
            p_data[kParamOffsetDQNom + static_cast<std::size_t>(i)] = ref.dq_nom(i);
            p_data[kParamOffsetDDQNom + static_cast<std::size_t>(i)] = ref.ddq_nom(i);
        }
        for (int i = 0; i < 3; ++i) {
            p_data[kParamOffsetEeLinVel + static_cast<std::size_t>(i)] = ref.ee_lin_vel_ref(i);
            p_data[kParamOffsetEeAngVel + static_cast<std::size_t>(i)] = ref.ee_ang_vel_ref(i);
        }
        return p_data;
    }

    void resetWarmStart(const Vec7& q, const Vec7& dq, const Vec7& ddq) {
        if (!has_warm_start_) {
            initializeWarmStart(q, dq, ddq);
            return;
        }

        std::array<double, PANDA_TASK_SPACE_NMPC_NX> x_init{};
        std::array<double, PANDA_TASK_SPACE_NMPC_NU> u_init{};

        for (int i = 0; i < 7; ++i) {
            x_init[static_cast<std::size_t>(i)] = q(i);
            x_init[static_cast<std::size_t>(i + 7)] = dq(i);
            x_init[static_cast<std::size_t>(i + 14)] = ddq(i);
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

    void initializeWarmStart(const Vec7& q, const Vec7& dq, const Vec7& ddq) {
        std::array<double, PANDA_TASK_SPACE_NMPC_NX> x_init{};
        std::array<double, PANDA_TASK_SPACE_NMPC_NU> u_init{};
        u_init.fill(0.0);

        for (int i = 0; i < 7; ++i) {
            x_init[static_cast<std::size_t>(i)] = q(i);
            x_init[static_cast<std::size_t>(i + 7)] = dq(i);
            x_init[static_cast<std::size_t>(i + 14)] = ddq(i);
        }

        for (int k = 0; k <= N_; ++k) {
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "x", x_init.data());
            if (k < N_) {
                ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "u", u_init.data());
            }
        }
    }

    void projectToHardLimits(NMPCResult& result) const {
        for (int i = 0; i < 7; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            result.q_ref(i) = std::clamp(
                result.q_ref(i),
                constraint_config_.hard_limits.q_lower[idx],
                constraint_config_.hard_limits.q_upper[idx]);
            result.v_ref(i) = std::clamp(
                result.v_ref(i),
                -constraint_config_.hard_limits.dq_abs[idx],
                constraint_config_.hard_limits.dq_abs[idx]);
            result.a_ref(i) = std::clamp(
                result.a_ref(i),
                -constraint_config_.hard_limits.ddq_abs[idx],
                constraint_config_.hard_limits.ddq_abs[idx]);
            result.jerk_cmd(i) = std::clamp(
                result.jerk_cmd(i),
                -constraint_config_.hard_limits.jerk_abs[idx],
                constraint_config_.hard_limits.jerk_abs[idx]);
        }
    }

    panda_task_space_nmpc_solver_capsule* capsule_{nullptr};
    ocp_nlp_config* nlp_config_{nullptr};
    ocp_nlp_dims* nlp_dims_{nullptr};
    ocp_nlp_in* nlp_in_{nullptr};
    ocp_nlp_out* nlp_out_{nullptr};

    int N_{0};
    double dt_{0.02};
    bool has_warm_start_{false};

    std::array<double, PANDA_TASK_SPACE_NMPC_NY> y_ref_{};
    std::array<double, PANDA_TASK_SPACE_NMPC_NYN> y_ref_e_{};

    CostWeights cost_weights_{defaultCostWeights()};
    ConstraintConfig constraint_config_{defaultConstraintConfig()};
    std::vector<NMPCStageReference> stage_refs_buffer_{};
};

#endif  // PANDA_NMPC_CONTROLLER_HPP_
