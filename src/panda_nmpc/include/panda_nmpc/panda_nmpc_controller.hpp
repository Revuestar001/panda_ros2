#ifndef PANDA_NMPC_CONTROLLER_HPP_
#define PANDA_NMPC_CONTROLLER_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>

#include "acados_c/external_function_interface.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_solver_panda_task_space_nmpc.h"

static_assert(PANDA_TASK_SPACE_NMPC_NX == 14, "Generated solver does not match the second-order NMPC model.");
static_assert(PANDA_TASK_SPACE_NMPC_NU == 7, "Unexpected Panda NMPC input dimension.");
static_assert(PANDA_TASK_SPACE_NMPC_NY == 33, "Unexpected Panda NMPC stage cost dimension.");
static_assert(PANDA_TASK_SPACE_NMPC_NYN == 26, "Unexpected Panda NMPC terminal cost dimension.");

constexpr std::size_t kPandaNmpcBaseParamDim = 25;
constexpr std::size_t kPandaNmpcObstacleValuesPerObstacle = 4;
static_assert(
    PANDA_TASK_SPACE_NMPC_NP >= static_cast<int>(kPandaNmpcBaseParamDim),
    "Generated solver parameter dimension is smaller than the fixed base parameter layout.");
constexpr std::size_t kPandaNmpcObstacleParamDim =
    static_cast<std::size_t>(PANDA_TASK_SPACE_NMPC_NP) - kPandaNmpcBaseParamDim;
static_assert(
    kPandaNmpcObstacleParamDim % kPandaNmpcObstacleValuesPerObstacle == 0,
    "Generated solver obstacle parameter block is inconsistent with xyzr packing.");

struct NMPCResult {
    Eigen::Matrix<double, 7, 1> q_ref{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> v_ref{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> a_ref{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 7, 1> jerk_cmd{Eigen::Matrix<double, 7, 1>::Zero()};
    int status{-1};
};

using PandaNmpcObstacleParamBlock = std::array<double, kPandaNmpcObstacleParamDim>;

struct NMPCStageReference {
    Eigen::Matrix<double, 3, 1> target_pos{Eigen::Matrix<double, 3, 1>::Zero()};
    Eigen::Matrix<double, 3, 3> target_rot{Eigen::Matrix<double, 3, 3>::Identity()};
    Eigen::Matrix<double, 7, 1> q_nom{Eigen::Matrix<double, 7, 1>::Zero()};
    Eigen::Matrix<double, 3, 1> ee_lin_vel_ref{Eigen::Matrix<double, 3, 1>::Zero()};
    Eigen::Matrix<double, 3, 1> ee_ang_vel_ref{Eigen::Matrix<double, 3, 1>::Zero()};
    PandaNmpcObstacleParamBlock obstacle_params{};
};

struct NMPCDebugStageSlackInfo {
    int stage{-1};
    int slack_dim{0};
    int active_count{0};
    double max_slack{0.0};
    double sum_slack{0.0};
};

struct NMPCSolveDebugInfo {
    bool available{false};
    bool qp_status_available{false};
    int solver_status{-1};
    int sqp_iter{0};
    int qp_status{0};
    int qp_iter{0};
    double cost_value{std::numeric_limits<double>::quiet_NaN()};
    double res_stat{std::numeric_limits<double>::quiet_NaN()};
    double res_eq{std::numeric_limits<double>::quiet_NaN()};
    double res_ineq{std::numeric_limits<double>::quiet_NaN()};
    double res_comp{std::numeric_limits<double>::quiet_NaN()};
    double time_tot{0.0};
    double time_lin{0.0};
    double time_qp{0.0};
    int total_slack_constraints{0};
    int active_slack_count{0};
    int active_stage_count{0};
    double max_slack{0.0};
    double sum_slack{0.0};
    std::vector<NMPCDebugStageSlackInfo> stage_slack_info{};
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
    };

    struct HardLimits {
        std::array<double, 7> q_lower{{-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973}};
        std::array<double, 7> q_upper{{2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973}};
        std::array<double, 7> dq_abs{{2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100}};
        std::array<double, 7> ddq_abs{{15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0}};
    };

    struct SoftConstraintPenalty {
        double slack_linear{0.0};
        double slack_quadratic{1.0e6};
    };

    struct ObstacleConstraintConfig {
        SoftConstraintPenalty stage_penalty{};
        SoftConstraintPenalty terminal_penalty{};
    };

    static CostWeights defaultCostWeights() { return CostWeights{}; }
    static HardLimits defaultHardLimits() { return HardLimits{}; }
    static ObstacleConstraintConfig defaultObstacleConstraintConfig() { return ObstacleConstraintConfig{}; }

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
        capsule_ = panda_task_space_nmpc_acados_create_capsule();
        if (capsule_ == nullptr) {
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
        nlp_solver_ = panda_task_space_nmpc_acados_get_nlp_solver(capsule_);
        nlp_plan_ = panda_task_space_nmpc_acados_get_nlp_plan(capsule_);

        N_ = PANDA_TASK_SPACE_NMPC_N;
        ocp_nlp_in_get(nlp_config_, nlp_dims_, nlp_in_, 0, "Ts", &dt_);
        qp_status_supported_ =
            (nlp_plan_ != nullptr) &&
            (nlp_plan_->ocp_qp_solver_plan.qp_solver != FULL_CONDENSING_HPIPM);

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

        setCostWeights(defaultCostWeights());
        setHardLimits(defaultHardLimits());
        setObstacleConstraintConfig(defaultObstacleConstraintConfig());
    }

    ~PandaNMPCController() {
        if (capsule_ != nullptr) {
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

    void setHardLimits(const HardLimits& limits) {
        hard_limits_ = limits;
        applyHardLimits();
    }

    const HardLimits& getHardLimits() const { return hard_limits_; }

    void setObstacleConstraintConfig(const ObstacleConstraintConfig& config) {
        obstacle_constraint_config_ = sanitizeObstacleConstraintConfig(config);
        applyObstacleConstraintPenalties();
    }

    const ObstacleConstraintConfig& getObstacleConstraintConfig() const {
        return obstacle_constraint_config_;
    }

    NMPCResult NMPCSolveSingleTarget(
        const Vec3& target_pos,
        const Mat3& target_rot,
        const Vec7& q_nom,
        const Vec3& ee_lin_vel_ref,
        const Vec3& ee_ang_vel_ref,
        const ObstacleParamBlock& obstacle_params,
        const Vec7& current_q,
        const Vec7& current_dq) {

        const std::size_t stage_count = static_cast<std::size_t>(N_ + 1);
        if (stage_refs_buffer_.size() != stage_count) {
            stage_refs_buffer_.resize(stage_count);
        }

        for (auto& stage_ref : stage_refs_buffer_) {
            stage_ref.target_pos = target_pos;
            stage_ref.target_rot = target_rot;
            stage_ref.q_nom = q_nom;
            stage_ref.ee_lin_vel_ref = ee_lin_vel_ref;
            stage_ref.ee_ang_vel_ref = ee_ang_vel_ref;
            stage_ref.obstacle_params = obstacle_params;
        }
        return NMPCSolveStageReferences(stage_refs_buffer_, current_q, current_dq);
    }

    NMPCResult NMPCSolveSingleTarget(
        const Vec3& target_pos,
        const Mat3& target_rot,
        const Vec7& q_nom,
        const Vec3& ee_lin_vel_ref,
        const Vec3& ee_ang_vel_ref,
        const Vec7& current_q,
        const Vec7& current_dq) {
        return NMPCSolveSingleTarget(
            target_pos,
            target_rot,
            q_nom,
            ee_lin_vel_ref,
            ee_ang_vel_ref,
            disabledObstacleParams(),
            current_q,
            current_dq);
    }

    NMPCResult NMPCSolveSingleTarget(
        const Vec3& target_pos,
        const Mat3& target_rot,
        const Vec7& q_nom,
        const Vec7& current_q,
        const Vec7& current_dq) {
        return NMPCSolveSingleTarget(
            target_pos,
            target_rot,
            q_nom,
            Vec3::Zero(),
            Vec3::Zero(),
            disabledObstacleParams(),
            current_q,
            current_dq);
    }

    NMPCResult NMPCSolveStageReferences(
        const std::vector<NMPCStageReference>& stage_refs,
        const Vec7& current_q,
        const Vec7& current_dq) {

        NMPCResult result;
        result.q_ref = current_q;
        result.v_ref = current_dq;
        result.a_ref.setZero();
        result.jerk_cmd.setZero();
        result.status = -1;

        if (stage_refs.size() != static_cast<std::size_t>(N_ + 1)) {
            result.status = -100;
            return result;
        }

        has_last_solve_ = false;
        last_solve_status_ = result.status;

        std::array<double, PANDA_TASK_SPACE_NMPC_NX> x0{};
        for (int i = 0; i < 7; ++i) {
            x0[static_cast<std::size_t>(i)] = current_q(i);
            x0[static_cast<std::size_t>(i + 7)] = current_dq(i);
        }

        ocp_nlp_constraints_model_set(
            nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "lbx",
            const_cast<double*>(x0.data()));
        ocp_nlp_constraints_model_set(
            nlp_config_, nlp_dims_, nlp_in_, nlp_out_, 0, "ubx",
            const_cast<double*>(x0.data()));

        for (int k = 0; k <= N_; ++k) {
            const ParamVector parameters = buildStageParamVector(stage_refs[static_cast<std::size_t>(k)]);
            const int st = panda_task_space_nmpc_acados_update_params(
                capsule_, k, const_cast<double*>(parameters.data()), PANDA_TASK_SPACE_NMPC_NP);
            if (st != 0) {
                result.status = st;
                return result;
            }
        }

        resetWarmStart(current_q, current_dq);

        result.status = panda_task_space_nmpc_acados_solve(capsule_);
        last_solve_status_ = result.status;
        has_last_solve_ = true;
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
            result.a_ref(i) = u0[static_cast<std::size_t>(i)];
            result.jerk_cmd(i) = 0.0;
        }

        projectToHardLimits(result);
        return result;
    }

    NMPCResult NMPCSolve(
        const Vec3& target_pos,
        const Mat3& target_rot,
        const Vec7& q_nom,
        const Vec7& current_q,
        const Vec7& current_dq) {
        return NMPCSolveSingleTarget(target_pos, target_rot, q_nom, current_q, current_dq);
    }

    NMPCResult NMPCSolve(
        const Vec3& target_pos,
        const Mat3& target_rot,
        const Vec7& q_nom,
        const Vec3& ee_lin_vel_ref,
        const Vec3& ee_ang_vel_ref,
        const Vec7& current_q,
        const Vec7& current_dq) {
        return NMPCSolveSingleTarget(
            target_pos, target_rot, q_nom, ee_lin_vel_ref, ee_ang_vel_ref, current_q, current_dq);
    }

    NMPCResult NMPCSolve(
        const Vec3& target_pos,
        const Mat3& target_rot,
        const Vec7& q_nom,
        const Vec3& ee_lin_vel_ref,
        const Vec3& ee_ang_vel_ref,
        const ObstacleParamBlock& obstacle_params,
        const Vec7& current_q,
        const Vec7& current_dq) {
        return NMPCSolveSingleTarget(
            target_pos,
            target_rot,
            q_nom,
            ee_lin_vel_ref,
            ee_ang_vel_ref,
            obstacle_params,
            current_q,
            current_dq);
    }

    double getDt() const { return dt_; }
    int getN() const { return N_; }
    NMPCSolveDebugInfo collectLastSolveDebugInfo(
        double slack_activation_threshold = 1.0e-6,
        bool include_stage_slack_info = false) {
        NMPCSolveDebugInfo info;
        if (!has_last_solve_) {
            return info;
        }

        info.available = true;
        info.qp_status_available = qp_status_supported_;
        info.solver_status = last_solve_status_;

        ocp_nlp_eval_cost(nlp_solver_, nlp_in_, nlp_out_);
        ocp_nlp_eval_residuals(nlp_solver_, nlp_in_, nlp_out_);

        ocp_nlp_get(nlp_solver_, "cost_value", &info.cost_value);
        ocp_nlp_get(nlp_solver_, "sqp_iter", &info.sqp_iter);
        if (qp_status_supported_) {
            ocp_nlp_get(nlp_solver_, "qp_status", &info.qp_status);
        }
        ocp_nlp_get(nlp_solver_, "qp_iter", &info.qp_iter);
        ocp_nlp_get(nlp_solver_, "time_tot", &info.time_tot);
        ocp_nlp_get(nlp_solver_, "time_lin", &info.time_lin);
        ocp_nlp_get(nlp_solver_, "time_qp", &info.time_qp);
        ocp_nlp_get(nlp_solver_, "res_stat", &info.res_stat);
        ocp_nlp_get(nlp_solver_, "res_eq", &info.res_eq);
        ocp_nlp_get(nlp_solver_, "res_ineq", &info.res_ineq);
        ocp_nlp_get(nlp_solver_, "res_comp", &info.res_comp);

        for (int stage = 0; stage <= N_; ++stage) {
            const int ns = ocp_nlp_dims_get_from_attr(nlp_config_, nlp_dims_, nlp_out_, stage, "sl");
            if (ns <= 0) {
                continue;
            }

            std::vector<double> sl(static_cast<std::size_t>(ns), 0.0);
            std::vector<double> su(static_cast<std::size_t>(ns), 0.0);
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, stage, "sl", sl.data());
            ocp_nlp_out_get(nlp_config_, nlp_dims_, nlp_out_, stage, "su", su.data());

            NMPCDebugStageSlackInfo stage_info;
            stage_info.stage = stage;
            stage_info.slack_dim = ns;

            for (int slack_idx = 0; slack_idx < ns; ++slack_idx) {
                const double lower_slack = std::max(0.0, sl[static_cast<std::size_t>(slack_idx)]);
                const double upper_slack = std::max(0.0, su[static_cast<std::size_t>(slack_idx)]);
                const double combined_slack = std::max(lower_slack, upper_slack);

                stage_info.max_slack = std::max(stage_info.max_slack, combined_slack);
                stage_info.sum_slack += lower_slack + upper_slack;
                if (combined_slack > slack_activation_threshold) {
                    stage_info.active_count += 1;
                }
            }

            info.total_slack_constraints += ns;
            info.active_slack_count += stage_info.active_count;
            info.max_slack = std::max(info.max_slack, stage_info.max_slack);
            info.sum_slack += stage_info.sum_slack;
            if (stage_info.active_count > 0) {
                info.active_stage_count += 1;
            }
            if (include_stage_slack_info) {
                info.stage_slack_info.push_back(stage_info);
            }
        }

        return info;
    }

private:
    static constexpr std::size_t kParamOffsetQNom = 12;
    static constexpr std::size_t kParamOffsetEeLinVel = 19;
    static constexpr std::size_t kParamOffsetEeAngVel = 22;
    static constexpr std::size_t kParamOffsetObstacles = kPandaNmpcBaseParamDim;
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

    static ObstacleConstraintConfig sanitizeObstacleConstraintConfig(const ObstacleConstraintConfig& raw) {
        ObstacleConstraintConfig config = raw;
        config.stage_penalty.slack_linear = std::max(0.0, config.stage_penalty.slack_linear);
        config.stage_penalty.slack_quadratic = std::max(0.0, config.stage_penalty.slack_quadratic);
        config.terminal_penalty.slack_linear = std::max(0.0, config.terminal_penalty.slack_linear);
        config.terminal_penalty.slack_quadratic = std::max(0.0, config.terminal_penalty.slack_quadratic);
        return config;
    }

    static StageWeights buildStageWeightVector(const CostWeights& weights) {
        StageWeights result{};
        result.fill(0.0);

        std::size_t offset = 0;
        appendWeights(result, offset, weights.pos);
        appendWeights(result, offset, weights.rot);
        appendWeights(result, offset, weights.ee_lin_vel);
        appendWeights(result, offset, weights.ee_ang_vel);
        appendWeights(result, offset, weights.q_reg);
        appendWeights(result, offset, weights.dq_reg);
        appendWeights(result, offset, weights.ddq_reg);
        return result;
    }

    static TerminalWeights buildTerminalWeightVector(const CostWeights& weights) {
        TerminalWeights result{};
        result.fill(0.0);

        std::size_t offset = 0;
        appendWeights(result, offset, weights.pos_e);
        appendWeights(result, offset, weights.rot_e);
        appendWeights(result, offset, weights.ee_lin_vel_e);
        appendWeights(result, offset, weights.ee_ang_vel_e);
        appendWeights(result, offset, weights.q_reg_e);
        appendWeights(result, offset, weights.dq_reg_e);
        return result;
    }

    void applyCostWeights() {
        const auto stage_weights = makeDiagonalMatrix(buildStageWeightVector(cost_weights_));
        const auto terminal_weights = makeDiagonalMatrix(buildTerminalWeightVector(cost_weights_));

        for (int k = 0; k < N_; ++k) {
            ocp_nlp_cost_model_set(
                nlp_config_, nlp_dims_, nlp_in_, k, "W",
                const_cast<double*>(stage_weights.data()));
        }
        ocp_nlp_cost_model_set(
            nlp_config_, nlp_dims_, nlp_in_, N_, "W",
            const_cast<double*>(terminal_weights.data()));
    }

    void applyHardLimits() {
        std::array<double, PANDA_TASK_SPACE_NMPC_NX> lbx{};
        std::array<double, PANDA_TASK_SPACE_NMPC_NX> ubx{};
        std::array<double, PANDA_TASK_SPACE_NMPC_NU> lbu{};
        std::array<double, PANDA_TASK_SPACE_NMPC_NU> ubu{};

        for (int i = 0; i < 7; ++i) {
            const std::size_t idx = static_cast<std::size_t>(i);
            lbx[idx] = hard_limits_.q_lower[idx];
            ubx[idx] = hard_limits_.q_upper[idx];
            lbx[idx + 7] = -hard_limits_.dq_abs[idx];
            ubx[idx + 7] = hard_limits_.dq_abs[idx];
            lbu[idx] = -hard_limits_.ddq_abs[idx];
            ubu[idx] = hard_limits_.ddq_abs[idx];
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

    void applyObstacleConstraintPenalties() {
        if constexpr (PANDA_TASK_SPACE_NMPC_NS > 0) {
            std::array<double, PANDA_TASK_SPACE_NMPC_NS> zl{};
            std::array<double, PANDA_TASK_SPACE_NMPC_NS> zu{};
            std::array<double, PANDA_TASK_SPACE_NMPC_NS> Zl{};
            std::array<double, PANDA_TASK_SPACE_NMPC_NS> Zu{};
            zl.fill(obstacle_constraint_config_.stage_penalty.slack_linear);
            zu.fill(obstacle_constraint_config_.stage_penalty.slack_linear);
            Zl.fill(obstacle_constraint_config_.stage_penalty.slack_quadratic);
            Zu.fill(obstacle_constraint_config_.stage_penalty.slack_quadratic);

            for (int k = 0; k < N_; ++k) {
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

        if constexpr (PANDA_TASK_SPACE_NMPC_NSN > 0) {
            std::array<double, PANDA_TASK_SPACE_NMPC_NSN> zl_e{};
            std::array<double, PANDA_TASK_SPACE_NMPC_NSN> zu_e{};
            std::array<double, PANDA_TASK_SPACE_NMPC_NSN> Zl_e{};
            std::array<double, PANDA_TASK_SPACE_NMPC_NSN> Zu_e{};
            zl_e.fill(obstacle_constraint_config_.terminal_penalty.slack_linear);
            zu_e.fill(obstacle_constraint_config_.terminal_penalty.slack_linear);
            Zl_e.fill(obstacle_constraint_config_.terminal_penalty.slack_quadratic);
            Zu_e.fill(obstacle_constraint_config_.terminal_penalty.slack_quadratic);

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
    }

    ParamVector buildStageParamVector(const NMPCStageReference& ref) const {
        ParamVector parameters{};
        parameters.fill(0.0);

        parameters[0] = ref.target_pos(0);
        parameters[1] = ref.target_pos(1);
        parameters[2] = ref.target_pos(2);

        std::size_t idx = 3;
        for (int col = 0; col < 3; ++col) {
            for (int row = 0; row < 3; ++row) {
                parameters[idx++] = ref.target_rot(row, col);
            }
        }
        for (int i = 0; i < 7; ++i) {
            parameters[kParamOffsetQNom + static_cast<std::size_t>(i)] = ref.q_nom(i);
        }
        for (int i = 0; i < 3; ++i) {
            parameters[kParamOffsetEeLinVel + static_cast<std::size_t>(i)] = ref.ee_lin_vel_ref(i);
            parameters[kParamOffsetEeAngVel + static_cast<std::size_t>(i)] = ref.ee_ang_vel_ref(i);
        }
        for (std::size_t i = 0; i < ref.obstacle_params.size(); ++i) {
            parameters[kParamOffsetObstacles + i] = ref.obstacle_params[i];
        }
        return parameters;
    }

    void resetWarmStart(const Vec7& q, const Vec7& dq) {
        if (!has_warm_start_) {
            initializeWarmStart(q, dq);
            return;
        }

        std::array<double, PANDA_TASK_SPACE_NMPC_NX> x_init{};
        std::array<double, PANDA_TASK_SPACE_NMPC_NU> u_init{};

        for (int i = 0; i < 7; ++i) {
            x_init[static_cast<std::size_t>(i)] = q(i);
            x_init[static_cast<std::size_t>(i + 7)] = dq(i);
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

    void initializeWarmStart(const Vec7& q, const Vec7& dq) {
        std::array<double, PANDA_TASK_SPACE_NMPC_NX> x_init{};
        std::array<double, PANDA_TASK_SPACE_NMPC_NU> u_init{};
        u_init.fill(0.0);

        for (int i = 0; i < 7; ++i) {
            x_init[static_cast<std::size_t>(i)] = q(i);
            x_init[static_cast<std::size_t>(i + 7)] = dq(i);
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
            result.q_ref(i) = std::clamp(result.q_ref(i), hard_limits_.q_lower[idx], hard_limits_.q_upper[idx]);
            result.v_ref(i) = std::clamp(result.v_ref(i), -hard_limits_.dq_abs[idx], hard_limits_.dq_abs[idx]);
            result.a_ref(i) = std::clamp(result.a_ref(i), -hard_limits_.ddq_abs[idx], hard_limits_.ddq_abs[idx]);
            result.jerk_cmd(i) = 0.0;
        }
    }

    panda_task_space_nmpc_solver_capsule* capsule_{nullptr};
    ocp_nlp_config* nlp_config_{nullptr};
    ocp_nlp_dims* nlp_dims_{nullptr};
    ocp_nlp_in* nlp_in_{nullptr};
    ocp_nlp_out* nlp_out_{nullptr};
    ocp_nlp_solver* nlp_solver_{nullptr};
    ocp_nlp_plan_t* nlp_plan_{nullptr};

    int N_{0};
    double dt_{0.02};
    bool has_warm_start_{false};
    bool has_last_solve_{false};
    bool qp_status_supported_{true};
    int last_solve_status_{-1};

    std::array<double, PANDA_TASK_SPACE_NMPC_NY> y_ref_{};
    std::array<double, PANDA_TASK_SPACE_NMPC_NYN> y_ref_e_{};

    CostWeights cost_weights_{defaultCostWeights()};
    HardLimits hard_limits_{defaultHardLimits()};
    ObstacleConstraintConfig obstacle_constraint_config_{defaultObstacleConstraintConfig()};
    std::vector<NMPCStageReference> stage_refs_buffer_{};
};

#endif  // PANDA_NMPC_CONTROLLER_HPP_
