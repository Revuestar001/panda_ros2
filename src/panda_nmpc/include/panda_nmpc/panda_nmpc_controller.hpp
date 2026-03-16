#ifndef PANDA_NMPC_CONTROLLER_HPP_
#define PANDA_NMPC_CONTROLLER_HPP_

#include <array>
#include <iostream>
#include <stdexcept>

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

    NMPCResult NMPCSolve(
        const Vec3& target_pos,
        const Mat3& target_rot,
        const Vec7& q_nom,
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
            p_data[12 + i] = q_nom(i);
        }

        for (int i = 0; i < 2; ++i) {
            const int base = 19 + i * 4;
            p_data[base + 0] = 0.4;  // obs_x
            p_data[base + 1] = (i == 0) ? -0.1 : 0.1;  // obs_y
            p_data[base + 2] = 0.35;  // obs_z
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
            return result;
        }

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

private:
    void resetWarmStart(const Vec7& q, const Vec7& v) {
        Vec7 qk = q;
        Vec7 vk = v;
        const double dt = 0.02;

        for (int k = 0; k <= N_; ++k) {
            double x_init[PANDA_TASK_SPACE_NMPC_NX] = {0.0};
            for (int i = 0; i < 7; ++i) {
                x_init[i] = qk(i);
                x_init[i + 7] = vk(i);
            }
            ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "x", x_init);

            if (k < N_) {
                double u_init[PANDA_TASK_SPACE_NMPC_NU] = {0.0};
                ocp_nlp_out_set(nlp_config_, nlp_dims_, nlp_out_, nlp_in_, k, "u", u_init);
                qk = qk + dt * vk;
            }
        }
    }

    panda_task_space_nmpc_solver_capsule* capsule_ = nullptr;
    ocp_nlp_config* nlp_config_ = nullptr;
    ocp_nlp_dims*   nlp_dims_   = nullptr;
    ocp_nlp_in*     nlp_in_     = nullptr;
    ocp_nlp_out*    nlp_out_    = nullptr;
    int N_ = 0;

    std::array<double, PANDA_TASK_SPACE_NMPC_NY>  y_ref_{};
    std::array<double, PANDA_TASK_SPACE_NMPC_NYN> y_ref_e_{};
};

#endif  // PANDA_NMPC_CONTROLLER_HPP_
