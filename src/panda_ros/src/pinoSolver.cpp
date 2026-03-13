#include "pinoSolver.hpp"

pinoSolver::pinoSolver(const std::string& urdf_path, const std::string& ee_frame_name) {
    // 1. 加载 URDF 模型
    pinocchio::urdf::buildModel(urdf_path, model);
    data = pinocchio::Data(model);

    // 2. 获取末端执行器的 Frame ID
    if (model.existFrame(ee_frame_name)) {
        ee_frame_id = model.getFrameId(ee_frame_name);
    } else {
        throw std::runtime_error("End-effector frame not found in URDF!");
    }

    q_min = model.lowerPositionLimit.head(7);
    q_max = model.upperPositionLimit.head(7);
    q_mean = 0.5 * (q_max + q_min); // 关节舒适区中心

    std::cout << "joint min limit : " << q_min << std::endl;
    std::cout << "joint max limit : " << q_max << std::endl;
    std::cout << "joint mean : " << q_mean << std::endl;
}

Eigen::VectorXd pinoSolver::CLIKSolve(const Eigen::Vector3d& target_pos, 
                          const Eigen::Matrix3d& target_rot, 
                          const Eigen::VectorXd& q_init, 
                          bool& success) {
    
    // 构建目标的 SE3 位姿
    pinocchio::SE3 oMdes(target_rot, target_pos);
    Eigen::VectorXd q = q_init;

    pinocchio::Data::Matrix6x J(6, model.nv);
    J.setZero();

    success = false;

    for (int i = 0; i < IT_MAX; i++) {
        // 1. 正运动学更新所有关节和 Frame 的位姿
        pinocchio::forwardKinematics(model, data, q);
        pinocchio::updateFramePlacement(model, data, ee_frame_id);

        // 2. 计算当前位姿到目标位姿的误差 (在局部坐标系下)
        const pinocchio::SE3 iMd = data.oMf[ee_frame_id].actInv(oMdes);
        Eigen::VectorXd err = pinocchio::log6(iMd).toVector(); // 6维误差向量 [线速度, 角速度]

        // std::cout << "error twist: " << err << std::endl;

        // 3. 判断是否收敛
        if (err.norm() < eps) {
            // std::cout << "IK step complete! " << std::endl;
            success = true;
            break;
        }

        // 4. 计算末端执行器在局部坐标系下的雅可比矩阵
        pinocchio::computeFrameJacobian(model, data, q, ee_frame_id, pinocchio::LOCAL, J);

        Eigen::MatrixXd J_arm = J.leftCols(7);

        // 5. 阻尼最小二乘法求伪逆 (Damped Pseudo-inverse): J^T * (J * J^T + damp * I)^-1
        Eigen::Matrix<double, 6, 6> Jlog;
        pinocchio::Jlog6(iMd.inverse(), Jlog); // 将 Jlog6 的结果写入 Jlog 矩阵
        J_arm = -Jlog * J_arm;

        Eigen::MatrixXd JJt = J_arm * J_arm.transpose();
        JJt.diagonal().array() += damp; 
        Eigen::MatrixXd J_pinv = J_arm.transpose() * JJt.ldlt().solve(Eigen::MatrixXd::Identity(6, 6));

        // 6. 主任务速度：末端追踪
        Eigen::VectorXd v_task = -J_pinv * err;

        // 7. 次级任务：零空间计算 (避开奇异点和关节限位)
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(7, 7);
        Eigen::MatrixXd N = I - J_pinv * J_arm; // 零空间投影矩阵
        
        // 构造一个吸引力，把当前关节拉向中心点 q_mean
        Eigen::VectorXd v_null = -k_null * (q.head(7) - q_mean); 
        
        // 8. 融合速度
        Eigen::VectorXd v_arm = v_task + N * v_null;

        // 9. 还原维度
        Eigen::VectorXd v_full = Eigen::VectorXd::Zero(model.nv);
        v_full.head(7) = v_arm;

        // 10. 更新关节角
        q = pinocchio::integrate(model, q, v_full * DT);

        // 11. 【硬限位安全锁】防止步长过大导致的积分穿透
        q.head(7) = q.head(7).cwiseMax(q_min).cwiseMin(q_max);
    }

    return q;
}

Eigen::VectorXd pinoSolver::impedanceControlSolver(const Eigen::Vector3d& target_pos, 
                                                   const Eigen::Matrix3d& target_rot,
                                                   const Eigen::VectorXd& target_vel_twist,
                                                   const Eigen::VectorXd& target_acc_twist,
                                                   const Eigen::VectorXd& jq_curr, 
                                                   const Eigen::VectorXd& jv_curr,
                                                   const bool J_dot_enable) 
{
    pinocchio::computeAllTerms(model, data, jq_curr, jv_curr);
    pinocchio::computeJointJacobiansTimeVariation(model, data, jq_curr, jv_curr);
    pinocchio::updateFramePlacement(model, data, ee_frame_id);

    Eigen::VectorXd jq_arm_curr = jq_curr.head<7>();
    Eigen::VectorXd jv_arm_curr = jv_curr.head<7>();

    Eigen::MatrixXd J_full_lwa = pinocchio::getFrameJacobian(model, data, ee_frame_id, pinocchio::LOCAL_WORLD_ALIGNED);
    Eigen::MatrixXd J_full_dot_lwa = Eigen::MatrixXd::Zero(6, model.nv);;
    pinocchio::getFrameJacobianTimeVariation(model, data, ee_frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J_full_dot_lwa);
    Eigen::MatrixXd J_lwa = J_full_lwa.leftCols(7);
    Eigen::MatrixXd J_dot_lwa = J_full_dot_lwa.leftCols(7);
    Eigen::MatrixXd J_T_lwa = J_lwa.transpose();

    pinocchio::SE3 X_ee_sb = data.oMf[ee_frame_id];
    pinocchio::SE3 X_ee_sd(target_rot, target_pos);
    auto X_err_bd = X_ee_sb.actInv(X_ee_sd);
    Eigen::VectorXd X_err_b_6d = pinocchio::log6(X_err_bd).toVector();

    Eigen::VectorXd X_err_lwa = Eigen::VectorXd::Zero(6);
    // equal to X_err_lwa = T_lwa_to_b * X_err_b_6d
    X_err_lwa.head<3>() = X_ee_sb.rotation() * X_err_b_6d.head<3>();
    X_err_lwa.tail<3>() = X_ee_sb.rotation() * X_err_b_6d.tail<3>();

    Eigen::VectorXd V_ee_lwa = pinocchio::getFrameVelocity(model, data, ee_frame_id, pinocchio::LOCAL_WORLD_ALIGNED).toVector();
    Eigen::VectorXd V_err_lwa = target_vel_twist - V_ee_lwa;

    Eigen::MatrixXd M_arm = data.M.topLeftCorner<7, 7>();
    Eigen::MatrixXd M_inv = M_arm.inverse();
    Eigen::VectorXd nonlinear_terms = data.nle.head<7>();

    Eigen::MatrixXd lamda_per_pinv = J_lwa * M_inv * J_T_lwa;
    Eigen::MatrixXd lamda = (lamda_per_pinv + Eigen::Matrix<double, 6, 6>::Identity() * lamda_damp).inverse();;

    // set M_desire = lamda to avoid measuring F_ext
    Eigen::Matrix<double, 6, 6> C_d = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 6> K_d = Eigen::Matrix<double, 6, 6>::Zero();
    C_d.diagonal().head<3>().setConstant(desire_pos_damp);
    C_d.diagonal().tail<3>().setConstant(desire_rot_damp);
    K_d.diagonal().head<3>().setConstant(desire_pos_stiff);
    K_d.diagonal().tail<3>().setConstant(desire_rot_stiff);
    
    Eigen::VectorXd F_imp = C_d * V_err_lwa + K_d * X_err_lwa;

    // dynamics consistent pseudo-inverse
    Eigen::MatrixXd J_hat_lwa = M_inv * J_T_lwa * lamda;

    Eigen::VectorXd eta = -(J_dot_enable ? 1.0 : 0.0) * lamda * J_dot_lwa * jv_arm_curr + J_hat_lwa.transpose() * nonlinear_terms;
    Eigen::VectorXd F_comp = lamda * target_acc_twist + eta;

    Eigen::VectorXd tau_task = J_T_lwa * (F_imp + F_comp);

    // add null space torque ... 
    // dynamics consistent null space projetion matrix
    Eigen::MatrixXd N_dyn = Eigen::Matrix<double, 7, 7>::Identity() - J_hat_lwa * J_lwa;

    // 零空间下，我们一般不使用PD控制器，因为设定一个关节角度期望q_d既困难又不合理
    // 设计代价函数(势能函数)，并用梯度替换PD控制器的比例项，因为线性误差e=q_d-q本质上就是q_d以为中心(势能最低点)的二次型势能函数的梯度
    // 当然，在这里和PD控制器没有本质区别，但是可以推广到其他形式的势能函数
    Eigen::Matrix<double, 7, 1> grad_q = Eigen::Matrix<double, 7, 1>::Zero();
    for(int i = 0; i < 7; ++i) {
        grad_q(i) = (jq_curr(i) - q_mean(i)) / std::pow((q_max(i) - q_min(i)), 2);
    }

    const double k_null = 20.0;
    const double d_null = 2.0 * std::sqrt(k_null);

    // -k_null是因为我们要把正梯度变为负梯度，下降最快
    Eigen::Matrix<double, 7, 1> tau_null_desired = M_arm * (-0.0 * k_null * grad_q - d_null * jv_curr.head<7>());
    Eigen::Matrix<double, 7, 1> tau_null = N_dyn.transpose() * (tau_null_desired + nonlinear_terms);

    Eigen::Matrix<double, 7, 1> tau_arm = tau_task + tau_null;

    const Eigen::Matrix<double, 7, 1> tau_limit = (Eigen::Matrix<double, 7, 1>() << 87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0).finished();
    tau_arm = tau_arm.cwiseMax(-tau_limit).cwiseMin(tau_limit);

    Eigen::VectorXd tau_total(model.nv);
    tau_total.setZero();
    tau_total.head<7>() = tau_arm; // 仅填充前 7 个手臂关节，保留夹爪 0 力矩

    return tau_total;
}

Eigen::Vector2d pinoSolver::getYoshikawaManipulabilityMeasure(const Eigen::VectorXd& jq_curr) 
{
    pinocchio::computeJointJacobians(model, data, jq_curr);
    pinocchio::updateFramePlacements(model, data);

    Eigen::Matrix<double, 6, Eigen::Dynamic> J(6, model.nv);
    J.setZero();
    pinocchio::getFrameJacobian(model, data, ee_frame_id, pinocchio::LOCAL_WORLD_ALIGNED, J);

    Eigen::Matrix<double, 3, Eigen::Dynamic> J_v = J.topRows<3>();
    Eigen::Matrix<double, 3, Eigen::Dynamic> J_w = J.bottomRows<3>();

    Eigen::Vector2d mani_measure;

    double m_v = std::sqrt(std::max(0.0, (J_v * J_v.transpose()).determinant())); // m/s
    double m_w = std::sqrt(std::max(0.0, (J_w * J_w.transpose()).determinant())); // rad/s

    mani_measure << m_v, m_w;

    return mani_measure;
}