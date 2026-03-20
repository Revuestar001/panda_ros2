#include <iostream>
#include <Eigen/Dense>

#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/cholesky.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/spatial/explog.hpp>               // 提供 pinocchio::log6 和 pinocchio::Jlog6

class pinoSolver {
private:
    pinocchio::Model model;
    pinocchio::Data data;
    int ee_frame_id; // 末端执行器 (End-Effector) 的 Frame ID

    // IK 超参数
    const double eps = 1e-4;       // 收敛阈值
    const int IT_MAX = 2;       // 最大迭代次数
    const double DT = 1e-1;        // 步长 (Learning rate)
    const double damp = 1e-6;      // 阻尼系数，防止奇异点崩溃
    const double k_null = 1.0;             // 零空间优化增益，可调

    Eigen::VectorXd q_min;
    Eigen::VectorXd q_max;
    Eigen::VectorXd q_mean;

    // cartesian impedance Control 
    const double lamda_damp = 1e-6;
    const double desire_pos_damp = 50.0;
    const double desire_rot_damp = 5.0;
    const double desire_pos_stiff = 500.0;
    const double desire_rot_stiff = 50.0;

    // c-space impedance Control 
    const double desire_joints_damp = 63.0;
    const double desire_joints_stiff = 1000.0;

public:
    pinoSolver(const std::string& urdf_path, const std::string& ee_frame_name);

    /**
     * @brief 求解逆运动学
     * @param target_pos 目标位置 3x1
     * @param target_rot 目标旋转矩阵 3x3
     * @param q_init 初始关节角 (通常是当前机器人的实际关节角)
     * @param success 引用返回是否成功收敛
     * @return 求解出的目标关节角 q_target
     */
    Eigen::VectorXd CLIKSolve(const Eigen::Vector3d& target_pos, 
                          const Eigen::Matrix3d& target_rot, 
                          const Eigen::VectorXd& q_init, 
                          bool& success);

    template <int T>
    Eigen::Vector<double, T> cSpaceImpedanceControlSolver(const Eigen::Vector<double, T>& target_jq,
                                                                        const Eigen::Vector<double, T>& target_jv,
                                                                        const Eigen::Vector<double, T>& target_ja,
                                                                        const Eigen::Vector<double, T>& jq_curr, 
                                                                        const Eigen::Vector<double, T>& jv_curr);


    Eigen::VectorXd impedanceControlSolver(const Eigen::Vector3d& target_pos, 
                                                   const Eigen::Matrix3d& target_rot,
                                                   const Eigen::VectorXd& target_vel_twist,
                                                   const Eigen::VectorXd& target_acc_twist,
                                                   const Eigen::VectorXd& jq_curr, 
                                                   const Eigen::VectorXd& jv_curr,
                                                   const bool J_dot_enable);

    Eigen::Vector2d getYoshikawaManipulabilityMeasure(const Eigen::VectorXd& jq_curr);

    double getMinSingularValue(const Eigen::VectorXd& jq_curr);

    double getConditionNumber(const Eigen::VectorXd& jq_curr);

    int getNumJoints() const { return model.nv; }
    Eigen::VectorXd getJointsMeanValue() const { return this->q_mean; }
    pinocchio::Model& getPinocchioModel() { return model; }
    pinocchio::Data& getPinocchioData() { return data; }
};