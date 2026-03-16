#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <Eigen/Dense>

#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

#include "acados_solver_panda_task_space_nmpc.h"

#include "panda_nmpc_controller.hpp"
#include "panda_interfaces/msg/result_nmpc.hpp"

class nmpc_tau_node : public rclcpp::Node
{
private:
    PandaNMPCController nmpc_solver_;
    NMPCResult nmpc_res_last_;

    Eigen::Vector3d target_pos_;
    Eigen::Matrix3d target_rot_;
    Eigen::VectorXd reference_q_;

    Eigen::VectorXd jq_;
    Eigen::VectorXd jv_;

    bool joint_state_ready_{false};

    rclcpp::Publisher<panda_interfaces::msg::ResultNMPC>::SharedPtr nmpc_res_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

    std::vector<std::string> ordered_names_ = {
        "joint1", "joint2", "joint3", "joint4",
        "joint5", "joint6", "joint7"
    };

    void publish_result(const NMPCResult& res)
    {
        auto msg = panda_interfaces::msg::ResultNMPC();

        Eigen::Map<Eigen::Matrix<double, 7, 1>>(msg.q_ref.data())    = res.q_ref;
        Eigen::Map<Eigen::Matrix<double, 7, 1>>(msg.v_ref.data())    = res.v_ref;
        Eigen::Map<Eigen::Matrix<double, 7, 1>>(msg.a_ref.data())    = res.a_ref;
        Eigen::Map<Eigen::Matrix<double, 7, 1>>(msg.jerk_cmd.data()) = res.jerk_cmd;
        msg.status = res.status;

        nmpc_res_pub_->publish(msg);
    }

    void timer_effort_cmd_callback()
    {
        if (!joint_state_ready_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "joint_states 尚未准备好，跳过本次 NMPC 求解。");
            return;
        }

        // 调试阶段：不要把上一拍失败/预测的 a_ref 回灌成当前测量加速度
        Eigen::Matrix<double, 7, 1> current_a = Eigen::Matrix<double, 7, 1>::Zero();

        // auto nmpc_res = nmpc_solver_.NMPCSolve(
        //     target_pos_,
        //     target_rot_,
        //     reference_q_.head<7>(),
        //     jq_.head<7>(),
        //     jv_.head<7>(),
        //     current_a
        // );
        auto nmpc_res = nmpc_solver_.NMPCSolve(
            target_pos_,
            target_rot_,
            jq_.head<7>(),   // 先用当前关节角做 q_nom
            jq_.head<7>(),
            jv_.head<7>(),
            current_a
        );

        if (nmpc_res.status == 0) {
            nmpc_res_last_ = nmpc_res;     // 只有成功才覆盖 last good
            publish_result(nmpc_res);
        } else {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "NMPC solve failed, status=%d, keep last good command.",
                nmpc_res.status);

            publish_result(nmpc_res_last_); // 失败就保持上一次成功解
        }
    }

    void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
    {
        for (size_t i = 0; i < ordered_names_.size(); ++i) {
            auto it = std::find(msg->name.begin(), msg->name.end(), ordered_names_[i]);

            if (it != msg->name.end()) {
                int index = std::distance(msg->name.begin(), it);
                jq_[i] = msg->position[index];
                jv_[i] = msg->velocity[index];
            } else {
                RCLCPP_ERROR(this->get_logger(),
                             "Fatal Error: %s not found in joint_states!",
                             ordered_names_[i].c_str());
                return;
            }
        }
        
        joint_state_ready_ = true;
    }

public:
    nmpc_tau_node(const std::string& urdf_path, const std::string& ee_frame_name)
        : Node("nmpc_tau_node"),
          nmpc_solver_(urdf_path, ee_frame_name)
        //   target_pos_(0.2288, 0.0168, 0.2454)   // 这里改成你当前真正想要的目标
    {
        // target_rot_ = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix();

        target_pos_ << 0.29299, 0.00020, 0.41004;

        // 注意 Eigen::Quaterniond 的构造顺序是 (w, x, y, z)，不是 (x, y, z, w)
        Eigen::Quaterniond q_des(
            -0.00051,   // w
            0.99938,     // x
            -0.0377,     // y
            -0.00125     // z
        );
        q_des.normalize();
        target_rot_ = q_des.toRotationMatrix();

        reference_q_ = Eigen::VectorXd::Zero(ordered_names_.size());
        reference_q_ << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;

        jq_ = Eigen::VectorXd::Zero(ordered_names_.size());
        jv_ = Eigen::VectorXd::Zero(ordered_names_.size());

        // 初始化 last good command
        nmpc_res_last_.q_ref = reference_q_.head<7>();
        nmpc_res_last_.v_ref.setZero();
        nmpc_res_last_.a_ref.setZero();
        nmpc_res_last_.jerk_cmd.setZero();
        nmpc_res_last_.status = -1;

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
                this->joint_states_callback(msg);
            });

        nmpc_res_pub_ = this->create_publisher<panda_interfaces::msg::ResultNMPC>(
            "/nmpc_result", 1);

        // 和生成器里的 dt=0.02 对齐
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),
            [this]() { this->timer_effort_cmd_callback(); });
    }

    ~nmpc_tau_node() {}
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);

    const std::string urdfFilePath =
        "/home/cyh/panda_mujoco/franka_panda_urdf/robots/panda_arm_tau.urdf";
    const std::string urdfEEName = "ee_center_body";

    auto node = std::make_shared<nmpc_tau_node>(urdfFilePath, urdfEEName);

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}