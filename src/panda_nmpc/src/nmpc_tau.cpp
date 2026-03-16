#include <rclcpp/rclcpp.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <Eigen/Dense>

// 引入 acados 核心底层接口
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
    Eigen::Vector<double, 6> target_vel_;
    Eigen::Vector<double, 6> target_acc_;
    Eigen::Matrix3d target_rot_;
    Eigen::VectorXd reference_q_;

    Eigen::VectorXd jq_;
    Eigen::VectorXd jv_;

    rclcpp::Publisher<panda_interfaces::msg::ResultNMPC>::SharedPtr nmpc_res_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

    std::vector<std::string> ordered_names_ = {
                "joint1", "joint2", "joint3", "joint4", 
                "joint5", "joint6", "joint7"
            };
    
    // 不推荐定时计算控制指令
    void timer_effort_cmd_callback() {
        auto nmpc_res = nmpc_solver_.NMPCSolve(target_pos_, target_rot_, reference_q_.head<7>(), jq_.head<7>(), jv_.head<7>(), nmpc_res_last_.a_ref.head<7>());

        auto msg = panda_interfaces::msg::ResultNMPC();
        Eigen::Map<Eigen::Vector<double, 7>> map_q_ref(msg.q_ref.data());
        map_q_ref = nmpc_res.q_ref;
        Eigen::Map<Eigen::Vector<double, 7>> map_v_ref(msg.v_ref.data());
        map_v_ref = nmpc_res.v_ref;
        Eigen::Map<Eigen::Vector<double, 7>> map_a_ref(msg.a_ref.data());
        map_a_ref = nmpc_res.a_ref;
        Eigen::Map<Eigen::Vector<double, 7>> map_jerk_cmd(msg.jerk_cmd.data());
        map_jerk_cmd = nmpc_res.jerk_cmd;
        msg.status = nmpc_res.status;

        nmpc_res_pub_->publish(msg);

        nmpc_res_last_ = nmpc_res;
    }

    void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        
        for (size_t i = 0; i < ordered_names_.size(); ++i) {
            auto it = std::find(msg->name.begin(), msg->name.end(), ordered_names_[i]);
            
            if (it != msg->name.end()) {
                int index = std::distance(msg->name.begin(), it);
                jq_[i] = msg->position[index];
                jv_[i] = msg->velocity[index];
            } else {
                RCLCPP_ERROR(this->get_logger(), "Fatal Error: %s not found in joint_states!", ordered_names_[i].c_str());
                return;
            }
        }

    }

public:
    nmpc_tau_node(const std::string& urdf_path, const std::string& ee_frame_name) : Node("nmpc_tau_node"),  
                                                                                    nmpc_solver_(),
                                                                                    target_pos_(0.2288, 0.0168, 0.2454) {
        target_rot_ = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix();
        target_vel_.setZero();
        target_acc_.setZero();

        reference_q_ = Eigen::VectorXd::Zero(ordered_names_.size());
        reference_q_ << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;
        RCLCPP_INFO_STREAM_ONCE(this->get_logger(), reference_q_);

        jq_ = Eigen::VectorXd::Zero(ordered_names_.size());
        jv_ = Eigen::VectorXd::Zero(ordered_names_.size());

        nmpc_res_last_.q_ref = Eigen::Vector<double, 7>::Zero();
        nmpc_res_last_.v_ref = Eigen::Vector<double, 7>::Zero();
        nmpc_res_last_.a_ref = Eigen::Vector<double, 7>::Zero();
        nmpc_res_last_.jerk_cmd = Eigen::Vector<double, 7>::Zero();
        nmpc_res_last_.status = -1;

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>("/joint_states", 10, [this] (const sensor_msgs::msg::JointState::SharedPtr msg) {this->joint_states_callback(msg);});
        nmpc_res_pub_ = this->create_publisher<panda_interfaces::msg::ResultNMPC>("/nmpc_result", 1);
        timer_ = this->create_wall_timer(std::chrono::milliseconds(10), [this] () {this->timer_effort_cmd_callback();});
    }

    ~nmpc_tau_node() {}
};



int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);

    const std::string urdfFilePath = "/home/cyh/panda_mujoco/franka_panda_urdf/robots/panda_arm_tau.urdf";
    const std::string urdfEEName = "ee_center_body";
    auto node = std::make_shared<nmpc_tau_node>(urdfFilePath, urdfEEName);

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}