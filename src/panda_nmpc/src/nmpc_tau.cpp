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

class nmpc_tau_node : public rclcpp::Node
{
private:

    PandaNMPCController nmpc_solver_;

    Eigen::Vector3d target_pos_;
    Eigen::Vector<double, 6> target_vel_;
    Eigen::Vector<double, 6> target_acc_;
    Eigen::Matrix3d target_rot_;

    Eigen::VectorXd jq_;
    Eigen::VectorXd jv_;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr effort_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

    std::vector<std::string> ordered_names_ = {
                "joint1", "joint2", "joint3", "joint4", 
                "joint5", "joint6", "joint7"
            };
    
    // 不推荐定时计算控制指令
    void timer_effort_cmd_callback() {
        Eigen::VectorXd torque = nmpc_solver_.NMPCSolve(target_pos_, target_vel_, target_acc_, target_rot_, jq_, jv_);
        auto effort_msg = std_msgs::msg::Float64MultiArray();
        for (size_t i = 0; i < ordered_names_.size(); ++i) {
            effort_msg.data.push_back(torque[i]);
        }

        // RCLCPP_INFO(this->get_logger(), "Publish effort message");
        effort_pub_->publish(effort_msg);
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
                                                                                    target_pos_(0.3, 0.2, 0.4) {
        target_rot_ = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix();
        target_vel_.setZero();
        target_acc_.setZero();

        jq_ = Eigen::VectorXd::Zero(ordered_names_.size());
        jv_ = Eigen::VectorXd::Zero(ordered_names_.size());

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>("/joint_states", 10, [this] (const sensor_msgs::msg::JointState::SharedPtr msg) {this->joint_states_callback(msg);});
        effort_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/effort_controller/commands", 1);
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