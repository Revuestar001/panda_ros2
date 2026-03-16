#include <rclcpp/rclcpp.hpp>
// #include <mujoco/mujoco.h>
// #include <pinocchio/multibody/model.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <Eigen/Dense>

#include "pinoSolver.hpp"
#include "panda_interfaces/msg/result_nmpc.hpp"

class panda_test_node : public rclcpp::Node
{
private:

    pinoSolver solver_;

    Eigen::Vector3d target_pos_;
    Eigen::Vector<double, 6> target_vel_;
    Eigen::Vector<double, 6> target_acc_;
    Eigen::Matrix3d target_rot_;

    Eigen::VectorXd jq_;
    Eigen::VectorXd jv_;

    panda_interfaces::msg::ResultNMPC nmpc_result_;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr effort_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<panda_interfaces::msg::ResultNMPC>::SharedPtr nmpc_result_sub_;

    std::vector<std::string> ordered_names_ = {
                "joint1", "joint2", "joint3", "joint4", 
                "joint5", "joint6", "joint7"
            };
    
    // 不推荐定时计算控制指令
    void timer_effort_cmd_callback() {

        // Eigen::VectorXd torque = solver_.impedanceControlSolver(target_pos_, target_rot_, target_vel_, target_acc_, jq_, jv_, false);

        // auto msg = std_msgs::msg::Float64MultiArray();
        // for (size_t i = 0; i < ordered_names_.size(); ++i) {
        //     msg.data.push_back(torque[i]);
        // }

        // // RCLCPP_INFO(this->get_logger(), "Publish effort message");
        // effort_pub_->publish(msg);
        
        // auto mani_meas = solver_.getYoshikawaManipulabilityMeasure(jq_);
        // RCLCPP_INFO(this->get_logger(), "Yoshikawa Manipulability Measure : %lf, %lf", mani_meas[0], mani_meas[1]);

        // auto min_sin = solver_.getMinSingularValue(jq_);
        // RCLCPP_INFO(this->get_logger(), "Min Singular Value : %lf", min_sin);

        // auto condi_num = solver_.getConditionNumber(jq_);
        // RCLCPP_INFO(this->get_logger(), "Condition Number : %lf", condi_num);
        
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

        // Eigen::VectorXd torque = solver_.impedanceControlSolver(target_pos_, target_rot_, target_vel_, target_acc_, jq_, jv_, false);
        
        Eigen::Vector<double, 7> target_jq;
        target_jq = Eigen::Map<Eigen::Vector<double, 7>>(nmpc_result_.q_ref.data());
        // target_jq << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;

        Eigen::Vector<double, 7> target_jv;
        target_jv = Eigen::Map<Eigen::Vector<double, 7>>(nmpc_result_.v_ref.data());

        Eigen::Vector<double, 7> target_ja;
        target_ja = Eigen::Map<Eigen::Vector<double, 7>>(nmpc_result_.a_ref.data());

        auto torque = solver_.cSpaceImpedanceControlSolver<7>(target_jq, 
                                                            target_jq,
                                                            target_ja,
                                                            jq_.head<7>(),
                                                            jv_.head<7>());

        auto effort_msg = std_msgs::msg::Float64MultiArray();
        for (size_t i = 0; i < ordered_names_.size(); ++i) {
            effort_msg.data.push_back(torque[i]);
        }

        // RCLCPP_INFO(this->get_logger(), "Publish effort message");
        effort_pub_->publish(effort_msg);

    }

    void nmpc_result_callback(const panda_interfaces::msg::ResultNMPC::SharedPtr msg) {
        nmpc_result_ = *msg;
    }

public:
    panda_test_node(const std::string& urdf_path, const std::string& ee_frame_name) : Node("pinoSolver_node"), 
                                                                                    solver_(urdf_path, ee_frame_name), 
                                                                                    target_pos_(0.3, 0.2, 0.4) {
        target_rot_ = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix();
        target_vel_.setZero();
        target_acc_.setZero();

        jq_ = Eigen::VectorXd::Zero(ordered_names_.size());
        jv_ = Eigen::VectorXd::Zero(ordered_names_.size());
        
        Eigen::Vector<double, 7> reference_q;
        reference_q << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;
        for (size_t i = 0; i < 7; ++i) {
            nmpc_result_.q_ref.data()[i] = reference_q[i];
        }

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>("/joint_states", 10, [this] (const sensor_msgs::msg::JointState::SharedPtr msg) {this->joint_states_callback(msg);});
        nmpc_result_sub_ = this->create_subscription<panda_interfaces::msg::ResultNMPC>("/nmpc_result", 10, [this] (const panda_interfaces::msg::ResultNMPC::SharedPtr msg) {this->nmpc_result_callback(msg);});
        effort_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/effort_controller/commands", 1);
        timer_ = this->create_wall_timer(std::chrono::milliseconds(500), [this] () {this->timer_effort_cmd_callback();});
    }

    ~panda_test_node() {}
};



int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);

    const std::string urdfFilePath = "/home/cyh/panda_mujoco/franka_panda_urdf/robots/panda_arm_tau.urdf";
    const std::string urdfEEName = "ee_center_body";
    auto node = std::make_shared<panda_test_node>(urdfFilePath, urdfEEName);

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}