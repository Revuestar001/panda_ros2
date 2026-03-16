#include <algorithm>
#include <chrono>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "panda_interfaces/msg/result_nmpc.hpp"
#include "pinoSolver.hpp"

class PandaControlNode : public rclcpp::Node {
private:
    pinoSolver solver_;
    Eigen::VectorXd jq_;
    Eigen::VectorXd jv_;
    bool joint_state_ready_{false};

    panda_interfaces::msg::ResultNMPC nmpc_result_;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr effort_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<panda_interfaces::msg::ResultNMPC>::SharedPtr nmpc_result_sub_;

    std::vector<std::string> ordered_names_ = {
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"
    };

    void joint_states_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        for (size_t i = 0; i < ordered_names_.size(); ++i) {
            auto it = std::find(msg->name.begin(), msg->name.end(), ordered_names_[i]);
            if (it == msg->name.end()) {
                RCLCPP_ERROR(this->get_logger(), "Joint %s not found in joint_states", ordered_names_[i].c_str());
                return;
            }
            const int index = static_cast<int>(std::distance(msg->name.begin(), it));
            jq_[i] = msg->position[index];
            jv_[i] = msg->velocity[index];
        }
        joint_state_ready_ = true;

        Eigen::Matrix<double, 7, 1> target_jq = Eigen::Map<Eigen::Matrix<double, 7, 1>>(nmpc_result_.q_ref.data());
        Eigen::Matrix<double, 7, 1> target_jv = Eigen::Map<Eigen::Matrix<double, 7, 1>>(nmpc_result_.v_ref.data());
        Eigen::Matrix<double, 7, 1> target_ja = Eigen::Map<Eigen::Matrix<double, 7, 1>>(nmpc_result_.a_ref.data());

        auto torque = solver_.cSpaceImpedanceControlSolver<7>(
            target_jq,
            target_jv,
            target_ja,
            jq_.head<7>(),
            jv_.head<7>());
        // auto torque = solver_.impedanceControlSolver(
        //     Eigen::Vector3d(0.15, 0.0, 0.25),
        //     Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix(),
        //     Eigen::Vector<double, 6>::Zero(),
        //     Eigen::Vector<double, 6>::Zero(),
        //     jq_.head<7>(),
        //     jv_.head<7>(),
        //     false);

        auto effort_msg = std_msgs::msg::Float64MultiArray();
        for (int i = 0; i < 7; ++i) {
            effort_msg.data.push_back(torque[i]);
        }
        effort_pub_->publish(effort_msg);
    }

    void nmpc_result_callback(const panda_interfaces::msg::ResultNMPC::SharedPtr msg) {
        nmpc_result_ = *msg;
    }

public:
    PandaControlNode(const std::string& urdf_path, const std::string& ee_frame_name)
        : Node("pinoSolver_node"), solver_(urdf_path, ee_frame_name) {
        jq_ = Eigen::VectorXd::Zero(7);
        jv_ = Eigen::VectorXd::Zero(7);

        Eigen::Matrix<double, 7, 1> reference_q;
        // reference_q << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;
        reference_q << 0.008, -1.382, -0.008, -3.072, -0.007, 1.615, 0.792;
        for (int i = 0; i < 7; ++i) {
            nmpc_result_.q_ref.data()[i] = reference_q[i];
            nmpc_result_.v_ref.data()[i] = 0.0;
            nmpc_result_.a_ref.data()[i] = 0.0;
            nmpc_result_.jerk_cmd.data()[i] = 0.0;
        }
        nmpc_result_.status = -1;

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) { joint_states_callback(msg); });
        nmpc_result_sub_ = this->create_subscription<panda_interfaces::msg::ResultNMPC>(
            "/nmpc_result", 10,
            [this](const panda_interfaces::msg::ResultNMPC::SharedPtr msg) { nmpc_result_callback(msg); });
        effort_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("/effort_controller/commands", 1);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    const std::string urdfFilePath = "/home/cyh/panda_ros2/src/panda_ros/model/panda_tau_sim.urdf";
    const std::string urdfEEName = "ee_center_body";
    auto node = std::make_shared<PandaControlNode>(urdfFilePath, urdfEEName);
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
