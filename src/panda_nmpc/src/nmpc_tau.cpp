#include <algorithm>
#include <chrono>
#include <vector>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "panda_interfaces/msg/result_nmpc.hpp"
#include "panda_nmpc_controller.hpp"

class NMPCNode : public rclcpp::Node {
private:
    PandaNMPCController nmpc_solver_;
    NMPCResult nmpc_res_last_;

    Eigen::Vector3d target_pos_;
    Eigen::Matrix3d target_rot_;
    Eigen::VectorXd reference_q_;
    Eigen::VectorXd jq_;
    Eigen::VectorXd jv_;

    bool joint_state_ready_{false};

    double node_initial_time_;

    rclcpp::Publisher<panda_interfaces::msg::ResultNMPC>::SharedPtr nmpc_res_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

    std::vector<std::string> ordered_names_ = {
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"
    };

    void publish_result(const NMPCResult& res) {
        auto msg = panda_interfaces::msg::ResultNMPC();
        Eigen::Map<Eigen::Matrix<double, 7, 1>>(msg.q_ref.data()) = res.q_ref;
        Eigen::Map<Eigen::Matrix<double, 7, 1>>(msg.v_ref.data()) = res.v_ref;
        Eigen::Map<Eigen::Matrix<double, 7, 1>>(msg.a_ref.data()) = res.a_ref;
        Eigen::Map<Eigen::Matrix<double, 7, 1>>(msg.jerk_cmd.data()) = res.jerk_cmd;
        msg.status = res.status;
        nmpc_res_pub_->publish(msg);
    }

    void update_target() {
        double dt = this->now().seconds() - node_initial_time_;
        target_pos_[0] = -0.27 * std::cos(dt * 2 * M_PI / 5.0) + 0.47;
        target_pos_[1] = 0.0 * std::cos(dt * 2 * M_PI / 5.0) + 0.0;
        target_pos_[2] = 0.32;
    }

    void timer_callback() {
        if (!joint_state_ready_) {
            return;
        }

        update_target();

        auto res = nmpc_solver_.NMPCSolve(
            target_pos_,
            target_rot_,
            jq_.head<7>(),     // 先继续使用当前关节角作为 q_nom，避免零空间突然拉扯
            jq_.head<7>(),
            jv_.head<7>()
        );

        if (res.status == 0) {
            nmpc_res_last_ = res;
            publish_result(res);
        } else {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "NMPC solve failed, status=%d, keep last good command.", res.status);
            publish_result(nmpc_res_last_);
        }
    }

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
    }

public:
    NMPCNode() : Node("nmpc_tau_node") {
        // target_pos_ << 0.30, 0.20, 0.40;
        // target_pos_ << 0.15, 0.0, 0.25;
        target_pos_ << 0.74, 0.0, 0.32;
        target_rot_ = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()).toRotationMatrix();

        reference_q_ = Eigen::VectorXd::Zero(7);
        // reference_q_ << 0.0, -1.57, 0.785, -2.356, 0.0, 1.571, 0.785;
        reference_q_ << 0.008, -1.382, -0.008, -3.072, -0.007, 1.615, 0.792;
        jq_ = Eigen::VectorXd::Zero(7);
        jv_ = Eigen::VectorXd::Zero(7);

        nmpc_res_last_.q_ref = reference_q_.head<7>();
        nmpc_res_last_.v_ref.setZero();
        nmpc_res_last_.a_ref.setZero();
        nmpc_res_last_.jerk_cmd.setZero();
        nmpc_res_last_.status = -1;

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) { joint_states_callback(msg); });

        nmpc_res_pub_ = this->create_publisher<panda_interfaces::msg::ResultNMPC>("/nmpc_result", 1);
        timer_ = this->create_wall_timer(std::chrono::milliseconds(20), [this]() { timer_callback(); });

        node_initial_time_ = this->now().seconds();
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NMPCNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
