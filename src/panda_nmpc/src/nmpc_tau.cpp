#include <algorithm>
#include <cmath>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "panda_interfaces/msg/result_nmpc.hpp"
#include "panda_nmpc_controller.hpp"
#include "second_order_reference_regulator.hpp"

class NMPCNode : public rclcpp::Node {
private:
    using Vec7 = Eigen::Matrix<double, 7, 1>;
    using Vec3 = Eigen::Vector3d;
    using Mat3 = Eigen::Matrix3d;

    PandaNMPCController nmpc_solver_;
    NMPCResult nmpc_res_last_;

    SecondOrderReferenceRegulator sorr_pos_;
    SecondOrderReferenceRegulator sorr_rot_; 

    Eigen::Vector3d target_pos_;
    Eigen::Matrix3d target_rot_;
    Eigen::VectorXd reference_q_;
    Eigen::VectorXd jq_;
    Eigen::VectorXd jv_;

    bool joint_state_ready_{false};
    bool sorr_ready_{false};

    double node_initial_time_;

    pinocchio::Model pin_model_;
    std::unique_ptr<pinocchio::Data> pin_data_;
    pinocchio::FrameIndex ee_frame_id_{0};

    rclcpp::Publisher<panda_interfaces::msg::ResultNMPC>::SharedPtr nmpc_res_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

    std::vector<std::string> ordered_names_ = {
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"
    };

    void initialize_pinocchio() {
        const std::string panda_ros_share = "/home/cyh/panda_ros2/src/panda_ros";
        const std::string urdf_path = panda_ros_share + "/model/panda_tau_sim.urdf";

        pinocchio::urdf::buildModel(urdf_path, pin_model_);
        pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);

        if (!pin_model_.existFrame("ee_center_body")) {
            throw std::runtime_error("ee_center_body frame not found in Panda URDF.");
        }
        ee_frame_id_ = pin_model_.getFrameId("ee_center_body");
    }

    void set_sorr_config() {
        SecondOrderReferenceRegulator::Config sorr_pos_config;
        sorr_pos_config.dt = nmpc_solver_.getDt();
        sorr_pos_config.damping_ratio = 1.0;
        sorr_pos_config.natural_frequency = 10.0;
        sorr_pos_config.max_linear_velocity = 1.0;
        sorr_pos_config.max_linear_acceleration = 10.0;
        sorr_pos_.setConfig(sorr_pos_config);

        SecondOrderReferenceRegulator::Config sorr_rot_config;
        sorr_rot_config.dt = nmpc_solver_.getDt();
        sorr_rot_config.damping_ratio = 1.0;
        sorr_rot_config.natural_frequency = 10.0;
        sorr_rot_config.max_angular_velocity = 0.7845;
        sorr_rot_config.max_angular_acceleration = 0.7845;
        sorr_rot_.setConfig(sorr_rot_config);
    }

    void compute_current_ee_pose(Vec3& ee_pos, Mat3& ee_rot) {
        if (!pin_data_) {
            throw std::runtime_error("Pinocchio data is not initialized.");
        }

        const Vec7 q_curr = jq_.head<7>();
        pinocchio::forwardKinematics(pin_model_, *pin_data_, q_curr);
        pinocchio::updateFramePlacements(pin_model_, *pin_data_);

        const pinocchio::SE3& oMf_ee = pin_data_->oMf[ee_frame_id_];
        ee_pos = oMf_ee.translation();
        ee_rot = oMf_ee.rotation();
    }

    void initialize_sorr_from_current_ee_pose() {
        Vec3 ee_pos = Vec3::Zero();
        Mat3 ee_rot = Mat3::Identity();
        compute_current_ee_pose(ee_pos, ee_rot);

        // 用当前实际末端状态初始化 SORR，避免第一次阶跃直接穿透到目标。
        sorr_pos_.resetPosition(ee_pos);
        sorr_rot_.resetOrientation(Eigen::Quaterniond(ee_rot));
        sorr_ready_ = true;
        node_initial_time_ = this->now().seconds();

        RCLCPP_INFO(this->get_logger(), "SORR initialized from current EE pose.");
    }

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
        if (!joint_state_ready_ || !sorr_ready_) {
            return;
        }

        update_target();

        const Vec3 filtered_target_pos = sorr_pos_.updatePosition(target_pos_);
        const Eigen::Quaterniond filtered_target_quat = sorr_rot_.updateOrientation(target_rot_);
        const Mat3 filtered_target_rot = filtered_target_quat.toRotationMatrix();

        const Vec7 q_nom = jq_.head<7>();
        const Vec7 current_q = jq_.head<7>();
        const Vec7 current_v = jv_.head<7>();

        auto res = nmpc_solver_.NMPCSolve(
            filtered_target_pos,
            filtered_target_rot,
            q_nom,     // 先继续使用当前关节角作为 q_nom，避免零空间突然拉扯
            sorr_pos_.getLinearVelocity(),
            sorr_rot_.getAngularVelocity(),
            current_q,
            current_v
        );
        // auto res = nmpc_solver_.NMPCSolve(
        //     target_pos_,
        //     target_rot_,
        //     q_nom,     // 先继续使用当前关节角作为 q_nom，避免零空间突然拉扯
        //     sorr_pos_.getLinearVelocity(),
        //     sorr_rot_.getAngularVelocity(),
        //     current_q,
        //     current_v
        // );

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

        if (!sorr_ready_) {
            initialize_sorr_from_current_ee_pose();
        }
    }

public:
    NMPCNode() : Node("nmpc_tau_node") {
        initialize_pinocchio();

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

        // SORR
        set_sorr_config();

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) { joint_states_callback(msg); });

        nmpc_res_pub_ = this->create_publisher<panda_interfaces::msg::ResultNMPC>("/nmpc_result", 1);
        timer_ = this->create_wall_timer(std::chrono::milliseconds(static_cast<int64_t>(nmpc_solver_.getDt() * 1000)), [this]() { timer_callback(); });

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
