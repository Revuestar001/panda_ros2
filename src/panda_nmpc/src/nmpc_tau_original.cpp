#include <algorithm>
#include <chrono>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "panda_interfaces/msg/result_nmpc.hpp"
#include "panda_nmpc_controller.hpp"
#include "second_order_reference_regulator.hpp"

class NMPCOriginalNode : public rclcpp::Node {
private:
    using Vec7 = Eigen::Matrix<double, 7, 1>;
    using Vec3 = Eigen::Vector3d;
    using Mat3 = Eigen::Matrix3d;

    PandaNMPCController nmpc_solver_;
    NMPCResult nmpc_res_last_;

    SecondOrderReferenceRegulator sorr_pos_;
    SecondOrderReferenceRegulator sorr_rot_;

    Vec3 target_pos_{Vec3::Zero()};
    Mat3 target_rot_{Mat3::Identity()};
    Eigen::VectorXd reference_q_;
    Eigen::VectorXd jq_;
    Eigen::VectorXd jv_;

    std::string urdf_path_;
    std::string ee_frame_name_;
    std::string joint_states_topic_;
    std::string nmpc_result_topic_;
    std::string target_pose_topic_;

    double sorr_pos_damping_ratio_{1.0};
    double sorr_pos_natural_frequency_{10.0};
    double sorr_pos_max_linear_velocity_{1.0};
    double sorr_pos_max_linear_acceleration_{10.0};

    double sorr_rot_damping_ratio_{1.0};
    double sorr_rot_natural_frequency_{10.0};
    double sorr_rot_max_angular_velocity_{0.7845};
    double sorr_rot_max_angular_acceleration_{0.7845};

    bool joint_state_ready_{false};
    bool sorr_ready_{false};
    double node_initial_time_{0.0};

    pinocchio::Model pin_model_;
    std::unique_ptr<pinocchio::Data> pin_data_;
    pinocchio::FrameIndex ee_frame_id_{0};

    rclcpp::Publisher<panda_interfaces::msg::ResultNMPC>::SharedPtr nmpc_res_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;

    std::vector<std::string> ordered_names_ = {
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"
    };

    static std::string default_panda_urdf_path() {
        try {
            return ament_index_cpp::get_package_share_directory("panda_ros") + "/model/panda_tau_sim.urdf";
        } catch (const std::exception&) {
            return "/home/cyh/panda_ros2/src/panda_ros/model/panda_tau_sim.urdf";
        }
    }

    void declare_startup_parameters() {
        urdf_path_ = this->declare_parameter<std::string>("urdf_path", default_panda_urdf_path());
        ee_frame_name_ = this->declare_parameter<std::string>("ee_frame_name", "ee_center_body");
        joint_states_topic_ = this->declare_parameter<std::string>("joint_states_topic", "/joint_states");
        nmpc_result_topic_ = this->declare_parameter<std::string>("nmpc_result_topic", "/nmpc_result");
        target_pose_topic_ = this->declare_parameter<std::string>("target_pose_topic", "/global_target_pose");

        target_pos_.x() = this->declare_parameter<double>("target_x", 0.25);
        target_pos_.y() = this->declare_parameter<double>("target_y", 0.3);
        target_pos_.z() = this->declare_parameter<double>("target_z", 0.25);

        const double target_qx = this->declare_parameter<double>("target_qx", 1.0);
        const double target_qy = this->declare_parameter<double>("target_qy", 0.0);
        const double target_qz = this->declare_parameter<double>("target_qz", 0.0);
        const double target_qw = this->declare_parameter<double>("target_qw", 0.0);
        const Eigen::Quaterniond target_quat(target_qw, target_qx, target_qy, target_qz);
        target_rot_ = target_quat.normalized().toRotationMatrix();

        sorr_pos_damping_ratio_ = this->declare_parameter<double>("sorr_pos_damping_ratio", 1.0);
        sorr_pos_natural_frequency_ = this->declare_parameter<double>("sorr_pos_natural_frequency", 10.0);
        sorr_pos_max_linear_velocity_ =
            this->declare_parameter<double>("sorr_pos_max_linear_velocity", 1.0);
        sorr_pos_max_linear_acceleration_ =
            this->declare_parameter<double>("sorr_pos_max_linear_acceleration", 10.0);

        sorr_rot_damping_ratio_ = this->declare_parameter<double>("sorr_rot_damping_ratio", 1.0);
        sorr_rot_natural_frequency_ = this->declare_parameter<double>("sorr_rot_natural_frequency", 10.0);
        sorr_rot_max_angular_velocity_ =
            this->declare_parameter<double>("sorr_rot_max_angular_velocity", 0.7845);
        sorr_rot_max_angular_acceleration_ =
            this->declare_parameter<double>("sorr_rot_max_angular_acceleration", 0.7845);
    }

    void initialize_pinocchio() {
        pinocchio::urdf::buildModel(urdf_path_, pin_model_);
        pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);

        if (!pin_model_.existFrame(ee_frame_name_)) {
            throw std::runtime_error(ee_frame_name_ + " frame not found in Panda URDF.");
        }
        ee_frame_id_ = pin_model_.getFrameId(ee_frame_name_);
    }

    void set_sorr_config() {
        SecondOrderReferenceRegulator::Config sorr_pos_config;
        sorr_pos_config.dt = nmpc_solver_.getDt();
        sorr_pos_config.damping_ratio = sorr_pos_damping_ratio_;
        sorr_pos_config.natural_frequency = sorr_pos_natural_frequency_;
        sorr_pos_config.max_linear_velocity = sorr_pos_max_linear_velocity_;
        sorr_pos_config.max_linear_acceleration = sorr_pos_max_linear_acceleration_;
        sorr_pos_.setConfig(sorr_pos_config);

        SecondOrderReferenceRegulator::Config sorr_rot_config;
        sorr_rot_config.dt = nmpc_solver_.getDt();
        sorr_rot_config.damping_ratio = sorr_rot_damping_ratio_;
        sorr_rot_config.natural_frequency = sorr_rot_natural_frequency_;
        sorr_rot_config.max_angular_velocity = sorr_rot_max_angular_velocity_;
        sorr_rot_config.max_angular_acceleration = sorr_rot_max_angular_acceleration_;
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

    void timer_callback() {
        if (!joint_state_ready_ || !sorr_ready_) {
            return;
        }

        const Vec7 current_q = jq_.head<7>();
        const Vec7 current_v = jv_.head<7>();
        const Vec7 q_nom = current_q;
        const Vec7 dq_nom = Vec7::Zero();
        const Vec7 ddq_nom = Vec7::Zero();
        const Vec7 current_a = Vec7::Zero();

        const Vec3 filtered_target_pos = sorr_pos_.updatePosition(target_pos_);
        const Eigen::Quaterniond filtered_target_quat = sorr_rot_.updateOrientation(target_rot_);
        const Mat3 filtered_target_rot = filtered_target_quat.toRotationMatrix();

        auto res = nmpc_solver_.NMPCSolveSingleTarget(
            filtered_target_pos,
            filtered_target_rot,
            q_nom,
            dq_nom,
            ddq_nom,
            sorr_pos_.getLinearVelocity(),
            sorr_rot_.getAngularVelocity(),
            current_q,
            current_v,
            current_a
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

    void target_pose_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        target_pos_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;

        Eigen::Quaterniond target_quat(
            msg->pose.orientation.w,
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z);
        if (target_quat.norm() < 1.0e-9) {
            RCLCPP_WARN(this->get_logger(), "Received invalid target pose quaternion, ignore orientation update.");
            return;
        }
        target_rot_ = target_quat.normalized().toRotationMatrix();
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
    NMPCOriginalNode() : Node("nmpc_tau_original_node") {
        declare_startup_parameters();
        initialize_pinocchio();

        reference_q_ = Eigen::VectorXd::Zero(7);
        reference_q_ << 0.008, -1.382, -0.008, -3.072, -0.007, 1.615, 0.792;
        jq_ = Eigen::VectorXd::Zero(7);
        jv_ = Eigen::VectorXd::Zero(7);

        nmpc_res_last_.q_ref = reference_q_.head<7>();
        nmpc_res_last_.v_ref.setZero();
        nmpc_res_last_.a_ref.setZero();
        nmpc_res_last_.jerk_cmd.setZero();
        nmpc_res_last_.status = -1;

        set_sorr_config();

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            joint_states_topic_, 10,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) { joint_states_callback(msg); });

        target_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            target_pose_topic_, 10,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { target_pose_callback(msg); });

        nmpc_res_pub_ = this->create_publisher<panda_interfaces::msg::ResultNMPC>(nmpc_result_topic_, 1);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int64_t>(nmpc_solver_.getDt() * 1000)),
            [this]() { timer_callback(); });

        node_initial_time_ = this->now().seconds();

        RCLCPP_INFO(
            this->get_logger(),
            "Original NMPC node configured. urdf=%s, ee_frame=%s, joint_states_topic=%s, target_pose_topic=%s, result_topic=%s",
            urdf_path_.c_str(),
            ee_frame_name_.c_str(),
            joint_states_topic_.c_str(),
            target_pose_topic_.c_str(),
            nmpc_result_topic_.c_str());
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NMPCOriginalNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
