#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "panda_interfaces/msg/result_nmpc.hpp"
#include "panda_nmpc_controller.hpp"
#include "second_order_reference_regulator.hpp"

class NMPCNode : public rclcpp::Node {
private:
    using Vec7 = Eigen::Matrix<double, 7, 1>;
    using Vec3 = Eigen::Vector3d;
    using Mat3 = Eigen::Matrix3d;
    using Quat = Eigen::Quaterniond;

    PandaNMPCController nmpc_solver_;
    NMPCResult last_valid_result_;

    SecondOrderReferenceRegulator sorr_pos_;
    SecondOrderReferenceRegulator sorr_rot_;

    Vec3 target_pos_{Vec3::Zero()};
    Mat3 target_rot_{Mat3::Identity()};

    Vec7 q_meas_{Vec7::Zero()};
    Vec7 dq_meas_{Vec7::Zero()};
    Vec7 ddq_est_{Vec7::Zero()};
    Vec7 q_nominal_{Vec7::Zero()};
    Vec7 dq_nominal_{Vec7::Zero()};
    Vec7 ddq_nominal_{Vec7::Zero()};

    Vec7 prev_dq_meas_{Vec7::Zero()};
    bool has_prev_joint_sample_{false};
    rclcpp::Time prev_joint_stamp_{0, 0, RCL_ROS_TIME};

    bool joint_state_ready_{false};
    bool sorr_ready_{false};
    bool nominal_initialized_from_state_{false};
    int consecutive_failure_count_{0};

    std::string urdf_path_;
    std::string obstacle_config_path_;
    std::string scene_xml_path_;
    std::string ee_frame_name_;
    std::string joint_states_topic_;
    std::string joint_trajectory_topic_;
    std::string nmpc_result_topic_;
    std::string target_pose_topic_;

    bool use_global_trajectory_{false};
    bool use_current_q_as_nominal_on_startup_{true};
    int max_consecutive_failures_before_hold_{2};

    double ddq_estimator_cutoff_hz_{25.0};
    double ddq_estimator_raw_clip_scale_{1.5};
    double max_joint_state_dt_{0.05};
    double min_joint_state_dt_{1.0e-4};

    double sorr_pos_damping_ratio_{1.0};
    double sorr_pos_natural_frequency_{8.0};
    double sorr_pos_max_linear_velocity_{0.3};
    double sorr_pos_max_linear_acceleration_{1.5};
    double sorr_rot_damping_ratio_{1.0};
    double sorr_rot_natural_frequency_{8.0};
    double sorr_rot_max_angular_velocity_{0.8};
    double sorr_rot_max_angular_acceleration_{2.0};

    pinocchio::Model pin_model_;
    std::unique_ptr<pinocchio::Data> pin_data_;
    pinocchio::FrameIndex ee_frame_id_{0};

    rclcpp::Publisher<panda_interfaces::msg::ResultNMPC>::SharedPtr nmpc_res_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::vector<std::string> ordered_names_{
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"
    };

    static std::string default_panda_urdf_path() {
        try {
            return ament_index_cpp::get_package_share_directory("panda_ros") + "/model/panda_tau_sim.urdf";
        } catch (const std::exception&) {
            return "/home/cyh/panda_ros2/src/panda_ros/model/panda_tau_sim.urdf";
        }
    }

    static std::string default_scene_xml_path() {
        return "/home/cyh/panda_ros2/model/franka_emika_panda/scene_tau_ros.xml";
    }

    static std::string default_obstacle_config_path() {
        return "/home/cyh/panda_ros2/model/franka_emika_panda/static_sphere_obstacles.xml";
    }

    template <std::size_t N>
    static std::vector<double> toParameterVector(const std::array<double, N>& values) {
        return std::vector<double>(values.begin(), values.end());
    }

    template <std::size_t N>
    std::array<double, N> declareFixedSizeArrayParameter(
        const std::string& param_name,
        const std::array<double, N>& defaults) {
        const auto raw = this->declare_parameter<std::vector<double>>(param_name, toParameterVector(defaults));
        if (raw.size() != N) {
            throw std::runtime_error(
                "Parameter '" + param_name + "' must contain exactly " + std::to_string(N) + " values.");
        }
        std::array<double, N> values{};
        std::copy(raw.begin(), raw.end(), values.begin());
        return values;
    }

    static PandaNMPCController::SoftLimitMode parseSoftLimitMode(const std::string& value) {
        if (value == "hard" || value == "hard_only") {
            return PandaNMPCController::SoftLimitMode::kHardOnly;
        }
        if (value == "soft") {
            return PandaNMPCController::SoftLimitMode::kSoft;
        }
        throw std::runtime_error(
            "Unsupported soft-limit mode '" + value + "'. Use 'hard' or 'soft'.");
    }

    void declareStartupParameters() {
        urdf_path_ = this->declare_parameter<std::string>("urdf_path", default_panda_urdf_path());
        obstacle_config_path_ =
            this->declare_parameter<std::string>("obstacle_config_path", default_obstacle_config_path());
        scene_xml_path_ = this->declare_parameter<std::string>("scene_xml_path", default_scene_xml_path());
        ee_frame_name_ = this->declare_parameter<std::string>("ee_frame_name", "ee_center_body");

        joint_states_topic_ = this->declare_parameter<std::string>("joint_states_topic", "/joint_states");
        joint_trajectory_topic_ =
            this->declare_parameter<std::string>("joint_trajectory_topic", "/global_joint_trajectory");
        nmpc_result_topic_ = this->declare_parameter<std::string>("nmpc_result_topic", "/nmpc_result");
        target_pose_topic_ = this->declare_parameter<std::string>("target_pose_topic", "/global_target_pose");

        target_pos_.x() = this->declare_parameter<double>("target_x", 0.55);
        target_pos_.y() = this->declare_parameter<double>("target_y", 0.0);
        target_pos_.z() = this->declare_parameter<double>("target_z", 0.35);
        const double target_qx = this->declare_parameter<double>("target_qx", 1.0);
        const double target_qy = this->declare_parameter<double>("target_qy", 0.0);
        const double target_qz = this->declare_parameter<double>("target_qz", 0.0);
        const double target_qw = this->declare_parameter<double>("target_qw", 0.0);
        const Quat quat(target_qw, target_qx, target_qy, target_qz);
        if (quat.norm() < 1.0e-9) {
            throw std::runtime_error("Initial target quaternion is invalid.");
        }
        target_rot_ = quat.normalized().toRotationMatrix();

        use_global_trajectory_ = this->declare_parameter<bool>("use_global_trajectory", false);
        use_current_q_as_nominal_on_startup_ =
            this->declare_parameter<bool>("use_current_q_as_nominal_on_startup", true);
        max_consecutive_failures_before_hold_ =
            this->declare_parameter<int>("max_consecutive_failures_before_hold", 2);

        ddq_estimator_cutoff_hz_ = this->declare_parameter<double>("ddq_estimator_cutoff_hz", 25.0);
        ddq_estimator_raw_clip_scale_ = this->declare_parameter<double>("ddq_estimator_raw_clip_scale", 1.5);
        max_joint_state_dt_ = this->declare_parameter<double>("max_joint_state_dt", 0.05);
        min_joint_state_dt_ = this->declare_parameter<double>("min_joint_state_dt", 1.0e-4);

        sorr_pos_damping_ratio_ = this->declare_parameter<double>("sorr_pos_damping_ratio", 1.0);
        sorr_pos_natural_frequency_ = this->declare_parameter<double>("sorr_pos_natural_frequency", 8.0);
        sorr_pos_max_linear_velocity_ = this->declare_parameter<double>("sorr_pos_max_linear_velocity", 0.3);
        sorr_pos_max_linear_acceleration_ = this->declare_parameter<double>("sorr_pos_max_linear_acceleration", 1.5);
        sorr_rot_damping_ratio_ = this->declare_parameter<double>("sorr_rot_damping_ratio", 1.0);
        sorr_rot_natural_frequency_ = this->declare_parameter<double>("sorr_rot_natural_frequency", 8.0);
        sorr_rot_max_angular_velocity_ = this->declare_parameter<double>("sorr_rot_max_angular_velocity", 0.8);
        sorr_rot_max_angular_acceleration_ = this->declare_parameter<double>("sorr_rot_max_angular_acceleration", 2.0);

        const auto default_weights = PandaNMPCController::defaultCostWeights();
        PandaNMPCController::CostWeights weights;
        weights.pos = declareFixedSizeArrayParameter("nmpc_weights.stage.pos", default_weights.pos);
        weights.rot = declareFixedSizeArrayParameter("nmpc_weights.stage.rot", default_weights.rot);
        weights.ee_lin_vel = declareFixedSizeArrayParameter("nmpc_weights.stage.ee_lin_vel", default_weights.ee_lin_vel);
        weights.ee_ang_vel = declareFixedSizeArrayParameter("nmpc_weights.stage.ee_ang_vel", default_weights.ee_ang_vel);
        weights.q_reg = declareFixedSizeArrayParameter("nmpc_weights.stage.q_reg", default_weights.q_reg);
        weights.dq_reg = declareFixedSizeArrayParameter("nmpc_weights.stage.dq_reg", default_weights.dq_reg);
        weights.ddq_reg = declareFixedSizeArrayParameter("nmpc_weights.stage.ddq_reg", default_weights.ddq_reg);
        weights.jerk_reg = declareFixedSizeArrayParameter("nmpc_weights.stage.jerk_reg", default_weights.jerk_reg);
        weights.pos_e = declareFixedSizeArrayParameter("nmpc_weights.terminal.pos", default_weights.pos_e);
        weights.rot_e = declareFixedSizeArrayParameter("nmpc_weights.terminal.rot", default_weights.rot_e);
        weights.ee_lin_vel_e = declareFixedSizeArrayParameter("nmpc_weights.terminal.ee_lin_vel", default_weights.ee_lin_vel_e);
        weights.ee_ang_vel_e = declareFixedSizeArrayParameter("nmpc_weights.terminal.ee_ang_vel", default_weights.ee_ang_vel_e);
        weights.q_reg_e = declareFixedSizeArrayParameter("nmpc_weights.terminal.q_reg", default_weights.q_reg_e);
        weights.dq_reg_e = declareFixedSizeArrayParameter("nmpc_weights.terminal.dq_reg", default_weights.dq_reg_e);
        weights.ddq_reg_e = declareFixedSizeArrayParameter("nmpc_weights.terminal.ddq_reg", default_weights.ddq_reg_e);
        nmpc_solver_.setCostWeights(weights);

        const auto default_constraints = PandaNMPCController::defaultConstraintConfig();
        PandaNMPCController::ConstraintConfig constraints;
        constraints.hard_limits.q_lower = declareFixedSizeArrayParameter("limits.hard.q_lower", default_constraints.hard_limits.q_lower);
        constraints.hard_limits.q_upper = declareFixedSizeArrayParameter("limits.hard.q_upper", default_constraints.hard_limits.q_upper);
        constraints.hard_limits.dq_abs = declareFixedSizeArrayParameter("limits.hard.dq_abs", default_constraints.hard_limits.dq_abs);
        constraints.hard_limits.ddq_abs = declareFixedSizeArrayParameter("limits.hard.ddq_abs", default_constraints.hard_limits.ddq_abs);
        constraints.hard_limits.jerk_abs = declareFixedSizeArrayParameter("limits.hard.jerk_abs", default_constraints.hard_limits.jerk_abs);

        constraints.soft_bounds.q_lower = declareFixedSizeArrayParameter("limits.soft.q_lower", default_constraints.soft_bounds.q_lower);
        constraints.soft_bounds.q_upper = declareFixedSizeArrayParameter("limits.soft.q_upper", default_constraints.soft_bounds.q_upper);
        constraints.soft_bounds.dq_abs = declareFixedSizeArrayParameter("limits.soft.dq_abs", default_constraints.soft_bounds.dq_abs);
        constraints.soft_bounds.ddq_abs = declareFixedSizeArrayParameter("limits.soft.ddq_abs", default_constraints.soft_bounds.ddq_abs);
        constraints.soft_bounds.jerk_abs = declareFixedSizeArrayParameter("limits.soft.jerk_abs", default_constraints.soft_bounds.jerk_abs);

        constraints.q_mode = parseSoftLimitMode(this->declare_parameter<std::string>("constraint_modes.q", "soft"));
        constraints.dq_mode = parseSoftLimitMode(this->declare_parameter<std::string>("constraint_modes.dq", "soft"));
        constraints.ddq_mode = parseSoftLimitMode(this->declare_parameter<std::string>("constraint_modes.ddq", "hard"));
        constraints.jerk_mode = parseSoftLimitMode(this->declare_parameter<std::string>("constraint_modes.jerk", "hard"));

        constraints.stage_penalty.slack_linear =
            this->declare_parameter<double>("soft_constraint.stage.slack_linear", default_constraints.stage_penalty.slack_linear);
        constraints.stage_penalty.slack_quadratic =
            this->declare_parameter<double>("soft_constraint.stage.slack_quadratic", default_constraints.stage_penalty.slack_quadratic);
        constraints.terminal_penalty.slack_linear =
            this->declare_parameter<double>("soft_constraint.terminal.slack_linear", default_constraints.terminal_penalty.slack_linear);
        constraints.terminal_penalty.slack_quadratic =
            this->declare_parameter<double>("soft_constraint.terminal.slack_quadratic", default_constraints.terminal_penalty.slack_quadratic);
        nmpc_solver_.setConstraintConfig(constraints);

        const std::array<double, 7> default_q_nominal{{0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0}};
        q_nominal_ = toEigen(declareFixedSizeArrayParameter("nominal_posture.q", default_q_nominal));
        dq_nominal_ = toEigen(declareFixedSizeArrayParameter("nominal_posture.dq", std::array<double, 7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
        ddq_nominal_ = toEigen(declareFixedSizeArrayParameter("nominal_posture.ddq", std::array<double, 7>{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));
    }

    static Vec7 toEigen(const std::array<double, 7>& values) {
        Vec7 out;
        for (int i = 0; i < 7; ++i) {
            out(i) = values[static_cast<std::size_t>(i)];
        }
        return out;
    }

    void initializePinocchio() {
        pinocchio::urdf::buildModel(urdf_path_, pin_model_);
        pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);
        if (!pin_model_.existFrame(ee_frame_name_)) {
            throw std::runtime_error("EE frame '" + ee_frame_name_ + "' not found in URDF.");
        }
        ee_frame_id_ = pin_model_.getFrameId(ee_frame_name_);
    }

    void setSorrConfig() {
        SecondOrderReferenceRegulator::Config pos_cfg;
        pos_cfg.dt = nmpc_solver_.getDt();
        pos_cfg.damping_ratio = sorr_pos_damping_ratio_;
        pos_cfg.natural_frequency = sorr_pos_natural_frequency_;
        pos_cfg.max_linear_velocity = sorr_pos_max_linear_velocity_;
        pos_cfg.max_linear_acceleration = sorr_pos_max_linear_acceleration_;
        sorr_pos_.setConfig(pos_cfg);

        SecondOrderReferenceRegulator::Config rot_cfg;
        rot_cfg.dt = nmpc_solver_.getDt();
        rot_cfg.damping_ratio = sorr_rot_damping_ratio_;
        rot_cfg.natural_frequency = sorr_rot_natural_frequency_;
        rot_cfg.max_angular_velocity = sorr_rot_max_angular_velocity_;
        rot_cfg.max_angular_acceleration = sorr_rot_max_angular_acceleration_;
        sorr_rot_.setConfig(rot_cfg);
    }

    void computeCurrentEePose(Vec3& ee_pos, Mat3& ee_rot) {
        pinocchio::forwardKinematics(pin_model_, *pin_data_, q_meas_);
        pinocchio::updateFramePlacements(pin_model_, *pin_data_);
        const auto& oMf = pin_data_->oMf[ee_frame_id_];
        ee_pos = oMf.translation();
        ee_rot = oMf.rotation();
    }

    void initializeSorrFromCurrentEePose() {
        Vec3 ee_pos = Vec3::Zero();
        Mat3 ee_rot = Mat3::Identity();
        computeCurrentEePose(ee_pos, ee_rot);
        sorr_pos_.resetPosition(ee_pos);
        sorr_rot_.resetOrientation(Quat(ee_rot));
        sorr_ready_ = true;
        RCLCPP_INFO(this->get_logger(), "SORR initialized from current EE pose.");
    }

    void publishResult(const NMPCResult& result) {
        panda_interfaces::msg::ResultNMPC msg;
        Eigen::Map<Vec7>(msg.q_ref.data()) = result.q_ref;
        Eigen::Map<Vec7>(msg.v_ref.data()) = result.v_ref;
        Eigen::Map<Vec7>(msg.a_ref.data()) = result.a_ref;
        Eigen::Map<Vec7>(msg.jerk_cmd.data()) = result.jerk_cmd;
        msg.status = result.status;
        nmpc_res_pub_->publish(msg);
    }

    static constexpr double kPi = 3.14159265358979323846;

    static double clampPositive(double value, double lower, double upper) {
        return std::clamp(value, lower, upper);
    }

    double computeJointStateDt(const sensor_msgs::msg::JointState& msg) {
        rclcpp::Time stamp = msg.header.stamp;
        if (stamp.nanoseconds() == 0) {
            stamp = this->now();
        }
        double dt = nmpc_solver_.getDt();
        if (has_prev_joint_sample_) {
            dt = (stamp - prev_joint_stamp_).seconds();
        }
        prev_joint_stamp_ = stamp;
        return clampPositive(dt, min_joint_state_dt_, max_joint_state_dt_);
    }

    void updateAccelerationEstimate(double dt) {
        if (!has_prev_joint_sample_) {
            ddq_est_.setZero();
            prev_dq_meas_ = dq_meas_;
            has_prev_joint_sample_ = true;
            return;
        }

        Vec7 raw_ddq = (dq_meas_ - prev_dq_meas_) / dt;
        const auto& hard_limits = nmpc_solver_.getConstraintConfig().hard_limits.ddq_abs;
        for (int i = 0; i < 7; ++i) {
            const double clip = ddq_estimator_raw_clip_scale_ * hard_limits[static_cast<std::size_t>(i)];
            raw_ddq(i) = std::clamp(raw_ddq(i), -clip, clip);
        }

        const double tau = 1.0 / std::max(2.0 * kPi * ddq_estimator_cutoff_hz_, 1.0e-6);
        const double alpha = std::clamp(dt / (tau + dt), 0.0, 1.0);
        ddq_est_ = (1.0 - alpha) * ddq_est_ + alpha * raw_ddq;
        prev_dq_meas_ = dq_meas_;
    }

    void timerCallback() {
        if (!joint_state_ready_ || !sorr_ready_) {
            return;
        }

        const Vec3 filtered_target_pos = sorr_pos_.updatePosition(target_pos_);
        const Quat filtered_target_quat = sorr_rot_.updateOrientation(target_rot_);
        const Mat3 filtered_target_rot = filtered_target_quat.toRotationMatrix();

        NMPCResult result = nmpc_solver_.NMPCSolveSingleTarget(
            filtered_target_pos,
            filtered_target_rot,
            q_nominal_,
            dq_nominal_,
            ddq_nominal_,
            sorr_pos_.getLinearVelocity(),
            sorr_rot_.getAngularVelocity(),
            q_meas_,
            dq_meas_,
            ddq_est_);

        if (result.status == 0) {
            consecutive_failure_count_ = 0;
            last_valid_result_ = result;
            publishResult(result);
            return;
        }

        ++consecutive_failure_count_;
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 1000,
            "NMPC solve failed, status=%d, consecutive_failures=%d",
            result.status, consecutive_failure_count_);

        if (consecutive_failure_count_ <= max_consecutive_failures_before_hold_ && last_valid_result_.status == 0) {
            publishResult(last_valid_result_);
            return;
        }

        publishResult(buildSafeHoldResult());
    }

    NMPCResult buildSafeHoldResult() const {
        NMPCResult result;
        result.q_ref = q_meas_;
        result.v_ref.setZero();
        result.a_ref.setZero();
        result.jerk_cmd.setZero();
        result.status = -999;
        return result;
    }

    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        if (msg->position.size() < ordered_names_.size()) {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "joint_states position size %zu < %zu",
                msg->position.size(), ordered_names_.size());
            return;
        }
        if (msg->velocity.size() < ordered_names_.size()) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "joint_states velocity size %zu < %zu, missing joints are treated as zero velocity.",
                msg->velocity.size(), ordered_names_.size());
        }

        for (std::size_t i = 0; i < ordered_names_.size(); ++i) {
            const auto it = std::find(msg->name.begin(), msg->name.end(), ordered_names_[i]);
            if (it == msg->name.end()) {
                RCLCPP_ERROR_THROTTLE(
                    this->get_logger(), *this->get_clock(), 1000,
                    "Joint %s not found in joint_states", ordered_names_[i].c_str());
                return;
            }
            const std::size_t idx = static_cast<std::size_t>(std::distance(msg->name.begin(), it));
            q_meas_(static_cast<Eigen::Index>(i)) = msg->position[idx];
            dq_meas_(static_cast<Eigen::Index>(i)) = (idx < msg->velocity.size()) ? msg->velocity[idx] : 0.0;
        }

        const double dt = computeJointStateDt(*msg);
        updateAccelerationEstimate(dt);
        joint_state_ready_ = true;

        if (use_current_q_as_nominal_on_startup_ && !nominal_initialized_from_state_) {
            q_nominal_ = q_meas_;
            nominal_initialized_from_state_ = true;
        }

        if (!sorr_ready_) {
            initializeSorrFromCurrentEePose();
        }
    }

    void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        target_pos_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
        const Quat quat(
            msg->pose.orientation.w,
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z);
        if (quat.norm() < 1.0e-9) {
            RCLCPP_WARN(this->get_logger(), "Received invalid target quaternion, ignore orientation update.");
            return;
        }
        target_rot_ = quat.normalized().toRotationMatrix();
    }

    void jointTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr /*msg*/) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "This NMPC build is single-pose only. /global_joint_trajectory is intentionally ignored.");
    }

public:
    NMPCNode() : Node("nmpc_tau_node") {
        declareStartupParameters();
        initializePinocchio();
        setSorrConfig();

        last_valid_result_.status = -1;
        last_valid_result_.q_ref = q_nominal_;
        last_valid_result_.v_ref.setZero();
        last_valid_result_.a_ref.setZero();
        last_valid_result_.jerk_cmd.setZero();

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            joint_states_topic_, 10,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) { jointStateCallback(msg); });

        joint_trajectory_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            joint_trajectory_topic_, 10,
            [this](const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) { jointTrajectoryCallback(msg); });

        target_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            target_pose_topic_, 10,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { targetPoseCallback(msg); });

        nmpc_res_pub_ = this->create_publisher<panda_interfaces::msg::ResultNMPC>(nmpc_result_topic_, 1);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int64_t>(1000.0 * nmpc_solver_.getDt())),
            [this]() { timerCallback(); });

        RCLCPP_INFO(
            this->get_logger(),
            "NMPC single-pose node started. urdf=%s, ee_frame=%s, joint_states_topic=%s, target_pose_topic=%s, result_topic=%s, joint_trajectory_topic=%s (ignored), use_global_trajectory=%s (ignored), obstacle_config=%s (unused), scene_xml=%s (unused)",
            urdf_path_.c_str(),
            ee_frame_name_.c_str(),
            joint_states_topic_.c_str(),
            target_pose_topic_.c_str(),
            nmpc_result_topic_.c_str(),
            joint_trajectory_topic_.c_str(),
            use_global_trajectory_ ? "true" : "false",
            obstacle_config_path_.c_str(),
            scene_xml_path_.c_str());
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NMPCNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
