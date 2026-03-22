#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
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
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "panda_interfaces/msg/result_nmpc.hpp"
#include "panda_nmpc_controller.hpp"
#include "second_order_reference_regulator.hpp"
#include "static_sphere_scene.hpp"

class NMPCNode : public rclcpp::Node {
private:
    using Vec7 = Eigen::Matrix<double, 7, 1>;
    using Vec3 = Eigen::Vector3d;
    using Mat3 = Eigen::Matrix3d;
    using Quat = Eigen::Quaterniond;

    struct NodeParameters {
        std::string urdf_path;
        std::string obstacle_config_path;
        std::string scene_xml_path;
        std::string ee_frame_name;

        std::string joint_states_topic;
        std::string joint_trajectory_topic;
        std::string nmpc_result_topic;
        std::string target_pose_topic;

        Vec3 target_pos{Vec3::Zero()};
        Mat3 target_rot{Mat3::Identity()};

        bool use_global_trajectory{false};
        bool use_current_q_as_nominal_on_startup{true};
        int max_consecutive_failures_before_hold{2};

        double sorr_pos_damping_ratio{1.0};
        double sorr_pos_natural_frequency{10.0};
        double sorr_pos_max_linear_velocity{0.1};
        double sorr_pos_max_linear_acceleration{0.5};

        double sorr_rot_damping_ratio{1.0};
        double sorr_rot_natural_frequency{10.0};
        double sorr_rot_max_angular_velocity{0.07845};
        double sorr_rot_max_angular_acceleration{0.7845};

        PandaNMPCController::CostWeights cost_weights{PandaNMPCController::defaultCostWeights()};
        PandaNMPCController::HardLimits hard_limits{PandaNMPCController::defaultHardLimits()};
        PandaNMPCController::ObstacleConstraintConfig obstacle_constraint{
            PandaNMPCController::defaultObstacleConstraintConfig()};
    };

    PandaNMPCController nmpc_solver_;
    NodeParameters params_;
    NMPCResult last_valid_result_;

    SecondOrderReferenceRegulator sorr_pos_;
    SecondOrderReferenceRegulator sorr_rot_;

    Vec3 target_pos_{Vec3::Zero()};
    Mat3 target_rot_{Mat3::Identity()};
    Vec7 q_meas_{Vec7::Zero()};
    Vec7 dq_meas_{Vec7::Zero()};
    Vec7 q_nominal_{Vec7::Zero()};

    bool joint_state_ready_{false};
    bool sorr_ready_{false};
    bool nominal_initialized_from_state_{false};
    int consecutive_failure_count_{0};

    pinocchio::Model pin_model_;
    std::unique_ptr<pinocchio::Data> pin_data_;
    pinocchio::FrameIndex ee_frame_id_{0};

    std::vector<panda_nmpc::StaticSphereObstacle> scene_obstacles_;
    PandaNMPCController::ObstacleParamBlock runtime_obstacle_params_{
        PandaNMPCController::disabledObstacleParams()};

    rclcpp::Publisher<panda_interfaces::msg::ResultNMPC>::SharedPtr nmpc_res_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::array<std::string, 7> ordered_names_{{
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"
    }};

    static std::string defaultPandaUrdfPath() {
        try {
            return ament_index_cpp::get_package_share_directory("panda_ros") + "/model/panda_tau_sim.urdf";
        } catch (const std::exception&) {
            return "/home/cyh/panda_ros2/src/panda_ros/model/panda_tau_sim.urdf";
        }
    }

    static std::string defaultSceneXmlPath() {
        return "/home/cyh/panda_ros2/model/franka_emika_panda/scene_tau_ros.xml";
    }

    static std::string defaultObstacleConfigPath() {
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
        const auto raw =
            this->declare_parameter<std::vector<double>>(param_name, toParameterVector(defaults));
        if (raw.size() != N) {
            throw std::runtime_error(
                "Parameter '" + param_name + "' must contain exactly " + std::to_string(N) + " values.");
        }

        std::array<double, N> values{};
        std::copy(raw.begin(), raw.end(), values.begin());
        return values;
    }

    static Vec7 toEigen(const std::array<double, 7>& values) {
        Vec7 result;
        for (int i = 0; i < 7; ++i) {
            result(i) = values[static_cast<std::size_t>(i)];
        }
        return result;
    }

    void declareStartupParameters() {
        params_.urdf_path = this->declare_parameter<std::string>("urdf_path", defaultPandaUrdfPath());
        params_.obstacle_config_path =
            this->declare_parameter<std::string>("obstacle_config_path", defaultObstacleConfigPath());
        params_.scene_xml_path =
            this->declare_parameter<std::string>("scene_xml_path", defaultSceneXmlPath());
        params_.ee_frame_name = this->declare_parameter<std::string>("ee_frame_name", "ee_center_body");

        params_.joint_states_topic =
            this->declare_parameter<std::string>("joint_states_topic", "/joint_states");
        params_.joint_trajectory_topic =
            this->declare_parameter<std::string>("joint_trajectory_topic", "/global_joint_trajectory");
        params_.nmpc_result_topic =
            this->declare_parameter<std::string>("nmpc_result_topic", "/nmpc_result");
        params_.target_pose_topic =
            this->declare_parameter<std::string>("target_pose_topic", "/global_target_pose");

        params_.target_pos.x() = this->declare_parameter<double>("target_x", 0.6);
        params_.target_pos.y() = this->declare_parameter<double>("target_y", 0.0);
        params_.target_pos.z() = this->declare_parameter<double>("target_z", 0.5);

        const double target_qx = this->declare_parameter<double>("target_qx", 1.0);
        const double target_qy = this->declare_parameter<double>("target_qy", 0.0);
        const double target_qz = this->declare_parameter<double>("target_qz", 0.0);
        const double target_qw = this->declare_parameter<double>("target_qw", 0.0);
        const Quat target_quat(target_qw, target_qx, target_qy, target_qz);
        if (target_quat.norm() < 1.0e-9) {
            throw std::runtime_error("Initial target quaternion is invalid.");
        }
        params_.target_rot = target_quat.normalized().toRotationMatrix();

        params_.use_global_trajectory = this->declare_parameter<bool>("use_global_trajectory", false);
        params_.use_current_q_as_nominal_on_startup =
            this->declare_parameter<bool>("use_current_q_as_nominal_on_startup", true);
        params_.max_consecutive_failures_before_hold =
            this->declare_parameter<int>("max_consecutive_failures_before_hold", 2);

        params_.sorr_pos_damping_ratio =
            this->declare_parameter<double>("sorr_pos_damping_ratio", params_.sorr_pos_damping_ratio);
        params_.sorr_pos_natural_frequency =
            this->declare_parameter<double>("sorr_pos_natural_frequency", params_.sorr_pos_natural_frequency);
        params_.sorr_pos_max_linear_velocity =
            this->declare_parameter<double>("sorr_pos_max_linear_velocity", params_.sorr_pos_max_linear_velocity);
        params_.sorr_pos_max_linear_acceleration =
            this->declare_parameter<double>("sorr_pos_max_linear_acceleration", params_.sorr_pos_max_linear_acceleration);

        params_.sorr_rot_damping_ratio =
            this->declare_parameter<double>("sorr_rot_damping_ratio", params_.sorr_rot_damping_ratio);
        params_.sorr_rot_natural_frequency =
            this->declare_parameter<double>("sorr_rot_natural_frequency", params_.sorr_rot_natural_frequency);
        params_.sorr_rot_max_angular_velocity =
            this->declare_parameter<double>("sorr_rot_max_angular_velocity", params_.sorr_rot_max_angular_velocity);
        params_.sorr_rot_max_angular_acceleration =
            this->declare_parameter<double>("sorr_rot_max_angular_acceleration", params_.sorr_rot_max_angular_acceleration);

        const auto default_weights = PandaNMPCController::defaultCostWeights();
        params_.cost_weights.pos = declareFixedSizeArrayParameter("nmpc_weights.stage.pos", default_weights.pos);
        params_.cost_weights.rot = declareFixedSizeArrayParameter("nmpc_weights.stage.rot", default_weights.rot);
        params_.cost_weights.ee_lin_vel =
            declareFixedSizeArrayParameter("nmpc_weights.stage.ee_lin_vel", default_weights.ee_lin_vel);
        params_.cost_weights.ee_ang_vel =
            declareFixedSizeArrayParameter("nmpc_weights.stage.ee_ang_vel", default_weights.ee_ang_vel);
        params_.cost_weights.q_reg =
            declareFixedSizeArrayParameter("nmpc_weights.stage.q_reg", default_weights.q_reg);
        params_.cost_weights.dq_reg =
            declareFixedSizeArrayParameter("nmpc_weights.stage.dq_reg", default_weights.dq_reg);
        params_.cost_weights.ddq_reg =
            declareFixedSizeArrayParameter("nmpc_weights.stage.ddq_reg", default_weights.ddq_reg);

        params_.cost_weights.pos_e =
            declareFixedSizeArrayParameter("nmpc_weights.terminal.pos", default_weights.pos_e);
        params_.cost_weights.rot_e =
            declareFixedSizeArrayParameter("nmpc_weights.terminal.rot", default_weights.rot_e);
        params_.cost_weights.ee_lin_vel_e =
            declareFixedSizeArrayParameter("nmpc_weights.terminal.ee_lin_vel", default_weights.ee_lin_vel_e);
        params_.cost_weights.ee_ang_vel_e =
            declareFixedSizeArrayParameter("nmpc_weights.terminal.ee_ang_vel", default_weights.ee_ang_vel_e);
        params_.cost_weights.q_reg_e =
            declareFixedSizeArrayParameter("nmpc_weights.terminal.q_reg", default_weights.q_reg_e);
        params_.cost_weights.dq_reg_e =
            declareFixedSizeArrayParameter("nmpc_weights.terminal.dq_reg", default_weights.dq_reg_e);

        const auto default_hard_limits = PandaNMPCController::defaultHardLimits();
        params_.hard_limits.q_lower =
            declareFixedSizeArrayParameter("limits.hard.q_lower", default_hard_limits.q_lower);
        params_.hard_limits.q_upper =
            declareFixedSizeArrayParameter("limits.hard.q_upper", default_hard_limits.q_upper);
        params_.hard_limits.dq_abs =
            declareFixedSizeArrayParameter("limits.hard.dq_abs", default_hard_limits.dq_abs);
        params_.hard_limits.ddq_abs =
            declareFixedSizeArrayParameter("limits.hard.ddq_abs", default_hard_limits.ddq_abs);

        const auto default_obstacle_constraint =
            PandaNMPCController::defaultObstacleConstraintConfig();
        params_.obstacle_constraint.stage_penalty.slack_linear = this->declare_parameter<double>(
            "obstacle_constraint.stage.slack_linear",
            default_obstacle_constraint.stage_penalty.slack_linear);
        params_.obstacle_constraint.stage_penalty.slack_quadratic = this->declare_parameter<double>(
            "obstacle_constraint.stage.slack_quadratic",
            default_obstacle_constraint.stage_penalty.slack_quadratic);
        params_.obstacle_constraint.terminal_penalty.slack_linear = this->declare_parameter<double>(
            "obstacle_constraint.terminal.slack_linear",
            default_obstacle_constraint.terminal_penalty.slack_linear);
        params_.obstacle_constraint.terminal_penalty.slack_quadratic = this->declare_parameter<double>(
            "obstacle_constraint.terminal.slack_quadratic",
            default_obstacle_constraint.terminal_penalty.slack_quadratic);

        const std::array<double, 7> default_q_nominal{{0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0}};
        q_nominal_ = toEigen(declareFixedSizeArrayParameter("nominal_posture.q", default_q_nominal));
    }

    std::string selectObstacleSourcePath() const {
        if (!params_.obstacle_config_path.empty() &&
            std::ifstream(params_.obstacle_config_path).good()) {
            return params_.obstacle_config_path;
        }
        return params_.scene_xml_path;
    }

    void loadStaticObstacles() {
        runtime_obstacle_params_ = PandaNMPCController::disabledObstacleParams();
        const std::string obstacle_source_path = selectObstacleSourcePath();
        scene_obstacles_ = panda_nmpc::loadStaticSphereObstaclesFromSceneXml(obstacle_source_path);

        if (scene_obstacles_.empty()) {
            RCLCPP_WARN(
                this->get_logger(),
                "No static sphere obstacles found in '%s'. Obstacle constraints are disabled.",
                obstacle_source_path.c_str());
            return;
        }

        if (scene_obstacles_.size() > PandaNMPCController::kNumRuntimeObstacles) {
            RCLCPP_WARN(
                this->get_logger(),
                "Obstacle config defines %zu obstacle(s) but the solver supports %zu. Extra obstacles will be ignored.",
                scene_obstacles_.size(),
                PandaNMPCController::kNumRuntimeObstacles);
        }

        const std::size_t obstacle_count =
            std::min(scene_obstacles_.size(), PandaNMPCController::kNumRuntimeObstacles);
        for (std::size_t obstacle_idx = 0; obstacle_idx < obstacle_count; ++obstacle_idx) {
            const auto& obstacle = scene_obstacles_[obstacle_idx];
            const std::size_t offset = obstacle_idx * 4;
            runtime_obstacle_params_[offset + 0] = obstacle.center.x();
            runtime_obstacle_params_[offset + 1] = obstacle.center.y();
            runtime_obstacle_params_[offset + 2] = obstacle.center.z();
            runtime_obstacle_params_[offset + 3] = obstacle.radius;
        }

        std::ostringstream stream;
        stream << "Loaded " << scene_obstacles_.size()
               << " obstacle(s) from '" << obstacle_source_path
               << "'. Applied " << obstacle_count << " obstacle(s): ";
        for (std::size_t obstacle_idx = 0; obstacle_idx < obstacle_count; ++obstacle_idx) {
            const auto& obstacle = scene_obstacles_[obstacle_idx];
            if (obstacle_idx > 0) {
                stream << "; ";
            }
            stream << "[" << obstacle_idx << "] "
                   << obstacle.name << " pos=("
                   << obstacle.center.x() << ", "
                   << obstacle.center.y() << ", "
                   << obstacle.center.z() << ") radius=" << obstacle.radius;
        }
        RCLCPP_INFO(this->get_logger(), "%s", stream.str().c_str());
    }

    void initializePinocchio() {
        pinocchio::urdf::buildModel(params_.urdf_path, pin_model_);
        pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);

        if (!pin_model_.existFrame(params_.ee_frame_name)) {
            throw std::runtime_error("EE frame '" + params_.ee_frame_name + "' not found in URDF.");
        }
        ee_frame_id_ = pin_model_.getFrameId(params_.ee_frame_name);
    }

    void setSorrConfig() {
        SecondOrderReferenceRegulator::Config pos_config;
        pos_config.dt = nmpc_solver_.getDt();
        pos_config.damping_ratio = params_.sorr_pos_damping_ratio;
        pos_config.natural_frequency = params_.sorr_pos_natural_frequency;
        pos_config.max_linear_velocity = params_.sorr_pos_max_linear_velocity;
        pos_config.max_linear_acceleration = params_.sorr_pos_max_linear_acceleration;
        sorr_pos_.setConfig(pos_config);

        SecondOrderReferenceRegulator::Config rot_config;
        rot_config.dt = nmpc_solver_.getDt();
        rot_config.damping_ratio = params_.sorr_rot_damping_ratio;
        rot_config.natural_frequency = params_.sorr_rot_natural_frequency;
        rot_config.max_angular_velocity = params_.sorr_rot_max_angular_velocity;
        rot_config.max_angular_acceleration = params_.sorr_rot_max_angular_acceleration;
        sorr_rot_.setConfig(rot_config);
    }

    void computeCurrentEePose(Vec3& ee_pos, Mat3& ee_rot) {
        pinocchio::forwardKinematics(pin_model_, *pin_data_, q_meas_);
        pinocchio::updateFramePlacements(pin_model_, *pin_data_);
        const auto& transform = pin_data_->oMf[ee_frame_id_];
        ee_pos = transform.translation();
        ee_rot = transform.rotation();
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

    NMPCResult buildSafeHoldResult() const {
        NMPCResult result;
        result.q_ref = q_meas_;
        result.v_ref.setZero();
        result.a_ref.setZero();
        result.jerk_cmd.setZero();
        result.status = -999;
        return result;
    }

    void timerCallback() {
        if (!joint_state_ready_ || !sorr_ready_) {
            return;
        }

        const Vec3 filtered_target_pos = sorr_pos_.updatePosition(target_pos_);
        const Quat filtered_target_quat = sorr_rot_.updateOrientation(target_rot_);
        const Mat3 filtered_target_rot = filtered_target_quat.toRotationMatrix();

        const NMPCResult result = nmpc_solver_.NMPCSolveSingleTarget(
            filtered_target_pos,
            filtered_target_rot,
            q_nominal_,
            sorr_pos_.getLinearVelocity(),
            sorr_rot_.getAngularVelocity(),
            runtime_obstacle_params_,
            q_meas_,
            dq_meas_);

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

        if (consecutive_failure_count_ <= params_.max_consecutive_failures_before_hold &&
            last_valid_result_.status == 0) {
            publishResult(last_valid_result_);
            return;
        }

        publishResult(buildSafeHoldResult());
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

        joint_state_ready_ = true;

        if (params_.use_current_q_as_nominal_on_startup && !nominal_initialized_from_state_) {
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
            RCLCPP_WARN(this->get_logger(), "Received invalid target quaternion, ignoring orientation update.");
            return;
        }
        target_rot_ = quat.normalized().toRotationMatrix();
    }

    void jointTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr /*msg*/) {
        RCLCPP_WARN_THROTTLE(
            this->get_logger(), *this->get_clock(), 2000,
            "This NMPC node is single-pose only. /global_joint_trajectory is intentionally ignored.");
    }

public:
    NMPCNode() : Node("nmpc_tau_node") {
        declareStartupParameters();

        nmpc_solver_.setCostWeights(params_.cost_weights);
        nmpc_solver_.setHardLimits(params_.hard_limits);
        nmpc_solver_.setObstacleConstraintConfig(params_.obstacle_constraint);

        target_pos_ = params_.target_pos;
        target_rot_ = params_.target_rot;

        initializePinocchio();
        loadStaticObstacles();
        setSorrConfig();

        last_valid_result_.status = -1;
        last_valid_result_.q_ref = q_nominal_;
        last_valid_result_.v_ref.setZero();
        last_valid_result_.a_ref.setZero();
        last_valid_result_.jerk_cmd.setZero();

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            params_.joint_states_topic, 10,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) { jointStateCallback(msg); });

        joint_trajectory_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            params_.joint_trajectory_topic, 10,
            [this](const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) { jointTrajectoryCallback(msg); });

        target_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            params_.target_pose_topic, 10,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { targetPoseCallback(msg); });

        nmpc_res_pub_ =
            this->create_publisher<panda_interfaces::msg::ResultNMPC>(params_.nmpc_result_topic, 1);
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int64_t>(1000.0 * nmpc_solver_.getDt())),
            [this]() { timerCallback(); });

        RCLCPP_INFO(
            this->get_logger(),
            "NMPC single-pose node started. urdf=%s, ee_frame=%s, obstacle_source=%s, joint_states_topic=%s, target_pose_topic=%s, result_topic=%s, joint_trajectory_topic=%s (ignored), use_global_trajectory=%s (ignored)",
            params_.urdf_path.c_str(),
            params_.ee_frame_name.c_str(),
            selectObstacleSourcePath().c_str(),
            params_.joint_states_topic.c_str(),
            params_.target_pose_topic.c_str(),
            params_.nmpc_result_topic.c_str(),
            params_.joint_trajectory_topic.c_str(),
            params_.use_global_trajectory ? "true" : "false");
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NMPCNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
