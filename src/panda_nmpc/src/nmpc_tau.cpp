#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
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
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "panda_interfaces/msg/dynamic_sphere_array.hpp"
#include "panda_interfaces/msg/result_nmpc.hpp"
#include "panda_nmpc_controller.hpp"
#include "second_order_reference_regulator.hpp"
#include "static_sphere_scene.hpp"

class NMPCNode : public rclcpp::Node {
private:
    // =========================================================================
    // Basic Type Aliases
    // =========================================================================
    // Keep the frequently used fixed-size types local to the node so the rest of
    // the implementation can stay compact and consistent.
    using Vec7 = Eigen::Matrix<double, 7, 1>;
    using Vec3 = Eigen::Vector3d;
    using Mat3 = Eigen::Matrix3d;
    using Quat = Eigen::Quaterniond;
    using DynamicSphereArrayMsg = panda_interfaces::msg::DynamicSphereArray;
    using Float64Msg = std_msgs::msg::Float64;

    // =========================================================================
    // Parameter and Runtime Configuration Types
    // =========================================================================
    enum class DynamicObstaclePredictionMode {
        kZeroOrderHold = 0,
        kConstantVelocity,
    };

    enum class PathTrackingState {
        kTargetOnly = 0,
        kFollowingPath,
        kTerminalTarget,
    };

    // Dynamic obstacle subscription policy shared by the timer loop and the
    // obstacle callback.
    struct DynamicObstacleConfig {
        bool enabled{false};
        std::string topic{"/dynamic_sphere_obstacles"};
        std::string prediction_mode{"constant_velocity"};
        double timeout_sec{0.2};
    };

    // Runtime cache for one received moving sphere obstacle.
    struct DynamicSphereState {
        std::string name;
        Vec3 center{Vec3::Zero()};
        Vec3 velocity{Vec3::Zero()};
        double radius{0.0};
    };

    struct RobotCollisionSphere {
        std::string name;
        std::string frame_name;
        pinocchio::FrameIndex frame_id{0};
        Vec3 offset{Vec3::Zero()};
        double radius{0.0};
    };

    struct NearestObstacleDistanceInfo {
        bool available{false};
        double signed_distance{std::numeric_limits<double>::infinity()};
        std::string robot_sphere_name;
        std::string obstacle_name;
        bool obstacle_is_dynamic{false};
    };

    // Low-frequency acados debug logging policy.
    struct AcadosDebugLogConfig {
        bool enabled{false};
        double period_sec{2.0};
        bool print_on_failure{true};
        double slack_activation_threshold{1.0e-5};
        bool include_stage_slack_summary{true};
        int max_reported_stages{4};
    };

    struct ObstacleDistanceMonitorConfig {
        bool enabled{true};
        std::string topic{"/nmpc_nearest_sphere_obstacle_distance"};
        bool include_in_acados_report{true};
    };

    // Carrot-point path-following configuration. Distances are measured in a
    // pose-space metric: translation norm plus a weighted orientation error.
    struct PathFollowingConfig {
        bool enabled{true};
        bool wait_for_fresh_path_after_target_update{true};
        double lookahead_distance{0.03};
        double desired_stage_spacing{0.015};
        double max_stage_average_speed{0.08};
        double max_head_advance_speed{0.06};
        double terminal_target_segment_length{0.03};
        double completion_path_tolerance{0.005};
        double completion_position_tolerance{0.01};
        double completion_orientation_tolerance{0.10};
        double rotation_metric_weight{0.02};
    };

    // Minimal input sanitation for externally commanded target poses.
    struct TargetPoseValidationConfig {
        bool enabled{true};
        std::string expected_frame_id{"link0"};
        bool allow_empty_frame_id{true};
        double max_position_norm{2.0};
        double min_quaternion_norm{1.0e-9};
    };

    struct PathWaypoint {
        double path_s{0.0};
        Vec7 q{Vec7::Zero()};
        Vec3 ee_pos{Vec3::Zero()};
        Mat3 ee_rot{Mat3::Identity()};
    };

    struct JointPathReference {
        std::vector<PathWaypoint> waypoints;
        double total_path_s{0.0};
        Vec7 terminal_q{Vec7::Zero()};
    };

    struct PathReferenceSample {
        Vec7 q_nom{Vec7::Zero()};
        Vec3 ee_pos{Vec3::Zero()};
        Mat3 ee_rot{Mat3::Identity()};
    };

    struct PathTrackingSnapshot {
        std::shared_ptr<const JointPathReference> active_path;
        PathTrackingState state{PathTrackingState::kTargetOnly};
        double reference_head_s{0.0};
        std::size_t closest_index_hint{0};
        Vec7 terminal_q_nominal{Vec7::Zero()};
        bool waiting_for_fresh_path{false};
        rclcpp::Time last_target_pose_update_stamp{0, 0, RCL_ROS_TIME};
        Vec3 latched_wait_target_pos{Vec3::Zero()};
        Mat3 latched_wait_target_rot{Mat3::Identity()};
        Vec7 latched_wait_q_nominal{Vec7::Zero()};
        bool latched_wait_reference_valid{false};
    };

    // Centralized runtime parameter bundle. All ROS parameters are declared once
    // and then fanned out from here to the controller, SORR and callbacks.
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

        DynamicObstacleConfig dynamic_obstacles{};
        AcadosDebugLogConfig acados_debug{};
        ObstacleDistanceMonitorConfig obstacle_distance_monitor{};
        TargetPoseValidationConfig target_pose_validation{};
        PathFollowingConfig path_following{};

        PandaNMPCController::CostWeights cost_weights{PandaNMPCController::defaultCostWeights()};
        PandaNMPCController::HardLimits hard_limits{PandaNMPCController::defaultHardLimits()};
        PandaNMPCController::ObstacleConstraintConfig obstacle_constraint{
            PandaNMPCController::defaultObstacleConstraintConfig()};
    };

    // =========================================================================
    // Core Controller and Regulation Objects
    // =========================================================================
    PandaNMPCController nmpc_solver_;
    NodeParameters params_;
    NMPCResult last_valid_result_;

    SecondOrderReferenceRegulator sorr_pos_;
    SecondOrderReferenceRegulator sorr_rot_;

    // =========================================================================
    // Shared Runtime State
    // =========================================================================
    // These fields are updated by subscriptions and consumed by the timer loop.
    // Access is synchronized via state_mutex_ because the node uses a
    // MultiThreadedExecutor.
    Vec3 target_pos_{Vec3::Zero()};
    Mat3 target_rot_{Mat3::Identity()};
    Vec7 q_meas_{Vec7::Zero()};
    Vec7 dq_meas_{Vec7::Zero()};
    Vec7 q_nominal_{Vec7::Zero()};

    bool joint_state_ready_{false};
    bool sorr_ready_{false};
    bool nominal_initialized_from_state_{false};
    int consecutive_failure_count_{0};

    // =========================================================================
    // Robot Model and Obstacle State
    // =========================================================================
    pinocchio::Model pin_model_;
    std::unique_ptr<pinocchio::Data> pin_data_;
    pinocchio::FrameIndex ee_frame_id_{0};
    mutable std::mutex pinocchio_mutex_;

    std::vector<panda_nmpc::StaticSphereObstacle> scene_obstacles_;
    std::vector<RobotCollisionSphere> robot_collision_spheres_;
    PandaNMPCController::ObstacleParamBlock static_obstacle_params_{
        PandaNMPCController::disabledObstacleParams()};
    std::size_t static_obstacle_slot_count_{0};
    std::vector<DynamicSphereState> dynamic_obstacles_;
    rclcpp::Time dynamic_obstacle_stamp_{0, 0, RCL_ROS_TIME};
    bool dynamic_obstacles_ready_{false};
    DynamicObstaclePredictionMode dynamic_prediction_mode_{DynamicObstaclePredictionMode::kConstantVelocity};
    std::vector<NMPCStageReference> stage_refs_buffer_;
    rclcpp::Time last_acados_debug_log_stamp_{0, 0, RCL_ROS_TIME};
    double min_recorded_nearest_obstacle_distance_{std::numeric_limits<double>::infinity()};
    std::string min_recorded_robot_sphere_name_;
    std::string min_recorded_obstacle_name_;
    bool min_recorded_obstacle_is_dynamic_{false};
    std::shared_ptr<const JointPathReference> active_joint_path_;
    PathTrackingState path_tracking_state_{PathTrackingState::kTargetOnly};
    double path_reference_head_s_{0.0};
    std::size_t path_closest_index_hint_{0};
    Vec7 path_terminal_q_nominal_{Vec7::Zero()};
    bool waiting_for_fresh_path_after_target_update_{false};
    rclcpp::Time last_target_pose_update_stamp_{0, 0, RCL_ROS_TIME};
    Vec3 latched_wait_target_pos_{Vec3::Zero()};
    Mat3 latched_wait_target_rot_{Mat3::Identity()};
    Vec7 latched_wait_q_nominal_{Vec7::Zero()};
    bool latched_wait_reference_valid_{false};
    mutable std::mutex state_mutex_;

    // =========================================================================
    // ROS Interfaces
    // =========================================================================
    rclcpp::Publisher<panda_interfaces::msg::ResultNMPC>::SharedPtr nmpc_res_pub_;
    rclcpp::Publisher<Float64Msg>::SharedPtr nearest_obstacle_distance_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;
    rclcpp::Subscription<DynamicSphereArrayMsg>::SharedPtr dynamic_obstacle_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::CallbackGroup::SharedPtr timer_callback_group_;
    rclcpp::CallbackGroup::SharedPtr state_callback_group_;
    rclcpp::CallbackGroup::SharedPtr dynamic_obstacle_callback_group_;

    // Fixed name ordering used to align ROS joint state messages with the Panda
    // solver state layout.
    std::array<std::string, 7> ordered_names_{{
        "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"
    }};

    // =========================================================================
    // Path and Formatting Helpers
    // =========================================================================
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

    // Declare a fixed-length numeric ROS parameter and fail fast if the YAML
    // shape does not match the expected solver dimension.
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

    // Parse the stage-wise obstacle prediction policy once at startup so the
    // timer loop only needs a cheap enum branch.
    static DynamicObstaclePredictionMode parseDynamicObstaclePredictionMode(const std::string& value) {
        if (value == "constant_velocity") {
            return DynamicObstaclePredictionMode::kConstantVelocity;
        }
        if (value == "zero_order_hold" || value == "zoh") {
            return DynamicObstaclePredictionMode::kZeroOrderHold;
        }
        throw std::runtime_error(
            "Unsupported dynamic obstacle prediction mode '" + value +
            "'. Use 'constant_velocity' or 'zero_order_hold'.");
    }

    static double rotationAngularDistance(const Mat3& rot_a, const Mat3& rot_b) {
        const Quat q_a(rot_a);
        const Quat q_b(rot_b);
        const double dot = std::clamp(std::abs(q_a.dot(q_b)), 0.0, 1.0);
        return 2.0 * std::acos(dot);
    }

    // Render debug scalars in a stable way so acados logs remain readable even
    // when values span multiple orders of magnitude.
    static std::string formatDebugFloat(const double value) {
        if (!std::isfinite(value)) {
            return "nan";
        }

        std::ostringstream stream;
        const double abs_value = std::abs(value);
        if ((abs_value > 0.0 && abs_value < 1.0e-3) || abs_value >= 1.0e4) {
            stream << std::scientific << std::setprecision(4) << value;
        } else {
            stream << std::fixed << std::setprecision(4) << value;
        }
        return stream.str();
    }

    static void appendDebugLine(
        std::ostringstream& stream,
        const std::string& label,
        const std::string& value) {
        constexpr int kLabelWidth = 38;
        stream << "| " << std::left << std::setw(kLabelWidth) << label
               << " : " << value << '\n';
    }

    static void appendDebugLine(
        std::ostringstream& stream,
        const std::string& label,
        const double value) {
        appendDebugLine(stream, label, formatDebugFloat(value));
    }

    static void appendDebugLine(
        std::ostringstream& stream,
        const std::string& label,
        const int value) {
        appendDebugLine(stream, label, std::to_string(value));
    }

    // =========================================================================
    // Parameter Declaration and Startup Configuration
    // =========================================================================
    // This is the single place where ROS parameters are declared. Keeping that
    // work centralized makes it much easier to audit how YAML flows into the
    // controller and the runtime node state.
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

        params_.target_pose_validation.enabled =
            this->declare_parameter<bool>("target_pose_validation.enabled", params_.target_pose_validation.enabled);
        params_.target_pose_validation.expected_frame_id = this->declare_parameter<std::string>(
            "target_pose_validation.expected_frame_id", params_.target_pose_validation.expected_frame_id);
        params_.target_pose_validation.allow_empty_frame_id = this->declare_parameter<bool>(
            "target_pose_validation.allow_empty_frame_id", params_.target_pose_validation.allow_empty_frame_id);
        params_.target_pose_validation.max_position_norm = this->declare_parameter<double>(
            "target_pose_validation.max_position_norm", params_.target_pose_validation.max_position_norm);
        params_.target_pose_validation.min_quaternion_norm = this->declare_parameter<double>(
            "target_pose_validation.min_quaternion_norm", params_.target_pose_validation.min_quaternion_norm);
        if (params_.target_pose_validation.max_position_norm <= 0.0) {
            throw std::runtime_error("Parameter 'target_pose_validation.max_position_norm' must be > 0.");
        }
        if (params_.target_pose_validation.min_quaternion_norm <= 0.0) {
            throw std::runtime_error("Parameter 'target_pose_validation.min_quaternion_norm' must be > 0.");
        }

        params_.dynamic_obstacles.enabled =
            this->declare_parameter<bool>("dynamic_obstacles.enabled", params_.dynamic_obstacles.enabled);
        params_.dynamic_obstacles.topic =
            this->declare_parameter<std::string>("dynamic_obstacles.topic", params_.dynamic_obstacles.topic);
        params_.dynamic_obstacles.prediction_mode =
            this->declare_parameter<std::string>(
                "dynamic_obstacles.prediction_mode", params_.dynamic_obstacles.prediction_mode);
        params_.dynamic_obstacles.timeout_sec = std::max(
            1.0e-3,
            this->declare_parameter<double>("dynamic_obstacles.timeout_sec", params_.dynamic_obstacles.timeout_sec));
        dynamic_prediction_mode_ =
            parseDynamicObstaclePredictionMode(params_.dynamic_obstacles.prediction_mode);

        params_.acados_debug.enabled =
            this->declare_parameter<bool>("debug.acados.enabled", params_.acados_debug.enabled);
        params_.acados_debug.period_sec = std::max(
            0.1,
            this->declare_parameter<double>("debug.acados.period_sec", params_.acados_debug.period_sec));
        params_.acados_debug.print_on_failure =
            this->declare_parameter<bool>("debug.acados.print_on_failure", params_.acados_debug.print_on_failure);
        params_.acados_debug.slack_activation_threshold = std::max(
            0.0,
            this->declare_parameter<double>(
                "debug.acados.slack_activation_threshold",
                params_.acados_debug.slack_activation_threshold));
        params_.acados_debug.include_stage_slack_summary = this->declare_parameter<bool>(
            "debug.acados.include_stage_slack_summary",
            params_.acados_debug.include_stage_slack_summary);
        const auto max_reported_stages = this->declare_parameter<int64_t>(
            "debug.acados.max_reported_stages",
            static_cast<int64_t>(params_.acados_debug.max_reported_stages));
        params_.acados_debug.max_reported_stages =
            static_cast<int>(std::max<int64_t>(1, max_reported_stages));

        params_.obstacle_distance_monitor.enabled = this->declare_parameter<bool>(
            "debug.nearest_obstacle_distance.enabled",
            params_.obstacle_distance_monitor.enabled);
        params_.obstacle_distance_monitor.topic = this->declare_parameter<std::string>(
            "debug.nearest_obstacle_distance.topic",
            params_.obstacle_distance_monitor.topic);
        params_.obstacle_distance_monitor.include_in_acados_report =
            this->declare_parameter<bool>(
                "debug.nearest_obstacle_distance.include_in_acados_report",
                params_.obstacle_distance_monitor.include_in_acados_report);

        params_.path_following.enabled =
            this->declare_parameter<bool>("path_following.enabled", params_.path_following.enabled);
        params_.path_following.wait_for_fresh_path_after_target_update =
            this->declare_parameter<bool>(
                "path_following.wait_for_fresh_path_after_target_update",
                params_.path_following.wait_for_fresh_path_after_target_update);
        params_.path_following.lookahead_distance = this->declare_parameter<double>(
            "path_following.lookahead_distance", params_.path_following.lookahead_distance);
        params_.path_following.desired_stage_spacing = this->declare_parameter<double>(
            "path_following.desired_stage_spacing", params_.path_following.desired_stage_spacing);
        params_.path_following.max_stage_average_speed = this->declare_parameter<double>(
            "path_following.max_stage_average_speed", params_.path_following.max_stage_average_speed);
        params_.path_following.max_head_advance_speed = this->declare_parameter<double>(
            "path_following.max_head_advance_speed", params_.path_following.max_head_advance_speed);
        params_.path_following.terminal_target_segment_length = this->declare_parameter<double>(
            "path_following.terminal_target_segment_length",
            params_.path_following.terminal_target_segment_length);
        params_.path_following.completion_path_tolerance = this->declare_parameter<double>(
            "path_following.completion_path_tolerance",
            params_.path_following.completion_path_tolerance);
        params_.path_following.completion_position_tolerance = this->declare_parameter<double>(
            "path_following.completion_position_tolerance",
            params_.path_following.completion_position_tolerance);
        params_.path_following.completion_orientation_tolerance = this->declare_parameter<double>(
            "path_following.completion_orientation_tolerance",
            params_.path_following.completion_orientation_tolerance);
        params_.path_following.rotation_metric_weight = this->declare_parameter<double>(
            "path_following.rotation_metric_weight",
            params_.path_following.rotation_metric_weight);
        if (params_.path_following.lookahead_distance < 0.0) {
            throw std::runtime_error("Parameter 'path_following.lookahead_distance' must be >= 0.");
        }
        if (params_.path_following.desired_stage_spacing <= 0.0) {
            throw std::runtime_error("Parameter 'path_following.desired_stage_spacing' must be > 0.");
        }
        if (params_.path_following.max_stage_average_speed <= 0.0) {
            throw std::runtime_error("Parameter 'path_following.max_stage_average_speed' must be > 0.");
        }
        if (params_.path_following.max_head_advance_speed <= 0.0) {
            throw std::runtime_error("Parameter 'path_following.max_head_advance_speed' must be > 0.");
        }
        if (params_.path_following.terminal_target_segment_length <= 0.0) {
            throw std::runtime_error(
                "Parameter 'path_following.terminal_target_segment_length' must be > 0.");
        }
        if (params_.path_following.completion_path_tolerance < 0.0) {
            throw std::runtime_error(
                "Parameter 'path_following.completion_path_tolerance' must be >= 0.");
        }
        if (params_.path_following.completion_position_tolerance <= 0.0) {
            throw std::runtime_error(
                "Parameter 'path_following.completion_position_tolerance' must be > 0.");
        }
        if (params_.path_following.completion_orientation_tolerance <= 0.0) {
            throw std::runtime_error(
                "Parameter 'path_following.completion_orientation_tolerance' must be > 0.");
        }
        if (params_.path_following.rotation_metric_weight < 0.0) {
            throw std::runtime_error("Parameter 'path_following.rotation_metric_weight' must be >= 0.");
        }

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

    // Prefer the explicit obstacle XML if it exists; otherwise fall back to the
    // MuJoCo scene XML so one launch file can still boot with minimal setup.
    std::string selectObstacleSourcePath() const {
        if (!params_.obstacle_config_path.empty() &&
            std::ifstream(params_.obstacle_config_path).good()) {
            return params_.obstacle_config_path;
        }
        return params_.scene_xml_path;
    }

    // =========================================================================
    // Robot Model Initialization and Static Obstacle Loading
    // =========================================================================
    // Everything here runs once during construction and produces immutable
    // resources that the timer loop later reuses.
    void loadRobotCollisionSpheres() {
        robot_collision_spheres_.clear();

        tinyxml2::XMLDocument doc;
        const tinyxml2::XMLError load_status = doc.LoadFile(params_.urdf_path.c_str());
        if (load_status != tinyxml2::XML_SUCCESS) {
            throw std::runtime_error(
                "Failed to load Panda URDF '" + params_.urdf_path + "' for robot collision spheres: " +
                (doc.ErrorStr() == nullptr ? std::string("unknown error") : std::string(doc.ErrorStr())));
        }

        const tinyxml2::XMLElement* root = doc.RootElement();
        if (root == nullptr) {
            throw std::runtime_error("URDF '" + params_.urdf_path + "' has no root element.");
        }

        for (const tinyxml2::XMLElement* link_elem = root->FirstChildElement("link");
             link_elem != nullptr;
             link_elem = link_elem->NextSiblingElement("link")) {
            const char* link_name_attr = link_elem->Attribute("name");
            if (link_name_attr == nullptr) {
                continue;
            }

            const std::string frame_name = link_name_attr;
            if (!pin_model_.existFrame(frame_name)) {
                continue;
            }
            const pinocchio::FrameIndex frame_id = pin_model_.getFrameId(frame_name);

            int sphere_index = 0;
            for (const tinyxml2::XMLElement* visual_elem = link_elem->FirstChildElement("visual");
                 visual_elem != nullptr;
                 visual_elem = visual_elem->NextSiblingElement("visual")) {
                const tinyxml2::XMLElement* geometry_elem = visual_elem->FirstChildElement("geometry");
                if (geometry_elem == nullptr) {
                    continue;
                }

                const tinyxml2::XMLElement* sphere_elem = geometry_elem->FirstChildElement("sphere");
                if (sphere_elem == nullptr) {
                    continue;
                }

                const char* radius_attr = sphere_elem->Attribute("radius");
                if (radius_attr == nullptr) {
                    continue;
                }

                double radius = 0.0;
                std::istringstream radius_stream(radius_attr);
                if (!(radius_stream >> radius) || !(radius > 0.0)) {
                    continue;
                }

                Vec3 offset = Vec3::Zero();
                const tinyxml2::XMLElement* origin_elem = visual_elem->FirstChildElement("origin");
                if (origin_elem != nullptr) {
                    const char* xyz_attr = origin_elem->Attribute("xyz");
                    if (xyz_attr != nullptr) {
                        offset = panda_nmpc::parse_xyz_attribute(xyz_attr);
                    }
                }

                RobotCollisionSphere sphere;
                sphere.name = frame_name + "::sphere_" + std::to_string(sphere_index++);
                sphere.frame_name = frame_name;
                sphere.frame_id = frame_id;
                sphere.offset = offset;
                sphere.radius = radius;
                robot_collision_spheres_.push_back(sphere);
            }
        }

        if (robot_collision_spheres_.empty()) {
            RCLCPP_WARN(
                this->get_logger(),
                "No robot collision spheres were found in URDF visuals. Nearest obstacle distance monitoring will be unavailable.");
            return;
        }

        RCLCPP_INFO(
            this->get_logger(),
            "Loaded %zu robot collision sphere(s) from '%s' for nearest-distance monitoring.",
            robot_collision_spheres_.size(),
            params_.urdf_path.c_str());
    }

    void loadStaticObstacles() {
        static_obstacle_params_ = PandaNMPCController::disabledObstacleParams();
        static_obstacle_slot_count_ = 0;
        const std::string obstacle_source_path = selectObstacleSourcePath();
        scene_obstacles_ = panda_nmpc::loadStaticSphereObstaclesFromSceneXml(obstacle_source_path);

        if (scene_obstacles_.empty()) {
            RCLCPP_WARN(
                this->get_logger(),
                "No static sphere obstacles found in '%s'. Obstacle constraints are disabled until runtime data arrives.",
                obstacle_source_path.c_str());
            return;
        }

        if (scene_obstacles_.size() > PandaNMPCController::kNumRuntimeObstacles) {
            RCLCPP_WARN(
                this->get_logger(),
                "Obstacle config defines %zu obstacle(s) but solver only supports %zu. Extra obstacles will be ignored.",
                scene_obstacles_.size(),
                PandaNMPCController::kNumRuntimeObstacles);
        }

        const std::size_t obstacle_count =
            std::min(scene_obstacles_.size(), PandaNMPCController::kNumRuntimeObstacles);
        static_obstacle_slot_count_ = obstacle_count;
        for (std::size_t obstacle_idx = 0; obstacle_idx < obstacle_count; ++obstacle_idx) {
            const auto& obstacle = scene_obstacles_[obstacle_idx];
            const std::size_t offset = obstacle_idx * 4;
            static_obstacle_params_[offset + 0] = obstacle.center.x();
            static_obstacle_params_[offset + 1] = obstacle.center.y();
            static_obstacle_params_[offset + 2] = obstacle.center.z();
            static_obstacle_params_[offset + 3] = obstacle.radius;
        }

        std::ostringstream stream;
        stream << "Loaded " << scene_obstacles_.size()
               << " static obstacle(s) from '" << obstacle_source_path
               << "'. Applied " << obstacle_count << " slot(s), reserved "
               << (PandaNMPCController::kNumRuntimeObstacles - static_obstacle_slot_count_)
               << " slot(s) for dynamic obstacles: ";
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

    // Build the Pinocchio model once and cache the EE frame id so later FK
    // queries only need lightweight data updates.
    void initializePinocchio() {
        pinocchio::urdf::buildModel(params_.urdf_path, pin_model_);
        pin_data_ = std::make_unique<pinocchio::Data>(pin_model_);

        if (!pin_model_.existFrame(params_.ee_frame_name)) {
            throw std::runtime_error("EE frame '" + params_.ee_frame_name + "' not found in URDF.");
        }
        ee_frame_id_ = pin_model_.getFrameId(params_.ee_frame_name);
        loadRobotCollisionSpheres();
    }

    // =========================================================================
    // SORR and Kinematics Utilities
    // =========================================================================
    // These helpers keep all target filtering and current-EE-state estimation in
    // one place, separate from the solver orchestration logic.
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

    void computeEePoseForConfiguration(const Vec7& q, Vec3& ee_pos, Mat3& ee_rot) const {
        std::lock_guard<std::mutex> lock(pinocchio_mutex_);
        pinocchio::forwardKinematics(pin_model_, *pin_data_, q);
        pinocchio::updateFramePlacements(pin_model_, *pin_data_);
        const auto& transform = pin_data_->oMf[ee_frame_id_];
        ee_pos = transform.translation();
        ee_rot = transform.rotation();
    }

    void computeCurrentGeometrySnapshot(
        const Vec7& q,
        Vec3& ee_pos,
        Mat3& ee_rot,
        const std::vector<DynamicSphereState>& dynamic_obstacles,
        bool use_dynamic_obstacles,
        NearestObstacleDistanceInfo* nearest_distance_info) const {
        std::lock_guard<std::mutex> lock(pinocchio_mutex_);
        pinocchio::forwardKinematics(pin_model_, *pin_data_, q);
        pinocchio::updateFramePlacements(pin_model_, *pin_data_);
        const auto& ee_transform = pin_data_->oMf[ee_frame_id_];
        ee_pos = ee_transform.translation();
        ee_rot = ee_transform.rotation();

        if (nearest_distance_info == nullptr) {
            return;
        }

        *nearest_distance_info = NearestObstacleDistanceInfo{};
        if (robot_collision_spheres_.empty()) {
            return;
        }

        auto try_update_min_distance =
            [&nearest_distance_info](
                const std::string& robot_sphere_name,
                const std::string& obstacle_name,
                bool obstacle_is_dynamic,
                double signed_distance) {
                if (!nearest_distance_info->available ||
                    signed_distance < nearest_distance_info->signed_distance) {
                    nearest_distance_info->available = true;
                    nearest_distance_info->signed_distance = signed_distance;
                    nearest_distance_info->robot_sphere_name = robot_sphere_name;
                    nearest_distance_info->obstacle_name = obstacle_name;
                    nearest_distance_info->obstacle_is_dynamic = obstacle_is_dynamic;
                }
            };

        for (const auto& robot_sphere : robot_collision_spheres_) {
            const auto& sphere_transform = pin_data_->oMf[robot_sphere.frame_id];
            const Vec3 sphere_center =
                sphere_transform.translation() + sphere_transform.rotation() * robot_sphere.offset;

            for (const auto& obstacle : scene_obstacles_) {
                const double signed_distance =
                    (sphere_center - obstacle.center).norm() - (robot_sphere.radius + obstacle.radius);
                try_update_min_distance(
                    robot_sphere.name,
                    obstacle.name,
                    false,
                    signed_distance);
            }

            if (!use_dynamic_obstacles) {
                continue;
            }

            for (const auto& obstacle : dynamic_obstacles) {
                const double signed_distance =
                    (sphere_center - obstacle.center).norm() - (robot_sphere.radius + obstacle.radius);
                try_update_min_distance(
                    robot_sphere.name,
                    obstacle.name,
                    true,
                    signed_distance);
            }
        }
    }

    // Compute the current end-effector pose from the latest joint measurement.
    void computeCurrentEePose(Vec3& ee_pos, Mat3& ee_rot) {
        computeEePoseForConfiguration(q_meas_, ee_pos, ee_rot);
    }

    // Seed the second-order reference regulators from the measured EE pose so
    // the first control cycle does not see an artificial target step.
    void initializeSorrFromCurrentEePose() {
        Vec3 ee_pos = Vec3::Zero();
        Mat3 ee_rot = Mat3::Identity();
        computeCurrentEePose(ee_pos, ee_rot);
        sorr_pos_.resetPosition(ee_pos);
        sorr_rot_.resetOrientation(Quat(ee_rot));
        sorr_ready_ = true;
        RCLCPP_INFO(this->get_logger(), "SORR initialized from current EE pose.");
    }

    // =========================================================================
    // Obstacle Projection and Reference Assembly
    // =========================================================================
    // The timer loop asks this block to build stage-wise references and obstacle
    // parameters without having to care about XML-vs-topic details.
    bool hasFreshDynamicObstacles(
        const rclcpp::Time& dynamic_obstacle_stamp,
        bool dynamic_obstacles_ready) const {
        if (!params_.dynamic_obstacles.enabled || !dynamic_obstacles_ready) {
            return false;
        }
        return (this->now() - dynamic_obstacle_stamp).seconds() <= params_.dynamic_obstacles.timeout_sec;
    }

    // Merge static obstacle slots with predicted dynamic obstacle slots for one
    // horizon stage.
    PandaNMPCController::ObstacleParamBlock buildObstacleParamsForStage(
        const int stage,
        const std::vector<DynamicSphereState>& dynamic_obstacles,
        bool dynamic_obstacles_ready,
        const rclcpp::Time& dynamic_obstacle_stamp) const {
        PandaNMPCController::ObstacleParamBlock params = static_obstacle_params_;
        if (!hasFreshDynamicObstacles(dynamic_obstacle_stamp, dynamic_obstacles_ready)) {
            return params;
        }

        const double prediction_time = static_cast<double>(stage) * nmpc_solver_.getDt();
        const std::size_t dynamic_slot_count =
            PandaNMPCController::kNumRuntimeObstacles - static_obstacle_slot_count_;
        const std::size_t obstacle_count = std::min(dynamic_obstacles.size(), dynamic_slot_count);

        for (std::size_t obstacle_idx = 0; obstacle_idx < obstacle_count; ++obstacle_idx) {
            Vec3 center = dynamic_obstacles[obstacle_idx].center;
            if (dynamic_prediction_mode_ == DynamicObstaclePredictionMode::kConstantVelocity) {
                center += prediction_time * dynamic_obstacles[obstacle_idx].velocity;
            }

            const std::size_t offset =
                (static_obstacle_slot_count_ + obstacle_idx) * 4;
            params[offset + 0] = center.x();
            params[offset + 1] = center.y();
            params[offset + 2] = center.z();
            params[offset + 3] = dynamic_obstacles[obstacle_idx].radius;
        }
        return params;
    }

    double posePathMetric(
        const Vec3& pos_a,
        const Mat3& rot_a,
        const Vec3& pos_b,
        const Mat3& rot_b) const {
        return (pos_a - pos_b).norm() +
               params_.path_following.rotation_metric_weight * rotationAngularDistance(rot_a, rot_b);
    }

    bool findOrderedJointIndices(
        const std::vector<std::string>& joint_names,
        std::array<std::size_t, 7>& indices) const {
        for (std::size_t joint = 0; joint < ordered_names_.size(); ++joint) {
            const auto it = std::find(joint_names.begin(), joint_names.end(), ordered_names_[joint]);
            if (it == joint_names.end()) {
                return false;
            }
            indices[joint] = static_cast<std::size_t>(std::distance(joint_names.begin(), it));
        }
        return true;
    }

    Vec7 extractOrderedJointPosition(
        const trajectory_msgs::msg::JointTrajectoryPoint& point,
        const std::array<std::size_t, 7>& indices) const {
        Vec7 q = Vec7::Zero();
        for (std::size_t joint = 0; joint < ordered_names_.size(); ++joint) {
            q(static_cast<Eigen::Index>(joint)) = point.positions[indices[joint]];
        }
        return q;
    }

    std::shared_ptr<JointPathReference> buildJointPathFromTrajectory(
        const trajectory_msgs::msg::JointTrajectory& trajectory) {
        if (trajectory.points.empty()) {
            return nullptr;
        }

        std::array<std::size_t, 7> joint_indices{};
        if (!findOrderedJointIndices(trajectory.joint_names, joint_indices)) {
            RCLCPP_WARN(
                this->get_logger(),
                "Received JointTrajectory that does not contain the expected Panda joint names. Path update is ignored.");
            return nullptr;
        }

        auto path = std::make_shared<JointPathReference>();
        path->waypoints.reserve(trajectory.points.size());

        for (const auto& point : trajectory.points) {
            if (point.positions.size() < ordered_names_.size()) {
                RCLCPP_WARN(
                    this->get_logger(),
                    "Ignoring JointTrajectory point with %zu positions; expected at least %zu.",
                    point.positions.size(),
                    ordered_names_.size());
                return nullptr;
            }

            PathWaypoint waypoint;
            waypoint.q = extractOrderedJointPosition(point, joint_indices);
            computeEePoseForConfiguration(waypoint.q, waypoint.ee_pos, waypoint.ee_rot);

            if (path->waypoints.empty()) {
                waypoint.path_s = 0.0;
                path->waypoints.push_back(waypoint);
                continue;
            }

            const auto& previous = path->waypoints.back();
            waypoint.path_s = previous.path_s +
                              posePathMetric(previous.ee_pos, previous.ee_rot, waypoint.ee_pos, waypoint.ee_rot);
            path->waypoints.push_back(waypoint);
        }

        if (path->waypoints.empty()) {
            return nullptr;
        }

        path->terminal_q = path->waypoints.back().q;
        path->total_path_s = path->waypoints.back().path_s;
        return path;
    }

    double computeEffectivePathStageSpacing() const {
        return std::min(
            params_.path_following.desired_stage_spacing,
            params_.path_following.max_stage_average_speed * nmpc_solver_.getDt());
    }

    std::size_t findClosestPathWaypointIndex(
        const JointPathReference& path,
        const Vec3& ee_pos,
        const Mat3& ee_rot,
        std::size_t start_hint) const {
        if (path.waypoints.empty()) {
            return 0;
        }

        const std::size_t clamped_start_hint =
            std::min(start_hint, path.waypoints.size() - 1);
        std::size_t best_index = clamped_start_hint;
        double best_metric = std::numeric_limits<double>::infinity();
        for (std::size_t idx = clamped_start_hint; idx < path.waypoints.size(); ++idx) {
            const auto& waypoint = path.waypoints[idx];
            const double metric =
                posePathMetric(ee_pos, ee_rot, waypoint.ee_pos, waypoint.ee_rot);
            if (metric < best_metric) {
                best_metric = metric;
                best_index = idx;
            }
        }
        return best_index;
    }

    PathWaypoint interpolatePathWaypoint(
        const JointPathReference& path,
        double query_s,
        std::size_t& segment_index_hint) const {
        if (path.waypoints.empty()) {
            return PathWaypoint{};
        }
        if (query_s <= 0.0 || path.waypoints.size() == 1) {
            return path.waypoints.front();
        }
        if (query_s >= path.total_path_s) {
            return path.waypoints.back();
        }

        const std::size_t max_segment_index = path.waypoints.size() - 2;
        segment_index_hint = std::min(segment_index_hint, max_segment_index);
        while (segment_index_hint < max_segment_index &&
               path.waypoints[segment_index_hint + 1].path_s < query_s) {
            ++segment_index_hint;
        }
        while (segment_index_hint > 0 &&
               path.waypoints[segment_index_hint].path_s > query_s) {
            --segment_index_hint;
        }

        const auto& waypoint_a = path.waypoints[segment_index_hint];
        const auto& waypoint_b = path.waypoints[segment_index_hint + 1];
        const double segment_length =
            std::max(1.0e-9, waypoint_b.path_s - waypoint_a.path_s);
        const double alpha =
            std::clamp((query_s - waypoint_a.path_s) / segment_length, 0.0, 1.0);

        PathWaypoint interpolated;
        interpolated.path_s = query_s;
        interpolated.q = (1.0 - alpha) * waypoint_a.q + alpha * waypoint_b.q;
        computeEePoseForConfiguration(interpolated.q, interpolated.ee_pos, interpolated.ee_rot);
        return interpolated;
    }

    PathReferenceSample samplePathReferenceAtS(
        const JointPathReference& path,
        double query_s,
        const Vec3& final_target_pos,
        const Mat3& final_target_rot,
        std::size_t& segment_index_hint) const {
        PathReferenceSample sample;
        if (query_s <= path.total_path_s || path.waypoints.empty()) {
            const PathWaypoint waypoint = interpolatePathWaypoint(path, query_s, segment_index_hint);
            sample.q_nom = waypoint.q;
            sample.ee_pos = waypoint.ee_pos;
            sample.ee_rot = waypoint.ee_rot;
            return sample;
        }

        sample.q_nom = path.terminal_q;
        sample.ee_pos = final_target_pos;
        sample.ee_rot = final_target_rot;
        return sample;
    }

    Vec3 approximateAngularVelocity(
        const Mat3& current_rot,
        const Mat3& next_rot,
        double dt) const {
        const Mat3 rotation_delta = current_rot.transpose() * next_rot;
        Eigen::AngleAxisd angle_axis(rotation_delta);
        if (!std::isfinite(angle_axis.angle()) || angle_axis.angle() < 1.0e-9) {
            return Vec3::Zero();
        }
        const Vec3 omega_local = angle_axis.axis() * (angle_axis.angle() / std::max(dt, 1.0e-6));
        return current_rot * omega_local;
    }

    PathTrackingSnapshot snapshotPathTrackingState() const {
        PathTrackingSnapshot snapshot;
        snapshot.active_path = active_joint_path_;
        snapshot.state = path_tracking_state_;
        snapshot.reference_head_s = path_reference_head_s_;
        snapshot.closest_index_hint = path_closest_index_hint_;
        snapshot.terminal_q_nominal = path_terminal_q_nominal_;
        snapshot.waiting_for_fresh_path = waiting_for_fresh_path_after_target_update_;
        snapshot.last_target_pose_update_stamp = last_target_pose_update_stamp_;
        snapshot.latched_wait_target_pos = latched_wait_target_pos_;
        snapshot.latched_wait_target_rot = latched_wait_target_rot_;
        snapshot.latched_wait_q_nominal = latched_wait_q_nominal_;
        snapshot.latched_wait_reference_valid = latched_wait_reference_valid_;
        return snapshot;
    }

    void resetPathTrackingStateLocked() {
        active_joint_path_.reset();
        path_tracking_state_ = PathTrackingState::kTargetOnly;
        path_reference_head_s_ = 0.0;
        path_closest_index_hint_ = 0;
        path_terminal_q_nominal_ = q_nominal_;
    }

    void commitPathTrackingStateIfUnchanged(
        const PathTrackingSnapshot& snapshot,
        PathTrackingState updated_state,
        double updated_reference_head_s,
        std::size_t updated_closest_index_hint) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (active_joint_path_.get() != snapshot.active_path.get()) {
            return;
        }

        if (updated_state == PathTrackingState::kTargetOnly) {
            resetPathTrackingStateLocked();
            return;
        }

        path_reference_head_s_ = updated_reference_head_s;
        path_closest_index_hint_ = updated_closest_index_hint;
        path_tracking_state_ = updated_state;
        path_terminal_q_nominal_ =
            snapshot.active_path ? snapshot.active_path->terminal_q : q_nominal_;
    }

    // Build a horizon of single-target references. Every stage shares the same
    // filtered task-space target, while obstacle parameters can vary by stage.
    const std::vector<NMPCStageReference>& buildUniformStageReferences(
        const Vec3& filtered_target_pos,
        const Mat3& filtered_target_rot,
        const Vec7& q_nominal,
        const Vec3& ee_lin_vel_ref,
        const Vec3& ee_ang_vel_ref,
        const std::vector<DynamicSphereState>& dynamic_obstacles,
        bool dynamic_obstacles_ready,
        const rclcpp::Time& dynamic_obstacle_stamp) {

        const std::size_t stage_count = static_cast<std::size_t>(nmpc_solver_.getN() + 1);
        if (stage_refs_buffer_.size() != stage_count) {
            stage_refs_buffer_.resize(stage_count);
        }

        for (std::size_t stage = 0; stage < stage_count; ++stage) {
            auto& stage_ref = stage_refs_buffer_[stage];
            stage_ref.target_pos = filtered_target_pos;
            stage_ref.target_rot = filtered_target_rot;
            stage_ref.q_nom = q_nominal;
            stage_ref.ee_lin_vel_ref = ee_lin_vel_ref;
            stage_ref.ee_ang_vel_ref = ee_ang_vel_ref;
            stage_ref.obstacle_params = buildObstacleParamsForStage(
                static_cast<int>(stage),
                dynamic_obstacles,
                dynamic_obstacles_ready,
                dynamic_obstacle_stamp);
        }
        return stage_refs_buffer_;
    }

    const std::vector<NMPCStageReference>& buildPathTrackingStageReferences(
        const JointPathReference& path,
        double reference_head_s,
        const Vec3& final_target_pos,
        const Mat3& final_target_rot,
        const std::vector<DynamicSphereState>& dynamic_obstacles,
        bool dynamic_obstacles_ready,
        const rclcpp::Time& dynamic_obstacle_stamp) {

        const std::size_t stage_count = static_cast<std::size_t>(nmpc_solver_.getN() + 1);
        if (stage_refs_buffer_.size() != stage_count) {
            stage_refs_buffer_.resize(stage_count);
        }

        const double stage_spacing = computeEffectivePathStageSpacing();
        const double dt = nmpc_solver_.getDt();
        const double total_reference_length =
            path.total_path_s + params_.path_following.terminal_target_segment_length;

        std::vector<PathReferenceSample> reference_samples(stage_count + 1);
        std::size_t segment_index_hint = 0;
        for (std::size_t sample_idx = 0; sample_idx < reference_samples.size(); ++sample_idx) {
            const double query_s = std::clamp(
                reference_head_s + static_cast<double>(sample_idx) * stage_spacing,
                0.0,
                total_reference_length);
            reference_samples[sample_idx] = samplePathReferenceAtS(
                path,
                query_s,
                final_target_pos,
                final_target_rot,
                segment_index_hint);
        }

        for (std::size_t stage = 0; stage < stage_count; ++stage) {
            auto& stage_ref = stage_refs_buffer_[stage];
            const auto& current = reference_samples[stage];
            const auto& next = reference_samples[stage + 1];
            stage_ref.target_pos = current.ee_pos;
            stage_ref.target_rot = current.ee_rot;
            stage_ref.q_nom = current.q_nom;
            stage_ref.ee_lin_vel_ref = (next.ee_pos - current.ee_pos) / std::max(dt, 1.0e-6);
            stage_ref.ee_ang_vel_ref = approximateAngularVelocity(current.ee_rot, next.ee_rot, dt);
            stage_ref.obstacle_params = buildObstacleParamsForStage(
                static_cast<int>(stage),
                dynamic_obstacles,
                dynamic_obstacles_ready,
                dynamic_obstacle_stamp);
        }

        return stage_refs_buffer_;
    }

    // =========================================================================
    // Debug Reporting
    // =========================================================================
    // acados introspection is intentionally kept out of the control logic so the
    // main timer path stays easy to read.
    bool shouldLogAcadosDebug(const int solve_status) const {
        const bool periodic_logging_enabled = params_.acados_debug.enabled;
        const bool failure_logging_enabled =
            params_.acados_debug.print_on_failure && solve_status != 0;
        if (!periodic_logging_enabled && !failure_logging_enabled) {
            return false;
        }

        if (last_acados_debug_log_stamp_.nanoseconds() == 0) {
            return true;
        }

        return (this->now() - last_acados_debug_log_stamp_).seconds() >=
               params_.acados_debug.period_sec;
    }

    void maybeLogAcadosDebug(
        const int solve_status,
        const NearestObstacleDistanceInfo* nearest_distance_info = nullptr) {
        if (!shouldLogAcadosDebug(solve_status)) {
            return;
        }

        NMPCSolveDebugInfo debug_info = nmpc_solver_.collectLastSolveDebugInfo(
            params_.acados_debug.slack_activation_threshold,
            params_.acados_debug.include_stage_slack_summary);
        if (!debug_info.available) {
            return;
        }

        last_acados_debug_log_stamp_ = this->now();

        std::ostringstream stream;
        const std::string divider =
            "======================================================================";
        const std::string subsection_divider =
            "----------------------------------------------------------------------";
        const int solver_status = debug_info.solver_status;
        const std::string report_prefix =
            (solver_status == 0)
                ? ">>> [INFO][ACADOS_NMPC_DEBUG] <<<"
                : "!!! [WARN][ACADOS_NMPC_DEBUG] !!!";

        stream << report_prefix << '\n';
        stream << divider << '\n';
        stream << "ACADOS NMPC DEBUG REPORT\n";
        stream << subsection_divider << '\n';
        appendDebugLine(stream, "Nonlinear Solver Status", solver_status);
        appendDebugLine(stream, "Sequential Quadratic Programming Iterations", debug_info.sqp_iter);
        appendDebugLine(
            stream,
            "Quadratic Program Solver Status",
            debug_info.qp_status_available ? std::to_string(debug_info.qp_status) : "not available");
        appendDebugLine(stream, "Quadratic Program Iterations", debug_info.qp_iter);
        appendDebugLine(stream, "Objective Cost Value", debug_info.cost_value);
        appendDebugLine(stream, "Stationarity Residual", debug_info.res_stat);
        appendDebugLine(stream, "Dynamics Equality Residual", debug_info.res_eq);
        appendDebugLine(stream, "Inequality Residual", debug_info.res_ineq);
        appendDebugLine(stream, "Complementarity Residual", debug_info.res_comp);
        appendDebugLine(stream, "Total Solver Time [ms]", 1.0e3 * debug_info.time_tot);
        appendDebugLine(stream, "Linearization Time [ms]", 1.0e3 * debug_info.time_lin);
        appendDebugLine(stream, "Quadratic Program Time [ms]", 1.0e3 * debug_info.time_qp);
        appendDebugLine(
            stream,
            "Active Soft Constraint Count",
            std::to_string(debug_info.active_slack_count) + " / " +
                std::to_string(debug_info.total_slack_constraints));
        appendDebugLine(
            stream,
            "Stages With Soft Constraint Activity",
            std::to_string(debug_info.active_stage_count) + " / " +
                std::to_string(nmpc_solver_.getN() + 1));
        appendDebugLine(stream, "Maximum Soft Constraint Violation", debug_info.max_slack);
        appendDebugLine(stream, "Accumulated Soft Constraint Violation", debug_info.sum_slack);
        appendDebugLine(
            stream,
            "Soft Constraint Activation Threshold",
            params_.acados_debug.slack_activation_threshold);
        if (params_.obstacle_distance_monitor.include_in_acados_report) {
            if (nearest_distance_info != nullptr && nearest_distance_info->available) {
                appendDebugLine(
                    stream,
                    "Nearest Sphere-Obstacle Clearance [m]",
                    nearest_distance_info->signed_distance);
                appendDebugLine(
                    stream,
                    "Nearest Sphere Pair",
                    nearest_distance_info->robot_sphere_name + " <-> " +
                        (nearest_distance_info->obstacle_is_dynamic ? "dynamic:" : "static:") +
                        nearest_distance_info->obstacle_name);
            } else {
                appendDebugLine(
                    stream,
                    "Nearest Sphere-Obstacle Clearance [m]",
                    "not available");
            }

            if (std::isfinite(min_recorded_nearest_obstacle_distance_)) {
                appendDebugLine(
                    stream,
                    "Minimum Recorded Sphere-Obstacle Clearance [m]",
                    min_recorded_nearest_obstacle_distance_);
                appendDebugLine(
                    stream,
                    "Minimum Recorded Sphere Pair",
                    min_recorded_robot_sphere_name_ + " <-> " +
                        (min_recorded_obstacle_is_dynamic_ ? "dynamic:" : "static:") +
                        min_recorded_obstacle_name_);
            }
        }

        if (params_.acados_debug.include_stage_slack_summary &&
            !debug_info.stage_slack_info.empty()) {
            auto stage_slack_info = debug_info.stage_slack_info;
            std::sort(
                stage_slack_info.begin(),
                stage_slack_info.end(),
                [](const NMPCDebugStageSlackInfo& lhs, const NMPCDebugStageSlackInfo& rhs) {
                    if (lhs.max_slack != rhs.max_slack) {
                        return lhs.max_slack > rhs.max_slack;
                    }
                    return lhs.stage < rhs.stage;
                });

            stream << subsection_divider << '\n';
            stream << "Soft Constraint Stage Summary\n";
            int reported_stage_count = 0;
            for (const auto& stage_info : stage_slack_info) {
                if (stage_info.max_slack <= params_.acados_debug.slack_activation_threshold &&
                    stage_info.active_count == 0) {
                    continue;
                }
                appendDebugLine(
                    stream,
                    "Prediction Stage " + std::to_string(stage_info.stage),
                    "maximum_violation=" + formatDebugFloat(stage_info.max_slack) +
                        ", accumulated_violation=" + formatDebugFloat(stage_info.sum_slack) +
                        ", active_constraints=" + std::to_string(stage_info.active_count) +
                        " / " + std::to_string(stage_info.slack_dim));
                ++reported_stage_count;
                if (reported_stage_count >= params_.acados_debug.max_reported_stages) {
                    break;
                }
            }
            if (reported_stage_count == 0) {
                appendDebugLine(stream, "Soft Constraint Stage Summary", "no active stage exceeded the threshold");
            }
        }

        stream << divider;

        if (solver_status == 0) {
            RCLCPP_INFO(this->get_logger(), "%s", stream.str().c_str());
        } else {
            RCLCPP_WARN(this->get_logger(), "%s", stream.str().c_str());
        }
    }

    void recordNearestDistanceMinimum(
        const NearestObstacleDistanceInfo& nearest_distance_info) {
        if (!nearest_distance_info.available) {
            return;
        }

        constexpr double kDistanceUpdateEpsilon = 1.0e-9;
        if (nearest_distance_info.signed_distance >=
            min_recorded_nearest_obstacle_distance_ - kDistanceUpdateEpsilon) {
            return;
        }

        min_recorded_nearest_obstacle_distance_ = nearest_distance_info.signed_distance;
        min_recorded_robot_sphere_name_ = nearest_distance_info.robot_sphere_name;
        min_recorded_obstacle_name_ = nearest_distance_info.obstacle_name;
        min_recorded_obstacle_is_dynamic_ = nearest_distance_info.obstacle_is_dynamic;
    }

    // =========================================================================
    // Result Publication and Main Control Loop
    // =========================================================================
    // These helpers form the real-time path: assemble data, solve NMPC, publish
    // the result and fall back to the last safe command when needed.
    void publishResult(const NMPCResult& result) {
        panda_interfaces::msg::ResultNMPC msg;
        Eigen::Map<Vec7>(msg.q_ref.data()) = result.q_ref;
        Eigen::Map<Vec7>(msg.v_ref.data()) = result.v_ref;
        Eigen::Map<Vec7>(msg.a_ref.data()) = result.a_ref;
        Eigen::Map<Vec7>(msg.jerk_cmd.data()) = result.jerk_cmd;
        msg.status = result.status;
        nmpc_res_pub_->publish(msg);
    }

    NMPCResult buildHoldResult(const Vec7& q_meas, int status_code) const {
        NMPCResult result;
        result.q_ref = q_meas;
        result.v_ref.setZero();
        result.a_ref.setZero();
        result.jerk_cmd.setZero();
        result.status = status_code;
        return result;
    }

    // Snapshot shared state, solve the current NMPC problem and handle failure
    // recovery in a bounded, easy-to-follow sequence.
    void timerCallback() {
        Vec3 target_pos = Vec3::Zero();
        Mat3 target_rot = Mat3::Identity();
        Vec7 q_meas = Vec7::Zero();
        Vec7 dq_meas = Vec7::Zero();
        Vec7 q_nominal = Vec7::Zero();
        std::vector<DynamicSphereState> dynamic_obstacles;
        rclcpp::Time dynamic_obstacle_stamp(0, 0, RCL_ROS_TIME);
        bool dynamic_obstacles_ready = false;
        PathTrackingSnapshot path_snapshot;

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (!joint_state_ready_ || !sorr_ready_) {
                return;
            }

            target_pos = target_pos_;
            target_rot = target_rot_;
            q_meas = q_meas_;
            dq_meas = dq_meas_;
            q_nominal = q_nominal_;
            dynamic_obstacles = dynamic_obstacles_;
            dynamic_obstacles_ready = dynamic_obstacles_ready_;
            dynamic_obstacle_stamp = dynamic_obstacle_stamp_;
            path_snapshot = snapshotPathTrackingState();
        }

        const bool dynamic_obstacles_fresh =
            params_.dynamic_obstacles.enabled &&
            dynamic_obstacles_ready &&
            hasFreshDynamicObstacles(dynamic_obstacle_stamp, dynamic_obstacles_ready);
        const bool should_monitor_nearest_distance =
            params_.obstacle_distance_monitor.enabled &&
            ((nearest_obstacle_distance_pub_ != nullptr &&
              nearest_obstacle_distance_pub_->get_subscription_count() > 0) ||
             params_.obstacle_distance_monitor.include_in_acados_report);

        Vec3 current_ee_pos = Vec3::Zero();
        Mat3 current_ee_rot = Mat3::Identity();
        NearestObstacleDistanceInfo nearest_distance_info;
        computeCurrentGeometrySnapshot(
            q_meas,
            current_ee_pos,
            current_ee_rot,
            dynamic_obstacles,
            dynamic_obstacles_fresh,
            should_monitor_nearest_distance ? &nearest_distance_info : nullptr);

        if (should_monitor_nearest_distance &&
            nearest_obstacle_distance_pub_ != nullptr &&
            nearest_obstacle_distance_pub_->get_subscription_count() > 0 &&
            nearest_distance_info.available) {
            Float64Msg distance_msg;
            distance_msg.data = nearest_distance_info.signed_distance;
            nearest_obstacle_distance_pub_->publish(distance_msg);
        }

        if (should_monitor_nearest_distance) {
            recordNearestDistanceMinimum(nearest_distance_info);
        }

        if (params_.dynamic_obstacles.enabled &&
            dynamic_obstacles_ready &&
            !dynamic_obstacles_fresh) {
            const double age_sec = (this->now() - dynamic_obstacle_stamp).seconds();
            const double timeout_sec = params_.dynamic_obstacles.timeout_sec;
            const std::size_t obstacle_count = dynamic_obstacles.size();
            // Keep the warning throttled, but include the measured age to make scheduler issues visible.
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Dynamic obstacle data is stale (age=%.3f s, timeout=%.3f s, obstacles=%zu). Falling back to static obstacle parameters.",
                age_sec, timeout_sec, obstacle_count);
        }

        const double dt = nmpc_solver_.getDt();
        const double total_terminal_length = params_.path_following.terminal_target_segment_length;
        SecondOrderReferenceRegulator candidate_sorr_pos = sorr_pos_;
        SecondOrderReferenceRegulator candidate_sorr_rot = sorr_rot_;

        bool use_path_tracking = false;
        bool use_waiting_reference = false;
        bool reset_sorr_from_current_pose = false;
        PathTrackingState updated_path_state = path_snapshot.state;
        double updated_reference_head_s = path_snapshot.reference_head_s;
        std::size_t updated_closest_index_hint = path_snapshot.closest_index_hint;

        if (params_.use_global_trajectory &&
            params_.path_following.enabled &&
            path_snapshot.waiting_for_fresh_path) {
            if (!path_snapshot.latched_wait_reference_valid) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 1000,
                    "Waiting for a fresh /global_joint_trajectory after a target update, but no latched wait reference is available. Falling back to a direct hold command.");
                consecutive_failure_count_ = 0;
                publishResult(buildHoldResult(q_meas, -998));
                return;
            }

            use_waiting_reference = true;
            RCLCPP_INFO_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "A new target pose has arrived and NMPC is waiting for a fresh /global_joint_trajectory. Solving against the latched current EE pose until MoveIt publishes the new path.");
        }

        if (!use_waiting_reference &&
            params_.use_global_trajectory &&
            params_.path_following.enabled &&
            path_snapshot.active_path) {
            if (updated_path_state == PathTrackingState::kFollowingPath) {
                updated_closest_index_hint = findClosestPathWaypointIndex(
                    *path_snapshot.active_path,
                    current_ee_pos,
                    current_ee_rot,
                    updated_closest_index_hint);

                const double projected_path_s =
                    path_snapshot.active_path->waypoints[updated_closest_index_hint].path_s;
                const double desired_head_s = projected_path_s + params_.path_following.lookahead_distance;
                const double limited_head_s =
                    std::min(desired_head_s, updated_reference_head_s +
                                                params_.path_following.max_head_advance_speed * dt);
                updated_reference_head_s = std::max(projected_path_s, limited_head_s);

                if (updated_reference_head_s >=
                    path_snapshot.active_path->total_path_s + total_terminal_length -
                        params_.path_following.completion_path_tolerance) {
                    updated_path_state = PathTrackingState::kTerminalTarget;
                }
            }

            if (updated_path_state == PathTrackingState::kTerminalTarget) {
                const double target_position_error = (current_ee_pos - target_pos).norm();
                const double target_orientation_error =
                    rotationAngularDistance(current_ee_rot, target_rot);
                if (target_position_error <= params_.path_following.completion_position_tolerance &&
                    target_orientation_error <= params_.path_following.completion_orientation_tolerance) {
                    updated_path_state = PathTrackingState::kTargetOnly;
                    reset_sorr_from_current_pose = true;
                }
            }

            use_path_tracking = updated_path_state == PathTrackingState::kFollowingPath;
        }

        if (reset_sorr_from_current_pose) {
            candidate_sorr_pos.resetPosition(current_ee_pos);
            candidate_sorr_rot.resetOrientation(Quat(current_ee_rot));
        }

        const std::vector<NMPCStageReference>* stage_refs = nullptr;
        if (use_waiting_reference) {
            stage_refs = &buildUniformStageReferences(
                path_snapshot.latched_wait_target_pos,
                path_snapshot.latched_wait_target_rot,
                path_snapshot.latched_wait_q_nominal,
                Vec3::Zero(),
                Vec3::Zero(),
                dynamic_obstacles,
                dynamic_obstacles_ready,
                dynamic_obstacle_stamp);
        } else if (use_path_tracking && path_snapshot.active_path) {
            stage_refs = &buildPathTrackingStageReferences(
                *path_snapshot.active_path,
                updated_reference_head_s,
                target_pos,
                target_rot,
                dynamic_obstacles,
                dynamic_obstacles_ready,
                dynamic_obstacle_stamp);
        } else if (updated_path_state == PathTrackingState::kTerminalTarget &&
                   path_snapshot.active_path) {
            stage_refs = &buildUniformStageReferences(
                target_pos,
                target_rot,
                path_snapshot.active_path->terminal_q,
                Vec3::Zero(),
                Vec3::Zero(),
                dynamic_obstacles,
                dynamic_obstacles_ready,
                dynamic_obstacle_stamp);
        } else {
            const Vec3 filtered_target_pos = candidate_sorr_pos.updatePosition(target_pos);
            const Quat filtered_target_quat = candidate_sorr_rot.updateOrientation(target_rot);
            const Mat3 filtered_target_rot = filtered_target_quat.toRotationMatrix();
            stage_refs = &buildUniformStageReferences(
                filtered_target_pos,
                filtered_target_rot,
                q_nominal,
                candidate_sorr_pos.getLinearVelocity(),
                candidate_sorr_rot.getAngularVelocity(),
                dynamic_obstacles,
                dynamic_obstacles_ready,
                dynamic_obstacle_stamp);
        }

        const NMPCResult result = nmpc_solver_.NMPCSolveStageReferences(
            *stage_refs,
            q_meas,
            dq_meas);

        maybeLogAcadosDebug(
            result.status,
            should_monitor_nearest_distance ? &nearest_distance_info : nullptr);

        if (result.status == 0) {
            consecutive_failure_count_ = 0;
            last_valid_result_ = result;
            if (path_snapshot.active_path) {
                commitPathTrackingStateIfUnchanged(
                    path_snapshot,
                    updated_path_state,
                    updated_reference_head_s,
                    updated_closest_index_hint);
            }
            sorr_pos_ = candidate_sorr_pos;
            sorr_rot_ = candidate_sorr_rot;
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

        publishResult(buildHoldResult(q_meas, -999));
    }

    // =========================================================================
    // ROS Topic Callbacks
    // =========================================================================
    // All subscriptions feed the shared runtime state here. Each callback keeps
    // its validation local before touching the shared cache.
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

        Vec7 q_meas = Vec7::Zero();
        Vec7 dq_meas = Vec7::Zero();
        for (std::size_t i = 0; i < ordered_names_.size(); ++i) {
            const auto it = std::find(msg->name.begin(), msg->name.end(), ordered_names_[i]);
            if (it == msg->name.end()) {
                RCLCPP_ERROR_THROTTLE(
                    this->get_logger(), *this->get_clock(), 1000,
                    "Joint %s not found in joint_states", ordered_names_[i].c_str());
                return;
            }

            const std::size_t idx = static_cast<std::size_t>(std::distance(msg->name.begin(), it));
            q_meas(static_cast<Eigen::Index>(i)) = msg->position[idx];
            dq_meas(static_cast<Eigen::Index>(i)) = (idx < msg->velocity.size()) ? msg->velocity[idx] : 0.0;
        }

        std::lock_guard<std::mutex> lock(state_mutex_);
        q_meas_ = q_meas;
        dq_meas_ = dq_meas;
        joint_state_ready_ = true;

        if (params_.use_current_q_as_nominal_on_startup && !nominal_initialized_from_state_) {
            q_nominal_ = q_meas_;
            nominal_initialized_from_state_ = true;
        }

        if (!sorr_ready_) {
            initializeSorrFromCurrentEePose();
        }
    }

    // Accept a new target pose only after lightweight sanity checks so a bad UI
    // message cannot poison the controller state.
    void targetPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        const Vec3 position(
            msg->pose.position.x,
            msg->pose.position.y,
            msg->pose.position.z);
        const Quat quat(
            msg->pose.orientation.w,
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z);

        const auto finite = [](double value) { return std::isfinite(value); };
        if (!finite(position.x()) || !finite(position.y()) || !finite(position.z())) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Received invalid target pose with non-finite position. Ignoring message.");
            return;
        }
        if (!finite(quat.w()) || !finite(quat.x()) || !finite(quat.y()) || !finite(quat.z())) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Received invalid target pose with non-finite quaternion. Ignoring message.");
            return;
        }

        if (params_.target_pose_validation.enabled) {
            const auto& validation = params_.target_pose_validation;
            if (!msg->header.frame_id.empty() && !validation.expected_frame_id.empty() &&
                msg->header.frame_id != validation.expected_frame_id) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 1000,
                    "Received target pose in frame '%s' but expected '%s'. Ignoring message.",
                    msg->header.frame_id.c_str(),
                    validation.expected_frame_id.c_str());
                return;
            }

            if (msg->header.frame_id.empty() && !validation.allow_empty_frame_id) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 1000,
                    "Received target pose with empty frame_id while target_pose_validation.allow_empty_frame_id=false. Ignoring message.");
                return;
            }

            if (position.norm() > validation.max_position_norm) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 1000,
                    "Received target pose with position norm %.3f > %.3f. Ignoring message.",
                    position.norm(),
                    validation.max_position_norm);
                return;
            }

            if (quat.norm() < validation.min_quaternion_norm) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 1000,
                    "Received invalid target quaternion with norm %.3e < %.3e. Ignoring message.",
                    quat.norm(),
                    validation.min_quaternion_norm);
                return;
            }
        } else if (quat.norm() < 1.0e-9) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Received invalid target quaternion, ignoring message.");
            return;
        }

        const rclcpp::Time target_update_stamp = this->now();
        const Mat3 normalized_target_rot = quat.normalized().toRotationMatrix();

        bool should_wait_for_fresh_path = false;
        bool have_latched_wait_reference = false;
        Vec7 latched_wait_q = Vec7::Zero();
        Vec3 latched_wait_pos = Vec3::Zero();
        Mat3 latched_wait_rot = Mat3::Identity();

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            should_wait_for_fresh_path =
                params_.use_global_trajectory &&
                params_.path_following.enabled &&
                params_.path_following.wait_for_fresh_path_after_target_update;
            if (should_wait_for_fresh_path && joint_state_ready_) {
                latched_wait_q = q_meas_;
                have_latched_wait_reference = true;
            }
        }

        if (have_latched_wait_reference) {
            computeEePoseForConfiguration(latched_wait_q, latched_wait_pos, latched_wait_rot);
        }

        std::lock_guard<std::mutex> lock(state_mutex_);
        target_pos_ = position;
        target_rot_ = normalized_target_rot;
        last_target_pose_update_stamp_ = target_update_stamp;

        if (should_wait_for_fresh_path) {
            resetPathTrackingStateLocked();
            waiting_for_fresh_path_after_target_update_ = true;
            latched_wait_reference_valid_ = have_latched_wait_reference;
            if (have_latched_wait_reference) {
                latched_wait_target_pos_ = latched_wait_pos;
                latched_wait_target_rot_ = latched_wait_rot;
                latched_wait_q_nominal_ = latched_wait_q;
            } else {
                latched_wait_target_pos_ = position;
                latched_wait_target_rot_ = normalized_target_rot;
                latched_wait_q_nominal_ = q_nominal_;
            }
        }
    }

    // Cache the latest moving obstacles so the timer loop can do stage-wise
    // prediction without any ROS work in the hot path.
    void dynamicObstacleCallback(const DynamicSphereArrayMsg::SharedPtr msg) {
        std::vector<DynamicSphereState> dynamic_obstacles;
        rclcpp::Time dynamic_obstacle_stamp(0, 0, RCL_ROS_TIME);
        if (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0) {
            dynamic_obstacle_stamp = this->now();
        } else {
            dynamic_obstacle_stamp = rclcpp::Time(msg->header.stamp);
        }

        const std::size_t dynamic_slot_count =
            PandaNMPCController::kNumRuntimeObstacles - static_obstacle_slot_count_;
        if (dynamic_slot_count == 0) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Static obstacles already occupy all %zu solver slots. Dynamic obstacles are ignored.",
                PandaNMPCController::kNumRuntimeObstacles);
            std::lock_guard<std::mutex> lock(state_mutex_);
            dynamic_obstacles_.clear();
            dynamic_obstacle_stamp_ = dynamic_obstacle_stamp;
            dynamic_obstacles_ready_ = false;
            return;
        }

        if (msg->spheres.size() > dynamic_slot_count) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Received %zu dynamic obstacle(s) but only %zu dynamic slot(s) are free after reserving %zu static slot(s). Extra obstacles are ignored.",
                msg->spheres.size(),
                dynamic_slot_count,
                static_obstacle_slot_count_);
        }

        const std::size_t obstacle_count = std::min(msg->spheres.size(), dynamic_slot_count);
        dynamic_obstacles.reserve(obstacle_count);
        for (std::size_t obstacle_idx = 0; obstacle_idx < obstacle_count; ++obstacle_idx) {
            const auto& sphere = msg->spheres[obstacle_idx];
            if (!(sphere.radius > 0.0)) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(), *this->get_clock(), 1000,
                    "Ignoring dynamic obstacle '%s' with non-positive radius %.6f.",
                    sphere.name.c_str(),
                    sphere.radius);
                continue;
            }

            DynamicSphereState state;
            state.name = sphere.name;
            state.center << sphere.center.x, sphere.center.y, sphere.center.z;
            state.velocity << sphere.velocity.x, sphere.velocity.y, sphere.velocity.z;
            state.radius = sphere.radius;
            dynamic_obstacles.push_back(state);
        }

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            dynamic_obstacles_ = dynamic_obstacles;
            dynamic_obstacle_stamp_ = dynamic_obstacle_stamp;
            dynamic_obstacles_ready_ = !dynamic_obstacles_.empty();
        }

        // std::ostringstream stream;
        // stream << "Updated " << dynamic_obstacles.size()
        //        << " dynamic obstacle(s); static obstacles keep the first "
        //        << static_obstacle_slot_count_ << " slot(s)";
        // if (dynamic_prediction_mode_ == DynamicObstaclePredictionMode::kConstantVelocity) {
        //     stream << " using constant-velocity prediction.";
        // } else {
        //     stream << " using zero-order-hold prediction.";
        // }
        // RCLCPP_INFO_THROTTLE(
        //     this->get_logger(), *this->get_clock(), 2000, "%s", stream.str().c_str());
    }

    void jointTrajectoryCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
        if (!params_.use_global_trajectory || !params_.path_following.enabled) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "Received /global_joint_trajectory but path following is disabled. Set use_global_trajectory=true and path_following.enabled=true to consume MoveIt paths.");
            return;
        }

        const rclcpp::Time trajectory_stamp =
            (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
                ? this->now()
                : rclcpp::Time(msg->header.stamp);

        const auto path = buildJointPathFromTrajectory(*msg);
        if (!path || path->waypoints.empty()) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 2000,
                "Received an invalid or empty JointTrajectory. Path update is ignored.");
            return;
        }

        Vec3 current_ee_pos = Vec3::Zero();
        Mat3 current_ee_rot = Mat3::Identity();
        Vec7 current_q = Vec7::Zero();
        bool have_current_state = false;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (joint_state_ready_) {
                have_current_state = true;
                current_q = q_meas_;
            }
        }
        if (have_current_state) {
            computeEePoseForConfiguration(current_q, current_ee_pos, current_ee_rot);
        }

        std::size_t initial_closest_index = 0;
        double initial_projected_s = 0.0;
        if (have_current_state) {
            initial_closest_index = findClosestPathWaypointIndex(*path, current_ee_pos, current_ee_rot, 0);
            initial_projected_s = path->waypoints[initial_closest_index].path_s;
        }

        bool rejected_as_stale = false;
        rclcpp::Time latest_target_stamp(0, 0, RCL_ROS_TIME);
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            latest_target_stamp = last_target_pose_update_stamp_;
            if (waiting_for_fresh_path_after_target_update_ &&
                params_.path_following.wait_for_fresh_path_after_target_update &&
                trajectory_stamp < last_target_pose_update_stamp_) {
                rejected_as_stale = true;
            }
            if (!rejected_as_stale) {
                active_joint_path_ = path;
                path_tracking_state_ = PathTrackingState::kFollowingPath;
                path_closest_index_hint_ = initial_closest_index;
                path_reference_head_s_ = std::max(
                    0.0,
                    initial_projected_s + params_.path_following.lookahead_distance);
                path_terminal_q_nominal_ = path->terminal_q;
                waiting_for_fresh_path_after_target_update_ = false;
            }
        }

        if (rejected_as_stale) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(), *this->get_clock(), 1000,
                "Ignored a stale /global_joint_trajectory (stamp=%.3f) because the latest target pose was updated at %.3f and NMPC is waiting for a fresh path.",
                trajectory_stamp.seconds(),
                latest_target_stamp.seconds());
            return;
        }

        RCLCPP_INFO(
            this->get_logger(),
            "Loaded global joint path with %zu waypoint(s), path_length=%.4f, initial_head_s=%.4f, terminal_q_from_path=true",
            path->waypoints.size(),
            path->total_path_s,
            initial_projected_s + params_.path_following.lookahead_distance);
    }

public:
    // =========================================================================
    // Node Construction
    // =========================================================================
    // The constructor wires together parameters, robot model, controller
    // settings and ROS interfaces. No NMPC logic should leak into this section.
    NMPCNode() : Node("nmpc_tau_node") {
        // 1) Pull all runtime configuration from ROS parameters.
        declareStartupParameters();

        // 2) Push configuration into the acados wrapper before any solve call.
        nmpc_solver_.setCostWeights(params_.cost_weights);
        nmpc_solver_.setHardLimits(params_.hard_limits);
        nmpc_solver_.setObstacleConstraintConfig(params_.obstacle_constraint);

        target_pos_ = params_.target_pos;
        target_rot_ = params_.target_rot;
        path_terminal_q_nominal_ = q_nominal_;

        // 3) Initialize model-side resources and reusable reference buffers.
        initializePinocchio();
        loadStaticObstacles();
        setSorrConfig();
        stage_refs_buffer_.resize(static_cast<std::size_t>(nmpc_solver_.getN() + 1));

        // 4) Prepare a conservative default output until the first successful solve.
        last_valid_result_.status = -1;
        last_valid_result_.q_ref = q_nominal_;
        last_valid_result_.v_ref.setZero();
        last_valid_result_.a_ref.setZero();
        last_valid_result_.jerk_cmd.setZero();

        // 5) Separate callbacks into groups so subscriptions stay responsive
        // while the control timer is solving.
        timer_callback_group_ =
            this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        state_callback_group_ =
            this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        dynamic_obstacle_callback_group_ =
            this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        rclcpp::SubscriptionOptions state_sub_options;
        state_sub_options.callback_group = state_callback_group_;

        joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            params_.joint_states_topic, 10,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) { jointStateCallback(msg); },
            state_sub_options);

        joint_trajectory_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
            params_.joint_trajectory_topic, 10,
            [this](const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) { jointTrajectoryCallback(msg); },
            state_sub_options);

        target_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            params_.target_pose_topic, 10,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { targetPoseCallback(msg); },
            state_sub_options);

        if (params_.dynamic_obstacles.enabled) {
            rclcpp::SubscriptionOptions dynamic_obstacle_sub_options;
            dynamic_obstacle_sub_options.callback_group = dynamic_obstacle_callback_group_;
            dynamic_obstacle_sub_ = this->create_subscription<DynamicSphereArrayMsg>(
                params_.dynamic_obstacles.topic, 10,
                [this](const DynamicSphereArrayMsg::SharedPtr msg) { dynamicObstacleCallback(msg); },
                dynamic_obstacle_sub_options);
        }

        // 6) Start the output publisher and fixed-rate control timer.
        nmpc_res_pub_ =
            this->create_publisher<panda_interfaces::msg::ResultNMPC>(params_.nmpc_result_topic, 1);
        if (params_.obstacle_distance_monitor.enabled) {
            nearest_obstacle_distance_pub_ =
                this->create_publisher<Float64Msg>(params_.obstacle_distance_monitor.topic, 1);
        }
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int64_t>(1000.0 * nmpc_solver_.getDt())),
            [this]() { timerCallback(); },
            timer_callback_group_);

        RCLCPP_INFO(
            this->get_logger(),
            "NMPC node started. urdf=%s, ee_frame=%s, static_obstacle_source=%s, dynamic_obstacles=%s, dynamic_topic=%s, dynamic_prediction=%s, joint_states_topic=%s, target_pose_topic=%s, result_topic=%s, joint_trajectory_topic=%s, use_global_trajectory=%s, path_following=%s, acados_debug=%s, nearest_obstacle_distance=%s, nearest_distance_topic=%s, executor=MultiThreadedExecutor",
            params_.urdf_path.c_str(),
            params_.ee_frame_name.c_str(),
            selectObstacleSourcePath().c_str(),
            params_.dynamic_obstacles.enabled ? "enabled" : "disabled",
            params_.dynamic_obstacles.topic.c_str(),
            params_.dynamic_obstacles.prediction_mode.c_str(),
            params_.joint_states_topic.c_str(),
            params_.target_pose_topic.c_str(),
            params_.nmpc_result_topic.c_str(),
            params_.joint_trajectory_topic.c_str(),
            params_.use_global_trajectory ? "true" : "false",
            params_.path_following.enabled ? "enabled" : "disabled",
            params_.acados_debug.enabled ? "enabled" : "disabled",
            params_.obstacle_distance_monitor.enabled ? "enabled" : "disabled",
            params_.obstacle_distance_monitor.topic.c_str());
    }
};

// ============================================================================
// Program Entry Point
// ============================================================================
// Use a MultiThreadedExecutor so the timer callback does not block state and
// obstacle subscriptions.
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NMPCNode>();
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3);
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}
