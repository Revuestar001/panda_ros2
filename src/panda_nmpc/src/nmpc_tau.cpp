#include <array>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "panda_interfaces/msg/result_nmpc.hpp"
#include "panda_nmpc_controller.hpp"
#include "static_sphere_scene.hpp"
#include "second_order_reference_regulator.hpp"

class NMPCNode : public rclcpp::Node {
private:
    using Vec7 = Eigen::Matrix<double, 7, 1>;
    using Vec3 = Eigen::Vector3d;
    using Mat3 = Eigen::Matrix3d;
    using ObstacleParamBlock = PandaNMPCController::ObstacleParamBlock;
    using CostWeights = PandaNMPCController::CostWeights;

    struct TaskSpacePathSample {
        double path_s{0.0};
        Vec3 pos{Vec3::Zero()};
        Vec7 q{Vec7::Zero()};
    };

    enum class JointTrajectoryState {
        kTrajectoryNull = 0,
        kTrajectoryReady,
        kTrajectoryUsing,
        kTrajectoryComplete,
    };

    // nmpc
    PandaNMPCController nmpc_solver_;
    NMPCResult nmpc_res_last_;

    // sorr
    SecondOrderReferenceRegulator sorr_pos_;
    SecondOrderReferenceRegulator sorr_rot_;

    Eigen::Vector3d target_pos_;
    Eigen::Matrix3d target_rot_;
    Eigen::VectorXd initial_q_;
    Eigen::VectorXd reference_q_;
    Eigen::VectorXd jq_;
    Eigen::VectorXd jv_;

    std::string urdf_path_;
    std::string obstacle_config_path_;
    std::string scene_xml_path_;
    std::string ee_frame_name_;
    std::string joint_states_topic_;
    std::string joint_trajectory_topic_;
    std::string nmpc_result_topic_;
    std::string target_pose_topic_;

    // global joint trajectory
    trajectory_msgs::msg::JointTrajectory joint_trajectory_;
    JointTrajectoryState trajectory_state_{JointTrajectoryState::kTrajectoryNull};
    double last_joint_trajectory_start_time_{-1.0};
    std::array<int, 7> trajectory_joint_index_map_{};

    // [修改] 预先把 MoveIt 关节轨迹转成任务空间路径样本；NMPC 后续按“路径进度”而不是“绝对时间”取参考。
    std::vector<TaskSpacePathSample> task_space_path_samples_;
    std::size_t task_space_path_progress_index_{0};
    double trajectory_path_stage_spacing_{0.015};

    double ee_frame_pos_tolerance_{1e-3};
    double ee_frame_rot_tolerance_{1e-1};
    double ee_frame_lin_vel_tolerance_{2e-2};
    double ee_frame_ang_vel_tolerance_{1e-1};

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
    bool use_global_trajectory_{false};

    double node_initial_time_;

    // pinocchio
    pinocchio::Model pin_model_;
    std::unique_ptr<pinocchio::Data> pin_data_;
    pinocchio::FrameIndex ee_frame_id_{0};
    std::vector<panda_nmpc::StaticSphereObstacle> scene_obstacles_;
    ObstacleParamBlock runtime_obstacle_params_{PandaNMPCController::disabledObstacleParams()};
    CostWeights nmpc_cost_weights_{PandaNMPCController::defaultCostWeights()};
    std::vector<NMPCStageReference> stage_refs_buffer_;

    // topic & timer
    rclcpp::Publisher<panda_interfaces::msg::ResultNMPC>::SharedPtr nmpc_res_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr joint_trajectory_sub_;
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

    static std::string default_scene_xml_path() {
        return "/home/cyh/panda_ros2/model/franka_emika_panda/scene_tau_ros.xml";
    }

    static std::string default_obstacle_config_path() {
        return "/home/cyh/panda_ros2/model/franka_emika_panda/static_sphere_obstacles.xml";
    }

    template <std::size_t N>
    static std::vector<double> to_parameter_vector(const std::array<double, N>& values) {
        return std::vector<double>(values.begin(), values.end());
    }

    template <std::size_t N>
    std::array<double, N> declare_fixed_size_array_parameter(
        const std::string& param_name,
        const std::array<double, N>& defaults) {
        const auto raw_values =
            this->declare_parameter<std::vector<double>>(param_name, to_parameter_vector(defaults));
        if (raw_values.size() != N) {
            throw std::runtime_error(
                "Parameter '" + param_name + "' must contain exactly " + std::to_string(N) + " values.");
        }

        std::array<double, N> values{};
        std::copy(raw_values.begin(), raw_values.end(), values.begin());
        return values;
    }

    void load_static_obstacles_from_scene() {
        runtime_obstacle_params_ = PandaNMPCController::disabledObstacleParams();
        const std::string obstacle_source_path =
            obstacle_config_path_.empty() ? scene_xml_path_ : obstacle_config_path_;
        scene_obstacles_ = panda_nmpc::loadStaticSphereObstaclesFromSceneXml(obstacle_source_path);

        if (scene_obstacles_.empty()) {
            RCLCPP_WARN(
                this->get_logger(),
                "No static sphere obstacles found in '%s'. NMPC obstacle constraints are disabled.",
                obstacle_source_path.c_str());
            return;
        }

        if (scene_obstacles_.size() > PandaNMPCController::kNumRuntimeObstacles) {
            RCLCPP_WARN(
                this->get_logger(),
                "Obstacle config defines %zu sphere obstacles but solver only supports %zu. Extra obstacles will be ignored until acados is regenerated.",
                scene_obstacles_.size(),
                PandaNMPCController::kNumRuntimeObstacles);
        }

        const std::size_t obstacle_count =
            std::min(scene_obstacles_.size(), PandaNMPCController::kNumRuntimeObstacles);
        for (std::size_t obstacle_idx = 0; obstacle_idx < obstacle_count; ++obstacle_idx) {
            const std::size_t offset = obstacle_idx * 4;
            const auto& obstacle = scene_obstacles_[obstacle_idx];
            runtime_obstacle_params_[offset + 0] = obstacle.center.x();
            runtime_obstacle_params_[offset + 1] = obstacle.center.y();
            runtime_obstacle_params_[offset + 2] = obstacle.center.z();
            runtime_obstacle_params_[offset + 3] = obstacle.radius;
        }
    }

    void declare_startup_parameters() {
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

        target_pos_.x() = this->declare_parameter<double>("target_x", 0.25);
        target_pos_.y() = this->declare_parameter<double>("target_y", 0.3);
        target_pos_.z() = this->declare_parameter<double>("target_z", 0.25);

        const double target_qx = this->declare_parameter<double>("target_qx", 1.0);
        const double target_qy = this->declare_parameter<double>("target_qy", 0.0);
        const double target_qz = this->declare_parameter<double>("target_qz", 0.0);
        const double target_qw = this->declare_parameter<double>("target_qw", 0.0);
        const Eigen::Quaterniond target_quat(target_qw, target_qx, target_qy, target_qz);
        target_rot_ = target_quat.normalized().toRotationMatrix();

        use_global_trajectory_ = this->declare_parameter<bool>("use_global_trajectory", false);

        ee_frame_pos_tolerance_ = this->declare_parameter<double>("ee_frame_pos_tolerance", 1.0e-3);
        ee_frame_rot_tolerance_ = this->declare_parameter<double>("ee_frame_rot_tolerance", 1.0e-1);
        ee_frame_lin_vel_tolerance_ = this->declare_parameter<double>("ee_frame_lin_vel_tolerance", 2.0e-2);
        ee_frame_ang_vel_tolerance_ = this->declare_parameter<double>("ee_frame_ang_vel_tolerance", 1.0e-1);

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

        const CostWeights default_cost_weights = PandaNMPCController::defaultCostWeights();
        nmpc_cost_weights_.pos = declare_fixed_size_array_parameter(
            "nmpc_weights.stage.pos", default_cost_weights.pos);
        nmpc_cost_weights_.rot = declare_fixed_size_array_parameter(
            "nmpc_weights.stage.rot", default_cost_weights.rot);
        nmpc_cost_weights_.ee_lin_vel = declare_fixed_size_array_parameter(
            "nmpc_weights.stage.ee_lin_vel", default_cost_weights.ee_lin_vel);
        nmpc_cost_weights_.ee_ang_vel = declare_fixed_size_array_parameter(
            "nmpc_weights.stage.ee_ang_vel", default_cost_weights.ee_ang_vel);
        nmpc_cost_weights_.q_reg = declare_fixed_size_array_parameter(
            "nmpc_weights.stage.q_reg", default_cost_weights.q_reg);
        nmpc_cost_weights_.dq_reg = declare_fixed_size_array_parameter(
            "nmpc_weights.stage.dq_reg", default_cost_weights.dq_reg);
        nmpc_cost_weights_.ddq_reg = declare_fixed_size_array_parameter(
            "nmpc_weights.stage.ddq_reg", default_cost_weights.ddq_reg);

        nmpc_cost_weights_.pos_e = declare_fixed_size_array_parameter(
            "nmpc_weights.terminal.pos", default_cost_weights.pos_e);
        nmpc_cost_weights_.rot_e = declare_fixed_size_array_parameter(
            "nmpc_weights.terminal.rot", default_cost_weights.rot_e);
        nmpc_cost_weights_.ee_lin_vel_e = declare_fixed_size_array_parameter(
            "nmpc_weights.terminal.ee_lin_vel", default_cost_weights.ee_lin_vel_e);
        nmpc_cost_weights_.ee_ang_vel_e = declare_fixed_size_array_parameter(
            "nmpc_weights.terminal.ee_ang_vel", default_cost_weights.ee_ang_vel_e);
        nmpc_cost_weights_.q_reg_e = declare_fixed_size_array_parameter(
            "nmpc_weights.terminal.q_reg", default_cost_weights.q_reg_e);
        nmpc_cost_weights_.dq_reg_e = declare_fixed_size_array_parameter(
            "nmpc_weights.terminal.dq_reg", default_cost_weights.dq_reg_e);
        nmpc_cost_weights_.soft_constraint_stage.slack_linear =
            this->declare_parameter<double>(
                "nmpc_weights.soft_constraint.stage.slack_linear",
                default_cost_weights.soft_constraint_stage.slack_linear);
        nmpc_cost_weights_.soft_constraint_stage.slack_quadratic =
            this->declare_parameter<double>(
                "nmpc_weights.soft_constraint.stage.slack_quadratic",
                default_cost_weights.soft_constraint_stage.slack_quadratic);
        nmpc_cost_weights_.soft_constraint_terminal.slack_linear =
            this->declare_parameter<double>(
                "nmpc_weights.soft_constraint.terminal.slack_linear",
                default_cost_weights.soft_constraint_terminal.slack_linear);
        nmpc_cost_weights_.soft_constraint_terminal.slack_quadratic =
            this->declare_parameter<double>(
                "nmpc_weights.soft_constraint.terminal.slack_quadratic",
                default_cost_weights.soft_constraint_terminal.slack_quadratic);

        // [修改] 用路径弧长而不是 time_from_start 推 horizon 参考。
        trajectory_path_stage_spacing_ =
            this->declare_parameter<double>("trajectory_path_stage_spacing", 0.015);
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

        // 用当前实际末端状态初始化 SORR，避免第一次阶跃直接穿透到目标。
        sorr_pos_.resetPosition(ee_pos);
        sorr_rot_.resetOrientation(Eigen::Quaterniond(ee_rot));
        sorr_ready_ = true;
        node_initial_time_ = this->now().seconds();

        RCLCPP_INFO(this->get_logger(), "SORR initialized from current EE pose.");
    }

    // [修改] 统一管理“真正想达到的末端目标 pose”：
    // 始终使用当前存储的 target_pos_/target_rot_，不再退回到 MoveIt 轨迹末端 FK。
    void get_desired_goal_pose(Vec3& goal_pos, Mat3& goal_rot) const {
        goal_pos = target_pos_;
        goal_rot = target_rot_;
    }

    // [修改] 在全局路径模式下，中间 stage 的姿态参考不再直接跟随 MoveIt 的 sample.rot，
    // 而是沿“当前姿态 -> 最终目标姿态”做渐进插值，从而只借用 MoveIt 的位置路径，不借用其整段姿态时间表。
    Mat3 build_progress_based_orientation_reference(
        const Mat3& current_ee_rot,
        double query_s) const {
        Vec3 goal_pos = Vec3::Zero();
        Mat3 goal_rot = Mat3::Identity();
        get_desired_goal_pose(goal_pos, goal_rot);

        const double total_s =
            task_space_path_samples_.empty() ? 0.0 : task_space_path_samples_.back().path_s;
        const double alpha =
            (total_s > 1.0e-9) ? std::clamp(query_s / total_s, 0.0, 1.0) : 1.0;

        const Eigen::Quaterniond q_curr(current_ee_rot);
        const Eigen::Quaterniond q_goal(goal_rot);
        return q_curr.slerp(alpha, q_goal).normalized().toRotationMatrix();
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

    // [修改] 统一的关节采样函数，供轨迹接收时预计算任务空间路径使用。
    void sample_joint_state_from_trajectory_point(
        const trajectory_msgs::msg::JointTrajectoryPoint& point,
        Vec7& q,
        Vec7& dq) const {
        q.setZero();
        dq.setZero();
        for (size_t joint = 0; joint < ordered_names_.size(); ++joint) {
            const int msg_index = trajectory_joint_index_map_[joint];
            const std::size_t idx = static_cast<std::size_t>(msg_index);
            q(static_cast<Eigen::Index>(joint)) = point.positions[idx];
            if (point.velocities.size() > idx) {
                dq(static_cast<Eigen::Index>(joint)) = point.velocities[idx];
            }
        }
    }

    // [修改] 路径进度只使用位置距离，彻底去掉 MoveIt 中间姿态对路径跟踪的影响。
    double task_space_metric(const Vec3& pos_a, const Vec3& pos_b) const {
        return (pos_a - pos_b).norm();
    }

    // [修改] horizon stage_refs 在实时环中反复复用，避免每个 control tick 重新分配 vector。
    std::vector<NMPCStageReference>& stage_ref_buffer() {
        const std::size_t required_size = static_cast<std::size_t>(nmpc_solver_.getN() + 1);
        if (stage_refs_buffer_.size() != required_size) {
            stage_refs_buffer_.resize(required_size);
        }
        return stage_refs_buffer_;
    }

    // [修改] 收到 MoveIt 轨迹后，只预计算一遍 task-space path samples。
    void build_task_space_path_samples_from_joint_trajectory() {
        task_space_path_samples_.clear();
        task_space_path_progress_index_ = 0;

        if (joint_trajectory_.points.empty()) {
            return;
        }

        task_space_path_samples_.reserve(joint_trajectory_.points.size());

        double cumulative_s = 0.0;
        Vec3 prev_pos = Vec3::Zero();
        bool has_prev = false;

        for (const auto& point : joint_trajectory_.points) {
            Vec7 q_stage = Vec7::Zero();
            Vec7 dq_stage = Vec7::Zero();
            sample_joint_state_from_trajectory_point(point, q_stage, dq_stage);

            pinocchio::forwardKinematics(pin_model_, *pin_data_, q_stage);
            pinocchio::updateFramePlacements(pin_model_, *pin_data_);
            const pinocchio::SE3& oMf_ee = pin_data_->oMf[ee_frame_id_];

            TaskSpacePathSample sample;
            sample.pos = oMf_ee.translation();
            sample.q = q_stage;

            if (has_prev) {
                cumulative_s += task_space_metric(prev_pos, sample.pos);
            }
            sample.path_s = cumulative_s;

            task_space_path_samples_.push_back(sample);
            prev_pos = sample.pos;
            has_prev = true;
        }
    }

    // [修改] 根据当前末端位置，在“尚未走过的路径后缀”里寻找最近样本，保证进度单调前进。
    std::size_t find_closest_path_sample_index(const Vec3& current_ee_pos) {
        if (task_space_path_samples_.empty()) {
            return 0;
        }

        std::size_t best_index = task_space_path_progress_index_;
        double best_metric = 1.0e18;

        for (std::size_t i = task_space_path_progress_index_; i < task_space_path_samples_.size(); ++i) {
            const auto& sample = task_space_path_samples_[i];
            const double metric = task_space_metric(current_ee_pos, sample.pos);
            if (metric < best_metric) {
                best_metric = metric;
                best_index = i;
            }
        }

        return best_index;
    }

    // [修改] 按路径弧长插值，并复用单调递增的 segment hint，
    // 避免每个 stage 都从路径起点重新线性扫描一遍。
    TaskSpacePathSample interpolate_task_space_sample_by_path_s(
        double query_s,
        std::size_t& segment_index_hint) const {
        if (task_space_path_samples_.empty()) {
            return TaskSpacePathSample{};
        }
        if (query_s <= task_space_path_samples_.front().path_s ||
            task_space_path_samples_.size() == 1) {
            segment_index_hint = 0;
            return task_space_path_samples_.front();
        }
        if (query_s >= task_space_path_samples_.back().path_s) {
            segment_index_hint = task_space_path_samples_.size() - 1;
            return task_space_path_samples_.back();
        }

        segment_index_hint = std::min(segment_index_hint, task_space_path_samples_.size() - 2);
        while (segment_index_hint + 1 < task_space_path_samples_.size() &&
               task_space_path_samples_[segment_index_hint + 1].path_s < query_s) {
            ++segment_index_hint;
        }

        const auto& sample_0 = task_space_path_samples_[segment_index_hint];
        const auto& sample_1 = task_space_path_samples_[segment_index_hint + 1];
        const double s0 = sample_0.path_s;
        const double s1 = sample_1.path_s;
        const double alpha = (s1 > s0) ? std::clamp((query_s - s0) / (s1 - s0), 0.0, 1.0) : 0.0;

        TaskSpacePathSample sample;
        sample.path_s = query_s;
        sample.pos = (1.0 - alpha) * sample_0.pos + alpha * sample_1.pos;
        sample.q = (1.0 - alpha) * sample_0.q + alpha * sample_1.q;
        return sample;
    }

    const std::vector<NMPCStageReference>& build_terminal_hold_stage_refs(const Vec7& current_q) {
        auto& stage_refs = stage_ref_buffer();

        Vec3 ee_goal_pos = Vec3::Zero();
        Mat3 ee_goal_rot = Mat3::Identity();
        get_desired_goal_pose(ee_goal_pos, ee_goal_rot);

        for (auto& stage_ref : stage_refs) {
            stage_ref.target_pos = ee_goal_pos;
            stage_ref.target_rot = ee_goal_rot;
            stage_ref.q_nom = current_q;  // [修改] 终端保持不再被 MoveIt goal q 或 ready pose 额外拉扯。
            stage_ref.ee_lin_vel_ref.setZero();
            stage_ref.ee_ang_vel_ref.setZero();
            stage_ref.obstacle_params = runtime_obstacle_params_;
        }

        return stage_refs;
    }

    // [修改] 用“路径进度 + 前视距离”生成整条 horizon 参考。
    const std::vector<NMPCStageReference>& build_path_progress_stage_refs(
        const Vec7& current_q,
        const Vec7& /*current_v*/) {
        const int N = nmpc_solver_.getN();
        auto& stage_refs = stage_ref_buffer();

        if (task_space_path_samples_.empty()) {
            Vec3 ee_pos = Vec3::Zero();
            Mat3 ee_rot = Mat3::Identity();
            compute_current_ee_pose(ee_pos, ee_rot);

            for (auto& stage_ref : stage_refs) {
                stage_ref.target_pos = ee_pos;
                stage_ref.target_rot = ee_rot;
                stage_ref.q_nom = current_q;
                stage_ref.ee_lin_vel_ref.setZero();
                stage_ref.ee_ang_vel_ref.setZero();
                stage_ref.obstacle_params = runtime_obstacle_params_;
            }
            return stage_refs;
        }

        Vec3 current_ee_pos = Vec3::Zero();
        Mat3 current_ee_rot = Mat3::Identity();
        compute_current_ee_pose(current_ee_pos, current_ee_rot);

        task_space_path_progress_index_ = find_closest_path_sample_index(current_ee_pos);
        const double current_path_s = task_space_path_samples_[task_space_path_progress_index_].path_s;
        std::size_t segment_index_hint = task_space_path_progress_index_;

        for (int stage = 0; stage <= N; ++stage) {
            const double query_s = current_path_s + stage * trajectory_path_stage_spacing_;
            const TaskSpacePathSample sample =
                interpolate_task_space_sample_by_path_s(query_s, segment_index_hint);

            auto& stage_ref = stage_refs[static_cast<std::size_t>(stage)];
            stage_ref.target_pos = sample.pos;
            stage_ref.target_rot =
                build_progress_based_orientation_reference(current_ee_rot, query_s);  // [修改] 不再跟随 MoveIt 的 sample.rot。

            // [修改] q_nom 改为当前关节角，作为纯局部平滑/正则项，不再引入与当前路径无关的 ready pose 偏置。
            stage_ref.q_nom = current_q;

            // [修改] 当前阶段先不给绝对时间意义上的末端速度参考，避免重新把 NMPC 拉回 time tracking。
            stage_ref.ee_lin_vel_ref.setZero();
            stage_ref.ee_ang_vel_ref.setZero();
            stage_ref.obstacle_params = runtime_obstacle_params_;
        }

        // [修改] 强制把 terminal stage 锚定到手动设置/外部订阅得到的最终目标位姿。
        // 这样 acados 的 terminal cost 始终盯住真实 goal，而不是盯住 MoveIt 路径前视点。
        Vec3 desired_goal_pos = Vec3::Zero();
        Mat3 desired_goal_rot = Mat3::Identity();
        get_desired_goal_pose(desired_goal_pos, desired_goal_rot);

        auto& terminal_ref = stage_refs.back();
        terminal_ref.target_pos = desired_goal_pos;
        terminal_ref.target_rot = desired_goal_rot;
        terminal_ref.q_nom = current_q;
        terminal_ref.ee_lin_vel_ref.setZero();
        terminal_ref.ee_ang_vel_ref.setZero();
        terminal_ref.obstacle_params = runtime_obstacle_params_;

        return stage_refs;
    }

    bool is_trajectory_tracking_complete(
        const Vec7& current_q,
        const Vec7& current_v
    ) {
        if (task_space_path_samples_.empty()) {
            return false;
        }

        // [修改] 不再要求“墙钟时间超过 MoveIt 轨迹总时长”；只要已经推进到路径末端附近，就按终点误差判断完成。
        if (task_space_path_progress_index_ + 1 < task_space_path_samples_.size()) {
            return false;
        }

        pinocchio::forwardKinematics(pin_model_, *pin_data_, current_q, current_v);
        pinocchio::computeJointJacobians(pin_model_, *pin_data_, current_q);
        pinocchio::updateFramePlacements(pin_model_, *pin_data_);

        Eigen::Matrix<double, 6, Eigen::Dynamic> J_lwa(6, pin_model_.nv);
        J_lwa.setZero();
        pinocchio::getFrameJacobian(
            pin_model_, *pin_data_, ee_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED, J_lwa);

        const pinocchio::SE3& oMf_current = pin_data_->oMf[ee_frame_id_];

        Vec3 desired_goal_pos = Vec3::Zero();
        Mat3 desired_goal_rot = Mat3::Identity();
        get_desired_goal_pose(desired_goal_pos, desired_goal_rot);
        const pinocchio::SE3 desired_goal_pose(desired_goal_rot, desired_goal_pos);

        const pinocchio::SE3 err_M = oMf_current.actInv(desired_goal_pose);
        const Eigen::Matrix<double, 6, 1> err = pinocchio::log6(err_M).toVector();
        const Eigen::Matrix<double, 6, 1> ee_twist = J_lwa.leftCols<7>() * current_v;

        return err.head<3>().norm() < ee_frame_pos_tolerance_ &&
               err.tail<3>().norm() < ee_frame_rot_tolerance_ &&
               ee_twist.head<3>().norm() < ee_frame_lin_vel_tolerance_ &&
               ee_twist.tail<3>().norm() < ee_frame_ang_vel_tolerance_;
    }

    void timer_callback() {
        if (!joint_state_ready_ || !sorr_ready_) {
            return;
        }

        const Vec7 current_q = jq_.head<7>();
        const Vec7 current_v = jv_.head<7>();

        auto res = nmpc_res_last_;

        if (!use_global_trajectory_) {
            // update_target();

            const Vec7 q_nom = jq_.head<7>();

            const Vec3 filtered_target_pos = sorr_pos_.updatePosition(target_pos_);
            const Eigen::Quaterniond filtered_target_quat = sorr_rot_.updateOrientation(target_rot_);
            const Mat3 filtered_target_rot = filtered_target_quat.toRotationMatrix();

            res = nmpc_solver_.NMPCSolve(
                filtered_target_pos,
                filtered_target_rot,
                reference_q_.head<7>(),     // 先继续使用当前关节角作为 q_nom，避免零空间突然拉扯
                sorr_pos_.getLinearVelocity(),
                sorr_rot_.getAngularVelocity(),
                runtime_obstacle_params_,
                current_q,
                current_v
            );
        } else {
            // 使用全局关节参考轨迹
            switch (trajectory_state_) {
                case JointTrajectoryState::kTrajectoryNull :
                    RCLCPP_WARN_THROTTLE(
                        this->get_logger(), *this->get_clock(), 1000,
                        "Joints trajectory is null, hold on.");
                    break;
                case JointTrajectoryState::kTrajectoryReady :
                    last_joint_trajectory_start_time_ = this->now().seconds();
                    trajectory_state_ = JointTrajectoryState::kTrajectoryUsing;
                    [[fallthrough]];
                case JointTrajectoryState::kTrajectoryUsing :
                {
                    // [修改] 由“按 time_duration 重采样”改成“按当前路径进度生成前视参考”。
                    res = nmpc_solver_.NMPCSolveTrajectory(
                        build_path_progress_stage_refs(current_q, current_v), current_q, current_v);
                    if (is_trajectory_tracking_complete(current_q, current_v)) {
                        trajectory_state_ = JointTrajectoryState::kTrajectoryComplete;
                        RCLCPP_INFO(this->get_logger(), "Global path tracking completed, switching to terminal hold.");
                    }
                    break;
                }
                case JointTrajectoryState::kTrajectoryComplete :
                    res = nmpc_solver_.NMPCSolveTrajectory(
                        build_terminal_hold_stage_refs(current_q), current_q, current_v);
                    break;
            }
        }

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
        } else {
            target_rot_ = target_quat.normalized().toRotationMatrix();
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

    void joint_trajectory_callback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
        if (msg->points.size() < 2) {
            RCLCPP_WARN(this->get_logger(), "Invalid joints trajectory, size less than 2.");
            return;
        }

        trajectory_joint_index_map_.fill(-1);
        for (size_t i = 0; i < ordered_names_.size(); ++i) {
            auto it = std::find(msg->joint_names.begin(), msg->joint_names.end(), ordered_names_[i]);
            if (it == msg->joint_names.end()) {
                RCLCPP_ERROR(this->get_logger(), "Joint %s not found in joint_trajectory", ordered_names_[i].c_str());
                return;
            }
            const int index = static_cast<int>(std::distance(msg->joint_names.begin(), it));
            trajectory_joint_index_map_[i] = index;
        }

        // [修改] 虽然不再按 time_from_start 做在线重采样，但仍保留严格递增检查，防止上游发来的轨迹异常。
        for (size_t i = 0; i < msg->points.size(); ++i) {
            const auto& point = msg->points[i];
            if (point.positions.size() < ordered_names_.size()) {
                RCLCPP_ERROR(
                    this->get_logger(),
                    "Joint trajectory point %zu has only %zu positions, expected at least %zu.",
                    i, point.positions.size(), ordered_names_.size());
                return;
            }
            if (i > 0 &&
                rclcpp::Duration(point.time_from_start) <=
                    rclcpp::Duration(msg->points[i - 1].time_from_start)) {
                RCLCPP_ERROR(
                    this->get_logger(),
                    "Joint trajectory time_from_start is not strictly increasing at point %zu.",
                    i);
                return;
            }
        }

        joint_trajectory_ = *msg;

        // [修改] 新增：在收到新轨迹时，把 joint trajectory 一次性转成 task-space path samples。
        build_task_space_path_samples_from_joint_trajectory();

        // [修改] 收到新轨迹后总是重置为 ready，并切到全局轨迹模式。
        last_joint_trajectory_start_time_ = -1.0;
        use_global_trajectory_ = true;
        trajectory_state_ = JointTrajectoryState::kTrajectoryReady;

        RCLCPP_INFO(
            this->get_logger(),
            "Received global joint trajectory with %zu points and built %zu task-space path samples. Switched to global trajectory mode.",
            joint_trajectory_.points.size(), task_space_path_samples_.size());
    }

public:
    NMPCNode() : Node("nmpc_tau_node") {
        declare_startup_parameters();
        nmpc_solver_.setCostWeights(nmpc_cost_weights_);
        load_static_obstacles_from_scene();
        initialize_pinocchio();
        stage_refs_buffer_.resize(static_cast<std::size_t>(nmpc_solver_.getN() + 1));

        initial_q_ = Eigen::VectorXd::Zero(7);
        reference_q_ = Eigen::VectorXd::Zero(7);
        // reference_q_ << 0.0, -1.57, 0.785, -2.356, 0.0, 1.571, 0.785;
        initial_q_ << 0.008, -1.382, -0.008, -3.072, -0.007, 1.615, 0.792;
        reference_q_ << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785;
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
            joint_states_topic_, 10,
            [this](const sensor_msgs::msg::JointState::SharedPtr msg) { joint_states_callback(msg); });

        joint_trajectory_sub_ = this->create_subscription<trajectory_msgs::msg::JointTrajectory> (
            joint_trajectory_topic_, 10,
            [this](const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) { joint_trajectory_callback(msg); });

        target_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            target_pose_topic_, 10,
            [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { target_pose_callback(msg); });

        nmpc_res_pub_ = this->create_publisher<panda_interfaces::msg::ResultNMPC>(nmpc_result_topic_, 1);
        timer_ = this->create_wall_timer(std::chrono::milliseconds(static_cast<int64_t>(nmpc_solver_.getDt() * 1000)), [this]() { timer_callback(); });

        node_initial_time_ = this->now().seconds();

        RCLCPP_INFO(
            this->get_logger(),
            "NMPC node configured. use_global_trajectory=%s, obstacle_config=%s, scene_xml=%s, loaded_sphere_obstacles=%zu, urdf=%s, ee_frame=%s, joint_states_topic=%s, joint_trajectory_topic=%s, target_pose_topic=%s, result_topic=%s",
            use_global_trajectory_ ? "true" : "false",
            obstacle_config_path_.c_str(),
            scene_xml_path_.c_str(),
            scene_obstacles_.size(),
            urdf_path_.c_str(),
            ee_frame_name_.c_str(),
            joint_states_topic_.c_str(),
            joint_trajectory_topic_.c_str(),
            target_pose_topic_.c_str(),
            nmpc_result_topic_.c_str());
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NMPCNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
