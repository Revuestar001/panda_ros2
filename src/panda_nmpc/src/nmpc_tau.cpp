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
#include "second_order_reference_regulator.hpp"

class NMPCNode : public rclcpp::Node {
private:
    using Vec7 = Eigen::Matrix<double, 7, 1>;
    using Vec3 = Eigen::Vector3d;
    using Mat3 = Eigen::Matrix3d;

    struct TaskSpacePathSample {
        double path_s{0.0};
        Vec3 pos{Vec3::Zero()};
        Mat3 rot{Mat3::Identity()};
        Vec7 q{Vec7::Zero()};
    };

    struct CarrotTarget {
        Vec3 target_pos{Vec3::Zero()};
        Mat3 target_rot{Mat3::Identity()};
        Vec7 q_nom{Vec7::Zero()};
        std::size_t progress_index{0};
        std::size_t carrot_index{0};
        double lookahead_distance{0.0};
        bool terminal_hold{false};
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
    NMPCRuntimeCostConfig nmpc_cost_config_;

    // sorr
    SecondOrderReferenceRegulator sorr_pos_;
    SecondOrderReferenceRegulator sorr_rot_;

    Eigen::Vector3d target_pos_;
    Eigen::Matrix3d target_rot_;
    Eigen::VectorXd reference_q_;
    Eigen::VectorXd jq_;
    Eigen::VectorXd jv_;

    std::string urdf_path_;
    std::string ee_frame_name_;
    std::string joint_states_topic_;
    std::string joint_trajectory_topic_;
    std::string nmpc_result_topic_;
    std::string target_pose_topic_;

    // global joint trajectory
    trajectory_msgs::msg::JointTrajectory joint_trajectory_;
    JointTrajectoryState trajectory_state_{JointTrajectoryState::kTrajectoryNull};
    std::array<int, 7> trajectory_joint_index_map_{};

    // 关节路径只作为无时间的全局几何引导；每个控制周期只选一个前视胡萝卜点。
    std::vector<TaskSpacePathSample> task_space_path_samples_;
    std::size_t task_space_path_progress_index_{0};
    std::size_t active_carrot_index_{0};
    double carrot_lookahead_distance_{0.06};
    double carrot_min_lookahead_distance_{0.04};
    double carrot_max_lookahead_distance_{0.12};
    double carrot_lookahead_velocity_gain_{0.08};
    double carrot_reached_tolerance_{0.015};
    double carrot_goal_switch_distance_{0.03};
    double carrot_joint_metric_weight_{0.02};
    bool carrot_use_path_orientation_{false};
    double progress_search_window_distance_{0.12};
    int progress_search_window_points_{12};

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

    template <int N>
    Eigen::Matrix<double, N, 1> declare_fixed_vector_parameter(
        const std::string& name,
        const std::array<double, N>& default_values
    ) {
        const std::vector<double> default_vec(default_values.begin(), default_values.end());
        const auto value = this->declare_parameter<std::vector<double>>(name, default_vec);
        if (value.size() != static_cast<std::size_t>(N)) {
            throw std::runtime_error(
                "Parameter '" + name + "' must have exactly " + std::to_string(N) + " elements.");
        }

        Eigen::Matrix<double, N, 1> result;
        for (int i = 0; i < N; ++i) {
            result(i) = value[static_cast<std::size_t>(i)];
        }
        return result;
    }

    void load_nmpc_runtime_cost_parameters() {
        nmpc_cost_config_.pos = declare_fixed_vector_parameter<3>(
            "cost_pos", {500.0, 500.0, 500.0});
        nmpc_cost_config_.rot = declare_fixed_vector_parameter<3>(
            "cost_rot", {100.0, 100.0, 100.0});
        nmpc_cost_config_.ee_lin_vel = declare_fixed_vector_parameter<3>(
            "cost_ee_lin_vel", {5.0, 5.0, 5.0});
        nmpc_cost_config_.ee_ang_vel = declare_fixed_vector_parameter<3>(
            "cost_ee_ang_vel", {2.0, 2.0, 2.0});
        nmpc_cost_config_.dq_reg = declare_fixed_vector_parameter<7>(
            "cost_dq_reg", {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1});
        nmpc_cost_config_.ddq_reg = declare_fixed_vector_parameter<7>(
            "cost_ddq_reg", {0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05});
        nmpc_cost_config_.neutral_q_reg = declare_fixed_vector_parameter<7>(
            "cost_neutral_q_reg", {15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0});

        nmpc_cost_config_.pos_e = declare_fixed_vector_parameter<3>(
            "cost_pos_e", {750.0, 750.0, 750.0});
        nmpc_cost_config_.rot_e = declare_fixed_vector_parameter<3>(
            "cost_rot_e", {200.0, 200.0, 200.0});
        nmpc_cost_config_.ee_lin_vel_e = declare_fixed_vector_parameter<3>(
            "cost_ee_lin_vel_e", {10.0, 10.0, 10.0});
        nmpc_cost_config_.ee_ang_vel_e = declare_fixed_vector_parameter<3>(
            "cost_ee_ang_vel_e", {4.0, 4.0, 4.0});
        nmpc_cost_config_.dq_reg_e = declare_fixed_vector_parameter<7>(
            "cost_dq_reg_e", {0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2});
        nmpc_cost_config_.neutral_q_reg_e = declare_fixed_vector_parameter<7>(
            "cost_neutral_q_reg_e", {30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0});

        nmpc_cost_config_.q_neutral = declare_fixed_vector_parameter<7>(
            "cost_q_neutral", {0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854});

        nmpc_cost_config_.joint_limit_barrier =
            this->declare_parameter<double>("cost_joint_limit_barrier", 0.05);
        nmpc_cost_config_.joint_limit_barrier_e =
            this->declare_parameter<double>("cost_joint_limit_barrier_e", 0.08);
        nmpc_cost_config_.manipulability =
            this->declare_parameter<double>("cost_manipulability", 5.0);
        nmpc_cost_config_.manipulability_e =
            this->declare_parameter<double>("cost_manipulability_e", 8.0);
        nmpc_cost_config_.clearance =
            this->declare_parameter<double>("cost_clearance", 180.0);
        nmpc_cost_config_.clearance_e =
            this->declare_parameter<double>("cost_clearance_e", 260.0);
        nmpc_cost_config_.barrier_eps =
            this->declare_parameter<double>("cost_barrier_eps", 1.0e-4);
        nmpc_cost_config_.manipulability_eps =
            this->declare_parameter<double>("cost_manipulability_eps", 1.0e-6);
        nmpc_cost_config_.clearance_activation_margin =
            this->declare_parameter<double>("cost_clearance_activation_margin", 0.06);
        nmpc_cost_config_.clearance_softplus_gain =
            this->declare_parameter<double>("cost_clearance_softplus_gain", 40.0);

        nmpc_cost_config_.safe_slack_linear =
            this->declare_parameter<double>("slack_safe_linear", 1.0e4);
        nmpc_cost_config_.safe_slack_quadratic =
            this->declare_parameter<double>("slack_safe_quadratic", 1.0e6);
        nmpc_cost_config_.approach_slack_linear =
            this->declare_parameter<double>("slack_approach_linear", 2.0e3);
        nmpc_cost_config_.approach_slack_quadratic =
            this->declare_parameter<double>("slack_approach_quadratic", 2.0e5);
    }

    void declare_startup_parameters() {
        urdf_path_ = this->declare_parameter<std::string>("urdf_path", default_panda_urdf_path());
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

        carrot_lookahead_distance_ =
            this->declare_parameter<double>("carrot_lookahead_distance", 0.06);
        carrot_min_lookahead_distance_ =
            this->declare_parameter<double>("carrot_min_lookahead_distance", 0.04);
        carrot_max_lookahead_distance_ =
            this->declare_parameter<double>("carrot_max_lookahead_distance", 0.12);
        carrot_lookahead_velocity_gain_ =
            this->declare_parameter<double>("carrot_lookahead_velocity_gain", 0.08);
        carrot_reached_tolerance_ =
            this->declare_parameter<double>("carrot_reached_tolerance", 0.015);
        carrot_goal_switch_distance_ =
            this->declare_parameter<double>("carrot_goal_switch_distance", 0.03);
        carrot_joint_metric_weight_ =
            this->declare_parameter<double>("carrot_joint_metric_weight", 0.02);
        carrot_use_path_orientation_ =
            this->declare_parameter<bool>("carrot_use_path_orientation", false);
        progress_search_window_distance_ =
            this->declare_parameter<double>("progress_search_window_distance", 0.12);
        progress_search_window_points_ =
            this->declare_parameter<int>("progress_search_window_points", 12);
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

    double compute_current_ee_linear_speed(const Vec7& current_q, const Vec7& current_v) {
        pinocchio::forwardKinematics(pin_model_, *pin_data_, current_q, current_v);
        pinocchio::computeJointJacobians(pin_model_, *pin_data_, current_q);
        pinocchio::updateFramePlacements(pin_model_, *pin_data_);

        Eigen::Matrix<double, 6, Eigen::Dynamic> J_lwa(6, pin_model_.nv);
        J_lwa.setZero();
        pinocchio::getFrameJacobian(
            pin_model_, *pin_data_, ee_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED, J_lwa);

        return (J_lwa.leftCols<7>().topRows<3>() * current_v).norm();
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

    void sample_joint_state_from_trajectory_point(
        const trajectory_msgs::msg::JointTrajectoryPoint& point,
        Vec7& q) const {
        q.setZero();
        for (size_t joint = 0; joint < ordered_names_.size(); ++joint) {
            const int msg_index = trajectory_joint_index_map_[joint];
            const std::size_t idx = static_cast<std::size_t>(msg_index);
            q(static_cast<Eigen::Index>(joint)) = point.positions[idx];
        }
    }

    TaskSpacePathSample build_task_space_sample_from_joint_q(const Vec7& q, double path_s) {
        pinocchio::forwardKinematics(pin_model_, *pin_data_, q);
        pinocchio::updateFramePlacements(pin_model_, *pin_data_);
        const pinocchio::SE3& oMf_ee = pin_data_->oMf[ee_frame_id_];

        TaskSpacePathSample sample;
        sample.path_s = path_s;
        sample.pos = oMf_ee.translation();
        sample.rot = oMf_ee.rotation();
        sample.q = q;
        return sample;
    }

    double task_space_metric(const Vec3& pos_a, const Vec3& pos_b) const {
        return (pos_a - pos_b).norm();
    }

    double path_progress_metric(const TaskSpacePathSample& sample, const Vec7& current_q, const Vec3& current_ee_pos) const {
        const double pos_metric = (sample.pos - current_ee_pos).norm();
        const double joint_metric = (sample.q - current_q).norm();
        return pos_metric + carrot_joint_metric_weight_ * joint_metric;
    }

    void build_task_space_path_samples_from_joint_trajectory() {
        task_space_path_samples_.clear();
        task_space_path_progress_index_ = 0;
        active_carrot_index_ = 0;

        if (joint_trajectory_.points.empty()) {
            return;
        }

        task_space_path_samples_.reserve(joint_trajectory_.points.size());

        double cumulative_s = 0.0;
        Vec3 prev_pos = Vec3::Zero();
        bool has_prev = false;

        for (const auto& point : joint_trajectory_.points) {
            Vec7 q_stage = Vec7::Zero();
            sample_joint_state_from_trajectory_point(point, q_stage);

            TaskSpacePathSample sample = build_task_space_sample_from_joint_q(q_stage, cumulative_s);
            if (has_prev) {
                cumulative_s += task_space_metric(prev_pos, sample.pos);
                sample.path_s = cumulative_s;
            }

            task_space_path_samples_.push_back(sample);
            prev_pos = sample.pos;
            has_prev = true;
        }
    }

    std::size_t find_closest_path_sample_index(const Vec7& current_q, const Vec3& current_ee_pos) {
        if (task_space_path_samples_.empty()) {
            return 0;
        }

        const std::size_t start_index =
            std::min(task_space_path_progress_index_, task_space_path_samples_.size() - 1);
        const std::size_t max_point_advance =
            static_cast<std::size_t>(std::max(progress_search_window_points_, 0));
        const double max_window_distance = std::max(progress_search_window_distance_, 0.0);
        const double start_path_s = task_space_path_samples_[start_index].path_s;

        std::size_t end_index = start_index;
        while (end_index + 1 < task_space_path_samples_.size()) {
            const std::size_t next_index = end_index + 1;
            if (next_index - start_index > max_point_advance) {
                break;
            }
            if (task_space_path_samples_[next_index].path_s - start_path_s > max_window_distance) {
                break;
            }
            end_index = next_index;
        }

        std::size_t best_index = start_index;
        double best_metric = 1.0e18;

        // 只允许在当前进度前方的局部窗口里选最近点，避免在自交/折返路径上直接跳到很后面的段。
        for (std::size_t i = start_index; i <= end_index; ++i) {
            const auto& sample = task_space_path_samples_[i];
            const double metric = path_progress_metric(sample, current_q, current_ee_pos);
            if (metric < best_metric) {
                best_metric = metric;
                best_index = i;
            }
        }

        return best_index;
    }

    std::size_t advance_path_index_by_distance(std::size_t start_index, double lookahead_distance) const {
        if (task_space_path_samples_.empty()) {
            return 0;
        }

        const std::size_t clamped_start = std::min(start_index, task_space_path_samples_.size() - 1);
        const double target_s = task_space_path_samples_[clamped_start].path_s + std::max(lookahead_distance, 0.0);

        std::size_t idx = clamped_start;
        while (idx + 1 < task_space_path_samples_.size() &&
               task_space_path_samples_[idx].path_s < target_s) {
            ++idx;
        }
        return idx;
    }

    CarrotTarget build_carrot_target(const Vec7& current_q, const Vec7& current_v) {
        CarrotTarget carrot;

        Vec3 goal_pos = Vec3::Zero();
        Mat3 goal_rot = Mat3::Identity();
        get_desired_goal_pose(goal_pos, goal_rot);

        if (task_space_path_samples_.empty()) {
            carrot.target_pos = goal_pos;
            carrot.target_rot = goal_rot;
            carrot.q_nom = current_q;
            carrot.terminal_hold = true;
            return carrot;
        }

        Vec3 current_ee_pos = Vec3::Zero();
        Mat3 current_ee_rot = Mat3::Identity();
        compute_current_ee_pose(current_ee_pos, current_ee_rot);

        task_space_path_progress_index_ = find_closest_path_sample_index(current_q, current_ee_pos);
        const auto& progress_sample = task_space_path_samples_[task_space_path_progress_index_];
        const double remaining_s = task_space_path_samples_.back().path_s - progress_sample.path_s;

        carrot.progress_index = task_space_path_progress_index_;

        if (remaining_s <= carrot_goal_switch_distance_) {
            carrot.target_pos = goal_pos;
            carrot.target_rot = goal_rot;
            carrot.q_nom = task_space_path_samples_.back().q;
            carrot.carrot_index = task_space_path_samples_.size() - 1;
            carrot.lookahead_distance = 0.0;
            carrot.terminal_hold = true;
            active_carrot_index_ = carrot.carrot_index;
            return carrot;
        }

        const double ee_speed = compute_current_ee_linear_speed(current_q, current_v);
        carrot.lookahead_distance = std::clamp(
            carrot_lookahead_distance_ + carrot_lookahead_velocity_gain_ * ee_speed,
            carrot_min_lookahead_distance_, carrot_max_lookahead_distance_);

        std::size_t carrot_index = advance_path_index_by_distance(task_space_path_progress_index_, carrot.lookahead_distance);
        carrot_index = std::max(carrot_index, task_space_path_progress_index_);

        while (carrot_index + 1 < task_space_path_samples_.size() &&
               (task_space_path_samples_[carrot_index].pos - current_ee_pos).norm() < carrot_reached_tolerance_) {
            ++carrot_index;
        }

        active_carrot_index_ = carrot_index;
        carrot.carrot_index = carrot_index;
        carrot.target_pos = task_space_path_samples_[carrot_index].pos;
        carrot.target_rot = carrot_use_path_orientation_ && carrot_index + 1 < task_space_path_samples_.size()
            ? task_space_path_samples_[carrot_index].rot
            : goal_rot;
        carrot.q_nom = task_space_path_samples_[carrot_index].q;
        carrot.terminal_hold = false;
        return carrot;
    }

    bool is_trajectory_tracking_complete(
        const Vec7& current_q,
        const Vec7& current_v
    ) {
        if (task_space_path_samples_.empty()) {
            return false;
        }

        const double remaining_s =
            task_space_path_samples_.back().path_s - task_space_path_samples_[task_space_path_progress_index_].path_s;
        if (remaining_s > carrot_goal_switch_distance_) {
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
                q_nom,     // 先继续使用当前关节角作为 q_nom，避免零空间突然拉扯
                sorr_pos_.getLinearVelocity(),
                sorr_rot_.getAngularVelocity(),
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
                    trajectory_state_ = JointTrajectoryState::kTrajectoryUsing;
                    [[fallthrough]];
                case JointTrajectoryState::kTrajectoryUsing :
                {
                    const CarrotTarget carrot = build_carrot_target(current_q, current_v);
                    res = nmpc_solver_.NMPCSolve(
                        carrot.target_pos,
                        carrot.target_rot,
                        carrot.q_nom,
                        Vec3::Zero(),
                        Vec3::Zero(),
                        current_q,
                        current_v);
                    if (is_trajectory_tracking_complete(current_q, current_v)) {
                        trajectory_state_ = JointTrajectoryState::kTrajectoryComplete;
                        RCLCPP_INFO(this->get_logger(), "Global path tracking completed, switching to terminal hold.");
                    } else {
                        RCLCPP_DEBUG_THROTTLE(
                            this->get_logger(), *this->get_clock(), 500,
                            "Carrot tracking: progress=%zu/%zu carrot=%zu/%zu lookahead=%.3f m",
                            task_space_path_progress_index_,
                            task_space_path_samples_.empty() ? std::size_t{0} : task_space_path_samples_.size() - 1,
                            active_carrot_index_,
                            task_space_path_samples_.empty() ? std::size_t{0} : task_space_path_samples_.size() - 1,
                            carrot.lookahead_distance);
                    }
                    break;
                }
                case JointTrajectoryState::kTrajectoryComplete :
                    res = nmpc_solver_.NMPCSolve(
                        target_pos_,
                        target_rot_,
                        current_q,
                        Vec3::Zero(),
                        Vec3::Zero(),
                        current_q,
                        current_v);
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

        for (size_t i = 0; i < msg->points.size(); ++i) {
            const auto& point = msg->points[i];
            if (point.positions.size() < ordered_names_.size()) {
                RCLCPP_ERROR(
                    this->get_logger(),
                    "Joint trajectory point %zu has only %zu positions, expected at least %zu.",
                    i, point.positions.size(), ordered_names_.size());
                return;
            }
        }

        joint_trajectory_ = *msg;

        // [修改] 新增：在收到新轨迹时，把 joint trajectory 一次性转成 task-space path samples。
        build_task_space_path_samples_from_joint_trajectory();

        use_global_trajectory_ = true;
        trajectory_state_ = JointTrajectoryState::kTrajectoryReady;
        task_space_path_progress_index_ = 0;
        active_carrot_index_ = 0;

        RCLCPP_INFO(
            this->get_logger(),
            "Received global joint path with %zu points and built %zu path samples. Switched to carrot-following mode.",
            joint_trajectory_.points.size(), task_space_path_samples_.size());
    }

public:
    NMPCNode() : Node("nmpc_tau_node") {
        declare_startup_parameters();
        load_nmpc_runtime_cost_parameters();
        nmpc_solver_.setRuntimeCostConfig(nmpc_cost_config_);
        initialize_pinocchio();

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
            "NMPC node configured. use_global_trajectory=%s, urdf=%s, ee_frame=%s, joint_states_topic=%s, joint_trajectory_topic=%s, target_pose_topic=%s, result_topic=%s, cost_pos=(%.1f, %.1f, %.1f), cost_clearance=%.1f",
            use_global_trajectory_ ? "true" : "false",
            urdf_path_.c_str(),
            ee_frame_name_.c_str(),
            joint_states_topic_.c_str(),
            joint_trajectory_topic_.c_str(),
            target_pose_topic_.c_str(),
            nmpc_result_topic_.c_str(),
            nmpc_cost_config_.pos.x(),
            nmpc_cost_config_.pos.y(),
            nmpc_cost_config_.pos.z(),
            nmpc_cost_config_.clearance);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<NMPCNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
