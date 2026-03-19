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
    Eigen::VectorXd reference_q_;
    Eigen::VectorXd jq_;
    Eigen::VectorXd jv_;

    std::string urdf_path_;
    std::string ee_frame_name_;
    std::string joint_states_topic_;
    std::string joint_trajectory_topic_;
    std::string nmpc_result_topic_;

    // global joint trajectory
    trajectory_msgs::msg::JointTrajectory joint_trajectory_;
    JointTrajectoryState trajectory_state_{JointTrajectoryState::kTrajectoryNull};
    double last_joint_trajectory_start_time_{-1.0};
    std::array<int, 7> trajectory_joint_index_map_{};
    pinocchio::SE3 joint_trajectory_ee_goal_pose_;
    Vec7 joint_trajectory_goal_q_{Vec7::Zero()};

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
        joint_trajectory_topic_ =
            this->declare_parameter<std::string>("joint_trajectory_topic", "/global_joint_trajectory");
        nmpc_result_topic_ = this->declare_parameter<std::string>("nmpc_result_topic", "/nmpc_result");

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

    double get_joint_trajectory_total_duration() const {
        if (joint_trajectory_.points.empty()) {
            return 0.0;
        }
        return rclcpp::Duration(joint_trajectory_.points.back().time_from_start).seconds();
    }

    std::vector<NMPCStageReference> build_terminal_hold_stage_refs() {
        const int N = nmpc_solver_.getN();
        std::vector<NMPCStageReference> stage_refs(static_cast<std::size_t>(N + 1));

        const Vec3 ee_goal_pos = joint_trajectory_ee_goal_pose_.translation();
        const Mat3 ee_goal_rot = joint_trajectory_ee_goal_pose_.rotation();

        for (auto& stage_ref : stage_refs) {
            stage_ref.target_pos = ee_goal_pos;
            stage_ref.target_rot = ee_goal_rot;
            stage_ref.q_nom = joint_trajectory_goal_q_;
            stage_ref.ee_lin_vel_ref.setZero();
            stage_ref.ee_ang_vel_ref.setZero();
        }

        return stage_refs;
    }

    bool is_trajectory_tracking_complete(
        const double time_duration,
        const Vec7& current_q,
        const Vec7& current_v
    ) {
        if (joint_trajectory_.points.empty()) {
            return false;
        }
        if (time_duration < get_joint_trajectory_total_duration()) {
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
        const pinocchio::SE3 err_M = oMf_current.actInv(joint_trajectory_ee_goal_pose_);
        const Eigen::Matrix<double, 6, 1> err = pinocchio::log6(err_M).toVector();
        const Eigen::Matrix<double, 6, 1> ee_twist = J_lwa.leftCols<7>() * current_v;

        return err.head<3>().norm() < ee_frame_pos_tolerance_ &&
               err.tail<3>().norm() < ee_frame_rot_tolerance_ &&
               ee_twist.head<3>().norm() < ee_frame_lin_vel_tolerance_ &&
               ee_twist.tail<3>().norm() < ee_frame_ang_vel_tolerance_;
    }

    std::vector<NMPCStageReference> joint_trajectory_resample(const double& time_duration) {
        const double dt = nmpc_solver_.getDt();
        const int N = nmpc_solver_.getN();

        std::vector<NMPCStageReference> stage_refs(static_cast<std::size_t>(N + 1));
        if (joint_trajectory_.points.empty()) {
            Vec3 ee_pos = Vec3::Zero();
            Mat3 ee_rot = Mat3::Identity();
            compute_current_ee_pose(ee_pos, ee_rot);

            for (auto& stage_ref : stage_refs) {
                // 这里不会导致控制发散？
                stage_ref.target_pos = ee_pos;
                stage_ref.target_rot = ee_rot;
                stage_ref.q_nom = jq_.head<7>();
                stage_ref.ee_lin_vel_ref.setZero();
                stage_ref.ee_ang_vel_ref.setZero();
            }
            return stage_refs;
        }

        const auto point_time_sec = [](const trajectory_msgs::msg::JointTrajectoryPoint& point) {
            return rclcpp::Duration(point.time_from_start).seconds();
        };

        const auto sample_joint_state =
            [this](const trajectory_msgs::msg::JointTrajectoryPoint& point, Vec7& q, Vec7& dq) {
                q.setZero();
                dq.setZero();
                for (size_t joint = 0; joint < ordered_names_.size(); ++joint) {
                    const int msg_index = trajectory_joint_index_map_[joint];
                    q(static_cast<Eigen::Index>(joint)) = point.positions[static_cast<std::size_t>(msg_index)];
                    if (point.velocities.size() > static_cast<std::size_t>(msg_index)) {
                        dq(static_cast<Eigen::Index>(joint)) =
                            point.velocities[static_cast<std::size_t>(msg_index)];
                    }
                }
            };

        const auto& points = joint_trajectory_.points;
        const double first_time = point_time_sec(points.front());
        const double last_time = point_time_sec(points.back());
        std::size_t seg_idx = 0;

        for (int stage = 0; stage <= N; ++stage) {
            const double query_time = time_duration + stage * dt;

            Vec7 q_stage = Vec7::Zero();
            Vec7 dq_stage = Vec7::Zero();

            if (query_time <= first_time) {
                sample_joint_state(points.front(), q_stage, dq_stage);
            } else if (query_time >= last_time) {
                sample_joint_state(points.back(), q_stage, dq_stage);
                dq_stage.setZero();  // 超出轨迹末端后保持末端姿态，速度参考收敛到 0。
            } else {
                while (seg_idx + 1 < points.size() && point_time_sec(points[seg_idx + 1]) < query_time) {
                    ++seg_idx;
                }

                const auto& point_0 = points[seg_idx];
                const auto& point_1 = points[seg_idx + 1];
                const double t0 = point_time_sec(point_0);
                const double t1 = point_time_sec(point_1);
                const double alpha = std::clamp((query_time - t0) / (t1 - t0), 0.0, 1.0);
                const double inv_alpha = 1.0 - alpha;

                for (size_t joint = 0; joint < ordered_names_.size(); ++joint) {
                    const int msg_index = trajectory_joint_index_map_[joint];
                    const std::size_t idx = static_cast<std::size_t>(msg_index);

                    const double q0 = point_0.positions[idx];
                    const double q1 = point_1.positions[idx];
                    q_stage(static_cast<Eigen::Index>(joint)) = inv_alpha * q0 + alpha * q1;

                    if (point_0.velocities.size() > idx && point_1.velocities.size() > idx) {
                        const double dq0 = point_0.velocities[idx];
                        const double dq1 = point_1.velocities[idx];
                        dq_stage(static_cast<Eigen::Index>(joint)) = inv_alpha * dq0 + alpha * dq1;
                    } else {
                        dq_stage(static_cast<Eigen::Index>(joint)) = (q1 - q0) / (t1 - t0);
                    }
                }
            }

            pinocchio::forwardKinematics(pin_model_, *pin_data_, q_stage, dq_stage);
            pinocchio::computeJointJacobians(pin_model_, *pin_data_, q_stage);
            pinocchio::updateFramePlacements(pin_model_, *pin_data_);

            Eigen::Matrix<double, 6, Eigen::Dynamic> J_lwa(6, pin_model_.nv);
            J_lwa.setZero();
            pinocchio::getFrameJacobian(
                pin_model_, *pin_data_, ee_frame_id_, pinocchio::LOCAL_WORLD_ALIGNED, J_lwa);

            const pinocchio::SE3& oMf_ee = pin_data_->oMf[ee_frame_id_];
            const Eigen::Matrix<double, 6, 1> ee_twist = J_lwa.leftCols<7>() * dq_stage;

            auto& stage_ref = stage_refs[static_cast<std::size_t>(stage)];
            stage_ref.target_pos = oMf_ee.translation();
            stage_ref.target_rot = oMf_ee.rotation();
            stage_ref.q_nom = q_stage;
            stage_ref.ee_lin_vel_ref = ee_twist.head<3>();
            stage_ref.ee_ang_vel_ref = ee_twist.tail<3>();
        }

        return stage_refs;
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
                    last_joint_trajectory_start_time_ = this->now().seconds();
                    trajectory_state_ = JointTrajectoryState::kTrajectoryUsing;
                    [[fallthrough]];
                case JointTrajectoryState::kTrajectoryUsing :
                {
                    const double time_duration = this->now().seconds() - last_joint_trajectory_start_time_;
                    res = nmpc_solver_.NMPCSolveTrajectory(joint_trajectory_resample(time_duration), current_q, current_v);
                    if (is_trajectory_tracking_complete(time_duration, current_q, current_v)) {
                        trajectory_state_ = JointTrajectoryState::kTrajectoryComplete;
                        RCLCPP_INFO(this->get_logger(), "Global joint trajectory completed, switching to terminal hold.");
                    }
                    break;
                }
                case JointTrajectoryState::kTrajectoryComplete :
                    res = nmpc_solver_.NMPCSolveTrajectory(
                        build_terminal_hold_stage_refs(), current_q, current_v);
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

        // [修改] 轨迹采样依赖 time_from_start 做插值，必须保证严格递增。
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

        const auto& last_point = joint_trajectory_.points.back();
        for (size_t i = 0; i < ordered_names_.size(); ++i) {
            const int joint_msg_index = trajectory_joint_index_map_[i];
            joint_trajectory_goal_q_(static_cast<Eigen::Index>(i)) = last_point.positions[joint_msg_index];
        }
        pinocchio::forwardKinematics(pin_model_, *pin_data_, joint_trajectory_goal_q_);
        pinocchio::updateFramePlacements(pin_model_, *pin_data_);
        joint_trajectory_ee_goal_pose_ = pin_data_->oMf[ee_frame_id_];

        // [修改] 收到新轨迹后总是重置为 ready，并切到全局轨迹模式。
        last_joint_trajectory_start_time_ = -1.0;
        use_global_trajectory_ = true;
        trajectory_state_ = JointTrajectoryState::kTrajectoryReady;

        RCLCPP_INFO(
            this->get_logger(),
            "Received global joint trajectory with %zu points. Switched to global trajectory mode.",
            joint_trajectory_.points.size());
    }

public:
    NMPCNode() : Node("nmpc_tau_node") {
        declare_startup_parameters();
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

        nmpc_res_pub_ = this->create_publisher<panda_interfaces::msg::ResultNMPC>(nmpc_result_topic_, 1);
        timer_ = this->create_wall_timer(std::chrono::milliseconds(static_cast<int64_t>(nmpc_solver_.getDt() * 1000)), [this]() { timer_callback(); });

        node_initial_time_ = this->now().seconds();

        RCLCPP_INFO(
            this->get_logger(),
            "NMPC node configured. use_global_trajectory=%s, urdf=%s, ee_frame=%s, joint_states_topic=%s, joint_trajectory_topic=%s, result_topic=%s",
            use_global_trajectory_ ? "true" : "false",
            urdf_path_.c_str(),
            ee_frame_name_.c_str(),
            joint_states_topic_.c_str(),
            joint_trajectory_topic_.c_str(),
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
