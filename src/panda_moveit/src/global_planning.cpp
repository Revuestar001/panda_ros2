#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include "panda_moveit/static_sphere_scene.hpp"

class GlobalPlanningNode : public rclcpp::Node {
public:
  explicit GlobalPlanningNode(const rclcpp::NodeOptions& options)
  : Node("global_planning_node", options),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_) {
    declareParameters();
    sanitizePlannerSelection();

    trajectory_pub_ =
        create_publisher<trajectory_msgs::msg::JointTrajectory>(trajectory_topic_, rclcpp::QoS(1).reliable());

    target_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        target_topic_, rclcpp::QoS(10),
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { onTargetPose(msg); });

    planning_scene_client_ =
        create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");

    init_timer_ = create_wall_timer(
        std::chrono::milliseconds(500), [this]() { tryInitializeMoveIt(); });

    planning_worker_thread_ = std::thread([this]() { planningWorkerLoop(); });

    RCLCPP_INFO(
        get_logger(),
        "Global planner node created. planning_group=%s, planner_id=%s, target_topic=%s, trajectory_topic=%s, obstacle_config=%s, scene_xml=%s, obstacle_radius_padding=%.4f, use_multi_seed_ik=%s",
        planning_group_.c_str(), planner_id_.c_str(), target_topic_.c_str(),
        trajectory_topic_.c_str(), obstacle_config_path_.c_str(),
        scene_xml_path_.c_str(), obstacle_radius_padding_,
        use_multi_seed_ik_ ? "true" : "false");
  }

  ~GlobalPlanningNode() override {
    stopPlanningWorker();
  }

private:
  struct PlanCandidate {
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    double terminal_position_error{std::numeric_limits<double>::infinity()};
    double terminal_orientation_error{std::numeric_limits<double>::infinity()};
    double joint_path_length{std::numeric_limits<double>::infinity()};
    int round_index{0};
    bool valid{false};
  };

  struct PlanningRequest {
    geometry_msgs::msg::PoseStamped target_pose;
    std::uint64_t revision{0};
  };

  static constexpr const char* kRequiredPlannerId = "RRTstarkConfigDefault";

  static double quaternionAngularDistance(
      const geometry_msgs::msg::Quaternion& a_msg,
      const geometry_msgs::msg::Quaternion& b_msg) {
    tf2::Quaternion a;
    tf2::Quaternion b;
    tf2::fromMsg(a_msg, a);
    tf2::fromMsg(b_msg, b);
    a.normalize();
    b.normalize();

    const double dot = std::clamp(std::abs(a.dot(b)), 0.0, 1.0);
    return 2.0 * std::acos(dot);
  }

  static double computeJointPathLength(const trajectory_msgs::msg::JointTrajectory& trajectory) {
    if (trajectory.points.size() < 2) {
      return 0.0;
    }

    double length = 0.0;
    for (std::size_t i = 1; i < trajectory.points.size(); ++i) {
      const auto& prev = trajectory.points[i - 1].positions;
      const auto& curr = trajectory.points[i].positions;
      const std::size_t dof = std::min(prev.size(), curr.size());
      double segment_sq_norm = 0.0;
      for (std::size_t joint = 0; joint < dof; ++joint) {
        const double diff = curr[joint] - prev[joint];
        segment_sq_norm += diff * diff;
      }
      length += std::sqrt(segment_sq_norm);
    }
    return length;
  }

  void declareParameters() {
    planning_group_ = declare_parameter<std::string>("planning_group", "panda_arm");
    target_topic_ = declare_parameter<std::string>("target_topic", "/global_target_pose");
    trajectory_topic_ = declare_parameter<std::string>("trajectory_topic", "/global_joint_trajectory");

    planning_time_ = declare_parameter<double>("planning_time", 3.0);
    num_planning_attempts_ = declare_parameter<int>("num_planning_attempts", 1);
    planner_id_ = declare_parameter<std::string>("planner_id", kRequiredPlannerId);

    max_multi_plan_rounds_ = declare_parameter<int>("max_multi_plan_rounds", 20);
    min_rounds_before_early_stop_ = declare_parameter<int>("min_rounds_before_early_stop", 3);
    accepted_terminal_position_error_ =
        declare_parameter<double>("accepted_terminal_position_error", 5.0e-4);
    accepted_terminal_orientation_error_ =
        declare_parameter<double>("accepted_terminal_orientation_error", 5.0e-3);

    position_tolerance_ = declare_parameter<double>("goal_position_tolerance", 5.0e-4);
    orientation_tolerance_ = declare_parameter<double>("goal_orientation_tolerance", 5.0e-3);
    target_frame_ = declare_parameter<std::string>("target_frame", "");

    use_multi_seed_ik_ = declare_parameter<bool>("use_multi_seed_ik", true);
    ik_timeout_per_seed_ = declare_parameter<double>("ik_timeout_per_seed", 0.02);
    current_state_wait_time_ = declare_parameter<double>("current_state_wait_time", 0.2);

    obstacle_config_path_ = declare_parameter<std::string>(
        "obstacle_config_path",
        "/home/cyh/panda_ros2/model/franka_emika_panda/static_sphere_obstacles.xml");
    scene_xml_path_ = declare_parameter<std::string>(
        "scene_xml_path", "/home/cyh/panda_ros2/model/franka_emika_panda/scene_tau_ros.xml");
    scene_world_frame_ = declare_parameter<std::string>("scene_world_frame", "world");
    obstacle_radius_padding_ = declare_parameter<double>("obstacle_radius_padding", 0.0);

    if (planning_time_ <= 0.0) {
      throw std::runtime_error("Parameter 'planning_time' must be > 0.");
    }
    if (num_planning_attempts_ <= 0) {
      throw std::runtime_error("Parameter 'num_planning_attempts' must be >= 1.");
    }
    if (max_multi_plan_rounds_ <= 0) {
      throw std::runtime_error("Parameter 'max_multi_plan_rounds' must be >= 1.");
    }
    if (min_rounds_before_early_stop_ <= 0) {
      throw std::runtime_error("Parameter 'min_rounds_before_early_stop' must be >= 1.");
    }
    if (position_tolerance_ <= 0.0 || orientation_tolerance_ <= 0.0) {
      throw std::runtime_error("Goal tolerances must be > 0.");
    }
    if (accepted_terminal_position_error_ <= 0.0 ||
        accepted_terminal_orientation_error_ <= 0.0) {
      throw std::runtime_error("Accepted terminal errors must be > 0.");
    }
    if (ik_timeout_per_seed_ <= 0.0) {
      throw std::runtime_error("Parameter 'ik_timeout_per_seed' must be > 0.");
    }
    if (current_state_wait_time_ <= 0.0) {
      throw std::runtime_error("Parameter 'current_state_wait_time' must be > 0.");
    }
    if (obstacle_radius_padding_ < 0.0) {
      throw std::runtime_error("Parameter 'obstacle_radius_padding' must be >= 0.");
    }
  }

  void sanitizePlannerSelection() {
    if (planner_id_ != kRequiredPlannerId) {
      RCLCPP_WARN(
          get_logger(),
          "Planner '%s' is not allowed. Overriding to '%s' because this node only supports RRT*.",
          planner_id_.c_str(),
          kRequiredPlannerId);
      planner_id_ = kRequiredPlannerId;
    }
  }

  std::string selectObstacleSourcePath() const {
    if (!obstacle_config_path_.empty() && std::ifstream(obstacle_config_path_).good()) {
      return obstacle_config_path_;
    }
    return scene_xml_path_;
  }

  bool transformTargetPoseToPlanningFrame(
      const geometry_msgs::msg::PoseStamped& input_pose,
      geometry_msgs::msg::PoseStamped& output_pose) {
    if (!move_group_) {
      return false;
    }

    const std::string planning_frame =
        target_frame_.empty() ? move_group_->getPlanningFrame() : target_frame_;
    if (input_pose.header.frame_id.empty()) {
      RCLCPP_WARN(
          get_logger(),
          "Target pose frame_id is empty. Expected a valid frame, preferably '%s'.",
          planning_frame.c_str());
      return false;
    }

    output_pose = input_pose;
    if (output_pose.header.frame_id == planning_frame) {
      return true;
    }

    try {
      if (!tf_buffer_.canTransform(
              planning_frame,
              output_pose.header.frame_id,
              tf2::TimePointZero,
              tf2::durationFromSec(0.2))) {
        RCLCPP_WARN(
            get_logger(),
            "Cannot transform target pose from frame '%s' to planning frame '%s'.",
            output_pose.header.frame_id.c_str(),
            planning_frame.c_str());
        return false;
      }

      output_pose = tf_buffer_.transform(
          output_pose, planning_frame, tf2::durationFromSec(0.2));
      return true;
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN(
          get_logger(),
          "Failed to transform target pose from frame '%s' to planning frame '%s': %s",
          output_pose.header.frame_id.c_str(),
          planning_frame.c_str(),
          ex.what());
      return false;
    }
  }

  void tryInitializeMoveIt() {
    if (moveit_ready_.load()) {
      return;
    }

    if (!planning_scene_client_->wait_for_service(std::chrono::seconds(0))) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 3000,
          "Waiting for MoveIt '/get_planning_scene' service. Start move_group before using the global planner.");
      return;
    }

    try {
      planning_scene_interface_ =
          std::make_unique<moveit::planning_interface::PlanningSceneInterface>();
      move_group_ =
          std::make_unique<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), planning_group_);

      move_group_->setPlanningTime(planning_time_);
      move_group_->setNumPlanningAttempts(num_planning_attempts_);
      move_group_->setGoalPositionTolerance(position_tolerance_);
      move_group_->setGoalOrientationTolerance(orientation_tolerance_);
      move_group_->setPlannerId(planner_id_);
      move_group_->allowReplanning(false);
      move_group_->setStartStateToCurrentState();

      validateIkConfiguration();

      if (!applyStaticObstaclesFromScene()) {
        planning_scene_interface_.reset();
        move_group_.reset();
        return;
      }

      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        moveit_ready_.store(true);
        if (latest_target_pose_.has_value()) {
          planning_request_pending_ = true;
        }
      }
      init_timer_->cancel();
      planning_request_cv_.notify_one();

      RCLCPP_INFO(
          get_logger(),
          "Global planner ready. planner_id=%s, planning_frame=%s, ee_link=%s",
          planner_id_.c_str(),
          move_group_->getPlanningFrame().c_str(),
          move_group_->getEndEffectorLink().c_str());
    } catch (const std::exception& ex) {
      planning_scene_interface_.reset();
      move_group_.reset();
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 3000,
          "Failed to initialize MoveIt interfaces: %s", ex.what());
    }
  }

  bool applyStaticObstaclesFromScene() {
    if (!planning_scene_interface_) {
      return false;
    }

    const std::string obstacle_source_path = selectObstacleSourcePath();
    std::vector<panda_moveit::StaticSphereObstacle> obstacles;
    try {
      obstacles = panda_moveit::loadStaticSphereObstaclesFromSceneXml(obstacle_source_path);
    } catch (const std::exception& ex) {
      RCLCPP_ERROR(
          get_logger(),
          "Failed to load static sphere obstacles from '%s': %s",
          obstacle_source_path.c_str(), ex.what());
      return false;
    }

    if (obstacles.empty()) {
      RCLCPP_WARN(
          get_logger(),
          "Obstacle source '%s' does not contain any static sphere obstacles. Planning will continue without scene obstacles.",
          obstacle_source_path.c_str());
      return true;
    }

    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    collision_objects.reserve(obstacles.size());
    for (const auto& obstacle : obstacles) {
      collision_objects.push_back(
          panda_moveit::toCollisionObject(obstacle, scene_world_frame_, obstacle_radius_padding_));
    }

    planning_scene_interface_->applyCollisionObjects(collision_objects);
    RCLCPP_INFO(
        get_logger(),
        "Applied %zu static sphere obstacles from '%s' with radius padding %.4f m.",
        collision_objects.size(), obstacle_source_path.c_str(), obstacle_radius_padding_);
    return true;
  }

  void onTargetPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    bool cancel_current_plan = false;
    bool moveit_ready = false;
    std::uint64_t revision = 0;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      latest_target_pose_ = *msg;
      revision = ++latest_target_revision_;
      planning_request_pending_ = true;
      cancel_current_plan = planning_in_progress_;
      moveit_ready = moveit_ready_.load();
    }

    RCLCPP_INFO(
        get_logger(),
        "Received target pose in frame '%s'. Queued planning request revision=%llu.",
        msg->header.frame_id.c_str(),
        static_cast<unsigned long long>(revision));

    if (!moveit_ready) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "MoveIt is not ready yet. Stored the latest target pose and will plan once initialization finishes.");
      return;
    }

    if (cancel_current_plan) {
      requestPlanningCancel("a newer target pose arrived");
    }
    planning_request_cv_.notify_one();
  }

  void planningWorkerLoop() {
    while (rclcpp::ok()) {
      const std::optional<PlanningRequest> request = waitForPlanningRequest();
      if (!request.has_value()) {
        return;
      }

      (void)planAndPublishForTarget(request->target_pose, request->revision);

      bool notify_again = false;
      {
        std::lock_guard<std::mutex> lock(state_mutex_);
        planning_in_progress_ = false;
        if (!shutdown_requested_ &&
            latest_target_pose_.has_value() &&
            latest_target_revision_ != request->revision) {
          planning_request_pending_ = true;
          notify_again = true;
        }
      }

      if (notify_again) {
        planning_request_cv_.notify_one();
      }
    }
  }

  std::optional<PlanningRequest> waitForPlanningRequest() {
    std::unique_lock<std::mutex> lock(state_mutex_);
    planning_request_cv_.wait(
        lock,
        [this]() {
          return shutdown_requested_ ||
                 (moveit_ready_.load() && planning_request_pending_ && latest_target_pose_.has_value());
        });

    if (shutdown_requested_) {
      return std::nullopt;
    }

    PlanningRequest request;
    request.target_pose = *latest_target_pose_;
    request.revision = latest_target_revision_;
    planning_request_pending_ = false;
    planning_in_progress_ = true;
    return request;
  }

  void stopPlanningWorker() {
    bool cancel_current_plan = false;
    {
      std::lock_guard<std::mutex> lock(state_mutex_);
      if (shutdown_requested_) {
        if (planning_worker_thread_.joinable()) {
          planning_worker_thread_.join();
        }
        return;
      }
      shutdown_requested_ = true;
      cancel_current_plan = planning_in_progress_;
    }

    if (cancel_current_plan) {
      requestPlanningCancel("the global planner node is shutting down");
    }
    planning_request_cv_.notify_all();

    if (planning_worker_thread_.joinable()) {
      planning_worker_thread_.join();
    }
  }

  void requestPlanningCancel(const std::string& reason) {
    if (!move_group_) {
      return;
    }

    auto& action_client = move_group_->getMoveGroupClient();
    if (!action_client.action_server_is_ready()) {
      return;
    }

    (void)action_client.async_cancel_all_goals();
    RCLCPP_INFO(
        get_logger(),
        "Requested cancellation of the active MoveIt planning goal because %s.",
        reason.c_str());
  }

  bool shouldAbortPlanningRevision(std::uint64_t revision) const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return shutdown_requested_ || latest_target_revision_ != revision;
  }

  bool computeJointGoalFromIK(
      const geometry_msgs::msg::PoseStamped& goal_pose,
      int round_index,
      std::vector<double>& joint_goal) {
    if (!move_group_ || !use_multi_seed_ik_) {
      return false;
    }

    const moveit::core::RobotStatePtr current_state =
        move_group_->getCurrentState(current_state_wait_time_);
    if (!current_state) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Failed to fetch the current robot state for IK seeding.");
      return false;
    }

    const moveit::core::JointModelGroup* joint_model_group =
        current_state->getJointModelGroup(planning_group_);
    if (joint_model_group == nullptr) {
      RCLCPP_ERROR(get_logger(), "Joint model group '%s' not found.", planning_group_.c_str());
      return false;
    }

    moveit::core::RobotState ik_state(*current_state);
    if (round_index > 1) {
      ik_state.setToRandomPositions(joint_model_group);
    }

    const std::string tip_link = move_group_->getEndEffectorLink();
    if (tip_link.empty()) {
      RCLCPP_ERROR(get_logger(), "Move group '%s' has no end effector link.", planning_group_.c_str());
      return false;
    }

    if (!joint_model_group->canSetStateFromIK(tip_link)) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 3000,
          "IK solver is not available for group '%s'. Falling back to direct pose-target planning.",
          planning_group_.c_str());
      return false;
    }

    if (!ik_state.setFromIK(joint_model_group, goal_pose.pose, tip_link, ik_timeout_per_seed_)) {
      return false;
    }

    ik_state.update();
    ik_state.copyJointGroupPositions(joint_model_group, joint_goal);
    return true;
  }

  void validateIkConfiguration() {
    if (!move_group_ || !use_multi_seed_ik_) {
      return;
    }

    const moveit::core::RobotModelConstPtr robot_model = move_group_->getRobotModel();
    if (robot_model == nullptr) {
      RCLCPP_WARN(
          get_logger(),
          "MoveIt robot model is not available while validating IK. Multi-seed IK will be disabled.");
      use_multi_seed_ik_ = false;
      return;
    }

    const moveit::core::JointModelGroup* joint_model_group =
        robot_model->getJointModelGroup(planning_group_);
    if (joint_model_group == nullptr) {
      RCLCPP_WARN(
          get_logger(),
          "Joint model group '%s' is missing in the local MoveIt model. Multi-seed IK will be disabled.",
          planning_group_.c_str());
      use_multi_seed_ik_ = false;
      return;
    }

    const std::string tip_link = move_group_->getEndEffectorLink();
    if (tip_link.empty() || !joint_model_group->canSetStateFromIK(tip_link)) {
      RCLCPP_WARN(
          get_logger(),
          "No local IK solver is configured for group '%s'. Make sure robot_description_kinematics is passed to this node. Multi-seed IK will be disabled for now.",
          planning_group_.c_str());
      use_multi_seed_ik_ = false;
      return;
    }

    RCLCPP_INFO(
        get_logger(),
        "IK solver is available for group '%s'. Multi-seed IK planning is enabled.",
        planning_group_.c_str());
  }

  bool planToPoseGoal(
      const geometry_msgs::msg::PoseStamped& goal_pose,
      PlanCandidate& candidate) {
    move_group_->setStartStateToCurrentState();
    move_group_->clearPoseTargets();

    if (!move_group_->setPoseTarget(goal_pose)) {
      RCLCPP_WARN(
          get_logger(),
          "Failed to set pose target for planning frame '%s'.",
          goal_pose.header.frame_id.c_str());
      move_group_->clearPoseTargets();
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool success = static_cast<bool>(move_group_->plan(plan));
    move_group_->clearPoseTargets();
    if (!success) {
      return false;
    }
    return evaluatePlanCandidate(plan, goal_pose, candidate);
  }

  bool planToJointGoal(
      const std::vector<double>& joint_goal,
      const geometry_msgs::msg::PoseStamped& goal_pose,
      PlanCandidate& candidate) {
    move_group_->setStartStateToCurrentState();
    move_group_->clearPoseTargets();

    if (!move_group_->setJointValueTarget(joint_goal)) {
      RCLCPP_WARN(get_logger(), "Failed to set the joint-value target generated from IK.");
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool success = static_cast<bool>(move_group_->plan(plan));
    move_group_->clearPoseTargets();
    if (!success) {
      return false;
    }
    return evaluatePlanCandidate(plan, goal_pose, candidate);
  }

  bool evaluatePlanCandidate(
      const moveit::planning_interface::MoveGroupInterface::Plan& plan,
      const geometry_msgs::msg::PoseStamped& goal_pose,
      PlanCandidate& candidate) const {
    const auto& trajectory = plan.trajectory_.joint_trajectory;
    if (trajectory.points.empty()) {
      return false;
    }

    const moveit::core::RobotModelConstPtr robot_model = move_group_->getRobotModel();
    const moveit::core::JointModelGroup* joint_model_group =
        robot_model->getJointModelGroup(planning_group_);
    if (joint_model_group == nullptr) {
      RCLCPP_ERROR(get_logger(), "Joint model group '%s' not found.", planning_group_.c_str());
      return false;
    }

    const std::string ee_link = move_group_->getEndEffectorLink();
    if (ee_link.empty()) {
      RCLCPP_ERROR(get_logger(), "Move group '%s' has no end effector link.", planning_group_.c_str());
      return false;
    }

    moveit::core::RobotState final_state(robot_model);
    final_state.setToDefaultValues();

    const auto& joint_names = trajectory.joint_names;
    const auto& final_positions = trajectory.points.back().positions;
    if (joint_names.size() != final_positions.size()) {
      RCLCPP_ERROR(
          get_logger(),
          "JointTrajectory final point dimension mismatch: names=%zu positions=%zu",
          joint_names.size(), final_positions.size());
      return false;
    }

    for (std::size_t i = 0; i < joint_names.size(); ++i) {
      final_state.setVariablePosition(joint_names[i], final_positions[i]);
    }
    final_state.update();

    const Eigen::Isometry3d final_transform = final_state.getGlobalLinkTransform(ee_link);
    const Eigen::Quaterniond final_quaternion(final_transform.rotation());

    geometry_msgs::msg::PoseStamped final_pose;
    final_pose.header.frame_id = move_group_->getPlanningFrame();
    final_pose.pose.position.x = final_transform.translation().x();
    final_pose.pose.position.y = final_transform.translation().y();
    final_pose.pose.position.z = final_transform.translation().z();
    final_pose.pose.orientation.x = final_quaternion.x();
    final_pose.pose.orientation.y = final_quaternion.y();
    final_pose.pose.orientation.z = final_quaternion.z();
    final_pose.pose.orientation.w = final_quaternion.w();

    candidate.plan = plan;
    candidate.terminal_position_error =
        std::sqrt(
            std::pow(final_pose.pose.position.x - goal_pose.pose.position.x, 2.0) +
            std::pow(final_pose.pose.position.y - goal_pose.pose.position.y, 2.0) +
            std::pow(final_pose.pose.position.z - goal_pose.pose.position.z, 2.0));
    candidate.terminal_orientation_error =
        quaternionAngularDistance(final_pose.pose.orientation, goal_pose.pose.orientation);
    candidate.joint_path_length = computeJointPathLength(trajectory);
    candidate.valid = true;
    return true;
  }

  bool isBetterCandidate(const PlanCandidate& lhs, const PlanCandidate& rhs) const {
    if (!lhs.valid) {
      return false;
    }
    if (!rhs.valid) {
      return true;
    }

    constexpr double kPosEps = 1.0e-9;
    constexpr double kRotEps = 1.0e-9;
    constexpr double kPathEps = 1.0e-9;

    if (lhs.terminal_position_error + kPosEps < rhs.terminal_position_error) {
      return true;
    }
    if (rhs.terminal_position_error + kPosEps < lhs.terminal_position_error) {
      return false;
    }

    if (lhs.terminal_orientation_error + kRotEps < rhs.terminal_orientation_error) {
      return true;
    }
    if (rhs.terminal_orientation_error + kRotEps < lhs.terminal_orientation_error) {
      return false;
    }

    if (lhs.joint_path_length + kPathEps < rhs.joint_path_length) {
      return true;
    }
    if (rhs.joint_path_length + kPathEps < lhs.joint_path_length) {
      return false;
    }

    return lhs.round_index < rhs.round_index;
  }

  bool candidateMeetsAcceptanceThreshold(const PlanCandidate& candidate) const {
    return candidate.valid &&
           candidate.terminal_position_error <= accepted_terminal_position_error_ &&
           candidate.terminal_orientation_error <= accepted_terminal_orientation_error_;
  }

  bool planAndPublishForTarget(
      const geometry_msgs::msg::PoseStamped& target_pose,
      std::uint64_t revision) {
    if (!move_group_) {
      RCLCPP_WARN(
          get_logger(),
          "MoveIt is not ready yet. Planning request ignored until initialization completes.");
      return false;
    }

    geometry_msgs::msg::PoseStamped pose_for_planning;
    if (!transformTargetPoseToPlanningFrame(target_pose, pose_for_planning)) {
      return false;
    }

    RCLCPP_INFO(
        get_logger(),
        "Starting planning for target revision=%llu using planner '%s'.",
        static_cast<unsigned long long>(revision),
        planner_id_.c_str());

    PlanCandidate best_candidate;
    int successful_rounds = 0;
    bool aborted_for_newer_target = false;

    for (int round = 1; round <= max_multi_plan_rounds_; ++round) {
      if (shouldAbortPlanningRevision(revision)) {
        aborted_for_newer_target = true;
        break;
      }

      PlanCandidate candidate;
      candidate.round_index = round;

      bool success = false;
      if (use_multi_seed_ik_) {
        std::vector<double> joint_goal;
        if (!computeJointGoalFromIK(pose_for_planning, round, joint_goal)) {
          continue;
        }
        success = planToJointGoal(joint_goal, pose_for_planning, candidate);
      } else {
        success = planToPoseGoal(pose_for_planning, candidate);
      }

      if (!success) {
        if (shouldAbortPlanningRevision(revision)) {
          aborted_for_newer_target = true;
          break;
        }
        continue;
      }

      ++successful_rounds;
      if (isBetterCandidate(candidate, best_candidate)) {
        best_candidate = std::move(candidate);
      }

      if (successful_rounds >= min_rounds_before_early_stop_ &&
          candidateMeetsAcceptanceThreshold(best_candidate)) {
        break;
      }
    }

    move_group_->clearPoseTargets();

    if (aborted_for_newer_target) {
      RCLCPP_INFO(
          get_logger(),
          "Discarded planning work for target revision=%llu because a newer target arrived.",
          static_cast<unsigned long long>(revision));
      return false;
    }

    if (!best_candidate.valid) {
      RCLCPP_WARN(
          get_logger(),
          "MoveIt planning failed after %d RRT* round(s) for target revision=%llu in frame '%s'.",
          max_multi_plan_rounds_,
          static_cast<unsigned long long>(revision),
          pose_for_planning.header.frame_id.c_str());
      return false;
    }

    trajectory_msgs::msg::JointTrajectory trajectory = best_candidate.plan.trajectory_.joint_trajectory;
    if (trajectory.points.empty()) {
      RCLCPP_WARN(get_logger(), "Best plan returned an empty JointTrajectory. Nothing will be published.");
      return false;
    }

    for (std::size_t i = 1; i < trajectory.points.size(); ++i) {
      if (static_cast<rclcpp::Duration>(trajectory.points[i].time_from_start) <=
          static_cast<rclcpp::Duration>(trajectory.points[i - 1].time_from_start)) {
        RCLCPP_WARN(
            get_logger(),
            "JointTrajectory has non-increasing time_from_start at point %zu. Nothing will be published.",
            i);
        return false;
      }
    }

    if (shouldAbortPlanningRevision(revision)) {
      RCLCPP_INFO(
          get_logger(),
          "A newer target arrived before publishing trajectory for revision=%llu. The old trajectory will be dropped.",
          static_cast<unsigned long long>(revision));
      return false;
    }

    trajectory.header.stamp = now();
    trajectory_pub_->publish(trajectory);

    const auto total_duration = trajectory.points.back().time_from_start;
    RCLCPP_INFO(
        get_logger(),
        "Published RRT* trajectory for revision=%llu: points=%zu, duration=%.3f s, terminal_pos_err=%.6f m, terminal_rot_err=%.6f rad, joint_path_len=%.6f, successful_rounds=%d/%d, ik_mode=%s",
        static_cast<unsigned long long>(revision),
        trajectory.points.size(),
        rclcpp::Duration(total_duration).seconds(),
        best_candidate.terminal_position_error,
        best_candidate.terminal_orientation_error,
        best_candidate.joint_path_length,
        successful_rounds,
        max_multi_plan_rounds_,
        use_multi_seed_ik_ ? "multi_seed" : "direct_pose");
    return true;
  }

  std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::unique_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_sub_;
  rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr planning_scene_client_;
  rclcpp::TimerBase::SharedPtr init_timer_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  mutable std::mutex state_mutex_;
  std::condition_variable planning_request_cv_;
  std::optional<geometry_msgs::msg::PoseStamped> latest_target_pose_;
  std::thread planning_worker_thread_;

  std::string planning_group_;
  std::string target_topic_;
  std::string trajectory_topic_;
  std::string planner_id_;
  std::string target_frame_;
  std::string obstacle_config_path_;
  std::string scene_xml_path_;
  std::string scene_world_frame_;

  std::atomic<bool> moveit_ready_{false};
  bool use_multi_seed_ik_{true};
  bool planning_request_pending_{false};
  bool planning_in_progress_{false};
  bool shutdown_requested_{false};

  std::uint64_t latest_target_revision_{0};

  double planning_time_{3.0};
  int num_planning_attempts_{1};
  int max_multi_plan_rounds_{20};
  int min_rounds_before_early_stop_{3};
  double accepted_terminal_position_error_{5.0e-4};
  double accepted_terminal_orientation_error_{5.0e-3};
  double position_tolerance_{5.0e-4};
  double orientation_tolerance_{5.0e-3};
  double obstacle_radius_padding_{0.0};
  double ik_timeout_per_seed_{0.02};
  double current_state_wait_time_{0.2};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<GlobalPlanningNode>(rclcpp::NodeOptions());

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
