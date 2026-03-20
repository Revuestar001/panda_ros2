#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
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
    planning_group_ = declare_parameter<std::string>("planning_group", "panda_arm");
    target_topic_ = declare_parameter<std::string>("target_topic", "/global_target_pose");
    trajectory_topic_ = declare_parameter<std::string>("trajectory_topic", "/global_joint_trajectory");
    collision_object_topic_ =
        declare_parameter<std::string>("collision_object_topic", "/global_collision_object");
    plan_service_name_ = declare_parameter<std::string>("plan_service", "/plan_global_trajectory");
    plan_on_target_update_ = declare_parameter<bool>("plan_on_target_update", false);

    planning_time_ = declare_parameter<double>("planning_time", 3.0);
    num_planning_attempts_ = declare_parameter<int>("num_planning_attempts", 5);
    planner_id_ = declare_parameter<std::string>("planner_id", "RRTConnectkConfigDefault");

    // 多轮规划：不断尝试并保留“终点误差更小，且路径更短”的那条。
    max_multi_plan_rounds_ = declare_parameter<int>("max_multi_plan_rounds", 20);
    min_rounds_before_early_stop_ = declare_parameter<int>("min_rounds_before_early_stop", 3);
    accepted_terminal_position_error_ =
        declare_parameter<double>("accepted_terminal_position_error", 5.0e-4);
    accepted_terminal_orientation_error_ =
        declare_parameter<double>("accepted_terminal_orientation_error", 5.0e-3);

    position_tolerance_ = declare_parameter<double>("goal_position_tolerance", 5.0e-4);
    orientation_tolerance_ = declare_parameter<double>("goal_orientation_tolerance", 5.0e-3);
    target_frame_ = declare_parameter<std::string>("target_frame", "");

    load_static_obstacles_from_scene_ =
        declare_parameter<bool>("load_static_obstacles_from_scene", true);
    obstacle_config_path_ = declare_parameter<std::string>(
        "obstacle_config_path",
        "/home/cyh/panda_ros2/model/franka_emika_panda/static_sphere_obstacles.xml");
    scene_xml_path_ = declare_parameter<std::string>(
        "scene_xml_path", "/home/cyh/panda_ros2/model/franka_emika_panda/scene_tau_ros.xml");
    scene_world_frame_ = declare_parameter<std::string>("scene_world_frame", "world");

    use_exact_ik_joint_targets_ = declare_parameter<bool>("use_exact_ik_joint_targets", true);
    ik_seed_attempts_ = declare_parameter<int>("ik_seed_attempts", 80);
    max_ik_solutions_ = declare_parameter<int>("max_ik_solutions", 12);
    ik_timeout_ = declare_parameter<double>("ik_timeout", 0.05);
    ik_min_solution_distance_ = declare_parameter<double>("ik_min_solution_distance", 0.1);
    planning_retries_per_ik_solution_ =
        declare_parameter<int>("planning_retries_per_ik_solution", 3);

    trajectory_pub_ =
        create_publisher<trajectory_msgs::msg::JointTrajectory>(trajectory_topic_, rclcpp::QoS(1).reliable());

    target_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        target_topic_, rclcpp::QoS(10),
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { onTargetPose(msg); });

    collision_object_sub_ = create_subscription<moveit_msgs::msg::CollisionObject>(
        collision_object_topic_, rclcpp::QoS(10).reliable().transient_local(),
        [this](const moveit_msgs::msg::CollisionObject::SharedPtr msg) { onCollisionObject(msg); });

    plan_srv_ = create_service<std_srvs::srv::Trigger>(
        plan_service_name_,
        [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
               std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
          const bool ok = planAndPublish();
          response->success = ok;
          response->message = ok ? "Global trajectory planned and published."
                                 : "Global planning failed. See node logs.";
        });

    planning_scene_client_ =
        create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");

    init_timer_ = create_wall_timer(
        std::chrono::milliseconds(500), [this]() { tryInitializeMoveIt(); });

    RCLCPP_INFO(
        get_logger(),
        "Global planner node created. planning_group=%s, target_topic=%s, trajectory_topic=%s, collision_object_topic=%s, service=%s, obstacle_config=%s, scene_xml=%s",
        planning_group_.c_str(), target_topic_.c_str(), trajectory_topic_.c_str(),
        collision_object_topic_.c_str(), plan_service_name_.c_str(),
        obstacle_config_path_.c_str(), scene_xml_path_.c_str());
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

  const moveit::core::JointModelGroup* getPlanningJointModelGroup() const {
    if (!move_group_) {
      return nullptr;
    }
    const moveit::core::RobotModelConstPtr robot_model = move_group_->getRobotModel();
    return robot_model == nullptr ? nullptr : robot_model->getJointModelGroup(planning_group_);
  }

  bool isDistinctJointTarget(
      const std::vector<double>& candidate,
      const std::vector<std::vector<double>>& solutions) const {
    for (const auto& solution : solutions) {
      const std::size_t dof = std::min(candidate.size(), solution.size());
      double sq_norm = 0.0;
      for (std::size_t i = 0; i < dof; ++i) {
        const double diff = candidate[i] - solution[i];
        sq_norm += diff * diff;
      }
      if (std::sqrt(sq_norm) < ik_min_solution_distance_) {
        return false;
      }
    }
    return true;
  }

  std::vector<std::vector<double>> collectExactIkJointTargets(
      const geometry_msgs::msg::PoseStamped& goal_pose) const {
    std::vector<std::vector<double>> solutions;
    if (!move_group_ || !use_exact_ik_joint_targets_) {
      return solutions;
    }

    const moveit::core::JointModelGroup* joint_model_group = getPlanningJointModelGroup();
    if (joint_model_group == nullptr) {
      RCLCPP_ERROR(get_logger(), "Joint model group '%s' not found.", planning_group_.c_str());
      return solutions;
    }

    const std::string ee_link = move_group_->getEndEffectorLink();
    if (ee_link.empty()) {
      RCLCPP_ERROR(get_logger(), "Move group '%s' has no end effector link.", planning_group_.c_str());
      return solutions;
    }

    const moveit::core::RobotStatePtr current_state = move_group_->getCurrentState(1.0);
    if (!current_state) {
      RCLCPP_WARN(get_logger(), "Failed to query current robot state for IK seeding.");
      return solutions;
    }

    std::vector<double> current_joint_values;
    current_state->copyJointGroupPositions(joint_model_group, current_joint_values);

    struct IkSolutionCandidate {
      std::vector<double> joint_values;
      double distance_to_current{0.0};
    };
    std::vector<IkSolutionCandidate> collected_candidates;

    for (int attempt = 0; attempt < ik_seed_attempts_; ++attempt) {
      moveit::core::RobotState ik_state(*current_state);
      if (attempt > 0) {
        ik_state.setToRandomPositions(joint_model_group);
      }

      const bool found_ik =
          ik_state.setFromIK(joint_model_group, goal_pose.pose, ee_link, ik_timeout_);
      if (!found_ik) {
        continue;
      }

      std::vector<double> joint_values;
      ik_state.copyJointGroupPositions(joint_model_group, joint_values);
      if (!isDistinctJointTarget(joint_values, solutions)) {
        continue;
      }

      double sq_norm = 0.0;
      const std::size_t dof = std::min(joint_values.size(), current_joint_values.size());
      for (std::size_t i = 0; i < dof; ++i) {
        const double diff = joint_values[i] - current_joint_values[i];
        sq_norm += diff * diff;
      }

      collected_candidates.push_back(
          IkSolutionCandidate{joint_values, std::sqrt(sq_norm)});
      solutions.push_back(joint_values);

      if (static_cast<int>(solutions.size()) >= max_ik_solutions_) {
        break;
      }
    }

    std::sort(
        collected_candidates.begin(), collected_candidates.end(),
        [](const IkSolutionCandidate& lhs, const IkSolutionCandidate& rhs) {
          return lhs.distance_to_current < rhs.distance_to_current;
        });

    solutions.clear();
    solutions.reserve(collected_candidates.size());
    for (const auto& candidate : collected_candidates) {
      solutions.push_back(candidate.joint_values);
    }
    return solutions;
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
      return false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool success = static_cast<bool>(move_group_->plan(plan));
    if (!success) {
      return false;
    }
    return evaluatePlanCandidate(plan, goal_pose, candidate);
  }

  void tryInitializeMoveIt() {
    if (moveit_ready_) {
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
      move_group_->allowReplanning(true);
      move_group_->setStartStateToCurrentState();

      applyStaticObstaclesFromScene();

      {
        std::lock_guard<std::mutex> lock(collision_object_mutex_);
        for (const auto& collision_object : pending_collision_objects_) {
          planning_scene_interface_->applyCollisionObject(collision_object);
        }
        pending_collision_objects_.clear();
      }

      moveit_ready_ = true;
      init_timer_->cancel();

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

  void applyStaticObstaclesFromScene() {
    if (!load_static_obstacles_from_scene_ || !planning_scene_interface_) {
      return;
    }

    const std::string obstacle_source_path =
        obstacle_config_path_.empty() ? scene_xml_path_ : obstacle_config_path_;
    std::vector<panda_moveit::StaticSphereObstacle> obstacles;
    try {
      obstacles = panda_moveit::loadStaticSphereObstaclesFromSceneXml(obstacle_source_path);
    } catch (const std::exception& ex) {
      RCLCPP_ERROR(
          get_logger(),
          "Failed to load static sphere obstacles from '%s': %s",
          obstacle_source_path.c_str(), ex.what());
      return;
    }

    if (obstacles.empty()) {
      RCLCPP_WARN(
          get_logger(),
          "Obstacle source '%s' does not contain any static sphere obstacles.",
          obstacle_source_path.c_str());
      return;
    }

    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    collision_objects.reserve(obstacles.size());
    for (const auto& obstacle : obstacles) {
      collision_objects.push_back(panda_moveit::toCollisionObject(obstacle, scene_world_frame_));
    }
    planning_scene_interface_->applyCollisionObjects(collision_objects);

    RCLCPP_INFO(
        get_logger(),
        "Applied %zu static sphere obstacles from '%s'.",
        collision_objects.size(), obstacle_source_path.c_str());
  }

  void onTargetPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      latest_target_pose_ = *msg;
    }

    RCLCPP_INFO(
        get_logger(),
        "Received target pose in frame '%s'. plan_on_target_update=%s",
        msg->header.frame_id.c_str(),
        plan_on_target_update_ ? "true" : "false");

    if (plan_on_target_update_) {
      (void)planAndPublish();
    }
  }

  void onCollisionObject(const moveit_msgs::msg::CollisionObject::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(collision_object_mutex_);

    if (!planning_scene_interface_) {
      pending_collision_objects_.push_back(*msg);
      RCLCPP_WARN(
          get_logger(),
          "MoveIt is not ready yet. Queued collision object '%s' for later application.",
          msg->id.c_str());
      return;
    }

    planning_scene_interface_->applyCollisionObject(*msg);
    RCLCPP_INFO(
        get_logger(),
        "Applied collision object '%s' in frame '%s' with operation=%d",
        msg->id.c_str(),
        msg->header.frame_id.c_str(),
        static_cast<int>(msg->operation));
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

  bool planAndPublish() {
    std::unique_lock<std::mutex> planning_lock(planning_mutex_, std::try_to_lock);
    if (!planning_lock.owns_lock()) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "A planning request is already running. Skip this request.");
      return false;
    }

    std::optional<geometry_msgs::msg::PoseStamped> target_pose;
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      target_pose = latest_target_pose_;
    }

    if (!target_pose.has_value()) {
      RCLCPP_WARN(get_logger(), "No target pose received yet. Planning request ignored.");
      return false;
    }

    if (!moveit_ready_ || !move_group_) {
      RCLCPP_WARN(
          get_logger(),
          "MoveIt is not ready yet. Make sure 'move_group' is running before triggering planning.");
      return false;
    }

    const std::string planning_frame =
        target_frame_.empty() ? move_group_->getPlanningFrame() : target_frame_;
    if (target_pose->header.frame_id.empty()) {
      RCLCPP_WARN(
          get_logger(),
          "Target pose frame_id is empty. Expected a valid frame, preferably '%s'.",
          planning_frame.c_str());
      return false;
    }

    geometry_msgs::msg::PoseStamped pose_for_planning = *target_pose;
    if (pose_for_planning.header.frame_id != planning_frame) {
      try {
        if (!tf_buffer_.canTransform(
                planning_frame,
                pose_for_planning.header.frame_id,
                tf2::TimePointZero,
                tf2::durationFromSec(0.2))) {
          RCLCPP_WARN(
              get_logger(),
              "Cannot transform target pose from frame '%s' to planning frame '%s'.",
              pose_for_planning.header.frame_id.c_str(),
              planning_frame.c_str());
          return false;
        }

        pose_for_planning = tf_buffer_.transform(
            pose_for_planning, planning_frame, tf2::durationFromSec(0.2));
      } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(
            get_logger(),
            "Failed to transform target pose from frame '%s' to planning frame '%s': %s",
            pose_for_planning.header.frame_id.c_str(),
            planning_frame.c_str(),
            ex.what());
        return false;
      }
    }

    PlanCandidate best_candidate;
    int successful_rounds = 0;
    int total_plan_attempts = 0;

    const auto ik_joint_targets = collectExactIkJointTargets(pose_for_planning);
    if (!ik_joint_targets.empty()) {
      RCLCPP_INFO(
          get_logger(),
          "Collected %zu distinct IK goal states for exact end-pose matching.",
          ik_joint_targets.size());
    } else if (use_exact_ik_joint_targets_) {
      RCLCPP_WARN(
          get_logger(),
          "No exact IK goal states found for the requested pose. Falling back to pose-target planning.");
    }

    // 先尽量把目标位姿转成一批精确的 joint target，再对每个终点做多轮规划筛选，
    // 让“路径尽量短”和“终点尽量精确贴合”同时成立。
    for (std::size_t target_index = 0;
         target_index < ik_joint_targets.size() && total_plan_attempts < max_multi_plan_rounds_;
         ++target_index) {
      for (int retry = 0;
           retry < planning_retries_per_ik_solution_ && total_plan_attempts < max_multi_plan_rounds_;
           ++retry) {
        ++total_plan_attempts;

        PlanCandidate candidate;
        candidate.round_index = total_plan_attempts;
        if (!planToJointGoal(ik_joint_targets[target_index], pose_for_planning, candidate)) {
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

      if (successful_rounds >= min_rounds_before_early_stop_ &&
          candidateMeetsAcceptanceThreshold(best_candidate)) {
        break;
      }
    }

    // 如果 IK joint target 没拿到有效解，再退回传统 pose target 重复规划。
    while (!best_candidate.valid && total_plan_attempts < max_multi_plan_rounds_) {
      ++total_plan_attempts;
      PlanCandidate candidate;
      candidate.round_index = total_plan_attempts;
      if (!planToPoseGoal(pose_for_planning, candidate)) {
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

    if (!best_candidate.valid) {
      RCLCPP_WARN(
          get_logger(),
          "MoveIt planning failed after %d attempts for target frame '%s'.",
          total_plan_attempts,
          planning_frame.c_str());
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

    trajectory.header.stamp = now();
    trajectory_pub_->publish(trajectory);

    const auto total_duration = trajectory.points.back().time_from_start;
    RCLCPP_INFO(
        get_logger(),
        "Published best trajectory after %d attempts: points=%zu, duration=%.3f s, terminal_pos_err=%.6f m, terminal_rot_err=%.6f rad, joint_path_len=%.6f",
        total_plan_attempts,
        trajectory.points.size(),
        rclcpp::Duration(total_duration).seconds(),
        best_candidate.terminal_position_error,
        best_candidate.terminal_orientation_error,
        best_candidate.joint_path_length);
    return true;
  }

  std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  std::unique_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;

  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_pub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_sub_;
  rclcpp::Subscription<moveit_msgs::msg::CollisionObject>::SharedPtr collision_object_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr plan_srv_;
  rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr planning_scene_client_;
  rclcpp::TimerBase::SharedPtr init_timer_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::mutex target_mutex_;
  std::optional<geometry_msgs::msg::PoseStamped> latest_target_pose_;
  std::mutex collision_object_mutex_;
  std::vector<moveit_msgs::msg::CollisionObject> pending_collision_objects_;
  std::mutex planning_mutex_;

  std::string target_topic_;
  std::string trajectory_topic_;
  std::string collision_object_topic_;
  std::string plan_service_name_;
  std::string planning_group_;
  std::string planner_id_;
  std::string target_frame_;
  std::string obstacle_config_path_;
  std::string scene_xml_path_;
  std::string scene_world_frame_;

  bool moveit_ready_{false};
  bool plan_on_target_update_{false};
  bool load_static_obstacles_from_scene_{true};

  double planning_time_{3.0};
  int num_planning_attempts_{5};
  int max_multi_plan_rounds_{20};
  int min_rounds_before_early_stop_{3};
  double accepted_terminal_position_error_{5.0e-4};
  double accepted_terminal_orientation_error_{5.0e-3};
  double position_tolerance_{5.0e-4};
  double orientation_tolerance_{5.0e-3};
  bool use_exact_ik_joint_targets_{true};
  int ik_seed_attempts_{80};
  int max_ik_solutions_{12};
  double ik_timeout_{0.05};
  double ik_min_solution_distance_{0.1};
  int planning_retries_per_ik_solution_{3};
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
