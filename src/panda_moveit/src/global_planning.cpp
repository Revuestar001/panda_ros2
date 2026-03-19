#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

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

    // 你若想更偏“最短路”，优先把 planner_id 设成 RRTstarkConfigDefault / PRMstarkConfigDefault。
    // 若对应 planner 没在 ompl_planning.yaml 里配置，节点会自动回退到 fallback_planner_id。
    planner_id_ = declare_parameter<std::string>("planner_id", "RRTstarkConfigDefault");
    fallback_planner_id_ = declare_parameter<std::string>("fallback_planner_id", "RRTConnectkConfigDefault");

    planning_time_ = declare_parameter<double>("planning_time", 4.0);
    num_planning_attempts_ = declare_parameter<int>("num_planning_attempts", 8);

    position_tolerance_ = declare_parameter<double>("goal_position_tolerance", 5.0e-3);
    orientation_tolerance_ = declare_parameter<double>("goal_orientation_tolerance", 5.0e-2);
    target_frame_ = declare_parameter<std::string>("target_frame", "");

    // true: 发布给下游的仅保留 positions，把 vel/acc/effort/time 都清空。
    // 这样 MoveIt 仍可能在内部做时间参数化，但你下游不会再被它的时间律牵着走。
    publish_positions_only_ = declare_parameter<bool>("publish_positions_only", true);

    // 若只是想观测 MoveIt 端总耗时，打开这个日志即可。
    log_plan_latency_ = declare_parameter<bool>("log_plan_latency", true);

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
          response->message = ok ? "Global path planned and published."
                                 : "Global planning failed. See node logs.";
        });

    planning_scene_client_ =
        create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");

    init_timer_ = create_wall_timer(
        std::chrono::milliseconds(500), [this]() { tryInitializeMoveIt(); });

    RCLCPP_INFO(
        get_logger(),
        "Global planner node created. group=%s, planner_id=%s, fallback=%s, attempts=%d, planning_time=%.3f, publish_positions_only=%s",
        planning_group_.c_str(), planner_id_.c_str(), fallback_planner_id_.c_str(),
        num_planning_attempts_, planning_time_, publish_positions_only_ ? "true" : "false");
  }

private:
  static double computeJointPathLength(const trajectory_msgs::msg::JointTrajectory& trajectory) {
    if (trajectory.points.size() < 2) {
      return 0.0;
    }

    double total = 0.0;
    for (size_t i = 1; i < trajectory.points.size(); ++i) {
      const auto& prev = trajectory.points[i - 1].positions;
      const auto& curr = trajectory.points[i].positions;
      if (prev.size() != curr.size()) {
        return std::numeric_limits<double>::infinity();
      }
      double seg = 0.0;
      for (size_t j = 0; j < curr.size(); ++j) {
        const double d = curr[j] - prev[j];
        seg += d * d;
      }
      total += std::sqrt(seg);
    }
    return total;
  }

  static void stripTimingAndDerivatives(trajectory_msgs::msg::JointTrajectory& trajectory) {
    for (auto& point : trajectory.points) {
      point.velocities.clear();
      point.accelerations.clear();
      point.effort.clear();
      point.time_from_start.sec = 0;
      point.time_from_start.nanosec = 0;
    }
  }

  void tryInitializeMoveIt() {
    if (!planning_scene_client_->wait_for_service(std::chrono::seconds(0))) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 3000,
          "Waiting for MoveIt '/get_planning_scene' service. Start move_group before using the global planner.");
      return;
    }

    // 串行化 MoveIt 接口初始化、规划调用和场景更新，避免多线程执行器下的共享状态竞争。
    std::lock_guard<std::mutex> planning_lock(planning_mutex_);
    if (moveit_ready_) {
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
      move_group_->setStartStateToCurrentState();

      for (const auto& collision_object : pending_collision_objects_) {
        planning_scene_interface_->applyCollisionObject(collision_object);
      }
      pending_collision_objects_.clear();

      moveit_ready_ = true;
      init_timer_->cancel();

      RCLCPP_INFO(
          get_logger(),
          "Global planner ready. group=%s, planner_id=%s, attempts=%d, planning_time=%.3f, planning_frame=%s",
          planning_group_.c_str(), planner_id_.c_str(), num_planning_attempts_, planning_time_,
          move_group_->getPlanningFrame().c_str());
    } catch (const std::exception& ex) {
      planning_scene_interface_.reset();
      move_group_.reset();
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 3000,
          "Failed to initialize MoveIt interfaces: %s", ex.what());
    }
  }

  void onTargetPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    {
      std::lock_guard<std::mutex> lock(target_mutex_);
      latest_target_pose_ = *msg;
    }

    RCLCPP_INFO(
        get_logger(), "Received target pose in frame '%s'. plan_on_target_update=%s",
        msg->header.frame_id.c_str(), plan_on_target_update_ ? "true" : "false");

    if (plan_on_target_update_) {
      (void)planAndPublish();
    }
  }

  void onCollisionObject(const moveit_msgs::msg::CollisionObject::SharedPtr msg) {
    std::lock_guard<std::mutex> planning_lock(planning_mutex_);

    if (!planning_scene_interface_) {
      pending_collision_objects_.push_back(*msg);
      RCLCPP_WARN(
          get_logger(), "MoveIt is not ready yet. Queued collision object '%s' for later application.",
          msg->id.c_str());
      return;
    }

    planning_scene_interface_->applyCollisionObject(*msg);
    RCLCPP_INFO(
        get_logger(), "Applied collision object '%s' in frame '%s' with operation=%d",
        msg->id.c_str(), msg->header.frame_id.c_str(), static_cast<int>(msg->operation));
  }

  bool runSinglePlan(const geometry_msgs::msg::PoseStamped& pose_for_planning,
                     const std::string& planning_frame,
                     const std::string& planner_id,
                     moveit::planning_interface::MoveGroupInterface::Plan& plan_out,
                     double& elapsed_ms_out) {
    move_group_->setPlannerId(planner_id);
    move_group_->setStartStateToCurrentState();
    move_group_->clearPoseTargets();

    const bool target_set = move_group_->setPoseTarget(pose_for_planning);
    if (!target_set) {
      RCLCPP_WARN(
          get_logger(), "Failed to set pose target for frame '%s'. Planning request ignored.",
          pose_for_planning.header.frame_id.c_str());
      move_group_->clearPoseTargets();
      return false;
    }

    const auto t0 = std::chrono::steady_clock::now();
    moveit::planning_interface::MoveGroupInterface::Plan local_plan;
    const bool success = static_cast<bool>(move_group_->plan(local_plan));
    const auto t1 = std::chrono::steady_clock::now();
    elapsed_ms_out =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

    if (!success) {
      RCLCPP_WARN(
          get_logger(), "MoveIt planning failed. planner_id=%s, target_frame='%s', planning_frame='%s'.",
          planner_id.c_str(), pose_for_planning.header.frame_id.c_str(), planning_frame.c_str());
      move_group_->clearPoseTargets();
      return false;
    }

    plan_out = std::move(local_plan);
    move_group_->clearPoseTargets();
    return true;
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
          "MoveIt is not ready yet. Make sure move_group is running before triggering planning.");
      return false;
    }

    const std::string planning_frame =
        target_frame_.empty() ? move_group_->getPlanningFrame() : target_frame_;

    if (target_pose->header.frame_id.empty()) {
      RCLCPP_WARN(
          get_logger(),
          "Target pose frame_id is empty. Expected a valid frame, preferably '%s'. Planning request ignored.",
          planning_frame.c_str());
      return false;
    }

    geometry_msgs::msg::PoseStamped pose_for_planning = *target_pose;
    if (pose_for_planning.header.frame_id != planning_frame) {
      try {
        if (!tf_buffer_.canTransform(
                planning_frame, pose_for_planning.header.frame_id, tf2::TimePointZero,
                tf2::durationFromSec(0.2))) {
          RCLCPP_WARN(
              get_logger(),
              "Cannot transform target pose from frame '%s' to planning frame '%s'. Planning request ignored.",
              pose_for_planning.header.frame_id.c_str(), planning_frame.c_str());
          return false;
        }

        pose_for_planning = tf_buffer_.transform(
            pose_for_planning, planning_frame, tf2::durationFromSec(0.2));
      } catch (const tf2::TransformException& ex) {
        RCLCPP_WARN(
            get_logger(),
            "Failed to transform target pose from frame '%s' to planning frame '%s': %s",
            pose_for_planning.header.frame_id.c_str(), planning_frame.c_str(), ex.what());
        return false;
      }
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    double elapsed_ms = 0.0;

    bool success = runSinglePlan(pose_for_planning, planning_frame, planner_id_, plan, elapsed_ms);
    std::string used_planner = planner_id_;

    if (!success && !fallback_planner_id_.empty() && fallback_planner_id_ != planner_id_) {
      success = runSinglePlan(pose_for_planning, planning_frame, fallback_planner_id_, plan, elapsed_ms);
      used_planner = fallback_planner_id_;
    }

    if (!success) {
      return false;
    }

    trajectory_msgs::msg::JointTrajectory trajectory = plan.trajectory_.joint_trajectory;
    if (trajectory.points.empty()) {
      RCLCPP_WARN(get_logger(), "MoveIt returned an empty JointTrajectory. Nothing will be published.");
      return false;
    }

    const double joint_path_length = computeJointPathLength(trajectory);
    const auto total_duration = trajectory.points.back().time_from_start;

    if (publish_positions_only_) {
      stripTimingAndDerivatives(trajectory);
    } else {
      for (size_t i = 1; i < trajectory.points.size(); ++i) {
        if (static_cast<rclcpp::Duration>(trajectory.points[i].time_from_start) <=
            static_cast<rclcpp::Duration>(trajectory.points[i - 1].time_from_start)) {
          RCLCPP_WARN(
              get_logger(),
              "JointTrajectory has non-increasing time_from_start at point %zu. Nothing will be published.", i);
          return false;
        }
      }
    }

    trajectory.header.stamp = now();
    trajectory_pub_->publish(trajectory);

    if (log_plan_latency_) {
      RCLCPP_INFO(
          get_logger(),
          "Published path with %zu points on %s. planner_id=%s, joint_path_length=%.6f rad, raw_duration=%.3f s, plan_latency=%.3f ms, publish_positions_only=%s",
          trajectory.points.size(), trajectory_topic_.c_str(), used_planner.c_str(), joint_path_length,
          rclcpp::Duration(total_duration).seconds(), elapsed_ms,
          publish_positions_only_ ? "true" : "false");
    }
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

  std::vector<moveit_msgs::msg::CollisionObject> pending_collision_objects_;

  // 统一保护 MoveIt 接口对象、初始化状态以及 planning scene 更新。
  std::mutex planning_mutex_;

  std::string target_topic_;
  std::string trajectory_topic_;
  std::string collision_object_topic_;
  std::string plan_service_name_;
  std::string planning_group_;
  std::string target_frame_;
  std::string planner_id_;
  std::string fallback_planner_id_;

  bool moveit_ready_{false};
  bool plan_on_target_update_{false};
  bool publish_positions_only_{true};
  bool log_plan_latency_{true};

  double planning_time_{4.0};
  int num_planning_attempts_{8};
  double position_tolerance_{5.0e-3};
  double orientation_tolerance_{5.0e-2};
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
