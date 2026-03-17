#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// 目标位姿消息：用于接收末端目标位姿
#include <geometry_msgs/msg/pose_stamped.hpp>

// MoveIt2 高层规划接口：用于设置目标并规划
#include <moveit/move_group_interface/move_group_interface.h>

// MoveIt2 场景接口：用于向 planning scene 中添加障碍物
#include <moveit/planning_scene_interface/planning_scene_interface.h>

// 碰撞物体消息：用于向 MoveIt 场景中添加/删除障碍物
#include <moveit_msgs/msg/collision_object.hpp>

// MoveIt 规划场景服务：这里主要用于检测 move_group 是否已启动
#include <moveit_msgs/srv/get_planning_scene.hpp>

// ROS2 基础节点接口
#include <rclcpp/rclcpp.hpp>

// 简单触发服务：用于外部请求“规划一次”
#include <std_srvs/srv/trigger.hpp>

// 规划输出：MoveIt 规划出的关节轨迹
#include <trajectory_msgs/msg/joint_trajectory.hpp>

// 全局规划节点：
// 职责：
// 1. 接收目标位姿
// 2. 接收障碍物
// 3. 调用 MoveIt2 做全局规划
// 4. 发布规划得到的关节轨迹
class GlobalPlanningNode : public rclcpp::Node {
public:
  explicit GlobalPlanningNode(const rclcpp::NodeOptions& options)
  : Node("global_planning_node", options) {
    // -----------------------------
    // 读取参数
    // -----------------------------

    // MoveIt 规划组名称，例如 panda_arm
    planning_group_ = declare_parameter<std::string>("planning_group", "panda_arm");

    // 目标位姿订阅 topic
    target_topic_ = declare_parameter<std::string>("target_topic", "/global_target_pose");

    // 规划结果（关节轨迹）发布 topic
    trajectory_topic_ = declare_parameter<std::string>("trajectory_topic", "/global_joint_trajectory");

    // 障碍物订阅 topic
    collision_object_topic_ =
        declare_parameter<std::string>("collision_object_topic", "/global_collision_object");

    // 触发规划服务名称
    plan_service_name_ = declare_parameter<std::string>("plan_service", "/plan_global_trajectory");

    // 是否在收到目标后立即自动规划
    plan_on_target_update_ = declare_parameter<bool>("plan_on_target_update", false);

    // MoveIt 规划时间上限
    planning_time_ = declare_parameter<double>("planning_time", 2.0);

    // MoveIt 规划尝试次数
    num_planning_attempts_ = declare_parameter<int>("num_planning_attempts", 3);

    // 目标位置容差
    position_tolerance_ = declare_parameter<double>("goal_position_tolerance", 1.0e-3);

    // 目标姿态容差
    orientation_tolerance_ = declare_parameter<double>("goal_orientation_tolerance", 1.0e-3);

    // -----------------------------
    // 创建 publisher
    // -----------------------------

    // 发布 MoveIt 规划后的 joint trajectory
    trajectory_pub_ =
        create_publisher<trajectory_msgs::msg::JointTrajectory>(trajectory_topic_, rclcpp::QoS(1).reliable());

    // -----------------------------
    // 创建 subscriber：目标位姿
    // -----------------------------
    target_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        target_topic_, rclcpp::QoS(10),
        [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) { onTargetPose(msg); });

    // -----------------------------
    // 创建 subscriber：碰撞物体
    // -----------------------------
    collision_object_sub_ = create_subscription<moveit_msgs::msg::CollisionObject>(
        collision_object_topic_, rclcpp::QoS(10),
        [this](const moveit_msgs::msg::CollisionObject::SharedPtr msg) { onCollisionObject(msg); });

    // -----------------------------
    // 创建 service：触发规划
    // -----------------------------
    plan_srv_ = create_service<std_srvs::srv::Trigger>(
        plan_service_name_,
        [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
               std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
          const bool ok = planAndPublish();
          response->success = ok;
          response->message = ok ? "Global trajectory planned and published."
                                 : "Global planning failed. See node logs.";
        });

    // -----------------------------
    // 创建 client：用于检测 MoveIt 规划场景服务是否上线
    // -----------------------------
    planning_scene_client_ =
        create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");

    // -----------------------------
    // 创建定时器：周期性尝试初始化 MoveIt
    // -----------------------------
    // 为什么不在构造函数里直接初始化？
    // 因为 move_group 可能还没启动，需要等它可用
    init_timer_ = create_wall_timer(
        std::chrono::milliseconds(500), [this]() { tryInitializeMoveIt(); });

    RCLCPP_INFO(
        get_logger(),
        "Global planner node created. planning_group=%s, target_topic=%s, trajectory_topic=%s, collision_object_topic=%s, service=%s",
        planning_group_.c_str(), target_topic_.c_str(), trajectory_topic_.c_str(), collision_object_topic_.c_str(),
        plan_service_name_.c_str());
  }

private:
  // 周期性尝试初始化 MoveIt 接口
  void tryInitializeMoveIt() {
    // 已经初始化成功则直接返回
    if (moveit_ready_) {
      return;
    }

    // 检查 MoveIt 的 /get_planning_scene 服务是否存在
    if (!planning_scene_client_->wait_for_service(std::chrono::seconds(0))) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 3000,
          "Waiting for MoveIt '/get_planning_scene' service. Start move_group before using the global planner.");
      return;
    }

    try {
      // 创建 planning scene interface，用于管理障碍物
      planning_scene_interface_ =
          std::make_unique<moveit::planning_interface::PlanningSceneInterface>();

      // 创建 move group interface，用于设置目标并调用规划器
      move_group_ =
          std::make_unique<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), planning_group_);

      // 设置 MoveIt 规划参数
      move_group_->setPlanningTime(planning_time_);
      move_group_->setNumPlanningAttempts(num_planning_attempts_);
      move_group_->setGoalPositionTolerance(position_tolerance_);
      move_group_->setGoalOrientationTolerance(orientation_tolerance_);
      move_group_->setPlannerId("RRTConnectkConfigDefault");

      // 将规划起始状态设置为当前机器人状态
      move_group_->setStartStateToCurrentState();

      // 如果在 MoveIt 未就绪前收到过碰撞物体，则现在统一补进 planning scene
      {
        std::lock_guard<std::mutex> lock(collision_object_mutex_);
        for (const auto& collision_object : pending_collision_objects_) {
          planning_scene_interface_->applyCollisionObject(collision_object);
        }
        pending_collision_objects_.clear();
      }

      // 初始化成功
      moveit_ready_ = true;

      // 停止定时重试
      init_timer_->cancel();

      RCLCPP_INFO(
          get_logger(),
          "Global planner node ready. planning_group=%s, target_topic=%s, trajectory_topic=%s, collision_object_topic=%s, service=%s",
          planning_group_.c_str(), target_topic_.c_str(), trajectory_topic_.c_str(), collision_object_topic_.c_str(),
          plan_service_name_.c_str());
    } catch (const std::exception& ex) {
      // 初始化失败则清空对象，等待下次 timer 再重试
      planning_scene_interface_.reset();
      move_group_.reset();
      RCLCPP_ERROR_THROTTLE(
          get_logger(), *get_clock(), 3000,
          "Failed to initialize MoveIt interfaces: %s", ex.what());
    }
  }

  // 收到新的目标位姿
  void onTargetPose(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
    {
      // 用互斥锁保护 latest_target_pose_
      std::lock_guard<std::mutex> lock(target_mutex_);
      latest_target_pose_ = *msg;
    }

    RCLCPP_INFO(
        get_logger(),
        "Received target pose in frame '%s'. plan_on_target_update=%s",
        msg->header.frame_id.c_str(),
        plan_on_target_update_ ? "true" : "false");

    // 如果配置为“目标一更新就规划”，则立即触发规划
    if (plan_on_target_update_) {
      (void)planAndPublish();
    }
  }

  // 收到新的碰撞物体
  void onCollisionObject(const moveit_msgs::msg::CollisionObject::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(collision_object_mutex_);

    // 如果 MoveIt 还没 ready，则先缓存
    if (!planning_scene_interface_) {
      pending_collision_objects_.push_back(*msg);
      RCLCPP_WARN(
          get_logger(),
          "MoveIt is not ready yet. Queued collision object '%s' for later application.",
          msg->id.c_str());
      return;
    }

    // MoveIt 已经 ready，直接把碰撞物体应用到 planning scene
    planning_scene_interface_->applyCollisionObject(*msg);
    RCLCPP_INFO(
        get_logger(),
        "Applied collision object '%s' in frame '%s' with operation=%d",
        msg->id.c_str(),
        msg->header.frame_id.c_str(),
        static_cast<int>(msg->operation));
  }

  // 核心函数：读取最近目标 -> 调 MoveIt 规划 -> 发布轨迹
  bool planAndPublish() {
    std::optional<geometry_msgs::msg::PoseStamped> target_pose;

    {
      // 线程安全地读取最新目标位姿
      std::lock_guard<std::mutex> lock(target_mutex_);
      target_pose = latest_target_pose_;
    }

    // 如果压根没有收到过目标，则不能规划
    if (!target_pose.has_value()) {
      RCLCPP_WARN(get_logger(), "No target pose received yet. Planning request ignored.");
      return false;
    }

    // 如果 MoveIt 还没 ready，则不能规划
    if (!moveit_ready_ || !move_group_) {
      RCLCPP_WARN(
          get_logger(),
          "MoveIt is not ready yet. Make sure 'demo.launch.py' or 'move_group' is running before triggering planning.");
      return false;
    }

    // 规划起点设为当前机器人状态
    move_group_->setStartStateToCurrentState();

    // 规划目标设为末端位姿
    move_group_->setPoseTarget(*target_pose);

    // 调用 MoveIt 规划
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    const bool success = static_cast<bool>(move_group_->plan(plan));
    if (!success) {
      RCLCPP_WARN(get_logger(), "MoveIt planning failed.");
      move_group_->clearPoseTargets();
      return false;
    }

    // 取出规划结果中的 joint trajectory
    trajectory_msgs::msg::JointTrajectory trajectory = plan.trajectory_.joint_trajectory;

    // 用当前时间给消息打时间戳
    trajectory.header.stamp = now();

    // 发布轨迹
    trajectory_pub_->publish(trajectory);

    // 清掉 pose target，避免影响后续规划
    move_group_->clearPoseTargets();

    RCLCPP_INFO(
        get_logger(),
        "Published joint trajectory with %zu points on %s",
        trajectory.points.size(),
        trajectory_topic_.c_str());
    return true;
  }

  // -----------------------------
  // MoveIt 核心接口
  // -----------------------------

  // 用于进行规划
  std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;

  // 用于管理 planning scene 中的障碍物
  std::unique_ptr<moveit::planning_interface::PlanningSceneInterface> planning_scene_interface_;

  // -----------------------------
  // ROS2 通信对象
  // -----------------------------

  // 发布全局规划轨迹
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_pub_;

  // 订阅末端目标位姿
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_sub_;

  // 订阅碰撞物体
  rclcpp::Subscription<moveit_msgs::msg::CollisionObject>::SharedPtr collision_object_sub_;

  // 提供“触发规划”服务
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr plan_srv_;

  // 用于检查 /get_planning_scene 服务是否存在
  rclcpp::Client<moveit_msgs::srv::GetPlanningScene>::SharedPtr planning_scene_client_;

  // 周期性尝试初始化 MoveIt
  rclcpp::TimerBase::SharedPtr init_timer_;

  // -----------------------------
  // 线程同步相关
  // -----------------------------

  // 保护最新目标位姿
  std::mutex target_mutex_;

  // 最近收到的目标位姿
  std::optional<geometry_msgs::msg::PoseStamped> latest_target_pose_;

  // 保护碰撞物体缓存
  std::mutex collision_object_mutex_;

  // MoveIt 未 ready 前收到的碰撞物体先缓存在这里
  std::vector<moveit_msgs::msg::CollisionObject> pending_collision_objects_;

  // -----------------------------
  // 参数与配置
  // -----------------------------
  std::string target_topic_;
  std::string trajectory_topic_;
  std::string collision_object_topic_;
  std::string plan_service_name_;
  std::string planning_group_;

  // MoveIt 是否已初始化成功
  bool moveit_ready_{false};

  // 收到目标时是否自动规划
  bool plan_on_target_update_{false};

  // MoveIt 规划时间上限
  double planning_time_{2.0};

  // MoveIt 规划尝试次数
  int num_planning_attempts_{3};

  // 末端位置容差
  double position_tolerance_{1.0e-3};

  // 末端姿态容差
  double orientation_tolerance_{1.0e-3};
};

// 主函数：
// 1. 初始化 ROS2
// 2. 创建节点
// 3. 用多线程执行器运行
// 4. 退出时关闭 ROS2
int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<GlobalPlanningNode>(
      rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true));

  // 多线程执行器：允许订阅、服务、timer 等回调并发执行
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}