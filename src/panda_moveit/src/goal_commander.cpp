#include <chrono>
#include <memory>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

class GoalCommanderNode : public rclcpp::Node {
public:
  explicit GoalCommanderNode(const rclcpp::NodeOptions& options)
  : Node("goal_commander_node", options) {
    target_topic_ = declare_parameter<std::string>("target_topic", "/global_target_pose");
    plan_service_name_ = declare_parameter<std::string>("plan_service", "/plan_global_trajectory");

    target_frame_ = declare_parameter<std::string>("target_frame", "world");
    target_x_ = declare_parameter<double>("target_x", 0.74);
    target_y_ = declare_parameter<double>("target_y", 0.0);
    target_z_ = declare_parameter<double>("target_z", 0.32);
    target_qx_ = declare_parameter<double>("target_qx", 1.0);
    target_qy_ = declare_parameter<double>("target_qy", 0.0);
    target_qz_ = declare_parameter<double>("target_qz", 0.0);
    target_qw_ = declare_parameter<double>("target_qw", 0.0);

    auto_plan_ = declare_parameter<bool>("auto_plan", true);
    stop_after_success_ = declare_parameter<bool>("stop_after_success", true);
    command_period_ms_ = declare_parameter<int>("command_period_ms", 500);
    plan_delay_after_publish_s_ = declare_parameter<double>("plan_delay_after_publish_s", 0.2);

    target_pub_ =
        create_publisher<geometry_msgs::msg::PoseStamped>(target_topic_, rclcpp::QoS(1).reliable());
    plan_client_ = create_client<std_srvs::srv::Trigger>(plan_service_name_);

    timer_ = create_wall_timer(
        std::chrono::milliseconds(command_period_ms_), [this]() { timerCallback(); });

    RCLCPP_INFO(
        get_logger(),
        "Goal commander created. target_topic=%s, plan_service=%s, auto_plan=%s, target=(%.3f, %.3f, %.3f) frame=%s",
        target_topic_.c_str(), plan_service_name_.c_str(), auto_plan_ ? "true" : "false",
        target_x_, target_y_, target_z_, target_frame_.c_str());
  }

private:
  enum class CommandStage {
    kWaitingForPlanner,
    kTargetPublished,
    kPlanningRequested,
    kCompleted,
  };

  geometry_msgs::msg::PoseStamped buildTargetPose() const {
    geometry_msgs::msg::PoseStamped pose;
    pose.header.stamp = now();
    pose.header.frame_id = target_frame_;
    pose.pose.position.x = target_x_;
    pose.pose.position.y = target_y_;
    pose.pose.position.z = target_z_;
    pose.pose.orientation.x = target_qx_;
    pose.pose.orientation.y = target_qy_;
    pose.pose.orientation.z = target_qz_;
    pose.pose.orientation.w = target_qw_;
    return pose;
  }

  void publishTargetPose() {
    auto pose = buildTargetPose();
    target_pub_->publish(pose);
    last_target_publish_time_ = this->now();

    RCLCPP_INFO(
        get_logger(),
        "Published target pose to %s: frame=%s, position=(%.3f, %.3f, %.3f)",
        target_topic_.c_str(), pose.header.frame_id.c_str(),
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);
  }

  void timerCallback() {
    if (stage_ == CommandStage::kCompleted || stage_ == CommandStage::kPlanningRequested) {
      return;
    }

    if (!plan_client_->wait_for_service(std::chrono::seconds(0))) {
      publishTargetPose();
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 3000,
          "Waiting for planning service '%s'.", plan_service_name_.c_str());
      return;
    }

    if (stage_ == CommandStage::kWaitingForPlanner) {
      publishTargetPose();
      if (!auto_plan_) {
        stage_ = CommandStage::kCompleted;
        if (stop_after_success_) {
          timer_->cancel();
        }
        return;
      }

      stage_ = CommandStage::kTargetPublished;
      return;
    }

    if (stage_ != CommandStage::kTargetPublished) {
      return;
    }

    if ((now() - last_target_publish_time_).seconds() < plan_delay_after_publish_s_) {
      return;
    }

    stage_ = CommandStage::kPlanningRequested;
    auto request = std::make_shared<std_srvs::srv::Trigger::Request>();
    (void)plan_client_->async_send_request(
        request,
        [this](rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture future) {
          handlePlanResponse(future);
        });

    RCLCPP_INFO(
        get_logger(),
        "Requested planning through service '%s'.", plan_service_name_.c_str());
  }

  void handlePlanResponse(
      const rclcpp::Client<std_srvs::srv::Trigger>::SharedFuture& future) {
    try {
      const auto response = future.get();
      if (response->success) {
        stage_ = CommandStage::kCompleted;
        RCLCPP_INFO(
            get_logger(),
            "Planning succeeded: %s", response->message.c_str());
        if (stop_after_success_) {
          timer_->cancel();
        }
        return;
      }

      RCLCPP_WARN(
          get_logger(),
          "Planning request returned failure: %s. The commander will retry.",
          response->message.c_str());
    } catch (const std::exception& ex) {
      RCLCPP_WARN(
          get_logger(),
          "Planning request failed with exception: %s. The commander will retry.",
          ex.what());
    }

    // Re-publish the target on the next timer tick before retrying the plan request.
    stage_ = CommandStage::kWaitingForPlanner;
  }

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_pub_;
  rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr plan_client_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::string target_topic_;
  std::string plan_service_name_;
  std::string target_frame_;

  double target_x_{0.74};
  double target_y_{0.0};
  double target_z_{0.32};
  double target_qx_{1.0};
  double target_qy_{0.0};
  double target_qz_{0.0};
  double target_qw_{0.0};

  bool auto_plan_{true};
  bool stop_after_success_{true};
  int command_period_ms_{500};
  double plan_delay_after_publish_s_{0.2};

  CommandStage stage_{CommandStage::kWaitingForPlanner};
  rclcpp::Time last_target_publish_time_{0, 0, RCL_ROS_TIME};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<GoalCommanderNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
