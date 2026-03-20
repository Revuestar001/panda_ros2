#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>

#include "panda_moveit/static_sphere_scene.hpp"

class StaticObstaclesPublisherNode : public rclcpp::Node {
public:
  explicit StaticObstaclesPublisherNode(const rclcpp::NodeOptions& options)
  : Node("static_obstacles_publisher_node", options) {
    collision_object_topic_ =
        declare_parameter<std::string>("collision_object_topic", "/global_collision_object");
    world_frame_ = declare_parameter<std::string>("world_frame", "world");
    publish_delay_ms_ = declare_parameter<int>("publish_delay_ms", 500);
    obstacle_config_path_ = declare_parameter<std::string>(
        "obstacle_config_path",
        "/home/cyh/panda_ros2/model/franka_emika_panda/static_sphere_obstacles.xml");
    scene_xml_path_ = declare_parameter<std::string>(
        "scene_xml_path", "/home/cyh/panda_ros2/model/franka_emika_panda/scene_tau_ros.xml");
    obstacle_radius_padding_ = declare_parameter<double>("obstacle_radius_padding", 0.0);

    collision_object_pub_ = create_publisher<moveit_msgs::msg::CollisionObject>(
        collision_object_topic_, rclcpp::QoS(10).reliable().transient_local());

    publish_timer_ = create_wall_timer(
        std::chrono::milliseconds(publish_delay_ms_), [this]() { publishObstacles(); });

    RCLCPP_INFO(
        get_logger(),
        "Static obstacles publisher created. topic=%s, frame=%s, obstacle_config=%s, scene_xml=%s, obstacle_radius_padding=%.4f",
        collision_object_topic_.c_str(), world_frame_.c_str(),
        obstacle_config_path_.c_str(), scene_xml_path_.c_str(), obstacle_radius_padding_);
  }

private:
  void publishObstacles() {
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
      publish_timer_->cancel();
      return;
    }

    if (obstacles.empty()) {
      RCLCPP_WARN(
          get_logger(),
          "Obstacle source '%s' does not contain any static sphere obstacles. Nothing will be published.",
          obstacle_source_path.c_str());
      publish_timer_->cancel();
      return;
    }

    for (const auto& obstacle : obstacles) {
      auto object = panda_moveit::toCollisionObject(
          obstacle, world_frame_, obstacle_radius_padding_);
      object.header.stamp = now();
      collision_object_pub_->publish(object);
      RCLCPP_INFO(
          get_logger(),
          "Published static obstacle '%s' from '%s' in frame '%s' with radius padding %.4f m.",
          object.id.c_str(), obstacle_source_path.c_str(), object.header.frame_id.c_str(),
          obstacle_radius_padding_);
    }

    // QoS 使用 transient_local，发布一次后保留样本供晚加入订阅者获取。
    publish_timer_->cancel();
  }

  rclcpp::Publisher<moveit_msgs::msg::CollisionObject>::SharedPtr collision_object_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;

  std::string collision_object_topic_;
  std::string world_frame_;
  std::string obstacle_config_path_;
  std::string scene_xml_path_;
  double obstacle_radius_padding_{0.0};
  int publish_delay_ms_{500};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<StaticObstaclesPublisherNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
