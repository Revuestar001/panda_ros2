#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <rclcpp/rclcpp.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>

class StaticObstaclesPublisherNode : public rclcpp::Node {
public:
  explicit StaticObstaclesPublisherNode(const rclcpp::NodeOptions& options)
  : Node("static_obstacles_publisher_node", options) {
    collision_object_topic_ =
        declare_parameter<std::string>("collision_object_topic", "/global_collision_object");
    world_frame_ = declare_parameter<std::string>("world_frame", "world");
    publish_delay_ms_ = declare_parameter<int>("publish_delay_ms", 500);

    // 默认与 panda_nmpc_controller.hpp 中当前写死的两个球一致。
    obstacle_1_enabled_ = declare_parameter<bool>("obstacle_1.enabled", true);
    obstacle_1_x_ = declare_parameter<double>("obstacle_1.x", 0.4);
    obstacle_1_y_ = declare_parameter<double>("obstacle_1.y", -0.1);
    obstacle_1_z_ = declare_parameter<double>("obstacle_1.z", 0.35);
    obstacle_1_radius_ = declare_parameter<double>("obstacle_1.radius", 0.1);

    obstacle_2_enabled_ = declare_parameter<bool>("obstacle_2.enabled", true);
    obstacle_2_x_ = declare_parameter<double>("obstacle_2.x", 0.4);
    obstacle_2_y_ = declare_parameter<double>("obstacle_2.y", 0.1);
    obstacle_2_z_ = declare_parameter<double>("obstacle_2.z", 0.35);
    obstacle_2_radius_ = declare_parameter<double>("obstacle_2.radius", 0.1);

    collision_object_pub_ = create_publisher<moveit_msgs::msg::CollisionObject>(
        collision_object_topic_, rclcpp::QoS(10).reliable().transient_local());

    publish_timer_ = create_wall_timer(
        std::chrono::milliseconds(publish_delay_ms_), [this]() { publishObstacles(); });

    RCLCPP_INFO(
        get_logger(),
        "Static obstacles publisher created. topic=%s, frame=%s",
        collision_object_topic_.c_str(), world_frame_.c_str());
  }

private:
  moveit_msgs::msg::CollisionObject makeSphereObstacle(
      const std::string& id, double x, double y, double z, double radius) const {
    moveit_msgs::msg::CollisionObject object;
    object.header.stamp = now();
    object.header.frame_id = world_frame_;
    object.id = id;
    object.operation = moveit_msgs::msg::CollisionObject::ADD;

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = shape_msgs::msg::SolidPrimitive::SPHERE;
    primitive.dimensions = {radius};

    geometry_msgs::msg::Pose pose;
    pose.position.x = x;
    pose.position.y = y;
    pose.position.z = z;
    pose.orientation.w = 1.0;

    object.primitives.push_back(primitive);
    object.primitive_poses.push_back(pose);
    return object;
  }

  void publishObstacles() {
    std::vector<moveit_msgs::msg::CollisionObject> objects;

    if (obstacle_1_enabled_) {
      objects.push_back(
          makeSphereObstacle("obs_sphere_1", obstacle_1_x_, obstacle_1_y_, obstacle_1_z_, obstacle_1_radius_));
    }
    if (obstacle_2_enabled_) {
      objects.push_back(
          makeSphereObstacle("obs_sphere_2", obstacle_2_x_, obstacle_2_y_, obstacle_2_z_, obstacle_2_radius_));
    }

    if (objects.empty()) {
      RCLCPP_WARN(get_logger(), "No enabled static obstacles configured. Nothing will be published.");
      publish_timer_->cancel();
      return;
    }

    for (const auto& object : objects) {
      collision_object_pub_->publish(object);
      RCLCPP_INFO(
          get_logger(),
          "Published static obstacle '%s' in frame '%s'.",
          object.id.c_str(), object.header.frame_id.c_str());
    }

    // QoS 使用 transient_local，发布一次后保留样本供晚加入订阅者获取。
    publish_timer_->cancel();
  }

  rclcpp::Publisher<moveit_msgs::msg::CollisionObject>::SharedPtr collision_object_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;

  std::string collision_object_topic_;
  std::string world_frame_;
  int publish_delay_ms_{500};

  bool obstacle_1_enabled_{true};
  double obstacle_1_x_{0.4};
  double obstacle_1_y_{-0.1};
  double obstacle_1_z_{0.35};
  double obstacle_1_radius_{0.1};

  bool obstacle_2_enabled_{true};
  double obstacle_2_x_{0.4};
  double obstacle_2_y_{0.1};
  double obstacle_2_z_{0.35};
  double obstacle_2_radius_{0.1};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);

  auto node = std::make_shared<StaticObstaclesPublisherNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
