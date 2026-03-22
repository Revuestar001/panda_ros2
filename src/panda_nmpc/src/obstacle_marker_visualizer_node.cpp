#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "panda_interfaces/msg/dynamic_sphere_array.hpp"
#include "static_sphere_scene.hpp"

class ObstacleMarkerVisualizerNode : public rclcpp::Node {
private:
    using DynamicSphereArrayMsg = panda_interfaces::msg::DynamicSphereArray;
    using Marker = visualization_msgs::msg::Marker;
    using MarkerArray = visualization_msgs::msg::MarkerArray;

    struct DynamicSphereState {
        std::string name;
        double x{0.0};
        double y{0.0};
        double z{0.0};
        double radius{0.0};
    };

    std::string frame_id_{"link0"};
    std::string obstacle_config_path_{
        "/home/cyh/panda_ros2/model/franka_emika_panda/static_sphere_obstacles.xml"};
    std::string dynamic_obstacle_topic_{"/dynamic_sphere_obstacles"};
    std::string marker_topic_{"/obstacle_markers"};
    double publish_rate_hz_{10.0};
    double dynamic_timeout_sec_{0.5};
    std::array<double, 4> static_rgba_{{1.0, 0.2, 0.2, 0.55}};
    std::array<double, 4> dynamic_rgba_{{0.2, 0.8, 1.0, 0.55}};

    std::vector<panda_nmpc::StaticSphereObstacle> static_obstacles_;
    std::vector<DynamicSphereState> dynamic_obstacles_;
    rclcpp::Time dynamic_obstacle_stamp_{0, 0, RCL_ROS_TIME};
    mutable std::mutex dynamic_mutex_;

    rclcpp::Publisher<MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Subscription<DynamicSphereArrayMsg>::SharedPtr dynamic_obstacle_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    static std::vector<double> toVector(const std::array<double, 4>& rgba) {
        return std::vector<double>(rgba.begin(), rgba.end());
    }

    std::array<double, 4> declareRgbaParameter(
        const std::string& name,
        const std::array<double, 4>& defaults) {
        const auto raw = this->declare_parameter<std::vector<double>>(name, toVector(defaults));
        if (raw.size() != 4) {
            throw std::runtime_error("Parameter '" + name + "' must contain exactly 4 values.");
        }
        return {{raw[0], raw[1], raw[2], raw[3]}};
    }

    void declareParameters() {
        frame_id_ = this->declare_parameter<std::string>("frame_id", frame_id_);
        obstacle_config_path_ =
            this->declare_parameter<std::string>("obstacle_config_path", obstacle_config_path_);
        dynamic_obstacle_topic_ =
            this->declare_parameter<std::string>("dynamic_obstacle_topic", dynamic_obstacle_topic_);
        marker_topic_ = this->declare_parameter<std::string>("marker_topic", marker_topic_);
        publish_rate_hz_ = std::max(1.0e-3, this->declare_parameter<double>("publish_rate_hz", publish_rate_hz_));
        dynamic_timeout_sec_ =
            std::max(0.0, this->declare_parameter<double>("dynamic_timeout_sec", dynamic_timeout_sec_));
        static_rgba_ = declareRgbaParameter("static_rgba", static_rgba_);
        dynamic_rgba_ = declareRgbaParameter("dynamic_rgba", dynamic_rgba_);
    }

    void loadStaticObstacles() {
        static_obstacles_ = panda_nmpc::loadStaticSphereObstaclesFromSceneXml(obstacle_config_path_);
        RCLCPP_INFO(
            this->get_logger(),
            "Obstacle marker visualizer loaded %zu static obstacle(s) from %s and listens to %s",
            static_obstacles_.size(),
            obstacle_config_path_.c_str(),
            dynamic_obstacle_topic_.c_str());
    }

    void dynamicObstacleCallback(const DynamicSphereArrayMsg::SharedPtr msg) {
        std::vector<DynamicSphereState> parsed;
        parsed.reserve(msg->spheres.size());

        for (const auto& sphere_msg : msg->spheres) {
            const auto finite = [](double value) { return std::isfinite(value); };
            if (!finite(sphere_msg.center.x) || !finite(sphere_msg.center.y) || !finite(sphere_msg.center.z) ||
                !finite(sphere_msg.radius) || sphere_msg.radius <= 0.0) {
                continue;
            }

            DynamicSphereState state;
            state.name = sphere_msg.name;
            state.x = sphere_msg.center.x;
            state.y = sphere_msg.center.y;
            state.z = sphere_msg.center.z;
            state.radius = sphere_msg.radius;
            parsed.push_back(state);
        }

        rclcpp::Time stamp = this->now();
        if (msg->header.stamp.sec != 0 || msg->header.stamp.nanosec != 0) {
            stamp = rclcpp::Time(msg->header.stamp);
        }

        std::lock_guard<std::mutex> lock(dynamic_mutex_);
        dynamic_obstacles_ = std::move(parsed);
        dynamic_obstacle_stamp_ = stamp;
    }

    Marker makeDeleteAllMarker() const {
        Marker marker;
        marker.action = Marker::DELETEALL;
        return marker;
    }

    Marker makeSphereMarker(
        const std::string& ns,
        int id,
        double x,
        double y,
        double z,
        double radius,
        const std::array<double, 4>& rgba,
        const rclcpp::Time& stamp) const {
        Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = stamp;
        marker.ns = ns;
        marker.id = id;
        marker.type = Marker::SPHERE;
        marker.action = Marker::ADD;
        marker.pose.position.x = x;
        marker.pose.position.y = y;
        marker.pose.position.z = z;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 2.0 * radius;
        marker.scale.y = 2.0 * radius;
        marker.scale.z = 2.0 * radius;
        marker.color.r = static_cast<float>(rgba[0]);
        marker.color.g = static_cast<float>(rgba[1]);
        marker.color.b = static_cast<float>(rgba[2]);
        marker.color.a = static_cast<float>(rgba[3]);
        return marker;
    }

    void publishMarkers() {
        std::vector<DynamicSphereState> dynamic_snapshot;
        rclcpp::Time dynamic_stamp = dynamic_obstacle_stamp_;
        {
            std::lock_guard<std::mutex> lock(dynamic_mutex_);
            dynamic_snapshot = dynamic_obstacles_;
            dynamic_stamp = dynamic_obstacle_stamp_;
        }

        const rclcpp::Time now = this->now();
        const bool dynamic_fresh =
            dynamic_stamp.nanoseconds() > 0 &&
            (dynamic_timeout_sec_ <= 0.0 || (now - dynamic_stamp).seconds() <= dynamic_timeout_sec_);

        MarkerArray marker_array;
        marker_array.markers.push_back(makeDeleteAllMarker());

        int marker_id = 0;
        for (const auto& obstacle : static_obstacles_) {
            marker_array.markers.push_back(makeSphereMarker(
                "static_obstacles",
                marker_id++,
                obstacle.center.x(),
                obstacle.center.y(),
                obstacle.center.z(),
                obstacle.radius,
                static_rgba_,
                now));
        }

        if (dynamic_fresh) {
            marker_id = 0;
            for (const auto& obstacle : dynamic_snapshot) {
                marker_array.markers.push_back(makeSphereMarker(
                    "dynamic_obstacles",
                    marker_id++,
                    obstacle.x,
                    obstacle.y,
                    obstacle.z,
                    obstacle.radius,
                    dynamic_rgba_,
                    now));
            }
        }

        marker_pub_->publish(marker_array);
    }

public:
    ObstacleMarkerVisualizerNode()
        : rclcpp::Node("obstacle_marker_visualizer_node") {
        declareParameters();
        loadStaticObstacles();

        marker_pub_ = this->create_publisher<MarkerArray>(marker_topic_, 10);
        dynamic_obstacle_sub_ = this->create_subscription<DynamicSphereArrayMsg>(
            dynamic_obstacle_topic_, 10,
            [this](const DynamicSphereArrayMsg::SharedPtr msg) { dynamicObstacleCallback(msg); });

        const auto publish_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / publish_rate_hz_));
        timer_ = this->create_wall_timer(
            publish_period, [this]() { publishMarkers(); });
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ObstacleMarkerVisualizerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
