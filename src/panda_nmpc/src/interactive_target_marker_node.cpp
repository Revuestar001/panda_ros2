#include <array>
#include <memory>
#include <mutex>
#include <string>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <interactive_markers/interactive_marker_server.hpp>
#include <interactive_markers/menu_handler.hpp>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/interactive_marker.hpp>
#include <visualization_msgs/msg/interactive_marker_control.hpp>
#include <visualization_msgs/msg/interactive_marker_feedback.hpp>
#include <visualization_msgs/msg/marker.hpp>

class InteractiveTargetMarkerNode : public rclcpp::Node {
private:
    struct Parameters {
        std::string frame_id{"link0"};
        std::string target_pose_topic{"/global_target_pose"};
        std::string marker_namespace{"target_pose_marker"};
        std::string marker_name{"nmpc_target_pose"};
        std::string marker_description{"NMPC Target Pose"};
        double marker_scale{0.24};
        double sphere_radius{0.035};
        std::array<double, 3> initial_position{{0.55, -0.30, 0.20}};
        std::array<double, 4> initial_quaternion_xyzw{{1.0, 0.0, 0.0, 0.0}};
    };

    using InteractiveMarker = visualization_msgs::msg::InteractiveMarker;
    using InteractiveMarkerControl = visualization_msgs::msg::InteractiveMarkerControl;
    using InteractiveMarkerFeedback = visualization_msgs::msg::InteractiveMarkerFeedback;
    using Marker = visualization_msgs::msg::Marker;

    Parameters params_;
    geometry_msgs::msg::Pose current_pose_{};
    geometry_msgs::msg::Pose last_sent_pose_{};
    interactive_markers::MenuHandler menu_handler_;
    interactive_markers::MenuHandler::EntryHandle send_pose_handle_{0};
    interactive_markers::MenuHandler::EntryHandle reset_pose_handle_{0};
    std::shared_ptr<interactive_markers::InteractiveMarkerServer> marker_server_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_pub_;
    std::mutex pose_mutex_;

    static std::vector<double> toVector(const std::array<double, 3>& values) {
        return std::vector<double>(values.begin(), values.end());
    }

    static std::vector<double> toVectorQuat(const std::array<double, 4>& values) {
        return std::vector<double>(values.begin(), values.end());
    }

    std::array<double, 3> declareVec3Parameter(
        const std::string& name,
        const std::array<double, 3>& defaults) {
        const auto raw = this->declare_parameter<std::vector<double>>(name, toVector(defaults));
        if (raw.size() != 3) {
            throw std::runtime_error("Parameter '" + name + "' must contain exactly 3 values.");
        }
        return {{raw[0], raw[1], raw[2]}};
    }

    std::array<double, 4> declareQuatParameter(
        const std::string& name,
        const std::array<double, 4>& defaults) {
        const auto raw = this->declare_parameter<std::vector<double>>(name, toVectorQuat(defaults));
        if (raw.size() != 4) {
            throw std::runtime_error("Parameter '" + name + "' must contain exactly 4 values.");
        }
        return {{raw[0], raw[1], raw[2], raw[3]}};
    }

    void declareParameters() {
        params_.frame_id = this->declare_parameter<std::string>("frame_id", params_.frame_id);
        params_.target_pose_topic =
            this->declare_parameter<std::string>("target_pose_topic", params_.target_pose_topic);
        params_.marker_namespace =
            this->declare_parameter<std::string>("marker_namespace", params_.marker_namespace);
        params_.marker_name = this->declare_parameter<std::string>("marker_name", params_.marker_name);
        params_.marker_description =
            this->declare_parameter<std::string>("marker_description", params_.marker_description);
        params_.marker_scale = this->declare_parameter<double>("marker_scale", params_.marker_scale);
        params_.sphere_radius = this->declare_parameter<double>("sphere_radius", params_.sphere_radius);
        params_.initial_position =
            declareVec3Parameter("initial_position", params_.initial_position);
        params_.initial_quaternion_xyzw =
            declareQuatParameter("initial_quaternion_xyzw", params_.initial_quaternion_xyzw);
    }

    void initializePoses() {
        current_pose_.position.x = params_.initial_position[0];
        current_pose_.position.y = params_.initial_position[1];
        current_pose_.position.z = params_.initial_position[2];
        current_pose_.orientation.x = params_.initial_quaternion_xyzw[0];
        current_pose_.orientation.y = params_.initial_quaternion_xyzw[1];
        current_pose_.orientation.z = params_.initial_quaternion_xyzw[2];
        current_pose_.orientation.w = params_.initial_quaternion_xyzw[3];
        last_sent_pose_ = current_pose_;
    }

    Marker makeVisualMarker() const {
        Marker marker;
        marker.type = Marker::SPHERE;
        marker.scale.x = 2.0 * params_.sphere_radius;
        marker.scale.y = 2.0 * params_.sphere_radius;
        marker.scale.z = 2.0 * params_.sphere_radius;
        marker.color.r = 1.0f;
        marker.color.g = 0.75f;
        marker.color.b = 0.15f;
        marker.color.a = 0.85f;
        return marker;
    }

    void addAxisControl(
        InteractiveMarker& marker,
        const std::string& name,
        double ox,
        double oy,
        double oz,
        double ow,
        uint8_t interaction_mode) const {
        InteractiveMarkerControl control;
        control.name = name;
        control.orientation.x = ox;
        control.orientation.y = oy;
        control.orientation.z = oz;
        control.orientation.w = ow;
        control.interaction_mode = interaction_mode;
        marker.controls.push_back(control);
    }

    InteractiveMarker buildInteractiveMarker() const {
        InteractiveMarker marker;
        marker.header.frame_id = params_.frame_id;
        marker.name = params_.marker_name;
        marker.description = params_.marker_description;
        marker.scale = params_.marker_scale;
        marker.pose = current_pose_;

        InteractiveMarkerControl visual_control;
        visual_control.name = "menu";
        visual_control.always_visible = true;
        visual_control.interaction_mode = InteractiveMarkerControl::MENU;
        visual_control.markers.push_back(makeVisualMarker());
        marker.controls.push_back(visual_control);

        addAxisControl(
            marker, "rotate_x", 1.0, 0.0, 0.0, 1.0,
            InteractiveMarkerControl::ROTATE_AXIS);
        addAxisControl(
            marker, "move_x", 1.0, 0.0, 0.0, 1.0,
            InteractiveMarkerControl::MOVE_AXIS);
        addAxisControl(
            marker, "rotate_y", 0.0, 1.0, 0.0, 1.0,
            InteractiveMarkerControl::ROTATE_AXIS);
        addAxisControl(
            marker, "move_y", 0.0, 1.0, 0.0, 1.0,
            InteractiveMarkerControl::MOVE_AXIS);
        addAxisControl(
            marker, "rotate_z", 0.0, 0.0, 1.0, 1.0,
            InteractiveMarkerControl::ROTATE_AXIS);
        addAxisControl(
            marker, "move_z", 0.0, 0.0, 1.0, 1.0,
            InteractiveMarkerControl::MOVE_AXIS);

        return marker;
    }

    void insertMarker() {
        auto marker = buildInteractiveMarker();
        marker_server_->insert(
            marker,
            [this](const InteractiveMarkerFeedback::ConstSharedPtr& feedback) {
                processFeedback(feedback);
            });
        menu_handler_.apply(*marker_server_, params_.marker_name);
        marker_server_->applyChanges();
    }

    void publishCurrentPose() {
        geometry_msgs::msg::Pose pose_to_send;
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            pose_to_send = current_pose_;
            last_sent_pose_ = current_pose_;
        }

        geometry_msgs::msg::PoseStamped msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = params_.frame_id;
        msg.pose = pose_to_send;
        target_pose_pub_->publish(msg);

        RCLCPP_INFO(
            this->get_logger(),
            "Published target pose to %s: p=(%.4f, %.4f, %.4f), q_xyzw=(%.4f, %.4f, %.4f, %.4f)",
            params_.target_pose_topic.c_str(),
            pose_to_send.position.x,
            pose_to_send.position.y,
            pose_to_send.position.z,
            pose_to_send.orientation.x,
            pose_to_send.orientation.y,
            pose_to_send.orientation.z,
            pose_to_send.orientation.w);
    }

    void resetToLastSentPose() {
        geometry_msgs::msg::Pose pose_to_restore;
        {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            current_pose_ = last_sent_pose_;
            pose_to_restore = current_pose_;
        }

        marker_server_->setPose(params_.marker_name, pose_to_restore);
        marker_server_->applyChanges();
        RCLCPP_INFO(this->get_logger(), "Interactive target marker reset to the last sent pose.");
    }

    void processFeedback(const InteractiveMarkerFeedback::ConstSharedPtr& feedback) {
        if (feedback->event_type == InteractiveMarkerFeedback::POSE_UPDATE ||
            feedback->event_type == InteractiveMarkerFeedback::MOUSE_UP) {
            std::lock_guard<std::mutex> lock(pose_mutex_);
            current_pose_ = feedback->pose;
            return;
        }

        if (feedback->event_type == InteractiveMarkerFeedback::MENU_SELECT) {
            if (feedback->menu_entry_id == send_pose_handle_) {
                publishCurrentPose();
            } else if (feedback->menu_entry_id == reset_pose_handle_) {
                resetToLastSentPose();
            }
        }
    }

public:
    InteractiveTargetMarkerNode()
        : Node("interactive_target_marker_node") {
        declareParameters();
        initializePoses();

        target_pose_pub_ =
            this->create_publisher<geometry_msgs::msg::PoseStamped>(params_.target_pose_topic, 10);
        marker_server_ = std::make_shared<interactive_markers::InteractiveMarkerServer>(
            params_.marker_namespace,
            this->get_node_base_interface(),
            this->get_node_clock_interface(),
            this->get_node_logging_interface(),
            this->get_node_topics_interface(),
            this->get_node_services_interface());

        send_pose_handle_ = menu_handler_.insert(
            "Send Target Pose",
            [this](const InteractiveMarkerFeedback::ConstSharedPtr& feedback) {
                processFeedback(feedback);
            });
        reset_pose_handle_ = menu_handler_.insert(
            "Reset To Last Sent Pose",
            [this](const InteractiveMarkerFeedback::ConstSharedPtr& feedback) {
                processFeedback(feedback);
            });

        insertMarker();

        RCLCPP_INFO(
            this->get_logger(),
            "Interactive target marker ready. frame=%s, namespace=%s, topic=%s. Drag in RViz and right-click -> Send Target Pose.",
            params_.frame_id.c_str(),
            params_.marker_namespace.c_str(),
            params_.target_pose_topic.c_str());
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InteractiveTargetMarkerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
