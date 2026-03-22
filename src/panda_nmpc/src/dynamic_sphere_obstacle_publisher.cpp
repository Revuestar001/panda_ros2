#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "panda_interfaces/msg/dynamic_sphere.hpp"
#include "panda_interfaces/msg/dynamic_sphere_array.hpp"

class DynamicSphereObstaclePublisher : public rclcpp::Node {
private:
    struct OscillatingSphereConfig {
        std::string name;
        double radius{0.05};
        std::array<double, 3> reference_position{{0.0, 0.0, 0.0}};
        std::array<double, 3> amplitude{{0.0, 0.0, 0.0}};
        std::array<double, 3> frequency_hz{{0.0, 0.0, 0.0}};
        std::array<double, 3> phase_offset{{0.0, 0.0, 0.0}};
    };

    using DynamicSphereMsg = panda_interfaces::msg::DynamicSphere;
    using DynamicSphereArrayMsg = panda_interfaces::msg::DynamicSphereArray;

    rclcpp::Publisher<DynamicSphereArrayMsg>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::vector<OscillatingSphereConfig> obstacle_configs_;

    std::string topic_{"/dynamic_sphere_obstacles"};
    std::string frame_id_{"world"};
    double publish_rate_hz_{30.0};
    std::chrono::steady_clock::time_point steady_start_time_{std::chrono::steady_clock::now()};

    static std::vector<std::string> defaultObstacleNames() {
        return {"dyn_obs_left", "dyn_obs_right"};
    }

    static std::vector<double> defaultObstacleRadii() {
        return {0.07, 0.06};
    }

    static std::vector<double> defaultReferencePositions() {
        return {
            0.50, -0.18, 0.35,
            0.55,  0.18, 0.45,
        };
    }

    static std::vector<double> defaultAmplitudes() {
        return {
            0.00, 0.12, 0.00,
            0.08, 0.00, 0.05,
        };
    }

    static std::vector<double> defaultFrequenciesHz() {
        return {
            0.00, 0.08, 0.00,
            0.06, 0.00, 0.12,
        };
    }

    static std::vector<double> defaultPhaseOffsets() {
        constexpr double half_pi = 1.5707963267948966;
        constexpr double pi = 3.1415926535897932;
        return {
            0.0, 0.0, 0.0,
            half_pi, 0.0, pi,
        };
    }

    static std::array<double, 3> unpackTriplet(
        const std::vector<double>& values,
        std::size_t obstacle_index,
        const std::string& param_name) {
        const std::size_t offset = 3 * obstacle_index;
        if (offset + 2 >= values.size()) {
            throw std::runtime_error(
                "Parameter '" + param_name + "' does not contain enough values for obstacle index " +
                std::to_string(obstacle_index) + ".");
        }
        return {values[offset + 0], values[offset + 1], values[offset + 2]};
    }

    static void validateVectorSize(
        const std::string& param_name,
        std::size_t actual_size,
        std::size_t expected_size) {
        if (actual_size != expected_size) {
            throw std::runtime_error(
                "Parameter '" + param_name + "' must contain exactly " +
                std::to_string(expected_size) + " values, but got " + std::to_string(actual_size) + ".");
        }
    }

    double motionTimeSeconds() const {
        const rclcpp::Time ros_time_now = this->now();
        if (ros_time_now.nanoseconds() != 0) {
            return ros_time_now.seconds();
        }

        return std::chrono::duration<double>(std::chrono::steady_clock::now() - steady_start_time_).count();
    }

    void declareParameters() {
        topic_ = this->declare_parameter<std::string>("topic", topic_);
        frame_id_ = this->declare_parameter<std::string>("frame_id", frame_id_);
        publish_rate_hz_ = std::max(
            1.0e-3, this->declare_parameter<double>("publish_rate_hz", publish_rate_hz_));

        const auto names =
            this->declare_parameter<std::vector<std::string>>("obstacles.names", defaultObstacleNames());
        const auto radii =
            this->declare_parameter<std::vector<double>>("obstacles.radii", defaultObstacleRadii());
        const auto reference_positions = this->declare_parameter<std::vector<double>>(
            "obstacles.reference_positions", defaultReferencePositions());
        const auto amplitudes =
            this->declare_parameter<std::vector<double>>("obstacles.amplitudes", defaultAmplitudes());
        const auto frequencies_hz = this->declare_parameter<std::vector<double>>(
            "obstacles.frequencies_hz", defaultFrequenciesHz());
        const auto phase_offsets = this->declare_parameter<std::vector<double>>(
            "obstacles.phase_offsets", defaultPhaseOffsets());

        if (names.empty()) {
            throw std::runtime_error("Parameter 'obstacles.names' must contain at least one obstacle.");
        }

        validateVectorSize("obstacles.radii", radii.size(), names.size());
        validateVectorSize("obstacles.reference_positions", reference_positions.size(), 3 * names.size());
        validateVectorSize("obstacles.amplitudes", amplitudes.size(), 3 * names.size());
        validateVectorSize("obstacles.frequencies_hz", frequencies_hz.size(), 3 * names.size());
        validateVectorSize("obstacles.phase_offsets", phase_offsets.size(), 3 * names.size());

        obstacle_configs_.clear();
        obstacle_configs_.reserve(names.size());

        for (std::size_t obstacle_index = 0; obstacle_index < names.size(); ++obstacle_index) {
            if (radii[obstacle_index] <= 0.0) {
                throw std::runtime_error(
                    "Parameter 'obstacles.radii' must be strictly positive for obstacle '" +
                    names[obstacle_index] + "'.");
            }

            OscillatingSphereConfig config;
            config.name = names[obstacle_index];
            config.radius = radii[obstacle_index];
            config.reference_position =
                unpackTriplet(reference_positions, obstacle_index, "obstacles.reference_positions");
            config.amplitude = unpackTriplet(amplitudes, obstacle_index, "obstacles.amplitudes");
            config.frequency_hz = unpackTriplet(frequencies_hz, obstacle_index, "obstacles.frequencies_hz");
            config.phase_offset = unpackTriplet(phase_offsets, obstacle_index, "obstacles.phase_offsets");
            obstacle_configs_.push_back(config);
        }
    }

    void logConfiguration() const {
        std::ostringstream stream;
        stream << "Dynamic sphere obstacle publisher started. topic=" << topic_
               << ", frame_id=" << frame_id_
               << ", publish_rate_hz=" << publish_rate_hz_
               << ", obstacles=" << obstacle_configs_.size() << ": ";

        for (std::size_t obstacle_index = 0; obstacle_index < obstacle_configs_.size(); ++obstacle_index) {
            const auto& obstacle = obstacle_configs_[obstacle_index];
            if (obstacle_index > 0) {
                stream << "; ";
            }
            stream << obstacle.name
                   << " ref=("
                   << obstacle.reference_position[0] << ", "
                   << obstacle.reference_position[1] << ", "
                   << obstacle.reference_position[2] << ")"
                   << " amp=("
                   << obstacle.amplitude[0] << ", "
                   << obstacle.amplitude[1] << ", "
                   << obstacle.amplitude[2] << ")"
                   << " freq_hz=("
                   << obstacle.frequency_hz[0] << ", "
                   << obstacle.frequency_hz[1] << ", "
                   << obstacle.frequency_hz[2] << ")"
                   << " radius=" << obstacle.radius;
        }

        RCLCPP_INFO(this->get_logger(), "%s", stream.str().c_str());
    }

    void publishDynamicObstacles() {
        constexpr double kTwoPi = 6.2831853071795865;

        DynamicSphereArrayMsg msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = frame_id_;
        msg.spheres.reserve(obstacle_configs_.size());

        const double time_sec = motionTimeSeconds();
        for (const auto& obstacle : obstacle_configs_) {
            DynamicSphereMsg sphere_msg;
            sphere_msg.name = obstacle.name;
            sphere_msg.radius = obstacle.radius;

            for (int axis = 0; axis < 3; ++axis) {
                const double omega = kTwoPi * obstacle.frequency_hz[static_cast<std::size_t>(axis)];
                const double argument = omega * time_sec + obstacle.phase_offset[static_cast<std::size_t>(axis)];
                const double position = obstacle.reference_position[static_cast<std::size_t>(axis)] +
                                        obstacle.amplitude[static_cast<std::size_t>(axis)] * std::sin(argument);
                const double velocity =
                    obstacle.amplitude[static_cast<std::size_t>(axis)] * omega * std::cos(argument);

                if (axis == 0) {
                    sphere_msg.center.x = position;
                    sphere_msg.velocity.x = velocity;
                } else if (axis == 1) {
                    sphere_msg.center.y = position;
                    sphere_msg.velocity.y = velocity;
                } else {
                    sphere_msg.center.z = position;
                    sphere_msg.velocity.z = velocity;
                }
            }

            msg.spheres.push_back(sphere_msg);
        }

        publisher_->publish(msg);
    }

public:
    DynamicSphereObstaclePublisher()
        : rclcpp::Node("dynamic_sphere_obstacle_publisher") {
        declareParameters();

        publisher_ = this->create_publisher<DynamicSphereArrayMsg>(topic_, 10);

        const auto publish_period = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(1.0 / publish_rate_hz_));
        timer_ = this->create_wall_timer(
            publish_period, [this]() { publishDynamicObstacles(); });

        logConfiguration();
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DynamicSphereObstaclePublisher>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
