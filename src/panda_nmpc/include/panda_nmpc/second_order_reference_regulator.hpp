#ifndef PANDA_NMPC_SECOND_ORDER_REFERENCE_REGULATOR_HPP_
#define PANDA_NMPC_SECOND_ORDER_REFERENCE_REGULATOR_HPP_

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include <Eigen/Dense>
#include <Eigen/Geometry>

// 一个实例只处理一种参考类型:
// 1) 位置模式: 输出平滑的位置/速度/加速度参考
// 2) 姿态模式: 输出平滑的四元数/角速度/角加速度参考
// 如果后续要同时滤位置和姿态，建议分别创建两个实例。
class SecondOrderReferenceRegulator {
public:
    using Vec3 = Eigen::Vector3d;
    using Mat3 = Eigen::Matrix3d;
    using Quat = Eigen::Quaterniond;

    enum class Mode {
        kUninitialized = 0,
        kPosition = 1,
        kOrientation = 2,
    };

    struct Config {
        double dt = 0.02;
        double natural_frequency = 6.0;
        double damping_ratio = 1.0;
        double max_linear_velocity = std::numeric_limits<double>::infinity();
        double max_linear_acceleration = std::numeric_limits<double>::infinity();
        double max_angular_velocity = std::numeric_limits<double>::infinity();
        double max_angular_acceleration = std::numeric_limits<double>::infinity();
    };

    SecondOrderReferenceRegulator()
    : SecondOrderReferenceRegulator(Config{}) {}

    explicit SecondOrderReferenceRegulator(const Config& config)
    : config_(config) {
        validateConfig();
        resetInternalState();
    }

    void setConfig(const Config& config) {
        config_ = config;
        validateConfig();
    }

    const Config& getConfig() const {
        return config_;
    }

    Mode getMode() const {
        return mode_;
    }

    bool isInitialized() const {
        return mode_ != Mode::kUninitialized;
    }

    static const char* modeToString(const Mode mode) {
        switch (mode) {
            case Mode::kPosition:
                return "position";
            case Mode::kOrientation:
                return "orientation";
            default:
                return "uninitialized";
        }
    }

    void resetPosition(const Vec3& position, const Vec3& velocity = Vec3::Zero()) {
        mode_ = Mode::kPosition;
        position_ = position;
        linear_velocity_ = velocity;
        linear_acceleration_.setZero();
    }

    void resetOrientation(const Quat& orientation, const Vec3& angular_velocity = Vec3::Zero()) {
        mode_ = Mode::kOrientation;
        orientation_ = normalizeQuaternion(orientation);
        angular_velocity_ = angular_velocity;
        angular_acceleration_.setZero();
    }

    const Vec3& updatePosition(const Vec3& target_position) {
        if (mode_ == Mode::kUninitialized) {
            throw std::logic_error(
                "SecondOrderReferenceRegulator::updatePosition() called before resetPosition(). "
                "Please initialize with the current position first.");
        }
        if (mode_ != Mode::kPosition) {
            throw std::logic_error(
                "SecondOrderReferenceRegulator is in orientation mode. "
                "Use a dedicated position regulator instance or call resetPosition().");
        }

        linear_acceleration_ =
            config_.natural_frequency * config_.natural_frequency * (target_position - position_) -
            2.0 * config_.damping_ratio * config_.natural_frequency * linear_velocity_;

        linear_acceleration_ = clampNorm(linear_acceleration_, config_.max_linear_acceleration);
        linear_velocity_ += linear_acceleration_ * config_.dt;
        linear_velocity_ = clampNorm(linear_velocity_, config_.max_linear_velocity);
        position_ += linear_velocity_ * config_.dt;
        return position_;
    }

    const Quat& updateOrientation(const Quat& target_orientation) {
        if (mode_ == Mode::kUninitialized) {
            throw std::logic_error(
                "SecondOrderReferenceRegulator::updateOrientation() called before resetOrientation(). "
                "Please initialize with the current orientation first.");
        }
        if (mode_ != Mode::kOrientation) {
            throw std::logic_error(
                "SecondOrderReferenceRegulator is in position mode. "
                "Use a dedicated orientation regulator instance or call resetOrientation().");
        }

        Quat target = normalizeQuaternion(target_orientation);
        if (orientation_.coeffs().dot(target.coeffs()) < 0.0) {
            target.coeffs() *= -1.0;
        }

        // 姿态误差在世界坐标系表达，对应 NMPC 常用的 world-aligned 角速度参考更直观。
        const Quat q_error = normalizeQuaternion(target * orientation_.conjugate());
        const Vec3 rotation_error = quaternionLog(q_error);

        angular_acceleration_ =
            config_.natural_frequency * config_.natural_frequency * rotation_error -
            2.0 * config_.damping_ratio * config_.natural_frequency * angular_velocity_;

        angular_acceleration_ = clampNorm(angular_acceleration_, config_.max_angular_acceleration);
        angular_velocity_ += angular_acceleration_ * config_.dt;
        angular_velocity_ = clampNorm(angular_velocity_, config_.max_angular_velocity);

        orientation_ = normalizeQuaternion(quaternionExp(angular_velocity_ * config_.dt) * orientation_);
        return orientation_;
    }

    const Quat& updateOrientation(const Mat3& target_rotation) {
        return updateOrientation(Quat(target_rotation));
    }

    const Vec3& getPosition() const {
        return position_;
    }

    const Vec3& getLinearVelocity() const {
        return linear_velocity_;
    }

    const Vec3& getLinearAcceleration() const {
        return linear_acceleration_;
    }

    const Quat& getOrientation() const {
        return orientation_;
    }

    Mat3 getOrientationMatrix() const {
        return orientation_.toRotationMatrix();
    }

    const Vec3& getAngularVelocity() const {
        return angular_velocity_;
    }

    const Vec3& getAngularAcceleration() const {
        return angular_acceleration_;
    }

    std::string debugString() const {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(6);
        oss << "SecondOrderReferenceRegulator{mode=" << modeToString(mode_)
            << ", dt=" << config_.dt
            << ", wn=" << config_.natural_frequency
            << ", zeta=" << config_.damping_ratio;

        if (mode_ == Mode::kPosition) {
            oss << ", pos=[" << position_.transpose() << "]"
                << ", vel=[" << linear_velocity_.transpose() << "]"
                << ", acc=[" << linear_acceleration_.transpose() << "]";
        } else if (mode_ == Mode::kOrientation) {
            oss << ", quat=["
                << orientation_.w() << ", "
                << orientation_.x() << ", "
                << orientation_.y() << ", "
                << orientation_.z() << "]"
                << ", ang_vel=[" << angular_velocity_.transpose() << "]"
                << ", ang_acc=[" << angular_acceleration_.transpose() << "]";
        }

        oss << "}";
        return oss.str();
    }

    void printDebug(std::ostream& os = std::cout) const {
        os << debugString() << std::endl;
    }

private:
    static Vec3 clampNorm(const Vec3& value, const double max_norm) {
        if (!std::isfinite(max_norm)) {
            return value;
        }
        if (max_norm <= 0.0) {
            return Vec3::Zero();
        }

        const double norm = value.norm();
        if (norm <= max_norm || norm < 1.0e-12) {
            return value;
        }
        return value * (max_norm / norm);
    }

    static Quat normalizeQuaternion(const Quat& q) {
        Quat normalized = q;
        normalized.normalize();
        if (normalized.w() < 0.0) {
            normalized.coeffs() *= -1.0;
        }
        return normalized;
    }

    static Vec3 quaternionLog(const Quat& q_in) {
        Quat q = normalizeQuaternion(q_in);
        const Vec3 imag(q.x(), q.y(), q.z());
        const double imag_norm = imag.norm();

        if (imag_norm < 1.0e-12) {
            return 2.0 * imag;
        }

        const double angle = 2.0 * std::atan2(imag_norm, q.w());
        return imag * (angle / imag_norm);
    }

    static Quat quaternionExp(const Vec3& rotation_vector) {
        const double theta = rotation_vector.norm();
        if (theta < 1.0e-12) {
            Quat q(1.0,
                   0.5 * rotation_vector.x(),
                   0.5 * rotation_vector.y(),
                   0.5 * rotation_vector.z());
            return normalizeQuaternion(q);
        }

        const Vec3 axis = rotation_vector / theta;
        const double half_theta = 0.5 * theta;
        const double sin_half_theta = std::sin(half_theta);

        Quat q(std::cos(half_theta),
               axis.x() * sin_half_theta,
               axis.y() * sin_half_theta,
               axis.z() * sin_half_theta);
        return normalizeQuaternion(q);
    }

    void validateConfig() const {
        if (config_.dt <= 0.0) {
            throw std::invalid_argument("SecondOrderReferenceRegulator: dt must be > 0.");
        }
        if (config_.natural_frequency <= 0.0) {
            throw std::invalid_argument("SecondOrderReferenceRegulator: natural_frequency must be > 0.");
        }
        if (config_.damping_ratio < 0.0) {
            throw std::invalid_argument("SecondOrderReferenceRegulator: damping_ratio must be >= 0.");
        }
    }

    void resetInternalState() {
        mode_ = Mode::kUninitialized;
        position_.setZero();
        linear_velocity_.setZero();
        linear_acceleration_.setZero();
        orientation_.setIdentity();
        angular_velocity_.setZero();
        angular_acceleration_.setZero();
    }

    Config config_;
    Mode mode_ = Mode::kUninitialized;

    Vec3 position_ = Vec3::Zero();
    Vec3 linear_velocity_ = Vec3::Zero();
    Vec3 linear_acceleration_ = Vec3::Zero();

    Quat orientation_ = Quat::Identity();
    Vec3 angular_velocity_ = Vec3::Zero();
    Vec3 angular_acceleration_ = Vec3::Zero();
};

#endif  // PANDA_NMPC_SECOND_ORDER_REFERENCE_REGULATOR_HPP_
