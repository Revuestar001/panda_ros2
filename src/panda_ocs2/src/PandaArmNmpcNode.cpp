#include "PandaArmNmpcInterface.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <ocs2_core/thread_support/ExecuteAndSleep.h>
#include <ocs2_ddp/GaussNewtonDDP_MPC.h>
#include <ocs2_mpc/MPC_MRT_Interface.h>

#include <panda_interfaces/msg/result_nmpc.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include "rclcpp/rclcpp.hpp"

namespace ocs2 {
namespace panda_arm_nmpc {

namespace {

constexpr size_t kArmDim = 7;
constexpr size_t kStateDim = 14;

template <typename Derived>
std::string formatVectorHead(const Eigen::MatrixBase<Derived>& value, size_t count = 3) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3) << "[";
  const size_t limit = std::min<size_t>(count, value.size());
  for (size_t i = 0; i < limit; ++i) {
    if (i > 0) {
      stream << ", ";
    }
    stream << value(i);
  }
  if (value.size() > static_cast<Eigen::Index>(limit)) {
    stream << ", ...";
  }
  stream << "]";
  return stream.str();
}

std::string formatJointNames(const std::vector<std::string>& jointNames) {
  std::ostringstream stream;
  for (size_t i = 0; i < jointNames.size(); ++i) {
    if (i > 0) {
      stream << ", ";
    }
    stream << jointNames[i];
  }
  return stream.str();
}

double quaternionDistanceDeg(const Eigen::Quaterniond& lhs, const Eigen::Quaterniond& rhs) {
  Eigen::Quaterniond qLhs(lhs.normalized());
  Eigen::Quaterniond qRhs(rhs.normalized());
  const double dot = std::clamp(std::abs(qLhs.dot(qRhs)), 0.0, 1.0);
  return 2.0 * std::acos(dot) * 180.0 / M_PI;
}

double maxAbs(const vector_t& value) {
  return value.size() > 0 ? value.cwiseAbs().maxCoeff() : 0.0;
}

}  // namespace

class PandaArmNmpcNode final : public rclcpp::Node {
 public:
  PandaArmNmpcNode() : Node("panda_arm_nmpc_node") {
    const std::string defaultUrdf = "/home/cyh/panda_ros2/src/panda_ros/model/panda_tau_sim.urdf";

    const auto urdfFile = this->declare_parameter<std::string>("urdfFile", defaultUrdf);
    const auto eeFrame = this->declare_parameter<std::string>("eeFrame", "ee_center_body");
    const auto fallbackEeFrame = this->declare_parameter<std::string>("fallbackEeFrame", "panda_hand_tcp");
    jointStateTopic_ = this->declare_parameter<std::string>("jointStateTopic", "/joint_states");
    resultTopic_ = this->declare_parameter<std::string>("resultTopic", "/nmpc_result");

    auto targetPosition = this->declare_parameter<std::vector<double>>("targetPosition", {0.4, 0.0, 0.5});
    auto targetOrientation = this->declare_parameter<std::vector<double>>("targetOrientation", {1.0, 0.0, 0.0, 0.0});
    if (targetPosition.size() != 3 || targetOrientation.size() != 4) {
      throw std::runtime_error("[PandaArmNmpcNode] 'targetPosition' must have 3 entries and 'targetOrientation' must have 4 entries.");
    }

    PandaArmNmpcConfig config;
    config.timeHorizon = this->declare_parameter<double>("timeHorizon", config.timeHorizon);
    config.solutionTimeWindow = this->declare_parameter<double>("solutionTimeWindow", config.solutionTimeWindow);
    config.mpcDesiredFrequency = this->declare_parameter<double>("mpcFrequency", config.mpcDesiredFrequency);
    config.controlDesiredFrequency = this->declare_parameter<double>("controlFrequency", config.controlDesiredFrequency);
    config.rolloutTimeStep = this->declare_parameter<double>("rolloutTimeStep", config.rolloutTimeStep);
    config.eePositionWeight = this->declare_parameter<double>("eePositionWeight", config.eePositionWeight);
    config.eeOrientationWeight = this->declare_parameter<double>("eeOrientationWeight", config.eeOrientationWeight);
    config.jointVelocityWeight = this->declare_parameter<double>("jointVelocityWeight", config.jointVelocityWeight);
    config.jointAccelerationWeight = this->declare_parameter<double>("jointAccelerationWeight", config.jointAccelerationWeight);
    config.terminalEePositionWeight =
        this->declare_parameter<double>("terminalEePositionWeight", config.terminalEePositionWeight);
    config.terminalEeOrientationWeight =
        this->declare_parameter<double>("terminalEeOrientationWeight", config.terminalEeOrientationWeight);
    config.terminalJointVelocityWeight =
        this->declare_parameter<double>("terminalJointVelocityWeight", config.terminalJointVelocityWeight);
    config.nThreads = this->declare_parameter<int>("solverThreads", static_cast<int>(config.nThreads));
    config.maxNumIterations =
        this->declare_parameter<int>("maxNumIterations", static_cast<int>(config.maxNumIterations));
    config.useFeedbackPolicy = this->declare_parameter<bool>("useFeedbackPolicy", config.useFeedbackPolicy);
    config.solverDisplayInfo = this->declare_parameter<bool>("solverDisplayInfo", config.solverDisplayInfo);
    config.solverShortSummary = this->declare_parameter<bool>("solverShortSummary", config.solverShortSummary);
    config.solverDebugRollout = this->declare_parameter<bool>("solverDebugRollout", config.solverDebugRollout);
    config.mpcDebugPrint = this->declare_parameter<bool>("mpcDebugPrint", config.mpcDebugPrint);

    runtimeDebug_ = this->declare_parameter<bool>("runtimeDebug", true);
    controlDebugStride_ = std::max<int>(1, this->declare_parameter<int>("controlDebugStride", 20));
    mpcDebugStride_ = std::max<int>(1, this->declare_parameter<int>("mpcDebugStride", 25));

    interface_ = std::make_unique<PandaArmNmpcInterface>(config, urdfFile, eeFrame, fallbackEeFrame);
    mpc_ = std::make_unique<GaussNewtonDDP_MPC>(interface_->mpcSettings(), interface_->ddpSettings(), interface_->getRollout(),
                                                interface_->getOptimalControlProblem(), interface_->getInitializer());
    mpcMrtInterface_ = std::make_unique<MPC_MRT_Interface>(*mpc_);
    mpcMrtInterface_->initRollout(&interface_->getRollout());

    targetPose_.resize(7);
    targetPose_.head<3>() = Eigen::Map<const Eigen::Vector3d>(targetPosition.data());
    Eigen::Quaterniond targetQuaternion(targetOrientation[3], targetOrientation[0], targetOrientation[1], targetOrientation[2]);
    targetQuaternion.normalize();
    targetPose_.tail<4>() = targetQuaternion.coeffs();

    lastInput_ = vector_t::Zero(kArmDim);
    controlPeriod_ = 1.0 / interface_->mpcSettings().mrtDesiredFrequency_;

    resultPublisher_ = this->create_publisher<panda_interfaces::msg::ResultNMPC>(resultTopic_, 1);
    jointStateSubscriber_ = this->create_subscription<sensor_msgs::msg::JointState>(
        jointStateTopic_, 10, std::bind(&PandaArmNmpcNode::jointStateCallback, this, std::placeholders::_1));

    controlTimer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(controlPeriod_)),
        std::bind(&PandaArmNmpcNode::controlTimerCallback, this));

    running_ = true;
    mpcThread_ = std::thread([this]() { mpcWorker(); });

    RCLCPP_INFO(this->get_logger(), "Panda arm NMPC node ready. joint_states='%s', nmpc_result='%s', ee_frame='%s'",
                jointStateTopic_.c_str(), resultTopic_.c_str(), interface_->getEeFrame().c_str());
    RCLCPP_INFO(this->get_logger(), "Target pose: position=%s orientation(xyzw)=%s",
                formatVectorHead(targetPose_.head<3>(), 3).c_str(), formatVectorHead(targetPose_.tail<4>(), 4).c_str());
    RCLCPP_INFO(this->get_logger(), "Joint order: %s", formatJointNames(interface_->getJointNames()).c_str());
    RCLCPP_INFO(this->get_logger(),
                "MPC config: horizon=%.3f window=%.3f mpc_hz=%.1f control_hz=%.1f pos_w=%.2f rot_w=%.2f input_w=%.5f "
                "terminal_pos_w=%.2f terminal_rot_w=%.2f",
                config.timeHorizon, config.solutionTimeWindow, config.mpcDesiredFrequency, config.controlDesiredFrequency,
                config.eePositionWeight, config.eeOrientationWeight, config.jointAccelerationWeight,
                config.terminalEePositionWeight, config.terminalEeOrientationWeight);
  }

  ~PandaArmNmpcNode() override {
    running_ = false;
    if (mpcThread_.joinable()) {
      mpcThread_.join();
    }
  }

 private:
  void jointStateCallback(const sensor_msgs::msg::JointState::ConstSharedPtr& msg) {
    vector_t q(kArmDim);
    vector_t v = vector_t::Zero(kArmDim);

    for (size_t i = 0; i < kArmDim; ++i) {
      const auto& jointName = interface_->getJointNames()[i];
      const auto it = std::find(msg->name.begin(), msg->name.end(), jointName);
      if (it == msg->name.end()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Joint '%s' not found in '%s'.",
                             jointName.c_str(), jointStateTopic_.c_str());
        return;
      }

      const auto index = static_cast<size_t>(std::distance(msg->name.begin(), it));
      if (index >= msg->position.size()) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Joint '%s' has no position entry.", jointName.c_str());
        return;
      }

      q(i) = msg->position[index];
      if (index < msg->velocity.size()) {
        v(i) = msg->velocity[index];
      } else if (!missingVelocityWarned_) {
        RCLCPP_WARN(this->get_logger(),
                    "Incoming JointState has no velocity field for all arm joints. The example will fall back to zero velocity.");
        missingVelocityWarned_ = true;
      }
    }

    SystemObservation observation;
    observation.mode = 0;
    if (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0) {
      observation.time = this->get_clock()->now().seconds();
    } else {
      observation.time = rclcpp::Time(msg->header.stamp).seconds();
    }
    observation.state.resize(kStateDim);
    observation.state.head(kArmDim) = q;
    observation.state.tail(kArmDim) = v;

    {
      std::lock_guard<std::mutex> lock(observationMutex_);
      observation.input = lastInput_;
      latestObservation_ = observation;
      hasObservation_ = true;
    }

    if (runtimeDebug_ && !firstObservationLogged_) {
      RCLCPP_INFO(this->get_logger(), "First joint observation received. q=%s v=%s",
                  formatVectorHead(q, 7).c_str(), formatVectorHead(v, 7).c_str());
      firstObservationLogged_ = true;
    }
  }

  void controlTimerCallback() {
    ++controlCycleCount_;
    SystemObservation observation;
    {
      std::lock_guard<std::mutex> lock(observationMutex_);
      if (!hasObservation_) {
        return;
      }
      observation = latestObservation_;
      observation.input = lastInput_;
    }

    if (!targetInitialized_) {
      mpcMrtInterface_->getReferenceManager().setTargetTrajectories(makeTargetTrajectories(observation.time));
      targetInitialized_ = true;
      RCLCPP_INFO(this->get_logger(), "Initialized target trajectory at t=%.3f", observation.time);
    }

    mpcMrtInterface_->setCurrentObservation(observation);
    if (!mpcMrtInterface_->initialPolicyReceived()) {
      if (runtimeDebug_ && controlCycleCount_ % static_cast<size_t>(controlDebugStride_) == 0) {
        RCLCPP_INFO(this->get_logger(), "Waiting for initial NMPC policy... t=%.3f", observation.time);
      }
      return;
    }

    const bool policyUpdated = mpcMrtInterface_->updatePolicy();
    if (policyUpdated && !initialPolicyLogged_) {
      RCLCPP_INFO(this->get_logger(), "Received first NMPC policy.");
      initialPolicyLogged_ = true;
    }

    vector_t optimizedState;
    vector_t optimizedInput;
    size_t plannedMode = 0;

    try {
      mpcMrtInterface_->evaluatePolicy(observation.time, observation.state, optimizedState, optimizedInput, plannedMode);
    } catch (const std::exception& error) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Failed to evaluate policy: %s", error.what());
      return;
    }

    logControlDebug(observation, optimizedState, optimizedInput);
    publishResults(optimizedState, optimizedInput);

    std::lock_guard<std::mutex> lock(observationMutex_);
    lastInput_ = optimizedInput;
  }

  void mpcWorker() {
    while (running_ && rclcpp::ok()) {
      if (!hasObservation_ || !targetInitialized_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        continue;
      }

      try {
        executeAndSleep([this]() { mpcMrtInterface_->advanceMpc(); }, interface_->mpcSettings().mpcDesiredFrequency_);
        ++mpcCycleCount_;
        if (runtimeDebug_ && mpcCycleCount_ % static_cast<size_t>(mpcDebugStride_) == 0) {
          double observationTime = 0.0;
          double lastInputMax = 0.0;
          {
            std::lock_guard<std::mutex> lock(observationMutex_);
            if (hasObservation_) {
              observationTime = latestObservation_.time;
            }
            lastInputMax = maxAbs(lastInput_);
          }
          RCLCPP_INFO(this->get_logger(), "MPC advance #%zu: obs_t=%.3f last_input_max=%.3f", mpcCycleCount_, observationTime,
                      lastInputMax);
        }
      } catch (const std::exception& error) {
        RCLCPP_ERROR(this->get_logger(), "MPC thread failed: %s", error.what());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
  }

  TargetTrajectories makeTargetTrajectories(scalar_t time) const {
    return TargetTrajectories({time}, {targetPose_}, {vector_t::Zero(kArmDim)});
  }

  void logControlDebug(const SystemObservation& observation, const vector_t& optimizedState, const vector_t& optimizedInput) {
    if (!runtimeDebug_ || controlCycleCount_ % static_cast<size_t>(controlDebugStride_) != 0) {
      return;
    }

    const auto currentQ = observation.state.head(kArmDim);
    const auto currentV = observation.state.tail(kArmDim);
    const auto qRef = optimizedState.head(kArmDim);
    const auto vRef = optimizedState.tail(kArmDim);
    const double qErrNorm = (qRef - currentQ).norm();
    const double vErrNorm = (vRef - currentV).norm();
    const Eigen::Vector3d currentEePosition = interface_->getEePosition(observation.state);
    const Eigen::Vector3d targetEePosition = targetPose_.head<3>();
    const double eePositionError = (currentEePosition - targetEePosition).norm();
    const Eigen::Quaterniond currentEeOrientation = interface_->getEeOrientation(observation.state);
    const Eigen::Quaterniond targetEeOrientation(targetPose_(6), targetPose_(3), targetPose_(4), targetPose_(5));
    const double eeOrientationErrorDeg = quaternionDistanceDeg(currentEeOrientation, targetEeOrientation);

    RCLCPP_INFO(this->get_logger(),
                "NMPC ctrl #%zu t=%.3f q_err=%.4f v_err=%.4f ee_pos_err=%.4f "
                "ee_rot_err_deg=%.2f a_max=%.3f q=%s q_ref=%s",
                controlCycleCount_, observation.time, qErrNorm, vErrNorm, eePositionError, eeOrientationErrorDeg,
                maxAbs(optimizedInput), formatVectorHead(currentQ, 4).c_str(), formatVectorHead(qRef, 4).c_str());
  }

  void publishResults(const vector_t& optimizedState, const vector_t& optimizedInput) {
    if (optimizedState.size() != kStateDim || optimizedInput.size() != kArmDim) {
      return;
    }

    panda_interfaces::msg::ResultNMPC msg;
    for (size_t i = 0; i < kArmDim; ++i) {
      msg.q_ref[i] = optimizedState(i);
      msg.v_ref[i] = optimizedState(kArmDim + i);
      msg.a_ref[i] = optimizedInput(i);
      msg.jerk_cmd[i] = 0.0;
    }
    msg.status = 0;
    resultPublisher_->publish(std::move(msg));
  }

  std::unique_ptr<PandaArmNmpcInterface> interface_;
  std::unique_ptr<GaussNewtonDDP_MPC> mpc_;
  std::unique_ptr<MPC_MRT_Interface> mpcMrtInterface_;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr jointStateSubscriber_;
  rclcpp::Publisher<panda_interfaces::msg::ResultNMPC>::SharedPtr resultPublisher_;
  rclcpp::TimerBase::SharedPtr controlTimer_;

  std::atomic_bool running_{false};
  std::thread mpcThread_;

  mutable std::mutex observationMutex_;
  SystemObservation latestObservation_;
  vector_t lastInput_;
  std::atomic_bool hasObservation_{false};
  std::atomic_bool targetInitialized_{false};
  bool initialPolicyLogged_ = false;
  bool missingVelocityWarned_ = false;
  bool firstObservationLogged_ = false;

  vector_t targetPose_;
  scalar_t controlPeriod_ = 0.01;
  bool runtimeDebug_ = true;
  int controlDebugStride_ = 20;
  int mpcDebugStride_ = 25;
  size_t controlCycleCount_ = 0;
  size_t mpcCycleCount_ = 0;

  std::string jointStateTopic_;
  std::string resultTopic_;
};
}  // namespace panda_arm_nmpc
}  // namespace ocs2

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ocs2::panda_arm_nmpc::PandaArmNmpcNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
