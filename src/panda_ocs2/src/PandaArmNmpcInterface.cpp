#include "PandaArmNmpcInterface.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include <ocs2_core/PreComputation.h>
#include <ocs2_core/cost/StateCost.h>
#include <ocs2_core/cost/StateInputCost.h>
#include <ocs2_core/dynamics/LinearSystemDynamics.h>
#include <ocs2_core/initialization/DefaultInitializer.h>
#include <ocs2_core/misc/LinearInterpolation.h>
#include <ocs2_mobile_manipulator/FactoryFunctions.h>
#include <ocs2_mobile_manipulator/ManipulatorModelInfo.h>
#include <ocs2_oc/rollout/TimeTriggeredRollout.h>
#include <ocs2_pinocchio_interface/PinocchioEndEffectorKinematics.h>
#include <ocs2_pinocchio_interface/PinocchioStateInputMapping.h>

namespace ocs2 {
namespace panda_arm_nmpc {

namespace {

using vector3_t = Eigen::Matrix<scalar_t, 3, 1>;
using quaternion_t = Eigen::Quaternion<scalar_t>;

constexpr size_t kArmDim = 7;
constexpr size_t kStateDim = 14;
constexpr size_t kInputDim = 7;

bool hasFrame(const pinocchio::Model& model, const std::string& frameName) {
  return std::any_of(model.frames.begin(), model.frames.end(),
                     [&](const auto& frame) { return frame.name == frameName; });
}

matrix_t scaledIdentity(size_t dim, scalar_t weight) {
  return weight * matrix_t::Identity(dim, dim);
}

std::pair<vector_t, quaternion_t> interpolateTargetPose(scalar_t time, const TargetTrajectories& targetTrajectories) {
  if (targetTrajectories.stateTrajectory.empty()) {
    throw std::runtime_error("[PandaArmNmpc] Target trajectories are empty.");
  }

  const auto& timeTrajectory = targetTrajectories.timeTrajectory;
  const auto& stateTrajectory = targetTrajectories.stateTrajectory;

  vector_t position;
  quaternion_t orientation;

  if (stateTrajectory.size() > 1) {
    int index;
    scalar_t alpha;
    std::tie(index, alpha) = LinearInterpolation::timeSegment(time, timeTrajectory);

    const auto& lhs = stateTrajectory[index];
    const auto& rhs = stateTrajectory[index + 1];
    const quaternion_t qLhs(lhs.tail<4>());
    const quaternion_t qRhs(rhs.tail<4>());

    position = alpha * lhs.head<3>() + (1.0 - alpha) * rhs.head<3>();
    orientation = qLhs.slerp((1.0 - alpha), qRhs);
  } else {
    position = stateTrajectory.front().head<3>();
    orientation = quaternion_t(stateTrajectory.front().tail<4>());
  }

  orientation.normalize();
  return {position, orientation};
}

void addResidualCost(const VectorFunctionLinearApproximation& residualApproximation, const matrix_t& weights,
                     ScalarFunctionQuadraticApproximation& costApproximation) {
  costApproximation.f += 0.5 * residualApproximation.f.dot(weights * residualApproximation.f);
  costApproximation.dfdx.noalias() += residualApproximation.dfdx.transpose() * weights * residualApproximation.f;
  costApproximation.dfdxx.noalias() += residualApproximation.dfdx.transpose() * weights * residualApproximation.dfdx;

  if (residualApproximation.dfdu.size() > 0) {
    costApproximation.dfdu.noalias() += residualApproximation.dfdu.transpose() * weights * residualApproximation.f;
    costApproximation.dfduu.noalias() += residualApproximation.dfdu.transpose() * weights * residualApproximation.dfdu;
    costApproximation.dfdux.noalias() += residualApproximation.dfdu.transpose() * weights * residualApproximation.dfdx;
  }
}

class PandaArmPinocchioMapping final : public PinocchioStateInputMapping<scalar_t> {
 public:
  PandaArmPinocchioMapping() = default;
  ~PandaArmPinocchioMapping() override = default;

  PandaArmPinocchioMapping* clone() const override { return new PandaArmPinocchioMapping(*this); }

  vector_t getPinocchioJointPosition(const vector_t& state) const override { return state.head(kArmDim); }

  vector_t getPinocchioJointVelocity(const vector_t& state, const vector_t& input) const override {
    return state.tail(kArmDim);
  }

  std::pair<matrix_t, matrix_t> getOcs2Jacobian(const vector_t& state, const matrix_t& Jq, const matrix_t& Jv) const override {
    matrix_t dfdx = matrix_t::Zero(Jq.rows(), kStateDim);
    dfdx.leftCols(kArmDim) = Jq;
    dfdx.rightCols(kArmDim) = Jv;
    matrix_t dfdu = matrix_t::Zero(Jq.rows(), kInputDim);
    return {dfdx, dfdu};
  }
};

class PandaArmPreComputation final : public PreComputation {
 public:
  explicit PandaArmPreComputation(PinocchioInterface pinocchioInterface) : pinocchioInterface_(std::move(pinocchioInterface)) {}

  PandaArmPreComputation* clone() const override { return new PandaArmPreComputation(pinocchioInterface_); }

  void request(RequestSet request, scalar_t t, const vector_t& x, const vector_t& u) override {
    if (!request.containsAny(Request::Cost + Request::Constraint + Request::SoftConstraint)) {
      return;
    }
    updatePinocchio(request, x);
  }

  void requestFinal(RequestSet request, scalar_t t, const vector_t& x) override {
    if (!request.containsAny(Request::Cost + Request::Constraint + Request::SoftConstraint)) {
      return;
    }
    updatePinocchio(request, x);
  }

  const PinocchioInterface& getPinocchioInterface() const { return pinocchioInterface_; }

 private:
  void updatePinocchio(RequestSet request, const vector_t& state) {
    const auto q = state.head(kArmDim);
    const auto v = state.tail(kArmDim);

    const auto& model = pinocchioInterface_.getModel();
    auto& data = pinocchioInterface_.getData();

    pinocchio::forwardKinematics(model, data, q, v);
    pinocchio::updateFramePlacements(model, data);

    if (request.contains(Request::Approximation)) {
      pinocchio::computeJointJacobians(model, data);
    }
  }

  PinocchioInterface pinocchioInterface_;
};

class PandaArmStageCost final : public StateInputCost {
 public:
  PandaArmStageCost(const EndEffectorKinematics<scalar_t>& eeKinematics, matrix_t positionWeights, matrix_t orientationWeights,
                    matrix_t jointVelocityWeights, matrix_t jointAccelerationWeights)
      : eeKinematicsPtr_(eeKinematics.clone()),
        positionWeights_(std::move(positionWeights)),
        orientationWeights_(std::move(orientationWeights)),
        jointVelocityWeights_(std::move(jointVelocityWeights)),
        jointAccelerationWeights_(std::move(jointAccelerationWeights)) {
    pinocchioEEKinematicsPtr_ = dynamic_cast<PinocchioEndEffectorKinematics*>(eeKinematicsPtr_.get());
  }

  PandaArmStageCost* clone() const override {
    return new PandaArmStageCost(*eeKinematicsPtr_, positionWeights_, orientationWeights_, jointVelocityWeights_,
                                 jointAccelerationWeights_);
  }

  scalar_t getValue(scalar_t time, const vector_t& state, const vector_t& input, const TargetTrajectories& targetTrajectories,
                    const PreComputation& preComputation) const override {
    setPinocchioInterface(preComputation);
    const auto [targetPosition, targetOrientation] = interpolateTargetPose(time, targetTrajectories);

    const vector3_t positionError = eeKinematicsPtr_->getPosition(state).front() - targetPosition;
    const vector3_t orientationError = eeKinematicsPtr_->getOrientationError(state, {targetOrientation}).front();
    const auto jointVelocity = state.tail(kArmDim);

    return 0.5 * positionError.dot(positionWeights_ * positionError) +
           0.5 * orientationError.dot(orientationWeights_ * orientationError) +
           0.5 * jointVelocity.dot(jointVelocityWeights_ * jointVelocity) +
           0.5 * input.dot(jointAccelerationWeights_ * input);
  }

  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t time, const vector_t& state, const vector_t& input,
                                                                 const TargetTrajectories& targetTrajectories,
                                                                 const PreComputation& preComputation) const override {
    setPinocchioInterface(preComputation);
    const auto [targetPosition, targetOrientation] = interpolateTargetPose(time, targetTrajectories);

    auto costApproximation = ScalarFunctionQuadraticApproximation::Zero(kStateDim, kInputDim);

    auto positionApproximation = eeKinematicsPtr_->getPositionLinearApproximation(state).front();
    positionApproximation.f.noalias() -= targetPosition;
    addResidualCost(positionApproximation, positionWeights_, costApproximation);

    auto orientationApproximation =
        eeKinematicsPtr_->getOrientationErrorLinearApproximation(state, {targetOrientation}).front();
    addResidualCost(orientationApproximation, orientationWeights_, costApproximation);

    const auto jointVelocity = state.tail(kArmDim);
    costApproximation.f += 0.5 * jointVelocity.dot(jointVelocityWeights_ * jointVelocity);
    costApproximation.dfdx.tail(kArmDim).noalias() += jointVelocityWeights_ * jointVelocity;
    costApproximation.dfdxx.bottomRightCorner(kArmDim, kArmDim).noalias() += jointVelocityWeights_;

    costApproximation.f += 0.5 * input.dot(jointAccelerationWeights_ * input);
    costApproximation.dfdu.noalias() += jointAccelerationWeights_ * input;
    costApproximation.dfduu.noalias() += jointAccelerationWeights_;

    return costApproximation;
  }

 private:
  void setPinocchioInterface(const PreComputation& preComputation) const {
    if (pinocchioEEKinematicsPtr_ == nullptr) {
      return;
    }
    const auto& pandaArmPreComputation = dynamic_cast<const PandaArmPreComputation&>(preComputation);
    pinocchioEEKinematicsPtr_->setPinocchioInterface(pandaArmPreComputation.getPinocchioInterface());
  }

  std::unique_ptr<EndEffectorKinematics<scalar_t>> eeKinematicsPtr_;
  mutable PinocchioEndEffectorKinematics* pinocchioEEKinematicsPtr_ = nullptr;
  matrix_t positionWeights_;
  matrix_t orientationWeights_;
  matrix_t jointVelocityWeights_;
  matrix_t jointAccelerationWeights_;
};

class PandaArmTerminalCost final : public StateCost {
 public:
  PandaArmTerminalCost(const EndEffectorKinematics<scalar_t>& eeKinematics, matrix_t positionWeights, matrix_t orientationWeights,
                       matrix_t jointVelocityWeights)
      : eeKinematicsPtr_(eeKinematics.clone()),
        positionWeights_(std::move(positionWeights)),
        orientationWeights_(std::move(orientationWeights)),
        jointVelocityWeights_(std::move(jointVelocityWeights)) {
    pinocchioEEKinematicsPtr_ = dynamic_cast<PinocchioEndEffectorKinematics*>(eeKinematicsPtr_.get());
  }

  PandaArmTerminalCost* clone() const override {
    return new PandaArmTerminalCost(*eeKinematicsPtr_, positionWeights_, orientationWeights_, jointVelocityWeights_);
  }

  scalar_t getValue(scalar_t time, const vector_t& state, const TargetTrajectories& targetTrajectories,
                    const PreComputation& preComputation) const override {
    setPinocchioInterface(preComputation);
    const auto [targetPosition, targetOrientation] = interpolateTargetPose(time, targetTrajectories);

    const vector3_t positionError = eeKinematicsPtr_->getPosition(state).front() - targetPosition;
    const vector3_t orientationError = eeKinematicsPtr_->getOrientationError(state, {targetOrientation}).front();
    const auto jointVelocity = state.tail(kArmDim);

    return 0.5 * positionError.dot(positionWeights_ * positionError) +
           0.5 * orientationError.dot(orientationWeights_ * orientationError) +
           0.5 * jointVelocity.dot(jointVelocityWeights_ * jointVelocity);
  }

  ScalarFunctionQuadraticApproximation getQuadraticApproximation(scalar_t time, const vector_t& state,
                                                                 const TargetTrajectories& targetTrajectories,
                                                                 const PreComputation& preComputation) const override {
    setPinocchioInterface(preComputation);
    const auto [targetPosition, targetOrientation] = interpolateTargetPose(time, targetTrajectories);

    auto costApproximation = ScalarFunctionQuadraticApproximation::Zero(kStateDim);

    auto positionApproximation = eeKinematicsPtr_->getPositionLinearApproximation(state).front();
    positionApproximation.f.noalias() -= targetPosition;
    addResidualCost(positionApproximation, positionWeights_, costApproximation);

    auto orientationApproximation =
        eeKinematicsPtr_->getOrientationErrorLinearApproximation(state, {targetOrientation}).front();
    addResidualCost(orientationApproximation, orientationWeights_, costApproximation);

    const auto jointVelocity = state.tail(kArmDim);
    costApproximation.f += 0.5 * jointVelocity.dot(jointVelocityWeights_ * jointVelocity);
    costApproximation.dfdx.tail(kArmDim).noalias() += jointVelocityWeights_ * jointVelocity;
    costApproximation.dfdxx.bottomRightCorner(kArmDim, kArmDim).noalias() += jointVelocityWeights_;

    return costApproximation;
  }

 private:
  void setPinocchioInterface(const PreComputation& preComputation) const {
    if (pinocchioEEKinematicsPtr_ == nullptr) {
      return;
    }
    const auto& pandaArmPreComputation = dynamic_cast<const PandaArmPreComputation&>(preComputation);
    pinocchioEEKinematicsPtr_->setPinocchioInterface(pandaArmPreComputation.getPinocchioInterface());
  }

  std::unique_ptr<EndEffectorKinematics<scalar_t>> eeKinematicsPtr_;
  mutable PinocchioEndEffectorKinematics* pinocchioEEKinematicsPtr_ = nullptr;
  matrix_t positionWeights_;
  matrix_t orientationWeights_;
  matrix_t jointVelocityWeights_;
};

}  // namespace

PandaArmNmpcInterface::PandaArmNmpcInterface(const PandaArmNmpcConfig& config, const std::string& urdfFile, std::string eeFrame,
                                             std::string fallbackEeFrame) {
  boost::filesystem::path urdfFilePath(urdfFile);
  if (!boost::filesystem::exists(urdfFilePath)) {
    throw std::invalid_argument("[PandaArmNmpcInterface] URDF file not found: " + urdfFilePath.string());
  }

  const std::vector<std::string> removeJointNames = {"panda_finger_joint1", "panda_finger_joint2"};
  pinocchioInterfacePtr_ = std::make_unique<PinocchioInterface>(mobile_manipulator::createPinocchioInterface(
      urdfFile, mobile_manipulator::ManipulatorModelType::DefaultManipulator, removeJointNames));

  const auto& model = pinocchioInterfacePtr_->getModel();
  if (model.nq != kArmDim || model.nv != kArmDim) {
    throw std::runtime_error("[PandaArmNmpcInterface] Expected a fixed-base 7-DoF Panda arm model after removing finger joints.");
  }

  if (!hasFrame(model, eeFrame)) {
    if (!fallbackEeFrame.empty() && hasFrame(model, fallbackEeFrame)) {
      eeFrame = std::move(fallbackEeFrame);
    } else {
      throw std::runtime_error("[PandaArmNmpcInterface] End-effector frame '" + eeFrame + "' not found in the URDF model.");
    }
  }
  eeFrame_ = std::move(eeFrame);

  jointNames_ = std::vector<std::string>(model.names.end() - kArmDim, model.names.end());
  initialState_ = vector_t::Zero(kStateDim);

  ddpSettings_.algorithm_ = ddp::Algorithm::SLQ;
  ddpSettings_.nThreads_ = config.nThreads;
  ddpSettings_.threadPriority_ = 50;
  ddpSettings_.maxNumIterations_ = config.maxNumIterations;
  ddpSettings_.minRelCost_ = 1e-1;
  ddpSettings_.constraintTolerance_ = 1e-3;
  ddpSettings_.displayInfo_ = config.solverDisplayInfo;
  ddpSettings_.displayShortSummary_ = config.solverShortSummary;
  ddpSettings_.checkNumericalStability_ = false;
  ddpSettings_.debugPrintRollout_ = config.solverDebugRollout;
  ddpSettings_.absTolODE_ = 1e-5;
  ddpSettings_.relTolODE_ = 1e-3;
  ddpSettings_.maxNumStepsPerSecond_ = 100000;
  ddpSettings_.timeStep_ = 1e-3;
  ddpSettings_.preComputeRiccatiTerms_ = true;
  ddpSettings_.useFeedbackPolicy_ = config.useFeedbackPolicy;
  ddpSettings_.constraintPenaltyInitialValue_ = 20.0;
  ddpSettings_.constraintPenaltyIncreaseRate_ = 2.0;

  mpcSettings_.timeHorizon_ = config.timeHorizon;
  mpcSettings_.solutionTimeWindow_ = config.solutionTimeWindow;
  mpcSettings_.debugPrint_ = config.mpcDebugPrint;
  mpcSettings_.coldStart_ = false;
  mpcSettings_.mpcDesiredFrequency_ = config.mpcDesiredFrequency;
  mpcSettings_.mrtDesiredFrequency_ = config.controlDesiredFrequency;

  referenceManagerPtr_ = std::make_shared<ReferenceManager>();

  matrix_t A = matrix_t::Zero(kStateDim, kStateDim);
  A.topRightCorner(kArmDim, kArmDim).setIdentity();
  matrix_t B = matrix_t::Zero(kStateDim, kInputDim);
  B.bottomRows(kArmDim).setIdentity();
  problem_.dynamicsPtr = std::make_unique<LinearSystemDynamics>(A, B);

  PandaArmPinocchioMapping pinocchioMapping;
  PinocchioEndEffectorKinematics eeKinematics(*pinocchioInterfacePtr_, pinocchioMapping, {eeFrame_});

  problem_.costPtr->add("eeTracking", std::make_unique<PandaArmStageCost>(
                                          eeKinematics, scaledIdentity(3, config.eePositionWeight),
                                          scaledIdentity(3, config.eeOrientationWeight), scaledIdentity(kArmDim, config.jointVelocityWeight),
                                          scaledIdentity(kArmDim, config.jointAccelerationWeight)));
  problem_.finalCostPtr->add("terminalEeTracking",
                             std::make_unique<PandaArmTerminalCost>(eeKinematics, scaledIdentity(3, config.terminalEePositionWeight),
                                                                    scaledIdentity(3, config.terminalEeOrientationWeight),
                                                                    scaledIdentity(kArmDim, config.terminalJointVelocityWeight)));

  problem_.preComputationPtr = std::make_unique<PandaArmPreComputation>(*pinocchioInterfacePtr_);

  rollout::Settings rolloutSettings;
  rolloutSettings.absTolODE = 1e-5;
  rolloutSettings.relTolODE = 1e-3;
  rolloutSettings.timeStep = config.rolloutTimeStep;
  rolloutSettings.integratorType = IntegratorType::ODE45;
  rolloutSettings.maxNumStepsPerSecond = 100000;
  rolloutSettings.checkNumericalStability = false;
  rolloutPtr_ = std::make_unique<TimeTriggeredRollout>(*problem_.dynamicsPtr, rolloutSettings);

  initializerPtr_ = std::make_unique<DefaultInitializer>(kInputDim);
}

Eigen::Vector3d PandaArmNmpcInterface::getEePosition(const vector_t& state) const {
  PinocchioInterface pinocchioInterface = *pinocchioInterfacePtr_;
  const auto& model = pinocchioInterface.getModel();
  auto& data = pinocchioInterface.getData();

  pinocchio::forwardKinematics(model, data, state.head(kArmDim), state.tail(kArmDim));
  pinocchio::updateFramePlacements(model, data);

  const auto frameId = model.getFrameId(eeFrame_);
  return data.oMf[frameId].translation();
}

Eigen::Quaterniond PandaArmNmpcInterface::getEeOrientation(const vector_t& state) const {
  PinocchioInterface pinocchioInterface = *pinocchioInterfacePtr_;
  const auto& model = pinocchioInterface.getModel();
  auto& data = pinocchioInterface.getData();

  pinocchio::forwardKinematics(model, data, state.head(kArmDim), state.tail(kArmDim));
  pinocchio::updateFramePlacements(model, data);

  const auto frameId = model.getFrameId(eeFrame_);
  return Eigen::Quaterniond(data.oMf[frameId].rotation());
}

}  // namespace panda_arm_nmpc
}  // namespace ocs2
