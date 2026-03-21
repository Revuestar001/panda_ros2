#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <ocs2_core/Types.h>
#include <ocs2_core/initialization/Initializer.h>
#include <ocs2_ddp/DDP_Settings.h>
#include <ocs2_mpc/MPC_Settings.h>
#include <ocs2_oc/rollout/TimeTriggeredRollout.h>
#include <ocs2_oc/synchronized_module/ReferenceManager.h>
#include <ocs2_pinocchio_interface/PinocchioInterface.h>
#include <ocs2_robotic_tools/common/RobotInterface.h>

namespace ocs2 {
namespace panda_arm_nmpc {

struct PandaArmNmpcConfig {
  scalar_t timeHorizon = 1.0;
  scalar_t solutionTimeWindow = 0.2;
  scalar_t mpcDesiredFrequency = 50.0;
  scalar_t controlDesiredFrequency = 100.0;
  scalar_t rolloutTimeStep = 1e-2;

  scalar_t eePositionWeight = 200.0;
  scalar_t eeOrientationWeight = 60.0;
  scalar_t jointVelocityWeight = 0.05;
  scalar_t jointAccelerationWeight = 0.1;

  scalar_t terminalEePositionWeight = 4000.0;
  scalar_t terminalEeOrientationWeight = 1200.0;
  scalar_t terminalJointVelocityWeight = 2.0;

  size_t nThreads = 3;
  size_t maxNumIterations = 1;
  bool useFeedbackPolicy = true;
  bool solverDisplayInfo = false;
  bool solverShortSummary = false;
  bool solverDebugRollout = false;
  bool mpcDebugPrint = false;
};

class PandaArmNmpcInterface final : public RobotInterface {
 public:
  PandaArmNmpcInterface(const PandaArmNmpcConfig& config, const std::string& urdfFile, std::string eeFrame,
                        std::string fallbackEeFrame = "panda_hand_tcp");

  const vector_t& getInitialState() const { return initialState_; }

  ddp::Settings& ddpSettings() { return ddpSettings_; }
  mpc::Settings& mpcSettings() { return mpcSettings_; }

  const OptimalControlProblem& getOptimalControlProblem() const override { return problem_; }
  std::shared_ptr<ReferenceManagerInterface> getReferenceManagerPtr() const override { return referenceManagerPtr_; }
  const Initializer& getInitializer() const override { return *initializerPtr_; }

  const RolloutBase& getRollout() const { return *rolloutPtr_; }
  const PinocchioInterface& getPinocchioInterface() const { return *pinocchioInterfacePtr_; }

  const std::vector<std::string>& getJointNames() const { return jointNames_; }
  const std::string& getEeFrame() const { return eeFrame_; }
  Eigen::Vector3d getEePosition(const vector_t& state) const;
  Eigen::Quaterniond getEeOrientation(const vector_t& state) const;

  size_t getStateDim() const { return initialState_.size(); }
  size_t getInputDim() const { return jointNames_.size(); }

 private:
  ddp::Settings ddpSettings_;
  mpc::Settings mpcSettings_;

  OptimalControlProblem problem_;
  std::shared_ptr<ReferenceManager> referenceManagerPtr_;

  std::unique_ptr<RolloutBase> rolloutPtr_;
  std::unique_ptr<Initializer> initializerPtr_;
  std::unique_ptr<PinocchioInterface> pinocchioInterfacePtr_;

  vector_t initialState_;
  std::vector<std::string> jointNames_;
  std::string eeFrame_;
};

}  // namespace panda_arm_nmpc
}  // namespace ocs2
