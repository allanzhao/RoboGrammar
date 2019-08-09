#include <iostream>
#include <robot_design/control.h>

namespace robot_design {

PDController::PDController(const Robot &robot, Simulation &sim)
    : robot_(robot), sim_(sim) {
  Index robot_idx = sim_.findRobotIndex(robot);
  int dof_count = sim_.getRobotDofCount(robot_idx);
  kp_ = VectorX::Zero(dof_count);
  kd_ = VectorX::Zero(dof_count);
  pos_target_ = VectorX::Zero(dof_count);
  vel_target_ = VectorX::Zero(dof_count);
}

void PDController::update() {
  Index robot_idx = sim_.findRobotIndex(robot_);
  sim_.getJointPositions(robot_idx, pos_);
  sim_.getJointVelocities(robot_idx, vel_);
  torque_ = kp_.cwiseProduct(pos_target_ - pos_) +
            kd_.cwiseProduct(vel_target_ - vel_);
  sim_.addJointTorques(robot_idx, torque_);
}

}  // namespace robot_design
