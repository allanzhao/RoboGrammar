#include <iostream>
#include <robot_design/control.h>

namespace robot_design {

void PDController::update() {
  Index robot_idx = sim_.findRobotIndex(robot_);
  sim_.getJointPositions(robot_idx, pos_);
  sim_.getJointVelocities(robot_idx, vel_);
  torque_ = kp_.cwiseProduct(pos_target_ - pos_) +
            kd_.cwiseProduct(vel_target_ - vel_);
  sim_.addJointTorques(robot_idx, torque_);
}

}  // namespace robot_design
