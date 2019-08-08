#pragma once

#include <robot_design/robot.h>
#include <robot_design/sim.h>
#include <robot_design/types.h>

namespace robot_design {

class PDController {
public:
  PDController(const Robot &robot, Simulation &sim) : robot_(robot), sim_(sim) {
    Index robot_idx = sim_.findRobotIndex(robot);
    int dof_count = sim_.getRobotDofCount(robot_idx);
    kp_ = VectorX::Zero(dof_count);
    kd_ = VectorX::Zero(dof_count);
    pos_target_ = VectorX::Zero(dof_count);
    vel_target_ = VectorX::Zero(dof_count);
  }
  void update();

  const Robot &robot_;
  Simulation &sim_;
  VectorX kp_;
  VectorX kd_;
  VectorX pos_target_;
  VectorX vel_target_;

private:
  VectorX pos_;
  VectorX vel_;
  VectorX torque_;
};

}  // namespace robot_design
