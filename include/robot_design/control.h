#pragma once

#include <robot_design/robot.h>
#include <robot_design/sim.h>
#include <robot_design/types.h>

namespace robot_design {

class PDController {
public:
  PDController(const Robot &robot, Simulation &sim);
  void update();

  VectorX kp_;
  VectorX kd_;
  VectorX pos_target_;
  VectorX vel_target_;

private:
  const Robot &robot_;
  Simulation &sim_;
  VectorX pos_;
  VectorX vel_;
  VectorX torque_;
};

}  // namespace robot_design
