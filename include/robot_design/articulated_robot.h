#pragma once

#include <memory>
#include <robot_design/design.h>
#include <robot_design/grammar.h>
#include <robot_design/robot.h>

namespace robot_design {

struct ArticulatedRobotGrammar : Grammar {
  ArticulatedRobotGrammar();

  Symbol robot_;
  Symbol segment_;
  Symbol segment1_;
  Symbol segment2_;
  Symbol segment3_;
  Symbol leg_;
  Symbol wheg_;
  Symbol wheel_;
  Symbol link_;
  Symbol cylinder_;
  Symbol hinge_;
  Symbol push_state_;
  Symbol pop_state_;
  Symbol left_;
  Symbol right_;
  Symbol up_;
  Symbol down_;
};

std::shared_ptr<Robot> buildArticulatedRobot(
    const Design &design, const ArticulatedRobotGrammar &grammar);

}  // namespace robot_design
