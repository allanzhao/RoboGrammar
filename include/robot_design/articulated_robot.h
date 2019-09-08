#pragma once

#include <robot_design/grammar.h>

namespace robot_design {

struct ArticulatedRobotGrammar : Grammar {
  ArticulatedRobotGrammar();

  Symbol articulated_robot_;
  Symbol spine_;
  Symbol leg_;
  Symbol link_;
  Symbol joint_;
  Symbol push_tf_;
  Symbol pop_tf_;
};

}  // namespace robot_design
