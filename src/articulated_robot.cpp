#include <robot_design/articulated_robot.h>

namespace robot_design {

ArticulatedRobotGrammar::ArticulatedRobotGrammar() {
  // Define symbols
  articulated_robot_ = addSymbol("ArticulatedRobot", {});
  spine_ = addSymbol("Spine", {});
  leg_ = addSymbol("Leg", {});
  link_ = addSymbol("Link", {{"displacement", 3}});
  joint_ = addSymbol("Joint", {{"location", 1}, {"axis", 3}, {"gear", 1}});
  push_tf_ = addSymbol("PushTF", {});
  pop_tf_ = addSymbol("PopTF", {});

  // Define rules
  addRule(articulated_robot_, {spine_});
  addRule(spine_, {leg_, link_, spine_});
  addRule(spine_, {leg_});
  addRule(leg_, {push_tf_, joint_, link_, joint_, link_, pop_tf_});
}

}  // namespace robot_design
