#include <deque>
#include <Eigen/StdVector>
#include <robot_design/articulated_robot.h>
#include <robot_design/design.h>
#include <robot_design/grammar.h>
#include <stdexcept>

namespace robot_design {

ArticulatedRobotGrammar::ArticulatedRobotGrammar() {
  // Define nonterminal symbols
  robot_ = addSymbol("Robot");
  segment_ = addSymbol("Segment");
  segment1_ = addSymbol("Segment1");
  segment2_ = addSymbol("Segment2");
  segment3_ = addSymbol("Segment3");
  leg_ = addSymbol("Leg");
  wheg_ = addSymbol("Wheg");
  wheel_ = addSymbol("Wheel");

  // Define terminal symbols
  link_ = addSymbol("Link");
  cylinder_ = addSymbol("Cylinder");
  hinge_ = addSymbol("Hinge");
  push_state_ = addSymbol("PushState");
  pop_state_ = addSymbol("PopState");
  left_ = addSymbol("Left");
  right_ = addSymbol("Right");
  up_ = addSymbol("Up");
  down_ = addSymbol("Down");

  // Define rules
  addRule(robot_, {segment_, robot_});
  addRule(robot_, {segment_});
  addRule(segment_, {segment1_, leg_, segment2_, leg_, segment3_});
  addRule(segment_, {segment1_, wheg_, segment2_, wheg_, segment3_});
  //addRule(segment_, {segment1_, wheel_, segment2_, wheel_, segment3_});
  addRule(segment1_, {link_, push_state_, left_, link_});
  addRule(segment2_, {pop_state_, push_state_, right_, link_});
  addRule(segment3_, {pop_state_, link_});
  addRule(leg_, {down_, hinge_, link_, hinge_, link_});
  addRule(wheg_, {down_, hinge_, link_});
  addRule(wheel_, {hinge_, cylinder_});
}

std::shared_ptr<Robot> buildArticulatedRobot(
    const Design &design, const ArticulatedRobotGrammar &grammar) {
  struct BuildState {
    Index parent_link_;
    JointType joint_type_;
    Scalar joint_pos_;
    Quaternion joint_rot_;
    Vector3 joint_axis_;
  };
  std::vector<BuildState, Eigen::aligned_allocator<BuildState>> state_stack;
  BuildState state = {/*parent_link=*/-1, /*joint_type=*/JointType::FREE,
                      /*joint_pos=*/0.0, /*joint_rot=*/Quaternion::Identity(),
                      /*joint_axis=*/Vector3::UnitX()};
  auto robot = std::make_shared<Robot>(
      /*link_density=*/1.0, /*link_radius=*/0.05, /*friction=*/0.9,
      /*motor_kp=*/2.0, /*motor_kd=*/0.1);
  std::deque<Symbol> symbols_to_expand = {grammar.getStartSymbol()};
  std::size_t rule_idx = 0;

  while (!symbols_to_expand.empty()) {
    Symbol symbol = symbols_to_expand.front();
    symbols_to_expand.pop_front();
    if (grammar.isTerminalSymbol(symbol)) {
      // Terminal symbols affect the robot's construction
      if (symbol == grammar.link_) {
        robot->links_.emplace_back(
            /*parent=*/state.parent_link_, /*joint_type=*/state.joint_type_,
            /*joint_pos=*/state.joint_pos_, /*joint_rot=*/state.joint_rot_,
            /*joint_axis=*/state.joint_axis_, /*shape=*/LinkShape::CAPSULE,
            /*length=*/0.2);
        state.parent_link_ = robot->links_.size() - 1;
        state.joint_type_ = JointType::FIXED;
        state.joint_pos_ = 1.0;
        state.joint_rot_ = Quaternion::Identity();
        state.joint_axis_ = Vector3::UnitX();
      } else if (symbol == grammar.hinge_) {
        state.joint_type_ = JointType::HINGE;
        state.joint_axis_ = Vector3::UnitY();
      } else if (symbol == grammar.push_state_) {
        state_stack.push_back(state);
      } else if (symbol == grammar.pop_state_) {
        if (state_stack.empty()) {
          throw std::runtime_error("State stack is empty");
        }
        state = state_stack.back();
        state_stack.pop_back();
      } else if (symbol == grammar.left_) {
        state.joint_rot_ = state.joint_rot_ * Eigen::AngleAxis<Scalar>(
            -M_PI / 2, Vector3::UnitY());
      } else if (symbol == grammar.right_) {
        state.joint_rot_ = state.joint_rot_ * Eigen::AngleAxis<Scalar>(
            M_PI / 2, Vector3::UnitY());
      } else if (symbol == grammar.up_) {
        state.joint_rot_ = state.joint_rot_ * Eigen::AngleAxis<Scalar>(
            -M_PI / 2, Vector3::UnitZ());
      } else if (symbol == grammar.down_) {
        state.joint_rot_ = state.joint_rot_ * Eigen::AngleAxis<Scalar>(
            M_PI / 2, Vector3::UnitZ());
      }
    } else {
      // Nonterminal symbols should be expanded
      if (rule_idx < design.derivation_.size()) {
        Rule rule = design.derivation_[rule_idx++];
        const RuleDef &rule_def = grammar.rule_defs_[rule];
        if (rule_def.lhs_ != symbol) {
          throw std::runtime_error(
              "Rule in derivation does not apply to symbol");
        }
        symbols_to_expand.insert(symbols_to_expand.begin(),
                                 rule_def.rhs_.begin(), rule_def.rhs_.end());
      } else {
        throw std::runtime_error("Not enough rules in derivation");
      }
    }
  }

  return robot;
}

}  // namespace robot_design
