#include <robot_design/grammar.h>

namespace robot_design {

bool Grammar::isTerminalSymbol(Symbol symbol) const {
  // Terminal symbols do not appear on the LHS of any rule
  for (const auto &rule_def : rule_defs_) {
    if (rule_def.lhs_ == symbol) {
      return false;
    }
  }
  return true;
}

}  // namespace robot_design
