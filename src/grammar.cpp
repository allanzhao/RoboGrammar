#include <deque>
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

Design DesignSampler::sampleDesign(Symbol start_symbol) {
  std::deque<Symbol> symbols_to_expand = {start_symbol};
  std::vector<Rule> applicable_rules;
  std::vector<Rule> derivation;
  VectorX attr_vals;

  while (!symbols_to_expand.empty()) {
    // Expand the leftmost symbol first
    Symbol symbol = symbols_to_expand.front();
    symbols_to_expand.pop_front();
    // Find applicable rules
    applicable_rules.clear();
    for (Rule rule = 0; rule < grammar_->rule_defs_.size(); ++rule) {
      if (grammar_->rule_defs_[rule].lhs_ == symbol) {
        applicable_rules.push_back(rule);
      }
    }
    if (!applicable_rules.empty()) {
      // Select one of the applicable rules at random
      std::uniform_int_distribution<Rule> distribution(
          0, applicable_rules.size() - 1);
      Rule rule = applicable_rules[distribution(generator_)];
      derivation.push_back(rule);
      const RuleDef &rule_def = grammar_->rule_defs_[rule];
      // Insert RHS of rule at front of symbol stack
      symbols_to_expand.insert(symbols_to_expand.begin(),
                               rule_def.rhs_.begin(), rule_def.rhs_.end());
    }
    // If no rules can be applied (symbol is terminal), move to the next symbol
  }

  return Design(std::move(derivation), std::move(attr_vals));
}

}  // namespace robot_design
