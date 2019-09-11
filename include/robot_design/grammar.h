#pragma once

#include <memory>
#include <random>
#include <robot_design/types.h>
#include <string>
#include <vector>

namespace robot_design {

using Symbol = unsigned int;
using Rule = unsigned int;

struct AttributeDef {
  AttributeDef(std::string name, int dim)
      : name_(std::move(name)), dim_(dim) {}

  std::string name_;
  int dim_;
};

struct SymbolDef {
  SymbolDef(std::string name, std::vector<AttributeDef> attr_defs)
      : name_(std::move(name)),
        attr_defs_(std::move(attr_defs)) {}

  std::string name_;
  std::vector<AttributeDef> attr_defs_;
};

struct RuleDef {
  RuleDef(Symbol lhs, std::vector<Symbol> rhs)
      : lhs_(lhs), rhs_(std::move(rhs)) {}

  Symbol lhs_;
  std::vector<Symbol> rhs_;
};

struct Grammar {
  Grammar() {}
  Grammar(std::vector<SymbolDef> symbol_defs, std::vector<RuleDef> rule_defs)
      : symbol_defs_(std::move(symbol_defs)),
        rule_defs_(std::move(rule_defs)) {}
  Symbol addSymbol(std::string &&name, std::vector<AttributeDef> &&attr_defs) {
    symbol_defs_.emplace_back(
        std::forward<std::string>(name),
        std::forward<std::vector<AttributeDef>>(attr_defs));
    return symbol_defs_.size() - 1;
  }
  Rule addRule(Symbol lhs, std::vector<Symbol> &&rhs) {
    rule_defs_.emplace_back(lhs, std::forward<std::vector<Symbol>>(rhs));
    return rule_defs_.size() - 1;
  }
  bool isTerminalSymbol(Symbol symbol) const;

  std::vector<SymbolDef> symbol_defs_;
  std::vector<RuleDef> rule_defs_;
};

struct Design {
  Design(std::vector<Rule> derivation, VectorX attr_vals)
      : derivation_(std::move(derivation)), attr_vals_(std::move(attr_vals)) {}

  std::vector<Rule> derivation_;
  VectorX attr_vals_;
};

class DesignSampler {
public:
  DesignSampler(std::shared_ptr<const Grammar> grammar, unsigned int seed)
      : grammar_(grammar), generator_(seed) {}
  Design sampleDesign(Symbol start_symbol);

  std::shared_ptr<const Grammar> grammar_;
  std::mt19937 generator_;
};

}  // namespace robot_design
