#pragma once

#include <memory>
#include <random>
#include <robot_design/grammar.h>
#include <robot_design/types.h>
#include <vector>

namespace robot_design {

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
