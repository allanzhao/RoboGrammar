#pragma once

#include <memory>
#include <string>

namespace robot_design {

struct Rule {
};

std::shared_ptr<Rule> loadRule(const std::string &filename);

}  // namespace robot_design
