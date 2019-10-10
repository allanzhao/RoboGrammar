#include <iostream>
#include <robot_design/grammar.h>
#include <robot_design/internal/dot_parsing.h>
#include <robot_design/internal/dot_rules.h>
#include <tao/pegtl.hpp>
#include <tao/pegtl/contrib/parse_tree.hpp>
#include <tao/pegtl/contrib/parse_tree_to_dot.hpp>

namespace robot_design {

std::shared_ptr<Rule> loadRule(const std::string &filename) {
  tao::pegtl::file_input<> input(filename);
  auto root = tao::pegtl::parse_tree::parse<
      tao::pegtl::pad<dot_rules::graph, dot_rules::sep>,
      dot_parsing::dot_selector>(input);
  tao::pegtl::parse_tree::print_dot(std::cout, *root);
  // TODO
  return std::make_shared<Rule>();
}

}  // namespace robot_design
