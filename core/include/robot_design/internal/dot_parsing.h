#pragma once

#include <iostream>  // TODO
#include <map>
#include <robot_design/graph.h>
#include <robot_design/internal/dot_rules.h>
#include <string>
#include <tao/pegtl.hpp>
#include <vector>

namespace robot_design {
namespace dot_parsing {

struct State {
  std::vector<NodeAttributes> node_attr_stack_;
  std::vector<EdgeAttributes> edge_attr_stack_;
  std::map<std::string, NodeIndex> node_indices_;
  std::string a_list_key;
  std::string a_list_value;
};

template <typename Rule>
struct dot_action : tao::pegtl::nothing<Rule> {};

template <>
struct dot_action<dot_rules::begin_attr_list> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    std::cout << "begin_attr_list" << std::endl;
  }
};

template <>
struct dot_action<dot_rules::a_list_key> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    std::cout << "key: " << input.string() << std::endl;
  }
};

template <>
struct dot_action<dot_rules::a_list_value> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    std::cout << "value: " << input.string() << std::endl;
  }
};

template <>
struct dot_action<dot_rules::a_list_item> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    std::cout << "a_list_item" << std::endl;
  }
};

template <>
struct dot_action<dot_rules::attr_list> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    std::cout << "attr_list" << std::endl;
  }
};

template <>
struct dot_action<dot_rules::begin_edge_stmt> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    std::cout << "begin_edge_stmt" << std::endl;
  }
};

template <>
struct dot_action<dot_rules::edge_stmt> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    std::cout << "edge_stmt" << std::endl;
  }
};

template <>
struct dot_action<dot_rules::begin_node_stmt> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    std::cout << "begin_node_stmt" << std::endl;
  }
};

template <>
struct dot_action<dot_rules::node_stmt> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    std::cout << "node_stmt" << std::endl;
  }
};

template <>
struct dot_action<dot_rules::begin_subgraph> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    std::cout << "begin_subgraph" << std::endl;
  }
};

template <>
struct dot_action<dot_rules::subgraph> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    std::cout << "subgraph" << std::endl;
  }
};

template <>
struct dot_action<dot_rules::graph> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    std::cout << "graph" << std::endl;
  }
};

}  // namespace dot_parsing
}  // namespace robot_design
