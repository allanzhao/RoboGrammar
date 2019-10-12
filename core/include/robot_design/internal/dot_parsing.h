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

// Additional state is required, because subgraphs provide default attributes
// for nodes and edges defined within them
struct SubgraphState {
  Subgraph result_;
  NodeAttributes node_attributes_;
  EdgeAttributes edge_attributes_;
};

struct NodeState {
  Node result_;
};

// Edge statements may define multiple edges
struct EdgeState {
  std::vector<Edge> result_;
};

struct State {
  Graph result_;
  std::vector<SubgraphState> subgraph_states_;
  std::vector<NodeState> node_states_;
  std::vector<EdgeState> edge_states_;
};

template <typename Rule>
struct dot_action : tao::pegtl::nothing<Rule> {};

template <>
struct dot_action<dot_rules::begin_subgraph> {
  static void apply0(State &state) {
    // Copy current attribute values to the new subgraph
    SubgraphState &parent_subgraph_state = state.subgraph_states_.back();
    state.subgraph_states_.push_back({
        /*result=*/{},
        /*node_attributes=*/parent_subgraph_state.node_attributes_,
        /*edge_attributes=*/parent_subgraph_state.edge_attributes_});
  }
};

template <>
struct dot_action<dot_rules::subgraph_id> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    SubgraphState &subgraph_state = state.subgraph_states_.back();
    subgraph_state.result_.name_ = input.string();
  }
};

template <>
struct dot_action<dot_rules::subgraph> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    SubgraphState &subgraph_state = state.subgraph_states_.back();
    state.result_.subgraphs_.push_back(std::move(subgraph_state.result_));
    state.subgraph_states_.pop_back();
  }
};

template <>
struct dot_action<dot_rules::begin_node_stmt> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    // Copy current node attribute values into the new node
    SubgraphState &subgraph_state = state.subgraph_states_.back();
    state.node_states_.push_back({/*result=*/{
        /*name=*/"", /*attrs=*/subgraph_state.node_attributes_}});
  }
};

template <>
struct dot_action<dot_rules::node_stmt> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    NodeState &node_state = state.node_states_.back();
    state.result_.nodes_.push_back(std::move(node_state.result_));
    state.node_states_.pop_back();

    // Add this node to every subgraph on the stack
    NodeIndex node_index = state.result_.nodes_.size() - 1;
    for (auto &subgraph_state : state.subgraph_states_) {
      subgraph_state.result_.nodes_.push_back(
    SubgraphState &subgraph_state = state.subgraph_states_.back();
    // TODO
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
struct dot_action<dot_rules::graph> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    std::cout << "graph" << std::endl;
  }
};

}  // namespace dot_parsing
}  // namespace robot_design
