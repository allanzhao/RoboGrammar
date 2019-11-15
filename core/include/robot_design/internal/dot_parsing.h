#pragma once

#include <iterator>
#include <robot_design/graph.h>
#include <robot_design/internal/dot_rules.h>
#include <string>
#include <tao/pegtl.hpp>
#include <unordered_map>
#include <utility>
#include <vector>

namespace robot_design {
namespace dot_parsing {

struct SubgraphState {
  Subgraph result_;
};

struct NodeState {
  Node result_;
};

struct EdgeState {
  // Edge statements may define multiple edges
  std::vector<Edge> results_;
  // Indices of nodes in the most recent edge_arg
  std::vector<NodeIndex> edge_arg_nodes_;
};

struct State {
  Graph result_;
  // Contents of the most recent id
  std::string id_content_;
  // Index of the most recently referenced subgraph
  SubgraphIndex latest_subgraph_index_;
  // Index of the most recently referenced node
  NodeIndex latest_node_index_;
  // Most recent attribute list
  std::vector<std::pair<std::string, std::string>> attr_list_;
  // Most recent attribute key
  std::string attr_key_;
  // Most recent attribute value
  std::string attr_value_;
  std::vector<SubgraphState> subgraph_states_;
  std::vector<NodeState> node_states_;
  std::vector<EdgeState> edge_states_;
  std::unordered_map<std::string, SubgraphIndex> subgraph_indices_;
  std::unordered_map<std::string, NodeIndex> node_indices_;
};

template <typename Rule> struct dot_action : tao::pegtl::nothing<Rule> {};

template <> struct dot_action<dot_rules::idstring> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    state.id_content_ = input.string();
  }
};

template <> struct dot_action<dot_rules::numeral> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    state.id_content_ = input.string();
  }
};

template <> struct dot_action<dot_rules::dqstring::content> {
  template <typename Input>
  static void apply(const Input &input, State &state) {
    state.id_content_ = input.string();
  }
};

template <> struct dot_action<dot_rules::begin_subgraph> {
  static void apply0(State &state) {
    state.subgraph_states_.emplace_back();
    // Get the second to last element
    // Need to do this after emplace_back, since the vector can resize
    SubgraphState &parent_subgraph_state = state.subgraph_states_.end()[-2];
    SubgraphState &subgraph_state = state.subgraph_states_.back();
    // Copy the parent subgraph's attribute values into the new subgraph
    subgraph_state.result_.node_attrs_ =
        parent_subgraph_state.result_.node_attrs_;
    subgraph_state.result_.edge_attrs_ =
        parent_subgraph_state.result_.edge_attrs_;
  }
};

template <> struct dot_action<dot_rules::subgraph_id> {
  static void apply0(State &state) {
    SubgraphState &subgraph_state = state.subgraph_states_.back();
    std::string &subgraph_name = state.id_content_;
    auto elem = state.subgraph_indices_.find(subgraph_name);
    if (elem != state.subgraph_indices_.end()) {
      // Subgraph with this name already exists, copy it
      Subgraph &existing_subgraph = state.result_.subgraphs_[elem->second];
      subgraph_state.result_ = existing_subgraph;
    }
    subgraph_state.result_.name_ = std::move(subgraph_name);
  }
};

template <> struct dot_action<dot_rules::subgraph> {
  static void apply0(State &state) {
    SubgraphState &subgraph_state = state.subgraph_states_.back();
    auto elem = state.subgraph_indices_.find(subgraph_state.result_.name_);
    SubgraphIndex subgraph_index;
    if (elem != state.subgraph_indices_.end()) {
      // Subgraph with this name already exists, update it
      subgraph_index = elem->second;
      Subgraph &existing_subgraph = state.result_.subgraphs_[subgraph_index];
      existing_subgraph = std::move(subgraph_state.result_);
    } else {
      // Add a new subgraph
      subgraph_index = state.result_.subgraphs_.size();
      if (!subgraph_state.result_.name_.empty()) {
        // Store a mapping from this subgraph's name to its index
        state.subgraph_indices_.insert(
            {subgraph_state.result_.name_, subgraph_index});
      }
      state.result_.subgraphs_.push_back(std::move(subgraph_state.result_));
    }
    state.subgraph_states_.pop_back();
    state.latest_subgraph_index_ = subgraph_index;
  }
};

template <> struct dot_action<dot_rules::begin_node_stmt> {
  static void apply0(State &state) {
    SubgraphState &subgraph_state = state.subgraph_states_.back();
    // Copy current node attribute values into the new node state
    state.node_states_.push_back({/*result=*/{
        /*name=*/"", /*attrs=*/subgraph_state.result_.node_attrs_}});
  }
};

template <> struct dot_action<dot_rules::node_id> {
  static void apply0(State &state) {
    NodeState &node_state = state.node_states_.back();
    std::string &node_name = state.id_content_;
    auto elem = state.node_indices_.find(node_name);
    if (elem != state.node_indices_.end()) {
      // Node with this name already exists, copy it
      Node &existing_node = state.result_.nodes_[elem->second];
      node_state.result_ = existing_node;
    }
    node_state.result_.name_ = std::move(node_name);
  }
};

template <> struct dot_action<dot_rules::node_stmt> {
  static void apply0(State &state) {
    NodeState &node_state = state.node_states_.back();
    auto elem = state.node_indices_.find(node_state.result_.name_);
    NodeIndex node_index;
    if (elem != state.node_indices_.end()) {
      // Node with this name already exists, update it
      node_index = elem->second;
      Node &existing_node = state.result_.nodes_[node_index];
      existing_node = std::move(node_state.result_);
    } else {
      // Add a new node
      node_index = state.result_.nodes_.size();
      if (!node_state.result_.name_.empty()) {
        // Store a mapping from this node's name to its index
        state.node_indices_.insert({node_state.result_.name_, node_index});
      }
      state.result_.nodes_.push_back(std::move(node_state.result_));
    }
    state.node_states_.pop_back();
    state.latest_node_index_ = node_index;

    // Add this node to all of its containing subgraphs
    for (auto &subgraph_state : state.subgraph_states_) {
      subgraph_state.result_.nodes_.insert(node_index);
    }
  }
};

template <>
struct dot_action<dot_rules::edge_node_stmt>
    : dot_action<dot_rules::node_stmt> {};

template <> struct dot_action<dot_rules::edge_node_arg> {
  static void apply0(State &state) {
    EdgeState &edge_state = state.edge_states_.back();
    if (!edge_state.edge_arg_nodes_.empty()) {
      // This is not the first argument
      // Create an edge from every node in the previous argument to this node,
      // using the current edge attribute values
      SubgraphState &subgraph_state = state.subgraph_states_.back();
      for (NodeIndex tail_index : edge_state.edge_arg_nodes_) {
        edge_state.results_.push_back(
            {/*head=*/state.latest_node_index_,
             /*tail=*/tail_index,
             /*attrs=*/subgraph_state.result_.edge_attrs_});
      }
    }
    // Store this argument
    edge_state.edge_arg_nodes_.clear();
    edge_state.edge_arg_nodes_.push_back(state.latest_node_index_);
  }
};

template <> struct dot_action<dot_rules::edge_subgraph_arg> {
  static void apply0(State &state) {
    EdgeState &edge_state = state.edge_states_.back();
    Subgraph &arg_subgraph =
        state.result_.subgraphs_[state.latest_subgraph_index_];
    if (!edge_state.edge_arg_nodes_.empty()) {
      // This is not the first argument
      // Create an edge from every node in the previous argument to every node
      // in this argument, using the current edge attribute values
      SubgraphState &subgraph_state = state.subgraph_states_.back();
      for (NodeIndex tail_index : edge_state.edge_arg_nodes_) {
        for (NodeIndex head_index : arg_subgraph.nodes_) {
          edge_state.results_.push_back(
              {/*head=*/head_index,
               /*tail=*/tail_index,
               /*attrs=*/subgraph_state.result_.edge_attrs_});
        }
      }
    }
    // Store this argument
    edge_state.edge_arg_nodes_.clear();
    edge_state.edge_arg_nodes_.insert(edge_state.edge_arg_nodes_.end(),
                                      arg_subgraph.nodes_.begin(),
                                      arg_subgraph.nodes_.end());
  }
};

template <> struct dot_action<dot_rules::begin_edge_stmt> {
  static void apply0(State &state) { state.edge_states_.emplace_back(); }
};

template <> struct dot_action<dot_rules::edge_stmt> {
  static void apply0(State &state) {
    EdgeState &edge_state = state.edge_states_.back();
    EdgeIndex start_edge_index = state.result_.edges_.size();
    // Add the new edges
    state.result_.edges_.insert(
        state.result_.edges_.end(),
        std::make_move_iterator(edge_state.results_.begin()),
        std::make_move_iterator(edge_state.results_.end()));
    EdgeIndex end_edge_index = state.result_.edges_.size();
    state.edge_states_.pop_back();

    // Add these edges to all of their containing subgraphs
    for (auto &subgraph_state : state.subgraph_states_) {
      for (EdgeIndex edge_index = start_edge_index; edge_index < end_edge_index;
           ++edge_index) {
        subgraph_state.result_.edges_.insert(edge_index);
      }
    }
  }
};

template <> struct dot_action<dot_rules::begin_attr_list> {
  static void apply0(State &state) { state.attr_list_.clear(); }
};

template <> struct dot_action<dot_rules::a_list_key> {
  static void apply0(State &state) {
    state.attr_key_ = std::move(state.id_content_);
  }
};

template <> struct dot_action<dot_rules::a_list_value> {
  static void apply0(State &state) {
    state.attr_value_ = std::move(state.id_content_);
  }
};

template <> struct dot_action<dot_rules::a_list_item> {
  static void apply0(State &state) {
    state.attr_list_.emplace_back(std::move(state.attr_key_),
                                  std::move(state.attr_value_));
  }
};

template <> struct dot_action<dot_rules::node_attr_list> {
  static void apply0(State &state) {
    NodeState &node_state = state.node_states_.back();
    // Set attributes on the current node
    updateNodeAttributes(node_state.result_.attrs_, state.attr_list_);
  }
};

template <> struct dot_action<dot_rules::edge_attr_list> {
  static void apply0(State &state) {
    EdgeState &edge_state = state.edge_states_.back();
    // Set attributes on all of the current edges
    for (Edge &edge : edge_state.results_) {
      updateEdgeAttributes(edge.attrs_, state.attr_list_);
    }
  }
};

template <> struct dot_action<dot_rules::node_def_attr_list> {
  static void apply0(State &state) {
    SubgraphState &subgraph_state = state.subgraph_states_.back();
    // Set default node attributes on the current subgraph
    updateNodeAttributes(subgraph_state.result_.node_attrs_, state.attr_list_);
  }
};

template <> struct dot_action<dot_rules::edge_def_attr_list> {
  static void apply0(State &state) {
    SubgraphState &subgraph_state = state.subgraph_states_.back();
    // Set default edge attributes on the current subgraph
    updateEdgeAttributes(subgraph_state.result_.edge_attrs_, state.attr_list_);
  }
};

template <> struct dot_action<dot_rules::graph_id> {
  static void apply0(State &state) {
    state.result_.name_ = std::move(state.id_content_);
  }
};

} // namespace dot_parsing
} // namespace robot_design
