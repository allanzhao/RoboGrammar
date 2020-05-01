#include <algorithm>
#include <iterator>
#include <robot_design/graph.h>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace robot_design {

Rule createRuleFromGraph(const Graph &graph) {
  Rule rule;
  rule.name_ = graph.name_;

  // Graph must have subgraphs named "L" and "R"
  const auto lhs_subgraph = std::find_if(
      graph.subgraphs_.begin(), graph.subgraphs_.end(),
      [](const Subgraph &subgraph) { return subgraph.name_ == "L"; });
  const auto rhs_subgraph = std::find_if(
      graph.subgraphs_.begin(), graph.subgraphs_.end(),
      [](const Subgraph &subgraph) { return subgraph.name_ == "R"; });
  if (lhs_subgraph == graph.subgraphs_.end() ||
      rhs_subgraph == graph.subgraphs_.end()) {
    throw std::runtime_error(
        "Graph must contain subgraphs named \"L\" and \"R\"");
  }

  // Mappings from graph node indices to LHS, RHS, and common node indices
  std::vector<NodeIndex> graph_to_lhs_node(graph.nodes_.size(), -1);
  std::vector<NodeIndex> graph_to_rhs_node(graph.nodes_.size(), -1);
  std::vector<NodeIndex> graph_to_common_node(graph.nodes_.size(), -1);

  // Copy nodes into the appropriate graphs in rule, and update the mappings
  for (NodeIndex i = 0; i < graph.nodes_.size(); ++i) {
    const Node &node = graph.nodes_[i];
    bool node_in_lhs = lhs_subgraph->nodes_.count(i) != 0;
    bool node_in_rhs = rhs_subgraph->nodes_.count(i) != 0;
    if (node_in_lhs) {
      rule.lhs_.nodes_.push_back(node);
      graph_to_lhs_node[i] = rule.lhs_.nodes_.size() - 1;
    }
    if (node_in_rhs) {
      rule.rhs_.nodes_.push_back(node);
      graph_to_rhs_node[i] = rule.rhs_.nodes_.size() - 1;
    }
    if (node_in_lhs && node_in_rhs) {
      rule.common_.nodes_.push_back(node);
      rule.common_to_lhs_.node_mapping_.push_back(rule.lhs_.nodes_.size() - 1);
      rule.common_to_rhs_.node_mapping_.push_back(rule.rhs_.nodes_.size() - 1);
      graph_to_common_node[i] = rule.common_.nodes_.size() - 1;
    }
    if (!node_in_lhs && !node_in_rhs) {
      throw std::runtime_error("Node \"" + node.name_ +
                               "\" is in neither the LHS nor the RHS");
    }
  }

  // Mappings from IDs to edges on the LHS and RHS
  std::unordered_map<std::string, EdgeIndex> lhs_id_to_edge;
  std::unordered_map<std::string, EdgeIndex> rhs_id_to_edge;

  // Copy edges into the appropriate graphs in rule, and update the mappings
  for (EdgeIndex m = 0; m < graph.edges_.size(); ++m) {
    const Edge &edge = graph.edges_[m];
    bool edge_in_lhs = lhs_subgraph->edges_.count(m) != 0;
    bool edge_in_rhs = rhs_subgraph->edges_.count(m) != 0;
    const std::string &id = edge.attrs_.id_;
    if (edge_in_lhs) {
      rule.lhs_.edges_.push_back(edge);
      Edge &lhs_edge = rule.lhs_.edges_.back();
      lhs_edge.head_ = graph_to_lhs_node[lhs_edge.head_];
      lhs_edge.tail_ = graph_to_lhs_node[lhs_edge.tail_];
      if (!id.empty()) {
        auto result = lhs_id_to_edge.emplace(id, rule.lhs_.edges_.size() - 1);
        if (!result.second) {
          throw std::runtime_error("Edge ID \"" + id +
                                   "\" is used more than once in the LHS");
        }
      }
    }
    if (edge_in_rhs) {
      rule.rhs_.edges_.push_back(edge);
      Edge &rhs_edge = rule.rhs_.edges_.back();
      rhs_edge.head_ = graph_to_rhs_node[rhs_edge.head_];
      rhs_edge.tail_ = graph_to_rhs_node[rhs_edge.tail_];
      if (!id.empty()) {
        auto result = rhs_id_to_edge.emplace(id, rule.rhs_.edges_.size() - 1);
        if (!result.second) {
          throw std::runtime_error("Edge ID \"" + id +
                                   "\" is used more than once in the RHS");
        }
      }
    }
    if (edge_in_lhs && edge_in_rhs) {
      // Possible using nested subgraphs, but discouraged
      throw std::runtime_error(
          "Edge is in both the \"L\" and \"R\" subgraphs, use separate edges "
          "with the same ID instead");
    }
    if (!edge_in_lhs && !edge_in_rhs) {
      throw std::runtime_error("Edge is in neither the LHS nor the RHS");
    }
  }

  // Add a common edge for every ID which appears on both the LHS and RHS
  for (const auto &elem_lhs : lhs_id_to_edge) {
    const std::string &id = elem_lhs.first;
    EdgeIndex m_lhs = elem_lhs.second;
    auto it_rhs = rhs_id_to_edge.find(id);
    if (it_rhs != rhs_id_to_edge.end()) {
      EdgeIndex m_rhs = it_rhs->second;
      // Edges in common are not connected to any nodes, use bogus node indices
      rule.common_.edges_.push_back({/*head=*/0, /*tail=*/0, /*attrs=*/{}});
      rule.common_.edges_.back().attrs_.id_ = id;
      rule.common_to_lhs_.edge_mapping_.push_back({m_lhs});
      rule.common_to_rhs_.edge_mapping_.push_back({m_rhs});
    }
  }

  return rule;
}

std::vector<GraphMapping> findMatches(const Graph &pattern,
                                      const Graph &target) {
  assert(pattern.nodes_.size() >= 1);

  // Stack for backtracking, initialized with the first partial match to try
  // The last entry of each partial match is speculative
  std::vector<GraphMapping> partial_matches = {GraphMapping{{0}}};
  std::vector<GraphMapping> matches;

  while (!partial_matches.empty()) {
    GraphMapping &pm = partial_matches.back();
    NodeIndex i = pm.node_mapping_.size() - 1;
    NodeIndex &j = pm.node_mapping_.back();

    // Try to map node i in pattern to node j in target

    if (j >= target.nodes_.size()) {
      // No more possible matches with this prefix, backtrack
      partial_matches.pop_back();
      if (!partial_matches.empty()) {
        GraphMapping &parent_pm = partial_matches.back();
        ++parent_pm.node_mapping_.back();
      }
      continue;
    }

    // No two pattern nodes can map to the same target node (injective mapping)
    if (std::find(pm.node_mapping_.begin(), std::prev(pm.node_mapping_.end()),
                  j) != std::prev(pm.node_mapping_.end())) {
      ++j;
      continue;
    }

    // If pattern node i has a required label, target node j must have a label
    // with the same value
    const std::string &pattern_node_label =
        pattern.nodes_[i].attrs_.require_label_;
    const std::string &target_node_label = target.nodes_[j].attrs_.label_;
    if (!pattern_node_label.empty() &&
        pattern_node_label != target_node_label) {
      ++j;
      continue;
    }

    // Edges in pattern incident on i must also be present in target
    bool edge_fail = false;
    for (const Edge &pattern_edge : pattern.edges_) {
      const std::string &pattern_label = pattern_edge.attrs_.require_label_;
      if (pattern_edge.head_ == i && pattern_edge.tail_ <= i) {
        // Pattern edge i_tail -> i requires target edge j_tail -> j
        NodeIndex j_tail = pm.node_mapping_[pattern_edge.tail_];
        auto it = std::find_if(
            target.edges_.begin(), target.edges_.end(),
            [&, j, j_tail](const Edge &target_edge) {
              const std::string &target_label = target_edge.attrs_.label_;
              return target_edge.head_ == j && target_edge.tail_ == j_tail &&
                     (pattern_label.empty() || pattern_label == target_label);
            });
        if (it == target.edges_.end()) {
          // No such target edge exists
          edge_fail = true;
          break;
        }
      } else if (pattern_edge.tail_ == i && pattern_edge.head_ <= i) {
        // Pattern edge i -> i_head requires target edge j -> j_head
        NodeIndex j_head = pm.node_mapping_[pattern_edge.head_];
        auto it = std::find_if(
            target.edges_.begin(), target.edges_.end(),
            [&, j, j_head](const Edge &target_edge) {
              const std::string &target_label = target_edge.attrs_.label_;
              return target_edge.tail_ == j && target_edge.head_ == j_head &&
                     (pattern_label.empty() || pattern_label == target_label);
            });
        if (it == target.edges_.end()) {
          // No such target edge exists
          edge_fail = true;
          break;
        }
      }
    }
    if (edge_fail) {
      ++j;
      continue;
    }

    // Partial match is consistent with pattern

    if (pm.node_mapping_.size() == pattern.nodes_.size()) {
      // Node matching is complete, fill in edge matches
      matches.push_back(pm);
      GraphMapping &new_match = matches.back();
      new_match.edge_mapping_.resize(pattern.edges_.size());
      for (EdgeIndex m = 0; m < pattern.edges_.size(); ++m) {
        const Edge &pattern_edge = pattern.edges_[m];
        const std::string &pattern_label = pattern_edge.attrs_.require_label_;
        NodeIndex j_head = new_match.node_mapping_[pattern_edge.head_];
        NodeIndex j_tail = new_match.node_mapping_[pattern_edge.tail_];
        for (EdgeIndex n = 0; n < target.edges_.size(); ++n) {
          const Edge &target_edge = target.edges_[n];
          const std::string &target_label = target_edge.attrs_.label_;
          if (target_edge.head_ == j_head && target_edge.tail_ == j_tail &&
              (pattern_label.empty() || pattern_label == target_label)) {
            new_match.edge_mapping_[m].push_back(n);
          }
        }
      }
      ++j;
    } else {
      // Recurse
      partial_matches.push_back(pm);
      GraphMapping &child_pm = partial_matches.back();
      child_pm.node_mapping_.push_back(0);
    }
  }

  return matches;
}

bool checkRuleApplicability(const Rule &rule, const Graph &target,
                            const GraphMapping &lhs_to_target) {
  // Target nodes in the LHS but not in common with the RHS will be removed
  std::unordered_set<NodeIndex> target_nodes_to_remove(
      lhs_to_target.node_mapping_.begin(), lhs_to_target.node_mapping_.end());
  for (NodeIndex i = 0; i < rule.common_.nodes_.size(); ++i) {
    NodeIndex lhs_node = rule.common_to_lhs_.node_mapping_[i];
    target_nodes_to_remove.erase(lhs_to_target.node_mapping_[lhs_node]);
  }

  std::unordered_set<EdgeIndex> target_edges_in_lhs;
  for (const auto &target_edges : lhs_to_target.edge_mapping_) {
    target_edges_in_lhs.insert(target_edges.begin(), target_edges.end());
  }

  // Check that no target edges that are not in the LHS are incident on a
  // node to be removed (dangling condition)
  for (EdgeIndex m = 0; m < target.edges_.size(); ++m) {
    if (target_edges_in_lhs.count(m) == 0) {
      // Target edge is not in the LHS
      if (target_nodes_to_remove.count(target.edges_[m].head_) > 0 ||
          target_nodes_to_remove.count(target.edges_[m].tail_) > 0) {
        // Head or tail node would be removed, making this edge a dangling edge
        return false;
      }
    }
  }

  // All checks passed
  return true;
}

Graph applyRule(const Rule &rule, const Graph &target,
                const GraphMapping &lhs_to_target) {
  Graph result;

  // Mappings from target and RHS node indices to result node indices
  std::vector<NodeIndex> target_to_result_node(target.nodes_.size(), -1);
  std::vector<NodeIndex> rhs_to_result_node(rule.rhs_.nodes_.size(), -1);

  // Copy target nodes not in LHS to result
  std::unordered_set<NodeIndex> target_nodes_in_lhs(
      lhs_to_target.node_mapping_.begin(), lhs_to_target.node_mapping_.end());
  for (NodeIndex i = 0; i < target.nodes_.size(); ++i) {
    if (target_nodes_in_lhs.count(i) == 0) {
      result.nodes_.push_back(target.nodes_[i]);
      target_to_result_node[i] = result.nodes_.size() - 1;
    }
  }

  // Copy target nodes in LHS to result if they are in common with the RHS
  for (NodeIndex i = 0; i < rule.common_.nodes_.size(); ++i) {
    NodeIndex lhs_node = rule.common_to_lhs_.node_mapping_[i];
    NodeIndex rhs_node = rule.common_to_rhs_.node_mapping_[i];
    NodeIndex target_node = lhs_to_target.node_mapping_[lhs_node];
    result.nodes_.push_back(target.nodes_[target_node]);
    target_to_result_node[target_node] = result.nodes_.size() - 1;
    rhs_to_result_node[rhs_node] = result.nodes_.size() - 1;
    // Update the result node's attributes
    Node &result_node = result.nodes_.back();
    const Node &common_node = rule.common_.nodes_[i];
    copyNondefaultAttributes(result_node.attrs_, common_node.attrs_);
    // If the rule requires a label to match, replace it with a new label
    // Otherwise, keep the original target node's label
    if (!common_node.attrs_.require_label_.empty()) {
      result_node.attrs_.label_ = common_node.attrs_.label_;
    }
    // Nodes in the result graph should never have require_label set
    result_node.attrs_.require_label_.clear();
  }

  // Add RHS nodes which are not in common with the LHS
  std::unordered_set<NodeIndex> rhs_nodes_in_common(
      rule.common_to_rhs_.node_mapping_.begin(),
      rule.common_to_rhs_.node_mapping_.end());
  for (NodeIndex i = 0; i < rule.rhs_.nodes_.size(); ++i) {
    if (rhs_nodes_in_common.count(i) == 0) {
      result.nodes_.push_back(rule.rhs_.nodes_[i]);
      rhs_to_result_node[i] = result.nodes_.size() - 1;
    }
  }

  // Copy target edges not in LHS to result
  std::unordered_set<EdgeIndex> target_edges_in_lhs;
  for (const auto &target_edges : lhs_to_target.edge_mapping_) {
    target_edges_in_lhs.insert(target_edges.begin(), target_edges.end());
  }
  for (EdgeIndex m = 0; m < target.edges_.size(); ++m) {
    if (target_edges_in_lhs.count(m) == 0) {
      result.edges_.push_back(target.edges_[m]);
      Edge &edge = result.edges_.back();
      edge.head_ = target_to_result_node[edge.head_];
      edge.tail_ = target_to_result_node[edge.tail_];
      if (edge.head_ == NodeIndex(-1) || edge.tail_ == NodeIndex(-1)) {
        // The head or tail node was removed without also removing this edge
        // (violates dangling condition)
        throw std::invalid_argument(
            "Resulting graph would have a dangling edge");
      }
    }
  }

  // Copy target edges in LHS to result if they are in common with the RHS
  for (EdgeIndex m = 0; m < rule.common_.edges_.size(); ++m) {
    // A common edge maps to exactly one LHS and one RHS edge, get the first
    EdgeIndex lhs_edge = rule.common_to_lhs_.edge_mapping_[m][0];
    EdgeIndex rhs_edge = rule.common_to_rhs_.edge_mapping_[m][0];
    // An LHS edge may map to multiple target edges
    const auto &target_edges = lhs_to_target.edge_mapping_[lhs_edge];
    for (EdgeIndex target_edge : target_edges) {
      result.edges_.push_back(target.edges_[target_edge]);
      Edge &edge = result.edges_.back();
      edge.head_ = rhs_to_result_node[rule.rhs_.edges_[rhs_edge].head_];
      edge.tail_ = rhs_to_result_node[rule.rhs_.edges_[rhs_edge].tail_];
    }
  }

  // Add RHS edges which are not in common with the LHS
  std::unordered_set<EdgeIndex> rhs_edges_in_common;
  for (const auto &rhs_edges : rule.common_to_rhs_.edge_mapping_) {
    rhs_edges_in_common.insert(rhs_edges.begin(), rhs_edges.end());
  }
  for (EdgeIndex m = 0; m < rule.rhs_.edges_.size(); ++m) {
    if (rhs_edges_in_common.count(m) == 0) {
      result.edges_.push_back(rule.rhs_.edges_[m]);
      Edge &edge = result.edges_.back();
      edge.head_ = rhs_to_result_node[edge.head_];
      edge.tail_ = rhs_to_result_node[edge.tail_];
    }
  }

  return result;
}

void copyNondefaultAttributes(NodeAttributes &dest, const NodeAttributes &src) {
  static const NodeAttributes defaults{};

  NodeAttributes::accept(
      [](auto &dest_value, auto &&src_value, auto &&default_value) {
        if (src_value != default_value) {
          dest_value = src_value;
        }
      },
      dest, src, defaults);
}

} // namespace robot_design
