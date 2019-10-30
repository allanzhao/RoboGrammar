#include <algorithm>
#include <robot_design/graph.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace robot_design {

std::vector<GraphMapping> findMatches(
    const Graph &pattern, const Graph &target) {
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

    // Edges in pattern incident on i must also be present in target
    bool edge_fail = false;
    for (const Edge &pattern_edge : pattern.edges_) {
      if (pattern_edge.head_ == i && pattern_edge.tail_ <= i) {
        // Pattern edge i_tail -> i requires target edge j_tail -> j
        NodeIndex j_tail = pm.node_mapping_[pattern_edge.tail_];
        auto it = std::find_if(target.edges_.begin(), target.edges_.end(),
            [=] (const Edge &target_edge) {
              return target_edge.head_ == j &&
                     target_edge.tail_ == j_tail; });
        if (it == target.edges_.end()) {
          // No such target edge exists
          edge_fail = true;
          break;
        }
      } else if (pattern_edge.tail_ == i && pattern_edge.head_ <= i) {
        // Pattern edge i -> i_head requires target edge j -> j_head
        NodeIndex j_head = pm.node_mapping_[pattern_edge.head_];
        auto it = std::find_if(target.edges_.begin(), target.edges_.end(),
            [=] (const Edge &target_edge) {
              return target_edge.tail_ == j &&
                     target_edge.head_ == j_head; });
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
        NodeIndex j_head = new_match.node_mapping_[pattern_edge.head_];
        NodeIndex j_tail = new_match.node_mapping_[pattern_edge.tail_];
        for (EdgeIndex n = 0; n < target.edges_.size(); ++n) {
          const Edge &target_edge = target.edges_[n];
          if (target_edge.head_ == j_head && target_edge.tail_ == j_tail) {
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

Graph applyRule(
    const Rule &rule, const Graph &target, const GraphMapping &lhs_to_target) {
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
    NodeIndex target_node = lhs_to_target.node_mapping_[lhs_node];
    result.nodes_.push_back(target.nodes_[target_node]);
    target_to_result_node[target_node] = result.nodes_.size() - 1;
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
    }
  }

  // Copy target edges in LHS to result if they are in common with the RHS
  for (EdgeIndex m = 0; m < rule.common_.edges_.size(); ++m) {
    // A common edge maps to exactly one LHS edge, just get the first one
    EdgeIndex lhs_edge = rule.common_to_lhs_.edge_mapping_[m][0];
    // An LHS edge may map to multiple target edges
    const auto &target_edges = lhs_to_target.edge_mapping_[lhs_edge];
    for (EdgeIndex target_edge : target_edges) {
      result.edges_.push_back(target.edges_[target_edge]);
      Edge &edge = result.edges_.back();
      edge.head_ = target_to_result_node[edge.head_];
      edge.tail_ = target_to_result_node[edge.tail_];
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

}  // namespace robot_design
