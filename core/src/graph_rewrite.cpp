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
    NodeIndex &k = pm.node_mapping_.back();

    // Try to map node i in pattern to node k in target

    if (k >= target.nodes_.size()) {
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
        // Pattern edge i_tail -> i requires target edge k_tail -> k
        NodeIndex k_tail = pm.node_mapping_[pattern_edge.tail_];
        auto it = std::find_if(target.edges_.begin(), target.edges_.end(),
            [=] (const Edge &target_edge) {
              return target_edge.head_ == k &&
                     target_edge.tail_ == k_tail; });
        if (it == target.edges_.end()) {
          // No such target edge exists
          edge_fail = true;
          break;
        }
      } else if (pattern_edge.tail_ == i && pattern_edge.head_ <= i) {
        // Pattern edge i -> i_head requires target edge k -> k_head
        NodeIndex k_head = pm.node_mapping_[pattern_edge.head_];
        auto it = std::find_if(target.edges_.begin(), target.edges_.end(),
            [=] (const Edge &target_edge) {
              return target_edge.tail_ == k &&
                     target_edge.head_ == k_head; });
        if (it == target.edges_.end()) {
          // No such target edge exists
          edge_fail = true;
          break;
        }
      }
    }
    if (edge_fail) {
      ++k;
      continue;
    }

    // Partial match is consistent with pattern

    if (pm.node_mapping_.size() == pattern.nodes_.size()) {
      // Node matching is complete, fill in edge matches
      matches.push_back(pm);
      GraphMapping &new_match = matches.back();
      new_match.edge_mapping_.resize(pattern.edges_.size());
      for (EdgeIndex j = 0; j < pattern.edges_.size(); ++j) {
        const Edge &pattern_edge = pattern.edges_[j];
        NodeIndex k_head = new_match.node_mapping_[pattern_edge.head_];
        NodeIndex k_tail = new_match.node_mapping_[pattern_edge.tail_];
        for (EdgeIndex l = 0; l < target.edges_.size(); ++l) {
          const Edge &target_edge = target.edges_[l];
          if (target_edge.head_ == k_head && target_edge.tail_ == k_tail) {
            new_match.edge_mapping_[j].push_back(l);
          }
        }
      }
      ++k;
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

/*
  // Construct sets of node and edge indices to remove from target
  std::unordered_set<NodeIndex> target_nodes_to_remove;
  for (NodeIndex i : rule.lhs_nodes_to_remove_) {
    target_nodes_to_remove.insert(lhs_match.node_mapping_[i]);
  }
  std::unordered_set<EdgeIndex> target_edges_to_remove;
  for (EdgeIndex j : rule.lhs_edges_to_remove_) {
    target_edges_to_remove.insert(lhs_match.edge_mapping_[j].begin(),
                                  lhs_match.edge_mapping_[j].end());
  }

  // Copy nodes not being removed to the result graph
  // Node k in target maps to target_node_mapping[k] in result
  std::vector<NodeIndex> target_node_mapping(target.nodes_.size(), -1);
  for (NodeIndex k = 0; k < target.nodes_.size(); ++k) {
    if (target_nodes_to_remove.count(k) == 0) {
      result.nodes_.push_back(target.nodes_[k]);
      target_node_mapping[k] = result.nodes_.size() - 1;
    }
  }

  // Add any new nodes from the RHS
  // Node m in RHS maps to rhs_node_mapping[m] in result
  std::vector<NodeIndex> rhs_node_mapping(rule.rhs_.nodes_.size(), -1);
  for (NodeIndex m : rule.rhs_nodes_to_add_) {
    result.nodes_.push_back(rule.rhs_.nodes_[m]);
    rhs_node_mapping[m] = result.nodes_.size() - 1;
  }

  // Construct a mapping from edge indices in target to edge indices in the RHS
  std::unordered_map<EdgeIndex, EdgeIndex> target_edge_mapping;
  for (EdgeIndex j = 0; j < lhs_match.edge_mapping_.size(); ++j) {
    for (EdgeIndex l : lhs_match.edge_mapping_[j]) {

      target_edge_mapping.emplace(l, j);
    }
  }

  // Copy edges not being removed to the result graph
  for (EdgeIndex l = 0; l < target.edges_.size(); ++l) {
    if (target_edges_to_remove.count(l) == 0) {
      result.edges_.push_back(target.edges_[l]);
      Edge &edge = result.edges_.back();
      if (target_edges_in_lhs.count(l) == 0) {
        // Edge is not in the LHS match, its endpoints are the same as in target
        edge.head_ = target_node_mapping[edge.head_];
        edge.tail_ = target_node_mapping[edge.tail_];
      } else {
        // Edge is in the LHS match, its endpoints are now determined by the RHS
        const auto it = std::find(
*/



  return result;
}

}  // namespace robot_design
