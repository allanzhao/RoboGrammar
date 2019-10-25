#include <algorithm>
#include <robot_design/graph.h>
#include <vector>

namespace robot_design {

std::vector<GraphMatch> findMatches(const Graph &target, const Graph &pattern) {
  assert(pattern.nodes_.size() >= 1);

  // Stack for backtracking, initialized with the first partial match to try
  // The last entry of each partial match is speculative
  std::vector<GraphMatch> partial_matches = {GraphMatch{{0}}};
  std::vector<GraphMatch> matches;

  while (!partial_matches.empty()) {
    GraphMatch &pm = partial_matches.back();
    NodeIndex i = pm.node_mapping_.size() - 1;
    NodeIndex &k = pm.node_mapping_.back();

    // Try to map node i in pattern to node k in target

    if (k >= target.nodes_.size()) {
      // No more possible matches with this prefix, backtrack
      partial_matches.pop_back();
      if (!partial_matches.empty()) {
        GraphMatch &parent_pm = partial_matches.back();
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
      // Match is complete
      matches.push_back(pm);
      ++k;
    } else {
      // Recurse
      partial_matches.push_back(pm);
      GraphMatch &child_pm = partial_matches.back();
      child_pm.node_mapping_.push_back(0);
    }
  }

  return matches;
}

}  // namespace robot_design
