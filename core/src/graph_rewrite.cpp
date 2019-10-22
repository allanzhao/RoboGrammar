#include <algorithm>
#include <robot_design/graph.h>
#include <vector>

namespace robot_design {

std::vector<Subgraph> findMatches(const Graph &source, const Graph &pattern) {
  // Assume there is at least one node in pattern
  struct PartialMatch {
    // Node i in pattern maps to node_mapping_[i] in source
    // The last entry is speculative
    std::vector<NodeIndex> node_mapping_;
  };
  // Stack for backtracking, initialized with the first partial match to try
  std::vector<PartialMatch> partial_matches = {PartialMatch{{0}}};

  while (!partial_matches.empty()) {
    PartialMatch &pm = partial_matches.back();
    NodeIndex i = pm.node_mapping_.size() - 1;
    NodeIndex &k = pm.node_mapping_.back();

    // Try to map node i in pattern to node k in source

    if (k >= source.nodes_.size()) {
      // No more possible matches with this prefix, backtrack
      partial_matches.pop_back();
      if (!partial_matches.empty()) {
        PartialMatch &parent_pm = partial_matches.back();
        ++parent_pm.node_mapping_.back();
      }
      continue;
    }

    if (std::find(pm.node_mapping_.begin(), pm.node_mapping_.end() - 1, k)) {
      // Node k in source was already used
      ++k;
      continue;
    }

    // Edges in pattern incident on i must also be present in source
    bool edge_fail = false;
    for (const Edge &pattern_edge : pattern.edges_) {
      if (pattern_edge.head_ == i && pattern_edge.tail_ <= i) {
        // Pattern edge i_tail -> i requires source edge k_tail -> k
        NodeIndex k_tail = pm.node_mapping_[pattern_edge.tail_];
        auto it = std::find_if(source.edges_.begin(), source.edges_.end(),
            [=] (const Edge &source_edge) {
                source_edge.head_ == k && source_edge_.tail_ == k_tail; }
        if (it == source.edges_.end()) {
          // No such source edge exists
          edge_fail = true;
          break;
        }
      } else if (pattern_edge.tail_ == i && pattern_edge.head_ <= i) {
        // Pattern edge i -> i_head requires source edge k -> k_head
        NodeIndex k_head = pm.node_mapping_[pattern_edge.head_];
        auto it = std::find_if(source.edges_.begin(), source.edges_.end(),
            [=] (const Edge &source_edge) {
                source_edge.tail_ == k && source_edge_.head_ == k_head; }
        if (it == source.edges_.end()) {
          // No such source edge exists
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
      // TODO: output match
      ++k;
    } else {
      // Recurse
      partial_matches.push_back(pm);
      PartialMatch &child_pm = partial_matches.back();
      child_pm.node_mapping_.push_back(0);
    }
  }
}

}  // namespace robot_design
