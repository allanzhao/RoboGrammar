#include <algorithm>
#include <deque>
#include <iterator>
#include <memory>
#include <robot_design/graph.h>
#include <robot_design/robot.h>
#include <stdexcept>

namespace robot_design {

Robot buildRobot(const Graph &graph) {
  struct NodeEntry {
    NodeIndex node_;
    // Arguments for link construction
    Index parent_link_;
    JointType joint_type_;
    Scalar joint_pos_;
    Quaternion joint_rot_;
    Vector3 joint_axis_;
    // Cumulative scaling factor
    Scalar scale_;
  };
  Robot robot(/*link_density=*/1.0, /*link_radius=*/0.05, /*friction=*/0.9,
              /*motor_kp=*/2.0, /*motor_kd=*/0.1);
  // The first node with base == true is the starting node
  const auto it = std::find_if(graph.nodes_.begin(), graph.nodes_.end(),
      [] (const Node &node) {
        return node.attrs_.base_;
      });
  if (it == graph.nodes_.end()) {
    throw std::runtime_error(
        "Graph has no suitable starting node (no node has base == true)");
  }
  NodeIndex starting_node = std::distance(graph.nodes_.begin(), it);
  std::deque<NodeEntry> entries_to_expand = {NodeEntry{
      /*node=*/starting_node, /*parent_link=*/-1,
      /*joint_type=*/JointType::FREE, /*joint_pos=*/0.0,
      /*joint_rot=*/Quaternion::Identity(), /*joint_axis=*/Vector3::Zero(),
      /*scale=*/1.0}};

  while (!entries_to_expand.empty()) {
    NodeEntry &entry = entries_to_expand.front();
    const Node &node = graph.nodes_[entry.node_];
    // Add a link corresponding to this node
    Index link_index = robot.links_.size();
    robot.links_.emplace_back(
        /*parent=*/entry.parent_link_, /*joint_type=*/entry.joint_type_,
        /*joint_pos=*/entry.joint_pos_, /*joint_rot=*/entry.joint_rot_,
        /*joint_axis=*/entry.joint_axis_, /*shape=*/node.attrs_.shape_,
        /*length=*/node.attrs_.length_);

    for (const Edge &edge : graph.edges_) {
      if (edge.tail_ == entry.node_) {
        // Outgoing edge from this node, push an entry for the node it points to
        entries_to_expand.push_back({
            /*node=*/edge.head_, /*parent_link=*/link_index,
            /*joint_type=*/edge.attrs_.joint_type_,
            /*joint_pos=*/edge.attrs_.joint_pos_,
            /*joint_rot=*/edge.attrs_.joint_rot_,
            /*joint_axis=*/edge.attrs_.joint_axis_,
            /*scale=*/entry.scale_ * edge.attrs_.scale_});
      }
    }

    entries_to_expand.pop_front();
  }

  return robot;
}

}  // namespace robot_design
