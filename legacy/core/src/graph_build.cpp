#include <algorithm>
#include <cstddef>
#include <deque>
#include <iterator>
#include <memory>
#include <robot_design/graph.h>
#include <robot_design/robot.h>
#include <stdexcept>

namespace robot_design {

Robot buildRobot(const Graph &graph) {
  // State struct for graph traversal
  struct NodeEntry {
    NodeIndex node_;
    // Arguments for link construction
    Index parent_link_;
    JointType joint_type_;
    Scalar joint_pos_;
    Quaternion joint_rot_;
    Vector3 joint_axis_;
    Scalar joint_kp_;
    Scalar joint_kd_;
    Scalar joint_torque_;
    JointControlMode joint_control_mode_;
    Color joint_color_;
    std::string joint_label_;
    // Cumulative scaling factor
    Scalar scale_;
    // Mirror subsequent links/joints across the xy plane
    bool mirror_;
  };

  assert(!graph.nodes_.empty());

  // Find a root for the kinematic tree (which will become the base link)
  NodeIndex root_node = 0;
  // If a node has base == true, use it as the root
  const auto it =
      std::find_if(graph.nodes_.begin(), graph.nodes_.end(),
                   [](const Node &node) { return node.attrs_.base_; });
  if (it != graph.nodes_.end()) {
    root_node = std::distance(graph.nodes_.begin(), it);
  } else {
    // Follow edges backwards in the graph to find a root
    // Limit the number of iterations in case the graph contains a cycle
    for (std::size_t i = 0; i < graph.edges_.size(); ++i) {
      // Find an edge pointing towards this node
      const auto it = std::find_if(
          graph.edges_.begin(), graph.edges_.end(),
          [root_node](const Edge &edge) { return edge.head_ == root_node; });
      if (it != graph.edges_.end()) {
        root_node = it->tail_; // Follow edge backwards
      } else {
        break; // Node is a root
      }
    }
  }

  // Build the (simulated) robot using breadth-first traversal
  Robot robot;
  // The base link doesn't have a joint, so joint-related params don't matter
  std::deque<NodeEntry> entries_to_expand = {NodeEntry{
      /*node=*/root_node, /*parent_link=*/-1, /*joint_type=*/JointType::FREE,
      /*joint_pos=*/0.0, /*joint_rot=*/Quaternion::Identity(),
      /*joint_axis=*/Vector3::Zero(), /*joint_kp=*/0.0, /*joint_kd=*/0.0,
      /*joint_torque=*/1.0, /*joint_control_mode=*/JointControlMode::POSITION,
      /*joint_color=*/Color::Zero(), /*joint_label=*/"", /*scale=*/1.0,
      /*mirror=*/false}};
  while (!entries_to_expand.empty()) {
    NodeEntry &entry = entries_to_expand.front();
    const Node &node = graph.nodes_[entry.node_];
    // Add a link corresponding to this node
    Index link_index = robot.links_.size();
    Quaternion joint_rot = entry.joint_rot_;
    Vector3 joint_axis = entry.joint_axis_;
    if (entry.mirror_) {
      // Mirror parameters across xy plane
      joint_rot.x() = -joint_rot.x();
      joint_rot.y() = -joint_rot.y();
      joint_axis(2) = -joint_axis(2);
    }
    robot.links_.emplace_back(
        /*parent=*/entry.parent_link_, /*joint_type=*/entry.joint_type_,
        /*joint_pos=*/entry.joint_pos_, /*joint_rot=*/joint_rot,
        /*joint_axis=*/joint_axis, /*shape=*/node.attrs_.shape_,
        /*length=*/node.attrs_.length_, /*radius=*/node.attrs_.radius_,
        /*density=*/node.attrs_.density_, /*friction=*/node.attrs_.friction_,
        /*joint_kp=*/entry.joint_kp_, /*joint_kd=*/entry.joint_kd_,
        /*joint_torque=*/entry.joint_torque_,
        /*joint_control_mode=*/entry.joint_control_mode_,
        /*color=*/node.attrs_.color_, /*joint_color=*/entry.joint_color_,
        /*label=*/node.attrs_.label_, /*joint_label=*/entry.joint_label_);

    for (const Edge &edge : graph.edges_) {
      if (edge.tail_ == entry.node_) {
        // Outgoing edge from this node, push an entry for the node it points to
        entries_to_expand.push_back(
            {/*node=*/edge.head_, /*parent_link=*/link_index,
             /*joint_type=*/edge.attrs_.joint_type_,
             /*joint_pos=*/edge.attrs_.joint_pos_,
             /*joint_rot=*/edge.attrs_.joint_rot_,
             /*joint_axis=*/edge.attrs_.joint_axis_,
             /*joint_kp=*/edge.attrs_.joint_kp_,
             /*joint_kd=*/edge.attrs_.joint_kd_,
             /*joint_torque=*/edge.attrs_.joint_torque_,
             /*joint_control_mode=*/edge.attrs_.joint_control_mode_,
             /*joint_color=*/edge.attrs_.color_,
             /*joint_label=*/edge.attrs_.label_,
             /*scale=*/entry.scale_ * edge.attrs_.scale_,
             /*mirror=*/entry.mirror_ != edge.attrs_.mirror_});
      }
    }

    entries_to_expand.pop_front();
  }

  return robot;
}

} // namespace robot_design
