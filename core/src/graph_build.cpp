#include <deque>
#include <memory>
#include <robot_design/graph.h>
#include <robot_design/robot.h>

namespace robot_design {

std::shared_ptr<Robot> buildRobot(const Graph &graph) {
  auto robot = std::make_shared<Robot>(
      /*link_density=*/1.0, /*link_radius=*/0.025, /*friction=*/0.9,
      /*motor_kp=*/2.0, /*motor_kd=*/0.1);
  struct NodeEntry {
    NodeIndex node_;
    // Arguments for link construction
    Index parent_link_;
    Scalar joint_pos_;
    Quaternion joint_rot_;
    // Cumulative scaling factor
    Scalar scale_;
  };
  std::deque<NodeEntry> entries_to_expand = {NodeEntry{
      /*node=*/0, /*parent_link=*/-1, /*joint_pos=*/0.0,
      /*joint_rot=*/Quaternion::Identity(), /*scale=*/1.0}};

  while (!entries_to_expand.empty()) {
    NodeEntry &entry = entries_to_expand.front();
    const Node &node = graph.nodes_[entry.node_];
    // Add a link corresponding to this node
    Index link_index = robot->links_.size();
    robot->links_.emplace_back(
        /*parent=*/entry.parent_link_, /*joint_type=*/node.attrs_.joint_type_,
        /*joint_pos=*/entry.joint_pos_, /*joint_rot=*/entry.joint_rot_,
        /*joint_axis=*/node.attrs_.joint_axis_, /*shape=*/node.attrs_.shape_,
        /*length=*/node.attrs_.length_);

    for (const Edge &edge : graph.edges_) {
      if (edge.tail_ == entry.node_) {
        // Outgoing edge from this node, push an entry for the node it points to
        entries_to_expand.push_back({
            /*node=*/edge.head_, /*parent_link=*/link_index,
            /*joint_pos=*/edge.attrs_.joint_pos_,
            /*joint_rot=*/edge.attrs_.joint_rot_,
            /*scale=*/entry.scale_ * edge.attrs_.scale_});
      }
    }

    entries_to_expand.pop_front();
  }

  return robot;
}

}  // namespace robot_design
