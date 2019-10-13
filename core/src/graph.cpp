#include <iostream>
#include <memory>
#include <robot_design/graph.h>
#include <robot_design/internal/dot_parsing.h>
#include <robot_design/internal/dot_rules.h>
#include <robot_design/robot.h>
#include <robot_design/types.h>
#include <tao/pegtl.hpp>

namespace robot_design {

std::shared_ptr<Graph> loadGraph(const std::string &filename) {
  tao::pegtl::file_input<> input(filename);
  dot_parsing::State state;
  // Create a root subgraph with default attribute values
  state.subgraph_states_.push_back({
      /*result=*/{},
      /*node_attributes=*/{
          /*joint_type=*/JointType::HINGE,
          /*joint_axis=*/Vector3::UnitZ(),
          /*shape=*/LinkShape::CAPSULE,
          /*length=*/1.0},
      /*edge_attributes=*/{
          /*joint_pos=*/1.0,
          /*joint_rot=*/Quaternion::Identity(),
          /*scale=*/1.0}});
  tao::pegtl::parse<
      tao::pegtl::pad<dot_rules::graph, dot_rules::sep>,
      dot_parsing::dot_action>(input, state);
  // TODO
  return std::make_shared<Graph>();
}

std::ostream &operator<<(std::ostream &out, const Node &node) {
  out << "Node{\"" << node.name_ << "\"}";
  return out;
}

std::ostream &operator<<(std::ostream &out, const Edge &edge) {
  out << "Edge{\"" << edge.head_ << "\", \"" << edge.tail_ << "\"}";
  return out;
}

std::ostream &operator<<(std::ostream &out, const Graph &graph) {
  out << "Graph{\"" << graph.name_ << "\", {" << std::endl;
  for (const auto &node : graph.nodes_) {
    out << "    " << node << "," << std::endl;
  }
  out << "}, {" << std::endl;
  for (const auto &edge : graph.edges_) {
    out << "    " << edge << "," << std::endl;
  }
  out << "}}";
  return out;
}

}  // namespace robot_design
