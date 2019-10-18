#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <memory>
#include <robot_design/graph.h>
#include <robot_design/internal/dot_parsing.h>
#include <robot_design/internal/dot_rules.h>
#include <robot_design/robot.h>
#include <robot_design/types.h>
#include <sstream>
#include <stdexcept>
#include <tao/pegtl.hpp>

namespace robot_design {

std::shared_ptr<Graph> loadGraph(const std::string &filename) {
  tao::pegtl::file_input<> input(filename);
  dot_parsing::State state;
  // Create a root subgraph with default attribute values
  state.subgraph_states_.emplace_back();
  dot_parsing::SubgraphState &root_subgraph_state =
      state.subgraph_states_.back();
  root_subgraph_state.result_.node_attrs_ = {
      /*joint_type=*/JointType::HINGE,
      /*joint_axis=*/Vector3::UnitZ(),
      /*shape=*/LinkShape::NONE,
      /*length=*/1.0};
  root_subgraph_state.result_.edge_attrs_ = {
      /*joint_pos=*/1.0,
      /*joint_rot=*/Quaternion::Identity(),
      /*scale=*/1.0};
  tao::pegtl::parse<
      tao::pegtl::pad<dot_rules::graph, dot_rules::sep>,
      dot_parsing::dot_action>(input, state);
  return std::make_shared<Graph>(std::move(state.result_));
}

void NodeAttributes::load(
    const std::vector<std::pair<std::string, std::string>> &attr_list) {
  for (const auto &attr : attr_list) {
    const std::string &key = attr.first;
    const std::string &value = attr.second;
    if (key == "joint_type") {
      if (value == "free") {
        joint_type_ = JointType::FREE;
      } else if (value == "hinge") {
        joint_type_ = JointType::HINGE;
      } else if (value == "fixed") {
        joint_type_ = JointType::FIXED;
      } else {
        throw std::runtime_error(
            "Unexpected value \"" + value + "\" for joint_type");
      }
    } else if (key == "joint_axis") {
      std::istringstream in(value);
      in >> joint_axis_(0) >> joint_axis_(1) >> joint_axis_(2);
    } else if (key == "link_shape") {
      if (value == "capsule") {
        shape_ = LinkShape::CAPSULE;
      } else if (value == "cylinder") {
        shape_ = LinkShape::CYLINDER;
      } else {
        throw std::runtime_error(
            "Unexpected value \"" + value + "\" for link_shape");
      }
    } else if (key == "length") {
      std::istringstream in(value);
      in >> length_;
    }
  }
}

void EdgeAttributes::load(
    const std::vector<std::pair<std::string, std::string>> &attr_list) {
  for (const auto &attr : attr_list) {
    const std::string &key = attr.first;
    const std::string &value = attr.second;
    if (key == "offset") {
      std::istringstream in(value);
      in >> joint_pos_;
    } else if (key == "axis_angle") {
      std::istringstream in(value);
      Vector3 axis;
      Scalar angle;
      in >> axis(0) >> axis(1) >> axis(2) >> angle;
      joint_rot_ = Eigen::AngleAxis<Scalar>(angle * RAD_PER_DEG, axis);
    } else if (key == "scale") {
      std::istringstream in(value);
      in >> scale_;
    }
  }
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
