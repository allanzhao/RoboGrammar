#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <memory>
#include <robot_design/graph.h>
#include <robot_design/internal/dot_parsing.h>
#include <robot_design/internal/dot_rules.h>
#include <robot_design/robot.h>
#include <robot_design/types.h>
#include <sstream>
#include <stdexcept>
#include <tao/pegtl.hpp>
#include <vector>

namespace robot_design {

constexpr Scalar RAD_PER_DEG = M_PI / 180;

std::vector<Graph> loadGraphs(const std::string &path) {
  tao::pegtl::file_input<> input(path);
  std::vector<Graph> graphs;
  bool success;

  do {
    dot_parsing::State state;
    // Create a root subgraph with default attribute values
    state.subgraph_states_.emplace_back();
    success =
        tao::pegtl::parse<tao::pegtl::pad<dot_rules::graph, dot_rules::sep>,
                          dot_parsing::dot_action>(input, state);
    if (success) {
      graphs.push_back(std::move(state.result_));
    }
  } while (success);

  return graphs;
}

static bool parseDOTBool(const std::string &str) {
  // TODO: add case insensitivity
  if (str == "true" || str == "yes") {
    return true;
  } else if (str == "false" || str == "no") {
    return false;
  } else {
    std::istringstream in(str);
    bool value;
    in >> value;
    return value;
  }
}

void updateNodeAttributes(
    NodeAttributes &node_attrs,
    const std::vector<std::pair<std::string, std::string>> &attr_list) {
  for (const auto &attr : attr_list) {
    const std::string &key = attr.first;
    const std::string &value = attr.second;
    if (key == "label") {
      node_attrs.label_ = value;
    } else if (key == "shape") {
      if (value == "capsule") {
        node_attrs.shape_ = LinkShape::CAPSULE;
      } else if (value == "cylinder") {
        node_attrs.shape_ = LinkShape::CYLINDER;
      } else {
        throw std::runtime_error("Unexpected value \"" + value +
                                 "\" for link_shape");
      }
    } else if (key == "length") {
      std::istringstream in(value);
      in >> node_attrs.length_;
    } else if (key == "radius") {
      std::istringstream in(value);
      in >> node_attrs.radius_;
    } else if (key == "density") {
      std::istringstream in(value);
      in >> node_attrs.density_;
    } else if (key == "friction") {
      std::istringstream in(value);
      in >> node_attrs.friction_;
    } else if (key == "base") {
      node_attrs.base_ = parseDOTBool(value);
    } else if (key == "color") {
      std::istringstream in(value);
      Color &color = node_attrs.color_;
      in >> color(0) >> color(1) >> color(2);
    } else if (key == "require_label") {
      node_attrs.require_label_ = value;
    }
  }
}

void updateEdgeAttributes(
    EdgeAttributes &edge_attrs,
    const std::vector<std::pair<std::string, std::string>> &attr_list) {
  for (const auto &attr : attr_list) {
    const std::string &key = attr.first;
    const std::string &value = attr.second;
    if (key == "id") {
      edge_attrs.id_ = value;
    } else if (key == "label") {
      edge_attrs.label_ = value;
    } else if (key == "type") {
      if (value == "none") {
        edge_attrs.joint_type_ = JointType::NONE;
      } else if (value == "free") {
        edge_attrs.joint_type_ = JointType::FREE;
      } else if (value == "hinge") {
        edge_attrs.joint_type_ = JointType::HINGE;
      } else if (value == "fixed") {
        edge_attrs.joint_type_ = JointType::FIXED;
      } else {
        throw std::runtime_error("Unexpected value \"" + value +
                                 "\" for joint_type");
      }
    } else if (key == "offset") {
      std::istringstream in(value);
      in >> edge_attrs.joint_pos_;
    } else if (key == "axis_angle") {
      std::istringstream in(value);
      Vector3 axis;
      Scalar angle;
      in >> axis(0) >> axis(1) >> axis(2) >> angle;
      edge_attrs.joint_rot_ =
          Eigen::AngleAxis<Scalar>(angle * RAD_PER_DEG, axis);
    } else if (key == "joint_axis") {
      std::istringstream in(value);
      Vector3 &joint_axis = edge_attrs.joint_axis_;
      in >> joint_axis(0) >> joint_axis(1) >> joint_axis(2);
    } else if (key == "kp") {
      std::istringstream in(value);
      in >> edge_attrs.joint_kp_;
    } else if (key == "kd") {
      std::istringstream in(value);
      in >> edge_attrs.joint_kd_;
    } else if (key == "torque") {
      std::istringstream in(value);
      in >> edge_attrs.joint_torque_;
    } else if (key == "limits") {
      std::istringstream in(value);
      Scalar lower_limit, upper_limit;
      in >> lower_limit >> upper_limit;
      edge_attrs.joint_lower_limit_ = lower_limit * RAD_PER_DEG;
      edge_attrs.joint_upper_limit_ = upper_limit * RAD_PER_DEG;
    } else if (key == "control_mode") {
      if (value == "position") {
        edge_attrs.joint_control_mode_ = JointControlMode::POSITION;
      } else if (value == "velocity") {
        edge_attrs.joint_control_mode_ = JointControlMode::VELOCITY;
      } else {
        throw std::runtime_error("Unexpected value \"" + value +
                                 "\" for control_mode");
      }
    } else if (key == "scale") {
      std::istringstream in(value);
      in >> edge_attrs.scale_;
    } else if (key == "mirror") {
      edge_attrs.mirror_ = parseDOTBool(value);
    } else if (key == "color") {
      std::istringstream in(value);
      Color &color = edge_attrs.color_;
      in >> color(0) >> color(1) >> color(2);
    } else if (key == "require_label") {
      edge_attrs.require_label_ = value;
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

} // namespace robot_design
