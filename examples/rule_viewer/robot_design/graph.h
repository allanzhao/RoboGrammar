#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <functional>
#include <ostream>
#include <robot_design/eigen_hash.h>
#include <robot_design/robot.h>
#include <robot_design/utils.h>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace robot_design {

using SubgraphIndex = std::size_t;
using NodeIndex = std::size_t;
using EdgeIndex = std::size_t;

// Nodes contain Link attributes which should be shared across all instances
// See the definition of Link for more information about these attributes
struct NodeAttributes {
  NodeAttributes() = default;
  NodeAttributes(const std::string &label) : label_(label) {}

  std::string label_ = "";
  LinkShape shape_ = LinkShape::NONE;
  Scalar length_ = 1.0;
  Scalar radius_ = 0.05;
  Scalar density_ = 1.0;
  Scalar friction_ = 0.9;
  bool base_ = false;
  Color color_ = {0.45f, 0.5f, 0.55f}; // Slate gray
  std::string require_label_ = "";     // Only used for rule matching

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  template <typename Visitor, typename... Args>
  static void accept(Visitor &&visit, Args &&... args) {
    visit(std::forward<Args>(args).label_...);
    visit(std::forward<Args>(args).shape_...);
    visit(std::forward<Args>(args).length_...);
    visit(std::forward<Args>(args).radius_...);
    visit(std::forward<Args>(args).density_...);
    visit(std::forward<Args>(args).friction_...);
    visit(std::forward<Args>(args).base_...);
    visit(std::forward<Args>(args).color_...);
    visit(std::forward<Args>(args).require_label_...);
  }
};

struct Node {
  std::string name_;
  NodeAttributes attrs_;
};

// Edges contain Link attributes which are unique to each instance
// E.g. the rigid transformation relative to the parent link, uniform scaling
struct EdgeAttributes {
  EdgeAttributes() = default;

  std::string id_ = "";
  std::string label_ = "";
  JointType joint_type_ = JointType::NONE;
  Scalar joint_pos_ = 1.0;
  Quaternion joint_rot_ = Quaternion::Identity();
  Vector3 joint_axis_ = Vector3::UnitZ();
  Scalar joint_kp_ = 0.01;
  Scalar joint_kd_ = 0.5;
  Scalar joint_torque_ = 1.0;
  Scalar joint_lower_limit_ = 0.0;
  Scalar joint_upper_limit_ = 0.0;
  JointControlMode joint_control_mode_ = JointControlMode::POSITION;
  Scalar scale_ = 1.0;
  bool mirror_ = false;
  Color color_ = {1.0f, 0.5f, 0.3f}; // Coral
  std::string require_label_ = "";   // Only used for rule matching

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  template <typename Visitor, typename... Args>
  static void accept(Visitor &&visit, Args &&... args) {
    visit(std::forward<Args>(args).id_...);
    visit(std::forward<Args>(args).label_...);
    visit(std::forward<Args>(args).joint_type_...);
    visit(std::forward<Args>(args).joint_pos_...);
    visit(std::forward<Args>(args).joint_rot_...);
    visit(std::forward<Args>(args).joint_axis_...);
    visit(std::forward<Args>(args).joint_kp_...);
    visit(std::forward<Args>(args).joint_kd_...);
    visit(std::forward<Args>(args).joint_torque_...);
    visit(std::forward<Args>(args).joint_lower_limit_...);
    visit(std::forward<Args>(args).joint_upper_limit_...);
    visit(std::forward<Args>(args).joint_control_mode_...);
    visit(std::forward<Args>(args).scale_...);
    visit(std::forward<Args>(args).mirror_...);
    visit(std::forward<Args>(args).color_...);
    visit(std::forward<Args>(args).require_label_...);
  }
};

struct Edge {
  NodeIndex head_;
  NodeIndex tail_;
  EdgeAttributes attrs_;
};

// Subgraphs are collections of nodes and edges with default attributes
// Nodes and edges may belong to multiple subgraphs
struct Subgraph {
  std::string name_;
  std::set<NodeIndex> nodes_;
  std::set<EdgeIndex> edges_;
  NodeAttributes node_attrs_;
  EdgeAttributes edge_attrs_;
};

struct Graph {
  std::string name_;
  std::vector<Node> nodes_;
  std::vector<Edge> edges_;
  std::vector<Subgraph> subgraphs_;
};

struct GraphMapping {
  // Node i in the domain graph maps to node_mapping_[i] in the codomain graph
  std::vector<NodeIndex> node_mapping_;
  // Edge l in the domain graph maps to edge_mapping_[l] in the codomain graph
  std::vector<std::vector<EdgeIndex>> edge_mapping_;
};

struct Rule {
  std::string name_;
  Graph lhs_;
  Graph rhs_;
  Graph common_;
  GraphMapping common_to_lhs_;
  GraphMapping common_to_rhs_;
};

std::vector<Graph> loadGraphs(const std::string &path);

void updateNodeAttributes(
    NodeAttributes &node_attrs,
    const std::vector<std::pair<std::string, std::string>> &attr_list);

void updateEdgeAttributes(
    EdgeAttributes &edge_attrs,
    const std::vector<std::pair<std::string, std::string>> &attr_list);

std::ostream &operator<<(std::ostream &out, const Node &node);

std::ostream &operator<<(std::ostream &out, const Edge &edge);

std::ostream &operator<<(std::ostream &out, const Graph &graph);

Robot buildRobot(const Graph &graph);

Rule createRuleFromGraph(const Graph &graph);

std::vector<GraphMapping> findMatches(const Graph &pattern,
                                      const Graph &target);

bool checkRuleApplicability(const Rule &rule, const Graph &target,
                            const GraphMapping &lhs_to_target);

Graph applyRule(const Rule &rule, const Graph &target,
                const GraphMapping &lhs_to_target);

/**
 * Copy attributes that are different from their default values from src to
 * dest.
 */
void copyNondefaultAttributes(NodeAttributes &dest, const NodeAttributes &src);

} // namespace robot_design

namespace std {

template <> struct hash<robot_design::NodeAttributes> {
  std::size_t operator()(const robot_design::NodeAttributes &node_attrs) const {
    using robot_design::hashCombine;

    std::size_t seed = 0;
    robot_design::NodeAttributes::accept(
        [&](auto &&value) { hashCombine(seed, value); }, node_attrs);
    return seed;
  }
};

template <> struct hash<robot_design::Node> {
  std::size_t operator()(const robot_design::Node &node) const {
    using robot_design::hashCombine;

    std::size_t seed = 0;
    hashCombine(seed, node.name_);
    hashCombine(seed, node.attrs_);
    return seed;
  }
};

template <> struct hash<robot_design::EdgeAttributes> {
  std::size_t operator()(const robot_design::EdgeAttributes &edge_attrs) const {
    using robot_design::hashCombine;

    std::size_t seed = 0;
    robot_design::EdgeAttributes::accept(
        [&](auto &&value) { hashCombine(seed, value); }, edge_attrs);
    return seed;
  }
};

template <> struct hash<robot_design::Graph> {
  std::size_t operator()(const robot_design::Graph &graph) const {
    using robot_design::Edge;
    using robot_design::EdgeAttributes;
    using robot_design::EdgeIndex;
    using robot_design::hashCombine;
    using robot_design::Node;
    using robot_design::NodeIndex;
    using robot_design::Subgraph;

    std::size_t seed = 0;
    hashCombine(seed, graph.name_);

    // Hash nodes (order should not matter)
    std::size_t nodes_seed = 0;
    for (std::size_t i = 0; i < graph.nodes_.size(); ++i) {
      const Node &node = graph.nodes_[i];
      std::size_t node_seed = std::hash<Node>()(node);
      std::size_t head_node_seed = 0;
      std::size_t tail_node_seed = 0;
      for (const Edge &edge : graph.edges_) {
        if (edge.head_ == i) {
          head_node_seed += std::hash<EdgeAttributes>()(edge.attrs_);
        }
        if (edge.tail_ == i) {
          tail_node_seed += std::hash<EdgeAttributes>()(edge.attrs_);
        }
      }
      hashCombine(node_seed, head_node_seed);
      hashCombine(node_seed, tail_node_seed);
      nodes_seed += node_seed;
    }
    hashCombine(seed, nodes_seed);

    // Hash edges (order should not matter)
    std::size_t edges_seed = 0;
    for (const Edge &edge : graph.edges_) {
      std::size_t edge_seed = 0;
      hashCombine(edge_seed, graph.nodes_[edge.head_]);
      hashCombine(edge_seed, graph.nodes_[edge.tail_]);
      hashCombine(edge_seed, edge.attrs_);
      edges_seed += edge_seed;
    }
    hashCombine(seed, edges_seed);

    return seed;
  }
};

} // namespace std
