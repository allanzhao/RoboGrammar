#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <robot_design/graph.h>

namespace py = pybind11;
namespace rd = robot_design;

void initGraph(py::module &m) {
  py::class_<rd::NodeAttributes>(m, "NodeAttributes")
      .def(py::init<>())
      .def(py::init<const std::string &>())
      .def("__hash__",
           [](const rd::NodeAttributes *self) {
             return std::hash<rd::NodeAttributes>()(*self);
           })
      .def_readwrite("label", &rd::NodeAttributes::label_)
      .def_readwrite("shape", &rd::NodeAttributes::shape_)
      .def_readwrite("length", &rd::NodeAttributes::length_)
      .def_readwrite("radius", &rd::NodeAttributes::radius_)
      .def_readwrite("density", &rd::NodeAttributes::density_)
      .def_readwrite("friction", &rd::NodeAttributes::friction_)
      .def_readwrite("base", &rd::NodeAttributes::base_)
      .def_readwrite("color", &rd::NodeAttributes::color_)
      .def_readwrite("require_label", &rd::NodeAttributes::require_label_);

  py::class_<rd::Node>(m, "Node")
      .def(py::init<>())
      .def("__hash__",
           [](const rd::Node *self) { return std::hash<rd::Node>()(*self); })
      .def_readwrite("name", &rd::Node::name_)
      .def_readwrite("attrs", &rd::Node::attrs_);

  py::class_<rd::EdgeAttributes>(m, "EdgeAttributes")
      .def(py::init<>())
      .def("__hash__",
           [](const rd::EdgeAttributes *self) {
             return std::hash<rd::EdgeAttributes>()(*self);
           })
      .def_readwrite("id", &rd::EdgeAttributes::id_)
      .def_readwrite("label", &rd::EdgeAttributes::label_)
      .def_readwrite("joint_type", &rd::EdgeAttributes::joint_type_)
      .def_readwrite("joint_pos", &rd::EdgeAttributes::joint_pos_)
      .def_readwrite("joint_rot", &rd::EdgeAttributes::joint_rot_)
      .def_readwrite("joint_axis", &rd::EdgeAttributes::joint_axis_)
      .def_readwrite("joint_kp", &rd::EdgeAttributes::joint_kp_)
      .def_readwrite("joint_kd", &rd::EdgeAttributes::joint_kd_)
      .def_readwrite("joint_torque", &rd::EdgeAttributes::joint_torque_)
      .def_readwrite("joint_lower_limit",
                     &rd::EdgeAttributes::joint_lower_limit_)
      .def_readwrite("joint_upper_limit",
                     &rd::EdgeAttributes::joint_upper_limit_)
      .def_readwrite("joint_control_mode",
                     &rd::EdgeAttributes::joint_control_mode_)
      .def_readwrite("scale", &rd::EdgeAttributes::scale_)
      .def_readwrite("mirror", &rd::EdgeAttributes::mirror_)
      .def_readwrite("color", &rd::EdgeAttributes::color_)
      .def_readwrite("require_label", &rd::EdgeAttributes::require_label_);

  py::class_<rd::Edge>(m, "Edge")
      .def(py::init<>())
      .def_readwrite("head", &rd::Edge::head_)
      .def_readwrite("tail", &rd::Edge::tail_)
      .def_readwrite("attrs", &rd::Edge::attrs_);

  py::class_<rd::Subgraph>(m, "Subgraph")
      .def(py::init<>())
      .def_readwrite("name", &rd::Subgraph::name_)
      .def_readwrite("nodes", &rd::Subgraph::nodes_)
      .def_readwrite("edges", &rd::Subgraph::edges_)
      .def_readwrite("node_attrs", &rd::Subgraph::node_attrs_)
      .def_readwrite("edge_attrs", &rd::Subgraph::edge_attrs_);

  py::class_<rd::Graph>(m, "Graph")
      .def(py::init<>())
      .def("__hash__",
           [](const rd::Graph *self) { return std::hash<rd::Graph>()(*self); })
      .def_readwrite("name", &rd::Graph::name_)
      .def_readwrite("nodes", &rd::Graph::nodes_)
      .def_readwrite("edges", &rd::Graph::edges_)
      .def_readwrite("subgraphs", &rd::Graph::subgraphs_);

  py::class_<rd::GraphMapping>(m, "GraphMapping")
      .def(py::init<>())
      .def_readwrite("node_mapping", &rd::GraphMapping::node_mapping_)
      .def_readwrite("edge_mapping", &rd::GraphMapping::edge_mapping_);

  py::class_<rd::Rule>(m, "Rule")
      .def(py::init<>())
      .def_readwrite("name", &rd::Rule::name_)
      .def_readwrite("lhs", &rd::Rule::lhs_)
      .def_readwrite("rhs", &rd::Rule::rhs_)
      .def_readwrite("common", &rd::Rule::common_)
      .def_readwrite("common_to_lhs", &rd::Rule::common_to_lhs_)
      .def_readwrite("common_to_rhs", &rd::Rule::common_to_rhs_);

  m.def("load_graphs", &rd::loadGraphs);
  m.def("build_robot", &rd::buildRobot);
  m.def("create_rule_from_graph", &rd::createRuleFromGraph);
  m.def("find_matches", &rd::findMatches);
  m.def("check_rule_applicability", &rd::checkRuleApplicability);
  m.def("apply_rule", &rd::applyRule);
}
