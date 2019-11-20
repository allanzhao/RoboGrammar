#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <robot_design/graph.h>

namespace py = pybind11;

PYBIND11_MODULE(pyrobotdesign, m) {
  py::class_<robot_design::Graph>(m, "Graph")
      .def(py::init<>())
      .def_readwrite("name", &robot_design::Graph::name_)
      .def_readwrite("nodes", &robot_design::Graph::nodes_)
      .def_readwrite("edges", &robot_design::Graph::edges_)
      .def_readwrite("subgraphs", &robot_design::Graph::subgraphs_);
}
