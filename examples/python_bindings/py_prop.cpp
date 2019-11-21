#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <robot_design/prop.h>

namespace py = pybind11;
namespace rd = robot_design;

void initProp(py::module &m) {
  py::class_<rd::Prop>(m, "Prop")
      .def(py::init<>())
      .def_readwrite("density", &rd::Prop::density_)
      .def_readwrite("friction", &rd::Prop::friction_)
      .def_readwrite("half_extents", &rd::Prop::half_extents_);
}
