#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <robot_design/prop.h>

namespace py = pybind11;
namespace rd = robot_design;

void initProp(py::module &m) {
  py::enum_<rd::PropShape>(m, "PropShape")
      .value("BOX", rd::PropShape::BOX);

  py::class_<rd::Prop, std::shared_ptr<rd::Prop>>(m, "Prop")
      .def(py::init<>())
      .def(py::init<rd::PropShape, rd::Scalar, rd::Scalar, rd::Vector3>())
      .def_readwrite("shape", &rd::Prop::shape_)
      .def_readwrite("density", &rd::Prop::density_)
      .def_readwrite("friction", &rd::Prop::friction_)
      .def_readwrite("half_extents", &rd::Prop::half_extents_)
      .def_readwrite("color", &rd::Prop::color_);

  py::class_<rd::HeightfieldProp, rd::Prop,
             std::shared_ptr<rd::HeightfieldProp>>(m, "HeightfieldProp")
      .def(py::init<rd::Scalar, const rd::Vector3 &, const rd::MatrixX &>())
      .def_readwrite("heightfield", &rd::HeightfieldProp::heightfield_);
}
