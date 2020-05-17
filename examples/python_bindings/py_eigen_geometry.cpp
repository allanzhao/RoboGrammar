#include <Eigen/Geometry>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename T>
void initEigenQuaternion(py::module &m, const char *name) {
  // TODO: Make this useful, not just a fancy tuple
  using QuaternionT = Eigen::Quaternion<T>;
  using Vector3T = Eigen::Vector3<T>;
  py::class_<QuaternionT>(m, name, py::module_local())
      .def(py::init<const T &, const T &, const T &, const T &>())
      .def_property("w", [](const QuaternionT &self) { return self.w(); },
                    [](QuaternionT &self, T val) { self.w() = val; })
      .def_property("x", [](const QuaternionT &self) { return self.x(); },
                    [](QuaternionT &self, T val) { self.x() = val; })
      .def_property("y", [](const QuaternionT &self) { return self.y(); },
                    [](QuaternionT &self, T val) { self.y() = val; })
      .def_property("z", [](const QuaternionT &self) { return self.z(); },
                    [](QuaternionT &self, T val) { self.z() = val; })
      .def("to_rotation_matrix", &QuaternionT::toRotationMatrix)
      .def_static("from_two_vectors", [](const Vector3T &a, const Vector3T &b) {
        return QuaternionT::FromTwoVectors(a, b);
      });
}

void initEigenGeometry(py::module &m) {
  initEigenQuaternion<float>(m, "Quaternionf");
  initEigenQuaternion<double>(m, "Quaterniond");
}
