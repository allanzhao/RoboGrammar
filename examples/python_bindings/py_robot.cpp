#include <memory>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <robot_design/robot.h>

namespace py = pybind11;
namespace rd = robot_design;

void initRobot(py::module &m) {
  py::enum_<rd::LinkShape>(m, "LinkShape")
      .value("NONE", rd::LinkShape::NONE)
      .value("CAPSULE", rd::LinkShape::CAPSULE)
      .value("CYLINDER", rd::LinkShape::CYLINDER);

  py::enum_<rd::JointType>(m, "JointType")
      .value("NONE", rd::JointType::NONE)
      .value("FREE", rd::JointType::FREE)
      .value("HINGE", rd::JointType::HINGE)
      .value("FIXED", rd::JointType::FIXED);

  py::enum_<rd::JointControlMode>(m, "JointControlMode")
      .value("POSITION", rd::JointControlMode::POSITION)
      .value("VELOCITY", rd::JointControlMode::VELOCITY);

  py::class_<rd::Link>(m, "Link")
      .def(py::init<rd::Index, rd::JointType, rd::Scalar,
                    const rd::Quaternion &, const rd::Vector3 &, rd::LinkShape,
                    rd::Scalar, rd::Scalar, rd::Scalar, rd::Scalar, rd::Scalar,
                    rd::Scalar, rd::Scalar, rd::JointControlMode,
                    const rd::Color &, const rd::Color &, const std::string &,
                    const std::string &>())
      .def_readwrite("parent", &rd::Link::parent_)
      .def_readwrite("joint_type", &rd::Link::joint_type_)
      .def_readwrite("joint_pos", &rd::Link::joint_pos_)
      .def_readwrite("joint_rot", &rd::Link::joint_rot_)
      .def_readwrite("joint_axis", &rd::Link::joint_axis_)
      .def_readwrite("shape", &rd::Link::shape_)
      .def_readwrite("length", &rd::Link::length_)
      .def_readwrite("radius", &rd::Link::radius_)
      .def_readwrite("density", &rd::Link::density_)
      .def_readwrite("friction", &rd::Link::friction_)
      .def_readwrite("joint_kp", &rd::Link::joint_kp_)
      .def_readwrite("joint_kd", &rd::Link::joint_kd_)
      .def_readwrite("joint_torque", &rd::Link::joint_torque_)
      .def_readwrite("joint_control_mode", &rd::Link::joint_control_mode_)
      .def_readwrite("color", &rd::Link::color_)
      .def_readwrite("joint_color", &rd::Link::joint_color_)
      .def_readwrite("label", &rd::Link::label_)
      .def_readwrite("joint_label", &rd::Link::joint_label_);

  py::class_<rd::Robot, std::shared_ptr<rd::Robot>>(m, "Robot")
      .def(py::init<>())
      .def_readwrite("links", &rd::Robot::links_);
}
