#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <robot_design/glfw_viewer.h>
#include <robot_design/render.h>

namespace py = pybind11;
namespace rd = robot_design;

void initRender(py::module &m) {
  py::class_<rd::CameraParameters>(m, "CameraParameters")
      .def(py::init<>())
      .def("get_proj_matrix", &rd::CameraParameters::getProjMatrix)
      .def("get_view_matrix", &rd::CameraParameters::getViewMatrix)
      .def_readwrite("aspect_ratio", &rd::CameraParameters::aspect_ratio_)
      .def_readwrite("z_near", &rd::CameraParameters::z_near_)
      .def_readwrite("z_far", &rd::CameraParameters::z_far_)
      .def_readwrite("fov", &rd::CameraParameters::fov_)
      .def_readwrite("position", &rd::CameraParameters::position_)
      .def_readwrite("yaw", &rd::CameraParameters::yaw_)
      .def_readwrite("pitch", &rd::CameraParameters::pitch_)
      .def_readwrite("distance", &rd::CameraParameters::distance_);

  py::class_<rd::FPSCameraController>(m, "FPSCameraController")
      .def(py::init<>())
      .def(py::init<float, float, float>())
      .def("update", &rd::FPSCameraController::update)
      .def_readwrite("move_speed", &rd::FPSCameraController::move_speed_)
      .def_readwrite("mouse_sensitivity",
                     &rd::FPSCameraController::mouse_sensitivity_)
      .def_readwrite("scroll_sensitivity",
                     &rd::FPSCameraController::scroll_sensitivity_);

  py::class_<rd::GLFWViewer>(m, "GLFWViewer")
      .def(py::init<>())
      .def(py::init<bool>())
      .def("update", &rd::GLFWViewer::update)
      .def("render", [](rd::GLFWViewer *self,
                        const rd::Simulation &sim) { self->render(sim); })
      .def("get_framebuffer_size", &rd::GLFWViewer::getFramebufferSize)
      .def("should_close", &rd::GLFWViewer::shouldClose)
      .def_readwrite("camera_params", &rd::GLFWViewer::camera_params_)
      .def_readwrite("camera_controller",
                     &rd::GLFWViewer::camera_controller_);
}
