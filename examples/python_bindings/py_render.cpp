#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <robot_design/render.h>

namespace py = pybind11;
namespace rd = robot_design;

void initRender(py::module &m) {
  py::class_<rd::FPSCameraController>(m, "FPSCameraController")
      .def(py::init<>())
      .def(py::init<const Eigen::Vector3f &, float, float, float, float, float,
                    float>())
      .def("update", &rd::FPSCameraController::update)
      .def("get_view_matrix", &rd::FPSCameraController::getViewMatrix)
      .def_readwrite("position", &rd::FPSCameraController::position_)
      .def_readwrite("yaw", &rd::FPSCameraController::yaw_)
      .def_readwrite("pitch", &rd::FPSCameraController::pitch_)
      .def_readwrite("distance", &rd::FPSCameraController::distance_)
      .def_readwrite("move_speed", &rd::FPSCameraController::move_speed_)
      .def_readwrite("mouse_sensitivity",
                     &rd::FPSCameraController::mouse_sensitivity_)
      .def_readwrite("scroll_sensitivity",
                     &rd::FPSCameraController::scroll_sensitivity_);

  py::class_<rd::GLFWRenderer>(m, "GLFWRenderer")
      .def(py::init<>())
      .def(py::init<bool>())
      .def("update", &rd::GLFWRenderer::update)
      .def("render", [](rd::GLFWRenderer *self,
                        const rd::Simulation &sim) { self->render(sim); })
      .def("get_framebuffer_size", &rd::GLFWRenderer::getFramebufferSize)
      .def("should_close", &rd::GLFWRenderer::shouldClose)
      .def_readwrite("camera_controller",
                     &rd::GLFWRenderer::camera_controller_);
}
