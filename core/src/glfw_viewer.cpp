#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <robot_design/glfw_viewer.h>
#include <robot_design/render.h>
#include <stdexcept>

namespace robot_design {

const std::array<int, FPSCameraController::ACTION_COUNT>
    FPSCameraController::DEFAULT_KEY_BINDINGS = {GLFW_KEY_W,
                                                 GLFW_KEY_A,
                                                 GLFW_KEY_S,
                                                 GLFW_KEY_D,
                                                 GLFW_KEY_Q,
                                                 GLFW_KEY_E,
                                                 GLFW_MOUSE_BUTTON_LEFT,
                                                 GLFW_KEY_SPACE};

void FPSCameraController::handleKey(int key, int scancode, int action,
                                    int mods) {
  for (int i = 0; i < ACTION_COUNT; ++i) {
    if (key == key_bindings_[i]) {
      action_flags_[i] = (action != GLFW_RELEASE);
    }
  }
}

void FPSCameraController::handleMouseButton(int button, int action, int mods) {
  // Handle mouse buttons the same way as keys (no conflicting codes)
  for (int i = 0; i < ACTION_COUNT; ++i) {
    if (button == key_bindings_[i]) {
      action_flags_[i] = (action != GLFW_RELEASE);
    }
  }
}

void FPSCameraController::handleCursorPosition(double xpos, double ypos) {
  cursor_x_ = xpos;
  cursor_y_ = ypos;
}

void FPSCameraController::handleScroll(double xoffset, double yoffset) {
  scroll_y_offset_ += yoffset;
}

void FPSCameraController::update(CameraParameters &camera_params, double dt) {
  Eigen::Vector3f offset = Eigen::Vector3f::Zero();
  float pan = 0.0f;
  float tilt = 0.0f;

  if (action_flags_[ACTION_MOVE_FORWARD]) {
    offset(2) -= move_speed_ * dt;
  }
  if (action_flags_[ACTION_MOVE_LEFT]) {
    offset(0) -= move_speed_ * dt;
  }
  if (action_flags_[ACTION_MOVE_BACKWARD]) {
    offset(2) += move_speed_ * dt;
  }
  if (action_flags_[ACTION_MOVE_RIGHT]) {
    offset(0) += move_speed_ * dt;
  }
  if (action_flags_[ACTION_MOVE_UP]) {
    offset(1) += move_speed_ * dt;
  }
  if (action_flags_[ACTION_MOVE_DOWN]) {
    offset(1) -= move_speed_ * dt;
  }
  if (action_flags_[ACTION_PAN_TILT]) {
    pan -= (cursor_x_ - last_cursor_x_) * mouse_sensitivity_;
    tilt -= (cursor_y_ - last_cursor_y_) * mouse_sensitivity_;
  }

  camera_params.position_ +=
      Eigen::AngleAxisf(camera_params.yaw_, Eigen::Vector3f::UnitY()) * offset;
  camera_params.yaw_ += pan;
  camera_params.pitch_ += tilt;
  camera_params.distance_ *=
      std::pow(1.0 - scroll_sensitivity_, scroll_y_offset_);
  last_cursor_x_ = cursor_x_;
  last_cursor_y_ = cursor_y_;
  scroll_y_offset_ = 0;
}

bool FPSCameraController::shouldRecord() const {
  return action_flags_[ACTION_RECORD];
}

GLFWViewer::GLFWViewer(bool hidden) {
  glfwSetErrorCallback(errorCallback);

  if (!glfwInit()) {
    throw std::runtime_error("Could not initialize GLFW");
  }

  // Require OpenGL 3.2 or higher
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  // Enable 4x MSAA
  glfwWindowHint(GLFW_SAMPLES, 4);
  if (hidden) {
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
  }
  window_ = glfwCreateWindow(1280, 720, "Robot Design Viewer", NULL, NULL);
  if (!window_) {
    throw std::runtime_error("Could not create GLFW window");
  }

  glfwMakeContextCurrent(window_);

  // Load all available extensions even if they are not in the extensions string
  glewExperimental = GL_TRUE;
  glewInit();

  // Create renderer (holder for OpenGL resources)
  const char *data_dir = std::getenv("ROBOT_DESIGN_DATA_DIR");
  if (!data_dir) {
    // Data directory was not provided, use the default
    data_dir = "data/";
  }
  renderer_ = std::make_shared<GLRenderer>(data_dir);

  // Set up callbacks
  // Allow accessing "this" from static callbacks
  glfwSetWindowUserPointer(window_, this);
  glfwSetKeyCallback(window_, keyCallback);
  glfwSetMouseButtonCallback(window_, mouseButtonCallback);
  glfwSetCursorPosCallback(window_, cursorPositionCallback);
  glfwSetScrollCallback(window_, scrollCallback);
}

GLFWViewer::~GLFWViewer() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void GLFWViewer::update(double dt) {
  camera_controller_.update(camera_params_, dt);
}

void GLFWViewer::render(const Simulation &sim, unsigned char *pixels) {
  int width, height;
  getFramebufferSize(width, height);
  float aspect_ratio = static_cast<float>(width) / height;
  camera_params_.aspect_ratio_ = aspect_ratio;

  renderer_->render(sim, camera_params_, render_params_, width, height);

  if (pixels) {
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
  } else {
    glfwSwapBuffers(window_);
    glfwPollEvents();
  }
}

void GLFWViewer::getFramebufferSize(int &width, int &height) const {
  glfwGetFramebufferSize(window_, &width, &height);
}

void GLFWViewer::setFramebufferSize(int width, int height) {
  glfwSetWindowSize(window_, width, height);
}

bool GLFWViewer::shouldClose() const { return glfwWindowShouldClose(window_); }

void GLFWViewer::errorCallback(int error, const char *description) {
  std::cerr << "GLFW error: " << description << std::endl;
}

void GLFWViewer::keyCallback(GLFWwindow *window, int key, int scancode,
                             int action, int mods) {
  GLFWViewer *viewer =
      static_cast<GLFWViewer *>(glfwGetWindowUserPointer(window));
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GL_TRUE);
  }
  viewer->camera_controller_.handleKey(key, scancode, action, mods);
}

void GLFWViewer::mouseButtonCallback(GLFWwindow *window, int button, int action,
                                     int mods) {
  GLFWViewer *viewer =
      static_cast<GLFWViewer *>(glfwGetWindowUserPointer(window));
  viewer->camera_controller_.handleMouseButton(button, action, mods);
}

void GLFWViewer::cursorPositionCallback(GLFWwindow *window, double xpos,
                                        double ypos) {
  GLFWViewer *viewer =
      static_cast<GLFWViewer *>(glfwGetWindowUserPointer(window));
  viewer->camera_controller_.handleCursorPosition(xpos, ypos);
}

void GLFWViewer::scrollCallback(GLFWwindow *window, double xoffset,
                                double yoffset) {
  GLFWViewer *viewer =
      static_cast<GLFWViewer *>(glfwGetWindowUserPointer(window));
  viewer->camera_controller_.handleScroll(xoffset, yoffset);
}

} // namespace robot_design
