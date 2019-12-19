#include <cmath>
#include <iostream>
#include <robot_design/render.h>
#include <stdexcept>

namespace robot_design {

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
  renderer_ = std::make_shared<GLRenderer>();

  // Set up callbacks
  // Allow accessing "this" from static callbacks
  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, framebufferSizeCallback);
  glfwSetKeyCallback(window_, keyCallback);
  glfwSetMouseButtonCallback(window_, mouseButtonCallback);
  glfwSetCursorPosCallback(window_, cursorPositionCallback);
  glfwSetScrollCallback(window_, scrollCallback);

  // Get initial framebuffer size
  glfwGetFramebufferSize(window_, &framebuffer_width_, &framebuffer_height_);

  // Set default camera parameters
  camera_params_.pitch_ = -M_PI / 6;
  camera_params_.distance_ = 2.0;
}

GLFWViewer::~GLFWViewer() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void GLFWViewer::update(double dt) {
  camera_controller_.update(camera_params_, dt);
}

void GLFWViewer::render(const Simulation &sim, int width, int height,
                        const Framebuffer *target_framebuffer) {
  if (width < 0 || height < 0) {
    // Use default framebuffer size
    width = framebuffer_width_;
    height = framebuffer_height_;
  }
  float aspect_ratio = static_cast<float>(width) / height;
  camera_params_.aspect_ratio_ = aspect_ratio;

  renderer_->render(sim, camera_params_, width, height, target_framebuffer);

  glfwSwapBuffers(window_);
  glfwPollEvents();
}

void GLFWViewer::getFramebufferSize(int &width, int &height) const {
  width = framebuffer_width_;
  height = framebuffer_height_;
}

bool GLFWViewer::shouldClose() const { return glfwWindowShouldClose(window_); }

void GLFWViewer::errorCallback(int error, const char *description) {
  std::cerr << "GLFW error: " << description << std::endl;
}

void GLFWViewer::framebufferSizeCallback(GLFWwindow *window, int width,
                                         int height) {
  GLFWViewer *viewer =
      static_cast<GLFWViewer *>(glfwGetWindowUserPointer(window));
  viewer->framebuffer_width_ = width;
  viewer->framebuffer_height_ = height;
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
