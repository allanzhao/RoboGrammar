#include <robot_design/render.h>

namespace robot_design {

GLFWRenderer::GLFWRenderer() {
  if (!glfwInit()) {
    return;
  }

  window_ = glfwCreateWindow(640, 480, "GLFW Renderer", NULL, NULL);
  if (!window_) {
    return;
  }

  glfwMakeContextCurrent(window_);
  glfwSetKeyCallback(window_, keyCallback);
}

GLFWRenderer::~GLFWRenderer() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void GLFWRenderer::render(Simulation &sim) {
  glClearColor(0.5, 0.8, 0.8, 1.0);
  while (!glfwWindowShouldClose(window_)) {
    glClear(GL_COLOR_BUFFER_BIT);
    glfwSwapBuffers(window_);
    glfwPollEvents();
  }
}

void GLFWRenderer::keyCallback(GLFWwindow *window, int key, int scancode,
    int action, int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, true);
  }
}

}  // namespace robot_design
