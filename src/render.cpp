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
  // Load all available extensions even if they are not in the extensions string
  glewExperimental = GL_TRUE;
  glewInit();

  glfwSetKeyCallback(window_, keyCallback);
}

GLFWRenderer::~GLFWRenderer() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void GLFWRenderer::run(Simulation &sim) {
  double last_time = glfwGetTime();
  while (!glfwWindowShouldClose(window_)) {
    double current_time = glfwGetTime();
    sim.step(current_time - last_time);
    last_time = current_time;

    glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    for (Index i = 0; i < sim.getRobotCount(); ++i) {
      renderRobot(*sim.getRobot(i), sim);
    }

    glfwSwapBuffers(window_);
    glfwPollEvents();
  }
}

void GLFWRenderer::renderRobot(const Robot &robot, const Simulation &sim) {
  for (const auto &link : robot.links_) {
  }
}

void GLFWRenderer::keyCallback(GLFWwindow *window, int key, int scancode,
    int action, int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GL_TRUE);
  }
}

}  // namespace robot_design
