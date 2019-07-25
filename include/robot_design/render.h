#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <robot_design/sim.h>

namespace robot_design {

class GLFWRenderer {
public:
  GLFWRenderer();
  virtual ~GLFWRenderer();
  GLFWRenderer(const GLFWRenderer &other) = delete;
  GLFWRenderer &operator=(const GLFWRenderer &other) = delete;
  void render(Simulation &sim);
  static void keyCallback(GLFWwindow *window, int key, int scancode, int action,
                          int mods);

private:
  GLFWwindow *window_;
};

}  // namespace robot_design
