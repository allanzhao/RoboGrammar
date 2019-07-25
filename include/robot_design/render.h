#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <robot_design/sim.h>
#include <robot_design/utils.h>
#include <typeindex>

namespace robot_design {

struct Visual {
  virtual std::size_t hash() const = 0;
};

struct CapsuleVisual : public Visual {
  virtual std::size_t hash() const {
    return hashCombine(typeid(CapsuleVisual).hash_code(), radius_, length_);
  }
  float radius_;
  float length_;
};

class GLFWRenderer {
public:
  GLFWRenderer();
  virtual ~GLFWRenderer();
  GLFWRenderer(const GLFWRenderer &other) = delete;
  GLFWRenderer &operator=(const GLFWRenderer &other) = delete;
  void run(Simulation &sim);
  void renderRobot(const Robot &robot, const Simulation &sim);
  static void keyCallback(GLFWwindow *window, int key, int scancode, int action,
                          int mods);

private:
  GLFWwindow *window_;
};

}  // namespace robot_design

// Hash function specialization for visuals
namespace std {
  template <>
  struct hash<robot_design::Visual> {
    std::size_t operator()(const robot_design::Visual &visual) const {
      return visual.hash();
    }
  };
}  // namespace std
