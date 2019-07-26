#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <robot_design/sim.h>

namespace robot_design {

struct VertexAttribute {
  VertexAttribute(GLuint index, const std::string &name)
      : index_(index), name_(name) {}
  GLuint index_;
  std::string name_;
};

extern const VertexAttribute ATTRIB_POSITION;

class Program {
public:
  Program(const std::string &vertex_shader_source,
          const std::string &fragment_shader_source);
  virtual ~Program();
  Program(const Program &other) = delete;
  Program &operator=(const Program &other) = delete;
  void use() const;

private:
  GLuint program_;
  GLuint vertex_shader_;
  GLuint fragment_shader_;
};

class Mesh {
public:
  Mesh(const std::vector<GLfloat> &positions,
       const std::vector<GLint> &indices);
  virtual ~Mesh();
  Mesh(const Mesh &other) = delete;
  Mesh &operator=(const Mesh &other) = delete;
  void bind() const;

private:
  GLuint vertex_array_;
  GLuint position_buffer_;
  GLuint index_buffer_;
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
  static std::string loadString(const std::string &path);
  GLFWwindow *window_;
  std::shared_ptr<Program> default_program_;
};

}  // namespace robot_design
