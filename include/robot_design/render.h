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
  void setProjectionMatrix(const Eigen::Matrix4f &proj_matrix) const;
  void setViewMatrix(const Eigen::Matrix4f &view_matrix) const;
  void setModelMatrix(const Eigen::Matrix4f &model_matrix) const;

private:
  GLuint program_;
  GLuint vertex_shader_;
  GLuint fragment_shader_;
  GLint proj_matrix_index_;
  GLint view_matrix_index_;
  GLint model_matrix_index_;
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
  static void windowSizeCallback(GLFWwindow *window, int width, int height);
  static void framebufferSizeCallback(GLFWwindow *window, int width, int height);
  static void keyCallback(GLFWwindow *window, int key, int scancode, int action,
                          int mods);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

private:
  void updateProjectionMatrix();
  static std::string loadString(const std::string &path);
  float z_near_;
  float z_far_;
  float fov_;
  int window_width_;
  int window_height_;
  Eigen::Matrix4f proj_matrix_;
  Eigen::Matrix4f view_matrix_;
  GLFWwindow *window_;
  std::shared_ptr<Program> default_program_;
};

void makePerspectiveProjection(float aspect_ratio, float z_near, float z_far,
                               float fov, Eigen::Matrix4f &matrix);

}  // namespace robot_design
