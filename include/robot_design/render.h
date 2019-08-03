#pragma once

#include <array>
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
extern const VertexAttribute ATTRIB_NORMAL;

class Program {
public:
  Program(const std::string &vertex_shader_source,
          const std::string &fragment_shader_source);
  virtual ~Program();
  Program(const Program &other) = delete;
  Program &operator=(const Program &other) = delete;
  void use() const;
  void setProjectionMatrix(const Eigen::Matrix4f &proj_matrix) const;
  void setModelViewMatrix(const Eigen::Matrix4f &model_view_matrix) const;

private:
  GLuint program_;
  GLuint vertex_shader_;
  GLuint fragment_shader_;
  GLint proj_matrix_index_;
  GLint model_view_matrix_index_;
  GLint normal_matrix_index_;
};

class Mesh {
public:
  Mesh(const std::vector<GLfloat> &positions,
       const std::vector<GLfloat> &normals,
       const std::vector<GLint> &indices);
  virtual ~Mesh();
  Mesh(const Mesh &other) = delete;
  Mesh &operator=(const Mesh &other) = delete;
  void bind() const;
  void draw() const;

private:
  GLuint vertex_array_;
  GLuint position_buffer_;
  GLuint normal_buffer_;
  GLuint index_buffer_;
  GLsizei index_count_;
};

class FPSCameraController {
public:
  FPSCameraController(
      const Eigen::Vector3f &position = Eigen::Vector3f::Zero(),
      float yaw = 0.0f, float pitch = 0.0f, float move_speed = 2.0f,
      float mouse_sensitivity = 0.005f)
      : position_(position), yaw_(yaw), pitch_(pitch), move_speed_(move_speed),
        mouse_sensitivity_(mouse_sensitivity), cursor_x_(0), cursor_y_(0),
        last_cursor_x_(0), last_cursor_y_(0), action_flags_(),
        key_bindings_(DEFAULT_KEY_BINDINGS) {}
  void handleKey(int key, int scancode, int action, int mods);
  void handleMouseButton(int button, int action, int mods);
  void handleCursorPosition(double xpos, double ypos);
  void update(float dt);
  void getViewMatrix(Eigen::Matrix4f &view_matrix) const;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

private:
  enum Action {
      ACTION_MOVE_FORWARD, ACTION_MOVE_LEFT, ACTION_MOVE_BACKWARD,
      ACTION_MOVE_RIGHT, ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_PAN_TILT,
      ACTION_COUNT};
  static constexpr std::array<int, ACTION_COUNT> DEFAULT_KEY_BINDINGS = {
      GLFW_KEY_W, GLFW_KEY_A, GLFW_KEY_S, GLFW_KEY_D, GLFW_KEY_Q, GLFW_KEY_E,
      GLFW_MOUSE_BUTTON_LEFT};
  Eigen::Vector3f position_;
  float yaw_, pitch_;
  float move_speed_;
  float mouse_sensitivity_;
  double cursor_x_, cursor_y_;
  double last_cursor_x_, last_cursor_y_;
  std::array<bool, ACTION_COUNT> action_flags_;
  std::array<int, ACTION_COUNT> key_bindings_;
};

class GLFWRenderer {
public:
  GLFWRenderer();
  virtual ~GLFWRenderer();
  GLFWRenderer(const GLFWRenderer &other) = delete;
  GLFWRenderer &operator=(const GLFWRenderer &other) = delete;
  void run(Simulation &sim);
  void render(const Simulation &sim);
  static void windowSizeCallback(GLFWwindow *window, int width, int height);
  static void framebufferSizeCallback(GLFWwindow *window, int width, int height);
  static void keyCallback(GLFWwindow *window, int key, int scancode, int action,
                          int mods);
  static void mouseButtonCallback(GLFWwindow *window, int button, int action,
                                  int mods);
  static void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

private:
  void drawBox(const Eigen::Matrix4f &transform,
               const Eigen::Vector3f &half_extents,
               const Program &program) const;
  void drawCapsule(const Eigen::Matrix4f &transform, float half_length,
                   float radius, const Program &program) const;
  void updateProjectionMatrix();
  static std::string loadString(const std::string &path);
  GLFWwindow *window_;
  float z_near_;
  float z_far_;
  float fov_;
  int window_width_;
  int window_height_;
  FPSCameraController camera_controller_;
  Eigen::Matrix4f proj_matrix_;
  Eigen::Matrix4f view_matrix_;
  std::shared_ptr<Program> default_program_;
  std::shared_ptr<Mesh> box_mesh_;
  std::shared_ptr<Mesh> capsule_end_mesh_;
  std::shared_ptr<Mesh> capsule_middle_mesh_;
};

void makePerspectiveProjection(float aspect_ratio, float z_near, float z_far,
                               float fov, Eigen::Matrix4f &matrix);

std::shared_ptr<Mesh> makeBoxMesh();

std::shared_ptr<Mesh> makeCapsuleEndMesh(int n_segments, int n_rings);

std::shared_ptr<Mesh> makeCapsuleMiddleMesh(int n_segments);

}  // namespace robot_design
