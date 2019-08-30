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

struct DirectionalLight;

struct Program {
  Program(const std::string &vertex_shader_source,
          const std::string &fragment_shader_source);
  virtual ~Program();
  Program(const Program &other) = delete;
  Program &operator=(const Program &other) = delete;
  void use() const;
  void setProjectionMatrix(const Eigen::Matrix4f &proj_matrix) const;
  void setModelViewMatrices(const Eigen::Matrix4f &model_matrix,
                            const Eigen::Matrix4f &view_matrix,
                            const Eigen::Matrix4f &light_view_matrix) const;
  void setObjectColor(const Eigen::Vector3f &object_color) const;
  void setDirectionalLight(const DirectionalLight &dir_light) const;

  GLuint program_;
  GLuint vertex_shader_;
  GLuint fragment_shader_;
  GLint proj_matrix_index_;
  GLint view_matrix_index_;
  GLint model_view_matrix_index_;
  GLint normal_matrix_index_;
  GLint object_color_index_;
  GLint world_light_dir_index_;
  GLint light_proj_matrix_index_;
  GLint light_model_view_matrix_index_;
  GLint light_color_index_;
  GLint shadow_map_index_;
};

struct Mesh {
  Mesh(const std::vector<GLfloat> &positions,
       const std::vector<GLfloat> &normals,
       const std::vector<GLint> &indices);
  virtual ~Mesh();
  Mesh(const Mesh &other) = delete;
  Mesh &operator=(const Mesh &other) = delete;
  void bind() const;
  void draw() const;

  GLuint vertex_array_;
  GLuint position_buffer_;
  GLuint normal_buffer_;
  GLuint index_buffer_;
  GLsizei index_count_;
};

struct Texture2D {
  Texture2D(GLenum target, GLint level, GLint internal_format, GLsizei width,
            GLsizei height, GLenum format, GLenum type,
            const GLvoid *data = NULL);
  virtual ~Texture2D();
  Texture2D(const Texture2D &other) = delete;
  Texture2D &operator=(const Texture2D &other) = delete;
  void bind() const;

  GLenum target_;
  GLuint texture_;
};

struct Framebuffer {
  Framebuffer(const Texture2D *color_texture, const Texture2D *depth_texture);
  virtual ~Framebuffer();
  Framebuffer(const Framebuffer &other) = delete;
  Framebuffer &operator=(const Framebuffer &other) = delete;
  void bind() const;

  GLuint framebuffer_;
};

class FPSCameraController {
public:
  FPSCameraController(
      const Eigen::Vector3f &position = Eigen::Vector3f::Zero(),
      float yaw = 0.0f, float pitch = 0.0f, float distance = 1.0f,
      float move_speed = 2.0f, float mouse_sensitivity = 0.005f,
      float scroll_sensitivity = 0.1f)
      : position_(position), yaw_(yaw), pitch_(pitch), distance_(distance),
        move_speed_(move_speed), mouse_sensitivity_(mouse_sensitivity),
        scroll_sensitivity_(scroll_sensitivity), cursor_x_(0), cursor_y_(0),
        last_cursor_x_(0), last_cursor_y_(0), action_flags_(),
        key_bindings_(DEFAULT_KEY_BINDINGS) {}
  void handleKey(int key, int scancode, int action, int mods);
  void handleMouseButton(int button, int action, int mods);
  void handleCursorPosition(double xpos, double ypos);
  void handleScroll(double xoffset, double yoffset);
  void update(double dt);
  void getViewMatrix(Eigen::Matrix4f &view_matrix) const;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Eigen::Vector3f position_;
  float yaw_, pitch_;
  float distance_;
  float move_speed_;
  float mouse_sensitivity_;
  float scroll_sensitivity_;

private:
  enum Action {
      ACTION_MOVE_FORWARD, ACTION_MOVE_LEFT, ACTION_MOVE_BACKWARD,
      ACTION_MOVE_RIGHT, ACTION_MOVE_UP, ACTION_MOVE_DOWN, ACTION_PAN_TILT,
      ACTION_COUNT};
  static const std::array<int, ACTION_COUNT> DEFAULT_KEY_BINDINGS;
  double cursor_x_, cursor_y_;
  double last_cursor_x_, last_cursor_y_;
  std::array<bool, ACTION_COUNT> action_flags_;
  std::array<int, ACTION_COUNT> key_bindings_;
};

struct DirectionalLight {
  DirectionalLight(
      const Eigen::Vector3f &color, const Eigen::Vector3f &pos,
      const Eigen::Vector3f &dir, const Eigen::Vector3f &up, GLsizei sm_width,
      GLsizei sm_height);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Eigen::Vector3f color_;
  Eigen::Vector3f pos_;
  Eigen::Vector3f dir_;
  GLsizei sm_width_;
  GLsizei sm_height_;
  Eigen::Matrix4f proj_matrix_;
  Eigen::Matrix4f view_matrix_;
  std::shared_ptr<Texture2D> sm_depth_texture_;
  std::shared_ptr<Framebuffer> sm_framebuffer_;
};

class GLFWRenderer {
public:
  GLFWRenderer();
  virtual ~GLFWRenderer();
  GLFWRenderer(const GLFWRenderer &other) = delete;
  GLFWRenderer &operator=(const GLFWRenderer &other) = delete;
  void update(double dt);
  void render(const Simulation &sim);
  bool shouldClose() const;
  static void windowSizeCallback(GLFWwindow *window, int width, int height);
  static void framebufferSizeCallback(GLFWwindow *window, int width, int height);
  static void keyCallback(GLFWwindow *window, int key, int scancode, int action,
                          int mods);
  static void mouseButtonCallback(GLFWwindow *window, int button, int action,
                                  int mods);
  static void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos);
  static void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

private:
  void draw(const Simulation &sim, const Program &program,
            const Eigen::Matrix4f &view_matrix) const;
  void drawBox(const Eigen::Matrix4f &transform,
               const Eigen::Vector3f &half_extents,
               const Program &program,
               const Eigen::Matrix4f &view_matrix) const;
  void drawCapsule(const Eigen::Matrix4f &transform, float half_length,
                   float radius, const Program &program,
                   const Eigen::Matrix4f &view_matrix) const;
  void updateProjectionMatrix();
  static std::string loadString(const std::string &path);
  GLFWwindow *window_;
  float z_near_;
  float z_far_;
  float fov_;
  int framebuffer_width_;
  int framebuffer_height_;
  FPSCameraController camera_controller_;
  Eigen::Matrix4f proj_matrix_;
  std::shared_ptr<Program> default_program_;
  std::shared_ptr<Program> depth_program_;
  std::shared_ptr<Mesh> box_mesh_;
  std::shared_ptr<Mesh> capsule_end_mesh_;
  std::shared_ptr<Mesh> capsule_middle_mesh_;
  std::shared_ptr<DirectionalLight> dir_light_;
};

void makeOrthographicProjection(float aspect_ratio, float z_near, float z_far,
                                Eigen::Matrix4f &matrix);

void makePerspectiveProjection(float aspect_ratio, float z_near, float z_far,
                               float fov, Eigen::Matrix4f &matrix);

std::shared_ptr<Mesh> makeBoxMesh();

std::shared_ptr<Mesh> makeCapsuleEndMesh(int n_segments, int n_rings);

std::shared_ptr<Mesh> makeCapsuleMiddleMesh(int n_segments);

}  // namespace robot_design
