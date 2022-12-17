#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <array>
#include <memory>
#include <robot_design/render.h>

namespace robot_design {

class FPSCameraController {
public:
  FPSCameraController(float move_speed = 2.0f, float mouse_sensitivity = 0.005f,
                      float scroll_sensitivity = 0.1f)
      : move_speed_(move_speed), mouse_sensitivity_(mouse_sensitivity),
        scroll_sensitivity_(scroll_sensitivity), cursor_x_(0), cursor_y_(0),
        last_cursor_x_(0), last_cursor_y_(0), scroll_y_offset_(0),
        action_flags_(), key_bindings_(DEFAULT_KEY_BINDINGS) {}
  void handleKey(int key, int scancode, int action, int mods);
  void handleMouseButton(int button, int action, int mods);
  void handleCursorPosition(double xpos, double ypos);
  void handleScroll(double xoffset, double yoffset);
  void update(CameraParameters &camera_params, double dt);
  bool shouldRecord() const;

  float move_speed_;
  float mouse_sensitivity_;
  float scroll_sensitivity_;

private:
  enum Action {
    ACTION_MOVE_FORWARD,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_BACKWARD,
    ACTION_MOVE_RIGHT,
    ACTION_MOVE_UP,
    ACTION_MOVE_DOWN,
    ACTION_PAN_TILT,
    ACTION_RECORD,
    ACTION_COUNT
  };
  static const std::array<int, ACTION_COUNT> DEFAULT_KEY_BINDINGS;
  double cursor_x_, cursor_y_;
  double last_cursor_x_, last_cursor_y_;
  double scroll_y_offset_;
  std::array<bool, ACTION_COUNT> action_flags_;
  std::array<int, ACTION_COUNT> key_bindings_;
};

class GLFWViewer : public Viewer {
public:
  explicit GLFWViewer(bool hidden = false);
  virtual ~GLFWViewer();
  GLFWViewer(const GLFWViewer &other) = delete;
  GLFWViewer &operator=(const GLFWViewer &other) = delete;
  virtual void update(double dt) override;
  virtual void render(const Simulation &sim,
                      unsigned char *pixels = nullptr) override;
  virtual void getFramebufferSize(int &width, int &height) const override;
  virtual void setFramebufferSize(int width, int height) override;
  bool shouldClose() const;
  static void errorCallback(int error, const char *description);
  static void keyCallback(GLFWwindow *window, int key, int scancode, int action,
                          int mods);
  static void mouseButtonCallback(GLFWwindow *window, int button, int action,
                                  int mods);
  static void cursorPositionCallback(GLFWwindow *window, double xpos,
                                     double ypos);
  static void scrollCallback(GLFWwindow *window, double xoffset,
                             double yoffset);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  CameraParameters camera_params_;
  RenderParameters render_params_;
  FPSCameraController camera_controller_;

private:
  GLFWwindow *window_ = nullptr;
  std::shared_ptr<GLRenderer> renderer_;
};

} // namespace robot_design
