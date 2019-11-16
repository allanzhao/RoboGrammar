#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <lodepng.h>
#include <robot_design/render.h>
#include <robot_design/utils.h>
#include <sstream>
#include <stdexcept>
#include <string>

namespace robot_design {

const VertexAttribute ATTRIB_POSITION(0, "model_position");
const VertexAttribute ATTRIB_NORMAL(1, "model_normal");
const VertexAttribute ATTRIB_TEXCOORD(2, "model_texcoord");

Program::Program(const std::string &vertex_shader_source,
                 const std::string &fragment_shader_source)
    : program_(0), vertex_shader_(0), fragment_shader_(0) {
  // Create vertex shader
  vertex_shader_ = glCreateShader(GL_VERTEX_SHADER);
  const GLchar *vertex_shader_source_ptr = vertex_shader_source.c_str();
  glShaderSource(vertex_shader_, 1, &vertex_shader_source_ptr, NULL);
  glCompileShader(vertex_shader_);
  // Check for compile errors
  GLint status;
  glGetShaderiv(vertex_shader_, GL_COMPILE_STATUS, &status);
  if (!status) {
    char buffer[512];
    glGetShaderInfoLog(vertex_shader_, sizeof(buffer), NULL, buffer);
    throw std::runtime_error(std::string("Failed to compile vertex shader: ") +
                             buffer);
  }

  // Create fragment shader
  // Fragment shader is optional (source may be an empty string)
  if (!fragment_shader_source.empty()) {
    fragment_shader_ = glCreateShader(GL_FRAGMENT_SHADER);
    const GLchar *fragment_shader_source_ptr = fragment_shader_source.c_str();
    glShaderSource(fragment_shader_, 1, &fragment_shader_source_ptr, NULL);
    glCompileShader(fragment_shader_);
    // Check for compile errors
    glGetShaderiv(fragment_shader_, GL_COMPILE_STATUS, &status);
    if (!status) {
      char buffer[512];
      glGetShaderInfoLog(fragment_shader_, sizeof(buffer), NULL, buffer);
      throw std::runtime_error(
          std::string("Failed to compile fragment shader: ") + buffer);
    }
  }

  // Create program
  program_ = glCreateProgram();
  glAttachShader(program_, vertex_shader_);
  if (fragment_shader_) {
    glAttachShader(program_, fragment_shader_);
  }

  // Define fixed attribute indices
  glBindAttribLocation(program_, ATTRIB_POSITION.index_,
                       ATTRIB_POSITION.name_.c_str());
  glBindAttribLocation(program_, ATTRIB_NORMAL.index_,
                       ATTRIB_NORMAL.name_.c_str());
  glBindAttribLocation(program_, ATTRIB_TEXCOORD.index_,
                       ATTRIB_TEXCOORD.name_.c_str());

  glLinkProgram(program_);
  // Check for link errors
  glGetProgramiv(program_, GL_LINK_STATUS, &status);
  if (!status) {
    char buffer[512];
    glGetProgramInfoLog(program_, sizeof(buffer), NULL, buffer);
    throw std::runtime_error(std::string("Failed to link shader program: ") +
                             buffer);
  }

  // Find uniform indices
  proj_matrix_index_ = glGetUniformLocation(program_, "proj_matrix");
  view_matrix_index_ = glGetUniformLocation(program_, "view_matrix");
  model_view_matrix_index_ =
      glGetUniformLocation(program_, "model_view_matrix");
  normal_matrix_index_ = glGetUniformLocation(program_, "normal_matrix");
  object_color_index_ = glGetUniformLocation(program_, "object_color");
  world_light_dir_index_ = glGetUniformLocation(program_, "world_light_dir");
  light_proj_matrix_index_ =
      glGetUniformLocation(program_, "light_proj_matrix");
  light_model_view_matrices_index_ =
      glGetUniformLocation(program_, "light_model_view_matrices");
  light_color_index_ = glGetUniformLocation(program_, "light_color");
  shadow_map_index_ = glGetUniformLocation(program_, "shadow_map");
  cascade_far_splits_index_ =
      glGetUniformLocation(program_, "cascade_far_splits");
}

Program::~Program() {
  glDetachShader(program_, fragment_shader_);
  glDetachShader(program_, vertex_shader_);
  glDeleteProgram(program_);
  glDeleteShader(fragment_shader_);
  glDeleteShader(vertex_shader_);
}

Mesh::Mesh(const std::vector<GLfloat> &positions,
           const std::vector<GLfloat> &normals,
           const std::vector<GLint> &indices)
    : vertex_array_(0), position_buffer_(0), normal_buffer_(0),
      index_buffer_(0), index_count_(indices.size()) {
  // Create vertex array object (VAO)
  glGenVertexArrays(1, &vertex_array_);
  glBindVertexArray(vertex_array_);

  // Create vertex position buffer
  glGenBuffers(1, &position_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, position_buffer_);
  glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(positions[0]),
               positions.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(ATTRIB_POSITION.index_, 3, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(ATTRIB_POSITION.index_);

  // Create vertex normal buffer
  glGenBuffers(1, &normal_buffer_);
  glBindBuffer(GL_ARRAY_BUFFER, normal_buffer_);
  glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(normals[0]),
               normals.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(ATTRIB_NORMAL.index_, 3, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(ATTRIB_NORMAL.index_);

  // Create index buffer
  glGenBuffers(1, &index_buffer_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(indices[0]),
               indices.data(), GL_STATIC_DRAW);
}

Mesh::~Mesh() {
  glDeleteBuffers(1, &index_buffer_);
  glDeleteBuffers(1, &normal_buffer_);
  glDeleteBuffers(1, &position_buffer_);
  glDeleteVertexArrays(1, &vertex_array_);
}

Texture2D::Texture2D(GLenum target, GLint level, GLint internal_format,
                     GLsizei width, GLsizei height, GLenum format, GLenum type,
                     const GLvoid *data)
    : target_(target), texture_(0) {
  glGenTextures(1, &texture_);
  glBindTexture(target, texture_);
  glTexImage2D(target, level, internal_format, width, height, 0, format, type,
               data);
  glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
}

Texture2D::~Texture2D() { glDeleteTextures(1, &texture_); }

void Texture2D::getImage(unsigned char *pixels) const {
  bind();
  glGetTexImage(target_, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
}

Texture3D::Texture3D(GLenum target, GLint level, GLint internal_format,
                     GLsizei width, GLsizei height, GLsizei depth,
                     GLenum format, GLenum type, const GLvoid *data)
    : target_(target), texture_(0) {
  glGenTextures(1, &texture_);
  glBindTexture(target, texture_);
  glTexImage3D(target, level, internal_format, width, height, depth, 0, format,
               type, data);
  glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
  glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
  glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
}

Texture3D::~Texture3D() { glDeleteTextures(1, &texture_); }

void Texture3D::getImage(unsigned char *pixels) const {
  bind();
  glGetTexImage(target_, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
}

Framebuffer::Framebuffer() : framebuffer_(0) {
  glGenFramebuffers(1, &framebuffer_);
  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_);
  // Necessary to make framebuffer complete without a color attachment
  glDrawBuffer(GL_NONE);
  glReadBuffer(GL_NONE);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

Framebuffer::~Framebuffer() { glDeleteFramebuffers(1, &framebuffer_); }

void Framebuffer::attachColorTexture(const Texture2D &color_texture) const {
  bind();
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                         color_texture.target_, color_texture.texture_, 0);
  glDrawBuffer(GL_COLOR_ATTACHMENT0);
  glReadBuffer(GL_COLOR_ATTACHMENT0);
}

void Framebuffer::attachColorTextureLayer(const Texture3D &color_texture,
                                          GLint layer) const {
  bind();
  glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            color_texture.texture_, 0, layer);
  glDrawBuffer(GL_COLOR_ATTACHMENT0);
  glReadBuffer(GL_COLOR_ATTACHMENT0);
}

void Framebuffer::attachDepthTexture(const Texture2D &depth_texture) const {
  bind();
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                         depth_texture.target_, depth_texture.texture_, 0);
}

void Framebuffer::attachDepthTextureLayer(const Texture3D &depth_texture,
                                          GLint layer) const {
  bind();
  glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            depth_texture.texture_, 0, layer);
}

const std::array<int, FPSCameraController::ACTION_COUNT>
    FPSCameraController::DEFAULT_KEY_BINDINGS = {GLFW_KEY_W,
                                                 GLFW_KEY_A,
                                                 GLFW_KEY_S,
                                                 GLFW_KEY_D,
                                                 GLFW_KEY_Q,
                                                 GLFW_KEY_E,
                                                 GLFW_MOUSE_BUTTON_LEFT};

void FPSCameraController::handleKey(int key, int scancode, int action,
                                    int mods) {
  for (int i = 0; i < ACTION_COUNT; ++i) {
    if (key == key_bindings_[i]) {
      action_flags_[i] = (action != GLFW_RELEASE);
    }
  }
}

void FPSCameraController::handleMouseButton(int button, int action, int mods) {
  // Handle mouse buttons the same way as keys (no conflicting codes)
  for (int i = 0; i < ACTION_COUNT; ++i) {
    if (button == key_bindings_[i]) {
      action_flags_[i] = (action != GLFW_RELEASE);
    }
  }
}

void FPSCameraController::handleCursorPosition(double xpos, double ypos) {
  cursor_x_ = xpos;
  cursor_y_ = ypos;
}

void FPSCameraController::handleScroll(double xoffset, double yoffset) {
  distance_ *= std::pow(1.0 - scroll_sensitivity_, yoffset);
}

void FPSCameraController::update(double dt) {
  Eigen::Vector3f offset = Eigen::Vector3f::Zero();
  float pan = 0.0f;
  float tilt = 0.0f;

  if (action_flags_[ACTION_MOVE_FORWARD]) {
    offset(2) -= move_speed_ * dt;
  }
  if (action_flags_[ACTION_MOVE_LEFT]) {
    offset(0) -= move_speed_ * dt;
  }
  if (action_flags_[ACTION_MOVE_BACKWARD]) {
    offset(2) += move_speed_ * dt;
  }
  if (action_flags_[ACTION_MOVE_RIGHT]) {
    offset(0) += move_speed_ * dt;
  }
  if (action_flags_[ACTION_MOVE_UP]) {
    offset(1) += move_speed_ * dt;
  }
  if (action_flags_[ACTION_MOVE_DOWN]) {
    offset(1) -= move_speed_ * dt;
  }
  if (action_flags_[ACTION_PAN_TILT]) {
    pan -= (cursor_x_ - last_cursor_x_) * mouse_sensitivity_;
    tilt -= (cursor_y_ - last_cursor_y_) * mouse_sensitivity_;
  }

  position_ += Eigen::AngleAxisf(yaw_, Eigen::Vector3f::UnitY()) * offset;
  yaw_ += pan;
  pitch_ += tilt;
  last_cursor_x_ = cursor_x_;
  last_cursor_y_ = cursor_y_;
}

void FPSCameraController::getViewMatrix(Eigen::Matrix4f &view_matrix) const {
  Eigen::Affine3f view_transform(
      Eigen::Translation3f(0.0f, 0.0f, -distance_) *
      Eigen::AngleAxisf(-pitch_, Eigen::Vector3f::UnitX()) *
      Eigen::AngleAxisf(-yaw_, Eigen::Vector3f::UnitY()) *
      Eigen::Translation3f(-position_));
  view_matrix = view_transform.matrix();
}

DirectionalLight::DirectionalLight(const Eigen::Vector3f &color,
                                   const Eigen::Vector3f &dir,
                                   const Eigen::Vector3f &up, GLsizei sm_width,
                                   GLsizei sm_height, int sm_cascade_count)
    : color_(color), dir_(dir.normalized()), sm_width_(sm_width),
      sm_height_(sm_height), sm_cascade_count_(sm_cascade_count) {
  makeOrthographicProjection(/*aspect_ratio=*/1.0f, /*z_near=*/-100.0f,
                             /*z_far=*/100.0f, /*matrix=*/proj_matrix_);
  view_matrices_.resize(4, 4 * sm_cascade_count);
  sm_cascade_splits_.resize(sm_cascade_count + 1);
  Eigen::Vector3f norm_dir = dir_;
  Eigen::Vector3f norm_up = (up - norm_dir * up.dot(norm_dir)).normalized();
  Eigen::Matrix3f inv_view_rot_matrix;
  // clang-format off
  inv_view_rot_matrix << norm_up.cross(norm_dir),
                         norm_up,
                         norm_dir;
  // clang-format on
  view_rot_matrix_ = inv_view_rot_matrix.transpose();

  sm_depth_array_texture_ = std::make_shared<Texture3D>(
      GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT, sm_width, sm_height,
      sm_cascade_count, GL_DEPTH_COMPONENT, GL_FLOAT);
  sm_framebuffer_ = std::make_shared<Framebuffer>();
}

void DirectionalLight::updateViewMatricesAndSplits(
    const Eigen::Matrix4f &camera_view_matrix, float aspect_ratio, float z_near,
    float z_far, float fov) {
  // Calculate cascade splits in view space
  for (int i = 0; i < sm_cascade_count_ + 1; ++i) {
    float t = static_cast<float>(i) / sm_cascade_count_;
    sm_cascade_splits_(i) = z_near * std::pow(z_far / z_near, t);
  }

  Eigen::Affine3f inv_camera_view_tf(camera_view_matrix.inverse());
  // https://lxjk.github.io/2017/04/15/Calculate-Minimal-Bounding-Sphere-of-Frustum.html
  float k =
      std::sqrt(1.0f + aspect_ratio * aspect_ratio) * std::tan(0.5f * fov);
  float k_sq = k * k;
  for (int i = 0; i < sm_cascade_count_; ++i) {
    float z_sn = sm_cascade_splits_(i);     // Near z of frustum segment
    float z_sf = sm_cascade_splits_(i + 1); // Far z of frustum segment
    float z_range = z_sf - z_sn;
    float z_sum = z_sf + z_sn;
    // Find a bounding sphere in view space
    Eigen::Vector3f center;
    float radius;
    if (k_sq >= z_range / z_sum) {
      center = Eigen::Vector3f{0.0f, 0.0f, -z_sf};
      radius = z_sf * k;
    } else {
      center = Eigen::Vector3f{0.0f, 0.0f, -0.5f * z_sum * (1.0f + k_sq)};
      radius = 0.5f * std::sqrt(z_range * z_range +
                                2.0f * (z_sf * z_sf + z_sn * z_sn) * k_sq +
                                z_sum * z_sum * k_sq * k_sq);
    }
    // Transform center of sphere into world space
    Eigen::Vector3f center_world = inv_camera_view_tf * center;
    view_matrices_.block<4, 4>(0, 4 * i) =
        Eigen::Affine3f(Eigen::Scaling(1.0f / radius) * view_rot_matrix_ *
                        Eigen::Translation3f(-center_world))
            .matrix();
  }
}

void ProgramState::updateUniforms(const Program &program) {
  if (proj_matrix_.dirty_) {
    program.setProjectionMatrix(proj_matrix_.value_);
  }
  if (view_matrix_.dirty_) {
    program.setViewMatrix(view_matrix_.value_);
  }
  if (view_matrix_.dirty_ || model_matrix_.dirty_) {
    Eigen::Matrix4f model_view_matrix =
        view_matrix_.value_ * model_matrix_.value_;
    program.setModelViewMatrix(model_view_matrix);
    program.setNormalMatrix(
        model_view_matrix.topLeftCorner<3, 3>().inverse().transpose());
  }
  if (object_color_.dirty_) {
    program.setObjectColor(object_color_.value_);
  }
  if (dir_light_dir_.dirty_) {
    program.setLightDir(dir_light_dir_.value_);
  }
  if (dir_light_proj_matrix_.dirty_) {
    program.setLightProjMatrix(dir_light_proj_matrix_.value_);
  }
  if (dir_light_color_.dirty_) {
    program.setLightColor(dir_light_color_.value_);
  }
  if (dir_light_view_matrices_.dirty_ || model_matrix_.dirty_) {
    Eigen::Matrix<float, 4, Eigen::Dynamic> light_mv_matrices(
        4, dir_light_view_matrices_.value_.cols());
    for (int j = 0; j < light_mv_matrices.cols(); j += 4) {
      light_mv_matrices.block<4, 4>(0, j) =
          dir_light_view_matrices_.value_.block<4, 4>(0, j) *
          model_matrix_.value_;
    }
    program.setLightModelViewMatrices(light_mv_matrices);
  }
  if (dir_light_sm_cascade_splits_.dirty_) {
    // Shader only supports up to 5 shadow map cascades at the moment
    // Only 4 splits are needed to describe 5 cascades, starting from index 1
    program.setCascadeFarSplits(
        dir_light_sm_cascade_splits_.value_.segment<4>(1));
  }

  proj_matrix_.dirty_ = false;
  view_matrix_.dirty_ = false;
  model_matrix_.dirty_ = false;
  object_color_.dirty_ = false;
  dir_light_color_.dirty_ = false;
  dir_light_dir_.dirty_ = false;
  dir_light_proj_matrix_.dirty_ = false;
  dir_light_view_matrices_.dirty_ = false;
  dir_light_sm_cascade_splits_.dirty_ = false;
}

GLFWRenderer::GLFWRenderer(bool hidden)
    : z_near_(0.1f), z_far_(100.0f), fov_(M_PI / 3) {
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
  window_ = glfwCreateWindow(640, 480, "GLFW Renderer", NULL, NULL);
  if (!window_) {
    throw std::runtime_error("Could not create GLFW window");
  }

  glfwMakeContextCurrent(window_);
  // Load all available extensions even if they are not in the extensions string
  glewExperimental = GL_TRUE;
  glewInit();

  // Create default shader program
  std::string default_vs_source = loadString("data/shaders/default.vert.glsl");
  std::string default_fs_source = loadString("data/shaders/default.frag.glsl");
  default_program_ =
      std::make_shared<Program>(default_vs_source, default_fs_source);

  // Create depth shader program
  std::string depth_vs_source = loadString("data/shaders/depth.vert.glsl");
  std::string depth_fs_source; // Empty string (no fragment shader needed)
  depth_program_ = std::make_shared<Program>(depth_vs_source, depth_fs_source);

  // Create meshes
  box_mesh_ = makeBoxMesh();
  tube_mesh_ = makeTubeMesh(/*n_segments=*/32);
  capsule_end_mesh_ = makeCapsuleEndMesh(/*n_segments=*/32, /*n_rings=*/8);
  cylinder_end_mesh_ = makeCylinderEndMesh(/*n_segments=*/32);

  // Create directional light
  dir_light_ = std::make_shared<DirectionalLight>(
      /*color=*/Eigen::Vector3f{1.0f, 1.0f, 1.0f},
      /*dir=*/Eigen::Vector3f{1.0f, 2.0f, 3.0f},
      /*up=*/Eigen::Vector3f{0.0f, 1.0f, 0.0f},
      /*sm_width=*/2048, /*sm_height=*/2048, /*sm_cascade_count=*/5);

  // Load font
  font_ = std::make_shared<BitmapFont>("data/fonts/OpenSans-Regular.fnt");

  // Set up callbacks
  // Allow accessing "this" from static callbacks
  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, framebufferSizeCallback);
  glfwSetKeyCallback(window_, keyCallback);
  glfwSetMouseButtonCallback(window_, mouseButtonCallback);
  glfwSetCursorPosCallback(window_, cursorPositionCallback);
  glfwSetScrollCallback(window_, scrollCallback);

  // Initialize projection matrix
  glfwGetFramebufferSize(window_, &framebuffer_width_, &framebuffer_height_);
  updateProjectionMatrix();

  // Enable depth test
  glEnable(GL_DEPTH_TEST);

  // Set default camera parameters
  camera_controller_.pitch_ = -M_PI / 6;
  camera_controller_.distance_ = 2.0;
}

GLFWRenderer::~GLFWRenderer() {
  glfwDestroyWindow(window_);
  glfwTerminate();
}

void GLFWRenderer::update(double dt) {
  camera_controller_.update(dt);
  camera_controller_.getViewMatrix(view_matrix_);
}

void GLFWRenderer::render(const Simulation &sim, int width, int height,
                          const Framebuffer *target_framebuffer) {
  if (width < 0 || height < 0) {
    // Use default framebuffer size
    width = framebuffer_width_;
    height = framebuffer_height_;
  }
  float aspect_ratio = static_cast<float>(width) / height;
  dir_light_->updateViewMatricesAndSplits(view_matrix_, aspect_ratio, z_near_,
                                          z_far_, fov_);

  // Render shadow map
  dir_light_->sm_framebuffer_->bind();
  glViewport(0, 0, dir_light_->sm_width_, dir_light_->sm_height_);
  glEnable(GL_POLYGON_OFFSET_FILL);
  glPolygonOffset(2.0f, 1.0f);
  depth_program_->use();
  ProgramState depth_program_state;
  for (int i = 0; i < dir_light_->sm_cascade_count_; ++i) {
    dir_light_->sm_framebuffer_->attachDepthTextureLayer(
        *dir_light_->sm_depth_array_texture_, i);
    glClear(GL_DEPTH_BUFFER_BIT);
    depth_program_state.setProjectionMatrix(dir_light_->proj_matrix_);
    depth_program_state.setViewMatrix(
        dir_light_->view_matrices_.block<4, 4>(0, 4 * i));
    draw(sim, *depth_program_, depth_program_state);
  }

  // Render main window
  if (target_framebuffer) {
    target_framebuffer->bind();
  } else {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  }
  glViewport(0, 0, width, height);
  glClearColor(0.4f, 0.6f, 0.8f, 1.0f); // Cornflower blue
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_POLYGON_OFFSET_FILL);
  default_program_->use();
  ProgramState default_program_state;
  default_program_state.setProjectionMatrix(proj_matrix_);
  default_program_state.setViewMatrix(view_matrix_);
  default_program_state.setDirectionalLight(*dir_light_);
  dir_light_->sm_depth_array_texture_->bind();
  draw(sim, *default_program_, default_program_state);

  glfwSwapBuffers(window_);
  glfwPollEvents();
}

void GLFWRenderer::readPixels(int x, int y, int width, int height,
                              unsigned char *data) const {
  glReadPixels(x, y, width, height, GL_RGBA, GL_UNSIGNED_BYTE, data);
}

void GLFWRenderer::getFramebufferSize(int &width, int &height) const {
  width = framebuffer_width_;
  height = framebuffer_height_;
}

bool GLFWRenderer::shouldClose() const {
  return glfwWindowShouldClose(window_);
}

void GLFWRenderer::draw(const Simulation &sim, const Program &program,
                        ProgramState &program_state) const {
  // Draw robots
  for (Index robot_idx = 0; robot_idx < sim.getRobotCount(); ++robot_idx) {
    const Robot &robot = *sim.getRobot(robot_idx);
    for (std::size_t link_idx = 0; link_idx < robot.links_.size(); ++link_idx) {
      const Link &link = robot.links_[link_idx];
      Matrix4 link_transform;
      sim.getLinkTransform(robot_idx, link_idx, link_transform);

      // Draw the link's collision shape
      program_state.setObjectColor(link.color_);
      switch (link.shape_) {
      case LinkShape::CAPSULE:
        drawCapsule(link_transform.cast<float>(), link.length_ / 2,
                    robot.link_radius_, program, program_state);
        break;
      case LinkShape::CYLINDER:
        drawCylinder(link_transform.cast<float>(), link.length_ / 2,
                     robot.link_radius_, program, program_state);
        break;
      default:
        throw std::runtime_error("Unexpected link shape");
      }

      // Draw the link's joint
      program_state.setObjectColor(link.joint_color_);
      Matrix3 joint_axis_rotation =
          makeVectorToVectorRotation(link.joint_axis_, Vector3::UnitX());
      Matrix4 joint_transform =
          (Affine3(link_transform) * Translation3(-link.length_ / 2, 0, 0) *
           Affine3(joint_axis_rotation))
              .matrix();
      switch (link.joint_type_) {
      case JointType::FREE:
        // Nothing to draw
        break;
      case JointType::HINGE:
        drawCylinder(joint_transform.cast<float>(), robot.link_radius_,
                     robot.link_radius_, program, program_state);
        break;
      case JointType::FIXED:
        drawBox(joint_transform.cast<float>(),
                Eigen::Vector3f::Constant(robot.link_radius_), program,
                program_state);
        break;
      default:
        throw std::runtime_error("Unexpected joint type");
      }
    }
  }

  // Draw props
  program_state.setObjectColor({0.8f, 0.7f, 0.6f}); // Tan
  for (Index prop_idx = 0; prop_idx < sim.getPropCount(); ++prop_idx) {
    const Prop &prop = *sim.getProp(prop_idx);
    Matrix4 prop_transform;
    sim.getPropTransform(prop_idx, prop_transform);
    drawBox(prop_transform.cast<float>(), prop.half_extents_.cast<float>(),
            program, program_state);
  }
}

void GLFWRenderer::drawBox(const Eigen::Matrix4f &transform,
                           const Eigen::Vector3f &half_extents,
                           const Program &program,
                           ProgramState &program_state) const {
  Eigen::Affine3f model_transform =
      Eigen::Affine3f(transform) * Eigen::Scaling(half_extents);
  box_mesh_->bind();
  program_state.setModelMatrix(model_transform.matrix());
  program_state.updateUniforms(program);
  box_mesh_->draw();
}

void GLFWRenderer::drawCapsule(const Eigen::Matrix4f &transform,
                               float half_length, float radius,
                               const Program &program,
                               ProgramState &program_state) const {
  drawTubeBasedShape(transform, half_length, radius, program, program_state,
                     *capsule_end_mesh_);
}

void GLFWRenderer::drawCylinder(const Eigen::Matrix4f &transform,
                                float half_length, float radius,
                                const Program &program,
                                ProgramState &program_state) const {
  drawTubeBasedShape(transform, half_length, radius, program, program_state,
                     *cylinder_end_mesh_);
}

void GLFWRenderer::drawTubeBasedShape(const Eigen::Matrix4f &transform,
                                      float half_length, float radius,
                                      const Program &program,
                                      ProgramState &program_state,
                                      const Mesh &end_mesh) const {
  Eigen::Affine3f right_end_model_transform =
      Eigen::Affine3f(transform) * Eigen::Translation3f(half_length, 0, 0) *
      Eigen::Scaling(radius, radius, radius);
  end_mesh.bind();
  program_state.setModelMatrix(right_end_model_transform.matrix());
  program_state.updateUniforms(program);
  end_mesh.draw();

  Eigen::Affine3f left_end_model_transform =
      Eigen::Affine3f(transform) * Eigen::Translation3f(-half_length, 0, 0) *
      Eigen::Scaling(-radius, radius, -radius);
  end_mesh.bind();
  program_state.setModelMatrix(left_end_model_transform.matrix());
  program_state.updateUniforms(program);
  end_mesh.draw();

  Eigen::Affine3f middle_model_transform =
      Eigen::Affine3f(transform) * Eigen::Scaling(half_length, radius, radius);
  tube_mesh_->bind();
  program_state.setModelMatrix(middle_model_transform.matrix());
  program_state.updateUniforms(program);
  tube_mesh_->draw();
}

void GLFWRenderer::errorCallback(int error, const char *description) {
  std::cerr << "GLFW error: " << description << std::endl;
}

void GLFWRenderer::framebufferSizeCallback(GLFWwindow *window, int width,
                                           int height) {
  GLFWRenderer *renderer =
      static_cast<GLFWRenderer *>(glfwGetWindowUserPointer(window));
  renderer->framebuffer_width_ = width;
  renderer->framebuffer_height_ = height;
  renderer->updateProjectionMatrix();
}

void GLFWRenderer::keyCallback(GLFWwindow *window, int key, int scancode,
                               int action, int mods) {
  GLFWRenderer *renderer =
      static_cast<GLFWRenderer *>(glfwGetWindowUserPointer(window));
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GL_TRUE);
  }
  renderer->camera_controller_.handleKey(key, scancode, action, mods);
}

void GLFWRenderer::mouseButtonCallback(GLFWwindow *window, int button,
                                       int action, int mods) {
  GLFWRenderer *renderer =
      static_cast<GLFWRenderer *>(glfwGetWindowUserPointer(window));
  renderer->camera_controller_.handleMouseButton(button, action, mods);
}

void GLFWRenderer::cursorPositionCallback(GLFWwindow *window, double xpos,
                                          double ypos) {
  GLFWRenderer *renderer =
      static_cast<GLFWRenderer *>(glfwGetWindowUserPointer(window));
  renderer->camera_controller_.handleCursorPosition(xpos, ypos);
}

void GLFWRenderer::scrollCallback(GLFWwindow *window, double xoffset,
                                  double yoffset) {
  GLFWRenderer *renderer =
      static_cast<GLFWRenderer *>(glfwGetWindowUserPointer(window));
  renderer->camera_controller_.handleScroll(xoffset, yoffset);
}

void GLFWRenderer::updateProjectionMatrix() {
  float aspect_ratio =
      static_cast<float>(framebuffer_width_) / framebuffer_height_;
  makePerspectiveProjection(aspect_ratio, z_near_, z_far_, fov_, proj_matrix_);
}

std::string GLFWRenderer::loadString(const std::string &path) {
  std::ifstream ifs(path);
  std::stringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

void makeOrthographicProjection(float aspect_ratio, float z_near, float z_far,
                                Eigen::Matrix4f &matrix) {
  float z_range = z_far - z_near;
  // clang-format off
  matrix << 1 / aspect_ratio, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, -2 / z_range, -(z_far + z_near) / z_range,
            0, 0, 0, 1;
  // clang-format on
}

void makePerspectiveProjection(float aspect_ratio, float z_near, float z_far,
                               float fov, Eigen::Matrix4f &matrix) {
  float z_range = z_far - z_near;
  float tan_half_fov = std::tan(0.5f * fov);
  // clang-format off
  matrix << 1 / (tan_half_fov * aspect_ratio), 0, 0, 0,
            0, 1 / tan_half_fov, 0, 0,
            0, 0, -(z_far + z_near) / z_range, -2 * z_far * z_near / z_range,
            0, 0, -1, 0;
  // clang-format on
}

std::shared_ptr<Mesh> makeBoxMesh() {
  // clang-format off
  std::vector<float> positions = {
      -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, // -X face
      -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, // -Y face
      -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, // -Z face
      1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1,     // +X face
      1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1,     // +Y face
      1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1};    // +Z face
  std::vector<float> normals = {
      -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,     // -X face
      0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,     // -Y face
      0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,     // -Z face
      1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,         // +X face
      0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,         // +Y face
      0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};        // +Z face
  std::vector<int> indices = {
      0, 1, 2, 3, 0, 2,                           // -X face
      4, 5, 6, 7, 4, 6,                           // -Y face
      8, 9, 10, 11, 8, 10,                        // -Z face
      12, 13, 14, 15, 12, 14,                     // +X face
      16, 17, 18, 19, 16, 18,                     // +Y face
      20, 21, 22, 23, 20, 22};                    // +Z face
  // clang-format on

  return std::make_shared<Mesh>(positions, normals, indices);
}

std::shared_ptr<Mesh> makeTubeMesh(int n_segments) {
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;

  // Define two rings of vertices
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < n_segments; ++j) {
      float theta = (2 * M_PI) * j / n_segments;
      float pos[3] = {(i == 0) ? -1.0f : 1.0f, std::cos(theta),
                      std::sin(theta)};
      float normal[3] = {0, std::cos(theta), std::sin(theta)};
      positions.insert(positions.end(), std::begin(pos), std::end(pos));
      normals.insert(normals.end(), std::begin(normal), std::end(normal));
    }
  }

  // Define triangles
  for (int j = 0; j < n_segments; ++j) {
    int idx_00 = j;
    int idx_01 = (j + 1) % n_segments;
    int idx_10 = n_segments + j;
    int idx_11 = n_segments + (j + 1) % n_segments;
    int idx[6] = {idx_00, idx_01, idx_10, idx_11, idx_10, idx_01};
    indices.insert(indices.end(), std::begin(idx), std::end(idx));
  }

  return std::make_shared<Mesh>(positions, normals, indices);
}

std::shared_ptr<Mesh> makeCapsuleEndMesh(int n_segments, int n_rings) {
  std::vector<float> positions;
  std::vector<int> indices;

  // Define rings of vertices
  for (int i = 0; i < n_rings; ++i) {
    for (int j = 0; j < n_segments; ++j) {
      float theta = (2 * M_PI) * j / n_segments;
      float phi = (M_PI / 2) * i / n_rings;
      float pos[3] = {std::sin(phi), std::cos(phi) * std::cos(theta),
                      std::cos(phi) * std::sin(theta)};
      positions.insert(positions.end(), std::begin(pos), std::end(pos));
    }
  }
  // Define zenith vertex
  float pos[3] = {1.0f, 0.0f, 0.0f};
  positions.insert(positions.end(), std::begin(pos), std::end(pos));

  // Define triangles for every ring except the last
  for (int i = 0; i < (n_rings - 1); ++i) {
    for (int j = 0; j < n_segments; ++j) {
      int idx_00 = i * n_segments + j;
      int idx_01 = i * n_segments + (j + 1) % n_segments;
      int idx_10 = (i + 1) * n_segments + j;
      int idx_11 = (i + 1) * n_segments + (j + 1) % n_segments;
      int idx[6] = {idx_00, idx_01, idx_10, idx_11, idx_10, idx_01};
      indices.insert(indices.end(), std::begin(idx), std::end(idx));
    }
  }
  // Define triangles for last ring
  for (int j = 0; j < n_segments; ++j) {
    int idx[3] = {(n_rings - 1) * n_segments + j,
                  (n_rings - 1) * n_segments + (j + 1) % n_segments,
                  n_rings * n_segments};
    indices.insert(indices.end(), std::begin(idx), std::end(idx));
  }

  // The positions and normals of points on a unit sphere are equal
  return std::make_shared<Mesh>(positions, positions, indices);
}

std::shared_ptr<Mesh> makeCylinderEndMesh(int n_segments) {
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;

  // Define a ring of vertices
  for (int j = 0; j < n_segments; ++j) {
    float theta = (2 * M_PI) * j / n_segments;
    float pos[3] = {0.0f, std::cos(theta), std::sin(theta)};
    float normal[3] = {1.0f, 0.0f, 0.0f};
    positions.insert(positions.end(), std::begin(pos), std::end(pos));
    normals.insert(normals.end(), std::begin(normal), std::end(normal));
  }

  // Define a center vertex
  float pos[3] = {0.0f, 0.0f, 0.0f};
  float normal[3] = {1.0f, 0.0f, 0.0f};
  positions.insert(positions.end(), std::begin(pos), std::end(pos));
  normals.insert(normals.end(), std::begin(normal), std::end(normal));

  // Define triangles
  for (int j = 0; j < n_segments; ++j) {
    int idx[3] = {j, (j + 1) % n_segments, n_segments};
    indices.insert(indices.end(), std::begin(idx), std::end(idx));
  }

  return std::make_shared<Mesh>(positions, normals, indices);
}

Matrix3 makeVectorToVectorRotation(Vector3 from, Vector3 to) {
  // https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
  from.normalize();
  to.normalize();
  Vector3 v = from.cross(to);
  Scalar c = from.dot(to);
  Matrix3 v_cross;
  // clang-format off
  v_cross << 0, -v(2), v(1),
             v(2), 0, -v(0),
             -v(1), v(0), 0;
  // clang-format on
  return Matrix3::Identity() + v_cross + v_cross * v_cross / (1 + c);
}

std::shared_ptr<Texture2D> loadTexture(const std::string &path) {
  unsigned char *rgba_raw = nullptr;
  unsigned int width, height;
  unsigned int error =
      lodepng_decode32_file(&rgba_raw, &width, &height, path.c_str());
  std::unique_ptr<unsigned char[], decltype(std::free) *> rgba(rgba_raw,
                                                               std::free);
  if (error) {
    throw std::runtime_error("Could not load texture from file \"" + path +
                             "\": " + lodepng_error_text(error));
  }
  return std::make_shared<Texture2D>(
      /*target=*/GL_TEXTURE_2D, /*level=*/0, /*internal_format=*/GL_RGBA,
      /*width=*/width, /*height=*/height, /*format=*/GL_RGBA,
      /*type=*/GL_UNSIGNED_BYTE, /*data=*/rgba.get());
}

} // namespace robot_design
