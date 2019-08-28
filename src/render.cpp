#include <cmath>
#include <exception>
#include <fstream>
#include <robot_design/render.h>
#include <robot_design/utils.h>
#include <sstream>
#include <string>

namespace robot_design {

const VertexAttribute ATTRIB_POSITION(0, "model_position");
const VertexAttribute ATTRIB_NORMAL(1, "model_normal");

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
    throw std::runtime_error(std::string("Failed to compile vertex shader: ") + buffer);
  }

  // Create fragment shader
  fragment_shader_ = glCreateShader(GL_FRAGMENT_SHADER);
  const GLchar *fragment_shader_source_ptr = fragment_shader_source.c_str();
  glShaderSource(fragment_shader_, 1, &fragment_shader_source_ptr, NULL);
  glCompileShader(fragment_shader_);
  // Check for compile errors
  glGetShaderiv(fragment_shader_, GL_COMPILE_STATUS, &status);
  if (!status) {
    char buffer[512];
    glGetShaderInfoLog(fragment_shader_, sizeof(buffer), NULL, buffer);
    throw std::runtime_error(std::string("Failed to compile fragment shader: ") + buffer);
  }

  // Create program
  program_ = glCreateProgram();
  glAttachShader(program_, vertex_shader_);
  glAttachShader(program_, fragment_shader_);

  // Define fixed attribute indices
  glBindAttribLocation(program_, ATTRIB_POSITION.index_, ATTRIB_POSITION.name_.c_str());
  glBindAttribLocation(program_, ATTRIB_NORMAL.index_, ATTRIB_NORMAL.name_.c_str());

  glLinkProgram(program_);

  // Find uniform indices
  proj_matrix_index_ = glGetUniformLocation(program_, "proj_matrix");
  view_matrix_index_ = glGetUniformLocation(program_, "view_matrix");
  model_view_matrix_index_ = glGetUniformLocation(program_, "model_view_matrix");
  normal_matrix_index_ = glGetUniformLocation(program_, "normal_matrix");
  object_color_index_ = glGetUniformLocation(program_, "object_color");
}

Program::~Program() {
  glDetachShader(program_, fragment_shader_);
  glDetachShader(program_, vertex_shader_);
  glDeleteProgram(program_);
  glDeleteShader(fragment_shader_);
  glDeleteShader(vertex_shader_);
}

void Program::use() const {
  glUseProgram(program_);
}

void Program::setProjectionMatrix(const Eigen::Matrix4f &proj_matrix) const {
  glUniformMatrix4fv(proj_matrix_index_, 1, GL_FALSE, proj_matrix.data());
}

void Program::setModelViewMatrices(const Eigen::Matrix4f &model_matrix,
                                   const Eigen::Matrix4f &view_matrix) const {
  Eigen::Matrix4f model_view_matrix = view_matrix * model_matrix;
  Eigen::Matrix3f normal_matrix = model_view_matrix.topLeftCorner<3, 3>().inverse().transpose();
  glUniformMatrix4fv(view_matrix_index_, 1, GL_FALSE, view_matrix.data());
  glUniformMatrix4fv(model_view_matrix_index_, 1, GL_FALSE, model_view_matrix.data());
  glUniformMatrix3fv(normal_matrix_index_, 1, GL_FALSE, normal_matrix.data());
}

void Program::setObjectColor(const Eigen::Vector3f &object_color) const {
  glUniform3fv(object_color_index_, 1, object_color.data());
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

void Mesh::bind() const {
  glBindVertexArray(vertex_array_);
}

void Mesh::draw() const {
  glDrawElements(GL_TRIANGLES, index_count_, GL_UNSIGNED_INT, 0);
}

const std::array<int, FPSCameraController::ACTION_COUNT> FPSCameraController::DEFAULT_KEY_BINDINGS = {
    GLFW_KEY_W, GLFW_KEY_A, GLFW_KEY_S, GLFW_KEY_D, GLFW_KEY_Q, GLFW_KEY_E,
    GLFW_MOUSE_BUTTON_LEFT};

void FPSCameraController::handleKey(int key, int scancode, int action, int mods) {
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

  if (action_flags_[ACTION_MOVE_FORWARD]) { offset(2) -= move_speed_ * dt; }
  if (action_flags_[ACTION_MOVE_LEFT]) { offset(0) -= move_speed_ * dt; }
  if (action_flags_[ACTION_MOVE_BACKWARD]) { offset(2) += move_speed_ * dt; }
  if (action_flags_[ACTION_MOVE_RIGHT]) { offset(0) += move_speed_ * dt; }
  if (action_flags_[ACTION_MOVE_UP]) { offset(1) += move_speed_ * dt; }
  if (action_flags_[ACTION_MOVE_DOWN]) { offset(1) -= move_speed_ * dt; }
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

GLFWRenderer::GLFWRenderer() : z_near_(0.1f), z_far_(100.0f), fov_(M_PI / 3),
                               camera_controller_() {
  if (!glfwInit()) {
    return;
  }

  // Require OpenGL 3.2 or higher
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  // Enable 4x MSAA
  glfwWindowHint(GLFW_SAMPLES, 4);
  window_ = glfwCreateWindow(640, 480, "GLFW Renderer", NULL, NULL);
  if (!window_) {
    return;
  }

  glfwMakeContextCurrent(window_);
  // Load all available extensions even if they are not in the extensions string
  glewExperimental = GL_TRUE;
  glewInit();

  // Create default shader program
  std::string default_vs_source = loadString("data/default.vert.glsl");
  std::string default_fs_source = loadString("data/default.frag.glsl");
  default_program_ = std::make_shared<Program>(default_vs_source, default_fs_source);

  // Create meshes
  box_mesh_ = makeBoxMesh();
  capsule_end_mesh_ = makeCapsuleEndMesh(/*n_segments=*/32, /*n_rings=*/8);
  capsule_middle_mesh_ = makeCapsuleMiddleMesh(/*n_segments=*/32);

  // Set up callbacks
  // Allow accessing "this" from static callbacks
  glfwSetWindowUserPointer(window_, this);
  glfwSetWindowSizeCallback(window_, windowSizeCallback);
  glfwSetFramebufferSizeCallback(window_, framebufferSizeCallback);
  glfwSetKeyCallback(window_, keyCallback);
  glfwSetMouseButtonCallback(window_, mouseButtonCallback);
  glfwSetCursorPosCallback(window_, cursorPositionCallback);
  glfwSetScrollCallback(window_, scrollCallback);

  // Initialize projection matrix
  glfwGetWindowSize(window_, &window_width_, &window_height_);
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
}

void GLFWRenderer::render(const Simulation &sim) {
  camera_controller_.getViewMatrix(view_matrix_);

  glClearColor(0.4f, 0.6f, 0.8f, 1.0f);  // Cornflower blue
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  default_program_->use();
  default_program_->setProjectionMatrix(proj_matrix_);
  draw(sim, *default_program_);

  glfwSwapBuffers(window_);
  glfwPollEvents();
}

bool GLFWRenderer::shouldClose() const {
  return glfwWindowShouldClose(window_);
}

void GLFWRenderer::draw(const Simulation &sim, const Program &program) const {
  program.setObjectColor({0.45f, 0.5f, 0.55f});  // Slate gray
  for (Index robot_idx = 0; robot_idx < sim.getRobotCount(); ++robot_idx) {
    const Robot &robot = *sim.getRobot(robot_idx);
    for (Index link_idx = 0; link_idx < robot.links_.size(); ++link_idx) {
      const Link &link = robot.links_[link_idx];
      Matrix4 link_transform;
      sim.getLinkTransform(robot_idx, link_idx, link_transform);
      drawCapsule(link_transform.cast<float>(), link.length_ / 2,
                  robot.link_radius_, program);
    }
  }

  program.setObjectColor({0.8f, 0.7f, 0.6f});  // Tan
  for (Index prop_idx = 0; prop_idx < sim.getPropCount(); ++prop_idx) {
    const Prop &prop = *sim.getProp(prop_idx);
    Matrix4 prop_transform;
    sim.getPropTransform(prop_idx, prop_transform);
    drawBox(prop_transform.cast<float>(), prop.half_extents_.cast<float>(),
            program);
  }
}

void GLFWRenderer::drawBox(const Eigen::Matrix4f &transform,
                           const Eigen::Vector3f &half_extents,
                           const Program &program) const {
  Eigen::Affine3f model_transform = Eigen::Affine3f(transform) *
      Eigen::DiagonalMatrix<float, 3>(half_extents);
  box_mesh_->bind();
  program.setModelViewMatrices(model_transform.matrix(), view_matrix_);
  box_mesh_->draw();
}

void GLFWRenderer::drawCapsule(const Eigen::Matrix4f &transform,
                               float half_length, float radius,
                               const Program &program) const {
  Eigen::Affine3f right_end_model_transform = Eigen::Affine3f(transform) *
      Eigen::Translation3f(half_length, 0, 0) *
      Eigen::DiagonalMatrix<float, 3>(radius, radius, radius);
  capsule_end_mesh_->bind();
  program.setModelViewMatrices(right_end_model_transform.matrix(), view_matrix_);
  capsule_end_mesh_->draw();

  Eigen::Affine3f left_end_model_transform = Eigen::Affine3f(transform) *
      Eigen::Translation3f(-half_length, 0, 0) *
      Eigen::DiagonalMatrix<float, 3>(-radius, radius, -radius);
  capsule_end_mesh_->bind();
  program.setModelViewMatrices(left_end_model_transform.matrix(), view_matrix_);
  capsule_end_mesh_->draw();

  Eigen::Affine3f middle_model_transform = Eigen::Affine3f(transform) *
      Eigen::DiagonalMatrix<float, 3>(half_length, radius, radius);
  capsule_middle_mesh_->bind();
  program.setModelViewMatrices(middle_model_transform.matrix(), view_matrix_);
  capsule_middle_mesh_->draw();
}

void GLFWRenderer::windowSizeCallback(GLFWwindow *window, int width, int height) {
  GLFWRenderer *renderer = static_cast<GLFWRenderer*>(glfwGetWindowUserPointer(window));
  renderer->window_width_ = width;
  renderer->window_height_ = height;
  renderer->updateProjectionMatrix();
}

void GLFWRenderer::framebufferSizeCallback(GLFWwindow *window, int width, int height) {
  glViewport(0, 0, width, height);
}

void GLFWRenderer::keyCallback(GLFWwindow *window, int key, int scancode,
                               int action, int mods) {
  GLFWRenderer *renderer = static_cast<GLFWRenderer*>(glfwGetWindowUserPointer(window));
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GL_TRUE);
  }
  renderer->camera_controller_.handleKey(key, scancode, action, mods);
}

void GLFWRenderer::mouseButtonCallback(GLFWwindow *window, int button,
                                       int action, int mods) {
  GLFWRenderer *renderer = static_cast<GLFWRenderer*>(glfwGetWindowUserPointer(window));
  renderer->camera_controller_.handleMouseButton(button, action, mods);
}

void GLFWRenderer::cursorPositionCallback(GLFWwindow *window, double xpos,
                                          double ypos) {
  GLFWRenderer *renderer = static_cast<GLFWRenderer*>(glfwGetWindowUserPointer(window));
  renderer->camera_controller_.handleCursorPosition(xpos, ypos);
}

void GLFWRenderer::scrollCallback(GLFWwindow *window, double xoffset,
                                  double yoffset) {
  GLFWRenderer *renderer = static_cast<GLFWRenderer*>(glfwGetWindowUserPointer(window));
  renderer->camera_controller_.handleScroll(xoffset, yoffset);
}

void GLFWRenderer::updateProjectionMatrix() {
  float aspect_ratio = static_cast<float>(window_width_) / window_height_;
  makePerspectiveProjection(aspect_ratio, z_near_, z_far_, fov_, proj_matrix_);
}

std::string GLFWRenderer::loadString(const std::string &path) {
  std::ifstream ifs(path);
  std::stringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

void makePerspectiveProjection(float aspect_ratio, float z_near, float z_far,
                               float fov, Eigen::Matrix4f &matrix) {
  float z_range = z_far - z_near;
  float tan_half_fov = std::tan(fov / 2);
  matrix << 1 / (tan_half_fov * aspect_ratio), 0, 0, 0,
            0, 1 / tan_half_fov, 0, 0,
            0, 0, -(z_far + z_near) / z_range, -2 * z_far * z_near / z_range,
            0, 0, -1, 0;
}

std::shared_ptr<Mesh> makeBoxMesh() {
  std::vector<float> positions = {
      -1, -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1,  // -X face
      -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1,  // -Y face
      -1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1,  // -Z face
      1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1,      // +X face
      1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1,      // +Y face
      1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1};     // +Z face
  std::vector<float> normals = {
      -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,      // -X face
      0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,      // -Y face
      0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,      // -Z face
      1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,          // +X face
      0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0,          // +Y face
      0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1};         // +Z face
  std::vector<int> indices = {
      0, 1, 2, 3, 0, 2,                            // -X face
      4, 5, 6, 7, 4, 6,                            // -Y face
      8, 9, 10, 11, 8, 10,                         // -Z face
      12, 13, 14, 15, 12, 14,                      // +X face
      16, 17, 18, 19, 16, 18,                      // +Y face
      20, 21, 22, 23, 20, 22};                     // +Z face

  return std::move(std::make_shared<Mesh>(positions, normals, indices));
}

std::shared_ptr<Mesh> makeCapsuleEndMesh(int n_segments, int n_rings) {
  std::vector<float> positions;
  std::vector<int> indices;

  // Define rings of vertices
  for (int i = 0; i < n_rings; ++i) {
    for (int j = 0; j < n_segments; ++j) {
      float theta = (2 * M_PI) * j / n_segments;
      float phi = (M_PI / 2) * i / n_rings;
      float pos[3] = {std::sin(phi),
                      std::cos(phi) * std::cos(theta),
                      std::cos(phi) * std::sin(theta)};
      positions.insert(positions.end(), std::begin(pos), std::end(pos));
    }
  }
  // Define zenith vertex
  float pos[3] = {1, 0, 0};
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
  return std::move(std::make_shared<Mesh>(positions, positions, indices));
}

std::shared_ptr<Mesh> makeCapsuleMiddleMesh(int n_segments) {
  std::vector<float> positions;
  std::vector<float> normals;
  std::vector<int> indices;

  // Define two rings of vertices
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < n_segments; ++j) {
      float theta = (2 * M_PI) * j / n_segments;
      float pos[3] = {(i == 0) ? -1.0f : 1.0f, std::cos(theta), std::sin(theta)};
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

  return std::move(std::make_shared<Mesh>(positions, normals, indices));
}

}  // namespace robot_design
