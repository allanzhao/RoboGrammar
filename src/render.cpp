#include <cmath>
#include <exception>
#include <fstream>
#include <robot_design/render.h>
#include <robot_design/utils.h>
#include <sstream>
#include <string>

namespace robot_design {

const VertexAttribute ATTRIB_POSITION(0, "position");

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

  glLinkProgram(program_);

  // Find uniform indices
  proj_matrix_index_ = glGetUniformLocation(program_, "proj_matrix");
  view_matrix_index_ = glGetUniformLocation(program_, "view_matrix");
  model_matrix_index_ = glGetUniformLocation(program_, "model_matrix");
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

void Program::setViewMatrix(const Eigen::Matrix4f &view_matrix) const {
  glUniformMatrix4fv(view_matrix_index_, 1, GL_FALSE, view_matrix.data());
}

void Program::setModelMatrix(const Eigen::Matrix4f &model_matrix) const {
  glUniformMatrix4fv(model_matrix_index_, 1, GL_FALSE, model_matrix.data());
}

Mesh::Mesh(const std::vector<GLfloat> &positions,
           const std::vector<GLint> &indices)
    : vertex_array_(0), position_buffer_(0), index_buffer_(0) {
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

  // Create index buffer
  glGenBuffers(1, &index_buffer_);
  // TODO
}

Mesh::~Mesh() {
  glDeleteBuffers(1, &index_buffer_);
  glDeleteBuffers(1, &position_buffer_);
  glDeleteVertexArrays(1, &vertex_array_);
}

void Mesh::bind() const {
  glBindVertexArray(vertex_array_);
}

GLFWRenderer::GLFWRenderer() : z_near_(1.0f), z_far_(1000.0f), fov_(M_PI / 3) {
  if (!glfwInit()) {
    return;
  }

  // Require OpenGL 3.2 or higher
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
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

  // Allow accessing "this" from static callbacks
  glfwSetWindowUserPointer(window_, this);
  glfwSetWindowSizeCallback(window_, windowSizeCallback);
  glfwSetFramebufferSizeCallback(window_, framebufferSizeCallback);
  glfwSetKeyCallback(window_, keyCallback);

  // Initialize matrices
  glfwGetWindowSize(window_, &window_width_, &window_height_);
  updateProjectionMatrix();
  view_matrix_ = Eigen::Matrix4f::Identity();
}

GLFWRenderer::~GLFWRenderer() {
  default_program_.reset();

  glfwDestroyWindow(window_);
  glfwTerminate();
}

void GLFWRenderer::run(Simulation &sim) {
  double last_time = glfwGetTime();
  while (!glfwWindowShouldClose(window_)) {
    double current_time = glfwGetTime();
    sim.advance(current_time - last_time);
    last_time = current_time;

    glClearColor(0.8f, 0.8f, 0.8f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    default_program_->use();
    default_program_->setProjectionMatrix(proj_matrix_);
    default_program_->setViewMatrix(view_matrix_);

    for (Index i = 0; i < sim.getRobotCount(); ++i) {
      renderRobot(*sim.getRobot(i), sim);
    }

    glfwSwapBuffers(window_);
    glfwPollEvents();
  }
}

void GLFWRenderer::renderRobot(const Robot &robot, const Simulation &sim) {
  default_program_->setModelMatrix(Eigen::Matrix4f::Identity());

  for (const auto &link : robot.links_) {
  }

  // TODO
  float time = glfwGetTime();
  std::vector<float> positions = {
      0, 0, 10,
      std::cos(time), 0, 10,
      0, std::sin(time), 10};
  std::vector<int> indices;
  Mesh mesh(positions, indices);
  mesh.bind();
  glDrawArrays(GL_TRIANGLES, 0, positions.size());
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
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GL_TRUE);
  }
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
  float z_range = z_near - z_far;
  float tan_half_fov = std::tan(fov / 2);
  matrix << 1 / (tan_half_fov * aspect_ratio), 0, 0, 0,
            0, 1 / tan_half_fov, 0, 0,
            0, 0, (-z_near - z_far) / z_range, 2 * z_far * z_near / z_range,
            0, 0, 1, 0;
}

}  // namespace robot_design
