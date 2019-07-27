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

GLFWRenderer::GLFWRenderer() {
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

  glfwSetKeyCallback(window_, keyCallback);
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

    for (Index i = 0; i < sim.getRobotCount(); ++i) {
      renderRobot(*sim.getRobot(i), sim);
    }

    glfwSwapBuffers(window_);
    glfwPollEvents();
  }
}

void GLFWRenderer::renderRobot(const Robot &robot, const Simulation &sim) {
  for (const auto &link : robot.links_) {
  }

  // TODO
  default_program_->use();
  float time = glfwGetTime();
  std::vector<float> positions = {
      0, 0, 0,
      std::cos(time), 0, 0,
      0, std::sin(time), 0};
  std::vector<int> indices;
  Mesh mesh(positions, indices);
  mesh.bind();
  glDrawArrays(GL_TRIANGLES, 0, positions.size());
}

void GLFWRenderer::keyCallback(GLFWwindow *window, int key, int scancode,
    int action, int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GL_TRUE);
  }
}

std::string GLFWRenderer::loadString(const std::string &path) {
  std::ifstream ifs(path);
  std::stringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

}  // namespace robot_design
