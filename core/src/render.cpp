#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <cstdlib>
#include <lodepng.h>
#include <memory>
#include <robot_design/render.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace robot_design {

const VertexAttribute ATTRIB_POSITION(0, "model_position");
const VertexAttribute ATTRIB_NORMAL(1, "model_normal");
const VertexAttribute ATTRIB_TEX_COORD(2, "model_tex_coord");

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
  glBindAttribLocation(program_, ATTRIB_TEX_COORD.index_,
                       ATTRIB_TEX_COORD.name_.c_str());

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
  tex_coords_matrix_index_ =
      glGetUniformLocation(program_, "tex_coords_matrix");
  model_view_matrix_index_ =
      glGetUniformLocation(program_, "model_view_matrix");
  normal_matrix_index_ = glGetUniformLocation(program_, "normal_matrix");
  proc_texture_type_index_ =
      glGetUniformLocation(program_, "proc_texture_type");
  object_color_index_ = glGetUniformLocation(program_, "object_color");
  world_light_dir_index_ = glGetUniformLocation(program_, "world_light_dir");
  light_proj_matrix_index_ =
      glGetUniformLocation(program_, "light_proj_matrix");
  light_model_view_matrices_index_ =
      glGetUniformLocation(program_, "light_model_view_matrices");
  light_color_index_ = glGetUniformLocation(program_, "light_color");
  shadow_map_index_ = glGetUniformLocation(program_, "shadow_map");
  msdf_index_ = glGetUniformLocation(program_, "msdf");
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

Mesh::Mesh(GLenum usage)
    : usage_(usage), vertex_array_(0), position_buffer_(0), normal_buffer_(0),
      tex_coord_buffer_(0), index_buffer_(0) {
  // Create vertex array object (VAO)
  glGenVertexArrays(1, &vertex_array_);
}

Mesh::~Mesh() {
  glDeleteBuffers(1, &index_buffer_);
  glDeleteBuffers(1, &tex_coord_buffer_);
  glDeleteBuffers(1, &normal_buffer_);
  glDeleteBuffers(1, &position_buffer_);
  glDeleteVertexArrays(1, &vertex_array_);
}

void Mesh::setPositions(const std::vector<GLfloat> &positions) {
  bind();
  if (!position_buffer_) {
    glGenBuffers(1, &position_buffer_);
  }
  glBindBuffer(GL_ARRAY_BUFFER, position_buffer_);
  glBufferData(GL_ARRAY_BUFFER, positions.size() * sizeof(positions[0]),
               positions.data(), usage_);
  glVertexAttribPointer(ATTRIB_POSITION.index_, 3, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(ATTRIB_POSITION.index_);
}

void Mesh::setNormals(const std::vector<GLfloat> &normals) {
  bind();
  if (!normal_buffer_) {
    glGenBuffers(1, &normal_buffer_);
  }
  glBindBuffer(GL_ARRAY_BUFFER, normal_buffer_);
  glBufferData(GL_ARRAY_BUFFER, normals.size() * sizeof(normals[0]),
               normals.data(), usage_);
  glVertexAttribPointer(ATTRIB_NORMAL.index_, 3, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(ATTRIB_NORMAL.index_);
}

void Mesh::setTexCoords(const std::vector<GLfloat> &tex_coords) {
  bind();
  if (!tex_coord_buffer_) {
    glGenBuffers(1, &tex_coord_buffer_);
  }
  glBindBuffer(GL_ARRAY_BUFFER, tex_coord_buffer_);
  glBufferData(GL_ARRAY_BUFFER, tex_coords.size() * sizeof(tex_coords[0]),
               tex_coords.data(), usage_);
  glVertexAttribPointer(ATTRIB_TEX_COORD.index_, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(ATTRIB_TEX_COORD.index_);
}

void Mesh::setIndices(const std::vector<GLint> &indices) {
  bind();
  if (!index_buffer_) {
    glGenBuffers(1, &index_buffer_);
  }
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(indices[0]),
               indices.data(), usage_);
  index_count_ = indices.size();
}

Texture2D::Texture2D(GLenum target, GLint level, GLint internal_format,
                     GLsizei width, GLsizei height, GLenum format, GLenum type,
                     const GLvoid *data)
    : target_(target), texture_(0) {
  glGenTextures(1, &texture_);
  glBindTexture(target, texture_);
  glTexImage2D(target, level, internal_format, width, height, 0, format, type,
               data);
}

Texture2D::~Texture2D() { glDeleteTextures(1, &texture_); }

void Texture2D::setParameter(GLenum name, GLint value) const {
  bind();
  glTexParameteri(target_, name, value);
}

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
}

Texture3D::~Texture3D() { glDeleteTextures(1, &texture_); }

void Texture3D::setParameter(GLenum name, GLint value) const {
  bind();
  glTexParameteri(target_, name, value);
}

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

Eigen::Matrix4f CameraParameters::getProjMatrix() const {
  Eigen::Matrix4f proj_matrix;
  makePerspectiveProjection(aspect_ratio_, z_near_, z_far_, fov_, proj_matrix);
  return proj_matrix;
}

Eigen::Matrix4f CameraParameters::getViewMatrix() const {
  Eigen::Affine3f view_transform(
      Eigen::Translation3f(0.0f, 0.0f, -distance_) *
      Eigen::AngleAxisf(-pitch_, Eigen::Vector3f::UnitX()) *
      Eigen::AngleAxisf(-yaw_, Eigen::Vector3f::UnitY()) *
      Eigen::Translation3f(-position_));
  return view_transform.matrix();
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
  sm_depth_array_texture_->setParameter(GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  sm_depth_array_texture_->setParameter(GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  sm_depth_array_texture_->setParameter(GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  sm_depth_array_texture_->setParameter(GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  sm_depth_array_texture_->setParameter(GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  sm_depth_array_texture_->setParameter(GL_TEXTURE_COMPARE_MODE,
                                        GL_COMPARE_R_TO_TEXTURE);
  sm_depth_array_texture_->setParameter(GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
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
  if (tex_coords_matrix_.dirty_) {
    program.setTexCoordsMatrix(tex_coords_matrix_.value_);
  }
  if (view_matrix_.dirty_ || model_matrix_.dirty_) {
    Eigen::Matrix4f model_view_matrix =
        view_matrix_.value_ * model_matrix_.value_;
    program.setModelViewMatrix(model_view_matrix);
    program.setNormalMatrix(
        model_view_matrix.topLeftCorner<3, 3>().inverse().transpose());
  }
  if (proc_texture_type_.dirty_) {
    program.setProcTextureType(proc_texture_type_.value_);
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
  proc_texture_type_.dirty_ = false;
  object_color_.dirty_ = false;
  dir_light_color_.dirty_ = false;
  dir_light_dir_.dirty_ = false;
  dir_light_proj_matrix_.dirty_ = false;
  dir_light_view_matrices_.dirty_ = false;
  dir_light_sm_cascade_splits_.dirty_ = false;
}

void makeOrthographicProjection(float aspect_ratio, float z_near, float z_far,
                                Ref<Eigen::Matrix4f> matrix) {
  float z_range = z_far - z_near;
  // clang-format off
  matrix << 1 / aspect_ratio, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, -2 / z_range, -(z_far + z_near) / z_range,
            0, 0, 0, 1;
  // clang-format on
}

void makePerspectiveProjection(float aspect_ratio, float z_near, float z_far,
                               float fov, Ref<Eigen::Matrix4f> matrix) {
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

  auto mesh = std::make_shared<Mesh>(GL_STATIC_DRAW);
  mesh->setPositions(positions);
  mesh->setNormals(normals);
  mesh->setIndices(indices);
  return mesh;
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

  auto mesh = std::make_shared<Mesh>(GL_STATIC_DRAW);
  mesh->setPositions(positions);
  mesh->setNormals(normals);
  mesh->setIndices(indices);
  return mesh;
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
  auto mesh = std::make_shared<Mesh>(GL_STATIC_DRAW);
  mesh->setPositions(positions);
  mesh->setNormals(positions);
  mesh->setIndices(indices);
  return mesh;
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

  auto mesh = std::make_shared<Mesh>(GL_STATIC_DRAW);
  mesh->setPositions(positions);
  mesh->setNormals(normals);
  mesh->setIndices(indices);
  return mesh;
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
