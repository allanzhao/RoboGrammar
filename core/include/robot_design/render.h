#pragma once

#include <Eigen/Dense>
#include <GL/glew.h>
#include <array>
#include <cmath>
#include <memory>
#include <robot_design/sim.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace robot_design {

using Eigen::Ref;

struct VertexAttribute {
  VertexAttribute(GLuint index, const std::string &name)
      : index_(index), name_(name) {}
  GLuint index_;
  std::string name_;
};

extern const VertexAttribute ATTRIB_POSITION;
extern const VertexAttribute ATTRIB_NORMAL;
extern const VertexAttribute ATTRIB_TEX_COORD;

struct DirectionalLight;

struct Program {
  Program(const std::string &vertex_shader_source,
          const std::string &fragment_shader_source);
  virtual ~Program();
  Program(const Program &other) = delete;
  Program &operator=(const Program &other) = delete;
  void use() const { glUseProgram(program_); }
  void setProjectionMatrix(const Eigen::Matrix4f &proj_matrix) const {
    glUniformMatrix4fv(proj_matrix_index_, 1, GL_FALSE, proj_matrix.data());
  }
  void setViewMatrix(const Eigen::Matrix4f &view_matrix) const {
    glUniformMatrix4fv(view_matrix_index_, 1, GL_FALSE, view_matrix.data());
  }
  void setTexCoordsMatrix(const Eigen::Matrix4f &tex_coords_matrix) const {
    glUniformMatrix4fv(tex_coords_matrix_index_, 1, GL_FALSE,
                       tex_coords_matrix.data());
  }
  void setModelViewMatrix(const Eigen::Matrix4f &model_view_matrix) const {
    glUniformMatrix4fv(model_view_matrix_index_, 1, GL_FALSE,
                       model_view_matrix.data());
  }
  void setNormalMatrix(const Eigen::Matrix3f &normal_matrix) const {
    glUniformMatrix3fv(normal_matrix_index_, 1, GL_FALSE, normal_matrix.data());
  }
  void setProcTextureType(int proc_texture_type) const {
    glUniform1i(proc_texture_type_index_, proc_texture_type);
  }
  void setObjectColor(const Eigen::Vector3f &object_color) const {
    glUniform3fv(object_color_index_, 1, object_color.data());
  }
  void setLightDir(const Eigen::Vector3f &light_dir) const {
    glUniform3fv(world_light_dir_index_, 1, light_dir.data());
  }
  void setLightProjMatrix(const Eigen::Matrix4f &light_proj_matrix) const {
    glUniformMatrix4fv(light_proj_matrix_index_, 1, GL_FALSE,
                       light_proj_matrix.data());
  }
  void setLightModelViewMatrices(
      const Eigen::Matrix<float, 4, Eigen::Dynamic> &light_mv_matrices) const {
    GLsizei count = light_mv_matrices.cols() / 4;
    glUniformMatrix4fv(light_model_view_matrices_index_, count, GL_FALSE,
                       light_mv_matrices.data());
  }
  void setLightColor(const Eigen::Vector3f &light_color) const {
    glUniform3fv(light_color_index_, 1, light_color.data());
  }
  void setShadowMapTextureUnit(GLint unit) const {
    glUniform1i(shadow_map_index_, unit);
  }
  void setMSDFTextureUnit(GLint unit) const { glUniform1i(msdf_index_, unit); }
  void setCascadeFarSplits(const Eigen::Vector4f &far_splits) const {
    glUniform4fv(cascade_far_splits_index_, 1, far_splits.data());
  }

  GLuint program_;
  GLuint vertex_shader_;
  GLuint fragment_shader_;
  GLint proj_matrix_index_;
  GLint view_matrix_index_;
  GLint tex_coords_matrix_index_;
  GLint model_view_matrix_index_;
  GLint normal_matrix_index_;
  GLint proc_texture_type_index_;
  GLint object_color_index_;
  GLint world_light_dir_index_;
  GLint light_proj_matrix_index_;
  GLint light_model_view_matrices_index_;
  GLint light_color_index_;
  GLint shadow_map_index_;
  GLint msdf_index_;
  GLint cascade_far_splits_index_;
};

struct Mesh {
  Mesh(GLenum usage = GL_DYNAMIC_DRAW);
  virtual ~Mesh();
  Mesh(const Mesh &other) = delete;
  Mesh &operator=(const Mesh &other) = delete;
  void bind() const { glBindVertexArray(vertex_array_); }
  void setPositions(const std::vector<GLfloat> &positions);
  void setNormals(const std::vector<GLfloat> &normals);
  void setTexCoords(const std::vector<GLfloat> &tex_coords);
  void setIndices(const std::vector<GLint> &indices);
  void draw() const {
    glDrawElements(GL_TRIANGLES, index_count_, GL_UNSIGNED_INT, 0);
  }

  GLenum usage_;
  GLuint vertex_array_;
  GLuint position_buffer_;
  GLuint normal_buffer_;
  GLuint tex_coord_buffer_;
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
  void bind() const { glBindTexture(target_, texture_); }
  void setParameter(GLenum name, GLint value) const;
  void getImage(unsigned char *pixels) const;

  GLenum target_;
  GLuint texture_;
};

struct Texture3D {
  Texture3D(GLenum target, GLint level, GLint internal_format, GLsizei width,
            GLsizei height, GLsizei depth, GLenum format, GLenum type,
            const GLvoid *data = NULL);
  virtual ~Texture3D();
  Texture3D(const Texture3D &other) = delete;
  Texture3D &operator=(const Texture3D &other) = delete;
  void bind() const { glBindTexture(target_, texture_); }
  void setParameter(GLenum name, GLint value) const;
  void getImage(unsigned char *pixels) const;

  GLenum target_;
  GLuint texture_;
};

struct Framebuffer {
  Framebuffer();
  virtual ~Framebuffer();
  Framebuffer(const Framebuffer &other) = delete;
  Framebuffer &operator=(const Framebuffer &other) = delete;
  void bind() const { glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_); }
  void attachColorTexture(const Texture2D &color_texture) const;
  void attachColorTextureLayer(const Texture3D &color_texture,
                               GLint layer) const;
  void attachDepthTexture(const Texture2D &depth_texture) const;
  void attachDepthTextureLayer(const Texture3D &depth_texture,
                               GLint layer) const;

  GLuint framebuffer_;
};

struct CameraParameters {
  Eigen::Matrix4f getProjMatrix() const;
  Eigen::Matrix4f getViewMatrix() const;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  float aspect_ratio_ = 1.0f;
  float z_near_ = 0.1f;
  float z_far_ = 100.0f;
  float fov_ = M_PI / 3;
  Eigen::Vector3f position_ = Eigen::Vector3f::Zero();
  float yaw_ = 0.0f;
  float pitch_ = -M_PI / 6;
  float distance_ = 2.0f;
};

struct RenderParameters {
  // Cornflower blue
  Eigen::Vector4f background_color_ = Eigen::Vector4f(0.4f, 0.6f, 0.8f, 1.0f);
};

struct DirectionalLight {
  DirectionalLight() {}
  DirectionalLight(const Eigen::Vector3f &color, const Eigen::Vector3f &dir,
                   const Eigen::Vector3f &up, GLsizei sm_width,
                   GLsizei sm_height, int sm_cascade_count);
  void updateViewMatricesAndSplits(const Eigen::Matrix4f &camera_view_matrix,
                                   float aspect_ratio, float z_near,
                                   float z_far, float fov);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Eigen::Vector3f color_;
  Eigen::Vector3f dir_;
  GLsizei sm_width_;
  GLsizei sm_height_;
  int sm_cascade_count_;
  Eigen::Matrix3f view_rot_matrix_;
  Eigen::Matrix4f proj_matrix_;
  Eigen::Matrix<float, 4, Eigen::Dynamic> view_matrices_;
  Eigen::VectorXf sm_cascade_splits_;
  std::shared_ptr<Texture3D> sm_depth_array_texture_;
  std::shared_ptr<Framebuffer> sm_framebuffer_;
};

struct BitmapFontChar {
  char char_;
  unsigned int width_;
  unsigned int height_;
  int xoffset_;
  int yoffset_;
  int xadvance_;
  int x_;
  int y_;
  unsigned int page_;
};

struct BitmapFont {
  explicit BitmapFont(const std::string &path, const std::string &resource_dir);
  unsigned int getStringWidth(const std::string &str) const;

  unsigned int line_height_ = 0;
  unsigned int base_ = 0;
  unsigned int page_width_ = 0;
  unsigned int page_height_ = 0;
  std::unordered_map<char, BitmapFontChar> chars_;
  std::vector<std::shared_ptr<Texture2D>> page_textures_;
};

template <typename T> struct ProgramParameter {
  ProgramParameter() : dirty_(false) {}
  void setValue(const T &value) {
    value_ = value;
    dirty_ = true;
  }

  T value_;
  bool dirty_;
};

struct ProgramState {
  ProgramState() {}
  void setProjectionMatrix(const Eigen::Matrix4f &proj_matrix) {
    proj_matrix_.setValue(proj_matrix);
  }
  void setViewMatrix(const Eigen::Matrix4f &view_matrix) {
    view_matrix_.setValue(view_matrix);
  }
  void setModelMatrix(const Eigen::Matrix4f &model_matrix) {
    model_matrix_.setValue(model_matrix);
  }
  void setTexCoordsMatrix(const Eigen::Matrix4f &tex_coords_matrix) {
    tex_coords_matrix_.setValue(tex_coords_matrix);
  }
  void setProcTextureType(int proc_texture_type) {
    proc_texture_type_.setValue(proc_texture_type);
  }
  void setObjectColor(const Eigen::Vector3f &object_color) {
    object_color_.setValue(object_color);
  }
  void setDirectionalLight(const DirectionalLight &dir_light) {
    dir_light_color_.setValue(dir_light.color_);
    dir_light_dir_.setValue(dir_light.dir_);
    dir_light_proj_matrix_.setValue(dir_light.proj_matrix_);
    dir_light_view_matrices_.setValue(dir_light.view_matrices_);
    dir_light_sm_cascade_splits_.setValue(dir_light.sm_cascade_splits_);
  }
  void setDirectionalLightViewMatrices(
      const Eigen::Matrix<float, 4, Eigen::Dynamic> &view_matrices) {
    dir_light_view_matrices_.setValue(view_matrices);
  }
  void updateUniforms(const Program &program);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  ProgramParameter<Eigen::Matrix4f> proj_matrix_;
  ProgramParameter<Eigen::Matrix4f> view_matrix_;
  ProgramParameter<Eigen::Matrix4f> model_matrix_;
  ProgramParameter<Eigen::Matrix4f> tex_coords_matrix_;
  ProgramParameter<int> proc_texture_type_;
  ProgramParameter<Eigen::Vector3f> object_color_;
  ProgramParameter<Eigen::Vector3f> dir_light_color_;
  ProgramParameter<Eigen::Vector3f> dir_light_dir_;
  ProgramParameter<Eigen::Matrix4f> dir_light_proj_matrix_;
  ProgramParameter<Eigen::Matrix<float, 4, Eigen::Dynamic>>
      dir_light_view_matrices_;
  ProgramParameter<Eigen::VectorXf> dir_light_sm_cascade_splits_;
};

class GLRenderer {
public:
  GLRenderer(const std::string &data_dir);
  void render(const Simulation &sim, const CameraParameters &camera_params,
              const RenderParameters &render_params, int width, int height,
              const Framebuffer *target_framebuffer = nullptr);

private:
  void drawOpaque(const Simulation &sim, const Program &program,
                  ProgramState &program_state) const;
  void drawLabels(const Simulation &sim, const Program &program,
                  ProgramState &program_state) const;
  void drawBox(const Eigen::Matrix4f &transform,
               const Eigen::Vector3f &half_extents, const Program &program,
               ProgramState &program_state) const;
  void drawCapsule(const Eigen::Matrix4f &transform, float half_length,
                   float radius, const Program &program,
                   ProgramState &program_state) const;
  void drawCylinder(const Eigen::Matrix4f &transform, float half_length,
                    float radius, const Program &program,
                    ProgramState &program_state) const;
  void drawTubeBasedShape(const Eigen::Matrix4f &transform, float half_length,
                          float radius, const Program &program,
                          ProgramState &program_state,
                          const Mesh &end_mesh) const;
  void drawText(const Eigen::Matrix4f &transform, float half_height,
                const Program &program, ProgramState &program_state,
                const std::string &str) const;
  void drawHeightfield(const Eigen::Matrix4f &transform,
                       const Eigen::Vector3f &half_extents,
                       const Program &program, ProgramState &program_state,
                       const Eigen::MatrixXf &heightfield) const;
  static std::string loadString(const std::string &path);

  std::shared_ptr<Program> default_program_;
  std::shared_ptr<Program> flat_program_;
  std::shared_ptr<Program> depth_program_;
  std::shared_ptr<Program> msdf_program_;
  std::shared_ptr<Mesh> box_mesh_;
  std::shared_ptr<Mesh> tube_mesh_;
  std::shared_ptr<Mesh> capsule_end_mesh_;
  std::shared_ptr<Mesh> cylinder_end_mesh_;
  std::shared_ptr<Mesh> text_mesh_;
  std::shared_ptr<Mesh> heightfield_mesh_;
  std::shared_ptr<DirectionalLight> dir_light_;
  std::shared_ptr<BitmapFont> font_;
};

class Viewer {
public:
  virtual ~Viewer() {}
  virtual void update(double dt) = 0;
  virtual void render(const Simulation &sim,
                      unsigned char *pixels = nullptr) = 0;
  virtual void getFramebufferSize(int &width, int &height) const = 0;
  virtual void setFramebufferSize(int width, int height) = 0;
};

void makeOrthographicProjection(float aspect_ratio, float z_near, float z_far,
                                Ref<Eigen::Matrix4f> matrix);

void makePerspectiveProjection(float aspect_ratio, float z_near, float z_far,
                               float fov, Ref<Eigen::Matrix4f> matrix);

std::shared_ptr<Mesh> makeBoxMesh();

std::shared_ptr<Mesh> makeTubeMesh(int n_segments);

std::shared_ptr<Mesh> makeCapsuleEndMesh(int n_segments, int n_rings);

std::shared_ptr<Mesh> makeCylinderEndMesh(int n_segments);

std::shared_ptr<Texture2D> loadTexture(const std::string &path);

} // namespace robot_design
