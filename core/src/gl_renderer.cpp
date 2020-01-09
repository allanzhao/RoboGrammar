#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cstddef>
#include <fstream>
#include <memory>
#include <robot_design/render.h>
#include <sstream>
#include <string>
#include <vector>

namespace robot_design {

GLRenderer::GLRenderer(const std::string &data_dir) {
  // Create default shader program
  std::string default_vs_source = loadString(data_dir +
                                             "shaders/default.vert.glsl");
  std::string default_fs_source = loadString(data_dir +
                                             "shaders/default.frag.glsl");
  default_program_ =
      std::make_shared<Program>(default_vs_source, default_fs_source);

  // Create depth shader program
  std::string depth_vs_source = loadString(data_dir +
                                           "shaders/depth.vert.glsl");
  std::string depth_fs_source; // Empty string (no fragment shader needed)
  depth_program_ = std::make_shared<Program>(depth_vs_source, depth_fs_source);

  // Create MSDF text shader program
  std::string msdf_vs_source = loadString(data_dir + "shaders/msdf.vert.glsl");
  std::string msdf_fs_source = loadString(data_dir + "shaders/msdf.frag.glsl");
  msdf_program_ = std::make_shared<Program>(msdf_vs_source, msdf_fs_source);

  // Create meshes
  box_mesh_ = makeBoxMesh();
  tube_mesh_ = makeTubeMesh(/*n_segments=*/32);
  capsule_end_mesh_ = makeCapsuleEndMesh(/*n_segments=*/32, /*n_rings=*/8);
  cylinder_end_mesh_ = makeCylinderEndMesh(/*n_segments=*/32);
  // drawText will create new vertex data for each string
  text_mesh_ = std::make_shared<Mesh>(/*usage=*/GL_STREAM_DRAW);

  // Create directional light
  dir_light_ = std::make_shared<DirectionalLight>(
      /*color=*/Eigen::Vector3f{1.0f, 1.0f, 1.0f},
      /*dir=*/Eigen::Vector3f{1.0f, 2.0f, 3.0f},
      /*up=*/Eigen::Vector3f{0.0f, 1.0f, 0.0f},
      /*sm_width=*/2048, /*sm_height=*/2048, /*sm_cascade_count=*/5);

  // Load font
  font_ = std::make_shared<BitmapFont>(data_dir + "fonts/OpenSans-Regular.fnt",
                                       data_dir + "fonts");

  // Enable depth test
  glEnable(GL_DEPTH_TEST);

  // Enable alpha blending
  glEnable(GL_BLEND);
  // Use premultiplied alpha
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
}

void GLRenderer::render(const Simulation &sim,
                        const CameraParameters &camera_params, int width,
                        int height, const Framebuffer *target_framebuffer) {
  Eigen::Matrix4f proj_matrix = camera_params.getProjMatrix();
  Eigen::Matrix4f view_matrix = camera_params.getViewMatrix();
  dir_light_->updateViewMatricesAndSplits(
      view_matrix, camera_params.aspect_ratio_, camera_params.z_near_,
      camera_params.z_far_, camera_params.fov_);

  // Render shadow maps
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
    drawOpaque(sim, *depth_program_, depth_program_state);
  }

  // Render camera view, using the shadow maps as input
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
  default_program_state.setProjectionMatrix(proj_matrix);
  default_program_state.setViewMatrix(view_matrix);
  default_program_state.setDirectionalLight(*dir_light_);
  dir_light_->sm_depth_array_texture_->bind();
  drawOpaque(sim, *default_program_, default_program_state);

  // Render label text with depth testing/writing turned off
  glDisable(GL_DEPTH_TEST);
  glDepthMask(GL_FALSE);
  msdf_program_->use();
  ProgramState msdf_program_state;
  msdf_program_state.setProjectionMatrix(proj_matrix);
  msdf_program_state.setViewMatrix(view_matrix);
  font_->page_textures_.at(0)->bind();
  drawLabels(sim, *msdf_program_, msdf_program_state);
  glDepthMask(GL_TRUE);
  glEnable(GL_DEPTH_TEST);
}

void GLRenderer::drawOpaque(const Simulation &sim, const Program &program,
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
                    link.radius_, program, program_state);
        break;
      case LinkShape::CYLINDER:
        drawCylinder(link_transform.cast<float>(), link.length_ / 2,
                     link.radius_, program, program_state);
        break;
      default:
        throw std::runtime_error("Unexpected link shape");
      }

      // Draw the link's joint
      program_state.setObjectColor(link.joint_color_);
      Matrix3 joint_axis_rotation(
          Quaternion::FromTwoVectors(link.joint_axis_, Vector3::UnitX()));
      Matrix4 joint_transform =
          (Affine3(link_transform) * Translation3(-link.length_ / 2, 0, 0) *
           Affine3(joint_axis_rotation))
              .matrix();
      float joint_size = 1.05 * link.radius_;
      switch (link.joint_type_) {
      case JointType::FREE:
        // Nothing to draw
        break;
      case JointType::HINGE:
        drawCylinder(joint_transform.cast<float>(), joint_size, joint_size,
                     program, program_state);
        break;
      case JointType::FIXED:
        drawBox(joint_transform.cast<float>(),
                Eigen::Vector3f::Constant(joint_size), program, program_state);
        break;
      default:
        throw std::runtime_error("Unexpected joint type");
      }
    }
  }

  // Draw props
  for (Index prop_idx = 0; prop_idx < sim.getPropCount(); ++prop_idx) {
    const Prop &prop = *sim.getProp(prop_idx);
    program_state.setObjectColor(prop.color_);
    Matrix4 prop_transform;
    sim.getPropTransform(prop_idx, prop_transform);
    drawBox(prop_transform.cast<float>(), prop.half_extents_.cast<float>(),
            program, program_state);
  }
}

void GLRenderer::drawLabels(const Simulation &sim, const Program &program,
                            ProgramState &program_state) const {
  // Draw robot labels
  for (Index robot_idx = 0; robot_idx < sim.getRobotCount(); ++robot_idx) {
    const Robot &robot = *sim.getRobot(robot_idx);
    for (std::size_t link_idx = 0; link_idx < robot.links_.size(); ++link_idx) {
      const Link &link = robot.links_[link_idx];
      Matrix4 link_transform;
      sim.getLinkTransform(robot_idx, link_idx, link_transform);

      // Draw the link's label, if it has one
      if (!link.label_.empty()) {
        drawText(link_transform.cast<float>(), link.radius_, program,
                 program_state, link.label_);
      }

      // Draw the joint's label, if it has one
      if (!link.joint_label_.empty()) {
        Matrix4 joint_transform =
            (Affine3(link_transform) * Translation3(-link.length_ / 2, 0, 0))
                .matrix();
        drawText(joint_transform.cast<float>(), link.radius_, program,
                 program_state, link.joint_label_);
      }
    }
  }
}

void GLRenderer::drawBox(const Eigen::Matrix4f &transform,
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

void GLRenderer::drawCapsule(const Eigen::Matrix4f &transform,
                             float half_length, float radius,
                             const Program &program,
                             ProgramState &program_state) const {
  drawTubeBasedShape(transform, half_length, radius, program, program_state,
                     *capsule_end_mesh_);
}

void GLRenderer::drawCylinder(const Eigen::Matrix4f &transform,
                              float half_length, float radius,
                              const Program &program,
                              ProgramState &program_state) const {
  drawTubeBasedShape(transform, half_length, radius, program, program_state,
                     *cylinder_end_mesh_);
}

void GLRenderer::drawTubeBasedShape(const Eigen::Matrix4f &transform,
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

void GLRenderer::drawText(const Eigen::Matrix4f &transform, float half_height,
                          const Program &program, ProgramState &program_state,
                          const std::string &str) const {
  std::vector<float> positions;
  std::vector<float> tex_coords;
  std::vector<int> indices;
  float xy_scale = 2.0f * half_height / font_->line_height_;
  float u_scale = 1.0f / font_->page_width_;
  float v_scale = 1.0f / font_->page_height_;
  float x = -0.5f * font_->getStringWidth(str) * xy_scale;
  float y = -half_height;
  int j = 0; // Vertex index
  for (char c : str) {
    const auto it = font_->chars_.find(c);
    if (it != font_->chars_.end()) {
      const BitmapFontChar &char_info = it->second;
      float x_min = x + char_info.xoffset_ * xy_scale;
      float x_max = x_min + char_info.width_ * xy_scale;
      float y_max = y + (font_->line_height_ - char_info.yoffset_) * xy_scale;
      float y_min = y_max - char_info.height_ * xy_scale;
      float u_min = char_info.x_ * u_scale;
      float u_max = u_min + char_info.width_ * u_scale;
      float v_max = (font_->page_height_ - char_info.y_) * v_scale;
      float v_min = v_max - char_info.height_ * v_scale;
      // Reverse v coordinates, since textures are loaded upside-down
      v_min = 1.0f - v_min;
      v_max = 1.0f - v_max;
      // Define a textured rectangle
      // clang-format off
      float pos[12] = {x_min, y_min, 0.0f, x_max, y_min, 0.0f,
                       x_max, y_max, 0.0f, x_min, y_max, 0.0f};
      float tex_coord[8] = {u_min, v_min, u_max, v_min,
                            u_max, v_max, u_min, v_max};
      int idx[6] = {j, j + 1, j + 2, j + 3, j, j + 2};
      // clang-format on
      positions.insert(positions.end(), std::begin(pos), std::end(pos));
      tex_coords.insert(tex_coords.end(), std::begin(tex_coord),
                        std::end(tex_coord));
      indices.insert(indices.end(), std::begin(idx), std::end(idx));
      x += char_info.xadvance_ * xy_scale;
      j += 4;
    }
  }
  text_mesh_->setPositions(positions);
  text_mesh_->setTexCoords(tex_coords);
  text_mesh_->setIndices(indices);

  text_mesh_->bind();
  program_state.setModelMatrix(transform.matrix());
  program_state.updateUniforms(program);
  text_mesh_->draw();
}

std::string GLRenderer::loadString(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs.is_open()) {
    throw std::runtime_error("Could not open file \"" + path + "\"");
  }
  std::stringstream ss;
  ss << ifs.rdbuf();
  return ss.str();
}

} // namespace robot_design
