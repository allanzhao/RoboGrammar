#version 150 core

in vec3 model_position;
in vec2 model_tex_coord;

out vec2 texture_coords;

uniform mat4 proj_matrix;
uniform mat4 model_view_matrix;

void main() {
  // Use only the translational component from model_view_matrix, so text always
  // faces the camera
  vec3 model_view_offset = model_view_matrix[3].xyz;
  gl_Position = proj_matrix * vec4(model_position + model_view_offset, 1.0);
  texture_coords = model_tex_coord;
}
