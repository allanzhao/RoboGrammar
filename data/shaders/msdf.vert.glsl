#version 150 core

in vec3 model_position;
in vec2 model_tex_coord;

out vec2 texture_coords;

uniform mat4 proj_matrix;
uniform mat4 model_view_matrix;

void main() {
  gl_Position = proj_matrix * model_view_matrix * vec4(model_position, 1.0);
  texture_coords = model_tex_coord;
}
