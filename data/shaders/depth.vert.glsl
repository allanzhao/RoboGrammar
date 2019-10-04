#version 150 core

in vec3 model_position;

uniform mat4 proj_matrix;
uniform mat4 model_view_matrix;

void main() {
  gl_Position = proj_matrix * model_view_matrix * vec4(model_position, 1.0);
}
