#version 150 core

in vec3 position;
in vec3 normal;

out vec3 frag_normal;

uniform mat4 proj_matrix;
uniform mat4 view_matrix;
uniform mat4 model_matrix;

void main() {
  gl_Position = proj_matrix * view_matrix * model_matrix * vec4(position, 1.0);
  frag_normal = normal;
}
