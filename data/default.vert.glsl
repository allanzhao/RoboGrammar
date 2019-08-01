#version 150 core

in vec3 position;
in vec3 normal;

out vec3 view_normal;

uniform mat4 proj_matrix;
uniform mat4 model_view_matrix;
uniform mat3 normal_matrix;

void main() {
  gl_Position = proj_matrix * model_view_matrix * vec4(position, 1.0);
  view_normal = normalize(normal_matrix * normal);
}
