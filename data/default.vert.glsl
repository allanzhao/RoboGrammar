#version 150 core

in vec3 model_position;
in vec3 model_normal;

out vec3 view_normal;
out vec3 view_light_dir;

uniform mat4 proj_matrix;
uniform mat4 view_matrix;
uniform mat4 model_view_matrix;
uniform mat3 normal_matrix;
uniform vec3 world_light_dir = vec3(0.0, 1.0, 0.0);

void main() {
  gl_Position = proj_matrix * model_view_matrix * vec4(model_position, 1.0);
  view_normal = normalize(normal_matrix * model_normal);
  view_light_dir = (view_matrix * vec4(world_light_dir, 0.0)).xyz;
}
