#version 150 core

const int SHADOW_MAP_CASCADE_COUNT = 5;

in vec3 model_position;
in vec3 model_normal;

out vec3 view_normal;
out vec3 view_light_dir;
out vec4 light_frag_pos[SHADOW_MAP_CASCADE_COUNT];

uniform mat4 proj_matrix;
uniform mat4 view_matrix;
uniform mat4 model_view_matrix;
uniform mat3 normal_matrix;
uniform vec3 world_light_dir = vec3(0.0, 1.0, 0.0);
uniform mat4 light_proj_matrix;
uniform mat4 light_model_view_matrices[SHADOW_MAP_CASCADE_COUNT];

void main() {
  gl_Position = proj_matrix * model_view_matrix * vec4(model_position, 1.0);
  view_normal = normalize(normal_matrix * model_normal);
  view_light_dir = (view_matrix * vec4(world_light_dir, 0.0)).xyz;
  for (int i = 0; i < SHADOW_MAP_CASCADE_COUNT; ++i) {
    light_frag_pos[i] = light_proj_matrix * light_model_view_matrices[i] *
                        vec4(model_position, 1.0);
  }
}
