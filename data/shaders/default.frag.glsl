#version 150 core

const int SHADOW_MAP_CASCADE_COUNT = 5;

in vec3 view_pos;
in vec3 view_normal;
in vec3 view_light_dir;
in vec4 light_frag_pos[SHADOW_MAP_CASCADE_COUNT];

out vec4 frag_color;

uniform vec3 object_color = vec3(0.5, 0.5, 0.5);
uniform vec3 light_color = vec3(1.0, 1.0, 1.0);
uniform sampler2DArrayShadow shadow_map;
uniform vec4 cascade_far_splits;

const vec3 view_camera_dir = vec3(0.0, 0.0, 1.0);

float computeShadowFactor(vec4 light_frag_pos, int cascade_idx) {
  vec3 proj_light_frag_pos = light_frag_pos.xyz / light_frag_pos.w;
  vec4 shadow_map_coords;
  shadow_map_coords.xyw = 0.5 * proj_light_frag_pos + 0.5;
  shadow_map_coords.z = cascade_idx;
  return texture(shadow_map, shadow_map_coords);
}

void main() {
  vec3 normal = normalize(view_normal);
  vec3 reflect_dir = reflect(-view_light_dir, normal);
  int cascade_idx = int(dot(
      vec4(greaterThan(-view_pos.zzzz, cascade_far_splits)), vec4(1.0)));
  float shadow_factor = computeShadowFactor(
      light_frag_pos[cascade_idx], cascade_idx);

  vec3 ambient = 0.2 * light_color;
  vec3 diffuse = max(dot(normal, view_light_dir), 0.0) * light_color *
                 shadow_factor;
  float spec_factor = pow(max(dot(view_camera_dir, reflect_dir), 0.0), 32);
  vec3 specular = 0.5 * spec_factor * light_color * shadow_factor;

  frag_color = vec4((ambient + diffuse + specular) * object_color, 1.0);
}
