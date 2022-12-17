#version 150 core

const int SHADOW_MAP_CASCADE_COUNT = 5;

in vec3 texture_coords;
in vec3 view_pos;
in vec3 view_normal;
in vec3 view_light_dir;
in vec4 light_frag_pos[SHADOW_MAP_CASCADE_COUNT];

out vec4 frag_color;

uniform int proc_texture_type = 0;
uniform vec3 object_color = vec3(0.5, 0.5, 0.5);
uniform vec3 light_color = vec3(1.0, 1.0, 1.0);
uniform sampler2DArrayShadow shadow_map;
uniform vec4 cascade_far_splits;

const vec3 view_camera_dir = vec3(0.0, 0.0, 1.0);

vec4 computeShadowMapCoords(int cascade_idx) {
  vec3 proj_light_frag_pos =
      light_frag_pos[cascade_idx].xyz / light_frag_pos[cascade_idx].w;
  vec4 shadow_map_coords;
  shadow_map_coords.xyw = 0.5 * proj_light_frag_pos + 0.5;
  shadow_map_coords.z = cascade_idx;
  return shadow_map_coords;
}

float computeShadowFactor(int cascade_idx) {
  vec4 sm_coords = computeShadowMapCoords(cascade_idx);
  vec4 closer_sm_coords = computeShadowMapCoords(max(cascade_idx - 1, 0));
  // If within bounds of a closer shadow map level, use it instead
  bool closer_in_bounds = fract(closer_sm_coords.xy) == closer_sm_coords.xy;
  return texture(shadow_map, closer_in_bounds ? closer_sm_coords : sm_coords);
}

vec2 tri(vec2 x) {
  // Triangle wave
  return 1.0 - 2.0 * abs(fract(0.5 * x) - 0.5);
}

float checkerboardGrad(vec2 p, vec2 dpdx, vec2 dpdy) {
  // Antialiased procedural checkerboard texture
  // https://www.iquilezles.org/www/articles/checkerfiltering/checkerfiltering.htm
  vec2 w = max(abs(dpdx), abs(dpdy)) + 1e-3; // Width of filter kernel
  vec2 i = (tri(p + 0.5 * w) - tri(p - 0.5 * w)) / w; // Integral of square wave
  return 0.5 - 0.5 * i.x * i.y; // XOR, rescale to [0.0, 1.0] range
}

float procTextureGrad(vec3 p, vec3 dpdx, vec3 dpdy, int type) {
  switch (type) {
  case 0:
  default:
    return 1.0; // Solid color
  case 1:
    return checkerboardGrad(p.yz, dpdx.yz, dpdy.yz); // Checkerboard in YZ plane
  case 2:
    return checkerboardGrad(p.xz, dpdx.xz, dpdy.xz); // Checkerboard in XZ plane
  case 3:
    return checkerboardGrad(p.xy, dpdx.xy, dpdy.xy); // Checkerboard in XY plane
  }
}

void main() {
  vec3 normal = normalize(view_normal);
  vec3 reflect_dir = reflect(-view_light_dir, normal);
  int cascade_idx = int(dot(
      vec4(greaterThan(-view_pos.zzzz, cascade_far_splits)), vec4(1.0)));
  float shadow_factor = computeShadowFactor(cascade_idx);

  vec3 ambient = 0.5 * light_color;
  vec3 diffuse = 0.5 * max(dot(normal, view_light_dir), 0.0) * light_color *
                 shadow_factor;
  float spec_factor = pow(max(dot(view_camera_dir, reflect_dir), 0.0), 32);
  vec3 specular = 0.25 * spec_factor * light_color * shadow_factor;

  vec3 p = 2.0 * texture_coords;
  float color_factor = 0.9 + 0.1 * procTextureGrad(p, dFdx(p), dFdy(p),
                                                   proc_texture_type);
  vec3 texture_color = object_color * color_factor;
  frag_color = vec4((ambient + diffuse + specular) * texture_color, 1.0);
}
