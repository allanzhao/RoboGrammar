#version 150 core

in vec2 texture_coords;

out vec4 frag_color;

uniform vec3 object_color = vec3(0.5, 0.5, 0.5);
uniform sampler2D msdf;

float median(float r, float g, float b) {
  return max(min(r, g), min(max(r, g), b));
}

void main() {
  vec3 sample = texture(msdf, texture_coords).rgb;
  float signed_dist = median(sample.r, sample.g, sample.b) - 0.5;
  float alpha = clamp(signed_dist / fwidth(signed_dist) + 0.5, 0.0, 1.0);
  frag_color = vec4(object_color, 1.0) * alpha;
}
