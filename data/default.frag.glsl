#version 150 core

in vec3 view_normal;

out vec4 out_color;

void main() {
  vec3 object_color = vec3(0.5, 0.5, 0.5);
  out_color = vec4(normalize(view_normal).z * object_color + 0.25, 1.0);
}
