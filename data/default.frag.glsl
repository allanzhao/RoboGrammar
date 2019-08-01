#version 150 core

in vec3 view_normal;

out vec4 out_color;

void main() {
  out_color = vec4(view_normal, 1.0);
}
