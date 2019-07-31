#version 150 core

in vec3 frag_normal;

out vec4 out_color;

void main() {
  out_color = vec4(frag_normal, 1.0);
}
