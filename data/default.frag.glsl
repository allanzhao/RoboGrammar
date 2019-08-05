#version 150 core

in vec3 view_normal;
in vec3 view_light_dir;

out vec4 frag_color;

uniform vec3 object_color = vec3(0.5, 0.5, 0.5);
uniform vec3 light_color = vec3(1.0, 1.0, 1.0);

const vec3 view_camera_dir = vec3(0.0, 0.0, 1.0);

void main() {
  vec3 normal = normalize(view_normal);
  vec3 reflect_dir = reflect(-view_light_dir, normal);

  vec3 ambient = 0.1 * light_color;
  vec3 diffuse = max(dot(normal, view_light_dir), 0.0) * light_color;
  float spec_factor = pow(max(dot(view_camera_dir, reflect_dir), 0.0), 32);
  vec3 specular = 0.5 * spec_factor * light_color;

  frag_color = vec4((ambient + diffuse + specular) * object_color, 1.0);
}
