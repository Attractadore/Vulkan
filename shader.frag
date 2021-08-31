#version 450

layout(location = 0) in vec3 vcolor;
layout(location = 0) out vec4 fcolor;

void main() {
    fcolor = vec4(vcolor, 1.0f);
}
