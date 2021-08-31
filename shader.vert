#version 450

const vec2 positions[3] = vec2[](
    vec2( 0.0f, -0.5f),
    vec2( 0.5f,  0.5f),
    vec2(-0.5f,  0.5f)
);

const vec3 colors[3] = vec3[](
    vec3(1.0f, 0.0f, 0.0f),
    vec3(0.0f, 1.0f, 0.0f),
    vec3(0.0f, 0.0f, 1.0f)
);

layout(location = 0) out vec3 vcolor;

void main() {
    const uint i = gl_VertexIndex;
    gl_Position = vec4(positions[i], 0.0f, 1.0f);
    vcolor = colors[i];
}
