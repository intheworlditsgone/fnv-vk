#version 450
// Screen-space textured quad (D3D9 pretransformed XYZRHW|TEX1 strip, as Bink's Submit_vertices draws it).
// rect: NDC x0, y0, x1, y1. Strip order matches Submit_vertices: (0,0) (0,1) (1,0) (1,1) in uv.
layout(push_constant) uniform PC { vec4 rect; } pc;
layout(location = 0) out vec2 uv;
void main() {
  vec2 t = vec2(float(gl_VertexIndex >> 1), float(gl_VertexIndex & 1));
  uv = t;
  gl_Position = vec4(mix(pc.rect.xy, pc.rect.zw, t), 0.0, 1.0);
}
