#version 450
// Present with the game's gamma ramp (IDirect3DDevice9::SetGammaRamp from Renderer::UpdateGamma): every 8-bit output
// channel value v becomes ramp[v]. lut is 256x1, its texel v holding (ramp.r[v], ramp.g[v], ramp.b[v]) >> 8.
layout(set = 1, binding = 16) uniform texture2D src;
layout(set = 1, binding = 17) uniform texture2D lut;
layout(set = 1, binding = 32) uniform sampler smp;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() {
  vec3 c = clamp(texture(sampler2D(src, smp), uv).rgb, 0.0, 1.0);
  ivec3 i = ivec3(c * 255.0 + 0.5);
  color = vec4(texelFetch(sampler2D(lut, smp), ivec2(i.r, 0), 0).r, texelFetch(sampler2D(lut, smp), ivec2(i.g, 0), 0).g,
               texelFetch(sampler2D(lut, smp), ivec2(i.b, 0), 0).b, 1.0);
}
