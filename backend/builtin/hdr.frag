#version 450
// HDR present. The finished frame is gamma 2.2 encoded with 1.0 = SDR white (paper white); tonemapped highlights are
// above 1.0 (wide targets). The menu layer is premultiplied and gamma encoded like SDR content. Both are decoded to
// light relative to paper white, scaled to nits, limited to the peak, and encoded for the swapchain:
//   HDR10: BT.709 -> BT.2020 primaries, SMPTE ST 2084 (PQ) of nits / 10000.
//   scRGB: linear BT.709, 1.0 = 80 nits.
layout(set = 1, binding = 16) uniform texture2D src;
layout(set = 1, binding = 17) uniform texture2D menu;
layout(set = 1, binding = 32) uniform sampler smp;
// x: output (1 = HDR10, 2 = scRGB), y: paper white nits, z: peak nits, w: 1 = menu layer present
layout(push_constant) uniform Hdr { layout(offset = 16) vec4 p; } pc;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;

// ITU-R BT.2087 (BT.709 to BT.2020, linear light); columns are the BT.709 R, G, B primaries.
const mat3 kBt709To2020 = mat3(0.627404, 0.069097, 0.016391,
                               0.329283, 0.919540, 0.088013,
                               0.043313, 0.011362, 0.895595);

vec3 pq(vec3 nits) {  // SMPTE ST 2084 inverse EOTF
  const float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
  vec3 y = pow(clamp(nits / 10000.0, 0.0, 1.0), vec3(m1));
  return pow((c1 + c2 * y) / (1.0 + c3 * y), vec3(m2));
}

void main() {
  vec3 c = texture(sampler2D(src, smp), uv).rgb;
  if (pc.p.w != 0.0) {
    vec4 m = texture(sampler2D(menu, smp), uv);
    c = c * (1.0 - m.a) + m.rgb;
  }
  vec3 nits = min(pow(max(c, 0.0), vec3(2.2)) * pc.p.y, vec3(pc.p.z));
  if (pc.p.x == 1.0) color = vec4(pq(kBt709To2020 * nits), 1.0);
  else color = vec4(nits / 80.0, 1.0);
}
