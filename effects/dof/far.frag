// Depth of field, pass 6(half resolution): the far layer, the blur of what lies behind the focus. Scatter as gather:
// the pixel's own blur disc is sampled (a Vogel spiral, p_samples taps), and a sample counts where its own blur, behind
// the focus too, reaches this pixel (Jimenez 2014): an in-focus edge in front adds nothing to the background's blur.
// Radii are in full-resolution pixels (half of them in this image).
layout(location = 0) out vec4 outFar;

void main() {
  const ivec2 h = ivec2(gl_FragCoord.xy);
  const ivec2 size = textureSize(sampler2D(img_half, fnvPoint), 0);
  const vec4 c0 = texelFetch(sampler2D(img_half, fnvPoint), h, 0);
  if (c0.a <= 0.5) {  // in focus or in front of it: the far layer is not used here (composite.frag)
    outFar = vec4(c0.rgb, 1.0);
    return;
  }
  const float radius = 0.5 * c0.a;
  const int n = clamp(p_samples, 4, 128);
  const float rot = 6.2831853 * fract(52.9829189 * fract(dot(vec2(h), vec2(0.06711056, 0.00583715))));
  const ivec2 lo = ivec2(fnv.rect.xy) / 2, hi = max(ivec2(fnv.rect.xy + fnv.rect.zw) / 2, lo + 1) - 1;
  vec3 sum = c0.rgb;
  float weights = 1.0;
  for (int i = 0; i < n; i++) {
    const float a = float(i) * 2.39996323 + rot;  // golden angle
    const vec2 o = radius * sqrt((float(i) + 0.5) / float(n)) * vec2(cos(a), sin(a));
    const ivec2 q = clamp(ivec2(floor(vec2(h) + 0.5 + o)), max(lo, ivec2(0)), min(hi, size - 1));
    const vec4 s = texelFetch(sampler2D(img_half, fnvPoint), q, 0);
    if (s.a < 1.0) continue;  // in focus (under a pixel) or in front of it
    const float w = clamp(0.5 * s.a - length(o) + 1.0, 0.0, 1.0);
    sum += s.rgb * w;
    weights += w;
  }
  outFar = vec4(sum / weights, 1.0);
}
