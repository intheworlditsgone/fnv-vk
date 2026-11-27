// Depth of field, pass 7(half resolution): the near layer, the blur of what lies in front of the focus, which spreads
// over whatever is behind it. Scatter as gather over the largest near radius that reaches this tile (dilate.frag): a
// sample in front of the focus counts where its own blur reaches this pixel. Each of the n taps stands for 1 / n of
// the gather disc's area (pi R^2), and a sample spreads its light over its own disc (pi r^2): it covers this pixel by
// R^2 / (n r^2). The coverage sums to 1 inside a blurred near surface and falls off past its edge over its blur radius.
// Out: rgb the near layer's colour premultiplied by its coverage, a the coverage (bilinear upsampling stays correct).
layout(location = 0) out vec4 outNear;

void main() {
  if (!p_nearBlur) {
    outNear = vec4(0.0);
    return;
  }
  const ivec2 h = ivec2(gl_FragCoord.xy);
  const ivec2 size = textureSize(sampler2D(img_half, fnvPoint), 0);
  const ivec2 tiles = textureSize(sampler2D(img_near, fnvPoint), 0);
  const float radius = 0.5 * texelFetch(sampler2D(img_near, fnvPoint), min((h * tiles) / size, tiles - 1), 0).r;
  if (radius < 0.5) {  // (under a full-resolution pixel: no sample would count)
    outNear = vec4(0.0);
    return;
  }
  const int n = clamp(p_samples, 4, 128);
  const float rot = 6.2831853 * fract(52.9829189 * fract(dot(vec2(h), vec2(0.06711056, 0.00583715))));
  const ivec2 lo = ivec2(fnv.rect.xy) / 2, hi = max(ivec2(fnv.rect.xy + fnv.rect.zw) / 2, lo + 1) - 1;
  vec3 sum = vec3(0.0);
  float coverage = 0.0;
  for (int i = 0; i < n; i++) {
    const float a = float(i) * 2.39996323 + rot;  // golden angle
    const vec2 o = radius * sqrt((float(i) + 0.5) / float(n)) * vec2(cos(a), sin(a));
    const ivec2 q = clamp(ivec2(floor(vec2(h) + 0.5 + o)), max(lo, ivec2(0)), min(hi, size - 1));
    const vec4 s = texelFetch(sampler2D(img_half, fnvPoint), q, 0);
    if (s.a > -1.0) continue;  // in focus (under a pixel; also the depth's rounding about the focus) or behind it
    const float r = -0.5 * s.a;
    const float w = clamp(r - length(o) + 1.0, 0.0, 1.0) * (radius * radius) / (float(n) * r * r);
    sum += s.rgb * w;
    coverage += w;
  }
  const float a = min(coverage, 1.0);
  outNear = coverage > 0.0 ? vec4(sum / coverage * a, a) : vec4(0.0);
}
