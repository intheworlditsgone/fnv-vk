// Karis weights suppress bright outliers. Apply the same weights to RGB and the miss fraction.
// img_gidepth still holds the previous frame's depth until gidepth.frag runs.
layout(location = 0) out vec4 outGi;

float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
  const ivec2 h = ivec2(gl_FragCoord.xy);
  const ivec2 pix = ivec2(fnvPixel());
  if (!fnvInsideRect(pix) || fnv.depthRange.w == 0.0) { outGi = vec4(0.0, 0.0, 0.0, 1.0); return; }
  const float d = fnvRawDepth(pix);
  if (fnvIsSky(d)) { outGi = vec4(0.0, 0.0, 0.0, 1.0); return; }
  const float zc = fnvLinearZ(d);
  const ivec2 size = ivec2(fnvPass.size.xy);
  vec4 sum = vec4(0.0);
  float weights = 0.0;
  for (int y = -2; y <= 2; y++)
    for (int x = -2; x <= 2; x++) {
      const ivec2 hq = clamp(h + ivec2(x, y), ivec2(0), size - 1);
      const ivec2 q = fnvClampRect(ivec2((vec2(hq) + 0.5) / fnvPass.info.x));
      const float z = fnvLinearZ(fnvRawDepth(q));
      const vec4 s = texelFetch(sampler2D(img_giraw, fnvPoint), hq, 0);
      const float w = max(0.0, 1.0 - abs(z - zc) / (0.04 * zc + 1.0)) / (1.0 + luma(s.rgb));
      sum += s * w;
      weights += w;
    }
  const vec4 current = weights > 0.0 ? sum / weights : texelFetch(sampler2D(img_giraw, fnvPoint), h, 0);

  float alpha = 1.0;
  vec4 history = current;
  vec2 prev;
  float prevZ;
  if (fnvPrevPixel(fnvViewPos(vec2(pix) + 0.5, zc), prev, prevZ)) {
    const ivec2 pp = ivec2(floor(prev));
    const ivec2 hp = clamp(ivec2(floor(prev * fnvPass.info.x)), ivec2(0), size - 1);
    if (fnvInsideRect(pp) && abs(texelFetch(sampler2D(img_gidepth, fnvPoint), hp, 0).r - log2(prevZ)) <= 0.03) {
      history = texelFetch(sampler2D(img_gihist, fnvPoint), hp, 0);
      alpha = clamp(p_giResponse, 0.02, 1.0);
    }
  }
  outGi = mix(history, current, alpha);
}
