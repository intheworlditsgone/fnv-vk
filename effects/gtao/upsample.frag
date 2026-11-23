// GTAO, the half-resolution visibility (blur.frag) at every full-resolution pixel: the four nearest texels, bilinear
// weights times depth similarity (each texel's g is the log2 view distance of the pixel it was computed for), so
// occlusion does not cross silhouettes. Where none of the four is at this pixel's depth (a feature thinner than a
// texel), the one nearest in depth. Output as the published AO image: visibility, log2 of this pixel's view distance.
layout(location = 0) out vec2 outVisibility;

const float kScale = 0.5;  // effect.ini: [Image.half] Scale

void main() {
  const ivec2 pix = ivec2(gl_FragCoord.xy);
  if (!fnvInsideRect(pix)) { outVisibility = vec2(1.0, 0.0); return; }
  const float d = fnvRawDepth(pix);
  const float zc = fnvLinearZ(d);
  if (fnvIsSky(d)) { outVisibility = vec2(1.0, log2(zc)); return; }  // (gtao.frag: the sky is unoccluded)
  // The half-resolution texels blur.frag wrote (the camera viewport's part, as the backend scales it).
  const ivec2 lo = ivec2(fnv.rect.xy * kScale);
  const ivec2 hi = lo + max(ivec2(fnv.rect.zw * kScale), ivec2(1)) - 1;
  // Texel t's centre is full-resolution position (t + 0.5) / kScale.
  const vec2 hp = (vec2(pix) + 0.5) * kScale - 0.5;
  const ivec2 base = ivec2(floor(hp));
  const vec2 f = hp - vec2(base);
  float sum = 0.0, weights = 0.0, depthWeights = 0.0, bestDiff = 1e30, best = 1.0;
  for (int y = 0; y < 2; y++)
    for (int x = 0; x < 2; x++) {
      const vec2 s = texelFetch(sampler2D(img_half, fnvPoint), clamp(base + ivec2(x, y), lo, hi), 0).rg;
      const float diff = abs(exp2(s.g) - zc);
      const float dw = max(0.0, 1.0 - diff / (0.04 * zc + 1.0));  // the blur's depth weight
      const float bilinear = (x == 0 ? 1.0 - f.x : f.x) * (y == 0 ? 1.0 - f.y : f.y);
      sum += s.r * bilinear * dw;
      weights += bilinear * dw;
      depthWeights += dw;
      if (diff < bestDiff) { bestDiff = diff; best = s.r; }
    }
  outVisibility = vec2(weights > 1e-4 && depthWeights > 1e-2 ? sum / weights : best, log2(zc));
}
