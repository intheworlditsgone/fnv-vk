// GTAO denoise, at half resolution: the 4x4 neighbourhood of texels (one tile of the slice rotation), weighted by depth
// similarity so occlusion does not bleed across depth discontinuities. Each texel stands for the full-resolution pixel
// under its centre (gtao.frag: fnvPixel), whose depth it is weighed by.
layout(location = 0) out vec2 outVisibility;  // visibility, log2 of the view distance

// The full-resolution pixel of texel t of this pass's grid (the same as gtao.frag's fnvPixel there).
ivec2 pixelOf(ivec2 t) { return ivec2((vec2(t) + 0.5) / fnvPass.info.x); }

void main() {
  const ivec2 h = ivec2(gl_FragCoord.xy);
  const ivec2 pix = pixelOf(h);
  if (!fnvInsideRect(pix)) { outVisibility = vec2(1.0, 0.0); return; }
  // The texels gtao.frag wrote: the camera viewport's part of the image (as the backend scales it).
  const ivec2 lo = ivec2(fnv.rect.xy * fnvPass.info.x);
  const ivec2 hi = lo + max(ivec2(fnv.rect.zw * fnvPass.info.x), ivec2(1)) - 1;
  const float zc = fnvLinearZ(fnvRawDepth(pix));
  float sum = 0.0, weights = 0.0;
  for (int y = -1; y <= 2; y++)
    for (int x = -1; x <= 2; x++) {
      const ivec2 q = clamp(h + ivec2(x, y), lo, hi);
      const float z = fnvLinearZ(fnvRawDepth(fnvClampRect(pixelOf(q))));
      const float w = max(0.0, 1.0 - abs(z - zc) / (0.04 * zc + 1.0));
      sum += texelFetch(sampler2D(img_raw, fnvPoint), q, 0).r * w;
      weights += w;
    }
  outVisibility = vec2(weights > 0.0 ? sum / weights : texelFetch(sampler2D(img_raw, fnvPoint), h, 0).r, log2(zc));
}
