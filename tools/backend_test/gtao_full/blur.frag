// GTAO denoise: the 4x4 neighbourhood (one tile of the slice rotation), weighted by depth similarity so occlusion
// does not bleed across depth discontinuities.
layout(location = 0) out vec2 outVisibility;  // visibility, log2 of the view distance

void main() {
  const ivec2 pix = ivec2(gl_FragCoord.xy);
  if (!fnvInsideRect(pix)) { outVisibility = vec2(1.0, 0.0); return; }
  const float zc = fnvLinearZ(fnvRawDepth(pix));
  float sum = 0.0, weights = 0.0;
  for (int y = -1; y <= 2; y++)
    for (int x = -1; x <= 2; x++) {
      const ivec2 q = fnvClampRect(pix + ivec2(x, y));
      const float z = fnvLinearZ(fnvRawDepth(q));
      const float w = max(0.0, 1.0 - abs(z - zc) / (0.04 * zc + 1.0));
      sum += texelFetch(sampler2D(img_raw, fnvPoint), q, 0).r * w;
      weights += w;
    }
  outVisibility = vec2(weights > 0.0 ? sum / weights : texelFetch(sampler2D(img_raw, fnvPoint), pix, 0).r, log2(zc));
}
