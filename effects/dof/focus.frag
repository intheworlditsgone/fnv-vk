// Depth of field, pass 0 (1 x 1): the focus distance in game units.
// Autofocus: the log-average view distance of a 7 x 7 grid over a square of p_focusArea % of the image height at the
// centre of the view, Gaussian weighted toward its centre (sky counts as the far plane); the focus moves toward it in
// log distance by 1 - exp(-speed * dt), dt the seconds since the previous run (fnv.time.z; the measured value itself
// without one). Without autofocus: p_focusDistance metres.
layout(location = 0) out float outFocus;

const float kUnitsPerMetre = 1.0 / 0.0142875;  // 64 units per yard

void main() {
  if (!p_autofocus) {
    outFocus = p_focusDistance * kUnitsPerMetre;
    return;
  }
  const vec2 centre = fnv.rect.xy + 0.5 * fnv.rect.zw;
  const float step = 0.01 * p_focusArea * fnv.rect.w / 6.0;
  float sum = 0.0, weights = 0.0;
  for (int y = -3; y <= 3; y++)
    for (int x = -3; x <= 3; x++) {
      const ivec2 p = fnvClampRect(ivec2(centre + vec2(x, y) * step));
      const float w = exp(-float(x * x + y * y) / 8.0);
      sum += w * log2(max(fnvLinearZ(fnvRawDepth(p)), fnv.depthRange.x));
      weights += w;
    }
  const float measured = sum / weights;
  const float prev = texelFetch(sampler2D(img_focusprev, fnvPoint), ivec2(0), 0).r;
  const float dt = fnv.time.z;
  float focus = measured;
  if (dt > 0.0 && prev > 0.0 && prev < 1e9) {
    const float lp = log2(prev);
    focus = lp + (measured - lp) * (1.0 - exp(-max(p_focusSpeed, 0.0) * dt));
  }
  outFocus = exp2(focus);
}
