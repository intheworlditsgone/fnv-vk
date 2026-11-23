// Auto-exposure, pass 2 (1 x 1): this frame's metered log2 luminance (the weighted mean of the 16 x 16 blocks), and
// the adapted one: the previous frame's moved toward it by 1 - exp(-speed * dt), with dt the seconds since this
// effect's previous run (fnv.time.z) and the speed toward brighter or darker scenes; the measured value itself on the
// first run (dt 0) or when the previous value is not a usable number.
layout(location = 0) out float outAdapt;

void main() {
  float sum = 0.0, weights = 0.0;
  for (int y = 0; y < 16; y++)
    for (int x = 0; x < 16; x++) {
      const vec2 b = texelFetch(sampler2D(img_aeblocks, fnvPoint), ivec2(x, y), 0).rg;
      sum += b.r * b.g;
      weights += b.g;
    }
  const float measured = weights > 0.0 ? sum / weights : log2(0.18);
  const float prev = texelFetch(sampler2D(img_aeprev, fnvPoint), ivec2(0), 0).r;
  const float dt = fnv.time.z;
  float adapted = measured;
  if (dt > 0.0 && abs(prev) < 30.0) {
    const float speed = measured > prev ? p_aeSpeedUp : p_aeSpeedDown;
    adapted = prev + (measured - prev) * (1.0 - exp(-max(speed, 0.0) * dt));
  }
  outAdapt = adapted;
}
