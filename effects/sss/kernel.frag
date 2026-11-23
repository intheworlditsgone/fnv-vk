// Subsurface scattering, pass 0: the separable blur's 21 taps, as Community Shaders' SubsurfaceScattering::CalculateKernel
// (GPL-3.0; after Jimenez et al.'s Separable SSS) computes them for its human profile (strength 0.48 0.41 0.28, falloff
// 1 0.37 0.3): offsets from -3 to 3 spaced quadratically, weights the d'Eon skin profile (its red channel, the falloff
// widening it per channel) times each tap's span, normalised, then blended toward no blur by the strength. Texel i: rgb
// its weight, a its offset; the centre tap first.
layout(location = 0) out vec4 outKernel;

const int N = 21;
const float RANGE = 3.0;
const vec3 kStrength = vec3(0.48, 0.41, 0.28), kFalloff = vec3(1.0, 0.37, 0.3);

vec3 gaussian(float variance, float r) {
  const vec3 rr = r / (0.001 + kFalloff);
  return exp(-(rr * rr) / (2.0 * variance)) / (2.0 * 3.14 * variance);
}
vec3 profile(float r) {
  // (0.233 * gaussian(0.0064, r), the directly bounced light, is the strength's share)
  return 0.100 * gaussian(0.0484, r) + 0.118 * gaussian(0.187, r) + 0.113 * gaussian(0.567, r) +
         0.358 * gaussian(1.99, r) + 0.078 * gaussian(7.41, r);
}
float offsetOf(int i) {  // before reordering
  const float o = -RANGE + float(i) * (2.0 * RANGE / float(N - 1));
  return RANGE * (o < 0.0 ? -1.0 : 1.0) * (o * o) / (RANGE * RANGE);
}
vec4 tap(int i) {  // before reordering and normalisation
  const float w0 = i > 0 ? abs(offsetOf(i) - offsetOf(i - 1)) : 0.0;
  const float w1 = i < N - 1 ? abs(offsetOf(i) - offsetOf(i + 1)) : 0.0;
  return vec4((w0 + w1) * 0.5 * profile(offsetOf(i)), offsetOf(i));
}

void main() {
  const int k = int(gl_FragCoord.x);
  // Reordered: the centre (N / 2) first, then 0 .. N / 2 - 1, then the rest in place.
  const int src = k == 0 ? N / 2 : k <= N / 2 ? k - 1 : k;
  vec3 sum = vec3(0.0);
  for (int i = 0; i < N; i++) sum += tap(i).rgb;
  vec4 t = tap(src);
  t.rgb /= sum;
  t.rgb = k == 0 ? mix(vec3(1.0), t.rgb, kStrength) : t.rgb * kStrength;
  outKernel = t;
}
