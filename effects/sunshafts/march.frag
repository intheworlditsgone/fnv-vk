// Sun shafts, pass 0 (half resolution): single scattering of sunlight along the view ray through a uniform medium.
// The ray runs from the camera to the scene (the sky: to the end of the shadow range); at each step the sun's
// visibility comes from the shadow cascades. Light scattered at distance t reaches the camera attenuated by
// exp(-sigma t):
//   S = sum_i vis(t_i) * sigma * exp(-sigma t_i) * dt * phase(view . sun)
// Past the shadow range nothing is added (no shafts form there; the game's own fog and sun glare cover the far air).
// Output: r = S without the sun's colour (apply.frag adds it), g = the pixel's view distance (for the upsampling).
layout(location = 0) out vec2 outScatter;

// Visibility of one view-space point: one compare tap (a hardware 2 x 2 filter) in the cascade containing it.
float sunVisibility(vec3 p) {
  int n = int(fnv.shadowInfo.x);
  int c = 0;
  while (c < n - 1 && p.z > fnv.shadowSplits[c]) c++;
  vec4 q = fnv.shadowFromView[c] * vec4(p, 1.0);
  return texture(sampler2DArrayShadow(fnvShadowTex, fnvShadowCmp), vec4(q.xy * 0.5 + 0.5, float(c), q.z));
}

void main() {
  const ivec2 pix = fnvClampRect(ivec2(fnvPixel()));
  const float d = fnvRawDepth(pix);
  const float dist = fnvIsSky(d) ? 1.0e9 : fnvLinearZ(d);
  outScatter = vec2(0.0, min(dist, 60000.0));  // half-float range (sky: far)
  // No cascades (indoors, night, shadows off) or the camera under water: nothing.
  if (fnv.shadowInfo.x == 0.0 || fnv.depthRange.z > 0.0) return;
  const int n = int(fnv.shadowInfo.x);
  // View-space direction of this pixel; z is the view distance per unit along it.
  const vec3 ray = fnvViewPos(vec2(pix) + 0.5, 1.0);
  const float range = min(dist, fnv.shadowSplits[n - 1]);  // view distance (z) the march covers
  if (range <= 0.0) return;
  const vec3 dir = normalize(ray);
  const float cosTheta = dot(dir, normalize(fnv.sunView.xyz));
  const float g = p_anisotropy;
  const float phase = (1.0 - g * g) / (12.566371 * pow(max(1.0 + g * g - 2.0 * g * cosTheta, 1e-4), 1.5));
  const float sigma = p_density * 1.0e-4;  // per game unit
  // Steps in view distance; the start is jittered per pixel (interleaved gradient noise) so the steps' banding becomes
  // fine noise the blur removes.
  const float jitter = fract(52.9829189 * fract(dot(vec2(pix), vec2(0.06711056, 0.00583715))));
  const int steps = p_steps;
  const float dz = range / float(steps);
  const float dt = dz * length(ray);  // length along the ray of one step
  float sum = 0.0;
  for (int i = 0; i < steps; i++) {
    const float z = (float(i) + jitter) * dz;
    const vec3 p = ray * z;
    sum += sunVisibility(p) * exp(-sigma * z * length(ray));
  }
  outScatter.x = sum * sigma * dt * phase;
}
