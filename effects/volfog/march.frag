// Volumetric fog, pass 0 (half resolution): the sun's light scattered toward the camera along the view ray.
//
// The medium: extinction sigma(h) = sigma0 exp(-(h - base) / H) with h the world height (z), base the camera's height
// plus the base height parameter, H the falloff. Along a ray from the camera with unit direction d (world z component
// b), sigma(s) = sigma_c exp(-k s), k = b / H, sigma_c = sigma0 exp(-(h_cam - base) / H); the optical depth to s is
// sigma_c (1 - exp(-k s)) / k (sigma_c s when k s is tiny) and the transmittance T(s) = exp(-optical depth).
// Single scattering with albedo 1: the light scattered toward the camera between s0 and s1 from a source of constant
// visibility is (T(s0) - T(s1)) times its radiance (d T / d s = -sigma T, exactly). The sun's radiance there is
// phase(view . sun) * pi * sunColor (the game's light colour is the diffuse light on a white surface facing it, so the
// irradiance is pi times it); the sun's visibility comes from the shadow cascades inside their range, 1 beyond.
// Without cascades (night, shadows off) there is no sunlight; indoors and under water there is no fog (apply.frag).
// Output: r = the sun's in-scattered light without the sun's colour, g = the pixel's view distance (upsample.frag).
// The same optical depth as upsample.frag.
layout(location = 0) out vec2 outScatter;

float opticalDepth(float sigmaC, float k, float s) {
  const float x = k * s;
  return abs(x) < 1e-4 ? sigmaC * s * (1.0 - 0.5 * x) : sigmaC * (1.0 - exp(-min(x, 80.0))) / k;
}

// Visibility of one view-space point: one compare tap (a hardware 2 x 2 filter) in the cascade containing it.
float sunVisibility(vec3 p) {
  const int n = int(fnv.shadowInfo.x);
  int c = 0;
  while (c < n - 1 && p.z > fnv.shadowSplits[c]) c++;
  const vec4 q = fnv.shadowFromView[c] * vec4(p, 1.0);
  return texture(sampler2DArrayShadow(fnvShadowTex, fnvShadowCmp), vec4(q.xy * 0.5 + 0.5, float(c), q.z));
}

void main() {
  const ivec2 pix = fnvClampRect(ivec2(fnvPixel()));
  const float d = fnvRawDepth(pix);
  const bool sky = fnvIsSky(d);
  const vec3 ray = fnvViewPos(vec2(pix) + 0.5, 1.0);  // view-space direction; z is the view distance per unit along it
  const float rayLen = length(ray);
  // View distance (z) of the end of the ray: the surface, or the range over the sky.
  const float zEnd = sky ? p_range / rayLen : fnvLinearZ(d);
  outScatter = vec2(0.0, min(zEnd, 60000.0));  // half-float range
  if (fnv.shadowInfo.x == 0.0 || fnv.shadowInfo.z != 0.0 || fnv.depthRange.z > 0.0 || p_density <= 0.0) return;

  const vec3 dirWorld = fnvWorldPos(ray / rayLen) - fnvCameraWorldPos();
  const float H = max(p_falloff, 1.0);
  const float k = dirWorld.z / H;
  const float sigmaC = p_density * 1.0e-4 * exp(p_baseHeight / H);  // the camera is base height below the base
  const float sEnd = zEnd * rayLen;
  const vec3 dir = ray / rayLen;
  const float cosTheta = dot(dir, normalize(fnv.sunView.xyz));
  const float g = p_anisotropy;
  const float phase = (1.0 - g * g) / (12.566371 * pow(max(1.0 + g * g - 2.0 * g * cosTheta, 1e-4), 1.5));

  // Inside the shadow range: segments of equal view distance, the visibility sampled at a jittered point in each
  // (interleaved gradient noise: the steps' banding becomes fine noise that the upsampling's neighbours average).
  const int n = int(fnv.shadowInfo.x);
  const float zShadow = min(zEnd, fnv.shadowSplits[n - 1]);
  const int steps = max(p_steps, 1);
  const float jitter = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  const float dz = zShadow / float(steps);
  float scatter = 0.0;
  float tPrev = 1.0;
  for (int i = 0; i < steps; i++) {
    const float z1 = float(i + 1) * dz;
    const float t1 = exp(-opticalDepth(sigmaC, k, z1 * rayLen));
    scatter += sunVisibility(ray * ((float(i) + jitter) * dz)) * (tPrev - t1);
    tPrev = t1;
  }
  // Beyond the shadow range: lit.
  scatter += tPrev - exp(-opticalDepth(sigmaC, k, sEnd));
  outScatter.x = scatter * phase * 3.14159265 * p_sun;
}
