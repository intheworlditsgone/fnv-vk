// Volumetric fog, pass 1 (full resolution): r = the sun's in-scattered light from the half-resolution march: the four
// nearest half-resolution texels, bilinear weights times a depth weight that is 1 at this pixel's view distance and 0
// beyond 5% from it (no light across silhouettes; smooth depth, the sky and slanted ground, interpolates);
// g = the transmittance from the camera to this pixel, exact (march.frag's medium and optical depth).
// Indoors, under water or without fog: no light, transmittance 1.
layout(location = 0) out vec2 outFog;

float opticalDepth(float sigmaC, float k, float s) {
  const float x = k * s;
  return abs(x) < 1e-4 ? sigmaC * s * (1.0 - 0.5 * x) : sigmaC * (1.0 - exp(-min(x, 80.0))) / k;
}

void main() {
  outFog = vec2(0.0, 1.0);
  const ivec2 pix = ivec2(gl_FragCoord.xy);
  if (!fnvInsideRect(pix) || fnv.shadowInfo.z != 0.0 || fnv.depthRange.z > 0.0 || p_density <= 0.0) return;
  const float d = fnvRawDepth(pix);
  const vec3 ray = fnvViewPos(vec2(pix) + 0.5, 1.0);
  const float rayLen = length(ray);
  const float zEnd = fnvIsSky(d) ? p_range / rayLen : fnvLinearZ(d);

  const vec3 dirWorld = fnvWorldPos(ray / rayLen) - fnvCameraWorldPos();
  const float H = max(p_falloff, 1.0);
  const float sigmaC = p_density * 1.0e-4 * exp(p_baseHeight / H);
  outFog.y = exp(-opticalDepth(sigmaC, dirWorld.z / H, zEnd * rayLen));

  // The half-resolution texels around this pixel (texel centres at (2i + 1) full-resolution pixels).
  const ivec2 size = textureSize(sampler2D(img_scatter, fnvPoint), 0);
  const vec2 h = (vec2(pix) + 0.5) * 0.5 - 0.5;
  const ivec2 base = ivec2(floor(h));
  const vec2 f = h - vec2(base);
  const float zc = min(zEnd, 60000.0);
  float sum = 0.0, weights = 0.0;
  for (int y = 0; y < 2; y++)
    for (int x = 0; x < 2; x++) {
      const ivec2 q = clamp(base + ivec2(x, y), ivec2(0), size - 1);
      const vec2 s = texelFetch(sampler2D(img_scatter, fnvPoint), q, 0).rg;
      const float bilinear = (x == 0 ? 1.0 - f.x : f.x) * (y == 0 ? 1.0 - f.y : f.y);
      const float w = bilinear * max(1.0 - abs(s.g - zc) / (0.05 * zc + 1.0), 1e-3);
      sum += s.r * w;
      weights += w;
    }
  outFog.x = weights > 0.0 ? sum / weights : 0.0;
}
