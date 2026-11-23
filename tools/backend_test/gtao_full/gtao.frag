// GTAO: per pixel, a few slices through the view vector; in each, the highest horizon on both sides is found by marching
// the depth buffer, and the cosine-weighted visible arc relative to the normal projected into the slice is integrated
// analytically (Jimenez et al., "Practical Real-Time Strategies for Accurate Indirect Occlusion", 2016).
layout(location = 0) out float outVisibility;

const float PI = 3.14159265358979;
const float HALF_PI = 1.57079632679490;

void main() {
  const ivec2 pix = ivec2(gl_FragCoord.xy);
  if (!fnvInsideRect(pix)) { outVisibility = 1.0; return; }
  const float d = fnvRawDepth(pix);
  if (fnvIsSky(d)) { outVisibility = 1.0; return; }
  const vec3 P = fnvViewPos(vec2(pix) + 0.5, fnvLinearZ(d));
  if (P.z >= p_fadeEnd) { outVisibility = 1.0; return; }

  // Normal from depth: per axis the neighbour with the smaller depth step (clean silhouettes); at the viewport border
  // only the inner neighbour exists.
  const bool hasL = fnvInsideRect(pix + ivec2(-1, 0)), hasR = fnvInsideRect(pix + ivec2(1, 0));
  const bool hasT = fnvInsideRect(pix + ivec2(0, -1)), hasB = fnvInsideRect(pix + ivec2(0, 1));
  const vec3 pl = fnvViewPosAt(pix + ivec2(-1, 0)), pr = fnvViewPosAt(pix + ivec2(1, 0));
  const vec3 pt = fnvViewPosAt(pix + ivec2(0, -1)), pb = fnvViewPosAt(pix + ivec2(0, 1));
  const vec3 dx = (hasR && (!hasL || abs(pr.z - P.z) < abs(P.z - pl.z))) ? pr - P : P - pl;
  const vec3 dy = (hasB && (!hasT || abs(pb.z - P.z) < abs(P.z - pt.z))) ? pb - P : P - pt;
  const vec3 V = normalize(-P);
  vec3 N = normalize(cross(dx, dy));
  if (dot(N, V) < 0.0) N = -N;

  const float radius = max(p_radius, 1.0);
  const float pixelsPerUnit = fnv.rect.z / ((fnv.frustum.y - fnv.frustum.x) * P.z);
  const float radiusPx = min(radius * pixelsPerUnit, 256.0);
  if (radiusPx < 2.0) { outVisibility = 1.0; return; }

  // 4x4 interleaved slice rotation (the blur averages one tile) and a per-pixel step jitter.
  const int tile = (pix.x & 3) | ((pix.y & 3) << 2);
  const float sliceNoise = float((tile * 7) & 15) / 16.0;
  const float stepNoise = fract(52.9829189 * fract(dot(vec2(pix), vec2(0.06711056, 0.00583715))));
  const int slices = clamp(p_slices, 1, 8), steps = clamp(p_steps, 1, 16);
  const float falloffRange = 0.6 * radius;

  // Slices spread evenly around the view vector, each mapped to the screen direction its plane projects to.
  const vec3 T = normalize(cross(V, abs(V.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
  const vec3 B = cross(V, T);
  const vec2 pixelsPerTan = vec2(fnv.rect.z / (fnv.frustum.y - fnv.frustum.x), fnv.rect.w / (fnv.frustum.z - fnv.frustum.w));
  float visibility = 0.0;
  for (int s = 0; s < slices; s++) {
    const float phi = (float(s) + sliceNoise) / float(slices) * PI;
    const vec3 orthoDir = cos(phi) * T + sin(phi) * B;
    const vec2 screenStep = vec2(orthoDir.x - P.x / P.z * orthoDir.z, -(orthoDir.y - P.y / P.z * orthoDir.z)) * pixelsPerTan;
    const vec2 screenDir = screenStep / max(length(screenStep), 1e-6);
    const vec3 axis = normalize(cross(orthoDir, V));
    const vec3 projN = N - axis * dot(N, axis);
    const float projNLen = length(projN);
    const float cosN = clamp(dot(projN, V) / max(projNLen, 1e-4), 0.0, 1.0);
    const float n = sign(dot(orthoDir, projN)) * acos(cosN);
    const float low0 = cos(n + HALF_PI), low1 = cos(n - HALF_PI);
    float horizonCos0 = low0, horizonCos1 = low1;
    for (int j = 0; j < steps; j++) {
      float t = (float(j) + stepNoise) / float(steps);
      t *= t;
      const vec2 offset = screenDir * (1.3 + t * (radiusPx - 1.3));
      // Outside the viewport: skipped (clamping would move the sample off the slice: a false horizon). Positions are
      // rebuilt at the exact screen offset so samples stay on the slice; only the depth comes from the texel.
      const vec2 f0 = vec2(pix) + 0.5 + offset, f1 = vec2(pix) + 0.5 - offset;
      const ivec2 q0 = ivec2(floor(f0)), q1 = ivec2(floor(f1));
      if (fnvInsideRect(q0)) {
        const vec3 d0 = fnvViewPos(f0, fnvLinearZ(fnvRawDepth(q0))) - P;
        const float l0 = length(d0);
        const float w0 = clamp((radius - l0) / falloffRange, 0.0, 1.0);
        horizonCos0 = max(horizonCos0, mix(low0, dot(d0, V) / max(l0, 1e-4), w0));
      }
      if (fnvInsideRect(q1)) {
        const vec3 d1 = fnvViewPos(f1, fnvLinearZ(fnvRawDepth(q1))) - P;
        const float l1 = length(d1);
        const float w1 = clamp((radius - l1) / falloffRange, 0.0, 1.0);
        horizonCos1 = max(horizonCos1, mix(low1, dot(d1, V) / max(l1, 1e-4), w1));
      }
    }
    float h0 = -acos(clamp(horizonCos1, -1.0, 1.0));
    float h1 = acos(clamp(horizonCos0, -1.0, 1.0));
    h0 = n + clamp(h0 - n, -HALF_PI, HALF_PI);
    h1 = n + clamp(h1 - n, -HALF_PI, HALF_PI);
    const float sinN = sin(n);
    const float arc0 = (cosN + 2.0 * h0 * sinN - cos(2.0 * h0 - n)) * 0.25;
    const float arc1 = (cosN + 2.0 * h1 * sinN - cos(2.0 * h1 - n)) * 0.25;
    visibility += projNLen * (arc0 + arc1);
  }
  visibility = clamp(visibility / float(slices), 0.0, 1.0);
  const float fade = 1.0 - smoothstep(p_fadeStart, max(p_fadeEnd, p_fadeStart + 1.0), P.z);
  outVisibility = mix(1.0, pow(visibility, max(p_intensity, 0.01)), fade);
}
