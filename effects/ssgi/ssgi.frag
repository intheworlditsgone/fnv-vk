// Screen-space indirect light (docs/INDIRECT.md "SSGI"), at half resolution: from each pixel's surface (its view position
// and the normal reconstructed from depth, as gtao.frag), cosine-distributed rays about the normal are marched through
// the depth buffer in view space; the first point that lies behind the visible surface (and not far behind it) is a
// hit, and the ray takes that pixel's lit colour (the frame's scene colour, decoded from gamma 2.2 to linear light).
// The mean over the rays of the light they found (0 for a miss) is irradiance / pi, the units of the game's light
// colours, which the lighting multiplies by the surface's albedo; a is the fraction of rays that found nothing (outside
// the screen, beyond the ray length, or the open sky), which the lighting gives the probe's ambient. Without the
// lighting using it (fnv.depthRange.w 0; effect.ini Requires=pbr already keeps the effect from running then), nothing is
// computed.
layout(location = 0) out vec4 outGi;

const float PI = 3.14159265358979;

// The full-resolution pixel a view-space point projects to (the inverse of fnvViewPos); false outside the viewport.
bool project(vec3 q, out ivec2 pix) {
  const vec2 uv = vec2((q.x / q.z - fnv.frustum.x) / (fnv.frustum.y - fnv.frustum.x),
                       (q.y / q.z - fnv.frustum.z) / (fnv.frustum.w - fnv.frustum.z));
  pix = ivec2(floor(fnv.rect.xy + uv * fnv.rect.zw));
  return all(greaterThanEqual(uv, vec2(0.0))) && all(lessThan(uv, vec2(1.0)));
}

void main() {
  outGi = vec4(0.0, 0.0, 0.0, 1.0);
  if (fnv.depthRange.w == 0.0) return;
  const ivec2 pix = ivec2(fnvPixel());
  if (!fnvInsideRect(pix)) return;
  const float d = fnvRawDepth(pix);
  if (fnvIsSky(d)) return;
  const vec3 P = fnvViewPos(vec2(pix) + 0.5, fnvLinearZ(d));
  if (P.z >= p_giFadeEnd) return;

  // Normal from depth (gtao.frag).
  const bool hasL = fnvInsideRect(pix + ivec2(-1, 0)), hasR = fnvInsideRect(pix + ivec2(1, 0));
  const bool hasT = fnvInsideRect(pix + ivec2(0, -1)), hasB = fnvInsideRect(pix + ivec2(0, 1));
  const vec3 pl = fnvViewPosAt(pix + ivec2(-1, 0)), pr = fnvViewPosAt(pix + ivec2(1, 0));
  const vec3 pt = fnvViewPosAt(pix + ivec2(0, -1)), pb = fnvViewPosAt(pix + ivec2(0, 1));
  const vec3 dx = (hasR && (!hasL || abs(pr.z - P.z) < abs(P.z - pl.z))) ? pr - P : P - pl;
  const vec3 dy = (hasB && (!hasT || abs(pb.z - P.z) < abs(P.z - pt.z))) ? pb - P : P - pt;
  vec3 N = normalize(cross(dx, dy));
  if (dot(N, -P) < 0.0) N = -N;
  const vec3 T = normalize(cross(N, abs(N.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0)));
  const vec3 B = cross(N, T);

  // Per-pixel rotation from a 4x4 tile (the denoiser averages across it) and a step jitter, both advanced every frame
  // by golden-ratio steps (the temporal accumulation then averages other directions each frame).
  const ivec2 h = ivec2(gl_FragCoord.xy);
  const int tile = (h.x & 3) | ((h.y & 3) << 2);
  const float frame = mod(fnv.time.y, 1024.0);
  const float rot = fract(float((tile * 7) & 15) / 16.0 + 0.618034 * frame);
  const float jitter = fract(52.9829189 * fract(dot(vec2(h), vec2(0.06711056, 0.00583715))) + 0.754878 * frame);
  const int rays = clamp(p_giRays, 1, 16), steps = clamp(p_giSteps, 2, 32);
  const float radius = max(p_giRadius, 1.0);
  const float pixelSize = (fnv.frustum.y - fnv.frustum.x) * P.z / fnv.rect.z;  // view units per pixel at P
  const vec3 origin = P + N * (2.0 * pixelSize);  // off the surface, against self-hits from the depth quantisation

  vec3 sum = vec3(0.0);
  float misses = 0.0;
  for (int r = 0; r < rays; r++) {
    // Cosine-distributed direction: a stratified azimuth (rotated per pixel) and a golden-ratio sequence for the radius.
    const float u1 = (float(r) + rot) / float(rays), u2 = fract(0.5 + 0.618034 * float(r) + jitter);
    const float phi = 2.0 * PI * u1, sinT = sqrt(u2), cosT = sqrt(1.0 - u2);
    const vec3 dir = T * (cos(phi) * sinT) + B * (sin(phi) * sinT) + N * cosT;
    bool hit = false;
    for (int s = 0; s < steps && !hit; s++) {
      float t = (float(s) + jitter) / float(steps);
      t = t * t * radius + 2.0 * pixelSize;
      const vec3 q = origin + dir * t;
      if (q.z <= fnv.depthRange.x) break;  // in front of the near plane: a miss
      ivec2 qp;
      if (!project(q, qp) || !fnvInsideRect(qp)) break;  // left the screen: a miss
      const float ds = fnvRawDepth(qp);
      if (fnvIsSky(ds)) continue;
      const float zs = fnvLinearZ(ds);
      if (q.z > zs * 1.002 && q.z - zs < max(radius * 0.25, 4.0 * pixelSize)) {
        // Behind the visible surface there, and not far behind it (a thick object in front is not a hit): that
        // surface's lit colour comes toward P.
        sum += pow(max(fnvSceneColor(qp).rgb, vec3(0.0)), vec3(2.2));
        hit = true;
      }
    }
    if (!hit) misses += 1.0;
  }
  // Fade to the probe toward the end of the range (no seam where the screen-space light stops).
  const float fade = clamp((p_giFadeEnd - P.z) / (0.25 * p_giFadeEnd), 0.0, 1.0);
  outGi = mix(vec4(0.0, 0.0, 0.0, 1.0), vec4(sum / float(rays), misses / float(rays)), fade);
}
