#pragma once
// Sun shadow cascades: the math of sunshadows.cpp with no game access, so the test harness runs exactly this code.
//
// Spaces: absolute world (game units, z up); camera-relative world (absolute - camera position: what the casters'
// rows are in); view space (x right, y up, z forward: the effects' fnvViewPos). Clip: x, y in -1..1 over the cascade,
// z 0..1 along the light from the near plane (on the sun's side). Matrices are row-major, clip = M * (p, 1).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace fnvvk::cascades {

struct D3 {
  double x = 0, y = 0, z = 0;
};
inline D3 operator+(D3 a, D3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
inline D3 operator-(D3 a, D3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
inline D3 operator*(D3 a, double s) { return { a.x * s, a.y * s, a.z * s }; }
inline double dot(D3 a, D3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline D3 cross(D3 a, D3 b) { return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x }; }
inline D3 normalize(D3 a) { const double l = std::sqrt(dot(a, a)); return l > 0 ? a * (1.0 / l) : a; }

struct Camera {
  D3 position;          // absolute world
  D3 right, up, forward;
  float frustum[4];     // tangents: left, right, top, bottom
  double nearZ, farZ;
};

struct Cascade {
  D3 xl, yl, zl;          // light basis: right, up, forward (the direction light travels)
  double cx = 0, cy = 0;  // snapped centre in light space (absolute world)
  double eye = 0;         // light-space depth of the near plane (absolute world)
  double radius = 0, depthRange = 0;
  float minRadius = 0;    // smaller casters are skipped in this cascade
  float split = 0;        // far view distance
  float viewProj[16];     // camera-relative world -> clip
  float fromView[16];     // view space -> clip
};

// The cascade's matrices for camera `cam` from its absolute box (light basis xl yl zl, snapped centre cx cy, near plane
// eye, radius, depthRange): camera-relative world -> clip, and view space -> clip. A cascade drawn for an earlier camera
// keeps its box; these make it usable from the current one (the clip position of a world point does not change).
inline void setMatrices(Cascade& c, const Camera& cam) {
  const D3 xl = c.xl, yl = c.yl, zl = c.zl;
  const double radius = c.radius;
  float* m = c.viewProj;
  const D3 rows[3] = { xl * (1.0 / radius), yl * (1.0 / radius), zl * (1.0 / c.depthRange) };
  const double offs[3] = { (dot(xl, cam.position) - c.cx) / radius, (dot(yl, cam.position) - c.cy) / radius,
                           (dot(zl, cam.position) - c.eye) / c.depthRange };
  for (int r = 0; r < 3; r++) {
    m[r * 4 + 0] = float(rows[r].x);
    m[r * 4 + 1] = float(rows[r].y);
    m[r * 4 + 2] = float(rows[r].z);
    m[r * 4 + 3] = float(offs[r]);
  }
  m[12] = m[13] = m[14] = 0.0f;
  m[15] = 1.0f;
  // View space -> clip: M * [right up forward | 0] (view-space points are camera-relative world directions).
  for (int r = 0; r < 4; r++) {
    const D3 mr{ m[r * 4 + 0], m[r * 4 + 1], m[r * 4 + 2] };
    c.fromView[r * 4 + 0] = float(dot(mr, cam.right));
    c.fromView[r * 4 + 1] = float(dot(mr, cam.up));
    c.fromView[r * 4 + 2] = float(dot(mr, cam.forward));
    c.fromView[r * 4 + 3] = m[r * 4 + 3];
  }
}

// Whether cascade `k` (drawn for an earlier camera) still covers the slice of `cam`'s view between view distances z0
// and z1: all eight corners of the slice inside its box, `margin` texels in from its sides (the shadow lookup's filter
// footprint) and within its depth range. The slice is convex, so then every point of it is.
inline bool covers(const Cascade& k, const Camera& cam, double z0, double z1, uint32_t size, double margin) {
  const double limit = k.radius - margin * 2.0 * k.radius / size;
  for (double z : { z0, z1 })
    for (float x : { cam.frustum[0], cam.frustum[1] })
      for (float y : { cam.frustum[2], cam.frustum[3] }) {
        const D3 p = cam.position + cam.right * (x * z) + cam.up * (y * z) + cam.forward * z;
        const double lx = dot(k.xl, p) - k.cx, ly = dot(k.yl, p) - k.cy, lz = dot(k.zl, p) - k.eye;
        if (std::fabs(lx) > limit || std::fabs(ly) > limit || lz < 0.0 || lz > k.depthRange) return false;
      }
  return true;
}

// Cascades covering view distances camera.nearZ .. min(distance, camera.farZ), lit from `toSun` (normalised).
// `reach`: how far toward the sun beyond a cascade's bounding sphere casters are included.
inline std::vector<Cascade> build(const Camera& cam, D3 toSun, uint32_t count, uint32_t size, double distance, double reach) {
  std::vector<Cascade> cs(count);
  const double nearZ = std::max(1.0, cam.nearZ), farZ = std::min(distance, cam.farZ);
  if (!(farZ > nearZ * 2) || count == 0) return {};
  const D3 zl = toSun * -1.0;
  const D3 xl = normalize(cross(zl, std::fabs(zl.z) < 0.99 ? D3{ 0, 0, 1 } : D3{ 0, 1, 0 }));
  const D3 yl = cross(xl, zl);
  static const float kMinRadius[4] = { 0.0f, 16.0f, 48.0f, 128.0f };
  double zPrev = nearZ;
  for (uint32_t i = 0; i < count; i++) {
    // Practical split scheme: mostly logarithmic, partly uniform.
    const double t = double(i + 1) / count;
    const double z1 = 0.8 * nearZ * std::pow(farZ / nearZ, t) + 0.2 * (nearZ + (farZ - nearZ) * t);
    // Bounding sphere of the slice of the view frustum between zPrev and z1 (camera-relative): its radius does not
    // change as the camera turns, so the cascade's texel size is constant.
    D3 corners[8];
    int k = 0;
    for (double z : { zPrev, z1 })
      for (float x : { cam.frustum[0], cam.frustum[1] })
        for (float y : { cam.frustum[2], cam.frustum[3] }) corners[k++] = cam.right * (x * z) + cam.up * (y * z) + cam.forward * z;
    D3 centre;
    for (const D3& c : corners) centre = centre + c * 0.125;
    double radius = 0;
    for (const D3& c : corners) radius = std::max(radius, std::sqrt(dot(c - centre, c - centre)));
    radius = std::ceil(radius / 16.0) * 16.0;
    Cascade& c = cs[i];
    c.xl = xl;
    c.yl = yl;
    c.zl = zl;
    c.radius = radius;
    c.depthRange = 2.0 * radius + reach;
    c.minRadius = kMinRadius[std::min<uint32_t>(i, 3)];
    c.split = float(z1);
    // Snap the centre to whole texels (absolute world), so shadows do not crawl as the camera moves.
    const double texel = 2.0 * radius / size;
    const D3 abs = cam.position + centre;
    c.cx = std::floor(dot(xl, abs) / texel) * texel;
    c.cy = std::floor(dot(yl, abs) / texel) * texel;
    c.eye = dot(zl, abs) - radius - reach;
    setMatrices(c, cam);
    zPrev = z1;
  }
  return cs;
}

// Wet world's rain cover (docs/WETWORLD.md): a box looking straight down, `radius` game units each way around the
// camera (its centre snapped to whole texels), from `above` units over the camera to `below` units under it. Clip z
// runs 0 at the top to 1 at the bottom, so the map holds the highest surface over each point: rain reaches a point
// whose depth is not below it.
inline Cascade overhead(const Camera& cam, double radius, uint32_t size, double above, double below) {
  Cascade c;
  c.zl = { 0, 0, -1 };  // the rain falls down
  c.xl = { 1, 0, 0 };
  c.yl = cross(c.xl, c.zl);  // (0, 1, 0)
  c.radius = radius;
  c.depthRange = above + below;
  c.minRadius = 0.0f;
  c.split = 0.0f;
  const double texel = 2.0 * radius / size;
  c.cx = std::floor(dot(c.xl, cam.position) / texel) * texel;
  c.cy = std::floor(dot(c.yl, cam.position) / texel) * texel;
  c.eye = dot(c.zl, cam.position) - above;
  setMatrices(c, cam);
  return c;
}

// A caster bound (absolute world centre, radius) against a cascade's box.
inline bool inside(const Cascade& k, D3 centre, double radius) {
  if (radius < k.minRadius) return false;
  const double lx = dot(k.xl, centre) - k.cx, ly = dot(k.yl, centre) - k.cy, lz = dot(k.zl, centre) - k.eye;
  return std::fabs(lx) <= k.radius + radius && std::fabs(ly) <= k.radius + radius && lz >= -radius && lz <= k.depthRange + radius;
}

}  // namespace fnvvk::cascades
