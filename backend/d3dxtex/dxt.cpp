// Behaviour-equivalent reimplementation of d3dx9_38's DXT codec. See dxt.h.
//
// Reproducing x87 results exactly:
//  * Precision. An x87 operation under 24-bit precision control returns the IEEE single-precision result; under
//    53-bit it returns the IEEE double-precision result (the extended exponent range never matters for these value
//    ranges). Every function is a template on R: values the original keeps in x87 registers are R, values it stores
//    to 32-bit memory are float (the store rounds). Loads are exact.
//  * Rounding. The original's compiler hoisted the __ftol idiom (control word | 0xC00) around whole loops, so all
//    arithmetic in those spans rounds toward zero. The spans are reproduced with MXCSR (RoundToZero scopes), at
//    the instruction boundaries of the original (addresses noted per scope).
//  * No contraction or reordering: this file is compiled with /fp:strict and fenv_access on.
//  * Constants are the binary's own (float values stored as doubles; all exactly representable in float).
#include "dxt.h"

#include <atomic>
#include <bit>
#include <cstring>
#include <utility>
#include <xmmintrin.h>

#pragma fenv_access(on)

namespace fnvvk::d3dx {

namespace {

std::atomic<uint64_t> g_outOfRange{ 0 };

constexpr float fb(uint32_t u) { return std::bit_cast<float>(u); }
constexpr float kInv31 = fb(0x3D042108), kInv63 = fb(0x3C820821), kInv15 = fb(0x3D888889), kInv255 = fb(0x3B808081);
constexpr float kThird = fb(0x3EAAAAAB), kTwoThirds = fb(0x3F2AAAAB), kSeventh = fb(0x3E124925), kFifth = fb(0x3E4CCCCD);
// g_Luminance and its inverse (d3dx9_38 0x00787E44 / 0x00787E54).
constexpr float kLum[3] = { fb(0x3E981530), 1.0f, fb(0x3DCE6734) };
constexpr float kLumInv[3] = { fb(0x4057762E), 1.0f, fb(0x411EC1DD) };
constexpr float kFltMin = fb(0x00800000);       // __real@00800000
constexpr float kEpsAB = fb(0x39800000);        // 1/4096, __real@39800000
// Interpolation weights (0x004080D4..), pC = weight of the first endpoint, pD = of the second.
constexpr float kPC3[3] = { 1.0f, 0.5f, 0.0f }, kPD3[3] = { 0.0f, 0.5f, 1.0f };
constexpr float kPC4[4] = { 1.0f, kTwoThirds, kThird, 0.0f }, kPD4[4] = { 0.0f, kThird, kTwoThirds, 1.0f };
constexpr float kPC6[6] = { 1.0f, fb(0x3F4CCCCD), fb(0x3F19999A), fb(0x3ECCCCCD), fb(0x3E4CCCCD), 0.0f };
constexpr float kPD6[6] = { 0.0f, fb(0x3E4CCCCD), fb(0x3ECCCCCD), fb(0x3F19999A), fb(0x3F4CCCCD), 1.0f };
constexpr float kPC8[8] = { 1.0f, fb(0x3F5B6DB7), fb(0x3F36DB6E), fb(0x3F124925), fb(0x3EDB6DB7), fb(0x3E924925), fb(0x3E124925), 0.0f };
// The binary's last entry is 8/7 (0x3F924925), not 1.0 -- reproduced as is (0x0040807C).
constexpr float kPD8[8] = { 0.0f, fb(0x3E124925), fb(0x3E924925), fb(0x3EDB6DB7), fb(0x3F124925), fb(0x3F36DB6E), fb(0x3F5B6DB7), fb(0x3F924925) };
// Index remaps from palette order to DXT index (0x0040810C, 0x0040811C, 0x00408130, 0x00408150).
constexpr uint32_t kRemap4[4] = { 0, 2, 3, 1 }, kRemap3[3] = { 0, 2, 1 };
constexpr uint32_t kRemapA8[8] = { 0, 2, 3, 4, 5, 6, 7, 1 }, kRemapA6[6] = { 0, 2, 3, 4, 5, 1 };

struct RoundToZero {
  unsigned saved;
  RoundToZero() : saved(_mm_getcsr()) { _mm_setcsr((saved & ~_MM_ROUND_MASK) | _MM_ROUND_TOWARD_ZERO); }
  ~RoundToZero() { _mm_setcsr(saved); }
};

template <class R> inline R r(float v) { return R(v); }

// Floyd-Steinberg error diffusion over a 4x4 block as the original does it (weights 7/16, 3/16, 5/16, 1/16;
// the error value is always a float the original stored first). `err` has stride `stride` floats per texel.
template <class R>
inline void diffuse(float* err, uint32_t i, uint32_t stride, float e) {
  if ((i & 3) != 3) err[(i + 1) * stride] = float(r<R>(e) * R(0.4375) + r<R>(err[(i + 1) * stride]));
  if (i < 12) {
    if ((i & 3) != 0) err[(i + 3) * stride] = float(r<R>(e) * R(0.1875) + r<R>(err[(i + 3) * stride]));
    err[(i + 4) * stride] = float(r<R>(e) * R(0.3125) + r<R>(err[(i + 4) * stride]));
    if ((i & 3) != 3) err[(i + 5) * stride] = float(r<R>(e) * R(0.0625) + r<R>(err[(i + 5) * stride]));
  }
}

// sub_627A50: 565 -> color, alpha 1.
template <class R>
inline void decode565(float out[4], uint32_t c) {
  out[0] = float(R((c & 0xFFFF) >> 11) * r<R>(kInv31));
  out[1] = float(R((c >> 5) & 0x3F) * r<R>(kInv63));
  out[2] = float(r<R>(kInv31) * R(c & 0x1F));
  out[3] = 1.0f;
}

// sub_628493: clamp to [0,1], then round each channel (inside a round-toward-zero span, 0x0062851C-0x00628576).
template <class R>
inline uint32_t encode565(const float c[3]) {
  float v[3];
  for (int k = 0; k < 3; k++) v[k] = c[k] >= 0.0f ? (c[k] <= 1.0f ? c[k] : 1.0f) : 0.0f;
  RoundToZero rz;
  float s0 = float(r<R>(v[0]) * R(31.0) + R(0.5));
  int ir = int(s0);
  float s1 = float(r<R>(v[1]) * R(63.0) + R(0.5));
  int ig = int(s1);
  float s2 = float(r<R>(v[2]) * R(31.0) + R(0.5));
  int ib = int(s2);
  return uint32_t(ib | (32 * (ig | (ir << 6))));
}

// sub_6288D0: optimise RGB endpoints (luminance-weighted points, alpha component unused).
template <class R>
void optimizeRGB(float X[3], float Y[3], const Color* pts, uint32_t steps) {
  const float* pC = steps == 3 ? kPC3 : kPC4;
  const float* pD = steps == 3 ? kPD3 : kPD4;
  float x[3] = { kLum[0], kLum[1], kLum[2] }, y[3] = { 0.0f, 0.0f, 0.0f };
  for (int i = 0; i < 16; i++) {
    const float* p = &pts[i].r;
    for (int k = 0; k < 3; k++) {
      if (x[k] > p[k]) x[k] = p[k];
    }
    for (int k = 0; k < 3; k++) {
      if (y[k] < p[k]) y[k] = p[k];
    }
  }
  float ab0 = float(r<R>(y[0]) - r<R>(x[0])), ab1 = float(r<R>(y[1]) - r<R>(x[1])), ab2 = float(r<R>(y[2]) - r<R>(x[2]));
  float fAB = float(r<R>(ab1) * r<R>(ab1) + r<R>(ab0) * r<R>(ab0) + r<R>(ab2) * r<R>(ab2));
  if (fAB < kFltMin) {
    for (int k = 0; k < 3; k++) { X[k] = x[k]; Y[k] = y[k]; }
    return;
  }
  float inv = float(R(1.0) / r<R>(fAB));
  float dir0 = float(r<R>(inv) * r<R>(ab0)), dir1 = float(r<R>(inv) * r<R>(ab1)), dir2 = float(r<R>(inv) * r<R>(ab2));
  float mid0 = float((r<R>(y[0]) + r<R>(x[0])) * R(0.5));
  float mid1 = float((r<R>(y[1]) + r<R>(x[1])) * R(0.5));
  float mid2 = float(R(0.5) * (r<R>(y[2]) + r<R>(x[2])));
  float fDir[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  for (int i = 0; i < 16; i++) {
    float d0 = float((r<R>(pts[i].r) - r<R>(mid0)) * r<R>(dir0));
    float d1 = float((r<R>(pts[i].g) - r<R>(mid1)) * r<R>(dir1));
    float d2 = float((r<R>(pts[i].b) - r<R>(mid2)) * r<R>(dir2));
    R s01 = r<R>(d1) + r<R>(d0);
    float f;
    f = float(r<R>(d2) + s01);
    fDir[0] = float(r<R>(f) * r<R>(f) + r<R>(fDir[0]));
    f = float(s01 - r<R>(d2));
    fDir[1] = float(r<R>(f) * r<R>(f) + r<R>(fDir[1]));
    R m01 = r<R>(d0) - r<R>(d1);
    f = float(r<R>(d2) + m01);
    fDir[2] = float(r<R>(f) * r<R>(f) + r<R>(fDir[2]));
    f = float(m01 - r<R>(d2));
    fDir[3] = float(r<R>(f) * r<R>(f) + r<R>(fDir[3]));
  }
  float best = fDir[0];
  uint32_t iMax = 0;
  for (uint32_t i = 1; i < 4; i++)
    if (best < fDir[i]) { iMax = i; best = fDir[i]; }
  if (iMax & 2) std::swap(x[1], y[1]);
  if (iMax & 1) std::swap(x[2], y[2]);
  if (fAB < kEpsAB) {
    for (int k = 0; k < 3; k++) { X[k] = x[k]; Y[k] = y[k]; }
    return;
  }

  float fSteps = float(steps - 1);
  RoundToZero rz;  // 0x00628BB8 - 0x00628EC9
  for (uint32_t iter = 0; iter < 8; iter++) {
    float st[4][3];
    for (uint32_t k = 0; k < steps; k++)
      for (int c = 0; c < 3; c++) st[k][c] = float(r<R>(pC[k]) * r<R>(x[c]) + r<R>(y[c]) * r<R>(pD[k]));
    float dir[3] = { float(r<R>(y[0]) - r<R>(x[0])), float(r<R>(y[1]) - r<R>(x[1])), float(r<R>(y[2]) - r<R>(x[2])) };
    float fLen = float(r<R>(dir[1]) * r<R>(dir[1]) + r<R>(dir[0]) * r<R>(dir[0]) + r<R>(dir[2]) * r<R>(dir[2]));
    if (fLen < kEpsAB) break;
    float scale = float(r<R>(fSteps) / r<R>(fLen));
    for (int c = 0; c < 3; c++) dir[c] = float(r<R>(scale) * r<R>(dir[c]));
    float d2X = 0.0f, d2Y = 0.0f, dX[3] = { 0.0f, 0.0f, 0.0f }, dY[3] = { 0.0f, 0.0f, 0.0f };
    for (int i = 0; i < 16; i++) {
      const float* p = &pts[i].r;
      float dot = float((r<R>(p[1]) - r<R>(x[1])) * r<R>(dir[1]) + (r<R>(p[0]) - r<R>(x[0])) * r<R>(dir[0]) +
                        (r<R>(p[2]) - r<R>(x[2])) * r<R>(dir[2]));
      int s;
      if (fSteps > dot) {
        float h = float(r<R>(dot) + R(0.5));
        s = int(h);
      } else {
        s = int(steps) - 1;
      }
      if (s < 0) {
        // The original reads step colours and weights outside its arrays here (stack / neighbouring constants).
        g_outOfRange++;
        s = 0;
      }
      float diff[3] = { float(r<R>(st[s][0]) - r<R>(p[0])), float(r<R>(st[s][1]) - r<R>(p[1])),
                        float(r<R>(st[s][2]) - r<R>(p[2])) };
      float fC = float(r<R>(pC[s]) * R(0.125));
      float fD = float(R(0.125) * r<R>(pD[s]));
      d2X = float(r<R>(pC[s]) * r<R>(fC) + r<R>(d2X));
      for (int c = 0; c < 3; c++) dX[c] = float(r<R>(fC) * r<R>(diff[c]) + r<R>(dX[c]));
      d2Y = float(r<R>(fD) * r<R>(pD[s]) + r<R>(d2Y));
      for (int c = 0; c < 3; c++) dY[c] = float(r<R>(fD) * r<R>(diff[c]) + r<R>(dY[c]));
    }
    if (0.0f < d2X) {
      float f = float(R(-1.0) / r<R>(d2X));
      for (int c = 0; c < 3; c++) x[c] = float(r<R>(f) * r<R>(dX[c]) + r<R>(x[c]));
    }
    if (0.0f < d2Y) {
      float f = float(R(-1.0) / r<R>(d2Y));
      for (int c = 0; c < 3; c++) y[c] = float(r<R>(f) * r<R>(dY[c]) + r<R>(y[c]));
    }
    const R eps = R(1.0 / 65536.0);
    if (r<R>(dX[0]) * r<R>(dX[0]) < eps && r<R>(dX[1]) * r<R>(dX[1]) < eps && r<R>(dX[2]) * r<R>(dX[2]) < eps &&
        r<R>(dY[0]) * r<R>(dY[0]) < eps && r<R>(dY[1]) * r<R>(dY[1]) < eps && r<R>(dY[2]) * r<R>(dY[2]) < eps)
      break;
  }
  for (int k = 0; k < 3; k++) { X[k] = x[k]; Y[k] = y[k]; }
}

// sub_628F03: encode the colour half of a DXT block (8 bytes). colorKey: DXT1 alpha handling (3-colour mode).
template <class R>
void encodeColor(uint8_t* out, const Color* in, bool colorKey, bool dither) {
  uint32_t steps = 4;
  if (colorKey) {
    uint32_t transparent = 0;
    for (int i = 0; i < 16; i++)
      if (in[i].a < 0.5f) transparent++;
    if (transparent == 16) {
      const uint16_t c0 = 0, c1 = 0xFFFF;
      const uint32_t idx = 0xFFFFFFFF;
      memcpy(out, &c0, 2);
      memcpy(out + 2, &c1, 2);
      memcpy(out + 4, &idx, 4);
      return;
    }
    steps = transparent ? 3 : 4;
  }

  Color pts[16];
  float err[16 * 4];
  if (dither) memset(err, 0, sizeof(err));
  {
    RoundToZero rz;  // 0x00628F94 - 0x006292C8
    for (uint32_t i = 0; i < 16; i++) {
      float c[3] = { in[i].r, in[i].g, in[i].b };
      if (dither)
        for (int k = 0; k < 3; k++) c[k] = float(r<R>(err[i * 4 + k]) + r<R>(c[k]));
      float s = float(r<R>(c[0]) * R(31.0) + R(0.5));
      float q0 = float(R(int(s)) * r<R>(kInv31));
      s = float(r<R>(c[1]) * R(63.0) + R(0.5));
      float q1 = float(R(int(s)) * r<R>(kInv63));
      s = float(r<R>(c[2]) * R(31.0) + R(0.5));
      float q2 = float(R(int(s)) * r<R>(kInv31));
      pts[i].a = 1.0f;
      if (dither) {
        float e[3] = { float(r<R>(c[0]) - r<R>(q0)), float(r<R>(c[1]) - r<R>(q1)), float(r<R>(c[2]) - r<R>(q2)) };
        for (int k = 0; k < 3; k++) diffuse<R>(err + k, i, 4, e[k]);
      }
      pts[i].r = float(r<R>(q0) * r<R>(kLum[0]));
      pts[i].g = float(r<R>(q1) * r<R>(kLum[1]));
      pts[i].b = float(r<R>(q2) * r<R>(kLum[2]));
    }
  }

  float X[3], Y[3];
  optimizeRGB<R>(X, Y, pts, steps);
  float cA[3] = { float(r<R>(kLumInv[0]) * r<R>(X[0])), float(r<R>(X[1]) * r<R>(kLumInv[1])),
                  float(r<R>(X[2]) * r<R>(kLumInv[2])) };
  float cB[3] = { float(r<R>(kLumInv[0]) * r<R>(Y[0])), float(r<R>(kLumInv[1]) * r<R>(Y[1])),
                  float(r<R>(kLumInv[2]) * r<R>(Y[2])) };
  uint32_t wA = encode565<R>(cA) & 0xFFFF, wB = encode565<R>(cB) & 0xFFFF;
  if (steps == 4 && wA == wB) {
    const uint16_t a = uint16_t(wA), b = uint16_t(wB);
    const uint32_t idx = 0;
    memcpy(out, &a, 2);
    memcpy(out + 2, &b, 2);
    memcpy(out + 4, &idx, 4);
    return;
  }
  float dA[4], dB[4];
  decode565<R>(dA, wA);
  decode565<R>(dB, wB);
  float A[4] = { float(r<R>(kLum[0]) * r<R>(dA[0])), float(r<R>(dA[1]) * r<R>(kLum[1])), float(r<R>(dA[2]) * r<R>(kLum[2])), 0.0f };
  float B[4] = { float(r<R>(kLum[0]) * r<R>(dB[0])), float(r<R>(kLum[1]) * r<R>(dB[1])), float(r<R>(kLum[2]) * r<R>(dB[2])), 0.0f };
  // The original's endpoint alphas are never written (uninitialised stack); they only feed palette alphas that
  // nothing reads.
  float step[4][4];
  uint16_t c0, c1;
  if ((steps == 3) == (wB >= wA)) {
    c0 = uint16_t(wA); c1 = uint16_t(wB);
    memcpy(step[0], A, sizeof(A)); memcpy(step[1], B, sizeof(B));
  } else {
    c0 = uint16_t(wB); c1 = uint16_t(wA);
    memcpy(step[0], B, sizeof(B)); memcpy(step[1], A, sizeof(A));
  }
  memcpy(out, &c0, 2);
  memcpy(out + 2, &c1, 2);
  const uint32_t* remap;
  R dr = r<R>(step[1][0]) - r<R>(step[0][0]);
  R dg, db;
  if (steps == 3) {
    remap = kRemap3;
    step[2][0] = float(dr * R(0.5) + r<R>(step[0][0]));
    dg = r<R>(step[1][1]) - r<R>(step[0][1]);
    step[2][1] = float(dg * R(0.5) + r<R>(step[0][1]));
    db = r<R>(step[1][2]) - r<R>(step[0][2]);
    step[2][2] = float(db * R(0.5) + r<R>(step[0][2]));
  } else {
    remap = kRemap4;
    step[2][0] = float(dr * r<R>(kThird) + r<R>(step[0][0]));
    dg = r<R>(step[1][1]) - r<R>(step[0][1]);
    step[2][1] = float(dg * r<R>(kThird) + r<R>(step[0][1]));
    db = r<R>(step[1][2]) - r<R>(step[0][2]);
    step[2][2] = float(db * r<R>(kThird) + r<R>(step[0][2]));
    step[3][0] = float(dr * r<R>(kTwoThirds) + r<R>(step[0][0]));
    step[3][1] = float(dg * r<R>(kTwoThirds) + r<R>(step[0][1]));
    step[3][2] = float(db * r<R>(kTwoThirds) + r<R>(step[0][2]));
  }
  float dir[3] = { float(dr), float(dg), float(db) };
  float fSteps = float(steps - 1);
  R scaleR = (wA == wB) ? R(0.0)
                        : r<R>(fSteps) / (r<R>(dir[1]) * r<R>(dir[1]) + r<R>(dir[0]) * r<R>(dir[0]) +
                                          r<R>(dir[2]) * r<R>(dir[2]));
  float scale = float(scaleR);
  for (int k = 0; k < 3; k++) dir[k] = float(r<R>(scale) * r<R>(dir[k]));

  if (dither) memset(err, 0, sizeof(err));
  uint32_t indices = 0;
  RoundToZero rz;  // 0x00629636 - 0x006298F3
  for (uint32_t i = 0; i < 16; i++) {
    if (steps == 3 && in[i].a < 0.5f) {
      indices = (indices >> 2) | 0xC0000000u;
      continue;
    }
    float c0f = float(r<R>(in[i].r) * r<R>(kLum[0]));
    float c1f = float(r<R>(kLum[1]) * r<R>(in[i].g));
    float c2f = float(r<R>(in[i].b) * r<R>(kLum[2]));
    if (dither) {
      c0f = float(r<R>(c0f) + r<R>(err[i * 4 + 0]));
      c1f = float(r<R>(err[i * 4 + 1]) + r<R>(c1f));
      c2f = float(r<R>(err[i * 4 + 2]) + r<R>(c2f));
    }
    float dot = float((r<R>(c1f) - r<R>(step[0][1])) * r<R>(dir[1]) + (r<R>(c0f) - r<R>(step[0][0])) * r<R>(dir[0]) +
                      (r<R>(c2f) - r<R>(step[0][2])) * r<R>(dir[2]));
    uint32_t s;
    if (dot > 0.0f) {
      if (fSteps > dot) {
        float h = float(r<R>(dot) + R(0.5));
        s = remap[int(h)];
      } else {
        s = 1;
      }
    } else {
      s = 0;
    }
    indices = (indices >> 2) | (s << 30);
    if (dither) {
      const float w = pts[i].a;  // 1.0, as the original multiplies by the point's alpha
      float e[3] = { float((r<R>(c0f) - r<R>(step[s][0])) * r<R>(w)), float((r<R>(c1f) - r<R>(step[s][1])) * r<R>(w)),
                     float((r<R>(c2f) - r<R>(step[s][2])) * r<R>(w)) };
      for (int k = 0; k < 3; k++) diffuse<R>(err + k, i, 4, e[k]);
    }
  }
  memcpy(out + 4, &indices, 4);
}

// sub_62858A: optimise alpha endpoints. steps 6 or 8.
template <class R>
void optimizeAlpha(float* outX, float* outY, const float* pts, uint32_t steps) {
  const float* pC = steps == 6 ? kPC6 : kPC8;
  const float* pD = steps == 6 ? kPD6 : kPD8;
  float x = 1.0f, y = 0.0f;
  if (steps == 8) {
    for (int i = 0; i < 16; i++) {
      if (x > pts[i]) x = pts[i];
      if (y < pts[i]) y = pts[i];
    }
  } else {
    for (int i = 0; i < 16; i++) {
      if (x > pts[i] && 0.0f < pts[i]) x = pts[i];
      if (y < pts[i] && 1.0f > pts[i]) y = pts[i];
    }
    if (y == x) y = 1.0f;
  }
  float fSteps = float(steps - 1);
  {
    RoundToZero rz;  // 0x00628680 - 0x00628864
    for (uint32_t iter = 0; iter < 8; iter++) {
      R len = r<R>(y) - r<R>(x);
      if (len < R(0.00390625)) break;
      float scale = float(r<R>(fSteps) / len);
      float st[8];
      for (uint32_t k = 0; k < steps; k++) st[k] = float(r<R>(pD[k]) * r<R>(y) + r<R>(pC[k]) * r<R>(x));
      if (steps == 6) { st[6] = 0.0f; st[7] = 1.0f; }
      float dX = 0.0f, dY = 0.0f, d2X = 0.0f, d2Y = 0.0f;
      for (int i = 0; i < 16; i++) {
        float dot = float((r<R>(pts[i]) - r<R>(x)) * r<R>(scale));
        uint32_t s;
        if (0.0f < dot) {
          if (fSteps > dot) {
            float h = float(r<R>(dot) + R(0.5));
            s = uint32_t(int(h));
          } else {
            if (steps == 6) {
              float t = float((r<R>(y) + R(1.0)) * R(0.5));
              if (t <= pts[i]) continue;
            }
            s = steps - 1;
          }
        } else {
          if (steps == 6) {
            float t = float(r<R>(x) * R(0.5));
            if (t >= pts[i]) continue;
          }
          s = 0;
        }
        if (s < steps) {
          float diff = float(r<R>(pts[i]) - r<R>(st[s]));
          dX = float(r<R>(diff) * r<R>(pC[s]) + r<R>(dX));
          d2X = float(r<R>(pC[s]) * r<R>(pC[s]) + r<R>(d2X));
          dY = float(r<R>(pD[s]) * r<R>(diff) + r<R>(dY));
          d2Y = float(r<R>(pD[s]) * r<R>(pD[s]) + r<R>(d2Y));
        }
      }
      if (0.0f < d2X) x = float(r<R>(x) - r<R>(dX) / r<R>(d2X));
      if (0.0f < d2Y) y = float(r<R>(y) - r<R>(dY) / r<R>(d2Y));
      if (y < x) std::swap(x, y);
      if (r<R>(dX) * r<R>(dX) < R(0.015625) && r<R>(dY) * r<R>(dY) < R(0.015625)) break;
    }
  }
  *outX = x >= 0.0f ? (x <= 1.0f ? x : 1.0f) : 0.0f;
  *outY = y >= 0.0f ? (1.0f >= y ? y : 1.0f) : 0.0f;
}

template <class R>
void decodeDXT1T(Color out[16], const uint8_t* block) {
  uint16_t w0, w1;
  uint32_t bits;
  memcpy(&w0, block, 2);
  memcpy(&w1, block + 2, 2);
  memcpy(&bits, block + 4, 4);
  float pal[4][4];
  decode565<R>(pal[0], w0);
  decode565<R>(pal[1], w1);
  R d0 = r<R>(pal[1][0]) - r<R>(pal[0][0]);
  if (w0 > w1) {
    R d1 = r<R>(pal[1][1]) - r<R>(pal[0][1]), d2 = r<R>(pal[1][2]) - r<R>(pal[0][2]), d3 = r<R>(pal[1][3]) - r<R>(pal[0][3]);
    pal[2][0] = float(d0 * r<R>(kThird) + r<R>(pal[0][0]));
    pal[2][1] = float(d1 * r<R>(kThird) + r<R>(pal[0][1]));
    pal[2][2] = float(d2 * r<R>(kThird) + r<R>(pal[0][2]));
    pal[2][3] = float(r<R>(kThird) * d3 + r<R>(pal[0][3]));
    pal[3][0] = float(d0 * r<R>(kTwoThirds) + r<R>(pal[0][0]));
    pal[3][1] = float(d1 * r<R>(kTwoThirds) + r<R>(pal[0][1]));
    pal[3][2] = float(d2 * r<R>(kTwoThirds) + r<R>(pal[0][2]));
    pal[3][3] = float(r<R>(kTwoThirds) * d3 + r<R>(pal[0][3]));
  } else {
    pal[2][0] = float(d0 * R(0.5) + r<R>(pal[0][0]));
    pal[2][1] = float((r<R>(pal[1][1]) - r<R>(pal[0][1])) * R(0.5) + r<R>(pal[0][1]));
    pal[2][2] = float((r<R>(pal[1][2]) - r<R>(pal[0][2])) * R(0.5) + r<R>(pal[0][2]));
    pal[2][3] = float(R(0.5) * (r<R>(pal[1][3]) - r<R>(pal[0][3])) + r<R>(pal[0][3]));
    pal[3][0] = pal[3][1] = pal[3][2] = pal[3][3] = 0.0f;
  }
  for (int i = 0; i < 16; i++, bits >>= 2) {
    const float* p = pal[bits & 3];
    out[i] = { p[0], p[1], p[2], p[3] };
  }
}

template <class R>
void decodeDXT3T(Color out[16], const uint8_t* block) {
  decodeDXT1T<R>(out, block + 8);
  uint32_t a0, a1;
  memcpy(&a0, block, 4);
  memcpy(&a1, block + 4, 4);
  for (int i = 0; i < 8; i++, a0 >>= 4) out[i].a = float(R(a0 & 0xF) * r<R>(kInv15));
  for (int i = 8; i < 16; i++, a1 >>= 4) out[i].a = float(R(a1 & 0xF) * r<R>(kInv15));
}

template <class R>
void alphaPalette(float pal[8], uint8_t a0, uint8_t a1) {
  pal[0] = float(R(a0) * r<R>(kInv255));
  pal[1] = float(r<R>(kInv255) * R(a1));
  if (a0 <= a1) {
    for (int i = 1; i < 5; i++) pal[i + 1] = float((R(5 - i) * r<R>(pal[0]) + R(i) * r<R>(pal[1])) * r<R>(kFifth));
    pal[6] = 0.0f;
    pal[7] = 1.0f;
  } else {
    for (int i = 1; i < 7; i++) pal[i + 1] = float((R(7 - i) * r<R>(pal[0]) + R(i) * r<R>(pal[1])) * r<R>(kSeventh));
  }
}

template <class R>
void decodeDXT5T(Color out[16], const uint8_t* block) {
  decodeDXT1T<R>(out, block + 8);
  float pal[8];
  alphaPalette<R>(pal, block[0], block[1]);
  uint32_t b0 = block[2] | (uint32_t(block[3]) << 8) | (uint32_t(block[4]) << 16);
  uint32_t b1 = block[5] | (uint32_t(block[6]) << 8) | (uint32_t(block[7]) << 16);
  for (int i = 0; i < 8; i++, b0 >>= 3) out[i].a = pal[b0 & 7];
  for (int i = 8; i < 16; i++, b1 >>= 3) out[i].a = pal[b1 & 7];
}

// D3DXEncodeDXT1 (0x00629C16): with dither, alpha is first quantised to 0/1 with error diffusion
// (round-toward-zero span 0x00629C4C - 0x00629D25).
template <class R>
void encodeDXT1T(uint8_t* out, const Color* in, bool dither) {
  if (!dither) {
    encodeColor<R>(out, in, true, false);
    return;
  }
  Color c[16];
  float err[16];
  memset(err, 0, sizeof(err));
  {
    RoundToZero rz;
    for (uint32_t i = 0; i < 16; i++) {
      R a = r<R>(err[i]) + r<R>(in[i].a);
      float stored = float(a);
      c[i] = in[i];
      float h = float(a + R(0.5));
      float q = float(int(h));
      c[i].a = q;
      float e = float(r<R>(stored) - r<R>(q));
      diffuse<R>(err, i, 1, e);
    }
  }
  encodeColor<R>(out, c, true, true);
}

// D3DXEncodeDXT3 (0x00629D52): 4-bit alpha (round-toward-zero span 0x00629D89 - 0x00629E75).
template <class R>
void encodeDXT3T(uint8_t* out, const Color* in, bool dither) {
  uint32_t a[2] = { 0, 0 };
  float err[16];
  if (dither) memset(err, 0, sizeof(err));
  {
    RoundToZero rz;
    for (uint32_t i = 0; i < 16; i++) {
      float v = in[i].a;
      if (dither) v = float(r<R>(err[i]) + r<R>(v));
      float h = float(r<R>(v) * R(15.0) + R(0.5));
      int q = int(h);
      a[i >> 3] = (a[i >> 3] >> 4) | (uint32_t(q) << 28);
      if (dither) {
        float e = float(r<R>(v) - R(uint32_t(q)) * r<R>(kInv15));
        diffuse<R>(err, i, 1, e);
      }
    }
  }
  memcpy(out, a, 8);
  encodeColor<R>(out + 8, in, false, dither);
}

// D3DXEncodeDXT5 (0x00629E94).
template <class R>
void encodeDXT5T(uint8_t* out, const Color* in, bool dither) {
  float q[16], err[16];
  float minA = in[0].a, maxA = in[0].a;  // the original seeds both with texel 0's unquantised alpha
  if (dither) memset(err, 0, sizeof(err));
  {
    RoundToZero rz;  // 0x00629EDC - 0x00629FD8
    for (uint32_t i = 0; i < 16; i++) {
      float v = in[i].a;
      if (dither) v = float(r<R>(err[i]) + r<R>(v));
      float h = float(r<R>(v) * R(255.0) + R(0.5));
      int iq = int(h);
      float qa = float(R(iq) * r<R>(kInv255));
      q[i] = qa;
      if (minA <= qa) {
        if (maxA < qa) maxA = qa;
      } else {
        minA = qa;
      }
      if (dither) diffuse<R>(err, i, 1, float(r<R>(v) - r<R>(qa)));
    }
  }
  encodeColor<R>(out + 8, in, false, dither);
  if (1.0f == minA) {
    out[0] = 0xFF;
    out[1] = 0xFF;
    memset(out + 2, 0, 6);
    return;
  }
  uint32_t steps = (minA == 0.0f || maxA == 1.0f) ? 6 : 8;
  float X, Y;
  optimizeAlpha<R>(&X, &Y, q, steps);
  int ia, ib;
  {
    RoundToZero rz;  // 0x0062A06E - 0x0062A0AD
    float h = float(r<R>(X) * R(255.0) + R(0.5));
    ia = int(h);
    h = float(r<R>(Y) * R(255.0) + R(0.5));
    ib = int(h);
  }
  uint8_t ba = uint8_t(ia), bb = uint8_t(ib);
  float fa = float(R(ba) * r<R>(kInv255)), fbv = float(r<R>(kInv255) * R(bb));
  float pal[8];
  const uint32_t* remap;
  if (steps != 8) {
    out[0] = ba;
    out[1] = bb;
    pal[0] = fa;
    pal[1] = fbv;
    for (int i = 1; i < 5; i++) pal[i + 1] = float((R(5 - i) * r<R>(pal[0]) + R(i) * r<R>(pal[1])) * r<R>(kFifth));
    pal[6] = 0.0f;
    pal[7] = 1.0f;
    remap = kRemapA6;
  } else {
    if (ba == bb) {
      out[0] = ba;
      out[1] = bb;
      memset(out + 2, 0, 6);
      return;
    }
    out[1] = ba;
    out[0] = bb;
    pal[0] = fbv;
    pal[1] = fa;
    for (int i = 1; i < 7; i++) pal[i + 1] = float((R(7 - i) * r<R>(pal[0]) + R(i) * r<R>(pal[1])) * r<R>(kSeventh));
    remap = kRemapA8;
  }
  float fSteps = float(steps - 1);
  float scale = pal[1] == pal[0] ? 0.0f : float(r<R>(fSteps) / (r<R>(pal[1]) - r<R>(pal[0])));
  if (dither) memset(err, 0, sizeof(err));
  RoundToZero rz;  // 0x0062A208 - 0x0062A38E
  for (uint32_t half = 0; half < 2; half++) {
    uint32_t bits = 0;
    for (uint32_t i = half * 8; i < half * 8 + 8; i++) {
      float v = in[i].a;
      if (dither) v = float(r<R>(err[i]) + r<R>(v));
      float dot = float((r<R>(v) - r<R>(pal[0])) * r<R>(scale));
      uint32_t s;
      if (dot > 0.0f) {
        if (fSteps > dot) {
          float h = float(r<R>(dot) + R(0.5));
          s = remap[int(h)];
        } else if (steps != 6 || (r<R>(pal[1]) + R(1.0)) * R(0.5) > r<R>(v)) {
          s = 1;
        } else {
          s = 7;
        }
      } else if (steps != 6 || r<R>(pal[0]) * R(0.5) < r<R>(v)) {
        s = 0;
      } else {
        s = 6;
      }
      bits = (s << 21) | (bits >> 3);
      if (dither) diffuse<R>(err, i, 1, float(r<R>(v) - r<R>(pal[s])));
    }
    out[2 + half * 3] = uint8_t(bits);
    out[3 + half * 3] = uint8_t(bits >> 8);
    out[4 + half * 3] = uint8_t(bits >> 16);
  }
}

}  // namespace

void decodeDXT1(Color out[16], const uint8_t block[8], Precision p) {
  p == Precision::Single ? decodeDXT1T<float>(out, block) : decodeDXT1T<double>(out, block);
}
void decodeDXT3(Color out[16], const uint8_t block[16], Precision p) {
  p == Precision::Single ? decodeDXT3T<float>(out, block) : decodeDXT3T<double>(out, block);
}
void decodeDXT5(Color out[16], const uint8_t block[16], Precision p) {
  p == Precision::Single ? decodeDXT5T<float>(out, block) : decodeDXT5T<double>(out, block);
}
void encodeDXT1(uint8_t block[8], const Color in[16], bool dither, Precision p) {
  p == Precision::Single ? encodeDXT1T<float>(block, in, dither) : encodeDXT1T<double>(block, in, dither);
}
void encodeDXT3(uint8_t block[16], const Color in[16], bool dither, Precision p) {
  p == Precision::Single ? encodeDXT3T<float>(block, in, dither) : encodeDXT3T<double>(block, in, dither);
}
void encodeDXT5(uint8_t block[16], const Color in[16], bool dither, Precision p) {
  p == Precision::Single ? encodeDXT5T<float>(block, in, dither) : encodeDXT5T<double>(block, in, dither);
}
uint64_t outOfRangeStepCount() { return g_outOfRange.load(); }

}  // namespace fnvvk::d3dx
