// HDR output and tonemapping (included by backend_test.cpp):
// - effects/tonemap through the effect data path, every operator, SDR (and the HDR highlight branch when the test
//   window's surface offers an HDR format), against a CPU reference whose anchor values were computed independently;
// - the present encode (HDR10 PQ / scRGB) through Backend::readbackHdrEncode against the formulas;
// - wide targets (FNV_WIDE_TARGETS=1): an 8-bit target of the window size keeps its D3D9 layout for the CPU
//   (writeTarget, readback) and holds values above 1.0.
namespace hdrtest {

struct V3 { float r, g, b; };
static V3 operator*(V3 a, float s) { return { a.r * s, a.g * s, a.b * s }; }
static V3 operator*(float s, V3 a) { return a * s; }
static V3 operator+(V3 a, V3 b) { return { a.r + b.r, a.g + b.g, a.b + b.b }; }
static V3 operator-(V3 a, V3 b) { return { a.r - b.r, a.g - b.g, a.b - b.b }; }
static V3 operator/(V3 a, V3 b) { return { a.r / b.r, a.g / b.g, a.b / b.b }; }
template <typename F> static V3 map(V3 a, F f) { return { f(a.r), f(a.g), f(a.b) }; }
static V3 mul(const float m[9], V3 v) {  // m: GLSL column-major mat3
  return { m[0] * v.r + m[3] * v.g + m[6] * v.b, m[1] * v.r + m[4] * v.g + m[7] * v.b, m[2] * v.r + m[5] * v.g + m[8] * v.b };
}
static float clamp01(float v) { return std::clamp(v, 0.0f, 1.0f); }
static float dot709(V3 v) { return 0.2126f * v.r + 0.7152f * v.g + 0.0722f * v.b; }

// The reference of effects/tonemap/tonemap.frag.
static float hableCurve(float x) {
  const float A = 0.15f, B = 0.50f, C = 0.10f, D = 0.20f, E = 0.02f, F = 0.30f;
  return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}
static float uchimura(float x) {
  const float P = 1, a = 1, m = 0.22f, l = 0.4f, c = 1.33f, b = 0;
  const float l0 = ((P - m) * l) / a, S0 = m + l0, S1 = m + a * l0, C2 = (a * P) / (P - S1), CP = -C2 / P;
  const float t = clamp01(x / m), w0 = 1 - t * t * (3 - 2 * t), w2 = x >= m + l0 ? 1.0f : 0.0f, w1 = 1 - w0 - w2;
  const float T = m * std::pow(x / m, c) + b, S = P - (P - S1) * std::exp(CP * (x - S0)), L = m + a * (x - m);
  return clamp01(T * w0 + L * w1 + S * w2);
}
static float lottes(float x) {
  const float a = 1.6f, d = 0.977f, hm = 8.0f, mi = 0.18f, mo = 0.267f;
  const float b = (-std::pow(mi, a) + std::pow(hm, a) * mo) / ((std::pow(hm, a * d) - std::pow(mi, a * d)) * mo);
  const float c = (std::pow(hm, a * d) * std::pow(mi, a) - std::pow(hm, a) * std::pow(mi, a * d) * mo) /
                  ((std::pow(hm, a * d) - std::pow(mi, a * d)) * mo);
  return clamp01(std::pow(x, a) / (std::pow(x, a * d) * b + c));
}
static V3 agx(V3 x, bool punchy) {
  static const float in[9] = { 0.842479062253094f, 0.0423282422610123f, 0.0423756549057051f, 0.0784335999999992f, 0.878468636469772f,
                               0.0784336f, 0.0792237451477643f, 0.0791661274605434f, 0.879142973793104f };
  static const float out[9] = { 1.19687900512017f, -0.0528968517574562f, -0.0529716355144438f, -0.0980208811401368f, 1.15190312990417f,
                                -0.0980434501171241f, -0.0990297440797205f, -0.0989611768448433f, 1.15107367264116f };
  const float minEv = -12.47393f, maxEv = 4.026069f;
  V3 v = mul(in, x);
  v = map(v, [&](float c) {
    c = (std::clamp(std::log2(std::max(c, 1e-10f)), minEv, maxEv) - minEv) / (maxEv - minEv);
    const float c2 = c * c, c4 = c2 * c2;
    return 15.5f * c4 * c2 - 40.14f * c4 * c + 31.96f * c4 - 6.868f * c2 * c + 0.4298f * c2 + 0.1191f * c - 0.00232f;
  });
  if (punchy) {
    v = map(v, [](float c) { return std::pow(std::max(c, 0.0f), 1.35f); });
    const float l = dot709(v);
    v = map(v, [&](float c) { return l + 1.4f * (c - l); });
  }
  v = mul(out, v);
  return map(v, [](float c) { return clamp01(std::pow(std::max(c, 0.0f), 2.2f)); });
}
static V3 acesHill(V3 x) {
  static const float in[9] = { 0.59719f, 0.07600f, 0.02840f, 0.35458f, 0.90834f, 0.13383f, 0.04823f, 0.01566f, 0.83777f };
  static const float out[9] = { 1.60475f, -0.10208f, -0.00327f, -0.53108f, 1.10813f, -0.07276f, -0.07367f, -0.00605f, 1.07602f };
  V3 v = mul(in, x);
  v = map(v, [](float c) { return (c * (c + 0.0245786f) - 0.000090537f) / (c * (0.983729f * c + 0.4329510f) + 0.238081f); });
  return map(mul(out, v), clamp01);
}
static V3 tonemapRef(int op, V3 x, float white) {
  switch (op) {
    case 1: return map(x, [](float c) { return c / (1 + c); });
    case 2: return map(x, [&](float c) { return std::min(c * (1 + c / (white * white)) / (1 + c), 1.0f); });
    case 3: return map(x, [](float c) { c *= 0.6f; return clamp01((c * (2.51f * c + 0.03f)) / (c * (2.43f * c + 0.59f) + 0.14f)); });
    case 4: return acesHill(x);
    case 5: return agx(x, false);
    case 6: return agx(x, true);
    case 7: return map(x, [](float c) { return clamp01(hableCurve(2 * c) / hableCurve(11.2f)); });
    case 8: return map(x, uchimura);
    case 9: return map(x, lottes);
    default: return map(x, clamp01);
  }
}
static float rolloff(float y, float k, float m) {
  if (y <= k) return y;
  const float t = (y - k) / (m - k);
  return k + (m - k) * t / (1 + t);
}
// Scene value (gamma encoded) -> the effect's output (gamma encoded).
static V3 effectRef(int op, V3 s, float exposure, float contrast, float saturation, float white, float knee, bool hdr,
                    float peakOverPaper) {
  V3 x = map(s, [&](float c) { return std::pow(std::max(c, 0.0f), 2.2f) * std::exp2(exposure); });
  if (contrast != 1.0f) x = map(x, [&](float c) { return 0.18f * std::pow(std::max(c, 0.0f) / 0.18f, contrast); });
  if (saturation != 1.0f) {
    const float l = dot709(x);
    x = map(x, [&](float c) { return std::max(l + saturation * (c - l), 0.0f); });
  }
  V3 t = tonemapRef(op, x, white);
  if (hdr) {
    const float y = dot709(x);
    if (y > knee) t = t * (rolloff(y, knee, std::max(peakOverPaper, 1.001f)) / rolloff(y, knee, 1.0f));
  }
  return map(t, [](float c) { return std::pow(c, 1.0f / 2.2f); });
}

static float halfF(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
  float v = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? INFINITY : std::ldexp(float(m | 0x400), int(e) - 25);
  return sign ? -v : v;
}
static uint16_t toHalf(float f) {  // round to nearest, normal range only (test values)
  if (f == 0.0f) return 0;
  int e = 0;
  const float m = std::frexp(f, &e);  // f = m * 2^e, m in [0.5, 1)
  int he = e - 1 + 15;
  uint32_t man = uint32_t(std::lround((m * 2.0f - 1.0f) * 1024.0f));
  if (man == 1024) { man = 0; he++; }
  return uint16_t((he << 10) | man);
}

static const char* kOps[] = { "vanilla", "reinhard", "reinhard ext", "aces narkowicz", "aces hill", "agx", "agx punchy",
                              "hable", "uchimura", "lottes" };

// Scene test pattern: gamma-encoded values from black to far above SDR white, grey and two colours.
static V3 patternAt(uint32_t x, uint32_t y) {
  static const float v[16] = { 0.0f, 0.02f, 0.05f, 0.1f, 0.2f, 0.3f, 0.46f, 0.6f, 0.73f, 0.9f, 1.0f, 1.3f, 1.8f, 2.5f, 4.0f, 8.0f };
  const float s = v[x % 16];
  switch (y % 3) {
    case 0: return { s, s, s };
    case 1: return { s, 0.6f * s, 0.25f * s };
    default: return { 0.2f * s, 0.5f * s, s };
  }
}

static void run(Backend& be, RtKey bb, RtKey bbDepth) {
  // CPU reference anchors, computed independently (numpy) from the published formulas.
  struct Anchor { const char* what; float got, want; } anchors[] = {
    { "narkowicz(1)", tonemapRef(3, { 1, 1, 1 }, 4).r, 0.6732905f },
    { "hill(0.18)", tonemapRef(4, { 0.18f, 0.18f, 0.18f }, 4).r, 0.1055912f },
    { "agx(0.18)", tonemapRef(5, { 0.18f, 0.18f, 0.18f }, 4).r, 0.2144674f },
    { "hable(0.18)", tonemapRef(7, { 0.18f, 0.18f, 0.18f }, 4).r, 0.1283384f },
    { "uchimura(0.18)", tonemapRef(8, { 0.18f, 0.18f, 0.18f }, 4).r, 0.1789948f },
    { "uchimura(1)", tonemapRef(8, { 1, 1, 1 }, 4).r, 0.8278324f },
    { "lottes(0.18)", tonemapRef(9, { 0.18f, 0.18f, 0.18f }, 4).r, 0.267f },
    { "lottes(8)", tonemapRef(9, { 8, 8, 8 }, 4).r, 1.0f },
  };
  for (const Anchor& a : anchors)
    if (std::fabs(a.got - a.want) > 1e-4f) fail("tonemap reference: %s = %.6f, expected %.6f", a.what, a.got, a.want);

  std::vector<fx::Effect> loaded = fx::loadEffects("effects", "build/fx_cache_test", [](const char* s) { printf("[fx] %s\n", s); });
  const fx::Effect* tm = nullptr;
  for (const fx::Effect& e : loaded)
    if (e.id == "tonemap" && e.error.empty()) tm = &e;
  if (!tm) { fail("hdr: effects/tonemap did not load"); return; }
  if (tm->point != "tonemap") fail("hdr: tonemap effect point is '%s'", tm->point.c_str());
  int opIndex = -1;
  for (size_t k = 0; k < tm->params.size(); k++)
    if (tm->params[k].name == "op") opIndex = int(k);
  if (opIndex < 0 || tm->params[opIndex].type != fx::Param::Type::Enum || tm->params[opIndex].options.size() != 10)
    fail("hdr: tonemap 'op' is not an enum of 10 operators");
  Backend::EffectDesc d;
  d.sceneColor = tm->sceneColor;
  for (const fx::Image& im : tm->images) d.images.push_back({ im.vkFormat, im.scale, Backend::EffectDesc::None, im.width, im.height });
  for (const fx::Pass& p : tm->passes)
    d.passes.push_back({ { p.spv[0].data(), p.spv[1].data() }, { p.spv[0].size(), p.spv[1].size() }, p.output, uint32_t(p.blend) });
  const uint32_t id = be.createEffect(d);
  if (!id) { fail("hdr: the backend did not create the tonemap effect"); return; }

  const uint32_t W = 32, H = 32;
  const RtKey rt = newKey();
  be.attach(rt, { W, H, kD3DFMT_A16B16G16R16F, 1, false });
  std::vector<uint16_t> scene(size_t(W) * H * 4);
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++) {
      const V3 c = patternAt(x, y);
      uint16_t* t = &scene[(size_t(y) * W + x) * 4];
      t[0] = toHalf(c.r); t[1] = toHalf(c.g); t[2] = toHalf(c.b); t[3] = toHalf(1.0f);
    }
  Backend::EffectFrame fr{};
  const float port[4] = { 0, 1, 1, 0 }, frustum[4] = { -1, 1, 1, -1 };
  memcpy(fr.port, port, 16);
  memcpy(fr.frustum, frustum, 16);
  fr.nearZ = 1;
  fr.farZ = 2;

  // Runs every operator with the backend's current output; returns the worst difference.
  auto runAll = [&](bool hdr, float peakOverPaper, const char* label) {
    float worstAll = 0;
    for (int op = 0; op < 10; op++) {
      float params[32][4] = {};
      // Vanilla at the defaults (bit-exact clip); the others graded. Odd operators desaturate, even ones saturate.
      const float exposure = op == 0 ? 0.0f : 0.5f, white = 3.0f, knee = 0.5f;
      const float contrast = op == 0 ? 1.0f : op % 2 ? 1.25f : 0.8f, saturation = op == 0 ? 1.0f : op % 2 ? 0.6f : 1.5f;
      for (size_t k = 0; k < tm->params.size(); k++) {
        const std::string& n = tm->params[k].name;
        params[k][0] = n == "op" ? float(op) : n == "exposure" ? exposure : n == "white" ? white : n == "knee" ? knee
                     : n == "contrast" ? contrast : n == "saturation" ? saturation : tm->params[k].def;
      }
      be.writeTarget(rt, scene.data(), W * 8);
      be.beginGroup(&rt, 1, 0);
      const bool ok = be.runEffect(id, fr, &params[0][0], uint32_t(tm->params.size()));
      be.endGroup();
      std::vector<uint8_t> got;
      uint32_t vf = 0, ts = 0;
      if (!ok || !be.readback(rt, got, vf, ts) || ts != 8) { fail("hdr: %s %s: run %d, readback texel %u", label, kOps[op], ok, ts); continue; }
      float worst = 0;
      uint32_t wx = 0, wy = 0;
      V3 wantW{}, gotW{};
      for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x++) {
          const uint16_t* t = reinterpret_cast<const uint16_t*>(got.data()) + (size_t(y) * W + x) * 4;
          const V3 g{ halfF(t[0]), halfF(t[1]), halfF(t[2]) };
          const uint16_t* sIn = &scene[(size_t(y) * W + x) * 4];
          const V3 want = effectRef(op, { halfF(sIn[0]), halfF(sIn[1]), halfF(sIn[2]) }, exposure, contrast, saturation, white, knee,
                                    hdr, peakOverPaper);
          // Relative to the value (half precision: 2^-11 relative), absolute near black.
          const float e = std::max({ std::fabs(g.r - want.r) / std::max(1.0f, want.r), std::fabs(g.g - want.g) / std::max(1.0f, want.g),
                                     std::fabs(g.b - want.b) / std::max(1.0f, want.b) });
          if (e > worst) { worst = e; wx = x; wy = y; wantW = want; gotW = g; }
        }
      worstAll = std::max(worstAll, worst);
      if (worst > 4e-3f)
        fail("hdr: %s %s: texel (%u,%u) got %.4f %.4f %.4f, expected %.4f %.4f %.4f", label, kOps[op], wx, wy, gotW.r, gotW.g,
             gotW.b, wantW.r, wantW.g, wantW.b);
    }
    printf("hdr: tonemap %s: 10 operators, worst difference from the reference %.5f\n", label, worstAll);
  };

  // SDR.
  runAll(false, 1.0f, "SDR");

  // HDR output on the test window: present until the swapchain is recreated with it.
  HdrSettings hs;
  hs.output = HdrSettings::Hdr10;
  hs.paperWhiteNits = 200;
  hs.peakNits = 1000;
  be.setHdr(hs);
  auto presentFrames = [&](int n, bool menu) {
    for (int f = 0; f < n; f++) {
      if (menu) {
        uint32_t w = 0, h = 0;
        if (be.overlayBeginFrame(w, h)) {
          ImGuiIO& io = ImGui::GetIO();
          io.DisplaySize = ImVec2(float(w), float(h));
          io.DeltaTime = 1.0f / 60.0f;
          ImGui::NewFrame();
          ImGui::Begin("hdr test menu");
          ImGui::Text("HDR frame %d", f);
          ImGui::End();
          ImGui::Render();
          be.overlaySubmit(true);
        }
      }
      be.beginGroup(&bb, 1, bbDepth);
      int32_t r[4] = { 0, 0, 1280, 720 };
      be.clear(kClearTarget | kClearZBuffer, 0xFF808080, 1.0f, 0, r);
      be.endGroup();
      be.present(bb);
    }
  };
  presentFrames(2, false);
  const HdrStatus st = be.hdrStatus();
  printf("hdr: test window surface offers HDR10 %d, scRGB %d; active output %u\n", st.hdr10Available, st.scRgbAvailable, st.active);
  if (st.active != HdrSettings::Off) {
    runAll(true, 1000.0f / 200.0f, st.active == HdrSettings::Hdr10 ? "HDR10" : "scRGB");
    const uint64_t before = be.stats().overlayFrames;
    presentFrames(4, true);
    const uint64_t drawn = be.stats().overlayFrames - before;
    printf("hdr: menu layer drawn on %llu of 4 HDR presents\n", drawn);
    if (drawn != 4) fail("hdr: menu layer drawn on %llu of 4 HDR presents", drawn);
  } else {
    printf("hdr: NOTE the test window's surface offers no HDR format: the HDR branch of the tonemap and the HDR present "
           "were not exercised on a swapchain (the encode is verified offscreen below)\n");
  }

  // The present encode, offscreen, for both outputs, of the plain pattern, against the formulas.
  float worstPq = 0, worstSc = 0;
  for (uint32_t output : { uint32_t(HdrSettings::Hdr10), uint32_t(HdrSettings::ScRgb) }) {
    be.writeTarget(rt, scene.data(), W * 8);
    std::vector<uint8_t> got;
    uint32_t vf = 0;
    if (!be.readbackHdrEncode(rt, output, got, vf)) { fail("hdr: encode readback (output %u) failed", output); continue; }
    static const float m709to2020[9] = { 0.627404f, 0.069097f, 0.016391f, 0.329283f, 0.919540f, 0.088013f, 0.043313f, 0.011362f, 0.895595f };
    for (uint32_t y = 0; y < H; y++)
      for (uint32_t x = 0; x < W; x++) {
        const uint16_t* sIn = &scene[(size_t(y) * W + x) * 4];
        const V3 s{ halfF(sIn[0]), halfF(sIn[1]), halfF(sIn[2]) };
        const V3 nits = map(s, [&](float c) { return std::min(std::pow(std::max(c, 0.0f), 2.2f) * hs.paperWhiteNits, hs.peakNits); });
        if (output == HdrSettings::Hdr10) {
          const V3 n2020 = mul(m709to2020, nits);
          auto pq = [](float n) {
            const float m1 = 0.1593017578125f, m2 = 78.84375f, c1 = 0.8359375f, c2 = 18.8515625f, c3 = 18.6875f;
            const float yy = std::pow(clamp01(n / 10000.0f), m1);
            return std::pow((c1 + c2 * yy) / (1 + c3 * yy), m2);
          };
          const uint32_t px = reinterpret_cast<const uint32_t*>(got.data())[size_t(y) * W + x];
          const float gr = float(px & 0x3FF) / 1023.0f, gg = float((px >> 10) & 0x3FF) / 1023.0f, gb = float((px >> 20) & 0x3FF) / 1023.0f;
          const float e = std::max({ std::fabs(gr - pq(n2020.r)), std::fabs(gg - pq(n2020.g)), std::fabs(gb - pq(n2020.b)) }) * 1023.0f;
          worstPq = std::max(worstPq, e);
        } else {
          const uint16_t* t = reinterpret_cast<const uint16_t*>(got.data()) + (size_t(y) * W + x) * 4;
          const V3 want = nits * (1.0f / 80.0f);
          const float e = std::max({ std::fabs(halfF(t[0]) - want.r) / std::max(1.0f, want.r), std::fabs(halfF(t[1]) - want.g) / std::max(1.0f, want.g),
                                     std::fabs(halfF(t[2]) - want.b) / std::max(1.0f, want.b) });
          worstSc = std::max(worstSc, e);
        }
      }
  }
  printf("hdr: encode: HDR10 worst %.3f codes (of 1023), scRGB worst relative %.5f\n", worstPq, worstSc);
  if (worstPq > 1.0f) fail("hdr: HDR10 encode differs by %.3f codes", worstPq);
  if (worstSc > 2e-3f) fail("hdr: scRGB encode differs by %.5f", worstSc);

  // Wide targets: an A8R8G8B8 target of the window size.
  if (be.hdrStatus().wideTargets) {
    const uint32_t ww = 1176, wh = 664;
    const RtKey wide = newKey();
    be.attach(wide, { ww, wh, kD3DFMT_A8R8G8B8, 1, false });
    std::vector<uint8_t> bgra(size_t(ww) * wh * 4);
    for (size_t i = 0; i < bgra.size(); i++) bgra[i] = uint8_t((i * 2654435761u) >> 13);
    be.writeTarget(wide, bgra.data(), ww * 4);
    std::vector<uint8_t> back;
    uint32_t vf = 0, ts = 0;
    if (!be.readback(wide, back, vf, ts) || ts != 4 || vf != 44 /* VK_FORMAT_B8G8R8A8_UNORM */ || back != bgra)
      fail("wide target: writeTarget/readback do not round-trip the D3D9 layout (texel %u, format %u, %zu bytes)", ts, vf, back.size());
    else
      printf("wide target: %ux%u A8R8G8B8 round-trips its D3D9 layout through writeTarget/readback\n", ww, wh);
    // Values above 1.0 survive: copy the HDR pattern in, encode, check a texel that is 4.0 (gamma encoded).
    be.writeTarget(rt, scene.data(), W * 8);
    const int32_t sr[4] = { 0, 0, int32_t(W), int32_t(H) }, dr[4] = { 0, 0, int32_t(W), int32_t(H) };
    be.copyTarget(rt, wide, sr, dr, 1);
    std::vector<uint8_t> enc;
    if (!be.readbackHdrEncode(wide, HdrSettings::ScRgb, enc, vf)) {
      fail("wide target: encode readback failed");
    } else {
      const uint16_t* t = reinterpret_cast<const uint16_t*>(enc.data()) + 14 * 4;  // texel (14, 0): grey 4.0
      const float nits = halfF(t[0]) * 80.0f;
      printf("wide target: a value of 4.0 copied in is shown at %.0f nits (peak %.0f)\n", nits, hs.peakNits);
      if (std::fabs(nits - hs.peakNits) > 2.0f) fail("wide target: value above 1.0 was not kept (%.1f nits)", nits);
    }
    be.destroy(wide);
  } else {
    printf("wide target: not enabled (FNV_WIDE_TARGETS=1 runs the suite with wide targets)\n");
  }

  hs.output = HdrSettings::Off;
  be.setHdr(hs);
  presentFrames(2, false);
  if (be.hdrStatus().active != HdrSettings::Off) fail("hdr: output did not return to SDR");
  be.destroyEffect(id);
  be.destroy(rt);
}

}  // namespace hdrtest
