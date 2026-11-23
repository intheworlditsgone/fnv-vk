// Volumetric fog (effects/volfog) and auto-exposure (effects/tonemap), against answers computed here (included by
// backend_test.cpp, after ssgi_test.inl).
namespace atmostest {

using sunshaftstest::halfToFloat;

static const fx::Effect* findEffect(const std::vector<fx::Effect>& all, const char* id) {
  for (const fx::Effect& x : all)
    if (x.id == id) return x.error.empty() ? &x : nullptr;
  return nullptr;
}
static uint32_t createEffect(Backend& be, const fx::Effect& e) {
  Backend::EffectDesc d;
  d.sceneColor = e.sceneColor;
  d.skinBuffer = e.skinBuffer;  // as the plugin registers it (effects.cpp backendId)
  for (const fx::Image& im : e.images) d.images.push_back({ im.vkFormat, im.scale, Backend::EffectDesc::None, im.width, im.height });
  for (const fx::Pass& p : e.passes)
    d.passes.push_back({ { p.spv[0].data(), p.spv[1].data() }, { p.spv[0].size(), p.spv[1].size() }, p.output, uint32_t(p.blend),
                         p.skipPbr });
  return be.createEffect(d);
}
static void setParam(const fx::Effect& e, float (*params)[4], const char* name, float v) {
  for (size_t i = 0; i < e.params.size(); i++)
    if (e.params[i].name == name) { params[i][0] = v; return; }
  fail("atmos: no parameter '%s' in '%s'", name, e.id.c_str());
}
static double lin(double c) { return std::pow(std::max(c, 0.0), 2.2); }
static double gam(double c) { return std::pow(std::max(c, 0.0), 1.0 / 2.2); }

// Volumetric fog over a scene of known depth: the camera at height 500 looking along world x (its up is world z), a
// grey wall at view distance 2000 in the left half, open sky in the right half. Per pixel the oracle is exact: the
// transmittance of the exponential height fog along the pixel's ray (to the wall, or the range over the sky), the
// sky light (1 - T) * ambient, and with the sun visible everywhere (empty cascades) the sun's in-scattered light
// phase * pi * (1 - T) (the marched shadowed part and the lit remainder add up to that exactly). Checks: without
// cascades (no sunlight) every pixel to 1e-3 relative; with empty cascades the half-resolution march at each of its
// sample pixels, and the final image against the oracle's sun term passed through upsample.frag's interpolation (both
// to 2e-3; the silhouette columns, where the depth weights choose sides, are left out); with cascades that shadow
// everything, the march's light is only what comes from beyond their range (view distance 1500), phase * pi *
// (T(1500) - T(end)); indoors and at density 0 the image unchanged.
static void runFog(Backend& be, const std::vector<fx::Effect>& all) {
  const fx::Effect* e = findEffect(all, "volfog");
  if (!e) { fail("volfog: effects/volfog did not load"); return; }
  const uint32_t id = createEffect(be, *e);
  if (!id) { fail("volfog: effect not created"); return; }
  constexpr uint32_t W = 128, H = 64;
  const double zn = 10.0, zf = 100000.0, zWall = 2000.0;
  auto depthOf = [&](double z) { return float(zf * (z - zn) / (z * (zf - zn))); };
  float params[32][4] = {};
  for (size_t i = 0; i < e->params.size() && i < 32; i++) params[i][0] = e->params[i].def;
  const double density = 2.0, falloff = 1000.0, base = 300.0, range = 50000.0, g = 0.6;
  setParam(*e, params, "density", float(density));
  setParam(*e, params, "falloff", float(falloff));
  setParam(*e, params, "baseHeight", float(base));
  setParam(*e, params, "range", float(range));
  setParam(*e, params, "anisotropy", float(g));
  setParam(*e, params, "sun", 1.0f);
  setParam(*e, params, "ambient", 1.0f);

  Backend::EffectFrame fr{};
  const float frustum[4] = { -1.0f, 1.0f, 1.0f, -1.0f }, port[4] = { 0.0f, 1.0f, 1.0f, 0.0f };
  memcpy(fr.frustum, frustum, 16);
  memcpy(fr.port, port, 16);
  fr.nearZ = float(zn);
  fr.farZ = float(zf);
  // Camera rows (world = T + R (z, y, x)): forward world x, up world z, right world y; at height 500.
  const float rows[3][4] = { { 1, 0, 0, 0 }, { 0, 0, 1, 0 }, { 0, 1, 0, 500 } };
  memcpy(fr.cameraRows, rows, sizeof(rows));
  const float sun[3] = { 0.9f, 0.8f, 0.6f }, amb[3] = { 0.5f, 0.6f, 0.7f };
  memcpy(fr.sunColor, sun, 12);
  memcpy(fr.ambientColor, amb, 12);
  const double sv[3] = { 0.3, 0.4, 0.866 };
  const double svl = std::sqrt(sv[0] * sv[0] + sv[1] * sv[1] + sv[2] * sv[2]);
  for (int i = 0; i < 3; i++) fr.sunView[i] = float(sv[i] / svl);
  // Empty cascades: every view point maps to the middle of cascade 0 at depth 0 (in front of everything: lit).
  float toCentre[16] = {};
  toCentre[15] = 1.0f;
  // Full cascades: every view point maps to depth 2, behind everything in the map: in shadow.
  float behind[16] = {};
  behind[11] = 2.0f;
  behind[15] = 1.0f;

  enum Mode { NoSun, Sun, Shadowed, Interior, Off };
  std::vector<uint8_t> scatter, shadowed;  // the half-resolution march of the sun and shadowed runs
  uint32_t sw = 0, sh = 0;
  auto runScene = [&](Mode mode, std::vector<uint8_t>& img) {
    const RtKey rt = newKey(), depth = newKey();
    be.attach(rt, { W, H, kD3DFMT_A16B16G16R16F, 1, false });
    be.attach(depth, { W, H, kD3DFMT_D24S8, 1, true });
    be.beginGroup(&rt, 1, depth);
    const int32_t left[4] = { 0, 0, int32_t(W / 2), int32_t(H) }, right[4] = { int32_t(W / 2), 0, int32_t(W), int32_t(H) };
    be.clear(kClearTarget | kClearZBuffer, 0xFF808080u, depthOf(zWall), 0, left);
    be.clear(kClearTarget | kClearZBuffer, 0xFF808080u, 1.0f, 0, right);
    fr.shadowCascades = 0;
    if (mode == Sun || mode == Shadowed) {
      if (!be.beginShadowMaps(1, 256)) fail("volfog: shadow maps");
      be.beginShadowCascade(0, toCentre);
      be.endShadowMaps();
      fr.shadowCascades = 1;
      memcpy(fr.shadowFromView[0], mode == Sun ? toCentre : behind, 64);
      fr.shadowSplits[0] = 1500.0f;  // the wall (2000) and the sky lie partly beyond the shadow range
    }
    fr.interior = mode == Interior;
    float p[32][4];
    memcpy(p, params, sizeof(p));
    if (mode == Off) setParam(*e, p, "density", 0.0f);
    if (!be.runEffect(id, fr, &p[0][0], uint32_t(std::min<size_t>(e->params.size(), 32)))) fail("volfog: run failed");
    be.endGroup();
    uint32_t vf = 0, ts = 0;
    bool ok = be.readback(rt, img, vf, ts) && img.size() == size_t(W) * H * 8;
    if (mode == Sun || mode == Shadowed) {
      std::vector<uint8_t>& out = mode == Sun ? scatter : shadowed;
      ok = ok && be.readbackEffectImage(id, 0, out, sw, sh) && sw == W / 2 && sh == H / 2 && out.size() == size_t(sw) * sh * 4;
    }
    be.destroy(rt);
    be.destroy(depth);
    if (!ok) fail("volfog: readback");
    return ok;
  };
  auto at = [&](const std::vector<uint8_t>& img, uint32_t x, uint32_t y, int c) {
    return double(halfToFloat(reinterpret_cast<const uint16_t*>(img.data())[(size_t(y) * W + x) * 4 + c]));
  };
  // The oracle along the ray through the centre of full-resolution pixel (x, y): transmittance, the sun's light with
  // the sun visible everywhere, and the view distance of the ray's end.
  struct Ray { double T, S, zEnd, shadowedS; };
  auto oracle = [&](uint32_t x, uint32_t y) {
    const double u = (x + 0.5) / W, v = (y + 0.5) / H;
    const double ray[3] = { -1.0 + 2.0 * u, 1.0 - 2.0 * v, 1.0 };
    const double rayLen = std::sqrt(ray[0] * ray[0] + ray[1] * ray[1] + 1.0);
    const double zEnd = x < W / 2 ? zWall : range / rayLen;
    const double s = zEnd * rayLen, k = (ray[1] / rayLen) / falloff;  // world z = view y
    const double sigmaC = density * 1e-4 * std::exp(base / falloff);
    const double xk = k * s;
    const double tau = std::abs(xk) < 1e-9 ? sigmaC * s : sigmaC * (1.0 - std::exp(-std::min(xk, 80.0))) / k;
    const double T = std::exp(-tau);
    double cosT = 0;
    for (int i = 0; i < 3; i++) cosT += ray[i] / rayLen * fr.sunView[i];
    const double phase = (1 - g * g) / (12.566371 * std::pow(std::max(1 + g * g - 2 * g * cosT, 1e-4), 1.5));
    // In shadow inside the cascades' range (view distance 1500): only the light from beyond it.
    const double s1 = std::min(zEnd, 1500.0) * rayLen, x1 = k * s1;
    const double T1 = std::exp(-(std::abs(x1) < 1e-9 ? sigmaC * s1 : sigmaC * (1.0 - std::exp(-std::min(x1, 80.0))) / k));
    return Ray{ T, phase * 3.14159265 * (1.0 - T), zEnd, phase * 3.14159265 * (T1 - T) };
  };
  // The march samples half-resolution texel (i, j) at full-resolution pixel (2i + 1, 2j + 1) (march.frag: fnvPixel()).
  auto sample = [&](int i, int j) {
    i = std::clamp(i, 0, int(W / 2) - 1);
    j = std::clamp(j, 0, int(H / 2) - 1);
    return oracle(uint32_t(std::min<int>(2 * i + 1, W - 1)), uint32_t(std::min<int>(2 * j + 1, H - 1)));
  };
  // upsample.frag's interpolation of the samples' sun light at pixel (x, y).
  auto upsampled = [&](uint32_t x, uint32_t y) {
    const double hx = (x + 0.5) * 0.5 - 0.5, hy = (y + 0.5) * 0.5 - 0.5;
    const int bx = int(std::floor(hx)), by = int(std::floor(hy));
    const double fx = hx - bx, fy = hy - by, zc = std::min(oracle(x, y).zEnd, 60000.0);
    double sum = 0, weights = 0;
    for (int j = 0; j < 2; j++)
      for (int i = 0; i < 2; i++) {
        const Ray r = sample(bx + i, by + j);
        const double bil = (i == 0 ? 1 - fx : fx) * (j == 0 ? 1 - fy : fy);
        const double w = bil * std::max(1.0 - std::abs(std::min(r.zEnd, 60000.0) - zc) / (0.05 * zc + 1.0), 1e-3);
        sum += r.S * w;
        weights += w;
      }
    return sum / weights;
  };
  auto expected = [&](uint32_t x, uint32_t y, bool withSun, double out[3]) {
    const Ray r = oracle(x, y);
    const double S = withSun ? upsampled(x, y) : 0.0;
    for (int c = 0; c < 3; c++) out[c] = gam(lin(128.0 / 255.0) * r.T + lin(sun[c]) * S + lin(amb[c]) * (1.0 - r.T));
  };
  std::vector<uint8_t> noSun, withSun, inShadow, interior, off;
  if (!runScene(NoSun, noSun) || !runScene(Sun, withSun) || !runScene(Shadowed, inShadow) || !runScene(Interior, interior) ||
      !runScene(Off, off)) {
    be.destroyEffect(id);
    return;
  }
  double errNoSun = 0, errSun = 0, errMarch = 0, errShadowed = 0, minT = 1, maxT = 0;
  for (uint32_t j = 1; j + 1 < H / 2; j++)
    for (uint32_t i = 1; i + 1 < W / 2; i++) {
      if (i >= W / 4 - 2 && i <= W / 4 + 1) continue;
      const double want = sample(int(i), int(j)).S;
      const double got = halfToFloat(reinterpret_cast<const uint16_t*>(scatter.data())[(size_t(j) * sw + i) * 2]);
      errMarch = std::max(errMarch, std::abs(got - want) / std::max(want, 1e-3));
      const double wantS = sample(int(i), int(j)).shadowedS;
      const double gotS = halfToFloat(reinterpret_cast<const uint16_t*>(shadowed.data())[(size_t(j) * sw + i) * 2]);
      errShadowed = std::max(errShadowed, std::abs(gotS - wantS) / std::max(want, 1e-3));
    }
  for (uint32_t y = 1; y + 1 < H; y++)
    for (uint32_t x = 1; x + 1 < W; x++) {
      if (x >= W / 2 - 3 && x <= W / 2 + 2) continue;  // the silhouette (half-resolution upsampling there)
      double a[3], b[3];
      expected(x, y, false, a);
      expected(x, y, true, b);
      for (int c = 0; c < 3; c++) {
        errNoSun = std::max(errNoSun, std::abs(at(noSun, x, y, c) - a[c]) / a[c]);
        errSun = std::max(errSun, std::abs(at(withSun, x, y, c) - b[c]) / b[c]);
      }
    }
  for (uint32_t x : { 10u, 100u }) {  // the transmittance range covered (wall, sky), for the log
    const double u = (x + 0.5) / W, ray0 = -1.0 + 2.0 * u, rayLen = std::sqrt(ray0 * ray0 + 1.0);
    const double s = (x < W / 2 ? zWall : range / rayLen) * rayLen;
    const double T = std::exp(-density * 1e-4 * std::exp(base / falloff) * s);
    minT = std::min(minT, T);
    maxT = std::max(maxT, T);
  }
  bool unchanged = true;
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++)
      for (int c = 0; c < 3; c++) {
        unchanged &= std::abs(at(interior, x, y, c) - 128.0 / 255.0) < 1e-3;
        unchanged &= std::abs(at(off, x, y, c) - 128.0 / 255.0) < 1e-3;
      }
  printf("volfog: transmittance on the middle row %.3f (sky) .. %.3f (wall); max relative error without sun %.5f; with "
         "the sun (empty cascades): march %.5f, image %.5f; in shadow inside the cascades' range: march %.5f (of the lit "
         "light); indoors and density 0 unchanged: %s\n", minT, maxT, errNoSun, errMarch, errSun, errShadowed,
         unchanged ? "yes" : "NO");
  if (errNoSun > 1e-3) fail("volfog: the fog without sunlight is not the exact transmittance and sky light");
  if (errMarch > 2e-3) fail("volfog: the march's sun light is not phase * pi * (1 - T) under empty cascades");
  if (errSun > 2e-3) fail("volfog: the image with the sun is not the march's light upsampled");
  if (errShadowed > 2e-3) fail("volfog: in shadow, the march's light is not only the light from beyond the cascades");
  if (!unchanged) fail("volfog: the image changed indoors or with density 0");
  be.destroyEffect(id);
}

// Auto-exposure over uniform grey images at the tonemap point (operator Vanilla: the image is the exposed colour, below
// 1 here): the first frame adapts to the measured luminance at once, so the grey comes out at mid grey (0.18 linear);
// after a step to a brighter grey and 0.1 s, the adapted value has moved by 1 - exp(-speed up * 0.1) of the way; after
// 3 s it is at the target again; a very dark grey gets at most the allowed brightening (+2 EV).
static void runExposure(Backend& be, const std::vector<fx::Effect>& all) {
  const fx::Effect* e = findEffect(all, "tonemap");
  if (!e) { fail("auto-exposure: effects/tonemap did not load"); return; }
  const uint32_t id = createEffect(be, *e);
  if (!id) { fail("auto-exposure: effect not created"); return; }
  constexpr uint32_t W = 512, H = 256;
  float params[32][4] = {};
  for (size_t i = 0; i < e->params.size() && i < 32; i++) params[i][0] = e->params[i].def;
  setParam(*e, params, "op", 0.0f);
  setParam(*e, params, "autoExposure", 1.0f);
  const double speedUp = 2.0, minEv = -2.0, maxEv = 2.0;
  setParam(*e, params, "aeSpeedUp", float(speedUp));
  setParam(*e, params, "aeMin", float(minEv));
  setParam(*e, params, "aeMax", float(maxEv));
  Backend::EffectFrame fr{};
  const float port[4] = { 0.0f, 1.0f, 1.0f, 0.0f }, frustum[4] = { -1.0f, 1.0f, 1.0f, -1.0f };
  memcpy(fr.port, port, 16);
  memcpy(fr.frustum, frustum, 16);
  fr.nearZ = 1.0f;
  fr.farZ = 2.0f;
  auto frame = [&](uint8_t grey, float time) {
    const RtKey rt = newKey();
    be.attach(rt, { W, H, kD3DFMT_A16B16G16R16F, 1, false });
    be.beginGroup(&rt, 1, 0);
    const int32_t full[4] = { 0, 0, int32_t(W), int32_t(H) };
    be.clear(kClearTarget, 0xFF000000u | uint32_t(grey) * 0x010101u, 1.0f, 0, full);
    fr.time = time;
    if (!be.runEffect(id, fr, &params[0][0], uint32_t(std::min<size_t>(e->params.size(), 32)))) fail("auto-exposure: run failed");
    be.endGroup();
    std::vector<uint8_t> img;
    uint32_t vf = 0, ts = 0;
    double v = -1.0;
    if (be.readback(rt, img, vf, ts) && img.size() == size_t(W) * H * 8)
      v = halfToFloat(reinterpret_cast<const uint16_t*>(img.data())[(size_t(H / 2) * W + W / 2) * 4 + 1]);
    be.destroy(rt);
    return v;
  };
  auto logLum = [](uint8_t grey) { return std::log2(lin(grey / 255.0)); };  // a grey's luminance is its channel value
  auto out = [&](uint8_t grey, double adapted) {
    const double ev = std::clamp(std::log2(0.18) - adapted, minEv, maxEv);
    return gam(lin(grey / 255.0) * std::exp2(ev));
  };
  const uint8_t dark = 77, bright = 153, night = 10;  // 0.30, 0.60, 0.04
  const double first = frame(dark, 1.0f);
  const double a1 = logLum(dark), m2 = logLum(bright);
  const double a2 = a1 + (m2 - a1) * (1.0 - std::exp(-speedUp * 0.1));
  const double second = frame(bright, 1.1f);
  double settled = 0.0;
  for (int i = 0; i < 30; i++) settled = frame(bright, 1.2f + 0.1f * float(i));
  double clamped = 0.0;
  for (int i = 0; i < 60; i++) clamped = frame(night, 4.2f + 0.1f * float(i));
  const double eFirst = out(dark, a1), eSecond = out(bright, a2), eSettled = gam(0.18), eClamped = gam(lin(night / 255.0) * 4.0);
  printf("auto-exposure: first frame %.4f (expected %.4f, mid grey), 0.1 s after a step to a brighter grey %.4f (expected "
         "%.4f), after 3 s %.4f (expected %.4f), a very dark grey %.4f (expected %.4f, +2 EV at most)\n",
         first, eFirst, second, eSecond, settled, eSettled, clamped, eClamped);
  auto close = [](double a, double b, double tol) { return std::abs(a - b) <= tol; };
  if (!close(first, eFirst, 2e-3)) fail("auto-exposure: the first frame is not at the target");
  if (!close(second, eSecond, 2e-3)) fail("auto-exposure: the adaptation step is not 1 - exp(-speed dt) of the way");
  if (!close(settled, eSettled, 2e-3)) fail("auto-exposure: it does not settle at the target");
  if (!close(clamped, eClamped, 2e-3)) fail("auto-exposure: the brightening is not limited to the allowed range");
  be.destroyEffect(id);
}

static void run(Backend& be) {
  std::vector<fx::Effect> all = fx::loadEffects("effects", "build/fx_cache_test", [](const char*) {});
  runFog(be, all);
  runExposure(be, all);
}

}  // namespace atmostest
