// Depth of field (effects/dof) and motion blur (effects/motionblur) on scenes with a known answer (included by
// backend_test.cpp). Frustum -1..1, near 10, far 100000; the scene colour is written (rgba16f), the depth cleared per
// region. Colours are gamma encoded; checks are in linear light (value^2.2), where both effects conserve energy.
//
// Depth of field, manual focus at 140 units, f/1.4, 100 mm (radius r(z) from the thin lens, dofRef):
// - a plane at the focus: the image unchanged;
// - a 4 x 4 white square on a plane behind the focus (r 8 px) and in front of it (r -7.3 px): its light spreads over a
//   disc of that radius (energy within 10%, nothing beyond r + 4.5, the disc lit at half its radius);
// - a near plane's edge over an in-focus one: the near blur spreads over the in-focus side (partial coverage there,
//   untouched beyond the blur radius);
// - an in-focus white half in front of a blurred black background: the in-focus pixels stay white (no halo);
// - autofocus on a plane at 300 units: the focus image holds 300.
// Motion blur (shutter 1, longest blur 10% of the height), a 2 px white column on a plane at 1000:
// - first run (no previous camera): unchanged;
// - the camera moved 125 units right (the column moves 16 px): the column's light spreads over the 16 px streak (energy
//   within 5% on every row, nothing beyond 10 px, both sides lit);
// - the camera still: unchanged; moved 1000 units (a cut): unchanged.
namespace dofmbtest {

using hdrtest::halfF;
using hdrtest::toHalf;

static uint32_t create(Backend& be, const fx::Effect& e) {
  Backend::EffectDesc d;
  d.sceneColor = e.sceneColor;
  for (const fx::Image& im : e.images) d.images.push_back({ im.vkFormat, im.scale, Backend::EffectDesc::None, im.width, im.height });
  for (const fx::Pass& p : e.passes)
    d.passes.push_back({ { p.spv[0].data(), p.spv[1].data() }, { p.spv[0].size(), p.spv[1].size() }, p.output, uint32_t(p.blend) });
  return be.createEffect(d);
}

struct Scene {
  uint32_t W, H;
  std::vector<float> lin;  // linear light, one value per pixel (grey)
  std::vector<int32_t> rects;  // depth regions: left, top, right, bottom
  std::vector<double> depths;
};

constexpr double kNear = 10.0, kFar = 100000.0;
static float depthOf(double z) { return float(kFar * (z - kNear) / (z * (kFar - kNear))); }

// Runs effect `id` once over the scene; the result in linear light.
static bool runScene(Backend& be, uint32_t id, const Scene& s, Backend::EffectFrame& fr, const float (*params)[4], uint32_t n,
                     std::vector<float>& out) {
  const RtKey rt = newKey(), depth = newKey();
  be.attach(rt, { s.W, s.H, kD3DFMT_A16B16G16R16F, 1, false });
  be.attach(depth, { s.W, s.H, kD3DFMT_D24S8, 1, true });
  std::vector<uint16_t> texels(size_t(s.W) * s.H * 4);
  for (size_t i = 0; i < s.lin.size(); i++) {
    const uint16_t v = toHalf(std::pow(s.lin[i], 1.0f / 2.2f));
    texels[i * 4] = texels[i * 4 + 1] = texels[i * 4 + 2] = v;
    texels[i * 4 + 3] = toHalf(1.0f);
  }
  be.writeTarget(rt, texels.data(), s.W * 8);
  be.beginGroup(&rt, 1, depth);
  for (size_t r = 0; r < s.depths.size(); r++) be.clear(kClearZBuffer, 0, depthOf(s.depths[r]), 0, &s.rects[r * 4]);
  const bool ok = be.runEffect(id, fr, &params[0][0], n);
  be.endGroup();
  std::vector<uint8_t> got;
  uint32_t vf = 0, ts = 0;
  const bool read = ok && be.readback(rt, got, vf, ts) && ts == 8;
  be.destroy(rt);
  be.destroy(depth);
  if (!read) return false;
  out.resize(size_t(s.W) * s.H);
  for (size_t i = 0; i < out.size(); i++) out[i] = std::pow(std::max(halfF(reinterpret_cast<const uint16_t*>(got.data())[i * 4]), 0.0f), 2.2f);
  return true;
}

static Backend::EffectFrame frameFor() {
  Backend::EffectFrame fr{};
  const float frustum[4] = { -1.0f, 1.0f, 1.0f, -1.0f }, port[4] = { 0.0f, 1.0f, 1.0f, 0.0f };
  memcpy(fr.frustum, frustum, 16);
  memcpy(fr.port, port, 16);
  fr.nearZ = float(kNear);
  fr.farZ = float(kFar);
  for (int r = 0; r < 3; r++) fr.cameraRows[r][r] = 1.0f;
  return fr;
}

static const fx::Effect* find(const std::vector<fx::Effect>& all, const char* id) {
  for (const fx::Effect& x : all)
    if (x.id == id) {
      if (!x.error.empty()) fail("%s: did not load: %s", id, x.error.c_str());
      return x.error.empty() ? &x : nullptr;
    }
  fail("%s: not found", id);
  return nullptr;
}

static void setParams(const fx::Effect& e, float (*params)[4], std::initializer_list<std::pair<const char*, float>> values) {
  for (size_t k = 0; k < e.params.size() && k < 32; k++) {
    params[k][0] = e.params[k].def;
    for (const auto& [name, v] : values)
      if (e.params[k].name == name) params[k][0] = v;
  }
}

// The thin lens of effects/dof (coc.frag): blur radius in full-resolution pixels at view distance z.
static double dofRef(double z, double focus, double focal, double fNumber, double width) {
  const double a = focal / fNumber, zf = focus * 14.2875, zm = z * 14.2875;
  return 0.5 * (a * focal * (zm - zf) / (zm * (zf - focal))) / 36.0 * width;
}

static void dof(Backend& be, const fx::Effect& e) {
  const uint32_t id = create(be, e);
  if (!id) { fail("dof: effect not created"); return; }
  constexpr uint32_t W = 512, H = 256;
  const double focus = 140.0, focal = 100.0, fNumber = 1.4;
  float params[32][4] = {};
  setParams(e, params, { { "fNumber", float(fNumber) }, { "focalLength", float(focal) }, { "autofocus", 0.0f },
                         { "focusDistance", float(focus * 0.0142875) }, { "maxBlur", 4.0f }, { "nearBlur", 1.0f }, { "samples", 64.0f } });
  const uint32_t n = uint32_t(std::min<size_t>(e.params.size(), 32));
  Backend::EffectFrame fr = frameFor();
  const int32_t full[4] = { 0, 0, int32_t(W), int32_t(H) };
  auto square = [&](double z) {
    Scene s{ W, H, std::vector<float>(size_t(W) * H, 0.0f), { full[0], full[1], full[2], full[3] }, { z } };
    for (uint32_t y = H / 2 - 2; y < H / 2 + 2; y++)
      for (uint32_t x = W / 2 - 2; x < W / 2 + 2; x++) s.lin[size_t(y) * W + x] = 1.0f;
    return s;
  };

  // In focus: unchanged.
  {
    const Scene s = square(focus);
    std::vector<float> out;
    if (!runScene(be, id, s, fr, params, n, out)) { fail("dof: run failed"); return; }
    float worst = 0;
    for (size_t i = 0; i < out.size(); i++) worst = std::max(worst, std::fabs(out[i] - s.lin[i]));
    printf("dof: plane at the focus: worst change %.5f\n", worst);
    if (worst > 4e-3f) fail("dof: an in-focus plane changed (by %.4f)", worst);
  }
  // A square behind and in front of the focus.
  for (const double z : { 200.0, 110.0 }) {
    const double r = dofRef(z, focus, focal, fNumber, W);
    const Scene s = square(z);
    std::vector<float> out;
    if (!runScene(be, id, s, fr, params, n, out)) { fail("dof: run failed"); return; }
    double energy = 0, beyond = 0;
    float atHalf = 0;
    for (uint32_t y = 0; y < H; y++)
      for (uint32_t x = 0; x < W; x++) {
        const double d = std::hypot(x + 0.5 - W / 2.0, y + 0.5 - H / 2.0);
        const float v = out[size_t(y) * W + x];
        energy += v;
        // The square's corners are 2.83 px from its centre; half resolution and the bilinear upsampling reach 1.5 more.
        if (d > std::fabs(r) + 4.5) beyond = std::max(beyond, double(v));
      }
    atHalf = out[size_t(H / 2) * W + W / 2 + uint32_t(std::fabs(r) * 0.5)];
    const double disc = 16.0 / (3.14159265 * r * r);  // the square's light over the disc
    printf("dof: square at %.0f (radius %.2f px): energy %.3f (16), beyond r + 4.5 px %.5f, at r / 2 %.4f (disc %.4f), centre %.4f\n",
           z, r, energy, beyond, atHalf, disc, out[size_t(H / 2) * W + W / 2]);
    if (std::fabs(energy - 16.0) > 1.6) fail("dof: square at %.0f: energy %.3f, expected 16", z, energy);
    if (beyond > 2e-3) fail("dof: square at %.0f: light %.4f beyond the blur radius", z, beyond);
    if (!(atHalf > 0.3 * disc && atHalf < 3.0 * disc)) fail("dof: square at %.0f: %.4f at half the radius, the disc is %.4f", z, atHalf, disc);
  }
  // Near blur over the in-focus: left half white at 110 (r -7.3), right half black in focus.
  {
    const double r = std::fabs(dofRef(110.0, focus, focal, fNumber, W));
    Scene s{ W, H, std::vector<float>(size_t(W) * H, 0.0f), { 0, 0, int32_t(W / 2), int32_t(H), int32_t(W / 2), 0, int32_t(W), int32_t(H) }, { 110.0, focus } };
    for (uint32_t y = 0; y < H; y++)
      for (uint32_t x = 0; x < W / 2; x++) s.lin[size_t(y) * W + x] = 1.0f;
    std::vector<float> out;
    if (!runScene(be, id, s, fr, params, n, out)) { fail("dof: run failed"); return; }
    const size_t row = size_t(H / 2) * W;
    const float edge = out[row + W / 2 + 1], past = out[row + W / 2 + uint32_t(r) + 4], inside = out[row + W / 4];
    printf("dof: near edge over in-focus: right of the edge %.3f, past the radius %.5f, inside %.3f\n", edge, past, inside);
    // 1.5 px past the edge of a half plane blurred by radius r: about 1/2 - (2/pi)(1.5/r) of its disc covers it (0.37).
    if (!(edge > 0.2f && edge < 0.55f)) fail("dof: the near blur over the in-focus side 1.5 px past the edge is %.3f, expected ~0.37", edge);
    if (past > 2e-3f) fail("dof: light past the near blur radius (%.4f)", past);
    if (std::fabs(inside - 1.0f) > 0.02f) fail("dof: inside the near plane %.3f, expected 1", inside);
  }
  // In-focus white in front of a blurred black background (at 400, far behind the focus): no halo onto the white.
  {
    Scene s{ W, H, std::vector<float>(size_t(W) * H, 0.0f), { 0, 0, int32_t(W / 2), int32_t(H), int32_t(W / 2), 0, int32_t(W), int32_t(H) }, { focus, 400.0 } };
    for (uint32_t y = 0; y < H; y++)
      for (uint32_t x = 0; x < W / 2; x++) s.lin[size_t(y) * W + x] = 1.0f;
    std::vector<float> out;
    if (!runScene(be, id, s, fr, params, n, out)) { fail("dof: run failed"); return; }
    const size_t row = size_t(H / 2) * W;
    float worst = 0;
    for (uint32_t x = 0; x < W / 2; x++) worst = std::max(worst, std::fabs(out[row + x] - 1.0f));
    printf("dof: in-focus edge over a blurred background: worst change on the in-focus side %.4f, background next to it %.3f\n", worst,
           out[row + W / 2 + 1]);
    if (worst > 4e-3f) fail("dof: the in-focus side changed by %.4f (background halo)", worst);
  }
  // Autofocus: the focus image (index 0) holds the plane's distance.
  {
    float af[32][4];
    memcpy(af, params, sizeof(af));
    setParams(e, af, { { "fNumber", float(fNumber) }, { "focalLength", float(focal) }, { "autofocus", 1.0f } });
    const Scene s = square(300.0);
    std::vector<float> out;
    std::vector<uint8_t> f;
    uint32_t fw = 0, fh = 0;
    if (!runScene(be, id, s, fr, af, n, out) || !be.readbackEffectImage(id, 0, f, fw, fh) || f.size() < 4) {
      fail("dof: autofocus run failed");
    } else {
      float got;
      memcpy(&got, f.data(), 4);
      printf("dof: autofocus on a plane at 300: focus %.2f\n", got);
      if (std::fabs(got - 300.0f) > 1.5f) fail("dof: autofocus %.2f, expected 300", got);
    }
  }
  be.destroyEffect(id);
}

static void motionBlur(Backend& be, const fx::Effect& e) {
  const uint32_t id = create(be, e);
  if (!id) { fail("motion blur: effect not created"); return; }
  constexpr uint32_t W = 256, H = 128;
  float params[32][4] = {};
  setParams(e, params, { { "shutter", 1.0f }, { "maxBlur", 10.0f }, { "samples", 32.0f } });
  const uint32_t n = uint32_t(std::min<size_t>(e.params.size(), 32));
  Scene s{ W, H, std::vector<float>(size_t(W) * H, 0.0f), { 0, 0, int32_t(W), int32_t(H) }, { 1000.0 } };
  for (uint32_t y = 0; y < H; y++) s.lin[size_t(y) * W + W / 2] = s.lin[size_t(y) * W + W / 2 + 1] = 1.0f;
  Backend::EffectFrame fr = frameFor();
  auto unchanged = [&](const std::vector<float>& out) {
    float worst = 0;
    for (size_t i = 0; i < out.size(); i++) worst = std::max(worst, std::fabs(out[i] - s.lin[i]));
    return worst;
  };
  std::vector<float> out;
  // First run: no previous camera.
  fr.time = 1.0f;
  if (!runScene(be, id, s, fr, params, n, out)) { fail("motion blur: run failed"); return; }
  const float first = unchanged(out);
  // Moved 125 units along the camera's right (cameraRows column 2 = world z): the column moves 125 / 1000 * 128 px.
  fr.cameraRows[2][3] += 125.0f;
  fr.time = 1.016f;
  if (!runScene(be, id, s, fr, params, n, out)) { fail("motion blur: run failed"); return; }
  // Every row: its energy (the motion is horizontal: no light moves between rows), the light beyond the streak.
  double energyMin = 1e9, energyMax = 0, beyond = 0, rowDiff = 0;
  const size_t mid = size_t(H / 2) * W;
  for (uint32_t y = 0; y < H; y++) {
    double energy = 0;
    for (uint32_t x = 0; x < W; x++) {
      const float v = out[size_t(y) * W + x];
      energy += v;
      if (std::fabs(x + 0.5 - (W / 2 + 1.0)) > 10.0) beyond = std::max(beyond, double(v));
      rowDiff = std::max(rowDiff, double(std::fabs(v - out[mid + x])));
    }
    energyMin = std::min(energyMin, energy);
    energyMax = std::max(energyMax, energy);
  }
  const float left = out[mid + W / 2 - 5], right = out[mid + W / 2 + 6];
  printf("motion blur: first run worst change %.5f | moved: row energy %.3f .. %.3f (2), beyond 10 px %.5f, at -5 / +5 px %.3f %.3f, "
         "rows differ by up to %.4f (the samples' jitter)\n", first, energyMin, energyMax, beyond, left, right, rowDiff);
  if (first > 4e-3f) fail("motion blur: changed the image without a previous camera (%.4f)", first);
  if (std::fabs(energyMin - 2.0) > 0.1 || std::fabs(energyMax - 2.0) > 0.1) fail("motion blur: row energy %.3f .. %.3f, expected 2", energyMin, energyMax);
  if (beyond > 2e-3) fail("motion blur: light %.4f beyond the streak", beyond);
  if (!(left > 0.06f && right > 0.06f)) fail("motion blur: the streak does not reach both ways (%.3f, %.3f)", left, right);
  // The per-pixel jitter of the sample positions (interleaved gradient noise) varies a pixel by a fraction of the streak's
  // value (2 / 16 here); temporal upscaling averages it.
  if (rowDiff > 0.04) fail("motion blur: rows differ by %.4f, more than the jitter explains", rowDiff);
  // Still, then a cut.
  fr.time = 1.032f;
  if (!runScene(be, id, s, fr, params, n, out)) { fail("motion blur: run failed"); return; }
  const float still = unchanged(out);
  fr.cameraRows[2][3] += 1000.0f;
  fr.time = 1.048f;
  if (!runScene(be, id, s, fr, params, n, out)) { fail("motion blur: run failed"); return; }
  const float cut = unchanged(out);
  printf("motion blur: still camera worst change %.5f, camera cut %.5f\n", still, cut);
  if (still > 4e-3f) fail("motion blur: changed the image with a still camera (%.4f)", still);
  if (cut > 4e-3f) fail("motion blur: blurred across a camera cut (%.4f)", cut);
  be.destroyEffect(id);
}

static void run(Backend& be) {
  std::vector<fx::Effect> all = fx::loadEffects("effects", "build/fx_cache_test", [](const char* s) { printf("[fx] %s\n", s); });
  // Both off until enabled in the menu (effect.ini Enable=0); effects without the key stay on.
  for (const fx::Effect& e : all)
    if (e.enabled != (e.id != "dof" && e.id != "motionblur")) fail("%s: enabled by default is %d", e.id.c_str(), int(e.enabled));
  if (const fx::Effect* e = find(all, "dof")) dof(be, *e);
  if (const fx::Effect* e = find(all, "motionblur")) motionBlur(be, *e);
}

}  // namespace dofmbtest
