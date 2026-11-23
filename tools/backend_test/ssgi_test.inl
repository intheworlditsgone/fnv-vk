// Screen-space indirect light (effects/ssgi: ssgi.frag, gitemporal.frag, gidepth.frag; docs/INDIRECT.md "SSGI") on a scene with a known
// answer, read back from the effect's published image: two planes facing the camera, the left half of the view at
// distance 500 (lit colour red 0.5), the right half at 800 (black); near 10, far 100000, frustum -1..1.
// - left-half pixels (the nearer plane): every ray goes toward the camera and nothing lies in front of it: no hits
//   (rgb 0, a 1);
// - right-half pixels far from the step: no hits either;
// - right-half pixels next to the step: rays toward the camera pass behind the nearer plane (within the thickness):
//   hits, which bring its red (linear 0.5^2.2 each), so r = lin(0.5) * (1 - a) exactly and g = b = 0;
// - with the lighting not using it (aoApplied false): nothing computed (rgb 0, a 1).
namespace ssgitest {

static void run(Backend& be) {
  std::vector<fx::Effect> all = fx::loadEffects("effects", "build/fx_cache_test", [](const char* s) { printf("[fx] %s\n", s); });
  const fx::Effect* e = nullptr;
  for (const fx::Effect& x : all)
    if (x.id == "ssgi") e = &x;
  if (!e || !e->error.empty()) { fail("ssgi: effects/ssgi did not load%s%s", e ? ": " : "", e ? e->error.c_str() : ""); return; }
  Backend::EffectDesc d;
  d.sceneColor = e->sceneColor;
  int giIndex = -1;
  for (size_t i = 0; i < e->images.size(); i++) {
    const fx::Image& im = e->images[i];
    if (im.publish == "gi") giIndex = int(i);
    d.images.push_back({ im.vkFormat, im.scale,
                         uint8_t(im.publish == "ao" ? Backend::EffectDesc::Ao
                                                    : im.publish == "gi" ? Backend::EffectDesc::Gi : im.publish == "gidepth" ? Backend::EffectDesc::GiDepth : Backend::EffectDesc::None) });
  }
  for (const fx::Pass& p : e->passes)
    d.passes.push_back({ { p.spv[0].data(), p.spv[1].data() }, { p.spv[0].size(), p.spv[1].size() }, p.output, uint32_t(p.blend) });
  const uint32_t id = be.createEffect(d);
  if (!id || giIndex < 0) { fail("ssgi: effect not created or no gi image"); return; }
  float params[32][4] = {};
  for (size_t i = 0; i < e->params.size() && i < 32; i++) params[i][0] = e->params[i].def;

  constexpr uint32_t W = 128, H = 64;
  const double zn = 10.0, zf = 100000.0, zNear = 500.0, zFar = 800.0;
  auto depthOf = [&](double z) { return float(zf * (z - zn) / (z * (zf - zn))); };
  Backend::EffectFrame fr{};
  const float frustum[4] = { -1.0f, 1.0f, 1.0f, -1.0f }, port[4] = { 0.0f, 1.0f, 1.0f, 0.0f };
  memcpy(fr.frustum, frustum, 16);
  memcpy(fr.port, port, 16);
  fr.nearZ = float(zn);
  fr.farZ = float(zf);
  for (int r = 0; r < 3; r++) fr.cameraRows[r][r] = 1.0f;

  auto runScene = [&](bool applied, std::vector<uint8_t>& gi, uint32_t& gw, uint32_t& gh, uint32_t leftColor = 0xFF800000u,
                      double zl = 500.0, double zr = 800.0) {
    const RtKey rt = newKey(), depth = newKey();
    be.attach(rt, { W, H, kD3DFMT_A16B16G16R16F, 1, false });
    be.attach(depth, { W, H, kD3DFMT_D24S8, 1, true });
    be.beginGroup(&rt, 1, depth);
    const int32_t left[4] = { 0, 0, int32_t(W / 2), int32_t(H) }, right[4] = { int32_t(W / 2), 0, int32_t(W), int32_t(H) };
    be.clear(kClearTarget | kClearZBuffer, leftColor, depthOf(zl), 0, left);   // default red 128 / 255
    be.clear(kClearTarget | kClearZBuffer, 0xFF000000u, depthOf(zr), 0, right);
    fr.aoApplied = applied;
    if (!be.runEffect(id, fr, &params[0][0], uint32_t(std::min<size_t>(e->params.size(), 32)))) fail("ssgi: run failed");
    be.endGroup();
    const bool ok = be.readbackEffectImage(id, uint32_t(giIndex), gi, gw, gh);
    be.destroy(rt);
    be.destroy(depth);
    return ok;
  };
  std::vector<uint8_t> gi, off;
  uint32_t gw = 0, gh = 0, ow = 0, oh = 0;
  if (!runScene(true, gi, gw, gh) || !runScene(false, off, ow, oh) || gi.size() != size_t(gw) * gh * 8) {
    fail("ssgi: readback failed");
    be.destroyEffect(id);
    return;
  }
  auto at = [&](const std::vector<uint8_t>& img, uint32_t x, uint32_t y, int c) {
    return sunshaftstest::halfToFloat(reinterpret_cast<const uint16_t*>(img.data())[(size_t(y) * gw + x) * 4 + c]);
  };
  const float red = std::pow(128.0f / 255.0f, 2.2f);
  // Columns of the half-resolution image: the step is at gw / 2.
  const uint32_t y = gh / 2, xLeft = gw / 4, xFarRight = gw - 2, xStep = gw / 2 + 1;
  const float leftA = at(gi, xLeft, y, 3), leftR = at(gi, xLeft, y, 0);
  const float farA = at(gi, xFarRight, y, 3), farR = at(gi, xFarRight, y, 0);
  const float stepR = at(gi, xStep, y, 0), stepG = at(gi, xStep, y, 1), stepA = at(gi, xStep, y, 3);
  float maxRelation = 0.0f;  // |r - lin(0.5) (1 - a)| over the right half
  for (uint32_t x = gw / 2; x < gw; x++)
    for (uint32_t yy = 1; yy + 1 < gh; yy++)
      maxRelation = std::max(maxRelation, std::abs(at(gi, x, yy, 0) - red * (1.0f - at(gi, x, yy, 3))));
  float offMax = 0.0f;
  for (uint32_t x = 0; x < ow && off.size() == size_t(ow) * oh * 8; x++)
    offMax = std::max(offMax, std::abs(at(off, x, oh / 2, 0)) + std::abs(1.0f - at(off, x, oh / 2, 3)));
  printf("ssgi: left (nearer plane) r %.4f miss %.3f | right, far from the step r %.4f miss %.3f | right, at the step r %.4f "
         "g %.4f miss %.3f | max |r - lin(red)(1 - miss)| %.4f | unused: max deviation %.4f\n",
         leftR, leftA, farR, farA, stepR, stepG, stepA, maxRelation, offMax);
  if (leftR > 1e-3f || leftA < 0.999f) fail("ssgi: the nearer plane found light where none can come from");
  if (farR > 1e-3f || farA < 0.999f) fail("ssgi: the farther plane away from the step found light");
  if (!(stepA < 0.95f) || stepR < 0.01f || stepG > 1e-3f) fail("ssgi: next to the step the farther plane does not see the nearer plane's red");
  if (maxRelation > 2e-3f) fail("ssgi: the light found is not the hit surface's linear colour times the hit fraction");
  if (offMax > 1e-4f) fail("ssgi: computed although the lighting does not use it");

  // Temporal accumulation (gitemporal.frag, response alpha = the parameter's default): with the same camera and depths
  // the new result is blended into the previous one, out = (1 - alpha) previous + alpha current. Frame 1: the scene as
  // above (red); frame 2: the nearer plane green: at the step, red persists as exactly (1 - alpha) of frame 1's, and the
  // green is alpha times green times the current hit fraction (recovered from the blended miss fraction). Frame 3: the
  // planes moved (500 -> 600, 800 -> 900): the previous result no longer applies (depth mismatch), so no red remains.
  float response = 0.15f;
  for (const fx::Param& p : e->params)
    if (p.name == "giResponse") response = p.def;
  std::vector<uint8_t> f1, f2, f3;
  uint32_t w1, h1, w2, h2, w3, h3;
  const bool ran = runScene(true, f1, w1, h1) && runScene(true, f2, w2, h2, 0xFF008000u) &&
                   runScene(true, f3, w3, h3, 0xFF008000u, 600.0, 900.0);
  if (!ran || f2.size() != gi.size() || f3.size() != gi.size()) {
    fail("ssgi: temporal readbacks failed");
  } else {
    const float green = std::pow(128.0f / 255.0f, 2.2f);
    float maxRed = 0.0f, maxGreen = 0.0f, red3 = 0.0f;
    uint32_t checked = 0;
    for (uint32_t x = gw / 2; x < gw; x++)
      for (uint32_t yy = 1; yy + 1 < gh; yy++) {
        const float hr = at(f1, x, yy, 0), ha = at(f1, x, yy, 3);
        const float or2 = at(f2, x, yy, 0), og = at(f2, x, yy, 1), oa = at(f2, x, yy, 3);
        const float ca = (oa - (1.0f - response) * ha) / response;  // the current miss fraction
        maxRed = std::max(maxRed, std::abs(or2 - (1.0f - response) * hr));
        maxGreen = std::max(maxGreen, std::abs(og - response * green * (1.0f - ca)));
        red3 = std::max(red3, at(f3, x, yy, 0));
        checked += hr > 0.0f;
      }
    printf("ssgi temporal (response %.2f): %u pixels with light; frame 2: max |red - (1 - a) red1| %.5f, max |green - a "
           "green (1 - miss)| %.5f; frame 3 (moved planes): max red %.5f\n", response, checked, maxRed, maxGreen, red3);
    if (!checked) fail("ssgi temporal: no light found to accumulate");
    if (maxRed > 2e-3f) fail("ssgi temporal: the previous result is not kept with weight 1 - response");
    if (maxGreen > 2e-3f) fail("ssgi temporal: the new result does not enter with weight response");
    if (red3 > 1e-3f) fail("ssgi temporal: a previous result that no longer applies was kept");
  }
  be.destroyEffect(id);
}

}  // namespace ssgitest
