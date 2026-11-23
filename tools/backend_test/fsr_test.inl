// Upscaling (backend_ffx.inl; included by backend_test.cpp):
// - render sizes and the jitter sequence against FSR's own formulas (ffxFsr3UpscalerGetRenderResolutionFromQualityMode,
//   Halton 2 / 3);
// - with fnv_vk_fidelityfx.dll (32-bit harness only: the DLL is built for the game's process; FNV_FFX_DLL overrides its
//   path): a scaled scene end to end. The window-sized targets render at the render size (clear rectangles given in window
//   pixels), FSR upscales a uniform scene colour (it must stay that colour), a render-size target written in the scene is
//   filtered up, the first-person view (a depth clear and a nearer clear after captureWorldDepth) gets its own depth and
//   no motion, and the world's motion vectors match the closed form of a camera translation.
namespace fsrtest {

static bool closeTo(float a, float b, float tol) { return std::fabs(a - b) <= tol; }
static float halfToF(uint16_t h) {
  const uint32_t s = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF;
  uint32_t bits = e == 0 ? (m ? 0 : s) : e == 31 ? (s | 0x7F800000u | m << 13) : (s | (e + 112) << 23 | m << 13);
  if (e == 0 && m) {  // subnormal
    float f = std::ldexp(float(m), -24);
    return (h & 0x8000) ? -f : f;
  }
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

static void helpers() {
  struct { UpscaleSettings::QualityMode q; uint32_t w, h; } modes[] = {
    { UpscaleSettings::NativeAa, 1176, 664 },  { UpscaleSettings::Quality, 784, 442 },
    { UpscaleSettings::Balanced, 691, 390 },   { UpscaleSettings::Performance, 588, 332 },
    { UpscaleSettings::UltraPerformance, 392, 221 },
  };
  for (const auto& m : modes) {
    UpscaleSettings s;
    s.upscaler = UpscaleSettings::Fsr3;
    s.quality = m.q;
    uint32_t w = 0, h = 0;
    Backend::renderSize(s, 1176, 664, w, h);
    if (w != m.w || h != m.h) fail("fsr: render size of mode %u: %ux%u, expected %ux%u", unsigned(m.q), w, h, m.w, m.h);
  }
  UpscaleSettings off;
  uint32_t w = 0, h = 0;
  Backend::renderSize(off, 1176, 664, w, h);
  if (w != 1176 || h != 664) fail("fsr: render size without an upscaler %ux%u", w, h);
  // Halton(1, 2) = 1/2, Halton(1, 3) = 1/3, Halton(2, 2) = 1/4, Halton(2, 3) = 2/3, Halton(3, 2) = 3/4, Halton(3, 3) = 1/9.
  const float expect[3][2] = { { 0.0f, 1.0f / 3 - 0.5f }, { 0.25f - 0.5f, 2.0f / 3 - 0.5f }, { 0.75f - 0.5f, 1.0f / 9 - 0.5f } };
  for (uint32_t i = 0; i < 3; i++) {
    float x, y;
    Backend::jitterOffset(i, 588, 1176, x, y);
    if (!closeTo(x, expect[i][0], 1e-6f) || !closeTo(y, expect[i][1], 1e-6f))
      fail("fsr: jitter %u = (%f, %f), expected (%f, %f)", i, x, y, expect[i][0], expect[i][1]);
  }
  // Performance at 2x: 8 * 2^2 = 32 phases: index 32 repeats index 0.
  float x0, y0, x32, y32;
  Backend::jitterOffset(0, 588, 1176, x0, y0);
  Backend::jitterOffset(32, 588, 1176, x32, y32);
  if (x0 != x32 || y0 != y32) fail("fsr: the jitter sequence of 2x does not repeat after 32 phases");
  printf("fsr: render sizes and jitter sequence checked\n");
}

// Camera: looking along +x (direction), up +z, right -y (Gamebryo's rows: columns direction, up, right).
static SceneView camera(float ox, float oy, float oz, bool reset) {
  SceneView v;
  const float dir[3] = { 1, 0, 0 }, up[3] = { 0, 0, 1 }, right[3] = { 0, -1, 0 };
  const float o[3] = { ox, oy, oz };
  for (int r = 0; r < 3; r++) {
    v.cameraRows[r][0] = dir[r];
    v.cameraRows[r][1] = up[r];
    v.cameraRows[r][2] = right[r];
    v.cameraRows[r][3] = o[r];
  }
  v.frustum[0] = -0.6f;  // left, right, top, bottom tangents
  v.frustum[1] = 0.6f;
  v.frustum[2] = 0.35f;
  v.frustum[3] = -0.35f;
  v.nearZ = 10.0f;
  v.farZ = 10000.0f;
  v.port[0] = 0.0f;
  v.port[1] = 1.0f;
  v.port[2] = 1.0f;
  v.port[3] = 0.0f;
  v.frameTimeMs = 16.7f;
  v.reset = reset;
  return v;
}

struct Scene {
  RtKey color = 0, depth = 0, other = 0;
};

// One world frame: the world (colour 0x40 grey, depth 0.5), the first-person view (depth cleared, 0.2 in a rectangle),
// another window-sized target written green; upscaled with `view`.
static void frame(Backend& be, const Scene& s, const SceneView& view, uint32_t jitterIndex, uint32_t renderW) {
  be.beginScaledScene();
  const int32_t full[4] = { 0, 0, 1176, 664 };
  be.beginGroup(&s.color, 1, s.depth);
  be.clear(kClearTarget | kClearZBuffer | kClearStencil, 0xFF404040, 0.5f, 0, full);
  be.endGroup();
  be.captureWorldDepth();
  be.beginGroup(&s.color, 1, s.depth);
  be.clear(kClearZBuffer, 0, 1.0f, 0, full);
  const int32_t fp[4] = { 100, 100, 300, 300 };
  be.clear(kClearZBuffer, 0, 0.2f, 0, fp);
  be.endGroup();
  be.beginGroup(&s.other, 1, 0);
  be.clear(kClearTarget, 0xFF00FF00, 1.0f, 0, full);
  be.endGroup();
  SceneView v = view;
  Backend::jitterOffset(jitterIndex, renderW, 1176, v.jitter[0], v.jitter[1]);
  be.upscaleScene(s.color, s.depth, v);
}

static void upscaling(Backend& be, const std::string& dll) {
  UpscaleSettings us;
  us.upscaler = UpscaleSettings::Fsr3;
  us.quality = UpscaleSettings::Performance;
  us.sharpen = true;
  us.sharpness = 0.5f;
  us.library = dll;
  be.setUpscaling(us);
  UpscaleStatus st = be.upscaleStatus();
  if (!st.libraryLoaded) {
    fail("fsr: %s did not load (%s)", dll.c_str(), st.error.c_str());
    return;
  }
  Scene s{ newKey(), newKey(), newKey() };
  be.attach(s.color, { 1176, 664, kD3DFMT_A16B16G16R16F, 1, false });
  be.attach(s.depth, { 1176, 664, kD3DFMT_D24S8, 1, true });
  be.attach(s.other, { 1176, 664, kD3DFMT_A8R8G8B8, 1, false });
  const BackendStats before = be.stats();

  // Frame 1 (reset): the camera at the origin.
  frame(be, s, camera(0, 0, 0, true), 0, 588);
  be.finishFrame();
  st = be.upscaleStatus();
  if (!st.upscaling || st.renderW != 588 || st.renderH != 332 || st.displayW != 1176 || st.displayH != 664)
    fail("fsr: status after the first frame: upscaling %d, render %ux%u, display %ux%u (%s)", int(st.upscaling), st.renderW,
         st.renderH, st.displayW, st.displayH, st.error.c_str());
  std::vector<uint8_t> px;
  uint32_t vf = 0, ts = 0;
  // The scene colour, upscaled into the window-sized target: uniform 0x40 / 255.
  if (!be.readback(s.color, px, vf, ts) || ts != 8) {
    fail("fsr: no readback of the upscaled scene");
  } else {
    const float want = 64.0f / 255.0f;
    size_t bad = 0;
    float worst = 0.0f;
    for (size_t i = 0; i < px.size() / 8; i++)
      for (int c = 0; c < 3; c++) {
        uint16_t h;
        memcpy(&h, &px[i * 8 + c * 2], 2);
        const float d = std::fabs(halfToF(h) - want);
        worst = std::max(worst, d);
        if (d > 0.01f) bad++;
      }
    printf("fsr: upscaled uniform scene: worst difference %.5f\n", worst);
    if (bad) fail("fsr: %zu channels of the upscaled scene differ from the uniform input (worst %.4f)", bad, worst);
  }
  // The other window-sized target, written at the render size during the scene: filtered up to the full size.
  if (!be.readback(s.other, px, vf, ts)) {
    fail("fsr: no readback of the filtered-up target");
  } else {
    size_t bad = 0;
    for (size_t i = 0; i < px.size() / ts; i++) {
      uint32_t t;
      memcpy(&t, &px[i * ts], 4);
      if (ts == 4 && t != 0xFF00FF00u) bad++;
      if (ts == 8) {  // wide target: RGBA16F
        uint16_t c[4];
        memcpy(c, &px[i * 8], 8);
        if (!closeTo(halfToF(c[0]), 0.0f, 1e-3f) || !closeTo(halfToF(c[1]), 1.0f, 1e-3f) || !closeTo(halfToF(c[2]), 0.0f, 1e-3f)) bad++;
      }
    }
    if (bad) fail("fsr: %zu texels of the filtered-up target are not the green it was cleared to (all of it, in window pixels)", bad);
  }
  // FSR's depth: the first-person view's (0.2) in its rectangle (window 100..300 = render 50..150), the world's (0.5)
  // elsewhere; no motion on the reset frame.
  uint32_t w = 0, h = 0;
  std::vector<uint8_t> depth;
  if (!be.readbackUpscaleInput(0, depth, w, h) || w != 588 || h != 332) {
    fail("fsr: no readback of FSR's depth (%ux%u)", w, h);
  } else {
    size_t bad = 0;
    for (uint32_t y = 0; y < h; y++)
      for (uint32_t x = 0; x < w; x++) {
        float d;
        memcpy(&d, &depth[(size_t(y) * w + x) * 4], 4);
        const bool fp = x >= 50 && x < 150 && y >= 50 && y < 150;
        if (!closeTo(d, fp ? 0.2f : 0.5f, 1e-5f)) bad++;
      }
    if (bad) fail("fsr: %zu texels of FSR's depth are not the first-person (0.2) / world (0.5) composite", bad);
  }
  std::vector<uint8_t> mv;
  if (!be.readbackUpscaleInput(1, mv, w, h)) {
    fail("fsr: no readback of the motion vectors");
  } else {
    size_t moving = 0;
    for (size_t i = 0; i < mv.size() / 2; i++) {
      uint16_t c;
      memcpy(&c, &mv[i * 2], 2);
      if (halfToF(c) != 0.0f) moving++;
    }
    if (moving) fail("fsr: the reset frame has %zu non-zero motion components", moving);
  }

  // Frame 2: the camera moved 20 units right and 10 up. A world point at view distance z was 20 units further right
  // (and 10 higher) in the previous camera's view: u_prev - u = +20 / (z (r - l)), v_prev - v = -10 / (z (t - b)) (v down).
  frame(be, s, camera(0, -20, 10, false), 1, 588);
  be.finishFrame();
  if (!be.readbackUpscaleInput(1, mv, w, h) || !be.readbackUpscaleInput(0, depth, w, h)) {
    fail("fsr: no readback after the moving frame");
  } else {
    const SceneView c = camera(0, 0, 0, false);
    size_t bad = 0, fpMoving = 0;
    float worst = 0.0f;
    for (uint32_t y = 0; y < h; y++)
      for (uint32_t x = 0; x < w; x++) {
        const size_t i = size_t(y) * w + x;
        uint16_t m2[2];
        memcpy(m2, &mv[i * 4], 4);
        const float mu = halfToF(m2[0]), mvv = halfToF(m2[1]);
        const bool fp = x >= 50 && x < 150 && y >= 50 && y < 150;
        if (fp) {
          if (mu != 0.0f || mvv != 0.0f) fpMoving++;
          continue;
        }
        float d;
        memcpy(&d, &depth[i * 4], 4);
        const float z = c.nearZ * c.farZ / (c.farZ - d * (c.farZ - c.nearZ));
        const float eu = 20.0f / (z * (c.frustum[1] - c.frustum[0]));
        const float ev = -10.0f / (z * (c.frustum[2] - c.frustum[3]));
        const float err = std::max(std::fabs(mu - eu), std::fabs(mvv - ev));
        worst = std::max(worst, err);
        if (err > 1e-4f + 2e-3f * std::max(std::fabs(eu), std::fabs(ev))) bad++;
      }
    printf("fsr: camera motion vectors: worst difference %.6f (expected u %.6f, v %.6f)\n", worst,
           20.0f / ((10.0f * 10000.0f / (10000.0f - 0.5f * 9990.0f)) * 1.2f),
           -10.0f / ((10.0f * 10000.0f / (10000.0f - 0.5f * 9990.0f)) * 0.7f));
    if (bad) fail("fsr: %zu world motion vectors differ from the camera translation's", bad);
    if (fpMoving) fail("fsr: %zu first-person texels have motion", fpMoving);
  }
  const BackendStats after = be.stats();
  if (after.fsrUpscales - before.fsrUpscales != 2) fail("fsr: %llu upscales for 2 frames", after.fsrUpscales - before.fsrUpscales);
  if (after.fsrTwinBlits == before.fsrTwinBlits) fail("fsr: no render-size target was filtered up");

  // Every quality mode, a frame each (context sizes stay the window's; the render size changes per mode).
  for (uint32_t q = 0; q < 5; q++) {
    us.quality = UpscaleSettings::QualityMode(q);
    be.setUpscaling(us);
    uint32_t rw = 0, rh = 0;
    Backend::renderSize(us, 1176, 664, rw, rh);
    frame(be, s, camera(float(q), 0, 0, false), q, rw);
    be.finishFrame();
    st = be.upscaleStatus();
    if (!st.upscaling || st.renderW != rw || st.renderH != rh)
      fail("fsr: mode %u: upscaling %d at %ux%u, expected %ux%u (%s)", q, int(st.upscaling), st.renderW, st.renderH, rw, rh,
           st.error.c_str());
  }
  printf("fsr: every quality mode upscaled\n");
  // The menu switches the upscaler off (and on again) after the frame's upscale was recorded, before the frame is
  // submitted (overlay::frame runs ahead of present): the FSR context the recorded dispatch uses must outlive that frame.
  {
    const uint64_t errorsBefore = be.stats().vkErrors, validationBefore = be.stats().validationMessages;
    for (int round = 0; round < 3; round++) {
      us.upscaler = UpscaleSettings::Fsr3;
      be.setUpscaling(us);
      uint32_t rw = 0, rh = 0;
      Backend::renderSize(us, 1176, 664, rw, rh);
      frame(be, s, camera(0, 0, 0, false), uint32_t(round), rw);
      us.upscaler = UpscaleSettings::Off;
      be.setUpscaling(us);  // the dispatch of the frame being recorded uses the context switched off here
      be.finishFrame();
      frame(be, s, camera(0, 0, 0, false), 0, 1176);  // a native frame after it
      be.finishFrame();
    }
    const BackendStats t = be.stats();
    printf("fsr: upscaler switched off mid-frame 3 times: %llu Vulkan errors, %llu validation messages\n",
           t.vkErrors - errorsBefore, t.validationMessages - validationBefore);
    if (t.vkErrors != errorsBefore) fail("fsr: switching the upscaler off mid-frame caused %llu Vulkan errors", t.vkErrors - errorsBefore);
    if (t.validationMessages != validationBefore)
      fail("fsr: switching the upscaler off mid-frame caused %llu validation messages", t.validationMessages - validationBefore);
  }
  us.upscaler = UpscaleSettings::Off;
  be.setUpscaling(us);
  be.destroy(s.color);
  be.destroy(s.depth);
  be.destroy(s.other);
}

static void run(Backend& be) {
  helpers();
  if (sizeof(void*) != 4) {
    printf("fsr: 64-bit harness: fnv_vk_fidelityfx.dll is 32-bit only: upscaling skipped\n");
    return;
  }
  const char* env = getenv("FNV_FFX_DLL");
  const std::string dll = env ? env : "build\\plugin\\fnv_vk_fidelityfx.dll";
  upscaling(be, dll);
}

}  // namespace fsrtest
