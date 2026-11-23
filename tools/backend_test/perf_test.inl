// Work the effect runner avoids (included by backend_test.cpp after sss_test.inl), with effects written here into
// build/fx_perftest:
// - scene copy reuse (EffectFrame::sceneUnchanged): the same sequence of runs with and without the flag gives
//   byte-identical images (the scene and an image copied from it), on a single-sample and a 4x multisampled target;
//   with the flag a copy is reused exactly while no pass wrote the scene since it was made (2 of 5 runs), and the
//   final scene is the two scene passes applied in turn (a stale copy would apply one);
// - Skip=pbr: the pass does not run while the lighting applies the AO (the scene as it was), and runs otherwise;
// - an effect reading the skin buffer, run with the buffer activated but no skin drawn: skipped, the scene as it was;
// - Skip= in effect.ini: tokens parsed, unknown ones and unknown parameters reported as load errors.
namespace perftest {

using sunshaftstest::halfToFloat;

static bool writeFile(const std::string& path, const std::string& text) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) return false;
  fwrite(text.data(), 1, text.size(), f);
  fclose(f);
  return true;
}
static bool writeEffect(const std::string& root, const std::string& id, const std::string& ini,
                        const std::vector<std::pair<std::string, std::string>>& shaders) {
  const std::string dir = root + "/" + id;
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  bool ok = writeFile(dir + "/effect.ini", ini);
  for (const auto& [name, src] : shaders) ok = ok && writeFile(dir + "/" + name, src);
  return ok;
}

static void run(Backend& be) {
  const std::string root = "build/fx_perftest";
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  bool written =
      writeEffect(root, "readimg",
                  "[Effect]\nPoint=world\nOrder=1\nSceneColor=1\n[Image.copy]\nFormat=rgba16f\n[Pass.0]\nShader=a.frag\nOutput=copy\n",
                  { { "a.frag", "layout(location = 0) out vec4 c;\nvoid main() { c = fnvSceneColor(ivec2(gl_FragCoord.xy)); }\n" } }) &&
      writeEffect(root, "half",
                  "[Effect]\nPoint=world\nOrder=2\nSceneColor=1\n[Pass.0]\nShader=a.frag\nOutput=scene\n",
                  { { "a.frag", "layout(location = 0) out vec4 c;\n"
                                "void main() { c = fnvSceneColor(ivec2(gl_FragCoord.xy)) * 0.5 + vec4(0.25, 0.125, 0.0625, 0.0); }\n" } }) &&
      writeEffect(root, "pbrskip",
                  "[Effect]\nPoint=world\nOrder=3\n[Pass.0]\nShader=a.frag\nOutput=scene\nBlend=add\nSkip=pbr\n",
                  { { "a.frag", "layout(location = 0) out vec4 c;\nvoid main() { c = vec4(0.5, 0.0, 0.0, 0.0); }\n" } }) &&
      writeEffect(root, "skipok",
                  "[Effect]\nPoint=world\nSkip= interior, Underwater ,abovewater,nocascades,strength=0\n[Param.strength]\nDefault=1\n"
                  "[Pass.0]\nShader=a.frag\n",
                  { { "a.frag", "layout(location = 0) out vec4 c;\nvoid main() { c = vec4(0.0); }\n" } }) &&
      writeEffect(root, "skipbad",
                  "[Effect]\nPoint=world\nSkip=interior,sometimes\n[Pass.0]\nShader=a.frag\n",
                  { { "a.frag", "layout(location = 0) out vec4 c;\nvoid main() { c = vec4(0.0); }\n" } }) &&
      writeEffect(root, "skipparam",
                  "[Effect]\nPoint=world\nSkip=nothere=0\n[Pass.0]\nShader=a.frag\n",
                  { { "a.frag", "layout(location = 0) out vec4 c;\nvoid main() { c = vec4(0.0); }\n" } }) &&
      writeEffect(root, "skippass",
                  "[Effect]\nPoint=world\n[Pass.0]\nShader=a.frag\nSkip=never\n",
                  { { "a.frag", "layout(location = 0) out vec4 c;\nvoid main() { c = vec4(0.0); }\n" } });
  if (!written) { fail("perf: could not write the test effects"); return; }
  const std::vector<fx::Effect> all = fx::loadEffects(root, "build/fx_cache_test", [](const char*) {});
  const fx::Effect* readimg = atmostest::findEffect(all, "readimg");
  const fx::Effect* half = atmostest::findEffect(all, "half");
  const fx::Effect* pbrskip = atmostest::findEffect(all, "pbrskip");
  const fx::Effect* skipok = atmostest::findEffect(all, "skipok");
  if (!readimg || !half || !pbrskip || !skipok) { fail("perf: test effects did not load"); return; }

  // Skip= parsing.
  {
    const uint32_t want = fx::Effect::SkipInterior | fx::Effect::SkipUnderwater | fx::Effect::SkipAboveWater |
                          fx::Effect::SkipNoCascades;
    if (skipok->skip != want || skipok->skipZero.size() != 1 || skipok->skipZero[0] != 0)
      fail("perf: Skip= parsed as %u with %zu parameters (expected %u, 1)", skipok->skip, skipok->skipZero.size(), want);
    if (!pbrskip->passes[0].skipPbr) fail("perf: [Pass.0] Skip=pbr not parsed");
    int errors = 0;
    for (const fx::Effect& e : all)
      if ((e.id == "skipbad" || e.id == "skipparam" || e.id == "skippass") && !e.error.empty()) errors++;
    if (errors != 3) fail("perf: %d of 3 bad Skip= values reported as errors", errors);
    printf("perf: Skip= parsed (flags %u, parameter %d); bad values reported: %d of 3\n", skipok->skip, skipok->skipZero[0], errors);
  }

  const uint32_t idRead = atmostest::createEffect(be, *readimg), idHalf = atmostest::createEffect(be, *half),
                 idPbr = atmostest::createEffect(be, *pbrskip);
  if (!idRead || !idHalf || !idPbr) { fail("perf: test effects not created"); return; }
  constexpr uint32_t W = 64, H = 32;
  Backend::EffectFrame base{};
  const float frustum[4] = { -1.0f, 1.0f, 1.0f, -1.0f }, port[4] = { 0.0f, 1.0f, 1.0f, 0.0f };
  memcpy(base.frustum, frustum, 16);
  memcpy(base.port, port, 16);
  base.nearZ = 1.0f;
  base.farZ = 1000.0f;
  float params[32][4] = {};
  auto at = [&](const std::vector<uint8_t>& img, uint32_t x, uint32_t y, int c) {
    return halfToFloat(reinterpret_cast<const uint16_t*>(img.data())[(size_t(y) * W + x) * 4 + c]);
  };

  // --- scene copy reuse ---------------------------------------------------------------------------------------------
  for (uint32_t samples : { 1u, 4u }) {
    std::vector<uint8_t> scene[2], copy[2];
    uint64_t copies[2] = {}, reused[2] = {};
    for (int flagged = 0; flagged < 2; flagged++) {
      const RtKey rt = newKey();
      be.attach(rt, { W, H, kD3DFMT_A16B16G16R16F, samples, false });
      be.beginGroup(&rt, 1, 0);
      const int32_t left[4] = { 0, 0, int32_t(W / 2), int32_t(H) }, right[4] = { int32_t(W / 2), 0, int32_t(W), int32_t(H) };
      be.clear(kClearTarget, 0xFF204080u, 1.0f, 0, left);
      be.clear(kClearTarget, 0xFFC0A060u, 1.0f, 0, right);
      const BackendStats s0 = be.stats();
      // readimg, readimg, half, half, readimg: copies at runs 1, 4, 5 with the flag (2 reuse 1's; 3 reuses it too, as
      // no pass wrote the scene; 3's scene pass makes 4 copy again, and 4's makes 5 copy again).
      const uint32_t seq[5] = { idRead, idRead, idHalf, idHalf, idRead };
      for (int i = 0; i < 5; i++) {
        Backend::EffectFrame f = base;
        f.sceneUnchanged = flagged && i > 0;
        if (!be.runEffect(seq[i], f, &params[0][0], 0)) fail("perf: run %d failed", i);
      }
      const BackendStats s1 = be.stats();
      copies[flagged] = s1.fxSceneCopies - s0.fxSceneCopies;
      reused[flagged] = s1.fxSceneCopiesReused - s0.fxSceneCopiesReused;
      be.endGroup();
      uint32_t vf = 0, ts = 0, w = 0, h = 0;
      if (!be.readback(rt, scene[flagged], vf, ts) || scene[flagged].size() != size_t(W) * H * 8) fail("perf: scene readback");
      if (!be.readbackEffectImage(idRead, 0, copy[flagged], w, h) || w != W || h != H) fail("perf: image readback");
      be.destroy(rt);
    }
    if (scene[0].size() != size_t(W) * H * 8 || copy[0].size() != size_t(W) * H * 8) return;
    const bool sameScene = scene[0] == scene[1], sameCopy = copy[0] == copy[1];
    // The final scene: both scene passes applied in turn to the cleared colour (ARGB 0xFF204080 on the left, 0xFFC0A060
    // on the right; the image is RGBA).
    auto twice = [](double v, double k) { return (v * 0.5 + k) * 0.5 + k; };
    const double lr = twice(0x20 / 255.0, 0.25), rr = twice(0xC0 / 255.0, 0.25), lb = twice(0x80 / 255.0, 0.0625);
    const double errL = std::abs(at(scene[1], 5, 5, 0) - lr), errR = std::abs(at(scene[1], 50, 20, 0) - rr),
                 errB = std::abs(at(scene[1], 5, 5, 2) - lb);
    const bool applied = errL < 2e-3 && errR < 2e-3 && errB < 2e-3;
    printf("perf: scene copies %ux: without reuse %llu copies, with it %llu copies + %llu reused; images identical: scene %s, "
           "copy %s; both passes applied: %s (red %.4f / %.4f)\n",
           samples, (unsigned long long)copies[0], (unsigned long long)copies[1], (unsigned long long)reused[1],
           sameScene ? "yes" : "NO", sameCopy ? "yes" : "NO", applied ? "yes" : "NO", at(scene[1], 5, 5, 0), lr);
    if (copies[0] != 5 || reused[0] != 0 || copies[1] != 3 || reused[1] != 2)
      fail("perf: scene copies %ux: %llu / %llu without the flag (expected 5 / 0), %llu / %llu with it (expected 3 / 2)",
           samples, (unsigned long long)copies[0], (unsigned long long)reused[0], (unsigned long long)copies[1],
           (unsigned long long)reused[1]);
    if (!sameScene || !sameCopy || !applied) fail("perf: scene copy reuse changed the image (%ux)", samples);
  }

  // --- Skip=pbr ---------------------------------------------------------------------------------------------------
  {
    std::vector<uint8_t> img[3];
    uint64_t skippedPasses[3] = {};
    for (int mode = 0; mode < 3; mode++) {  // 0: not run; 1: run with the AO applied (skipped); 2: run without
      const RtKey rt = newKey();
      be.attach(rt, { W, H, kD3DFMT_A16B16G16R16F, 4, false });
      be.beginGroup(&rt, 1, 0);
      const int32_t full[4] = { 0, 0, int32_t(W), int32_t(H) };
      be.clear(kClearTarget, 0xFF204080u, 1.0f, 0, full);
      const uint64_t p0 = be.stats().fxPassesSkipped;
      if (mode) {
        Backend::EffectFrame f = base;
        f.aoApplied = mode == 1;
        if (!be.runEffect(idPbr, f, &params[0][0], 0)) fail("perf: Skip=pbr run failed");
      }
      skippedPasses[mode] = be.stats().fxPassesSkipped - p0;
      be.endGroup();
      uint32_t vf = 0, ts = 0;
      if (!be.readback(rt, img[mode], vf, ts) || img[mode].size() != size_t(W) * H * 8) fail("perf: Skip=pbr readback");
      be.destroy(rt);
    }
    if (img[0].size() != size_t(W) * H * 8 || img[2].size() != img[0].size()) return;
    const bool unchanged = img[1] == img[0];
    const double added = at(img[2], 10, 10, 0) - at(img[0], 10, 10, 0);
    printf("perf: Skip=pbr: with the AO applied the scene %s (%llu pass skipped); without it the pass adds %.4f (expected 0.5)\n",
           unchanged ? "unchanged" : "CHANGED", (unsigned long long)skippedPasses[1], added);
    if (!unchanged || skippedPasses[1] != 1 || skippedPasses[2] != 0 || std::abs(added - 0.5) > 1e-3)
      fail("perf: Skip=pbr wrong");
  }

  // --- an effect reading the skin buffer, with no skin drawn ---------------------------------------------------------
  {
    std::vector<fx::Effect> sss = fx::loadEffects("effects", "build/fx_cache_test", [](const char*) {});
    const fx::Effect* e = atmostest::findEffect(sss, "sss");
    const uint32_t id = e ? atmostest::createEffect(be, *e) : 0;
    if (!id) { fail("perf: effects/sss not created"); return; }
    float p[32][4] = {};
    for (size_t i = 0; i < e->params.size() && i < 32; i++) p[i][0] = e->params[i].def;
    std::vector<uint8_t> before, after;
    const RtKey rt = newKey(), depth = newKey();
    be.attach(rt, { W, H, kD3DFMT_A16B16G16R16F, 1, false });
    be.attach(depth, { W, H, kD3DFMT_D24S8, 1, true });
    be.beginGroup(&rt, 1, depth);
    const int32_t full[4] = { 0, 0, int32_t(W), int32_t(H) };
    be.clear(kClearTarget | kClearZBuffer, 0xFF204080u, 0.5f, 0, full);
    be.setSkinBuffer(true);   // activated, as the plugin does for the world's opaque half ...
    be.setSkinBuffer(false);  // ... in which nothing drew skin
    const uint64_t n0 = be.stats().fxNoSkinRuns;
    Backend::EffectFrame f = base;
    f.aoApplied = true;
    if (!be.runEffect(id, f, &p[0][0], uint32_t(e->params.size()))) fail("perf: sss run failed");
    const uint64_t n1 = be.stats().fxNoSkinRuns;
    be.endGroup();
    uint32_t vf = 0, ts = 0;
    if (!be.readback(rt, after, vf, ts)) fail("perf: sss readback");
    // The same target cleared the same way, with no effect run.
    be.beginGroup(&rt, 1, depth);
    be.clear(kClearTarget | kClearZBuffer, 0xFF204080u, 0.5f, 0, full);
    be.endGroup();
    if (!be.readback(rt, before, vf, ts)) fail("perf: sss readback");
    be.destroy(rt);
    be.destroy(depth);
    be.destroyEffect(id);
    printf("perf: skin buffer active, no skin drawn: subsurface scattering skipped %llu time(s), scene %s\n",
           (unsigned long long)(n1 - n0), before == after ? "unchanged" : "CHANGED");
    if (n1 - n0 != 1 || before != after) fail("perf: the skin effect without skin");
  }
  be.destroyEffect(idRead);
  be.destroyEffect(idHalf);
  be.destroyEffect(idPbr);

  // --- how often the far cascade can be kept (cascades::covers, as sunshadows.cpp asks it) --------------------------
  // Random cameras (16:9, 75 degrees across, pitch -30..10 degrees) and suns (10..80 degrees up), 3 cascades of 2048
  // over 8000 units: the cascade drawn for one camera against the next frame's view after a move.
  {
    using namespace fnvvk::cascades;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double deg = 3.14159265358979 / 180.0;
    auto camera = [&](D3 pos, double yaw, double pitch) {
      Camera c;
      c.position = pos;
      c.forward = { std::cos(pitch) * std::cos(yaw), std::cos(pitch) * std::sin(yaw), std::sin(pitch) };
      c.right = normalize(cross(c.forward, { 0, 0, 1 }));
      c.up = cross(c.right, c.forward);
      const float tx = float(std::tan(37.5 * deg)), ty = tx * 9.0f / 16.0f;
      const float fr[4] = { -tx, tx, ty, -ty };
      memcpy(c.frustum, fr, sizeof(fr));
      c.nearZ = 5.0;
      c.farZ = 300000.0;
      return c;
    };
    struct Move { const char* name; double forward, side, turn; } moves[] = {
      { "still", 0, 0, 0 }, { "walking (4 units a frame)", 4, 0, 0 }, { "running sideways (10)", 0, 10, 0 },
      { "turning 0.5 degrees a frame", 0, 0, 0.5 }, { "turning 2", 0, 0, 2 }, { "turning 5", 0, 0, 5 } };
    constexpr int kTrials = 2000;
    int kept[6] = {};
    for (int t = 0; t < kTrials; t++) {
      const D3 pos{ u(rng) * 100000.0, u(rng) * 100000.0, 100.0 + u(rng) * 2000.0 };
      const double yaw = u(rng) * 360.0 * deg, pitch = (-30.0 + u(rng) * 40.0) * deg;
      const double sunYaw = u(rng) * 360.0 * deg, sunUp = (10.0 + u(rng) * 70.0) * deg;
      const D3 toSun{ std::cos(sunUp) * std::cos(sunYaw), std::cos(sunUp) * std::sin(sunYaw), std::sin(sunUp) };
      const Camera a = camera(pos, yaw, pitch);
      const std::vector<Cascade> ca = build(a, toSun, 3, 2048, 8000.0, 8000.0);
      for (int m = 0; m < 6; m++) {
        const D3 p2 = pos + a.forward * moves[m].forward + a.right * moves[m].side;
        const Camera b = camera(p2, yaw + moves[m].turn * deg, pitch);
        const std::vector<Cascade> cb = build(b, toSun, 3, 2048, 8000.0, 8000.0);
        if (covers(ca[2], b, cb[1].split, cb[2].split, 2048, 2.0)) kept[m]++;
      }
    }
    printf("perf: far cascade kept the next frame (of %d random views):", kTrials);
    for (int m = 0; m < 6; m++) printf(" %s %.0f%%%s", moves[m].name, 100.0 * kept[m] / kTrials, m < 5 ? "," : "\n");
    // (A few views are not covered even by their own cascade with the margin: the centre snapped to whole texels can put
    // a far corner of the slice within 2 texels of the box's side. Those frames draw the far cascade.)
    if (kept[0] < kTrials * 99 / 100 || kept[1] < kTrials * 95 / 100)
      fail("perf: the far cascade is kept too rarely (still %d, walking %d of %d)", kept[0], kept[1], kTrials);
  }

  // Queued texture uploads (record thread): a burst of large uploads in one frame, as loading a save makes on the game's
  // loader threads, keeps at most the budget (64 MiB) plus one texture of decoded data queued, and every texture arrives
  // (checked by reading back some of them: each texture's bytes are its index).
  {
    constexpr uint32_t kTex = 96, kDim = 1024;  // 4 MiB each, 384 MiB in all
    const BackendStats before = be.stats();
    for (uint32_t i = 0; i < kTex; i++) {
      d3dx::Texture t;
      t.format = d3dx::kFmtA8R8G8B8;
      t.width = t.height = kDim;
      t.levels = 1;
      t.cube = false;
      d3dx::Level l;
      l.width = l.height = kDim;
      l.rowPitch = kDim * 4;
      l.rows = kDim;
      l.data.assign(size_t(kDim) * kDim * 4, uint8_t(i));
      t.surfaces.push_back(std::move(l));
      if (!be.createTexture(TexKey(0x7E000000u + i), std::move(t))) fail("perf: texture burst: createTexture %u refused", i);
    }
    be.finishFrame();
    const BackendStats after = be.stats();
    const unsigned long long created = after.texturesCreated - before.texturesCreated;
    const unsigned long long failed = after.textureFailures - before.textureFailures;
    int wrongContent = 0;
    for (uint32_t i = 0; i < kTex; i += 19) {
      std::vector<uint8_t> px;
      if (!be.readbackTexture(TexKey(0x7E000000u + i), 0, 0, px) || px.size() < 4 || px[0] != uint8_t(i) ||
          px[px.size() - 1] != uint8_t(i))
        wrongContent++;
    }
    for (uint32_t i = 0; i < kTex; i++) be.destroyTexture(TexKey(0x7E000000u + i));
    const bool threaded = getenv("FNVVK_RECORD_THREAD") != nullptr;
    const double peakMiB = double(after.textureQueuePeakBytes) / (1 << 20);
    const unsigned long long waits = after.textureQueueWaits - before.textureQueueWaits;
    printf("perf: texture burst of %u x %u MiB: created %llu, failed %llu, content wrong %d; upload queue peak %.0f MiB, "
           "%llu waits\n",
           kTex, (kDim * kDim * 4) >> 20, created, failed, wrongContent, threaded ? peakMiB : 0.0, threaded ? waits : 0ull);
    if (created != kTex || failed || wrongContent)
      fail("perf: texture burst: %llu of %u created, %llu failed, %d wrong", created, kTex, failed, wrongContent);
    if (threaded && (after.textureQueuePeakBytes > (64ull << 20) + uint64_t(kDim) * kDim * 4 || !waits))
      fail("perf: texture burst: the upload queue is not bounded (peak %.0f MiB, %llu waits)", peakMiB, waits);
  }
}

}  // namespace perftest
