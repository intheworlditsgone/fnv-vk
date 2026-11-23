// Subsurface scattering (effects/sss, docs/SKIN.md; included by backend_test.cpp after skin_test.inl): the game's
// SKIN2000.vso with replacements/lighting's SKIN2000.pso (physically based) draws skin into a depth-tested HDR target
// while the skin buffer is active, then the effect runs (as at the "opaque" point). The skin's normal map faces the sun
// on its left half and turns 90 degrees from it on its right half (lit by the ambient only): a step in its light.
// - the skin buffer not active: the effect leaves every pixel as it was;
// - active: uniform skin far from the step keeps its colour (the kernel sums to 1); at the step the lit side darkens
//   and the dark side brightens, red the most (the human profile's widest channel);
// - the right half 80 units behind the left: nothing crosses the depth step (at the two pixels beside it the change is
//   under a tenth of the flat scene's, beyond the change 8 pixels away on the same side);
// - an object (SLS2000) drawn over the right half after the skin: it is left as it was, and so is the skin beside it.
namespace ssstest {

static void run(Backend& be) {
  const char* dir = getenv("FNV_GAME_SHADERS");
  if (!dir || repltest::g_lightingPs.empty()) { printf("sss: skipped\n"); return; }
  std::vector<fx::Effect> all = fx::loadEffects("effects", "build/fx_cache_test", [](const char*) {});
  const fx::Effect* e = atmostest::findEffect(all, "sss");
  if (!e) { fail("sss: effects/sss did not load"); return; }
  const uint32_t id = atmostest::createEffect(be, *e);
  if (!id) { fail("sss: effect not created"); return; }
  const ShaderKey vsKey = 0xC8000, psKey = 0xC8040, objVs = 0xC8080, objPs = 0xC80C0;
  bool ok = false, ok2 = false, ok3 = false, ok4 = false;
  for (const Md5& m : sdpShaderMd5s(dir, "SKIN2000.vso", { 13 })) ok = ok || be.createShader(vsKey, m.b.data());
  for (const Md5& m : sdpShaderMd5s(dir, "SKIN2000.pso", { 13 })) ok2 = ok2 || be.createShader(psKey, m.b.data());
  for (const Md5& m : sdpShaderMd5s(dir, "SLS2000.vso", { 13 })) ok3 = ok3 || be.createShader(objVs, m.b.data());
  for (const Md5& m : sdpShaderMd5s(dir, "SLS2000.pso", { 13 })) ok4 = ok4 || be.createShader(objPs, m.b.data());
  if (!ok || !ok2 || !ok3 || !ok4) { fail("sss: SKIN2000 / SLS2000 shaders not in the shader pack"); return; }
  D3DRef fresh;
  if (!fresh.init()) { fail("sss: no D3D9 device"); return; }
  static uint32_t rs[256], ss[16][14];
  for (uint32_t i = 0; i < 256; i++) { DWORD v = 0; fresh.dev->GetRenderState(D3DRENDERSTATETYPE(i), &v); rs[i] = v; }
  for (uint32_t s = 0; s < 16; s++)
    for (uint32_t t = 0; t < 14; t++) { DWORD v = 0; fresh.dev->GetSamplerState(s, D3DSAMPLERSTATETYPE(t), &v); ss[s][t] = v; }
  rs[D3DRS_ZENABLE] = 1;
  rs[D3DRS_CULLMODE] = D3DCULL_NONE;
  for (uint32_t s = 0; s < 16; s++) ss[s][D3DSAMP_MAGFILTER] = ss[s][D3DSAMP_MINFILTER] = D3DTEXF_POINT;
  auto texture = [&](TexKey key, const std::vector<uint8_t>& bgra, uint32_t w, uint32_t h) {
    d3dx::Texture t;
    t.format = 21;
    t.width = w;
    t.height = h;
    t.levels = 1;
    t.cube = false;
    d3dx::Level l;
    l.width = w;
    l.height = h;
    l.rowPitch = w * 4;
    l.rows = h;
    l.data = bgra;
    t.surfaces.push_back(std::move(l));
    return be.createTexture(key, t);
  };
  // Base grey 160; the normal map: left texel up (+z: facing the sun), right texel along +x; FaceGenMap0 128 (no
  // detail), FaceGenMap1 64 (x 1).
  if (!texture(0xC8100, { 160, 160, 160, 255 }, 1, 1) || !texture(0xC8140, { 255, 128, 128, 255, 128, 128, 255, 255 }, 2, 1) ||
      !texture(0xC8180, { 128, 128, 128, 255 }, 1, 1) || !texture(0xC81C0, { 64, 64, 64, 255 }, 1, 1)) {
    fail("sss: textures");
    return;
  }
  struct V { float p[3], t[3], b[3], n[3], uv[2], c[4]; };
  auto quad = [&](BufKey key, float x0, float x1, float z, float u0, float u1) {
    auto c = [&](float x, float y, float u) { return V{ { x, y, z }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { u, 0.5f }, { 1, 1, 1, 1 } }; };
    const V q[6] = { c(x0, -1, u0), c(x1, -1, u1), c(x1, 1, u1), c(x0, -1, u0), c(x1, 1, u1), c(x0, 1, u0) };
    be.createBuffer(key, sizeof(q), false);
    be.writeBuffer(key, 0, q, sizeof(q), 0);
  };
  const BufKey full = 0xC8200, leftNear = 0xC8240, rightFar = 0xC8280, rightCover = 0xC82C0;
  quad(full, -1, 1, 0.5f, 0.0f, 1.0f);
  quad(leftNear, -1, 0, 0.5f, 0.0f, 0.5f);
  quad(rightFar, 0, 1, 0.9f, 0.5f, 1.0f);
  quad(rightCover, 0, 1, 0.3f, 0.5f, 1.0f);
  const VertexElement ve[6] = { { 0, 0, 2, 0, 0, 0 }, { 0, 12, 2, 0, 6, 0 }, { 0, 24, 2, 0, 7, 0 },
                                { 0, 36, 2, 0, 3, 0 }, { 0, 48, 1, 0, 5, 0 }, { 0, 56, 3, 0, 10, 0 } };
  const DeclKey decl = 0xC8300;
  be.createDecl(decl, ve, 6);
  static float vsc[256][4], psc[224][4];
  memset(vsc, 0, sizeof(vsc));
  memset(psc, 0, sizeof(psc));
  for (int i = 0; i < 4; i++) vsc[i][i] = 1.0f;
  const float fog[4] = { 1.0e6f, 1.0f, 1.0f, 0.0f }, light[4] = { 0, 0, 1, 1 }, eye[4] = { 0.0f, 0.0f, 1000.0f, 1.0f };  // far: the same view at every pixel (uniform halves)
  memcpy(vsc[14], fog, 16);
  memcpy(vsc[16], eye, 16);
  memcpy(vsc[25], light, 16);
  for (int r = 0; r < 3; r++) vsc[250 + r][r] = 1.0f;
  const float ambient[4] = { 0.15f, 0.15f, 0.15f, 1.0f }, sun[4] = { 0.9f, 0.85f, 0.8f, 1.0f }, toggles[4] = { 0, 0, 30, 0 };
  memcpy(psc[1], ambient, 16);
  memcpy(psc[3], sun, 16);
  memcpy(psc[27], toggles, 16);

  constexpr uint32_t W = 64, H = 32;
  Backend::EffectFrame fr{};
  const float frustum[4] = { -1.0f, 1.0f, 1.0f, -1.0f }, port[4] = { 0.0f, 1.0f, 1.0f, 0.0f };
  memcpy(fr.frustum, frustum, 16);
  memcpy(fr.port, port, 16);
  fr.nearZ = 10.0f;
  fr.farZ = 100000.0f;
  const float rows[3][4] = { { 1, 0, 0, 0 }, { 0, 0, 1, 0 }, { 0, 1, 0, 0 } };
  memcpy(fr.cameraRows, rows, sizeof(rows));
  float params[32][4] = {};
  for (size_t i = 0; i < e->params.size() && i < 32; i++) params[i][0] = e->params[i].def;
  float noReject[32][4];  // the control: depth falloff 0 (no depth rejection)
  memcpy(noReject, params, sizeof(params));
  atmostest::setParam(*e, noReject, "thickness", 0.0f);
  float (*runParams)[4] = params;
  uint64_t version = 9900;
  auto draw = [&](BufKey vb, ShaderKey vs, ShaderKey ps) {
    const ShaderReplacement::Constant pbr[1] = { { 223, { 1.0f, 0, 0, 0 } } };
    for (const Md5& m : repltest::g_lightingPs) be.setShaderConstants(m.b.data(), pbr, 1);
    DrawSetup s;
    s.renderStates = rs;
    s.samplerStates = ss;
    s.textures[0] = { DrawSetup::Texture::Tex, 0xC8100 };
    s.textures[1] = { DrawSetup::Texture::Tex, 0xC8140 };
    s.textures[2] = { DrawSetup::Texture::Tex, 0xC8180 };
    s.textures[3] = { DrawSetup::Texture::Tex, 0xC81C0 };
    s.vs = vs;
    s.ps = ps;
    s.decl = decl;
    s.streams[0] = { vb, 0, sizeof(V) };
    s.vsConstants = &vsc[0][0];
    s.psConstants = &psc[0][0];
    s.vsConstantsVersion = s.psConstantsVersion = ++version;
    s.viewport = { 0, 0, W, H, 0.0f, 1.0f };
    s.scissor[2] = int32_t(W);
    s.scissor[3] = int32_t(H);
    be.draw(s, 4, 2, false, 0, 0);
  };
  enum Scene { Flat, DepthStep, Covered };
  // runEffect: whether the effect runs; buffer: whether the skin buffer takes the skin's pixels.
  auto scene = [&](Scene sc, bool buffer, bool runEffect, std::vector<uint8_t>& img) {
    const RtKey rt = newKey(), depth = newKey();
    be.attach(rt, { W, H, kD3DFMT_A16B16G16R16F, 1, false });
    be.attach(depth, { W, H, kD3DFMT_D24S8, 1, true });
    be.beginGroup(&rt, 1, depth);
    const int32_t all4[4] = { 0, 0, int32_t(W), int32_t(H) };
    be.clear(kClearTarget | kClearZBuffer, 0xFF000000u, 1.0f, 0, all4);
    if (buffer) be.setSkinBuffer(true);
    if (sc == DepthStep) {
      draw(leftNear, vsKey, psKey);
      draw(rightFar, vsKey, psKey);
    } else {
      draw(full, vsKey, psKey);
    }
    if (sc == Covered) draw(rightCover, objVs, objPs);
    if (buffer) be.setSkinBuffer(false);
    if (runEffect && !be.runEffect(id, fr, &runParams[0][0], uint32_t(std::min<size_t>(e->params.size(), 32)))) fail("sss: run failed");
    be.endGroup();
    uint32_t vf = 0, ts = 0;
    if (!be.readback(rt, img, vf, ts) || img.size() != size_t(W) * H * 8) fail("sss: readback");
    be.destroy(rt);
    be.destroy(depth);
  };
  auto at = [&](const std::vector<uint8_t>& img, uint32_t x, uint32_t y, int c) {
    return img.size() == size_t(W) * H * 8
               ? double(sunshaftstest::halfToFloat(reinterpret_cast<const uint16_t*>(img.data())[(size_t(y) * W + x) * 4 + c]))
               : -1.0;
  };
  auto maxDiff = [&](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, uint32_t x0, uint32_t x1) {
    double m = 0.0;
    for (uint32_t y = 0; y < H; y++)
      for (uint32_t x = x0; x < x1; x++)
        for (int c = 0; c < 3; c++) m = std::max(m, std::abs(at(a, x, y, c) - at(b, x, y, c)));
    return m;
  };
  std::vector<uint8_t> base, off, on, stepBase, stepOn, coverBase, coverOn;
  scene(Flat, false, false, base);
  scene(Flat, false, true, off);
  scene(Flat, true, true, on);
  scene(DepthStep, false, false, stepBase);
  scene(DepthStep, true, true, stepOn);
  std::vector<uint8_t> stepControl;
  runParams = noReject;
  scene(DepthStep, true, true, stepControl);
  runParams = params;
  scene(Covered, false, false, coverBase);
  scene(Covered, true, true, coverOn);

  const uint32_t y = H / 2, lit = W / 2 - 1, dark = W / 2;
  const double offDiff = maxDiff(off, base, 0, W);
  const double farDiff = maxDiff(on, base, 0, 8);
  double litDrop[3], darkRise[3], step[3];
  for (int c = 0; c < 3; c++) {
    step[c] = at(base, lit, y, c) - at(base, dark, y, c);
    litDrop[c] = (at(base, lit, y, c) - at(on, lit, y, c)) / step[c];
    darkRise[c] = (at(on, dark, y, c) - at(base, dark, y, c)) / step[c];
  }
  // The change beside the depth step less the change 8 pixels away on the same side (the half-float round trip shifts
  // every skin pixel by about one step of its precision), as a share of the flat scene's change there.
  double stepLeak = 0.0;
  auto leakOf = [&](const std::vector<uint8_t>& img) {
    auto change = [&](uint32_t x, int c) { return at(img, x, y, c) - at(stepBase, x, y, c); };
    double l = 0.0;
    for (int c = 0; c < 3; c++) {
      l = std::max(l, std::abs(change(lit, c) - change(lit - 8, c)) / (litDrop[c] * step[c]));
      l = std::max(l, std::abs(change(dark, c) - change(dark + 8, c)) / (darkRise[c] * step[c]));
    }
    return l;
  };
  stepLeak = leakOf(stepOn);
  const double controlLeak = leakOf(stepControl);
  const double coverDiff = maxDiff(coverOn, coverBase, W / 2, W), besideDiff = maxDiff(coverOn, coverBase, 0, W / 2);
  printf("sss: the buffer off %.2g apart; far from the step %.2g apart; at the step (lit %.3f %.3f %.3f | dark %.3f %.3f "
         "%.3f) the lit side drops %.3f %.3f %.3f and the dark side rises %.3f %.3f %.3f of the step; across a depth step "
         "%.3f of the flat scene's change (%.3f without depth rejection); an object over the skin %.2g apart, the skin beside it %.2g\n", offDiff, farDiff, at(base, lit, y, 0),
         at(base, lit, y, 1), at(base, lit, y, 2), at(base, dark, y, 0), at(base, dark, y, 1), at(base, dark, y, 2), litDrop[0],
         litDrop[1], litDrop[2], darkRise[0], darkRise[1], darkRise[2], stepLeak, controlLeak, coverDiff, besideDiff);
  if (offDiff != 0.0) fail("sss: without the skin buffer the effect changed the image");
  if (farDiff > 2e-3) fail("sss: uniform skin far from the step changed");
  if (!(step[0] > 0.05 && step[1] > 0.05 && step[2] > 0.05)) fail("sss: the test scene has no step in the skin's light");
  if (!(litDrop[0] > 0.02 && darkRise[0] > 0.02)) fail("sss: the step in the skin's light is not softened");
  if (!(darkRise[0] > darkRise[1] && darkRise[1] >= darkRise[2] - 1e-3)) fail("sss: red does not spread the furthest");
  if (!(stepLeak < 0.1)) fail("sss: light crossed a depth step");
  if (!(controlLeak > 0.3)) fail("sss: the depth-step check does not see light crossing it (control)");
  if (coverDiff != 0.0) fail("sss: an object drawn over the skin changed");
  if (besideDiff > 2e-3) fail("sss: skin beside an object drawn over it took its light");
  for (BufKey k : { full, leftNear, rightFar, rightCover }) be.destroyBuffer(k);
  for (TexKey k : { TexKey(0xC8100), TexKey(0xC8140), TexKey(0xC8180), TexKey(0xC81C0) }) be.destroyTexture(k);
  be.destroyEffect(id);
}

}  // namespace ssstest
