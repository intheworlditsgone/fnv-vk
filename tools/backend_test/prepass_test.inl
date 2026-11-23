// Depth pre-pass (backend_draw.inl "Depth pre-pass"; included by backend_test.cpp after skin_test.inl):
// - the shader traits it relies on, against the game's bytecode: every translated pixel shader of the cache counts as
//   discarding exactly when its D3D9 bytecode has texkill and as writing depth exactly when it writes oDepth, and every
//   vertex shader's position is Invariant;
// - one scene drawn with and without the pre-pass gives the same image, pixel for pixel: overlapping slanted quads of the
//   game's skin light pass (SKIN2004, ad.frag: eligible) in no depth order, among draws kept out of it (SKIN2000's adts.frag
//   discards; a blended draw; an alpha-tested skin draw, whose shader has no depth module), a clear between draws and an
//   occlusion query around one draw (whose sample count is the same too), and two overlapping alpha-tested quads of the
//   standard lighting (SLS2000, adts.frag; the nearer drawn first) drawn depth-first through their shader's depth module;
// - those alpha-tested quads are cut (the scene without their alpha test differs) and take the depth module's path;
// - the pre-pass is in effect: of two coplanar quads drawn with D3DCMP_LESS the first wins without it, the second with
//   it (the colour pass tests EQUAL: the one tie rule that differs, documented there); the held / depth-only draws and the
//   runs recorded are as counted here.
namespace prepasstest {

// D3D9 bytecode: whether a shader has texkill, writes oDepth (destination register type 9).
static void d3dTraits(const std::vector<uint8_t>& code, bool& kill, bool& depth) {
  kill = depth = false;
  const size_t n = code.size() / 4;
  const uint32_t* t = reinterpret_cast<const uint32_t*>(code.data());
  for (size_t i = 1; i < n;) {
    const uint32_t tok = t[i];
    if (tok == 0x0000FFFF) break;
    const uint32_t op = tok & 0xFFFF;
    if (op == 0xFFFE) { i += 1 + ((tok >> 16) & 0x7FFF); continue; }
    const uint32_t len = (tok >> 24) & 0xF;
    if (op == 0x41) kill = true;
    if (len >= 1 && op != 0x1F && op != 0x51 && i + 1 < n) {
      const uint32_t d = t[i + 1];
      if ((((d >> 28) & 7) | ((d >> 8) & 0x18)) == 9) depth = true;
    }
    i += 1 + len;
  }
}

static void traits(const char* packPath, const char* cacheDir) {
  ShaderPack pack;
  std::string err;
  if (!pack.load(packPath, &err)) { fail("prepass: shader pack: %s", err.c_str()); return; }
  WIN32_FIND_DATAA fd;
  const std::string spvDir = std::string(cacheDir) + "\\spv\\";
  HANDLE h = FindFirstFileA((spvDir + "*.dxso").c_str(), &fd);
  if (h == INVALID_HANDLE_VALUE) { fail("prepass: no .dxso files in %s", spvDir.c_str()); return; }
  uint32_t ps = 0, vs = 0, kills = 0, depths = 0, mismatches = 0, notInvariant = 0, missing = 0;
  do {
    std::string name = fd.cFileName;
    Md5 m;
    if (name.size() != 37 || !drawtest::hexMd5(name.substr(0, 32), m.b.data())) continue;
    const PackedShader* p = pack.find(m);
    if (!p) { missing++; continue; }
    if (!p->pixel) {
      vs++;
      if (!p->invariantPosition) notInvariant++;
      continue;
    }
    std::vector<uint8_t> code;
    if (!drawtest::readFile(spvDir + name, code)) { missing++; continue; }
    bool kill, depth;
    d3dTraits(code, kill, depth);
    ps++;
    kills += kill;
    depths += depth;
    if (kill != p->discards || depth != p->writesDepth) mismatches++;
  } while (FindNextFileA(h, &fd));
  FindClose(h);
  printf("prepass: traits of %u pixel shaders (%u with texkill, %u writing depth), %u vertex shaders: %u mismatches, %u "
         "positions not invariant, %u not in the pack\n", ps, kills, depths, vs, mismatches, notInvariant, missing);
  if (!ps || !vs || !kills) fail("prepass: no shaders checked");
  if (mismatches) fail("prepass: %u pixel shaders' discard / depth traits differ from their bytecode", mismatches);
  if (notInvariant) fail("prepass: %u vertex shaders without an invariant position", notInvariant);
}

static void run(Backend& be, const char* packPath, const char* cacheDir) {
  if (packPath && cacheDir) traits(packPath, cacheDir);
  const char* dir = getenv("FNV_GAME_SHADERS");
  if (!dir || repltest::g_lightingPs.empty()) { printf("prepass: scene skipped\n"); return; }
  const ShaderKey vsE = 0xD1000, psE = 0xD1040, vsN = 0xD1080, psN = 0xD10C0;
  bool ok = false;
  for (const Md5& m : sdpShaderMd5s(dir, "SKIN2004.vso", { 13 })) ok = be.createShader(vsE, m.b.data()) || ok;
  bool ok2 = false;
  for (const Md5& m : sdpShaderMd5s(dir, "SKIN2004.pso", { 13 })) ok2 = be.createShader(psE, m.b.data()) || ok2;
  bool ok3 = false;
  for (const Md5& m : sdpShaderMd5s(dir, "SKIN2000.vso", { 13 })) ok3 = be.createShader(vsN, m.b.data()) || ok3;
  bool ok4 = false;
  for (const Md5& m : sdpShaderMd5s(dir, "SKIN2000.pso", { 13 })) ok4 = be.createShader(psN, m.b.data()) || ok4;
  if (!ok || !ok2 || !ok3 || !ok4) { fail("prepass: SKIN2004 / SKIN2000 shaders not in the pack"); return; }
  // Alpha-tested draws through the standard lighting (SLS2000: not skin, so they have a depth module).
  const ShaderKey vsA = 0xD1300, psA = 0xD1340;
  bool ok5 = false, ok6 = false;
  for (const Md5& m : sdpShaderMd5s(dir, "SLS2000.vso", { 13 })) ok5 = be.createShader(vsA, m.b.data()) || ok5;
  for (const Md5& m : sdpShaderMd5s(dir, "SLS2000.pso", { 13 })) ok6 = be.createShader(psA, m.b.data()) || ok6;
  if (!ok5 || !ok6) { fail("prepass: SLS2000 shaders not in the pack"); return; }
  D3DRef fresh;
  if (!fresh.init()) { fail("prepass: no D3D9 device"); return; }
  static uint32_t rs[256], ss[16][14];
  for (uint32_t i = 0; i < 256; i++) { DWORD v = 0; fresh.dev->GetRenderState(D3DRENDERSTATETYPE(i), &v); rs[i] = v; }
  for (uint32_t s = 0; s < 16; s++)
    for (uint32_t t = 0; t < 14; t++) { DWORD v = 0; fresh.dev->GetSamplerState(s, D3DSAMPLERSTATETYPE(t), &v); ss[s][t] = v; }
  rs[D3DRS_CULLMODE] = D3DCULL_NONE;
  rs[D3DRS_ZENABLE] = 1;
  rs[D3DRS_ZWRITEENABLE] = 1;
  rs[D3DRS_ZFUNC] = D3DCMP_LESSEQUAL;
  auto texture = [&](TexKey key, uint8_t b, uint8_t g, uint8_t r) {
    d3dx::Texture t;
    t.format = 21;
    t.width = t.height = 4;
    t.levels = 1;
    t.cube = false;
    d3dx::Level l;
    l.width = l.height = 4;
    l.rowPitch = 16;
    l.rows = 4;
    for (int i = 0; i < 16; i++) l.data.insert(l.data.end(), { b, g, r, 255 });
    t.surfaces.push_back(std::move(l));
    return be.createTexture(key, t);
  };
  if (!texture(0xD1100, 128, 128, 128) || !texture(0xD1140, 255, 128, 128) || !texture(0xD1180, 128, 128, 128) ||
      !texture(0xD11C0, 64, 64, 64) || !texture(0xD1200, 255, 255, 255)) { fail("prepass: textures"); return; }
  {
    // The alpha-tested quads' diffuse map: 8 x 8 texels of scattered alpha (about half above the reference 128).
    d3dx::Texture t;
    t.format = 21;
    t.width = t.height = 8;
    t.levels = 1;
    t.cube = false;
    d3dx::Level l;
    l.width = l.height = 8;
    l.rowPitch = 32;
    l.rows = 8;
    for (uint32_t i = 0; i < 64; i++) l.data.insert(l.data.end(), { 200, 220, 240, uint8_t((i * 37u + (i >> 3) * 91u) & 255u) });
    t.surfaces.push_back(std::move(l));
    if (!be.createTexture(0xD1380, t)) { fail("prepass: alpha texture"); return; }
  }
  struct V { float p[3], t[3], b[3], n[3], uv[2], c[4]; };
  // A quad in clip space (x0..x1, y0..y1) with depth z00 (x0, y0) .. z11 (x1, y1), bilinear across it (two triangles:
  // slanted planes whose depths are not exact in floating point, as the game's are).
  std::vector<V> verts;
  auto quad = [&](float x0, float y0, float x1, float y1, float z00, float z10, float z01, float z11) {
    auto c = [](float x, float y, float z) {
      return V{ { x, y, z }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { (x + 1) * 0.5f, (y + 1) * 0.5f }, { 1, 1, 1, 1 } };
    };
    const uint32_t start = uint32_t(verts.size());
    verts.insert(verts.end(), { c(x0, y0, z00), c(x1, y0, z10), c(x1, y1, z11), c(x0, y0, z00), c(x1, y1, z11), c(x0, y1, z01) });
    return start;
  };
  // The scene, in drawing order (no depth order): what each draw is and its first vertex.
  const uint32_t qFar = quad(-1, -1, 1, 1, 0.81f, 0.93f, 0.87f, 0.79f);
  const uint32_t qMid = quad(-0.7f, -0.8f, 0.6f, 0.5f, 0.31f, 0.62f, 0.55f, 0.37f);
  const uint32_t qKill = quad(-0.2f, -0.9f, 0.9f, 0.2f, 0.45f, 0.52f, 0.41f, 0.58f);
  const uint32_t qBlend = quad(-0.9f, 0.1f, 0.1f, 0.9f, 0.43f, 0.47f, 0.39f, 0.51f);
  const uint32_t qQuery = quad(0.5f, 0.3f, 0.95f, 0.95f, 0.72f, 0.68f, 0.66f, 0.74f);  // partly behind qMid
  const uint32_t qNear = quad(-0.3f, -0.2f, 0.3f, 0.9f, 0.13f, 0.22f, 0.17f, 0.11f);
  const uint32_t qTest = quad(0.2f, -1.0f, 1.0f, -0.4f, 0.27f, 0.33f, 0.29f, 0.35f);
  const uint32_t qTie = quad(-1.0f, -1.0f, -0.6f, -0.6f, 0.05f, 0.05f, 0.05f, 0.05f);
  // Alpha-tested, overlapping, the nearer one drawn first (without the pre-pass the farther one fills its holes), the
  // farther one crossing qMid and qKill.
  const uint32_t qAlphaNear = quad(-0.85f, -0.6f, 0.25f, 0.75f, 0.20f, 0.28f, 0.24f, 0.21f);
  const uint32_t qAlphaFar = quad(-0.5f, -0.95f, 0.7f, 0.35f, 0.36f, 0.50f, 0.47f, 0.40f);
  const BufKey vb = 0xD1240;
  be.createBuffer(vb, uint32_t(verts.size() * sizeof(V)), false);
  be.writeBuffer(vb, 0, verts.data(), uint32_t(verts.size() * sizeof(V)), 0);
  const VertexElement ve[6] = { { 0, 0, 2, 0, 0, 0 }, { 0, 12, 2, 0, 6, 0 }, { 0, 24, 2, 0, 7, 0 },
                                { 0, 36, 2, 0, 3, 0 }, { 0, 48, 1, 0, 5, 0 }, { 0, 56, 3, 0, 10, 0 } };
  const DeclKey decl = 0xD1280;
  be.createDecl(decl, ve, 6);
  static float vsc[256][4], psc[224][4];
  memset(vsc, 0, sizeof(vsc));
  memset(psc, 0, sizeof(psc));
  for (int i = 0; i < 4; i++) vsc[i][i] = 1.0f;
  const float fog[4] = { 1.0e6f, 1.0f, 1.0f, 0.0f }, light[4] = { 0, 0, 1, 1 }, eye[4] = { 0.3f, -0.2f, 3.0f, 1.0f };
  memcpy(vsc[14], fog, 16);
  memcpy(vsc[16], eye, 16);
  memcpy(vsc[25], light, 16);
  for (int r = 0; r < 3; r++) vsc[250 + r][r] = 1.0f;
  const float ambient[4] = { 0.1f, 0.1f, 0.1f, 1.0f }, toggles[4] = { 0, 0, 30, 0 };
  memcpy(psc[1], ambient, 16);
  memcpy(psc[27], toggles, 16);
  constexpr uint32_t W = 64, H = 64;
  uint64_t version = 9900;
  struct Result {
    std::vector<uint8_t> img;
    uint32_t samples = ~0u;
    BackendStats d;
  };
  // cut: the alpha-tested quads' alpha test on (else they are plain opaque draws).
  auto scene = [&](bool prepass, bool cut, Result& out) {
    const ShaderReplacement::Constant params[1] = { { 223, { 1.0f, 0, 0, 0 } } };
    for (const Md5& m : repltest::g_lightingPs) be.setShaderConstants(m.b.data(), params, 1);
    const RtKey rt = newKey(), ds = newKey();
    be.attach(rt, { W, H, kD3DFMT_A8R8G8B8, 1, false });
    be.attach(ds, { W, H, kD3DFMT_D24S8, 1, true });
    const uint32_t q = be.createQuery();
    const BackendStats s0 = be.stats();
    be.beginGroup(&rt, 1, ds);
    int32_t full[4] = { 0, 0, int32_t(W), int32_t(H) };
    be.clear(kClearTarget | kClearZBuffer | kClearStencil, 0xFF102030, 1.0f, 0, full);
    if (prepass) be.beginDepthPrepass();
    uint32_t localRs[256];
    // shader: 0 SKIN2004 (eligible), 1 SKIN2000 (discards: kept out), 2 SLS2000 (the alpha-tested quads).
    auto draw = [&](uint32_t start, int shader, float r, float g, float b, auto&& states) {
      memcpy(localRs, rs, sizeof(localRs));
      states(localRs);
      const float sun[4] = { r, g, b, 1.0f };
      memcpy(psc[3], sun, 16);
      DrawSetup s;
      s.renderStates = localRs;
      s.samplerStates = ss;
      s.textures[0] = { DrawSetup::Texture::Tex, shader == 2 ? 0xD1380u : 0xD1100u };
      s.textures[1] = { DrawSetup::Texture::Tex, 0xD1140 };
      s.textures[2] = { DrawSetup::Texture::Tex, 0xD1180 };
      s.textures[3] = { DrawSetup::Texture::Tex, 0xD11C0 };
      s.textures[4] = { DrawSetup::Texture::Tex, 0xD1200 };
      s.vs = shader == 2 ? vsA : shader == 1 ? vsN : vsE;
      s.ps = shader == 2 ? psA : shader == 1 ? psN : psE;
      s.decl = decl;
      s.streams[0] = { vb, 0, sizeof(V) };
      s.vsConstants = &vsc[0][0];
      s.psConstants = &psc[0][0];
      s.vsConstantsVersion = s.psConstantsVersion = ++version;
      s.viewport = { 0, 0, W, H, 0.0f, 1.0f };
      s.scissor[2] = int32_t(W);
      s.scissor[3] = int32_t(H);
      be.draw(s, D3DPT_TRIANGLELIST, 2, false, 0, start);
    };
    auto plain = [](uint32_t*) {};
    auto alpha = [cut](uint32_t* s) {  // alpha-tested: drawn depth-first through the shader's depth module
      s[D3DRS_ALPHATESTENABLE] = cut ? 1 : 0;
      s[D3DRS_ALPHAFUNC] = D3DCMP_GREATER;
      s[D3DRS_ALPHAREF] = 128;
    };
    draw(qFar, 0, 0.9f, 0.2f, 0.1f, plain);
    draw(qAlphaNear, 2, 0.95f, 0.85f, 0.3f, alpha);
    draw(qMid, 0, 0.1f, 0.8f, 0.2f, plain);
    draw(qKill, 1, 0.3f, 0.3f, 0.9f, plain);  // its own discard: kept out
    draw(qAlphaFar, 2, 0.35f, 0.9f, 0.85f, alpha);
    draw(qBlend, 0, 0.8f, 0.8f, 0.1f, [](uint32_t* s) {  // blended: kept out
      s[D3DRS_ALPHABLENDENABLE] = 1;
      s[D3DRS_SRCBLEND] = D3DBLEND_ONE;
      s[D3DRS_DESTBLEND] = D3DBLEND_ONE;
    });
    const int32_t rect[4] = { 40, 4, 56, 12 };
    be.clear(kClearTarget, 0xFF00FF00, 1.0f, 0, rect);  // records the held draws first
    be.beginQuery(q);
    draw(qQuery, 0, 0.5f, 0.1f, 0.7f, plain);  // in a query: drawn as it comes
    be.endQuery(q);
    draw(qNear, 0, 0.2f, 0.6f, 0.9f, plain);
    draw(qTest, 0, 0.9f, 0.5f, 0.3f, [](uint32_t* s) {  // alpha tested, a skin shader (no depth module): kept out
      s[D3DRS_ALPHATESTENABLE] = 1;
      s[D3DRS_ALPHAFUNC] = D3DCMP_GREATER;
      s[D3DRS_ALPHAREF] = 0;
    });
    auto less = [](uint32_t* s) { s[D3DRS_ZFUNC] = D3DCMP_LESS; };
    draw(qTie, 0, 1.0f, 0.0f, 0.0f, less);
    draw(qTie, 0, 0.0f, 0.0f, 1.0f, less);
    if (prepass) be.endDepthPrepass();
    be.endGroup();
    uint32_t vf = 0, ts = 0;
    if (!be.readback(rt, out.img, vf, ts) || ts != 4 || out.img.size() != W * H * 4) fail("prepass: readback");
    be.finishFrame();
    if (be.queryResult(q, out.samples) != 1) fail("prepass: query not ready after the frame");
    const BackendStats s1 = be.stats();
    out.d.prepassDraws = s1.prepassDraws - s0.prepassDraws;
    out.d.prepassDepthDraws = s1.prepassDepthDraws - s0.prepassDepthDraws;
    out.d.prepassFlushes = s1.prepassFlushes - s0.prepassFlushes;
    out.d.prepassDiscardingDraws = s1.prepassDiscardingDraws - s0.prepassDiscardingDraws;
    out.d.prepassDiscardingRefused = s1.prepassDiscardingRefused - s0.prepassDiscardingRefused;
    out.d.draws = s1.draws - s0.draws;
    be.destroyQuery(q);
    be.destroy(rt);
    be.destroy(ds);
  };
  Result off, on, uncut;
  scene(false, true, off);
  scene(true, true, on);
  scene(true, false, uncut);  // control: the alpha quads' alpha test off
  // GPU time of draws that can discard, with and without the pre-pass (BENCH=1; a measurement, not a check): 16 layers of
  // full-screen alpha-tested standard lighting (SLS2000, physically based) at 2048 x 2048, drawn back to front (each
  // layer in front of the ones before: without the pre-pass every layer shades every pixel it keeps), best of 5 frames.
  if (getenv("BENCH")) {
    constexpr uint32_t BW = 2048, BH = 2048, kLayers = 16;
    std::vector<V> layers;
    for (uint32_t i = 0; i < kLayers; i++) {
      const float z = 0.9f - 0.05f * float(i);
      auto c = [z](float x, float y) {
        return V{ { x, y, z }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { (x + 1) * 2.0f, (y + 1) * 2.0f }, { 1, 1, 1, 1 } };
      };
      layers.insert(layers.end(), { c(-1, -1), c(1, -1), c(1, 1), c(-1, -1), c(1, 1), c(-1, 1) });
    }
    const BufKey bvb = 0xD13C0;
    be.createBuffer(bvb, uint32_t(layers.size() * sizeof(V)), false);
    be.writeBuffer(bvb, 0, layers.data(), uint32_t(layers.size() * sizeof(V)), 0);
    const ShaderReplacement::Constant params[1] = { { 223, { 1.0f, 0, 0, 0 } } };
    for (const Md5& m : repltest::g_lightingPs) be.setShaderConstants(m.b.data(), params, 1);
    uint32_t brs[256];
    memcpy(brs, rs, sizeof(brs));
    brs[D3DRS_ALPHATESTENABLE] = 1;
    brs[D3DRS_ALPHAFUNC] = D3DCMP_GREATER;
    brs[D3DRS_ALPHAREF] = 128;
    auto frameMs = [&](bool prepass, uint32_t nLayers) {
      const RtKey rt = newKey(), ds = newKey();
      be.attach(rt, { BW, BH, kD3DFMT_A8R8G8B8, 1, false });
      be.attach(ds, { BW, BH, kD3DFMT_D24S8, 1, true });
      double best = 1e30;
      for (int run = 0; run < 6; run++) {
        be.finishFrame();
        LARGE_INTEGER f, t0, t1;
        QueryPerformanceFrequency(&f);
        QueryPerformanceCounter(&t0);
        be.beginGroup(&rt, 1, ds);
        int32_t full[4] = { 0, 0, int32_t(BW), int32_t(BH) };
        be.clear(kClearTarget | kClearZBuffer | kClearStencil, 0xFF102030, 1.0f, 0, full);
        if (prepass) be.beginDepthPrepass();
        for (uint32_t i = kLayers - nLayers; i < kLayers; i++) {
          const float sun[4] = { 0.3f + 0.04f * i, 0.6f, 0.9f - 0.04f * i, 1.0f };
          memcpy(psc[3], sun, 16);
          DrawSetup s;
          s.renderStates = brs;
          s.samplerStates = ss;
          s.textures[0] = { DrawSetup::Texture::Tex, 0xD1380 };
          s.textures[1] = { DrawSetup::Texture::Tex, 0xD1140 };
          s.textures[2] = { DrawSetup::Texture::Tex, 0xD1180 };
          s.textures[3] = { DrawSetup::Texture::Tex, 0xD11C0 };
          s.textures[4] = { DrawSetup::Texture::Tex, 0xD1200 };
          s.vs = vsA;
          s.ps = psA;
          s.decl = decl;
          s.streams[0] = { bvb, 0, sizeof(V) };
          s.vsConstants = &vsc[0][0];
          s.psConstants = &psc[0][0];
          s.vsConstantsVersion = s.psConstantsVersion = ++version;
          s.viewport = { 0, 0, BW, BH, 0.0f, 1.0f };
          s.scissor[2] = int32_t(BW);
          s.scissor[3] = int32_t(BH);
          be.draw(s, D3DPT_TRIANGLELIST, 2, false, 0, i * 6);
        }
        if (prepass) be.endDepthPrepass();
        be.endGroup();
        be.finishFrame();
        QueryPerformanceCounter(&t1);
        if (run) best = std::min(best, double(t1.QuadPart - t0.QuadPart) * 1000.0 / double(f.QuadPart));  // (run 0: pipelines)
      }
      be.destroy(rt);
      be.destroy(ds);
      return best;
    };
    for (uint32_t n : { kLayers, 4u, 1u }) {
      const double without = frameMs(false, n), with = frameMs(true, n);
      printf("prepass bench: %u full-screen alpha-tested SLS2000 layers at %ux%u, back to front: %.2f ms without the "
             "pre-pass, %.2f ms with it\n", n, BW, BH, without, with);
    }
    be.destroyBuffer(bvb);
  }
  for (TexKey k : { 0xD1100, 0xD1140, 0xD1180, 0xD11C0, 0xD1200, 0xD1380 }) be.destroyTexture(k);
  be.destroyBuffer(vb);
  if (off.img.size() != W * H * 4 || on.img.size() != W * H * 4 || uncut.img.size() != W * H * 4) return;
  // The alpha test cuts the quads (the control differs where their holes show what is behind), and the pre-pass draws
  // them depth-first through their shader's depth module, with or without the alpha test (adts.frag keeps the game's
  // texkill: it can discard either way). The skin shaders (the killing quad's, the alpha-tested skin quad's) write the
  // skin buffer: no depth module, kept out.
  uint32_t holes = 0;
  for (size_t i = 0; i < size_t(W) * H; i++) holes += memcmp(&on.img[i * 4], &uncut.img[i * 4], 4) != 0;
  printf("prepass: alpha-tested: %u pixels differ from the same quads without the alpha test; drawn through the depth "
         "module %llu, kept out %llu (without the alpha test: %llu, %llu)\n", holes, on.d.prepassDiscardingDraws,
         on.d.prepassDiscardingRefused, uncut.d.prepassDiscardingDraws, uncut.d.prepassDiscardingRefused);
  if (holes < 100) fail("prepass: the alpha test cut only %u pixels: the alpha-tested quads are not tested", holes);
  for (const Result* r : { &on, &uncut })
    if (r->d.prepassDiscardingDraws != 2 || r->d.prepassDiscardingRefused != 2 || r->d.prepassDepthDraws != 7)
      fail("prepass (%s): through the depth module %llu (2), kept out %llu (2), depth-only %llu (7)",
           r == &on ? "alpha test" : "no alpha test", r->d.prepassDiscardingDraws, r->d.prepassDiscardingRefused,
           r->d.prepassDepthDraws);
  // The tie quad's pixels (clip x, y below -0.6, y up: the bottom-left corner), without its edge (D3D9's half-pixel
  // offset moves the edge by a pixel).
  auto inTie = [&](uint32_t x, uint32_t y) {
    const float cx = (x + 0.5f) / W * 2 - 1, cy = 1 - (y + 0.5f) / H * 2;
    return cx < -0.65f && cy < -0.65f;
  };
  uint32_t diff = 0, tiePixels = 0, tieFirst = 0;
  auto nearTie = [&](uint32_t x, uint32_t y) {
    const float cx = (x + 0.5f) / W * 2 - 1, cy = 1 - (y + 0.5f) / H * 2;
    return cx < -0.55f && cy < -0.55f;
  };
  std::set<uint32_t> colours;
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++) {
      const uint8_t* a = &off.img[(size_t(y) * W + x) * 4];
      const uint8_t* b = &on.img[(size_t(y) * W + x) * 4];
      colours.insert(uint32_t(a[0]) | uint32_t(a[1]) << 8 | uint32_t(a[2]) << 16);
      if (inTie(x, y)) {
        tiePixels++;
        // BGRA: the first (red sun) is more red than blue, the second (blue sun) more blue than red.
        tieFirst += a[2] > a[0] && b[0] > b[2];
        continue;
      }
      if (nearTie(x, y)) continue;  // (the tie's edge)
      if (memcmp(a, b, 4) != 0) {
        if (diff < 4) printf("prepass: pixel (%u, %u) without %02X%02X%02X with %02X%02X%02X\n", x, y, a[2], a[1], a[0], b[2], b[1], b[0]);
        diff++;
      }
    }
  printf("prepass: %zu colours; %u pixels differ outside the tie (of %u); tie: %u of %u pixels first without, second "
         "with; query %u / %u samples; held %llu draws, %llu depth-only, %llu runs (of %llu draws)\n",
         colours.size(), diff, W * H - tiePixels, tieFirst, tiePixels, off.samples, on.samples, on.d.prepassDraws,
         on.d.prepassDepthDraws, on.d.prepassFlushes, on.d.draws);
  if (colours.size() < 8) fail("prepass: the scene has %zu colours: too few to tell its draws apart", colours.size());
  if (diff) fail("prepass: %u pixels differ with the pre-pass", diff);
  if (!tiePixels || tieFirst != tiePixels) fail("prepass: the coplanar LESS quads: %u of %u pixels as expected", tieFirst, tiePixels);
  if (off.samples != on.samples || !off.samples) fail("prepass: query %u samples without, %u with", off.samples, on.samples);
  if (off.d.prepassDraws || off.d.prepassDepthDraws || off.d.prepassFlushes) fail("prepass: draws held without the pre-pass");
  // Held: all but the query's draw (10); depth-only: far, mid, near, the two tie quads and the two alpha-tested quads
  // (7); runs: before the clear, after the query (the query's begin found none held).
  if (on.d.prepassDraws != 10 || on.d.prepassDepthDraws != 7 || on.d.prepassFlushes != 2)
    fail("prepass: held %llu (10), depth-only %llu (7), runs %llu (2)", on.d.prepassDraws, on.d.prepassDepthDraws,
         on.d.prepassFlushes);
  if (on.d.draws != off.d.draws) fail("prepass: %llu draws with, %llu without", on.d.draws, off.d.draws);
}

}  // namespace prepasstest
