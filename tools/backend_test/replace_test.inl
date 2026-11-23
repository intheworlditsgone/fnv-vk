// Shader replacements (included by backend_test.cpp): replacements/water through the plugin's path (loadReplacements,
// game shader names -> MD5 from the game's shader packages, BackendConfig::shaderReplacements), then a water plane
// drawn with the game's WATER000.vso and the replaced WATER000.pso:
// - from straight above, over water whose depth rises from 0 (left) to 1 (right): the ground shows through at the
//   shore and gives way to the water's deep colour;
// - at a grazing angle: mostly the reflection (Fresnel), more so toward the horizon.
// FNV_GAME_SHADERS = the game's Data\Shaders folder (the test is skipped without it).
namespace repltest {

std::vector<fx::Replacement> g_loaded;
std::vector<Md5> g_vs, g_ps;  // WATER000.vso / WATER000.pso builds
std::vector<Md5> g_lightingPs;   // SLS2000.pso, SLS2017.pso builds replaced by replacements/lighting (pbr_test.inl)
std::vector<fx::Param> g_lightingParams;

static void prepare(BackendConfig& cfg) {
  g_loaded = fx::loadReplacements("replacements", "build/fx_cache_test", [](const char* s) { printf("[repl] %s\n", s); });
  const fx::Replacement* water = nullptr;
  for (const fx::Replacement& r : g_loaded)
    if (r.id == "water") water = &r;
  if (!water || !water->error.empty()) { fail("replacements: replacements/water did not load"); return; }
  if (water->targets.size() != 34) fail("replacements: water has %zu targets, expected 34", water->targets.size());
  // Every compiled variant, for spirv-val (run by the test script).
  CreateDirectoryA("build\\repl_spv", nullptr);
  for (const fx::Replacement::Target& t : water->targets)
    if (FILE* f = fopen(("build/repl_spv/" + t.shader + ".spv").c_str(), "wb")) {
      fwrite(t.spv.data(), 4, t.spv.size(), f);
      fclose(f);
    }
  const char* dir = getenv("FNV_GAME_SHADERS");
  if (!dir) { printf("replacements: FNV_GAME_SHADERS not set: the replaced draw is not tested\n"); return; }
  g_vs = sdpShaderMd5s(dir, "WATER000.vso");
  g_ps = sdpShaderMd5s(dir, "WATER000.pso");
  printf("replacements: WATER000.vso %zu builds, WATER000.pso %zu builds\n", g_vs.size(), g_ps.size());
  if (g_vs.empty() || g_ps.empty()) { fail("replacements: WATER000 not found in %s", dir); return; }
  // Every replacement other than water through ShaderPack::replace on its own copy of the pack, as the runtime installs
  // them: each build of each target (in the replacement's packages) must be accepted, and every texture it samples must
  // have the register and kind (2D or cube) of the translation it replaces.
  if (!cfg.shaderPack.empty()) {
    ShaderPack pack;
    std::string err;
    if (!pack.load(cfg.shaderPack, &err)) fail("replacements: pack %s: %s", cfg.shaderPack.c_str(), err.c_str());
    for (const fx::Replacement& r : g_loaded) {
      if (r.id == "water" || !r.error.empty()) continue;
      size_t installed = 0;
      for (const fx::Replacement::Target& t : r.targets)
        for (const Md5& m : sdpShaderMd5s(dir, t.shader, r.packages)) {
          const PackedShader* orig = pack.find(m);
          if (!orig) { fail("replacements: %s %s is not in the pack", r.id.c_str(), t.shader.c_str()); continue; }
          const std::vector<PackedShader::Tex> before = orig->textures;
          if (!pack.replace(m, t.spv, {})) { fail("replacements: %s %s rejected", r.id.c_str(), t.shader.c_str()); continue; }
          for (const PackedShader::Tex& x : pack.find(m)->textures) {
            bool same = false;
            for (const PackedShader::Tex& y : before) same |= y.reg == x.reg && y.kind == x.kind;
            if (!same) fail("replacements: %s %s s%u kind %u is not the original's", r.id.c_str(), t.shader.c_str(), x.reg, x.kind);
          }
          installed++;
        }
      printf("replacements: %s: %zu builds installed\n", r.id.c_str(), installed);
    }
  }
  // The lighting replacement's SLS2000.pso and SLS2017.pso (objects: the sun, ambient; and with specular) for
  // tools/backend_test/pbr_test.inl, installed
  // as the plugin does: its builds in the replacement's packages, parameters at their defaults.
  for (const fx::Replacement& r : g_loaded) {
    if (r.id != "lighting" || !r.error.empty()) continue;
    for (const fx::Replacement::Target& t : r.targets) {
      if (t.shader != "SLS2000.pso" && t.shader != "SLS2017.pso" && t.shader != "SLS2092.pso" && t.shader != "SKIN2000.pso" &&
          t.shader != "SKIN2004.pso" && t.shader != "GRASS2000.pso")
        continue;  // ADTS, ADTS SPECULAR, LandO, skin
      for (const Md5& m : sdpShaderMd5s(dir, t.shader, r.packages)) {
        ShaderReplacement sr;
        memcpy(sr.md5, m.b.data(), 16);
        sr.spv = t.spv;
        for (size_t i = 0; i < r.params.size(); i++)
          sr.constants.push_back({ uint32_t(fx::kReplacementParamTop - i), { r.params[i].def, 0, 0, 0 } });
        cfg.shaderReplacements.push_back(std::move(sr));
        g_lightingPs.push_back(m);
      }
      g_lightingParams = r.params;
    }
  }
  // Parameters at their defaults, as the plugin passes them.
  std::vector<ShaderReplacement::Constant> constants;
  for (size_t i = 0; i < water->params.size(); i++)
    constants.push_back({ uint32_t(fx::kReplacementParamTop - i), { water->params[i].def, 0, 0, 0 } });
  for (const fx::Replacement::Target& t : water->targets) {
    if (t.shader != "WATER000.pso") continue;
    for (const Md5& m : g_ps) {
      ShaderReplacement sr;
      memcpy(sr.md5, m.b.data(), 16);
      sr.spv = t.spv;
      sr.constants = constants;
      cfg.shaderReplacements.push_back(std::move(sr));
    }
  }
}

// D3D-style (row vectors dotted with rows) perspective view-projection, left-handed, z forward.
struct Mat { float r[4][4]; };
static Mat viewProj(const float eye[3], const float at[3], const float upHint[3], float fovY, float zn, float zf) {
  auto sub = [](const float* a, const float* b, float* o) { for (int i = 0; i < 3; i++) o[i] = a[i] - b[i]; };
  auto norm = [](float* v) { float l = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); for (int i = 0; i < 3; i++) v[i] /= l; };
  auto cross = [](const float* a, const float* b, float* o) {
    o[0] = a[1] * b[2] - a[2] * b[1]; o[1] = a[2] * b[0] - a[0] * b[2]; o[2] = a[0] * b[1] - a[1] * b[0];
  };
  auto dot = [](const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
  float f[3], r[3], u[3];
  sub(at, eye, f);
  norm(f);
  cross(upHint, f, r);
  norm(r);
  cross(f, r, u);
  const float sy = 1.0f / std::tan(fovY * 0.5f), sx = sy, A = zf / (zf - zn), B = -zn * zf / (zf - zn);
  Mat m{};
  for (int i = 0; i < 3; i++) {
    m.r[0][i] = sx * r[i];
    m.r[1][i] = sy * u[i];
    m.r[2][i] = A * f[i];
    m.r[3][i] = f[i];
  }
  m.r[0][3] = -sx * dot(r, eye);
  m.r[1][3] = -sy * dot(u, eye);
  m.r[2][3] = -A * dot(f, eye) + B;
  m.r[3][3] = -dot(f, eye);
  return m;
}

static void run(Backend& be) {
  if (g_vs.empty() || g_ps.empty()) return;
  // D3D9's default render and sampler states (a fresh device), as the game's draws start from.
  D3DRef fresh;
  if (!fresh.init()) { fail("replacements: no D3D9 device for default states"); return; }
  static uint32_t rs[256], ss[16][14];
  for (uint32_t i = 0; i < 256; i++) { DWORD v = 0; fresh.dev->GetRenderState(D3DRENDERSTATETYPE(i), &v); rs[i] = v; }
  for (uint32_t s = 0; s < 16; s++)
    for (uint32_t t = 0; t < 14; t++) { DWORD v = 0; fresh.dev->GetSamplerState(s, D3DSAMPLERSTATETYPE(t), &v); ss[s][t] = v; }
  rs[D3DRS_ZENABLE] = 0;
  rs[D3DRS_CULLMODE] = D3DCULL_NONE;
  for (uint32_t s = 0; s < 5; s++) {
    ss[s][D3DSAMP_ADDRESSU] = ss[s][D3DSAMP_ADDRESSV] = D3DTADDRESS_CLAMP;
    ss[s][D3DSAMP_MAGFILTER] = ss[s][D3DSAMP_MINFILTER] = D3DTEXF_LINEAR;
  }

  const ShaderKey vsKey = 0x9C000, psKey = 0x9C040;
  if (!be.createShader(vsKey, g_vs[0].b.data()) || !be.createShader(psKey, g_ps[0].b.data())) {
    fail("replacements: WATER000 is not in the shader pack");
    return;
  }
  // Textures: sky reflection, ground behind the water, a flat noise normal, depth rising left to right (.y).
  const float sky[3] = { 0.55f, 0.70f, 0.90f }, ground[3] = { 0.45f, 0.32f, 0.20f };
  auto makeTex = [&](TexKey key, auto texel) {
    d3dx::Texture t;
    t.format = 21;
    t.width = t.height = 64;
    t.levels = 1;
    t.cube = false;
    d3dx::Level l;
    l.width = l.height = 64;
    l.rowPitch = 256;
    l.rows = 64;
    l.data.resize(64 * 256);
    for (uint32_t y = 0; y < 64; y++)
      for (uint32_t x = 0; x < 64; x++) {
        float c[3];
        texel(x, c);
        uint8_t* p = &l.data[y * 256 + x * 4];
        p[0] = uint8_t(c[2] * 255 + 0.5f); p[1] = uint8_t(c[1] * 255 + 0.5f); p[2] = uint8_t(c[0] * 255 + 0.5f); p[3] = 255;
      }
    t.surfaces.push_back(std::move(l));
    if (!be.createTexture(key, t)) fail("replacements: texture %llx", (unsigned long long)key);
  };
  const TexKey reflKey = 0x9D000, refrKey = 0x9D040, noiseKey = 0x9D080, depthKey = 0x9D0C0;
  makeTex(reflKey, [&](uint32_t, float* c) { memcpy(c, sky, 12); });
  makeTex(refrKey, [&](uint32_t, float* c) { memcpy(c, ground, 12); });
  makeTex(noiseKey, [&](uint32_t, float* c) { c[0] = c[1] = 128.0f / 255.0f; c[2] = 1.0f; });
  makeTex(depthKey, [&](uint32_t x, float* c) { c[0] = c[1] = c[2] = float(x) / 63.0f; });

  // A 4000 x 4000 plane at z = 0: position, texcoord.
  const float h = 2000.0f;
  const float verts[6][5] = { { -h, -h, 0, 0, 0 }, { h, -h, 0, 1, 0 }, { h, h, 0, 1, 1 },
                              { -h, -h, 0, 0, 0 }, { h, h, 0, 1, 1 }, { -h, h, 0, 0, 1 } };
  const BufKey vb = 0x9E000;
  be.createBuffer(vb, sizeof(verts), false);
  be.writeBuffer(vb, 0, verts, sizeof(verts), 0);
  const VertexElement ve[2] = { { 0, 0, 2 /*FLOAT3*/, 0, 0 /*POSITION*/, 0 }, { 0, 12, 1 /*FLOAT2*/, 0, 5 /*TEXCOORD*/, 0 } };
  const DeclKey decl = 0x9E040;
  be.createDecl(decl, ve, 2);

  // The game's water constants (WaterShader::SetupGeometryConstants): no fog, depth falloff 0..1.
  static float vsc[256][4], psc[224][4];
  const float shallow[3] = { 0.30f, 0.50f, 0.50f }, deep[3] = { 0.05f, 0.15f, 0.20f };
  auto setC = [](float (*c)[4], int r, float x, float y, float z, float w) { c[r][0] = x; c[r][1] = y; c[r][2] = z; c[r][3] = w; };
  const uint32_t W = 128, H = 128;
  struct View { const char* name; float eye[3], at[3], up[3], sun[3]; } views[2] = {
    { "straight down", { 0, 0, 300 }, { 0, 0, 0 }, { 0, 1, 0 }, { 0.0f, 0.5f, 0.866f } },
    { "grazing", { 0, -1900, 8 }, { 0, 1900, 0 }, { 0, 0, 1 }, { 0.0f, -0.5f, 0.866f } },  // sun behind the camera
  };
  std::vector<uint8_t> img[2];
  for (int vi = 0; vi < 2; vi++) {
    const View& v = views[vi];
    memset(vsc, 0, sizeof(vsc));
    memset(psc, 0, sizeof(psc));
    const Mat m = viewProj(v.eye, v.at, v.up, 1.0472f, 1.0f, 10000.0f);
    for (int r = 0; r < 4; r++) setC(vsc, r, m.r[r][0], m.r[r][1], m.r[r][2], m.r[r][3]);
    setC(vsc, 4, 1, 0, 0, 0); setC(vsc, 5, 0, 1, 0, 0); setC(vsc, 6, 0, 0, 1, 0);  // WorldMat
    setC(vsc, 8, 1, 0, 0, 0);                                                     // TexScale
    setC(psc, 1, v.eye[0], v.eye[1], v.eye[2], 0);
    setC(psc, 2, shallow[0], shallow[1], shallow[2], 1);
    setC(psc, 3, deep[0], deep[1], deep[2], 1);
    setC(psc, 4, 0.5f, 0.6f, 0.7f, 1);              // ReflectionColor
    setC(psc, 5, 0.02f, 1.0f, 0.0f, 1.0f);          // FresnelRI: R0, fog power, -, reflectivity
    setC(psc, 8, 100.0f, 1.0f, 0.0f, 1.0f);         // VarAmounts: specular power, reflection amount, -, distortion
    setC(psc, 9, 1e7f, 1e7f, 0.0f, 1.0f);           // FogParam: no fog
    setC(psc, 10, 0.5f, 0.5f, 0.5f, 0.0f);          // FogColor, opacity 0
    setC(psc, 11, 0.0f, 1.0f, 0, 0);                // DepthFalloff
    setC(psc, 12, v.sun[0], v.sun[1], v.sun[2], 1.0f);
    setC(psc, 13, 1.0f, 0.95f, 0.9f, 1);
    DrawSetup s;
    s.renderStates = rs;
    s.samplerStates = ss;
    s.textures[0] = { DrawSetup::Texture::Tex, reflKey };
    s.textures[1] = { DrawSetup::Texture::Tex, refrKey };
    s.textures[2] = { DrawSetup::Texture::Tex, noiseKey };
    s.textures[4] = { DrawSetup::Texture::Tex, depthKey };
    s.vs = vsKey;
    s.ps = psKey;
    s.decl = decl;
    s.streams[0] = { vb, 0, 20 };
    s.vsConstants = &vsc[0][0];
    s.psConstants = &psc[0][0];
    s.vsConstantsVersion = s.psConstantsVersion = 5000 + vi;
    s.viewport = { 0, 0, W, H, 0.0f, 1.0f };
    s.scissor[2] = int32_t(W);
    s.scissor[3] = int32_t(H);
    const RtKey rt = newKey();
    be.attach(rt, { W, H, kD3DFMT_A8R8G8B8, 1, false });
    be.beginGroup(&rt, 1, 0);
    int32_t full[4] = { 0, 0, int32_t(W), int32_t(H) };
    be.clear(kClearTarget, 0xFFFF00FF, 1.0f, 0, full);  // magenta: not water
    be.draw(s, 4 /*D3DPT_TRIANGLELIST*/, 2, false, 0, 0);
    be.endGroup();
    uint32_t vf = 0, ts = 0;
    if (!be.readback(rt, img[vi], vf, ts) || ts != 4) fail("replacements: readback of view %s", v.name);
    be.destroy(rt);
  }
  for (int vi = 0; vi < 2; vi++)  // raw BGRA, for inspection
    if (FILE* f = fopen(("build/water_view" + std::to_string(vi) + ".bgra").c_str(), "wb")) {
      fwrite(img[vi].data(), 1, img[vi].size(), f);
      fclose(f);
    }
  auto px = [&](int vi, uint32_t x, uint32_t y) {
    const uint8_t* p = &img[vi][(size_t(y) * W + x) * 4];
    return std::array<float, 3>{ p[2] / 255.0f, p[1] / 255.0f, p[0] / 255.0f };
  };
  auto dist = [](std::array<float, 3> a, const float* b) {
    return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
  };
  auto isClear = [](std::array<float, 3> c) { return c[0] > 0.99f && c[1] < 0.01f && c[2] > 0.99f; };
  if (!img[0].empty()) {
    // Straight down: depth = screen x.
    const float light = 0.35f + 0.65f * 0.866f;
    const float deepLit[3] = { deep[0] * light, deep[1] * light, deep[2] * light };
    const auto shore = px(0, 4, H / 2), mid = px(0, W / 2, H / 2), deepPx = px(0, W - 5, H / 2);
    printf("water straight down: shore %.3f %.3f %.3f | middle %.3f %.3f %.3f | deep %.3f %.3f %.3f (ground %.2f %.2f %.2f, deep lit %.3f %.3f %.3f)\n",
           shore[0], shore[1], shore[2], mid[0], mid[1], mid[2], deepPx[0], deepPx[1], deepPx[2], ground[0], ground[1], ground[2],
           deepLit[0], deepLit[1], deepLit[2]);
    if (isClear(shore) || isClear(mid) || isClear(deepPx)) fail("water: the plane does not cover the view");
    if (!(dist(shore, ground) < dist(shore, deepLit))) fail("water: the ground does not show through at the shore");
    if (!(dist(deepPx, deepLit) < dist(deepPx, ground))) fail("water: deep water is not the deep colour");
    float prevRed = 2.0f;
    for (uint32_t x = 4; x < W; x += 12) {
      const float r = px(0, x, H / 2)[0];
      if (r > prevRed + 0.01f) { fail("water: red does not fall with depth (x %u: %.3f after %.3f)", x, r, prevRed); break; }
      prevRed = r;
    }
  }
  if (!img[1].empty()) {
    // Grazing: the plane is the lower half; near the horizon it reflects the sky more than further down.
    const auto horizon = px(1, W / 2, H / 2 + 6), lower = px(1, W / 2, H - 4);
    printf("water grazing: near the horizon %.3f %.3f %.3f | bottom of view %.3f %.3f %.3f (sky %.2f %.2f %.2f)\n",
           horizon[0], horizon[1], horizon[2], lower[0], lower[1], lower[2], sky[0], sky[1], sky[2]);
    if (isClear(horizon) || isClear(lower)) fail("water: the grazing view does not show the plane");
    if (!(dist(horizon, sky) < 0.12f)) fail("water: at a grazing angle the reflection does not dominate");
    if (!(dist(horizon, sky) < dist(lower, sky))) fail("water: reflection does not grow toward the horizon");
  }
  be.destroyTexture(reflKey);
  be.destroyTexture(refrKey);
  be.destroyTexture(noiseKey);
  be.destroyTexture(depthKey);
  be.destroyBuffer(vb);
  be.destroyDecl(decl);
  be.destroyShader(vsKey);
  be.destroyShader(psKey);
}

}  // namespace repltest
