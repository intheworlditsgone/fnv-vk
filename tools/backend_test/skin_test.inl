// Skin (docs/SKIN.md; included by backend_test.cpp after wet_test.inl): the game's SKIN2000.vso with
// replacements/lighting's SKIN2000.pso (adts.frag FACEGENBLEND, physically based) on a flat quad, as pbr_test.inl's
// (object space = clip space, the frame the identity, the sun along +z, the eye above), against the skin model computed
// here from Community Shaders' formulas (Skin.hlsli, BRDF.hlsli): every pixel within 2 of 255;
// - without a detail map (the zero dummy) and with a flat one (BC5, 0.5 everywhere): the same image (within 1);
// - with the shipped detail map (replacements/lighting/skin_detail_n.dds, read by plugin/src/bc5dds.h): a different one;
// - the multi-pass light pass (SKIN2004.vso + SKIN2004.pso, ad.frag): the model's albedo-scaled light plus its specular
//   divided by the base map (the Texture pass multiplies by the albedo after it), every pixel within 2;
// - rain (a wet state, wetness 1): wet skin differs from dry; with the local puddles full and covering everything (as
//   wet_test.inl's ground, which stands under water then) skin is the same as without them: water runs off skin.
namespace skintest {

struct Vec3 {
  double x, y, z;
};
static Vec3 operator+(Vec3 a, Vec3 b) { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
static Vec3 operator-(Vec3 a, Vec3 b) { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
static Vec3 operator*(Vec3 a, double s) { return { a.x * s, a.y * s, a.z * s }; }
static double dot(Vec3 a, Vec3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static Vec3 unit(Vec3 a) { return a * (1.0 / std::sqrt(dot(a, a))); }
static double sat(double v) { return std::clamp(v, 0.0, 1.0); }
static double schlick(double f0, double vh) { const double fc = std::pow(1.0 - vh, 5.0); return fc + (1.0 - fc) * f0; }
static void envBrdf(double r, double nv, double& a, double& b) {
  const double c0[4] = { -1, -0.0275, -0.572, 0.022 }, c1[4] = { 1, 0.0425, 1.04, -0.04 };
  double q[4];
  for (int i = 0; i < 4; i++) q[i] = r * c0[i] + c1[i];
  const double a004 = std::min(q[0] * q[0], std::exp2(-9.28 * nv)) * q[0] + q[1];
  a = -1.04 * a004 + q[2];
  b = 1.04 * a004 + q[3];
}
static double ggx(double r, double nh) {
  const double a = r * r, a2 = a * a, d = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / (3.14159265 * d * d);
}

// The skin's light at a pixel: N, V (unit, tangent = world here), the sun along +z of linear colour sun[ch], the
// ambient lin(amb) (no probe, no occlusion); albedo linear. Returns the linear colour.
static double skinModel(Vec3 N, Vec3 V, double sun, double amb, double albedo, double* scaled = nullptr,
                        double* unscaled = nullptr) {
  const double pi = 3.14159265, f0 = 0.0278;
  const double nvRaw = dot(N, V);
  const double edge = schlick(0.04, sat(nvRaw)) * 0.25;
  const double r1 = std::clamp(sat(0.7) + edge, 0.02, 1.0), r2 = std::clamp(0.35 + edge, 0.02, 1.0);
  const double avg = r1 + (r2 - r1) * 0.15;
  const double NdotV = sat(std::abs(nvRaw) + 1e-5);
  double energy;
  {
    const double a = avg * avg, t = NdotV / (NdotV + 1.103665 * a), x = 2.0 * t - 1.0;
    const double grazing = 1.424916 + x * (0.341704 + x * (-0.646581 - 0.742605 * x)) - avg * (0.847600 + 1.693598 * x);
    const double rough = 1.841562 + 0.210406 * NdotV - 0.899554 * avg;
    const double single = sat(1.0 - t * ((1.0 - t) * grazing + a * a * rough));
    energy = 1.0 + f0 * (1.0 / single - 1.0);
  }
  const Vec3 L{ 0, 0, 1 };
  const Vec3 H = unit(V + L);
  const double E = sun * pi;
  const double diffuseCos = std::max(sat(dot(N, L)), 1e-5), NdotL = std::clamp(dot(N, L), 1e-5, 1.0);
  const double NdotH = sat(dot(N, H)), VdotH = sat(dot(V, H));
  const double fd90 = 0.5 + 2.0 * VdotH * VdotH * avg;
  const double burley = (1.0 + (fd90 - 1.0) * std::pow(1.0 - NdotV, 5.0)) * (1.0 + (fd90 - 1.0) * std::pow(1.0 - diffuseCos, 5.0)) / pi;
  double d = E * diffuseCos * burley;
  const double D = ggx(r1, NdotH) + (ggx(r2, NdotH) - ggx(r1, NdotH)) * 0.15, a = avg * avg;
  const double G = 0.5 / std::max(NdotL * (NdotV * (1.0 - a) + a) + NdotV * (NdotL * (1.0 - a) + a), 1e-6);
  const double F = schlick(f0, VdotH);
  double s = E * D * G * F * NdotL * energy;
  d *= 1.0 - F;
  double fa, fb;
  envBrdf(0.35, NdotV, fa, fb);
  const double invAlpha = std::pow(0.35, -4.0);
  const double charlie = (2.0 + invAlpha) * std::pow(std::max(1.0 - NdotH * NdotH, 0.0), invAlpha * 0.5) / (2.0 * pi) *
                         (1.0 / (4.0 * std::max(NdotL + NdotV - NdotL * NdotV, 1e-6))) * schlick(0.045, VdotH) * NdotL;
  const double fuzz = E * charlie * (1.0 + 0.045 * (1.0 / (fa + fb) - 1.0));
  const double retained = 1.0 - sat(schlick(0.045, VdotH));
  const double diffuse = d * retained, spec = s * retained + fuzz;
  double ea, eb;
  const double nv = sat(nvRaw);
  envBrdf(avg, nv, ea, eb);
  const double ls = f0 * ea + eb;
  const double lobeD = 1.0 - sat(ls), lobeS = ls * (1.0 + f0 * (1.0 / (ea + eb) - 1.0));
  const Vec3 r = N * (2.0 * dot(N, V)) - V;
  const double horizon = sat(1.0 + r.z);
  if (scaled) *scaled = diffuse + lobeD * amb;
  if (unscaled) *unscaled = spec + amb * lobeS * horizon * horizon;
  return albedo * (diffuse + lobeD * amb) + spec + amb * lobeS * horizon * horizon;
}

static void run(Backend& be) {
  const char* dir = getenv("FNV_GAME_SHADERS");
  if (!dir || repltest::g_lightingPs.empty()) { printf("skin: skipped\n"); return; }
  const ShaderKey vsKey = 0xC7000, psKey = 0xC7040;
  bool haveVs = false, havePs = false;
  for (const Md5& m : sdpShaderMd5s(dir, "SKIN2000.vso", { 13 })) haveVs = haveVs || be.createShader(vsKey, m.b.data());
  for (const Md5& m : sdpShaderMd5s(dir, "SKIN2000.pso", { 13 })) havePs = havePs || be.createShader(psKey, m.b.data());
  if (!haveVs || !havePs) { fail("skin: SKIN2000.vso / SKIN2000.pso not in the shader pack"); return; }
  const ShaderKey mpVsKey = 0xC7280, mpPsKey = 0xC72C0;
  bool haveMpVs = false, haveMpPs = false;
  for (const Md5& m : sdpShaderMd5s(dir, "SKIN2004.vso", { 13 })) haveMpVs = haveMpVs || be.createShader(mpVsKey, m.b.data());
  for (const Md5& m : sdpShaderMd5s(dir, "SKIN2004.pso", { 13 })) haveMpPs = haveMpPs || be.createShader(mpPsKey, m.b.data());
  if (!haveMpVs || !haveMpPs) { fail("skin: SKIN2004.vso / SKIN2004.pso not in the shader pack"); return; }
  D3DRef fresh;
  if (!fresh.init()) { fail("skin: no D3D9 device"); return; }
  static uint32_t rs[256], ss[16][14];
  for (uint32_t i = 0; i < 256; i++) { DWORD v = 0; fresh.dev->GetRenderState(D3DRENDERSTATETYPE(i), &v); rs[i] = v; }
  for (uint32_t s = 0; s < 16; s++)
    for (uint32_t t = 0; t < 14; t++) { DWORD v = 0; fresh.dev->GetSamplerState(s, D3DSAMPLERSTATETYPE(t), &v); ss[s][t] = v; }
  rs[D3DRS_ZENABLE] = 0;
  rs[D3DRS_CULLMODE] = D3DCULL_NONE;
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
  // Base grey 128, a flat normal map, FaceGenMap0 128 (no detail) and FaceGenMap1 64: the albedo
  // (fg0 * 2 - 1 + base) * fg1 * 4.
  const uint8_t base = 128, fg0 = 128, fg1 = 64;
  if (!texture(0xC7080, base, base, base) || !texture(0xC70C0, 255, 128, 128) || !texture(0xC7100, fg0, fg0, fg0) ||
      !texture(0xC7140, fg1, fg1, fg1) || !texture(0xC7300, 255, 255, 255)) { fail("skin: textures"); return; }
  struct V { float p[3], t[3], b[3], n[3], uv[2], c[4]; };
  auto corner = [](float x, float y, float u, float v) {
    return V{ { x, y, 0.5f }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { u, v }, { 1, 1, 1, 1 } };
  };
  // Texture coordinates 0..0.01: the detail map (tiled 10 times per unit) covers a tenth of its width over the 32
  // pixels, about 3 of its texels per pixel, so its fine mips are sampled (over 0..1 it would read a 4 x 4 mip: flat).
  const float u1 = 0.01f;
  const V quad[6] = { corner(-1, -1, 0, 0), corner(1, -1, u1, 0), corner(1, 1, u1, u1),
                      corner(-1, -1, 0, 0), corner(1, 1, u1, u1), corner(-1, 1, 0, u1) };
  const BufKey vb = 0xC7180;
  be.createBuffer(vb, sizeof(quad), false);
  be.writeBuffer(vb, 0, quad, sizeof(quad), 0);
  const VertexElement ve[6] = { { 0, 0, 2, 0, 0, 0 }, { 0, 12, 2, 0, 6, 0 }, { 0, 24, 2, 0, 7, 0 },
                                { 0, 36, 2, 0, 3, 0 }, { 0, 48, 1, 0, 5, 0 }, { 0, 56, 3, 0, 10, 0 } };
  const DeclKey decl = 0xC71C0;
  be.createDecl(decl, ve, 6);
  static float vsc[256][4], psc[224][4];
  memset(vsc, 0, sizeof(vsc));
  memset(psc, 0, sizeof(psc));
  for (int i = 0; i < 4; i++) vsc[i][i] = 1.0f;
  const float fog[4] = { 1.0e6f, 1.0f, 1.0f, 0.0f }, light[4] = { 0, 0, 1, 1 }, eye[4] = { 0.3f, -0.2f, 3.0f, 1.0f };
  memcpy(vsc[14], fog, 16);
  memcpy(vsc[16], eye, 16);
  memcpy(vsc[25], light, 16);
  for (int r = 0; r < 3; r++) vsc[250 + r][r] = 1.0f;  // the world rows: the identity
  const float ambient[4] = { 0.1f, 0.1f, 0.1f, 1.0f }, sun[4] = { 0.8f, 0.7f, 0.6f, 1.0f }, toggles[4] = { 0, 0, 30, 0 };
  memcpy(psc[1], ambient, 16);
  memcpy(psc[3], sun, 16);
  memcpy(psc[27], toggles, 16);
  const uint32_t W = 32, H = 32;
  uint64_t version = 9800;
  // The wet block's camera (as wet_test.inl's): only the rain amount matters here (no cover, no puddle map).
  Backend::WetLighting wetBase{};
  {
    const float rows[3][4] = { { 1, 0, 0, 0 }, { 0, 0, 1, 0 }, { 0, 1, 0, 0 } };
    memcpy(wetBase.cameraRows, rows, sizeof(rows));
    const float frustum[4] = { -1, 1, 1, -1 }, port[4] = { 0, 1, 1, 0 };
    memcpy(wetBase.frustum, frustum, 16);
    memcpy(wetBase.port, port, 16);
    wetBase.nearZ = 1.0f;
    wetBase.farZ = 3.0f;
  }
  ShaderKey drawVs = vsKey, drawPs = psKey;
  const Backend::WetLighting* drawWet = nullptr;
  auto draw = [&](std::vector<uint8_t>& img) {
    const ShaderReplacement::Constant params[1] = { { 223, { 1.0f, 0, 0, 0 } } };
    for (const Md5& m : repltest::g_lightingPs) be.setShaderConstants(m.b.data(), params, 1);
    DrawSetup s;
    s.renderStates = rs;
    s.samplerStates = ss;
    s.textures[0] = { DrawSetup::Texture::Tex, 0xC7080 };
    s.textures[1] = { DrawSetup::Texture::Tex, 0xC70C0 };
    s.textures[2] = { DrawSetup::Texture::Tex, 0xC7100 };
    s.textures[3] = { DrawSetup::Texture::Tex, 0xC7140 };
    s.textures[4] = { DrawSetup::Texture::Tex, 0xC7300 };  // ad.frag's attenuation map: white, its point light out
    s.vs = drawVs;
    s.ps = drawPs;
    s.decl = decl;
    s.streams[0] = { vb, 0, sizeof(V) };
    s.vsConstants = &vsc[0][0];
    s.psConstants = &psc[0][0];
    s.vsConstantsVersion = s.psConstantsVersion = ++version;
    s.viewport = { 0, 0, W, H, 0.0f, 1.0f };
    s.scissor[2] = int32_t(W);
    s.scissor[3] = int32_t(H);
    const RtKey rt = newKey();
    be.attach(rt, { W, H, kD3DFMT_A8R8G8B8, 1, false });
    be.beginGroup(&rt, 1, 0);
    int32_t full[4] = { 0, 0, int32_t(W), int32_t(H) };
    be.clear(kClearTarget, 0xFFFF00FF, 1.0f, 0, full);
    be.setWetLighting(drawWet);
    be.draw(s, 4, 2, false, 0, 0);
    be.setWetLighting(nullptr);
    be.endGroup();
    uint32_t vf = 0, ts = 0;
    if (!be.readback(rt, img, vf, ts) || ts != 4) fail("skin: readback");
    be.destroy(rt);
  };
  auto px = [&](const std::vector<uint8_t>& img, uint32_t x, uint32_t y, int ch) {
    return img.size() == W * H * 4 ? int(img[(size_t(y) * W + x) * 4 + 2 - ch]) : -1;
  };
  std::vector<uint8_t> plain, flat, detailed;
  be.setSkinDetail(0);
  draw(plain);
  // A flat detail map: BC5 blocks whose two channels are 128 throughout (both endpoints 128, every index 0).
  {
    d3dx::Texture t;
    t.format = 0x55354342;  // "BC5U"
    t.width = t.height = 8;
    t.levels = 1;
    d3dx::Level l;
    l.width = l.height = 8;
    l.rowPitch = 32;
    l.rows = 2;
    for (int b = 0; b < 4; b++)
      for (int c = 0; c < 2; c++) l.data.insert(l.data.end(), { 128, 128, 0, 0, 0, 0, 0, 0 });
    t.surfaces.push_back(std::move(l));
    if (!be.createTexture(0xC7200, t)) fail("skin: a BC5 texture");
    be.setSkinDetail(0xC7200);
    draw(flat);
  }
  // The shipped detail map.
  bool haveDetail = false;
  {
    FILE* f = fopen("replacements/lighting/skin_detail_n.dds", "rb");
    if (f) {
      std::vector<uint8_t> file;
      fseek(f, 0, SEEK_END);
      file.resize(size_t(ftell(f)));
      fseek(f, 0, SEEK_SET);
      const bool read = fread(file.data(), 1, file.size(), f) == file.size();
      fclose(f);
      d3dx::Texture t;
      std::string why;
      if (!read || !bc5DdsTexture(file.data(), file.size(), t, &why)) fail(("skin: the shipped detail map: " + why).c_str());
      else if (!be.createTexture(0xC7240, t)) fail("skin: the shipped detail map does not upload");
      else {
        be.setSkinDetail(0xC7240);
        draw(detailed);
        haveDetail = true;
      }
    } else fail("skin: replacements/lighting/skin_detail_n.dds not found (run from the repository root)");
  }
  be.setSkinDetail(0);
  // Rain on skin.
  std::vector<uint8_t> wetImg, puddleImg, multi;
  Backend::WetLighting wetState = wetBase;
  wetState.wetness = 1.0f;
  drawWet = &wetState;
  draw(wetImg);
  Backend::WetLighting puddleState = wetState;
  puddleState.puddleLevel = 1.0f;
  puddleState.puddleCoverage = 2.0f;
  drawWet = &puddleState;
  draw(puddleImg);
  drawWet = nullptr;
  // The multi-pass light pass.
  drawVs = mpVsKey;
  drawPs = mpPsKey;
  draw(multi);
  drawVs = vsKey;
  drawPs = psKey;

  // The model at every pixel.
  const double nmapC = 128.0 / 255.0;
  const Vec3 N = unit({ (nmapC - 0.5) * 2.0, (nmapC - 0.5) * 2.0, (1.0 - 0.5) * 2.0 });
  const double b = base / 255.0, g0 = fg0 / 255.0, g1 = fg1 / 255.0;
  const double albedo = ((g0 - 0.5) * 2.0 + b) * (g1 * 2.0) * 2.0;
  int worst = 0, wx = 0, wy = 0, flatDiff = 0, detailDiff = 0, mpWorst = 0, wetDiff = 0, poolDiff = 0;
  double expC[3] = {};
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++) {
      const Vec3 P{ (x + 0.5) / W * 2.0 - 1.0, 1.0 - (y + 0.5) / H * 2.0, 0.5 };
      const Vec3 Vv = unit(Vec3{ eye[0], eye[1], eye[2] } - P);
      for (int ch = 0; ch < 3; ch++) {
        double scaled = 0, unscaled = 0;
        const double e = 255.0 * wettest::gam(skinModel(N, Vv, wettest::lin(sun[ch]), wettest::lin(ambient[ch]),
                                                        wettest::lin(albedo), &scaled, &unscaled));
        const double mp = 255.0 * wettest::gam(scaled + unscaled / std::max(wettest::lin(b), 0.02));
        mpWorst = std::max(mpWorst, std::abs(px(multi, x, y, ch) - int(std::lround(mp))));
        wetDiff = std::max(wetDiff, std::abs(px(wetImg, x, y, ch) - px(plain, x, y, ch)));
        poolDiff = std::max(poolDiff, std::abs(px(puddleImg, x, y, ch) - px(wetImg, x, y, ch)));
        const int d = std::abs(px(plain, x, y, ch) - int(std::lround(e)));
        if (d > worst) { worst = d; wx = x; wy = y; }
        if (x == W / 2 && y == H / 2) expC[ch] = e;
        flatDiff = std::max(flatDiff, std::abs(px(flat, x, y, ch) - px(plain, x, y, ch)));
        if (haveDetail) detailDiff = std::max(detailDiff, std::abs(px(detailed, x, y, ch) - px(plain, x, y, ch)));
      }
    }
  printf("skin: centre %d %d %d (model %.1f %.1f %.1f); worst difference from the model %d at (%d, %d); a flat detail map "
         "%d apart, the shipped one %d apart\n", px(plain, W / 2, H / 2, 0), px(plain, W / 2, H / 2, 1), px(plain, W / 2, H / 2, 2),
         expC[0], expC[1], expC[2], worst, wx, wy, flatDiff, detailDiff);
  if (worst > 2) fail("skin: the skin shading is not the skin model's");
  if (flatDiff > 1) fail("skin: a flat detail map changed the shading");
  if (haveDetail && detailDiff < 3) fail("skin: the shipped detail map does not change the shading");
  printf("skin: multi-pass light pass %d %d %d at the centre, worst difference from the model %d; rain: wet skin %d apart "
         "from dry (centre %d %d %d), with full local puddles %d apart from wet\n", px(multi, W / 2, H / 2, 0),
         px(multi, W / 2, H / 2, 1), px(multi, W / 2, H / 2, 2), mpWorst, wetDiff, px(wetImg, W / 2, H / 2, 0),
         px(wetImg, W / 2, H / 2, 1), px(wetImg, W / 2, H / 2, 2), poolDiff);
  if (mpWorst > 2) fail("skin: the multi-pass light pass is not the skin model's");
  if (wetDiff < 5) fail("skin: rain does not wet skin");
  if (poolDiff > 1) fail("skin: water stands on skin");
  be.destroyTexture(0xC7080);
  be.destroyTexture(0xC70C0);
  be.destroyTexture(0xC7100);
  be.destroyTexture(0xC7140);
  be.destroyTexture(0xC7200);
  be.destroyTexture(0xC7240);
  be.destroyTexture(0xC7300);
  be.destroyBuffer(vb);
}

}  // namespace skintest
