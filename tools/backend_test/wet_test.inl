// Wet world (docs/WETWORLD.md; included by backend_test.cpp after pbr_test.inl):
// - puddles: plugin/src/puddlemap.h on terrain with known answers: a bowl whose rim has a notch spills at the notch, one
//   basin from its bottom; the ground outside the rim stays dry; a hole of unloaded cells at its bottom drains it; a
//   basin fed by a hillside: its catchment, and its water level for a rain depth (volume = depth x (its cells + runoff
//   x the rest of its catchment)), at most its brim;
// - cover: plugin/src/cascades.h overhead(): the top and bottom of the box at clip depth 0 and 1, a point's place;
// - the terrain shader (SLS2100.vso with replacements/lighting's SLS2092.pso, pbr on) under a wet state, against the
//   model computed here: a state with nothing wet draws the dry image exactly; wet ground (no cover, no puddles); a
//   rain cover over the left half keeps it dry (the dry image) and the right half wet; a basin holding deep water
//   (the puddle map and the basins' levels) gives the standing-water film; so do local puddles at a full level and a
//   coverage past the whole field, with no map;
// - the object shader (SLS2000.vso with replacements/lighting's SLS2000.pso, adts.frag) under the same states, against
//   the same model; and a wall (its frame facing -y) at the cover's edge: the air in front of it is open, so it is wet.
namespace wettest {

static void runPuddles() {
  using namespace fnvvk::puddles;
  // A 41 x 41 heightfield: a bowl (100 at the centre, rising 1 per sample) inside a rim of height 120 at distance 10 to
  // 12, with a notch of height 115 cut through it east of the centre (a channel across the rim's width); flat ground of
  // 105 outside.
  const uint32_t n = 41;
  std::vector<float> h(n * n);
  std::vector<uint8_t> valid(n * n, 1);
  for (uint32_t y = 0; y < n; y++)
    for (uint32_t x = 0; x < n; x++) {
      const double d = std::hypot(double(x) - 20.0, double(y) - 20.0);
      h[y * n + x] = d < 10.0 ? float(100.0 + d) : d <= 12.0 ? 120.0f : 105.0f;
    }
  for (uint32_t x = 30; x <= 32; x++) h[20 * n + x] = 115.0f;  // the notch: the rim's cells east of the centre
  Map m = flood(h, valid, n, n);
  const size_t centre = 20 * n + 20, inside = 20 * n + 25, outside = 2 * n + 2;
  const bool bowl = m.basins.size() == 1 && m.basin[centre] == 1 && m.basin[inside] == 1 && m.basins[0].spill == 115.0f &&
                    m.basins[0].bottom == 100.0f;
  const bool dry = m.spill[outside] == h[outside] && m.basin[outside] == 0;
  const float brim = m.basins.empty() ? 0.0f : basinLevel(m.basins[0], 1000.0f, 0.3f);
  // The same with a hole of unloaded ground at the bowl's bottom: water drains into it, so the bowl holds none.
  std::vector<uint8_t> holed = valid;
  holed[centre] = 0;
  Map mh = flood(h, holed, n, n);
  const size_t beside = 20 * n + 21;
  const bool drained = mh.spill[beside] == h[beside] && mh.spill[inside] == h[inside] && mh.basins.empty();
  // A row (the rows above and below it are walls of 100) falling from its outlet (x 0, height 5) over a lip of 6 into a
  // pit of 0 (x 4..6), then rising 1 per cell to x 19 (13): the pit and the ground below the lip (x 4..11, heights
  // 0 0 0 1 2 3 4 5) are the basin, and the hillside above it (x 12..19) drains into it: a catchment of 16. After rain
  // 1 deep with runoff 0.3 it holds 8 + 0.3 x 8 = 10.4: five cells under water at 2.68 (5 x 2.68 - (0+0+0+1+2) = 10.4).
  const uint32_t rx = 21, ry = 3;
  std::vector<float> row(rx * ry, 100.0f);
  std::vector<uint8_t> rowValid(rx * ry, 1);
  const float line[21] = { 5, 5, 5, 6, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 100 };
  for (uint32_t x = 0; x < rx; x++) row[rx + x] = line[x];
  Map mr = flood(row, rowValid, rx, ry);
  const Map::Basin* hill = mr.basins.size() == 1 ? &mr.basins[0] : nullptr;
  const float rained = hill ? basinLevel(*hill, 1.0f, 0.3f) : -1.0f;
  printf("puddles: bowl spills at %.1f (notch 115), bottom %.1f, %zu basin(s); outside spill %.1f (ground %.1f), basin %u; "
         "filled by a flood of rain: %.1f (the brim 115); with a hole at the bottom: spill beside it %.1f (ground %.1f), %zu "
         "basin(s); hillside: %zu basin(s) of %zu cells, catchment %u (expected 8, 16), level after 1 unit of rain %.3f "
         "(expected 2.680)\n", m.basins.empty() ? 0.0f : m.basins[0].spill, m.basins.empty() ? 0.0f : m.basins[0].bottom,
         m.basins.size(), m.spill[outside], h[outside], m.basin[outside], brim, mh.spill[beside], h[beside], mh.basins.size(),
         mr.basins.size(), hill ? hill->heights.size() : size_t(0), hill ? hill->catchment : 0u, rained);
  if (!bowl) fail("puddles: the bowl does not fill to its notch from its bottom");
  if (!dry) fail("puddles: ground outside the rim holds water");
  if (brim != 115.0f) fail("puddles: a basin rises above its brim");
  if (!drained) fail("puddles: a hole of unloaded ground does not drain the bowl");
  if (!hill || hill->heights.size() != 8 || hill->catchment != 16) fail("puddles: the hillside's water does not reach the basin below it");
  if (std::abs(rained - 2.68f) > 1e-4f) fail("puddles: the basin's level is not where its water's volume puts it");
}

static void runCoverMath() {
  using namespace fnvvk::cascades;
  Camera cam;
  cam.position = { 1000.0, -2000.0, 300.0 };
  cam.forward = { 0, 1, 0 };
  cam.right = { 1, 0, 0 };
  cam.up = { 0, 0, 1 };
  const float fr[4] = { -1, 1, 1, -1 };
  memcpy(cam.frustum, fr, 16);
  cam.nearZ = 5;
  cam.farZ = 100000;
  const Cascade c = overhead(cam, 4096.0, 2048, 4000.0, 4000.0);
  auto clip = [&](double wx, double wy, double wz) {  // absolute world -> clip
    const double p[3] = { wx - cam.position.x, wy - cam.position.y, wz - cam.position.z };
    double q[3];
    for (int r = 0; r < 3; r++) q[r] = c.viewProj[r * 4] * p[0] + c.viewProj[r * 4 + 1] * p[1] + c.viewProj[r * 4 + 2] * p[2] + c.viewProj[r * 4 + 3];
    return std::array<double, 3>{ q[0], q[1], q[2] };
  };
  const auto top = clip(cam.position.x, cam.position.y, cam.position.z + 4000.0);
  const auto bottom = clip(cam.position.x, cam.position.y, cam.position.z - 4000.0);
  const auto east = clip(cam.position.x + 2048.0, cam.position.y, cam.position.z);
  const double texel = 2.0 * 4096.0 / 2048.0;
  printf("cover: top of the box at clip depth %.4f, bottom %.4f; 2048 units east at clip x %.4f (expected about 0.5, within "
         "a texel of snapping: %.4f)\n", top[2], bottom[2], east[0], texel / 4096.0);
  if (std::abs(top[2]) > 1e-4 || std::abs(bottom[2] - 1.0) > 1e-4) fail("cover: the box's depth range is not 0..1 top to bottom");
  if (std::abs(east[0] - 0.5) > texel / 4096.0 + 1e-4 || std::abs(east[1]) > texel / 4096.0 + 1e-4)
    fail("cover: a point east of the camera is not where the box puts it");
  if (!inside(c, { cam.position.x, cam.position.y, cam.position.z + 3000.0 }, 10.0) ||
      inside(c, { cam.position.x + 9000.0, cam.position.y, cam.position.z }, 10.0))
    fail("cover: the caster test does not follow the box");
}

// The wet model as effectlib.cpp's FNV_WET computes it, for one pixel without ripples (rain 0).
static double lin(double c) { return std::pow(std::max(c, 0.0), 2.2); }
static double gam(double c) { return std::pow(std::max(c, 0.0), 1.0 / 2.2); }
static double reflectance(double nv, double rough, double gloss) {
  const double c0[4] = { -1.0, -0.0275, -0.572, 0.022 }, c1[4] = { 1.0, 0.0425, 1.04, -0.04 };
  double r[4];
  for (int i = 0; i < 4; i++) r[i] = rough * c0[i] + c1[i];
  const double a004 = std::min(r[0] * r[0], std::exp2(-9.28 * nv)) * r[0] + r[1];
  return (0.02 * (-1.04 * a004 + r[2]) + (1.04 * a004 + r[3])) * gloss;
}
static double ggx(const double N[3], const double V[3], const double L[3], double rough) {  // times the light's colour
  double H[3] = { L[0] + V[0], L[1] + V[1], L[2] + V[2] };
  const double hl = std::sqrt(H[0] * H[0] + H[1] * H[1] + H[2] * H[2]);
  for (double& x : H) x /= hl;
  auto dot = [](const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
  const double nl = std::clamp(dot(N, L), 0.0, 1.0), nv = std::clamp(dot(N, V), 0.0, 1.0) + 1e-4;
  const double nh = std::clamp(dot(N, H), 0.0, 1.0), vh = std::clamp(dot(V, H), 0.0, 1.0);
  const double a = rough * rough, a2 = a * a, d = nh * nh * (a2 - 1.0) + 1.0;
  const double D = a2 / (3.14159265 * d * d), k = a * 0.5;
  const double vis = 0.25 / ((nl * (1.0 - k) + k) * (nv * (1.0 - k) + k));
  return 3.14159265 * D * vis * (0.02 + 0.98 * std::pow(1.0 - vh, 5.0)) * nl;
}

static void runTerrain(Backend& be) {
  const char* dir = getenv("FNV_GAME_SHADERS");
  if (!dir || repltest::g_lightingPs.empty()) { printf("wet terrain: skipped\n"); return; }
  const ShaderKey vsKey = 0xC5000, psKey = 0xC5040;
  bool haveVs = false, havePs = false;
  for (const Md5& m : sdpShaderMd5s(dir, "SLS2100.vso", { 13 })) haveVs = haveVs || be.createShader(vsKey, m.b.data());
  for (const Md5& m : sdpShaderMd5s(dir, "SLS2092.pso", { 13 })) havePs = havePs || be.createShader(psKey, m.b.data());
  if (!haveVs || !havePs) { fail("wet terrain: SLS2100.vso / SLS2092.pso not in the shader pack"); return; }
  D3DRef fresh;
  if (!fresh.init()) { fail("wet terrain: no D3D9 device"); return; }
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
  const uint8_t grey = 128;  // the ground's albedo (gamma); a white one would not show the darkening
  if (!texture(0xC5080, grey, grey, grey) || !texture(0xC50C0, 255, 128, 128)) { fail("wet terrain: textures"); return; }
  struct V { float p[3], t[3], b[3], n[3], uv[2], c[4], w[4], w2[4]; };
  auto corner = [](float x, float y) {
    return V{ { x, y, 0.5f }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 0 }, { 1, 1, 1, 1 }, { 1, 0, 0, 1 }, { 0, 0, 0, 0 } };
  };
  const V quad[6] = { corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, -1), corner(1, 1), corner(-1, 1) };
  const BufKey vb = 0xC5100;
  be.createBuffer(vb, sizeof(quad), false);
  be.writeBuffer(vb, 0, quad, sizeof(quad), 0);
  const VertexElement ve[8] = { { 0, 0, 2, 0, 0, 0 },  { 0, 12, 2, 0, 6, 0 },  { 0, 24, 2, 0, 7, 0 }, { 0, 36, 2, 0, 3, 0 },
                                { 0, 48, 1, 0, 5, 0 }, { 0, 56, 3, 0, 10, 0 }, { 0, 72, 3, 0, 5, 1 }, { 0, 88, 3, 0, 5, 2 } };
  const DeclKey decl = 0xC5140;
  be.createDecl(decl, ve, 8);
  static float vsc[256][4], psc[224][4];
  memset(vsc, 0, sizeof(vsc));
  memset(psc, 0, sizeof(psc));
  for (int i = 0; i < 4; i++) vsc[i][i] = 1.0f;
  const float fog[4] = { 1.0e6f, 1.0f, 1.0f, 0.0f };
  memcpy(vsc[14], fog, 16);
  const float ambient[4] = { 0.1f, 0.1f, 0.1f, 1.0f }, sun[4] = { 0.8f, 0.7f, 0.6f, 1.0f }, dirUp[4] = { 0, 0, 1, 0 };
  memcpy(psc[1], ambient, 16);
  memcpy(psc[3], sun, 16);
  memcpy(psc[18], dirUp, 16);  // PSLightDir: the sun straight above (world +z)
  const uint32_t W = 32, H = 32;
  uint64_t version = 9500;
  // The camera for the wet block: forward world x, up world z, right world y, at the origin; the quad's depth 0.5 with
  // near 1, far 3 is 1.5 in front. Pixel (x, y) lies at world (1.5, 1.5 (2u - 1), 1.5 (1 - 2v)).
  Backend::WetLighting base{};
  const float rows[3][4] = { { 1, 0, 0, 0 }, { 0, 0, 1, 0 }, { 0, 1, 0, 0 } };
  memcpy(base.cameraRows, rows, sizeof(rows));
  const float frustum[4] = { -1, 1, 1, -1 }, port[4] = { 0, 1, 1, 0 };
  memcpy(base.frustum, frustum, 16);
  memcpy(base.port, port, 16);
  base.nearZ = 1.0f;
  base.farZ = 3.0f;
  base.rippleStrength = 1.0f;
  auto draw = [&](const Backend::WetLighting* w, std::vector<uint8_t>& img) {
    const ShaderReplacement::Constant params[2] = { { 223, { 1.0f, 0, 0, 0 } }, { 222, { 1.0f, 0, 0, 0 } } };
    for (const Md5& m : repltest::g_lightingPs) be.setShaderConstants(m.b.data(), params, 2);
    DrawSetup s;
    s.renderStates = rs;
    s.samplerStates = ss;
    s.textures[0] = { DrawSetup::Texture::Tex, 0xC5080 };
    s.textures[7] = { DrawSetup::Texture::Tex, 0xC50C0 };
    s.vs = vsKey;
    s.ps = psKey;
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
    be.setWetLighting(w);
    be.draw(s, 4, 2, false, 0, 0);
    be.setWetLighting(nullptr);
    be.endGroup();
    uint32_t vf = 0, ts = 0;
    if (!be.readback(rt, img, vf, ts) || ts != 4) fail("wet terrain: readback");
    be.destroy(rt);
  };
  auto px = [&](const std::vector<uint8_t>& img, uint32_t x, uint32_t y, int ch) {
    return img.size() == W * H * 4 ? int(img[(size_t(y) * W + x) * 4 + 2 - ch]) : -1;
  };
  // The expected colour of pixel (x, y) for a wet surface state: darkening, roughness, gloss, murk.
  auto expect = [&](uint32_t x, uint32_t y, double darkening, double rough, double gloss, double murk, int ch) {
    const double u = (x + 0.5) / W, v = (y + 0.5) / H;
    const double wp[3] = { 1.5, 1.5 * (2.0 * u - 1.0), 1.5 * (1.0 - 2.0 * v) };
    const double len = std::sqrt(wp[0] * wp[0] + wp[1] * wp[1] + wp[2] * wp[2]);
    const double Vv[3] = { -wp[0] / len, -wp[1] / len, -wp[2] / len }, N[3] = { 0, 0, 1 }, L[3] = { 0, 0, 1 };
    const double a = lin(grey / 255.0), aw = 0.5 * (a + std::pow(a, 1.0 + darkening));
    const double baseColour = gam(aw) * gam(lin(sun[ch]) + lin(ambient[ch]));
    const double R = reflectance(std::clamp(Vv[2], 0.0, 1.0), rough, gloss);
    const double spec = ggx(N, Vv, L, rough) * lin(sun[ch]) * gloss;
    const double tint = 1.0 + (std::array<double, 3>{ 0.62, 0.55, 0.44 }[ch] - 1.0) * murk;
    return gam(lin(baseColour) * (1.0 - R) + (lin(ambient[ch]) * R + spec) * tint);
  };
  std::vector<uint8_t> dry, idle, wet, covered, puddle, local;
  draw(nullptr, dry);
  Backend::WetLighting none = base;  // a state with nothing wet
  draw(&none, idle);
  Backend::WetLighting w = base;
  w.wetness = 1.0f;
  draw(&w, wet);

  // The rain cover: one layer whose left half holds a caster at depth 0.2; the block maps every point to depth 0.5 and
  // its world y (right) to the layer's x (q.x = y / 1.5), so the left half of the image lies under the caster.
  {
    const float v3[4][3] = { { -1, -1, 0.2f }, { 0, -1, 0.2f }, { 0, 1, 0.2f }, { -1, 1, 0.2f } };
    const uint16_t idx[6] = { 0, 1, 2, 0, 2, 3 };
    const BufKey cvb = 0xC5180, cib = 0xC51C0;
    be.createBuffer(cvb, sizeof(v3), false);
    be.writeBuffer(cvb, 0, v3, sizeof(v3), 0);
    be.createBuffer(cib, sizeof(idx), true);
    be.writeBuffer(cib, 0, idx, sizeof(idx), 0);
    const VertexElement cve[1] = { { 0, 0, 2, 0, 0, 0 } };
    const DeclKey coverDecl = 0xC5200;
    be.createDecl(coverDecl, cve, 1);
    const float identity[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    const RtKey holder = newKey();  // a group open across the shadow pass, as in game
    be.attach(holder, { 8, 8, kD3DFMT_A8R8G8B8, 1, false });
    be.beginGroup(&holder, 1, 0);
    if (!be.beginShadowMaps(1, 256)) fail("wet terrain: shadow maps");
    be.beginShadowCascade(0, identity);
    Backend::ShadowCaster sc;
    sc.streams[0] = cvb;
    sc.strides[0] = 12;
    sc.streamCount = 1;
    sc.indices = cib;
    sc.decl = coverDecl;
    sc.d3dPrimitive = 4;
    sc.primitiveCount = 2;
    const float world[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
    memcpy(sc.world, world, sizeof(world));
    be.drawShadowCaster(sc);
    be.endShadowMaps();
    be.endGroup();
    be.destroy(holder);
    Backend::WetLighting wc = w;
    wc.cover = true;
    wc.coverLayer = 0;
    const float toCover[16] = { 0, 1.0f / 1.5f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5f, 0, 0, 0, 1 };
    memcpy(wc.coverFromWorld, toCover, sizeof(toCover));
    wc.coverBias = 0.0f;
    wc.coverTexel = 1.0f / 256.0f;
    draw(&wc, covered);
    be.destroyBuffer(cvb);
    be.destroyBuffer(cib);
  }

  // Standing water everywhere from a basin: a 2 x 2 puddle map over world x 0..4, y -2..2 whose texels are basin 1 with
  // the ground at z -10, basin 1's level at 20 (the level table): water 18.5 to 21.5 deep over the image.
  {
    const TexKey mapKey = 0xC5240, levelsKey = 0xC5280;
    const float texels[4][4] = { { 1, -10, 0, 0 }, { 1, -10, 0, 0 }, { 1, -10, 0, 0 }, { 1, -10, 0, 0 } };
    float levels[256] = {};
    levels[1] = 20.0f;
    if (!be.updateTexture2D(mapKey, kD3DFMT_A32B32G32R32F, 2, 2, texels, 32)) fail("wet terrain: the puddle map texture");
    be.uploadTexture2D(levelsKey, kD3DFMT_R32F, 256, 1, levels, 1024);
    Backend::WetLighting wp = w;
    wp.puddleMap = mapKey;
    wp.puddleLevels = levelsKey;
    wp.basinWater = true;
    wp.puddleOrigin[0] = 0.0f;
    wp.puddleOrigin[1] = -2.0f;
    wp.puddleSpacing = 4.0f;
    wp.puddleSize[0] = wp.puddleSize[1] = 2;
    draw(&wp, puddle);
    be.destroyTexture(mapKey);
    be.destroyTexture(levelsKey);
  }
  // Local puddles, no map: a full level with a coverage of 2 puts the threshold under the whole field (1 - 2 = -1), so
  // every flat open spot stands under water, deep (the field is at least 1 over the threshold, 5 x its softness).
  Backend::WetLighting wl = w;
  wl.puddleLevel = 1.0f;
  wl.puddleCoverage = 2.0f;
  draw(&wl, local);
  be.destroyTexture(0xC5080);
  be.destroyTexture(0xC50C0);

  const bool idleSame = !dry.empty() && dry == idle;
  int errWet = 0, errCoveredDry = 0, errCoveredWet = 0, errPuddle = 0, errLocal = 0;
  const uint32_t y = 28;  // a row where the view looks down at the ground (N.V about 0.6)
  for (int ch = 0; ch < 3; ch++) {
    errWet = std::max(errWet, std::abs(px(wet, 16, y, ch) - int(std::lround(255.0 * expect(16, y, 0.7, 0.22, 1.0, 0.0, ch)))));
    errCoveredDry = std::max(errCoveredDry, std::abs(px(covered, 6, y, ch) - px(dry, 6, y, ch)));
    errCoveredWet = std::max(errCoveredWet, std::abs(px(covered, 26, y, ch) - int(std::lround(255.0 * expect(26, y, 0.7, 0.22, 1.0, 0.0, ch)))));
    errPuddle = std::max(errPuddle, std::abs(px(puddle, 16, y, ch) - int(std::lround(255.0 * expect(16, y, 1.6, 0.02, 1.0, 1.0, ch)))));
    errLocal = std::max(errLocal, std::abs(px(local, 16, y, ch) - int(std::lround(255.0 * expect(16, y, 1.6, 0.02, 1.0, 1.0, ch)))));
  }
  printf("wet terrain: nothing wet: %s the dry image; wet ground at (16, %u) %d %d %d (expected %.0f %.0f %.0f, dry %d %d %d); "
         "under cover %d %d %d (dry %d %d %d), beside it %d %d %d; standing water %d %d %d (expected %.0f %.0f %.0f); worst "
         "differences: wet %d, covered %d, beside %d, standing (a basin) %d, local puddles %d\n",
         idleSame ? "same as" : "DIFFERENT from", y, px(wet, 16, y, 0), px(wet, 16, y, 1), px(wet, 16, y, 2),
         255.0 * expect(16, y, 0.7, 0.22, 1.0, 0.0, 0), 255.0 * expect(16, y, 0.7, 0.22, 1.0, 0.0, 1),
         255.0 * expect(16, y, 0.7, 0.22, 1.0, 0.0, 2), px(dry, 16, y, 0), px(dry, 16, y, 1), px(dry, 16, y, 2),
         px(covered, 6, y, 0), px(covered, 6, y, 1), px(covered, 6, y, 2), px(dry, 6, y, 0), px(dry, 6, y, 1), px(dry, 6, y, 2),
         px(covered, 26, y, 0), px(covered, 26, y, 1), px(covered, 26, y, 2), px(puddle, 16, y, 0), px(puddle, 16, y, 1),
         px(puddle, 16, y, 2), 255.0 * expect(16, y, 1.6, 0.02, 1.0, 1.0, 0), 255.0 * expect(16, y, 1.6, 0.02, 1.0, 1.0, 1),
         255.0 * expect(16, y, 1.6, 0.02, 1.0, 1.0, 2), errWet, errCoveredDry, errCoveredWet, errPuddle, errLocal);
  if (!idleSame) fail("wet terrain: a wet state with nothing wet changed the image");
  if (errWet > 2) fail("wet terrain: wet ground is not the wet model's");
  if (errCoveredDry > 1) fail("wet terrain: ground under the rain cover got wet");
  if (errCoveredWet > 2) fail("wet terrain: ground beside the rain cover is not wet");
  if (errPuddle > 2) fail("wet terrain: a basin's standing water is not the water film's");
  if (errLocal > 2) fail("wet terrain: local puddles are not the water film's");
}

static void runObjects(Backend& be) {
  const char* dir = getenv("FNV_GAME_SHADERS");
  if (!dir || repltest::g_lightingPs.empty()) { printf("wet objects: skipped\n"); return; }
  const ShaderKey vsKey = 0xC6000, psKey = 0xC6040;
  bool haveVs = false, havePs = false;
  for (const Md5& m : sdpShaderMd5s(dir, "SLS2000.vso", { 13 })) haveVs = haveVs || be.createShader(vsKey, m.b.data());
  for (const Md5& m : sdpShaderMd5s(dir, "SLS2000.pso", { 13 })) havePs = havePs || be.createShader(psKey, m.b.data());
  if (!haveVs || !havePs) { fail("wet objects: SLS2000.vso / SLS2000.pso not in the shader pack"); return; }
  D3DRef fresh;
  if (!fresh.init()) { fail("wet objects: no D3D9 device"); return; }
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
  const uint8_t grey = 128;  // the ground's albedo (gamma); a white one would not show the darkening
  if (!texture(0xC6080, grey, grey, grey) || !texture(0xC60C0, 255, 128, 128)) { fail("wet objects: textures"); return; }
  // position, tangent, binormal, normal, texcoord, colour (pbr_test.inl's layout). The quad faces up (+z); the wall is the
  // same quad with its frame turned to face world -y (its tangent x, binormal z, normal -y).
  struct V { float p[3], t[3], b[3], n[3], uv[2], c[4]; };
  auto corner = [](float x, float y, bool wall) {
    return wall ? V{ { x, y, 0.5f }, { 1, 0, 0 }, { 0, 0, 1 }, { 0, -1, 0 }, { 0, 0 }, { 1, 1, 1, 1 } }
                : V{ { x, y, 0.5f }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 0 }, { 1, 1, 1, 1 } };
  };
  const BufKey vb = 0xC6100, wallVb = 0xC6110;
  for (int k = 0; k < 2; k++) {
    const V quad[6] = { corner(-1, -1, k), corner(1, -1, k), corner(1, 1, k), corner(-1, -1, k), corner(1, 1, k), corner(-1, 1, k) };
    be.createBuffer(k ? wallVb : vb, sizeof(quad), false);
    be.writeBuffer(k ? wallVb : vb, 0, quad, sizeof(quad), 0);
  }
  const VertexElement ve[6] = { { 0, 0, 2, 0, 0, 0 }, { 0, 12, 2, 0, 6, 0 }, { 0, 24, 2, 0, 7, 0 },
                                { 0, 36, 2, 0, 3, 0 }, { 0, 48, 1, 0, 5, 0 }, { 0, 56, 3, 0, 10, 0 } };
  const DeclKey decl = 0xC6140;
  be.createDecl(decl, ve, 6);
  BufKey drawVb = vb;
  static float vsc[256][4], psc[224][4];
  memset(vsc, 0, sizeof(vsc));
  memset(psc, 0, sizeof(psc));
  for (int i = 0; i < 4; i++) vsc[i][i] = 1.0f;
  const float fog[4] = { 1.0e6f, 1.0f, 1.0f, 0.0f };
  memcpy(vsc[14], fog, 16);
  const float ambient[4] = { 0.1f, 0.1f, 0.1f, 1.0f }, sun[4] = { 0.8f, 0.7f, 0.6f, 1.0f }, dirUp[4] = { 0, 0, 1, 0 };
  memcpy(psc[1], ambient, 16);
  memcpy(psc[3], sun, 16);
  (void)dirUp;
  const float light[4] = { 0, 0, 1, 1 }, eye[4] = { 0, 0, 5, 1 }, toggles[4] = { 0, 0, 30, 0 };
  memcpy(vsc[25], light, 16);  // LightData: the sun along +z (object space = world here), no specular fade
  memcpy(vsc[16], eye, 16);
  memcpy(psc[27], toggles, 16);
  for (int r = 0; r < 3; r++) vsc[250 + r][r] = 1.0f;  // the world rows (docs/SHADER_ABI.md "Tangent-to-world frame"): identity
  const uint32_t W = 32, H = 32;
  uint64_t version = 9500;
  // The camera for the wet block: forward world x, up world z, right world y, at the origin; the quad's depth 0.5 with
  // near 1, far 3 is 1.5 in front. Pixel (x, y) lies at world (1.5, 1.5 (2u - 1), 1.5 (1 - 2v)).
  Backend::WetLighting base{};
  const float rows[3][4] = { { 1, 0, 0, 0 }, { 0, 0, 1, 0 }, { 0, 1, 0, 0 } };
  memcpy(base.cameraRows, rows, sizeof(rows));
  const float frustum[4] = { -1, 1, 1, -1 }, port[4] = { 0, 1, 1, 0 };
  memcpy(base.frustum, frustum, 16);
  memcpy(base.port, port, 16);
  base.nearZ = 1.0f;
  base.farZ = 3.0f;
  base.rippleStrength = 1.0f;
  auto draw = [&](const Backend::WetLighting* w, std::vector<uint8_t>& img) {
    const ShaderReplacement::Constant params[2] = { { 223, { 1.0f, 0, 0, 0 } }, { 222, { 1.0f, 0, 0, 0 } } };
    for (const Md5& m : repltest::g_lightingPs) be.setShaderConstants(m.b.data(), params, 2);
    DrawSetup s;
    s.renderStates = rs;
    s.samplerStates = ss;
    s.textures[0] = { DrawSetup::Texture::Tex, 0xC6080 };
    s.textures[1] = { DrawSetup::Texture::Tex, 0xC60C0 };
    s.vs = vsKey;
    s.ps = psKey;
    s.decl = decl;
    s.streams[0] = { drawVb, 0, sizeof(V) };
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
    be.setWetLighting(w);
    be.draw(s, 4, 2, false, 0, 0);
    be.setWetLighting(nullptr);
    be.endGroup();
    uint32_t vf = 0, ts = 0;
    if (!be.readback(rt, img, vf, ts) || ts != 4) fail("wet objects: readback");
    be.destroy(rt);
  };
  auto px = [&](const std::vector<uint8_t>& img, uint32_t x, uint32_t y, int ch) {
    return img.size() == W * H * 4 ? int(img[(size_t(y) * W + x) * 4 + 2 - ch]) : -1;
  };
  // The expected colour of pixel (x, y) for a wet surface state: darkening, roughness, gloss, murk.
  auto expect = [&](uint32_t x, uint32_t y, double darkening, double rough, double gloss, double murk, int ch) {
    const double u = (x + 0.5) / W, v = (y + 0.5) / H;
    const double wp[3] = { 1.5, 1.5 * (2.0 * u - 1.0), 1.5 * (1.0 - 2.0 * v) };
    const double len = std::sqrt(wp[0] * wp[0] + wp[1] * wp[1] + wp[2] * wp[2]);
    const double Vv[3] = { -wp[0] / len, -wp[1] / len, -wp[2] / len }, N[3] = { 0, 0, 1 }, L[3] = { 0, 0, 1 };
    const double a = lin(grey / 255.0), aw = 0.5 * (a + std::pow(a, 1.0 + darkening));
    const double baseColour = gam(aw) * gam(lin(sun[ch]) + lin(ambient[ch]));
    const double R = reflectance(std::clamp(Vv[2], 0.0, 1.0), rough, gloss);
    const double spec = ggx(N, Vv, L, rough) * lin(sun[ch]) * gloss;
    const double tint = 1.0 + (std::array<double, 3>{ 0.62, 0.55, 0.44 }[ch] - 1.0) * murk;
    return gam(lin(baseColour) * (1.0 - R) + (lin(ambient[ch]) * R + spec) * tint);
  };
  std::vector<uint8_t> dry, idle, wet, covered, puddle, local, wallCovered, wallOpen, wallDry;
  draw(nullptr, dry);
  Backend::WetLighting none = base;  // a state with nothing wet
  draw(&none, idle);
  Backend::WetLighting w = base;
  w.wetness = 1.0f;
  draw(&w, wet);

  // The rain cover: one layer whose left half holds a caster at depth 0.2; the block maps every point to depth 0.5 and
  // its world y (right) to the layer's x (q.x = y / 1.5), so the left half of the image lies under the caster.
  {
    const float v3[4][3] = { { -1, -1, 0.2f }, { 0, -1, 0.2f }, { 0, 1, 0.2f }, { -1, 1, 0.2f } };
    const uint16_t idx[6] = { 0, 1, 2, 0, 2, 3 };
    const BufKey cvb = 0xC6180, cib = 0xC61C0;
    be.createBuffer(cvb, sizeof(v3), false);
    be.writeBuffer(cvb, 0, v3, sizeof(v3), 0);
    be.createBuffer(cib, sizeof(idx), true);
    be.writeBuffer(cib, 0, idx, sizeof(idx), 0);
    const VertexElement cve[1] = { { 0, 0, 2, 0, 0, 0 } };
    const DeclKey coverDecl = 0xC6200;
    be.createDecl(coverDecl, cve, 1);
    const float identity[16] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    const RtKey holder = newKey();  // a group open across the shadow pass, as in game
    be.attach(holder, { 8, 8, kD3DFMT_A8R8G8B8, 1, false });
    be.beginGroup(&holder, 1, 0);
    if (!be.beginShadowMaps(1, 256)) fail("wet objects: shadow maps");
    be.beginShadowCascade(0, identity);
    Backend::ShadowCaster sc;
    sc.streams[0] = cvb;
    sc.strides[0] = 12;
    sc.streamCount = 1;
    sc.indices = cib;
    sc.decl = coverDecl;
    sc.d3dPrimitive = 4;
    sc.primitiveCount = 2;
    const float world[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
    memcpy(sc.world, world, sizeof(world));
    be.drawShadowCaster(sc);
    be.endShadowMaps();
    be.endGroup();
    be.destroy(holder);
    Backend::WetLighting wc = w;
    wc.cover = true;
    wc.coverLayer = 0;
    const float toCover[16] = { 0, 1.0f / 1.5f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.5f, 0, 0, 0, 1 };
    memcpy(wc.coverFromWorld, toCover, sizeof(toCover));
    wc.coverBias = 0.0f;
    wc.coverTexel = 1.0f / 256.0f;
    draw(&wc, covered);
    // The wall (lit by a brighter ambient, the sun along it): the air in front of it (32 units toward -y) lies outside the
    // cover, so under the cover it is exactly as wet as with no cover at all.
    drawVb = wallVb;
    const float bright[4] = { 0.6f, 0.6f, 0.6f, 1.0f };
    memcpy(psc[1], bright, 16);
    draw(&wc, wallCovered);
    draw(&w, wallOpen);
    draw(nullptr, wallDry);
    memcpy(psc[1], ambient, 16);
    drawVb = vb;
    be.destroyBuffer(cvb);
    be.destroyBuffer(cib);
  }

  // Standing water everywhere from a basin: a 2 x 2 puddle map over world x 0..4, y -2..2 whose texels are basin 1 with
  // the ground at z -10, basin 1's level at 20 (the level table): water 18.5 to 21.5 deep over the image.
  {
    const TexKey mapKey = 0xC6240, levelsKey = 0xC6280;
    const float texels[4][4] = { { 1, -10, 0, 0 }, { 1, -10, 0, 0 }, { 1, -10, 0, 0 }, { 1, -10, 0, 0 } };
    float levels[256] = {};
    levels[1] = 20.0f;
    if (!be.updateTexture2D(mapKey, kD3DFMT_A32B32G32R32F, 2, 2, texels, 32)) fail("wet objects: the puddle map texture");
    be.uploadTexture2D(levelsKey, kD3DFMT_R32F, 256, 1, levels, 1024);
    Backend::WetLighting wp = w;
    wp.puddleMap = mapKey;
    wp.puddleLevels = levelsKey;
    wp.basinWater = true;
    wp.puddleOrigin[0] = 0.0f;
    wp.puddleOrigin[1] = -2.0f;
    wp.puddleSpacing = 4.0f;
    wp.puddleSize[0] = wp.puddleSize[1] = 2;
    draw(&wp, puddle);
    be.destroyTexture(mapKey);
    be.destroyTexture(levelsKey);
  }
  // Local puddles, no map: a full level with a coverage of 2 puts the threshold under the whole field (1 - 2 = -1), so
  // every flat open spot stands under water, deep (the field is at least 1 over the threshold, 5 x its softness).
  Backend::WetLighting wl = w;
  wl.puddleLevel = 1.0f;
  wl.puddleCoverage = 2.0f;
  draw(&wl, local);
  be.destroyBuffer(vb);
  be.destroyBuffer(wallVb);
  be.destroyTexture(0xC6080);
  be.destroyTexture(0xC60C0);

  const bool idleSame = !dry.empty() && dry == idle;
  int errWet = 0, errCoveredDry = 0, errCoveredWet = 0, errPuddle = 0, errLocal = 0;
  const uint32_t y = 28;  // a row where the view looks down at the ground (N.V about 0.6)
  for (int ch = 0; ch < 3; ch++) {
    errWet = std::max(errWet, std::abs(px(wet, 16, y, ch) - int(std::lround(255.0 * expect(16, y, 0.7, 0.22, 1.0, 0.0, ch)))));
    errCoveredDry = std::max(errCoveredDry, std::abs(px(covered, 6, y, ch) - px(dry, 6, y, ch)));
    errCoveredWet = std::max(errCoveredWet, std::abs(px(covered, 26, y, ch) - int(std::lround(255.0 * expect(26, y, 0.7, 0.22, 1.0, 0.0, ch)))));
    errPuddle = std::max(errPuddle, std::abs(px(puddle, 16, y, ch) - int(std::lround(255.0 * expect(16, y, 1.6, 0.02, 1.0, 1.0, ch)))));
    errLocal = std::max(errLocal, std::abs(px(local, 16, y, ch) - int(std::lround(255.0 * expect(16, y, 1.6, 0.02, 1.0, 1.0, ch)))));
  }
  printf("wet objects: nothing wet: %s the dry image; wet ground at (16, %u) %d %d %d (expected %.0f %.0f %.0f, dry %d %d %d); "
         "under cover %d %d %d (dry %d %d %d), beside it %d %d %d; standing water %d %d %d (expected %.0f %.0f %.0f); worst "
         "differences: wet %d, covered %d, beside %d, standing (a basin) %d, local puddles %d\n",
         idleSame ? "same as" : "DIFFERENT from", y, px(wet, 16, y, 0), px(wet, 16, y, 1), px(wet, 16, y, 2),
         255.0 * expect(16, y, 0.7, 0.22, 1.0, 0.0, 0), 255.0 * expect(16, y, 0.7, 0.22, 1.0, 0.0, 1),
         255.0 * expect(16, y, 0.7, 0.22, 1.0, 0.0, 2), px(dry, 16, y, 0), px(dry, 16, y, 1), px(dry, 16, y, 2),
         px(covered, 6, y, 0), px(covered, 6, y, 1), px(covered, 6, y, 2), px(dry, 6, y, 0), px(dry, 6, y, 1), px(dry, 6, y, 2),
         px(covered, 26, y, 0), px(covered, 26, y, 1), px(covered, 26, y, 2), px(puddle, 16, y, 0), px(puddle, 16, y, 1),
         px(puddle, 16, y, 2), 255.0 * expect(16, y, 1.6, 0.02, 1.0, 1.0, 0), 255.0 * expect(16, y, 1.6, 0.02, 1.0, 1.0, 1),
         255.0 * expect(16, y, 1.6, 0.02, 1.0, 1.0, 2), errWet, errCoveredDry, errCoveredWet, errPuddle, errLocal);
  if (!idleSame) fail("wet objects: a wet state with nothing wet changed the image");
  if (errWet > 2) fail("wet objects: wet ground is not the wet model's");
  if (errCoveredDry > 1) fail("wet objects: ground under the rain cover got wet");
  if (errCoveredWet > 2) fail("wet objects: ground beside the rain cover is not wet");
  if (errPuddle > 2) fail("wet objects: a basin's standing water is not the water film's");
  if (errLocal > 2) fail("wet objects: local puddles are not the water film's");
  int wallWet = 0, wallSame = 0;
  for (int ch = 0; ch < 3; ch++) {
    wallWet = std::max(wallWet, std::abs(px(wallOpen, 6, y, ch) - px(wallDry, 6, y, ch)));
    wallSame = std::max(wallSame, std::abs(px(wallCovered, 6, y, ch) - px(wallOpen, 6, y, ch)));
  }
  printf("wet objects: a wall at the cover's edge %d %d %d; with no cover %d %d %d; dry %d %d %d (wet vs dry %d, covered vs "
         "open %d)\n", px(wallCovered, 6, y, 0), px(wallCovered, 6, y, 1), px(wallCovered, 6, y, 2), px(wallOpen, 6, y, 0),
         px(wallOpen, 6, y, 1), px(wallOpen, 6, y, 2), px(wallDry, 6, y, 0), px(wallDry, 6, y, 1), px(wallDry, 6, y, 2), wallWet,
         wallSame);
  if (wallWet < 5) fail("wet objects: a wall in the rain is not wet");
  if (wallSame > 1) fail("wet objects: a wall open to the rain beside a cover is not as wet as in the open");
}

static void run(Backend& be) {
  runPuddles();
  runCoverMath();
  runTerrain(be);
  runObjects(be);
}

}  // namespace wettest
