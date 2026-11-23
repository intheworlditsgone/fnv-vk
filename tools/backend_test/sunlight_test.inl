// Sun shadows in the game's lighting (included by backend_test.cpp): a real shadow pass, then the game's SLS1034.pso
// (out = DiffuseMap * (COLOR0 * PSLightColor[0] + AmbientColor), sun-lit by sdp2spv's sun shadow pass) drawn over
// the target with SKYQUADSI.vso (clip-space positions as given, depth 0.5), through the backend's own draw path.
//
// The scene is the one tools/shaderdiff's splitx test checks against D3D9: camera frustum tangents (-1, 1, 1, -1), near 1,
// far 2 (depth 0.5 is view distance 4/3); one cascade with clip x = 0.6 * view x, depth = 0.25 * z + 0.25; a caster
// quad over view x < 0 at z = 0. So the left columns are in shadow and the right ones lit, a band of 6 columns in the
// middle mixed by the filter. Checks: no lighting state and a state whose gate says no give the same image; with the
// gate open the shadowed side loses the sun's light (strength 1: black with no ambient; 0.5: half) and the lit side is
// unchanged; the backend counts exactly the shadowed draws; validation stays clean.
// FNV_GAME_SHADERS = the game's Data\Shaders folder (the test is skipped without it); the shader pack must hold
// package 013's SKYQUADSI.vso and SLS1034.pso.
namespace suntest {

static void run(Backend& be) {
  const char* dir = getenv("FNV_GAME_SHADERS");
  if (!dir) { printf("sun lighting: FNV_GAME_SHADERS not set: skipped\n"); return; }
  const std::vector<Md5> vs = sdpShaderMd5s(dir, "SKYQUADSI.vso"), ps = sdpShaderMd5s(dir, "SLS1034.pso");
  const ShaderKey vsKey = 0xC1000, psKey = 0xC1040;
  bool haveVs = false, havePs = false;
  for (const Md5& m : vs) haveVs = haveVs || be.createShader(vsKey, m.b.data());
  for (const Md5& m : ps) havePs = havePs || be.createShader(psKey, m.b.data());
  if (!haveVs || !havePs) { fail("sun lighting: SKYQUADSI.vso / SLS1034.pso not in the shader pack"); return; }

  D3DRef fresh;
  if (!fresh.init()) { fail("sun lighting: no D3D9 device for default states"); return; }
  static uint32_t rs[256], ss[16][14];
  for (uint32_t i = 0; i < 256; i++) { DWORD v = 0; fresh.dev->GetRenderState(D3DRENDERSTATETYPE(i), &v); rs[i] = v; }
  for (uint32_t s = 0; s < 16; s++)
    for (uint32_t t = 0; t < 14; t++) { DWORD v = 0; fresh.dev->GetSamplerState(s, D3DSAMPLERSTATETYPE(t), &v); ss[s][t] = v; }
  rs[D3DRS_ZENABLE] = 0;
  rs[D3DRS_CULLMODE] = D3DCULL_NONE;

  {
    d3dx::Texture t;
    t.format = 21;
    t.width = t.height = 4;
    t.levels = 1;
    t.cube = false;
    d3dx::Level l;
    l.width = l.height = 4;
    l.rowPitch = 16;
    l.rows = 4;
    l.data.assign(64, 255);
    t.surfaces.push_back(std::move(l));
    if (!be.createTexture(0xC1080, t)) { fail("sun lighting: texture"); return; }
  }
  // The receiver: the whole target at depth 0.5 (SKYQUADSI: oPos = (x, y, z, 1)).
  const float quad[6][3] = { { -1, -1, 0.5f }, { 1, -1, 0.5f }, { 1, 1, 0.5f }, { -1, -1, 0.5f }, { 1, 1, 0.5f }, { -1, 1, 0.5f } };
  // The caster: view x in [-10, 0], y in [-10, 10], z = 0 (world = view space here).
  const float caster[6][3] = { { -10, -10, 0 }, { 0, -10, 0 }, { 0, 10, 0 }, { -10, -10, 0 }, { 0, 10, 0 }, { -10, 10, 0 } };
  const BufKey vbQuad = 0xC10C0, vbCaster = 0xC1100;
  be.createBuffer(vbQuad, sizeof(quad), false);
  be.writeBuffer(vbQuad, 0, quad, sizeof(quad), 0);
  be.createBuffer(vbCaster, sizeof(caster), false);
  be.writeBuffer(vbCaster, 0, caster, sizeof(caster), 0);
  const VertexElement ve[1] = { { 0, 0, 2 /*FLOAT3*/, 0, 0 /*POSITION*/, 0 } };
  const DeclKey decl = 0xC1140;
  be.createDecl(decl, ve, 1);

  const float cascade[16] = { 0.6f, 0, 0, 0, 0, 0.6f, 0, 0, 0, 0, 0.25f, 0.25f, 0, 0, 0, 1 };
  if (!be.beginShadowMaps(1, 256)) { fail("sun lighting: beginShadowMaps"); return; }
  be.beginShadowCascade(0, cascade);
  Backend::ShadowCaster c;
  c.streams[0] = vbCaster;
  c.strides[0] = 12;
  c.streamCount = 1;
  c.decl = decl;
  c.d3dPrimitive = 4;
  c.primitiveCount = 2;
  const float identity[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
  memcpy(c.world, identity, sizeof(identity));
  be.drawShadowCaster(c);
  be.endShadowMaps();

  static float vsc[256][4], psc[224][4];
  memset(vsc, 0, sizeof(vsc));
  memset(psc, 0, sizeof(psc));
  const float sunColor[4] = { 0.8f, 0.6f, 0.4f, 1.0f };
  memcpy(psc[3], sunColor, 16);  // PSLightColor[0]; AmbientColor (c1) 0

  bool gateOpen = false;
  Backend::SunLighting sun;
  const float frustum[4] = { -1, 1, 1, -1 }, port[4] = { 0, 1, 1, 0 };
  memcpy(sun.frustum, frustum, 16);
  memcpy(sun.port, port, 16);
  sun.nearZ = 1.0f;
  sun.farZ = 2.0f;
  sun.cascades = 1;
  memcpy(sun.shadowFromView[0], cascade, 64);
  sun.splits[0] = 1.0e6f;
  sun.gate = [](void* ctx) { return *static_cast<bool*>(ctx); };
  sun.ctx = &gateOpen;

  const uint32_t W = 64, H = 64;
  uint64_t version = 7000;
  auto draw = [&](std::vector<uint8_t>& img) {
    DrawSetup s;
    s.renderStates = rs;
    s.samplerStates = ss;
    s.textures[0] = { DrawSetup::Texture::Tex, 0xC1080 };
    s.vs = vsKey;
    s.ps = psKey;
    s.decl = decl;
    s.streams[0] = { vbQuad, 0, 12 };
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
    be.draw(s, 4 /*D3DPT_TRIANGLELIST*/, 2, false, 0, 0);
    be.endGroup();
    uint32_t vf = 0, ts = 0;
    if (!be.readback(rt, img, vf, ts) || ts != 4) fail("sun lighting: readback");
    be.destroy(rt);
  };
  auto shadowedDraws = [&]() { return be.stats().sunShadowedDraws; };

  std::vector<uint8_t> base, closed, open, half;
  be.setSunLighting(nullptr);
  draw(base);
  const uint64_t n0 = shadowedDraws();
  be.setSunLighting(&sun);
  draw(closed);
  const uint64_t n1 = shadowedDraws();
  gateOpen = true;
  draw(open);
  const uint64_t n2 = shadowedDraws();
  sun.strength = 0.5f;
  be.setSunLighting(&sun);
  draw(half);
  const uint64_t n3 = shadowedDraws();
  be.setSunLighting(nullptr);
  if (base.size() != W * H * 4 || closed.size() != base.size() || open.size() != base.size() || half.size() != base.size()) {
    fail("sun lighting: image sizes");
    return;
  }
  if (n1 != n0 || n2 != n1 + 1 || n3 != n2 + 1)
    fail("sun lighting: shadowed draws counted %llu, %llu, %llu (expected +0, +1, +1)", (unsigned long long)(n1 - n0),
         (unsigned long long)(n2 - n1), (unsigned long long)(n3 - n2));
  // Base: the sun colour everywhere (BGRA bytes 0.4, 0.6, 0.8 * 255).
  const uint8_t* b0 = &base[(32 * W + 32) * 4];
  printf("sun lighting: unshadowed pixel BGRA %u %u %u %u\n", b0[0], b0[1], b0[2], b0[3]);
  if (std::abs(int(b0[2]) - 204) > 2 || std::abs(int(b0[1]) - 153) > 2 || std::abs(int(b0[0]) - 102) > 2)
    fail("sun lighting: unshadowed colour %u %u %u, expected 102 153 204", b0[0], b0[1], b0[2]);
  uint32_t badClosed = 0, badLit = 0, badDark = 0, badHalf = 0;
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++) {
      const size_t i = (size_t(y) * W + x) * 4;
      for (int ch = 0; ch < 3; ch++) {
        const int b = base[i + ch];
        if (closed[i + ch] != base[i + ch]) badClosed++;
        if (x > 34) {
          if (open[i + ch] != b) badLit++;
          if (half[i + ch] != b) badLit++;
        } else if (x < 29) {
          if (open[i + ch] > 1) badDark++;
          if (std::abs(int(half[i + ch]) - (b + 1) / 2) > 1) badHalf++;
        }
      }
    }
  printf("sun lighting: gate closed %u differing channels; open: lit side %u, shadowed side %u, half strength %u off\n",
         badClosed, badLit, badDark, badHalf);
  if (badClosed || badLit || badDark || badHalf) fail("sun lighting: forward sun shadows wrong");
  if (FILE* f = fopen("build/sunlight_open.bgra", "wb")) {  // raw 64x64 BGRA, for inspection
    fwrite(open.data(), 1, open.size(), f);
    fclose(f);
  }

  // The cascade blend: the next cascade's filter is computed only where it is blended in (sdp2spv's sun pass branches
  // on blend > 0; every check above has blend 0). Two cascades with the same projection, the caster drawn only into
  // cascade 0, and cascade 0's split placed so that the receiver's lookup point (view distance 4/3, moved 1 + 0.002 z
  // toward the camera along the normal) lies in the middle of its blend band (the last 20% of [0, split]): blend 0.5.
  // The shadowed side then gets half the sun's light (cascade 1 sees no caster), the lit side all of it; with the next
  // cascade never sampled it would stay black.
  {
    const double z = 4.0 / 3.0, qz = z - (1.0 + 0.002 * z);
    if (!be.beginShadowMaps(2, 256)) { fail("sun lighting: beginShadowMaps(2)"); return; }
    be.beginShadowCascade(0, cascade);
    be.drawShadowCaster(c);
    be.beginShadowCascade(1, cascade);
    be.endShadowMaps();
    sun.cascades = 2;
    sun.strength = 1.0f;
    memcpy(sun.shadowFromView[1], cascade, 64);
    sun.splits[0] = float(qz / 0.9);
    sun.splits[1] = 1.0e6f;
    gateOpen = true;
    be.setSunLighting(&sun);
    std::vector<uint8_t> blended;
    draw(blended);
    be.setSunLighting(nullptr);
    uint32_t badBlendLit = 0, badBlendDark = 0;
    if (blended.size() == base.size())
      for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x++)
          for (int ch = 0; ch < 3; ch++) {
            const size_t i = (size_t(y) * W + x) * 4 + ch;
            if (x > 34 && blended[i] != base[i]) badBlendLit++;
            if (x < 29 && std::abs(int(blended[i]) - (int(base[i]) + 1) / 2) > 1) badBlendDark++;
          }
    const uint8_t* d0 = blended.size() == base.size() ? &blended[(32 * W + 8) * 4] : b0;
    printf("sun lighting: in a cascade blend band (blend 0.5): shadowed side BGRA %u %u %u (half the sun), lit side %u off, "
           "shadowed side %u off\n", d0[0], d0[1], d0[2], badBlendLit, badBlendDark);
    if (blended.size() != base.size() || badBlendLit || badBlendDark) fail("sun lighting: cascade blend wrong");
  }
  be.destroyTexture(0xC1080);
}

}  // namespace suntest
