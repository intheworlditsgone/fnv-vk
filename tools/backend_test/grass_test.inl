// Grass (docs/LIGHTING.md "Grass"; included by backend_test.cpp after sss_test.inl): the game's GRASS2001.vso (lit,
// with mesh normals; translated with sdp2spv's grass pass) with replacements/lighting's GRASS2000.pso (grass.frag,
// physically based) through the backend. One instance (index 0 at the origin, occlusion frac(0.6) x 0.75 + 0.25 = 0.7),
// no wind, no fade, no fog, ModelViewProj the identity: a quad at z 0.5 facing +z whose mesh normal is +z, the sun
// along (0.3, 0.2, 0.9). Against Community Shaders' grass model computed here from the vertex shader's inputs:
// - every pixel within 2 of 255, with the canopy height (c250.x) 1 (the quad at canopy fraction 0.5) and 4 (0.125);
// - the sun behind the blade (along -z): the light through it (the transmission lobe), within 2;
// - the canopy height changes the image (its constant reaches the shader).
namespace grasstest {

static void run(Backend& be) {
  const char* dir = getenv("FNV_GAME_SHADERS");
  if (!dir || repltest::g_lightingPs.empty()) { printf("grass: skipped\n"); return; }
  const ShaderKey vsKey = 0xC9000, psKey = 0xC9040;
  bool haveVs = false, havePs = false;
  for (const Md5& m : sdpShaderMd5s(dir, "GRASS2001.vso", { 13 })) haveVs = haveVs || be.createShader(vsKey, m.b.data());
  for (const Md5& m : sdpShaderMd5s(dir, "GRASS2000.pso", { 13 })) havePs = havePs || be.createShader(psKey, m.b.data());
  if (!haveVs || !havePs) { fail("grass: GRASS2001.vso / GRASS2000.pso not in the shader pack"); return; }
  D3DRef fresh;
  if (!fresh.init()) { fail("grass: no D3D9 device"); return; }
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
    d3dx::Level l;
    l.width = l.height = 4;
    l.rowPitch = 16;
    l.rows = 4;
    for (int i = 0; i < 16; i++) l.data.insert(l.data.end(), { 160, 160, 160, 255 });
    t.surfaces.push_back(std::move(l));
    if (!be.createTexture(0xC9080, t)) { fail("grass: texture"); return; }
  }
  // position, normal, colour (float4), texcoord0, texcoord1 (x: the instance index)
  struct V { float p[3], n[3], c[4], uv[2], inst[2]; };
  const float vc[3] = { 0.8f, 0.6f, 0.4f };
  auto corner = [&](float x, float y) {
    return V{ { x, y, 0.5f }, { 0, 0, 1 }, { vc[0], vc[1], vc[2], 0.0f }, { 0, 0 }, { 0.0f, 0.0f } };
  };
  // Front faces are clockwise on screen (D3D9's convention, which the backend keeps): this quad faces the camera; the
  // other winding is its back (the blade seen from behind: its normal turned around).
  const V quad[6] = { corner(-1, -1), corner(1, 1), corner(1, -1), corner(-1, -1), corner(-1, 1), corner(1, 1) };
  const V backQuad[6] = { corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, -1), corner(1, 1), corner(-1, 1) };
  const BufKey vb = 0xC90C0, backVb = 0xC90E0;
  be.createBuffer(vb, sizeof(quad), false);
  be.writeBuffer(vb, 0, quad, sizeof(quad), 0);
  be.createBuffer(backVb, sizeof(backQuad), false);
  be.writeBuffer(backVb, 0, backQuad, sizeof(backQuad), 0);
  BufKey drawVb = vb;
  const VertexElement ve[5] = { { 0, 0, 2, 0, 0, 0 }, { 0, 12, 2, 0, 3, 0 }, { 0, 24, 3, 0, 10, 0 },
                                { 0, 40, 1, 0, 5, 0 }, { 0, 48, 1, 0, 5, 1 } };
  const DeclKey decl = 0xC9100;
  be.createDecl(decl, ve, 5);
  static float vsc[256][4], psc[224][4];
  memset(vsc, 0, sizeof(vsc));
  memset(psc, 0, sizeof(psc));
  float sunDir[4] = { 0.3f, 0.2f, 0.9f, 0.0f };  // DiffuseDir (the model reads it too)
  const float sunCol[4] = { 0.9f, 0.8f, 0.7f, 0.0f }, amb[4] = { 0.2f, 0.2f, 0.25f, 0.0f };
  memcpy(vsc[0], sunDir, 16);                                    // DiffuseDir
  memcpy(vsc[1], sunCol, 16);                                    // DiffuseColor
  const float fade[4] = { 0, 0, 1.0e6f, 1.0f }, addl[4] = { 1, 0, 0, 0 }, fog[4] = { 1.0e6f, 1.0f, 1.0f, 0.0f };
  memcpy(vsc[5], fade, 16);                                      // AlphaParam: no fade
  memcpy(vsc[6], amb, 16);                                       // AmbientColor
  memcpy(vsc[7], addl, 16);                                      // AddlParams.x
  for (int i = 0; i < 4; i++) vsc[9 + i][i] = 1.0f;             // ModelViewProj: the identity
  memcpy(vsc[15], fog, 16);                                      // FogParam: no fog
  const float inst[4] = { 0, 0, 0, 0.6f };
  memcpy(vsc[20], inst, 16);                                     // InstanceData[0]
  const double occlusion = 0.6 * 0.75 + 0.25;
  const uint32_t W = 32, H = 32;
  uint64_t version = 9950;
  auto draw = [&](float canopy, std::vector<uint8_t>& img) {
    vsc[250][0] = canopy;
    const ShaderReplacement::Constant pbr[1] = { { 223, { 1.0f, 0, 0, 0 } } };
    for (const Md5& m : repltest::g_lightingPs) be.setShaderConstants(m.b.data(), pbr, 1);
    DrawSetup s;
    s.renderStates = rs;
    s.samplerStates = ss;
    s.textures[0] = { DrawSetup::Texture::Tex, 0xC9080 };
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
    be.draw(s, 4, 2, false, 0, 0);
    be.endGroup();
    uint32_t vf = 0, ts = 0;
    if (!be.readback(rt, img, vf, ts) || ts != 4) fail("grass: readback");
    be.destroy(rt);
  };
  auto px = [&](const std::vector<uint8_t>& img, uint32_t x, uint32_t y, int ch) {
    return img.size() == W * H * 4 ? int(img[(size_t(y) * W + x) * 4 + 2 - ch]) : -1;
  };
  using wettest::gam;
  using wettest::lin;
  // The model (grass.frag's grassLight, after Community Shaders), at pixel (x, y): world (px, py, 0.5).
  auto model = [&](uint32_t x, uint32_t y, double h, int ch, bool backFace = false) {
    const double wx = (x + 0.5) / W * 2.0 - 1.0, wy = 1.0 - (y + 0.5) / H * 2.0;
    double n[3] = { 0, 0, backFace ? -1.0 : 1.0 };
    double cl[3] = { wx, wy, 0.5 + 0.5 * std::max(h, 1.0) };
    const double cll = std::sqrt(cl[0] * cl[0] + cl[1] * cl[1] + cl[2] * cl[2]);
    for (double& c : cl) c /= cll;
    const double nd = n[0] * cl[0] + n[1] * cl[1] + n[2] * cl[2];
    double b[3];
    for (int i = 0; i < 3; i++) b[i] = n[i] + (cl[i] - n[i] * nd) * 0.5;
    const double bl = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
    double L[3] = { sunDir[0], sunDir[1], sunDir[2] };
    const double ll = std::sqrt(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]);
    for (double& c : L) c /= ll;
    const double nl = (b[0] * L[0] + b[1] * L[1] + b[2] * L[2]) / bl;
    const double canopy = 1.0 - 0.5 * (1.0 - std::clamp(0.5 / std::max(h, 1.0), 0.0, 1.0));
    const double clumpVis = 1.0 + (std::clamp((cl[0] * L[0] + cl[1] * L[1] + cl[2] * L[2]) * 0.5 + 0.5, 0.0, 1.0) - 1.0) * 0.5;
    const double vao = lin(std::max(std::max(vc[0], vc[1]), vc[2]));
    const double albedo = lin(160.0 / 255.0) * lin(vc[ch]) / vao;
    const double E = lin(sunCol[ch] * occlusion) * canopy * clumpVis;
    double c = 0.5 * albedo * std::clamp(nl, 0.0, 1.0) * E + std::min(albedo, 1.0) * std::clamp(-nl, 0.0, 1.0) * E;
    c += albedo * lin(amb[ch] * occlusion) * canopy;
    return 255.0 * gam(c);
  };
  std::vector<uint8_t> h1, h4, behind;
  draw(1.0f, h1);
  draw(4.0f, h4);
  int worst1 = 0, worst4 = 0, canopyDiff = 0;
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++)
      for (int ch = 0; ch < 3; ch++) {
        worst1 = std::max(worst1, std::abs(px(h1, x, y, ch) - int(std::lround(model(x, y, 1.0, ch)))));
        worst4 = std::max(worst4, std::abs(px(h4, x, y, ch) - int(std::lround(model(x, y, 4.0, ch)))));
        canopyDiff = std::max(canopyDiff, std::abs(px(h1, x, y, ch) - px(h4, x, y, ch)));
      }
  // The sun behind the blade.
  sunDir[2] = -0.9f;
  memcpy(vsc[0], sunDir, 16);
  draw(1.0f, behind);
  int worstBack = 0;
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++)
      for (int ch = 0; ch < 3; ch++)
        worstBack = std::max(worstBack, std::abs(px(behind, x, y, ch) - int(std::lround(model(x, y, 1.0, ch)))));
  // The back face, the sun in front again: lit through the blade.
  sunDir[2] = 0.9f;
  memcpy(vsc[0], sunDir, 16);
  std::vector<uint8_t> backFace;
  drawVb = backVb;
  draw(1.0f, backFace);
  drawVb = vb;
  int worstBackFace = 0;
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++)
      for (int ch = 0; ch < 3; ch++)
        worstBackFace = std::max(worstBackFace, std::abs(px(backFace, x, y, ch) - int(std::lround(model(x, y, 1.0, ch, true)))));
  printf("grass: the back face worst %d; ", worstBackFace);
  printf("grass: centre %d %d %d (model %.1f %.1f %.1f); worst difference from the model %d (canopy height 1), %d (4); the "
         "canopy height changes it by %d; the sun behind: centre %d %d %d, worst %d\n", px(h1, W / 2, H / 2, 0),
         px(h1, W / 2, H / 2, 1), px(h1, W / 2, H / 2, 2), model(W / 2, H / 2, 1.0, 0), model(W / 2, H / 2, 1.0, 1),
         model(W / 2, H / 2, 1.0, 2), worst1, worst4, canopyDiff, px(behind, W / 2, H / 2, 0), px(behind, W / 2, H / 2, 1),
         px(behind, W / 2, H / 2, 2), worstBack);
  if (worst1 > 2 || worst4 > 2) fail("grass: the grass shading is not Community Shaders' grass model");
  if (canopyDiff < 5) fail("grass: the canopy height does not reach the shader");
  if (worstBack > 2) fail("grass: the light through the blade is not the model's");
  if (worstBackFace > 2) fail("grass: a blade's back face is not lit as its other side");
  be.destroyTexture(0xC9080);
  be.destroyBuffer(vb);
  be.destroyBuffer(backVb);
}

}  // namespace grasstest
