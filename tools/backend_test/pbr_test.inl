// Physically based lighting through the backend (included by backend_test.cpp): a flat quad drawn with the game's
// SLS2000.vso (translated, with sdp2spv's view vector at location 12) and replacements/lighting's SLS2000.pso
// (adts.frag), as the plugin installs it (replace_test.inl's prepare). Object space = clip space (ModelViewProj the
// identity), the quad at z 0.5 facing +z with the tangent frame x, y, z, the sun along +z, a white base map and a flat
// normal map.
// - pbr 0: the game's shading, sun + ambient on white, whatever the eye position;
// - pbr 1: a material without the game's specular flag (SLS2000) is diffuse only: gamma(lin(sun) N.L + lin(ambient)),
//   whatever the eye position;
// - a glossy material with the flag (SLS2012.vso + SLS2017.pso, ADTS SPECULAR, the normal map's alpha 1, power 30):
//   with the eye above the specular peak (N.H = N.V = N.L = 1: GGX D = 1 / (pi alpha^2), visibility 0.25, F 0.04, alpha
//   sqrt(2 / 32)) on top of the diffuse; at the end of the game's specular fade (LightData[0].w
//   0, where the game switches to the non-specular variants) it draws exactly what SLS2000 draws;
// - the view vector across large triangles: the glossy material with the eye low (0.1) over an off-centre point of the
//   quad (two triangles spanning the target): the specular peak lies under the eye. The vector interpolated after
//   normalising per vertex puts it elsewhere (and moves it with the eye).
// FNV_GAME_SHADERS = the game's Data\Shaders folder (skipped without it).
namespace pbrtest {

static float toLinear(float c) { return std::pow(std::max(c, 0.0f), 2.2f); }
static float toGamma(float c) { return std::pow(std::max(c, 0.0f), 1.0f / 2.2f); }

static void run(Backend& be) {
  const char* dir = getenv("FNV_GAME_SHADERS");
  if (!dir || repltest::g_lightingPs.empty()) { printf("pbr: no lighting replacement or FNV_GAME_SHADERS: skipped\n"); return; }
  const std::vector<Md5> vs = sdpShaderMd5s(dir, "SLS2000.vso", { 13 });
  const ShaderKey vsKey = 0xC2000, psKey = 0xC2040;
  bool haveVs = false, havePs = false;
  for (const Md5& m : vs) haveVs = haveVs || be.createShader(vsKey, m.b.data());
  for (const Md5& m : sdpShaderMd5s(dir, "SLS2000.pso", { 13 })) havePs = havePs || be.createShader(psKey, m.b.data());
  if (!haveVs || !havePs) { fail("pbr: SLS2000.vso / SLS2000.pso not in the shader pack"); return; }
  const ShaderKey specVsKey = 0xC2010, specPsKey = 0xC2050;
  bool haveSpecVs = false, haveSpecPs = false;
  for (const Md5& m : sdpShaderMd5s(dir, "SLS2012.vso", { 13 })) haveSpecVs = haveSpecVs || be.createShader(specVsKey, m.b.data());
  for (const Md5& m : sdpShaderMd5s(dir, "SLS2017.pso", { 13 })) haveSpecPs = haveSpecPs || be.createShader(specPsKey, m.b.data());
  if (!haveSpecVs || !haveSpecPs) { fail("pbr: SLS2012.vso / SLS2017.pso not in the shader pack"); return; }

  D3DRef fresh;
  if (!fresh.init()) { fail("pbr: no D3D9 device for default states"); return; }
  static uint32_t rs[256], ss[16][14];
  for (uint32_t i = 0; i < 256; i++) { DWORD v = 0; fresh.dev->GetRenderState(D3DRENDERSTATETYPE(i), &v); rs[i] = v; }
  for (uint32_t s = 0; s < 16; s++)
    for (uint32_t t = 0; t < 14; t++) { DWORD v = 0; fresh.dev->GetSamplerState(s, D3DSAMPLERSTATETYPE(t), &v); ss[s][t] = v; }
  rs[D3DRS_ZENABLE] = 0;
  rs[D3DRS_CULLMODE] = D3DCULL_NONE;

  auto texture = [&](TexKey key, uint8_t b, uint8_t g, uint8_t r) {  // 4 x 4 A8R8G8B8 of one colour
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
  if (!texture(0xC2080, 255, 255, 255) || !texture(0xC20C0, 255, 128, 128)) { fail("pbr: textures"); return; }

  // position, tangent, binormal, normal (float3), texcoord (float2), colour (float4): stride 72.
  struct V { float p[3], t[3], b[3], n[3], uv[2], c[4]; };
  const V corner[4] = { { { -1, -1, 0.5f }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 0 }, { 1, 1, 1, 1 } },
                        { { 1, -1, 0.5f }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0 }, { 1, 1, 1, 1 } },
                        { { 1, 1, 0.5f }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 1 }, { 1, 1, 1, 1 } },
                        { { -1, 1, 0.5f }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 1 }, { 1, 1, 1, 1 } } };
  const V quad[6] = { corner[0], corner[1], corner[2], corner[0], corner[2], corner[3] };
  const BufKey vb = 0xC2100;
  be.createBuffer(vb, sizeof(quad), false);
  be.writeBuffer(vb, 0, quad, sizeof(quad), 0);
  const VertexElement ve[6] = { { 0, 0, 2 /*FLOAT3*/, 0, 0 /*POSITION*/, 0 }, { 0, 12, 2, 0, 6 /*TANGENT*/, 0 },
                                { 0, 24, 2, 0, 7 /*BINORMAL*/, 0 },          { 0, 36, 2, 0, 3 /*NORMAL*/, 0 },
                                { 0, 48, 1 /*FLOAT2*/, 0, 5 /*TEXCOORD*/, 0 }, { 0, 56, 3 /*FLOAT4*/, 0, 10 /*COLOR*/, 0 } };
  const DeclKey decl = 0xC2140;
  be.createDecl(decl, ve, 6);

  static float vsc[256][4], psc[224][4];
  memset(vsc, 0, sizeof(vsc));
  memset(psc, 0, sizeof(psc));
  for (int i = 0; i < 4; i++) vsc[i][i] = 1.0f;                      // ModelViewProj: the identity
  const float fog[4] = { 1.0e6f, 1.0f, 1.0f, 0.0f }, light[4] = { 0, 0, 1, 1 };
  memcpy(vsc[14], fog, 16);                                          // FogParam (fog off in the pixel shader)
  memcpy(vsc[25], light, 16);                                        // LightData: the sun along +z
  const float ambient[4] = { 0.1f, 0.1f, 0.1f, 1.0f }, sun[4] = { 0.8f, 0.7f, 0.6f, 1.0f }, toggles[4] = { 0, 0, 30, 0 };
  memcpy(psc[1], ambient, 16);
  memcpy(psc[3], sun, 16);
  memcpy(psc[27], toggles, 16);  // no vertex colour, no fog, power 30, no alpha cut-off

  const uint32_t W = 32, H = 32;
  uint64_t version = 8000;
  std::function<void()> afterDraw;  // run inside the draw's group, after the draw (an effect over it)
  auto draw = [&](const float eye[3], float pbr, std::vector<uint8_t>& img, ShaderKey vk, ShaderKey pk, float lodFade,
                  float specular) {
    memcpy(vsc[16], eye, 12);  // EyePosition
    vsc[16][3] = 1.0f;
    vsc[25][3] = lodFade;      // LightData[0].w: the game's specular LOD fade (the SPECULAR vertex shaders' t1.w)
    // The replacement's parameters, as the plugin applies them: pbr c223, specular c222.
    const ShaderReplacement::Constant params[2] = { { 223, { pbr, 0, 0, 0 } }, { 222, { specular, 0, 0, 0 } } };
    for (const Md5& m : repltest::g_lightingPs) be.setShaderConstants(m.b.data(), params, 2);
    DrawSetup s;
    s.renderStates = rs;
    s.samplerStates = ss;
    s.textures[0] = { DrawSetup::Texture::Tex, 0xC2080 };
    s.textures[1] = { DrawSetup::Texture::Tex, 0xC20C0 };
    s.vs = vk;
    s.ps = pk;
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
    be.draw(s, 4 /*D3DPT_TRIANGLELIST*/, 2, false, 0, 0);
    if (afterDraw) afterDraw();
    be.endGroup();
    uint32_t vf = 0, ts = 0;
    if (!be.readback(rt, img, vf, ts) || ts != 4) fail("pbr: readback");
    be.destroy(rt);
  };
  const float above[3] = { 0, 0, 5 }, side[3] = { 3, 0, 1 }, lowEye[3] = { 0.5f, 0.0f, 0.6f };
  std::vector<uint8_t> vanAbove, vanSide, pbrAbove, pbrSide, specFaded, specFull, low;
  draw(above, 0.0f, vanAbove, vsKey, psKey, 1.0f, 1.0f);
  draw(side, 0.0f, vanSide, vsKey, psKey, 1.0f, 1.0f);
  draw(above, 1.0f, pbrAbove, vsKey, psKey, 1.0f, 1.0f);
  draw(side, 1.0f, pbrSide, vsKey, psKey, 1.0f, 1.0f);
  draw(above, 1.0f, specFaded, specVsKey, specPsKey, 0.0f, 1.0f);
  draw(above, 1.0f, specFull, specVsKey, specPsKey, 1.0f, 1.0f);
  draw(lowEye, 1.0f, low, specVsKey, specPsKey, 1.0f, 1.0f);
  for (const auto* img : { &vanAbove, &vanSide, &pbrAbove, &pbrSide, &specFaded, &specFull, &low })
    if (img->size() != W * H * 4) { fail("pbr: image sizes"); return; }
  const size_t c = ((H / 2) * W + W / 2) * 4;  // the centre texel, BGRA
  auto rgb = [&](const std::vector<uint8_t>& img, int ch) { return int(img[c + 2 - ch]); };
  auto within = [](int a, float b, int tol) { return std::abs(a - int(std::lround(b * 255.0f))) <= tol; };
  // The game's shading: (sun * N.L + ambient) * white, N.L = 1 (to the normal map's 8-bit precision); and the diffuse
  // of the physically based mode: gamma(lin(sun) + lin(ambient)). Neither depends on the eye.
  bool vanilla = true, diffuse = true, peak = true;
  for (int ch = 0; ch < 3; ch++) {
    vanilla &= within(rgb(vanAbove, ch), sun[ch] + ambient[ch], 2) && rgb(vanAbove, ch) == rgb(vanSide, ch);
    diffuse &= within(rgb(pbrAbove, ch), toGamma(toLinear(sun[ch]) + toLinear(ambient[ch])), 2) &&
               rgb(pbrAbove, ch) == rgb(pbrSide, ch);
    const float a2 = 2.0f / 32.0f, D = 1.0f / (3.14159265f * a2), lobe = 3.14159265f * D * 0.25f * 0.04f;
    peak &= within(rgb(specFull, ch), toGamma(toLinear(sun[ch]) * (1.0f + lobe) + toLinear(ambient[ch])), 2);
  }
  printf("pbr: centre vanilla %d %d %d (eye moved: %d %d %d), pbr %d %d %d (eye moved: %d %d %d), glossy eye above %d %d %d\n",
         rgb(vanAbove, 0), rgb(vanAbove, 1), rgb(vanAbove, 2), rgb(vanSide, 0), rgb(vanSide, 1), rgb(vanSide, 2),
         rgb(pbrAbove, 0), rgb(pbrAbove, 1), rgb(pbrAbove, 2), rgb(pbrSide, 0), rgb(pbrSide, 1), rgb(pbrSide, 2),
         rgb(specFull, 0), rgb(specFull, 1), rgb(specFull, 2));
  // The specular LOD switch: every texel of the faded specular material equals the non-specular draw (to 1 step).
  uint32_t seamBad = 0;
  for (size_t i = 0; i < specFaded.size(); i++)
    if (std::abs(int(specFaded[i]) - int(pbrAbove[i])) > 1) seamBad++;
  printf("pbr: specular LOD: fade 0 %d %d %d (%u channels off the non-specular draw)\n", rgb(specFaded, 0),
         rgb(specFaded, 1), rgb(specFaded, 2), seamBad);
  // The specular peak under the low eye: the centroid of the brightest texels (within 2 steps of the maximum) at the
  // texel under the eye, clip (0.5, 0): (0.75 W - 0.5, 0.5 H - 0.5) in texel centres.
  int maxLum = 0;
  for (size_t i = 0; i < low.size(); i += 4) maxLum = std::max(maxLum, int(low[i]) + low[i + 1] + low[i + 2]);
  double sx = 0, sy = 0, n = 0;
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++) {
      const size_t i = (size_t(y) * W + x) * 4;
      if (int(low[i]) + low[i + 1] + low[i + 2] >= maxLum - 6) { sx += x; sy += y; n++; }
    }
  const double px = sx / n, py = sy / n, ex = 0.75 * W - 0.5, ey = 0.5 * H - 0.5;
  printf("pbr: low eye over (%.1f, %.1f): specular peak at (%.2f, %.2f), %.0f texels\n", ex, ey, px, py, n);
  // Reflections (docs/INDIRECT.md): the glossy material under a uniform environment (a rendered cube of grey 128 on
  // every face, set as the environment and filtered, its harmonics read back after a frame) with the tangent-to-world
  // frame the identity (c250..c252), and no sun or ambient light: with the eye above, the colour is the environment's
  // radiance times the split-sum albedo (Karis's fit at N.V = 1, roughness sqrt(alpha) = 0.5), black without it. With
  // c219 1 (a material with its own environment-map pass) it is black.
  {
    RtKey envFaces[6];
    int32_t fullFace[4] = { 0, 0, 16, 16 };
    for (RtKey& f : envFaces) {
      f = newKey();
      be.attach(f, { 16, 16, kD3DFMT_A16B16G16R16F, 1, false });
      be.beginGroup(&f, 1, 0);
      be.clear(kClearTarget, 0xFF808080u, 1.0f, 0, fullFace);
      be.endGroup();
    }
    const TexKey envKey = 0xC0BE2000;
    be.attachCube(envKey, envFaces, 1);
    be.setEnvironment(envKey);
    for (int j = 0; j < 3; j++) vsc[250 + j][j] = 1.0f;
    for (int c = 0; c < 3; c++) psc[1][c] = psc[3][c] = 0.0f;
    // As in the game, the first draw that samples the new environment brings it up to date (the cube assembled and
    // prefiltered in the middle of the draw): that draw must still be drawn with its own pipeline (it shows the
    // environment as not yet valid, black here, since its harmonics are read back a frame later).
    std::vector<uint8_t> first;
    draw(above, 1.0f, first, specVsKey, specPsKey, 1.0f, 1.0f);
    be.finishFrame();  // the harmonics are read back when the frame that copied them has finished
    bool firstOk = first.size() == W * H * 4;
    for (int ch = 0; ch < 3 && firstOk; ch++) firstOk &= rgb(first, ch) == 0;
    if (!firstOk) fail("pbr: the draw that prefiltered the environment did not draw with its own pipeline");
    std::vector<uint8_t> refl, own;
    draw(above, 1.0f, refl, specVsKey, specPsKey, 1.0f, 1.0f);
    psc[219][0] = 1.0f;
    draw(above, 1.0f, own, specVsKey, specPsKey, 1.0f, 1.0f);
    psc[219][0] = 0.0f;
    for (int j = 0; j < 3; j++) vsc[250 + j][j] = 0.0f;
    memcpy(psc[1], ambient, 16);
    memcpy(psc[3], sun, 16);
    be.setEnvironment(0);
    for (RtKey f : envFaces) be.destroy(f);
    const float env = toLinear(128.0f / 255.0f);
    const float rough = 0.5f, nv = 1.0f;
    const float q[4] = { rough * -1.0f + 1.0f, rough * -0.0275f + 0.0425f, rough * -0.572f + 1.04f, rough * 0.022f - 0.04f };
    const float a004 = std::min(q[0] * q[0], std::exp2(-9.28f * nv)) * q[0] + q[1];
    const float A = -1.04f * a004 + q[2], B = 1.04f * a004 + q[3];
    const float dfg = 0.04f * A + B;
    bool reflOk = refl.size() == W * H * 4 && own.size() == refl.size(), ownOk = reflOk;
    for (int ch = 0; ch < 3 && reflOk; ch++) {
      reflOk &= within(rgb(refl, ch), toGamma(env * dfg), 1);
      ownOk &= rgb(own, ch) == 0;
    }
    printf("pbr: reflection of a uniform environment (radiance %.4f, albedo %.4f, expected %.1f): %d %d %d, own "
           "environment map %d %d %d\n", env, dfg, toGamma(env * dfg) * 255.0f, rgb(refl, 0), rgb(refl, 1), rgb(refl, 2),
           rgb(own, 0), rgb(own, 1), rgb(own, 2));
    if (!reflOk) fail("pbr: the reflection of a uniform environment is not the split-sum model's");
    if (!ownOk) fail("pbr: a material with its own environment-map pass still gets the probe's reflection");
  }
  // Ambient occlusion and screen-space indirect light in the lighting (docs/SHADER_ABI.md "Ambient occlusion"), published
  // by two separate effects: one (Publish=ao) visibility 0.25 in the left half of the target and 1 in the right, with log2
  // of the quad's view distance in g (the quad's depth is 0.5: with near 1, far 3 the distance is 1.5); the other
  // (Publish=gi, gidepth) indirect light 0.1 with half the rays missing in the left half, none found (all missing) in the
  // right, with the same distance in its depth image. The camera: rotation the identity (direction world x, up y, right
  // z), frustum -1..1. SLS2000 with pbr on, no sun, ambient 0.5: the colour is gamma(gi + miss * lin(0.5) * ao) where
  // each is valid, gamma(lin(0.5)) (128) where neither is or on the right. Run with both effects, the ambient occlusion
  // alone and the indirect light alone (the left half's expected value differs; the rest is the same):
  // - frame A: nothing published yet (128);
  // - frame B, the camera unchanged: the left half lit as published, the right half 128;
  // - frame C, the camera moved forward 0.2: the surface's distance in the previous frame (1.7) is not the published
  //   one (1.5): nothing applies (disocclusion);
  // - frame D, the camera moved right 0.75 (a quarter of the view at distance 1.5): a pixel at u is the previous
  //   frame's u + 0.25, so u 0.1 gets the left half's values (0.35 was in the left half) and u 0.4 does not (0.65);
  // - frames E, F: the camera back at the origin; E publishes a moved object: a strip about 3 pixels wide around u 0.25
  //   where both published distances are another surface's (twice as far). F, reading at u 0.25 (the strip): the values
  //   of the surface beside the strip, found by the neighbourhood search in each image's own depth.
  {
    namespace fs = std::filesystem;
    auto writeFile = [](const fs::path& path, const char* text) {
      if (FILE* f = fopen(path.string().c_str(), "wb")) {
        fputs(text, f);
        fclose(f);
      }
    };
    const fs::path aoDir = "build/fx_aotest/aoconst", giDir = "build/fx_aotest/giconst";
    fs::create_directories(aoDir);
    fs::create_directories(giDir);
    writeFile(aoDir / "effect.ini", "[Effect]\nName=AO test\nPoint=world\n[Param.hole]\nType=float\nDefault=0\n[Image.ao]\n"
                                    "Format=rg16f\nPublish=ao\n[Pass.0]\nShader=pass.frag\nOutput=ao\n");
    writeFile(aoDir / "pass.frag",
              "layout(location = 0) out vec2 o;\nvoid main() {\n"
              "  const bool hole = p_hole != 0.0 && abs(gl_FragCoord.x - fnv.target.x * 0.25) < 1.6;\n"
              "  o = vec2(gl_FragCoord.x < fnv.target.x * 0.5 ? 0.25 : 1.0, log2(fnvLinearZ(0.5)) + (hole ? 1.0 : 0.0));\n}\n");
    writeFile(giDir / "effect.ini", "[Effect]\nName=GI test\nPoint=world\n[Param.hole]\nType=float\nDefault=0\n[Image.gi]\n"
                                    "Format=rgba16f\nPublish=gi\n[Image.gidepth]\nFormat=r16f\nPublish=gidepth\n[Pass.0]\n"
                                    "Shader=gi.frag\nOutput=gi\n[Pass.1]\nShader=depth.frag\nOutput=gidepth\n");
    writeFile(giDir / "gi.frag",
              "layout(location = 0) out vec4 o;\nvoid main() {\n"
              "  o = gl_FragCoord.x < fnv.target.x * 0.5 ? vec4(0.1, 0.1, 0.1, 0.5) : vec4(0.0, 0.0, 0.0, 1.0);\n}\n");
    writeFile(giDir / "depth.frag",
              "layout(location = 0) out float o;\nvoid main() {\n"
              "  const bool hole = p_hole != 0.0 && abs(gl_FragCoord.x - fnv.target.x * 0.25) < 1.6;\n"
              "  o = log2(fnvLinearZ(0.5)) + (hole ? 1.0 : 0.0);\n}\n");
    std::vector<fx::Effect> all = fx::loadEffects("build/fx_aotest", "build/fx_cache_test", [](const char* m) { printf("[fx] %s\n", m); });
    uint32_t aoId = 0, giId = 0;
    for (const fx::Effect& e : all) {
      if (!e.error.empty()) { fail("pbr ao: test effect '%s': %s", e.id.c_str(), e.error.c_str()); continue; }
      Backend::EffectDesc d;
      for (const fx::Image& im : e.images)
        d.images.push_back({ im.vkFormat, im.scale,
                             uint8_t(im.publish == "ao" ? Backend::EffectDesc::Ao
                                                        : im.publish == "gi" ? Backend::EffectDesc::Gi : im.publish == "gidepth" ? Backend::EffectDesc::GiDepth : Backend::EffectDesc::None) });
      for (const fx::Pass& ps : e.passes)
        d.passes.push_back({ { ps.spv[0].data(), ps.spv[1].data() }, { ps.spv[0].size(), ps.spv[1].size() }, ps.output, uint32_t(ps.blend) });
      (e.id == "aoconst" ? aoId : giId) = be.createEffect(d);
    }
    if (!aoId || !giId) {
      fail("pbr ao: the test effects did not load");
    } else {
      const float frustum[4] = { -1.0f, 1.0f, 1.0f, -1.0f }, port[4] = { 0.0f, 1.0f, 1.0f, 0.0f };
      const float amb[4] = { 0.5f, 0.5f, 0.5f, 1.0f }, zero[4] = { 0, 0, 0, 1 };
      const int full = int(std::lround(255.0f * 0.5f));
      auto near1 = [](int x, int w) { return std::abs(x - w) <= 1; };
      auto at = [&](const std::vector<uint8_t>& img, float u) {  // green at (u W, H / 2)
        const uint32_t x = std::min(W - 1, uint32_t(u * W)), y = H / 2;
        return img.size() == size_t(W) * H * 4 ? int(img[(size_t(y) * W + x) * 4 + 1]) : -1;
      };
      struct Case { const char* name; bool ao, gi; float lit; };
      const Case cases[3] = { { "both", true, true, toGamma(0.1f + 0.5f * toLinear(0.5f) * 0.25f) },
                              { "ambient occlusion alone", true, false, toGamma(toLinear(0.5f) * 0.25f) },
                              { "indirect light alone", false, true, toGamma(0.1f + 0.5f * toLinear(0.5f)) } };
      for (const Case& k : cases) {
        Backend::EffectFrame fr{};
        memcpy(fr.frustum, frustum, 16);
        memcpy(fr.port, port, 16);
        fr.nearZ = 1.0f;
        fr.farZ = 3.0f;
        for (int r = 0; r < 3; r++) fr.cameraRows[r][r] = 1.0f;
        Backend::AoLighting ao{};
        memcpy(ao.frustum, frustum, 16);
        memcpy(ao.port, port, 16);
        ao.nearZ = 1.0f;
        ao.farZ = 3.0f;
        memcpy(ao.cameraRows, fr.cameraRows, sizeof(ao.cameraRows));
        float params[1][4] = {};
        afterDraw = [&] {
          if (k.ao && !be.runEffect(aoId, fr, &params[0][0], 1)) fail("pbr ao: effect run failed");
          if (k.gi && !be.runEffect(giId, fr, &params[0][0], 1)) fail("pbr ao: effect run failed");
        };
        memcpy(psc[1], amb, 16);
        memcpy(psc[3], zero, 16);
        std::vector<uint8_t> a, b, c, d, e1, f1;
        draw(above, 1.0f, a, vsKey, psKey, 1.0f, 1.0f);   // A: publishes
        be.setAoLighting(&ao);
        draw(above, 1.0f, b, vsKey, psKey, 1.0f, 1.0f);   // B: same camera
        ao.cameraRows[0][3] = 0.2f;                         // C: forward (the lighting reads B's results)
        fr.cameraRows[0][3] = 0.2f;                          // C's effects publish from the moved camera
        be.setAoLighting(&ao);
        draw(above, 1.0f, c, vsKey, psKey, 1.0f, 1.0f);
        ao.cameraRows[2][3] = 0.75f;                         // D: right
        be.setAoLighting(&ao);
        draw(above, 1.0f, d, vsKey, psKey, 1.0f, 1.0f);
        fr.cameraRows[0][3] = ao.cameraRows[0][3] = 0.0f;
        fr.cameraRows[2][3] = ao.cameraRows[2][3] = 0.0f;
        be.setAoLighting(&ao);
        params[0][0] = 1.0f;
        draw(above, 1.0f, e1, vsKey, psKey, 1.0f, 1.0f);  // E: publishes the moved object
        params[0][0] = 0.0f;
        draw(above, 1.0f, f1, vsKey, psKey, 1.0f, 1.0f);  // F
        be.setAoLighting(nullptr);
        afterDraw = nullptr;
        // Two frames with neither effect: what was published ages out before the next case.
        draw(above, 1.0f, e1, vsKey, psKey, 1.0f, 1.0f);
        draw(above, 1.0f, e1, vsKey, psKey, 1.0f, 1.0f);
        const int lit = int(std::lround(255.0f * k.lit));
        const int va[2] = { at(a, 0.25f), at(a, 0.75f) }, vb[2] = { at(b, 0.25f), at(b, 0.75f) },
                  vc[2] = { at(c, 0.25f), at(c, 0.75f) }, vd[2] = { at(d, 0.1f), at(d, 0.4f) };
        const int vf = at(f1, 0.25f);
        printf("pbr ao, %s: expected %d / %d | A (none) %d %d, B (same camera) %d %d, C (forward) %d %d, D (right: u 0.1, "
               "0.4) %d %d, F (moved object around u 0.25) %d\n", k.name, lit, full, va[0], va[1], vb[0], vb[1], vc[0], vc[1],
               vd[0], vd[1], vf);
        if (!near1(va[0], full) || !near1(va[1], full)) fail("pbr ao, %s: applied before anything was published", k.name);
        if (!near1(vb[0], lit) || !near1(vb[1], full)) fail("pbr ao, %s: what was published is not applied", k.name);
        if (!near1(vc[0], full) || !near1(vc[1], full)) fail("pbr ao, %s: applied where the depth does not match", k.name);
        if (!near1(vd[0], lit) || !near1(vd[1], full)) fail("pbr ao, %s: the reprojection does not follow the camera", k.name);
        if (!near1(vf, lit)) fail("pbr ao, %s: where an object moved, the surface beside it does not give its values", k.name);
      }
      memcpy(psc[1], ambient, 16);
      memcpy(psc[3], sun, 16);
      be.destroyEffect(aoId);
      be.destroyEffect(giId);
    }
  }
  // Reuse of the previous draw's pipeline and render-state blocks (DrawSetup::renderStatesVersion): three draws into one
  // target, the second with alpha test on (reference 255, less: every pixel discarded), the third either with the same
  // state and version as the second (reused: everything discarded) or back to the first state with a new version
  // (recomputed: drawn). Each final image must equal the same sequence drawn with the versions unknown (0).
  {
    const RtKey rt = newKey();
    be.attach(rt, { 32, 32, kD3DFMT_A8R8G8B8, 1, false });
    DrawSetup s;
    s.samplerStates = ss;
    s.textures[0] = { DrawSetup::Texture::Tex, 0xC2080 };
    s.textures[1] = { DrawSetup::Texture::Tex, 0xC20C0 };
    s.vs = vsKey;
    s.ps = psKey;
    s.decl = decl;
    s.streams[0] = { vb, 0, sizeof(V) };
    s.vsConstants = &vsc[0][0];
    s.psConstants = &psc[0][0];
    s.vsConstantsVersion = s.psConstantsVersion = ++version;
    s.viewport = { 0, 0, 32, 32, 0.0f, 1.0f };
    s.scissor[2] = s.scissor[3] = 32;
    static uint32_t rsA[256], rsB[256];
    memcpy(rsA, rs, sizeof(rsA));
    memcpy(rsB, rs, sizeof(rsB));
    rsB[D3DRS_ALPHATESTENABLE] = 1;
    rsB[D3DRS_ALPHAFUNC] = D3DCMP_LESS;
    rsB[D3DRS_ALPHAREF] = 255;
    auto sequence = [&](bool versions, bool thirdLikeSecond, std::vector<uint8_t>& img) {
      be.beginGroup(&rt, 1, 0);
      int32_t full[4] = { 0, 0, 32, 32 };
      be.clear(kClearTarget, 0xFF102030u, 1.0f, 0, full);
      s.renderStates = rsA;
      s.renderStatesVersion = versions ? 1001 : 0;
      s.clipPlanesVersion = versions ? 1 : 0;
      be.draw(s, 4, 2, false, 0, 0);
      be.clear(kClearTarget, 0xFF102030u, 1.0f, 0, full);
      s.renderStates = rsB;
      s.renderStatesVersion = versions ? 1002 : 0;
      be.draw(s, 4, 2, false, 0, 0);
      s.renderStates = thirdLikeSecond ? rsB : rsA;
      s.renderStatesVersion = versions ? (thirdLikeSecond ? 1002 : 1003) : 0;
      be.draw(s, 4, 2, false, 0, 0);
      be.endGroup();
      uint32_t vf = 0, ts = 0;
      if (!be.readback(rt, img, vf, ts)) fail("pbr reuse: readback");
    };
    const uint64_t reusesBefore = be.stats().pipelineReuses;
    std::vector<uint8_t> a0, a1, b0, b1;
    sequence(false, true, a0);
    sequence(true, true, a1);
    sequence(false, false, b0);
    sequence(true, false, b1);
    const uint64_t reuses = be.stats().pipelineReuses - reusesBefore;
    printf("pbr reuse: third draw like the second: %s (centre %d, clear 32), back to the first: %s (centre %d); %llu reuses\n",
           a0 == a1 ? "same" : "DIFFERENT", a1.empty() ? -1 : int(a1[(16 * 32 + 16) * 4 + 1]), b0 == b1 ? "same" : "DIFFERENT",
           b1.empty() ? -1 : int(b1[(16 * 32 + 16) * 4 + 1]), (unsigned long long)reuses);
    if (a0 != a1 || b0 != b1) fail("pbr reuse: reusing the previous draw's state changed the image");
    if (reuses != 1) fail("pbr reuse: expected exactly one reuse (the third draw like the second), got %llu", (unsigned long long)reuses);
    s.renderStatesVersion = s.clipPlanesVersion = 0;
    be.destroy(rt);
  }
  // A long run of draws with no wait between them, each into its own 8 x 8 cell of two 128 x 128 targets: random
  // constants (a new version with new values, or the same version with the same values), the two shader pairs, render
  // states (alpha test half way, version or unknown), sampler states, textures, the replacement's pbr parameter and
  // rewrites of the vertex buffer (discard) between draws. The hash of the images is printed: with the record thread
  // (FNVVK_RECORD_THREAD) it must be the one without (docs/PERFORMANCE.md "Record thread"; the logs are compared).
  {
    const RtKey rts[2] = { newKey(), newKey() };
    for (RtKey k : rts) be.attach(k, { 128, 128, kD3DFMT_A8R8G8B8, 1, false });
    static uint32_t rsA[256], rsC[256], ssA[16][14], ssB[16][14];
    memcpy(rsA, rs, sizeof(rsA));
    memcpy(rsC, rs, sizeof(rsC));
    rsC[D3DRS_ALPHATESTENABLE] = 1;
    rsC[D3DRS_ALPHAFUNC] = D3DCMP_GREATER;
    rsC[D3DRS_ALPHAREF] = 128;
    memcpy(ssA, ss, sizeof(ssA));
    memcpy(ssB, ss, sizeof(ssB));
    ssB[0][D3DSAMP_MAGFILTER] = D3DTEXF_POINT;
    ssB[0][D3DSAMP_MINFILTER] = D3DTEXF_POINT;
    V tilted[6];
    memcpy(tilted, quad, sizeof(quad));
    for (V& v : tilted) { v.n[0] = 0.6f; v.n[2] = 0.8f; }
    // A third vertex shader that does not read FogParam (c14, c15), which SLS2000.vso reads: switching between them
    // under the same constants version must still bring the worker's c14, c15 up to date (fog is on in some draws).
    const ShaderKey altVsKey = 0xC2020;
    bool haveAlt = false;
    for (const Md5& m : sdpShaderMd5s(dir, "SLS2028.vso", { 13 })) haveAlt = haveAlt || be.createShader(altVsKey, m.b.data());
    if (!haveAlt) fail("record stress: SLS2028.vso not in the shader pack");
    uint32_t seed = 12345;
    auto rnd = [&] { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
    auto unit = [&] { return float(rnd() & 0xFFFF) / 65535.0f; };
    uint64_t rsVersion = 5000;
    DrawSetup s;
    s.decl = decl;
    s.streams[0] = { vb, 0, sizeof(V) };
    s.vsConstants = &vsc[0][0];
    s.psConstants = &psc[0][0];
    s.vsConstantsVersion = s.psConstantsVersion = ++version;
    s.renderStates = rsA;
    s.renderStatesVersion = ++rsVersion;
    s.clipPlanesVersion = 1;
    s.samplerStates = ssA;
    s.textures[1] = { DrawSetup::Texture::Tex, 0xC20C0 };
    for (int t = 0; t < 2; t++) {
      be.beginGroup(&rts[t], 1, 0);
      int32_t full[4] = { 0, 0, 128, 128 };
      be.clear(kClearTarget, 0xFF000000u, 1.0f, 0, full);
      for (int i = 0; i < 256; i++) {
        const int n = t * 256 + i;
        if (n % 37 == 5) {
          const ShaderReplacement::Constant params[2] = { { 223, { float(rnd() & 1), 0, 0, 0 } }, { 222, { 0.5f + unit(), 0, 0, 0 } } };
          for (const Md5& m : repltest::g_lightingPs) be.setShaderConstants(m.b.data(), params, 2);
        }
        if (n % 50 == 17) be.writeBuffer(vb, 0, (rnd() & 1) ? tilted : quad, sizeof(quad), 0x2000 /*D3DLOCK_DISCARD*/);
        const uint32_t pick = rnd() % 4;  // 0: the specular pair, 1: the vertex shader without fog, else the base pair
        s.vs = pick == 0 ? specVsKey : pick == 1 ? altVsKey : vsKey;
        s.ps = pick == 0 ? specPsKey : psKey;
        if ((rnd() % 10) < 7) {  // new constants
          // Every register but the transform (c0..c3): the two shader pairs read different ones, so a register one
          // reads and the other does not is stale on the worker unless the switch sends it (the negative control).
          for (int k = 4; k < 64; k++)
            for (int c = 0; c < 4; c++) vsc[k][c] = unit();
          for (int k = 0; k < 41; k++)
            for (int c = 0; c < 4; c++) psc[k][c] = unit();
          psc[219][0] = float(rnd() & 1);  // SLS2017's replacement alone reads c219 (envmap flag, first specular pass)
          psc[219][1] = float(rnd() & 1);
          const float eye[4] = { unit() * 4 - 2, unit() * 4 - 2, 0.5f + unit() * 4, 1.0f };
          memcpy(vsc[16], eye, 16);
          vsc[25][3] = unit();
          psc[1][0] = psc[1][1] = psc[1][2] = 0.05f + 0.2f * unit();
          psc[3][0] = unit();
          psc[3][1] = unit();
          psc[3][2] = unit();
          psc[27][2] = 5.0f + 60.0f * unit();
          s.vsConstantsVersion = s.psConstantsVersion = ++version;
        }
        const uint32_t r = rnd() % 8;
        if (r == 0) { s.renderStates = rsC; s.renderStatesVersion = ++rsVersion; }
        else if (r == 1) { s.renderStates = rsA; s.renderStatesVersion = ++rsVersion; }
        else if (r == 2) { s.renderStatesVersion = 0; }
        else if (s.renderStatesVersion == 0) { s.renderStatesVersion = ++rsVersion; }
        s.samplerStates = (rnd() & 1) ? ssA : ssB;
        s.textures[0] = { DrawSetup::Texture::Tex, (rnd() & 1) ? 0xC2080u : 0xC20C0u };
        s.viewport = { (i % 16) * 8, (i / 16) * 8, 8, 8, 0.0f, 1.0f };
        s.scissor[0] = s.viewport.x;
        s.scissor[1] = s.viewport.y;
        s.scissor[2] = s.viewport.x + 8;
        s.scissor[3] = s.viewport.y + 8;
        be.draw(s, 4, 2, false, 0, 0);
      }
      be.endGroup();
    }
    uint32_t hash = 2166136261u, colours = 0;
    std::set<uint32_t> distinct;
    for (RtKey k : rts) {
      std::vector<uint8_t> img;
      uint32_t vf = 0, ts = 0;
      if (!be.readback(k, img, vf, ts) || img.size() != 128 * 128 * 4) { fail("record stress: readback"); break; }
      for (size_t p = 0; p < img.size(); p += 4) {
        uint32_t v;
        memcpy(&v, &img[p], 4);
        distinct.insert(v);
      }
      for (uint8_t b : img) hash = (hash ^ b) * 16777619u;
      be.destroy(k);
    }
    colours = uint32_t(distinct.size());
    printf("record stress: 512 draws, image hash %08X, %u distinct colours\n", hash, colours);
    if (colours < 64) fail("record stress: the draws did not vary the image (%u colours)", colours);
    be.writeBuffer(vb, 0, quad, sizeof(quad), 0x2000);
    psc[219][0] = psc[219][1] = 0.0f;
    const ShaderReplacement::Constant params[2] = { { 223, { 0.0f, 0, 0, 0 } }, { 222, { 1.0f, 0, 0, 0 } } };
    for (const Md5& m : repltest::g_lightingPs) be.setShaderConstants(m.b.data(), params, 2);
  }
  // A query polled with nothing else queued: issued around a full-target draw, the frame submitted (present without a
  // target), then queryResult until it is ready (2 s at most). With the record thread the worker must publish the result
  // on its own while the caller polls (Backend::queryResult, Recorder::run).
  {
    const RtKey rt = newKey();
    be.attach(rt, { 32, 32, kD3DFMT_A8R8G8B8, 1, false });
    DrawSetup s;
    s.renderStates = rs;
    s.samplerStates = ss;
    s.textures[0] = { DrawSetup::Texture::Tex, 0xC2080 };
    s.textures[1] = { DrawSetup::Texture::Tex, 0xC20C0 };
    s.vs = vsKey;
    s.ps = psKey;
    s.decl = decl;
    s.streams[0] = { vb, 0, sizeof(V) };
    s.vsConstants = &vsc[0][0];
    s.psConstants = &psc[0][0];
    s.vsConstantsVersion = s.psConstantsVersion = ++version;
    s.viewport = { 0, 0, 32, 32, 0.0f, 1.0f };
    s.scissor[2] = s.scissor[3] = 32;
    const uint32_t q = be.createQuery();
    be.beginGroup(&rt, 1, 0);
    int32_t full[4] = { 0, 0, 32, 32 };
    be.clear(kClearTarget, 0xFF000000u, 1.0f, 0, full);
    be.beginQuery(q);
    be.draw(s, 4, 2, false, 0, 0);
    be.endQuery(q);
    be.endGroup();
    be.present(0);
    uint32_t samples = 0;
    int got = 0;
    const ULONGLONG t0 = GetTickCount64();
    while (!(got = be.queryResult(q, samples)) && GetTickCount64() - t0 < 2000) Sleep(0);
    printf("query polled after the frame: %s, %u samples\n", got ? "ready" : "NOT READY after 2 s", samples);
    if (!got || samples != 32 * 32) fail("query: the result of a submitted query did not arrive while polled (%u samples)", samples);
    be.destroyQuery(q);
    be.destroy(rt);
  }
  // CPU cost of recording a game draw (Backend::draw, the lighting shaders replaced), as the game issues them: every draw
  // changes the vertex and pixel constants (new versions), half of them the texture. Timed on the recording thread only
  // (the GPU work is submitted after). BENCH=1 to run (a measurement, not a check).
  if (getenv("BENCH")) {
    const RtKey rt = newKey();
    be.attach(rt, { 256, 256, kD3DFMT_A8R8G8B8, 1, false });
    DrawSetup s;
    s.renderStates = rs;
    s.samplerStates = ss;
    s.textures[1] = { DrawSetup::Texture::Tex, 0xC20C0 };
    s.vs = vsKey;
    s.ps = psKey;
    s.decl = decl;
    s.streams[0] = { vb, 0, sizeof(V) };
    s.vsConstants = &vsc[0][0];
    s.psConstants = &psc[0][0];
    s.viewport = { 0, 0, 256, 256, 0.0f, 1.0f };
    s.scissor[2] = s.scissor[3] = 256;
    s.renderStatesVersion = 7;  // as the game's tracker gives them; the render states do not change in the benchmark
    s.clipPlanesVersion = 7;
    // Modes: 0 constants and textures change every draw (as the game); 1 constants fixed; 2 textures fixed; 3 nothing
    // changes (the fixed cost of a draw).
    for (int mode = 0; mode < 4; mode++)
    for (float pbr : { 0.0f, 1.0f }) {
      const ShaderReplacement::Constant params[2] = { { 223, { pbr, 0, 0, 0 } }, { 222, { 1.0f, 0, 0, 0 } } };
      for (const Md5& m : repltest::g_lightingPs) be.setShaderConstants(m.b.data(), params, 2);
      constexpr int kDraws = 20000;
      double best = 1e30;
      for (int rep = 0; rep < 5; rep++) {
        be.beginGroup(&rt, 1, 0);
        LARGE_INTEGER t0, t1, fq;
        QueryPerformanceFrequency(&fq);
        QueryPerformanceCounter(&t0);
        for (int i = 0; i < kDraws; i++) {
          if (mode == 0 || mode == 2) {
            vsc[0][3] = float(i & 7) * 1e-4f;
            psc[1][0] = 0.1f + float(i & 15) * 1e-3f;
            s.vsConstantsVersion = s.psConstantsVersion = ++version;
          }
          if (mode == 0 || mode == 1) s.textures[0] = { DrawSetup::Texture::Tex, (i & 1) ? 0xC2080u : 0xC20C0u };
          else s.textures[0] = { DrawSetup::Texture::Tex, 0xC2080u };
          be.draw(s, 4, 2, false, 0, 0);
        }
        QueryPerformanceCounter(&t1);
        be.endGroup();
        be.finishFrame();
        best = std::min(best, double(t1.QuadPart - t0.QuadPart) / double(fq.QuadPart) / kDraws * 1e9);
      }
      static const char* kModes[4] = { "constants + textures change", "textures change", "constants change", "nothing changes" };
      printf("bench: game draw recording, pbr %d, %s: %.0f ns per draw (best of 5 x %d)\n", int(pbr), kModes[mode], best, kDraws);
    }
    be.destroy(rt);
  }
  be.destroyTexture(0xC2080);
  be.destroyTexture(0xC20C0);
  if (!vanilla) fail("pbr: with pbr off the shading is not the game's, or depends on the eye");
  if (!diffuse) fail("pbr: the diffuse of a material without specular is not the linear sum, or depends on the eye");
  if (!peak) fail("pbr: the specular peak on the glossy material with the eye above is not the model's");
  if (seamBad) fail("pbr: at the end of the specular fade the specular variant does not draw what the non-specular one does");
  if (std::abs(px - ex) > 1.0 || std::abs(py - ey) > 1.0) fail("pbr: the specular peak is not under the eye (view vector across triangles)");
}

}  // namespace pbrtest

// Terrain (lando.frag): SLS2100.vso (LandO's vertex shader) with replacements/lighting's SLS2092.pso (one layer):
// with pbr off the game's shading (sun + ambient on white), with pbr on the linear sum, diffuse only (the terrain has
// no specular material).
namespace pbrterrain {

static void run(Backend& be) {
  const char* dir = getenv("FNV_GAME_SHADERS");
  if (!dir || repltest::g_lightingPs.empty()) { printf("pbr terrain: skipped\n"); return; }
  const ShaderKey vsKey = 0xC3000, psKey = 0xC3040;
  bool haveVs = false, havePs = false;
  for (const Md5& m : sdpShaderMd5s(dir, "SLS2100.vso", { 13 })) haveVs = haveVs || be.createShader(vsKey, m.b.data());
  for (const Md5& m : sdpShaderMd5s(dir, "SLS2092.pso", { 13 })) havePs = havePs || be.createShader(psKey, m.b.data());
  if (!haveVs || !havePs) { fail("pbr terrain: SLS2100.vso / SLS2092.pso not in the shader pack"); return; }
  D3DRef fresh;
  if (!fresh.init()) { fail("pbr terrain: no D3D9 device"); return; }
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
  if (!texture(0xC3080, 255, 255, 255) || !texture(0xC30C0, 255, 128, 128)) { fail("pbr terrain: textures"); return; }
  // position, tangent, binormal, normal, texcoord0 (float2), colour (the tint), texcoord1 (layer weights), texcoord2.
  struct V { float p[3], t[3], b[3], n[3], uv[2], c[4], w[4], w2[4]; };
  auto corner = [](float x, float y) {
    return V{ { x, y, 0.5f }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 0, 0 }, { 1, 1, 1, 1 }, { 1, 0, 0, 1 }, { 0, 0, 0, 0 } };
  };
  const V quad[6] = { corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, -1), corner(1, 1), corner(-1, 1) };
  const BufKey vb = 0xC3100;
  be.createBuffer(vb, sizeof(quad), false);
  be.writeBuffer(vb, 0, quad, sizeof(quad), 0);
  const VertexElement ve[8] = { { 0, 0, 2, 0, 0 /*POSITION*/, 0 },  { 0, 12, 2, 0, 6 /*TANGENT*/, 0 },
                                { 0, 24, 2, 0, 7 /*BINORMAL*/, 0 }, { 0, 36, 2, 0, 3 /*NORMAL*/, 0 },
                                { 0, 48, 1, 0, 5 /*TEXCOORD*/, 0 }, { 0, 56, 3, 0, 10 /*COLOR*/, 0 },
                                { 0, 72, 3, 0, 5, 1 },              { 0, 88, 3, 0, 5, 2 } };
  const DeclKey decl = 0xC3140;
  be.createDecl(decl, ve, 8);
  static float vsc[256][4], psc[224][4];
  memset(vsc, 0, sizeof(vsc));
  memset(psc, 0, sizeof(psc));
  for (int i = 0; i < 4; i++) vsc[i][i] = 1.0f;
  const float fog[4] = { 1.0e6f, 1.0f, 1.0f, 0.0f };  // no fog
  memcpy(vsc[14], fog, 16);
  const float ambient[4] = { 0.1f, 0.1f, 0.1f, 1.0f }, sun[4] = { 0.8f, 0.7f, 0.6f, 1.0f }, dirUp[4] = { 0, 0, 1, 0 };
  const float eye[4] = { 0.5f, 0.0f, 0.6f, 1.0f };  // 0.1 above the quad, over clip (0.5, 0)
  memcpy(psc[1], ambient, 16);
  memcpy(psc[3], sun, 16);
  memcpy(psc[17], eye, 16);
  memcpy(psc[18], dirUp, 16);  // PSLightDir: the sun along +z
  const uint32_t W = 32, H = 32;
  uint64_t version = 9000;
  auto draw = [&](float pbr, std::vector<uint8_t>& img) {
    const ShaderReplacement::Constant params[2] = { { 223, { pbr, 0, 0, 0 } }, { 222, { 1.0f, 0, 0, 0 } } };
    for (const Md5& m : repltest::g_lightingPs) be.setShaderConstants(m.b.data(), params, 2);
    DrawSetup s;
    s.renderStates = rs;
    s.samplerStates = ss;
    s.textures[0] = { DrawSetup::Texture::Tex, 0xC3080 };
    s.textures[7] = { DrawSetup::Texture::Tex, 0xC30C0 };
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
    be.draw(s, 4 /*D3DPT_TRIANGLELIST*/, 2, false, 0, 0);
    be.endGroup();
    uint32_t vf = 0, ts = 0;
    if (!be.readback(rt, img, vf, ts) || ts != 4) fail("pbr terrain: readback");
    be.destroy(rt);
  };
  std::vector<uint8_t> van, pbr;
  draw(0.0f, van);
  draw(1.0f, pbr);
  be.destroyTexture(0xC3080);
  be.destroyTexture(0xC30C0);
  if (van.size() != W * H * 4 || pbr.size() != van.size()) { fail("pbr terrain: image sizes"); return; }
  const size_t c = ((H / 2) * W + W / 2) * 4;
  bool vanilla = true;
  for (int ch = 0; ch < 3; ch++) vanilla &= std::abs(int(van[c + 2 - ch]) - int(std::lround((sun[ch] + ambient[ch]) * 255.0f))) <= 2;
  bool linearSum = true;
  for (int ch = 0; ch < 3; ch++)
    linearSum &= std::abs(int(pbr[c + 2 - ch]) -
                          int(std::lround(pbrtest::toGamma(pbrtest::toLinear(sun[ch]) + pbrtest::toLinear(ambient[ch])) * 255.0f))) <= 2;
  printf("pbr terrain: centre vanilla %d %d %d, pbr %d %d %d\n", van[c + 2], van[c + 1], van[c], pbr[c + 2], pbr[c + 1], pbr[c]);
  if (!vanilla) fail("pbr terrain: with pbr off the shading is not the game's");
  if (!linearSum) fail("pbr terrain: with pbr on the shading is not the linear sum");
}

}  // namespace pbrterrain
