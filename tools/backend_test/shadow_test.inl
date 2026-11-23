// Sun shadow maps (included by backend_test.cpp): a caster quad over the left half of a cascade, drawn through
// drawShadowCaster from its own vertex buffer and declaration; then fnvShadow (the effect header) samples it for view
// points behind the caster (left: shadow, right: lit) and in front of it (lit).
namespace shadowtest {

static void run(Backend& be) {
  // A probe effect: visibility of view point (x across the target, 0, z) -> red.
  CreateDirectoryA("build\\fx_shadowprobe", nullptr);
  CreateDirectoryA("build\\fx_shadowprobe\\probe", nullptr);
  if (FILE* f = fopen("build/fx_shadowprobe/probe/effect.ini", "wb")) {
    fputs("[Effect]\nPoint=world\n[Param.z]\nDefault=0.5\n[Pass.0]\nShader=probe.frag\nOutput=scene\n", f);
    fclose(f);
  }
  if (FILE* f = fopen("build/fx_shadowprobe/probe/probe.frag", "wb")) {
    fputs("layout(location = 0) out vec4 c;\n"
          "void main() {\n"
          "  vec2 u = fnvPixel() * fnv.target.zw;\n"
          "  c = vec4(fnvShadow(vec3(u.x * 2.0 - 1.0, 0.0, p_z)), 0.0, 0.0, 1.0);\n"
          "}\n", f);
    fclose(f);
  }
  std::vector<fx::Effect> probe = fx::loadEffects("build/fx_shadowprobe", "build/fx_cache_test", [](const char* s) { printf("[fx] %s\n", s); });
  if (probe.size() != 1 || !probe[0].error.empty()) { fail("shadows: probe effect did not compile"); return; }
  Backend::EffectDesc d;
  for (const fx::Pass& p : probe[0].passes)
    d.passes.push_back({ { p.spv[0].data(), p.spv[1].data() }, { p.spv[0].size(), p.spv[1].size() }, p.output, uint32_t(p.blend) });
  const uint32_t id = be.createEffect(d);
  if (!id) { fail("shadows: probe effect not created"); return; }

  // Caster: a quad at world z = 0.3, x in [-1, 0], y in [-1, 1] (triangle list, FLOAT3 positions).
  const float quad[6][3] = { { -1, -1, 0.3f }, { 0, -1, 0.3f }, { 0, 1, 0.3f }, { -1, -1, 0.3f }, { 0, 1, 0.3f }, { -1, 1, 0.3f } };
  const BufKey vb = 0xAE000;
  be.createBuffer(vb, sizeof(quad), false);
  be.writeBuffer(vb, 0, quad, sizeof(quad), 0);
  const VertexElement ve[1] = { { 0, 0, 2 /*FLOAT3*/, 0, 0 /*POSITION*/, 0 } };
  const DeclKey decl = 0xAE040;
  be.createDecl(decl, ve, 1);

  // One cascade with scale and translation in every row (row-major, clip = M * (world, 1)): a transposed or misread
  // matrix on either side (the caster push constant, the effects' uniform block) moves the shadow.
  // x: 0.5 * x + 0.25, y: 0.5 * y - 0.1, z: 0.8 * z + 0.05. The same matrix maps the probe's view points.
  const float identity[16] = { 0.5f, 0, 0, 0.25f, 0, 0.5f, 0, -0.1f, 0, 0, 0.8f, 0.05f, 0, 0, 0, 1 };
  const RtKey rt = newKey();
  be.attach(rt, { 64, 64, kD3DFMT_A8R8G8B8, 1, false });
  be.beginGroup(&rt, 1, 0);  // a group open across the shadow pass, as in game
  int32_t full[4] = { 0, 0, 64, 64 };
  be.clear(kClearTarget, 0xFF00FF00, 1.0f, 0, full);
  if (!be.beginShadowMaps(1, 512)) fail("shadows: beginShadowMaps failed");
  be.beginShadowCascade(0, identity);
  Backend::ShadowCaster c;
  c.streams[0] = vb;
  c.strides[0] = 12;
  c.streamCount = 1;
  c.decl = decl;
  c.d3dPrimitive = 4;
  c.primitiveCount = 2;
  const float world[12] = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 };
  memcpy(c.world, world, sizeof(world));
  const uint64_t before = be.stats().shadowCasters;
  be.drawShadowCaster(c);
  be.endShadowMaps();
  if (be.stats().shadowCasters != before + 1) fail("shadows: the caster was not drawn");
  // CPU cost of recording a sun shadow caster (BENCH=1): 3 cascades of 2048, each caster with its own world transform.
  if (getenv("BENCH")) {
    constexpr int kCasters = 5000;
    double best = 1e30;
    for (int rep = 0; rep < 5; rep++) {
      LARGE_INTEGER t0, t1, fq;
      QueryPerformanceFrequency(&fq);
      if (!be.beginShadowMaps(3, 2048)) break;
      QueryPerformanceCounter(&t0);
      for (uint32_t k = 0; k < 3; k++) {
        be.beginShadowCascade(k, identity);
        for (int i = 0; i < kCasters / 3; i++) {
          c.world[3] = float(i & 15) * 0.01f;
          be.drawShadowCaster(c);
        }
      }
      QueryPerformanceCounter(&t1);
      be.endShadowMaps();
      be.finishFrame();
      best = std::min(best, double(t1.QuadPart - t0.QuadPart) / double(fq.QuadPart) / (3 * (kCasters / 3)) * 1e9);
    }
    printf("bench: sun shadow caster recording: %.0f ns per caster draw (best of 5)\n", best);
  }

  Backend::EffectFrame fr{};
  const float port[4] = { 0, 1, 1, 0 }, frustum[4] = { -1, 1, 1, -1 };
  memcpy(fr.port, port, 16);
  memcpy(fr.frustum, frustum, 16);
  fr.nearZ = 1;
  fr.farZ = 2;
  memcpy(fr.shadowFromView[0], identity, 64);
  fr.shadowSplits[0] = 100.0f;
  fr.sunView[2] = 1.0f;
  fr.shadowCascades = 1;
  float results[2][2];  // [behind, in front][left, right]
  const float zs[2] = { 0.5f, 0.1f };
  for (int zi = 0; zi < 2; zi++) {
    float params[1][4] = { { zs[zi], 0, 0, 0 } };
    if (!be.runEffect(id, fr, &params[0][0], 1)) fail("shadows: probe run failed");
    be.endGroup();
    std::vector<uint8_t> px;
    uint32_t vf = 0, ts = 0;
    be.readback(rt, px, vf, ts);
    results[zi][0] = px.size() >= 64 * 64 * 4 ? px[(32 * 64 + 12) * 4 + 2] / 255.0f : -1.0f;  // red of BGRA
    results[zi][1] = px.size() >= 64 * 64 * 4 ? px[(32 * 64 + 52) * 4 + 2] / 255.0f : -1.0f;
    be.beginGroup(&rt, 1, 0);
  }
  be.endGroup();
  printf("shadows: behind the caster: left %.2f right %.2f | in front: left %.2f right %.2f (1 lit, 0 shadow)\n",
         results[0][0], results[0][1], results[1][0], results[1][1]);
  if (!(results[0][0] < 0.05f)) fail("shadows: no shadow behind the caster");
  if (!(results[0][1] > 0.95f)) fail("shadows: shadow beside the caster");
  if (!(results[1][0] > 0.95f && results[1][1] > 0.95f)) fail("shadows: shadow in front of the caster");

  // Strips in one call (ShadowCaster::stripLengths) against one call per strip: two quads as strips of 4 indices, x in
  // [-1, -0.5] and [0, 0.5] at z 0.3, with a strip of 2 (skipped, but its indices counted) between them. The shadow seen
  // behind them across a row of the target must be the same, with the two bands.
  {
    const float v[8][3] = { { -1, -1, 0.3f }, { -1, 1, 0.3f }, { -0.5f, -1, 0.3f }, { -0.5f, 1, 0.3f },
                            { 0, -1, 0.3f },  { 0, 1, 0.3f },  { 0.5f, -1, 0.3f },  { 0.5f, 1, 0.3f } };
    const uint16_t idx[10] = { 0, 1, 2, 3, 0, 0, 4, 5, 6, 7 };
    const uint16_t lengths[3] = { 4, 2, 4 };
    const BufKey svb = 0xAE100, sib = 0xAE140;
    be.createBuffer(svb, sizeof(v), false);
    be.writeBuffer(svb, 0, v, sizeof(v), 0);
    be.createBuffer(sib, sizeof(idx), true);
    be.writeBuffer(sib, 0, idx, sizeof(idx), 0);
    Backend::ShadowCaster sc;
    sc.streams[0] = svb;
    sc.strides[0] = 12;
    sc.streamCount = 1;
    sc.indices = sib;
    sc.decl = decl;
    sc.d3dPrimitive = 5;
    memcpy(sc.world, world, sizeof(world));
    std::vector<uint8_t> rows[2];
    for (int batched = 0; batched < 2; batched++) {
      be.beginGroup(&rt, 1, 0);
      be.clear(kClearTarget, 0xFF00FF00, 1.0f, 0, full);
      if (!be.beginShadowMaps(1, 512)) fail("shadows: strips: beginShadowMaps failed");
      be.beginShadowCascade(0, identity);
      if (batched) {
        sc.start = 0;
        sc.primitiveCount = 0;
        sc.stripLengths = lengths;
        sc.stripCount = 3;
        be.drawShadowCaster(sc);
      } else {
        sc.stripLengths = nullptr;
        sc.stripCount = 0;
        uint32_t first = 0;
        for (uint16_t n : lengths) {
          if (n >= 3) {
            sc.start = first;
            sc.primitiveCount = n - 2u;
            be.drawShadowCaster(sc);
          }
          first += n;
        }
      }
      be.endShadowMaps();
      float params[1][4] = { { 0.5f, 0, 0, 0 } };
      if (!be.runEffect(id, fr, &params[0][0], 1)) fail("shadows: strips: probe run failed");
      be.endGroup();
      std::vector<uint8_t> px;
      uint32_t vf = 0, ts = 0;
      be.readback(rt, px, vf, ts);
      for (uint32_t x = 0; x < 64 && px.size() >= 64 * 64 * 4; x++) rows[batched].push_back(px[(32 * 64 + x) * 4 + 2]);
    }
    int shadowed = 0;
    for (uint8_t r : rows[1]) shadowed += r < 13;
    printf("shadows: strips in one call vs one call each: %s, %d of 64 texels of the row shadowed\n",
           rows[0] == rows[1] && !rows[0].empty() ? "same" : "DIFFERENT", shadowed);
    if (rows[0] != rows[1] || rows[0].empty()) fail("shadows: strips drawn in one call differ from one call each");
    if (shadowed < 8 || shadowed > 40) fail("shadows: strips: the two bands are not in the shadow (%d texels)", shadowed);
    // CPU cost of a caster of 5 strips (BENCH=1): one call per strip against one call with the strips.
    if (getenv("BENCH")) {
      const uint16_t five[5] = { 4, 4, 4, 4, 4 };
      const uint16_t idx5[20] = { 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3 };
      be.writeBuffer(sib, 0, idx5, sizeof(idx5), 0);
      for (int batched = 0; batched < 2; batched++) {
        double best = 1e30;
        constexpr int kCasters = 1500;
        for (int rep = 0; rep < 5; rep++) {
          if (!be.beginShadowMaps(1, 512)) break;
          be.beginShadowCascade(0, identity);
          LARGE_INTEGER t0, t1, fq;
          QueryPerformanceFrequency(&fq);
          QueryPerformanceCounter(&t0);
          for (int i = 0; i < kCasters; i++) {
            sc.world[3] = float(i & 15) * 0.01f;
            if (batched) {
              sc.start = 0;
              sc.stripLengths = five;
              sc.stripCount = 5;
              be.drawShadowCaster(sc);
            } else {
              sc.stripLengths = nullptr;
              sc.stripCount = 0;
              for (uint32_t k = 0; k < 5; k++) {
                sc.start = 4 * k;
                sc.primitiveCount = 2;
                be.drawShadowCaster(sc);
              }
            }
          }
          QueryPerformanceCounter(&t1);
          be.endShadowMaps();
          be.finishFrame();
          best = std::min(best, double(t1.QuadPart - t0.QuadPart) / double(fq.QuadPart) / kCasters * 1e9);
        }
        printf("bench: sun shadow caster of 5 strips, %s: %.0f ns per caster (best of 5)\n",
               batched ? "one call" : "one call per strip", best);
      }
    }
    be.destroyBuffer(svb);
    be.destroyBuffer(sib);
  }
  be.destroyEffect(id);
  be.destroyBuffer(vb);
  be.destroyDecl(decl);
  be.destroy(rt);
}

}  // namespace shadowtest
