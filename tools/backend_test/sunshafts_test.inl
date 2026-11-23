// Sun shafts (included by backend_test.cpp, after shadow_scene_test.inl): effects/sunshafts over the shadow scene's
// cascades and casters (a ground square 4000 wide and a 80 x 80 x 200 box), every pixel open sky (depth cleared to the
// far plane), so each view ray is marched over the whole shadow range. The oracle integrates each ray exactly (2000
// steps): visibility 0 where the segment toward the sun hits the box or the ground square (rays dipping under the ground
// are shadowed from where they pass beneath it), 1 elsewhere; in-scattering sigma * exp(-sigma t) dt times the
// Henyey-Greenstein phase, times intensity and the sun's colour.
// Checks: the GPU light per pixel matches the oracle (median and 90th-percentile relative error), on rows from fully lit
// to mostly shadowed rays, so the march places the ground's shadow along the rays where the cascades do.
namespace sunshaftstest {

static float halfToFloat(uint16_t h) {
  const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
  float v = e == 0 ? std::ldexp(float(m), -24) : e == 31 ? INFINITY : std::ldexp(float(m | 1024), int(e) - 25);
  return s ? -v : v;
}

static void run(Backend& be) {
  using namespace cascades;
  std::vector<fx::Effect> all = fx::loadEffects("effects", "build/fx_cache_test", [](const char* s) { printf("[fx] %s\n", s); });
  const fx::Effect* e = nullptr;
  for (const fx::Effect& x : all)
    if (x.id == "sunshafts") e = &x;
  if (!e || !e->error.empty()) { fail("sun shafts: effects/sunshafts did not load%s%s", e ? ": " : "", e ? e->error.c_str() : ""); return; }
  Backend::EffectDesc d;
  for (const fx::Image& im : e->images) d.images.push_back({ im.vkFormat, im.scale });
  for (const fx::Pass& p : e->passes)
    d.passes.push_back({ { p.spv[0].data(), p.spv[1].data() }, { p.spv[0].size(), p.spv[1].size() }, p.output, uint32_t(p.blend) });
  const uint32_t id = be.createEffect(d);
  if (!id) { fail("sun shafts: effect not created"); return; }

  // The shadow scene (shadow_scene_test.inl): camera, sun, cascades, casters.
  Camera cam;
  cam.position = { 12000.0, -34000.0, 260.0 };
  cam.forward = normalize({ 0.0, 1.0, -0.3 });
  cam.right = normalize(cross(cam.forward, { 0, 0, 1 }));
  cam.up = cross(cam.right, cam.forward);
  const float frustum[4] = { -0.7f, 0.7f, 0.4f, -0.4f };
  memcpy(cam.frustum, frustum, sizeof(frustum));
  cam.nearZ = 5.0;
  cam.farZ = 300000.0;
  const D3 toSun = normalize({ 0.45, -0.35, 0.82 });
  const uint32_t kCascades = 3, kSize = 2048;
  const std::vector<Cascade> cs = build(cam, toSun, kCascades, kSize, 8000.0, 8000.0);
  if (cs.size() != kCascades) { fail("sun shafts: cascades not built"); return; }
  const D3 groundOrigin{ 12000.0, -33000.0, 0.0 }, boxOrigin{ 12000.0, -33400.0, 0.0 };
  const float g = 2000.0f, hx = 40.0f, hz = 200.0f;
  const float ground[6][3] = { { -g, -g, 0 }, { g, -g, 0 }, { g, g, 0 }, { -g, -g, 0 }, { g, g, 0 }, { -g, g, 0 } };
  const float bv[8][3] = { { -hx, -hx, 0 }, { hx, -hx, 0 }, { hx, hx, 0 }, { -hx, hx, 0 },
                           { -hx, -hx, hz }, { hx, -hx, hz }, { hx, hx, hz }, { -hx, hx, hz } };
  const int faces[12][3] = { { 0, 1, 2 }, { 0, 2, 3 }, { 4, 6, 5 }, { 4, 7, 6 }, { 0, 4, 5 }, { 0, 5, 1 },
                             { 1, 5, 6 }, { 1, 6, 2 }, { 2, 6, 7 }, { 2, 7, 3 }, { 3, 7, 4 }, { 3, 4, 0 } };
  float box[36][3];
  for (int f = 0; f < 12; f++)
    for (int k = 0; k < 3; k++) memcpy(box[f * 3 + k], bv[faces[f][k]], 12);
  const BufKey vbGround = 0xBF000, vbBox = 0xBF040;
  be.createBuffer(vbGround, sizeof(ground), false);
  be.writeBuffer(vbGround, 0, ground, sizeof(ground), 0);
  be.createBuffer(vbBox, sizeof(box), false);
  be.writeBuffer(vbBox, 0, box, sizeof(box), 0);
  const VertexElement ve[1] = { { 0, 0, 2 /*FLOAT3*/, 0, 0 /*POSITION*/, 0 } };
  const DeclKey decl = 0xBF080;
  be.createDecl(decl, ve, 1);
  auto caster = [&](BufKey vb, uint32_t tris, D3 origin) {
    Backend::ShadowCaster c;
    c.streams[0] = vb;
    c.strides[0] = 12;
    c.streamCount = 1;
    c.decl = decl;
    c.d3dPrimitive = 4;
    c.primitiveCount = tris;
    const D3 t = origin - cam.position;
    const float w[12] = { 1, 0, 0, float(t.x), 0, 1, 0, float(t.y), 0, 0, 1, float(t.z) };
    memcpy(c.world, w, sizeof(w));
    return c;
  };

  const uint32_t W = 320, H = 180;
  const RtKey rt = newKey(), depth = newKey();
  be.attach(rt, { W, H, kD3DFMT_A16B16G16R16F, 1, false });
  be.attach(depth, { W, H, kD3DFMT_D24S8, 1, true });
  be.beginGroup(&rt, 1, depth);
  int32_t full[4] = { 0, 0, int32_t(W), int32_t(H) };
  be.clear(kClearTarget | kClearZBuffer, 0x00000000, 1.0f, 0, full);
  if (!be.beginShadowMaps(kCascades, kSize)) fail("sun shafts: beginShadowMaps failed");
  for (uint32_t i = 0; i < kCascades; i++) {
    be.beginShadowCascade(i, cs[i].viewProj);
    be.drawShadowCaster(caster(vbGround, 2, groundOrigin));
    be.drawShadowCaster(caster(vbBox, 12, boxOrigin));
  }
  be.endShadowMaps();

  Backend::EffectFrame fr{};
  const float port[4] = { 0, 1, 1, 0 };
  memcpy(fr.port, port, 16);
  memcpy(fr.frustum, frustum, 16);
  fr.nearZ = float(cam.nearZ);
  fr.farZ = float(cam.farZ);
  for (uint32_t i = 0; i < kCascades; i++) {
    memcpy(fr.shadowFromView[i], cs[i].fromView, 64);
    fr.shadowSplits[i] = cs[i].split;
  }
  fr.sunView[0] = float(dot(toSun, cam.right));
  fr.sunView[1] = float(dot(toSun, cam.up));
  fr.sunView[2] = float(dot(toSun, cam.forward));
  fr.shadowCascades = kCascades;
  fr.sunColor[0] = fr.sunColor[1] = fr.sunColor[2] = 1.0f;
  // intensity, density, anisotropy, steps (effect.ini order)
  const float intensity = 4.0f, density = 2.0f, aniso = 0.5f, steps = 96.0f;
  float params[4][4] = { { intensity }, { density }, { aniso }, { steps } };
  for (size_t i = 0; i < e->params.size() && i < 4; i++) {
    static const char* names[4] = { "intensity", "density", "anisotropy", "steps" };
    if (e->params[i].name != names[i]) { fail("sun shafts: parameter %zu is '%s', expected '%s'", i, e->params[i].name.c_str(), names[i]); return; }
  }
  if (!be.runEffect(id, fr, &params[0][0], 4)) fail("sun shafts: run failed");
  be.endGroup();
  std::vector<uint8_t> px;
  uint32_t vf = 0, ts = 0;
  if (!be.readback(rt, px, vf, ts) || ts != 8) { fail("sun shafts: readback (texel size %u)", ts); return; }

  const D3 lo{ boxOrigin.x - hx, boxOrigin.y - hx, 0.0 }, hi{ boxOrigin.x + hx, boxOrigin.y + hx, hz };
  auto lit = [&](D3 p) {
    double t;
    if (shadowscene::rayBox(p, toSun, lo, hi, t)) return false;
    if (p.z < 0.0) {  // under the ground plane: shadowed where the way up to the sun crosses the ground square
      const D3 q = p + toSun * (-p.z / toSun.z);
      if (std::abs(q.x - groundOrigin.x) <= g && std::abs(q.y - groundOrigin.y) <= g) return false;
    }
    return true;
  };
  const double sigma = density * 1.0e-4, gg = aniso;
  const double range = cs[kCascades - 1].split;
  std::vector<double> rel;
  double maxWant = 0.0, minWant = 1e9;
  for (uint32_t y = 4; y < H; y += 8)
    for (uint32_t x = 4; x < W; x += 8) {
      const double u = (x + 0.5) / W, v = (y + 0.5) / H;
      const D3 ray{ frustum[0] + (frustum[1] - frustum[0]) * u, frustum[2] + (frustum[3] - frustum[2]) * v, 1.0 };
      const double len = std::sqrt(dot(ray, ray));
      const D3 dirW = (cam.right * ray.x + cam.up * ray.y + cam.forward * ray.z) * (1.0 / len);
      const double cosT = (ray.x * fr.sunView[0] + ray.y * fr.sunView[1] + ray.z * fr.sunView[2]) / len;
      const double phase = (1 - gg * gg) / (12.566371 * std::pow(1 + gg * gg - 2 * gg * cosT, 1.5));
      const int N = 2000;
      const double dz = range / N;
      double sum = 0.0;
      for (int i = 0; i < N; i++) {
        const double z = (i + 0.5) * dz, t = z * len;
        if (lit(cam.position + dirW * t)) sum += std::exp(-sigma * t);
      }
      const double want = intensity * sum * sigma * dz * len * phase;
      const uint16_t* t = reinterpret_cast<const uint16_t*>(&px[(size_t(y) * W + x) * 8]);
      const double got = halfToFloat(t[0]);  // R
      maxWant = std::max(maxWant, want);
      minWant = std::min(minWant, want);
      rel.push_back(std::abs(got - want) / std::max(want, 1e-4));
    }
  std::sort(rel.begin(), rel.end());
  const double median = rel[rel.size() / 2], p90 = rel[rel.size() * 9 / 10];
  printf("sun shafts: %zu pixels vs the exact integral: median error %.1f%%, 90th percentile %.1f%% (oracle range %.4f .. %.4f)\n",
         rel.size(), median * 100, p90 * 100, minWant, maxWant);
  if (median > 0.03 || p90 > 0.10) fail("sun shafts: the light does not follow the cascades' shadowing");
  if (maxWant < 2.0 * minWant) fail("sun shafts: the scene does not vary enough to test");
  be.destroyEffect(id);
  be.destroyBuffer(vbGround);
  be.destroyBuffer(vbBox);
  be.destroyDecl(decl);
  be.destroy(rt);
  be.destroy(depth);
}

}  // namespace sunshaftstest
