// Sun shadows end to end (included by backend_test.cpp): the plugin's cascade math (plugin/src/cascades.h, the same
// code sunshadows.cpp runs), casters drawn with camera-relative rows as the plugin submits them, and the effects'
// view reconstruction (fnvViewPos) into fnvShadow with the shadows effect's normal offset. A camera far from the
// origin looks over a ground square (which casts too, as terrain does) at a box standing on it; every ground pixel's
// shadow is compared with a ray cast toward the sun.
// Also the far cascade kept from an earlier frame (sunshadows.cpp [Shadows] FarInterval): a second box in the last
// cascade's range; the cascades drawn for a camera 40 units back (and a little turned), the frame submitted, then the
// near two drawn for the current camera with the far layer left as it was and its matrices rebuilt from its box
// (cascades::setMatrices, after cascades::covers accepts it). The same pixel check must pass; with the earlier camera's
// matrices used as they were (a negative control) the far box's shadow is misplaced and pixels must be wrong.
namespace shadowscene {
using namespace fnvvk::cascades;

static bool rayBox(D3 o, D3 d, D3 lo, D3 hi, double& tHit) {
  double t0 = 1e-3, t1 = 1e30;
  const double os[3] = { o.x, o.y, o.z }, ds[3] = { d.x, d.y, d.z }, ls[3] = { lo.x, lo.y, lo.z }, hs[3] = { hi.x, hi.y, hi.z };
  for (int a = 0; a < 3; a++) {
    if (std::fabs(ds[a]) < 1e-12) {
      if (os[a] < ls[a] || os[a] > hs[a]) return false;
      continue;
    }
    double ta = (ls[a] - os[a]) / ds[a], tb = (hs[a] - os[a]) / ds[a];
    if (ta > tb) std::swap(ta, tb);
    t0 = std::max(t0, ta);
    t1 = std::min(t1, tb);
    if (t0 > t1) return false;
  }
  tHit = t0;
  return true;
}

// mode 0: one frame, all cascades; 1: the far cascade kept from an earlier frame; 2: as 1 without rebuilding its
// matrices (control: must be wrong).
static void runMode(Backend& be, int mode) {
  // Probe: per pixel, the view ray to the ground plane (view-space normal n, camera height h), the shadows effect's
  // normal offset, fnvShadow. Red = visibility; 0.5 where the ray misses the ground.
  CreateDirectoryA("build\\fx_shadowscene", nullptr);
  CreateDirectoryA("build\\fx_shadowscene\\probe", nullptr);
  if (FILE* f = fopen("build/fx_shadowscene/probe/effect.ini", "wb")) {
    fputs("[Effect]\nPoint=world\n[Param.nx]\n[Param.ny]\n[Param.nz]\n[Param.h]\n[Pass.0]\nShader=probe.frag\nOutput=scene\n", f);
    fclose(f);
  }
  if (FILE* f = fopen("build/fx_shadowscene/probe/probe.frag", "wb")) {
    fputs("layout(location = 0) out vec4 c;\n"
          "void main() {\n"
          "  vec3 n = vec3(p_nx, p_ny, p_nz);\n"
          "  vec3 dir = fnvViewPos(fnvPixel(), 1.0);\n"
          "  float dn = dot(n, dir);\n"
          "  if (dn >= -1e-4) { c = vec4(0.5, 0, 0, 1); return; }\n"
          "  vec3 v = dir * (-p_h / dn);\n"
          "  c = vec4(fnvShadow(v + n * (1.0 + 0.002 * v.z)), 0, 0, 1);\n"
          "}\n", f);
    fclose(f);
  }
  std::vector<fx::Effect> probe = fx::loadEffects("build/fx_shadowscene", "build/fx_cache_test", [](const char* s) { printf("[fx] %s\n", s); });
  if (probe.size() != 1 || !probe[0].error.empty()) { fail("shadow scene: probe did not compile"); return; }
  const char* name = mode == 0 ? "shadow scene"
                     : mode == 1 ? "shadow scene, far cascade kept"
                                 : "shadow scene, far cascade kept, matrices not rebuilt (control)";
  Backend::EffectDesc d;
  for (const fx::Pass& p : probe[0].passes)
    d.passes.push_back({ { p.spv[0].data(), p.spv[1].data() }, { p.spv[0].size(), p.spv[1].size() }, p.output, uint32_t(p.blend) });
  const uint32_t id = be.createEffect(d);
  if (!id) { fail("shadow scene: probe not created"); return; }

  // Camera: far from the origin, 260 units up, looking north and down; z up, right-handed world.
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
  std::vector<Cascade> cs = build(cam, toSun, kCascades, kSize, 8000.0, 8000.0);
  if (cs.size() != kCascades) { fail("shadow scene: cascades not built"); return; }
  // The earlier frame's camera: mode 1 40 units back along the view, turned 0.3 degrees (a frame of movement the kept
  // cascade still covers); mode 2 (the control) 200 units to the side, turned 2 degrees, so stale matrices misplace the
  // far shadow by more than the pixel check's edge band (at that distance a pixel spans about 100 units of ground).
  Camera prevCam = cam;
  {
    const double a = (mode == 2 ? 2.0 : 0.3) * 3.14159265358979 / 180.0;
    prevCam.position = mode == 2 ? cam.position + cam.right * 200.0 : cam.position - cam.forward * 40.0;
    prevCam.forward = normalize(cam.forward * std::cos(a) + cam.right * std::sin(a));
    prevCam.right = normalize(cross(prevCam.forward, { 0, 0, 1 }));
    prevCam.up = cross(prevCam.right, prevCam.forward);
  }
  const std::vector<Cascade> prevCs = build(prevCam, toSun, kCascades, kSize, 8000.0, 8000.0);
  if (mode != 0) {
    const bool covered = covers(prevCs[kCascades - 1], cam, cs[kCascades - 2].split, cs[kCascades - 1].split, kSize, 2.0);
    if (mode == 1) printf("%s: the earlier far cascade covers this view's far slice: %s\n", name, covered ? "yes" : "NO");
    if (mode == 1 && !covered) { fail("%s: the earlier far cascade does not cover the view (the test needs it to)", name); return; }
    const float split = cs[kCascades - 1].split;
    cs[kCascades - 1] = prevCs[kCascades - 1];
    if (mode == 1) setMatrices(cs[kCascades - 1], cam);
    cs[kCascades - 1].split = split;
  }

  // Casters: a ground square (z = 0, 4000 wide) and a box (80 x 80 x 200) standing 600 units ahead, in object space
  // around their origins, placed by camera-relative rows (translation = origin - camera position).
  const D3 groundOrigin{ 12000.0, -33000.0, 0.0 }, boxOrigin{ 12000.0, -33400.0, 0.0 }, farBoxOrigin{ 12000.0, -31500.0, 0.0 };
  const float g = 2000.0f;
  const float ground[6][3] = { { -g, -g, 0 }, { g, -g, 0 }, { g, g, 0 }, { -g, -g, 0 }, { g, g, 0 }, { -g, g, 0 } };
  const float hx = 40.0f, hz = 200.0f;
  const float bv[8][3] = { { -hx, -hx, 0 }, { hx, -hx, 0 }, { hx, hx, 0 }, { -hx, hx, 0 },
                           { -hx, -hx, hz }, { hx, -hx, hz }, { hx, hx, hz }, { -hx, hx, hz } };
  const int faces[12][3] = { { 0, 1, 2 }, { 0, 2, 3 }, { 4, 6, 5 }, { 4, 7, 6 }, { 0, 4, 5 }, { 0, 5, 1 },
                             { 1, 5, 6 }, { 1, 6, 2 }, { 2, 6, 7 }, { 2, 7, 3 }, { 3, 7, 4 }, { 3, 4, 0 } };
  float box[36][3];
  for (int f = 0; f < 12; f++)
    for (int k = 0; k < 3; k++) memcpy(box[f * 3 + k], bv[faces[f][k]], 12);
  const BufKey vbGround = 0xBE000, vbBox = 0xBE040;
  be.createBuffer(vbGround, sizeof(ground), false);
  be.writeBuffer(vbGround, 0, ground, sizeof(ground), 0);
  be.createBuffer(vbBox, sizeof(box), false);
  be.writeBuffer(vbBox, 0, box, sizeof(box), 0);
  const VertexElement ve[1] = { { 0, 0, 2 /*FLOAT3*/, 0, 0 /*POSITION*/, 0 } };
  const DeclKey decl = 0xBE080;
  be.createDecl(decl, ve, 1);
  // The far box: the same box scaled 5 x 5 x 6 (400 x 400 x 1200), so its shadow covers pixels at its distance.
  constexpr float kFarXY = 5.0f, kFarZ = 6.0f;
  auto caster = [&](BufKey vb, uint32_t tris, D3 origin, const Camera& from, float sxy = 1.0f, float sz = 1.0f) {
    Backend::ShadowCaster c;
    c.streams[0] = vb;
    c.strides[0] = 12;
    c.streamCount = 1;
    c.decl = decl;
    c.d3dPrimitive = 4;
    c.primitiveCount = tris;
    const D3 t = origin - from.position;
    const float w[12] = { sxy, 0, 0, float(t.x), 0, sxy, 0, float(t.y), 0, 0, sz, float(t.z) };
    memcpy(c.world, w, sizeof(w));
    return c;
  };
  const Backend::ShadowCaster groundCaster = caster(vbGround, 2, groundOrigin, cam), boxCaster = caster(vbBox, 12, boxOrigin, cam),
                              farBoxCaster = caster(vbBox, 12, farBoxOrigin, cam, kFarXY, kFarZ);
  if (mode != 0) {
    // The earlier frame: every cascade for its camera (casters relative to it), submitted.
    if (!be.beginShadowMaps(kCascades, kSize)) fail("shadow scene: beginShadowMaps failed");
    for (uint32_t i = 0; i < kCascades; i++) {
      be.beginShadowCascade(i, prevCs[i].viewProj);
      be.drawShadowCaster(caster(vbGround, 2, groundOrigin, prevCam));
      be.drawShadowCaster(caster(vbBox, 12, boxOrigin, prevCam));
      be.drawShadowCaster(caster(vbBox, 12, farBoxOrigin, prevCam, kFarXY, kFarZ));
    }
    be.endShadowMaps();
    be.finishFrame();
  }

  const uint32_t W = 320, H = 180;
  const RtKey rt = newKey();
  be.attach(rt, { W, H, kD3DFMT_A8R8G8B8, 1, false });
  be.beginGroup(&rt, 1, 0);
  int32_t full[4] = { 0, 0, int32_t(W), int32_t(H) };
  be.clear(kClearTarget, 0xFF000000, 1.0f, 0, full);
  if (!be.beginShadowMaps(kCascades, kSize)) fail("shadow scene: beginShadowMaps failed");
  for (uint32_t i = 0; i < kCascades; i++) {
    if (mode != 0 && i == kCascades - 1) continue;  // kept from the earlier frame
    be.beginShadowCascade(i, cs[i].viewProj);
    be.drawShadowCaster(groundCaster);
    be.drawShadowCaster(boxCaster);
    be.drawShadowCaster(farBoxCaster);
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
  const D3 zUp{ 0, 0, 1 };
  float params[4][4] = { { float(dot(zUp, cam.right)) }, { float(dot(zUp, cam.up)) }, { float(dot(zUp, cam.forward)) }, { float(cam.position.z) } };
  if (!be.runEffect(id, fr, &params[0][0], 4)) fail("shadow scene: probe run failed");
  be.endGroup();
  std::vector<uint8_t> px;
  uint32_t vf = 0, ts = 0;
  be.readback(rt, px, vf, ts);

  // Expected, per pixel: the ground point under the view ray, shadowed if the ray toward the sun hits the box.
  // Pixels whose view ray hits the box first, or within 4 texels (of the cascade used) of a shadow edge, are skipped.
  const D3 lo{ boxOrigin.x - hx, boxOrigin.y - hx, 0.0 }, hi{ boxOrigin.x + hx, boxOrigin.y + hx, hz };
  const D3 flo{ farBoxOrigin.x - hx * kFarXY, farBoxOrigin.y - hx * kFarXY, 0.0 },
      fhi{ farBoxOrigin.x + hx * kFarXY, farBoxOrigin.y + hx * kFarXY, hz * kFarZ };
  int checked = 0, wrong = 0, expectShadow = 0, gotShadow = 0, firstX = -1, firstY = -1, farChecked = 0, farShadow = 0;
  float firstGot = 0;
  bool firstWant = false;
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++) {
      const double u = (x + 0.5) / W, v = (y + 0.5) / H;
      const D3 dirV{ frustum[0] + (frustum[1] - frustum[0]) * u, frustum[2] + (frustum[3] - frustum[2]) * v, 1.0 };
      const D3 dirW = cam.right * dirV.x + cam.up * dirV.y + cam.forward * dirV.z;
      if (dirW.z >= -1e-4) continue;
      const D3 p = cam.position + dirW * (-cam.position.z / dirW.z);
      double tb;
      if (rayBox(cam.position, dirW, lo, hi, tb) && tb < (-cam.position.z / dirW.z)) continue;
      if (rayBox(cam.position, dirW, flo, fhi, tb) && tb < (-cam.position.z / dirW.z)) continue;
      const double viewZ = dot(p - cam.position, cam.forward);
      uint32_t c = 0;
      while (c + 1 < kCascades && viewZ > cs[c].split) c++;
      if (viewZ > cs[kCascades - 1].split * 0.85) continue;  // the last cascade's fade
      const double texel = 2.0 * cs[c].radius / kSize;
      auto shadowed = [&](D3 q) { double t; return rayBox(q, toSun, lo, hi, t) || rayBox(q, toSun, flo, fhi, t); };
      const bool want = shadowed(p);
      bool edge = false;
      for (int k = 0; k < 8 && !edge; k++) {
        const double a = k * 0.7853981;
        const D3 q = p + D3{ std::cos(a), std::sin(a), 0.0 } * (4.0 * texel);
        edge = shadowed(q) != want;
      }
      if (edge) continue;
      const float got = px[(size_t(y) * W + x) * 4 + 2] / 255.0f;
      checked++;
      expectShadow += want;
      if (c == kCascades - 1) {
        farChecked++;
        farShadow += want;
      }
      gotShadow += got < 0.5f;
      if ((got < 0.5f) != want) {
        if (firstX < 0) { firstX = int(x); firstY = int(y); firstGot = got; firstWant = want; }
        wrong++;
      }
    }
  printf("%s: %d ground pixels checked (%d in the far cascade, %d of them in shadow), %d in shadow expected, %d found; %d wrong%s\n",
         name, checked, farChecked, farShadow, expectShadow, gotShadow, wrong,
         wrong ? "" : " (box shadows placed and shaped as the ray cast, no self-shadowing on the ground)");
  if (mode == 2) {
    if (!wrong) fail("%s: the stale matrices gave no wrong pixels: the kept-cascade check cannot see a misplaced shadow", name);
  } else if (wrong) {
    fail("%s: %d of %d pixels wrong (first (%d,%d): visibility %.2f, expected %s)", name, wrong, checked, firstX, firstY,
         firstGot, firstWant ? "shadow" : "lit");
  }
  if (expectShadow < 50 || farShadow < 20)
    fail("%s: the boxes' shadows are not in view (%d pixels, %d far)", name, expectShadow, farShadow);
  be.destroyEffect(id);
  be.destroyBuffer(vbGround);
  be.destroyBuffer(vbBox);
  be.destroyDecl(decl);
  be.destroy(rt);
}

static void run(Backend& be) {
  for (int mode = 0; mode < 3; mode++) runMode(be, mode);
}

}  // namespace shadowscene
