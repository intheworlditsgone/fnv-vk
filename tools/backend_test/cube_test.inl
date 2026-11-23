// Rendered cube maps (Backend::attachCube; included by backend_test.cpp after sunshafts_test.inl, whose halfToFloat it
// uses): six RGBA16F face targets cleared through ordinary groups, then read back from the cube texture.
// - every face holds its own clear colour at level 0, and the cube has the full mip chain (levels 0);
// - a face cleared half red, half blue averages to (0.5, 0, 0.5) at the 1x1 level (the mips are built from the faces);
// - after only one face is drawn again, that face changes and the others keep their contents (per-face updates);
// - destroying the faces destroys the cube.
namespace cubetest {

static void run(Backend& be) {
  constexpr uint32_t kSize = 16, kLevels = 5;
  RtKey faces[6];
  for (RtKey& f : faces) {
    f = newKey();
    be.attach(f, { kSize, kSize, kD3DFMT_A16B16G16R16F, 1, false });
  }
  const TexKey cube = 0xC0BE0000;
  be.attachCube(cube, faces, 0);
  // Face colours (ARGB): +X red, -X green, +Y blue, -Y yellow, +Z cyan, -Z magenta.
  const uint32_t colors[6] = { 0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFF00, 0xFF00FFFF, 0xFFFF00FF };
  int32_t full[4] = { 0, 0, int32_t(kSize), int32_t(kSize) };
  auto clearFace = [&](uint32_t f, uint32_t argb) {
    be.beginGroup(&faces[f], 1, 0);
    be.clear(kClearTarget, argb, 1.0f, 0, full);
    be.endGroup();
  };
  for (uint32_t f = 0; f < 6; f++) clearFace(f, colors[f]);
  if (!be.updateCube(cube)) { fail("cube: updateCube failed"); return; }
  auto texel = [&](uint32_t f, uint32_t level, uint32_t x, uint32_t y, float out[4]) {
    std::vector<uint8_t> d;
    if (!be.readbackTexture(cube, f, level, d)) return false;
    const uint32_t w = std::max(kSize >> level, 1u);
    if (d.size() != size_t(w) * w * 8) return false;
    const uint16_t* h = reinterpret_cast<const uint16_t*>(d.data()) + (size_t(y) * w + x) * 4;
    for (int c = 0; c < 4; c++) out[c] = sunshaftstest::halfToFloat(h[c]);
    return true;
  };
  auto is = [](const float v[4], uint32_t argb) {
    const float want[3] = { float((argb >> 16) & 255) / 255.0f, float((argb >> 8) & 255) / 255.0f, float(argb & 255) / 255.0f };
    for (int c = 0; c < 3; c++)
      if (std::abs(v[c] - want[c]) > 1e-3f) return false;
    return true;
  };
  uint32_t wrong = 0;
  float v[4];
  for (uint32_t f = 0; f < 6; f++) {
    if (!texel(f, 0, 3, 5, v)) { fail("cube: readback of face %u failed", f); return; }
    if (!is(v, colors[f])) wrong++;
    if (!texel(f, kLevels - 1, 0, 0, v) || !is(v, colors[f])) wrong++;
  }
  std::vector<uint8_t> beyond;
  const bool fullChain = !be.readbackTexture(cube, 0, kLevels, beyond);  // exactly kLevels levels
  // Face 0 half red (x < 8), half blue: the 1x1 level is their average.
  const int32_t left[4] = { 0, 0, int32_t(kSize / 2), int32_t(kSize) }, right[4] = { int32_t(kSize / 2), 0, int32_t(kSize), int32_t(kSize) };
  be.beginGroup(&faces[0], 1, 0);
  be.clear(kClearTarget, 0xFFFF0000, 1.0f, 0, left);
  be.clear(kClearTarget, 0xFF0000FF, 1.0f, 0, right);
  be.endGroup();
  clearFace(3, 0xFFFFFFFF);
  be.updateCube(cube);
  float avg[4], f3[4], f2[4];
  const bool readOk = texel(0, kLevels - 1, 0, 0, avg) && texel(3, 0, 0, 0, f3) && texel(2, 0, 0, 0, f2);
  const bool mean = readOk && std::abs(avg[0] - 0.5f) < 2e-3f && std::abs(avg[1]) < 2e-3f && std::abs(avg[2] - 0.5f) < 2e-3f;
  const bool update = readOk && is(f3, 0xFFFFFFFF) && is(f2, colors[2]);
  printf("cube: %u wrong face texels, full chain %d, 1x1 of the split face %.3f %.3f %.3f, redrawn face %.2f %.2f %.2f, "
         "untouched face %.2f %.2f %.2f\n", wrong, fullChain, avg[0], avg[1], avg[2], f3[0], f3[1], f3[2], f2[0], f2[1], f2[2]);
  if (wrong) fail("cube: faces do not hold their clear colours (level 0 or the last level)");
  if (!fullChain) fail("cube: levels 0 did not give the full chain");
  if (!mean) fail("cube: the mips are not built from the face contents");
  if (!update) fail("cube: redrawing one face did not update it alone");
  for (RtKey f : faces) be.destroy(f);
  if (be.destroyTexture(cube)) fail("cube: the cube outlived its faces");
}

}  // namespace cubetest

// The environment's irradiance harmonics (Backend::projectEnvironment), against environments with known answers, as
// the replaced shaders evaluate them (fnvEnvIrradiance: sum c_i Y_i(n), relative to the mean):
// - uniform radiance: 1 in every direction;
// - radiance 1 above the horizon (world z > 0), 0 below: exactly 2 straight up, 0 straight down, 1 at the horizon
//   (order 2 represents this irradiance exactly at those directions: pi, 0 and pi / 2, mean pi / 2).
namespace envtest {

static uint16_t toHalf(float f) { return f >= 1.0f ? 0x3C00 : 0; }

static void run() {
  constexpr uint32_t kSize = 32;
  std::vector<uint16_t> uniform(6 * kSize * kSize * 4), sky(6 * kSize * kSize * 4);
  for (uint32_t f = 0; f < 6; f++)
    for (uint32_t j = 0; j < kSize; j++)
      for (uint32_t i = 0; i < kSize; i++) {
        const float u = 2.0f * (i + 0.5f) / kSize - 1.0f, v = 2.0f * (j + 0.5f) / kSize - 1.0f;
        // World z of the texel's direction, from D3D9's cube face table (sc, tc): +X (-z, -y), -X (+z, -y),
        // +Y (+x, +z), -Y (+x, -z), +Z (+x, -y), -Z (-x, -y).
        const float z = f == 0 ? -u : f == 1 ? u : f == 2 ? v : f == 3 ? -v : f == 4 ? 1.0f : -1.0f;
        uint16_t* a = &uniform[((f * kSize + j) * kSize + i) * 4];
        uint16_t* b = &sky[((f * kSize + j) * kSize + i) * 4];
        for (int c = 0; c < 3; c++) { a[c] = toHalf(1.0f); b[c] = toHalf(z > 0.0f ? 1.0f : 0.0f); }
        a[3] = b[3] = 0x3C00;
      }
  auto eval = [](const float sh[9][4], float x, float y, float z) {
    const float Y[9] = { 0.282095f, 0.488603f * y, 0.488603f * z, 0.488603f * x, 1.092548f * x * y, 1.092548f * y * z,
                         0.315392f * (3.0f * z * z - 1.0f), 1.092548f * x * z, 0.546274f * (x * x - y * y) };
    float e = 0.0f;
    for (int k = 0; k < 9; k++) e += sh[k][1] * Y[k];  // green
    return e;
  };
  float shU[9][4], shS[9][4];
  const float meanU = Backend::projectEnvironment(uniform.data(), kSize, shU);
  const float meanS = Backend::projectEnvironment(sky.data(), kSize, shS);
  const float dirs[4][3] = { { 0, 0, 1 }, { 0, 0, -1 }, { 1, 0, 0 }, { 0.6f, -0.8f, 0 } };
  float maxU = 0.0f;
  for (const auto& d : dirs) maxU = std::max(maxU, std::abs(eval(shU, d[0], d[1], d[2]) - 1.0f));
  const float up = eval(shS, 0, 0, 1), down = eval(shS, 0, 0, -1), horizon = eval(shS, 1, 0, 0);
  printf("environment: uniform mean %.4f (pi %.4f), max deviation %.4f; sky up %.3f down %.3f horizon %.3f, mean %.4f\n",
         meanU, 3.14159265f, maxU, up, down, horizon, meanS);
  if (std::abs(meanU - 3.14159265f) > 0.01f || maxU > 0.005f) fail("environment: a uniform environment is not uniform irradiance");
  if (std::abs(up - 2.0f) > 0.03f || std::abs(down) > 0.03f || std::abs(horizon - 1.0f) > 0.03f || std::abs(meanS - 1.5708f) > 0.02f)
    fail("environment: the sky-above environment's irradiance is not the analytic one");
}

}  // namespace envtest

// The prefiltered environment (Backend::kEnvironmentFiltered) against a brute-force integral of the same lobe: a cube
// lit 1 on its +Z face and 0 elsewhere (a hard edge, where gamma-averaged mips would darken the blur), and a uniform cube.
// For each texel direction n and the level's alpha, the reference is
//   sum L(l) max(n.l, 0) D(h) dOmega / sum max(n.l, 0) D(h) dOmega,  h = normalize(n + l)
// over a 96 x 96 per face grid: the split-sum radiance term with N = V (its sample density is D(h) / 4).
namespace prefiltertest {

static void dirOf(uint32_t f, float u, float v, float d[3]) {
  switch (f) {
    case 0: d[0] = 1; d[1] = -v; d[2] = -u; break;
    case 1: d[0] = -1; d[1] = -v; d[2] = u; break;
    case 2: d[0] = u; d[1] = 1; d[2] = v; break;
    case 3: d[0] = u; d[1] = -1; d[2] = -v; break;
    case 4: d[0] = u; d[1] = -v; d[2] = 1; break;
    default: d[0] = -u; d[1] = -v; d[2] = -1; break;
  }
  const float l = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  for (int c = 0; c < 3; c++) d[c] /= l;
}

static double reference(const float n[3], double alpha, bool lidOnly) {
  constexpr uint32_t G = 96;
  const double a2 = alpha * alpha;
  double num = 0.0, den = 0.0;
  for (uint32_t f = 0; f < 6; f++)
    for (uint32_t j = 0; j < G; j++)
      for (uint32_t i = 0; i < G; i++) {
        const float u = 2.0f * (i + 0.5f) / G - 1.0f, v = 2.0f * (j + 0.5f) / G - 1.0f;
        float l[3];
        dirOf(f, u, v, l);
        const double len2 = 1.0 + double(u) * u + double(v) * v;
        const double dOmega = 1.0 / (len2 * std::sqrt(len2));
        const double nl = n[0] * l[0] + n[1] * l[1] + n[2] * l[2];
        if (nl <= 0.0) continue;
        double h[3] = { n[0] + l[0], n[1] + l[1], n[2] + l[2] };
        const double hl = std::sqrt(h[0] * h[0] + h[1] * h[1] + h[2] * h[2]);
        const double nh = (n[0] * h[0] + n[1] * h[1] + n[2] * h[2]) / hl;
        const double d = nh * nh * (a2 - 1.0) + 1.0;
        const double D = a2 / (3.14159265358979 * d * d);
        const double w = nl * D * dOmega;
        num += w * ((!lidOnly || f == 4) ? 1.0 : 0.0);
        den += w;
      }
  return num / den;
}

static void run(Backend& be) {
  constexpr uint32_t kSize = 32, kLevels = 4;  // filtered 32, 16, 8, 4: roughness 0, 1/3, 2/3, 1
  int32_t full[4] = { 0, 0, int32_t(kSize), int32_t(kSize) };
  for (int scene = 0; scene < 2; scene++) {
    const bool lid = scene == 0;
    RtKey faces[6];
    for (uint32_t f = 0; f < 6; f++) {
      faces[f] = newKey();
      be.attach(faces[f], { kSize, kSize, kD3DFMT_A16B16G16R16F, 1, false });
      be.beginGroup(&faces[f], 1, 0);
      be.clear(kClearTarget, (!lid || f == 4) ? 0xFFFFFFFFu : 0xFF000000u, 1.0f, 0, full);
      be.endGroup();
    }
    const TexKey cube = 0xC0BE1000 + scene;
    be.attachCube(cube, faces, 1);
    be.setEnvironment(cube);
    if (!be.updateCube(cube)) { fail("prefilter: updateCube failed"); return; }
    double maxErr[kLevels] = {};
    bool read = true;
    for (uint32_t l = 0; l < kLevels && read; l++) {
      const uint32_t edge = kSize >> l;
      const double alpha = std::pow(double(l) / (kLevels - 1), 2.0);
      for (uint32_t f : { 0u, 4u, 5u }) {
        std::vector<uint8_t> d;
        if (!be.readbackTexture(Backend::kEnvironmentFiltered, f, l, d) || d.size() != size_t(edge) * edge * 8) {
          read = false;
          break;
        }
        const uint16_t* h = reinterpret_cast<const uint16_t*>(d.data());
        const uint32_t step = std::max(edge / 8, 1u);
        for (uint32_t j = step / 2; j < edge; j += step)
          for (uint32_t i = step / 2; i < edge; i += step) {
            float n[3];
            dirOf(f, 2.0f * (i + 0.5f) / edge - 1.0f, 2.0f * (j + 0.5f) / edge - 1.0f, n);
            const double want = l == 0 ? ((!lid || f == 4) ? 1.0 : 0.0) : reference(n, alpha, lid);
            const double got = sunshaftstest::halfToFloat(h[(size_t(j) * edge + i) * 4 + 1]);
            maxErr[l] = std::max(maxErr[l], std::abs(got - want));
          }
      }
    }
    printf("prefilter (%s): max |error| per level %.4f %.4f %.4f %.4f\n", lid ? "lit +Z face" : "uniform", maxErr[0],
           maxErr[1], maxErr[2], maxErr[3]);
    if (!read) fail("prefilter: readback of the filtered cube failed");
    if (maxErr[0] > 1e-3) fail("prefilter: level 0 is not the environment itself");
    for (uint32_t l = 1; l < kLevels; l++)
      if (maxErr[l] > 0.03) fail("prefilter: level %u is not the GGX integral (max error %.4f)", l, maxErr[l]);
    be.setEnvironment(0);
    for (RtKey f : faces) be.destroy(f);
  }
}

}  // namespace prefiltertest
