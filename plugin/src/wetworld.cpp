// Wet world (wetworld.h, docs/WETWORLD.md). After VegasShaders' RainState / Puddles (GPL-3.0): the rain as the game
// decides it, and water that fills and dries over game hours: the local puddles where the rain lands (a level), and the
// rain that runs off into the terrain's basins (a depth taken in, which fills each basin by its catchment).
//
// Game data (FNV 1.4.0.525, from the decomp):
//   TES* 0x011DEA10: +0x24 / +0x28 the current cell grid's centre (x, y), +0x34 the interior cell (null outdoors), +0x68
//     Sky*, +0x88 the worldspace. uGridsToLoad: setting 0x011C63CC, value +4.
//   Sky: +0x10 the current weather, +0x14 the last one, +0xEC the game hour (0..24), +0xF4 the weather transition
//     (0..1, from the last weather to the current one).
//   TESWeather +0xE0 ucWeatherData[15]: [6] / 255 how far into the transition the rain begins, [7] / 255 how far into it
//     a rainy last weather's rain ends, [11] & 4 rainy (TESWeather::IsRainy). Sky::IsRaining (0x0059E950) compares the
//     transition with GetWeatherData(6, 0.999, 0) and GetWeatherData(7, 1.0, 0.001).
//   TES::GetLandHeight (0x004572E0, thiscall (const NiPoint3&, float&) -> bool): the land's height at a world position,
//     false where the cell is not loaded (the body in the byte-matching decomp: src/fallout_shared/tes.cpp).
#include "wetworld.h"

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "addresses.h"
#include "common.h"
#include "effectlib.h"
#include "puddlemap.h"
#include "replacements.h"
#include "sunshadows.h"

namespace fnvvk::wetworld {
namespace {

float clamp01(float v) { return v > 0.0f ? (v < 1.0f ? v : 1.0f) : 0.0f; }

// Game hours for the levels (VegasShaders' Puddles / WetWorld defaults): open ground wet through in 0.25 h of rain and
// dry in 1.5 h of daylight; the local puddles full in 0.5 h and gone in 3 h; at night drying runs at 0.35 of the day's
// pace.
constexpr float kWetHours = 0.25f, kDryHours = 1.5f, kFillHours = 0.5f, kDrainHours = 3.0f, kNightDrying = 0.35f;
// The rain that runs off into the terrain basins, as a depth on flat ground (game units, 1 = 1.43 cm): full rain adds
// 0.7 an hour (10 mm/h, heavy rain), and it soaks away and evaporates at 0.25 an hour in daylight. 30% of the rain on a
// basin's catchment outside the basin reaches it (the rest soaks into the desert ground).
constexpr float kRainPerHour = 0.7f, kSoakPerHour = 0.25f, kRunoff = 0.3f;
constexpr float kLocalCoverage = 0.25f;   // the share of flat open ground local puddles cover at a full level
constexpr TexKey kPuddleMapKey = TexKey(~uintptr_t(0) - 0x3F), kLevelsKey = TexKey(~uintptr_t(0) - 0x4F);

struct Settings { bool on = false; float puddles = 1.0f; };
Settings settings() {
  Settings s;
  for (const fx::Replacement& r : replacements::list()) {
    if (r.id != "lighting" || !replacements::active(r)) continue;
    bool pbr = false, wet = false;
    for (const fx::Param& p : r.params) {
      if (p.name == "pbr") pbr = p.value != 0.0f;
      if (p.name == "wet") wet = p.value != 0.0f;
      if (p.name == "puddles") s.puddles = std::max(0.0f, p.value);
    }
    s.on = pbr && wet;
  }
  return s;
}

struct State {
  float rain = 0, wetness = 0, puddles = 0;
  float rainDepth = 0;  // the rain the basins have taken in, as a depth on flat ground (game units)
  float lastHour = -1.0f;
  bool outdoors = false;
  bool wasRaining = false;
} g;

struct PuddleMap {
  const void* worldspace = nullptr;
  int gridX = 0x7FFFFFFF, gridY = 0x7FFFFFFF, cells = 0;
  float originX = 0, originY = 0;
  uint32_t n = 0;
  uint32_t missing = 0, retries = 0, basins = 0;
  ULONGLONG builtAt = 0;
  bool uploaded = false;
  puddles::Map map;          // the flood's basins (their heights and catchments), for the levels
  float levelsDepth = -1.0f; // the rain depth the uploaded levels are for
  bool basinWater = false;
} g_map;
uint64_t g_builds = 0, g_lightingFrames = 0;
uint32_t g_logLines = 0;
void note(const char* fmt, ...) {
  if (g_logLines >= 60) return;  // event lines only, at most 60 a session
  g_logLines++;
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  logf("wet world: %s", buf);
}

// The rain falling now, 0..1: the current weather's rain from its begin point of the transition to full, the last
// weather's from full down to its end point (as Sky::IsRaining decides "raining", as an amount).
float rainNow(const void* sky) {
  const void* cur = rd<const void*>(sky, 0x10);
  const void* last = rd<const void*>(sky, 0x14);
  const float pct = clamp01(rd<float>(sky, 0xF4));
  float rain = 0.0f;
  if (cur && (rd<uint8_t>(cur, 0xE0 + 11) & 4)) {
    const float begin = rd<uint8_t>(cur, 0xE0 + 6) / 255.0f * 0.999f;
    rain = std::max(rain, clamp01((pct - begin) / std::max(1.0f - begin, 1e-3f)));
  }
  if (last && (rd<uint8_t>(last, 0xE0 + 11) & 4)) {
    const float end = 0.001f + rd<uint8_t>(last, 0xE0 + 7) / 255.0f * 0.999f;
    rain = std::max(rain, clamp01((end - pct) / std::max(end, 1e-3f)));
  }
  return rain;
}

// The terrain's heights over the loaded cells, every 128 units (the land's own vertex spacing), flooded into basins and
// uploaded as the puddle map. Rebuilt when the cell grid moves or changes worldspace; while some heights are missing
// (cells still loading) it is redone every 2 s, at most 8 times.
void buildPuddleMap(Backend* be, const void* tes) {
  const void* ws = rd<const void*>(tes, 0x88);
  const int gx = rd<int>(tes, 0x24), gy = rd<int>(tes, 0x28);
  const int cells = *reinterpret_cast<const int*>(0x011C63CC + 4);
  if (!ws || cells <= 0 || cells > 31) return;
  const ULONGLONG now = GetTickCount64();
  const bool moved = ws != g_map.worldspace || gx != g_map.gridX || gy != g_map.gridY || cells != g_map.cells;
  const bool retry = !moved && g_map.missing && g_map.retries < 8 && now - g_map.builtAt >= 2000;
  if (!moved && !retry) return;
  using LandHeightFn = bool(__thiscall*)(const void*, const float*, float*);
  const auto landHeight = reinterpret_cast<LandHeightFn>(0x004572E0);
  constexpr float kSpacing = 128.0f;
  const uint32_t n = uint32_t(cells) * 32 + 1;
  // The loaded grid: cells/2 each way around the centre cell (TES::GetGridCell's grid), in 4096-unit cells.
  const float ox = float(gx - cells / 2) * 4096.0f, oy = float(gy - cells / 2) * 4096.0f;
  std::vector<float> h(size_t(n) * n, 0.0f);
  std::vector<uint8_t> valid(size_t(n) * n, 0);
  uint32_t missing = 0;
  for (uint32_t y = 0; y < n; y++)
    for (uint32_t x = 0; x < n; x++) {
      const float p[3] = { ox + x * kSpacing, oy + y * kSpacing, 0.0f };
      float z = 0.0f;
      const size_t i = size_t(y) * n + x;
      if (landHeight(tes, p, &z) && std::isfinite(z)) {
        h[i] = z;
        valid[i] = 1;
      } else {
        missing++;
      }
    }
  puddles::Map m = puddles::flood(h, valid, n, n);
  std::vector<float> texels(size_t(n) * n * 4, 0.0f);
  for (size_t i = 0; i < size_t(n) * n; i++) {
    texels[i * 4 + 0] = float(m.basin[i]);
    texels[i * 4 + 1] = valid[i] ? h[i] : puddles::kNoGround;
  }
  g_map.uploaded = be->updateTexture2D(kPuddleMapKey, kD3DFMT_A32B32G32R32F, n, n, texels.data(), n * 16);
  g_map.retries = moved ? 0 : g_map.retries + 1;
  g_map.worldspace = ws;
  g_map.gridX = gx;
  g_map.gridY = gy;
  g_map.cells = cells;
  g_map.originX = ox;
  g_map.originY = oy;
  g_map.n = n;
  g_map.missing = missing;
  g_map.basins = uint32_t(m.basins.size());
  g_map.map = std::move(m);
  g_map.levelsDepth = -1.0f;  // the levels are uploaded for the new basins
  g_map.builtAt = now;
  g_builds++;
  if (moved || !missing) note("puddle map %ux%u around cell (%d, %d): %u basins, %u heights missing%s", n, n, gx, gy,
                              g_map.basins, missing, g_map.uploaded ? "" : " (NOT uploaded)");
}

// The basins' water levels for the rain depth now (puddlemap.h basinLevel), uploaded without waiting when the depth has
// moved by 0.02 units (a fortieth of an hour's heavy rain) or the map was rebuilt. Basin i's level at (i % 256, i / 256).
void updateLevels(Backend* be) {
  if (!g_map.uploaded) return;
  if (g_map.levelsDepth >= 0.0f && std::fabs(g.rainDepth - g_map.levelsDepth) < 0.02f) return;
  const uint32_t count = uint32_t(g_map.map.basins.size()) + 1;
  const uint32_t rows = (count + 255) / 256;
  std::vector<float> levels(size_t(rows) * 256, 0.0f);
  bool water = false;
  for (uint32_t i = 1; i < count; i++) {
    const puddles::Map::Basin& b = g_map.map.basins[i - 1];
    levels[i] = puddles::basinLevel(b, g.rainDepth, kRunoff);
    water = water || levels[i] > b.bottom + 0.01f;
  }
  be->uploadTexture2D(kLevelsKey, kD3DFMT_R32F, 256, rows, levels.data(), 256 * 4);
  g_map.levelsDepth = g.rainDepth;
  g_map.basinWater = water;
}

}  // namespace

bool enabled() { return settings().on; }

bool wantCover() {
  return settings().on && g.outdoors && (g.rain > 0.0f || g.wetness > 0.0f || g.puddles > 0.0f || g_map.basinWater);
}

void update(Backend* be) {
  const Settings s = settings();
  const void* tes = *reinterpret_cast<const void* const*>(fnv::TES_Singleton);
  const void* sky = tes ? rd<const void*>(tes, 0x68) : nullptr;
  g.outdoors = tes && sky && !rd<const void*>(tes, 0x34) && rd<const void*>(tes, 0x88);
  if (!g.outdoors) return;  // indoors the levels hold (no weather there)
  const float hour = rd<float>(sky, 0xEC);
  float hours = 0.0f;
  if (g.lastHour >= 0.0f && std::isfinite(hour)) {
    hours = hour - g.lastHour;
    if (hours < 0.0f) hours += 24.0f;  // past midnight
    if (hours > 12.0f) hours = 0.0f;   // a load or a jump backwards: not time that passed
  }
  g.lastHour = std::isfinite(hour) ? hour : g.lastHour;
  g.rain = rainNow(sky);
  const bool day = hour >= 6.0f && hour < 20.0f;
  const float drying = day ? 1.0f : kNightDrying;
  if (g.rain > 0.0f) {
    g.wetness = clamp01(g.wetness + g.rain * hours / kWetHours);
    g.puddles = clamp01(g.puddles + g.rain * hours / kFillHours);
    g.rainDepth += g.rain * hours * kRainPerHour * s.puddles;
  } else {
    g.wetness = clamp01(g.wetness - hours / kDryHours * drying);
    g.puddles = clamp01(g.puddles - hours / kDrainHours * drying);
    g.rainDepth = std::max(0.0f, g.rainDepth - hours * kSoakPerHour * drying);
  }
  if ((g.rain > 0.0f) != g.wasRaining)
    note("the rain %s (wetness %.2f, puddles %.2f)", g.rain > 0.0f ? "started" : "stopped", g.wetness, g.puddles);
  g.wasRaining = g.rain > 0.0f;
  if (be && s.on && (g.puddles > 0.0f || g.rain > 0.0f || g.rainDepth > 0.0f)) {
    buildPuddleMap(be, tes);
    updateLevels(be);
  }
}

void beginLighting(Backend* be) {
  if (!be) return;
  const Settings s = settings();
  if (!s.on || !g.outdoors || !(g.rain > 0.0f || g.wetness > 0.0f || g.puddles > 0.0f || g_map.basinWater)) return;
  const void* sceneGraph = *reinterpret_cast<const void* const*>(fnv::TESMain_spWorldRoot);
  const void* camera = sceneGraph ? rd<const void*>(sceneGraph, 0xAC) : nullptr;
  if (!camera || rd<bool>(camera, 0xDC + 24)) return;
  Backend::WetLighting w{};
  w.rain = g.rain;
  w.wetness = g.wetness;
  w.puddleLevel = g.puddles;
  w.puddleCoverage = kLocalCoverage * s.puddles;
  static LARGE_INTEGER t0{}, freq{};
  if (!freq.QuadPart) {
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);
  }
  LARGE_INTEGER now;
  QueryPerformanceCounter(&now);
  w.time = float(double(now.QuadPart - t0.QuadPart) / double(freq.QuadPart));
  for (int i = 0; i < 4; i++) w.frustum[i] = rd<float>(camera, 0xDC + 4 * i);
  w.nearZ = rd<float>(camera, 0xDC + 16);
  w.farZ = rd<float>(camera, 0xDC + 20);
  for (int i = 0; i < 4; i++) w.port[i] = rd<float>(camera, 0x100 + 4 * i);
  for (int row = 0; row < 3; ++row) {  // NiAVObject world rotation rows (columns direction, up, right), translation
    memcpy(w.cameraRows[row], static_cast<const char*>(camera) + 0x68 + 12 * row, 12);
    w.cameraRows[row][3] = rd<float>(camera, 0x8C + 4 * row);
  }
  const sunshadows::Cover& c = sunshadows::cover();
  w.cover = c.drawn;
  w.coverLayer = c.layer;
  memcpy(w.coverFromWorld, c.fromWorld, sizeof(w.coverFromWorld));
  w.coverBias = c.bias;
  w.coverTexel = c.texel;
  if (g_map.uploaded && g_map.levelsDepth >= 0.0f) {
    w.puddleMap = kPuddleMapKey;
    w.puddleLevels = kLevelsKey;
    w.basinWater = g_map.basinWater;
    w.puddleOrigin[0] = g_map.originX;
    w.puddleOrigin[1] = g_map.originY;
    w.puddleSpacing = 128.0f;
    w.puddleSize[0] = w.puddleSize[1] = g_map.n;
  }
  be->setWetLighting(&w);
  g_lightingFrames++;
}

void endLighting(Backend* be) {
  if (be) be->setWetLighting(nullptr);
}

std::string summary() {
  char b[256];
  snprintf(b, sizeof(b), "wet world: %s, %s; rain %.2f, wetness %.2f, local puddles %.2f, rain taken in %.2f units; "
           "puddle map %ux%u (%u basins%s, %u heights missing, %llu builds); %llu lit world passes", settings().on ? "on" : "off",
           g.outdoors ? "outdoors" : "indoors", g.rain, g.wetness, g.puddles, g.rainDepth, g_map.n, g_map.n, g_map.basins,
           g_map.basinWater ? ", some holding water" : "", g_map.missing, (unsigned long long)g_builds,
           (unsigned long long)g_lightingFrames);
  return b;
}

}  // namespace fnvvk::wetworld
