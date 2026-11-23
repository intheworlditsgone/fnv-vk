// Effects at fixed points of the game's frame (native mode). The effects themselves are data (backend/effectlib.h).
//
// - "world": TESMain::DrawWorldStandard 0x00870BD0 draws the world with TESMain::DrawWorld_DrawScene 0x00873200
//   (__thiscall, 4 stack arguments, retn 10h) and only then refraction, the first-person view, image space (HDR, bloom,
//   tonemapping) and the interface. Effects at "world" run right after DrawScene returns, on the render-target group the
//   world was drawn into, with the world camera: TESMain::spWorldRoot 0x011DEB7C -> BSSceneGraph::spCamera (+0xAC) ->
//   NiCamera::m_kViewFrustum (+0xDC: left, right, top, bottom, near, far, ortho) and m_kPort (+0x100: left, right, top,
//   bottom).
// - "tonemap": Main::DrawWorld_ImageSpace 0x00875FD0 calls BSShaderManager::RenderImageSpaceEffects 0x00B55AC0
//   (__cdecl: renderer, screen buffer, destination buffer), which runs ImageSpaceManager::RenderEndOfFrameEffects: every
//   active image-space effect (HDR exposure/bloom/cinematic, depth of field, blurs...), the last one writing the
//   destination. The interface is drawn afterwards. Effects at "tonemap" run right after it returns, on the
//   destination (the last render-target group), over the whole target; at most once per frame (DrawWorld_ImageSpace
//   has three callers, one per frame path).
// - "opaque": inside DrawWorld_DrawScene, BSShaderAccumulator::FinishAccumulating_Standard_PreResolveDepth 0x00B65AE0
//   (__cdecl, the accumulator) draws the opaque passes (1..0x287) and the decals; PostResolveDepth then draws water
//   (passes 0x288..0x2CE) and the blended geometry. Effects at "opaque" run right after it returns, for the world
//   pass only (the accumulator's camera, +8, is the world camera; modes 7 and 8, +0x19C, are water-depth passes),
//   once per frame: water and transparent geometry are drawn over their result (water refracts it).
#include "cputime.h"
#include "effects.h"

#include <windows.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "addresses.h"
#include "backend.h"
#include "common.h"
#include "replacements.h"
#include "shadow.h"
#include "sunshadows.h"
#include "upscale.h"
#include "viewvector.h"
#include "wetworld.h"

namespace fnvvk::effects {
namespace {

std::vector<fx::Effect> g_effects;
std::vector<uint32_t> g_backendIds;  // per effect: the backend's id (0 = not registered yet / failed)
std::string g_dir, g_cacheDir, g_ini;
std::atomic<uint64_t> g_runs{ 0 }, g_noCamera{ 0 }, g_skipped{ 0 };  // g_skipped: by Skip= (effect.ini)
uint32_t g_frame = 0;
LARGE_INTEGER g_t0{}, g_freq{};

void applySettings(fx::Effect& e) {
  const std::string sec = "Effect." + e.id;
  e.enabled = GetPrivateProfileIntA(sec.c_str(), "Enable", 1, g_ini.c_str()) != 0;
  for (fx::Param& p : e.params) {
    char buf[64] = "";
    GetPrivateProfileStringA(sec.c_str(), p.name.c_str(), "", buf, sizeof(buf), g_ini.c_str());
    p.value = buf[0] ? float(atof(buf)) : p.def;
  }
}

void unregisterAll() {
  if (Backend* be = shadow::backendIfReady())
    for (uint32_t id : g_backendIds)
      if (id) be->destroyEffect(id);
  g_backendIds.assign(g_effects.size(), 0);
}

void load() {
  g_effects = fx::loadEffects(g_dir, g_cacheDir, [](const char* s) { logf("%s", s); });
  for (fx::Effect& e : g_effects) applySettings(e);
  g_backendIds.assign(g_effects.size(), 0);
}

// Registers effect i with the backend once (its compiled passes become shader modules there).
uint32_t backendId(Backend* be, size_t i) {
  if (g_backendIds[i]) return g_backendIds[i];
  const fx::Effect& e = g_effects[i];
  Backend::EffectDesc d;
  d.sceneColor = e.sceneColor;
  d.skinBuffer = e.skinBuffer;
  for (const fx::Image& im : e.images)
    d.images.push_back({ im.vkFormat, im.scale,
                         uint8_t(im.publish == "ao" ? Backend::EffectDesc::Ao : im.publish == "gi" ? Backend::EffectDesc::Gi : im.publish == "gidepth" ? Backend::EffectDesc::GiDepth
                                                                                                  : Backend::EffectDesc::None), im.width, im.height });
  for (const fx::Pass& p : e.passes)
    d.passes.push_back({ { p.spv[0].data(), p.spv[1].data() }, { p.spv[0].size(), p.spv[1].size() }, p.output, uint32_t(p.blend),
                         p.skipPbr });
  g_backendIds[i] = be->createEffect(d);
  if (!g_backendIds[i]) {
    logf("effect '%s': the backend could not create it (disabled)", e.id.c_str());
    g_effects[i].enabled = false;
  }
  return g_backendIds[i];
}

// The world camera's frustum and viewport (effects at "world"). False without a perspective camera.
bool worldCamera(Backend::EffectFrame& f) {
  const void* sceneGraph = *reinterpret_cast<void* const*>(fnv::TESMain_spWorldRoot);
  const void* camera = sceneGraph ? rd<const void*>(sceneGraph, 0xAC) : nullptr;
  if (!camera || rd<bool>(camera, 0xDC + 24)) return false;  // no camera, or an orthographic frustum
  for (int i = 0; i < 4; i++) f.frustum[i] = rd<float>(camera, 0xDC + 4 * i);
  f.nearZ = rd<float>(camera, 0xDC + 16);
  f.farZ = rd<float>(camera, 0xDC + 20);
  for (int i = 0; i < 4; i++) f.port[i] = rd<float>(camera, 0x100 + 4 * i);
  // NiAVObject m_kWorld: rotation rows (+0x68; columns direction, up, right), translation (+0x8C).
  for (int row = 0; row < 3; ++row) {
    memcpy(f.cameraRows[row], static_cast<const char*>(camera) + 0x68 + 12 * row, 12);
    f.cameraRows[row][3] = rd<float>(camera, 0x8C + 4 * row);
  }
  // TES +0x34: the interior cell the player is in, null outdoors.
  const void* tes = *reinterpret_cast<const void* const*>(fnv::TES_Singleton);
  f.interior = tes && rd<const void*>(tes, 0x34);
  // Outdoors, the sun light's colours at any time of day (TES +0x1C, NiLight: +0xC4 dimmer, +0xC8 ambient, +0xD4
  // diffuse; Sun::Update sets them from the weather), also without shadow cascades (night): volumetric fog's ambient.
  // sunshadows::fill writes the same values when it has cascades.
  if (tes && !f.interior)
    if (const void* light = rd<const void*>(tes, 0x1C)) {
      const float dimmer = rd<float>(light, 0xC4);
      for (int i = 0; i < 3; i++) {
        f.sunColor[i] = rd<float>(light, 0xD4 + 4 * i) * dimmer;
        f.ambientColor[i] = rd<float>(light, 0xC8 + 4 * i);
      }
    }
  if (*reinterpret_cast<const bool*>(0x011C7A59)) { // TESWaterSystem::Update, camera below water
    memcpy(f.waterSun, reinterpret_cast<const void*>(0x011FF398), sizeof(f.waterSun));
    float height = *reinterpret_cast<const float*>(0x011C7A5C);
    if (!*reinterpret_cast<const bool*>(0x011C7A5A)) {
      void* player = *reinterpret_cast<void* const*>(0x011DEA3C);
      void* cell = player ? reinterpret_cast<void* (__thiscall*)(void*)>(0x008D6F30)(player) : nullptr;
      if (cell) height = reinterpret_cast<float (__thiscall*)(void*)>(0x005471E0)(cell);
    }
    f.submersion = height - f.cameraRows[2][3];
  }
  return f.nearZ > 0.0f && f.farZ > f.nearZ;
}

// The lighting applies the published ambient occlusion itself: the lighting replacement is active with its physically
// based mode on (its shaders read the AO block, docs/SHADER_ABI.md "Ambient occlusion").
bool lightingAppliesAo() {
  for (const fx::Replacement& r : replacements::list())
    if (r.id == "lighting" && replacements::active(r))
      for (const fx::Param& p : r.params)
        if (p.name == "pbr") return p.value != 0.0f;
  return false;
}

// The ambient occlusion state for the world pass's lighting (Backend::setAoLighting): the world camera now.
void beginAo(Backend* be) {
  Backend::EffectFrame f{};
  if (!worldCamera(f)) return;
  Backend::AoLighting a{};
  memcpy(a.frustum, f.frustum, sizeof(a.frustum));
  a.nearZ = f.nearZ;
  a.farZ = f.farZ;
  memcpy(a.port, f.port, sizeof(a.port));
  memcpy(a.cameraRows, f.cameraRows, sizeof(a.cameraRows));
  be->setAoLighting(&a);
}

// An enabled effect reads the skin buffer (SkinBuffer=1, physically based lighting on): the world's opaque half fills it.
bool wantSkinBuffer() {
  if (!lightingAppliesAo()) return false;
  for (const fx::Effect& e : g_effects)
    if (e.enabled && e.error.empty() && e.skinBuffer) return true;
  return false;
}

void runPoint(const char* point) {
  Backend* be = shadow::backendIfReady();
  if (!be) return;
  // Effects with Requires=pbr (effects/ssgi: only the physically based lighting uses what they publish) do not run
  // without it, not even their scene copy.
  const bool pbr = lightingAppliesAo();
  auto runs = [&](const fx::Effect& e) {
    return e.enabled && e.error.empty() && e.point == point && (e.requirement != "pbr" || pbr);
  };
  bool any = false;
  for (const fx::Effect& e : g_effects) any |= runs(e);
  if (!any) return;
  Backend::EffectFrame f{};
  if (strcmp(point, "tonemap") != 0) {  // "world", "opaque": the world camera
    if (!worldCamera(f)) {
      g_noCamera++;
      return;
    }
  } else {  // "tonemap": the whole image; no camera (the passes do not read depth)
    static uint64_t lastFrame = ~0ull;
    const uint64_t frame = be->frameIndex();
    if (frame == lastFrame) return;
    lastFrame = frame;
    const float port[4] = { 0.0f, 1.0f, 1.0f, 0.0f };  // NiViewport left, right, top, bottom
    const float frustum[4] = { -1.0f, 1.0f, 1.0f, -1.0f };
    memcpy(f.port, port, sizeof(port));
    memcpy(f.frustum, frustum, sizeof(frustum));
    f.nearZ = 1.0f;
    f.farZ = 2.0f;
  }
  LARGE_INTEGER now{};
  QueryPerformanceCounter(&now);
  f.time = float(double(now.QuadPart - g_t0.QuadPart) / double(g_freq.QuadPart));
  f.frame = g_frame++;
  f.aoApplied = lightingAppliesAo();
  sunshadows::fill(f);
  cputime::Scope timed(cputime::Effects);
  // Skip= (effect.ini): the conditions under which an effect would leave the image as it is.
  const bool underwater = f.submersion > 0.0f;
  auto skips = [&](const fx::Effect& e) {
    if ((e.skip & fx::Effect::SkipInterior) && f.interior) return true;
    if ((e.skip & fx::Effect::SkipUnderwater) && underwater) return true;
    if ((e.skip & fx::Effect::SkipAboveWater) && !underwater) return true;
    if ((e.skip & fx::Effect::SkipNoCascades) && f.shadowCascades == 0) return true;  // (fnv.shadowInfo.x 0)
    for (int k : e.skipZero)
      if (e.params[size_t(k)].value == 0.0f) return true;
    return false;
  };
  bool ranBefore = false;
  for (size_t i = 0; i < g_effects.size(); i++) {
    const fx::Effect& e = g_effects[i];
    if (!runs(e)) continue;
    if (skips(e)) {
      g_skipped++;
      continue;
    }
    const uint32_t id = backendId(be, i);
    if (!id) continue;
    float params[32][4] = {};
    for (size_t k = 0; k < e.params.size() && k < 32; k++) params[k][0] = e.params[k].value;
    // The effects of this point run back to back (nothing else draws in between): a scene copy made by an earlier one
    // is reused while no pass has written the scene since (Backend::EffectFrame::sceneUnchanged).
    f.sceneUnchanged = ranBefore;
    be->runEffect(id, f, &params[0][0], uint32_t(std::min<size_t>(e.params.size(), 32)));
    ranBefore = true;
    g_runs++;
  }
}

using DrawSceneFn = void(__fastcall*)(void* self, void* edx, void* sun, int firstPerson, bool wireframe, bool vatsQuery);
DrawSceneFn oDrawScene = nullptr;
void __fastcall hkDrawScene(void* self, void* edx, void* sun, int firstPerson, bool wireframe, bool vatsQuery) {
  oDrawScene(self, edx, sun, firstPerson, wireframe, vatsQuery);
  if (!wireframe) runPoint("world");
  if (!wireframe) upscale::afterDrawScene();  // the world's depth, before the first-person view clears it
}

// The world pass of a BSShaderAccumulator: its camera (+8) is the world camera, and it is not a water-depth pass
// (mode +0x19C 7 or 8).
bool isWorldPass(const void* accumulator) {
  const void* sceneGraph = *reinterpret_cast<void* const*>(fnv::TESMain_spWorldRoot);
  const void* camera = sceneGraph ? rd<const void*>(sceneGraph, 0xAC) : nullptr;
  if (!accumulator || !camera || rd<const void*>(accumulator, 8) != camera) return false;
  const int mode = rd<int>(accumulator, 0x19C);
  return mode != 7 && mode != 8;
}

// BSShaderAccumulator::FinishAccumulating_Standard (0x00B665A0) draws the accumulated world in two halves:
// PreResolveDepth (opaque passes and decals) and PostResolveDepth (water and blended geometry). The sun shadow cascades
// are drawn before the first half (the game's lighting samples them, sunshadows::beginLighting), and the lighting
// shadows the sun during both halves of the world pass only.
using FinishFn = void(__cdecl*)(void* accumulator);
FinishFn oFinishOpaque = nullptr, oFinishBlended = nullptr;
uint64_t g_opaqueFrame = ~0ull;
void __cdecl hkFinishOpaque(void* accumulator) {
  Backend* be = isWorldPass(accumulator) ? shadow::backendIfReady() : nullptr;
  if (!be) {
    oFinishOpaque(accumulator);
    return;
  }
  wetworld::update(be);    // the weather's wetness, before the shadow pass (which draws its rain cover)
  sunshadows::render(be);  // at most once per frame
  sunshadows::beginLighting(be);
  wetworld::beginLighting(be);
  beginAo(be);
  viewvector::beginWorld();
  const bool skin = wantSkinBuffer();
  if (skin) be->setSkinBuffer(true);
  {
    cputime::Scope timed(cputime::WorldOpaque);
    be->beginDepthPrepass();  // each pixel shaded once (BackendConfig::depthPrepass)
    oFinishOpaque(accumulator);
    be->endDepthPrepass();
  }
  if (skin) be->setSkinBuffer(false);
  viewvector::endWorld();
  be->setAoLighting(nullptr);
  sunshadows::endLighting(be);
  wetworld::endLighting(be);
  const uint64_t frame = be->frameIndex();
  if (frame == g_opaqueFrame) return;
  g_opaqueFrame = frame;
  runPoint("opaque");
}
void __cdecl hkFinishBlended(void* accumulator) {
  Backend* be = isWorldPass(accumulator) ? shadow::backendIfReady() : nullptr;
  if (be) {
    sunshadows::beginLighting(be);
    wetworld::beginLighting(be);
    beginAo(be);
  }
  {
    cputime::Scope timed(cputime::WorldBlended);  // (any accumulator: the world's is most of it)
    oFinishBlended(accumulator);  // (the world pass's reflections continue: g_reflected is kept from the opaque half)
  }
  if (be) {
    be->setAoLighting(nullptr);
    sunshadows::endLighting(be);
    wetworld::endLighting(be);
  }
}

using ImageSpaceFn = void(__cdecl*)(void* renderer, void* screenBuffer, void* destination);
ImageSpaceFn oImageSpace = nullptr;
void __cdecl hkImageSpace(void* renderer, void* screenBuffer, void* destination) {
  upscale::beforeImageSpace(screenBuffer, destination);  // the scene upscaled: image space runs at the window size
  oImageSpace(renderer, screenBuffer, destination);
  runPoint("tonemap");
}

}  // namespace

std::vector<fx::Effect>& list() { return g_effects; }
bool hasPoint(const char* point) {
  for (const fx::Effect& e : g_effects)
    if (e.error.empty() && e.point == point) return true;
  return false;
}
const char* effectsDir() { return g_dir.c_str(); }

bool save(const fx::Effect& e) {
  const std::string sec = "Effect." + e.id;
  bool ok = WritePrivateProfileStringA(sec.c_str(), "Enable", e.enabled ? "1" : "0", g_ini.c_str()) != 0;
  for (const fx::Param& p : e.params) {
    char v[32];
    if (p.type == fx::Param::Type::Float) snprintf(v, sizeof(v), "%g", p.value);
    else snprintf(v, sizeof(v), "%d", int(p.value));
    ok &= WritePrivateProfileStringA(sec.c_str(), p.name.c_str(), v, g_ini.c_str()) != 0;
  }
  logf("effect '%s': settings %s fnv_vk.ini", e.id.c_str(), ok ? "saved to" : "could NOT be saved to");
  return ok;
}

void reload() {
  // Current settings survive the reload (matched by effect id and parameter name).
  struct Kept { bool enabled; std::vector<std::pair<std::string, float>> values; };
  std::vector<std::pair<std::string, Kept>> kept;
  for (const fx::Effect& e : g_effects) {
    Kept k{ e.enabled, {} };
    for (const fx::Param& p : e.params) k.values.push_back({ p.name, p.value });
    kept.push_back({ e.id, k });
  }
  unregisterAll();
  load();
  for (fx::Effect& e : g_effects)
    for (const auto& [id, k] : kept)
      if (id == e.id) {
        e.enabled = k.enabled;
        for (fx::Param& p : e.params)
          for (const auto& [name, v] : k.values)
            if (name == p.name) p.value = v;
      }
  logf("effects reloaded: %zu", g_effects.size());
}

bool install() {
  g_ini = iniPath();
  g_dir = moduleDir() + "fnv_vk\\effects";
  g_cacheDir = moduleDir() + "fnv_vk\\cache";
  QueryPerformanceFrequency(&g_freq);
  QueryPerformanceCounter(&g_t0);
  load();
  bool ok = hook(0x00873200, &hkDrawScene, &oDrawScene, "TESMain::DrawWorld_DrawScene (effects at 'world')");
  ok &= hook(0x00B55AC0, &hkImageSpace, &oImageSpace, "BSShaderManager::RenderImageSpaceEffects (effects at 'tonemap')");
  ok &= hook(0x00B65AE0, &hkFinishOpaque, &oFinishOpaque,
             "BSShaderAccumulator::FinishAccumulating_Standard_PreResolveDepth (effects at 'opaque', sun shadows)");
  ok &= hook(0x00B65C60, &hkFinishBlended, &oFinishBlended,
             "BSShaderAccumulator::FinishAccumulating_Standard_PostResolveDepth (sun shadows)");
  return ok;
}

void logSummary() {
  Backend* be = shadow::backendIfReady();
  const BackendStats s = be ? be->stats() : BackendStats{};
  size_t on = 0;
  for (const fx::Effect& e : g_effects) on += e.enabled && e.error.empty();
  logf("effects: %zu loaded, %zu enabled | runs %llu: applied %llu, not possible %llu, no camera %llu | %s, %llu caster draws, "
       "%llu sun-shadowed draws",
       g_effects.size(), on, g_runs.load(), s.effectRuns, s.effectsSkipped, g_noCamera.load(), sunshadows::summary().c_str(),
       s.shadowCasters, s.sunShadowedDraws);
  logf("effects, work avoided: runs skipped by Skip= %llu, runs without skin %llu, passes skipped (Skip=pbr) %llu | scene "
       "copies %llu, reused %llu",
       g_skipped.load(), s.fxNoSkinRuns, s.fxPassesSkipped, s.fxSceneCopies, s.fxSceneCopiesReused);
  logf("%s", wetworld::summary().c_str());
}

bool pbrLighting() { return lightingAppliesAo(); }

}  // namespace fnvvk::effects
