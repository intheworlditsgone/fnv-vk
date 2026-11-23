// Environment probe: the game renders the scene around the camera into a cube map, one face per frame, with its own
// cube-map camera and scene path (BSCubeMapCamera; the body of BSCubeMapCamera::ClickCubeMap 0x00BD77C0 for one face).
// Two cubes alternate: one is being drawn, the other (complete) is what draws sample (current()).
//
// Engine facts (IDA, FalloutNV.exe 1.4.0.525):
// - BSCubeMapCamera (0x140 bytes; NiNew 0x00AA13E0, constructor 0x00BD8290 thiscall (this, type)): an NiCamera with
//   a 90-degree frustum (m_kViewFrustum +0xDC: near +0xEC, far +0xF0), its own BSShaderAccumulator (+0x13C, whose
//   pActiveShadowSceneNode is +0x194), the node it renders (+0x138). NiAVObject m_kLocal: rotate rows +0x34,
//   translate +0x58, scale +0x64; m_kWorld translate +0x8C.
// - The cube: BSTextureManager::CreateRenderedTexture 0x00B6D170 (thiscall on *0x011F91A8: renderer, width, height,
//   D3DFORMAT, EDRAM format, flags (0x10 cube), type, MSAA, alias, alias offset) -> BSRenderedTexture whose
//   spRenderedTextures[0] (+0x30) is the NiRenderedCubeMap: m_spBuffer +0x30, m_eFace +0x48, face buffers +0x4C.
//   BSRenderedTexture::GetGroup 0x00B6B260 picks the group of m_eFace.
// - One face, as ClickCubeMap: m_eFace and m_spBuffer set, BSRenderedTexture::StartOffscreen 0x00B6B8D0 (cdecl:
//   clear mode 7, group), NiAVObject::Update 0x00A59C60, the accumulator's StartAccumulating (vtable +0x8C, camera),
//   bAccumulate (+0x84) = 1, a BSCullingProcess on the stack (0xC8 bytes: NiCullingProcess constructor 0x00A69400
//   (this, 0), vtable 0x0101E2EC, eCullMode +0x90 / uiStackIndex +0xBC / pCompoundFrustum +0xC0 / spAccumulator +0xC4
//   zero, m_pkCamera +0xC, NiCullingProcess::SetFrustum 0x00A694A0 (camera +0xDC)), BSShaderUtil::AccumulateScene
//   0x00B6BEE0 (camera, node, culler), BSShaderUtil::RenderScene 0x00B6C0D0 (camera, accumulator, 0), the culler's
//   destructor 0x004A0F60, BSRenderedTexture::StopOffscreen 0x00B6B790.
// - The face orientation. The game's own (SetupTransformation 0x00BD7200) is its shadow cubes' convention; a cube
//   sampled with world-space directions (the game's environment-map shaders) needs each face's camera looking along
//   the face's axis with D3D9's up vector, and its right vector mirrored, which NiDX9Renderer::SetLeftRightSwap
//   0x00E757F0 (on BSShaderManager::spRenderer 0x011F9508) applies: for every face (rows below) the camera direction
//   (column 0) and up (column 1) are D3D9's face axis and up, and column 2 is minus D3D9's right.
// - While capturing, the world ShadowSceneNode (0x011F91C8[0]) keeps its lights (bDisableLightUpdate +0x130): the
//   lights are the ones chosen for the frame.
// - When: TESMain::DrawWorld_UpdateWater 0x008727D0, where the game renders its own off-screen views (water
//   reflections) before the world.
//
// The probe's draws are not the world pass (their accumulator's camera is the probe's), so effects and the sun shadow
// cascades, which are in the world camera's view, do not apply to them (effects.cpp isWorldPass).
#include "cputime.h"
#include "envprobe.h"

#include <windows.h>

#include <d3d9.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

#include "addresses.h"
#include "backend.h"
#include "common.h"
#include "shadow.h"

namespace fnvvk::envprobe {
namespace {

void addRef(void* o) { if (o) InterlockedIncrement(reinterpret_cast<volatile LONG*>(static_cast<char*>(o) + 4)); }
void release(void* o) {
  if (o && !InterlockedDecrement(reinterpret_cast<volatile LONG*>(static_cast<char*>(o) + 4)))
    (*reinterpret_cast<void(__thiscall***)(void*)>(o))[1](o);  // NiRefObject::DeleteThis (vtable +4)
}

struct Settings {
  bool enable = true;
  uint32_t size = 256;
  float distance = 30000.0f;
  uint32_t interval = 2;  // frames per face: the probe renders the scene once every `interval` frames (CPU cost)
} g_cfg;
uint32_t g_tick = 0;

void* g_camera = nullptr;     // BSCubeMapCamera
void* g_cubes[2] = {};        // BSRenderedTexture (cube)
uint32_t g_face = 0, g_drawing = 0;
std::atomic<uintptr_t> g_current{ 0 };
bool g_failed = false;

// The D3D9 cube texture of a rendered cube, the backend's key for it (shadow.cpp attaches cubes by it): face buffer 0
// (NiRenderedCubeMap +0x4C) -> its renderer data (Ni2DBuffer +0x10, an NiDX9TextureBufferData) -> m_pkRenderTexture
// (+0x20, NiDX9TextureBufferData::CreateCubeFace 0x00E7D390).
uintptr_t cubeKey(void* rendered) {
  void* cube = rd<void*>(rendered, 0x30);
  void* buffer = cube ? rd<void*>(cube, 0x4C) : nullptr;
  void* data = buffer ? rd<void*>(buffer, 0x10) : nullptr;
  return data ? rd<uintptr_t>(data, 0x20) : 0;
}

void* createCube(uint32_t size) {
  using CreateFn = void*(__thiscall*)(void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                                      void*, uint32_t);
  void* manager = *reinterpret_cast<void**>(0x011F91A8);
  void* renderer = *reinterpret_cast<void**>(0x011F9508);
  if (!manager || !renderer) return nullptr;
  void* t = reinterpret_cast<CreateFn>(0x00B6D170)(manager, renderer, size, size, D3DFMT_A16B16G16R16F, 0, 0x10, 0, 0,
                                                   nullptr, 0);
  addRef(t);
  return t;
}

bool create() {
  using NiNewFn = void*(__cdecl*)(uint32_t);
  using CtorFn = void*(__thiscall*)(void*, int);
  void* mem = reinterpret_cast<NiNewFn>(0x00AA13E0)(0x140);
  if (!mem) return false;
  g_camera = reinterpret_cast<CtorFn>(0x00BD8290)(mem, 0);
  addRef(g_camera);
  for (void*& c : g_cubes) c = createCube(g_cfg.size);
  if (!g_cubes[0] || !g_cubes[1] || !cubeKey(g_cubes[0]) || !cubeKey(g_cubes[1])) {
    logf("probe: the game could not create the %ux%u cube maps; probe off", g_cfg.size, g_cfg.size);
    return false;
  }
  wr<float>(g_camera, 0xEC, 5.0f);
  wr<float>(g_camera, 0xF0, g_cfg.distance);
  logf("probe: %ux%u RGBA16F cube maps, view distance %.0f, one face per frame", g_cfg.size, g_cfg.size, g_cfg.distance);
  return true;
}

// Camera rotation rows for each D3D9 face (+X -X +Y -Y +Z -Z), see the header comment.
constexpr float kFaceRows[6][9] = {
  { 1, 0, 0, 0, 1, 0, 0, 0, 1 },     { -1, 0, 0, 0, 1, 0, 0, 0, -1 }, { 0, 0, -1, 1, 0, 0, 0, -1, 0 },
  { 0, 0, -1, -1, 0, 0, 0, 1, 0 },   { 0, 0, -1, 0, 1, 0, 1, 0, 0 },  { 0, 0, 1, 0, 1, 0, -1, 0, 0 },
};

// The probe must not change what the world view culls or fades. Culling writes per-object state that the world view
// then reuses: BSMultiBound visibility, and BSFadeNode fade (distance) state, both from whatever camera culls. The
// game's own off-screen views do two things about it, and the probe does the same:
// - cull mode BSCP_CULL_FORCEMULTIBOUNDSNOUPDATE (4): the water's world and sky reflections pass it to
//   MTRenderingSystem::AddAccumTask (TESWaterSystem::MTSetupWorldReflectionTexture 0x004EAF18,
//   MTSetupSkyReflectionTexture 0x004EB1D7);
// - fading off while rendering, restored after: BSFadeNode::SetFadeEnabled 0x004E20B0 / GetFadeEnabled 0x0054F4B0,
//   as TESWaterSystem::UpdateWaterSystem, TESMain::DrawWorld_UpdateOffscreenBuffers and the local map pictures do.
// (BSCubeMapCamera::ClickCubeMap culls normally: the game only uses it for shadow depth cubes.) Without them the probe,
// culling from the camera's position in a new direction each frame, changed the world view's visibility of large
// multi-bound objects every frame (flicker).
constexpr uint32_t kCullForceMultiBoundsNoUpdate = 4;

// One face of g_cubes[g_drawing], from the world camera's position.
void captureFace(void* sceneGraph, void* worldCamera, void* ssn) {
  void* tex = g_cubes[g_drawing];
  void* cube = rd<void*>(tex, 0x30);
  const uint32_t face = g_face;
  // As ClickCubeMap: the cube's current face and its buffer.
  wr<uint32_t>(cube, 0x48, face);
  void* buffer = rd<void*>(cube, 0x4C + 4 * face);
  void* old = rd<void*>(cube, 0x30);
  if (old != buffer) {
    addRef(buffer);
    wr<void*>(cube, 0x30, buffer);
    release(old);
  }
  // The camera at the world camera's position, looking along the face.
  memcpy(static_cast<char*>(g_camera) + 0x34, kFaceRows[face], sizeof(kFaceRows[face]));
  memcpy(static_cast<char*>(g_camera) + 0x58, static_cast<char*>(worldCamera) + 0x8C, 12);
  wr<float>(g_camera, 0x64, 1.0f);
  struct UpdateData { float time; bool controllers, parallel, particles, morph, sceneGraphChange; } ud{};
  reinterpret_cast<void(__thiscall*)(void*, UpdateData*)>(0x00A59C60)(g_camera, &ud);
  // Its scene: the world root, lit by the world's shadow scene node.
  void* node = rd<void*>(g_camera, 0x138);
  if (node != sceneGraph) {
    addRef(sceneGraph);
    wr<void*>(g_camera, 0x138, sceneGraph);
    release(node);
  }
  void* acc = rd<void*>(g_camera, 0x13C);
  wr<void*>(acc, 0x194, ssn);

  void* renderer = *reinterpret_cast<void**>(0x011F9508);
  const bool lightsFrozen = rd<bool>(ssn, 0x130);
  wr<bool>(ssn, 0x130, true);
  reinterpret_cast<void(__thiscall*)(void*, bool)>(0x00E757F0)(renderer, true);
  void* group = reinterpret_cast<void*(__thiscall*)(void*)>(0x00B6B260)(tex);
  reinterpret_cast<void(__cdecl*)(uint32_t, void*)>(0x00B6B8D0)(7, group);

  if (rd<LONG>(acc, 4) == 1) addRef(acc);  // as ClickCubeMap
  (*reinterpret_cast<void(__thiscall***)(void*, void*)>(acc))[0x8C / 4](acc, g_camera);
  wr<bool>(acc, 0x84, true);
  alignas(16) char culler[0xC8];
  reinterpret_cast<void*(__thiscall*)(void*, void*)>(0x00A69400)(culler, nullptr);
  wr<uintptr_t>(culler, 0, 0x0101E2EC);
  wr<uint32_t>(culler, 0x90, kCullForceMultiBoundsNoUpdate);
  wr<uint32_t>(culler, 0xBC, 0);
  wr<uint32_t>(culler, 0xC0, 0);
  wr<void*>(culler, 0xC4, nullptr);
  wr<void*>(culler, 0x0C, g_camera);
  reinterpret_cast<void(__thiscall*)(void*, void*)>(0x00A694A0)(culler, static_cast<char*>(g_camera) + 0xDC);
  addRef(acc);
  wr<void*>(culler, 0xC4, acc);
  const bool fade = reinterpret_cast<bool(__cdecl*)()>(0x0054F4B0)();  // BSFadeNode::GetFadeEnabled
  reinterpret_cast<void(__cdecl*)(bool)>(0x004E20B0)(false);              // BSFadeNode::SetFadeEnabled
  reinterpret_cast<void(__cdecl*)(void*, void*, void*)>(0x00B6BEE0)(g_camera, sceneGraph, culler);
  reinterpret_cast<void(__cdecl*)(void*, void*, int)>(0x00B6C0D0)(g_camera, acc, 0);
  reinterpret_cast<void(__cdecl*)(bool)>(0x004E20B0)(fade);
  release(rd<void*>(culler, 0xC4));
  wr<void*>(culler, 0xC4, nullptr);
  reinterpret_cast<void(__thiscall*)(void*)>(0x004A0F60)(culler);

  reinterpret_cast<void(__cdecl*)()>(0x00B6B790)();
  reinterpret_cast<void(__thiscall*)(void*, bool)>(0x00E757F0)(renderer, false);
  wr<bool>(ssn, 0x130, lightsFrozen);

  if (++g_face == 6) {
    g_face = 0;
    g_current = cubeKey(tex);
    g_drawing ^= 1;
  }
}

bool inWorld() {
  void* player = *reinterpret_cast<void**>(0x011DEA3C);
  return player && reinterpret_cast<void*(__thiscall*)(void*)>(0x008D6F30)(player);
}

using UpdateWaterFn = void(__fastcall*)(void*, void*);
UpdateWaterFn oUpdateWater = nullptr;
void __fastcall hkUpdateWater(void* self, void* edx) {
  if (g_cfg.enable && !g_failed && inWorld()) {
    void* sceneGraph = *reinterpret_cast<void**>(fnv::TESMain_spWorldRoot);
    void* worldCamera = sceneGraph ? rd<void*>(sceneGraph, 0xAC) : nullptr;
    void* ssn = *reinterpret_cast<void**>(0x011F91C8);
    if (sceneGraph && worldCamera && ssn) {
      if (!g_camera && !create()) g_failed = true;
      else if (++g_tick >= g_cfg.interval) {
        g_tick = 0;
        cputime::Scope timed(cputime::Probe);
        captureFace(sceneGraph, worldCamera, ssn);
      }
    }
  }
  // The lighting samples the last complete cube (docs/INDIRECT.md); none outside the world.
  if (Backend* be = shadow::backendIfReady()) be->setEnvironment(g_cfg.enable && !g_failed && inWorld() ? g_current.load() : 0);
  oUpdateWater(self, edx);
}

}  // namespace

uintptr_t current() { return g_current; }

bool install() {
  const std::string ini = iniPath();
  g_cfg.enable = GetPrivateProfileIntA("Probe", "Enable", 1, ini.c_str()) != 0;
  const uint32_t size = GetPrivateProfileIntA("Probe", "Size", 256, ini.c_str());
  g_cfg.size = size >= 32 && size <= 1024 && !(size & (size - 1)) ? size : 256;
  char buf[32];
  GetPrivateProfileStringA("Probe", "Distance", "30000", buf, sizeof(buf), ini.c_str());
  g_cfg.distance = std::max(1000.0f, float(atof(buf)));
  g_cfg.interval = std::clamp<uint32_t>(GetPrivateProfileIntA("Probe", "Interval", 2, ini.c_str()), 1, 8);
  if (!g_cfg.enable) return true;
  return hook(0x008727D0, &hkUpdateWater, &oUpdateWater, "TESMain::DrawWorld_UpdateWater (environment probe)");
}

}  // namespace fnvvk::envprobe
