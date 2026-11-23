// Upscaling (backend.h UpscaleSettings; the work is the backend's, backend_ffx.inl).
//
// The game's frame (layouts and signatures from the IDA database of FalloutNV.exe 1.4.0.525):
// - TESMain::DrawWorldStandard 0x00870BD0 (__thiscall: BSRenderedTexture* apCustomRenderTarget, bool bIsPipBoyMode;
//   retn 8) draws the world into the scene texture: water and offscreen buffers, the scene (DrawWorld_DrawScene
//   0x00873200), refraction, the first-person view (after a depth clear), then image space (DrawWorld_ImageSpace ->
//   BSShaderManager::RenderImageSpaceEffects 0x00B55AC0, __cdecl: NiDX9Renderer*, BSRenderedTexture* source,
//   BSRenderedTexture* destination, null for the back buffer). The interface is drawn after it returns.
//   With an upscaler, the frame's world (a DrawWorldStandard without a custom target) is a scaled scene: from its entry to
//   image space. The scene is upscaled when image space is about to run (its source: BSRenderedTexture
//   spRenderTargetGroups[0] +0x08 -> NiRenderTargetGroup m_aspBuffers[0] +0x0C (colour), m_spDepthStencilBuffer +0x20
//   (depth) -> Ni2DBuffer +0x10 renderer data, the backend's key). A scaled scene that reaches no image space ends with
//   DrawWorldStandard.
// - Jitter: NiDX9Renderer::Do_SetCameraData 0x00E6C780 (__thiscall: NiPoint3* loc, dir, up, right, NiFrustum*,
//   NiRect<float>* port; retn 18h) builds m_kD3DView (+0x980), m_kD3DProj (+0x9C0; perspective: _31/_32 the frustum's
//   offset, _34 = 1) and m_kViewProj (+0xA00) = view * proj, which the shader constants are made from
//   (NiD3DShaderConstantMap::SetDefinedConstant, ShadowLightShader::SetupPassConstants). In a scaled scene, for a
//   perspective camera drawing into a window-sized group (m_pkCurrRenderTargetGroup +0x888), the projection is moved by
//   FSR's jitter (2 x / renderW, -2 y / renderH in clip space): _31, _32 and the matching view * proj terms, and the
//   device's D3DTS_PROJECTION again.
// - The world's depth before the first-person view: when DrawWorld_DrawScene returns (effects.cpp's hook).
#include "upscale.h"

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

#include "addresses.h"
#include "backend.h"
#include "common.h"
#include "shadow.h"

namespace fnvvk::upscale {
namespace {

constexpr uint32_t kRendDevice = 0x288, kRendDeviceLost = 0x6FC, kRendDefaultGroup = 0x884, kRendCurrentGroup = 0x888;
constexpr uint32_t kRendView = 0x980, kRendProj = 0x9C0, kRendViewProj = 0xA00;
constexpr uint32_t kGroupBuffers = 0x0C, kGroupDepth = 0x20, kBufWidth = 0x08, kBufHeight = 0x0C, kBufData = 0x10;
constexpr uint32_t kRenderedTextureGroups = 0x08;
constexpr uint32_t kFrustumOrtho = 0x18;

Settings g_settings;
std::string g_ini;

// The frame's world: a scaled scene with this jitter (render pixels) and render size.
bool g_scaled = false;
bool g_upscaledThisScene = false;
float g_jitter[2] = {};
uint32_t g_renderW = 0, g_renderH = 0, g_displayW = 0, g_displayH = 0;
uint32_t g_jitterIndex = 0;
uint64_t g_jitteredCameras = 0, g_scenes = 0;
// Frame timing and camera cuts.
LARGE_INTEGER g_freq{}, g_lastScene{};
bool g_havePrev = false;
float g_prevOrigin[3] = {};

UpscaleSettings toBackend(const Settings& s) {
  UpscaleSettings u;
  u.upscaler = s.upscaler == Settings::Fsr3 ? UpscaleSettings::Fsr3 : UpscaleSettings::Off;
  u.quality = UpscaleSettings::QualityMode(s.quality);
  u.sharpen = s.sharpen;
  u.sharpness = s.sharpness;
  u.library = moduleDir() + "fnv_vk\\fnv_vk_fidelityfx.dll";
  return u;
}

// The window (the default render-target group's buffer 0).
bool displaySize(const void* renderer, uint32_t& w, uint32_t& h) {
  const void* group = renderer ? rd<const void*>(renderer, kRendDefaultGroup) : nullptr;
  const void* buf = group ? rd<const void*>(group, kGroupBuffers) : nullptr;
  if (!buf) return false;
  w = rd<uint32_t>(buf, kBufWidth);
  h = rd<uint32_t>(buf, kBufHeight);
  return w && h;
}

RtKey bufferKey(const void* buffer) { return buffer ? RtKey(rd<const void*>(buffer, kBufData)) : 0; }

// The world camera now (as effects.cpp reads it for effects at "world").
bool worldCamera(SceneView& v) {
  const void* sceneGraph = *reinterpret_cast<void* const*>(fnv::TESMain_spWorldRoot);
  const void* camera = sceneGraph ? rd<const void*>(sceneGraph, 0xAC) : nullptr;
  if (!camera || rd<bool>(camera, 0xDC + 24)) return false;
  for (int i = 0; i < 4; i++) v.frustum[i] = rd<float>(camera, 0xDC + 4 * i);
  v.nearZ = rd<float>(camera, 0xDC + 16);
  v.farZ = rd<float>(camera, 0xDC + 20);
  for (int i = 0; i < 4; i++) v.port[i] = rd<float>(camera, 0x100 + 4 * i);
  for (int row = 0; row < 3; ++row) {
    memcpy(v.cameraRows[row], static_cast<const char*>(camera) + 0x68 + 12 * row, 12);
    v.cameraRows[row][3] = rd<float>(camera, 0x8C + 4 * row);
  }
  return v.nearZ > 0.0f && v.farZ > v.nearZ;
}

// --- TESMain::DrawWorldStandard: the scaled scene ---------------------------------------------------------------------
using DrawWorldFn = void(__fastcall*)(void* self, void* edx, void* customTarget, bool pipBoyMode);
DrawWorldFn oDrawWorld = nullptr;
void __fastcall hkDrawWorld(void* self, void* edx, void* customTarget, bool pipBoyMode) {
  Backend* be = shadow::backendIfReady();
  const void* renderer = *reinterpret_cast<void* const*>(fnv::NiRenderer_ms_pRenderer);
  g_scaled = false;
  g_upscaledThisScene = false;
  if (be && !customTarget && g_settings.upscaler != Settings::Off && be->upscaleStatus().libraryLoaded &&
      displaySize(renderer, g_displayW, g_displayH)) {
    Backend::renderSize(toBackend(g_settings), g_displayW, g_displayH, g_renderW, g_renderH);
    Backend::jitterOffset(g_jitterIndex++, g_renderW, g_displayW, g_jitter[0], g_jitter[1]);
    be->beginScaledScene();
    g_scaled = true;
    g_scenes++;
  }
  oDrawWorld(self, edx, customTarget, pipBoyMode);
  if (g_scaled && !g_upscaledThisScene && be) be->endScaledScene();  // no image space this frame
  g_scaled = false;
}

// --- NiDX9Renderer::Do_SetCameraData: jitter ---------------------------------------------------------------------------
using SetCameraFn = void(__fastcall*)(void* self, void* edx, const void* loc, const void* dir, const void* up,
                                      const void* right, const void* frustum, const void* port);
SetCameraFn oSetCamera = nullptr;
void __fastcall hkSetCamera(void* self, void* edx, const void* loc, const void* dir, const void* up, const void* right,
                            const void* frustum, const void* port) {
  oSetCamera(self, edx, loc, dir, up, right, frustum, port);
  if (!g_scaled || !self || !frustum || rd<bool>(frustum, kFrustumOrtho) || rd<bool>(self, kRendDeviceLost)) return;
  // Only cameras drawing into a window-sized group (the scene, the first-person view), not reflections or shadow maps.
  const void* group = rd<const void*>(self, kRendCurrentGroup);
  const void* buf = group ? rd<const void*>(group, kGroupBuffers) : nullptr;
  if (!buf || rd<uint32_t>(buf, kBufWidth) != g_displayW || rd<uint32_t>(buf, kBufHeight) != g_displayH) return;
  const float dx = 2.0f * g_jitter[0] / float(g_renderW), dy = -2.0f * g_jitter[1] / float(g_renderH);
  float* proj = reinterpret_cast<float*>(static_cast<char*>(self) + kRendProj);  // row-major 4x4
  const float* view = reinterpret_cast<const float*>(static_cast<char*>(self) + kRendView);
  float* viewProj = reinterpret_cast<float*>(static_cast<char*>(self) + kRendViewProj);
  // clip = v * P: with w = z (P._34 = 1), adding dx to P._31 moves x / w by dx. view * proj changes by view's third
  // column times the change of proj's third row.
  proj[2 * 4 + 0] += dx;
  proj[2 * 4 + 1] += dy;
  for (int r = 0; r < 4; r++) {
    viewProj[r * 4 + 0] += view[r * 4 + 2] * dx;
    viewProj[r * 4 + 1] += view[r * 4 + 2] * dy;
  }
  // The device's projection too (IDirect3DDevice9::SetTransform, vtable slot 44; D3DTS_PROJECTION = 3).
  if (void* device = rd<void*>(self, kRendDevice)) {
    using SetTransformFn = long(__stdcall*)(void*, uint32_t, const float*);
    reinterpret_cast<SetTransformFn>((*reinterpret_cast<void***>(device))[44])(device, 3, proj);
  }
  g_jitteredCameras++;
}

}  // namespace

Settings& settings() { return g_settings; }

void load() {
  g_ini = iniPath();
  const char* sec = "Upscaling";
  Settings s;
  s.upscaler = Settings::Upscaler(std::clamp(int(GetPrivateProfileIntA(sec, "Upscaler", 0, g_ini.c_str())), 0, 1));
  s.quality = Settings::QualityMode(std::clamp(int(GetPrivateProfileIntA(sec, "Quality", 1, g_ini.c_str())), 0, 4));
  s.sharpen = GetPrivateProfileIntA(sec, "Sharpen", 1, g_ini.c_str()) != 0;
  char buf[32] = "";
  GetPrivateProfileStringA(sec, "Sharpness", "0.5", buf, sizeof(buf), g_ini.c_str());
  s.sharpness = std::clamp(float(atof(buf)), 0.0f, 1.0f);
  g_settings = s;
  logf("upscaling: upscaler %s (quality %d, sharpen %d %.2f)", s.upscaler == Settings::Fsr3 ? "FSR 3.1" : "off",
       int(s.quality), int(s.sharpen), s.sharpness);
}

void apply(Backend* be) {
  if (!be) return;
  be->setUpscaling(toBackend(g_settings));
  const UpscaleStatus st = be->upscaleStatus();
  if (!st.error.empty()) logf("upscaling: %s", st.error.c_str());
}

bool save() {
  const char* sec = "Upscaling";
  char v[32];
  bool ok = true;
  auto put = [&](const char* key, const char* value) { ok &= WritePrivateProfileStringA(sec, key, value, g_ini.c_str()) != 0; };
  put("Upscaler", g_settings.upscaler == Settings::Fsr3 ? "1" : "0");
  snprintf(v, sizeof(v), "%d", int(g_settings.quality));
  put("Quality", v);
  put("Sharpen", g_settings.sharpen ? "1" : "0");
  snprintf(v, sizeof(v), "%.2f", g_settings.sharpness);
  put("Sharpness", v);
  WritePrivateProfileStringA(sec, "FrameGeneration", nullptr, g_ini.c_str());  // a removed setting: dropped
  logf("upscaling: settings %s fnv_vk.ini", ok ? "saved to" : "could NOT be saved to");
  return ok;
}

bool install() {
  QueryPerformanceFrequency(&g_freq);
  load();
  bool ok = hook(0x00870BD0, &hkDrawWorld, &oDrawWorld, "TESMain::DrawWorldStandard (upscaling: scaled scene)");
  ok &= hook(0x00E6C780, &hkSetCamera, &oSetCamera, "NiDX9Renderer::Do_SetCameraData (upscaling: jitter)");
  return ok;
}

void afterDrawScene() {
  Backend* be = shadow::backendIfReady();
  if (!be || !g_scaled) return;
  be->captureWorldDepth();
}

void beforeImageSpace(void* source, void* /*destination*/) {
  Backend* be = shadow::backendIfReady();
  if (!be || !g_scaled || !source || g_upscaledThisScene) return;
  const void* group = rd<const void*>(source, kRenderedTextureGroups);
  const void* colorBuf = group ? rd<const void*>(group, kGroupBuffers) : nullptr;
  const void* depthBuf = group ? rd<const void*>(group, kGroupDepth) : nullptr;
  if (!colorBuf || !depthBuf) {
    if (g_scaled) be->endScaledScene();
    g_upscaledThisScene = g_scaled;
    return;
  }
  SceneView v;
  if (!worldCamera(v)) {
    if (g_scaled) be->endScaledScene();
    g_upscaledThisScene = g_scaled;
    return;
  }
  LARGE_INTEGER now;
  QueryPerformanceCounter(&now);
  const double ms = g_lastScene.QuadPart ? double(now.QuadPart - g_lastScene.QuadPart) * 1000.0 / double(g_freq.QuadPart) : 16.7;
  g_lastScene = now;
  v.frameTimeMs = float(std::min(ms, 1000.0));
  if (g_scaled) {
    v.jitter[0] = g_jitter[0];
    v.jitter[1] = g_jitter[1];
  }
  // A camera cut: a long pause (loading, menus) or a jump no frame of movement makes (teleport, cell change). Bethesda
  // units: 70 to the metre; 700 units is 10 m in one frame.
  float jump = 0.0f;
  for (int i = 0; i < 3; i++) jump += (v.cameraRows[i][3] - g_prevOrigin[i]) * (v.cameraRows[i][3] - g_prevOrigin[i]);
  v.reset = !g_havePrev || ms > 500.0 || jump > 700.0f * 700.0f;
  for (int i = 0; i < 3; i++) g_prevOrigin[i] = v.cameraRows[i][3];
  g_havePrev = true;
  be->upscaleScene(bufferKey(colorBuf), bufferKey(depthBuf), v);
  g_upscaledThisScene = g_scaled;
  g_scaled = false;  // image space and what follows are not jittered
}

}  // namespace fnvvk::upscale
