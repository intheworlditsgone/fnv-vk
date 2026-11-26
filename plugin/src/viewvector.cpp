#include "viewvector.h"

#include <cstdint>
#include <unordered_map>
#include <unordered_set>

#include "common.h"

namespace fnvvk::viewvector {
namespace {

// ShadowLightShader's per-pass constant masks (.data, 759 passes; filled by the Preset*Passes functions as the passes
// are created, zeroed by ShadowLightShader::Initialize 0x00B7A170). Bit i + 1 enables entry i of
// spVertexConstantMapEntries (0x011FEBD8, in the order ShadowLightShader::CreateConstantMaps 0x00B7E430 registers
// them): entry 6 LightData, entry 9 EyePosition.
uint32_t* const kVertexConstantFlags = reinterpret_cast<uint32_t*>(0x011FCC80);
constexpr uint32_t kPassCount = 759;
constexpr uint32_t kLightData = 1u << 7;
constexpr uint32_t kEyePosition = 1u << 10;
// Every lighting pass uploads EyePosition (SetupGeometryConstants_EyePos 0x00B78EF0 computes it, in the object's
// space, or for land relative to NiRenderer::kPosAdjust like the land's vertices, whenever the entry is enabled).
// Applied where the masks are read, so it holds for every pass whenever the engine (re)creates them.
void ensure(uint32_t pass) {
  if (pass < kPassCount && (kVertexConstantFlags[pass] & kLightData)) kVertexConstantFlags[pass] |= kEyePosition;
}

// ShadowLightShader::SetupPassConstants (2.x passes) and ShadowLightShader::EnableConstantsForPass_1x: the two
// readers of the masks.
using SetupPassConstantsFn = void(__fastcall*)(void* self, void* edx, uint32_t pass);
SetupPassConstantsFn oSetupPassConstants = nullptr;
void __fastcall hkSetupPassConstants(void* self, void* edx, uint32_t pass) {
  ensure(pass);
  oSetupPassConstants(self, edx, pass);
}
using EnableConstantsFn = void(__cdecl*)(uint32_t pass);
EnableConstantsFn oEnableConstants = nullptr;
void __cdecl hkEnableConstants(uint32_t pass) {
  ensure(pass);
  oEnableConstants(pass);
}

// The optimised ADT passes (BSSM_ADT_Opt, BSSM_ADT_OptLOD, BSSM_ADT4_Opt; BSBatchRenderer::BeginPass 0x00B99390 skips
// SetupPassConstants for them) set their vertex constants directly, per geometry, in Setup_ADT_Opt 0x00B7DDE0 and
// Setup_ADT4_Opt 0x00B7DED0 (BSBatchRenderer::RenderPassImmediately_Standard), and neither writes EyePosition: c16
// keeps whatever an earlier draw left there (another object's eye position in that object's space). The other two do:
// Setup_ADTS10_Opt 0x00B7DFE0 and Setup_ADT_PxOpt 0x00B7E150 write c16 = the camera's world position
// (BSShaderLightingProperty::kShaderLightConstants[30], the same SetupGeometryConstants_EyePos reads) transformed by
// the geometry's inverse world matrix. After the two that do not, the same is written the same way.
struct Matrix4 { float m[16]; };
using GetInverseWorldFn = void(__cdecl*)(const void* transform, const void* skin, Matrix4* out);
using TransformCoordFn = float*(__stdcall*)(float* out, const float* v, const Matrix4* m);
const auto kGetInverseWorldMatrix = reinterpret_cast<GetInverseWorldFn>(0x00C4C2D0);  // BSUtilities::GetInverseWorldMatrix
const auto kTransformCoord = reinterpret_cast<TransformCoordFn>(0x00EE6DF8);           // D3DXVec3TransformCoord
const float* const kCameraWorldPosition = reinterpret_cast<const float*>(0x011FA2A0);  // kShaderLightConstants[30]
void* const* const kCurrentRenderPass = reinterpret_cast<void* const*>(0x011F91E0);  // BSShaderManager::pCurrentRenderPass
void* const* const kCurrentPass = reinterpret_cast<void* const*>(0x0126F74C);        // NiD3DShader::m_pCurrentPass

void writeEyePosition() {
  const void* renderPass = *kCurrentRenderPass;
  const void* d3dPass = *kCurrentPass;
  if (!renderPass || !d3dPass) return;
  const char* geometry = *reinterpret_cast<const char* const*>(renderPass);        // RenderPass.pGeometry (+0)
  const void* vs = *reinterpret_cast<void* const*>(static_cast<const char*>(d3dPass) + 0x5C);  // NiD3DPass.m_spVertexShader
  if (!geometry || !vs) return;
  void* device = *reinterpret_cast<void* const*>(static_cast<const char*>(vs) + 0x20);  // NiD3DVertexShader.m_pkD3DDevice
  if (!device) return;
  Matrix4 inverseWorld;
  kGetInverseWorldMatrix(geometry + 0x68, nullptr, &inverseWorld);                // NiAVObject::m_kWorld
  float eye[4] = { 0, 0, 0, 1 };
  kTransformCoord(eye, kCameraWorldPosition, &inverseWorld);
  using SetConstantsFn = long(__stdcall*)(void* device, uint32_t reg, const float* v, uint32_t count);
  const auto set = (*reinterpret_cast<SetConstantsFn* const*>(device))[0x178 / 4];  // SetVertexShaderConstantF
  set(device, 16, eye, 1);
}

// The tangent-to-world frame (sdp2spv: locations 13..15 from c250..c252, docs/SHADER_ABI.md "View vector"): the rows
// of the 3x3 part of the matrix the game makes the vertex shader's space with, BSUtilities::GetInverseWorldMatrix of
// the geometry's world transform and skin instance, as ShadowLightShader::SetupGeometryConstants (0x00B7DAB0) does
// for EyePosition. Written for every lighting draw: unskinned ones in ShadowLightShader::SetupGeometryOpt_WorldViewProj
// (0x00B7C750, called by SetupGeometryConstants and by all four Opt setups), skinned ones after SetupGeometryConstants.
constexpr uint32_t kFrameRegister = 250;
void writeFrame(const void* transform, const void* skin) {
  const void* d3dPass = *kCurrentPass;
  if (!d3dPass || !transform) return;
  const void* vs = *reinterpret_cast<void* const*>(static_cast<const char*>(d3dPass) + 0x5C);
  void* device = vs ? *reinterpret_cast<void* const*>(static_cast<const char*>(vs) + 0x20) : nullptr;
  if (!device) return;
  Matrix4 inverseWorld;
  kGetInverseWorldMatrix(transform, skin, &inverseWorld);
  float rows[3][4];
  for (int j = 0; j < 3; j++) {
    for (int c = 0; c < 3; c++) rows[j][c] = inverseWorld.m[4 * j + c];
    rows[j][3] = 0.0f;
  }
  using SetConstantsFn = long(__stdcall*)(void* device, uint32_t reg, const float* v, uint32_t count);
  const auto set = (*reinterpret_cast<SetConstantsFn* const*>(device))[0x178 / 4];  // SetVertexShaderConstantF
  set(device, kFrameRegister, &rows[0][0], 3);
}
// The material's own reflection (docs/INDIRECT.md "Reflections"): c219.x of the pixel shaders is 1 when the geometry's
// shader property has the game's environment-map flag (BSShaderProperty ulFlags[0] (+0x20) & 0x80: the game adds its
// 2x environment-map pass for it, BSShaderPPLightingProperty::GetRenderPasses_2x), else 0. The replaced lighting
// shaders then leave that material's reflection to its environment-map pass. Shade property: NiGeometry m_kProperties
// (+0x9C) m_spShadeProperty (+0xC).
//
// c219.y: 1 in the first specular pass of the geometry in this frame's world pass (multi-pass lighting draws a
// specular pass per group of lights; the environment's reflection is added in one of them only). The pass type:
// BSShaderManager::eCurrentPass (0x011F91E4); the specular passes are 340..375 (BSSM_SPECULARDIR .. 2x_SPECULARPT3_Sb).
constexpr uint32_t kMaterialRegister = 219;
const uint32_t* const kCurrentPassType = reinterpret_cast<const uint32_t*>(0x011F91E4);
bool g_worldPass = false;
std::unordered_set<const void*> g_reflected;  // geometry whose reflection this frame's world pass added
void writeMaterial(const void* geometry, void* device) {
  if (!geometry || !device) return;
  const char* shade = *reinterpret_cast<const char* const*>(static_cast<const char*>(geometry) + 0x9C + 0xC);
  const uint32_t flags = shade ? *reinterpret_cast<const uint32_t*>(shade + 0x20) : 0;
  const uint32_t pass = *kCurrentPassType;
  const bool first = g_worldPass && pass >= 340 && pass <= 375 && g_reflected.insert(geometry).second;
  const float v[4] = { (flags & 0x80) ? 1.0f : 0.0f, first ? 1.0f : 0.0f, 0.0f, 0.0f };
  using SetConstantsFn = long(__stdcall*)(void* device, uint32_t reg, const float* v, uint32_t count);
  const auto set = (*reinterpret_cast<SetConstantsFn* const*>(device))[0x1B4 / 4];  // SetPixelShaderConstantF
  set(device, kMaterialRegister, v, 1);
}
void* currentDevice() {
  const void* d3dPass = *kCurrentPass;
  const void* vs = d3dPass ? *reinterpret_cast<void* const*>(static_cast<const char*>(d3dPass) + 0x5C) : nullptr;
  return vs ? *reinterpret_cast<void* const*>(static_cast<const char*>(vs) + 0x20) : nullptr;
}

using WorldViewProjFn = void(__fastcall*)(void* self, void* edx, void* geometry, void* transform);
WorldViewProjFn oWorldViewProj = nullptr;
void __fastcall hkWorldViewProj(void* self, void* edx, void* geometry, void* transform) {
  oWorldViewProj(self, edx, geometry, transform);
  writeFrame(transform, nullptr);
  writeMaterial(geometry, currentDevice());
}
using SetupGeometryFn = void(__fastcall*)(void* self, void* edx, void* properties);
SetupGeometryFn oSetupGeometry = nullptr;
void __fastcall hkSetupGeometry(void* self, void* edx, void* properties) {
  oSetupGeometry(self, edx, properties);
  const void* renderPass = *kCurrentRenderPass;
  const char* geometry = renderPass ? *reinterpret_cast<const char* const*>(renderPass) : nullptr;
  const void* skin = geometry ? *reinterpret_cast<const void* const*>(geometry + 0xBC) : nullptr;  // m_spSkinInstance
  if (skin) {
    writeFrame(geometry + 0x68, skin);
    writeMaterial(geometry, currentDevice());
  }
}

using SetupOptFn = void(__fastcall*)(void* self, void* edx, void* properties);
SetupOptFn oSetupAdtOpt = nullptr, oSetupAdt4Opt = nullptr;
void __fastcall hkSetupAdtOpt(void* self, void* edx, void* properties) {
  oSetupAdtOpt(self, edx, properties);
  writeEyePosition();
}
void __fastcall hkSetupAdt4Opt(void* self, void* edx, void* properties) {
  oSetupAdt4Opt(self, edx, properties);
  writeEyePosition();
}

// The grass's canopy height (sdp2spv's grass pass: c250.x, docs/LIGHTING.md "Grass"): the top of the drawn mesh, its
// highest model-space vertex z (the grass vertex shaders take positions in model space, scaled and offset per
// instance). TallGrassShader::SetupGeometryConstants (0x00BAC9C0, vtable slot 31, as ShadowLightShader's) sets the
// geometry's constants; the geometry is BSShaderManager::pCurrentRenderPass->pGeometry. Its NiGeometry::m_spModelData
// (+0xB8): NiGeometryData m_usVertices (+0x8), m_kBound (+0x10: centre, radius), m_pkVertex (+0x20). Where the
// vertices are not kept, the bound's top. Cached per model data.
struct CanopyKey {
  const void* data;
  uint32_t vertices;
  bool operator==(const CanopyKey& o) const { return data == o.data && vertices == o.vertices; }
};
struct CanopyKeyHash {
  size_t operator()(const CanopyKey& k) const { return std::hash<const void*>()(k.data) ^ (size_t(k.vertices) << 16); }
};
std::unordered_map<CanopyKey, float, CanopyKeyHash> g_canopy;
float canopyHeight(const char* geometry) {
  const char* data = *reinterpret_cast<const char* const*>(geometry + 0xB8);
  if (!data) return 1.0f;
  const uint32_t n = *reinterpret_cast<const uint16_t*>(data + 0x8);
  const CanopyKey key{ data, n };
  auto it = g_canopy.find(key);
  if (it != g_canopy.end()) return it->second;
  const float* v = *reinterpret_cast<const float* const*>(data + 0x20);
  float top;
  if (v && n) {
    top = v[2];
    for (uint32_t i = 1; i < n; i++) top = v[3 * i + 2] > top ? v[3 * i + 2] : top;
  } else {
    const float* bound = reinterpret_cast<const float*>(data + 0x10);
    top = bound[2] + bound[3];
  }
  if (g_canopy.size() > 4096) g_canopy.clear();
  g_canopy[key] = top;
  return top;
}
using GrassSetupFn = void(__fastcall*)(void* self, void* edx, void* state);
GrassSetupFn oGrassSetup = nullptr;
void __fastcall hkGrassSetup(void* self, void* edx, void* state) {
  oGrassSetup(self, edx, state);
  const void* renderPass = *kCurrentRenderPass;
  const char* geometry = renderPass ? *reinterpret_cast<const char* const*>(renderPass) : nullptr;
  void* device = currentDevice();
  if (!geometry || !device) return;
  const float c[4] = { canopyHeight(geometry), 0.0f, 0.0f, 0.0f };
  using SetConstantsFn = long(__stdcall*)(void* device, uint32_t reg, const float* v, uint32_t count);
  const auto set = (*reinterpret_cast<SetConstantsFn* const*>(device))[0x178 / 4];  // SetVertexShaderConstantF
  set(device, kFrameRegister, c, 1);
}

// Lighting30Shader's per-pass pixel constant masks (.data 0x01200868, one per pass-table index, 112 up to the vertex
// masks at 0x01200A28). Lighting30Shader::SetupPassConstants (0x00BBE120) maps the pass to its index and sets each
// entry of the pixel constant map on or off from bit i + 1 (entries in the order Lighting30Shader::CreateConstantMaps
// 0x00BBF5A0 registers them: ambientcolor, eyepos, hairtint, matalpha, matemit, refractpower, parallaxdata, waterfog,
// envmaptoggles, lightdata). With eyepos on, UpdateTransformedPosition (0x00BBB890) computes it per geometry (the
// camera in the space of the shader's position input) and the map uploads it to c1.
uint32_t* const kLighting30PixelFlags = reinterpret_cast<uint32_t*>(0x01200868);
constexpr uint32_t kLighting30Entries = (0x01200A28 - 0x01200868) / 4;
constexpr uint32_t kL30EyePos = 1u << 2, kL30LightData = 1u << 10;
using L30SetupPassFn = void(__fastcall*)(void* self, void* edx, int pass);
L30SetupPassFn oL30SetupPass = nullptr;
void __fastcall hkL30SetupPass(void* self, void* edx, int pass) {
  for (uint32_t i = 0; i < kLighting30Entries; i++)
    if (kLighting30PixelFlags[i] & kL30LightData) kLighting30PixelFlags[i] |= kL30EyePos;
  oL30SetupPass(self, edx, pass);
}

}  // namespace

void beginWorld() {
  g_reflected.clear();
  g_worldPass = true;
}
void endWorld() { g_worldPass = false; }

bool install(bool needed) {
  if (!needed) return true;
  bool ok = hook(0x00B7B450, &hkSetupPassConstants, &oSetupPassConstants,
                 "ShadowLightShader::SetupPassConstants (EyePosition for the view vector)");
  ok &= hook(0x00C02F70, &hkEnableConstants, &oEnableConstants,
             "ShadowLightShader::EnableConstantsForPass_1x (EyePosition for the view vector)");
  ok &= hook(0x00B7DDE0, &hkSetupAdtOpt, &oSetupAdtOpt, "ShadowLightShader::Setup_ADT_Opt (EyePosition for the view vector)");
  ok &= hook(0x00B7DED0, &hkSetupAdt4Opt, &oSetupAdt4Opt,
             "ShadowLightShader::Setup_ADT4_Opt (EyePosition for the view vector)");
  ok &= hook(0x00B7C750, &hkWorldViewProj, &oWorldViewProj,
             "ShadowLightShader::SetupGeometryOpt_WorldViewProj (tangent-to-world frame)");
  ok &= hook(0x00B7DAB0, &hkSetupGeometry, &oSetupGeometry,
             "ShadowLightShader::SetupGeometryConstants (tangent-to-world frame, skinned)");
  ok &= hook(0x00BAC9C0, &hkGrassSetup, &oGrassSetup, "TallGrassShader::SetupGeometryConstants (the grass's canopy height)");
  return ok;
}

bool installLighting30(bool needed) {
  if (!needed) return true;
  return hook(0x00BBE120, &hkL30SetupPass, &oL30SetupPass, "Lighting30Shader::SetupPassConstants (eyepos for the hair)");
}

}  // namespace fnvvk::viewvector
