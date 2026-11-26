#include "cputime.h"
#include "shadow.h"

#include <windows.h>
#include <d3d9.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

#include "addresses.h"
#include "backend.h"
#include "common.h"
#include "display.h"
#include "effects.h"
#include "envprobe.h"
#include "gbbuffers.h"
#include "gbdevice.h"
#include "gbobjects.h"
#include "gbquery.h"
#include "gbshaders.h"
#include "gbstate.h"
#include "gbsurfaces.h"
#include "movie.h"
#include "observe.h"
#include "overlay.h"
#include "present.h"
#include "replacements.h"
#include "screenshot.h"
#include "sunshadows.h"
#include "textures.h"
#include "upscale.h"
#include "viewvector.h"

namespace fnvvk::shadow {

namespace {

// Layouts read from FalloutNV.exe machine code (docs/M2_RESULTS.md, M2_DESIGN.md).
constexpr uint32_t kGroupBuffers = 0x0C, kGroupCount = 0x1C, kGroupDepth = 0x20;
constexpr uint32_t kBufWidth = 0x08, kBufHeight = 0x0C, kBufData = 0x10;
constexpr uint32_t kDataPixelFormat = 0x0C, kDataMsaaPref = 0x10, kDataSurface = 0x14, kDataMsSurface = 0x18,
                   kDataMsaaLevel = 0x1C;  // D3DMULTISAMPLE_TYPE; what Do_EndUsingRenderTargetGroup resolves by
constexpr uint32_t kImplicitPresentParams = 0x20, kPPMultiSampleType = 0x10;
constexpr uint32_t kPixFmtRendererHint = 0x0C;
constexpr uint32_t kRendBackground = 0x5E0, kRendZClear = 0x5E4, kRendStencilClear = 0x5E8, kRendDeviceLost = 0x6FC,
                   kRendDefaultGroup = 0x884, kRendCurrentGroup = 0x888, kRendDevice = 0x288;

uint32_t g_compareEvery = 0, g_compareDumps = 0, g_framesSinceCompare = 0, g_comparesDone = 0;
Backend* g_be = nullptr;  // never deleted: tearing Vulkan down during process exit would run under the loader lock
bool g_enabled = true;

struct Counters {
  std::atomic<uint64_t> attachChecked{ 0 }, attachDescMismatch{ 0 }, attachUnknownClass{ 0 }, destroyUnknown{ 0 },
    groupsDeviceLost{ 0 }, clearsWithRect{ 0 };
} g_c;
std::atomic<uint32_t> g_mismatchLogs{ 0 };

Backend& be() {
  if (!g_be) g_be = new Backend();
  return *g_be;
}

// Sample count of a renderer-data object, from Gamebryo fields only.
uint32_t samplesOf(uintptr_t vt, const void* data) {
  auto fromType = [](uint32_t t) { return t >= 2 && t <= 16 ? t : 1u; };  // D3DMULTISAMPLE_n_SAMPLES == n
  switch (vt) {
    case fnv::vtbl_NiDX9TextureBufferData: return fromType(rd<uint32_t>(data, kDataMsaaLevel));
    case fnv::vtbl_NiDX9ImplicitBufferData:
      return fromType(rd<uint32_t>(data, kImplicitPresentParams + kPPMultiSampleType));
    case fnv::vtbl_NiDX9AdditionalDepthStencilBufferData: {
      // Ni2DBuffer::MultiSamplePreference -> D3DMULTISAMPLE_TYPE, as NiDX9AdditionalDepthStencilBufferData::Create
      // (0x00E7D9D0) maps it: 1 -> 2, 2 -> 4, 3 -> 8 samples.
      uint32_t pref = rd<uint32_t>(data, kDataMsaaPref);
      return pref == 1 ? 2 : pref == 2 ? 4 : pref == 3 ? 8 : 1;
    }
    default: return 1;
  }
}

bool isDepthClass(uintptr_t vt) {
  return vt == fnv::vtbl_NiDX9ImplicitDepthStencilBufferData || vt == fnv::vtbl_NiDX9AdditionalDepthStencilBufferData;
}
bool isKnownClass(uintptr_t vt) {
  return vt == fnv::vtbl_NiDX92DBufferData || vt == fnv::vtbl_NiDX9TextureBufferData ||
         vt == fnv::vtbl_NiDX9ImplicitBufferData || isDepthClass(vt);
}

// A face of a rendered cube map (NiDX9RenderedCubeMapData::Create 0x00E901A0: NiDX9TextureBufferData::CreateCubeFace
// 0x00E7D390 per face, m_pkRenderTexture (+0x20) the cube texture, m_pkSurface (+0x14) its level-0 face surface): the
// cube texture, the face index and the level count the cube is created with (0: the full chain, for
// D3DUSAGE_AUTOGENMIPMAP). Through the D3D9 interfaces only.
struct CubeFace { IDirect3DCubeTexture9* cube = nullptr; uint32_t face = 0, levels = 1; };
bool cubeFaceOf(const void* data, CubeFace& out) {
  if (rd<uintptr_t>(data, 0) != fnv::vtbl_NiDX9TextureBufferData) return false;
  auto* tex = rd<IDirect3DBaseTexture9*>(data, 0x20);
  void* surface = rd<void*>(data, kDataSurface);
  if (!tex || !surface || tex->GetType() != D3DRTYPE_CUBETEXTURE) return false;
  auto* cube = static_cast<IDirect3DCubeTexture9*>(tex);
  for (uint32_t f = 0; f < 6; f++) {
    IDirect3DSurface9* s = nullptr;
    if (cube->GetCubeMapSurface(D3DCUBEMAP_FACES(f), 0, &s) < 0 || !s) continue;
    const bool match = s == surface;
    s->Release();
    if (!match) continue;
    D3DSURFACE_DESC d{};
    cube->GetLevelDesc(0, &d);
    out = { cube, f, (d.Usage & D3DUSAGE_AUTOGENMIPMAP) ? 0u : cube->GetLevelCount() };
    return true;
  }
  return false;
}
// The faces of each rendered cube map seen so far; the backend gets the cube once all six are attached.
std::mutex g_cubeMutex;
std::vector<std::pair<IDirect3DCubeTexture9*, std::array<RtKey, 6>>> g_cubeFaces;
void noteCubeFace(const CubeFace& c, RtKey key) {
  std::lock_guard<std::mutex> lock(g_cubeMutex);
  auto it = std::find_if(g_cubeFaces.begin(), g_cubeFaces.end(), [&](const auto& e) { return e.first == c.cube; });
  if (it == g_cubeFaces.end()) it = g_cubeFaces.insert(g_cubeFaces.end(), { c.cube, {} });
  it->second[c.face] = key;
  if (std::all_of(it->second.begin(), it->second.end(), [](RtKey k) { return k != 0; }))
    be().attachCube(TexKey(c.cube), it->second.data(), c.levels);
}
void forgetCubeFace(RtKey key) {
  std::lock_guard<std::mutex> lock(g_cubeMutex);
  for (auto it = g_cubeFaces.begin(); it != g_cubeFaces.end(); ++it) {
    for (RtKey& k : it->second)
      if (k == key) k = 0;
    if (std::all_of(it->second.begin(), it->second.end(), [](RtKey k) { return k == 0; })) {
      g_cubeFaces.erase(it);
      return;
    }
  }
}

bool describe(const void* buffer, const void* data, RtDesc& d) {
  uintptr_t vt = rd<uintptr_t>(data, 0);
  const void* fmt = rd<const void*>(data, kDataPixelFormat);
  if (!fmt) return false;
  d.width = rd<uint32_t>(buffer, kBufWidth);
  d.height = rd<uint32_t>(buffer, kBufHeight);
  d.d3dFormat = rd<uint32_t>(fmt, kPixFmtRendererHint);
  d.samples = samplesOf(vt, data);
  d.depth = isDepthClass(vt);
  return true;
}

// Verification only: compares the Gamebryo-derived descriptor with the D3D9 surface the game created
// (IDirect3DSurface9::GetDesc, vtable index 12). For MSAA colour targets the multisampled surface is m_pkSurface2.
struct SurfDesc { uint32_t Format, Type, Usage, Pool, MultiSampleType, MultiSampleQuality, Width, Height; };
bool d3dDesc(void* surface, SurfDesc& out) {
  if (!surface) return false;
  using GetDescFn = long(__stdcall*)(void*, SurfDesc*);
  auto fn = reinterpret_cast<GetDescFn>((*reinterpret_cast<void***>(surface))[12]);
  return fn(surface, &out) >= 0;
}

void verifyAttach(const void* buffer, const void* data, const RtDesc& d) {
  uintptr_t vt = rd<uintptr_t>(data, 0);
  void* surf = rd<void*>(data, kDataSurface);
  void* ms = d.depth ? nullptr : rd<void*>(data, kDataMsSurface);
  SurfDesc s{}, sm{};
  if (!d3dDesc(surf, s)) return;
  g_c.attachChecked++;
  uint32_t d3dSamples = s.MultiSampleType >= 2 ? s.MultiSampleType : 1;
  if (ms && d3dDesc(ms, sm)) d3dSamples = sm.MultiSampleType >= 2 ? sm.MultiSampleType : 1;
  bool ok = s.Width == d.width && s.Height == d.height && s.Format == d.d3dFormat && d3dSamples == d.samples;
  if (!ok) {
    g_c.attachDescMismatch++;
    if (g_mismatchLogs++ < 32)
      logf("ATTACH MISMATCH buffer %p data %p class %08X: gamebryo %ux%u fmt %u samples %u depth %d | d3d9 %ux%u fmt %u "
           "samples %u (ms surface %p)", buffer, data, unsigned(vt), d.width, d.height, d.d3dFormat, d.samples, d.depth,
           s.Width, s.Height, s.Format, d3dSamples, ms);
  }
}

void attachFromBuffer(const void* buffer, const void* data, const char* why) {
  __try {
    uintptr_t vt = rd<uintptr_t>(data, 0);
    if (!isKnownClass(vt)) {
      g_c.attachUnknownClass++;
      logf("attach (%s): renderer data %p has unknown class %08X; not tracked", why, data, unsigned(vt));
      return;
    }
    CubeFace cube;
    const bool isCubeFace = cubeFaceOf(data, cube);
    if (isCubeFace) {
      // CreateCubeFace leaves m_pkSurface2 and the MSAA level uninitialised (NiNew'd, never written); a cube face is
      // single-sample, as NiDX9TextureBufferData::Create sets them for a texture without MSAA.
      wr<uint32_t>(const_cast<void*>(data), kDataMsSurface, 0);
      wr<uint32_t>(const_cast<void*>(data), kDataMsaaLevel, 0);
    }
    RtDesc d;
    if (!describe(buffer, data, d)) return;
    be().attach(RtKey(data), d);
    if (isCubeFace) noteCubeFace(cube, RtKey(data));
    observe::registerSurface(rd<void*>(data, kDataSurface), RtKey(data));
    // NiDX9TextureBufferData::Create 0x00E7D250: the render texture is m_pkRenderTexture (+0x20) of the 0x24-byte object.
    if (vt == fnv::vtbl_NiDX9TextureBufferData) observe::registerRtTexture(rd<void*>(data, 0x20), RtKey(data));
    verifyAttach(buffer, data, d);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    logf("attach (%s): exception reading buffer %p data %p", why, buffer, data);
  }
}

// --- lifetime hooks -------------------------------------------------------------------------------------------
using SetRendererDataFn = void(__fastcall*)(void*, void*, void*);
SetRendererDataFn oSetRendererData;
void __fastcall hkSetRendererData(void* buffer, void* edx, void* data) {
  oSetRendererData(buffer, edx, data);
  // Ni2DBuffer::Create stores the pointer before calling this (the setter then returns early), so attach whenever
  // the buffer holds `data` afterwards. Width, height and pixel format are set before every call (all six callers).
  if (g_enabled && data) attachFromBuffer(buffer, data, "SetRendererData");
}

using DeletingDtorFn = void*(__fastcall*)(void*, void*, unsigned);
DeletingDtorFn oDtor2D, oDtorTexture, oDtorImplicit, oDtorAdditionalDS;
void onDestroyed(void* data) {
  if (!g_enabled) return;
  observe::unregisterRtKey(RtKey(data));
  forgetCubeFace(RtKey(data));
  if (!be().destroy(RtKey(data))) g_c.destroyUnknown++;
}
void* __fastcall hkDtor2D(void* self, void* edx, unsigned flags) { onDestroyed(self); return oDtor2D(self, edx, flags); }
void* __fastcall hkDtorTexture(void* self, void* edx, unsigned flags) { onDestroyed(self); return oDtorTexture(self, edx, flags); }
void* __fastcall hkDtorImplicit(void* self, void* edx, unsigned flags) { onDestroyed(self); return oDtorImplicit(self, edx, flags); }
void* __fastcall hkDtorAdditionalDS(void* self, void* edx, unsigned flags) { onDestroyed(self); return oDtorAdditionalDS(self, edx, flags); }

using ResetDimensionsFn = void(__fastcall*)(void*, void*, unsigned, unsigned);
ResetDimensionsFn oResetDimensions;
void __fastcall hkResetDimensions(void* buffer, void* edx, unsigned w, unsigned h) {
  oResetDimensions(buffer, edx, w, h);
  if (!g_enabled) return;
  if (const void* data = rd<const void*>(buffer, kBufData)) be().resize(RtKey(data), w, h);
}

// Device reset: Recreate rebuilds the D3D surfaces inside existing renderer data; size and format may change.
using RecreateFn = char(__fastcall*)(void*, void*, void*);
RecreateFn oRecreateImplicit, oRecreateImplicitDS, oRecreateAdditionalDS;
void reattach(void* data) {
  if (!g_enabled) return;
  if (const void* buffer = rd<const void*>(data, 0x08)) attachFromBuffer(buffer, data, "Recreate");
}
char __fastcall hkRecreateImplicit(void* self, void* edx, void* dev) { char r = oRecreateImplicit(self, edx, dev); if (r) reattach(self); return r; }
char __fastcall hkRecreateImplicitDS(void* self, void* edx, void* dev) { char r = oRecreateImplicitDS(self, edx, dev); if (r) reattach(self); return r; }
char __fastcall hkRecreateAdditionalDS(void* self, void* edx, void* dev) { char r = oRecreateAdditionalDS(self, edx, dev); if (r) reattach(self); return r; }

// Do_ClearBuffer (0x00E6F310), mirrored: same early-outs, same rect arithmetic, same clear values.
using ClearBufferFn = void(__fastcall*)(void*, void*, const float*, unsigned);
ClearBufferFn oClearBuffer;
void __fastcall hkClearBuffer(void* renderer, void* edx, const float* r, unsigned mode) {
  oClearBuffer(renderer, edx, r, mode);
  if (!g_enabled || !g_be || !g_be->ready() || !mode || rd<bool>(renderer, kRendDeviceLost)) return;
  const void* group = rd<const void*>(renderer, kRendCurrentGroup);
  if (!group) return;
  const void* buf0 = rd<const void*>(group, kGroupBuffers);
  uint32_t w = buf0 ? rd<uint32_t>(buf0, kBufWidth) : 0, h = buf0 ? rd<uint32_t>(buf0, kBufHeight) : 0;
  int32_t rect[4] = { 0, 0, int32_t(w), int32_t(h) };
  if (r) {
    // NiRect<float> {left, right, top, bottom}: x1 = left*W, y1 = (1-top)*H, x2 = right*W, y2 = (1-bottom)*H,
    // truncated (_ftol2_sse).
    g_c.clearsWithRect++;
    float fw = float(w), fh = float(h);
    rect[0] = int32_t(r[0] * fw);
    rect[1] = int32_t((1.0f - r[2]) * fh);
    rect[2] = int32_t(r[1] * fw);
    rect[3] = int32_t((1.0f - r[3]) * fh);
  }
  uint32_t flags = ((mode & 1) ? kClearTarget : 0) | ((mode & 4) ? kClearZBuffer : 0) | ((mode & 2) ? kClearStencil : 0);
  g_be->clear(flags, rd<uint32_t>(renderer, kRendBackground), rd<float>(renderer, kRendZClear),
              rd<uint32_t>(renderer, kRendStencilClear), rect);
}

RtKey dataKey(const void* buffer) { return buffer ? RtKey(rd<const void*>(buffer, kBufData)) : 0; }

// d3dx9_38's texture code is x87 (docs/M2_RESULTS.md); exact native parity depends on the precision-control bits
// in effect inside its encoder. Measured at NiDX9SourceTextureData::LoadTextureFile entry and inside
// D3DXEncodeDXT1/3/5 (hooked by RVA from Microsoft's public PDB for this exact build).
uint16_t x87ControlWord() {
  uint16_t cw;
  __asm fnstcw cw
  return cw;
}
struct CwStats {
  std::atomic<uint64_t> byPc[4]{};  // precision control: 0 = 24-bit, 2 = 53-bit, 3 = 64-bit
  std::atomic<uint64_t> otherRounding{ 0 };
  std::atomic<uint32_t> logged{ 0 };
};
CwStats g_cwLoad, g_cwEncode;
void recordCw(CwStats& s, const char* where) {
  uint16_t cw = x87ControlWord();
  s.byPc[(cw >> 8) & 3]++;
  if (((cw >> 10) & 3) != 0) s.otherRounding++;
  if (s.logged++ < 6) logf("x87 control word at %s: %04X (thread %lu)", where, cw, GetCurrentThreadId());
}

using LoadTextureFileFn = bool(__fastcall*)(void*, void*, const char*, void*, void*, void*);
LoadTextureFileFn oLoadTextureFile;
bool __fastcall hkLoadTextureFile(void* self, void* edx, const char* name, void* renderer, void* prefs, void* file) {
  recordCw(g_cwLoad, "LoadTextureFile");
  const uint16_t cw = x87ControlWord();
  textures::beginLoad();
  bool ok = oLoadTextureFile(self, edx, name, renderer, prefs, file);
  if (g_enabled) textures::onLoadTextureFile(self, name, file, cw, ok);
  return ok;
}
using EncodeDxtFn = long(__stdcall*)(uint8_t*, const void*, int);
EncodeDxtFn oEncodeDxt1, oEncodeDxt3, oEncodeDxt5;
long __stdcall hkEncodeDxt1(uint8_t* d, const void* c, int f) { recordCw(g_cwEncode, "D3DXEncodeDXT1"); return oEncodeDxt1(d, c, f); }
long __stdcall hkEncodeDxt3(uint8_t* d, const void* c, int f) { recordCw(g_cwEncode, "D3DXEncodeDXT3"); return oEncodeDxt3(d, c, f); }
long __stdcall hkEncodeDxt5(uint8_t* d, const void* c, int f) { recordCw(g_cwEncode, "D3DXEncodeDXT5"); return oEncodeDxt5(d, c, f); }

bool hookD3dx() {
  HMODULE dx = GetModuleHandleA("d3dx9_38.dll");
  if (!dx) { logf("d3dx9_38.dll not loaded; encoder hooks skipped"); return false; }
  auto base = reinterpret_cast<const uint8_t*>(dx);
  auto nt = base + *reinterpret_cast<const uint32_t*>(base + 0x3C);
  uint32_t stamp = *reinterpret_cast<const uint32_t*>(nt + 8), image = *reinterpret_cast<const uint32_t*>(nt + 0x50);
  if (stamp != fnv::d3dx9_38_TimeDateStamp || image != fnv::d3dx9_38_SizeOfImage) {
    logf("d3dx9_38.dll is a different build (timestamp %08X size %08X); encoder hooks skipped", stamp, image);
    return false;
  }
  uintptr_t b = reinterpret_cast<uintptr_t>(dx);
  bool ok = hook(b + fnv::d3dx9_38_EncodeDXT1_RVA, &hkEncodeDxt1, &oEncodeDxt1, "d3dx9_38 D3DXEncodeDXT1");
  ok &= hook(b + fnv::d3dx9_38_EncodeDXT3_RVA, &hkEncodeDxt3, &oEncodeDxt3, "d3dx9_38 D3DXEncodeDXT3");
  ok &= hook(b + fnv::d3dx9_38_EncodeDXT5_RVA, &hkEncodeDxt5, &oEncodeDxt5, "d3dx9_38 D3DXEncodeDXT5");
  return ok;
}

}  // namespace

// --- draw-state measurement (design input for the draw path) ------------------------------------------------------
// Reads NiDX9RenderState (renderer +0x8B8) after each leaf draw. Offsets from the IDA types of NiD3DRenderState /
// NiDX9RenderState: +0x10A0 textures[16], +0x10E0 current VS handle, +0x10E8 current PS handle, +0x1100 using a
// declaration, +0x1104 FVF, +0x110C declaration handle.
namespace {
struct DrawStats {
  std::mutex m;
  uint64_t draws = 0, noVS = 0, noPS = 0, fvf = 0, decl = 0, byKind[8] = {};
  uint64_t texCount[17] = {};
  std::set<std::pair<uintptr_t, uintptr_t>> shaderPairs;
  std::set<uintptr_t> decls, fvfs;
} g_draw;

void logDrawStats() {
  std::lock_guard<std::mutex> lock(g_draw.m);
  uint64_t many = 0;
  for (int i = 8; i <= 16; i++) many += g_draw.texCount[i];
  logf("draw state: leaf draws %llu (tristrips/other %llu, TriShape2 %llu, partition %llu, mesh %llu, strip %llu) | "
       "no VS %llu, no PS %llu | declaration %llu (%zu distinct), FVF %llu (%zu distinct) | shader pairs %zu | "
       "textures bound 0:%llu 1:%llu 2:%llu 3:%llu 4:%llu 5:%llu 6:%llu 7:%llu 8+:%llu",
       g_draw.draws, g_draw.byKind[0], g_draw.byKind[1], g_draw.byKind[2], g_draw.byKind[3], g_draw.byKind[4],
       g_draw.noVS, g_draw.noPS, g_draw.decl, g_draw.decls.size(), g_draw.fvf, g_draw.fvfs.size(),
       g_draw.shaderPairs.size(), g_draw.texCount[0], g_draw.texCount[1], g_draw.texCount[2], g_draw.texCount[3],
       g_draw.texCount[4], g_draw.texCount[5], g_draw.texCount[6], g_draw.texCount[7], many);
}
}  // namespace

void onLeafDraw(void* renderer, int kind) {
  if (!g_enabled || !renderer) return;
  const char* rs = rd<const char*>(renderer, 0x8B8);
  if (!rs) return;
  uintptr_t vs = rd<uintptr_t>(rs, 0x10E0), ps = rd<uintptr_t>(rs, 0x10E8);
  bool usesDecl = rd<bool>(rs, 0x1100);
  uintptr_t fvf = rd<uint32_t>(rs, 0x1104), decl = rd<uintptr_t>(rs, 0x110C);
  int tex = 0;
  for (int i = 0; i < 16; i++) tex += rd<uintptr_t>(rs, 0x10A0 + 4 * i) ? 1 : 0;
  std::lock_guard<std::mutex> lock(g_draw.m);
  g_draw.draws++;
  g_draw.byKind[kind & 7]++;
  if (!vs) g_draw.noVS++;
  if (!ps) g_draw.noPS++;
  if (usesDecl) { g_draw.decl++; g_draw.decls.insert(decl); }
  else { g_draw.fvf++; g_draw.fvfs.insert(fvf); }
  g_draw.texCount[tex]++;
  g_draw.shaderPairs.insert({ vs, ps });
}

Backend* backendIfReady() { return g_enabled && g_be && g_be->ready() ? g_be : nullptr; }

uintptr_t defaultTargetKey() {
  const void* renderer = *reinterpret_cast<void* const*>(fnv::NiRenderer_ms_pRenderer);
  const void* group = renderer ? rd<const void*>(renderer, kRendDefaultGroup) : nullptr;
  return group ? dataKey(rd<const void*>(group, kGroupBuffers)) : 0;
}

bool __fastcall hkFastCopy(void* self, void* edx, void* src, void* dst, const int32_t* rect, int32_t x, int32_t y);
bool __fastcall hkCopy(void* self, void* edx, void* src, void* dst, const int32_t* srcRect, const int32_t* dstRect, int32_t pref);
extern bool(__fastcall* oFastCopy)(void*, void*, void*, void*, const int32_t*, int32_t, int32_t);
extern bool(__fastcall* oCopy)(void*, void*, void*, void*, const int32_t*, const int32_t*, int32_t);

bool patchGammaRamp();
bool patchScreenshotLock();
bool patchCanopyUnlock();

bool installHooks() {
  bool ok = true;
  ok &= hook(fnv::Ni2DBuffer_SetRendererData, &hkSetRendererData, &oSetRendererData, "Ni2DBuffer::SetRendererData");
  ok &= hook(fnv::NiDX92DBufferData_DeletingDtor, &hkDtor2D, &oDtor2D, "~NiDX92DBufferData (deleting)");
  ok &= hook(fnv::NiDX9TextureBufferData_DeletingDtor, &hkDtorTexture, &oDtorTexture,
             "~NiDX9TextureBufferData (deleting)");
  ok &= hook(fnv::NiDX9ImplicitBufferData_DeletingDtor, &hkDtorImplicit, &oDtorImplicit,
             "~NiDX9ImplicitBufferData (deleting)");
  ok &= hook(fnv::NiDX9AdditionalDepthStencilBufferData_DeletingDtor, &hkDtorAdditionalDS, &oDtorAdditionalDS,
             "~NiDX9AdditionalDepthStencilBufferData (deleting)");
  ok &= hook(fnv::Ni2DBuffer_ResetDimensions, &hkResetDimensions, &oResetDimensions, "Ni2DBuffer::ResetDimensions");
  ok &= hook(fnv::NiDX9ImplicitBufferData_Recreate, &hkRecreateImplicit, &oRecreateImplicit,
             "NiDX9ImplicitBufferData::Recreate");
  ok &= hook(fnv::NiDX9ImplicitDepthStencilBufferData_Recreate, &hkRecreateImplicitDS, &oRecreateImplicitDS,
             "NiDX9ImplicitDepthStencilBufferData::Recreate");
  ok &= hook(fnv::NiDX9AdditionalDepthStencilBufferData_Recreate, &hkRecreateAdditionalDS, &oRecreateAdditionalDS,
             "NiDX9AdditionalDepthStencilBufferData::Recreate");
  ok &= hook(fnv::Do_ClearBuffer, &hkClearBuffer, &oClearBuffer, "Do_ClearBuffer (slot 98)");
  ok &= hook(fnv::NiDX9SourceTextureData_LoadTextureFile, &hkLoadTextureFile, &oLoadTextureFile,
             "NiDX9SourceTextureData::LoadTextureFile");
  ok &= hookD3dx();
  ok &= textures::installHooks();
  ok &= gbshaders::installHooks();
  ok &= gbstate::installHooks();
  ok &= gbstate::patchCallSites();
  ok &= gbbuffers::installHooks();
  ok &= gbquery::installHooks();
  ok &= movie::install();
  ok &= patchCanopyUnlock();
  ok &= patchScreenshotLock();
  ok &= patchGammaRamp();
  ok &= hook(0x00E6F6E0, &hkFastCopy, &oFastCopy, "NiDX9Renderer::FastCopy (mirror)");
  ok &= hook(0x00E6F840, &hkCopy, &oCopy, "NiDX9Renderer::Copy (mirror)");
  {
    const std::string ini = iniPath();
    const bool on = GetPrivateProfileIntA("Backend", "Enable", 1, ini.c_str()) &&
                    GetPrivateProfileIntA("Backend", "Present", 0, ini.c_str());
    // Native mode: Vulkan presents and renders alone; D3D9 runs as the engine's Ex device holding placeholders.
    const bool native = on && !GetPrivateProfileIntA("Backend", "D3D9Draws", 1, ini.c_str());
    ok &= present::install(on, native);
    textures::setNative(native);
    if (native) {
      // From the start: D3D9 never receives state or draws, so it never sees a Vulkan-backed object.
      gbstate::setForwardDraws(false);
      gbbuffers::setNative(true);
      ok &= gbobjects::installBuffers();
      ok &= gbsurfaces::install();
      ok &= gbobjects::installPrograms();
      // No D3D9 at all: the renderer creates the plugin's IDirect3D9Ex / device instead of loading D3D9.DLL.
      gbstate::setNoD3D9();
      ok &= gbdevice::install();
      ok &= effects::install();
      ok &= upscale::install();
      replacements::load();
      ok &= viewvector::install(replacements::readsViewVector());
      ok &= viewvector::installLighting30(replacements::readsLighting30EyePosition());
      sunshadows::install();
      ok &= envprobe::install();
    }
  }
  return ok;
}

// Free address space and the largest free block (the game process has 2 GiB without LARGEADDRESSAWARE).
void measureAddressSpace(size_t& freeTotal, size_t& largest);
void logAddressSpace(const char* when) {
  size_t freeTotal = 0, largest = 0;
  measureAddressSpace(freeTotal, largest);
  logf("address space (%s): %zu MiB free, largest free block %zu MiB", when, freeTotal >> 20, largest >> 20);
}
void measureAddressSpace(size_t& freeTotal, size_t& largest) {
  MEMORY_BASIC_INFORMATION mbi{};
  uintptr_t a = 0x10000;
  freeTotal = largest = 0;
  while (a < 0xFFFF0000u && VirtualQuery(reinterpret_cast<void*>(a), &mbi, sizeof(mbi))) {
    if (mbi.State == MEM_FREE) {
      freeTotal += mbi.RegionSize;
      largest = std::max(largest, size_t(mbi.RegionSize));
    }
    uintptr_t next = reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
    if (next <= a) break;
    a = next;
  }
}

// Memory watch: a thread of its own measures the address space every 250 ms and logs it with what the backend holds
// (Backend::MemoryReport) whenever free space moved 128 MiB from the last line, or fell under 512 MiB, so a load that
// runs the process out of address space leaves its trail in the log even when it ends in a crash the summaries miss.
void logMemory(const char* when, size_t freeTotal, size_t largest) {
  constexpr double kMiB = 1.0 / (1 << 20);
  Backend::MemoryReport r;
  if (g_be) r = g_be->memoryReport();
  logf("memory (%s): address space %zu MiB free, largest free block %zu MiB | record queue %.1f MiB (peak %.1f, "
       "producers waited %llu, chunk allocations failed %llu) | decoded textures queued %.1f MiB (peak %.1f) | Vulkan "
       "memory held until a frame completes %.1f MiB | per-frame host rings %.1f MiB",
       when, freeTotal >> 20, largest >> 20, r.cmdQueued * kMiB, r.cmdQueuedPeak * kMiB, r.cmdQueueWaits,
       r.cmdAllocFailures, r.texQueued * kMiB, r.texQueuedPeak * kMiB, r.garbageBytes * kMiB, r.hostRingBytes * kMiB);
}
void startMemoryWatch() {
  std::thread([] {
    size_t lastFree = 0;
    for (;;) {
      Sleep(250);
      size_t freeTotal = 0, largest = 0;
      measureAddressSpace(freeTotal, largest);
      const size_t moved = freeTotal > lastFree ? freeTotal - lastFree : lastFree - freeTotal;
      if (moved >= (128u << 20) || (freeTotal < (512u << 20) && moved >= (16u << 20))) {
        logMemory("watch", freeTotal, largest);
        lastFree = freeTotal;
      }
    }
  }).detach();
}

void onRendererCreated(void* renderer) {
  installCrashLogger();
  const std::string ini = iniPath();
  g_enabled =GetPrivateProfileIntA("Backend", "Enable", 1, ini.c_str()) != 0;
  if (!g_enabled || !renderer) {
    logf("shadow backend disabled (%s [Backend] Enable=0)", ini.c_str());
    return;
  }
  BackendConfig cfg;
  cfg.shadowWindow = GetPrivateProfileIntA("Backend", "ShadowWindow", 1, ini.c_str()) != 0;
  cfg.validation = GetPrivateProfileIntA("Backend", "Validation", 1, ini.c_str()) != 0;
  // Record thread (docs/PERFORMANCE.md "Record thread"): the game's calls are recorded into Vulkan on a worker thread.
  cfg.recordThread = GetPrivateProfileIntA("Backend", "RecordThread", 1, ini.c_str()) != 0;
  // Depth pre-pass for the world's opaque pass (backend_draw.inl "Depth pre-pass").
  cfg.depthPrepass = GetPrivateProfileIntA("Backend", "DepthPrepass", 1, ini.c_str()) != 0;
  cfg.depthPrepassDiscarding = GetPrivateProfileIntA("Backend", "DepthPrepassDiscarding", 1, ini.c_str()) != 0;
  if (present::enabled() && present::window()) {
    cfg.presentWindow = present::window();
    cfg.vsync = present::vsync();
  }
  textures::configure(GetPrivateProfileIntA("Textures", "Verify", 1, ini.c_str()) != 0);
  // The Vulkan SDK ships no 32-bit layers; a 32-bit build of Vulkan-ValidationLayers (tag vulkan-sdk-1.4.335.0)
  // is found through the loader's VK_ADD_LAYER_PATH, read when the instance is created.
  char layerPath[MAX_PATH] = "";
  GetPrivateProfileStringA("Backend", "ValidationLayerPath", "", layerPath, sizeof(layerPath), ini.c_str());
  if (cfg.validation && layerPath[0]) {
    SetEnvironmentVariableA("VK_ADD_LAYER_PATH", layerPath);
    logf("validation layer path: %s", layerPath);
  }
  cfg.log = [](const char* s) { logf("[backend] %s", s); };
  cfg.shaderPack = moduleDir() + "fnv_vk_shaders.bin";
  cfg.pipelineFile = moduleDir() + "fnv_vk_pipelines.bin";
  g_compareEvery = GetPrivateProfileIntA("Verify", "CompareEvery", 0, ini.c_str());
  g_compareDumps = GetPrivateProfileIntA("Verify", "CompareDumps", 3, ini.c_str());
  const bool draws = GetPrivateProfileIntA("Backend", "Draws", 1, ini.c_str()) != 0;
  uint32_t w = 1280, h = 720;
  if (const void* group = rd<const void*>(renderer, kRendDefaultGroup))
    if (const void* buf0 = rd<const void*>(group, kGroupBuffers)) {
      w = rd<uint32_t>(buf0, kBufWidth);
      h = rd<uint32_t>(buf0, kBufHeight);
    }
  // HDR output and tonemapping need the image after the game's HDR exposure unclipped: its 8-bit targets become RGBA16F.
  display::load(present::window());
  cfg.wideTargets = effects::hasPoint("tonemap") || display::hdrWanted();
  replacements::fillConfig(cfg);
  logAddressSpace("before backend init");
  bool ok = be().init(cfg, w, h);
  if (ok) display::apply(g_be);
  if (ok) upscale::apply(g_be);
  if (ok) replacements::onBackendReady(g_be);
  logAddressSpace("after backend init");
  if (ok) startMemoryWatch();
  if (ok) gbstate::flushDeclarations();
  if (ok) gbbuffers::flushCreations();
  if (ok) gbobjects::onBackendReady(g_be);
  if (ok && present::enabled() && present::window()) overlay::init(present::window());
  BackendStats s = be().stats();
  logf("shadow backend init %s (window %ux%u, shadowWindow %d, validation requested %d); %u render targets attached "
       "during renderer creation", ok ? "OK" : "FAILED", w, h, cfg.shadowWindow, cfg.validation, s.live);
  bool d3dDraws = GetPrivateProfileIntA("Backend", "D3D9Draws", 1, ini.c_str()) != 0;
  if (!d3dDraws && !(ok && present::enabled() && present::window())) {
    // Native mode has no D3D9 to fall back to: nothing can be shown without the backend presenting.
    logf("NATIVE MODE WITHOUT A PRESENTING BACKEND (init %s, present %d, window %p): nothing will be shown",
         ok ? "ok" : "FAILED", int(present::enabled()), present::window());
  }
  // Native mode: no D3D9 observer or verification (nothing to verify against), no state sent to D3D9.
  if (ok) observe::install(rd<IDirect3DDevice9*>(renderer, kRendDevice), g_be, draws, d3dDraws);
  observe::enableProbe(d3dDraws && GetPrivateProfileIntA("Verify", "Probe", 1, ini.c_str()) != 0);
  gbstate::initFromDevice(rd<IDirect3DDevice9*>(renderer, kRendDevice));
  gbstate::setForwardDraws(d3dDraws);
  if (!d3dDraws) {
    g_compareEvery = 0;
    textures::configure(false);
    gbbuffers::setVerify(false);
    logf("native mode: the backend alone renders; D3D9 receives no draws or state; verification off");
  }
}

// NiDX9Renderer::FastCopy 0x00E6F6E0 (renderer vtable +0x114) and Copy 0x00E6F840 (+0x118): StretchRect between the
// m_pkSurface of two Ni2DBuffers. Mirrored after the original succeeds. NiRect<unsigned int> is {left, right, top,
// bottom}; FastCopy's destination rect is (x, y, x + w, y + h) with filter NONE, Copy maps COPY_FILTER_POINT (1) to
// D3DTEXF_POINT, COPY_FILTER_LINEAR (2) to D3DTEXF_LINEAR, anything else to NONE.
using FastCopyFn = bool(__fastcall*)(void*, void*, void* src, void* dst, const int32_t* rect, int32_t x, int32_t y);
using CopyFn = bool(__fastcall*)(void*, void*, void* src, void* dst, const int32_t* srcRect, const int32_t* dstRect, int32_t pref);
FastCopyFn oFastCopy = nullptr;
CopyFn oCopy = nullptr;
bool __fastcall hkFastCopy(void* self, void* edx, void* src, void* dst, const int32_t* rect, int32_t x, int32_t y) {
  const bool r = oFastCopy(self, edx, src, dst, rect, x, y);
  if (r && src && dst)
    if (Backend* be = backendIfReady()) {
      int32_t a[4], b[4];
      if (rect) {
        a[0] = rect[0]; a[1] = rect[2]; a[2] = rect[1]; a[3] = rect[3];
        b[0] = x; b[1] = y; b[2] = x + (rect[1] - rect[0]); b[3] = y + (rect[3] - rect[2]);
      }
      be->copyTarget(dataKey(src), dataKey(dst), rect ? a : nullptr, rect ? b : nullptr, 0);
    }
  return r;
}
bool __fastcall hkCopy(void* self, void* edx, void* src, void* dst, const int32_t* srcRect, const int32_t* dstRect, int32_t pref) {
  const bool r = oCopy(self, edx, src, dst, srcRect, dstRect, pref);
  if (r && src && dst)
    if (Backend* be = backendIfReady()) {
      int32_t a[4], b[4];
      if (srcRect) { a[0] = srcRect[0]; a[1] = srcRect[2]; a[2] = srcRect[1]; a[3] = srcRect[3]; }
      if (dstRect) { b[0] = dstRect[0]; b[1] = dstRect[2]; b[2] = dstRect[1]; b[3] = dstRect[3]; }
      be->copyTarget(dataKey(src), dataKey(dst), srcRect ? a : nullptr, dstRect ? b : nullptr,
                     pref == 1 ? 1u : pref == 2 ? 2u : 0u);
    }
  return r;
}

// GridCellArray::RecreateCanopyShadowMaskTexture unlocks each cell's CPU-written canopy shadow mask (64x64 L8
// NiRenderedTexture, written by TESObjectCELL::PlaceCanopyShadow*) at 0x004BB7A9 `mov eax,[ecx+50h]; call eax`
// (IDirect3DTexture9::UnlockRect) before drawing it. Patched to canopyUnlock: unlock, read the texels back
// (the texture is lockable: the game locked it) and write them into the backend's copy of that render target.
HRESULT __stdcall canopyUnlock(IDirect3DTexture9* t, UINT level) {
  const HRESULT hr = t->UnlockRect(level);
  if (Backend* be = backendIfReady())
    if (const uintptr_t key = observe::rtKeyForRtTexture(t)) {
      D3DLOCKED_RECT lr{};
      if (SUCCEEDED(t->LockRect(0, &lr, nullptr, D3DLOCK_READONLY))) {
        be->writeTarget(key, lr.pBits, uint32_t(lr.Pitch));
        t->UnlockRect(0);
      }
    }
  return hr;
}

bool patchCanopyUnlock() {
  uint8_t* p = reinterpret_cast<uint8_t*>(0x004BB7A9);
  static const uint8_t kExpect[5] = { 0x8B, 0x41, 0x50, 0xFF, 0xD0 };
  if (memcmp(p, kExpect, 5) != 0) {
    logf("canopy: unlock site 004BB7A9 does not match; canopy shadow masks are not mirrored");
    return false;
  }
  DWORD old = 0;
  if (!VirtualProtect(p, 5, PAGE_EXECUTE_READWRITE, &old)) return false;
  p[0] = 0xE8;
  const int32_t rel = int32_t(reinterpret_cast<uintptr_t>(&canopyUnlock) - (uintptr_t(p) + 5));
  memcpy(p + 1, &rel, 4);
  VirtualProtect(p, 5, old, &old);
  FlushInstructionCache(GetCurrentProcess(), p, 5);
  return true;
}

// ScreenShot::TakeScreenShot locks the D3D9 backbuffer (0x00878982 `mov edx,[eax+34h]; call edx`,
// IDirect3DSurface9::LockRect). Without D3D9 draws it is empty: the locked bits are filled from the backend's
// backbuffer (same size and 32-bit format) before the game reads them.
HRESULT __stdcall screenshotLock(IDirect3DSurface9* s, D3DLOCKED_RECT* lr, const RECT* r, DWORD flags) {
  const HRESULT hr = s->LockRect(lr, r, flags);
  if (SUCCEEDED(hr) && !r && !gbstate::forwardDraws())
    if (Backend* be = backendIfReady()) {
      D3DSURFACE_DESC d{};
      s->GetDesc(&d);
      std::vector<uint8_t> texels;
      uint32_t vf = 0, ts = 0;
      if (be->readback(defaultTargetKey(), texels, vf, ts) && ts == 4 && texels.size() >= size_t(d.Width) * d.Height * 4)
        for (uint32_t y = 0; y < d.Height; y++)
          memcpy(static_cast<uint8_t*>(lr->pBits) + size_t(y) * lr->Pitch, texels.data() + size_t(y) * d.Width * 4, size_t(d.Width) * 4);
    }
  return hr;
}

bool patchCall5(uintptr_t addr, const uint8_t expect[5], void* target, const char* what) {
  uint8_t* p = reinterpret_cast<uint8_t*>(addr);
  if (memcmp(p, expect, 5) != 0) {
    logf("%s: site %08X does not match; not patched", what, unsigned(addr));
    return false;
  }
  DWORD old = 0;
  if (!VirtualProtect(p, 5, PAGE_EXECUTE_READWRITE, &old)) return false;
  p[0] = 0xE8;
  const int32_t rel = int32_t(reinterpret_cast<uintptr_t>(target) - (addr + 5));
  memcpy(p + 1, &rel, 4);
  VirtualProtect(p, 5, old, &old);
  FlushInstructionCache(GetCurrentProcess(), p, 5);
  return true;
}
// Renderer::UpdateGamma's IDirect3DDevice9::SetGammaRamp (0x004DD119 `mov ecx,[eax+54h]; call ecx`): forwarded to
// D3D9 (it only takes effect on a fullscreen device) and given to the backend, which applies it when presenting.
HRESULT __stdcall gammaRamp(IDirect3DDevice9* d, UINT swapChain, DWORD flags, const D3DGAMMARAMP* ramp) {
  if (ramp) screenshot::setGammaRamp(ramp->red, ramp->green, ramp->blue);  // screenshots of SDR output apply it too
  if (Backend* be = backendIfReady())
    if (ramp) be->setGammaRamp(ramp->red, ramp->green, ramp->blue);
  d->SetGammaRamp(swapChain, flags, ramp);
  return D3D_OK;
}
bool patchGammaRamp() {
  static const uint8_t kExpect[5] = { 0x8B, 0x48, 0x54, 0xFF, 0xD1 };
  return patchCall5(0x004DD119, kExpect, reinterpret_cast<void*>(&gammaRamp), "SetGammaRamp");
}
bool patchScreenshotLock() {
  static const uint8_t kExpect[5] = { 0x8B, 0x50, 0x34, 0xFF, 0xD2 };
  return patchCall5(0x00878982, kExpect, reinterpret_cast<void*>(&screenshotLock), "screenshot LockRect");
}

void onBeforeDisplayFrame(void* renderer) {
  if (!g_enabled || !g_be || !g_be->ready() || !g_compareEvery) return;
  if (++g_framesSinceCompare < g_compareEvery) return;
  g_framesSinceCompare = 0;
  const void* group = rd<const void*>(renderer, kRendDefaultGroup);
  if (!group) return;
  char prefix[MAX_PATH] = "";
  if (g_comparesDone < g_compareDumps)
    snprintf(prefix, sizeof(prefix), "%sfnv_vk_compare_%u", moduleDir().c_str(), g_comparesDone);
  g_comparesDone++;
  observe::compareFrame(dataKey(rd<const void*>(group, kGroupBuffers)), prefix[0] ? prefix : nullptr);
}

void onBeginGroup(void* renderer, void* group) {
  if (!g_enabled || !g_be || !g_be->ready() || !group) return;
  if (rd<bool>(renderer, kRendDeviceLost)) { g_c.groupsDeviceLost++; return; }
  RtKey colors[4] = {};
  uint32_t n = std::min<uint32_t>(rd<uint32_t>(group, kGroupCount), 4);
  for (uint32_t i = 0; i < n; i++) colors[i] = dataKey(rd<const void*>(group, kGroupBuffers + 4 * i));
  g_be->beginGroup(colors, n, dataKey(rd<const void*>(group, kGroupDepth)));
}

void onEndGroup(void* renderer) {
  if (!g_enabled || !g_be || !g_be->ready() || rd<bool>(renderer, kRendDeviceLost)) return;
  g_be->endGroup();
}

void onDisplayFrame(void* renderer) {
  observe::onFrameStart();
  if (!g_enabled || !g_be || !g_be->ready()) return;
  const void* group = rd<const void*>(renderer, kRendDefaultGroup);
  const void* buf0 = group ? rd<const void*>(group, kGroupBuffers) : nullptr;
  if (buf0)  // a requested screenshot of the finished frame; messages of finished saves
    screenshot::frame(g_be, dataKey(buf0), rd<uint32_t>(buf0, kBufWidth), rd<uint32_t>(buf0, kBufHeight));
  overlay::frame(g_be);  // the menu (drawn over the finished frame at present)
  cputime::Scope timed(cputime::Present);
  g_be->present(group ? dataKey(rd<const void*>(group, kGroupBuffers)) : 0);
}

void logSummary(uint32_t frame) {
  if (!g_enabled || !g_be) return;
  {
    size_t freeTotal = 0, largest = 0;
    measureAddressSpace(freeTotal, largest);
    logMemory("summary", freeTotal, largest);
  }
  BackendStats s = g_be->stats();
  logf("shadow @%u: frames %llu groups %llu clears %llu (rect %llu) resolves %llu | targets attached %llu destroyed %llu "
       "resized %llu live %u realized %u | checks: attach vs D3D9 %llu checked, %llu MISMATCH; unknown class %llu; "
       "destroy without attach %llu; unknown keys %llu; sample mismatches %llu; size mismatches %llu; unsupported "
       "formats %llu; device-lost groups %llu | vk errors %llu, validation %llu, swapchains %llu",
       frame, s.frames, s.groups, s.clears, g_c.clearsWithRect.load(), s.resolves, s.attached, s.destroyed, s.resized,
       s.live, s.liveRealized, g_c.attachChecked.load(), g_c.attachDescMismatch.load(), g_c.attachUnknownClass.load(),
       g_c.destroyUnknown.load(), s.unknownKeys, s.sampleMismatches, s.sizeMismatches, s.unsupportedFormats,
       g_c.groupsDeviceLost.load(), s.vkErrors, s.validationMessages, s.swapchains);
  logf("x87 precision control: LoadTextureFile 24-bit %llu / 53-bit %llu / 64-bit %llu (non-nearest rounding %llu) | "
       "D3DX DXT encoder 24-bit %llu / 53-bit %llu / 64-bit %llu (non-nearest rounding %llu)",
       g_cwLoad.byPc[0].load(), g_cwLoad.byPc[2].load(), g_cwLoad.byPc[3].load(), g_cwLoad.otherRounding.load(),
       g_cwEncode.byPc[0].load(), g_cwEncode.byPc[2].load(), g_cwEncode.byPc[3].load(), g_cwEncode.otherRounding.load());
  textures::logSummary();
  logDrawStats();
  observe::logSummary();
  movie::logSummary();
  gbobjects::logSummary();
  gbsurfaces::logSummary();
  gbdevice::logSummary();
  effects::logSummary();
  {
    BackendStats bs = g_be->stats();
    logf("shadow @%u: copies %llu (skipped %llu), target writes %llu, groups reopened %llu, gamma presents %llu, occlusion queries %llu (unrecorded %llu)", frame, bs.copies, bs.copiesSkipped, bs.targetWrites, bs.groupsReopened, bs.gammaPresents, bs.queriesIssued, bs.queriesUnrecorded);
  }
  gbshaders::logSummary();
  gbbuffers::logSummary();
  logf("backend draws: %llu recorded | skipped %llu, no group %llu, no shader %llu, no pipeline %llu | pipelines %llu, "
       "samplers %llu | shaders bound %llu (missing from pack %llu) | buffers created %llu, writes %llu (%llu bytes, "
       "renames %llu: recycled %llu; direct %llu, pool full %llu), unknown %llu, live %u | inputs missing %llu, textures unknown %llu, feedback %llu, kind "
       "mismatch %llu, depth-texture samples %llu, table-fog draws %llu",
       s.draws, s.drawsSkipped, s.drawsNoGroup, s.drawsNoShader, s.drawsNoPipeline, s.pipelines, s.samplers,
       s.shadersCreated, s.shadersMissing, s.buffersCreated, s.bufferWrites, s.bufferBytesWritten, s.bufferRenames,
       s.bufferRecycles, s.bufferDirectWrites, s.mappedFallbacks, s.unknownBuffers, s.liveBuffers, s.missingInputs, s.unknownTextures, s.feedbackTextures, s.textureKindMismatch,
       s.depthTextureSamples, s.tableFogDraws);
  logf("draw recording reuse: pipeline from the previous draw %llu, render-state blocks %llu | descriptor sets written "
       "%llu, found written earlier in the frame %llu",
       s.pipelineReuses, s.stateBlockReuses, s.descriptorSetWrites, s.descriptorSetCacheHits);
  static uint32_t lastFrame = 0;
  static BackendStats last{};
  const uint32_t frames = frame - lastFrame;
  const std::string cpu = cputime::summary(frames);
  if (!cpu.empty()) logf("%s", cpu.c_str());
  if (!cpu.empty() && frames && s.recordBusyUs + s.recordPresentWaitUs + s.recordSyncWaitUs) {
    auto ms = [&](uint64_t now, uint64_t before) { return double(now - before) / 1000.0 / frames; };
    logf("record thread, ms per frame over %u frames: recording %.2f | the game waiting for it: at present %.2f, for "
         "results %.2f (%.1f waits per frame)", frames, ms(s.recordBusyUs, last.recordBusyUs),
         ms(s.recordPresentWaitUs, last.recordPresentWaitUs), ms(s.recordSyncWaitUs, last.recordSyncWaitUs),
         double(s.recordSyncs - last.recordSyncs) / frames);
  }
  if (frames && s.prepassDraws != last.prepassDraws)
    logf("depth pre-pass, per frame over %u frames: %.0f opaque draws held, %.0f of them drawn depth-only first (%.0f "
         "that can discard; %.0f that can discard kept out: no depth module), %.1f runs recorded", frames,
         double(s.prepassDraws - last.prepassDraws) / frames,
         double(s.prepassDepthDraws - last.prepassDepthDraws) / frames,
         double(s.prepassDiscardingDraws - last.prepassDiscardingDraws) / frames,
         double(s.prepassDiscardingRefused - last.prepassDiscardingRefused) / frames,
         double(s.prepassFlushes - last.prepassFlushes) / frames);
  last = s;
  lastFrame = frame;
}

}  // namespace fnvvk::shadow
