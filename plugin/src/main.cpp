#include <windows.h>

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <share.h>
#include <mutex>
#include <set>
#include <string>

#include "MinHook.h"
#include "addresses.h"
#include "nvse_abi.h"
#include "common.h"
#include "present.h"
#include "shadow.h"

namespace fnvvk {

std::mutex g_logMutex;
FILE* g_log = nullptr;

void logf(const char* fmt, ...) {
  std::lock_guard<std::mutex> lock(g_logMutex);
  if (!g_log) return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(g_log, fmt, ap);
  va_end(ap);
  fputc('\n', g_log);
  fflush(g_log);
}

std::string moduleDir() {
  HMODULE self = nullptr;
  GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                     reinterpret_cast<LPCSTR>(&moduleDir), &self);
  char path[MAX_PATH] = { };
  GetModuleFileNameA(self, path, MAX_PATH);
  std::string p(path);
  return p.substr(0, p.find_last_of("\\/") + 1);
}

bool hookFunction(uintptr_t addr, void* detour, void** original, const char* name) {
  MH_STATUS s = MH_CreateHook(reinterpret_cast<void*>(addr), detour, original);
  if (s == MH_OK) s = MH_EnableHook(reinterpret_cast<void*>(addr));
  logf("hook %-48s %08X: %s", name, unsigned(addr), MH_StatusToString(s));
  return s == MH_OK;
}

}  // namespace fnvvk

namespace {

using fnvvk::g_log;
using fnvvk::hook;
using fnvvk::logf;
using fnvvk::moduleDir;
using fnvvk::rd;

// Detailed events are traced for kTraceCount frames from frame 600, and again whenever fnv_vk_trace.txt appears
// next to the plugin (deleted when the trace starts), so an in-world trace can be requested from outside the game.
constexpr uint32_t kTraceCount = 3;
constexpr uint32_t kSummaryEvery = 600;

std::atomic<uint32_t> g_frame{ 0 };
std::atomic<uint32_t> g_traceStart{ 600 };
std::string g_triggerPath;
struct Counters {
  std::atomic<uint32_t> beginFrame{ 0 }, endFrame{ 0 }, beginRT{ 0 }, endRT{ 0 }, dip{ 0 }, triShape2{ 0 },
    partition{ 0 }, renderPass{ 0 }, swap{ 0 }, geometry{ 0 }, mesh{ 0 }, strip{ 0 }, particles{ 0 }, lines{ 0 },
    screenTex{ 0 }, slot107{ 0 }, slot114{ 0 };
} g_c;

bool tracing() {
  uint32_t f = g_frame.load(std::memory_order_relaxed);
  uint32_t s = g_traceStart.load(std::memory_order_relaxed);
  return f >= s && f < s + kTraceCount;
}

#define TRACE(...) do { if (tracing()) logf(__VA_ARGS__); } while (0)

// __thiscall members are detoured as __fastcall (ecx = this, edx unused).
using CreateFn = void*(__cdecl*)(unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
                                 unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned);
CreateFn oCreate;
void* __cdecl hkCreate(unsigned a1, unsigned a2, unsigned a3, unsigned a4, unsigned a5, unsigned a6, unsigned a7,
                       unsigned a8, unsigned a9, unsigned a10, unsigned a11, unsigned a12, unsigned a13,
                       unsigned a14, unsigned a15) {
  logf("NiDX9Renderer::Create(%u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u, %u)", a1, a2, a3, a4, a5, a6,
       a7, a8, a9, a10, a11, a12, a13, a14, a15);
  // Native mode forces the engine's own D3D9Ex path (last argument): the device is never lost, so DEFAULT-pool
  // resources (where the native texture placeholders live) survive Reset.
  if (fnvvk::present::forceD3D9Ex() && !(a15 & 0xFF)) {
    a15 = (a15 & ~0xFFu) | 1u;
    logf("  native mode: useD3D9ex forced on");
  }
  void* r = oCreate(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15);
  void* vtbl = r ? *reinterpret_cast<void**>(r) : nullptr;
  void* dev = r ? *reinterpret_cast<void**>(static_cast<char*>(r) + fnv::NiDX9Renderer_D3DDeviceOffset) : nullptr;
  logf("  -> renderer %p, vtable %p (expected %08X), IDirect3DDevice9* %p, ms_pRenderer %p", r, vtbl,
       unsigned(fnv::NiDX9Renderer_vtbl), dev, *reinterpret_cast<void**>(fnv::NiRenderer_ms_pRenderer));
  // Vulkan starts here rather than in NVSEPlugin_Load: xNVSE loads plugins under the loader lock, and creating a
  // Vulkan instance there loads the ICD DLLs and starts driver threads, which deadlocks.
  fnvvk::shadow::onRendererCreated(r);
  return r;
}

using BoolThisFn = bool(__fastcall*)(void*, void*);
BoolThisFn oBeginFrame, oEndFrame, oDisplayFrame, oEndRT;

bool __fastcall hkBeginFrame(void* self, void* edx) {
  g_c.beginFrame++;
  TRACE("  BeginFrame");
  return oBeginFrame(self, edx);
}
bool __fastcall hkEndFrame(void* self, void* edx) {
  g_c.endFrame++;
  TRACE("  EndFrame");
  return oEndFrame(self, edx);
}
bool __fastcall hkEndRT(void* self, void* edx) {
  g_c.endRT++;
  TRACE("  EndUsingRenderTargetGroup");
  bool r = oEndRT(self, edx);
  fnvvk::shadow::onEndGroup(self);
  return r;
}

// PC layouts from FalloutNV.exe. The Xbox-derived headers have different offsets.
//   NiRenderTargetGroup (vtable 0x011010EC, size 0x28): +0x0C Ni2DBuffer* buffers[4], +0x1C count, +0x20 depth
//     -- GetBufferCount 0x00BE0AD0, GetBuffer 0x00EE84F0, GetDepthStencilBufferRendererData 0x00EE8700
//   Ni2DBuffer: +0x08 width, +0x0C height (GetWidth 0x00EE8490 / GetHeight 0x00EE84B0), +0x10 renderer data
//   RendererData: +0x0C owned NiPixelFormat* (FastCopy 0x00E6F6E0, ~NiDX92DBufferData 0x00E7D620),
//     +0x14 IDirect3DSurface9* (SetRenderTarget 0x00E7CD50); +0x08 back pointer to the buffer is checked here
//   NiPixelFormat: +0x00 flags, +0x01 bpp, +0x04 format, +0x08 tiling, +0x0C renderer hint (IsNotEqual 0x00A765C0)

const char* dataClass(uintptr_t vt) {
  switch (vt) {
    case fnv::vtbl_NiDX92DBufferData: return "2DBufferData";
    case fnv::vtbl_NiDX9TextureBufferData: return "TextureBufferData";
    case fnv::vtbl_NiDX9ImplicitBufferData: return "ImplicitBufferData";
    case fnv::vtbl_NiDX9ImplicitDepthStencilBufferData: return "ImplicitDepthStencilBufferData";
    case fnv::vtbl_NiDX9AdditionalDepthStencilBufferData: return "AdditionalDepthStencilBufferData";
    default: return "?";
  }
}

int describeBuffer(const void* buf, char* out, size_t n) {
  // Plain C body: SEH (__try) cannot be combined with C++ objects that need unwinding.
  __try {
    uintptr_t bufVt = rd<uintptr_t>(buf, 0x00);
    uint32_t w = rd<uint32_t>(buf, 0x08), h = rd<uint32_t>(buf, 0x0C);
    const void* data = rd<const void*>(buf, 0x10);
    uintptr_t dataVt = data ? rd<uintptr_t>(data, 0x00) : 0;
    const void* back = data ? rd<const void*>(data, 0x08) : nullptr;
    const void* fmt = data ? rd<const void*>(data, 0x0C) : nullptr;
    const void* surf = data ? rd<const void*>(data, 0x14) : nullptr;
    int flags = fmt ? rd<uint8_t>(fmt, 0x00) : -1;
    int bpp = fmt ? rd<uint8_t>(fmt, 0x01) : -1;
    int fmtEnum = fmt ? rd<int>(fmt, 0x04) : -1;
    int tiling = fmt ? rd<int>(fmt, 0x08) : -1;
    unsigned hint = fmt ? rd<unsigned>(fmt, 0x0C) : 0;
    const char* kind = bufVt == fnv::vtbl_Ni2DBuffer ? "Ni2DBuffer"
                     : bufVt == fnv::vtbl_NiDepthStencilBuffer ? "NiDepthStencilBuffer" : "?";
    return _snprintf_s(out, n, _TRUNCATE,
                       "%s %p %ux%u data %s %p backptr %s surf %p fmt %d bpp %d flags 0x%x tiling %d hint %u (0x%x)",
                       kind, buf, w, h, dataClass(dataVt), data, back == buf ? "OK" : "MISMATCH", surf, fmtEnum, bpp,
                       flags, tiling, hint, hint);
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    return _snprintf_s(out, n, _TRUNCATE, "%p <unreadable>", buf);
  }
}

int describeTarget(const void* target, char* out, size_t n) {
  __try {
    uintptr_t vt = rd<uintptr_t>(target, 0x00);
    uint32_t count = rd<uint32_t>(target, 0x1C);
    const void* depth = rd<const void*>(target, 0x20);
    int len = _snprintf_s(out, n, _TRUNCATE, "group %p vtbl %s count %u", target,
                          vt == fnv::vtbl_NiRenderTargetGroup ? "OK" : "MISMATCH", count);
    for (uint32_t i = 0; i < count && i < 4 && len > 0 && size_t(len) < n; i++) {
      const void* b = rd<const void*>(target, 0x0C + 4 * i);
      len += _snprintf_s(out + len, n - len, _TRUNCATE, " | color%u ", i);
      if (len > 0 && size_t(len) < n) len += describeBuffer(b, out + len, n - len);
    }
    if (len > 0 && size_t(len) < n) {
      len += _snprintf_s(out + len, n - len, _TRUNCATE, " | depth ");
      if (depth && len > 0 && size_t(len) < n) len += describeBuffer(depth, out + len, n - len);
    }
    return len;
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    return _snprintf_s(out, n, _TRUNCATE, "group %p <unreadable>", target);
  }
}

std::mutex g_seenMutex;
std::set<const void*> g_seenTargets;

using BeginRTFn = bool(__fastcall*)(void*, void*, void*, unsigned);
BeginRTFn oBeginRT;
bool __fastcall hkBeginRT(void* self, void* edx, void* target, unsigned clearMode) {
  g_c.beginRT++;
  TRACE("  BeginUsingRenderTargetGroup(target %p, clear %u)", target, clearMode);
  bool first;
  { std::lock_guard<std::mutex> lock(g_seenMutex); first = g_seenTargets.insert(target).second; }
  if (first && target) {
    char desc[768];
    describeTarget(target, desc, sizeof(desc));
    logf("RT layout (frame %u): %s", g_frame.load(), desc);
  }
  // Before the original: it ends by calling Do_ClearBuffer, which the shadow clears into the group begun here.
  fnvvk::shadow::onBeginGroup(self, target);
  return oBeginRT(self, edx, target, clearMode);
}

using DipFn = int(__fastcall*)(void*, void*, int, int, int, int);
DipFn oDip;
int __fastcall hkDip(void* self, void* edx, int a1, int vertexCount, int a3, int primCount) {
  g_c.dip++;
  TRACE("    DIP trilist verts=%d prims=%d", vertexCount, primCount);
  return oDip(self, edx, a1, vertexCount, a3, primCount);
}

using Ptr1Fn = void(__fastcall*)(void*, void*, void*);
Ptr1Fn oTriShape2;
void __fastcall hkTriShape2(void* self, void* edx, void* src) {
  g_c.triShape2++;
  TRACE("    RenderTriShape2");
  oTriShape2(self, edx, src);
  fnvvk::shadow::onLeafDraw(self, 1);
}

using PartitionFn = void(__fastcall*)(void*, void*, void*, void*, int);
PartitionFn oPartition;
void __fastcall hkPartition(void* self, void* edx, void* stripSet, void* geometry, int unused) {
  g_c.partition++;
  TRACE("    Do_DrawPartition");
  oPartition(self, edx, stripSet, geometry, unused);
  fnvvk::shadow::onLeafDraw(self, 2);
}

using RenderPassFn = void(__cdecl*)(void*, unsigned, bool, bool, bool);
RenderPassFn oRenderPass;
void __cdecl hkRenderPass(void* pass, unsigned passId, bool alphaTest, bool alphaPass, bool geomStates) {
  g_c.renderPass++;
  TRACE("   RenderPassImmediately pass=%u alphaTest=%d alphaPass=%d", passId, alphaTest, alphaPass);
  oRenderPass(pass, passId, alphaTest, alphaPass, geomStates);
}

using VoidThisFn = void(__fastcall*)(void*, void*);
VoidThisFn oSwap;
void __fastcall hkSwap(void* self, void* edx) {
  g_c.swap++;
  oSwap(self, edx);
}

// The decomp types the argument as int, but its PDB annotation names it apFileName and the first run logged a
// pointer-like value, so it is read as a C string (guarded).
using LoadFn = bool(__fastcall*)(void*, void*, const char*);
LoadFn oShaderLoad;
bool safeCopy(const char* src, char* dst, size_t n) {
  __try {
    size_t i = 0;
    for (; i + 1 < n && src[i]; i++) dst[i] = src[i];
    dst[i] = 0;
    return true;
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    return false;
  }
}
bool __fastcall hkShaderLoad(void* self, void* edx, const char* fileName) {
  bool ok = oShaderLoad(self, edx, fileName);
  char name[MAX_PATH] = "<unreadable>";
  safeCopy(fileName, name, sizeof(name));
  logf("ShaderBuffer::Load(\"%s\") -> %d", name, ok);
  return ok;
}

// The remaining leaf draw paths, all void __thiscall(this, ptr).
#define PTR1_HOOK(name, counter, label)                                         \
  Ptr1Fn o##name;                                                               \
  void __fastcall hk##name(void* self, void* edx, void* a) {                    \
    g_c.counter++;                                                              \
    TRACE("    " label);                                                        \
    o##name(self, edx, a);                                                      \
    fnvvk::shadow::onLeafDraw(self, 0);                                         \
  }
PTR1_HOOK(DrawGeometry, geometry, "DrawGeometry")
PTR1_HOOK(Particles, particles, "RenderParticles")
PTR1_HOOK(Lines, lines, "RenderLines")
PTR1_HOOK(ScreenTex, screenTex, "RenderScreenTexture")
PTR1_HOOK(Slot107, slot107, "Render slot107 (0x00E744B0)")
PTR1_HOOK(Slot114, slot114, "Render slot114 (0x00E750E0)")

using MeshFn = void(__fastcall*)(void*, void*, void*, unsigned, unsigned, void*, void*, void*);
MeshFn oMesh;
void __fastcall hkMesh(void* self, void* edx, void* mesh, unsigned b, unsigned c, void* p, void* q, void* batch) {
  g_c.mesh++;
  TRACE("    DrawMesh");
  oMesh(self, edx, mesh, b, c, p, q, batch);
  fnvvk::shadow::onLeafDraw(self, 3);
}
using StripFn = void(__fastcall*)(void*, void*, void*, void*, void*, void*, void*);
StripFn oStrip;
void __fastcall hkStrip(void* self, void* edx, void* g, void* f, void* s, void* ctx, void* col) {
  g_c.strip++;
  TRACE("    DrawStrip");
  oStrip(self, edx, g, f, s, ctx, col);
  fnvvk::shadow::onLeafDraw(self, 4);
}

double perFrame(std::atomic<uint32_t>& c) { return c.exchange(0) / double(kSummaryEvery); }

bool __fastcall hkDisplayFrame(void* self, void* edx) {
  fnvvk::shadow::onBeforeDisplayFrame(self);
  bool r = oDisplayFrame(self, edx);
  fnvvk::shadow::onDisplayFrame(self);
  uint32_t f = g_frame.fetch_add(1) + 1;
  uint32_t s = g_traceStart.load();
  if (f > s && f <= s + kTraceCount) logf("--- end of frame %u ---", f - 1);
  if (f % 30 == 0 && !g_triggerPath.empty() && GetFileAttributesA(g_triggerPath.c_str()) != INVALID_FILE_ATTRIBUTES) {
    DeleteFileA(g_triggerPath.c_str());
    g_traceStart = f;
    logf("=== trace requested: frames %u..%u ===", f, f + kTraceCount - 1);
  }
  if (f % kSummaryEvery == 0) {
    logf("frames %u..%u per frame: BeginFrame %.1f EndFrame %.1f BeginRT %.1f EndRT %.1f RenderPass %.1f | "
         "TriShape2 %.1f Partition %.1f Geometry %.1f Mesh %.1f Strip %.1f Particles %.1f Lines %.1f ScreenTex %.1f "
         "slot107 %.1f slot114 %.1f DIP %.1f | Swap %.2f", f - kSummaryEvery, f - 1,
         perFrame(g_c.beginFrame), perFrame(g_c.endFrame), perFrame(g_c.beginRT), perFrame(g_c.endRT),
         perFrame(g_c.renderPass), perFrame(g_c.triShape2), perFrame(g_c.partition), perFrame(g_c.geometry),
         perFrame(g_c.mesh), perFrame(g_c.strip), perFrame(g_c.particles), perFrame(g_c.lines),
         perFrame(g_c.screenTex), perFrame(g_c.slot107), perFrame(g_c.slot114), perFrame(g_c.dip),
         perFrame(g_c.swap));
    fnvvk::shadow::logSummary(f);
  }
  return r;
}

bool installHooks() {
  if (MH_Initialize() != MH_OK) { logf("MH_Initialize failed"); return false; }
  bool ok = true;
  ok &= hook(fnv::NiDX9Renderer_Create, &hkCreate, &oCreate, "NiDX9Renderer::Create");
  ok &= hook(fnv::Do_BeginFrame, &hkBeginFrame, &oBeginFrame, "Do_BeginFrame (slot 95)");
  ok &= hook(fnv::Do_EndFrame, &hkEndFrame, &oEndFrame, "Do_EndFrame (slot 96)");
  ok &= hook(fnv::Do_DisplayFrame, &hkDisplayFrame, &oDisplayFrame, "Do_DisplayFrame (slot 97)");
  ok &= hook(fnv::Do_BeginUsingRenderTargetGroup, &hkBeginRT, &oBeginRT, "Do_BeginUsingRenderTargetGroup (101)");
  ok &= hook(fnv::Do_EndUsingRenderTargetGroup, &hkEndRT, &oEndRT, "Do_EndUsingRenderTargetGroup (102)");
  ok &= hook(fnv::DrawIndexedPrimitive_TriList, &hkDip, &oDip, "DrawIndexedPrimitive_TriList");
  ok &= hook(fnv::RenderTriShape2, &hkTriShape2, &oTriShape2, "RenderTriShape2");
  ok &= hook(fnv::Do_DrawPartition, &hkPartition, &oPartition, "Do_DrawPartition");
  ok &= hook(fnv::BSBatchRenderer_RenderPassImmediately, &hkRenderPass, &oRenderPass, "BSBatchRenderer::RenderPassImmediately");
  ok &= hook(fnv::Main_Swap, &hkSwap, &oSwap, "Main::Swap");
  ok &= hook(fnv::ShaderBuffer_Load, &hkShaderLoad, &oShaderLoad, "ShaderBuffer::Load");
  ok &= hook(fnv::DrawGeometry, &hkDrawGeometry, &oDrawGeometry, "DrawGeometry");
  ok &= hook(fnv::DrawMesh, &hkMesh, &oMesh, "DrawMesh");
  ok &= hook(fnv::DrawStrip, &hkStrip, &oStrip, "DrawStrip");
  ok &= hook(fnv::RenderParticles, &hkParticles, &oParticles, "RenderParticles");
  ok &= hook(fnv::RenderLines, &hkLines, &oLines, "RenderLines");
  ok &= hook(fnv::RenderScreenTexture, &hkScreenTex, &oScreenTex, "RenderScreenTexture");
  ok &= hook(fnv::RenderSlot107, &hkSlot107, &oSlot107, "Render slot 107");
  ok &= hook(fnv::RenderSlot114, &hkSlot114, &oSlot114, "Render slot 114");
  ok &= fnvvk::shadow::installHooks();
  return ok;
}

}  // namespace

extern "C" {

__declspec(dllexport) bool NVSEPlugin_Query(const nvse::Interface* nvse, nvse::PluginInfo* info) {
  info->infoVersion = nvse::PluginInfo::kInfoVersion;
  info->name = "fnv_vk";
  info->version = 1;
  // Every address is for 1.4.0.525.
  return !nvse->isEditor && nvse->runtimeVersion == fnv::kRuntimeVersion;
}

__declspec(dllexport) bool NVSEPlugin_Load(const nvse::Interface* nvse) {
  // Shared for reading so the log can be inspected while the game runs.
  g_log = _fsopen((moduleDir() + "fnv_vk.log").c_str(), "w", _SH_DENYWR);
  g_triggerPath = moduleDir() + "fnv_vk_trace.txt";
  logf("fnv_vk: nvse %08X runtime %08X", nvse->nvseVersion, nvse->runtimeVersion);
  logf("renderer at plugin load: ms_pRenderer = %p (expected null: renderer not created yet)",
       *reinterpret_cast<void**>(fnv::NiRenderer_ms_pRenderer));
  // Runs under the loader lock (see hkCreate): keep it minimal.
  if (!installHooks()) logf("WARNING: not all hooks installed");
  return true;
}

}  // extern "C"
