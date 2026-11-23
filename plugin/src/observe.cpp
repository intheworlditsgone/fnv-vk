#include "cputime.h"
#include "observe.h"

#include <windows.h>
#include <d3d9.h>

#include <array>
#include <atomic>
#include <string>
#include <intrin.h>
#include <utility>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "backend.h"
#include "common.h"
#include "gbbuffers.h"
#include "gbshaders.h"
#include "gbstate.h"
#include "shaderpack.h"
#include "translate.h"

namespace fnvvk::observe {

namespace {

Backend* g_be = nullptr;
bool g_draws = false;
std::mutex g_mutex;  // resource maps (resources are created and locked from loader threads too)
// Bumped on every change of the texture binding maps (g_texToKey, g_rtTexToRt, g_surfToRt): the per-sampler binding
// cache of drawFromTracker is valid while it is unchanged.
std::atomic<uint32_t> g_bindingGen{ 1 };

// --- the device's pipeline state, as D3D9 holds it (main thread) ------------------------------------------------
struct Mirror {
  uint32_t rs[256] = {};
  uint32_t ss[16][14] = {};
  IDirect3DBaseTexture9* tex[16] = {};
  IDirect3DVertexShader9* vs = nullptr;
  IDirect3DPixelShader9* ps = nullptr;
  IDirect3DVertexDeclaration9* decl = nullptr;
  DWORD fvf = 0;
  struct { IDirect3DVertexBuffer9* vb; UINT offset, stride; } streams[8] = {};
  IDirect3DIndexBuffer9* ib = nullptr;
  float vsc[256][4] = {};
  float psc[224][4] = {};
  uint64_t vscVersion = 1, pscVersion = 1;
  D3DVIEWPORT9 vp{};
  RECT scissor{};
  float clip[6][4] = {};
} g_m;

std::unordered_map<void*, Md5> g_devShaders;          // shader handle -> MD5 of the bytecode D3D9 received
std::unordered_map<void*, int> g_shaderChecked;        // handle -> 1 match, 2 mismatch, 3 not registered by the game side
std::unordered_map<void*, uintptr_t> g_texToKey;   // D3D texture -> TexKey
std::unordered_map<void*, uintptr_t> g_surfToRt;   // RT surface -> RtKey
std::unordered_map<void*, uintptr_t> g_rtTexToRt;  // RT texture (Gamebryo m_pkRenderTexture) -> RtKey
std::unordered_map<uintptr_t, void*> g_rtToRtTex;
std::unordered_map<uintptr_t, void*> g_rtToSurf;

struct LockInfo { void* ptr; UINT offset, size; DWORD flags; };
std::unordered_map<void*, LockInfo> g_locks;       // buffer -> its outstanding Lock

struct Counters {
  std::atomic<uint64_t> draws{ 0 }, drawsUP{ 0 }, unknownTex{ 0 }, rtTex{ 0 }, shadersNoPack{ 0 }, shadersCreated{ 0 },
    vbCreated{ 0 }, ibCreated{ 0 }, writes{ 0 }, released{ 0 }, fvfDraws{ 0 }, noDecl{ 0 }, clipPlaneSets{ 0 },
    scissorSets{ 0 }, vsTextures{ 0 }, compares{ 0 }, trackedDraws{ 0 }, unpatchedDraws{ 0 }, rtTexFallback{ 0 }, shaderMatch{ 0 }, shaderMismatch{ 0 }, declMatch{ 0 }, declMismatch{ 0 }, declUnregistered{ 0 },
    shaderUnregistered{ 0 };
} g_c;

using GetShaderSizeFn = UINT(WINAPI*)(const DWORD*);
GetShaderSizeFn g_getShaderSize = nullptr;

// --- device method detours (vtable indices of IDirect3DDevice9) -------------------------------------------------
#define DEV_FN(ret, name, ...)                       \
  using name##Fn = ret(STDMETHODCALLTYPE*)(IDirect3DDevice9*, __VA_ARGS__); \
  name##Fn o##name = nullptr;

DEV_FN(HRESULT, SetRenderState, D3DRENDERSTATETYPE, DWORD)
DEV_FN(HRESULT, SetSamplerState, DWORD, D3DSAMPLERSTATETYPE, DWORD)
DEV_FN(HRESULT, SetTexture, DWORD, IDirect3DBaseTexture9*)
DEV_FN(HRESULT, SetVertexShader, IDirect3DVertexShader9*)
DEV_FN(HRESULT, SetPixelShader, IDirect3DPixelShader9*)
DEV_FN(HRESULT, SetVertexDeclaration, IDirect3DVertexDeclaration9*)
DEV_FN(HRESULT, SetFVF, DWORD)
DEV_FN(HRESULT, SetStreamSource, UINT, IDirect3DVertexBuffer9*, UINT, UINT)
DEV_FN(HRESULT, SetIndices, IDirect3DIndexBuffer9*)
DEV_FN(HRESULT, SetViewport, const D3DVIEWPORT9*)
DEV_FN(HRESULT, SetScissorRect, const RECT*)
DEV_FN(HRESULT, SetClipPlane, DWORD, const float*)
DEV_FN(HRESULT, SetVertexShaderConstantF, UINT, const float*, UINT)
DEV_FN(HRESULT, SetPixelShaderConstantF, UINT, const float*, UINT)
DEV_FN(HRESULT, DrawPrimitive, D3DPRIMITIVETYPE, UINT, UINT)
DEV_FN(HRESULT, DrawIndexedPrimitive, D3DPRIMITIVETYPE, INT, UINT, UINT, UINT, UINT)
DEV_FN(HRESULT, DrawPrimitiveUP, D3DPRIMITIVETYPE, UINT, const void*, UINT)
DEV_FN(HRESULT, DrawIndexedPrimitiveUP, D3DPRIMITIVETYPE, UINT, UINT, UINT, const void*, D3DFORMAT, const void*, UINT)
DEV_FN(HRESULT, CreateVertexShader, const DWORD*, IDirect3DVertexShader9**)
DEV_FN(HRESULT, CreatePixelShader, const DWORD*, IDirect3DPixelShader9**)
DEV_FN(HRESULT, CreateVertexDeclaration, const D3DVERTEXELEMENT9*, IDirect3DVertexDeclaration9**)
DEV_FN(HRESULT, CreateVertexBuffer, UINT, DWORD, DWORD, D3DPOOL, IDirect3DVertexBuffer9**, HANDLE*)
DEV_FN(HRESULT, CreateTexture, UINT, UINT, UINT, DWORD, D3DFORMAT, D3DPOOL, IDirect3DTexture9**, HANDLE*)
DEV_FN(HRESULT, CreateCubeTexture, UINT, UINT, DWORD, D3DFORMAT, D3DPOOL, IDirect3DCubeTexture9**, HANDLE*)
DEV_FN(HRESULT, CreateIndexBuffer, UINT, DWORD, D3DFORMAT, D3DPOOL, IDirect3DIndexBuffer9**, HANDLE*)
#undef DEV_FN

// Resource methods (IDirect3DVertexBuffer9 / IndexBuffer9 share their method layout for these slots).
using ReleaseFn = ULONG(STDMETHODCALLTYPE*)(IUnknown*);
using LockFn = HRESULT(STDMETHODCALLTYPE*)(IUnknown*, UINT, UINT, void**, DWORD);
using UnlockFn = HRESULT(STDMETHODCALLTYPE*)(IUnknown*);
// Buffer methods. d3d9.dll has several buffer classes (by pool/usage, and per kind), some sharing method
// implementations. Every distinct implementation of Release / Lock / Unlock gets its own detour slot; a slot knows its
// original function. Buffer objects are told apart by their vtable (registered per class at install).
constexpr int kSlots = 8;
ReleaseFn g_origRelease[kSlots] = {};
LockFn g_origLock[kSlots] = {};
UnlockFn g_origUnlock[kSlots] = {};
std::unordered_map<void*, bool> g_bufVtbls;  // buffer class vtable -> is index buffer
bool isIndexBuffer(IUnknown* b) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_bufVtbls.find(*reinterpret_cast<void**>(b));
  return it != g_bufVtbls.end() && it->second;
}
bool isBuffer(void* vt) {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_bufVtbls.count(vt) != 0;
}

HRESULT STDMETHODCALLTYPE hkSetRenderState(IDirect3DDevice9* d, D3DRENDERSTATETYPE s, DWORD v) {
  if (uint32_t(s) < 256) g_m.rs[s] = v;
  return oSetRenderState(d, s, v);
}
HRESULT STDMETHODCALLTYPE hkSetSamplerState(IDirect3DDevice9* d, DWORD smp, D3DSAMPLERSTATETYPE s, DWORD v) {
  if (smp < 16 && uint32_t(s) < 14) g_m.ss[smp][s] = v;
  return oSetSamplerState(d, smp, s, v);
}
HRESULT STDMETHODCALLTYPE hkSetTexture(IDirect3DDevice9* d, DWORD s, IDirect3DBaseTexture9* t) {
  if (s < 16) g_m.tex[s] = t;
  else if (t) g_c.vsTextures++;
  return oSetTexture(d, s, t);
}
HRESULT STDMETHODCALLTYPE hkSetVertexShader(IDirect3DDevice9* d, IDirect3DVertexShader9* s) {
  g_m.vs = s;
  return oSetVertexShader(d, s);
}
HRESULT STDMETHODCALLTYPE hkSetPixelShader(IDirect3DDevice9* d, IDirect3DPixelShader9* s) {
  g_m.ps = s;
  return oSetPixelShader(d, s);
}
HRESULT STDMETHODCALLTYPE hkSetVertexDeclaration(IDirect3DDevice9* d, IDirect3DVertexDeclaration9* v) {
  g_m.decl = v;
  return oSetVertexDeclaration(d, v);
}
HRESULT STDMETHODCALLTYPE hkSetFVF(IDirect3DDevice9* d, DWORD f) {
  // D3D9: SetFVF replaces the current declaration with one derived from the FVF.
  g_m.fvf = f;
  g_m.decl = nullptr;
  return oSetFVF(d, f);
}
HRESULT STDMETHODCALLTYPE hkSetStreamSource(IDirect3DDevice9* d, UINT n, IDirect3DVertexBuffer9* vb, UINT off, UINT stride) {
  if (n < 8) g_m.streams[n] = { vb, off, stride };
  return oSetStreamSource(d, n, vb, off, stride);
}
HRESULT STDMETHODCALLTYPE hkSetIndices(IDirect3DDevice9* d, IDirect3DIndexBuffer9* ib) {
  g_m.ib = ib;
  return oSetIndices(d, ib);
}
HRESULT STDMETHODCALLTYPE hkSetViewport(IDirect3DDevice9* d, const D3DVIEWPORT9* vp) {
  if (vp) g_m.vp = *vp;
  return oSetViewport(d, vp);
}
HRESULT STDMETHODCALLTYPE hkSetScissorRect(IDirect3DDevice9* d, const RECT* r) {
  if (r) g_m.scissor = *r;
  g_c.scissorSets++;
  return oSetScissorRect(d, r);
}
HRESULT STDMETHODCALLTYPE hkSetClipPlane(IDirect3DDevice9* d, DWORD i, const float* p) {
  if (i < 6 && p) memcpy(g_m.clip[i], p, 16);
  g_c.clipPlaneSets++;
  return oSetClipPlane(d, i, p);
}
HRESULT STDMETHODCALLTYPE hkSetVertexShaderConstantF(IDirect3DDevice9* d, UINT start, const float* v, UINT n) {
  if (v && start < 256) {
    memcpy(g_m.vsc[start], v, std::min<UINT>(n, 256 - start) * 16);
    g_m.vscVersion++;
  }
  return oSetVertexShaderConstantF(d, start, v, n);
}
HRESULT STDMETHODCALLTYPE hkSetPixelShaderConstantF(IDirect3DDevice9* d, UINT start, const float* v, UINT n) {
  if (v && start < 224) {
    memcpy(g_m.psc[start], v, std::min<UINT>(n, 224 - start) * 16);
    g_m.pscVersion++;
  }
  return oSetPixelShaderConstantF(d, start, v, n);
}

// Creator of each texture: the first return addresses into FalloutNV.exe on the stack at creation (diagnostics for
// textures the backend does not have).
struct Creator { uint32_t ret[3]; };
std::unordered_map<void*, Creator> g_creators;
Creator gameCallers() {
  Creator c{};
  int n = 0;
  const uint32_t* sp = reinterpret_cast<const uint32_t*>(_AddressOfReturnAddress());
  for (int i = 0; i < 4096 && n < 3; i++) {
    uint32_t v = sp[i];
    if (v < 0x00401000 || v >= 0x01000000) continue;
    const uint8_t* code = reinterpret_cast<const uint8_t*>(uintptr_t(v));
    if (code[-5] == 0xE8 || (code[-6] == 0xFF && (code[-5] & 0x38) == 0x10) || (code[-2] == 0xFF && (code[-1] & 0x38) == 0x10) ||
        (code[-3] == 0xFF && (code[-2] & 0x38) == 0x10))
      c.ret[n++] = v;
  }
  return c;
}
HRESULT STDMETHODCALLTYPE hkCreateTexture(IDirect3DDevice9* d, UINT w, UINT h, UINT l, DWORD u, D3DFORMAT f, D3DPOOL p,
                                          IDirect3DTexture9** out, HANDLE* sh) {
  HRESULT r = oCreateTexture(d, w, h, l, u, f, p, out, sh);
  if (SUCCEEDED(r) && out && *out) {
    Creator c = gameCallers();
    std::lock_guard<std::mutex> lock(g_mutex);
    g_creators[*out] = c;
  }
  return r;
}
HRESULT STDMETHODCALLTYPE hkCreateCubeTexture(IDirect3DDevice9* d, UINT e, UINT l, DWORD u, D3DFORMAT f, D3DPOOL p,
                                              IDirect3DCubeTexture9** out, HANDLE* sh) {
  HRESULT r = oCreateCubeTexture(d, e, l, u, f, p, out, sh);
  if (SUCCEEDED(r) && out && *out) {
    Creator c = gameCallers();
    std::lock_guard<std::mutex> lock(g_mutex);
    g_creators[*out] = c;
  }
  return r;
}

// Texture binding for one sampler: a loaded texture (TexKey) or a render-target texture (its level-0 surface).
DrawSetup::Texture bindingFor(IDirect3DBaseTexture9* t) {
  DrawSetup::Texture b;
  if (!t) return b;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_texToKey.find(t);
    if (it != g_texToKey.end()) return { DrawSetup::Texture::Tex, it->second };
    auto rt = g_rtTexToRt.find(t);
    if (rt != g_rtTexToRt.end()) {
      g_c.rtTex++;
      return { DrawSetup::Texture::RenderTarget, rt->second };
    }
  }
  g_c.rtTexFallback++;  // not mapped game-side: resolved through D3D9 below (counted to show it is never needed)
  if (t->GetType() == D3DRTYPE_TEXTURE) {
    IDirect3DSurface9* s = nullptr;
    if (SUCCEEDED(static_cast<IDirect3DTexture9*>(t)->GetSurfaceLevel(0, &s)) && s) {
      std::lock_guard<std::mutex> lock(g_mutex);
      auto it = g_surfToRt.find(s);
      s->Release();
      if (it != g_surfToRt.end()) {
        g_c.rtTex++;
        return { DrawSetup::Texture::RenderTarget, it->second };
      }
    }
  }
  g_c.unknownTex++;
  {
    static std::unordered_map<void*, int> logged;
    std::lock_guard<std::mutex> lock(g_mutex);
    if (logged.size() < 60 && logged.emplace(t, 1).second) {
      D3DRESOURCETYPE type = t->GetType();
      char desc[200] = "";
      if (type == D3DRTYPE_TEXTURE) {
        D3DSURFACE_DESC d{};
        static_cast<IDirect3DTexture9*>(t)->GetLevelDesc(0, &d);
        snprintf(desc, sizeof(desc), "2D %ux%u fmt %d usage %08lX pool %d levels %lu", d.Width, d.Height, int(d.Format),
                 d.Usage, int(d.Pool), t->GetLevelCount());
      } else if (type == D3DRTYPE_CUBETEXTURE) {
        D3DSURFACE_DESC d{};
        static_cast<IDirect3DCubeTexture9*>(t)->GetLevelDesc(0, &d);
        snprintf(desc, sizeof(desc), "cube %u fmt %d usage %08lX pool %d", d.Width, int(d.Format), d.Usage, int(d.Pool));
      } else {
        snprintf(desc, sizeof(desc), "type %d", int(type));
      }
      Creator c{};
      auto ci = g_creators.find(t);
      if (ci != g_creators.end()) c = ci->second;
      logf("observer: unknown texture %p: %s | created from %08X <- %08X <- %08X", t, desc, c.ret[0], c.ret[1], c.ret[2]);
    }
  }
  return b;
}

// Verification of the game-side shader registration: the bytecode D3D9 received must be the one registered.
void checkShader(void* h) {
  if (!h) return;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_shaderChecked.count(h)) return;
  uint8_t game[16];
  auto it = g_devShaders.find(h);
  int r;
  if (!gbshaders::lookup(h, game)) r = 3;
  else r = (it != g_devShaders.end() && memcmp(game, it->second.b.data(), 16) == 0) ? 1 : 2;
  g_shaderChecked[h] = r;
  (r == 1 ? g_c.shaderMatch : r == 2 ? g_c.shaderMismatch : g_c.shaderUnregistered)++;
  if (r != 1) logf("observer: shader %p %s", h, r == 2 ? "MISMATCH: game-side bytecode differs from D3D9's" : "used but not registered by the game side");
  if (r == 3 && it != g_devShaders.end() && g_be) g_be->createShader(ShaderKey(h), it->second.b.data());  // keep drawing
}

// Verification of the game-side state tracker (gbstate.cpp) against what D3D9 holds at this draw.
struct TrackerDiffs {
  uint64_t draws = 0, rs = 0, ss = 0, tex = 0, shaders = 0, decl = 0, vsc = 0, psc = 0, other = 0;
  uint32_t logged = 0;
} g_td;
void verifyTracker() {
  const gbstate::State& t = gbstate::state();
  g_td.draws++;
  auto note = [&](uint64_t& counter, const char* what, uint32_t a, uint32_t b, uint32_t c) {
    counter++;
    if (g_td.logged < 40) {
      g_td.logged++;
      logf("tracker DIFF %s [%u]: tracker %08X, D3D9 %08X", what, a, b, c);
    }
  };
  for (uint32_t i = 0; i < 256; i++)
    if (t.rs[i] != g_m.rs[i]) { note(g_td.rs, "render state", i, t.rs[i], g_m.rs[i]); break; }
  for (uint32_t s = 0; s < 16; s++)
    for (uint32_t i = 1; i < 14; i++)
      if (t.ss[s][i] != g_m.ss[s][i]) { note(g_td.ss, "sampler state (stage*16+type)", s * 16 + i, t.ss[s][i], g_m.ss[s][i]); s = 16; break; }
  for (uint32_t i = 0; i < 16; i++)
    if (t.tex[i] != g_m.tex[i]) { note(g_td.tex, "texture", i, uint32_t(uintptr_t(t.tex[i])), uint32_t(uintptr_t(g_m.tex[i]))); break; }
  if (t.vs != g_m.vs || t.ps != g_m.ps) note(g_td.shaders, "shaders (vs)", 0, uint32_t(uintptr_t(t.vs)), uint32_t(uintptr_t(g_m.vs)));
  if (t.decl != g_m.decl) note(g_td.decl, "declaration", 0, uint32_t(uintptr_t(t.decl)), uint32_t(uintptr_t(g_m.decl)));
  if (t.viewport.x != g_m.vp.X || t.viewport.y != g_m.vp.Y || t.viewport.w != g_m.vp.Width || t.viewport.h != g_m.vp.Height ||
      t.viewport.minZ != g_m.vp.MinZ || t.viewport.maxZ != g_m.vp.MaxZ)
    note(g_td.other, "viewport (width)", 0, t.viewport.w, g_m.vp.Width);
  if (g_m.rs[D3DRS_SCISSORTESTENABLE] && (t.scissor[0] != g_m.scissor.left || t.scissor[1] != g_m.scissor.top ||
                                          t.scissor[2] != g_m.scissor.right || t.scissor[3] != g_m.scissor.bottom))
    note(g_td.other, "scissor (right)", 0, uint32_t(t.scissor[2]), uint32_t(g_m.scissor.right));
  for (uint32_t i = 0; i < 6; i++)
    if ((g_m.rs[D3DRS_CLIPPLANEENABLE] >> i & 1) && memcmp(t.clip[i], g_m.clip[i], 16)) { note(g_td.other, "clip plane", i, 0, 0); break; }
  for (uint32_t i = 0; i < 8; i++)
    if (t.streams[i].vb != g_m.streams[i].vb ||
        (g_m.streams[i].vb && (t.streams[i].offset != g_m.streams[i].offset || t.streams[i].stride != g_m.streams[i].stride))) {
      note(g_td.other, "stream", i, uint32_t(uintptr_t(t.streams[i].vb)), uint32_t(uintptr_t(g_m.streams[i].vb)));
      break;
    }
  if (t.ib != g_m.ib) note(g_td.other, "indices", 0, uint32_t(uintptr_t(t.ib)), uint32_t(uintptr_t(g_m.ib)));
  for (uint32_t r = 0; r < 256; r++)
    if (memcmp(t.vsc[r], g_m.vsc[r], 16)) { note(g_td.vsc, "VS constant register", r, 0, 0); break; }
  for (uint32_t r = 0; r < 224; r++)
    if (memcmp(t.psc[r], g_m.psc[r], 16)) { note(g_td.psc, "PS constant register", r, 0, 0); break; }
}

bool g_fromTracker = true;
void issue(uint32_t pt, uint32_t prims, bool indexed, int32_t baseVertex, uint32_t start, const DrawSetup::Stream* override0,
           BufKey indexOverride) {
  if (!g_draws || !g_be) return;
  checkShader(g_m.vs);
  checkShader(g_m.ps);
  verifyTracker();
  // Draws are issued from the game-side tracker by the draw-site thunks (gbstate.cpp); here only verification.
  // A draw that did not come through a thunk is from a site not yet patched: drawn from here, counted, reported.
  if (g_fromTracker && gbstate::inDrawThunk()) return;
  if (g_fromTracker) {
    g_c.unpatchedDraws++;
    void* caller = nullptr;
    const uint32_t* sp = reinterpret_cast<const uint32_t*>(_AddressOfReturnAddress());
    for (int i = 0; i < 64 && !caller; i++)
      if (sp[i] >= 0x00401000 && sp[i] < 0x01000000) caller = reinterpret_cast<void*>(uintptr_t(sp[i]));
    static std::unordered_map<void*, int> seen;
    if (seen.size() < 32 && seen.emplace(caller, 1).second) logf("observer: draw from an unpatched site (caller %p)", caller);
  }
  DrawSetup s;
  s.renderStates = g_m.rs;
  s.samplerStates = g_m.ss;
  for (int i = 0; i < 16; i++) s.textures[i] = bindingFor(g_m.tex[i]);
  s.vs = ShaderKey(g_m.vs);
  s.ps = ShaderKey(g_m.ps);
  s.decl = DeclKey(g_m.decl);
  if (!g_m.decl) g_c.fvfDraws++;
  for (int i = 0; i < 8; i++) s.streams[i] = { BufKey(g_m.streams[i].vb), g_m.streams[i].offset, g_m.streams[i].stride };
  if (override0) s.streams[0] = *override0;
  s.indices = indexOverride ? indexOverride : BufKey(g_m.ib);
  s.vsConstants = &g_m.vsc[0][0];
  s.psConstants = &g_m.psc[0][0];
  s.vsConstantsVersion = g_m.vscVersion;
  s.psConstantsVersion = g_m.pscVersion;
  s.viewport = { int32_t(g_m.vp.X), int32_t(g_m.vp.Y), g_m.vp.Width, g_m.vp.Height, g_m.vp.MinZ, g_m.vp.MaxZ };
  s.scissor[0] = g_m.scissor.left;
  s.scissor[1] = g_m.scissor.top;
  s.scissor[2] = g_m.scissor.right;
  s.scissor[3] = g_m.scissor.bottom;
  memcpy(s.clipPlanes, g_m.clip, sizeof(s.clipPlanes));
  g_be->draw(s, pt, prims, indexed, baseVertex, start);
  g_c.draws++;
}

HRESULT STDMETHODCALLTYPE hkDrawPrimitive(IDirect3DDevice9* d, D3DPRIMITIVETYPE pt, UINT start, UINT prims) {
  HRESULT r = oDrawPrimitive(d, pt, start, prims);
  issue(pt, prims, false, 0, start, nullptr, 0);
  return r;
}
HRESULT STDMETHODCALLTYPE hkDrawIndexedPrimitive(IDirect3DDevice9* d, D3DPRIMITIVETYPE pt, INT base, UINT minIdx, UINT nv,
                                                 UINT startIdx, UINT prims) {
  HRESULT r = oDrawIndexedPrimitive(d, pt, base, minIdx, nv, startIdx, prims);
  issue(pt, prims, true, base, startIdx, nullptr, 0);
  return r;
}

// User-pointer draws: the data goes through scratch buffers (written with DISCARD, as D3D9 itself streams it).
constexpr BufKey kScratchVb = 0xF0000001, kScratchIb = 0xF0000002;
uint32_t g_scratchVbSize = 0, g_scratchIbSize = 0;
void scratch(BufKey key, uint32_t& cap, const void* data, uint32_t size, bool index, bool index32) {
  if (size > cap) {
    cap = std::max<uint32_t>(size, 1u << 20);
    g_be->createBuffer(key, cap, index, index32);
  }
  g_be->writeBuffer(key, 0, data, size, kLockDiscard);
}
HRESULT STDMETHODCALLTYPE hkDrawPrimitiveUP(IDirect3DDevice9* d, D3DPRIMITIVETYPE pt, UINT prims, const void* data, UINT stride) {
  HRESULT r = oDrawPrimitiveUP(d, pt, prims, data, stride);
  // D3D9 resets stream 0 and the index buffer after a UP draw.
  g_m.streams[0] = {};
  g_m.ib = nullptr;
  g_c.drawsUP++;
  if (g_draws && g_be && data) {
    uint32_t bytes = uint32_t(stride) * xlat::vertexCount(pt, prims);
    scratch(kScratchVb, g_scratchVbSize, data, bytes, false, false);
    DrawSetup::Stream s0{ kScratchVb, 0, stride };
    issue(pt, prims, false, 0, 0, &s0, 0);
  }
  return r;
}
HRESULT STDMETHODCALLTYPE hkDrawIndexedPrimitiveUP(IDirect3DDevice9* d, D3DPRIMITIVETYPE pt, UINT minIdx, UINT nv, UINT prims,
                                                   const void* idx, D3DFORMAT fmt, const void* data, UINT stride) {
  HRESULT r = oDrawIndexedPrimitiveUP(d, pt, minIdx, nv, prims, idx, fmt, data, stride);
  g_m.streams[0] = {};
  g_m.ib = nullptr;
  g_c.drawsUP++;
  if (g_draws && g_be && data && idx) {
    bool i32 = fmt == D3DFMT_INDEX32;
    uint32_t icount = xlat::vertexCount(pt, prims);
    scratch(kScratchVb, g_scratchVbSize, data, uint32_t(stride) * (minIdx + nv), false, false);
    scratch(i32 ? kScratchIb + 1 : kScratchIb, g_scratchIbSize, idx, icount * (i32 ? 4 : 2), true, i32);
    DrawSetup::Stream s0{ kScratchVb, 0, stride };
    issue(pt, prims, true, 0, 0, &s0, i32 ? kScratchIb + 1 : kScratchIb);
  }
  return r;
}

// --- resources --------------------------------------------------------------------------------------------------
HRESULT STDMETHODCALLTYPE hkCreateVertexShader(IDirect3DDevice9* d, const DWORD* code, IDirect3DVertexShader9** out) {
  HRESULT r = oCreateVertexShader(d, code, out);
  if (SUCCEEDED(r) && out && *out && code && g_getShaderSize) {
    // Shaders are registered from the game side (gbshaders.cpp); here only the bytecode D3D9 received is recorded,
    // and checked against that registration at the shader's first draw.
    Md5 m;
    if (md5Of(code, g_getShaderSize(code), m)) {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_devShaders[*out] = m;
      g_c.shadersCreated++;
    }
  }
  return r;
}
HRESULT STDMETHODCALLTYPE hkCreatePixelShader(IDirect3DDevice9* d, const DWORD* code, IDirect3DPixelShader9** out) {
  HRESULT r = oCreatePixelShader(d, code, out);
  if (SUCCEEDED(r) && out && *out && code && g_getShaderSize) {
    // Shaders are registered from the game side (gbshaders.cpp); here only the bytecode D3D9 received is recorded,
    // and checked against that registration at the shader's first draw.
    Md5 m;
    if (md5Of(code, g_getShaderSize(code), m)) {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_devShaders[*out] = m;
      g_c.shadersCreated++;
    }
  }
  return r;
}
HRESULT STDMETHODCALLTYPE hkCreateVertexDeclaration(IDirect3DDevice9* d, const D3DVERTEXELEMENT9* e, IDirect3DVertexDeclaration9** out) {
  HRESULT r = oCreateVertexDeclaration(d, e, out);
  if (SUCCEEDED(r) && out && *out && e) {
    // Registered from the game side (gbstate.cpp); here the elements D3D9 received are checked against it.
    const D3DVERTEXELEMENT9* p = e;
    while (p->Stream != 0xFF && p - e < 64) p++;
    std::vector<uint8_t> game;
    if (gbstate::inDeclThunk()) {
      g_c.declMatch++;  // created through the game-side thunk: the same element array D3D9 receives
    } else if (!gbstate::declarationElements(*out, game)) {
      // Created before the game-side sites were patched (renderer creation): register from here, and count it.
      g_c.declUnregistered++;
      std::vector<VertexElement> v;
      for (const D3DVERTEXELEMENT9* q = e; q != p; q++) v.push_back({ q->Stream, q->Offset, q->Type, q->Method, q->Usage, q->UsageIndex });
      g_be->createDecl(DeclKey(*out), v.data(), uint32_t(v.size()));
    }
    else if (game.size() == size_t(p - e) * sizeof(D3DVERTEXELEMENT9) && memcmp(game.data(), e, game.size()) == 0) g_c.declMatch++;
    else g_c.declMismatch++;
  }
  return r;
}

template <int I>
ULONG STDMETHODCALLTYPE hkRelease(IUnknown* b) {
  void* vt = *reinterpret_cast<void**>(b);  // read before the object can be freed
  ULONG n = g_origRelease[I](b);
  if (n == 0 && isBuffer(vt)) {
    g_be->destroyBuffer(BufKey(b));
    g_c.released++;
  }
  return n;
}
HRESULT lockCommon(LockFn orig, IUnknown* b, UINT off, UINT size, void** p, DWORD flags) {
  HRESULT r = orig(b, off, size, p, flags);
  if (SUCCEEDED(r) && p && *p && !(flags & D3DLOCK_READONLY)) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_locks[b] = { *p, off, size, flags };
  }
  return r;
}
HRESULT unlockCommon(UnlockFn orig, IUnknown* b, bool index) {
  LockInfo li{};
  bool have = false;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_locks.find(b);
    if (it != g_locks.end()) {
      li = it->second;
      have = true;
      g_locks.erase(it);
    }
  }
  if (have) {
    UINT size = li.size;
    if (!size) {
      // Size 0 locks the whole buffer from `offset`.
      if (index) {
        D3DINDEXBUFFER_DESC desc{};
        static_cast<IDirect3DIndexBuffer9*>(static_cast<IUnknown*>(b))->GetDesc(&desc);
        size = desc.Size - li.offset;
      } else {
        D3DVERTEXBUFFER_DESC desc{};
        static_cast<IDirect3DVertexBuffer9*>(static_cast<IUnknown*>(b))->GetDesc(&desc);
        size = desc.Size - li.offset;
      }
    }
    // The locked pointer is valid until Unlock: copy before calling it.
    g_be->writeBuffer(BufKey(b), li.offset, li.ptr, size, li.flags);
    gbbuffers::verifyWrite(b, li.offset, li.ptr, size);
    g_c.writes++;
  }
  return orig(b);
}
template <int I>
HRESULT STDMETHODCALLTYPE hkLock(IUnknown* b, UINT off, UINT size, void** p, DWORD flags) {
  return lockCommon(g_origLock[I], b, off, size, p, flags);
}
template <int I>
HRESULT STDMETHODCALLTYPE hkUnlock(IUnknown* b) {
  return unlockCommon(g_origUnlock[I], b, isIndexBuffer(b));
}
template <int... I>
constexpr std::array<void*, sizeof...(I)> releaseDetours(std::integer_sequence<int, I...>) { return { reinterpret_cast<void*>(&hkRelease<I>)... }; }
template <int... I>
constexpr std::array<void*, sizeof...(I)> lockDetours(std::integer_sequence<int, I...>) { return { reinterpret_cast<void*>(&hkLock<I>)... }; }
template <int... I>
constexpr std::array<void*, sizeof...(I)> unlockDetours(std::integer_sequence<int, I...>) { return { reinterpret_cast<void*>(&hkUnlock<I>)... }; }

// Hooks the Release (slot 2), Lock (11) and Unlock (12) implementations of one buffer class, each distinct
// implementation once. Called at install for every pool/usage class (main thread, renderer creation).
std::vector<void*> g_hookedRelease, g_hookedLock, g_hookedUnlock;
void hookBufferClass(IUnknown* obj, bool index) {
  void** vt = *reinterpret_cast<void***>(obj);
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_bufVtbls.emplace(vt, index).second) return;
  }
  static const auto rel = releaseDetours(std::make_integer_sequence<int, kSlots>());
  static const auto lck = lockDetours(std::make_integer_sequence<int, kSlots>());
  static const auto unl = unlockDetours(std::make_integer_sequence<int, kSlots>());
  auto add = [](std::vector<void*>& hooked, void* fn, const auto& detours, auto* origs, const char* what) {
    for (void* h : hooked)
      if (h == fn) return;
    if (hooked.size() >= size_t(kSlots)) {
      logf("observer: more than %d %s implementations; one is not observed", kSlots, what);
      return;
    }
    size_t slot = hooked.size();
    hooked.push_back(fn);
    char name[96];
    snprintf(name, sizeof(name), "IDirect3D*Buffer9::%s impl %zu (observe)", what, slot);
    hookFunction(reinterpret_cast<uintptr_t>(fn), detours[slot], reinterpret_cast<void**>(&origs[slot]), name);
  };
  add(g_hookedRelease, vt[2], rel, g_origRelease, "Release");
  add(g_hookedLock, vt[11], lck, g_origLock, "Lock");
  add(g_hookedUnlock, vt[12], unl, g_origUnlock, "Unlock");
}

HRESULT STDMETHODCALLTYPE hkCreateVertexBuffer(IDirect3DDevice9* d, UINT len, DWORD usage, DWORD fvf, D3DPOOL pool,
                                               IDirect3DVertexBuffer9** out, HANDLE* sh) {
  HRESULT r = oCreateVertexBuffer(d, len, usage, fvf, pool, out, sh);
  if (SUCCEEDED(r) && out && *out) {
    if (!isBuffer(*reinterpret_cast<void**>(*out)))
      logf("observer: vertex buffer class %p (usage %08lX pool %d) was not hooked at install: not observed",
           *reinterpret_cast<void**>(*out), usage, int(pool));
    if (!gbbuffers::inCreator()) {
      g_be->createBuffer(BufKey(*out), len, false);  // not from a Gamebryo creator: registered here, counted
      g_c.vbCreated++;
    }
  }
  return r;
}
HRESULT STDMETHODCALLTYPE hkCreateIndexBuffer(IDirect3DDevice9* d, UINT len, DWORD usage, D3DFORMAT fmt, D3DPOOL pool,
                                              IDirect3DIndexBuffer9** out, HANDLE* sh) {
  HRESULT r = oCreateIndexBuffer(d, len, usage, fmt, pool, out, sh);
  if (SUCCEEDED(r) && out && *out) {
    if (!isBuffer(*reinterpret_cast<void**>(*out)))
      logf("observer: index buffer class %p (usage %08lX pool %d) was not hooked at install: not observed",
           *reinterpret_cast<void**>(*out), usage, int(pool));
    if (!gbbuffers::inCreator()) {
      g_be->createBuffer(BufKey(*out), len, true, fmt == D3DFMT_INDEX32);
      g_c.ibCreated++;
    }
  }
  return r;
}

// --- frame comparison -------------------------------------------------------------------------------------------
IDirect3DDevice9* g_dev = nullptr;
bool g_probeEnabled = false;

void writeBmp(const std::string& path, const std::vector<uint8_t>& bgra, uint32_t w, uint32_t h) {
  FILE* f = nullptr;
  if (fopen_s(&f, path.c_str(), "wb") || !f) return;
  uint32_t row = (w * 3 + 3) & ~3u, size = 54 + row * h;
  uint8_t hdr[54] = { 'B', 'M' };
  memcpy(hdr + 2, &size, 4);
  uint32_t off = 54, dib = 40, planesBpp = 1 | (24 << 16), imgSize = row * h;
  int32_t sw = int32_t(w), shNeg = -int32_t(h);
  memcpy(hdr + 10, &off, 4);
  memcpy(hdr + 14, &dib, 4);
  memcpy(hdr + 18, &sw, 4);
  memcpy(hdr + 22, &shNeg, 4);
  memcpy(hdr + 26, &planesBpp, 4);
  memcpy(hdr + 34, &imgSize, 4);
  fwrite(hdr, 1, 54, f);
  std::vector<uint8_t> line(row, 0);
  for (uint32_t y = 0; y < h; y++) {
    for (uint32_t x = 0; x < w; x++) memcpy(&line[x * 3], &bgra[(size_t(y) * w + x) * 4], 3);
    fwrite(line.data(), 1, row, f);
  }
  fclose(f);
}

}  // namespace

bool install(IDirect3DDevice9* dev, Backend* backend, bool drawsEnabled, bool installHooks) {
  static bool done = false;
  if (done || !dev) return done;
  done = true;
  g_be = backend;
  g_draws = drawsEnabled;
  g_dev = dev;
  if (HMODULE dx = GetModuleHandleA("d3dx9_38.dll"))
    g_getShaderSize = reinterpret_cast<GetShaderSizeFn>(GetProcAddress(dx, "D3DXGetShaderSize"));
  {
    D3DDEVICE_CREATION_PARAMETERS cp{};
    dev->GetCreationParameters(&cp);
    D3DPRESENT_PARAMETERS pp{};
    IDirect3DSwapChain9* sc = nullptr;
    if (SUCCEEDED(dev->GetSwapChain(0, &sc)) && sc) { sc->GetPresentParameters(&pp); sc->Release(); }
    logf("D3D9 device: adapter %u type %d behavior %08lX | present %ux%u fmt %d count %u ms %d/%lu swap %d windowed %d "
         "autoDS %d dsfmt %d flags %08lX refresh %u interval %08X",
         cp.AdapterOrdinal, int(cp.DeviceType), cp.BehaviorFlags, pp.BackBufferWidth, pp.BackBufferHeight,
         int(pp.BackBufferFormat), pp.BackBufferCount, int(pp.MultiSampleType), pp.MultiSampleQuality, int(pp.SwapEffect),
         pp.Windowed, pp.EnableAutoDepthStencil, int(pp.AutoDepthStencilFormat), pp.Flags, pp.FullScreen_RefreshRateInHz,
         pp.PresentationInterval);
  }
  // Native mode: no device hooks or state mirror; the buffer-class hooks below stay (they carry the game's buffer
  // writes to the backend until buffers are Vulkan-backed objects).
  if (installHooks) {
  // Start from the device's current state (the game set it up before this point).
  for (uint32_t i = 0; i < 256; i++) {
    DWORD v = 0;
    if (SUCCEEDED(dev->GetRenderState(D3DRENDERSTATETYPE(i), &v))) g_m.rs[i] = v;
  }
  for (uint32_t s = 0; s < 16; s++)
    for (uint32_t i = 1; i < 14; i++) {
      DWORD v = 0;
      if (SUCCEEDED(dev->GetSamplerState(s, D3DSAMPLERSTATETYPE(i), &v))) g_m.ss[s][i] = v;
    }
  dev->GetViewport(&g_m.vp);
  dev->GetScissorRect(&g_m.scissor);
  for (uint32_t i = 0; i < 6; i++) dev->GetClipPlane(i, g_m.clip[i]);
  dev->GetVertexShaderConstantF(0, &g_m.vsc[0][0], 256);
  dev->GetPixelShaderConstantF(0, &g_m.psc[0][0], 224);

  struct H { int index; void* detour; void** orig; const char* name; };
#define HK(i, n) { i, reinterpret_cast<void*>(&hk##n), reinterpret_cast<void**>(&o##n), "IDirect3DDevice9::" #n " (observe)" }
  const H hooks[] = { HK(57, SetRenderState), HK(69, SetSamplerState), HK(65, SetTexture), HK(92, SetVertexShader),
                      HK(107, SetPixelShader), HK(87, SetVertexDeclaration), HK(89, SetFVF), HK(100, SetStreamSource),
                      HK(104, SetIndices), HK(47, SetViewport), HK(75, SetScissorRect), HK(55, SetClipPlane),
                      HK(94, SetVertexShaderConstantF), HK(109, SetPixelShaderConstantF), HK(81, DrawPrimitive),
                      HK(82, DrawIndexedPrimitive), HK(83, DrawPrimitiveUP), HK(84, DrawIndexedPrimitiveUP),
                      HK(91, CreateVertexShader), HK(106, CreatePixelShader), HK(86, CreateVertexDeclaration),
                      HK(26, CreateVertexBuffer), HK(27, CreateIndexBuffer),
                      HK(23, CreateTexture), HK(25, CreateCubeTexture) };
#undef HK
  bool ok = true;
  for (const H& h : hooks) {
    void** vt = *reinterpret_cast<void***>(dev);
    ok &= hookFunction(reinterpret_cast<uintptr_t>(vt[h.index]), h.detour, h.orig, h.name);
  }
  } else {
    logf("D3D9 observer: device hooks not installed (native mode); buffer writes still observed");
  }
  bool ok = true;
  // Buffer method hooks from throwaway buffers of every pool/usage class (D3D9 only: in native mode buffers are the
  // plugin's objects, which upload on Unlock themselves), here on the main thread during renderer
  // creation (not lazily from whichever thread creates a buffer: MinHook suspends the other threads while patching).
  if (installHooks) {
    const struct { D3DPOOL pool; DWORD usage; } classes[] = {
      { D3DPOOL_DEFAULT, 0 }, { D3DPOOL_DEFAULT, D3DUSAGE_WRITEONLY }, { D3DPOOL_DEFAULT, D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY },
      { D3DPOOL_DEFAULT, D3DUSAGE_DYNAMIC }, { D3DPOOL_MANAGED, 0 }, { D3DPOOL_MANAGED, D3DUSAGE_WRITEONLY },
      { D3DPOOL_SYSTEMMEM, 0 }, { D3DPOOL_SYSTEMMEM, D3DUSAGE_WRITEONLY }, { D3DPOOL_SYSTEMMEM, D3DUSAGE_DYNAMIC },
      { D3DPOOL_DEFAULT, D3DUSAGE_WRITEONLY | D3DUSAGE_SOFTWAREPROCESSING },
      { D3DPOOL_DEFAULT, D3DUSAGE_DYNAMIC | D3DUSAGE_WRITEONLY | D3DUSAGE_SOFTWAREPROCESSING },
    };
    int made = 0;
    for (const auto& c : classes) {
      IDirect3DVertexBuffer9* vb = nullptr;
      IDirect3DIndexBuffer9* ib = nullptr;
      if (SUCCEEDED(oCreateVertexBuffer ? oCreateVertexBuffer(dev, 64, c.usage, 0, c.pool, &vb, nullptr)
                                        : dev->CreateVertexBuffer(64, c.usage, 0, c.pool, &vb, nullptr)) && vb) {
        hookBufferClass(vb, false);
        vb->Release();
        made++;
      }
      if (SUCCEEDED(oCreateIndexBuffer ? oCreateIndexBuffer(dev, 64, c.usage, D3DFMT_INDEX16, c.pool, &ib, nullptr)
                                       : dev->CreateIndexBuffer(64, c.usage, D3DFMT_INDEX16, c.pool, &ib, nullptr)) && ib) {
        hookBufferClass(ib, true);
        ib->Release();
        made++;
      }
    }
    logf("observer: %d probe buffers, %zu buffer classes, Release x%zu Lock x%zu Unlock x%zu implementations hooked", made,
         g_bufVtbls.size(), g_hookedRelease.size(), g_hookedLock.size(), g_hookedUnlock.size());
  }
  logf("D3D9 observer installed (%s); draws to the backend %s", ok ? "all hooks" : "SOME HOOKS FAILED",
       drawsEnabled ? "enabled" : "disabled");
  return ok;
}

// --- pixel probe (diagnostics) ------------------------------------------------------------------------------------
// After a frame comparison with a large difference, the next frame is probed at the worst pixel: after every draw the
// texel at that position of colour target 0 is read from D3D9 (1x1 StretchRect, which also resolves MSAA, then
// GetRenderTargetData) and from the backend, and every draw that changes the D3D9 texel or leaves the two different
// is logged with its shaders. Slow (two GPU round trips per draw) and limited to one frame.
struct Probe {
  bool armed = false, active = false;
  uint32_t x = 0, y = 0, draw = 0, logged = 0;
  std::vector<uint8_t> lastD3d;
  IDirect3DSurface9* rt1 = nullptr;  // 1x1 render target of the current format
  IDirect3DSurface9* sys1 = nullptr;
  D3DFORMAT fmt = D3DFMT_UNKNOWN;
} g_probe;

bool d3dTexel(uint32_t x, uint32_t y, std::vector<uint8_t>& out, D3DFORMAT& fmt) {
  IDirect3DSurface9* cur = nullptr;
  if (FAILED(g_dev->GetRenderTarget(0, &cur)) || !cur) return false;
  D3DSURFACE_DESC d{};
  cur->GetDesc(&d);
  bool ok = x < d.Width && y < d.Height;
  if (ok && d.Format != g_probe.fmt) {
    if (g_probe.rt1) g_probe.rt1->Release();
    if (g_probe.sys1) g_probe.sys1->Release();
    g_probe.rt1 = g_probe.sys1 = nullptr;
    g_probe.fmt = d.Format;
    ok = SUCCEEDED(g_dev->CreateRenderTarget(1, 1, d.Format, D3DMULTISAMPLE_NONE, 0, FALSE, &g_probe.rt1, nullptr)) &&
         SUCCEEDED(g_dev->CreateOffscreenPlainSurface(1, 1, d.Format, D3DPOOL_SYSTEMMEM, &g_probe.sys1, nullptr));
  }
  RECT src{ LONG(x), LONG(y), LONG(x + 1), LONG(y + 1) };
  ok = ok && g_probe.rt1 && SUCCEEDED(g_dev->StretchRect(cur, &src, g_probe.rt1, nullptr, D3DTEXF_NONE)) &&
       SUCCEEDED(g_dev->GetRenderTargetData(g_probe.rt1, g_probe.sys1));
  D3DLOCKED_RECT lr{};
  if (ok && SUCCEEDED(g_probe.sys1->LockRect(&lr, nullptr, D3DLOCK_READONLY))) {
    uint32_t bytes = d.Format == D3DFMT_A16B16G16R16F ? 8 : 4;
    out.assign(static_cast<uint8_t*>(lr.pBits), static_cast<uint8_t*>(lr.pBits) + bytes);
    g_probe.sys1->UnlockRect();
  } else {
    ok = false;
  }
  fmt = d.Format;
  cur->Release();
  return ok;
}

std::string hexBytes(const std::vector<uint8_t>& v) {
  std::string s;
  char b[4];
  for (uint8_t c : v) { snprintf(b, sizeof(b), "%02X", c); s += b; }
  return s;
}

void probeAfterDraw(const gbstate::State& t, uint32_t prims) {
  if (!g_probe.active) return;
  g_probe.draw++;
  std::vector<uint8_t> d3d, vk;
  D3DFORMAT fmt = D3DFMT_UNKNOWN;
  uint32_t vkFmt = 0;
  if (!d3dTexel(g_probe.x, g_probe.y, d3d, fmt) || !g_be->probePixel(g_probe.x, g_probe.y, vk, vkFmt)) return;
  bool same = d3d.size() == vk.size() && memcmp(d3d.data(), vk.data(), d3d.size()) == 0;
  if (!same && fmt == D3DFMT_X8R8G8B8 && d3d.size() == 4 && vk.size() == 4) same = memcmp(d3d.data(), vk.data(), 3) == 0;
  const bool changed = d3d != g_probe.lastD3d;
  g_probe.lastD3d = d3d;
  if ((changed || !same) && g_probe.logged < 300) {
    g_probe.logged++;
    uint8_t vsMd5[16] = {}, psMd5[16] = {};
    gbshaders::lookup(t.vs, vsMd5);
    gbshaders::lookup(t.ps, psMd5);
    std::vector<uint8_t> vsv(vsMd5, vsMd5 + 16), psv(psMd5, psMd5 + 16);
    logf("probe draw %u: d3d %s vk %s %s | fmt %d prims %u | vs %s ps %s | blend %u src %u dst %u alphatest %u ref %u",
         g_probe.draw, hexBytes(d3d).c_str(), hexBytes(vk).c_str(), same ? "same" : "DIFFERENT", int(fmt), prims,
         hexBytes(vsv).c_str(), hexBytes(psv).c_str(), t.rs[D3DRS_ALPHABLENDENABLE], t.rs[D3DRS_SRCBLEND],
         t.rs[D3DRS_DESTBLEND], t.rs[D3DRS_ALPHATESTENABLE], t.rs[D3DRS_ALPHAREF]);
  }
}

void drawFromTracker(uint32_t pt, uint32_t prims, bool indexed, int32_t baseVertex, uint32_t start, const void* upData,
                     uint32_t upStride) {
  if (!g_draws || !g_be) return;
  cputime::Scope timed(cputime::Draw);
  const gbstate::State& t = gbstate::state();
  DrawSetup s;
  s.renderStates = t.rs;
  s.samplerStates = t.ss;
  // Per sampler: the binding of the texture bound there, looked up again only when the texture or the maps changed
  // (draws come from the main thread only).
  {
    struct Slot {
      void* tex;
      uint32_t gen;
      DrawSetup::Texture b;
    };
    static Slot slots[16] = {};
    const uint32_t gen = g_bindingGen.load(std::memory_order_acquire);
    for (int i = 0; i < 16; i++) {
      Slot& sl = slots[i];
      if (sl.tex != t.tex[i] || sl.gen != gen) {
        sl.tex = t.tex[i];
        sl.gen = gen;
        sl.b = bindingFor(static_cast<IDirect3DBaseTexture9*>(t.tex[i]));
      }
      s.textures[i] = sl.b;
    }
  }
  s.vs = ShaderKey(t.vs);
  s.ps = ShaderKey(t.ps);
  s.decl = DeclKey(t.decl);
  for (int i = 0; i < 8; i++) s.streams[i] = { BufKey(t.streams[i].vb), t.streams[i].offset, t.streams[i].stride };
  s.indices = BufKey(t.ib);
  if (upData) {
    // User-pointer draw: the vertices go through a scratch buffer written with DISCARD, as D3D9 streams them.
    scratch(kScratchVb, g_scratchVbSize, upData, upStride * xlat::vertexCount(pt, prims), false, false);
    s.streams[0] = { kScratchVb, 0, upStride };
  }
  s.vsConstants = &t.vsc[0][0];
  s.psConstants = &t.psc[0][0];
  s.vsConstantsVersion = t.vscVersion;
  s.psConstantsVersion = t.pscVersion;
  s.renderStatesVersion = t.rsVersion;
  s.clipPlanesVersion = t.clipVersion;
  s.viewport = { int32_t(t.viewport.x), int32_t(t.viewport.y), t.viewport.w, t.viewport.h, t.viewport.minZ, t.viewport.maxZ };
  memcpy(s.scissor, t.scissor, sizeof(s.scissor));
  memcpy(s.clipPlanes, t.clip, sizeof(s.clipPlanes));
  {
    cputime::Scope backendTimed(cputime::DrawBackend);
    g_be->draw(s, pt, prims, indexed, baseVertex, start);
  }
  g_c.trackedDraws++;
  probeAfterDraw(t, prims);
}

void registerTexture(void* t, uintptr_t key) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_texToKey[t] = key;
  g_bindingGen++;
}
void unregisterTexture(void* t) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_texToKey.erase(t);
  g_bindingGen++;
}
void registerSurface(void* surface, uintptr_t rtKey) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto old = g_rtToSurf.find(rtKey);
  if (old != g_rtToSurf.end()) g_surfToRt.erase(old->second);
  g_rtToSurf[rtKey] = surface;
  if (surface) g_surfToRt[surface] = rtKey;
  g_bindingGen++;
}
uintptr_t texKeyForTexture(void* texture) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_texToKey.find(texture);
  return it == g_texToKey.end() ? 0 : it->second;
}
uintptr_t rtKeyForRtTexture(void* texture) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_rtTexToRt.find(texture);
  return it == g_rtTexToRt.end() ? 0 : it->second;
}
uintptr_t rtKeyForSurface(void* surface) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_surfToRt.find(surface);
  return it == g_surfToRt.end() ? 0 : it->second;
}
void registerRtTexture(void* t, uintptr_t rtKey) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto old = g_rtToRtTex.find(rtKey);
  if (old != g_rtToRtTex.end()) g_rtTexToRt.erase(old->second);
  g_rtToRtTex[rtKey] = t;
  if (t) g_rtTexToRt[t] = rtKey;
  g_bindingGen++;
}

void unregisterRtKey(uintptr_t rtKey) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_bindingGen++;
  auto rtt = g_rtToRtTex.find(rtKey);
  if (rtt != g_rtToRtTex.end()) {
    g_rtTexToRt.erase(rtt->second);
    g_rtToRtTex.erase(rtt);
  }
  auto old = g_rtToSurf.find(rtKey);
  if (old == g_rtToSurf.end()) return;
  g_surfToRt.erase(old->second);
  g_rtToSurf.erase(old);
}

void compareFrame(uintptr_t bbKey, const char* dumpPrefix) {
  if (!g_dev || !g_be) return;
  IDirect3DSurface9 *bb = nullptr, *plain = nullptr, *sys = nullptr;
  if (FAILED(g_dev->GetBackBuffer(0, 0, D3DBACKBUFFER_TYPE_MONO, &bb)) || !bb) return;
  D3DSURFACE_DESC d{};
  bb->GetDesc(&d);
  std::vector<uint8_t> ref;
  bool ok = SUCCEEDED(g_dev->CreateRenderTarget(d.Width, d.Height, d.Format, D3DMULTISAMPLE_NONE, 0, FALSE, &plain, nullptr)) &&
            SUCCEEDED(g_dev->StretchRect(bb, nullptr, plain, nullptr, D3DTEXF_NONE)) &&
            SUCCEEDED(g_dev->CreateOffscreenPlainSurface(d.Width, d.Height, d.Format, D3DPOOL_SYSTEMMEM, &sys, nullptr)) &&
            SUCCEEDED(g_dev->GetRenderTargetData(plain, sys));
  D3DLOCKED_RECT lr{};
  if (ok && SUCCEEDED(sys->LockRect(&lr, nullptr, D3DLOCK_READONLY))) {
    ref.resize(size_t(d.Width) * d.Height * 4);
    for (uint32_t y = 0; y < d.Height; y++)
      memcpy(&ref[size_t(y) * d.Width * 4], static_cast<uint8_t*>(lr.pBits) + size_t(y) * lr.Pitch, size_t(d.Width) * 4);
    sys->UnlockRect();
  }
  if (sys) sys->Release();
  if (plain) plain->Release();
  bb->Release();
  std::vector<uint8_t> got;
  uint32_t vf = 0, ts = 0;
  if (ref.empty() || !g_be->readback(bbKey, got, vf, ts) || got.size() != ref.size()) {
    logf("frame compare: readback failed (d3d %zu bytes, vk %zu bytes)", ref.size(), got.size());
    return;
  }
  uint64_t diff = 0, big = 0, sum = 0;
  int maxd = 0;
  size_t worst = 0;
  for (size_t i = 0; i < ref.size(); i += 4) {
    int m = 0;
    for (int c = 0; c < 3; c++) m = std::max(m, std::abs(int(ref[i + c]) - int(got[i + c])));
    if (m) diff++;
    if (m > 8) big++;
    sum += m;
    if (m > maxd) { maxd = m; worst = i / 4; }
  }
  if (maxd >= 64 && !g_probe.armed && g_probeEnabled) {
    g_probe.armed = true;
    g_probe.x = uint32_t(worst % d.Width);
    g_probe.y = uint32_t(worst / d.Width);
    logf("probe armed for the next frame at (%u, %u): difference %d", g_probe.x, g_probe.y, maxd);
  }
  g_c.compares++;
  uint64_t n = ref.size() / 4;
  logf("frame compare %llu (%ux%u): %llu of %llu pixels differ (%.2f%%), %llu by more than 8/255 (%.2f%%), max %d, mean %.3f",
       g_c.compares.load(), d.Width, d.Height, diff, n, 100.0 * diff / n, big, 100.0 * big / n, maxd, double(sum) / n);
  if (dumpPrefix) {
    writeBmp(std::string(dumpPrefix) + "_d3d9.bmp", ref, d.Width, d.Height);
    writeBmp(std::string(dumpPrefix) + "_vulkan.bmp", got, d.Width, d.Height);
  }
}

void onFrameStart() {
  if (g_probe.active) {
    g_probe.active = false;
    logf("probe finished: %u draws probed", g_probe.draw);
  }
  if (g_probe.armed && !g_probe.draw) {
    g_probe.active = true;
    g_probe.lastD3d.clear();
  }
}
void enableProbe(bool on) { g_probeEnabled = on; }

void logSummary() {
  logf("draws issued from the game-side tracker: %llu; from unpatched sites (drawn from D3D9 state) %llu; texture "
       "bindings not mapped game-side %llu", g_c.trackedDraws.load(), g_c.unpatchedDraws.load(), g_c.rtTexFallback.load());
  logf("game-side declarations vs D3D9: match %llu, MISMATCH %llu, unregistered %llu", g_c.declMatch.load(),
       g_c.declMismatch.load(), g_c.declUnregistered.load());
  logf("tracker vs D3D9 at %llu draws: render state %llu, sampler %llu, texture %llu, shaders %llu, declaration %llu, "
       "VS constants %llu, PS constants %llu, viewport/scissor/clip/streams/indices %llu draws differing",
       g_td.draws, g_td.rs, g_td.ss, g_td.tex, g_td.shaders, g_td.decl, g_td.vsc, g_td.psc, g_td.other);
  logf("observer: draws %llu (UP %llu, FVF %llu) | textures: render-target %llu, unknown %llu, vertex-stage %llu | "
       "shaders created %llu (not in pack %llu) | buffers VB %llu IB %llu, writes %llu, released %llu | clip-plane sets %llu, "
       "scissor sets %llu | game-side shaders at first draw: match %llu, MISMATCH %llu, unregistered %llu",
       g_c.draws.load(), g_c.drawsUP.load(), g_c.fvfDraws.load(), g_c.rtTex.load(), g_c.unknownTex.load(),
       g_c.vsTextures.load(), g_c.shadersCreated.load(), g_c.shadersNoPack.load(), g_c.vbCreated.load(),
       g_c.ibCreated.load(), g_c.writes.load(), g_c.released.load(), g_c.clipPlaneSets.load(), g_c.scissorSets.load(), g_c.shaderMatch.load(), g_c.shaderMismatch.load(),
       g_c.shaderUnregistered.load());
}

}  // namespace fnvvk::observe
