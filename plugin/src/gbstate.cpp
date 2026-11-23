// Ports of Gamebryo's render-state object (NiD3DRenderState / NiDX9RenderState, 0x00E88000-0x00E91200), from the
// decompilation and IDA (fnv_decomp, docs/M2_DESIGN.md). Each port reproduces the original's cache logic and data
// writes exactly and calls the device facade (gbstate.h) where the original called IDirect3DDevice9.
//
// Layout (IDA types, checked against the machine code of these functions): NiD3DRenderState
//   +0x120 NiRenderStateSetting m_akRenderStateSettings[256] { curr, prev }   +0x920 bool m_bSavedStates[256]
//   +0xA20 m_akTextureStageStateSettings[16][8]   +0xE20 m_akSamplerStateSettings[16][5]
//   +0x10A0 textures[16]  +0x10E0/+0x10E4 VS current/previous  +0x10E8/+0x10EC PS current/previous  +0x10F8 device
// NiDX9RenderState: +0x1100 bool m_bDeclaration, +0x1104/+0x1108 FVF current/previous, +0x110C/+0x1110 declaration.
#include "gbstate.h"

#include <windows.h>
#include <d3d9.h>

#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "backend.h"
#include "gbobjects.h"
#include "observe.h"
#include "shadow.h"

#include "common.h"

namespace fnvvk::gbstate {

namespace {

State g_state;
// Which entries the game has set through the tracker. initFromDevice fills only the others from the device: with
// forwarding off from plugin load (native mode) the device never received the game's state, only its defaults.
struct SetMask {
  bool rs[256] = {}, ss[16][14] = {}, viewport = false, scissor = false, clip[6] = {}, vsc[256] = {}, psc[224] = {};
} g_set;
bool g_forward = true;  // D3D9 exists (shadow phase)
bool g_forwardState = true;  // D3D9 also receives the state (off in native mode: nothing reads it)
bool g_forwardDraws = true;  // D3D9 also executes the draws ([Backend] D3D9Draws); off = Vulkan alone renders

template <typename T> T& at(void* p, uint32_t off) { return *reinterpret_cast<T*>(static_cast<char*>(p) + off); }
IDirect3DDevice9* devOf(void* rs) { return at<IDirect3DDevice9*>(rs, 0x10F8); }

struct Setting { uint32_t curr, prev; };
Setting& rsSetting(void* rs, uint32_t s) { return reinterpret_cast<Setting*>(static_cast<char*>(rs) + 0x120)[s]; }
bool& rsSaved(void* rs, uint32_t s) { return reinterpret_cast<bool*>(static_cast<char*>(rs) + 0x920)[s]; }
Setting& tssSetting(void* rs, uint32_t stage, uint32_t i) { return reinterpret_cast<Setting*>(static_cast<char*>(rs) + 0xA20)[stage * 8 + i]; }
Setting& sampSetting(void* rs, uint32_t stage, uint32_t i) { return reinterpret_cast<Setting*>(static_cast<char*>(rs) + 0xE20)[stage * 5 + i]; }
void*& texSlot(void* rs, uint32_t stage) { return reinterpret_cast<void**>(static_cast<char*>(rs) + 0x10A0)[stage]; }

const uint16_t* const kSampRevMap = reinterpret_cast<const uint16_t*>(0x0126F92C);  // filled at run time
const uint16_t* const kTssRevMap = reinterpret_cast<const uint16_t*>(0x0126F958);
struct InitialValue { uint32_t type, value; };
InitialValue* const kInitialRS = reinterpret_cast<InitialValue*>(0x011BFCE8);
InitialValue* const kInitialTSS = reinterpret_cast<InitialValue*>(0x011C04F0);
InitialValue* const kInitialSamp = reinterpret_cast<InitialValue*>(0x011C0538);
const uint32_t& kMaxBlendStages = *reinterpret_cast<const uint32_t*>(0x011BF558);
const uint32_t& kMaxSamplers = *reinterpret_cast<const uint32_t*>(0x011BF560);
constexpr uint32_t kEnd = 0xFFFFFFFF;  // D3D*_FORCE_DWORD | 0x80000000 terminator

// --- ports ------------------------------------------------------------------------------------------------------
// NiD3DRenderState::SetRenderState 0x00E88780
void __fastcall SetRenderState(void* self, void*, uint32_t s, uint32_t v, bool, bool bSave) {
  if (rsSaved(self, s)) return;
  rsSaved(self, s) = bSave;
  if (rsSetting(self, s).curr != v) {
    dev::setRenderState(devOf(self), s, v);
    rsSetting(self, s).curr = v;
  }
}
// NiD3DRenderState::SetDestAndSrcBlends 0x00E887D0
// retn 10h: four stack arguments (the last two unused).
void __fastcall SetDestAndSrcBlends(void* self, void*, uint32_t src, uint32_t dst, int, int) {
  if (rsSetting(self, D3DRS_SRCBLEND).curr != src) dev::setRenderState(devOf(self), D3DRS_SRCBLEND, src);
  rsSetting(self, D3DRS_SRCBLEND).curr = src;
  if (rsSetting(self, D3DRS_DESTBLEND).curr != dst) dev::setRenderState(devOf(self), D3DRS_DESTBLEND, dst);
  rsSetting(self, D3DRS_DESTBLEND).curr = dst;
}
// NiD3DRenderState::SetAlphaBlendEnable 0x00E88830
// retn 0Ch: three stack arguments (the last two unused).
void __fastcall SetAlphaBlendEnable(void* self, void*, uint32_t v, int, int) {
  if (rsSetting(self, D3DRS_ALPHABLENDENABLE).curr != v) dev::setRenderState(devOf(self), D3DRS_ALPHABLENDENABLE, v);
  rsSetting(self, D3DRS_ALPHABLENDENABLE).curr = v;
}
// NiD3DRenderState::InitRenderState 0x00E88000
void __fastcall InitRenderState(void* self, void*) {
  for (InitialValue* p = kInitialRS; p->type != kEnd; p++) {
    rsSetting(self, p->type).prev = p->value;
    rsSetting(self, p->type).curr = p->value;
    dev::setRenderState(devOf(self), p->type, p->value);
  }
}
// NiD3DRenderState::InitTextureStageState 0x00E88060 (the stage loop rewrites the static table as the original does)
using SetSamplerVFn = void(__thiscall*)(void*, uint32_t, uint32_t, uint32_t, bool);
void __fastcall InitTextureStageState(void* self, void*) {
  for (uint32_t i = 0; i < kMaxBlendStages; i++) {
    for (InitialValue* p = kInitialTSS; p->type != kEnd; p++) {
      if (p->type == D3DTSS_TEXCOORDINDEX) p->value = i;
      else if ((p->type == D3DTSS_COLOROP || p->type == D3DTSS_ALPHAOP) && i) p->value = 1;
      tssSetting(self, i, kTssRevMap[p->type]).prev = p->value;
      dev::setTextureStageState(devOf(self), i, p->type, p->value);
    }
  }
  for (uint32_t s = 0; s < kMaxSamplers; s++) {
    for (InitialValue* p = kInitialSamp; p->type != kEnd; p++) {
      sampSetting(self, s, kSampRevMap[p->type]).prev = p->value;
      // Virtual SetSamplerState (NiD3DRenderState_vtbl +0xCC, 0x00E8813F), then the device directly.
      void** vt = *reinterpret_cast<void***>(self);
      reinterpret_cast<SetSamplerVFn>(vt[0xCC / 4])(self, s, p->type, p->value, false);
      dev::setSamplerState(devOf(self), s, p->type, p->value);
    }
  }
}
// NiD3DRenderState::SetTextureStageState 0x00E88930
void __fastcall SetTextureStageState(void* self, void*, uint32_t stage, uint32_t s, uint32_t v, bool) {
  uint16_t i = kTssRevMap[s];
  if (i < 8) {
    if (tssSetting(self, stage, i).curr != v) dev::setTextureStageState(devOf(self), stage, s, v);
    tssSetting(self, stage, i).curr = v;
  }
}
// NiDX9RenderState::SetSamplerState 0x00E910A0 (types without a cache slot are dropped, as in the original)
void __fastcall SetSamplerState(void* self, void*, uint32_t stage, uint32_t s, uint32_t v, bool) {
  uint16_t i = kSampRevMap[s];
  if (i < 5 && sampSetting(self, stage, i).curr != v) {
    sampSetting(self, stage, i).curr = v;
    dev::setSamplerState(devOf(self), stage, s, v);
  }
}
// NiD3DRenderState::SetTexture 0x00E88A20 / ClearTexture 0x00E88A60
void __fastcall SetTexture(void* self, void*, uint32_t stage, void* tex) {
  if (texSlot(self, stage) != tex) {
    texSlot(self, stage) = tex;
    dev::setTexture(devOf(self), stage, tex);
  }
}
void __fastcall ClearTexture(void* self, void*, void* tex) {
  if (!tex) return;
  for (uint32_t i = 0; i < 16; i++)
    if (texSlot(self, i) == tex) {
      texSlot(self, i) = nullptr;
      dev::setTexture(devOf(self), i, nullptr);
    }
}
// NiD3DRenderState::SetPixelShader 0x00E88870 / ClearPixelShader 0x00E888E0
void __fastcall SetPixelShader(void* self, void*, void* h, bool bSave) {
  if (bSave) at<void*>(self, 0x10EC) = at<void*>(self, 0x10E8);
  if (at<void*>(self, 0x10E8) != h) {
    at<void*>(self, 0x10E8) = h;
    dev::setPixelShader(devOf(self), h);
  }
}
void __fastcall ClearPixelShader(void* self, void*, void* h) {
  if (at<void*>(self, 0x10E8) == h) {
    at<void*>(self, 0x10E8) = nullptr;
    dev::setPixelShader(devOf(self), nullptr);
  }
  if (at<void*>(self, 0x10EC) == h) at<void*>(self, 0x10EC) = nullptr;
}
// NiDX9RenderState::SetVertexShader 0x00E90E00 / ClearVertexShader 0x00E90E70
void __fastcall SetVertexShader(void* self, void*, void* h, bool bSave) {
  if (bSave) at<void*>(self, 0x10E4) = at<void*>(self, 0x10E0);
  if (at<void*>(self, 0x10E0) != h) {
    at<void*>(self, 0x10E0) = h;
    dev::setVertexShader(devOf(self), h);
  }
}
void __fastcall ClearVertexShader(void* self, void*, void* h) {
  if (at<void*>(self, 0x10E0) == h) {
    at<void*>(self, 0x10E0) = nullptr;
    dev::setVertexShader(devOf(self), nullptr);
  }
  if (at<void*>(self, 0x10E4) == h) at<void*>(self, 0x10E4) = nullptr;
}
// NiDX9RenderState::SetFVF 0x00E90EC0 / ClearFVF 0x00E90F60
void __fastcall SetFVF(void* self, void*, uint32_t fvf, bool bSave) {
  if (!fvf) return;
  if (at<bool>(self, 0x1100) || at<uint32_t>(self, 0x1104) != fvf) {
    at<bool>(self, 0x1100) = false;
    if (bSave) at<uint32_t>(self, 0x1108) = at<uint32_t>(self, 0x1104);
    at<uint32_t>(self, 0x1104) = fvf;
    dev::setFVF(devOf(self), fvf);
  }
}
void __fastcall ClearFVF(void* self, void*, uint32_t fvf) {
  if (!fvf || at<bool>(self, 0x1100)) return;
  if (at<uint32_t>(self, 0x1104) == fvf) {
    at<uint32_t>(self, 0x1104) = 2;
    dev::setFVF(devOf(self), 2);
  }
  if (at<uint32_t>(self, 0x1108) == fvf) at<uint32_t>(self, 0x1108) = 2;
}
// NiDX9RenderState::SetDeclaration 0x00E90FC0
void __fastcall SetDeclaration(void* self, void*, void* decl, bool bSave) {
  if (!decl) return;
  if (!at<bool>(self, 0x1100) || at<void*>(self, 0x110C) != decl) {
    at<bool>(self, 0x1100) = true;
    if (bSave) at<void*>(self, 0x1110) = at<void*>(self, 0x110C);
    at<void*>(self, 0x110C) = decl;
    dev::setDeclaration(devOf(self), decl);
  }
}
// Constants: NiD3DRenderState::SetVertexShaderConstant{B,F,I} 0x00E89AE0/0x00E89B10/0x00E89B40,
// NiDX9RenderState::SetPixelShaderConstant{B,F,I} 0x00E898E0/0x00E89910/0x00E89940: straight to the device.
bool __fastcall SetVSConstB(void* self, void*, uint32_t r, const int* v, uint32_t n, bool) { return dev::setVSConstB(devOf(self), r, v, n) >= 0; }
bool __fastcall SetVSConstF(void* self, void*, uint32_t r, const float* v, uint32_t n, bool) { return dev::setVSConstF(devOf(self), r, v, n) >= 0; }
bool __fastcall SetVSConstI(void* self, void*, uint32_t r, const int* v, uint32_t n, bool) { return dev::setVSConstI(devOf(self), r, v, n) >= 0; }
bool __fastcall SetPSConstB(void* self, void*, uint32_t r, const int* v, uint32_t n, bool) { return dev::setPSConstB(devOf(self), r, v, n) >= 0; }
bool __fastcall SetPSConstF(void* self, void*, uint32_t r, const float* v, uint32_t n, bool) { return dev::setPSConstF(devOf(self), r, v, n) >= 0; }
bool __fastcall SetPSConstI(void* self, void*, uint32_t r, const int* v, uint32_t n, bool) { return dev::setPSConstI(devOf(self), r, v, n) >= 0; }

}  // namespace

State& state() { return g_state; }

namespace dev {
void setRenderState(IDirect3DDevice9* d, uint32_t s, uint32_t v) {
  if (s < 256) {
    if (g_state.rs[s] != v) g_state.rsVersion++;
    g_state.rs[s] = v;
    g_set.rs[s] = true;
  }
  if (g_forwardState && d) d->SetRenderState(D3DRENDERSTATETYPE(s), v);
}
void setSamplerState(IDirect3DDevice9* d, uint32_t stage, uint32_t s, uint32_t v) {
  if (stage < 16 && s < 14) { g_state.ss[stage][s] = v; g_set.ss[stage][s] = true; }
  if (g_forwardState && d) d->SetSamplerState(stage, D3DSAMPLERSTATETYPE(s), v);
}
void setTexture(IDirect3DDevice9* d, uint32_t stage, void* tex) {
  if (stage < 16) g_state.tex[stage] = tex;
  if (g_forwardState && d) d->SetTexture(stage, static_cast<IDirect3DBaseTexture9*>(tex));
}
void setTextureStageState(IDirect3DDevice9* d, uint32_t stage, uint32_t s, uint32_t v) {
  // Fixed-function stage state: no translated shader reads it (SHADER_ABI: binding 8 unused by all modules).
  if (g_forwardState && d) d->SetTextureStageState(stage, D3DTEXTURESTAGESTATETYPE(s), v);
}
void setVertexShader(IDirect3DDevice9* d, void* vs) {
  g_state.vs = vs;
  if (g_forwardState && d) d->SetVertexShader(static_cast<IDirect3DVertexShader9*>(vs));
}
void setPixelShader(IDirect3DDevice9* d, void* ps) {
  g_state.ps = ps;
  if (g_forwardState && d) d->SetPixelShader(static_cast<IDirect3DPixelShader9*>(ps));
}
void setFVF(IDirect3DDevice9* d, uint32_t fvf) {
  g_state.fvf = fvf;
  g_state.decl = nullptr;  // D3D9: SetFVF replaces the declaration
  if (g_forwardState && d) d->SetFVF(fvf);
}
void setDeclaration(IDirect3DDevice9* d, void* decl) {
  g_state.decl = decl;
  if (g_forwardState && d) d->SetVertexDeclaration(static_cast<IDirect3DVertexDeclaration9*>(decl));
}
long setVSConstF(IDirect3DDevice9* d, uint32_t start, const float* v, uint32_t n) {
  if (v && start < 256) {
    memcpy(g_state.vsc[start], v, size_t(n < 256 - start ? n : 256 - start) * 16);
    for (uint32_t i = start; i < start + n && i < 256; i++) g_set.vsc[i] = true;
    g_state.vscVersion++;
  }
  return g_forwardState && d ? d->SetVertexShaderConstantF(start, v, n) : 0;
}
long setPSConstF(IDirect3DDevice9* d, uint32_t start, const float* v, uint32_t n) {
  if (v && start < 224) {
    memcpy(g_state.psc[start], v, size_t(n < 224 - start ? n : 224 - start) * 16);
    for (uint32_t i = start; i < start + n && i < 224; i++) g_set.psc[i] = true;
    g_state.pscVersion++;
  }
  return g_forwardState && d ? d->SetPixelShaderConstantF(start, v, n) : 0;
}
// Integer and boolean constants: no translated shader reads them (SHADER_ABI bindings 1 and 2 unused).
long setVSConstI(IDirect3DDevice9* d, uint32_t start, const int* v, uint32_t n) { return g_forwardState && d ? d->SetVertexShaderConstantI(start, v, n) : 0; }
long setPSConstI(IDirect3DDevice9* d, uint32_t start, const int* v, uint32_t n) { return g_forwardState && d ? d->SetPixelShaderConstantI(start, v, n) : 0; }
long setVSConstB(IDirect3DDevice9* d, uint32_t start, const int* v, uint32_t n) { return g_forwardState && d ? d->SetVertexShaderConstantB(start, reinterpret_cast<const BOOL*>(v), n) : 0; }
long setPSConstB(IDirect3DDevice9* d, uint32_t start, const int* v, uint32_t n) { return g_forwardState && d ? d->SetPixelShaderConstantB(start, reinterpret_cast<const BOOL*>(v), n) : 0; }
}  // namespace dev

void setForwardDraws(bool on) {
  g_forwardDraws = on;
  g_forwardState = on;
}
void setNoD3D9() {
  g_forward = false;
  g_forwardDraws = false;
  g_forwardState = false;
}
bool forwardDraws() { return g_forwardDraws; }

void initFromDevice(IDirect3DDevice9* d) {
  if (!d) return;
  if (!g_set.viewport) {
    D3DVIEWPORT9 vp{};
    d->GetViewport(&vp);
    g_state.viewport = { vp.X, vp.Y, vp.Width, vp.Height, vp.MinZ, vp.MaxZ };
  }
  if (!g_set.scissor) {
    RECT sc{};
    d->GetScissorRect(&sc);
    g_state.scissor[0] = sc.left;
    g_state.scissor[1] = sc.top;
    g_state.scissor[2] = sc.right;
    g_state.scissor[3] = sc.bottom;
  }
  for (uint32_t i = 0; i < 6; i++)
    if (!g_set.clip[i]) d->GetClipPlane(i, g_state.clip[i]);
  g_state.clipVersion++;
  for (uint32_t i = 0; i < 256; i++) {
    DWORD v = 0;
    if (!g_set.rs[i] && SUCCEEDED(d->GetRenderState(D3DRENDERSTATETYPE(i), &v))) g_state.rs[i] = v;
    g_state.rsVersion++;
  }
  for (uint32_t s = 0; s < 16; s++)
    for (uint32_t i = 1; i < 14; i++) {
      DWORD v = 0;
      if (!g_set.ss[s][i] && SUCCEEDED(d->GetSamplerState(s, D3DSAMPLERSTATETYPE(i), &v))) g_state.ss[s][i] = v;
    }
  for (uint32_t i = 0; i < 256; i++)
    if (!g_set.vsc[i]) d->GetVertexShaderConstantF(i, g_state.vsc[i], 1);
  for (uint32_t i = 0; i < 224; i++)
    if (!g_set.psc[i]) d->GetPixelShaderConstantF(i, g_state.psc[i], 1);
}

// --- direct device-call sites ------------------------------------------------------------------------------------
// Each site is a 6-byte `mov r32, [r32+disp32]` loading a method from the IDirect3DDevice9 vtable before `call r32`
// (IDA: operands typed IDirect3DDevice9_vtbl, plus the untyped loads in functions that call nothing but the device;
// candidates with the same displacement in other classes' vtables or fields are excluded). The load becomes
// `mov r32, imm32; nop` with a facade thunk of the identical COM signature: the function's call lands in the
// tracker, and the thunk forwards to D3D9 while it exists. Each site's bytes are checked before patching.
namespace {
// Forwarding goes through the calling device's own vtable (the patches are applied before the device exists).
template <typename F> F real(IDirect3DDevice9* d, int index) { return reinterpret_cast<F>((*reinterpret_cast<void***>(d))[index]); }

HRESULT STDMETHODCALLTYPE tSetRenderState(IDirect3DDevice9* d, D3DRENDERSTATETYPE s, DWORD v) {
  if (uint32_t(s) < 256) {
    if (g_state.rs[s] != v) g_state.rsVersion++;
    g_state.rs[s] = v;
    g_set.rs[s] = true;
  }
  return g_forwardState ? real<HRESULT(STDMETHODCALLTYPE*)(IDirect3DDevice9*, D3DRENDERSTATETYPE, DWORD)>(d, 57)(d, s, v) : 0;
}
HRESULT STDMETHODCALLTYPE tSetSamplerState(IDirect3DDevice9* d, DWORD st, D3DSAMPLERSTATETYPE s, DWORD v) {
  if (st < 16 && uint32_t(s) < 14) { g_state.ss[st][s] = v; g_set.ss[st][s] = true; }
  return g_forwardState ? real<HRESULT(STDMETHODCALLTYPE*)(IDirect3DDevice9*, DWORD, D3DSAMPLERSTATETYPE, DWORD)>(d, 69)(d, st, s, v) : 0;
}
HRESULT STDMETHODCALLTYPE tSetViewport(IDirect3DDevice9* d, const D3DVIEWPORT9* vp) {
  if (vp) { g_state.viewport = { vp->X, vp->Y, vp->Width, vp->Height, vp->MinZ, vp->MaxZ }; g_set.viewport = true; }
  return g_forwardState ? real<HRESULT(STDMETHODCALLTYPE*)(IDirect3DDevice9*, const D3DVIEWPORT9*)>(d, 47)(d, vp) : 0;
}
HRESULT STDMETHODCALLTYPE tSetClipPlane(IDirect3DDevice9* d, DWORD i, const float* p) {
  if (i < 6 && p) {
    if (memcmp(g_state.clip[i], p, 16) != 0) g_state.clipVersion++;
    memcpy(g_state.clip[i], p, 16);
    g_set.clip[i] = true;
  }
  return g_forwardState ? real<HRESULT(STDMETHODCALLTYPE*)(IDirect3DDevice9*, DWORD, const float*)>(d, 55)(d, i, p) : 0;
}
HRESULT STDMETHODCALLTYPE tSetScissorRect(IDirect3DDevice9* d, const RECT* r) {
  if (r) {
    g_set.scissor = true;
    g_state.scissor[0] = r->left;
    g_state.scissor[1] = r->top;
    g_state.scissor[2] = r->right;
    g_state.scissor[3] = r->bottom;
  }
  return g_forwardState ? real<HRESULT(STDMETHODCALLTYPE*)(IDirect3DDevice9*, const RECT*)>(d, 75)(d, r) : 0;
}
HRESULT STDMETHODCALLTYPE tSetVertexShader(IDirect3DDevice9* d, IDirect3DVertexShader9* v) {
  g_state.vs = v;
  return g_forwardState ? real<HRESULT(STDMETHODCALLTYPE*)(IDirect3DDevice9*, IDirect3DVertexShader9*)>(d, 92)(d, v) : 0;
}
HRESULT STDMETHODCALLTYPE tSetPixelShader(IDirect3DDevice9* d, IDirect3DPixelShader9* p) {
  g_state.ps = p;
  return g_forwardState ? real<HRESULT(STDMETHODCALLTYPE*)(IDirect3DDevice9*, IDirect3DPixelShader9*)>(d, 107)(d, p) : 0;
}
HRESULT STDMETHODCALLTYPE tSetVertexDeclaration(IDirect3DDevice9* d, IDirect3DVertexDeclaration9* v) {
  g_state.decl = v;
  return g_forwardState ? real<HRESULT(STDMETHODCALLTYPE*)(IDirect3DDevice9*, IDirect3DVertexDeclaration9*)>(d, 87)(d, v) : 0;
}
HRESULT STDMETHODCALLTYPE tSetStreamSource(IDirect3DDevice9* d, UINT n, IDirect3DVertexBuffer9* vb, UINT off, UINT stride) {
  if (n < 8) g_state.streams[n] = { vb, off, stride };
  return g_forwardState ? real<HRESULT(STDMETHODCALLTYPE*)(IDirect3DDevice9*, UINT, IDirect3DVertexBuffer9*, UINT, UINT)>(d, 100)(d, n, vb, off, stride) : 0;
}
HRESULT STDMETHODCALLTYPE tSetIndices(IDirect3DDevice9* d, IDirect3DIndexBuffer9* ib) {
  g_state.ib = ib;
  return g_forwardState ? real<HRESULT(STDMETHODCALLTYPE*)(IDirect3DDevice9*, IDirect3DIndexBuffer9*)>(d, 104)(d, ib) : 0;
}
HRESULT STDMETHODCALLTYPE tSetVSConstF(IDirect3DDevice9* d, UINT r, const float* v, UINT n) { return dev::setVSConstF(d, r, v, n); }
HRESULT STDMETHODCALLTYPE tSetPSConstF(IDirect3DDevice9* d, UINT r, const float* v, UINT n) { return dev::setPSConstF(d, r, v, n); }
HRESULT STDMETHODCALLTYPE tSetVSConstI(IDirect3DDevice9* d, UINT r, const int* v, UINT n) { return dev::setVSConstI(d, r, v, n); }
HRESULT STDMETHODCALLTYPE tSetPSConstI(IDirect3DDevice9* d, UINT r, const int* v, UINT n) { return dev::setPSConstI(d, r, v, n); }
HRESULT STDMETHODCALLTYPE tSetVSConstB(IDirect3DDevice9* d, UINT r, const BOOL* v, UINT n) { return dev::setVSConstB(d, r, reinterpret_cast<const int*>(v), n); }
HRESULT STDMETHODCALLTYPE tSetPSConstB(IDirect3DDevice9* d, UINT r, const BOOL* v, UINT n) { return dev::setPSConstB(d, r, reinterpret_cast<const int*>(v), n); }

// Declarations: created by NiDX9Renderer::GetDecl_ParticlesSubTexture / GetDecl_Particles and
// NiDX9ShaderDeclaration::GetD3DDeclaration (the only callers, IDA); the elements go to the backend.
std::mutex g_declMutex;
std::unordered_map<void*, std::vector<uint8_t>> g_declElements;  // handle -> D3DVERTEXELEMENT9 bytes (without END)
thread_local int t_inDeclThunk = 0;
HRESULT STDMETHODCALLTYPE tCreateVertexDeclaration(IDirect3DDevice9* d, const D3DVERTEXELEMENT9* e, IDirect3DVertexDeclaration9** out) {
  struct Scope { Scope() { t_inDeclThunk++; } ~Scope() { t_inDeclThunk--; } } scope;
  // Without D3D9 (native device) the declaration is a plugin object; otherwise D3D9 creates it.
  HRESULT r = g_forward ? real<HRESULT(STDMETHODCALLTYPE*)(IDirect3DDevice9*, const D3DVERTEXELEMENT9*, IDirect3DVertexDeclaration9**)>(d, 86)(d, e, out)
                        : gbobjects::createVertexDeclaration(d, e, reinterpret_cast<void**>(out));
  if (SUCCEEDED(r) && out && *out && e) {
    std::vector<VertexElement> v;
    const D3DVERTEXELEMENT9* p = e;
    for (; p->Stream != 0xFF && v.size() < 64; p++) v.push_back({ p->Stream, p->Offset, p->Type, p->Method, p->Usage, p->UsageIndex });
    {
      std::lock_guard<std::mutex> lock(g_declMutex);
      g_declElements[*out].assign(reinterpret_cast<const uint8_t*>(e), reinterpret_cast<const uint8_t*>(p));
    }
    if (Backend* be = shadow::backendIfReady()) be->createDecl(DeclKey(*out), v.data(), uint32_t(v.size()));
  }
  return r;
}

// Draws: D3D9 draws first (shadow phase), then the backend draws the same primitive from the tracker.
thread_local int t_inDrawThunk = 0;  // lets the observer tell patched draw sites from unpatched ones
struct DrawThunkScope { DrawThunkScope() { t_inDrawThunk++; } ~DrawThunkScope() { t_inDrawThunk--; } };
HRESULT STDMETHODCALLTYPE tDrawPrimitive(IDirect3DDevice9* d, D3DPRIMITIVETYPE pt, UINT start, UINT prims) {
  DrawThunkScope scope;
  HRESULT r = g_forward && g_forwardDraws ? real<HRESULT(STDMETHODCALLTYPE*)(IDirect3DDevice9*, D3DPRIMITIVETYPE, UINT, UINT)>(d, 81)(d, pt, start, prims) : 0;
  observe::drawFromTracker(pt, prims, false, 0, start);
  return r;
}
HRESULT STDMETHODCALLTYPE tDrawIndexedPrimitive(IDirect3DDevice9* d, D3DPRIMITIVETYPE pt, INT base, UINT minIdx, UINT nv, UINT startIdx,
                                                UINT prims) {
  DrawThunkScope scope;
  HRESULT r = g_forward && g_forwardDraws ? real<HRESULT(STDMETHODCALLTYPE*)(IDirect3DDevice9*, D3DPRIMITIVETYPE, INT, UINT, UINT, UINT, UINT)>(d, 82)(
                              d, pt, base, minIdx, nv, startIdx, prims)
                        : 0;
  observe::drawFromTracker(pt, prims, true, base, startIdx);
  return r;
}
HRESULT STDMETHODCALLTYPE tDrawPrimitiveUP(IDirect3DDevice9* d, D3DPRIMITIVETYPE pt, UINT prims, const void* data, UINT stride) {
  DrawThunkScope scope;
  HRESULT r = g_forward && g_forwardDraws ? real<HRESULT(STDMETHODCALLTYPE*)(IDirect3DDevice9*, D3DPRIMITIVETYPE, UINT, const void*, UINT)>(d, 83)(d, pt, prims, data, stride)
                        : 0;
  if (data) observe::drawFromTracker(pt, prims, false, 0, 0, data, stride);
  // D3D9 sets stream 0 and the indices to null after a user-pointer draw.
  g_state.streams[0] = {};
  g_state.ib = nullptr;
  return r;
}

// ScreenShot::TakeSaveGameScreenShot reads the rendered thumbnail with GetRenderTargetData. Without D3D9 draws the
// D3D9 surface is empty: the backend's copy of that render target is written into the system-memory surface.
HRESULT STDMETHODCALLTYPE tGetRenderTargetData(IDirect3DDevice9* d, IDirect3DSurface9* src, IDirect3DSurface9* dst) {
  if (!g_forwardDraws && src && dst)
    if (Backend* be = shadow::backendIfReady())
      if (const uintptr_t key = observe::rtKeyForSurface(src)) {
        std::vector<uint8_t> texels;
        uint32_t vf = 0, ts = 0;
        D3DSURFACE_DESC sd{};
        dst->GetDesc(&sd);
        D3DLOCKED_RECT lr{};
        if (be->readback(key, texels, vf, ts) && ts && texels.size() >= size_t(sd.Width) * sd.Height * ts &&
            SUCCEEDED(dst->LockRect(&lr, nullptr, 0))) {
          for (uint32_t y = 0; y < sd.Height; y++)
            memcpy(static_cast<uint8_t*>(lr.pBits) + size_t(y) * lr.Pitch, texels.data() + size_t(y) * sd.Width * ts, size_t(sd.Width) * ts);
          dst->UnlockRect();
          return D3D_OK;
        }
      }
  if (!g_forwardDraws) return D3DERR_INVALIDCALL;  // native mode: the source is a Vulkan-backed object
  return real<HRESULT(STDMETHODCALLTYPE*)(IDirect3DDevice9*, IDirect3DSurface9*, IDirect3DSurface9*)>(d, 32)(d, src, dst);
}

// InterfaceManager::Click saves the declaration with GetVertexDeclaration and restores it later through the tracker:
// served from the tracker (D3D9 does not receive the state in native mode). D3D9 AddRefs what it returns.
HRESULT STDMETHODCALLTYPE tGetVertexDeclaration(IDirect3DDevice9*, IDirect3DVertexDeclaration9** out) {
  if (!out) return D3DERR_INVALIDCALL;
  auto* decl = static_cast<IDirect3DVertexDeclaration9*>(g_state.decl);
  if (decl) decl->AddRef();
  *out = decl;
  return D3D_OK;
}

struct Site { uint32_t addr, disp; void* thunk; const char* where; };
#define T(f) reinterpret_cast<void*>(&f)
const Site kSites[] = {
  { 0x00714600, 0x160, T(tGetVertexDeclaration), "InterfaceManager::Click" },
  { 0x008792D5, 0x80, T(tGetRenderTargetData), "ScreenShot::TakeSaveGameScreenShot" },
  { 0x0071462D, 0xE4, T(tSetRenderState), "InterfaceManager::Click" },
  { 0x00C5D1E0, 0xE4, T(tSetRenderState), "BSScissorTriShape::RenderImmediate" },
  { 0x00C5D251, 0xE4, T(tSetRenderState), "BSScissorTriShape::RenderImmediate" },
  { 0x00B7C1E0, 0x114, T(tSetSamplerState), "ShadowLightShader::SetLODNoiseMap" },
  { 0x00B9894B, 0x114, T(tSetSamplerState), "BSBatchRenderer::EndPass" },
  { 0x00BADD22, 0x114, T(tSetSamplerState), "ImageSpaceEffectHDR::Render" },
  { 0x00BADE19, 0x114, T(tSetSamplerState), "ImageSpaceEffectHDR::Render" },
  { 0x00E7EAE5, 0x114, T(tSetSamplerState), "NiD3DTextureStage::ApplyTexture" },
  { 0x00714210, 0xBC, T(tSetViewport), "InterfaceManager::Click" },
  { 0x007145C6, 0xBC, T(tSetViewport), "InterfaceManager::Click" },
  { 0x00E6CEED, 0xBC, T(tSetViewport), "NiDX9Renderer::Do_SetCameraData" },
  { 0x00E6D2DB, 0xBC, T(tSetViewport), "NiDX9Renderer::Do_SetScreenSpaceCameraData" },
  { 0x00E6F496, 0xBC, T(tSetViewport), "NiDX9Renderer::Do_ClearBuffer" },
  { 0x00EC1329, 0xBC, T(tSetViewport), "Movie::Update" },
  { 0x00EC14A2, 0xBC, T(tSetViewport), "Movie::Update" },
  { 0x004ED1DC, 0xDC, T(tSetClipPlane), "NiDX9Renderer::SetClippingPlanes" },
  { 0x00B7C018, 0xDC, T(tSetClipPlane), "ShadowLightShader::SetupGeometryConstants_SimpleShadowClipPlanes" },
  { 0x00B7C077, 0xDC, T(tSetClipPlane), "ShadowLightShader::SetupGeometryConstants_SimpleShadowClipPlanes" },
  { 0x00BBF438, 0xDC, T(tSetClipPlane), "Lighting30Shader::SetupGeometryConstants" },
  { 0x00BBF4A6, 0xDC, T(tSetClipPlane), "Lighting30Shader::SetupGeometryConstants" },
  { 0x00B97D59, 0x12C, T(tSetScissorRect), "BSRenderState::SetScissorRect" },
  { 0x00C5D20D, 0x12C, T(tSetScissorRect), "BSScissorTriShape::RenderImmediate" },
  { 0x007146D8, 0x170, T(tSetVertexShader), "InterfaceManager::Click" },
  { 0x007146AD, 0x1AC, T(tSetPixelShader), "InterfaceManager::Click" },
  { 0x00714708, 0x15C, T(tSetVertexDeclaration), "InterfaceManager::Click" },
  { 0x00876948, 0x190, T(tSetStreamSource), "TESMain::DrawWorld_End" },
  { 0x00E7069C, 0x190, T(tSetStreamSource), "NiDX9Renderer::Do_RenderPoints" },
  { 0x00E70B42, 0x190, T(tSetStreamSource), "NiDX9Renderer::Do_RenderParticleStrips" },
  { 0x00E71157, 0x190, T(tSetStreamSource), "NiDX9Renderer::Do_RenderLines" },
  { 0x00E813E9, 0x190, T(tSetStreamSource), "NiD3DShader::PrepareGeometryForRendering" },
  { 0x0087692F, 0x1A0, T(tSetIndices), "TESMain::DrawWorld_End" },
  { 0x00E706BF, 0x1A0, T(tSetIndices), "NiDX9Renderer::Do_RenderPoints" },
  { 0x00E70B5C, 0x1A0, T(tSetIndices), "NiDX9Renderer::Do_RenderParticleStrips" },
  { 0x00E7117B, 0x1A0, T(tSetIndices), "NiDX9Renderer::Do_RenderLines" },
  { 0x00E81406, 0x1A0, T(tSetIndices), "NiD3DShader::PrepareGeometryForRendering" },
  { 0x00B7B6B5, 0x178, T(tSetVSConstF), "ShadowLightShader::SetupPassConstants" },
  { 0x00B7C4F9, 0x178, T(tSetVSConstF), "ShadowLightShader::SetupPassOpt_FogConstants" },
  { 0x00B7C79C, 0x178, T(tSetVSConstF), "ShadowLightShader::SetupGeometryOpt_WorldViewProj" },
  { 0x00B7C835, 0x178, T(tSetVSConstF), "BSShader::SetLightDirection" },
  { 0x00B7C89E, 0x178, T(tSetVSConstF), "ShadowLightShader::SetupGeometryOpt_Lights" },
  { 0x00B7C9E2, 0x178, T(tSetVSConstF), "ShadowLightShader::SetupGeometryOpt_Lights" },
  { 0x00B7CB4E, 0x178, T(tSetVSConstF), "ShadowLightShader::SetupGeometryOpt_LightsSpecular" },
  { 0x00B7CCB5, 0x178, T(tSetVSConstF), "ShadowLightShader::SetupGeometryOpt_LightsSpecular" },
  { 0x00B7E0FD, 0x178, T(tSetVSConstF), "ShadowLightShader::Setup_ADTS10_Opt" },
  { 0x00B7E28B, 0x178, T(tSetVSConstF), "ShadowLightShader::Setup_ADT_PxOpt" },
  { 0x00E827FE, 0x178, T(tSetVSConstF), "NiD3DShaderConstantMap::SetShaderConstants" },
  { 0x00E89C4D, 0x178, T(tSetVSConstF), "NiD3DVertexShader::SetShaderConstant" },
  { 0x00E89D12, 0x178, T(tSetVSConstF), "NiD3DVertexShader::SetShaderConstantArray" },
  { 0x00E974C2, 0x178, T(tSetVSConstF), "NiD3DHLSLVertexShader::SetShaderConstant" },
  { 0x00E976AF, 0x178, T(tSetVSConstF), "NiD3DHLSLVertexShader::SetShaderConstantArray" },
  { 0x00E97755, 0x178, T(tSetVSConstF), "NiD3DHLSLVertexShader::SetShaderConstantArray" },
  { 0x00E82831, 0x1B4, T(tSetPSConstF), "NiD3DShaderConstantMap::SetShaderConstants" },
  { 0x00E89A1F, 0x1B4, T(tSetPSConstF), "NiD3DPixelShader::SetShaderConstant" },
  { 0x00E89AA5, 0x1B4, T(tSetPSConstF), "NiD3DPixelShader::SetShaderConstantArray" },
  { 0x00E96F82, 0x1B4, T(tSetPSConstF), "NiD3DHLSLPixelShader::SetShaderConstant" },
  { 0x00E89C36, 0x180, T(tSetVSConstI), "NiD3DVertexShader::SetShaderConstant" },
  { 0x00E89A09, 0x1BC, T(tSetPSConstI), "NiD3DPixelShader::SetShaderConstant" },
  { 0x00E89C1A, 0x188, T(tSetVSConstB), "NiD3DVertexShader::SetShaderConstant" },
  { 0x00E899EC, 0x1C4, T(tSetPSConstB), "NiD3DPixelShader::SetShaderConstant" },
  { 0x00E6976A, 0x158, T(tCreateVertexDeclaration), "NiDX9Renderer::GetDecl_ParticlesSubTexture" },
  { 0x00E6987E, 0x158, T(tCreateVertexDeclaration), "NiDX9Renderer::GetDecl_Particles" },
  { 0x00E76C04, 0x158, T(tCreateVertexDeclaration), "NiDX9ShaderDeclaration::GetD3DDeclaration" },
  { 0x00E7181D, 0x144, T(tDrawPrimitive), "NiDX9Renderer::DrawPrimitive" },
  { 0x00E7481A, 0x144, T(tDrawPrimitive), "NiDX9Renderer::Do_OnlyRenderShape" },
  { 0x00E74A10, 0x144, T(tDrawPrimitive), "NiDX9Renderer::Do_OnlyRenderTristrips" },
  { 0x00E71447, 0x144, T(tDrawPrimitive), "NiDX9Renderer::Do_EndBatch" },
  { 0x00E698C1, 0x148, T(tDrawIndexedPrimitive), "NiDX9Renderer::DrawIndexedPrimitive_TRIANGLELIST" },
  { 0x00E698F1, 0x148, T(tDrawIndexedPrimitive), "NiDX9Renderer::DrawIndexedPrimitive_TRIANGLESTRIP" },
  { 0x00E6D36E, 0x148, T(tDrawIndexedPrimitive), "NiDX9Renderer::Do_DrawPartition" },
  { 0x00E706ED, 0x148, T(tDrawIndexedPrimitive), "NiDX9Renderer::Do_RenderPoints" },
  { 0x00E70B75, 0x148, T(tDrawIndexedPrimitive), "NiDX9Renderer::Do_RenderParticleStrips" },
  { 0x00E711AE, 0x148, T(tDrawIndexedPrimitive), "NiDX9Renderer::Do_RenderLines" },
  { 0x00E71403, 0x148, T(tDrawIndexedPrimitive), "NiDX9Renderer::Do_EndBatch" },
  { 0x00E7178D, 0x148, T(tDrawIndexedPrimitive), "NiDX9Renderer::DrawPrimitive" },
  { 0x00E717DC, 0x148, T(tDrawIndexedPrimitive), "NiDX9Renderer::DrawPrimitive" },
  { 0x00E71AE3, 0x148, T(tDrawIndexedPrimitive), "NiDX9Renderer::DrawSkinnedPrimitive" },
  { 0x00E71BD7, 0x148, T(tDrawIndexedPrimitive), "NiDX9Renderer::DrawSkinnedPrimitive" },
  { 0x00E746A5, 0x148, T(tDrawIndexedPrimitive), "NiDX9Renderer::Do_OnlyRenderShape" },
  { 0x00E7479E, 0x148, T(tDrawIndexedPrimitive), "NiDX9Renderer::Do_OnlyRenderShape" },
  { 0x00E747DD, 0x148, T(tDrawIndexedPrimitive), "NiDX9Renderer::Do_OnlyRenderShape" },
  { 0x00E74975, 0x148, T(tDrawIndexedPrimitive), "NiDX9Renderer::Do_OnlyRenderTristrips" },
  { 0x00E749D2, 0x148, T(tDrawIndexedPrimitive), "NiDX9Renderer::Do_OnlyRenderTristrips" },
  { 0x00EC03ED, 0x14C, T(tDrawPrimitiveUP), "Bink Submit_vertices" },
};
#undef T
}  // namespace

namespace api {
long setRenderState(IDirect3DDevice9* d, uint32_t s, uint32_t v) { return tSetRenderState(d, D3DRENDERSTATETYPE(s), v); }
long setSamplerState(IDirect3DDevice9* d, uint32_t st, uint32_t s, uint32_t v) { return tSetSamplerState(d, st, D3DSAMPLERSTATETYPE(s), v); }
long setViewport(IDirect3DDevice9* d, const void* vp) { return tSetViewport(d, static_cast<const D3DVIEWPORT9*>(vp)); }
long setClipPlane(IDirect3DDevice9* d, uint32_t i, const float* p) { return tSetClipPlane(d, i, p); }
long setScissorRect(IDirect3DDevice9* d, const void* r) { return tSetScissorRect(d, static_cast<const RECT*>(r)); }
long setVertexShader(IDirect3DDevice9* d, void* vs) { return tSetVertexShader(d, static_cast<IDirect3DVertexShader9*>(vs)); }
long setPixelShader(IDirect3DDevice9* d, void* ps) { return tSetPixelShader(d, static_cast<IDirect3DPixelShader9*>(ps)); }
long setVertexDeclaration(IDirect3DDevice9* d, void* decl) { return tSetVertexDeclaration(d, static_cast<IDirect3DVertexDeclaration9*>(decl)); }
long setStreamSource(IDirect3DDevice9* d, uint32_t n, void* vb, uint32_t off, uint32_t stride) {
  return tSetStreamSource(d, n, static_cast<IDirect3DVertexBuffer9*>(vb), off, stride);
}
long setIndices(IDirect3DDevice9* d, void* ib) { return tSetIndices(d, static_cast<IDirect3DIndexBuffer9*>(ib)); }
long createVertexDeclaration(IDirect3DDevice9* d, const void* e, void** out) {
  return tCreateVertexDeclaration(d, static_cast<const D3DVERTEXELEMENT9*>(e), reinterpret_cast<IDirect3DVertexDeclaration9**>(out));
}
long drawPrimitive(IDirect3DDevice9* d, uint32_t pt, uint32_t start, uint32_t prims) { return tDrawPrimitive(d, D3DPRIMITIVETYPE(pt), start, prims); }
long drawIndexedPrimitive(IDirect3DDevice9* d, uint32_t pt, int32_t base, uint32_t minIdx, uint32_t nv, uint32_t startIdx, uint32_t prims) {
  return tDrawIndexedPrimitive(d, D3DPRIMITIVETYPE(pt), base, minIdx, nv, startIdx, prims);
}
long drawPrimitiveUP(IDirect3DDevice9* d, uint32_t pt, uint32_t prims, const void* data, uint32_t stride) {
  return tDrawPrimitiveUP(d, D3DPRIMITIVETYPE(pt), prims, data, stride);
}
long getRenderTargetData(IDirect3DDevice9* d, void* src, void* dst) {
  return tGetRenderTargetData(d, static_cast<IDirect3DSurface9*>(src), static_cast<IDirect3DSurface9*>(dst));
}
}  // namespace api

bool inDrawThunk() { return t_inDrawThunk > 0; }
bool inDeclThunk() { return t_inDeclThunk > 0; }

void flushDeclarations() {
  Backend* be = shadow::backendIfReady();
  if (!be) return;
  std::lock_guard<std::mutex> lock(g_declMutex);
  for (auto& [handle, bytes] : g_declElements) {
    std::vector<VertexElement> v;
    const auto* e = reinterpret_cast<const D3DVERTEXELEMENT9*>(bytes.data());
    for (size_t i = 0; i < bytes.size() / sizeof(D3DVERTEXELEMENT9); i++)
      v.push_back({ e[i].Stream, e[i].Offset, e[i].Type, e[i].Method, e[i].Usage, e[i].UsageIndex });
    be->createDecl(DeclKey(handle), v.data(), uint32_t(v.size()));
  }
  logf("game-side declarations registered at backend start: %zu", g_declElements.size());
}

bool declarationElements(void* handle, std::vector<uint8_t>& out) {
  std::lock_guard<std::mutex> lock(g_declMutex);
  auto it = g_declElements.find(handle);
  if (it == g_declElements.end()) return false;
  out = it->second;
  return true;
}

bool patchCallSites() {
  int patched = 0, refused = 0;
  for (const Site& s : kSites) {
    uint8_t* p = reinterpret_cast<uint8_t*>(uintptr_t(s.addr));
    uint32_t disp;
    memcpy(&disp, p + 2, 4);
    const uint8_t modrm = p[1];
    // Exactly `8B modrm disp32` (mod 10, no SIB) with the method's vtable offset.
    if (p[0] != 0x8B || (modrm & 0xC0) != 0x80 || (modrm & 7) == 4 || disp != s.disp) {
      refused++;
      logf("call-site patch REFUSED at %08X (%s): %02X %02X disp %X, expected mov r32,[r32+%X]", s.addr, s.where, p[0],
           p[1], disp, s.disp);
      continue;
    }
    uint8_t code[6] = { uint8_t(0xB8 + ((modrm >> 3) & 7)), 0, 0, 0, 0, 0x90 };
    const uint32_t target = uint32_t(uintptr_t(s.thunk));
    memcpy(code + 1, &target, 4);
    DWORD old = 0;
    VirtualProtect(p, 6, PAGE_EXECUTE_READWRITE, &old);
    memcpy(p, code, 6);
    VirtualProtect(p, 6, old, &old);
    FlushInstructionCache(GetCurrentProcess(), p, 6);
    patched++;
  }
  logf("device call sites routed through the state tracker: %d patched, %d refused", patched, refused);
  return refused == 0;
}

bool installHooks() {
  struct H { uintptr_t addr; void* fn; const char* name; };
  const H hooks[] = {
    { 0x00E88780, reinterpret_cast<void*>(&SetRenderState), "NiD3DRenderState::SetRenderState (port)" },
    { 0x00E887D0, reinterpret_cast<void*>(&SetDestAndSrcBlends), "NiD3DRenderState::SetDestAndSrcBlends (port)" },
    { 0x00E88830, reinterpret_cast<void*>(&SetAlphaBlendEnable), "NiD3DRenderState::SetAlphaBlendEnable (port)" },
    { 0x00E88000, reinterpret_cast<void*>(&InitRenderState), "NiD3DRenderState::InitRenderState (port)" },
    { 0x00E88060, reinterpret_cast<void*>(&InitTextureStageState), "NiD3DRenderState::InitTextureStageState (port)" },
    { 0x00E88930, reinterpret_cast<void*>(&SetTextureStageState), "NiD3DRenderState::SetTextureStageState (port)" },
    { 0x00E910A0, reinterpret_cast<void*>(&SetSamplerState), "NiDX9RenderState::SetSamplerState (port)" },
    { 0x00E88A20, reinterpret_cast<void*>(&SetTexture), "NiD3DRenderState::SetTexture (port)" },
    { 0x00E88A60, reinterpret_cast<void*>(&ClearTexture), "NiD3DRenderState::ClearTexture (port)" },
    { 0x00E88870, reinterpret_cast<void*>(&SetPixelShader), "NiD3DRenderState::SetPixelShader (port)" },
    { 0x00E888E0, reinterpret_cast<void*>(&ClearPixelShader), "NiD3DRenderState::ClearPixelShader (port)" },
    { 0x00E90E00, reinterpret_cast<void*>(&SetVertexShader), "NiDX9RenderState::SetVertexShader (port)" },
    { 0x00E90E70, reinterpret_cast<void*>(&ClearVertexShader), "NiDX9RenderState::ClearVertexShader (port)" },
    { 0x00E90EC0, reinterpret_cast<void*>(&SetFVF), "NiDX9RenderState::SetFVF (port)" },
    { 0x00E90F60, reinterpret_cast<void*>(&ClearFVF), "NiDX9RenderState::ClearFVF (port)" },
    { 0x00E90FC0, reinterpret_cast<void*>(&SetDeclaration), "NiDX9RenderState::SetDeclaration (port)" },
    { 0x00E89AE0, reinterpret_cast<void*>(&SetVSConstB), "NiD3DRenderState::SetVertexShaderConstantB (port)" },
    { 0x00E89B10, reinterpret_cast<void*>(&SetVSConstF), "NiD3DRenderState::SetVertexShaderConstantF (port)" },
    { 0x00E89B40, reinterpret_cast<void*>(&SetVSConstI), "NiD3DRenderState::SetVertexShaderConstantI (port)" },
    { 0x00E898E0, reinterpret_cast<void*>(&SetPSConstB), "NiDX9RenderState::SetPixelShaderConstantB (port)" },
    { 0x00E89910, reinterpret_cast<void*>(&SetPSConstF), "NiDX9RenderState::SetPixelShaderConstantF (port)" },
    { 0x00E89940, reinterpret_cast<void*>(&SetPSConstI), "NiDX9RenderState::SetPixelShaderConstantI (port)" },
  };
  bool ok = true;
  for (const H& h : hooks) {
    void* orig = nullptr;  // replaced outright: the original is never called
    ok &= hookFunction(h.addr, h.fn, &orig, h.name);
  }
  return ok;
}

}  // namespace fnvvk::gbstate
