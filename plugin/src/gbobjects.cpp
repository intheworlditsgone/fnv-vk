// Vulkan-backed resource objects with the engine's resource ABI (native mode).
//
// Gamebryo keeps IDirect3D*9 pointers in its own objects and calls their methods through the vtable at many sites
// (typed and untyped in IDA). Keeping that ABI with Vulkan-backed contents (FEASIBILITY.md, constraint 2) covers every
// site by construction: the objects below implement the interfaces' vtables and own their backend resource.
//
// Buffers: the five creation sites (IDA: the only CreateVertexBuffer / CreateIndexBuffer callers) are rewritten to
// call createVertexBuffer / createIndexBuffer here.
// - Lock hands out host memory: the whole-buffer shadow for buffers that are not WRITEONLY (the game may read them),
//   a per-lock staging block for WRITEONLY ones (D3D9 forbids reading those).
// - Unlock writes the locked range to the backend with the lock's DISCARD / NOOVERWRITE flags (the backend renames as
//   D3D9 would).
// - Objects created before the backend exists keep a full shadow and are registered and uploaded when it starts.
#include <windows.h>

#include <d3d9.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "backend.h"
#include "common.h"
#include "gbobjects.h"
#include "shadow.h"

namespace fnvvk::gbobjects {
namespace {

std::atomic<uint64_t> g_vbs{ 0 }, g_ibs{ 0 }, g_live{ 0 }, g_locks{ 0 }, g_bytes{ 0 }, g_flushed{ 0 };

template <typename Iface, typename Desc>
class Buffer;
std::mutex g_registryMutex;
std::unordered_set<void*> g_unregistered;  // created before the backend existed
std::vector<std::pair<void*, bool>> g_kinds;  // the same objects, with index-buffer flag, in creation order

template <typename Iface, typename Desc>
class Buffer final : public Iface {
 public:
  Buffer(IDirect3DDevice9* dev, const Desc& d, bool index, bool index32)
      : dev_(dev), desc_(d), index_(index), index32_(index32), shadowed_(!(d.Usage & D3DUSAGE_WRITEONLY)) {
    if (Backend* be = shadow::backendIfReady()) {
      be->createBuffer(BufKey(this), desc_.Size, index_, index32_);
    } else {
      shadowed_ = true;  // keeps the contents until the backend exists
      std::lock_guard<std::mutex> lock(g_registryMutex);
      g_unregistered.insert(this);
      g_kinds.push_back({ this, index });
    }
    if (shadowed_) shadow_.assign(desc_.Size, 0);
    g_live++;
  }

  // IUnknown
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == __uuidof(IUnknown) || riid == __uuidof(IDirect3DResource9) || riid == __uuidof(Iface)) {
      AddRef();
      *out = this;
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
  ULONG STDMETHODCALLTYPE Release() override {
    const ULONG r = --refs_;
    if (r == 0) {
      {
        std::lock_guard<std::mutex> lock(g_registryMutex);
        g_unregistered.erase(this);
      }
      if (Backend* be = shadow::backendIfReady()) be->destroyBuffer(BufKey(this));
      g_live--;
      delete this;
    }
    return r;
  }
  // IDirect3DResource9
  HRESULT STDMETHODCALLTYPE GetDevice(IDirect3DDevice9** out) override {
    if (!out) return D3DERR_INVALIDCALL;
    *out = dev_;
    if (dev_) dev_->AddRef();
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, const void*, DWORD, DWORD) override { return D3D_OK; }
  HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, void*, DWORD*) override { return D3DERR_NOTFOUND; }
  HRESULT STDMETHODCALLTYPE FreePrivateData(REFGUID) override { return D3D_OK; }
  DWORD STDMETHODCALLTYPE SetPriority(DWORD p) override { std::swap(p, priority_); return p; }
  DWORD STDMETHODCALLTYPE GetPriority() override { return priority_; }
  void STDMETHODCALLTYPE PreLoad() override {}
  D3DRESOURCETYPE STDMETHODCALLTYPE GetType() override { return index_ ? D3DRTYPE_INDEXBUFFER : D3DRTYPE_VERTEXBUFFER; }
  // IDirect3D{Vertex,Index}Buffer9
  HRESULT STDMETHODCALLTYPE Lock(UINT offset, UINT size, void** out, DWORD flags) override {
    if (!out) return D3DERR_INVALIDCALL;
    if (offset > desc_.Size) offset = desc_.Size;
    if (!size || offset + size > desc_.Size) size = desc_.Size - offset;  // 0 = to the end of the buffer
    std::lock_guard<std::mutex> lock(m_);
    lockOffset_ = offset;
    lockSize_ = size;
    lockFlags_ = flags;
    locked_ = true;
    if (shadowed_) {
      *out = shadow_.data() + offset;
    } else {
      staging_.resize(size);
      *out = staging_.data();
    }
    g_locks++;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE Unlock() override {
    std::lock_guard<std::mutex> lock(m_);
    if (!locked_) return D3D_OK;
    locked_ = false;
    if (lockFlags_ & D3DLOCK_READONLY) return D3D_OK;
    const uint8_t* src = shadowed_ ? shadow_.data() + lockOffset_ : staging_.data();
    if (Backend* be = shadow::backendIfReady()) {
      bool registered = true;
      {
        std::lock_guard<std::mutex> rl(g_registryMutex);
        registered = !g_unregistered.count(this);
      }
      if (registered && lockSize_) {
        be->writeBuffer(BufKey(this), lockOffset_, src, lockSize_, lockFlags_);
        g_bytes += lockSize_;
      }
    }
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetDesc(Desc* out) override {
    if (!out) return D3DERR_INVALIDCALL;
    *out = desc_;
    return D3D_OK;
  }

  // Registration once the backend exists (objects created before it).
  void registerWithBackend(Backend* be) {
    std::lock_guard<std::mutex> lock(m_);
    be->createBuffer(BufKey(this), desc_.Size, index_, index32_);
    if (!shadow_.empty()) be->writeBuffer(BufKey(this), 0, shadow_.data(), desc_.Size, D3DLOCK_DISCARD);
  }

 private:
  ~Buffer() = default;
  std::atomic<ULONG> refs_{ 1 };
  IDirect3DDevice9* dev_;
  Desc desc_;
  bool index_, index32_, shadowed_;
  std::mutex m_;
  std::vector<uint8_t> shadow_, staging_;
  UINT lockOffset_ = 0, lockSize_ = 0;
  DWORD lockFlags_ = 0, priority_ = 0;
  bool locked_ = false;
};
using VertexBuffer = Buffer<IDirect3DVertexBuffer9, D3DVERTEXBUFFER_DESC>;
using IndexBuffer = Buffer<IDirect3DIndexBuffer9, D3DINDEXBUFFER_DESC>;


// --- creation thunks (stdcall, the device methods' signatures) ----------------------------------------------------
HRESULT STDMETHODCALLTYPE createVertexBufferThunk(IDirect3DDevice9* d, UINT len, DWORD usage, DWORD fvf, D3DPOOL pool,
                                             IDirect3DVertexBuffer9** out, HANDLE*) {
  if (!out || !len) return D3DERR_INVALIDCALL;
  D3DVERTEXBUFFER_DESC desc{ D3DFMT_VERTEXDATA, D3DRTYPE_VERTEXBUFFER, usage, pool, len, fvf };
  auto* b = new VertexBuffer(d, desc, false, false);
  *out = b;
  g_vbs++;
  return D3D_OK;
}
HRESULT STDMETHODCALLTYPE createIndexBufferThunk(IDirect3DDevice9* d, UINT len, DWORD usage, D3DFORMAT fmt, D3DPOOL pool,
                                            IDirect3DIndexBuffer9** out, HANDLE*) {
  if (!out || !len) return D3DERR_INVALIDCALL;
  D3DINDEXBUFFER_DESC desc{ fmt, D3DRTYPE_INDEXBUFFER, usage, pool, len };
  auto* b = new IndexBuffer(d, desc, true, fmt == D3DFMT_INDEX32);
  *out = b;
  g_ibs++;
  return D3D_OK;
}


// --- shader programs and vertex declarations ------------------------------------------------------------------
// The handles Gamebryo stores (NiD3DVertexShader +0x34, NiD3DPixelShader +0x2C, NiDX9ShaderDeclaration's declaration).
// gbshaders.cpp registers each shader handle with the backend from its bytecode (its creation hooks run after these
// objects are returned); declarations are registered by gbstate's creation thunk. Release destroys the backend side.
std::atomic<uint64_t> g_vss{ 0 }, g_pss{ 0 }, g_decls{ 0 };

using GetShaderSizeFn = UINT(WINAPI*)(const DWORD*);
UINT shaderSize(const DWORD* code) {
  static GetShaderSizeFn fn = [] {
    HMODULE dx = GetModuleHandleA("d3dx9_38.dll");
    return dx ? reinterpret_cast<GetShaderSizeFn>(GetProcAddress(dx, "D3DXGetShaderSize")) : nullptr;
  }();
  return fn && code ? fn(code) : 0;
}

template <typename Iface>
class Shader final : public Iface {
 public:
  Shader(IDirect3DDevice9* dev, const DWORD* code) : dev_(dev) {
    const UINT size = shaderSize(code);
    if (size) code_.assign(reinterpret_cast<const uint8_t*>(code), reinterpret_cast<const uint8_t*>(code) + size);
  }
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == __uuidof(IUnknown) || riid == __uuidof(Iface)) {
      AddRef();
      *out = this;
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
  ULONG STDMETHODCALLTYPE Release() override {
    const ULONG r = --refs_;
    if (r == 0) {
      if (Backend* be = shadow::backendIfReady()) be->destroyShader(ShaderKey(this));
      delete this;
    }
    return r;
  }
  HRESULT STDMETHODCALLTYPE GetDevice(IDirect3DDevice9** out) override {
    if (!out) return D3DERR_INVALIDCALL;
    *out = dev_;
    if (dev_) dev_->AddRef();
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetFunction(void* data, UINT* size) override {
    if (!size) return D3DERR_INVALIDCALL;
    if (!data) {
      *size = UINT(code_.size());
      return D3D_OK;
    }
    if (*size < code_.size()) return D3DERR_INVALIDCALL;
    memcpy(data, code_.data(), code_.size());
    *size = UINT(code_.size());
    return D3D_OK;
  }

 private:
  ~Shader() = default;
  std::atomic<ULONG> refs_{ 1 };
  IDirect3DDevice9* dev_;
  std::vector<uint8_t> code_;
};

class Declaration final : public IDirect3DVertexDeclaration9 {
 public:
  Declaration(IDirect3DDevice9* dev, const D3DVERTEXELEMENT9* e) : dev_(dev) {
    const D3DVERTEXELEMENT9* p = e;
    while (p->Stream != 0xFF && elements_.size() < 64) elements_.push_back(*p++);
    elements_.push_back(D3DVERTEXELEMENT9{ 0xFF, 0, D3DDECLTYPE_UNUSED, 0, 0, 0 });
  }
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == __uuidof(IUnknown) || riid == __uuidof(IDirect3DVertexDeclaration9)) {
      AddRef();
      *out = this;
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
  ULONG STDMETHODCALLTYPE Release() override {
    const ULONG r = --refs_;
    if (r == 0) {
      if (Backend* be = shadow::backendIfReady()) be->destroyDecl(DeclKey(this));
      delete this;
    }
    return r;
  }
  HRESULT STDMETHODCALLTYPE GetDevice(IDirect3DDevice9** out) override {
    if (!out) return D3DERR_INVALIDCALL;
    *out = dev_;
    if (dev_) dev_->AddRef();
    return D3D_OK;
  }
  // D3D9: the count includes the END element.
  HRESULT STDMETHODCALLTYPE GetDeclaration(D3DVERTEXELEMENT9* out, UINT* count) override {
    if (!count) return D3DERR_INVALIDCALL;
    if (out) memcpy(out, elements_.data(), elements_.size() * sizeof(D3DVERTEXELEMENT9));
    *count = UINT(elements_.size());
    return D3D_OK;
  }

 private:
  ~Declaration() = default;
  std::atomic<ULONG> refs_{ 1 };
  IDirect3DDevice9* dev_;
  std::vector<D3DVERTEXELEMENT9> elements_;
};

HRESULT STDMETHODCALLTYPE createVertexShaderThunk(IDirect3DDevice9* d, const DWORD* code, IDirect3DVertexShader9** out);
HRESULT STDMETHODCALLTYPE createPixelShaderThunk(IDirect3DDevice9* d, const DWORD* code, IDirect3DPixelShader9** out);

}  // namespace

// Rewrites `mov r32,[r32+disp]` (disp8 or disp32; the COM method load from a vtable) followed by pushes and other
// instructions that do not use that register, then `call r32`, into the same instructions followed by
// `call rel32 target` (padded with NOPs to the original length). The thunk has the method's stdcall signature.
bool patchVirtualCall(uint32_t site, uint32_t disp, void* target, const char* where) {
  uint8_t* p = reinterpret_cast<uint8_t*>(uintptr_t(site));
  const uint8_t modrm = p[1];
  const uint8_t mod = modrm >> 6, reg = (modrm >> 3) & 7;
  uint32_t got = 0;
  int movLen = 0;
  if (p[0] == 0x8B && mod == 1 && (modrm & 7) != 4) { got = p[2]; movLen = 3; }
  if (p[0] == 0x8B && mod == 2 && (modrm & 7) != 4) { memcpy(&got, p + 2, 4); movLen = 6; }
  if (!movLen || got != disp) {
    logf("object patch REFUSED at %08X (%s): %02X %02X, disp %X expected %X", site, where, p[0], p[1], got, disp);
    return false;
  }
  int callAt = -1;
  for (int i = movLen; i < 40; i++)
    if (p[i] == 0xFF && p[i + 1] == uint8_t(0xD0 + reg)) {
      callAt = i;
      break;
    }
  if (callAt < 0) {
    logf("object patch REFUSED at %08X (%s): no call through the loaded register", site, where);
    return false;
  }
  const int total = callAt + 2, between = callAt - movLen;
  uint8_t code[48];
  memcpy(code, p + movLen, between);
  code[between] = 0xE8;
  const int32_t rel = int32_t(uintptr_t(target) - (uintptr_t(p) + between + 5));
  memcpy(code + between + 1, &rel, 4);
  for (int i = between + 5; i < total; i++) code[i] = 0x90;
  DWORD old = 0;
  if (!VirtualProtect(p, total, PAGE_EXECUTE_READWRITE, &old)) return false;
  memcpy(p, code, total);
  VirtualProtect(p, total, old, &old);
  FlushInstructionCache(GetCurrentProcess(), p, total);
  return true;
}

bool installBuffers() {
  struct S { uint32_t site; uint8_t disp; void* fn; const char* where; };
  const S sites[] = {
    { 0x00E945BD, 0x68, reinterpret_cast<void*>(&createVertexBufferThunk), "NiVBSet::CreateBlock" },
    { 0x00E94E87, 0x68, reinterpret_cast<void*>(&createVertexBufferThunk), "NiUnsharedGeometryGroup::CreateChip" },
    { 0x00E98813, 0x68, reinterpret_cast<void*>(&createVertexBufferThunk), "NiVBDynamicSet::AllocateBufferSpace" },
    { 0x00E8F1BA, 0x6C, reinterpret_cast<void*>(&createIndexBufferThunk), "NiDX9IndexBufferManager::CreateIndexBuffer" },
    { 0x00E8F2A2, 0x6C, reinterpret_cast<void*>(&createIndexBufferThunk), "NiDX9IndexBufferManager::PackBuffer" },
  };
  int ok = 0;
  for (const S& s : sites) ok += patchVirtualCall(s.site, s.disp, s.fn, s.where);
  logf("native buffers: %d of %zu creation sites create Vulkan-backed buffer objects", ok, std::size(sites));
  return ok == int(std::size(sites));
}

void onBackendReady(Backend* be) {
  std::vector<std::pair<void*, bool>> kinds;
  {
    std::lock_guard<std::mutex> lock(g_registryMutex);
    kinds.swap(g_kinds);
  }
  for (auto& [obj, index] : kinds) {
    bool pending;
    {
      std::lock_guard<std::mutex> lock(g_registryMutex);
      pending = g_unregistered.erase(obj) != 0;
    }
    if (!pending) continue;  // released meanwhile
    if (index) static_cast<IndexBuffer*>(obj)->registerWithBackend(be);
    else static_cast<VertexBuffer*>(obj)->registerWithBackend(be);
    g_flushed++;
  }
}

HRESULT createVertexShader(IDirect3DDevice9* d, const DWORD* code, void** out) {
  if (!out || !code) return D3DERR_INVALIDCALL;
  *out = static_cast<IDirect3DVertexShader9*>(new Shader<IDirect3DVertexShader9>(d, code));
  g_vss++;
  return D3D_OK;
}
HRESULT createPixelShader(IDirect3DDevice9* d, const DWORD* code, void** out) {
  if (!out || !code) return D3DERR_INVALIDCALL;
  *out = static_cast<IDirect3DPixelShader9*>(new Shader<IDirect3DPixelShader9>(d, code));
  g_pss++;
  return D3D_OK;
}
HRESULT createVertexDeclaration(IDirect3DDevice9* d, const void* elements, void** out) {
  if (!out || !elements) return D3DERR_INVALIDCALL;
  *out = static_cast<IDirect3DVertexDeclaration9*>(new Declaration(d, static_cast<const D3DVERTEXELEMENT9*>(elements)));
  g_decls++;
  return D3D_OK;
}
HRESULT createVertexBuffer(IDirect3DDevice9* d, uint32_t len, uint32_t usage, uint32_t fvf, uint32_t pool, void** out) {
  return createVertexBufferThunk(d, len, usage, fvf, D3DPOOL(pool), reinterpret_cast<IDirect3DVertexBuffer9**>(out), nullptr);
}
HRESULT createIndexBuffer(IDirect3DDevice9* d, uint32_t len, uint32_t usage, uint32_t fmt, uint32_t pool, void** out) {
  return createIndexBufferThunk(d, len, usage, D3DFORMAT(fmt), D3DPOOL(pool), reinterpret_cast<IDirect3DIndexBuffer9**>(out), nullptr);
}
namespace {
HRESULT STDMETHODCALLTYPE createVertexShaderThunk(IDirect3DDevice9* d, const DWORD* code, IDirect3DVertexShader9** out) {
  return createVertexShader(d, code, reinterpret_cast<void**>(out));
}
HRESULT STDMETHODCALLTYPE createPixelShaderThunk(IDirect3DDevice9* d, const DWORD* code, IDirect3DPixelShader9** out) {
  return createPixelShader(d, code, reinterpret_cast<void**>(out));
}
}  // namespace

// Shader creation: BSShader::CompileVertexShaderHLSL / CompilePixelShaderHLSL and NiD3DShaderProgramCreator's
// CreateVertexShader / CreatePixelShader (IDA: every caller of the device's Create*Shader).
bool installPrograms() {
  struct S { uint32_t site, disp; void* fn; const char* where; };
  const S sites[] = {
    { 0x00BE1670, 0x16C, reinterpret_cast<void*>(&createVertexShaderThunk), "BSShader::CompileVertexShaderHLSL" },
    { 0x00E988A2, 0x16C, reinterpret_cast<void*>(&createVertexShaderThunk), "NiD3DShaderProgramCreator::CreateVertexShader" },
    { 0x00BE1DDB, 0x1A8, reinterpret_cast<void*>(&createPixelShaderThunk), "BSShader::CompilePixelShaderHLSL" },
    { 0x00E988E2, 0x1A8, reinterpret_cast<void*>(&createPixelShaderThunk), "NiD3DShaderProgramCreator::CreatePixelShader" },
  };
  int ok = 0;
  for (const S& s : sites) ok += patchVirtualCall(s.site, s.disp, s.fn, s.where);
  logf("native shaders: %d of %zu creation sites create shader objects", ok, std::size(sites));
  return ok == int(std::size(sites));
}

void logSummary() {
  logf("native buffers: VB %llu IB %llu created, %llu live | locks %llu, %llu bytes uploaded | %llu registered late",
       g_vbs.load(), g_ibs.load(), g_live.load(), g_locks.load(), g_bytes.load(), g_flushed.load());
  logf("native programs: vertex shaders %llu, pixel shaders %llu, declarations %llu created", g_vss.load(), g_pss.load(),
       g_decls.load());
}

}  // namespace fnvvk::gbobjects
