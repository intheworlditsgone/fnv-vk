// Vulkan-backed textures and surfaces with the engine's IDirect3D*9 ABI (native mode).
//
// Contents live in the backend, keyed by the Gamebryo objects (texture renderer data, 2D buffer data); the observe
// maps take these objects' pointers to those keys. The objects here provide what the engine reads through
// the interfaces: descriptions, level counts, surfaces of levels and cube faces (stable pointers: Gamebryo stores and
// registers them), reference counts, and host memory for the textures it locks (Bink tiles, canopy shadow masks,
// pixel-data uploads, the FaceGen conversion target).
//
// Sites (IDA, every typed caller):
// - creation: CreateTexture x6, CreateCubeTexture x2, CreateRenderTarget x1, CreateDepthStencilSurface x2;
// - consumers that the backend already mirrors and that become no-ops: SetRenderTarget x3, SetDepthStencilSurface x2,
//   Clear x2 (Do_ClearBuffer is mirrored by its hook, Movie::Update by movie.cpp), StretchRect x3 (Copy/FastCopy are
//   mirrored by their hooks, Do_EndUsingRenderTargetGroup's resolve by the backend's group end);
// - BSShaderUtil::ConvertTexture's D3DXLoadSurfaceFromSurface (FaceGen): the source level is read back from the backend
//   and converted with the D3DX-exact DXT codec.
#include <windows.h>

#include <d3d9.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <vector>

#include "backend.h"
#include "common.h"
#include "d3dxtex/dxt.h"
#include "gbobjects.h"
#include "gbsurfaces.h"
#include "observe.h"
#include "shadow.h"

namespace fnvvk::gbsurfaces {
namespace {

std::atomic<uint64_t> g_textures{ 0 }, g_cubes{ 0 }, g_surfaces{ 0 }, g_live{ 0 }, g_converted{ 0 }, g_convertFailed{ 0 };
IDirect3DDevice9* g_device = nullptr;

bool isDxt(D3DFORMAT f) { return f == D3DFMT_DXT1 || f == D3DFMT_DXT2 || f == D3DFMT_DXT3 || f == D3DFMT_DXT4 || f == D3DFMT_DXT5; }
uint32_t bytesPerTexel(D3DFORMAT f) {
  switch (f) {
    case D3DFMT_L8: case D3DFMT_A8: case D3DFMT_P8: return 1;
    case D3DFMT_R5G6B5: case D3DFMT_X1R5G5B5: case D3DFMT_A1R5G5B5: case D3DFMT_A4R4G4B4: case D3DFMT_L16:
    case D3DFMT_R16F: case D3DFMT_A8L8: case D3DFMT_V8U8: case D3DFMT_D16: return 2;
    case D3DFMT_A16B16G16R16F: case D3DFMT_A16B16G16R16: case D3DFMT_G32R32F: return 8;
    case D3DFMT_A32B32G32R32F: return 16;
    default: return 4;
  }
}
// D3D9 pitch of a level: bytes per texel row, or per row of 4x4 blocks for DXT.
uint32_t pitchOf(D3DFORMAT f, uint32_t w) {
  if (isDxt(f)) return std::max(1u, (w + 3) / 4) * (f == D3DFMT_DXT1 ? 8 : 16);
  return w * bytesPerTexel(f);
}
uint32_t rowsOf(D3DFORMAT f, uint32_t h) { return isDxt(f) ? std::max(1u, (h + 3) / 4) : h; }

// Refcounted base with the IDirect3DResource9 part.
template <typename Iface>
class Resource : public Iface {
 public:
  explicit Resource(IDirect3DDevice9* dev) : dev_(dev) { g_live++; }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
  ULONG STDMETHODCALLTYPE Release() override {
    const ULONG r = --refs_;
    if (r == 0) {
      g_live--;
      destroy();
    }
    return r;
  }
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

 protected:
  virtual void destroy() = 0;
  virtual ~Resource() = default;
  std::atomic<ULONG> refs_{ 1 };
  IDirect3DDevice9* dev_;
  DWORD priority_ = 0;
};

class Surface;
// Host memory of one level / face, allocated on the first lock and kept (later locks see earlier contents).
struct LevelMemory {
  std::vector<uint8_t> bytes;
  bool locked = false;
};

// A surface: standalone (render target, depth-stencil) or a level/face of a texture (reference counting goes to the
// container, as in D3D9).
class Surface final : public Resource<IDirect3DSurface9> {
 public:
  Surface(IDirect3DDevice9* dev, const D3DSURFACE_DESC& d, IUnknown* container, LevelMemory* memory)
      : Resource(dev), desc_(d), container_(container), memory_(memory ? memory : &own_) {
    g_surfaces++;
  }
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == __uuidof(IUnknown) || riid == __uuidof(IDirect3DResource9) || riid == __uuidof(IDirect3DSurface9)) {
      AddRef();
      *out = this;
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  ULONG STDMETHODCALLTYPE AddRef() override { return container_ ? container_->AddRef() : Resource::AddRef(); }
  ULONG STDMETHODCALLTYPE Release() override { return container_ ? container_->Release() : Resource::Release(); }
  D3DRESOURCETYPE STDMETHODCALLTYPE GetType() override { return D3DRTYPE_SURFACE; }
  HRESULT STDMETHODCALLTYPE GetContainer(REFIID riid, void** out) override {
    if (!out) return D3DERR_INVALIDCALL;
    if (container_) return container_->QueryInterface(riid, out);
    return dev_ ? dev_->QueryInterface(riid, out) : E_NOINTERFACE;
  }
  HRESULT STDMETHODCALLTYPE GetDesc(D3DSURFACE_DESC* out) override {
    if (!out) return D3DERR_INVALIDCALL;
    *out = desc_;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE LockRect(D3DLOCKED_RECT* lr, const RECT* r, DWORD) override {
    if (!lr) return D3DERR_INVALIDCALL;
    std::lock_guard<std::mutex> lock(m_);
    const D3DFORMAT f = desc_.Format;
    const uint32_t pitch = pitchOf(f, desc_.Width);
    if (memory_->bytes.empty()) memory_->bytes.assign(size_t(pitch) * rowsOf(f, desc_.Height), 0);
    size_t off = 0;
    if (r) {
      off = isDxt(f) ? size_t(r->top / 4) * pitch + size_t(r->left / 4) * (f == D3DFMT_DXT1 ? 8 : 16)
                     : size_t(r->top) * pitch + size_t(r->left) * bytesPerTexel(f);
    }
    lr->Pitch = INT(pitch);
    lr->pBits = memory_->bytes.data() + off;
    memory_->locked = true;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE UnlockRect() override {
    memory_->locked = false;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetDC(HDC*) override { return D3DERR_INVALIDCALL; }
  HRESULT STDMETHODCALLTYPE ReleaseDC(HDC) override { return D3DERR_INVALIDCALL; }

  const D3DSURFACE_DESC& desc() const { return desc_; }
  LevelMemory* memory() { return memory_; }
  IUnknown* container() const { return container_; }
  uint32_t level = 0, face = 0;  // within the container

 protected:
  void destroy() override { delete this; }

 private:
  D3DSURFACE_DESC desc_;
  IUnknown* container_;
  LevelMemory own_;
  LevelMemory* memory_;
  std::mutex m_;
};

// Shared part of 2D and cube textures.
struct Levels {
  D3DFORMAT format;
  DWORD usage;
  D3DPOOL pool;
  uint32_t width, height, levels, faces;
  std::vector<LevelMemory> memory;   // faces * levels
  std::vector<Surface*> surfaces;    // created on first request, owned (deleted with the texture)

  D3DSURFACE_DESC desc(uint32_t level) const {
    D3DSURFACE_DESC d{};
    d.Format = format;
    d.Type = D3DRTYPE_SURFACE;
    d.Usage = usage;
    d.Pool = pool;
    d.MultiSampleType = D3DMULTISAMPLE_NONE;
    d.Width = std::max(1u, width >> level);
    d.Height = std::max(1u, height >> level);
    return d;
  }
  Surface* surface(IDirect3DDevice9* dev, IUnknown* container, uint32_t face, uint32_t level) {
    const size_t i = size_t(face) * levels + level;
    if (!surfaces[i]) {
      surfaces[i] = new Surface(dev, desc(level), container, &memory[i]);
      surfaces[i]->level = level;
      surfaces[i]->face = face;
    }
    return surfaces[i];
  }
  void release() {
    for (Surface* s : surfaces)
      if (s) delete s;  // never handed out without the container's reference
  }
};

template <typename Iface>
class TextureBase : public Resource<Iface> {
 public:
  TextureBase(IDirect3DDevice9* dev, uint32_t w, uint32_t h, uint32_t levels, uint32_t faces, DWORD usage, D3DFORMAT f,
              D3DPOOL pool)
      : Resource<Iface>(dev) {
    if (!levels) {  // 0 = full chain
      levels = 1;
      for (uint32_t s = std::max(w, h); s > 1; s >>= 1) levels++;
    }
    l_ = Levels{ f, usage, pool, w, h, levels, faces, {}, {} };
    l_.memory.resize(size_t(faces) * levels);
    l_.surfaces.assign(size_t(faces) * levels, nullptr);
  }
  DWORD STDMETHODCALLTYPE SetLOD(DWORD lod) override { std::swap(lod, lod_); return lod; }
  DWORD STDMETHODCALLTYPE GetLOD() override { return lod_; }
  DWORD STDMETHODCALLTYPE GetLevelCount() override { return l_.levels; }
  HRESULT STDMETHODCALLTYPE SetAutoGenFilterType(D3DTEXTUREFILTERTYPE t) override { filter_ = t; return D3D_OK; }
  D3DTEXTUREFILTERTYPE STDMETHODCALLTYPE GetAutoGenFilterType() override { return filter_; }
  void STDMETHODCALLTYPE GenerateMipSubLevels() override {}
  HRESULT STDMETHODCALLTYPE GetLevelDesc(UINT level, D3DSURFACE_DESC* out) override {
    if (!out || level >= l_.levels) return D3DERR_INVALIDCALL;
    *out = l_.desc(level);
    return D3D_OK;
  }
  const Levels& levels() const { return l_; }

 protected:
  void destroy() override {
    l_.release();
    delete this;
  }
  Levels l_;
  DWORD lod_ = 0;
  D3DTEXTUREFILTERTYPE filter_ = D3DTEXF_LINEAR;
};

class Texture final : public TextureBase<IDirect3DTexture9> {
 public:
  using TextureBase::TextureBase;
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == __uuidof(IUnknown) || riid == __uuidof(IDirect3DResource9) || riid == __uuidof(IDirect3DBaseTexture9) ||
        riid == __uuidof(IDirect3DTexture9)) {
      AddRef();
      *out = this;
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  D3DRESOURCETYPE STDMETHODCALLTYPE GetType() override { return D3DRTYPE_TEXTURE; }
  HRESULT STDMETHODCALLTYPE GetSurfaceLevel(UINT level, IDirect3DSurface9** out) override {
    if (!out || level >= l_.levels) return D3DERR_INVALIDCALL;
    *out = l_.surface(dev_, this, 0, level);
    AddRef();
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE LockRect(UINT level, D3DLOCKED_RECT* lr, const RECT* r, DWORD flags) override {
    if (level >= l_.levels) return D3DERR_INVALIDCALL;
    return l_.surface(dev_, this, 0, level)->LockRect(lr, r, flags);
  }
  HRESULT STDMETHODCALLTYPE UnlockRect(UINT level) override {
    if (level >= l_.levels) return D3DERR_INVALIDCALL;
    return l_.surface(dev_, this, 0, level)->UnlockRect();
  }
  HRESULT STDMETHODCALLTYPE AddDirtyRect(const RECT*) override { return D3D_OK; }
};

class CubeTexture final : public TextureBase<IDirect3DCubeTexture9> {
 public:
  using TextureBase::TextureBase;
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == __uuidof(IUnknown) || riid == __uuidof(IDirect3DResource9) || riid == __uuidof(IDirect3DBaseTexture9) ||
        riid == __uuidof(IDirect3DCubeTexture9)) {
      AddRef();
      *out = this;
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  D3DRESOURCETYPE STDMETHODCALLTYPE GetType() override { return D3DRTYPE_CUBETEXTURE; }
  HRESULT STDMETHODCALLTYPE GetCubeMapSurface(D3DCUBEMAP_FACES face, UINT level, IDirect3DSurface9** out) override {
    if (!out || uint32_t(face) >= 6 || level >= l_.levels) return D3DERR_INVALIDCALL;
    *out = l_.surface(dev_, this, uint32_t(face), level);
    AddRef();
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE LockRect(D3DCUBEMAP_FACES face, UINT level, D3DLOCKED_RECT* lr, const RECT* r, DWORD f) override {
    if (uint32_t(face) >= 6 || level >= l_.levels) return D3DERR_INVALIDCALL;
    return l_.surface(dev_, this, uint32_t(face), level)->LockRect(lr, r, f);
  }
  HRESULT STDMETHODCALLTYPE UnlockRect(D3DCUBEMAP_FACES face, UINT level) override {
    if (uint32_t(face) >= 6 || level >= l_.levels) return D3DERR_INVALIDCALL;
    return l_.surface(dev_, this, uint32_t(face), level)->UnlockRect();
  }
  HRESULT STDMETHODCALLTYPE AddDirtyRect(D3DCUBEMAP_FACES, const RECT*) override { return D3D_OK; }
};

// --- device-method thunks (stdcall, the methods' signatures) ------------------------------------------------------
HRESULT STDMETHODCALLTYPE tCreateTexture(IDirect3DDevice9* d, UINT w, UINT h, UINT levels, DWORD usage, D3DFORMAT f,
                                         D3DPOOL pool, IDirect3DTexture9** out, HANDLE*) {
  if (!out || !w || !h) return D3DERR_INVALIDCALL;
  *out = new Texture(d, w, h, levels, 1, usage, f, pool);
  g_textures++;
  return D3D_OK;
}
HRESULT STDMETHODCALLTYPE tCreateCubeTexture(IDirect3DDevice9* d, UINT edge, UINT levels, DWORD usage, D3DFORMAT f,
                                             D3DPOOL pool, IDirect3DCubeTexture9** out, HANDLE*) {
  if (!out || !edge) return D3DERR_INVALIDCALL;
  *out = new CubeTexture(d, edge, edge, levels, 6, usage, f, pool);
  g_cubes++;
  return D3D_OK;
}
HRESULT STDMETHODCALLTYPE tCreateSurface(IDirect3DDevice9* d, UINT w, UINT h, D3DFORMAT f, D3DMULTISAMPLE_TYPE ms,
                                         DWORD quality, BOOL, IDirect3DSurface9** out, HANDLE*, DWORD usage) {
  if (!out || !w || !h) return D3DERR_INVALIDCALL;
  D3DSURFACE_DESC desc{ f, D3DRTYPE_SURFACE, usage, D3DPOOL_DEFAULT, ms, quality, w, h };
  *out = new Surface(d, desc, nullptr, nullptr);
  return D3D_OK;
}
HRESULT STDMETHODCALLTYPE tCreateRenderTarget(IDirect3DDevice9* d, UINT w, UINT h, D3DFORMAT f, D3DMULTISAMPLE_TYPE ms,
                                              DWORD q, BOOL lockable, IDirect3DSurface9** out, HANDLE* sh) {
  return tCreateSurface(d, w, h, f, ms, q, lockable, out, sh, D3DUSAGE_RENDERTARGET);
}
HRESULT STDMETHODCALLTYPE tCreateDepthStencilSurface(IDirect3DDevice9* d, UINT w, UINT h, D3DFORMAT f,
                                                     D3DMULTISAMPLE_TYPE ms, DWORD q, BOOL discard,
                                                     IDirect3DSurface9** out, HANDLE* sh) {
  return tCreateSurface(d, w, h, f, ms, q, discard, out, sh, D3DUSAGE_DEPTHSTENCIL);
}
// Mirrored by the backend (render-target groups, Do_ClearBuffer's hook, Copy/FastCopy hooks, group-end resolves).
HRESULT STDMETHODCALLTYPE tSetRenderTarget(IDirect3DDevice9*, DWORD, IDirect3DSurface9*) { return D3D_OK; }
HRESULT STDMETHODCALLTYPE tSetDepthStencilSurface(IDirect3DDevice9*, IDirect3DSurface9*) { return D3D_OK; }
HRESULT STDMETHODCALLTYPE tClear(IDirect3DDevice9*, DWORD, const D3DRECT*, DWORD, D3DCOLOR, float, DWORD) { return D3D_OK; }
HRESULT STDMETHODCALLTYPE tStretchRect(IDirect3DDevice9*, IDirect3DSurface9*, const RECT*, IDirect3DSurface9*, const RECT*,
                                       D3DTEXTUREFILTERTYPE) {
  return D3D_OK;
}

// --- FaceGen conversion (BSShaderUtil::ConvertTexture's D3DXLoadSurfaceFromSurface) ------------------------------
// Same-size format conversion of one level: the source texture's level from the backend, written into the destination
// surface's host memory (the system-memory texture ConvertTexture then locks and copies).
void toColors(D3DFORMAT f, const uint8_t* src, uint32_t pitch, uint32_t w, uint32_t h, std::vector<d3dx::Color>& out,
              d3dx::Precision p) {
  out.assign(size_t(w) * h, {});
  if (isDxt(f)) {
    const uint32_t bs = f == D3DFMT_DXT1 ? 8 : 16;
    for (uint32_t by = 0; by < (h + 3) / 4; by++)
      for (uint32_t bx = 0; bx < (w + 3) / 4; bx++) {
        d3dx::Color c[16];
        const uint8_t* b = src + size_t(by) * pitch + size_t(bx) * bs;
        if (f == D3DFMT_DXT1) d3dx::decodeDXT1(c, b, p);
        else if (f == D3DFMT_DXT5 || f == D3DFMT_DXT4) d3dx::decodeDXT5(c, b, p);
        else d3dx::decodeDXT3(c, b, p);
        for (uint32_t i = 0; i < 16; i++) {
          const uint32_t x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
          if (x < w && y < h) out[size_t(y) * w + x] = c[i];
        }
      }
    return;
  }
  for (uint32_t y = 0; y < h; y++)
    for (uint32_t x = 0; x < w; x++) {
      const uint8_t* t = src + size_t(y) * pitch + size_t(x) * 4;  // A8R8G8B8 / X8R8G8B8 memory: B G R A
      out[size_t(y) * w + x] = { t[2] / 255.0f, t[1] / 255.0f, t[0] / 255.0f, f == D3DFMT_X8R8G8B8 ? 1.0f : t[3] / 255.0f };
    }
}
void fromColors(D3DFORMAT f, const std::vector<d3dx::Color>& in, uint32_t w, uint32_t h, uint8_t* dst, uint32_t pitch,
                d3dx::Precision p) {
  if (isDxt(f)) {
    const uint32_t bs = f == D3DFMT_DXT1 ? 8 : 16;
    for (uint32_t by = 0; by < (h + 3) / 4; by++)
      for (uint32_t bx = 0; bx < (w + 3) / 4; bx++) {
        d3dx::Color c[16];
        for (uint32_t i = 0; i < 16; i++) {
          const uint32_t x = std::min(bx * 4 + (i & 3), w - 1), y = std::min(by * 4 + (i >> 2), h - 1);
          c[i] = in[size_t(y) * w + x];
        }
        uint8_t* b = dst + size_t(by) * pitch + size_t(bx) * bs;
        if (f == D3DFMT_DXT1) d3dx::encodeDXT1(b, c, true, p);
        else if (f == D3DFMT_DXT5 || f == D3DFMT_DXT4) d3dx::encodeDXT5(b, c, true, p);
        else d3dx::encodeDXT3(b, c, true, p);
      }
    return;
  }
  auto q = [](float v) { return uint8_t(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f); };
  for (uint32_t y = 0; y < h; y++)
    for (uint32_t x = 0; x < w; x++) {
      const d3dx::Color& c = in[size_t(y) * w + x];
      uint8_t* t = dst + size_t(y) * pitch + size_t(x) * 4;
      t[0] = q(c.b);
      t[1] = q(c.g);
      t[2] = q(c.r);
      t[3] = q(c.a);
    }
}
HRESULT WINAPI loadSurfaceFromSurface(IDirect3DSurface9* dst, const PALETTEENTRY*, const RECT*, IDirect3DSurface9* src,
                                      const PALETTEENTRY*, const RECT*, DWORD, D3DCOLOR) {
  auto* s = static_cast<Surface*>(src);
  auto* d = static_cast<Surface*>(dst);
  Backend* be = shadow::backendIfReady();
  const uintptr_t key = s && s->container() ? observe::texKeyForTexture(s->container()) : 0;
  std::vector<uint8_t> level;
  if (!be || !d || !key || !be->readbackTexture(TexKey(key), s->face, s->level, level)) {
    g_convertFailed++;
    return D3DERR_INVALIDCALL;
  }
  const D3DSURFACE_DESC& sd = s->desc();
  const D3DSURFACE_DESC& dd = d->desc();
  if (sd.Width != dd.Width || sd.Height != dd.Height) {
    g_convertFailed++;
    return D3DERR_INVALIDCALL;
  }
  D3DLOCKED_RECT lr{};
  d->LockRect(&lr, nullptr, 0);
  const uint32_t srcPitch = pitchOf(sd.Format, sd.Width);
  if (sd.Format == dd.Format) {
    for (uint32_t r = 0; r < rowsOf(dd.Format, dd.Height); r++)
      memcpy(static_cast<uint8_t*>(lr.pBits) + size_t(r) * lr.Pitch, level.data() + size_t(r) * srcPitch, srcPitch);
  } else {
    // The game's thread: D3DX would run under its x87 precision (main thread 24-bit).
    uint16_t cw = 0;
    __asm fnstcw cw;
    const d3dx::Precision p = ((cw >> 8) & 3) == 0 ? d3dx::Precision::Single : d3dx::Precision::Double;
    std::vector<d3dx::Color> colors;
    toColors(sd.Format, level.data(), srcPitch, sd.Width, sd.Height, colors, p);
    fromColors(dd.Format, colors, dd.Width, dd.Height, static_cast<uint8_t*>(lr.pBits), uint32_t(lr.Pitch), p);
  }
  d->UnlockRect();
  g_converted++;
  return D3D_OK;
}

}  // namespace

bool install() {
  struct S { uint32_t site, disp; void* fn; const char* where; };
#define F(x) reinterpret_cast<void*>(&x)
  const S sites[] = {
    { 0x00B6BCC2, 0x5C, F(tCreateTexture), "BSShaderUtil::ConvertTexture" },
    { 0x00E68102, 0x5C, F(tCreateTexture), "NiDX9SourceTextureData::CreateSurf" },
    { 0x00E6838A, 0x5C, F(tCreateTexture), "NiDX9SourceTextureData::CreateSurfFromRendererData" },
    { 0x00E8FF11, 0x5C, F(tCreateTexture), "NiDX9RenderedTextureData::CreateSurf" },
    { 0x00EBF801, 0x5C, F(tCreateTexture), "Bink Create_texture" },
    { 0x00EBF857, 0x5C, F(tCreateTexture), "Bink Create_texture" },
    { 0x00E8AAB7, 0x64, F(tCreateCubeTexture), "NiDX9SourceCubeMapData::CreateCubeMap" },
    { 0x00E90150, 0x64, F(tCreateCubeTexture), "NiDX9RenderedCubeMapData::CreateSurf" },
    { 0x00E8FEE1, 0x70, F(tCreateRenderTarget), "NiDX9RenderedTextureData::CreateSurf" },
    { 0x00E7D00C, 0x74, F(tCreateDepthStencilSurface), "NiDX9AdditionalDepthStencilBufferData::Recreate" },
    { 0x00E7DAE0, 0x74, F(tCreateDepthStencilSurface), "NiDX9AdditionalDepthStencilBufferData::Create" },
    { 0x00E7CC69, 0x94, F(tSetRenderTarget), "NiDX92DBufferData::ClearRenderTarget" },
    { 0x00E7CD7D, 0x94, F(tSetRenderTarget), "NiDX92DBufferData::SetRenderTarget" },
    { 0x00EC0F2C, 0x94, F(tSetRenderTarget), "Movie::Play" },
    { 0x00E7CCAF, 0x9C, F(tSetDepthStencilSurface), "NiDX9DepthStencilBufferData::SetNULLDepthStencilTarget" },
    { 0x00E7CDCF, 0x9C, F(tSetDepthStencilSurface), "NiDX9DepthStencilBufferData::SetDepthTarget" },
    { 0x00E6F4C7, 0xAC, F(tClear), "NiDX9Renderer::Do_ClearBuffer" },
    { 0x00EC1169, 0xAC, F(tClear), "Movie::Update" },
    { 0x00E6F7CA, 0x88, F(tStretchRect), "NiDX9Renderer::FastCopy" },
    { 0x00E6F93C, 0x88, F(tStretchRect), "NiDX9Renderer::Copy" },
    { 0x00E74E71, 0x88, F(tStretchRect), "NiDX9Renderer::Do_EndUsingRenderTargetGroup" },
  };
#undef F
  int ok = 0;
  for (const S& s : sites) ok += gbobjects::patchVirtualCall(s.site, s.disp, s.fn, s.where);
  // ConvertTexture's `call D3DXLoadSurfaceFromSurface` at 0x00B6BD75: E8 rel32 to the import thunk 0x00EE6DE6
  // (`jmp ds:__imp_D3DXLoadSurfaceFromSurface`, IAT 0x00FDF3D0) -> E8 rel32 to the conversion here.
  bool conv = false;
  {
    uint8_t* p = reinterpret_cast<uint8_t*>(uintptr_t(0x00B6BD75));
    int32_t rel = 0;
    memcpy(&rel, p + 1, 4);
    if (p[0] == 0xE8 && uintptr_t(p + 5 + rel) == 0x00EE6DE6) {
      DWORD old = 0;
      VirtualProtect(p, 5, PAGE_EXECUTE_READWRITE, &old);
      rel = int32_t(reinterpret_cast<uintptr_t>(&loadSurfaceFromSurface) - (uintptr_t(p) + 5));
      memcpy(p + 1, &rel, 4);
      VirtualProtect(p, 5, old, &old);
      FlushInstructionCache(GetCurrentProcess(), p, 5);
      conv = true;
    }
  }
  logf("native textures/surfaces: %d of %zu device call sites patched; ConvertTexture conversion %s", ok, std::size(sites),
       conv ? "patched" : "NOT FOUND");
  return ok == int(std::size(sites)) && conv;
}

HRESULT createTexture(IDirect3DDevice9* d, uint32_t w, uint32_t h, uint32_t levels, uint32_t usage, uint32_t format,
                      uint32_t pool, void** out) {
  return tCreateTexture(d, w, h, levels, usage, D3DFORMAT(format), D3DPOOL(pool), reinterpret_cast<IDirect3DTexture9**>(out), nullptr);
}
HRESULT createCubeTexture(IDirect3DDevice9* d, uint32_t edge, uint32_t levels, uint32_t usage, uint32_t format,
                          uint32_t pool, void** out) {
  return tCreateCubeTexture(d, edge, levels, usage, D3DFORMAT(format), D3DPOOL(pool),
                            reinterpret_cast<IDirect3DCubeTexture9**>(out), nullptr);
}
HRESULT createSurface(IDirect3DDevice9* d, uint32_t w, uint32_t h, uint32_t format, uint32_t ms, uint32_t quality,
                      uint32_t usage, uint32_t pool, void** out) {
  if (!out || !w || !h) return D3DERR_INVALIDCALL;
  D3DSURFACE_DESC desc{ D3DFORMAT(format), D3DRTYPE_SURFACE, usage, D3DPOOL(pool), D3DMULTISAMPLE_TYPE(ms), quality, w, h };
  *out = static_cast<IDirect3DSurface9*>(new Surface(d, desc, nullptr, nullptr));
  return D3D_OK;
}

// Host-memory copies between plugin surfaces (IDirect3DDevice9::UpdateSurface semantics: same format; rectangles in
// texels, DXT rectangles on 4x4 block boundaries).
HRESULT updateSurface(void* src, const RECT* srcRect, void* dst, const POINT* dstPoint) {
  auto* s = static_cast<Surface*>(static_cast<IDirect3DSurface9*>(src));
  auto* d = static_cast<Surface*>(static_cast<IDirect3DSurface9*>(dst));
  if (!s || !d || s->desc().Format != d->desc().Format) return D3DERR_INVALIDCALL;
  const D3DFORMAT f = s->desc().Format;
  RECT r = srcRect ? *srcRect : RECT{ 0, 0, LONG(s->desc().Width), LONG(s->desc().Height) };
  const LONG dx = dstPoint ? dstPoint->x : 0, dy = dstPoint ? dstPoint->y : 0;
  if (r.right - r.left + dx > LONG(d->desc().Width) || r.bottom - r.top + dy > LONG(d->desc().Height)) return D3DERR_INVALIDCALL;
  D3DLOCKED_RECT ls{}, ld{};
  RECT dr{ dx, dy, dx + (r.right - r.left), dy + (r.bottom - r.top) };
  s->LockRect(&ls, &r, D3DLOCK_READONLY);
  d->LockRect(&ld, &dr, 0);
  const bool dxt = isDxt(f);
  const uint32_t rows = dxt ? uint32_t((r.bottom - r.top + 3) / 4) : uint32_t(r.bottom - r.top);
  const uint32_t bytes = dxt ? uint32_t((r.right - r.left + 3) / 4) * (f == D3DFMT_DXT1 ? 8 : 16)
                             : uint32_t(r.right - r.left) * bytesPerTexel(f);
  for (uint32_t y = 0; y < rows; y++)
    memcpy(static_cast<uint8_t*>(ld.pBits) + size_t(y) * ld.Pitch, static_cast<const uint8_t*>(ls.pBits) + size_t(y) * ls.Pitch, bytes);
  s->UnlockRect();
  d->UnlockRect();
  return D3D_OK;
}
// IDirect3DDevice9::UpdateTexture: every level (and face) of the source into the destination's matching level.
HRESULT updateTexture(void* src, void* dst) {
  auto* sb = static_cast<IDirect3DBaseTexture9*>(src);
  auto* db = static_cast<IDirect3DBaseTexture9*>(dst);
  if (!sb || !db || sb->GetType() != db->GetType()) return D3DERR_INVALIDCALL;
  const DWORD levels = std::min(sb->GetLevelCount(), db->GetLevelCount());
  const uint32_t faces = sb->GetType() == D3DRTYPE_CUBETEXTURE ? 6 : 1;
  for (uint32_t f = 0; f < faces; f++)
    for (DWORD l = 0; l < levels; l++) {
      IDirect3DSurface9 *s = nullptr, *d = nullptr;
      if (faces == 6) {
        static_cast<IDirect3DCubeTexture9*>(sb)->GetCubeMapSurface(D3DCUBEMAP_FACES(f), l, &s);
        static_cast<IDirect3DCubeTexture9*>(db)->GetCubeMapSurface(D3DCUBEMAP_FACES(f), l, &d);
      } else {
        static_cast<IDirect3DTexture9*>(sb)->GetSurfaceLevel(l, &s);
        static_cast<IDirect3DTexture9*>(db)->GetSurfaceLevel(l, &d);
      }
      HRESULT hr = s && d ? updateSurface(s, nullptr, d, nullptr) : D3DERR_INVALIDCALL;
      if (s) s->Release();
      if (d) d->Release();
      if (FAILED(hr)) return hr;
    }
  return D3D_OK;
}

void logSummary() {
  logf("native textures/surfaces: textures %llu, cubes %llu, standalone+level surfaces %llu, live objects %llu | FaceGen "
       "conversions %llu (failed %llu)",
       g_textures.load(), g_cubes.load(), g_surfaces.load(), g_live.load(), g_converted.load(), g_convertFailed.load());
}

}  // namespace fnvvk::gbsurfaces
