#include "textures.h"

#include <windows.h>
#include <d3d9.h>

#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "addresses.h"
#include "backend.h"
#include "common.h"
#include "gbsurfaces.h"
#include "observe.h"
#include "d3dxtex/texture.h"
#include "shadow.h"

namespace fnvvk::textures {

namespace {

bool g_verify = true;

struct Counters {
  std::atomic<uint64_t> loads{ 0 }, loadsSingle{ 0 }, loadsDouble{ 0 }, loadsOtherPrecision{ 0 };
  std::atomic<uint64_t> fileObjectPassed{ 0 }, rereadFailed{ 0 }, mipSkipped{ 0 }, noCapture{ 0 };
  std::atomic<uint64_t> bytesCompared{ 0 }, bytesIdentical{ 0 }, bytesMismatch{ 0 };
  std::atomic<uint64_t> nativeOk{ 0 }, nativeUnsupported{ 0 }, nativeOtherError{ 0 };
  std::atomic<uint64_t> placeholders{ 0 };
  std::atomic<uint64_t> verified{ 0 }, verifyIdentical{ 0 }, verifyMismatch{ 0 }, verifySkipped{ 0 };
  std::atomic<uint64_t> uploaded{ 0 }, uploadFailed{ 0 }, destroyed{ 0 };
} g_c;
std::atomic<uint32_t> g_logged{ 0 };

// The file buffer the game hands to D3DX (after its mip skip), captured per thread for the call in flight. Captured
// at d3dx9_38's D3DXCreate[Cube]TextureFromFileInMemory exports, which only LoadTextureFile calls (IDA xrefs); the
// game frees the buffer right after, so it is copied. Observation only: the game's call is passed through unchanged.
thread_local std::vector<uint8_t> t_capture;
thread_local bool t_captured = false;
using CreateTexFn = long(__stdcall*)(void*, const void*, unsigned, void**);
CreateTexFn oCreateTex, oCreateCube;

// Native mode (D3D9 draws off): the file is decoded here with the D3DX-exact decoder and D3D9 gets an empty
// D3DPOOL_DEFAULT texture of the same description (what InitializeFromD3DTexture reads: type, size, levels, format),
// so no managed system-memory copy exists in the 32-bit address space. Needs the D3D9Ex device (present.cpp), on
// which DEFAULT resources survive Reset. Falls back to D3DX for anything the decoder does not produce.
bool g_native = false;
thread_local bool t_haveDecoded = false;
thread_local d3dx::Texture t_decoded;
uint16_t currentX87() {
  uint16_t cw = 0;
  __asm fnstcw cw;
  return cw;
}
bool createPlaceholder(void* dev, const void* data, unsigned size, void** out) {
  const uint32_t pc = (currentX87() >> 8) & 3;
  const d3dx::Precision prec = pc == 0 ? d3dx::Precision::Single : d3dx::Precision::Double;
  d3dx::Texture tex;
  if (d3dx::createTextureFromDds(static_cast<const uint8_t*>(data), size, prec, tex, nullptr) != d3dx::Status::Ok) return false;
  // Vulkan-backed texture objects with the D3D9 texture ABI (gbsurfaces): the description only.
  auto* d = static_cast<IDirect3DDevice9*>(dev);
  const HRESULT hr = tex.cube ? gbsurfaces::createCubeTexture(d, tex.width, tex.levels, 0, tex.format, D3DPOOL_DEFAULT, out)
                              : gbsurfaces::createTexture(d, tex.width, tex.height, tex.levels, 0, tex.format, D3DPOOL_DEFAULT, out);
  if (FAILED(hr) || !*out) {
    *out = nullptr;
    return false;
  }
  t_decoded = std::move(tex);
  t_haveDecoded = true;
  return true;
}
long __stdcall hkCreateTex(void* dev, const void* data, unsigned size, void** out) {
  t_capture.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
  t_captured = true;
  if (g_native && createPlaceholder(dev, data, size, out)) return 0;
  return oCreateTex(dev, data, size, out);
}
long __stdcall hkCreateCube(void* dev, const void* data, unsigned size, void** out) {
  t_capture.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
  t_captured = true;
  if (g_native && createPlaceholder(dev, data, size, out)) return 0;
  return oCreateCube(dev, data, size, out);
}

// Game functions (Gamebryo level; addresses.h has the references).
using StandardizeFn = bool(__cdecl*)(char*);
using GetFileFn = void*(__cdecl*)(const char*, int, unsigned);
using StreamReadFn = unsigned(__fastcall*)(void*, void*, void*, unsigned);

// NiFile::GetFile + Bool + GetFileSize + NiBinaryStream::Read + delete, as LoadTextureFile does it.
bool readGameFile(const char* name, std::vector<uint8_t>& out) {
  char path[MAX_PATH];
  if (strcpy_s(path, name)) return false;
  reinterpret_cast<StandardizeFn>(fnv::NiPath_Standardize)(path);
  void* file = reinterpret_cast<GetFileFn>(fnv::NiFile_GetFile)(path, 0, 0x8000);
  if (!file) return false;
  void** vt = *reinterpret_cast<void***>(file);
  using BoolFn = bool(__fastcall*)(void*, void*);
  using SizeFn = unsigned(__fastcall*)(void*, void*);
  using DeleteFn = void*(__fastcall*)(void*, void*, unsigned);
  bool ok = reinterpret_cast<BoolFn>(vt[1])(file, nullptr);
  if (ok) {
    unsigned size = reinterpret_cast<SizeFn>(vt[7])(file, nullptr);
    ok = size != 0;
    if (ok) {
      out.resize(size);
      reinterpret_cast<StreamReadFn>(fnv::NiBinaryStream_Read)(file, nullptr, out.data(), size);
    }
  }
  reinterpret_cast<DeleteFn>(vt[0])(file, nullptr, 1);
  return ok;
}

// LoadTextureFile's mip skip (0x00E68C57-0x00E68DBD), reproduced exactly: DXT1-5 DDS only; unsigned arithmetic;
// the block count uses w>>2 (floor); the rewritten header gets the new size, mip count and pitch = w*h.
bool applyMipSkip(std::vector<uint8_t>& f) {
  if (f.size() < 128 || memcmp(f.data(), "DDS ", 4)) return false;
  auto u32 = [&](size_t o) { uint32_t v; memcpy(&v, f.data() + o, 4); return v; };
  auto put = [&](size_t o, uint32_t v) { memcpy(f.data() + o, &v, 4); };
  const uint32_t caps2 = u32(112);
  if ((caps2 & 0x200) || (caps2 & 0x200000)) return false;  // D3DX reports cube/volume: not D3DRTYPE_TEXTURE
  if (!(u32(80) & 4)) return false;
  const uint32_t fourcc = u32(84);
  uint32_t div;
  if (fourcc == 0x31545844) div = 2;
  else if (fourcc == 0x32545844 || fourcc == 0x33545844 || fourcc == 0x34545844 || fourcc == 0x35545844) div = 1;
  else return false;
  const uint32_t skip = *reinterpret_cast<const uint32_t*>(fnv::NiDX9SourceTextureData_uiMipMapSkip);
  const uint32_t minimum = *reinterpret_cast<const uint32_t*>(fnv::NiDX9SourceTextureData_uiMipMapMinimum);
  const uint32_t mips = u32(28);
  uint32_t target = mips - skip;
  if (target <= minimum) target = minimum;
  if (mips <= skip) {
    target = mips;
    if (mips >= minimum) target = minimum;
  }
  if (target >= mips) return false;
  uint32_t w = u32(16), h = u32(12), count = mips;
  size_t src = 128, size = f.size();
  for (uint32_t n = mips - target; n; n--) {
    uint32_t bytes = 16 * std::max(w >> 2, 1u) * std::max(h >> 2, 1u) / div;
    src += bytes;
    size -= bytes;
    w = std::max(w >> 1, 1u);
    h = std::max(h >> 1, 1u);
    count--;
  }
  if (count == mips) return false;
  std::vector<uint8_t> out(size);
  memcpy(out.data(), f.data(), 128);
  memcpy(out.data() + 128, f.data() + src, size - 128);
  f.swap(out);
  put(16, w);
  put(12, h);
  put(28, count);
  put(20, w * h);
  return true;
}

// Verification only: every level (every face) of the game's D3D9 texture against the native result.
bool compareWithD3D(IDirect3DBaseTexture9* base, const d3dx::Texture& t, char* why, size_t whyLen) {
  if (base->GetLevelCount() != t.levels) {
    _snprintf_s(why, whyLen, _TRUNCATE, "levels d3d %lu native %u", base->GetLevelCount(), t.levels);
    return false;
  }
  const bool cube = base->GetType() == D3DRTYPE_CUBETEXTURE;
  if (cube != t.cube) {
    _snprintf_s(why, whyLen, _TRUNCATE, "type mismatch (d3d cube %d)", cube);
    return false;
  }
  const uint32_t faces = cube ? 6 : 1;
  for (uint32_t f = 0; f < faces; f++)
    for (uint32_t l = 0; l < t.levels; l++) {
      const d3dx::Level& nl = t.surfaces[size_t(f) * t.levels + l];
      D3DSURFACE_DESC d{};
      D3DLOCKED_RECT lr{};
      HRESULT hr;
      if (cube) {
        auto* c = static_cast<IDirect3DCubeTexture9*>(base);
        c->GetLevelDesc(l, &d);
        hr = c->LockRect(D3DCUBEMAP_FACES(f), l, &lr, nullptr, D3DLOCK_READONLY);
      } else {
        auto* x = static_cast<IDirect3DTexture9*>(base);
        x->GetLevelDesc(l, &d);
        hr = x->LockRect(l, &lr, nullptr, D3DLOCK_READONLY);
      }
      if (FAILED(hr)) {
        _snprintf_s(why, whyLen, _TRUNCATE, "LockRect failed face %u level %u", f, l);
        return false;
      }
      bool same = uint32_t(d.Format) == t.format && d.Width == nl.width && d.Height == nl.height;
      uint32_t badRow = UINT32_MAX;
      for (uint32_t r = 0; same && r < nl.rows; r++)
        if (memcmp(static_cast<const uint8_t*>(lr.pBits) + size_t(r) * lr.Pitch, nl.data.data() + size_t(r) * nl.rowPitch,
                   nl.rowPitch)) {
          same = false;
          badRow = r;
        }
      if (cube) static_cast<IDirect3DCubeTexture9*>(base)->UnlockRect(D3DCUBEMAP_FACES(f), l);
      else static_cast<IDirect3DTexture9*>(base)->UnlockRect(l);
      if (!same) {
        _snprintf_s(why, whyLen, _TRUNCATE, "face %u level %u differs (d3d %ux%u fmt %u, native %ux%u fmt %u, row %d)", f,
                    l, d.Width, d.Height, unsigned(d.Format), nl.width, nl.height, t.format, int(badRow));
        return false;
      }
    }
  return true;
}

using DeletingDtorFn = void*(__fastcall*)(void*, void*, unsigned);
DeletingDtorFn oDtorSourceTexture;
}  // namespace

void beginLoad() {
  t_captured = false;
  t_haveDecoded = false;
}
void setNative(bool on) { g_native = on; }

namespace {
void* __fastcall hkDtorSourceTexture(void* self, void* edx, unsigned flags) {
  if (Backend* be = shadow::backendIfReady())
    if (be->destroyTexture(TexKey(self))) g_c.destroyed++;
    observe::unregisterTexture(*reinterpret_cast<void**>(static_cast<char*>(self) + fnv::NiDX9TextureData_D3DTexture));
  return oDtorSourceTexture(self, edx, flags);
}

// --- textures from in-memory pixel data (UI, fonts, generated textures, cube maps) ----------------------------------
// Gamebryo creates these from NiPersistentSrcTextureRendererData and copies its levels raw into the D3D texture:
// CopyRendererDataToSurface 0x00E68920 (2D: levels m_uiLevelsSkipped.. m_uiLevels, face 0) and
// NiDX9SourceCubeMapData::CopyRendererDataToCubeMapFace 0x00E8AB70 (one face, levels 0.. m_uiLevels), both through
// CopyRendererDataToSurfaceLevel 0x00E683D0. The backend takes the same bytes. Offsets read from those functions'
// machine code: source data +0x18 NiPixelFormat::m_eFormat, +0x60 pixels, +0x64 widths[], +0x68 heights[],
// +0x6C offsets[] (offsets[mipCount] = face size), +0x74 mip count, +0x78 bytes per pixel; texture data +0x0C/+0x10
// width/height, +0x14 NiPixelFormat, +0x64 D3D texture, +0x68 levels, +0x78 levels skipped.
template <typename T> T rdp(const void* p, uint32_t off) { return *reinterpret_cast<const T*>(static_cast<const char*>(p) + off); }
using DetermineD3DFormatFn = uint32_t(__cdecl*)(const void*);
const auto DetermineD3DFormat = reinterpret_cast<DetermineD3DFormatFn>(0x00E7BD20);

struct PixelCounters {
  std::atomic<uint64_t> textures{ 0 }, cubes{ 0 }, unsupported{ 0 }, verified{ 0 }, identical{ 0 }, mismatch{ 0 },
    uploaded{ 0 }, failed{ 0 };
} g_p;

// Source layouts, read from the copy functions' machine code. NiPixelData (CopyDataToSurfaceLevel 0x00E68150, also
// IDA's type): format +0x0C, pixels +0x50, widths +0x54, heights +0x58, offsets +0x5C, mips +0x60, stride +0x64.
// NiPersistentSrcTextureRendererData (CopyRendererDataToSurfaceLevel 0x00E683D0): the same fields 0x10 later.
struct SrcLayout { uint32_t format, pixels; };
constexpr SrcLayout kPixelData{ 0x0C, 0x50 }, kPersistent{ 0x18, 0x60 };

// One level of the source, as the copy functions read it, repacked tightly (backend upload layout).
bool sourceLevel(const void* src, const SrcLayout& L, uint32_t level, uint32_t face, uint32_t d3dFmt, d3dx::Level& out) {
  const uint8_t* pixels = rdp<const uint8_t*>(src, L.pixels);
  const uint32_t* widths = rdp<const uint32_t*>(src, L.pixels + 4);
  const uint32_t* heights = rdp<const uint32_t*>(src, L.pixels + 8);
  const uint32_t* offsets = rdp<const uint32_t*>(src, L.pixels + 0xC);
  const uint32_t mips = rdp<uint32_t>(src, L.pixels + 0x10), stride = rdp<uint32_t>(src, L.pixels + 0x14),
                 nf = rdp<uint32_t>(src, L.format);
  if (!pixels || !widths || !heights || !offsets || level >= mips) return false;
  const uint32_t w = widths[level], h = heights[level];
  const bool dxt = nf >= 4 && nf <= 6;  // NiPixelFormat DXT1/3/5, as the copy function tests it
  uint32_t rowSize, rows;
  if (dxt) {
    rowSize = ((w + 3) / 4) * (d3dFmt == d3dx::kFmtDXT1 ? 8 : 16);
    rows = (h + 3) / 4;
  } else {
    rowSize = stride * w;
    rows = h;
  }
  out.width = w;
  out.height = h;
  out.rowPitch = rowSize;
  out.rows = rows;
  const uint8_t* p = pixels + offsets[level] + size_t(offsets[mips]) * face;
  out.data.assign(p, p + size_t(rowSize) * rows);
  return true;
}

void upload(void* data, d3dx::Texture& tex) {
  Backend* be = shadow::backendIfReady();
  if (!be) return;
  auto* d3dTex = rdp<IDirect3DBaseTexture9*>(data, fnv::NiDX9TextureData_D3DTexture);
  if (g_verify && d3dTex) {
    char msg[256] = "";
    g_p.verified++;
    if (compareWithD3D(d3dTex, tex, msg, sizeof(msg))) g_p.identical++;
    else {
      g_p.mismatch++;
      if (g_logged++ < 40) logf("PIXEL-DATA TEXTURE MISMATCH (fmt %u %ux%u cube %d): %s", tex.format, tex.width, tex.height, tex.cube, msg);
    }
  }
  if (be->createTexture(TexKey(data), std::move(tex))) {  // (tex is not used after: handed over)
    g_p.uploaded++;
    if (d3dTex) observe::registerTexture(d3dTex, TexKey(data));
  } else {
    g_p.failed++;
  }
}

using CopyToSurfaceFn = void(__fastcall*)(void*, void*, void*);
CopyToSurfaceFn oCopyToSurface;
// Reading game objects is guarded: a malformed object must not take the game down (SEH needs a frame without C++
// destructors, hence the wrapper).
bool guarded(void (*fn)(void*, void*, uint32_t), void* a, void* b, uint32_t c) {
  __try {
    fn(a, b, c);
    return true;
  } __except (EXCEPTION_EXECUTE_HANDLER) {
    return false;
  }
}

void copy2D(void* data, void* src, const SrcLayout& L) {
  {
    uint32_t levels = rdp<uint32_t>(data, 0x68), skipped = rdp<uint32_t>(data, 0x78);
    if (levels <= skipped) return;
    d3dx::Texture tex;
    tex.format = DetermineD3DFormat(static_cast<char*>(data) + 0x14);
    tex.width = rdp<uint32_t>(data, 0x0C);
    tex.height = rdp<uint32_t>(data, 0x10);
    tex.levels = levels - skipped;
    tex.cube = false;
    for (uint32_t i = 0; i < tex.levels; i++) {
      d3dx::Level l;
      if (!sourceLevel(src, L, skipped + i, 0, tex.format, l)) { g_p.unsupported++; return; }
      tex.surfaces.push_back(std::move(l));
    }
    g_p.textures++;
    upload(data, tex);
  }
}
void copy2DPersistent(void* data, void* src, uint32_t) { copy2D(data, src, kPersistent); }
void copy2DPixelData(void* data, void* src, uint32_t) { copy2D(data, src, kPixelData); }
void __fastcall hkCopyToSurface(void* data, void* edx, void* src) {
  oCopyToSurface(data, edx, src);
  if (!shadow::backendIfReady() || !data || !src) return;
  if (!guarded(&copy2DPersistent, data, src, 0)) logf("pixel-data texture: exception reading %p / %p", data, src);
}
CopyToSurfaceFn oCopyDataToSurface;
void __fastcall hkCopyDataToSurface(void* data, void* edx, void* src) {
  oCopyDataToSurface(data, edx, src);
  if (!shadow::backendIfReady() || !data || !src) return;
  if (!guarded(&copy2DPixelData, data, src, 0)) logf("pixel-data texture: exception reading %p / %p", data, src);
}

// Cube faces arrive one call each; the texture is created once all six are in.
struct PendingCube { d3dx::Texture tex; uint32_t faces = 0; };
std::mutex g_cubeMutex;
std::unordered_map<void*, PendingCube> g_cubes;
using CopyCubeFaceFn = void(__fastcall*)(void*, void*, void*, uint32_t);
CopyCubeFaceFn oCopyCubeFace;
void copyCubeFace(void* data, void* src, uint32_t face, const SrcLayout& L) {
  std::lock_guard<std::mutex> lock(g_cubeMutex);
  {
    PendingCube& pc = g_cubes[data];
    d3dx::Texture& tex = pc.tex;
    if (!pc.faces) {
      tex.format = DetermineD3DFormat(static_cast<char*>(data) + 0x14);
      tex.width = rdp<uint32_t>(data, 0x0C);
      tex.height = rdp<uint32_t>(data, 0x10);
      tex.levels = rdp<uint32_t>(data, 0x68);
      tex.cube = true;
      tex.surfaces.assign(size_t(6) * tex.levels, d3dx::Level{});
    }
    for (uint32_t i = 0; i < tex.levels; i++)
      if (!sourceLevel(src, L, i, face, tex.format, tex.surfaces[size_t(face) * tex.levels + i])) {
        g_p.unsupported++;
        g_cubes.erase(data);
        return;
      }
    pc.faces |= 1u << face;
    if (pc.faces == 0x3F) {
      g_p.cubes++;
      upload(data, tex);
      g_cubes.erase(data);
    }
  }
}
void cubePersistent(void* data, void* src, uint32_t face) { copyCubeFace(data, src, face, kPersistent); }
void cubePixelData(void* data, void* src, uint32_t face) { copyCubeFace(data, src, face, kPixelData); }
void __fastcall hkCopyCubeFace(void* data, void* edx, void* src, uint32_t face) {
  oCopyCubeFace(data, edx, src, face);
  if (!shadow::backendIfReady() || !data || !src || face >= 6) return;
  if (!guarded(&cubePersistent, data, src, face)) logf("pixel-data cube: exception reading %p / %p", data, src);
}
CopyCubeFaceFn oCopyCubeData;
void __fastcall hkCopyCubeData(void* data, void* edx, void* src, uint32_t face) {
  oCopyCubeData(data, edx, src, face);
  if (!shadow::backendIfReady() || !data || !src || face >= 6) return;
  if (!guarded(&cubePixelData, data, src, face)) logf("pixel-data cube: exception reading %p / %p", data, src);
}

}  // namespace

bool installHooks() {
  hook(0x00E68920, &hkCopyToSurface, &oCopyToSurface, "NiDX9SourceTextureData::CopyRendererDataToSurface");
  hook(0x00E8AB70, &hkCopyCubeFace, &oCopyCubeFace, "NiDX9SourceCubeMapData::CopyRendererDataToCubeMapFace");
  hook(0x00E68870, &hkCopyDataToSurface, &oCopyDataToSurface, "NiDX9SourceTextureData::CopyDataToSurface");
  hook(0x00E8AAF0, &hkCopyCubeData, &oCopyCubeData, "NiDX9SourceCubeMapData::CopyDataToSurface");
  bool ok = hook(fnv::NiDX9SourceTextureData_DeletingDtor, &hkDtorSourceTexture, &oDtorSourceTexture,
                 "~NiDX9SourceTextureData (deleting)");
  HMODULE dx = GetModuleHandleA("d3dx9_38.dll");
  auto tex = dx ? reinterpret_cast<uintptr_t>(GetProcAddress(dx, "D3DXCreateTextureFromFileInMemory")) : 0;
  auto cube = dx ? reinterpret_cast<uintptr_t>(GetProcAddress(dx, "D3DXCreateCubeTextureFromFileInMemory")) : 0;
  if (!tex || !cube) {
    logf("d3dx9_38 exports not found; texture capture disabled");
    return false;
  }
  ok &= hook(tex, &hkCreateTex, &oCreateTex, "d3dx9_38 D3DXCreateTextureFromFileInMemory (observe)");
  ok &= hook(cube, &hkCreateCube, &oCreateCube, "d3dx9_38 D3DXCreateCubeTextureFromFileInMemory (observe)");
  return ok;
}

void configure(bool verify) { g_verify = verify; }

void onLoadTextureFile(void* data, const char* name, void* file, uint16_t cw, bool loaded) {
  const bool captured = t_captured;
  const bool decoded = t_haveDecoded;
  t_captured = false;
  t_haveDecoded = false;
  if (!loaded || !data) return;
  Backend* be = shadow::backendIfReady();
  if (!be) return;
  g_c.loads++;
  if (decoded) {
    // Native mode: the D3D9 texture is an empty placeholder; the decode made at the D3DX call is the content.
    g_c.nativeOk++;
    g_c.placeholders++;
    if (be->createTexture(TexKey(data), std::move(t_decoded))) {
      g_c.uploaded++;
      observe::registerTexture(*reinterpret_cast<void**>(static_cast<char*>(data) + fnv::NiDX9TextureData_D3DTexture),
                               TexKey(data));
    } else {
      g_c.uploadFailed++;
    }
    t_decoded = d3dx::Texture{};
    return;
  }
  const uint32_t pc = (cw >> 8) & 3;
  d3dx::Precision prec = d3dx::Precision::Double;
  if (pc == 0) { prec = d3dx::Precision::Single; g_c.loadsSingle++; }
  else if (pc == 2) g_c.loadsDouble++;
  else g_c.loadsOtherPrecision++;  // 64-bit precision: not reproduced (never measured in game)

  // Bytes for D3DX. Named loads: the native path (NiFile reread + the game's mip skip), checked against what the
  // game passed to D3DX. Loads from an already-open NiFile (consumed by the original): the captured bytes.
  std::vector<uint8_t> bytes;
  if (file) {
    g_c.fileObjectPassed++;
    if (!captured) {
      g_c.noCapture++;
      return;
    }
    bytes = t_capture;
  } else {
    if (!name || !readGameFile(name, bytes)) {
      g_c.rereadFailed++;
      if (g_logged++ < 40) logf("texture: could not reread %s through NiFile", name ? name : "(null)");
      return;
    }
    if (applyMipSkip(bytes)) g_c.mipSkipped++;
    if (captured) {
      g_c.bytesCompared++;
      if (bytes == t_capture) {
        g_c.bytesIdentical++;
      } else {
        g_c.bytesMismatch++;
        if (g_logged++ < 40)
          logf("TEXTURE BYTES differ from the game's D3DX input: %s (native %zu bytes, game %zu)", name, bytes.size(),
               t_capture.size());
      }
    } else {
      g_c.noCapture++;
    }
  }
  d3dx::Texture tex;
  std::string why;
  d3dx::Status st = d3dx::createTextureFromDds(bytes.data(), bytes.size(), prec, tex, &why);
  if (st != d3dx::Status::Ok) {
    (st == d3dx::Status::UnsupportedFormat || st == d3dx::Status::UnsupportedLayout ? g_c.nativeUnsupported : g_c.nativeOtherError)++;
    if (g_logged++ < 40) logf("texture: native load of %s: %s", name ? name : "(NiFile)", why.c_str());
    return;
  }
  g_c.nativeOk++;
  if (g_verify) {
    auto* d3dTex = *reinterpret_cast<IDirect3DBaseTexture9**>(static_cast<char*>(data) + fnv::NiDX9TextureData_D3DTexture);
    if (!d3dTex) {
      g_c.verifySkipped++;
    } else {
      char msg[256] = "";
      g_c.verified++;
      if (compareWithD3D(d3dTex, tex, msg, sizeof(msg))) {
        g_c.verifyIdentical++;
      } else {
        g_c.verifyMismatch++;
        if (g_logged++ < 40) logf("TEXTURE MISMATCH %s (x87 cw %04X): %s", name ? name : "(NiFile)", cw, msg);
      }
    }
  }
  if (be->createTexture(TexKey(data), std::move(tex))) {  // (tex is not used after: handed over)
    g_c.uploaded++;
    observe::registerTexture(*reinterpret_cast<void**>(static_cast<char*>(data) + fnv::NiDX9TextureData_D3DTexture),
                             TexKey(data));
  }
  else g_c.uploadFailed++;
}

void logSummary() {
  logf("textures: loads %llu (x87 24-bit %llu, 53-bit %llu, other %llu; NiFile passed %llu) | reread failed %llu, "
       "mip-skipped %llu, no capture %llu | file bytes vs game's D3DX input %llu: identical %llu, MISMATCH %llu | native ok %llu, unsupported %llu, error %llu | verified vs D3D9 %llu: identical %llu, "
       "MISMATCH %llu, skipped %llu | uploaded %llu, upload failed %llu, destroyed %llu",
       g_c.loads.load(), g_c.loadsSingle.load(), g_c.loadsDouble.load(), g_c.loadsOtherPrecision.load(),
       g_c.fileObjectPassed.load(), g_c.rereadFailed.load(), g_c.mipSkipped.load(), g_c.noCapture.load(), g_c.bytesCompared.load(),
       g_c.bytesIdentical.load(), g_c.bytesMismatch.load(), g_c.nativeOk.load(),
       g_c.nativeUnsupported.load(), g_c.nativeOtherError.load(), g_c.verified.load(), g_c.verifyIdentical.load(),
       g_c.verifyMismatch.load(), g_c.verifySkipped.load(), g_c.uploaded.load(), g_c.uploadFailed.load(),
       g_c.destroyed.load());
  if (Backend* be = shadow::backendIfReady()) {
    const BackendStats s = be->statsSnapshot();
    logf("texture upload queue: peak %.1f MiB of decoded data queued, %llu uploads waited for the queue to drain",
         double(s.textureQueuePeakBytes) / (1 << 20), s.textureQueueWaits);
  }
  logf("pixel-data textures: 2D %llu, cube %llu, unsupported %llu | verified vs D3D9 %llu: identical %llu, MISMATCH %llu "
       "| uploaded %llu, failed %llu",
       g_p.textures.load(), g_p.cubes.load(), g_p.unsupported.load(), g_p.verified.load(), g_p.identical.load(),
       g_p.mismatch.load(), g_p.uploaded.load(), g_p.failed.load());
  if (g_c.placeholders) logf("textures: %llu loaded natively with empty D3D9 placeholders", g_c.placeholders.load());
}

}  // namespace fnvvk::textures
