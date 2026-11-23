// Bink movies on the backend (Movie::Play / Movie::Update, RAD's D3D9 "pow2 image" blitter at 0x00EBF7D0..).
// Bink decodes into tiles of D3D9 textures (image block: +0x04 D3DFORMAT, +0x14 lock cursor, +0x20 tile count,
// +0x24/+0x28 full tiles per row/column, +0x2C/+0x30 edge tile width/height (0 = none), +0x3C tile size,
// +0x40 IDirect3DTexture9* array) and draws them with fixed-function state as XYZRHW|TEX1 strips.
// - Lock_RAD_3D_image_pow2 0x00EBFDA0 / Unlock 0x00EBFF00: Bink writes one tile between them (Decompress_frame);
//   at unlock the written texels are uploaded as that tile's backend texture (key = its IDirect3DTexture9*).
// - Close_RAD_3D_image_pow2 0x00EBFD20: the tile textures go away.
// - Blit_RAD_3D_image_pow2 0x00EBFF30: its geometry is reproduced exactly (Submit_vertices 0x00EC0280) as backend
//   screen quads, clipped to the current viewport; drawn at once into an open group (background movies, inside the
//   renderer's frame), else queued for the movie frame's present.
// - Movie::Update's own IDirect3DDevice9::Present (call site 0x00EC13F1 `mov eax,[edx+44h]; call eax`) is patched to
//   a call of moviePresent, which renders the queued movie frame (black clear + quads) into the backbuffer and
//   presents it; D3D9 still presents too unless [Backend] Present=1.
#include <windows.h>

#include <d3d9.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "backend.h"
#include "common.h"
#include "gbstate.h"
#include "movie.h"
#include "present.h"
#include "shadow.h"

namespace fnvvk::movie {
namespace {

uint32_t field(const void* block, uint32_t off) { return *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(block) + off); }
void* tile(const void* block, uint32_t i) { return reinterpret_cast<void* const*>(uintptr_t(field(block, 0x40)))[i]; }

struct Locked { const void* block = nullptr; void* bits = nullptr; uint32_t pitch = 0, w = 0, h = 0; } g_locked;
std::vector<Backend::ScreenQuad> g_pending;
int32_t g_clip[4] = {};
uint64_t g_frames = 0, g_tiles = 0;

using LockFn = int(__cdecl*)(void*, uint32_t*, uint32_t*, uint32_t*, uint32_t*, uint32_t*, uint32_t*);
using UnlockFn = int(__cdecl*)(void*);
using CloseFn = void(__cdecl*)(void*);
using BlitFn = int(__cdecl*)(void*, float, float, float, float);
LockFn oLock = nullptr;
UnlockFn oUnlock = nullptr;
CloseFn oClose = nullptr;
BlitFn oBlit = nullptr;

int __cdecl hkLock(void* b, uint32_t* bits, uint32_t* pitch, uint32_t* x, uint32_t* y, uint32_t* w, uint32_t* h) {
  uint32_t lb = 0, lp = 0, lw = 0, lh = 0;
  const int r = oLock(b, bits ? bits : &lb, pitch ? pitch : &lp, x, y, w ? w : &lw, h ? h : &lh);
  g_locked = {};
  if (r) g_locked = { b, reinterpret_cast<void*>(uintptr_t(bits ? *bits : lb)), pitch ? *pitch : lp, w ? *w : lw, h ? *h : lh };
  return r;
}
int __cdecl hkUnlock(void* b) {
  if (g_locked.block == b && g_locked.bits)
    if (Backend* be = shadow::backendIfReady()) {
      const uint32_t cursor = field(b, 0x14);  // already advanced past the locked tile
      if (cursor) {
        be->updateTexture2D(TexKey(tile(b, cursor - 1)), field(b, 0x04), g_locked.w, g_locked.h, g_locked.bits, g_locked.pitch);
        g_tiles++;
      }
    }
  g_locked = {};
  return oUnlock(b);
}
void __cdecl hkClose(void* b) {
  if (b && field(b, 0x40))
    if (Backend* be = shadow::backendIfReady())
      for (uint32_t i = 0; i < field(b, 0x20); i++)
        if (void* t = tile(b, i)) be->destroyTexture(TexKey(t));
  oClose(b);
}

// Blit_RAD_3D_image_pow2 / Submit_vertices geometry: tile (cx, cy) of size (w, h) texels spans
// (x, y) .. (x + w * sx, y + h * sy); x advances by tile * sx per full tile, y by tile * sy per full row.
int __cdecl hkBlit(void* b, float x, float y, float sx, float sy) {
  const int r = oBlit(b, x, y, sx, sy);
  Backend* be = shadow::backendIfReady();
  if (!be) return r;
  const uint32_t across = field(b, 0x24), down = field(b, 0x28), edgeW = field(b, 0x2C), edgeH = field(b, 0x30),
                 size = field(b, 0x3C);
  std::vector<Backend::ScreenQuad> quads;
  uint32_t i = 0;
  float ty = y;
  auto row = [&](uint32_t th) {
    float tx = x;
    for (uint32_t c = 0; c < across; c++, tx += float(size) * sx)
      quads.push_back({ TexKey(tile(b, i++)), tx, ty, tx + float(size) * sx, ty + float(th) * sy });
    if (edgeW) quads.push_back({ TexKey(tile(b, i++)), tx, ty, tx + float(edgeW) * sx, ty + float(th) * sy });
  };
  for (uint32_t rr = 0; rr < down; rr++, ty += float(size) * sy) row(size);
  if (edgeH) row(edgeH);
  const auto& vp = gbstate::state().viewport;
  int32_t clip[4] = { int32_t(vp.x), int32_t(vp.y), int32_t(vp.x + vp.w), int32_t(vp.y + vp.h) };
  if (be->inGroup()) {
    be->drawScreenQuads(quads.data(), uint32_t(quads.size()), clip);
  } else {
    g_pending = std::move(quads);
    memcpy(g_clip, clip, sizeof(clip));
  }
  return r;
}

// Replaces Movie::Update's `call [vtbl+0x44]` (IDirect3DDevice9::Present, stdcall, args already pushed).
HRESULT __stdcall moviePresent(IDirect3DDevice9* dev, const RECT* a, const RECT* b, HWND c, const RGNDATA* d) {
  HRESULT hr = S_OK;
  if (!present::enabled()) hr = dev->Present(a, b, c, d);
  if (Backend* be = shadow::backendIfReady()) {
    RtKey bb = shadow::defaultTargetKey();
    if (bb) {
      be->beginGroup(&bb, 1, 0);
      int32_t full[4] = { 0, 0, 1 << 16, 1 << 16 };
      be->clear(kClearTarget, 0xFF000000, 1.0f, 0, full);  // Movie::Update's Clear(TARGET, black)
      if (!g_pending.empty()) be->drawScreenQuads(g_pending.data(), uint32_t(g_pending.size()), g_clip);
      be->endGroup();
      be->present(bb);
      g_frames++;
    }
  }
  g_pending.clear();
  return hr;
}

bool patchPresentSite() {
  uint8_t* p = reinterpret_cast<uint8_t*>(0x00EC13F1);
  static const uint8_t kExpect[5] = { 0x8B, 0x42, 0x44, 0xFF, 0xD0 };  // mov eax,[edx+44h]; call eax
  if (memcmp(p, kExpect, 5) != 0) {
    logf("movie: Present call site 00EC13F1 does not match; movie frames are not presented by the backend");
    return false;
  }
  DWORD old = 0;
  if (!VirtualProtect(p, 5, PAGE_EXECUTE_READWRITE, &old)) return false;
  p[0] = 0xE8;  // call rel32 moviePresent
  const int32_t rel = int32_t(reinterpret_cast<uintptr_t>(&moviePresent) - (uintptr_t(p) + 5));
  memcpy(p + 1, &rel, 4);
  VirtualProtect(p, 5, old, &old);
  FlushInstructionCache(GetCurrentProcess(), p, 5);
  return true;
}

}  // namespace

bool install() {
  bool ok = hook(0x00EBFDA0, &hkLock, &oLock, "Lock_RAD_3D_image_pow2");
  ok &= hook(0x00EBFF00, &hkUnlock, &oUnlock, "Unlock_RAD_3D_image_pow2");
  ok &= hook(0x00EBFD20, &hkClose, &oClose, "Close_RAD_3D_image_pow2");
  ok &= hook(0x00EBFF30, &hkBlit, &oBlit, "Blit_RAD_3D_image_pow2");
  ok &= patchPresentSite();
  return ok;
}

void logSummary() {
  if (g_frames || g_tiles) logf("movie: %llu frames presented, %llu tiles uploaded", g_frames, g_tiles);
}

}  // namespace fnvvk::movie
