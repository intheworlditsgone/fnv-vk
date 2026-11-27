// Offline x64 backend tests with the Khronos validation layer.
//  1. Clear oracle: every render-target format the game uses (docs/M2_RESULTS.md), with and without MSAA, cleared
//     by real D3D9 (IDirect3DDevice9::Clear) and by the backend with the same flags/color/rect; texels compared.
//  2. Depth/stencil clears checked against the conversion rule (D3DFMT_D24S8 is not lockable in D3D9).
//  3. Lifetime churn: the in-game frame structure replayed for many frames while targets are destroyed,
//     re-attached and resized, as the game does during play.
//  4. Present path: shadow window, swapchain, window resize.
// Exit code 0 only if every check passes and the layer reported nothing.
#include <windows.h>
#include <d3d9.h>
#include <dbghelp.h>

#include <cmath>
#include <set>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <array>
#include <filesystem>
#include <functional>

#include "backend.h"
#include "d3dxtex/texture.h"
#include "vkfn.h"  // VkFormat values for interpreting readbacks
#include "effectlib.h"
#include "shaderpack.h"
#include "imgui.h"

using namespace fnvvk;

static int g_failures = 0;
static void fail(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  printf("FAIL: ");
  vprintf(fmt, ap);
  printf("\n");
  va_end(ap);
  g_failures++;
}

// --- D3D9 reference ---------------------------------------------------------------------------------------------
struct D3DRef {
  IDirect3D9* d3d = nullptr;
  IDirect3DDevice9* dev = nullptr;
  HWND wnd = nullptr;

  bool init() {
    WNDCLASSA wc{};
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.lpszClassName = "backend_test_d3d";
    RegisterClassA(&wc);
    wnd = CreateWindowA(wc.lpszClassName, "d3d", WS_OVERLAPPEDWINDOW, 0, 0, 64, 64, nullptr, nullptr, wc.hInstance, nullptr);
    d3d = Direct3DCreate9(D3D_SDK_VERSION);
    if (!d3d) return false;
    D3DPRESENT_PARAMETERS pp{};
    pp.Windowed = TRUE;
    pp.SwapEffect = D3DSWAPEFFECT_DISCARD;
    pp.BackBufferFormat = D3DFMT_UNKNOWN;
    pp.hDeviceWindow = wnd;
    // FNV_GAME_DEVICE=1: the game's device (logged in game): behaviour 0x54 = hardware VP | pure | multithreaded,
    // no FPU_PRESERVE; FNV_GAME_DEVICE=2 also exclusive fullscreen at the desktop mode with vsync.
    DWORD flags = D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE;
    const char* gd = getenv("FNV_GAME_DEVICE");
    if (gd) flags = D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_PUREDEVICE | D3DCREATE_MULTITHREADED;
    if (gd && gd[0] == '2') {
      D3DDISPLAYMODE dm{};
      d3d->GetAdapterDisplayMode(D3DADAPTER_DEFAULT, &dm);
      pp.Windowed = FALSE;
      pp.BackBufferWidth = dm.Width;
      pp.BackBufferHeight = dm.Height;
      pp.BackBufferFormat = D3DFMT_X8R8G8B8;
      pp.BackBufferCount = 1;
      pp.FullScreen_RefreshRateInHz = dm.RefreshRate;
      pp.PresentationInterval = D3DPRESENT_INTERVAL_ONE;
      pp.Flags = D3DPRESENTFLAG_LOCKABLE_BACKBUFFER;
      pp.SwapEffect = D3DSWAPEFFECT_DISCARD;
    }
    return SUCCEEDED(d3d->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, wnd, flags, &pp, &dev));
  }

  // Clears a fresh render target of `fmt` (MSAA `samples`) that was first filled with `prefill`, then reads it back.
  bool clear(D3DFORMAT fmt, uint32_t w, uint32_t h, uint32_t samples, uint32_t prefill, uint32_t flags, uint32_t argb,
             const D3DRECT* rect, std::vector<uint8_t>& out, uint32_t texel) {
    IDirect3DSurface9 *rt = nullptr, *ms = nullptr, *sys = nullptr;
    bool ok = SUCCEEDED(dev->CreateRenderTarget(w, h, fmt, D3DMULTISAMPLE_NONE, 0, FALSE, &rt, nullptr));
    if (ok && samples > 1)
      ok = SUCCEEDED(dev->CreateRenderTarget(w, h, fmt, D3DMULTISAMPLE_TYPE(samples), 0, FALSE, &ms, nullptr));
    ok = ok && SUCCEEDED(dev->CreateOffscreenPlainSurface(w, h, fmt, D3DPOOL_SYSTEMMEM, &sys, nullptr));
    if (ok) {
      IDirect3DSurface9* target = ms ? ms : rt;
      dev->SetRenderTarget(0, target);
      dev->SetDepthStencilSurface(nullptr);
      D3DVIEWPORT9 vp{ 0, 0, w, h, 0.0f, 1.0f };  // Do_ClearBuffer sets the viewport to buffer 0's size
      dev->SetViewport(&vp);
      ok = SUCCEEDED(dev->Clear(0, nullptr, D3DCLEAR_TARGET, prefill, 1.0f, 0)) &&
           SUCCEEDED(dev->Clear(rect ? 1 : 0, rect, flags, argb, 1.0f, 0));
      if (ok && ms) ok = SUCCEEDED(dev->StretchRect(ms, nullptr, rt, nullptr, D3DTEXF_NONE));  // EndUsingRenderTargetGroup
      ok = ok && SUCCEEDED(dev->GetRenderTargetData(rt, sys));
      D3DLOCKED_RECT lr{};
      if (ok && SUCCEEDED(sys->LockRect(&lr, nullptr, D3DLOCK_READONLY))) {
        out.resize(size_t(w) * h * texel);
        for (uint32_t y = 0; y < h; y++)
          memcpy(out.data() + size_t(y) * w * texel, static_cast<uint8_t*>(lr.pBits) + size_t(y) * lr.Pitch, size_t(w) * texel);
        sys->UnlockRect();
      } else {
        ok = false;
      }
    }
    if (sys) sys->Release();
    if (ms) ms->Release();
    if (rt) rt->Release();
    return ok;
  }
};

// --- helpers ----------------------------------------------------------------------------------------------------
static uint32_t texelOf(uint32_t d3dFmt) {
  switch (d3dFmt) {
    case kD3DFMT_A8R8G8B8: case kD3DFMT_X8R8G8B8: case kD3DFMT_R32F: return 4;
    case kD3DFMT_A16B16G16R16F: return 8;
    default: return 0;
  }
}

static uint32_t keyCounter = 0x1000;
static RtKey newKey() { return RtKey(keyCounter += 0x40); }

// Clears a backend target exactly as the game does and reads it back.
static bool backendClear(Backend& be, const RtDesc& d, uint32_t prefill, uint32_t flags, uint32_t argb, const int32_t* rect,
                         std::vector<uint8_t>& out) {
  RtKey k = newKey();
  be.attach(k, d);
  int32_t full[4] = { 0, 0, int32_t(d.width), int32_t(d.height) };
  be.beginGroup(&k, 1, 0);
  be.clear(kClearTarget, prefill, 1.0f, 0, full);
  be.clear(flags, argb, 1.0f, 0, rect ? rect : full);
  be.endGroup();
  uint32_t vf = 0, ts = 0;
  bool ok = be.readback(k, out, vf, ts);
  be.destroy(k);
  return ok;
}

static void testClearOracle(Backend& be, D3DRef& ref) {
  const uint32_t formats[] = { kD3DFMT_A8R8G8B8, kD3DFMT_X8R8G8B8, kD3DFMT_A16B16G16R16F, kD3DFMT_R32F };
  const uint32_t colors[] = { 0x00000000, 0xFFFFFFFF, 0x80C04020, 0x01FE7F80, 0x3F00FF10 };
  const int32_t rects[][4] = { { 3, 5, 40, 29 }, { -10, -10, 200, 200 }, { 50, 50, 60, 52 }, { 70, 2, 90, 10 } };
  const uint32_t w = 64, h = 48;
  int cases = 0;
  for (uint32_t fmt : formats) {
    for (uint32_t samples : { 1u, 4u }) {
      DWORD q = 0;
      if (samples > 1 && FAILED(ref.d3d->CheckDeviceMultiSampleType(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, D3DFORMAT(fmt),
                                                                    TRUE, D3DMULTISAMPLE_TYPE(samples), &q)))
        continue;
      for (uint32_t argb : colors) {
        for (int r = -1; r < int(std::size(rects)); r++) {
          const int32_t* rect = r < 0 ? nullptr : rects[r];
          D3DRECT dr{};
          if (rect) dr = { rect[0], rect[1], rect[2], rect[3] };
          std::vector<uint8_t> a, b;
          uint32_t ts = texelOf(fmt);
          uint32_t prefill = 0xFF102030;
          if (!ref.clear(D3DFORMAT(fmt), w, h, samples, prefill, D3DCLEAR_TARGET, argb, rect ? &dr : nullptr, a, ts)) {
            fail("D3D9 clear failed fmt %u samples %u", fmt, samples);
            continue;
          }
          RtDesc d{ w, h, fmt, samples, false };
          if (!backendClear(be, d, prefill, kClearTarget, argb, rect, b)) {
            fail("backend clear/readback failed fmt %u samples %u", fmt, samples);
            continue;
          }
          cases++;
          // X8R8G8B8: the X byte is undefined in D3D9 and forced to 1 on sampling in Vulkan; compare RGB only.
          size_t mism = 0, first = SIZE_MAX;
          for (size_t i = 0; i < a.size(); i++) {
            if (fmt == kD3DFMT_X8R8G8B8 && i % 4 == 3) continue;
            if (a[i] != b[i]) { mism++; if (first == SIZE_MAX) first = i; }
          }
          if (mism)
            fail("clear mismatch fmt %u samples %u argb %08X rect %d: %zu bytes differ (first at texel %zu: d3d %02X vk %02X)",
                 fmt, samples, argb, r, mism, first / ts, a[first], b[first]);
        }
      }
    }
  }
  printf("clear oracle: %d cases compared against D3D9\n", cases);
}

// StretchRect between render targets (NiDX9Renderer::Copy / FastCopy) against D3D9: a grid of cleared cells is
// copied same-size (sub-rect), stretched with POINT and with LINEAR.
static void testCopyOracle(Backend& be, D3DRef& ref) {
  const uint32_t w = 96, h = 64;
  struct C { const char* name; int32_t s[4], d[4]; uint32_t filter; };
  const C cs[] = { { "sub-rect same size, NONE", { 8, 8, 56, 40 }, { 20, 10, 68, 42 }, 0 },
                   { "whole surface, NONE", { 0, 0, 96, 64 }, { 0, 0, 96, 64 }, 0 },
                   { "2x stretch, POINT", { 0, 0, 48, 32 }, { 0, 0, 96, 64 }, 1 },
                   { "2x stretch, LINEAR", { 0, 0, 48, 32 }, { 0, 0, 96, 64 }, 2 },
                   { "shrink, LINEAR", { 0, 0, 96, 64 }, { 10, 10, 58, 42 }, 2 } };
  auto cellColor = [](int i) { return 0xFF000000u | ((i * 0x3A5F17u) & 0xFFFFFFu); };
  int identical = 0;
  for (uint32_t fmt : { kD3DFMT_A8R8G8B8, kD3DFMT_A16B16G16R16F }) {
    const uint32_t ts = texelOf(fmt);
    for (const C& c : cs) {
      std::vector<uint8_t> a;
      {
        IDirect3DSurface9 *src = nullptr, *dst = nullptr, *sys = nullptr;
        bool ok = SUCCEEDED(ref.dev->CreateRenderTarget(w, h, D3DFORMAT(fmt), D3DMULTISAMPLE_NONE, 0, FALSE, &src, nullptr)) &&
                  SUCCEEDED(ref.dev->CreateRenderTarget(w, h, D3DFORMAT(fmt), D3DMULTISAMPLE_NONE, 0, FALSE, &dst, nullptr)) &&
                  SUCCEEDED(ref.dev->CreateOffscreenPlainSurface(w, h, D3DFORMAT(fmt), D3DPOOL_SYSTEMMEM, &sys, nullptr));
        if (ok) {
          D3DVIEWPORT9 vp{ 0, 0, w, h, 0.0f, 1.0f };
          ref.dev->SetRenderTarget(0, src);
          ref.dev->SetDepthStencilSurface(nullptr);
          ref.dev->SetViewport(&vp);
          for (int i = 0; i < 32; i++) {
            D3DRECT r{ LONG((i % 8) * 12), LONG((i / 8) * 16), LONG((i % 8) * 12 + 12), LONG((i / 8) * 16 + 16) };
            ref.dev->Clear(1, &r, D3DCLEAR_TARGET, cellColor(i), 1.0f, 0);
          }
          ref.dev->SetRenderTarget(0, dst);
          ref.dev->Clear(0, nullptr, D3DCLEAR_TARGET, 0xFF102030, 1.0f, 0);
          RECT sr{ c.s[0], c.s[1], c.s[2], c.s[3] }, dr{ c.d[0], c.d[1], c.d[2], c.d[3] };
          ok = SUCCEEDED(ref.dev->StretchRect(src, &sr, dst, &dr, D3DTEXTUREFILTERTYPE(c.filter))) &&
               SUCCEEDED(ref.dev->GetRenderTargetData(dst, sys));
          D3DLOCKED_RECT lr{};
          if (ok && SUCCEEDED(sys->LockRect(&lr, nullptr, D3DLOCK_READONLY))) {
            a.resize(size_t(w) * h * ts);
            for (uint32_t y = 0; y < h; y++) memcpy(a.data() + size_t(y) * w * ts, static_cast<uint8_t*>(lr.pBits) + size_t(y) * lr.Pitch, size_t(w) * ts);
            sys->UnlockRect();
          }
        }
        if (sys) sys->Release();
        if (dst) dst->Release();
        if (src) src->Release();
        if (a.empty()) { fail("copy test '%s' fmt %u: D3D9 failed", c.name, fmt); continue; }
      }
      std::vector<uint8_t> b;
      RtKey src = newKey(), dst = newKey();
      be.attach(src, { w, h, fmt, 1, false });
      be.attach(dst, { w, h, fmt, 1, false });
      be.beginGroup(&src, 1, 0);
      for (int i = 0; i < 32; i++) {
        int32_t r[4] = { (i % 8) * 12, (i / 8) * 16, (i % 8) * 12 + 12, (i / 8) * 16 + 16 };
        be.clear(kClearTarget, cellColor(i), 1.0f, 0, r);
      }
      be.endGroup();
      int32_t full[4] = { 0, 0, int32_t(w), int32_t(h) };
      be.beginGroup(&dst, 1, 0);
      be.clear(kClearTarget, 0xFF102030, 1.0f, 0, full);
      be.endGroup();
      be.copyTarget(src, dst, c.s, c.d, c.filter);
      uint32_t vf = 0, bts = 0;
      if (!be.readback(dst, b, vf, bts) || b.size() != a.size()) { fail("copy test '%s' fmt %u: backend readback", c.name, fmt); continue; }
      be.destroy(src);
      be.destroy(dst);
      size_t diff = 0, first = SIZE_MAX;
      for (size_t i = 0; i < a.size(); i++)
        if (a[i] != b[i]) { diff++; if (first == SIZE_MAX) first = i; }
      if (diff)
        fail("copy test '%s' fmt %u: %zu bytes differ (first texel (%zu,%zu) byte %zu: d3d %02X vk %02X)", c.name, fmt, diff,
             (first / ts) % w, (first / ts) / w, first % ts, a[first], b[first]);
      else
        identical++;
    }
  }
  printf("copy oracle: %d of %zu StretchRect cases identical to D3D9\n", identical, 2 * std::size(cs));
}

// Bink's fixed-function blit (Setup_Renderstate / Blit_RAD_3D_image_pow2 / Submit_vertices) against the backend's
// screen-quad path.
static void testScreenQuadOracle(Backend& be, D3DRef& ref) {
  const uint32_t w = 96, h = 64, tw = 32, th = 32;
  std::vector<uint32_t> tex(tw * th);
  for (uint32_t y = 0; y < th; y++)
    for (uint32_t x = 0; x < tw; x++) tex[y * tw + x] = 0xFF000000u | (x * 8) << 16 | (y * 8) << 8 | ((x ^ y) & 1) * 0xC0;
  struct C { const char* name; float x0, y0, sx, sy; int32_t vp[4]; };
  const C cs[] = { { "1:1 at integer offset", 10, 6, 1, 1, { 0, 0, 96, 64 } },
                   { "2x scale", 4, 0, 2, 2, { 0, 0, 96, 64 } },
                   { "fractional offset, 1.5x", 10.25f, 3.75f, 1.5f, 1.5f, { 0, 0, 96, 64 } },
                   { "letterbox viewport", 0, 0, 3, 2, { 0, 8, 96, 56 } } };
  IDirect3DTexture9* dt = nullptr;
  if (FAILED(ref.dev->CreateTexture(tw, th, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &dt, nullptr))) { fail("quad test: texture"); return; }
  D3DLOCKED_RECT lr{};
  dt->LockRect(0, &lr, nullptr, 0);
  for (uint32_t y = 0; y < th; y++) memcpy(static_cast<uint8_t*>(lr.pBits) + y * lr.Pitch, &tex[y * tw], tw * 4);
  dt->UnlockRect(0);
  const TexKey tk = 0x7D000;
  be.updateTexture2D(tk, kD3DFMT_A8R8G8B8, tw, th, tex.data(), tw * 4);
  int identical = 0;
  for (const C& c : cs) {
    std::vector<uint8_t> a, b;
    IDirect3DSurface9 *rt = nullptr, *sys = nullptr;
    bool ok = SUCCEEDED(ref.dev->CreateRenderTarget(w, h, D3DFMT_A8R8G8B8, D3DMULTISAMPLE_NONE, 0, FALSE, &rt, nullptr)) &&
              SUCCEEDED(ref.dev->CreateOffscreenPlainSurface(w, h, D3DFMT_A8R8G8B8, D3DPOOL_SYSTEMMEM, &sys, nullptr));
    if (ok) {
      ref.dev->SetRenderTarget(0, rt);
      ref.dev->SetDepthStencilSurface(nullptr);
      D3DVIEWPORT9 full{ 0, 0, w, h, 0, 1 };
      ref.dev->SetViewport(&full);
      ref.dev->Clear(0, nullptr, D3DCLEAR_TARGET, 0xFF000000, 1.0f, 0);
      D3DVIEWPORT9 vp{ DWORD(c.vp[0]), DWORD(c.vp[1]), DWORD(c.vp[2] - c.vp[0]), DWORD(c.vp[3] - c.vp[1]), 0, 1 };
      ref.dev->SetViewport(&vp);
      ref.dev->SetVertexShader(nullptr);
      ref.dev->SetPixelShader(nullptr);
      ref.dev->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
      ref.dev->SetRenderState(D3DRS_ZENABLE, FALSE);
      ref.dev->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
      ref.dev->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
      ref.dev->SetRenderState(D3DRS_LIGHTING, FALSE);
      ref.dev->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
      ref.dev->SetTextureStageState(0, D3DTSS_ALPHAOP, D3DTOP_SELECTARG1);
      ref.dev->SetTextureStageState(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
      ref.dev->SetTextureStageState(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);
      ref.dev->SetTextureStageState(0, D3DTSS_COLOROP, D3DTOP_MODULATE);
      ref.dev->SetTextureStageState(1, D3DTSS_COLOROP, D3DTOP_DISABLE);
      ref.dev->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
      ref.dev->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
      ref.dev->SetSamplerState(0, D3DSAMP_MIPFILTER, D3DTEXF_LINEAR);
      ref.dev->SetSamplerState(0, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
      ref.dev->SetSamplerState(0, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
      ref.dev->SetTexture(0, dt);
      ref.dev->SetFVF(D3DFVF_XYZRHW | D3DFVF_TEX1);
      const float x1 = c.x0 + tw * c.sx, y1 = c.y0 + th * c.sy;
      const float v[24] = { c.x0, c.y0, 0, 1, 0, 0, c.x0, y1, 0, 1, 0, 1, x1, c.y0, 0, 1, 1, 0, x1, y1, 0, 1, 1, 1 };
      ref.dev->BeginScene();
      ok = SUCCEEDED(ref.dev->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, v, 24));
      ref.dev->EndScene();
      ref.dev->SetTexture(0, nullptr);
      ok = ok && SUCCEEDED(ref.dev->GetRenderTargetData(rt, sys));
      if (ok && SUCCEEDED(sys->LockRect(&lr, nullptr, D3DLOCK_READONLY))) {
        a.resize(size_t(w) * h * 4);
        for (uint32_t y = 0; y < h; y++) memcpy(a.data() + size_t(y) * w * 4, static_cast<uint8_t*>(lr.pBits) + size_t(y) * lr.Pitch, w * 4);
        sys->UnlockRect();
      }
    }
    if (sys) sys->Release();
    if (rt) rt->Release();
    if (a.empty()) { fail("quad test '%s': D3D9 failed", c.name); continue; }
    RtKey k = newKey();
    be.attach(k, { w, h, kD3DFMT_A8R8G8B8, 1, false });
    be.beginGroup(&k, 1, 0);
    int32_t full[4] = { 0, 0, int32_t(w), int32_t(h) };
    be.clear(kClearTarget, 0xFF000000, 1.0f, 0, full);
    Backend::ScreenQuad q{ tk, c.x0, c.y0, c.x0 + tw * c.sx, c.y0 + th * c.sy };
    if (!be.drawScreenQuads(&q, 1, c.vp)) fail("quad test '%s': drawScreenQuads refused", c.name);
    be.endGroup();
    uint32_t vf = 0, ts = 0;
    if (!be.readback(k, b, vf, ts) || b.size() != a.size()) { fail("quad test '%s': readback", c.name); continue; }
    be.destroy(k);
    size_t diff = 0, first = SIZE_MAX;
    int maxd = 0;
    for (size_t i = 0; i < a.size(); i++)
      if (a[i] != b[i]) { diff++; maxd = std::max(maxd, std::abs(int(a[i]) - int(b[i]))); if (first == SIZE_MAX) first = i; }
    if (diff)
      fail("quad test '%s': %zu bytes differ, max %d (first texel (%zu,%zu): d3d %02X vk %02X)", c.name, diff, maxd,
           (first / 4) % w, (first / 4) / w, a[first], b[first]);
    else
      identical++;
  }
  be.destroyTexture(tk);
  dt->Release();
  printf("screen-quad oracle: %d of %zu Bink blit cases identical to D3D9" "\n", identical, std::size(cs));
}

static void testDepthClear(Backend& be) {
  int cases = 0;
  for (float z : { 0.0f, 1.0f, 0.5f, 0.25f, 0.123456f }) {
    RtKey c = newKey(), dk = newKey();
    be.attach(c, { 32, 16, kD3DFMT_A8R8G8B8, 1, false });
    be.attach(dk, { 32, 24, kD3DFMT_D24S8, 1, true });  // depth larger than color, as the game's shared depth buffer
    int32_t rect[4] = { 0, 0, 32, 16 };
    be.beginGroup(&c, 1, dk);
    be.clear(kClearZBuffer | kClearStencil, 0, 1.0f, 0x00, rect);  // everything inside the render area to 1.0
    be.clear(kClearZBuffer | kClearStencil, 0, z, 0x5A, rect);
    be.endGroup();
    std::vector<uint8_t> t;
    uint32_t vf = 0, ts = 0;
    if (!be.readback(dk, t, vf, ts)) { fail("depth readback failed"); continue; }
    cases++;
    for (uint32_t y = 0; y < 16; y++) {
      uint32_t v;
      memcpy(&v, t.data() + (y * 32 + 7) * 4, 4);
      if (vf == VK_FORMAT_D24_UNORM_S8_UINT) {
        uint32_t got = v & 0xFFFFFF, want = uint32_t(std::lround(double(z) * 16777215.0));
        // Vulkan allows the implementation's float->unorm conversion a tolerance; D3D9 rounds to nearest.
        if (got + 1 < want || got > want + 1) { fail("D24 depth clear z=%f: got %u want %u", z, got, want); break; }
      } else {
        float f;
        memcpy(&f, &v, 4);
        if (f != z) { fail("D32 depth clear z=%f: got %f", z, f); break; }
      }
    }
    be.destroy(c);
    be.destroy(dk);
  }
  printf("depth clears: %d cases\n", cases);
}

// The game's in-world frame (docs/M0_RESULTS.md, M2_RESULTS.md): reflection, copies, sky, shadow depth, HDR scene,
// downsample chain, image space, interface to the back buffer.
static void testFrameChurn(Backend& be, RtKey backbuffer, RtKey bbDepth, uint32_t frames) {
  struct T { RtKey key; RtDesc d; RtKey depth; };
  std::vector<T> targets;
  RtKey shadowDepth = newKey();
  be.attach(shadowDepth, { 1024, 1024, kD3DFMT_D24S8, 1, true });
  auto add = [&](uint32_t w, uint32_t h, uint32_t fmt, RtKey depth, uint32_t samples = 1) {
    T t{ newKey(), { w, h, fmt, samples, false }, depth };
    be.attach(t.key, t.d);
    targets.push_back(t);
  };
  add(1024, 1024, kD3DFMT_A8R8G8B8, shadowDepth);
  add(1024, 1024, kD3DFMT_R32F, bbDepth);  // shares the default depth buffer, which is larger in one axis
  // HDR scene with its own 1176x664 depth (as logged in game). MSAA is exercised with matching sample counts; which
  // in-game targets are multisampled is measured in game, not assumed here.
  RtKey hdrDepth = newKey();
  be.attach(hdrDepth, { 1176, 664, kD3DFMT_D24S8, 4, true });
  add(1176, 664, kD3DFMT_A16B16G16R16F, hdrDepth, 4);
  for (auto [w, h] : { std::pair{ 588u, 256u }, { 256u, 256u }, { 64u, 64u }, { 16u, 16u }, { 4u, 4u }, { 1u, 1u } })
    add(w, h, kD3DFMT_A16B16G16R16F, 0);
  add(1176, 664, kD3DFMT_A8R8G8B8, 0);
  add(294, 166, kD3DFMT_A8R8G8B8, 0);
  std::mt19937 rng(7);
  for (uint32_t f = 0; f < frames; f++) {
    for (T& t : targets) {
      be.beginGroup(&t.key, 1, t.depth);
      int32_t r[4] = { 0, 0, int32_t(t.d.width), int32_t(t.d.height) };
      be.clear(kClearTarget | kClearZBuffer | kClearStencil, rng(), 1.0f, 0, r);
      be.endGroup();
    }
    // The game recreates image-space targets during play.
    if (f % 7 == 3) {
      T& t = targets.back();
      be.destroy(t.key);
      t.key = newKey();
      be.attach(t.key, t.d);
    }
    be.beginGroup(&backbuffer, 1, bbDepth);
    int32_t r[4] = { 0, 0, 1176, 664 };
    be.clear(kClearTarget | kClearZBuffer, 0xFF000000 | (f * 3), 1.0f, 0, r);
    be.endGroup();
    be.present(backbuffer);
  }
  for (T& t : targets) be.destroy(t.key);
  be.destroy(shadowDepth);
  be.destroy(hdrDepth);
}

// Textures: native D3DX-equivalent load, upload, then every subresource read back from Vulkan must equal the
// uploaded bytes (BC blocks and packed texels are copied verbatim).
static void testTextures(Backend& be, const char* dir) {
  namespace fs = std::filesystem;
  int files = 0;
  uint64_t subresources = 0;
  TexKey key = 0x10000;
  for (auto& e : fs::directory_iterator(dir)) {
    if (_wcsicmp(e.path().extension().c_str(), L".dds") != 0) continue;
    FILE* f = _wfopen(e.path().c_str(), L"rb");
    if (!f) continue;
    std::vector<uint8_t> data;
    fseek(f, 0, SEEK_END);
    data.resize(size_t(ftell(f)));
    fseek(f, 0, SEEK_SET);
    fread(data.data(), 1, data.size(), f);
    fclose(f);
    fnvvk::d3dx::Texture tex;
    if (fnvvk::d3dx::createTextureFromDds(data.data(), data.size(), fnvvk::d3dx::Precision::Single, tex) !=
        fnvvk::d3dx::Status::Ok) {
      fail("native load failed: %s", e.path().filename().string().c_str());
      continue;
    }
    key += 0x40;
    if (!be.createTexture(key, tex)) { fail("createTexture failed: %s", e.path().filename().string().c_str()); continue; }
    files++;
    uint32_t faces = tex.cube ? 6 : 1;
    for (uint32_t fc = 0; fc < faces; fc++)
      for (uint32_t l = 0; l < tex.levels; l++) {
        std::vector<uint8_t> back;
        const auto& src = tex.surfaces[size_t(fc) * tex.levels + l];
        if (!be.readbackTexture(key, fc, l, back) || back != src.data)
          fail("texture readback differs: %s face %u level %u (%zu vs %zu bytes)", e.path().filename().string().c_str(),
               fc, l, back.size(), src.data.size());
        subresources++;
      }
    // Replace once (re-load path), then destroy.
    if (files % 5 == 0 && !be.createTexture(key, tex)) fail("texture replace failed");
    if (!be.destroyTexture(key)) fail("destroyTexture failed");
  }
  printf("textures: %d uploaded, %llu subresources read back and compared\n", files, subresources);
}

#include "draw_test.inl"
#include "hdr_test.inl"
#include "replace_test.inl"
#include "../../plugin/src/cascades.h"
#include "shadow_test.inl"
#include "shadow_scene_test.inl"
#include "sunlight_test.inl"
#include "pbr_test.inl"
#include "sunshafts_test.inl"
#include "cube_test.inl"
#include "ssgi_test.inl"
#include "atmos_test.inl"
#include "../../plugin/src/puddlemap.h"
#include "wet_test.inl"
#include "../../plugin/src/bc5dds.h"
#include "skin_test.inl"
#include "prepass_test.inl"
#include "sss_test.inl"
#include "perf_test.inl"
#include "grass_test.inl"
#include "fsr_test.inl"
#include "dofmb_test.inl"

// usage: backend_test [texture dir | -] [sdp2spv cache dir (manifest.json, spv/)] [shader pack]
static LONG CALLBACK firstChance(EXCEPTION_POINTERS* ep) {
  DWORD code = ep->ExceptionRecord->ExceptionCode;
  if (code != EXCEPTION_ACCESS_VIOLATION && code != 0xC0000409 && code != 0xC0000374) return EXCEPTION_CONTINUE_SEARCH;
  HMODULE m = nullptr;
  char name[MAX_PATH] = "?";
  void* at = ep->ExceptionRecord->ExceptionAddress;
  if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         static_cast<LPCSTR>(at), &m))
    GetModuleFileNameA(m, name, MAX_PATH);
  printf("FIRST-CHANCE %08lX at %s+0x%X params %08lX %08lX\n", code, strrchr(name, '\\') ? strrchr(name, '\\') + 1 : name,
         unsigned(uintptr_t(at) - uintptr_t(m)),
         ep->ExceptionRecord->NumberParameters > 0 ? ULONG(ep->ExceptionRecord->ExceptionInformation[0]) : 0,
         ep->ExceptionRecord->NumberParameters > 1 ? ULONG(ep->ExceptionRecord->ExceptionInformation[1]) : 0);
#if defined(_M_IX86)
  // The caller of a call through a null pointer, and the next frames (32-bit, frame pointers), with their symbols.
  static bool symbols = SymInitialize(GetCurrentProcess(), nullptr, TRUE) != FALSE;
  auto where = [&](void* a, const char* what) {
    HMODULE mod = nullptr;
    char mn[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCSTR>(a), &mod))
      GetModuleFileNameA(mod, mn, MAX_PATH);
    alignas(SYMBOL_INFO) char sb[sizeof(SYMBOL_INFO) + 256] = {};
    auto* si = reinterpret_cast<SYMBOL_INFO*>(sb);
    si->SizeOfStruct = sizeof(SYMBOL_INFO);
    si->MaxNameLen = 255;
    DWORD64 disp = 0;
    const bool named = symbols && SymFromAddr(GetCurrentProcess(), DWORD64(uintptr_t(a)), &disp, si);
    const char* base = strrchr(mn, '\\');
    printf("  %s %s+0x%X %s+0x%llX\n", what, base ? base + 1 : mn, unsigned(uintptr_t(a) - uintptr_t(mod)),
           named ? si->Name : "?", (unsigned long long)disp);
  };
  if (ep->ContextRecord && ep->ContextRecord->Esp) where(*reinterpret_cast<void**>(ep->ContextRecord->Esp), "return to");
  uintptr_t ebp = ep->ContextRecord ? ep->ContextRecord->Ebp : 0;
  for (int i = 0; i < 8 && ebp && !IsBadReadPtr(reinterpret_cast<void*>(ebp), 8); i++) {
    where(reinterpret_cast<void**>(ebp)[1], "frame");
    ebp = reinterpret_cast<uintptr_t*>(ebp)[0];
  }
#endif
  return EXCEPTION_CONTINUE_SEARCH;
}

int main(int argc, char** argv) {
  AddVectoredExceptionHandler(1, firstChance);
  setvbuf(stdout, nullptr, _IONBF, 0);
  // backend_test --compile-replacements <replacements dir> <out dir>: compile every replacement the way the plugin does
  // (loadReplacements) and write each target's SPIR-V to <out dir>/<replacement id>/<shader name>.spv, for
  // tools/replacediff.py. Exit code 1 if any replacement failed to compile.
  if (argc == 4 && strcmp(argv[1], "--compile-replacements") == 0) {
    const std::vector<fx::Replacement> all =
        fx::loadReplacements(argv[2], "build/fx_cache_test", [](const char* s) { printf("[repl] %s\n", s); });
    bool ok = true;
    for (const fx::Replacement& r : all) {
      ok &= r.error.empty();
      const std::string dir = std::string(argv[3]) + "\\" + r.id;
      CreateDirectoryA(argv[3], nullptr);
      CreateDirectoryA(dir.c_str(), nullptr);
      for (const fx::Replacement::Target& t : r.targets)
        if (FILE* f = fopen((dir + "\\" + t.shader + ".spv").c_str(), "wb")) {
          fwrite(t.spv.data(), 4, t.spv.size(), f);
          fclose(f);
        }
    }
    return ok ? 0 : 1;
  }
  // backend_test --depth-variants <shader pack> <shader cache> <replacements dir> <out dir>: the depth pre-pass's depth
  // and colour modules (spvDepthVariant, spvColourVariant) of every pixel shader of the pack (named by the cache's
  // spv\*.dxso) and of every replacement target, written to <out dir> (refused ones listed), for spirv-val. Exit code 1
  // if none were written.
  if (argc == 6 && strcmp(argv[1], "--depth-variants") == 0) {
    ShaderPack pack;
    std::string err;
    if (!pack.load(argv[2], &err)) { printf("pack: %s\n", err.c_str()); return 1; }
    CreateDirectoryA(argv[5], nullptr);
    uint32_t written = 0, refused = 0;
    // <name>.spv: the depth module, <name>.colour.spv: the colour module (spvColourVariant).
    auto variant = [&](const uint32_t* spv, size_t words, const std::string& name) {
      std::vector<uint32_t> depth, colour;
      const bool d = spvDepthVariant(spv, words, depth), c = spvColourVariant(spv, words, colour);
      if (!d || !c) {
        refused++;
        printf("refused: %s (depth module %d, colour module %d)\n", name.c_str(), int(d), int(c));
      }
      auto write = [&](const std::vector<uint32_t>& v, const std::string& file) {
        if (FILE* f = fopen((std::string(argv[5]) + "\\" + file).c_str(), "wb")) {
          fwrite(v.data(), 4, v.size(), f);
          fclose(f);
          written++;
        }
      };
      if (d) write(depth, name + ".spv");
      if (c) write(colour, name + ".colour.spv");
    };
    WIN32_FIND_DATAA fd;
    const std::string spvDir = std::string(argv[3]) + "\\spv\\";
    HANDLE h = FindFirstFileA((spvDir + "*.dxso").c_str(), &fd);
    if (h != INVALID_HANDLE_VALUE) {
      do {
        const std::string name = fd.cFileName;
        Md5 m;
        if (name.size() != 37 || !drawtest::hexMd5(name.substr(0, 32), m.b.data())) continue;
        const PackedShader* p = pack.find(m);
        if (p && p->pixel) variant(p->spv, p->spvBytes / 4, name.substr(0, 32));
      } while (FindNextFileA(h, &fd));
      FindClose(h);
    }
    for (const fx::Replacement& r : fx::loadReplacements(argv[4], "build/fx_cache_test", [](const char*) {}))
      for (const fx::Replacement::Target& t : r.targets)
        if (t.spv.size() > 5) variant(t.spv.data(), t.spv.size(), r.id + "_" + t.shader);  // (vertex targets: refused)
    printf("depth variants: %u written, %u refused\n", written, refused);
    return written ? 0 : 1;
  }
  // FNV_HIGH_ADDRESSES=1: reserve the low 2 GiB first (the game is large-address-aware, and its address space fills
  // up), so allocations made from here on land above 0x80000000 as they can in game.
  if (getenv("FNV_HIGH_ADDRESSES")) {
    size_t reserved = 0;
    for (size_t sz = size_t(64) << 20; sz >= 4096; sz /= 2)
      while (void* p = VirtualAlloc(nullptr, sz, MEM_RESERVE, PAGE_NOACCESS)) {
        if (uintptr_t(p) >= 0x80000000u) { VirtualFree(p, 0, MEM_RELEASE); break; }
        reserved += sz;
      }
    printf("reserved %zu MiB below 2 GiB\n", reserved >> 20);
  }
  D3DRef ref;
  if (!ref.init()) { printf("FAIL: D3D9 device\n"); return 2; }

  Backend be;
  BackendConfig cfg;
  cfg.shadowWindow = true;
  cfg.validation = true;
  cfg.wideTargets = getenv("FNV_WIDE_TARGETS") != nullptr;  // the whole suite on wide targets
  cfg.recordThread = getenv("FNVVK_RECORD_THREAD") != nullptr;  // the whole suite through the record thread
  if (argc > 3) cfg.shaderPack = argv[3];
  repltest::prepare(cfg);  // shader replacements are part of the configuration (applied as the pack loads)
  if (const char* pf = getenv("FNV_PIPELINE_FILE")) cfg.pipelineFile = pf;  // persistent pipelines (save/load/prebuild)
  cfg.log = [](const char* s) { printf("[backend] %s\n", s); };

  // Back buffer and default depth are attached before init, as in the game (they are created inside
  // NiDX9Renderer::Create, before the backend initialises).
  RtKey bb = newKey(), bbDepth = newKey();
  be.attach(bb, { 1176, 664, kD3DFMT_X8R8G8B8, 1, false });
  be.attach(bbDepth, { 1176, 1024, kD3DFMT_D24S8, 1, true });
  if (!be.init(cfg, 1176, 664)) { printf("FAIL: backend init\n"); return 2; }

  testClearOracle(be, ref);
  testCopyOracle(be, ref);
  testScreenQuadOracle(be, ref);
  testDepthClear(be);
  testFrameChurn(be, bb, bbDepth, 120);

  // Resize the shadow window mid-run: the swapchain must be recreated without errors.
  HWND shadow = FindWindowA("fnv_vk_shadow", nullptr);
  if (!shadow) fail("shadow window not found");
  else SetWindowPos(shadow, nullptr, 0, 0, 800, 500, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
  uint64_t before = be.stats().swapchains;
  testFrameChurn(be, bb, bbDepth, 60);
  if (be.stats().swapchains <= before) fail("window resize did not recreate the swapchain");

  // Implicit back buffer resize (device reset path: Recreate -> Ni2DBuffer::ResetDimensions).
  be.resize(bb, 1280, 720);
  be.resize(bbDepth, 1280, 1024);
  testFrameChurn(be, bb, bbDepth, 10);

  // In-game menu: ImGui frames drawn over presented frames (the plugin's sequence: overlayBeginFrame, NewFrame, widgets,
  // Render, overlaySubmit, present). The context stays alive: the backend's ImGui renderer shuts down with the backend.
  {
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.IniFilename = nullptr;
    const uint64_t before = be.stats().overlayFrames;
    int drawn = 0;
    for (int f = 0; f < 8; f++) {
      uint32_t w = 0, h = 0;
      if (!be.overlayBeginFrame(w, h)) { fail("menu: overlayBeginFrame failed (frame %d)", f); break; }
      io.DisplaySize = ImVec2(float(w), float(h));
      io.DeltaTime = 1.0f / 60.0f;
      io.AddMousePosEvent(float(w) * 0.5f, float(h) * 0.5f);
      io.MouseDrawCursor = true;
      ImGui::NewFrame();
      ImGui::Begin("fnv_vk test menu");
      ImGui::Text("frame %d", f);
      static float slider = 0.5f;
      ImGui::SliderFloat("value", &slider, 0.0f, 1.0f);
      static bool box = true;
      ImGui::Checkbox("enabled", &box);
      ImGui::End();
      ImGui::Render();
      be.overlaySubmit(true);
      be.beginGroup(&bb, 1, bbDepth);
      int32_t r[4] = { 0, 0, 1280, 720 };
      be.clear(kClearTarget | kClearZBuffer, 0xFF102030, 1.0f, 0, r);
      be.endGroup();
      be.present(bb);
      drawn++;
    }
    const uint64_t frames = be.stats().overlayFrames - before;
    printf("menu: %llu of %d presents drew the ImGui overlay\n", frames, drawn);
    if (frames != uint64_t(drawn)) fail("menu: overlay drawn on %llu of %d presents", frames, drawn);
  }
  hdrtest::run(be, bb, bbDepth);
  repltest::run(be);
  shadowtest::run(be);
  shadowscene::run(be);
  suntest::run(be);
  pbrtest::run(be);
  pbrterrain::run(be);
  sunshaftstest::run(be);
  cubetest::run(be);
  envtest::run();
  prefiltertest::run(be);
  ssgitest::run(be);
  atmostest::run(be);
  wettest::run(be);
  skintest::run(be);
  prepasstest::run(be, argc > 3 ? argv[3] : nullptr, argc > 2 ? argv[2] : nullptr);
  ssstest::run(be);
  perftest::run(be);
  grasstest::run(be);
  fsrtest::run(be);
  dofmbtest::run(be);
  if (argc > 1 && strcmp(argv[1], "-") != 0) testTextures(be, argv[1]);
  if (getenv("FNV_NO_2MB_BLOCKS")) {
    // Leave no free address-space block of 2 MiB or more (what the game's 2 GiB space looks like when full).
    size_t n = 0;
    while (VirtualAlloc(nullptr, size_t(2) << 20, MEM_RESERVE, PAGE_NOACCESS)) n++;
    printf("reserved %zu blocks of 2 MiB; no 2 MiB block left\n", n);
  }
  if (argc > 3) drawtest::run(be, ref, argv[2]);

  be.finishFrame();
  BackendStats s = be.stats();
  printf("stats: frames %llu groups %llu clears %llu resolves %llu attached %llu destroyed %llu resized %llu realized %llu "
         "live %u (realized %u) unknownKeys %llu sampleMismatches %llu sizeMismatches %llu unsupported %llu vkErrors %llu "
         "validation %llu swapchains %llu\n",
         s.frames, s.groups, s.clears, s.resolves, s.attached, s.destroyed, s.resized, s.realized, s.live, s.liveRealized,
         s.unknownKeys, s.sampleMismatches, s.sizeMismatches, s.unsupportedFormats, s.vkErrors, s.validationMessages, s.swapchains);
  if (s.liveTextures != 0) fail("%u textures leaked", s.liveTextures);
  if (s.texturesCreated != s.texturesDestroyed + 0 && s.texturesCreated < s.texturesDestroyed) fail("texture counts");
  if (s.live != 2) fail("expected 2 live targets (back buffer + depth), got %u", s.live);
  if (s.attached != s.destroyed + s.live) fail("lifetime imbalance: attached %llu destroyed %llu live %u", s.attached, s.destroyed, s.live);
  if (s.unknownKeys || s.sampleMismatches || s.sizeMismatches || s.unsupportedFormats) fail("backend reported anomalies");
  if (s.vkErrors) fail("%llu Vulkan errors", s.vkErrors);
  if (s.validationMessages) fail("%llu validation messages", s.validationMessages);
  be.shutdown();
  printf(g_failures ? "RESULT: %d failure(s)\n" : "RESULT: PASS\n", g_failures);
  return g_failures ? 1 : 0;
}
