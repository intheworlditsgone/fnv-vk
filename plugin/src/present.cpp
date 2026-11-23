// Vulkan presents into the game's window ([Backend] Present=1).
// - NiDX9Renderer::InitializePresentParams (0x00E6BF50, the only place the device's D3DPRESENT_PARAMETERS are built,
//   called from NiDX9Renderer::Initialize): records the device window and presentation interval, and clears the
//   fullscreen flag (uiUseFlags & 4) so D3D9 creates a windowed device; a fullscreen request becomes a borderless
//   window covering its monitor (Vulkan cannot present to a window D3D9 holds exclusively).
// - NiDX9ImplicitBufferData::DisplayFrame (0x00E7CCD0, the onscreen buffer's slot-20 virtual called by
//   Do_DisplayFrame; its only code is the device Present): replaced so D3D9 never presents; the backend presents the
//   same onscreen render target from shadow::onDisplayFrame.
// - Window geometry in fullscreen. The D3D9 runtime owns a fullscreen device's window: it keeps the window covering the
//   display mode and restores it on reactivation. The game relies on that. Its main loop (WinMain 0x0086A850) calls,
//   whenever the window loses or regains the foreground with bFull Screen set,
//     AdjustWindowRectEx(&r = {0, 0, width, height}, style, 0, 0);
//     SetWindowPos(wnd, 0, r.left, r.bottom, r.right - r.left, r.top - r.bottom, 0);
//   That is y = height and cx/cy = width/-height: the window moves below the monitor with no height. Under the D3D9
//   runtime the exclusive-mode window handling put it back; with a borderless window nothing did, so after alt-tab the
//   game kept running (audio) in an invisible window. This module takes the runtime's role: while fullscreen was
//   requested, every move or resize of the window, except minimising, is held to the monitor rectangle.
#include <windows.h>
#include <commctrl.h>

#include <cstdint>

#include "common.h"
#include "present.h"

namespace fnvvk::present {
namespace {

bool g_on = false;
HWND g_wnd = nullptr;
bool g_vsync = false;
bool g_fullscreenRequested = false;
RECT g_monitorRect{};

LRESULT CALLBACK fullscreenProc(HWND h, UINT msg, WPARAM wp, LPARAM lp, UINT_PTR, DWORD_PTR) {
  if (msg == WM_WINDOWPOSCHANGING && !IsIconic(h)) {
    auto* pos = reinterpret_cast<WINDOWPOS*>(lp);
    const bool minimizing = pos->x <= -32000 && pos->y <= -32000;
    if (!minimizing && (pos->flags & (SWP_NOMOVE | SWP_NOSIZE)) != (SWP_NOMOVE | SWP_NOSIZE)) {
      pos->x = g_monitorRect.left;
      pos->y = g_monitorRect.top;
      pos->cx = g_monitorRect.right - g_monitorRect.left;
      pos->cy = g_monitorRect.bottom - g_monitorRect.top;
      pos->flags &= ~UINT(SWP_NOMOVE | SWP_NOSIZE);
    }
  }
  return DefSubclassProc(h, msg, wp, lp);
}

using InitPPFn = bool(__fastcall*)(void* self, void* edx, HWND wnd, UINT w, UINT h, UINT useFlags, UINT fbMode,
                                   UINT fbFormat, UINT dsFormat, UINT backBuffers, UINT swapEffect, UINT refresh,
                                   UINT interval, void* pp);
InitPPFn oInitPP = nullptr;

bool __fastcall hkInitPP(void* self, void* edx, HWND wnd, UINT w, UINT h, UINT useFlags, UINT fbMode, UINT fbFormat,
                         UINT dsFormat, UINT backBuffers, UINT swapEffect, UINT refresh, UINT interval, void* pp) {
  g_fullscreenRequested = (useFlags & 4) != 0;
  const bool r = oInitPP(self, edx, wnd, w, h, useFlags & ~4u, fbMode, fbFormat, dsFormat, backBuffers, swapEffect,
                         refresh, interval, pp);
  g_wnd = wnd;
  // D3DPRESENT_PARAMETERS::PresentationInterval (+0x34): 0x80000000 = IMMEDIATE, otherwise synchronised.
  const uint32_t pi = *reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(pp) + 0x34);
  g_vsync = pi != 0x80000000u;
  if (g_fullscreenRequested && wnd) {
    HMONITOR mon = MonitorFromWindow(wnd, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFO mi{ sizeof(mi) };
    GetMonitorInfoA(mon, &mi);
    g_monitorRect = mi.rcMonitor;
    SetWindowSubclass(wnd, &fullscreenProc, 1, 0);
    SetWindowLongA(wnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);
    SetWindowLongA(wnd, GWL_EXSTYLE, 0);
    SetWindowPos(wnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top, mi.rcMonitor.right - mi.rcMonitor.left,
                 mi.rcMonitor.bottom - mi.rcMonitor.top, SWP_FRAMECHANGED | SWP_SHOWWINDOW);
  }
  logf("present: device window %p, %ux%u, fullscreen requested %d -> windowed device%s, interval %08X (vsync %d)", wnd,
       w, h, int(g_fullscreenRequested), g_fullscreenRequested ? " (borderless on its monitor)" : "", pi, int(g_vsync));
  return r;
}

// NiDX9Renderer::Create 0x00E76210 (cdecl, 15 args; the last, useD3D9ex, is set by TESRenderer::Init only for
// fullscreen on a multi-monitor system at a non-desktop size). Forced on in native mode: a D3D9Ex device is never
// lost and keeps D3DPOOL_DEFAULT resources across Reset, and the engine already runs its own Ex paths for it.
// The argument is set in main.cpp's NiDX9Renderer::Create detour (one detour per function).
bool g_forceEx = false;

using DisplayFn = bool(__fastcall*)(void* self, void* edx);
DisplayFn oDisplay = nullptr;
bool __fastcall hkDisplay(void*, void*) { return true; }  // D3D9 does not present; the backend does

}  // namespace

bool install(bool on, bool forceEx) {
  g_on = on;
  if (!on) return true;
  g_forceEx = forceEx;
  bool ok = hook(0x00E6BF50, &hkInitPP, &oInitPP, "NiDX9Renderer::InitializePresentParams (windowed device)");
  ok &= hook(0x00E7CCD0, &hkDisplay, &oDisplay, "NiDX9ImplicitBufferData::DisplayFrame (no D3D9 present)");
  return ok;
}
bool enabled() { return g_on; }
void* window() { return g_wnd; }
bool vsync() { return g_vsync; }
bool forceD3D9Ex() { return g_on && g_forceEx; }

}  // namespace fnvvk::present
