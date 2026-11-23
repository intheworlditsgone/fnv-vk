// HDR output settings and the monitor's HDR state (display.h).
//
// Windows HDR state and peak brightness: IDXGIOutput6::GetDesc1 of the output whose HMONITOR is the game window's:
// ColorSpace DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 when HDR is on; MaxLuminance in nits.
// SDR content brightness (what Windows shows SDR white at in HDR): DisplayConfigGetDeviceInfo with
// DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL for the path whose source is that output; SDRWhiteLevel is in units
// of 80/1000 nits.
#include "display.h"

#include <windows.h>
#include <dxgi1_6.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "backend.h"
#include "common.h"

namespace fnvvk::display {
namespace {

Settings g_settings;
Monitor g_monitor;
std::string g_ini;

Monitor queryMonitor(HWND wnd) {
  Monitor m;
  const HMONITOR mon = MonitorFromWindow(wnd, MONITOR_DEFAULTTOPRIMARY);
  using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);
  HMODULE dxgi = LoadLibraryA("dxgi.dll");
  auto create = dxgi ? reinterpret_cast<CreateFactoryFn>(GetProcAddress(dxgi, "CreateDXGIFactory1")) : nullptr;
  IDXGIFactory1* factory = nullptr;
  if (!create || FAILED(create(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) || !factory) return m;
  WCHAR deviceName[32] = L"";
  IDXGIAdapter1* adapter = nullptr;
  for (UINT a = 0; !m.known && factory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; a++) {
    IDXGIOutput* output = nullptr;
    for (UINT o = 0; !m.known && adapter->EnumOutputs(o, &output) != DXGI_ERROR_NOT_FOUND; o++) {
      IDXGIOutput6* out6 = nullptr;
      if (SUCCEEDED(output->QueryInterface(__uuidof(IDXGIOutput6), reinterpret_cast<void**>(&out6))) && out6) {
        DXGI_OUTPUT_DESC1 d{};
        if (SUCCEEDED(out6->GetDesc1(&d)) && d.Monitor == mon) {
          m.known = true;
          m.hdrOn = d.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
          m.peakNits = d.MaxLuminance;
          wcsncpy_s(deviceName, d.DeviceName, _TRUNCATE);
        }
        out6->Release();
      }
      output->Release();
    }
    adapter->Release();
  }
  factory->Release();
  if (!m.known) return m;

  UINT32 nPaths = 0, nModes = 0;
  if (GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, &nPaths, &nModes) != ERROR_SUCCESS) return m;
  std::vector<DISPLAYCONFIG_PATH_INFO> paths(nPaths);
  std::vector<DISPLAYCONFIG_MODE_INFO> modes(nModes);
  if (QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, &nPaths, paths.data(), &nModes, modes.data(), nullptr) != ERROR_SUCCESS)
    return m;
  for (UINT32 i = 0; i < nPaths; i++) {
    DISPLAYCONFIG_SOURCE_DEVICE_NAME src{};
    src.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME;
    src.header.size = sizeof(src);
    src.header.adapterId = paths[i].sourceInfo.adapterId;
    src.header.id = paths[i].sourceInfo.id;
    if (DisplayConfigGetDeviceInfo(&src.header) != ERROR_SUCCESS || wcscmp(src.viewGdiDeviceName, deviceName) != 0) continue;
    DISPLAYCONFIG_SDR_WHITE_LEVEL white{};
    white.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL;
    white.header.size = sizeof(white);
    white.header.adapterId = paths[i].targetInfo.adapterId;
    white.header.id = paths[i].targetInfo.id;
    if (DisplayConfigGetDeviceInfo(&white.header) == ERROR_SUCCESS) m.sdrWhiteNits = float(white.SDRWhiteLevel) * 80.0f / 1000.0f;
    break;
  }
  return m;
}

float iniFloat(const char* key, float def) {
  char buf[64] = "";
  GetPrivateProfileStringA("Display", key, "", buf, sizeof(buf), g_ini.c_str());
  return buf[0] ? float(atof(buf)) : def;
}

}  // namespace

Settings& settings() { return g_settings; }
const Monitor& monitor() { return g_monitor; }

void load(void* window) {
  g_ini = iniPath();
  char buf[32] = "";
  GetPrivateProfileStringA("Display", "HDR", "auto", buf, sizeof(buf), g_ini.c_str());
  const std::string v = buf;
  g_settings.hdr = v == "1" || _stricmp(buf, "on") == 0 ? Settings::On : v == "0" || _stricmp(buf, "off") == 0 ? Settings::Off : Settings::Auto;
  GetPrivateProfileStringA("Display", "HDRFormat", "hdr10", buf, sizeof(buf), g_ini.c_str());
  g_settings.format = _stricmp(buf, "scrgb") == 0 ? Settings::ScRgb : Settings::Hdr10;
  g_settings.peakNits = std::max(0.0f, iniFloat("PeakNits", 0.0f));
  g_settings.paperWhiteNits = std::max(0.0f, iniFloat("PaperWhiteNits", 0.0f));
  if (window) g_monitor = queryMonitor(static_cast<HWND>(window));
  logf("display: Windows HDR %s, peak %.0f nits, SDR content brightness %.0f nits%s | [Display] HDR=%s format %s, peak %s, "
       "paper white %s",
       g_monitor.hdrOn ? "on" : "off", g_monitor.peakNits, g_monitor.sdrWhiteNits, g_monitor.known ? "" : " (monitor not found)",
       g_settings.hdr == Settings::Auto ? "auto" : g_settings.hdr == Settings::On ? "on" : "off",
       g_settings.format == Settings::ScRgb ? "scRGB" : "HDR10", g_settings.peakNits > 0 ? "set" : "display",
       g_settings.paperWhiteNits > 0 ? "set" : "Windows");
}

bool hdrWanted() {
  return g_settings.hdr == Settings::On || (g_settings.hdr == Settings::Auto && g_monitor.hdrOn);
}
float effectivePeak() {
  if (g_settings.peakNits > 0) return g_settings.peakNits;
  return g_monitor.peakNits >= 100.0f ? g_monitor.peakNits : 1000.0f;
}
float effectivePaperWhite() {
  if (g_settings.paperWhiteNits > 0) return g_settings.paperWhiteNits;
  return g_monitor.sdrWhiteNits >= 40.0f ? g_monitor.sdrWhiteNits : 200.0f;
}

void apply(Backend* be) {
  if (!be) return;
  HdrSettings h;
  h.output = !hdrWanted() ? HdrSettings::Off : g_settings.format == Settings::ScRgb ? HdrSettings::ScRgb : HdrSettings::Hdr10;
  h.paperWhiteNits = effectivePaperWhite();
  h.peakNits = std::max(effectivePeak(), h.paperWhiteNits);
  be->setHdr(h);
}

bool save() {
  char v[32];
  bool ok = WritePrivateProfileStringA("Display", "HDR", g_settings.hdr == Settings::Auto ? "auto" : g_settings.hdr == Settings::On ? "on" : "off", g_ini.c_str()) != 0;
  ok &= WritePrivateProfileStringA("Display", "HDRFormat", g_settings.format == Settings::ScRgb ? "scrgb" : "hdr10", g_ini.c_str()) != 0;
  snprintf(v, sizeof(v), "%g", g_settings.peakNits);
  ok &= WritePrivateProfileStringA("Display", "PeakNits", v, g_ini.c_str()) != 0;
  snprintf(v, sizeof(v), "%g", g_settings.paperWhiteNits);
  ok &= WritePrivateProfileStringA("Display", "PaperWhiteNits", v, g_ini.c_str()) != 0;
  logf("display: settings %s fnv_vk.ini", ok ? "saved to" : "could NOT be saved to");
  return ok;
}

}  // namespace fnvvk::display
