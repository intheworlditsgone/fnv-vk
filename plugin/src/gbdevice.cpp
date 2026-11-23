// The plugin's IDirect3D9Ex and IDirect3DDevice9Ex (native mode): D3D9 is never created.
//
// NiDX9Renderer::CreateDirect3D9 0x00E69460 takes Direct3DCreate9 / Direct3DCreate9Ex from the pointers that
// NiDX9Renderer::IsD3D9Create 0x00E69410 resolves from D3D9.DLL. IsD3D9Create is replaced: it hands the engine the
// creators below, so the engine runs its own creation path (adapter enumeration, caps, CreateDeviceEx, the implicit
// buffers) against these objects, and every device call it makes (typed, untyped, D3DX's) lands here.
//
// - The adapter list, identifiers, monitors and display modes are the system's (DXGI adapters and outputs,
//   EnumDisplaySettings), one D3D9 adapter per output as D3D9 enumerates them.
// - The caps are the D3D9 caps DXVK reports for an SM3 device (src/d3d9/d3d9_adapter.cpp, the set FNV is known to run
//   with); formats and multisampling are answered with what the Vulkan backend renders and samples.
// - Resources are the plugin's objects (gbobjects.cpp, gbsurfaces.cpp); state and draws go to the game-side state
//   tracker (gbstate.cpp), which drives the backend; everything the backend mirrors from the Gamebryo layer
//   (render-target binding, clears, copies, present) is accepted here and does nothing else.
#include <windows.h>

#include <d3d9.h>
#include <dxgi.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "common.h"
#include "gbdevice.h"
#include "gbobjects.h"
#include "gbstate.h"
#include "gbsurfaces.h"

namespace fnvvk::gbdevice {
namespace {

std::atomic<uint64_t> g_unsupportedCalls{ 0 }, g_upDrawsDropped{ 0 }, g_volumeRefused{ 0 };

// --- adapters -----------------------------------------------------------------------------------------------------
struct Mode {
  UINT w, h, refresh;
  bool operator<(const Mode& o) const { return w != o.w ? w < o.w : h != o.h ? h < o.h : refresh < o.refresh; }
  bool operator==(const Mode& o) const { return w == o.w && h == o.h && refresh == o.refresh; }
};
struct Adapter {
  D3DADAPTER_IDENTIFIER9 id{};
  HMONITOR monitor = nullptr;
  std::wstring deviceName;  // \\.\DISPLAYn
  std::vector<Mode> modes;
  uint64_t dedicatedMemory = 0;
  LUID luid{};
};
std::vector<Adapter> g_adapters;

void enumerateModes(Adapter& a) {
  DEVMODEW dm{};
  dm.dmSize = sizeof(dm);
  for (DWORD i = 0; EnumDisplaySettingsExW(a.deviceName.c_str(), i, &dm, 0); i++) {
    if (dm.dmBitsPerPel != 32) continue;
    a.modes.push_back({ dm.dmPelsWidth, dm.dmPelsHeight, dm.dmDisplayFrequency });
  }
  std::sort(a.modes.begin(), a.modes.end());
  a.modes.erase(std::unique(a.modes.begin(), a.modes.end()), a.modes.end());
}

bool currentMode(const Adapter& a, D3DDISPLAYMODE& out) {
  DEVMODEW dm{};
  dm.dmSize = sizeof(dm);
  if (!EnumDisplaySettingsExW(a.deviceName.c_str(), ENUM_CURRENT_SETTINGS, &dm, 0)) return false;
  out = { dm.dmPelsWidth, dm.dmPelsHeight, dm.dmDisplayFrequency, D3DFMT_X8R8G8B8 };
  return true;
}

// The adapter's 32-bit user-mode driver file name (D3D9's D3DADAPTER_IDENTIFIER9::Driver, e.g. nvldumd.dll): the first
// UserModeDriverNameWow entry of the display-class key whose MatchingDeviceId is this PCI vendor/device. The engine
// keys its shader setup on its prefix (BSShaderManager::SetShaderVersion: "nv", "ati", "ig" -> GPU vendor, shader
// level, Lighting30); the driver's known names are the fallback.
std::string userModeDriverName(UINT vendor, UINT device) {
  char want[64];
  sprintf_s(want, "pci\\ven_%04x&dev_%04x", vendor, device);
  const char* cls = "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}";
  HKEY root = nullptr;
  std::string found;
  if (RegOpenKeyExA(HKEY_LOCAL_MACHINE, cls, 0, KEY_READ, &root) == ERROR_SUCCESS) {
    char sub[16];
    for (DWORD i = 0; found.empty() && RegEnumKeyA(root, i, sub, sizeof(sub)) == ERROR_SUCCESS; i++) {
      HKEY k = nullptr;
      if (RegOpenKeyExA(root, sub, 0, KEY_READ, &k) != ERROR_SUCCESS) continue;
      char match[256] = {};
      DWORD size = sizeof(match) - 1;
      if (RegQueryValueExA(k, "MatchingDeviceId", nullptr, nullptr, reinterpret_cast<BYTE*>(match), &size) == ERROR_SUCCESS) {
        for (char* c = match; *c; c++) *c = char(tolower(uint8_t(*c)));
        if (strncmp(match, want, strlen(want)) == 0) {
          char names[2048] = {};
          size = sizeof(names) - 2;
          if (RegQueryValueExA(k, "UserModeDriverNameWow", nullptr, nullptr, reinterpret_cast<BYTE*>(names), &size) == ERROR_SUCCESS &&
              names[0]) {
            const char* slash = strrchr(names, '\\');
            found = slash ? slash + 1 : names;
          }
        }
      }
      RegCloseKey(k);
    }
    RegCloseKey(root);
  }
  if (found.empty()) {
    switch (vendor) {
      case 0x10DE: found = "nvldumd.dll"; break;
      case 0x1002: found = "aticfx32.dll"; break;
      case 0x8086: found = "igdumdim32.dll"; break;
      default: found = "unknown.dll"; break;
    }
  }
  return found;
}

void enumerateAdapters() {
  if (!g_adapters.empty()) return;
  using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);
  HMODULE dxgi = LoadLibraryA("dxgi.dll");
  auto create = dxgi ? reinterpret_cast<CreateFactoryFn>(GetProcAddress(dxgi, "CreateDXGIFactory1")) : nullptr;
  IDXGIFactory1* factory = nullptr;
  if (create && SUCCEEDED(create(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) && factory) {
    IDXGIAdapter1* ad = nullptr;
    for (UINT i = 0; factory->EnumAdapters1(i, &ad) != DXGI_ERROR_NOT_FOUND; i++) {
      DXGI_ADAPTER_DESC1 desc{};
      ad->GetDesc1(&desc);
      // The user-mode driver version, as D3D9 reports it (DriverVersion): the D3D10 interface query returns it.
      static const GUID kID3D10Device = { 0x9B7E4C0F, 0x342C, 0x4106, { 0xA1, 0x9F, 0x4F, 0x27, 0x04, 0xF6, 0x89, 0xF0 } };
      LARGE_INTEGER umd{};
      ad->CheckInterfaceSupport(kID3D10Device, &umd);
      IDXGIOutput* out = nullptr;
      for (UINT o = 0; ad->EnumOutputs(o, &out) != DXGI_ERROR_NOT_FOUND; o++) {
        DXGI_OUTPUT_DESC od{};
        out->GetDesc(&od);
        out->Release();
        Adapter a;
        a.monitor = od.Monitor;
        a.deviceName = od.DeviceName;
        a.dedicatedMemory = desc.DedicatedVideoMemory;
        a.luid = desc.AdapterLuid;
        WideCharToMultiByte(CP_ACP, 0, desc.Description, -1, a.id.Description, MAX_DEVICE_IDENTIFIER_STRING, nullptr, nullptr);
        WideCharToMultiByte(CP_ACP, 0, od.DeviceName, -1, a.id.DeviceName, 32, nullptr, nullptr);
        strcpy_s(a.id.Driver, userModeDriverName(desc.VendorId, desc.DeviceId).c_str());
        a.id.DriverVersion = umd;
        a.id.VendorId = desc.VendorId;
        a.id.DeviceId = desc.DeviceId;
        a.id.SubSysId = desc.SubSysId;
        a.id.Revision = desc.Revision;
        // Stable per adapter and driver, as D3D9's identifier is.
        a.id.DeviceIdentifier.Data1 = desc.VendorId ^ (desc.DeviceId << 16);
        a.id.DeviceIdentifier.Data2 = WORD(desc.SubSysId);
        a.id.DeviceIdentifier.Data3 = WORD(desc.SubSysId >> 16);
        memcpy(a.id.DeviceIdentifier.Data4, &umd, 8);
        a.id.WHQLLevel = 1;
        enumerateModes(a);
        g_adapters.push_back(std::move(a));
      }
      ad->Release();
    }
    factory->Release();
  }
  if (g_adapters.empty()) {  // no DXGI outputs: the primary display
    Adapter a;
    a.monitor = MonitorFromPoint(POINT{ 0, 0 }, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFOEXW mi{};
    mi.cbSize = sizeof(mi);
    GetMonitorInfoW(a.monitor, &mi);
    a.deviceName = mi.szDevice;
    strcpy_s(a.id.Description, "Vulkan");
    strcpy_s(a.id.Driver, "fnv_vk");
    WideCharToMultiByte(CP_ACP, 0, mi.szDevice, -1, a.id.DeviceName, 32, nullptr, nullptr);
    enumerateModes(a);
    g_adapters.push_back(std::move(a));
  }
  // D3D9's adapter 0 is the primary display (WinMain's CreateDisplayDriver compares its description with sD3DDevice).
  const HMONITOR primary = MonitorFromPoint(POINT{ 0, 0 }, MONITOR_DEFAULTTOPRIMARY);
  std::stable_partition(g_adapters.begin(), g_adapters.end(), [&](const Adapter& a) { return a.monitor == primary; });
  for (size_t i = 0; i < g_adapters.size(); i++)
    logf("D3D9 adapter %zu: %s (vendor %04lX device %04lX, driver %s), %ls, %zu modes, %llu MB dedicated", i,
         g_adapters[i].id.Description, g_adapters[i].id.VendorId, g_adapters[i].id.DeviceId, g_adapters[i].id.Driver,
         g_adapters[i].deviceName.c_str(), g_adapters[i].modes.size(),
         g_adapters[i].dedicatedMemory >> 20);
}

// --- formats ------------------------------------------------------------------------------------------------------
constexpr D3DFORMAT fourcc(char a, char b, char c, char d) {
  return D3DFORMAT(uint32_t(uint8_t(a)) | (uint32_t(uint8_t(b)) << 8) | (uint32_t(uint8_t(c)) << 16) | (uint32_t(uint8_t(d)) << 24));
}
// Colour formats the backend renders to (backend.cpp vkFormatFor).
bool renderTargetFormat(D3DFORMAT f) {
  switch (f) {
    case D3DFMT_A8R8G8B8: case D3DFMT_X8R8G8B8: case D3DFMT_R5G6B5: case D3DFMT_A16B16G16R16F: case D3DFMT_G16R16F:
    case D3DFMT_R32F: case D3DFMT_A32B32G32R32F: return true;
    default: return false;
  }
}
bool depthFormat(D3DFORMAT f) { return f == D3DFMT_D24S8 || f == D3DFMT_D24X8; }
// Formats the backend uploads and samples as textures (backend.cpp texFormatFor), plus the render-target formats
// (sampled after rendering).
bool textureFormat(D3DFORMAT f) {
  switch (f) {
    case D3DFMT_DXT1: case D3DFMT_DXT2: case D3DFMT_DXT3: case D3DFMT_DXT4: case D3DFMT_DXT5:
    case D3DFMT_A4R4G4B4: return true;
    default: return renderTargetFormat(f);
  }
}
bool floatFormat(D3DFORMAT f) {
  return f == D3DFMT_A16B16G16R16F || f == D3DFMT_G16R16F || f == D3DFMT_R32F || f == D3DFMT_A32B32G32R32F;
}

UINT g_vendor = 0;

HRESULT checkDeviceFormat(D3DFORMAT adapterFmt, DWORD usage, D3DRESOURCETYPE rtype, D3DFORMAT f) {
  if (adapterFmt == D3DFMT_UNKNOWN) return D3DERR_INVALIDCALL;
  if (rtype == D3DRTYPE_VERTEXBUFFER || rtype == D3DRTYPE_INDEXBUFFER) return D3DERR_INVALIDCALL;
  if (adapterFmt != D3DFMT_X8R8G8B8) return D3DERR_NOTAVAILABLE;
  // Driver extension formats: alpha-to-coverage ('ATOC') as NVIDIA's D3D9 driver advertises it; none of the others.
  if (f == fourcc('A', 'T', 'O', 'C')) return rtype == D3DRTYPE_SURFACE && g_vendor == 0x10DE ? D3D_OK : D3DERR_NOTAVAILABLE;
  if (rtype == D3DRTYPE_VOLUMETEXTURE || rtype == D3DRTYPE_VOLUME) return D3DERR_NOTAVAILABLE;
  if (usage & (D3DUSAGE_QUERY_SRGBREAD | D3DUSAGE_QUERY_SRGBWRITE | D3DUSAGE_QUERY_LEGACYBUMPMAP | D3DUSAGE_DMAP))
    return D3DERR_NOTAVAILABLE;
  if (usage & D3DUSAGE_DEPTHSTENCIL) return depthFormat(f) && rtype == D3DRTYPE_SURFACE ? D3D_OK : D3DERR_NOTAVAILABLE;
  if (depthFormat(f)) return D3DERR_NOTAVAILABLE;
  if (usage & D3DUSAGE_RENDERTARGET) {
    if (!renderTargetFormat(f)) return D3DERR_NOTAVAILABLE;
  } else if (!textureFormat(f)) {
    return D3DERR_NOTAVAILABLE;
  }
  // Blending and linear filtering of 32-bit float formats are optional Vulkan features: not promised.
  const bool float32 = f == D3DFMT_R32F || f == D3DFMT_A32B32G32R32F;
  if ((usage & D3DUSAGE_QUERY_POSTPIXELSHADER_BLENDING) && (!renderTargetFormat(f) || float32)) return D3DERR_NOTAVAILABLE;
  if ((usage & D3DUSAGE_QUERY_FILTER) && float32) return D3DERR_NOTAVAILABLE;
  if ((usage & D3DUSAGE_QUERY_VERTEXTEXTURE) && !floatFormat(f)) return D3DERR_NOTAVAILABLE;
  // GenerateMipSubLevels is not implemented: D3D9's answer for a usable format without automatic mipmaps.
  if (usage & D3DUSAGE_AUTOGENMIPMAP) return D3DOK_NOAUTOGEN;
  return D3D_OK;
}

HRESULT checkMultiSample(D3DFORMAT f, D3DMULTISAMPLE_TYPE ms, DWORD* quality) {
  if (quality) *quality = 1;
  if (ms > D3DMULTISAMPLE_16_SAMPLES || f == D3DFMT_UNKNOWN) return D3DERR_INVALIDCALL;
  if (!renderTargetFormat(f) && !depthFormat(f)) return D3DERR_NOTAVAILABLE;
  // 1, 2, 4 and 8 samples: the counts desktop Vulkan drivers support for these formats.
  if (ms == D3DMULTISAMPLE_NONMASKABLE) {
    if (quality) *quality = 4;
    return D3D_OK;
  }
  if (ms == D3DMULTISAMPLE_NONE || ms == D3DMULTISAMPLE_2_SAMPLES || ms == D3DMULTISAMPLE_4_SAMPLES || ms == D3DMULTISAMPLE_8_SAMPLES)
    return D3D_OK;
  return D3DERR_NOTAVAILABLE;
}

void fillCaps(UINT adapter, D3DDEVTYPE type, D3DCAPS9& c) {
  c = D3DCAPS9{};
  c.DeviceType = type;
  c.AdapterOrdinal = adapter;
  c.Caps = D3DCAPS_READ_SCANLINE;
  c.Caps2 = D3DCAPS2_FULLSCREENGAMMA | D3DCAPS2_DYNAMICTEXTURES | D3DCAPS2_CANAUTOGENMIPMAP | D3DCAPS2_CANSHARERESOURCE;
  c.Caps3 = D3DCAPS3_ALPHA_FULLSCREEN_FLIP_OR_DISCARD | D3DCAPS3_LINEAR_TO_SRGB_PRESENTATION | D3DCAPS3_COPY_TO_VIDMEM |
            D3DCAPS3_COPY_TO_SYSTEMMEM;
  c.PresentationIntervals = D3DPRESENT_INTERVAL_DEFAULT | D3DPRESENT_INTERVAL_ONE | D3DPRESENT_INTERVAL_TWO |
                            D3DPRESENT_INTERVAL_THREE | D3DPRESENT_INTERVAL_FOUR | D3DPRESENT_INTERVAL_IMMEDIATE;
  c.CursorCaps = D3DCURSORCAPS_COLOR;
  c.DevCaps = D3DDEVCAPS_EXECUTESYSTEMMEMORY | D3DDEVCAPS_EXECUTEVIDEOMEMORY | D3DDEVCAPS_TLVERTEXSYSTEMMEMORY |
              D3DDEVCAPS_TLVERTEXVIDEOMEMORY | D3DDEVCAPS_TEXTUREVIDEOMEMORY | D3DDEVCAPS_DRAWPRIMTLVERTEX |
              D3DDEVCAPS_CANRENDERAFTERFLIP | D3DDEVCAPS_TEXTURENONLOCALVIDMEM | D3DDEVCAPS_DRAWPRIMITIVES2 |
              D3DDEVCAPS_DRAWPRIMITIVES2EX | D3DDEVCAPS_HWTRANSFORMANDLIGHT | D3DDEVCAPS_CANBLTSYSTONONLOCAL |
              D3DDEVCAPS_HWRASTERIZATION | D3DDEVCAPS_PUREDEVICE;
  c.PrimitiveMiscCaps = D3DPMISCCAPS_MASKZ | D3DPMISCCAPS_CULLNONE | D3DPMISCCAPS_CULLCW | D3DPMISCCAPS_CULLCCW |
                        D3DPMISCCAPS_COLORWRITEENABLE | D3DPMISCCAPS_CLIPPLANESCALEDPOINTS | D3DPMISCCAPS_CLIPTLVERTS |
                        D3DPMISCCAPS_TSSARGTEMP | D3DPMISCCAPS_BLENDOP | D3DPMISCCAPS_INDEPENDENTWRITEMASKS |
                        D3DPMISCCAPS_PERSTAGECONSTANT | D3DPMISCCAPS_FOGANDSPECULARALPHA | D3DPMISCCAPS_SEPARATEALPHABLEND |
                        D3DPMISCCAPS_MRTINDEPENDENTBITDEPTHS | D3DPMISCCAPS_MRTPOSTPIXELSHADERBLENDING |
                        D3DPMISCCAPS_FOGVERTEXCLAMPED | D3DPMISCCAPS_POSTBLENDSRGBCONVERT;
  c.RasterCaps = D3DPRASTERCAPS_DITHER | D3DPRASTERCAPS_ZTEST | D3DPRASTERCAPS_FOGVERTEX | D3DPRASTERCAPS_FOGTABLE |
                 D3DPRASTERCAPS_MIPMAPLODBIAS | D3DPRASTERCAPS_FOGRANGE | D3DPRASTERCAPS_ANISOTROPY | D3DPRASTERCAPS_WFOG |
                 D3DPRASTERCAPS_ZFOG | D3DPRASTERCAPS_COLORPERSPECTIVE | D3DPRASTERCAPS_SCISSORTEST |
                 D3DPRASTERCAPS_SLOPESCALEDEPTHBIAS | D3DPRASTERCAPS_DEPTHBIAS | D3DPRASTERCAPS_MULTISAMPLE_TOGGLE;
  c.ZCmpCaps = D3DPCMPCAPS_NEVER | D3DPCMPCAPS_LESS | D3DPCMPCAPS_EQUAL | D3DPCMPCAPS_LESSEQUAL | D3DPCMPCAPS_GREATER |
               D3DPCMPCAPS_NOTEQUAL | D3DPCMPCAPS_GREATEREQUAL | D3DPCMPCAPS_ALWAYS;
  c.SrcBlendCaps = D3DPBLENDCAPS_ZERO | D3DPBLENDCAPS_ONE | D3DPBLENDCAPS_SRCCOLOR | D3DPBLENDCAPS_INVSRCCOLOR |
                   D3DPBLENDCAPS_SRCALPHA | D3DPBLENDCAPS_INVSRCALPHA | D3DPBLENDCAPS_DESTALPHA | D3DPBLENDCAPS_INVDESTALPHA |
                   D3DPBLENDCAPS_DESTCOLOR | D3DPBLENDCAPS_INVDESTCOLOR | D3DPBLENDCAPS_SRCALPHASAT |
                   D3DPBLENDCAPS_BOTHSRCALPHA | D3DPBLENDCAPS_BOTHINVSRCALPHA | D3DPBLENDCAPS_BLENDFACTOR |
                   D3DPBLENDCAPS_SRCCOLOR2 | D3DPBLENDCAPS_INVSRCCOLOR2;
  c.DestBlendCaps = c.SrcBlendCaps;
  c.AlphaCmpCaps = c.ZCmpCaps;
  c.ShadeCaps = D3DPSHADECAPS_COLORGOURAUDRGB | D3DPSHADECAPS_SPECULARGOURAUDRGB | D3DPSHADECAPS_ALPHAGOURAUDBLEND |
                D3DPSHADECAPS_FOGGOURAUD;
  c.TextureCaps = D3DPTEXTURECAPS_PERSPECTIVE | D3DPTEXTURECAPS_ALPHA | D3DPTEXTURECAPS_TEXREPEATNOTSCALEDBYSIZE |
                  D3DPTEXTURECAPS_ALPHAPALETTE | D3DPTEXTURECAPS_PROJECTED | D3DPTEXTURECAPS_CUBEMAP |
                  D3DPTEXTURECAPS_MIPMAP | D3DPTEXTURECAPS_MIPCUBEMAP;  // no volume textures (not in the game's data)
  c.TextureFilterCaps = D3DPTFILTERCAPS_MINFPOINT | D3DPTFILTERCAPS_MINFLINEAR | D3DPTFILTERCAPS_MINFANISOTROPIC |
                        D3DPTFILTERCAPS_MIPFPOINT | D3DPTFILTERCAPS_MIPFLINEAR | D3DPTFILTERCAPS_MAGFPOINT |
                        D3DPTFILTERCAPS_MAGFLINEAR | D3DPTFILTERCAPS_MAGFANISOTROPIC;
  c.CubeTextureFilterCaps = c.TextureFilterCaps;
  c.TextureAddressCaps = D3DPTADDRESSCAPS_WRAP | D3DPTADDRESSCAPS_MIRROR | D3DPTADDRESSCAPS_CLAMP | D3DPTADDRESSCAPS_BORDER |
                         D3DPTADDRESSCAPS_INDEPENDENTUV | D3DPTADDRESSCAPS_MIRRORONCE;
  c.LineCaps = D3DLINECAPS_TEXTURE | D3DLINECAPS_ZTEST | D3DLINECAPS_BLEND | D3DLINECAPS_ALPHACMP | D3DLINECAPS_FOG |
               D3DLINECAPS_ANTIALIAS;
  c.MaxTextureWidth = c.MaxTextureHeight = 16384;
  c.MaxTextureRepeat = 8192;
  c.MaxTextureAspectRatio = 16384;
  c.MaxAnisotropy = 16;
  c.MaxVertexW = 1e10f;
  c.GuardBandLeft = c.GuardBandTop = -32768.0f;
  c.GuardBandRight = c.GuardBandBottom = 32768.0f;
  c.StencilCaps = D3DSTENCILCAPS_KEEP | D3DSTENCILCAPS_ZERO | D3DSTENCILCAPS_REPLACE | D3DSTENCILCAPS_INCRSAT |
                  D3DSTENCILCAPS_DECRSAT | D3DSTENCILCAPS_INVERT | D3DSTENCILCAPS_INCR | D3DSTENCILCAPS_DECR |
                  D3DSTENCILCAPS_TWOSIDED;
  c.FVFCaps = (8 & D3DFVFCAPS_TEXCOORDCOUNTMASK) | D3DFVFCAPS_PSIZE;
  c.TextureOpCaps = D3DTEXOPCAPS_DISABLE | D3DTEXOPCAPS_SELECTARG1 | D3DTEXOPCAPS_SELECTARG2 | D3DTEXOPCAPS_MODULATE |
                    D3DTEXOPCAPS_MODULATE2X | D3DTEXOPCAPS_MODULATE4X | D3DTEXOPCAPS_ADD | D3DTEXOPCAPS_ADDSIGNED |
                    D3DTEXOPCAPS_ADDSIGNED2X | D3DTEXOPCAPS_SUBTRACT | D3DTEXOPCAPS_ADDSMOOTH |
                    D3DTEXOPCAPS_BLENDDIFFUSEALPHA | D3DTEXOPCAPS_BLENDTEXTUREALPHA | D3DTEXOPCAPS_BLENDFACTORALPHA |
                    D3DTEXOPCAPS_BLENDTEXTUREALPHAPM | D3DTEXOPCAPS_BLENDCURRENTALPHA | D3DTEXOPCAPS_PREMODULATE |
                    D3DTEXOPCAPS_MODULATEALPHA_ADDCOLOR | D3DTEXOPCAPS_MODULATECOLOR_ADDALPHA |
                    D3DTEXOPCAPS_MODULATEINVALPHA_ADDCOLOR | D3DTEXOPCAPS_MODULATEINVCOLOR_ADDALPHA |
                    D3DTEXOPCAPS_BUMPENVMAP | D3DTEXOPCAPS_BUMPENVMAPLUMINANCE | D3DTEXOPCAPS_DOTPRODUCT3 |
                    D3DTEXOPCAPS_MULTIPLYADD | D3DTEXOPCAPS_LERP;
  c.MaxTextureBlendStages = 8;
  c.MaxSimultaneousTextures = 8;
  c.VertexProcessingCaps = D3DVTXPCAPS_TEXGEN | D3DVTXPCAPS_MATERIALSOURCE7 | D3DVTXPCAPS_DIRECTIONALLIGHTS |
                           D3DVTXPCAPS_POSITIONALLIGHTS | D3DVTXPCAPS_LOCALVIEWER | D3DVTXPCAPS_TWEENING |
                           D3DVTXPCAPS_TEXGEN_SPHEREMAP;
  c.MaxActiveLights = 8;
  c.MaxUserClipPlanes = 6;
  c.MaxVertexBlendMatrices = 4;
  c.MaxVertexBlendMatrixIndex = 0;
  c.MaxPointSize = 256.0f;
  c.MaxPrimitiveCount = 0x00555555;
  c.MaxVertexIndex = 0x00FFFFFF;
  c.MaxStreams = 16;
  c.MaxStreamStride = 508;
  c.VertexShaderVersion = D3DVS_VERSION(3, 0);
  c.MaxVertexShaderConst = 256;
  c.PixelShaderVersion = D3DPS_VERSION(3, 0);
  c.PixelShader1xMaxValue = 3.402823466e+38f;
  c.DevCaps2 = D3DDEVCAPS2_STREAMOFFSET | D3DDEVCAPS2_CAN_STRETCHRECT_FROM_TEXTURES |
               D3DDEVCAPS2_VERTEXELEMENTSCANSHARESTREAMOFFSET;
  c.MasterAdapterOrdinal = adapter;
  c.AdapterOrdinalInGroup = 0;
  c.NumberOfAdaptersInGroup = 1;
  c.DeclTypes = D3DDTCAPS_UBYTE4 | D3DDTCAPS_UBYTE4N | D3DDTCAPS_SHORT2N | D3DDTCAPS_SHORT4N | D3DDTCAPS_USHORT2N |
                D3DDTCAPS_USHORT4N | D3DDTCAPS_UDEC3 | D3DDTCAPS_DEC3N | D3DDTCAPS_FLOAT16_2 | D3DDTCAPS_FLOAT16_4;
  c.NumSimultaneousRTs = 4;
  c.StretchRectFilterCaps = D3DPTFILTERCAPS_MINFPOINT | D3DPTFILTERCAPS_MINFLINEAR | D3DPTFILTERCAPS_MAGFPOINT |
                            D3DPTFILTERCAPS_MAGFLINEAR;
  c.VS20Caps.Caps = D3DVS20CAPS_PREDICATION;
  c.VS20Caps.DynamicFlowControlDepth = D3DVS20_MAX_DYNAMICFLOWCONTROLDEPTH;
  c.VS20Caps.NumTemps = D3DVS20_MAX_NUMTEMPS;
  c.VS20Caps.StaticFlowControlDepth = D3DVS20_MAX_STATICFLOWCONTROLDEPTH;
  c.PS20Caps.Caps = D3DPS20CAPS_ARBITRARYSWIZZLE | D3DPS20CAPS_GRADIENTINSTRUCTIONS | D3DPS20CAPS_PREDICATION |
                    D3DPS20CAPS_NODEPENDENTREADLIMIT | D3DPS20CAPS_NOTEXINSTRUCTIONLIMIT;
  c.PS20Caps.DynamicFlowControlDepth = D3DPS20_MAX_DYNAMICFLOWCONTROLDEPTH;
  c.PS20Caps.NumTemps = D3DPS20_MAX_NUMTEMPS;
  c.PS20Caps.StaticFlowControlDepth = D3DPS20_MAX_STATICFLOWCONTROLDEPTH;
  c.PS20Caps.NumInstructionSlots = D3DPS20_MAX_NUMINSTRUCTIONSLOTS;
  c.VertexTextureFilterCaps = D3DPTFILTERCAPS_MINFPOINT | D3DPTFILTERCAPS_MINFLINEAR | D3DPTFILTERCAPS_MAGFPOINT |
                              D3DPTFILTERCAPS_MAGFLINEAR;
  c.MaxVShaderInstructionsExecuted = 4294967295u;
  c.MaxPShaderInstructionsExecuted = 4294967295u;
  c.MaxVertexShader30InstructionSlots = 32768;
  c.MaxPixelShader30InstructionSlots = 32768;
}

// --- device -------------------------------------------------------------------------------------------------------
class Direct3D;
class Device;

class SwapChain final : public IDirect3DSwapChain9Ex {
 public:
  explicit SwapChain(Device* dev) : dev_(dev) {}
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == __uuidof(IUnknown) || riid == __uuidof(IDirect3DSwapChain9) || riid == __uuidof(IDirect3DSwapChain9Ex)) {
      AddRef();
      *out = this;
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
  ULONG STDMETHODCALLTYPE Release() override { return --refs_; }  // owned by the device
  HRESULT STDMETHODCALLTYPE Present(const RECT*, const RECT*, HWND, const RGNDATA*, DWORD) override { return D3D_OK; }
  HRESULT STDMETHODCALLTYPE GetFrontBufferData(IDirect3DSurface9*) override { return D3DERR_INVALIDCALL; }
  HRESULT STDMETHODCALLTYPE GetBackBuffer(UINT i, D3DBACKBUFFER_TYPE t, IDirect3DSurface9** out) override;
  HRESULT STDMETHODCALLTYPE GetRasterStatus(D3DRASTER_STATUS* s) override {
    if (!s) return D3DERR_INVALIDCALL;
    *s = {};
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetDisplayMode(D3DDISPLAYMODE* m) override;
  HRESULT STDMETHODCALLTYPE GetDevice(IDirect3DDevice9** out) override;
  HRESULT STDMETHODCALLTYPE GetPresentParameters(D3DPRESENT_PARAMETERS* pp) override;
  HRESULT STDMETHODCALLTYPE GetLastPresentCount(UINT* n) override {
    if (n) *n = 0;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetPresentStats(D3DPRESENTSTATS* s) override {
    if (s) *s = {};
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetDisplayModeEx(D3DDISPLAYMODEEX* m, D3DDISPLAYROTATION* r) override;

 private:
  std::atomic<ULONG> refs_{ 1 };
  Device* dev_;
};

class Query final : public IDirect3DQuery9 {
 public:
  Query(IDirect3DDevice9* dev, D3DQUERYTYPE t) : dev_(dev), type_(t) {}
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == __uuidof(IUnknown) || riid == __uuidof(IDirect3DQuery9)) {
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
    if (!r) delete this;
    return r;
  }
  HRESULT STDMETHODCALLTYPE GetDevice(IDirect3DDevice9** out) override {
    if (!out) return D3DERR_INVALIDCALL;
    *out = dev_;
    dev_->AddRef();
    return D3D_OK;
  }
  D3DQUERYTYPE STDMETHODCALLTYPE GetType() override { return type_; }
  DWORD STDMETHODCALLTYPE GetDataSize() override { return type_ == D3DQUERYTYPE_EVENT ? sizeof(BOOL) : sizeof(DWORD); }
  HRESULT STDMETHODCALLTYPE Issue(DWORD) override { return D3D_OK; }
  // Event queries signal at once (the backend orders its own work); occlusion queries are BSOcclusionQuery's, which
  // runs on the backend's queries (gbquery.cpp) and never creates device queries: report everything visible.
  HRESULT STDMETHODCALLTYPE GetData(void* data, DWORD size, DWORD) override {
    if (data && size >= sizeof(DWORD)) *static_cast<DWORD*>(data) = type_ == D3DQUERYTYPE_EVENT ? TRUE : 0xFFFFFFFFu;
    return S_OK;
  }

 private:
  ~Query() = default;
  std::atomic<ULONG> refs_{ 1 };
  IDirect3DDevice9* dev_;
  D3DQUERYTYPE type_;
};

class Device final : public IDirect3DDevice9Ex {
 public:
  Device(Direct3D* parent, UINT adapter, D3DDEVTYPE type, HWND focus, DWORD behavior, const D3DPRESENT_PARAMETERS& pp)
      : parent_(parent), swapChain_(this) {
    cp_ = { adapter, type, focus, behavior };
    for (auto& r : gamma_.red) r = 0;
    for (int i = 0; i < 256; i++) gamma_.red[i] = gamma_.green[i] = gamma_.blue[i] = WORD(i * 257);
    createImplicit(pp);
  }

  // IUnknown
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == __uuidof(IUnknown) || riid == __uuidof(IDirect3DDevice9) || riid == __uuidof(IDirect3DDevice9Ex)) {
      AddRef();
      *out = this;
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
  // Kept alive to process exit: the engine's resources and D3DX hold the pointer past its last Release.
  ULONG STDMETHODCALLTYPE Release() override { return refs_ ? --refs_ : 0; }

  // IDirect3DDevice9
  HRESULT STDMETHODCALLTYPE TestCooperativeLevel() override { return D3D_OK; }
  UINT STDMETHODCALLTYPE GetAvailableTextureMem() override;
  HRESULT STDMETHODCALLTYPE EvictManagedResources() override { return D3D_OK; }
  HRESULT STDMETHODCALLTYPE GetDirect3D(IDirect3D9** out) override;
  HRESULT STDMETHODCALLTYPE GetDeviceCaps(D3DCAPS9* c) override {
    if (!c) return D3DERR_INVALIDCALL;
    fillCaps(cp_.AdapterOrdinal, cp_.DeviceType, *c);
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetDisplayMode(UINT, D3DDISPLAYMODE* m) override {
    if (!m || cp_.AdapterOrdinal >= g_adapters.size()) return D3DERR_INVALIDCALL;
    return currentMode(g_adapters[cp_.AdapterOrdinal], *m) ? D3D_OK : D3DERR_INVALIDCALL;
  }
  HRESULT STDMETHODCALLTYPE GetCreationParameters(D3DDEVICE_CREATION_PARAMETERS* p) override {
    if (!p) return D3DERR_INVALIDCALL;
    *p = cp_;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetCursorProperties(UINT, UINT, IDirect3DSurface9*) override { return D3D_OK; }
  void STDMETHODCALLTYPE SetCursorPosition(int, int, DWORD) override {}
  BOOL STDMETHODCALLTYPE ShowCursor(BOOL show) override {
    const BOOL prev = cursor_;
    cursor_ = show;
    return prev;
  }
  HRESULT STDMETHODCALLTYPE CreateAdditionalSwapChain(D3DPRESENT_PARAMETERS*, IDirect3DSwapChain9**) override {
    return unsupported(D3DERR_NOTAVAILABLE);
  }
  HRESULT STDMETHODCALLTYPE GetSwapChain(UINT i, IDirect3DSwapChain9** out) override {
    if (!out || i) return D3DERR_INVALIDCALL;
    swapChain_.AddRef();
    *out = &swapChain_;
    return D3D_OK;
  }
  UINT STDMETHODCALLTYPE GetNumberOfSwapChains() override { return 1; }
  HRESULT STDMETHODCALLTYPE Reset(D3DPRESENT_PARAMETERS* pp) override {
    if (!pp) return D3DERR_INVALIDCALL;
    createImplicit(*pp);
    *pp = pp_;
    return D3D_OK;
  }
  // Vulkan presents (present.cpp replaces NiDX9ImplicitBufferData::DisplayFrame; Bink's Present is movie.cpp's).
  HRESULT STDMETHODCALLTYPE Present(const RECT*, const RECT*, HWND, const RGNDATA*) override { return D3D_OK; }
  HRESULT STDMETHODCALLTYPE GetBackBuffer(UINT sc, UINT i, D3DBACKBUFFER_TYPE, IDirect3DSurface9** out) override {
    if (!out || sc || i >= pp_.BackBufferCount || !backBuffer_) return D3DERR_INVALIDCALL;
    backBuffer_->AddRef();
    *out = backBuffer_;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetRasterStatus(UINT, D3DRASTER_STATUS* s) override {
    if (!s) return D3DERR_INVALIDCALL;
    *s = {};
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetDialogBoxMode(BOOL) override { return D3D_OK; }
  void STDMETHODCALLTYPE SetGammaRamp(UINT, DWORD, const D3DGAMMARAMP* r) override {
    if (r) gamma_ = *r;  // the backend applies it (shadow.cpp gammaRamp)
  }
  void STDMETHODCALLTYPE GetGammaRamp(UINT, D3DGAMMARAMP* r) override {
    if (r) *r = gamma_;
  }
  HRESULT STDMETHODCALLTYPE CreateTexture(UINT w, UINT h, UINT levels, DWORD usage, D3DFORMAT f, D3DPOOL pool,
                                          IDirect3DTexture9** out, HANDLE*) override {
    return gbsurfaces::createTexture(this, w, h, levels, usage, f, pool, reinterpret_cast<void**>(out));
  }
  HRESULT STDMETHODCALLTYPE CreateVolumeTexture(UINT, UINT, UINT, UINT, DWORD, D3DFORMAT, D3DPOOL,
                                                IDirect3DVolumeTexture9** out, HANDLE*) override {
    if (out) *out = nullptr;
    g_volumeRefused++;
    return D3DERR_NOTAVAILABLE;
  }
  HRESULT STDMETHODCALLTYPE CreateCubeTexture(UINT edge, UINT levels, DWORD usage, D3DFORMAT f, D3DPOOL pool,
                                              IDirect3DCubeTexture9** out, HANDLE*) override {
    return gbsurfaces::createCubeTexture(this, edge, levels, usage, f, pool, reinterpret_cast<void**>(out));
  }
  HRESULT STDMETHODCALLTYPE CreateVertexBuffer(UINT len, DWORD usage, DWORD fvf, D3DPOOL pool, IDirect3DVertexBuffer9** out,
                                               HANDLE*) override {
    return gbobjects::createVertexBuffer(this, len, usage, fvf, pool, reinterpret_cast<void**>(out));
  }
  HRESULT STDMETHODCALLTYPE CreateIndexBuffer(UINT len, DWORD usage, D3DFORMAT f, D3DPOOL pool, IDirect3DIndexBuffer9** out,
                                              HANDLE*) override {
    return gbobjects::createIndexBuffer(this, len, usage, f, pool, reinterpret_cast<void**>(out));
  }
  HRESULT STDMETHODCALLTYPE CreateRenderTarget(UINT w, UINT h, D3DFORMAT f, D3DMULTISAMPLE_TYPE ms, DWORD q, BOOL,
                                               IDirect3DSurface9** out, HANDLE*) override {
    return gbsurfaces::createSurface(this, w, h, f, ms, q, D3DUSAGE_RENDERTARGET, D3DPOOL_DEFAULT, reinterpret_cast<void**>(out));
  }
  HRESULT STDMETHODCALLTYPE CreateDepthStencilSurface(UINT w, UINT h, D3DFORMAT f, D3DMULTISAMPLE_TYPE ms, DWORD q, BOOL,
                                                      IDirect3DSurface9** out, HANDLE*) override {
    return gbsurfaces::createSurface(this, w, h, f, ms, q, D3DUSAGE_DEPTHSTENCIL, D3DPOOL_DEFAULT, reinterpret_cast<void**>(out));
  }
  HRESULT STDMETHODCALLTYPE UpdateSurface(IDirect3DSurface9* s, const RECT* r, IDirect3DSurface9* d, const POINT* p) override {
    return gbsurfaces::updateSurface(s, r, d, p);
  }
  HRESULT STDMETHODCALLTYPE UpdateTexture(IDirect3DBaseTexture9* s, IDirect3DBaseTexture9* d) override {
    return gbsurfaces::updateTexture(s, d);
  }
  HRESULT STDMETHODCALLTYPE GetRenderTargetData(IDirect3DSurface9* s, IDirect3DSurface9* d) override {
    return gbstate::api::getRenderTargetData(this, s, d);
  }
  HRESULT STDMETHODCALLTYPE GetFrontBufferData(UINT, IDirect3DSurface9*) override { return unsupported(D3DERR_INVALIDCALL); }
  // Mirrored by the backend from the Gamebryo layer (NiDX9Renderer::Copy/FastCopy, render-target group ends).
  HRESULT STDMETHODCALLTYPE StretchRect(IDirect3DSurface9*, const RECT*, IDirect3DSurface9*, const RECT*,
                                        D3DTEXTUREFILTERTYPE) override {
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE ColorFill(IDirect3DSurface9*, const RECT*, D3DCOLOR) override { return unsupported(D3D_OK); }
  HRESULT STDMETHODCALLTYPE CreateOffscreenPlainSurface(UINT w, UINT h, D3DFORMAT f, D3DPOOL pool, IDirect3DSurface9** out,
                                                        HANDLE*) override {
    return gbsurfaces::createSurface(this, w, h, f, D3DMULTISAMPLE_NONE, 0, 0, pool, reinterpret_cast<void**>(out));
  }
  // Binding is mirrored by the backend from the Gamebryo render-target groups; tracked here for the getters.
  HRESULT STDMETHODCALLTYPE SetRenderTarget(DWORD i, IDirect3DSurface9* s) override {
    if (i >= 4 || (i == 0 && !s)) return D3DERR_INVALIDCALL;
    replace(rt_[i], s);
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetRenderTarget(DWORD i, IDirect3DSurface9** out) override {
    if (!out || i >= 4) return D3DERR_INVALIDCALL;
    *out = rt_[i];
    if (!rt_[i]) return D3DERR_NOTFOUND;
    rt_[i]->AddRef();
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetDepthStencilSurface(IDirect3DSurface9* s) override {
    replace(ds_, s);
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetDepthStencilSurface(IDirect3DSurface9** out) override {
    if (!out) return D3DERR_INVALIDCALL;
    *out = ds_;
    if (!ds_) return D3DERR_NOTFOUND;
    ds_->AddRef();
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE BeginScene() override { return D3D_OK; }
  HRESULT STDMETHODCALLTYPE EndScene() override { return D3D_OK; }
  // Do_ClearBuffer's clear is mirrored by its hook, Bink's by movie.cpp.
  HRESULT STDMETHODCALLTYPE Clear(DWORD, const D3DRECT*, DWORD, D3DCOLOR, float, DWORD) override { return D3D_OK; }
  // Fixed-function state: kept for the getters; no translated shader reads it.
  HRESULT STDMETHODCALLTYPE SetTransform(D3DTRANSFORMSTATETYPE t, const D3DMATRIX* m) override {
    if (!m) return D3DERR_INVALIDCALL;
    transforms_[t] = *m;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetTransform(D3DTRANSFORMSTATETYPE t, D3DMATRIX* m) override {
    if (!m) return D3DERR_INVALIDCALL;
    auto it = transforms_.find(t);
    if (it != transforms_.end()) {
      *m = it->second;
    } else {
      *m = {};
      m->_11 = m->_22 = m->_33 = m->_44 = 1.0f;
    }
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE MultiplyTransform(D3DTRANSFORMSTATETYPE t, const D3DMATRIX* m) override {
    if (!m) return D3DERR_INVALIDCALL;
    D3DMATRIX a{};
    GetTransform(t, &a);
    D3DMATRIX r{};
    for (int i = 0; i < 4; i++)
      for (int j = 0; j < 4; j++)
        for (int k = 0; k < 4; k++) r.m[i][j] += a.m[i][k] * m->m[k][j];
    transforms_[t] = r;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetViewport(const D3DVIEWPORT9* vp) override { return gbstate::api::setViewport(this, vp); }
  HRESULT STDMETHODCALLTYPE GetViewport(D3DVIEWPORT9* vp) override {
    if (!vp) return D3DERR_INVALIDCALL;
    const auto& v = gbstate::state().viewport;
    *vp = { v.x, v.y, v.w, v.h, v.minZ, v.maxZ };
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetMaterial(const D3DMATERIAL9* m) override {
    if (!m) return D3DERR_INVALIDCALL;
    material_ = *m;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetMaterial(D3DMATERIAL9* m) override {
    if (!m) return D3DERR_INVALIDCALL;
    *m = material_;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetLight(DWORD i, const D3DLIGHT9* l) override {
    if (!l) return D3DERR_INVALIDCALL;
    lights_[i].light = *l;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetLight(DWORD i, D3DLIGHT9* l) override {
    auto it = lights_.find(i);
    if (!l || it == lights_.end()) return D3DERR_INVALIDCALL;
    *l = it->second.light;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE LightEnable(DWORD i, BOOL on) override {
    auto it = lights_.find(i);
    if (it == lights_.end()) {  // D3D9 creates a default directional light
      D3DLIGHT9 l{};
      l.Type = D3DLIGHT_DIRECTIONAL;
      l.Diffuse = { 1, 1, 1, 0 };
      l.Direction = { 0, 0, 1 };
      it = lights_.emplace(i, LightSlot{ l, FALSE }).first;
    }
    it->second.enabled = on ? 128 : FALSE;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetLightEnable(DWORD i, BOOL* on) override {
    auto it = lights_.find(i);
    if (!on || it == lights_.end()) return D3DERR_INVALIDCALL;
    *on = it->second.enabled;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetClipPlane(DWORD i, const float* p) override {
    if (i >= 6 || !p) return D3DERR_INVALIDCALL;
    return gbstate::api::setClipPlane(this, i, p);
  }
  HRESULT STDMETHODCALLTYPE GetClipPlane(DWORD i, float* p) override {
    if (i >= 6 || !p) return D3DERR_INVALIDCALL;
    memcpy(p, gbstate::state().clip[i], 16);
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetRenderState(D3DRENDERSTATETYPE s, DWORD v) override {
    return gbstate::api::setRenderState(this, uint32_t(s), v);
  }
  HRESULT STDMETHODCALLTYPE GetRenderState(D3DRENDERSTATETYPE s, DWORD* v) override {
    if (!v || uint32_t(s) >= 256) return D3DERR_INVALIDCALL;
    *v = gbstate::state().rs[s];
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE CreateStateBlock(D3DSTATEBLOCKTYPE, IDirect3DStateBlock9** out) override {
    if (out) *out = nullptr;
    return unsupported(D3DERR_INVALIDCALL);
  }
  HRESULT STDMETHODCALLTYPE BeginStateBlock() override { return unsupported(D3DERR_INVALIDCALL); }
  HRESULT STDMETHODCALLTYPE EndStateBlock(IDirect3DStateBlock9** out) override {
    if (out) *out = nullptr;
    return unsupported(D3DERR_INVALIDCALL);
  }
  HRESULT STDMETHODCALLTYPE SetClipStatus(const D3DCLIPSTATUS9* s) override {
    if (!s) return D3DERR_INVALIDCALL;
    clipStatus_ = *s;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetClipStatus(D3DCLIPSTATUS9* s) override {
    if (!s) return D3DERR_INVALIDCALL;
    *s = clipStatus_;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetTexture(DWORD stage, IDirect3DBaseTexture9** out) override {
    if (!out || stage >= 16) return D3DERR_INVALIDCALL;
    *out = static_cast<IDirect3DBaseTexture9*>(gbstate::state().tex[stage]);
    if (*out) (*out)->AddRef();
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetTexture(DWORD stage, IDirect3DBaseTexture9* t) override {
    if (stage >= 16) return D3D_OK;  // vertex-texture samplers (D3DVERTEXTEXTURESAMPLER0+): no shader uses them
    gbstate::dev::setTexture(this, stage, t);
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetTextureStageState(DWORD stage, D3DTEXTURESTAGESTATETYPE t, DWORD* v) override {
    if (!v || stage >= 8 || uint32_t(t) >= 33) return D3DERR_INVALIDCALL;
    *v = tss_[stage][t];
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetTextureStageState(DWORD stage, D3DTEXTURESTAGESTATETYPE t, DWORD v) override {
    if (stage >= 8 || uint32_t(t) >= 33) return D3DERR_INVALIDCALL;
    tss_[stage][t] = v;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetSamplerState(DWORD s, D3DSAMPLERSTATETYPE t, DWORD* v) override {
    if (!v || s >= 16 || uint32_t(t) >= 14) return D3DERR_INVALIDCALL;
    *v = gbstate::state().ss[s][t];
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetSamplerState(DWORD s, D3DSAMPLERSTATETYPE t, DWORD v) override {
    return gbstate::api::setSamplerState(this, s, uint32_t(t), v);
  }
  HRESULT STDMETHODCALLTYPE ValidateDevice(DWORD* passes) override {
    if (passes) *passes = 1;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetPaletteEntries(UINT, const PALETTEENTRY*) override { return D3D_OK; }
  HRESULT STDMETHODCALLTYPE GetPaletteEntries(UINT, PALETTEENTRY*) override { return D3DERR_INVALIDCALL; }
  HRESULT STDMETHODCALLTYPE SetCurrentTexturePalette(UINT p) override {
    palette_ = p;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetCurrentTexturePalette(UINT* p) override {
    if (!p) return D3DERR_INVALIDCALL;
    *p = palette_;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetScissorRect(const RECT* r) override { return gbstate::api::setScissorRect(this, r); }
  HRESULT STDMETHODCALLTYPE GetScissorRect(RECT* r) override {
    if (!r) return D3DERR_INVALIDCALL;
    const auto& s = gbstate::state().scissor;
    *r = { s[0], s[1], s[2], s[3] };
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetSoftwareVertexProcessing(BOOL on) override {
    swvp_ = on;
    return D3D_OK;
  }
  BOOL STDMETHODCALLTYPE GetSoftwareVertexProcessing() override { return swvp_; }
  HRESULT STDMETHODCALLTYPE SetNPatchMode(float n) override {
    npatch_ = n;
    return D3D_OK;
  }
  float STDMETHODCALLTYPE GetNPatchMode() override { return npatch_; }
  HRESULT STDMETHODCALLTYPE DrawPrimitive(D3DPRIMITIVETYPE pt, UINT start, UINT prims) override {
    return gbstate::api::drawPrimitive(this, pt, start, prims);
  }
  HRESULT STDMETHODCALLTYPE DrawIndexedPrimitive(D3DPRIMITIVETYPE pt, INT base, UINT minIdx, UINT nv, UINT startIdx,
                                                 UINT prims) override {
    return gbstate::api::drawIndexedPrimitive(this, pt, base, minIdx, nv, startIdx, prims);
  }
  HRESULT STDMETHODCALLTYPE DrawPrimitiveUP(D3DPRIMITIVETYPE pt, UINT prims, const void* data, UINT stride) override {
    return gbstate::api::drawPrimitiveUP(this, pt, prims, data, stride);
  }
  // No caller in the game (IDA); counted.
  HRESULT STDMETHODCALLTYPE DrawIndexedPrimitiveUP(D3DPRIMITIVETYPE, UINT, UINT, UINT, const void*, D3DFORMAT, const void*,
                                                   UINT) override {
    g_upDrawsDropped++;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE ProcessVertices(UINT, UINT, UINT, IDirect3DVertexBuffer9*, IDirect3DVertexDeclaration9*,
                                            DWORD) override {
    return unsupported(D3DERR_INVALIDCALL);
  }
  HRESULT STDMETHODCALLTYPE CreateVertexDeclaration(const D3DVERTEXELEMENT9* e, IDirect3DVertexDeclaration9** out) override {
    return gbstate::api::createVertexDeclaration(this, e, reinterpret_cast<void**>(out));
  }
  HRESULT STDMETHODCALLTYPE SetVertexDeclaration(IDirect3DVertexDeclaration9* d) override {
    return gbstate::api::setVertexDeclaration(this, d);
  }
  HRESULT STDMETHODCALLTYPE GetVertexDeclaration(IDirect3DVertexDeclaration9** out) override {
    if (!out) return D3DERR_INVALIDCALL;
    *out = static_cast<IDirect3DVertexDeclaration9*>(gbstate::state().decl);
    if (*out) (*out)->AddRef();
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetFVF(DWORD fvf) override {
    gbstate::dev::setFVF(this, fvf);
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetFVF(DWORD* fvf) override {
    if (!fvf) return D3DERR_INVALIDCALL;
    *fvf = gbstate::state().fvf;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE CreateVertexShader(const DWORD* code, IDirect3DVertexShader9** out) override {
    return gbobjects::createVertexShader(this, code, reinterpret_cast<void**>(out));
  }
  HRESULT STDMETHODCALLTYPE SetVertexShader(IDirect3DVertexShader9* s) override { return gbstate::api::setVertexShader(this, s); }
  HRESULT STDMETHODCALLTYPE GetVertexShader(IDirect3DVertexShader9** out) override {
    if (!out) return D3DERR_INVALIDCALL;
    *out = static_cast<IDirect3DVertexShader9*>(gbstate::state().vs);
    if (*out) (*out)->AddRef();
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetVertexShaderConstantF(UINT r, const float* v, UINT n) override {
    return gbstate::dev::setVSConstF(this, r, v, n);
  }
  HRESULT STDMETHODCALLTYPE GetVertexShaderConstantF(UINT r, float* v, UINT n) override {
    if (!v || r + n > 256) return D3DERR_INVALIDCALL;
    memcpy(v, gbstate::state().vsc[r], size_t(n) * 16);
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetVertexShaderConstantI(UINT r, const int* v, UINT n) override {
    if (!v || r + n > 16) return D3DERR_INVALIDCALL;
    memcpy(vsI_[r], v, size_t(n) * 16);
    return gbstate::dev::setVSConstI(this, r, v, n);
  }
  HRESULT STDMETHODCALLTYPE GetVertexShaderConstantI(UINT r, int* v, UINT n) override {
    if (!v || r + n > 16) return D3DERR_INVALIDCALL;
    memcpy(v, vsI_[r], size_t(n) * 16);
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetVertexShaderConstantB(UINT r, const BOOL* v, UINT n) override {
    if (!v || r + n > 16) return D3DERR_INVALIDCALL;
    memcpy(vsB_ + r, v, size_t(n) * sizeof(BOOL));
    return gbstate::dev::setVSConstB(this, r, reinterpret_cast<const int*>(v), n);
  }
  HRESULT STDMETHODCALLTYPE GetVertexShaderConstantB(UINT r, BOOL* v, UINT n) override {
    if (!v || r + n > 16) return D3DERR_INVALIDCALL;
    memcpy(v, vsB_ + r, size_t(n) * sizeof(BOOL));
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetStreamSource(UINT n, IDirect3DVertexBuffer9* vb, UINT off, UINT stride) override {
    if (n >= 16) return D3DERR_INVALIDCALL;
    return gbstate::api::setStreamSource(this, n, vb, off, stride);
  }
  HRESULT STDMETHODCALLTYPE GetStreamSource(UINT n, IDirect3DVertexBuffer9** vb, UINT* off, UINT* stride) override {
    if (!vb || !off || !stride || n >= 8) return D3DERR_INVALIDCALL;
    const auto& s = gbstate::state().streams[n];
    *vb = static_cast<IDirect3DVertexBuffer9*>(s.vb);
    if (*vb) (*vb)->AddRef();
    *off = s.offset;
    *stride = s.stride;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetStreamSourceFreq(UINT n, UINT f) override {
    if (n >= 16) return D3DERR_INVALIDCALL;
    freq_[n] = f;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetStreamSourceFreq(UINT n, UINT* f) override {
    if (n >= 16 || !f) return D3DERR_INVALIDCALL;
    *f = freq_[n];
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetIndices(IDirect3DIndexBuffer9* ib) override { return gbstate::api::setIndices(this, ib); }
  HRESULT STDMETHODCALLTYPE GetIndices(IDirect3DIndexBuffer9** out) override {
    if (!out) return D3DERR_INVALIDCALL;
    *out = static_cast<IDirect3DIndexBuffer9*>(gbstate::state().ib);
    if (*out) (*out)->AddRef();
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE CreatePixelShader(const DWORD* code, IDirect3DPixelShader9** out) override {
    return gbobjects::createPixelShader(this, code, reinterpret_cast<void**>(out));
  }
  HRESULT STDMETHODCALLTYPE SetPixelShader(IDirect3DPixelShader9* s) override { return gbstate::api::setPixelShader(this, s); }
  HRESULT STDMETHODCALLTYPE GetPixelShader(IDirect3DPixelShader9** out) override {
    if (!out) return D3DERR_INVALIDCALL;
    *out = static_cast<IDirect3DPixelShader9*>(gbstate::state().ps);
    if (*out) (*out)->AddRef();
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetPixelShaderConstantF(UINT r, const float* v, UINT n) override {
    return gbstate::dev::setPSConstF(this, r, v, n);
  }
  HRESULT STDMETHODCALLTYPE GetPixelShaderConstantF(UINT r, float* v, UINT n) override {
    if (!v || r + n > 224) return D3DERR_INVALIDCALL;
    memcpy(v, gbstate::state().psc[r], size_t(n) * 16);
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetPixelShaderConstantI(UINT r, const int* v, UINT n) override {
    if (!v || r + n > 16) return D3DERR_INVALIDCALL;
    memcpy(psI_[r], v, size_t(n) * 16);
    return gbstate::dev::setPSConstI(this, r, v, n);
  }
  HRESULT STDMETHODCALLTYPE GetPixelShaderConstantI(UINT r, int* v, UINT n) override {
    if (!v || r + n > 16) return D3DERR_INVALIDCALL;
    memcpy(v, psI_[r], size_t(n) * 16);
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetPixelShaderConstantB(UINT r, const BOOL* v, UINT n) override {
    if (!v || r + n > 16) return D3DERR_INVALIDCALL;
    memcpy(psB_ + r, v, size_t(n) * sizeof(BOOL));
    return gbstate::dev::setPSConstB(this, r, reinterpret_cast<const int*>(v), n);
  }
  HRESULT STDMETHODCALLTYPE GetPixelShaderConstantB(UINT r, BOOL* v, UINT n) override {
    if (!v || r + n > 16) return D3DERR_INVALIDCALL;
    memcpy(v, psB_ + r, size_t(n) * sizeof(BOOL));
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE DrawRectPatch(UINT, const float*, const D3DRECTPATCH_INFO*) override { return unsupported(D3DERR_INVALIDCALL); }
  HRESULT STDMETHODCALLTYPE DrawTriPatch(UINT, const float*, const D3DTRIPATCH_INFO*) override { return unsupported(D3DERR_INVALIDCALL); }
  HRESULT STDMETHODCALLTYPE DeletePatch(UINT) override { return unsupported(D3DERR_INVALIDCALL); }
  HRESULT STDMETHODCALLTYPE CreateQuery(D3DQUERYTYPE t, IDirect3DQuery9** out) override {
    if (t != D3DQUERYTYPE_EVENT && t != D3DQUERYTYPE_OCCLUSION) return D3DERR_NOTAVAILABLE;
    if (!out) return D3D_OK;  // support check
    *out = new Query(this, t);
    return D3D_OK;
  }

  // IDirect3DDevice9Ex
  HRESULT STDMETHODCALLTYPE SetConvolutionMonoKernel(UINT, UINT, float*, float*) override { return D3D_OK; }
  HRESULT STDMETHODCALLTYPE ComposeRects(IDirect3DSurface9*, IDirect3DSurface9*, IDirect3DVertexBuffer9*, UINT,
                                         IDirect3DVertexBuffer9*, D3DCOMPOSERECTSOP, int, int) override {
    return unsupported(D3DERR_INVALIDCALL);
  }
  HRESULT STDMETHODCALLTYPE PresentEx(const RECT*, const RECT*, HWND, const RGNDATA*, DWORD) override { return D3D_OK; }
  HRESULT STDMETHODCALLTYPE GetGPUThreadPriority(INT* p) override {
    if (!p) return D3DERR_INVALIDCALL;
    *p = gpuPriority_;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE SetGPUThreadPriority(INT p) override {
    gpuPriority_ = p;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE WaitForVBlank(UINT) override { return D3D_OK; }
  HRESULT STDMETHODCALLTYPE CheckResourceResidency(IDirect3DResource9**, UINT32) override { return D3D_OK; }
  HRESULT STDMETHODCALLTYPE SetMaximumFrameLatency(UINT n) override {
    latency_ = n ? n : 3;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetMaximumFrameLatency(UINT* n) override {
    if (!n) return D3DERR_INVALIDCALL;
    *n = latency_;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE CheckDeviceState(HWND) override { return D3D_OK; }
  HRESULT STDMETHODCALLTYPE CreateRenderTargetEx(UINT w, UINT h, D3DFORMAT f, D3DMULTISAMPLE_TYPE ms, DWORD q, BOOL l,
                                                 IDirect3DSurface9** out, HANDLE* sh, DWORD) override {
    return CreateRenderTarget(w, h, f, ms, q, l, out, sh);
  }
  HRESULT STDMETHODCALLTYPE CreateOffscreenPlainSurfaceEx(UINT w, UINT h, D3DFORMAT f, D3DPOOL pool, IDirect3DSurface9** out,
                                                          HANDLE* sh, DWORD) override {
    return CreateOffscreenPlainSurface(w, h, f, pool, out, sh);
  }
  HRESULT STDMETHODCALLTYPE CreateDepthStencilSurfaceEx(UINT w, UINT h, D3DFORMAT f, D3DMULTISAMPLE_TYPE ms, DWORD q, BOOL d,
                                                        IDirect3DSurface9** out, HANDLE* sh, DWORD) override {
    return CreateDepthStencilSurface(w, h, f, ms, q, d, out, sh);
  }
  HRESULT STDMETHODCALLTYPE ResetEx(D3DPRESENT_PARAMETERS* pp, D3DDISPLAYMODEEX*) override { return Reset(pp); }
  HRESULT STDMETHODCALLTYPE GetDisplayModeEx(UINT, D3DDISPLAYMODEEX* m, D3DDISPLAYROTATION* r) override {
    D3DDISPLAYMODE dm{};
    if (FAILED(GetDisplayMode(0, &dm))) return D3DERR_INVALIDCALL;
    if (m) {
      m->Width = dm.Width;
      m->Height = dm.Height;
      m->RefreshRate = dm.RefreshRate;
      m->Format = dm.Format;
      m->ScanLineOrdering = D3DSCANLINEORDERING_PROGRESSIVE;
    }
    if (r) *r = D3DDISPLAYROTATION_IDENTITY;
    return D3D_OK;
  }

  const D3DPRESENT_PARAMETERS& presentParameters() const { return pp_; }
  IDirect3DSurface9* backBuffer() const { return backBuffer_; }
  Direct3D* parent() const { return parent_; }

 private:
  static HRESULT unsupported(HRESULT hr) {
    g_unsupportedCalls++;
    return hr;
  }
  static void replace(IDirect3DSurface9*& slot, IDirect3DSurface9* s) {
    if (s) s->AddRef();
    if (slot) slot->Release();
    slot = s;
  }
  // The implicit swap chain's buffers for these presentation parameters (creation, Reset); fills in the defaults D3D9
  // fills in (window-sized back buffer, display format, one buffer).
  void createImplicit(const D3DPRESENT_PARAMETERS& in) {
    pp_ = in;
    HWND wnd = pp_.hDeviceWindow ? pp_.hDeviceWindow : cp_.hFocusWindow;
    if (pp_.Windowed && (!pp_.BackBufferWidth || !pp_.BackBufferHeight) && wnd) {
      RECT rc{};
      GetClientRect(wnd, &rc);
      if (!pp_.BackBufferWidth) pp_.BackBufferWidth = std::max<LONG>(1, rc.right - rc.left);
      if (!pp_.BackBufferHeight) pp_.BackBufferHeight = std::max<LONG>(1, rc.bottom - rc.top);
    }
    if (pp_.BackBufferFormat == D3DFMT_UNKNOWN) pp_.BackBufferFormat = D3DFMT_X8R8G8B8;
    if (!pp_.BackBufferCount) pp_.BackBufferCount = 1;
    for (auto& r : rt_) replace(r, nullptr);
    replace(ds_, nullptr);
    if (backBuffer_) backBuffer_->Release();
    if (autoDepth_) autoDepth_->Release();
    backBuffer_ = autoDepth_ = nullptr;
    void* bb = nullptr;
    gbsurfaces::createSurface(this, pp_.BackBufferWidth, pp_.BackBufferHeight, pp_.BackBufferFormat, pp_.MultiSampleType,
                              pp_.MultiSampleQuality, D3DUSAGE_RENDERTARGET, D3DPOOL_DEFAULT, &bb);
    backBuffer_ = static_cast<IDirect3DSurface9*>(bb);
    replace(rt_[0], backBuffer_);
    if (pp_.EnableAutoDepthStencil) {
      void* ds = nullptr;
      gbsurfaces::createSurface(this, pp_.BackBufferWidth, pp_.BackBufferHeight, pp_.AutoDepthStencilFormat,
                                pp_.MultiSampleType, pp_.MultiSampleQuality, D3DUSAGE_DEPTHSTENCIL, D3DPOOL_DEFAULT, &ds);
      autoDepth_ = static_cast<IDirect3DSurface9*>(ds);
      replace(ds_, autoDepth_);
    }
    logf("plugin D3D9 device: back buffer %ux%u format %d x%u, ms %d, windowed %d, auto depth %d (format %d)", pp_.BackBufferWidth,
         pp_.BackBufferHeight, int(pp_.BackBufferFormat), pp_.BackBufferCount, int(pp_.MultiSampleType), pp_.Windowed,
         pp_.EnableAutoDepthStencil, int(pp_.AutoDepthStencilFormat));
  }

  std::atomic<ULONG> refs_{ 1 };
  Direct3D* parent_;
  D3DDEVICE_CREATION_PARAMETERS cp_{};
  D3DPRESENT_PARAMETERS pp_{};
  SwapChain swapChain_;
  IDirect3DSurface9* backBuffer_ = nullptr;
  IDirect3DSurface9* autoDepth_ = nullptr;
  IDirect3DSurface9* rt_[4] = {};
  IDirect3DSurface9* ds_ = nullptr;
  D3DGAMMARAMP gamma_{};
  BOOL cursor_ = FALSE, swvp_ = FALSE;
  float npatch_ = 0.0f;
  UINT palette_ = 0, latency_ = 3, freq_[16] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
  INT gpuPriority_ = 0;
  std::map<uint32_t, D3DMATRIX> transforms_;
  D3DMATERIAL9 material_{};
  struct LightSlot {
    D3DLIGHT9 light;
    BOOL enabled;
  };
  std::map<DWORD, LightSlot> lights_;
  D3DCLIPSTATUS9 clipStatus_{};
  DWORD tss_[8][33] = {};
  int vsI_[16][4] = {}, psI_[16][4] = {};
  BOOL vsB_[16] = {}, psB_[16] = {};
};

class Direct3D final : public IDirect3D9Ex {
 public:
  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** out) override {
    if (!out) return E_POINTER;
    if (riid == __uuidof(IUnknown) || riid == __uuidof(IDirect3D9) || riid == __uuidof(IDirect3D9Ex)) {
      AddRef();
      *out = this;
      return S_OK;
    }
    *out = nullptr;
    return E_NOINTERFACE;
  }
  ULONG STDMETHODCALLTYPE AddRef() override { return ++refs_; }
  ULONG STDMETHODCALLTYPE Release() override { return refs_ ? --refs_ : 0; }  // process lifetime
  HRESULT STDMETHODCALLTYPE RegisterSoftwareDevice(void*) override { return D3DERR_NOTAVAILABLE; }
  UINT STDMETHODCALLTYPE GetAdapterCount() override { return UINT(g_adapters.size()); }
  HRESULT STDMETHODCALLTYPE GetAdapterIdentifier(UINT a, DWORD, D3DADAPTER_IDENTIFIER9* id) override {
    if (!id || a >= g_adapters.size()) return D3DERR_INVALIDCALL;
    *id = g_adapters[a].id;
    return D3D_OK;
  }
  UINT STDMETHODCALLTYPE GetAdapterModeCount(UINT a, D3DFORMAT f) override {
    if (a >= g_adapters.size() || f != D3DFMT_X8R8G8B8) return 0;
    return UINT(g_adapters[a].modes.size());
  }
  HRESULT STDMETHODCALLTYPE EnumAdapterModes(UINT a, D3DFORMAT f, UINT i, D3DDISPLAYMODE* m) override {
    if (!m || a >= g_adapters.size() || f != D3DFMT_X8R8G8B8 || i >= g_adapters[a].modes.size()) return D3DERR_INVALIDCALL;
    const Mode& md = g_adapters[a].modes[i];
    *m = { md.w, md.h, md.refresh, D3DFMT_X8R8G8B8 };
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetAdapterDisplayMode(UINT a, D3DDISPLAYMODE* m) override {
    if (!m || a >= g_adapters.size()) return D3DERR_INVALIDCALL;
    return currentMode(g_adapters[a], *m) ? D3D_OK : D3DERR_INVALIDCALL;
  }
  HRESULT STDMETHODCALLTYPE CheckDeviceType(UINT a, D3DDEVTYPE t, D3DFORMAT adapterFmt, D3DFORMAT bbFmt, BOOL windowed) override {
    if (a >= g_adapters.size()) return D3DERR_INVALIDCALL;
    if (t != D3DDEVTYPE_HAL || adapterFmt != D3DFMT_X8R8G8B8) return D3DERR_NOTAVAILABLE;
    if (bbFmt == D3DFMT_X8R8G8B8 || bbFmt == D3DFMT_A8R8G8B8 || (windowed && bbFmt == D3DFMT_UNKNOWN)) return D3D_OK;
    return D3DERR_NOTAVAILABLE;
  }
  HRESULT STDMETHODCALLTYPE CheckDeviceFormat(UINT a, D3DDEVTYPE t, D3DFORMAT adapterFmt, DWORD usage, D3DRESOURCETYPE rt,
                                              D3DFORMAT f) override {
    if (a >= g_adapters.size()) return D3DERR_INVALIDCALL;
    if (t != D3DDEVTYPE_HAL) return D3DERR_NOTAVAILABLE;
    return checkDeviceFormat(adapterFmt, usage, rt, f);
  }
  HRESULT STDMETHODCALLTYPE CheckDeviceMultiSampleType(UINT a, D3DDEVTYPE t, D3DFORMAT f, BOOL, D3DMULTISAMPLE_TYPE ms,
                                                       DWORD* q) override {
    if (a >= g_adapters.size()) return D3DERR_INVALIDCALL;
    if (t != D3DDEVTYPE_HAL) return D3DERR_NOTAVAILABLE;
    return checkMultiSample(f, ms, q);
  }
  HRESULT STDMETHODCALLTYPE CheckDepthStencilMatch(UINT a, D3DDEVTYPE t, D3DFORMAT, D3DFORMAT rtFmt, D3DFORMAT dsFmt) override {
    if (a >= g_adapters.size()) return D3DERR_INVALIDCALL;
    if (t != D3DDEVTYPE_HAL || !depthFormat(dsFmt) || !renderTargetFormat(rtFmt)) return D3DERR_NOTAVAILABLE;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE CheckDeviceFormatConversion(UINT a, D3DDEVTYPE t, D3DFORMAT src, D3DFORMAT dst) override {
    if (a >= g_adapters.size()) return D3DERR_INVALIDCALL;
    if (t == D3DDEVTYPE_HAL && renderTargetFormat(src) && (dst == D3DFMT_X8R8G8B8 || dst == D3DFMT_A8R8G8B8)) return D3D_OK;
    return D3DERR_NOTAVAILABLE;
  }
  HRESULT STDMETHODCALLTYPE GetDeviceCaps(UINT a, D3DDEVTYPE t, D3DCAPS9* c) override {
    if (!c || a >= g_adapters.size()) return D3DERR_INVALIDCALL;
    if (t != D3DDEVTYPE_HAL) return D3DERR_NOTAVAILABLE;
    fillCaps(a, t, *c);
    return D3D_OK;
  }
  HMONITOR STDMETHODCALLTYPE GetAdapterMonitor(UINT a) override { return a < g_adapters.size() ? g_adapters[a].monitor : nullptr; }
  HRESULT STDMETHODCALLTYPE CreateDevice(UINT a, D3DDEVTYPE t, HWND focus, DWORD behavior, D3DPRESENT_PARAMETERS* pp,
                                         IDirect3DDevice9** out) override {
    IDirect3DDevice9Ex* d = nullptr;
    const HRESULT hr = CreateDeviceEx(a, t, focus, behavior, pp, nullptr, out ? &d : nullptr);
    if (out) *out = d;
    return hr;
  }
  UINT STDMETHODCALLTYPE GetAdapterModeCountEx(UINT a, const D3DDISPLAYMODEFILTER* f) override {
    return GetAdapterModeCount(a, f ? f->Format : D3DFMT_X8R8G8B8);
  }
  HRESULT STDMETHODCALLTYPE EnumAdapterModesEx(UINT a, const D3DDISPLAYMODEFILTER* f, UINT i, D3DDISPLAYMODEEX* m) override {
    D3DDISPLAYMODE dm{};
    const HRESULT hr = EnumAdapterModes(a, f ? f->Format : D3DFMT_X8R8G8B8, i, &dm);
    if (FAILED(hr) || !m) return FAILED(hr) ? hr : D3DERR_INVALIDCALL;
    m->Width = dm.Width;
    m->Height = dm.Height;
    m->RefreshRate = dm.RefreshRate;
    m->Format = dm.Format;
    m->ScanLineOrdering = D3DSCANLINEORDERING_PROGRESSIVE;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetAdapterDisplayModeEx(UINT a, D3DDISPLAYMODEEX* m, D3DDISPLAYROTATION* r) override {
    D3DDISPLAYMODE dm{};
    const HRESULT hr = GetAdapterDisplayMode(a, &dm);
    if (FAILED(hr)) return hr;
    if (m) {
      m->Width = dm.Width;
      m->Height = dm.Height;
      m->RefreshRate = dm.RefreshRate;
      m->Format = dm.Format;
      m->ScanLineOrdering = D3DSCANLINEORDERING_PROGRESSIVE;
    }
    if (r) *r = D3DDISPLAYROTATION_IDENTITY;
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE CreateDeviceEx(UINT a, D3DDEVTYPE t, HWND focus, DWORD behavior, D3DPRESENT_PARAMETERS* pp,
                                           D3DDISPLAYMODEEX*, IDirect3DDevice9Ex** out) override {
    if (!out || !pp || a >= g_adapters.size()) return D3DERR_INVALIDCALL;
    *out = nullptr;
    if (t != D3DDEVTYPE_HAL) return D3DERR_NOTAVAILABLE;
    auto* d = new Device(this, a, t, focus, behavior, *pp);
    *pp = d->presentParameters();  // D3D9 writes the filled-in parameters back
    *out = d;
    logf("plugin D3D9 device created (adapter %u, behavior %08lX); D3D9 is not loaded", a, behavior);
    return D3D_OK;
  }
  HRESULT STDMETHODCALLTYPE GetAdapterLUID(UINT a, LUID* l) override {
    if (!l || a >= g_adapters.size()) return D3DERR_INVALIDCALL;
    *l = g_adapters[a].luid;
    return D3D_OK;
  }

 private:
  std::atomic<ULONG> refs_{ 1 };
};

UINT Device::GetAvailableTextureMem() {
  const uint64_t m = cp_.AdapterOrdinal < g_adapters.size() ? g_adapters[cp_.AdapterOrdinal].dedicatedMemory : 0;
  return UINT(std::min<uint64_t>(m ? m : (512ull << 20), 0x7FF00000ull));  // a signed 32-bit value for 32-bit callers
}
HRESULT Device::GetDirect3D(IDirect3D9** out) {
  if (!out) return D3DERR_INVALIDCALL;
  parent_->AddRef();
  *out = parent_;
  return D3D_OK;
}

HRESULT SwapChain::GetBackBuffer(UINT i, D3DBACKBUFFER_TYPE t, IDirect3DSurface9** out) { return dev_->GetBackBuffer(0, i, t, out); }
HRESULT SwapChain::GetDisplayMode(D3DDISPLAYMODE* m) { return dev_->GetDisplayMode(0, m); }
HRESULT SwapChain::GetDevice(IDirect3DDevice9** out) {
  if (!out) return D3DERR_INVALIDCALL;
  dev_->AddRef();
  *out = dev_;
  return D3D_OK;
}
HRESULT SwapChain::GetPresentParameters(D3DPRESENT_PARAMETERS* pp) {
  if (!pp) return D3DERR_INVALIDCALL;
  *pp = dev_->presentParameters();
  return D3D_OK;
}
HRESULT SwapChain::GetDisplayModeEx(D3DDISPLAYMODEEX* m, D3DDISPLAYROTATION* r) { return dev_->GetDisplayModeEx(0, m, r); }

Direct3D* g_d3d = nullptr;
Direct3D* direct3d() {
  if (!g_d3d) {
    enumerateAdapters();
    if (!g_adapters.empty()) g_vendor = g_adapters[0].id.VendorId;
    g_d3d = new Direct3D();
  } else {
    g_d3d->AddRef();
  }
  return g_d3d;
}
IDirect3D9* WINAPI create9(UINT) { return direct3d(); }
HRESULT WINAPI create9Ex(UINT, IDirect3D9Ex** out) {
  if (!out) return D3DERR_INVALIDCALL;
  *out = direct3d();
  return D3D_OK;
}

// NiDX9Renderer::IsD3D9Create 0x00E69410: the creators are the plugin's (ms_pfnD3DCreate9 0x0126F0DC,
// Direct3DCreate9Ex 0x0126F0E0); D3D9.DLL is not loaded.
bool __cdecl isD3D9Create() {
  *reinterpret_cast<void**>(0x0126F0DC) = reinterpret_cast<void*>(&create9);
  *reinterpret_cast<void**>(0x0126F0E0) = reinterpret_cast<void*>(&create9Ex);
  return true;
}

// TESRenderer::Init's NVIDIA SLI check passes IDirect3D9* to NvAPI_D3D_GetCurrentSLIState (interface id 0x4B708B54,
// thunk 0x009FEBD8, called at 0x004DB479). That function takes a D3D device, so the driver refuses an IDirect3D9 with an
// error, which sends the engine down its GPU-count comparison (SLI only if fewer logical than physical GPUs). The
// plugin's object is not the driver's, so the call is not made: it returns NVAPI_INVALID_ARGUMENT (-5) for it.
int __cdecl nvGetCurrentSliState(void*, void*) { return -5; }

}  // namespace

bool install() {
  void* orig = nullptr;  // replaced outright
  bool ok = hookFunction(0x00E69410, reinterpret_cast<void*>(&isD3D9Create), &orig, "NiDX9Renderer::IsD3D9Create (plugin D3D9)");
  uint8_t* p = reinterpret_cast<uint8_t*>(0x004DB479);
  int32_t rel = 0;
  memcpy(&rel, p + 1, 4);
  if (p[0] == 0xE8 && uintptr_t(p + 5 + rel) == 0x009FEBD8) {
    DWORD old = 0;
    VirtualProtect(p, 5, PAGE_EXECUTE_READWRITE, &old);
    rel = int32_t(reinterpret_cast<uintptr_t>(&nvGetCurrentSliState) - uintptr_t(p + 5));
    memcpy(p + 1, &rel, 4);
    VirtualProtect(p, 5, old, &old);
    FlushInstructionCache(GetCurrentProcess(), p, 5);
  } else {
    logf("NVAPI SLI-state call at 004DB479 does not match; not patched");
    ok = false;
  }
  logf("plugin D3D9: %s", ok ? "installed (D3D9.DLL will not be loaded by the renderer)" : "INSTALL FAILED");
  return ok;
}

void logSummary() {
  logf("plugin D3D9 device: unsupported calls %llu, DrawIndexedPrimitiveUP dropped %llu, volume textures refused %llu",
       g_unsupportedCalls.load(), g_upDrawsDropped.load(), g_volumeRefused.load());
}

}  // namespace fnvvk::gbdevice
