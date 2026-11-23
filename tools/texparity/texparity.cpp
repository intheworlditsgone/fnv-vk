// Texture-level parity: native loader (backend/d3dxtex/texture.cpp) vs the game's d3dx9_38.dll
// D3DXCreateTextureFromFileInMemory / D3DXCreateCubeTextureFromFileInMemory, called as the game calls them, under
// each x87 precision the game uses. Every level of every face is compared byte for byte.
//
// usage: texparity <file.dds | directory>...      (x86; exit 0 only if everything is identical)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d9.h>

#include <float.h>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "d3dxtex/texture.h"

namespace fs = std::filesystem;
using namespace fnvvk::d3dx;

using CreateTexFn = HRESULT(WINAPI*)(IDirect3DDevice9*, const void*, UINT, IDirect3DTexture9**);
using CreateCubeFn = HRESULT(WINAPI*)(IDirect3DDevice9*, const void*, UINT, IDirect3DCubeTexture9**);

static void setPrecision(Precision p) {
  unsigned cur;
  _controlfp_s(&cur, p == Precision::Single ? _PC_24 : _PC_53, _MCW_PC);
}

static std::vector<uint8_t> readFile(const fs::path& p) {
  std::vector<uint8_t> v;
  FILE* f = _wfopen(p.c_str(), L"rb");
  if (!f) return v;
  fseek(f, 0, SEEK_END);
  v.resize(size_t(ftell(f)));
  fseek(f, 0, SEEK_SET);
  fread(v.data(), 1, v.size(), f);
  fclose(f);
  return v;
}

struct Totals { uint64_t files = 0, compared = 0, identical = 0, levelsCompared = 0, levelsDiffer = 0, unsupported = 0, d3dxFailed = 0; };

int main(int argc, char** argv) {
  HMODULE dx = LoadLibraryA("d3dx9_38.dll");
  if (!dx) { printf("d3dx9_38.dll missing\n"); return 2; }
  auto createTex = reinterpret_cast<CreateTexFn>(GetProcAddress(dx, "D3DXCreateTextureFromFileInMemory"));
  auto createCube = reinterpret_cast<CreateCubeFn>(GetProcAddress(dx, "D3DXCreateCubeTextureFromFileInMemory"));

  WNDCLASSA wc{};
  wc.lpfnWndProc = DefWindowProcA;
  wc.hInstance = GetModuleHandleA(nullptr);
  wc.lpszClassName = "texparity";
  RegisterClassA(&wc);
  HWND wnd = CreateWindowA(wc.lpszClassName, "", WS_OVERLAPPEDWINDOW, 0, 0, 64, 64, nullptr, nullptr, wc.hInstance, nullptr);
  IDirect3D9* d3d = Direct3DCreate9(D3D_SDK_VERSION);
  D3DPRESENT_PARAMETERS pp{};
  pp.Windowed = TRUE;
  pp.SwapEffect = D3DSWAPEFFECT_DISCARD;
  pp.hDeviceWindow = wnd;
  IDirect3DDevice9* dev = nullptr;
  // FPU_PRESERVE so the precision forced before each D3DX call is the one D3DX's code runs under.
  if (FAILED(d3d->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, wnd,
                               D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE | D3DCREATE_MULTITHREADED,
                               &pp, &dev))) {
    printf("no D3D9 device\n");
    return 2;
  }

  std::vector<fs::path> files;
  for (int a = 1; a < argc; a++) {
    fs::path p = argv[a];
    if (fs::is_directory(p)) {
      for (auto& e : fs::recursive_directory_iterator(p))
        if (e.is_regular_file() && _wcsicmp(e.path().extension().c_str(), L".dds") == 0) files.push_back(e.path());
    } else {
      files.push_back(p);
    }
  }

  Totals t;
  int printed = 0;
  for (const fs::path& path : files) {
    std::vector<uint8_t> data = readFile(path);
    t.files++;
    for (Precision prec : { Precision::Single, Precision::Double }) {
      Texture native;
      std::string why;
      Status st = createTextureFromDds(data.data(), data.size(), prec, native, &why);
      if (st != Status::Ok) {
        if (prec == Precision::Single) {
          t.unsupported++;
          if (printed++ < 20) printf("native: %s: %s\n", path.filename().string().c_str(), why.c_str());
        }
        continue;
      }
      setPrecision(prec);
      IDirect3DBaseTexture9* base = nullptr;
      HRESULT hr;
      if (native.cube) {
        IDirect3DCubeTexture9* c = nullptr;
        hr = createCube(dev, data.data(), UINT(data.size()), &c);
        base = c;
      } else {
        IDirect3DTexture9* tx = nullptr;
        hr = createTex(dev, data.data(), UINT(data.size()), &tx);
        base = tx;
      }
      setPrecision(Precision::Double);
      if (FAILED(hr) || !base) {
        t.d3dxFailed++;
        continue;
      }
      t.compared++;
      bool same = base->GetLevelCount() == native.levels;
      uint32_t faces = native.cube ? 6 : 1;
      for (uint32_t f = 0; f < faces && same; f++)
        for (uint32_t l = 0; l < native.levels; l++) {
          const Level& nl = native.surfaces[size_t(f) * native.levels + l];
          D3DLOCKED_RECT lr{};
          D3DSURFACE_DESC desc{};
          HRESULT lh;
          if (native.cube) {
            static_cast<IDirect3DCubeTexture9*>(base)->GetLevelDesc(l, &desc);
            lh = static_cast<IDirect3DCubeTexture9*>(base)->LockRect(D3DCUBEMAP_FACES(f), l, &lr, nullptr, D3DLOCK_READONLY);
          } else {
            static_cast<IDirect3DTexture9*>(base)->GetLevelDesc(l, &desc);
            lh = static_cast<IDirect3DTexture9*>(base)->LockRect(l, &lr, nullptr, D3DLOCK_READONLY);
          }
          if (FAILED(lh)) { same = false; break; }
          bool levelSame = desc.Format == D3DFORMAT(native.format) && desc.Width == nl.width && desc.Height == nl.height;
          size_t badRow = SIZE_MAX;
          for (uint32_t r = 0; r < nl.rows && levelSame; r++)
            if (memcmp(static_cast<uint8_t*>(lr.pBits) + size_t(r) * lr.Pitch, nl.data.data() + size_t(r) * nl.rowPitch, nl.rowPitch)) {
              levelSame = false;
              badRow = r;
            }
          if (native.cube) static_cast<IDirect3DCubeTexture9*>(base)->UnlockRect(D3DCUBEMAP_FACES(f), l);
          else static_cast<IDirect3DTexture9*>(base)->UnlockRect(l);
          t.levelsCompared++;
          if (!levelSame) {
            t.levelsDiffer++;
            same = false;
            if (printed++ < 40)
              printf("DIFF %s %s face %u level %u (%ux%u fmt %u vs d3dx %ux%u fmt %u) first differing row %zd\n",
                     path.filename().string().c_str(), prec == Precision::Single ? "PC24" : "PC53", f, l, nl.width,
                     nl.height, native.format, desc.Width, desc.Height, unsigned(desc.Format),
                     badRow == SIZE_MAX ? -1 : ptrdiff_t(badRow));
          }
        }
      if (same) t.identical++;
      base->Release();
    }
  }
  printf("files %llu | textures compared %llu (both precisions), identical %llu | levels compared %llu, differing %llu | "
         "native unsupported %llu, d3dx failed %llu | out-of-range step indices %llu\n",
         t.files, t.compared, t.identical, t.levelsCompared, t.levelsDiffer, t.unsupported, t.d3dxFailed,
         outOfRangeStepCount());
  bool pass = t.compared && t.identical == t.compared;
  printf(pass ? "RESULT: PASS\n" : "RESULT: FAIL\n");
  return pass ? 0 : 1;
}
