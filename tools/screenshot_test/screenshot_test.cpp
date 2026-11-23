// Screenshot files (plugin/src/imagefile.cpp, screenshot.h toBgr): each format written and decoded again by WIC.
//   PNG, BMP: exact. JPEG: close (lossy). JPEG XR scRGB: HDR values (above 1.0) survive, lossless at quality 100.
//   The gamma table applies per channel in B, G, R order, as backend/builtin/gamma.frag does.
//   A folder name outside the ANSI code page works.
// Usage: screenshot_test [output dir]   (default: %TEMP%\fnv_vk_screenshot_test)
#include <windows.h>
#include <objbase.h>
#include <shlobj.h>
#include <wincodec.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "imagefile.h"
#include "screenshot.h"

using namespace fnvvk;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
  if (!ok) g_failures++;
}

float halfToFloat(uint16_t h) {
  const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
  float v;
  if (e == 0) v = std::ldexp(float(m), -24);
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = std::ldexp(float(m | 0x400), int(e) - 25);
  return s ? -v : v;
}
uint16_t floatToHalf(float f) {  // round to nearest; the test values are normal halves
  uint32_t x;
  memcpy(&x, &f, 4);
  const uint32_t s = (x >> 16) & 0x8000;
  int e = int((x >> 23) & 0xFF) - 127 + 15;
  uint32_t m = x & 0x7FFFFF;
  if (e <= 0) return uint16_t(s);
  if (e >= 31) return uint16_t(s | 0x7C00);
  uint32_t h = s | (uint32_t(e) << 10) | (m >> 13);
  if ((m & 0x1FFF) > 0x1000 || ((m & 0x1FFF) == 0x1000 && (h & 1))) h++;
  return uint16_t(h);
}

// The pixel format WIC reports for the file's frame (what a viewer sees before any conversion).
GUID nativeFormat(const std::wstring& path) {
  GUID g = GUID_NULL;
  IWICImagingFactory* f = nullptr;
  IWICBitmapDecoder* d = nullptr;
  IWICBitmapFrameDecode* fr = nullptr;
  if (SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&f))) &&
      SUCCEEDED(f->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &d)) &&
      SUCCEEDED(d->GetFrame(0, &fr)))
    fr->GetPixelFormat(&g);
  if (fr) fr->Release();
  if (d) d->Release();
  if (f) f->Release();
  return g;
}

long long fileSize(const std::wstring& path) {
  WIN32_FILE_ATTRIBUTE_DATA a{};
  if (!GetFileAttributesExW(path.c_str(), GetFileExInfoStandard, &a)) return -1;
  return (long long(a.nFileSizeHigh) << 32) | a.nFileSizeLow;
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  CoInitializeEx(nullptr, COINIT_MULTITHREADED);
  std::wstring dir;
  if (argc > 1) {
    dir = argv[1];
  } else {
    wchar_t tmp[MAX_PATH];
    GetTempPathW(MAX_PATH, tmp);
    dir = std::wstring(tmp) + L"fnv_vk_screenshot_test";
  }
  // A name outside the ANSI code page (Cyrillic, Japanese): the paths are wide all the way.
  dir += L"\\Снимки スクリーン";
  const int mk = SHCreateDirectoryExW(nullptr, dir.c_str(), nullptr);
  check(mk == ERROR_SUCCESS || mk == ERROR_ALREADY_EXISTS, "create a folder with a non-ANSI name");

  // A frame as Backend::readback returns it: B, G, R, A with alpha not 255 (must not reach the file).
  const uint32_t W = 643, H = 361;  // odd sizes: rows are not 4-byte aligned in 24-bit formats
  std::vector<uint8_t> bgra(size_t(W) * H * 4);
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++) {
      uint8_t* p = &bgra[(size_t(y) * W + x) * 4];
      p[0] = uint8_t(x * 255 / (W - 1));          // B: horizontal ramp
      p[1] = uint8_t(y * 255 / (H - 1));          // G: vertical ramp
      p[2] = uint8_t(((x / 16 + y / 16) & 1) ? 230 : 20);  // R: checker
      p[3] = uint8_t(x & 0x7F);
    }

  // toBgr through a gamma table: distinct per channel, so a swapped channel shows.
  uint8_t lut[3][256];
  for (int i = 0; i < 256; i++) {
    lut[0][i] = uint8_t(255 - i);                                         // red: inverted
    lut[1][i] = uint8_t(std::lround(255.0 * std::pow(i / 255.0, 0.8)));   // green: brighter
    lut[2][i] = uint8_t(i / 2);                                           // blue: halved
  }
  std::vector<uint8_t> bgr(size_t(W) * H * 3);
  screenshot::toBgr(bgra.data(), size_t(W) * H, lut, bgr.data());
  bool lutOk = true;
  for (size_t i = 0; i < size_t(W) * H; i++)
    lutOk &= bgr[i * 3 + 0] == lut[2][bgra[i * 4 + 0]] && bgr[i * 3 + 1] == lut[1][bgra[i * 4 + 1]] &&
             bgr[i * 3 + 2] == lut[0][bgra[i * 4 + 2]];
  check(lutOk, "toBgr: red, green and blue each through their own table");
  uint8_t identity[3][256];
  for (int c = 0; c < 3; c++)
    for (int i = 0; i < 256; i++) identity[c][i] = uint8_t(i);
  screenshot::toBgr(bgra.data(), size_t(W) * H, identity, bgr.data());
  bool idOk = true;
  for (size_t i = 0; i < size_t(W) * H; i++)
    idOk &= bgr[i * 3] == bgra[i * 4] && bgr[i * 3 + 1] == bgra[i * 4 + 1] && bgr[i * 3 + 2] == bgra[i * 4 + 2];
  check(idOk, "toBgr: the identity table keeps the frame");

  struct Case { imagefile::Format f; int quality; int maxError; double maxMean; const char* name; };
  const Case cases[] = {
    { imagefile::Format::Png, 95, 0, 0.0, "PNG" },
    { imagefile::Format::Bmp, 95, 0, 0.0, "BMP" },
    { imagefile::Format::Jpeg, 95, 64, 2.0, "JPEG q95" },
    { imagefile::Format::Jpeg, 20, 160, 8.0, "JPEG q20" },
  };
  long long jpegSize[2] = {};
  int jpegIdx = 0;
  for (const Case& c : cases) {
    const std::wstring path = dir + L"\\shot_" + std::to_wstring(int(c.f)) + L"_" + std::to_wstring(c.quality) +
                              imagefile::extension(c.f);
    std::string err;
    const bool wrote = imagefile::writeSdr(path, c.f, bgr.data(), W, H, c.quality, err);
    char what[160];
    snprintf(what, sizeof(what), "%s: written%s%s", c.name, err.empty() ? "" : ": ", err.c_str());
    check(wrote, what);
    std::vector<uint8_t> back;
    uint32_t w = 0, h = 0;
    const bool read = wrote && imagefile::read(path, false, back, w, h, err);
    snprintf(what, sizeof(what), "%s: decoded at %ux%u", c.name, w, h);
    check(read && w == W && h == H, what);
    if (!read || w != W || h != H) continue;
    int worst = 0;
    double sum = 0;
    bool opaque = true;
    for (size_t i = 0; i < size_t(W) * H; i++) {
      for (int k = 0; k < 3; k++) {
        const int d = std::abs(int(back[i * 4 + k]) - int(bgr[i * 3 + k]));
        worst = d > worst ? d : worst;
        sum += d;
      }
      opaque &= back[i * 4 + 3] == 255;
    }
    const double mean = sum / (double(W) * H * 3);
    snprintf(what, sizeof(what), "%s: pixels match (worst %d <= %d, mean %.3f <= %.1f)", c.name, worst, c.maxError, mean,
             c.maxMean);
    check(worst <= c.maxError && mean <= c.maxMean, what);
    snprintf(what, sizeof(what), "%s: opaque", c.name);
    check(opaque, what);
    if (c.f == imagefile::Format::Jpeg) jpegSize[jpegIdx++] = fileSize(path);
  }
  char sizes[96];
  snprintf(sizes, sizeof(sizes), "JPEG quality changes the file (q95 %lld bytes > q20 %lld bytes)", jpegSize[0], jpegSize[1]);
  check(jpegSize[0] > jpegSize[1] && jpegSize[1] > 0, sizes);

  // HDR: scRGB as Backend::readbackHdrEncode(ScRgb) gives it, with highlights up to 12.5 (1000 nits) and alpha set by
  // the caller to 1.0.
  std::vector<uint16_t> hdr(size_t(W) * H * 4);
  for (uint32_t y = 0; y < H; y++)
    for (uint32_t x = 0; x < W; x++) {
      uint16_t* p = &hdr[(size_t(y) * W + x) * 4];
      const float t = float(x) / float(W - 1);
      p[0] = floatToHalf(t * 12.5f);                       // R: 0 .. 1000 nits
      p[1] = floatToHalf(float(y) / float(H - 1) * 2.0f);  // G: 0 .. 160 nits
      p[2] = floatToHalf(((x / 16 + y / 16) & 1) ? 0.25f : 4.0f);
      p[3] = 0x3C00;
    }
  for (int quality : { 100, 90 }) {
    const std::wstring path = dir + L"\\shot_hdr_" + std::to_wstring(quality) + L".jxr";
    std::string err;
    const bool wrote = imagefile::writeScRgbJxr(path, hdr.data(), W, H, quality, err);
    char what[200];
    snprintf(what, sizeof(what), "JPEG XR q%d: written%s%s", quality, err.empty() ? "" : ": ", err.c_str());
    check(wrote, what);
    const GUID native = wrote ? nativeFormat(path) : GUID_NULL;
    snprintf(what, sizeof(what), "JPEG XR q%d: stored as half-float RGB(A) (an HDR image to Windows)", quality);
    check(IsEqualGUID(native, GUID_WICPixelFormat64bppRGBAHalf) || IsEqualGUID(native, GUID_WICPixelFormat64bppRGBHalf), what);
    std::vector<uint8_t> back;
    uint32_t w = 0, h = 0;
    const bool read = wrote && imagefile::read(path, true, back, w, h, err);
    snprintf(what, sizeof(what), "JPEG XR q%d: decoded at %ux%u", quality, w, h);
    check(read && w == W && h == H, what);
    if (!read || w != W || h != H) continue;
    const uint16_t* b = reinterpret_cast<const uint16_t*>(back.data());
    double worstRel = 0, maxR = 0;
    for (size_t i = 0; i < size_t(W) * H; i++)
      for (int k = 0; k < 3; k++) {
        const float want = halfToFloat(hdr[i * 4 + k]), got = halfToFloat(b[i * 4 + k]);
        worstRel = std::max(worstRel, double(std::fabs(got - want)) / std::max(0.25, double(want)));
        if (k == 0) maxR = std::max(maxR, double(got));
      }
    const double limit = quality == 100 ? 0.001 : 0.05;
    snprintf(what, sizeof(what), "JPEG XR q%d: values match (worst relative error %.5f <= %.3f; brightest red %.2f)",
             quality, worstRel, limit, maxR);
    check(worstRel <= limit && maxR > 12.0, what);
  }

  // A failed write leaves no file behind.
  {
    const std::wstring path = dir + L"\\no_such_folder\\x.png";
    std::string err;
    const bool wrote = imagefile::writeSdr(path, imagefile::Format::Png, bgr.data(), W, H, 95, err);
    check(!wrote && !err.empty() && GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES,
          "a write into a missing folder fails with a message and no file");
    printf("      (%s)\n", err.c_str());
  }

  printf("\n%s: %d failure(s); files in the test folder\n", g_failures ? "FAILED" : "ALL PASSED", g_failures);
  return g_failures ? 1 : 0;
}
