#pragma once
// Image files through the Windows Imaging Component (the system's PNG, JPEG, BMP and JPEG XR codecs). No game or
// backend dependencies: the screenshot tool and its test (tools/screenshot_test) both use it.
#include <cstdint>
#include <string>
#include <vector>

namespace fnvvk::imagefile {

enum class Format : int { Png = 0, Jpeg = 1, Bmp = 2 };
const wchar_t* extension(Format f);  // L".png", ...

// 8-bit sRGB image, rows of width * 3 bytes in B, G, R order (no alpha: a screenshot is opaque).
// quality 1..100 (JPEG only). The calling thread must have COM initialised.
bool writeSdr(const std::wstring& path, Format f, const uint8_t* bgr, uint32_t width, uint32_t height, int quality,
              std::string& error);

// HDR image as JPEG XR, scRGB: linear BT.709 primaries, 1.0 = 80 nits, rows of width * 4 half floats (R, G, B, A),
// the format Windows (Game Bar, Photos) uses for HDR screenshots. quality 1..100; 100 is lossless.
bool writeScRgbJxr(const std::wstring& path, const uint16_t* rgbaHalf, uint32_t width, uint32_t height, int quality,
                   std::string& error);

// Decodes any WIC-readable file to 32bppBGRA (8-bit) or 64bppRGBAHalf (hdr). Verification.
bool read(const std::wstring& path, bool hdr, std::vector<uint8_t>& texels, uint32_t& width, uint32_t& height,
          std::string& error);

}  // namespace fnvvk::imagefile
