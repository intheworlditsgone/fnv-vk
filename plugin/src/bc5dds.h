#pragma once
// A DDS file holding a block-compressed BC4 (one channel: FOURCC "BC4U" or "ATI1") or BC5 (two channels: "BC5U" or
// "ATI2") texture in the legacy header (no DX10 header), as the skin's detail normal map (BC5, docs/SKIN.md) and the
// hair's tangent shift map (BC4, docs/LIGHTING.md "Hair") ship. The game's own loader (D3DX, d3dxtex) has neither; this
// reads the file's levels as they are (8 bytes per 4 x 4 block for BC4, 16 for BC5), for Backend::createTexture (the
// format is the FOURCC).
#include <cstdint>
#include <cstring>
#include <string>

#include "d3dxtex/texture.h"

namespace fnvvk {

// `formats`: which of the two formats the caller takes (kBcDdsBC4, kBcDdsBC5 or both).
constexpr uint32_t kBcDdsBC4 = 1, kBcDdsBC5 = 2;
inline bool bcDdsTexture(const uint8_t* file, size_t size, uint32_t formats, d3dx::Texture& out, std::string* why = nullptr) {
  auto fail = [&](const char* s) {
    if (why) *why = s;
    return false;
  };
  auto u32 = [&](size_t off) {
    uint32_t v;
    memcpy(&v, file + off, 4);
    return v;
  };
  if (size < 128 || memcmp(file, "DDS ", 4) != 0 || u32(4) != 124) return fail("not a DDS file");
  const uint32_t height = u32(12), width = u32(16), mips = u32(28), pfFlags = u32(80), fourcc = u32(84);
  const uint32_t kBC4U = 0x55344342, kATI1 = 0x31495441, kBC5U = 0x55354342, kATI2 = 0x32495441, kFourccFlag = 0x4;
  const bool bc4 = (formats & kBcDdsBC4) && (fourcc == kBC4U || fourcc == kATI1);
  const bool bc5 = (formats & kBcDdsBC5) && (fourcc == kBC5U || fourcc == kATI2);
  if (!(pfFlags & kFourccFlag) || !(bc4 || bc5))
    return fail(formats == kBcDdsBC4 ? "not BC4 (BC4U / ATI1)" : formats == kBcDdsBC5 ? "not BC5 (BC5U / ATI2)" : "not BC4 or BC5");
  if (u32(112) & 0x200) return fail("a cube map");
  if (!width || !height || width > 16384 || height > 16384) return fail("bad size");
  const uint32_t blockBytes = bc4 ? 8 : 16;
  out = d3dx::Texture{};
  out.format = fourcc;
  out.width = width;
  out.height = height;
  out.levels = mips ? mips : 1;
  out.cube = false;
  size_t off = 128;
  for (uint32_t l = 0; l < out.levels; l++) {
    d3dx::Level lv;
    lv.width = width >> l ? width >> l : 1;
    lv.height = height >> l ? height >> l : 1;
    lv.rowPitch = ((lv.width + 3) / 4) * blockBytes;
    lv.rows = (lv.height + 3) / 4;
    const size_t bytes = size_t(lv.rowPitch) * lv.rows;
    if (off + bytes > size) return fail("truncated");
    lv.data.assign(file + off, file + off + bytes);
    off += bytes;
    out.surfaces.push_back(std::move(lv));
  }
  return true;
}

inline bool bc5DdsTexture(const uint8_t* file, size_t size, d3dx::Texture& out, std::string* why = nullptr) {
  return bcDdsTexture(file, size, kBcDdsBC5, out, why);
}

}  // namespace fnvvk
