#pragma once
// A DDS file holding a two-channel BC5 texture (FOURCC "BC5U" or "ATI2" in the legacy header; no DX10 header), as the
// skin's detail normal map ships (docs/SKIN.md). The game's own loader (D3DX, d3dxtex) has no BC5; this reads the file's
// levels as they are, 16 bytes per 4 x 4 block, for Backend::createTexture (the format is the FOURCC).
#include <cstdint>
#include <cstring>
#include <string>

#include "d3dxtex/texture.h"

namespace fnvvk {

inline bool bc5DdsTexture(const uint8_t* file, size_t size, d3dx::Texture& out, std::string* why = nullptr) {
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
  const uint32_t kBC5U = 0x55354342, kATI2 = 0x32495441, kFourccFlag = 0x4;
  if (!(pfFlags & kFourccFlag) || (fourcc != kBC5U && fourcc != kATI2)) return fail("not BC5 (BC5U / ATI2)");
  if (u32(112) & 0x200) return fail("a cube map");
  if (!width || !height || width > 16384 || height > 16384) return fail("bad size");
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
    lv.rowPitch = ((lv.width + 3) / 4) * 16;
    lv.rows = (lv.height + 3) / 4;
    const size_t bytes = size_t(lv.rowPitch) * lv.rows;
    if (off + bytes > size) return fail("truncated");
    lv.data.assign(file + off, file + off + bytes);
    off += bytes;
    out.surfaces.push_back(std::move(lv));
  }
  return true;
}

}  // namespace fnvvk
