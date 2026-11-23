#pragma once
// Native equivalent of d3dx9_38's D3DXCreateTextureFromFileInMemory / D3DXCreateCubeTextureFromFileInMemory for DDS
// input, called exactly as NiDX9SourceTextureData::LoadTextureFile (0x00E68A80) calls it: default size, full mip
// chain, file format, filter TRIANGLE|DITHER, mip filter BOX. Produces the texel data D3D9 ends up holding, level
// by level, so the Vulkan backend can upload it. Behaviour follows the binary (docs/M2_RESULTS.md):
//  * CImage::LoadDDS parsing (mip count used even without DDSD_MIPMAPCOUNT, all-six-faces cubes, size checks,
//    the binary's pixel-format table);
//  * R8G8B8 is created as X8R8G8B8 (X written as 0);
//  * file levels: CBlt::BltSame / BltSame_DXTn byte copy when possible, otherwise CBlt::BltCopy (decode a row to
//    float, encode with the TRIANGLE|DITHER filter's dither: Bayer thresholds for 8-bit formats, the DXT encoder's
//    error diffusion for DXT);
//  * missing levels: D3DXFilterTexture with BOX, each level from the previous one (integer box filter for
//    uncompressed formats when the source is at least 2x2, otherwise the float path).
#include <cstdint>
#include <string>
#include <vector>

#include "dxt.h"

namespace fnvvk::d3dx {

// D3DFORMAT values handled.
enum : uint32_t {
  kFmtR8G8B8 = 20, kFmtA8R8G8B8 = 21, kFmtX8R8G8B8 = 22, kFmtA4R4G4B4 = 26,
  kFmtDXT1 = 0x31545844, kFmtDXT2 = 0x32545844, kFmtDXT3 = 0x33545844, kFmtDXT4 = 0x34545844, kFmtDXT5 = 0x35545844,
};

struct Level {
  uint32_t width = 0, height = 0;
  uint32_t rowPitch = 0;  // bytes per row of texels, or per row of 4x4 blocks for DXT
  uint32_t rows = 0;      // texel rows, or block rows for DXT
  std::vector<uint8_t> data;
};

struct Texture {
  uint32_t format = 0;  // D3DFORMAT of the created texture
  uint32_t width = 0, height = 0, levels = 0;
  bool cube = false;
  std::vector<Level> surfaces;  // face-major: surfaces[face * levels + level]
};

enum class Status { Ok, NotDds, Truncated, UnsupportedFormat, UnsupportedLayout };

// `precision` is the x87 precision control the original D3DX code would run under: that of the loading thread
// (the game: 24-bit on its main thread, 53-bit on other threads).
Status createTextureFromDds(const uint8_t* file, size_t size, Precision precision, Texture& out, std::string* why = nullptr);

}  // namespace fnvvk::d3dx
