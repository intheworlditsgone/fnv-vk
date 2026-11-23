// Native equivalent of d3dx9_38's DDS texture creation path. See texture.h and docs/M2_RESULTS.md.
#include "texture.h"

#include <algorithm>
#include <cstring>
#include <xmmintrin.h>

#pragma fenv_access(on)

namespace fnvvk::d3dx {

namespace {

constexpr uint32_t kDdsMagic = 0x20534444;  // "DDS "

// CImage::LoadDDS pixel-format table (d3dx9_38 0x00787758): only the formats this loader has codecs for. Matching
// follows the binary: pixel-format size 32, flags equal (FOURCC alone when set), and only the fields the flags select.
struct PfEntry { uint32_t format, flags, fourcc, bits, r, g, b, a; };
constexpr PfEntry kPixelFormats[] = {
  { kFmtR8G8B8, 0x40, 0, 24, 0x00FF0000, 0x0000FF00, 0x000000FF, 0 },
  { kFmtA8R8G8B8, 0x41, 0, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000 },
  { kFmtX8R8G8B8, 0x40, 0, 32, 0x00FF0000, 0x0000FF00, 0x000000FF, 0 },
  { kFmtA4R4G4B4, 0x41, 0, 16, 0x0F00, 0x00F0, 0x000F, 0xF000 },
  { kFmtDXT1, 0x04, kFmtDXT1, 0, 0, 0, 0, 0 },
  { kFmtDXT3, 0x04, kFmtDXT3, 0, 0, 0, 0, 0 },
  { kFmtDXT5, 0x04, kFmtDXT5, 0, 0, 0, 0, 0 },
};
// Formats the binary's table knows but this loader has no codec for (reported as unsupported, never approximated).
constexpr uint32_t kKnownButUnsupported[] = { 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 40, 41, 50, 51, 52, 81, 60,
                                              61, 62, 63, 64, 67, kFmtDXT2, kFmtDXT4, 70, 82, 36, 110, 111, 112, 113, 114,
                                              115, 116 };

bool isDxt(uint32_t f) { return f == kFmtDXT1 || f == kFmtDXT2 || f == kFmtDXT3 || f == kFmtDXT4 || f == kFmtDXT5; }
uint32_t bytesPerPixel(uint32_t f) {
  switch (f) {
    case kFmtR8G8B8: return 3;
    case kFmtA8R8G8B8: case kFmtX8R8G8B8: return 4;
    case kFmtA4R4G4B4: return 2;
    default: return 0;
  }
}

Level makeLevel(uint32_t fmt, uint32_t w, uint32_t h) {
  Level l;
  l.width = w;
  l.height = h;
  if (isDxt(fmt)) {
    l.rowPitch = (fmt == kFmtDXT1 ? 8 : 16) * ((w + 3) / 4);
    l.rows = (h + 3) / 4;
  } else {
    l.rowPitch = bytesPerPixel(fmt) * w;
    l.rows = h;
  }
  l.data.assign(size_t(l.rowPitch) * l.rows, 0);
  return l;
}

struct RoundToZero {
  unsigned saved;
  RoundToZero() : saved(_mm_getcsr()) { _mm_setcsr((saved & ~_MM_ROUND_MASK) | _MM_ROUND_TOWARD_ZERO); }
  ~RoundToZero() { _mm_setcsr(saved); }
};

constexpr float kInv255 = 0.0039215688593685627f;  // 0x3B808081, __real@3f70101020000000
constexpr float kInv15 = 0.066666670143604279f;    // 0x3D888889
// Encoder thresholds (d3dx9_38 0x00787510 no dither: 0.5; 0x00787590 with D3DX_FILTER_DITHER: 4x4 Bayer).
constexpr float kBayer[4][4] = { { 0.96875f, 0.46875f, 0.84375f, 0.34375f },
                                 { 0.21875f, 0.71875f, 0.09375f, 0.59375f },
                                 { 0.78125f, 0.28125f, 0.90625f, 0.40625f },
                                 { 0.03125f, 0.53125f, 0.15625f, 0.65625f } };

// --- uncompressed codecs (CCodec_*::Decode / Encode) -------------------------------------------------------------
void decodeRow(uint32_t fmt, const Level& l, uint32_t y, Color* out) {
  const uint8_t* p = l.data.data() + size_t(y) * l.rowPitch;
  for (uint32_t x = 0; x < l.width; x++) {
    Color& c = out[x];
    switch (fmt) {
      case kFmtA8R8G8B8:
        c = { float(p[4 * x + 2]) * kInv255, float(p[4 * x + 1]) * kInv255, float(p[4 * x + 0]) * kInv255,
              float(p[4 * x + 3]) * kInv255 };
        break;
      case kFmtX8R8G8B8:
        c = { float(p[4 * x + 2]) * kInv255, float(p[4 * x + 1]) * kInv255, float(p[4 * x + 0]) * kInv255, 1.0f };
        break;
      case kFmtR8G8B8:
        c = { float(p[3 * x + 2]) * kInv255, float(p[3 * x + 1]) * kInv255, float(p[3 * x + 0]) * kInv255, 1.0f };
        break;
      case kFmtA4R4G4B4: {
        uint32_t v = uint32_t(p[2 * x]) | (uint32_t(p[2 * x + 1]) << 8);
        c = { float((v >> 8) & 0xF) * kInv15, float((v >> 4) & 0xF) * kInv15, float(v & 0xF) * kInv15,
              float(v >> 12) * kInv15 };
        break;
      }
    }
  }
}

// int((c + 0) * scale + threshold), clamped; the whole pixel loop runs with round-toward-zero (e.g. 0x0061493F).
int quantize(float c, float scale, float threshold, int maxValue) {
  float s = (c + 0.0f) * scale;
  float t = s + threshold;
  int q = int(t);
  int clampHigh = q >= maxValue ? maxValue : q;
  return clampHigh <= 0 ? 0 : (q >= maxValue ? maxValue : q);
}

void encodeRow(uint32_t fmt, Level& l, uint32_t y, const Color* in, bool dither) {
  RoundToZero rz;
  uint8_t* p = l.data.data() + size_t(y) * l.rowPitch;
  for (uint32_t x = 0; x < l.width; x++) {
    const float thr = dither ? kBayer[y & 3][x & 3] : 0.5f;
    const Color& c = in[x];
    if (fmt == kFmtA4R4G4B4) {
      int r = quantize(c.r, 15.0f, thr, 15), g = quantize(c.g, 15.0f, thr, 15), b = quantize(c.b, 15.0f, thr, 15),
          a = quantize(c.a, 15.0f, thr, 15);
      uint16_t v = uint16_t(b | (16 * (g | (16 * (r | (16 * a))))));
      memcpy(p + 2 * x, &v, 2);
    } else {
      int r = quantize(c.r, 255.0f, thr, 255), g = quantize(c.g, 255.0f, thr, 255), b = quantize(c.b, 255.0f, thr, 255);
      int a = fmt == kFmtA8R8G8B8 ? quantize(c.a, 255.0f, thr, 255) : 0;  // X8R8G8B8 writes X = 0
      p[4 * x + 0] = uint8_t(b);
      p[4 * x + 1] = uint8_t(g);
      p[4 * x + 2] = uint8_t(r);
      p[4 * x + 3] = uint8_t(a);
    }
  }
}

// --- DXT surfaces (CCodecDXT::Decode / Encode) -------------------------------------------------------------------
void decodeBlock(uint32_t fmt, const uint8_t* b, Color out[16], Precision p) {
  if (fmt == kFmtDXT1) decodeDXT1(out, b, p);
  else if (fmt == kFmtDXT3) decodeDXT3(out, b, p);
  else decodeDXT5(out, b, p);
}
void encodeBlock(uint32_t fmt, uint8_t* b, const Color in[16], bool dither, Precision p) {
  if (fmt == kFmtDXT1) encodeDXT1(b, in, dither, p);
  else if (fmt == kFmtDXT3) encodeDXT3(b, in, dither, p);
  else encodeDXT5(b, in, dither, p);
}

// One DXT block row -> texel rows 4*by .. 4*by+3 (those inside the level), written to `out` (stride width).
// Levels are processed a block row at a time: the game is a 32-bit process and cannot always get one contiguous
// block for a whole large level of float texels (a 4096x4096 level would be 256 MiB). Blocks are independent, so
// banding does not change any result.
void decodeDxtBlockRow(uint32_t fmt, const Level& l, uint32_t by, Color* out, Precision p) {
  uint32_t bs = fmt == kFmtDXT1 ? 8 : 16;
  for (uint32_t bx = 0; bx < l.rowPitch / bs; bx++) {
    Color blk[16];
    decodeBlock(fmt, l.data.data() + size_t(by) * l.rowPitch + size_t(bx) * bs, blk, p);
    for (uint32_t y = 0; y < 4; y++)
      for (uint32_t x = 0; x < 4; x++) {
        uint32_t ix = bx * 4 + x, iy = by * 4 + y;
        if (ix < l.width && iy < l.height) out[size_t(y) * l.width + ix] = blk[y * 4 + x];
      }
  }
}

// CCodecDXT::Encode (0x0061D2BC) for a whole-level rect: blocks are filled from the texel rows; the parts of edge
// blocks outside the level are padded column-wise then row-wise from index pattern {0,0,0,1} (applied in order,
// so a 1-wide block replicates column 0), then each block goes to D3DXEncodeDXTn with the codec's dither flag.
// Encodes block row `by` from its texel rows (`rows`: rows 4*by.. of the level, stride width).
void encodeDxtBlockRow(uint32_t fmt, Level& l, uint32_t by, const Color* rows, bool dither, Precision p) {
  static constexpr uint32_t kPad[4] = { 0, 0, 0, 1 };
  uint32_t bs = fmt == kFmtDXT1 ? 8 : 16;
  {
    for (uint32_t bx = 0; bx < l.rowPitch / bs; bx++) {
      Color blk[16];
      uint32_t w = std::min(4u, l.width - bx * 4), h = std::min(4u, l.height - by * 4);
      for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) blk[y * 4 + x] = rows[size_t(y) * l.width + bx * 4 + x];
      if (w < 4)
        for (uint32_t y = 0; y < h && y < 4; y++)
          for (uint32_t x = w; x < 4; x++) blk[y * 4 + x] = blk[y * 4 + kPad[x]];
      if (h < 4)
        for (uint32_t y = h; y < 4; y++)
          for (uint32_t x = 0; x < 4; x++) blk[y * 4 + x] = blk[kPad[y] * 4 + x];
      encodeBlock(fmt, l.data.data() + size_t(by) * l.rowPitch + size_t(bx) * bs, blk, dither, p);
    }
  }
}

// --- blits -------------------------------------------------------------------------------------------------------
// D3DXLoadSurfaceFromMemory of a file level into the same-size destination level with TRIANGLE|DITHER.
void loadFileLevel(uint32_t srcFmt, const Level& src, uint32_t dstFmt, Level& dst, Precision p) {
  if (srcFmt == dstFmt && (!isDxt(srcFmt) || (src.width % 4 == 0 && src.height % 4 == 0))) {
    dst.data = src.data;  // CBlt::BltSame / BltSame_DXTn
    return;
  }
  // CBlt::BltCopy: same dimensions, any formats: decode each row to float and encode it (dither on).
  if (isDxt(srcFmt)) {
    std::vector<Color> band(size_t(src.width) * 4);
    for (uint32_t by = 0; by < src.rows; by++) {
      decodeDxtBlockRow(srcFmt, src, by, band.data(), p);
      if (isDxt(dstFmt)) {
        encodeDxtBlockRow(dstFmt, dst, by, band.data(), true, p);
      } else {
        for (uint32_t y = by * 4; y < std::min(dst.height, by * 4 + 4); y++)
          encodeRow(dstFmt, dst, y, band.data() + size_t(y - by * 4) * dst.width, true);
      }
    }
    return;
  }
  std::vector<Color> row(src.width);
  for (uint32_t y = 0; y < src.height; y++) {
    decodeRow(srcFmt, src, y, row.data());
    encodeRow(dstFmt, dst, y, row.data(), true);
  }
}

// D3DXFilterTexture BOX step: next level from the previous one (CBlt::BltBox2D, no dither).
void boxLevel(uint32_t fmt, const Level& src, Level& dst, Precision p) {
  bool integer = !isDxt(fmt) && src.width >= 2 && src.height >= 2;
  if (integer) {
    // BltBox2D_A8R8G8B8 / _X8R8G8B8 (MMX, every byte including X) and _A4R4G4B4: per channel (a+b+c+d+2)>>2.
    // Equivalence of the binary's bit-trick C fallbacks with this formula was checked exhaustively on random input.
    uint32_t bpp = bytesPerPixel(fmt);
    for (uint32_t y = 0; y < dst.height; y++) {
      const uint8_t* r0 = src.data.data() + size_t(2 * y) * src.rowPitch;
      const uint8_t* r1 = r0 + src.rowPitch;
      uint8_t* o = dst.data.data() + size_t(y) * dst.rowPitch;
      for (uint32_t x = 0; x < dst.width; x++) {
        if (bpp == 4) {
          for (int k = 0; k < 4; k++)
            o[4 * x + k] = uint8_t((r0[8 * x + k] + r0[8 * x + 4 + k] + r1[8 * x + k] + r1[8 * x + 4 + k] + 2) >> 2);
        } else {
          auto px = [](const uint8_t* q) { return uint32_t(q[0]) | (uint32_t(q[1]) << 8); };
          uint32_t a = px(r0 + 4 * x), b = px(r0 + 4 * x + 2), c = px(r1 + 4 * x), d = px(r1 + 4 * x + 2), v = 0;
          for (int s = 0; s < 16; s += 4)
            v |= ((((a >> s) & 15) + ((b >> s) & 15) + ((c >> s) & 15) + ((d >> s) & 15) + 2) >> 2) << s;
          o[2 * x] = uint8_t(v);
          o[2 * x + 1] = uint8_t(v >> 8);
        }
      }
    }
    return;
  }
  // Generic path: decode source rows 2y and 2y+1 (a 1-high source uses its row twice; a 1-wide source its texel
  // twice), sum ((a+b)+c)+d with each add stored to float, times 0.25, encode the row (no dither).
  // Processed one destination band at a time: 4 rows for DXT (one block row), 1 row otherwise; the band reads
  // source rows 2*y and 2*y+1 of each destination row.
  const uint32_t bandRows = isDxt(fmt) ? 4 : 1;
  std::vector<Color> srcRows(size_t(src.width) * 2 * bandRows);  // source rows 2*y0 .. 2*y0 + 2*bandRows - 1
  std::vector<Color> out(size_t(dst.width) * bandRows);
  for (uint32_t y0 = 0; y0 < dst.height; y0 += bandRows) {
    const uint32_t s0row = 2 * y0, sRows = std::min(src.height - s0row, 2 * bandRows);
    if (isDxt(fmt)) {
      // Source block rows covering texel rows s0row .. s0row + sRows - 1 (s0row is a multiple of 8 here).
      for (uint32_t r = 0; r < sRows; r += 4) decodeDxtBlockRow(fmt, src, (s0row + r) / 4, srcRows.data() + size_t(r) * src.width, p);
    } else {
      for (uint32_t r = 0; r < sRows; r++) decodeRow(fmt, src, s0row + r, srcRows.data() + size_t(r) * src.width);
    }
    for (uint32_t y = y0; y < std::min(dst.height, y0 + bandRows); y++) {
      const Color* r0 = srcRows.data() + size_t(2 * y - s0row) * src.width;
      const Color* r1 = src.height == 1 ? r0 : r0 + src.width;
      for (uint32_t x = 0; x < dst.width; x++) {
        const Color& a = r0[2 * x];
        const Color& b = src.width == 1 ? r0[2 * x] : r0[2 * x + 1];
        const Color& c = r1[2 * x];
        const Color& d = src.width == 1 ? r1[2 * x] : r1[2 * x + 1];
        // Each addition is a single operation on two floats stored to float: identical at 24- and 53-bit precision.
        float s0 = b.r + a.r, s1 = b.g + a.g, s2 = b.b + a.b, s3 = b.a + a.a;
        s0 = c.r + s0; s1 = c.g + s1; s2 = c.b + s2; s3 = c.a + s3;
        s0 = d.r + s0; s1 = d.g + s1; s2 = d.b + s2; s3 = d.a + s3;
        out[size_t(y - y0) * dst.width + x] = { s0 * 0.25f, s1 * 0.25f, s2 * 0.25f, s3 * 0.25f };
      }
    }
    if (isDxt(fmt)) {
      encodeDxtBlockRow(fmt, dst, y0 / 4, out.data(), false, p);
    } else {
      encodeRow(fmt, dst, y0, out.data(), false);
    }
  }
}

}  // namespace

Status createTextureFromDds(const uint8_t* file, size_t size, Precision precision, Texture& out, std::string* why) {
  auto fail = [&](Status s, const char* msg) {
    if (why) *why = msg;
    return s;
  };
  auto u32 = [&](size_t off) { uint32_t v; memcpy(&v, file + off, 4); return v; };
  if (size < 4 || u32(0) != kDdsMagic) return fail(Status::NotDds, "not a DDS file");
  if (size - 4 < 124) return fail(Status::Truncated, "DDS header truncated");
  const uint32_t flags = u32(8), height = u32(12), width = u32(16), depth = (flags & 0x800000) ? u32(24) : 1;
  uint32_t fileMips = u32(28) ? u32(28) : 1;  // dwMipMapCount is used whether or not DDSD_MIPMAPCOUNT is set
  const uint32_t caps2 = u32(112);
  uint32_t faces;
  if ((caps2 & 0xFC00) == 0xFC00) faces = 6;
  else if (caps2 & 0xFC00) return fail(Status::UnsupportedLayout, "cube map without all six faces");
  else faces = 1;
  if (depth > 1 || (caps2 & 0x200000)) return fail(Status::UnsupportedLayout, "volume texture");

  // Pixel format (CImage::LoadDDS matching rules).
  const uint32_t pfSize = u32(76), pfFlagsRaw = u32(80);
  const uint32_t pfFlags = (pfFlagsRaw & 4) ? 4 : pfFlagsRaw;
  uint32_t srcFmt = 0;
  for (const PfEntry& e : kPixelFormats) {
    if (pfSize != 32 || e.flags != pfFlags) continue;
    if ((pfFlags & 4) && u32(84) != e.fourcc) continue;
    if ((pfFlags & 0xC4462) && u32(88) != e.bits) continue;
    if ((pfFlags & 0xE4040) && u32(92) != e.r) continue;
    if ((pfFlags & 0xC4440) && u32(96) != e.g) continue;
    if ((pfFlags & 0xC4040) && u32(100) != e.b) continue;
    if ((pfFlags & 0x80003) && u32(104) != e.a) continue;
    srcFmt = e.format;
    break;
  }
  if (!srcFmt) {
    if (pfSize == 32 && (pfFlags & 4))
      for (uint32_t f : kKnownButUnsupported)
        if (f == u32(84)) return fail(Status::UnsupportedFormat, "known D3DX format without a native codec");
    return fail(Status::UnsupportedFormat, "pixel format not handled");
  }
  if ((width & (width - 1)) || (height & (height - 1)) || !width || !height)
    return fail(Status::UnsupportedLayout, "non-power-of-two size (D3DX would rescale with its triangle filter)");

  // Destination: the file's format, except R8G8B8 -> X8R8G8B8; full mip chain (D3DX_DEFAULT).
  const uint32_t dstFmt = srcFmt == kFmtR8G8B8 ? kFmtX8R8G8B8 : srcFmt;
  uint32_t levels = 1;
  for (uint32_t w = width, h = height; w > 1 || h > 1; w = std::max(w / 2, 1u), h = std::max(h / 2, 1u)) levels++;

  // File levels (all faces, each a full declared chain), with the binary's size checks.
  size_t off = 128;
  std::vector<Level> fileLevels;
  for (uint32_t f = 0; f < faces; f++)
    for (uint32_t m = 0, w = width, h = height; m < fileMips; m++, w = std::max(w / 2, 1u), h = std::max(h / 2, 1u)) {
      Level l = makeLevel(srcFmt, w, h);
      if (size - off < l.data.size()) return fail(Status::Truncated, "DDS data shorter than its declared levels");
      memcpy(l.data.data(), file + off, l.data.size());
      off += l.data.size();
      fileLevels.push_back(std::move(l));
    }

  out = Texture{};
  out.format = dstFmt;
  out.width = width;
  out.height = height;
  out.levels = levels;
  out.cube = faces == 6;
  out.surfaces.reserve(size_t(faces) * levels);
  const uint32_t loaded = std::min(fileMips, levels);
  for (uint32_t f = 0; f < faces; f++) {
    for (uint32_t m = 0, w = width, h = height; m < levels; m++, w = std::max(w / 2, 1u), h = std::max(h / 2, 1u)) {
      Level dst = makeLevel(dstFmt, w, h);
      if (m < loaded) loadFileLevel(srcFmt, fileLevels[size_t(f) * fileMips + m], dstFmt, dst, precision);
      else boxLevel(dstFmt, out.surfaces[size_t(f) * levels + m - 1], dst, precision);
      out.surfaces.push_back(std::move(dst));
    }
  }
  return Status::Ok;
}

}  // namespace fnvvk::d3dx
