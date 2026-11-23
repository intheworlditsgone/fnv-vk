// Parity test: native DXT codec (backend/d3dxtex) vs the game's own d3dx9_38.dll (x86), function by function.
// D3DX's internal D3DXEncodeDXTn / D3DXDecodeDXTn are called by RVA (from Microsoft's public PDB for this build)
// with the x87 precision control forced to 24-bit or 53-bit, the two modes measured in game.
//
// usage: dxtparity [blocks-per-case] [dds files...]   exit 0 only if every output is bit-identical
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <float.h>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "d3dxtex/dxt.h"

using fnvvk::d3dx::Color;
using fnvvk::d3dx::Precision;

using EncodeFn = long(__stdcall*)(uint8_t*, const Color*, int);
using DecodeFn = long(__stdcall*)(Color*, const uint8_t*);

static EncodeFn dxEnc[3];
static DecodeFn dxDec[3];

static uint16_t x87cw() {
  uint16_t cw;
  __asm fnstcw cw
  return cw;
}
static void setPrecision(Precision p) {
  unsigned cur;
  _controlfp_s(&cur, p == Precision::Single ? _PC_24 : _PC_53, _MCW_PC);
}

static bool loadD3dx() {
  HMODULE dx = LoadLibraryA("d3dx9_38.dll");
  if (!dx) return false;
  auto base = reinterpret_cast<const uint8_t*>(dx);
  auto nt = base + *reinterpret_cast<const uint32_t*>(base + 0x3C);
  if (*reinterpret_cast<const uint32_t*>(nt + 8) != 0x4833557D || *reinterpret_cast<const uint32_t*>(nt + 0x50) != 0x3C8000) {
    printf("d3dx9_38.dll is not the game's build\n");
    return false;
  }
  uintptr_t b = reinterpret_cast<uintptr_t>(dx);
  dxEnc[0] = reinterpret_cast<EncodeFn>(b + 0x229C16);
  dxEnc[1] = reinterpret_cast<EncodeFn>(b + 0x229D52);
  dxEnc[2] = reinterpret_cast<EncodeFn>(b + 0x229E94);
  dxDec[0] = reinterpret_cast<DecodeFn>(b + 0x22990D);
  dxDec[1] = reinterpret_cast<DecodeFn>(b + 0x229A3D);
  dxDec[2] = reinterpret_cast<DecodeFn>(b + 0x229AC7);
  return true;
}

struct Stats { uint64_t cases = 0, mismatches = 0; };
static const char* kFmt[3] = { "DXT1", "DXT3", "DXT5" };
static const uint32_t kBlockBytes[3] = { 8, 16, 16 };

static void nativeEncode(int f, uint8_t* out, const Color* in, bool dither, Precision p) {
  if (f == 0) fnvvk::d3dx::encodeDXT1(out, in, dither, p);
  else if (f == 1) fnvvk::d3dx::encodeDXT3(out, in, dither, p);
  else fnvvk::d3dx::encodeDXT5(out, in, dither, p);
}
static void nativeDecode(int f, Color* out, const uint8_t* in, Precision p) {
  if (f == 0) fnvvk::d3dx::decodeDXT1(out, in, p);
  else if (f == 1) fnvvk::d3dx::decodeDXT3(out, in, p);
  else fnvvk::d3dx::decodeDXT5(out, in, p);
}

static Stats g_enc[3][2][2], g_dec[3][2];
static int g_printed = 0;

static void checkEncode(int f, const Color* in, bool dither, Precision p, const char* origin) {
  uint8_t a[16] = {}, b[16] = {};
  setPrecision(p);
  uint16_t cw = x87cw();
  dxEnc[f](a, in, dither ? 1 : 0);
  setPrecision(Precision::Double);
  nativeEncode(f, b, in, dither, p);
  Stats& s = g_enc[f][int(p)][dither];
  s.cases++;
  if (memcmp(a, b, kBlockBytes[f])) {
    s.mismatches++;
    if (g_printed++ < 12) {
      printf("ENCODE MISMATCH %s %s dither %d (cw %04X, %s)\n  d3dx  ", kFmt[f], p == Precision::Single ? "PC24" : "PC53",
             dither, cw, origin);
      for (uint32_t i = 0; i < kBlockBytes[f]; i++) printf("%02X", a[i]);
      printf("\n  native ");
      for (uint32_t i = 0; i < kBlockBytes[f]; i++) printf("%02X", b[i]);
      printf("\n  input:");
      for (int i = 0; i < 16; i++) printf(" (%a %a %a %a)", in[i].r, in[i].g, in[i].b, in[i].a);
      printf("\n");
    }
  }
}

static void checkDecode(int f, const uint8_t* block, Precision p) {
  Color a[16], b[16];
  setPrecision(p);
  dxDec[f](a, block);
  setPrecision(Precision::Double);
  nativeDecode(f, b, block, p);
  Stats& s = g_dec[f][int(p)];
  s.cases++;
  if (memcmp(a, b, sizeof(a))) {
    s.mismatches++;
    if (g_printed++ < 12) {
      printf("DECODE MISMATCH %s %s block ", kFmt[f], p == Precision::Single ? "PC24" : "PC53");
      for (uint32_t i = 0; i < kBlockBytes[f]; i++) printf("%02X", block[i]);
      printf("\n");
    }
  }
}

static void checkAll(const Color* in, const char* origin) {
  for (int f = 0; f < 3; f++)
    for (Precision p : { Precision::Single, Precision::Double })
      for (bool d : { false, true }) checkEncode(f, in, d, p, origin);
}

static std::vector<uint8_t> readFile(const char* path) {
  std::vector<uint8_t> v;
  FILE* fp = nullptr;
  if (fopen_s(&fp, path, "rb") || !fp) return v;
  fseek(fp, 0, SEEK_END);
  v.resize(size_t(ftell(fp)));
  fseek(fp, 0, SEEK_SET);
  fread(v.data(), 1, v.size(), fp);
  fclose(fp);
  return v;
}

int main(int argc, char** argv) {
  if (!loadD3dx()) { printf("FAIL: cannot load the game's d3dx9_38.dll\n"); return 2; }
  int perCase = argc > 1 ? atoi(argv[1]) : 20000;
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> u01(0.0f, 1.0f);
  Color in[16];

  // 1. Random inputs: uniform, few-colour, gradients, alpha extremes, 8-bit quantised values (as decoded data is).
  for (int n = 0; n < perCase; n++) {
    int kind = n % 6;
    Color base[4];
    for (auto& c : base) c = { u01(rng), u01(rng), u01(rng), u01(rng) };
    for (int i = 0; i < 16; i++) {
      Color c;
      switch (kind) {
        case 0: c = { u01(rng), u01(rng), u01(rng), u01(rng) }; break;
        case 1: c = base[rng() % 2]; break;
        case 2: { float t = (i % 4) / 3.0f; c = { base[0].r * (1 - t) + base[1].r * t, base[0].g * (1 - t) + base[1].g * t,
                                                 base[0].b * (1 - t) + base[1].b * t, base[0].a * (1 - t) + base[1].a * t }; } break;
        case 3: c = { u01(rng), u01(rng), u01(rng), (rng() & 1) ? 1.0f : 0.0f }; break;
        case 4: c = { float(rng() % 256) / 255.0f, float(rng() % 256) / 255.0f, float(rng() % 256) / 255.0f, float(rng() % 256) / 255.0f }; break;
        default: c = base[rng() % 4]; c.a = (rng() % 3) ? 1.0f : float(rng() % 256) / 255.0f; break;
      }
      in[i] = c;
    }
    checkAll(in, "random");
  }

  // 2. Random blocks through the decoders.
  for (int n = 0; n < perCase; n++) {
    uint8_t blk[16];
    for (auto& b : blk) b = uint8_t(rng());
    if (n % 4 == 0) { blk[0] = blk[1]; }  // DXT5 alpha a0 == a1 and edge cases
    for (int f = 0; f < 3; f++)
      for (Precision p : { Precision::Single, Precision::Double }) checkDecode(f, f == 0 ? blk + 8 : blk, p);
  }

  // 3. Real game blocks: decode with D3DX (as its own filter/re-encode paths do), re-encode, and 2x2 box averages
  //    of neighbouring decoded blocks (the shape of generated mip data).
  int files = 0;
  uint64_t realBlocks = 0;
  for (int a = 2; a < argc; a++) {
    std::vector<uint8_t> d = readFile(argv[a]);
    if (d.size() < 128 || memcmp(d.data(), "DDS ", 4)) continue;
    uint32_t h, w, fourcc;
    memcpy(&h, &d[12], 4);
    memcpy(&w, &d[16], 4);
    memcpy(&fourcc, &d[84], 4);
    int f = fourcc == 0x31545844 ? 0 : fourcc == 0x33545844 ? 1 : fourcc == 0x35545844 ? 2 : -1;
    if (f < 0) continue;
    files++;
    uint32_t bs = kBlockBytes[f], bw = std::max(1u, (w + 3) / 4), bh = std::max(1u, (h + 3) / 4);
    size_t nBlocks = std::min<size_t>(size_t(bw) * bh, (d.size() - 128) / bs);
    size_t stride = std::max<size_t>(1, nBlocks / 400);
    for (size_t bi = 0; bi + 1 < nBlocks; bi += stride) {
      const uint8_t* blk = d.data() + 128 + bi * bs;
      for (Precision p : { Precision::Single, Precision::Double }) checkDecode(f, blk, p);
      Color dec[16], dec2[16], avg[16];
      setPrecision(Precision::Single);
      dxDec[f](dec, blk);
      dxDec[f](dec2, blk + bs);
      setPrecision(Precision::Double);
      checkAll(dec, "game block");
      for (int i = 0; i < 16; i++) {
        const Color& p0 = dec[i];
        const Color& p1 = dec2[(i + 5) % 16];
        avg[i] = { (p0.r + p1.r) * 0.5f, (p0.g + p1.g) * 0.5f, (p0.b + p1.b) * 0.5f, (p0.a + p1.a) * 0.5f };
      }
      checkAll(avg, "game block average");
      realBlocks++;
    }
  }

  uint64_t total = 0, bad = 0;
  printf("%-5s %-5s %-7s %12s %10s\n", "fmt", "x87", "dither", "cases", "mismatch");
  for (int f = 0; f < 3; f++)
    for (int p = 0; p < 2; p++)
      for (int d = 0; d < 2; d++) {
        Stats& s = g_enc[f][p][d];
        printf("%-5s %-5s %-7d %12llu %10llu  (encode)\n", kFmt[f], p ? "PC53" : "PC24", d, s.cases, s.mismatches);
        total += s.cases;
        bad += s.mismatches;
      }
  for (int f = 0; f < 3; f++)
    for (int p = 0; p < 2; p++) {
      Stats& s = g_dec[f][p];
      printf("%-5s %-5s %-7s %12llu %10llu  (decode)\n", kFmt[f], p ? "PC53" : "PC24", "-", s.cases, s.mismatches);
      total += s.cases;
      bad += s.mismatches;
    }
  printf("game files %d, game blocks %llu; out-of-range step indices (native) %llu\n", files, realBlocks,
         fnvvk::d3dx::outOfRangeStepCount());
  printf(bad ? "RESULT: %llu of %llu differ\n" : "RESULT: PASS (%llu of %llu identical)\n", bad ? bad : total, total);
  return bad ? 1 : 0;
}
