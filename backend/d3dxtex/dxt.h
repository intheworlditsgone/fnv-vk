#pragma once
// Native reimplementation of d3dx9_38's DXT1/DXT3/DXT5 block codec (D3DXTex::D3DXEncodeDXTn / D3DXDecodeDXTn),
// behaviour-equivalent to the x86 build the game imports (PE timestamp 0x4833557D). Written from the observed
// behaviour of that binary (decompilation + disassembly with Microsoft's public symbols); no Microsoft code is used.
//
// The original is x87 code, so its results depend on the x87 precision control of the calling thread, which the
// game leaves at 24-bit on its main thread and 53-bit on other threads (docs/M2_RESULTS.md). Every function takes
// the precision to reproduce. The original also switches the x87 rounding mode to round-toward-zero around its
// quantisation loops; those spans are reproduced exactly (via MXCSR).
#include <cstdint>

namespace fnvvk::d3dx {

struct Color { float r, g, b, a; };  // D3DXCOLOR

enum class Precision : uint8_t {
  Single,  // x87 precision control 24-bit (control word 0x007F)
  Double,  // x87 precision control 53-bit (control word 0x027F, the Windows thread default)
};

void decodeDXT1(Color out[16], const uint8_t block[8], Precision p);
void decodeDXT3(Color out[16], const uint8_t block[16], Precision p);
void decodeDXT5(Color out[16], const uint8_t block[16], Precision p);

void encodeDXT1(uint8_t block[8], const Color in[16], bool dither, Precision p);
void encodeDXT3(uint8_t block[16], const Color in[16], bool dither, Precision p);
void encodeDXT5(uint8_t block[16], const Color in[16], bool dither, Precision p);

// Number of times the RGB optimiser computed a negative step index. The original indexes outside its step array
// (stack memory) in that case; the port uses a defined substitute and counts it so verification can find it.
uint64_t outOfRangeStepCount();

}  // namespace fnvvk::d3dx
