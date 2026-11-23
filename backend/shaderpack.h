#pragma once
// Runtime shader pack (tools/packshaders.py): SPIR-V translations of game shaders, keyed by the MD5 of
// the D3D9 bytecode, with the reflection the runtime needs (docs/SHADER_ABI.md).
#include "construns.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

namespace fnvvk {

struct Md5;
struct PackedShader {
  uint8_t md5[16] = {};
  struct Def { uint32_t reg; float v[4]; };
  struct Tex { uint8_t reg, kind; };                       // kind 5 = 2D, 9 = cube
  struct Input { uint8_t usage, usageIndex, location, components; };  // usage = D3DDECLUSAGE
  bool pixel = false;
  uint32_t cbvMask = 0;       // uniform bindings 0..8 declared by the module (9: sun shadows, 10: environment)
  uint32_t textureMask = 0;   // sampler registers with image binding 16+s / sampler binding 32+s
  bool skinDetail = false;    // the skin's detail normal map (59) and its sampler (60)
  bool skinStore = false;     // the skin buffer (block 13, storage images 61 / 62)
  // Pixel shaders (spvPixelTraits): kills pixels other than by the alpha test (D3D9 texkill, a replacement's own
  // discard), or writes depth. Either keeps its draws out of the depth pre-pass (backend_draw.inl).
  bool discards = false, writesDepth = false;
  // Vertex shaders: the position is declared Invariant (every translation's is: the same position in the pre-pass and
  // the colour pass, whose depth test is EQUAL). A shader without it keeps its draws out of the pre-pass.
  bool invariantPosition = false;
  std::vector<Def> defs;
  std::vector<Tex> textures;
  std::vector<Input> inputs;
  // The float-constant registers the module reads (construns.h): only these runs are copied into its constant block;
  // constAll: every register (a dynamic index or another use of the block).
  std::vector<ConstRun> constRuns;
  bool constAll = true;
  const uint32_t* spv = nullptr;  // points into ShaderPack::blob
  uint32_t spvBytes = 0;
};

struct Md5 {
  std::array<uint8_t, 16> b{};
  bool operator==(const Md5&) const = default;
};
struct Md5Hash {
  size_t operator()(const Md5& m) const {
    size_t h;
    static_assert(sizeof(h) <= 16);
    memcpy(&h, m.b.data(), sizeof(h));
    return h;
  }
};

class ShaderPack {
public:
  bool load(const std::string& path, std::string* error);
  const PackedShader* find(const Md5& md5) const;
  PackedShader* findMutable(const Md5& md5);
  size_t size() const { return shaders_.size(); }
  // Shader replacement: `md5`'s module becomes `spv` (same interface: the bindings and texture registers of the
  // translation, a subset of them used), and `extraDefs` are overlaid on its float constants like its own `def`s.
  bool replace(const Md5& md5, std::vector<uint32_t> spv, const std::vector<PackedShader::Def>& extraDefs);

private:
  std::vector<uint8_t> blob_;
  std::deque<std::vector<uint32_t>> replacements_;  // owned SPIR-V of replaced shaders (stable addresses)
  std::unordered_map<Md5, PackedShader, Md5Hash> shaders_;
};

// Whether a SPIR-V module declares an Input variable at `location` (e.g. a pixel shader reading the lighting vertex
// shaders' view vector, location 12).
bool spvDeclaresInputLocation(const std::vector<uint32_t>& spv, uint32_t location);

// The depth pre-pass's module of a pixel shader (backend_draw.inl "depth pre-pass"): the same code with its colour
// outputs made Private variables, so it keeps or discards each pixel exactly as the shader does and the driver drops
// the work that only fed the colours. False (out empty) for a module with built-in outputs (FragDepth, SampleMask) or
// side effects (storage image writes, atomics), or that is not a fragment shader.
bool spvDepthVariant(const uint32_t* spv, size_t words, std::vector<uint32_t>& out);
// Its colour module: the same code without its discards (every one is OpDemoteToHelperInvocation, which the module
// simply drops), so it shades every pixel it is given: the pixels its depth module kept, by an EQUAL depth test. False
// for a module with a discard that ends the invocation (OpKill, OpTerminateInvocation) or that tests for one
// (OpIsHelperInvocationEXT), or that is not a fragment shader.
bool spvColourVariant(const uint32_t* spv, size_t words, std::vector<uint32_t>& out);

// MD5 of a D3D9 shader token stream (the pack key). Uses the Windows CNG provider.
bool md5Of(const void* data, size_t size, Md5& out);

// The game's shader packages (Data\Shaders\shaderpackage*.sdp: u32 100, u32 count, u32 size, then per shader a
// 256-byte name, u32 size and the bytecode): the distinct MD5s of every shader called `name` (packages hold
// different builds of a shader). Empty if none.
// `packages`: only these shaderpackageNNN.sdp (empty: all).
std::vector<Md5> sdpShaderMd5s(const std::string& shaderDir, const std::string& name,
                               const std::vector<int>& packages = {});

}  // namespace fnvvk
