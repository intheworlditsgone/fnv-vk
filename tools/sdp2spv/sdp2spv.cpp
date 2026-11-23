// sdp2spv: translate FNV's D3D9 shader packages (Data/Shaders/shaderpackageNNN.sdp) to SPIR-V for the
// Vulkan backend, and emit reflection for the runtime.
//
// Package format (ShaderBuffer::Load, 0x00B7FE40 in FalloutNV.exe 1.4.0.525):
//   u32 tag (= 100), u32 count, u32 dataSize, then count x { char name[256]; u32 size; u8 code[size]; }
//
// Translation uses dxbc-spirv's SM3 frontend (the library DXVK 3.x uses for D3D9). D3D9 legacy state is lowered
// to constant buffers with the fixed binding ABI documented in docs/SHADER_ABI.md.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <functional>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "ir/ir.h"
#include "ir/ir_builder.h"
#include "ir/ir_disasm.h"
#include "ir/ir_legalize.h"
#include "ir/passes/ir_pass_lower_io.h"
#include "sm3/sm3_converter.h"
#include "sm3/sm3_parser.h"
#include "spirv/spirv_builder.h"
#include "spirv/spirv_mapping.h"
#include "util/util_byte_stream.h"
#include "util/util_log.h"
#include "util/util_md5.h"

namespace fs = std::filesystem;
using namespace dxbc_spv;

// ---------------------------------------------------------------------------------------------------------------
// Binding ABI (keep in sync with docs/SHADER_ABI.md and the runtime's pipeline layout).
namespace abi {
  constexpr uint32_t SetVertex = 0u;
  constexpr uint32_t SetPixel  = 1u;

  // Constant buffer register indices (register space 0) used for lowered legacy built-ins.
  constexpr uint32_t CbvConstFloat   = 0u;
  constexpr uint32_t CbvConstInt     = 1u;
  constexpr uint32_t CbvConstBool    = 2u;
  constexpr uint32_t CbvAlphaTest    = 3u;
  constexpr uint32_t CbvFog          = 4u;
  constexpr uint32_t CbvClipPlanes   = 5u;
  constexpr uint32_t CbvPointArgs    = 6u;
  constexpr uint32_t CbvSamplerState = 7u;
  constexpr uint32_t CbvTextureStage = 8u;

  constexpr uint32_t BindingCbvBase     = 0u;   // binding = regIndex
  constexpr uint32_t BindingTextureBase = 16u;  // binding = 16 + sampler register
  constexpr uint32_t BindingSamplerBase = 32u;  // binding = 32 + sampler register
  constexpr uint32_t MaxSamplerRegs     = 16u;

  // Sun shadows in the game's lighting (docs/SHADER_ABI.md, "Sun shadows"): pixel shaders that light with the sun get
  // a uniform block at binding 9 and the shadow cascades (a depth 2D array) and their compare sampler at 48 and 49.
  constexpr uint32_t CbvSunShadow       = 9u;
  constexpr uint32_t SunRegSpace        = 7u;   // register space of the cascade image and sampler declarations
  constexpr uint32_t BindingSunImage    = 48u;
  constexpr uint32_t BindingSunSampler  = 49u;
}

struct BindingRecord {
  std::string kind;  // "cbv" | "texture" | "sampler"
  uint32_t regSpace = 0u;
  uint32_t regIndex = 0u;
  uint32_t set = 0u;
  uint32_t binding = 0u;
};

class FnvResourceMapping : public spirv::ResourceMapping {
public:
  explicit FnvResourceMapping(uint32_t set) : m_set(set) { }

  spirv::DescriptorBinding mapDescriptor(ir::ScalarType type, uint32_t regSpace, uint32_t regIndex) override {
    BindingRecord rec;
    rec.regSpace = regSpace;
    rec.regIndex = regIndex;
    rec.set = m_set;

    switch (type) {
      case ir::ScalarType::eCbv:
        rec.kind = "cbv";
        rec.binding = abi::BindingCbvBase + regIndex;
        if (regSpace != 0u || regIndex >= abi::BindingTextureBase) fail(rec);
        break;
      case ir::ScalarType::eSrv:
        if (regSpace == abi::SunRegSpace) { rec.kind = "sun-image"; rec.binding = abi::BindingSunImage; break; }
        rec.kind = "texture";
        rec.binding = abi::BindingTextureBase + regIndex;
        if (regIndex >= abi::MaxSamplerRegs) fail(rec);
        break;
      case ir::ScalarType::eSampler:
        if (regSpace == abi::SunRegSpace) { rec.kind = "sun-sampler"; rec.binding = abi::BindingSunSampler; break; }
        rec.kind = "sampler";
        rec.binding = abi::BindingSamplerBase + regIndex;
        if (regIndex >= abi::MaxSamplerRegs) fail(rec);
        break;
      default:
        rec.kind = "unexpected";
        fail(rec);
    }

    records.push_back(rec);
    return spirv::DescriptorBinding { rec.set, rec.binding };
  }

  uint32_t mapPushData(ir::ShaderStageMask) override {
    // The SM3 frontend with the built-in CBV lowering does not emit push data. Record if it ever does.
    pushDataRequested = true;
    return 0u;
  }

  std::vector<BindingRecord> records;
  bool pushDataRequested = false;
  bool abiViolation = false;

private:
  void fail(const BindingRecord&) { abiViolation = true; }
  uint32_t m_set;
};

// ---------------------------------------------------------------------------------------------------------------
// Float-controls capabilities of the target device (VkPhysicalDeviceFloatControlsProperties). The SPIR-V builder
// only emits a RoundingMode/Denorm execution mode the device supports, so the output is device-dependent: the
// offline cache records the profile it was built for, and the runtime compiles for the live device. The default
// (empty) profile emits no float-control modes and is valid on every Vulkan 1.3 device.
struct FpCaps {
  ir::RoundModes round16 = 0u, round32 = 0u, round64 = 0u;
  ir::DenormModes denorm16 = 0u, denorm32 = 0u, denorm64 = 0u;
  std::string spec;
};

static bool parseFpCaps(const std::string& spec, FpCaps& caps) {
  caps.spec = spec;
  std::stringstream s(spec);
  std::string tok;
  while (std::getline(s, tok, ',')) {
    if (tok.empty()) continue;
    if (tok.size() < 4) return false;
    std::string kind = tok.substr(0, tok.size() - 2), bits = tok.substr(tok.size() - 2);
    ir::RoundModes* r = bits == "16" ? &caps.round16 : bits == "32" ? &caps.round32 : bits == "64" ? &caps.round64 : nullptr;
    ir::DenormModes* d = bits == "16" ? &caps.denorm16 : bits == "32" ? &caps.denorm32 : bits == "64" ? &caps.denorm64 : nullptr;
    if (!r) return false;
    if (kind == "rte") *r |= ir::RoundMode::eNearestEven;
    else if (kind == "rtz") *r |= ir::RoundMode::eZero;
    else if (kind == "ftz") *d |= ir::DenormMode::eFlush;
    else if (kind == "dp") *d |= ir::DenormMode::ePreserve;
    else return false;
  }
  return true;
}

static FpCaps g_fpCaps;
static fs::path g_dumpIr;  // --dump-ir: the IR of the translated shader(s) after lowering, appended here

// ---------------------------------------------------------------------------------------------------------------
class CaptureLogger : public util::Logger {
public:
  LogLevel getMinimumSeverity() override { return LogLevel::eWarn; }
  void message(LogLevel severity, const char* msg) override {
    std::stringstream s;
    s << severity << msg;
    lines.push_back(s.str());
  }
  std::vector<std::string> lines;
};

// ---------------------------------------------------------------------------------------------------------------
struct PackageEntry {
  std::string name;
  std::vector<uint8_t> code;
};

static bool readPackage(const fs::path& path, std::vector<PackageEntry>& out, std::string& err) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { err = "cannot open"; return false; }
  std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

  auto rd32 = [&](size_t off, uint32_t& v) {
    if (off + 4u > data.size()) return false;
    std::memcpy(&v, &data[off], 4u);
    return true;
  };

  uint32_t tag = 0u, count = 0u, size = 0u;
  if (!rd32(0, tag) || !rd32(4, count) || !rd32(8, size)) { err = "truncated header"; return false; }
  if (tag != 100u) { err = "bad tag " + std::to_string(tag); return false; }
  if (size + 12u != data.size()) { err = "dataSize mismatch"; return false; }

  size_t off = 12u;
  for (uint32_t i = 0; i < count; i++) {
    if (off + 260u > data.size()) { err = "truncated record header"; return false; }
    PackageEntry e;
    const char* nameBegin = reinterpret_cast<const char*>(&data[off]);
    e.name.assign(nameBegin, strnlen(nameBegin, 256u));
    uint32_t sz = 0u;
    rd32(off + 256u, sz);
    off += 260u;
    if (off + sz > data.size()) { err = "truncated record " + e.name; return false; }
    e.code.assign(data.begin() + off, data.begin() + off + sz);
    off += sz;
    out.push_back(std::move(e));
  }
  if (off != data.size()) { err = "trailing bytes"; return false; }
  return true;
}

static std::string hexDigest(const util::md5::Digest& d) {
  std::stringstream s;
  for (auto b : d.data) s << std::hex << std::setw(2) << std::setfill('0') << uint32_t(b);
  return s.str();
}

static std::string jsonEscape(const std::string& in) {
  std::string out;
  for (char c : in) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': break;
      case '\t': out += "\\t"; break;
      default:
        if (uint8_t(c) < 0x20) { char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", c); out += buf; }
        else out += c;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------
struct IoVar {
  bool output = false;
  bool builtin = false;
  uint32_t location = 0u;     // or BuiltIn enum value
  uint32_t components = 0u;
  std::string semantic;       // e.g. "texcoord"
  uint32_t semanticIndex = 0u;
};

struct CtabEntry {
  std::string name;
  std::string set;
  uint32_t index = 0u;
  uint32_t count = 0u;
};

struct TextureDecl {
  uint32_t reg = 0u;
  uint32_t kind = 0u;  // ir::ResourceKind
};

// Float constants the shader defines itself (`def cN, ...`). D3D9 hardware returns these values for relative
// reads (c[a0.x + n]) that land on a def'd register -- verified against hardware D3D9 -- while the translated
// shader reads relative addresses from the constant buffer. The runtime therefore overlays these values onto
// the uploaded float constants, as DXVK does (D3D9ConstantBufferLayout, isShaderDefined ranges).
struct FloatDef {
  uint32_t reg = 0u;
  float v[4] = { };
};

static void readFloatDefs(const std::vector<uint8_t>& code, std::vector<FloatDef>& out) {
  sm3::Parser parser(util::ByteReader(code.data(), code.size()));
  while (parser) {
    sm3::Instruction op = parser.parseInstruction();
    if (op.getOpCode() != sm3::OpCode::eDef || !op.hasDst() || !op.hasImm()) continue;
    if (op.getDst().getRegisterType() != sm3::RegisterType::eConst) continue;
    FloatDef d;
    d.reg = op.getDst().getIndex();
    for (uint32_t c = 0; c < 4; c++) d.v[c] = op.getImm().getImmediate<float>(c);
    out.push_back(d);
  }
}

struct ShaderResult {
  std::string hash;
  std::string profile;
  bool isVertex = false;
  bool ok = false;
  std::vector<std::string> log;
  std::vector<uint32_t> spirvWords;
  std::vector<IoVar> io;
  std::vector<CtabEntry> ctab;
  std::vector<TextureDecl> textures;
  std::vector<FloatDef> defs;
  std::vector<BindingRecord> bindings;
  bool pushData = false;
  bool abiViolation = false;
  bool nonFixedIo = false;
  int sunLight = -1;  // register of the sun light colour this shader shadows (sun shadow pass), -1: none
  bool viewVector = false;  // the view vector output was added (view vector pass)
  bool grassVector = false; // the grass lighting inputs were added (grass pass)
};

static const char* ctabSetName(sm3::ConstantType t) {
  switch (t) {
    case sm3::ConstantType::eBool: return "bool";
    case sm3::ConstantType::eInt4: return "int4";
    case sm3::ConstantType::eFloat4: return "float4";
    case sm3::ConstantType::eSampler: return "sampler";
  }
  return "?";
}

// Finds the CTAB comment block. D3D9 bytecode: version token, then instructions; comments are
// 0xFFFE | (dwordCount << 16) followed by the payload, CTAB payload starting with the FourCC "CTAB".
static void readCtab(const std::vector<uint8_t>& code, std::vector<CtabEntry>& out) {
  size_t n = code.size() / 4u;
  const uint32_t* tok = reinterpret_cast<const uint32_t*>(code.data());
  for (size_t i = 1; i < n; i++) {
    if ((tok[i] & 0xFFFFu) != 0xFFFEu) continue;
    uint32_t len = (tok[i] >> 16) & 0x7FFFu;
    if (i + 1 + len > n) return;
    if (len >= 1u && std::memcmp(&tok[i + 1], "CTAB", 4u) == 0) {
      sm3::ConstantTable table(util::ByteReader(&tok[i + 1], len * 4u));
      for (const auto& e : table.entries())
        out.push_back(CtabEntry { e.name, ctabSetName(e.registerSet), e.index, e.count });
      return;
    }
    i += len;
  }
}

static bool isFixedFunctionSemantic(const std::string& name, uint32_t index) {
  // dxbc-spirv's sm3 IoMap gives these semantics fixed VS-output/PS-input locations (s_ffLocations).
  // Anything else is numbered in declaration order and needs explicit VS/PS linkage checks.
  if (name == "texcoord") return index < 8u;
  if (name == "color") return index < 2u;
  if (name == "normal" || name == "fog") return index == 0u;
  return false;
}

static void collectIo(const ir::Builder& builder, std::vector<IoVar>& out) {
  auto [begin, end] = builder.getDeclarations();
  std::map<uint32_t, std::pair<std::string, uint32_t>> semantics;  // decl def -> (name, index)

  for (auto it = begin; it != end; ++it) {
    if (it->getOpCode() == ir::OpCode::eSemantic) {
      uint32_t target = uint32_t(ir::SsaDef(it->getOperand(0u)).getId());
      semantics[target] = { it->getLiteralString(2u), uint32_t(it->getOperand(1u)) };
    }
  }

  for (auto it = begin; it != end; ++it) {
    auto op = it->getOpCode();
    bool isIn = op == ir::OpCode::eDclInput || op == ir::OpCode::eDclInputBuiltIn;
    bool isOut = op == ir::OpCode::eDclOutput || op == ir::OpCode::eDclOutputBuiltIn;
    if (!isIn && !isOut) continue;

    IoVar v;
    v.output = isOut;
    v.builtin = op == ir::OpCode::eDclInputBuiltIn || op == ir::OpCode::eDclOutputBuiltIn;
    v.location = uint32_t(it->getOperand(it->getFirstLiteralOperandIndex()));
    v.components = it->getType().getBaseType(0u).getVectorSize();

    auto s = semantics.find(it->getDef().getId());
    if (s != semantics.end()) {
      v.semantic = s->second.first;
      v.semanticIndex = s->second.second;
    }
    out.push_back(v);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Sun shadows in the game's lighting.
//
// The game hands each lighting pass its lights in order: ShadowLightShader::InitPass (0x00B78A90) writes light i's
// colour to PSLightColor[i], and a directional light is always a pass's first light (BSRenderPass lights[0]). Outdoors
// the only directional light is the sun, so in a pixel shader that declares PSLightColor (CTAB), the colour at its first
// register is the sun's whenever the pass's first light is the sun. This pass multiplies the rgb of every load of that
// register by the sun's visibility at the fragment. The runtime says per draw whether the pass's first light is the sun
// (block 0.x); when it is 0 the visibility is exactly 1, so the shader computes what the game's does.
//
// Visibility (as fnvShadow in backend/effectlib.cpp): the fragment's view-space position from gl_FragCoord and the
// world camera (the game's projection: z = n*f / (f - d*(f-n))), pushed off the surface along the face normal (the
// position's screen derivatives) by 1 + 0.002 * z, then a 3 x 3 compare filter in the cascade whose far split contains
// it, blended with the next cascade over the last 20% of its depth range, the last cascade fading out from 85% of its
// distance. The block at binding 9, one vec4 each:
//   0: enabled (0/1), the last cascade's far distance, texel size (1 / cascade size), strength
//   1: the camera viewport in target pixels (x, y, w, h)    2: frustum tangents (left, right, top, bottom)
//   3: near, far, 0, 0                                      4: far split of cascades 0..3 (huge past the last cascade)
//   5 + 4c .. 7 + 4c: cascade c's rows, view space -> clip (x, y in -1..1, z = depth 0..1)
namespace sunshadow {
constexpr uint32_t kBlockVec4s = 21u;

// The function each op of the code belongs to.
static std::map<uint32_t, ir::SsaDef> functionOfOps(const ir::Builder& b) {
  std::map<uint32_t, ir::SsaDef> out;
  ir::SsaDef fn;
  auto [begin, end] = b.getCode();
  for (auto it = begin; it != end; ++it) {
    if (it->getOpCode() == ir::OpCode::eFunction) fn = it->getDef();
    out[it->getDef().getId()] = fn;
    if (it->getOpCode() == ir::OpCode::eFunctionEnd) fn = ir::SsaDef();
  }
  return out;
}

// Returns the register of the shadowed colour, -1 when the shader has no sun light colour, or -2 (with `err`) when it
// reads one where the pass cannot reach.
static int apply(ir::Builder& b, const std::vector<CtabEntry>& ctab, std::string& err) {
  int reg = -1;
  for (const auto& c : ctab)
    if (c.name == "PSLightColor" && c.set == "float4") reg = int(c.index);
  if (reg < 0) return -1;

  using ir::Op;
  using ir::SsaDef;
  const auto F = ir::ScalarType::eF32;
  const auto U = ir::ScalarType::eU32;
  const auto B = ir::ScalarType::eBool;
  const ir::BasicType F4(F, 4u), F2(F, 2u);

  // The float constants' buffer (the lowered c#), the position input, the entry point and its function.
  SsaDef cfDecl, posDecl, entryPoint, mainFn;
  {
    auto [begin, end] = b.getDeclarations();
    for (auto it = begin; it != end; ++it) {
      if (it->getOpCode() == ir::OpCode::eDclCbv && uint32_t(it->getOperand(1u)) == 0u &&
          uint32_t(it->getOperand(2u)) == abi::CbvConstFloat)
        cfDecl = it->getDef();
      if (it->getOpCode() == ir::OpCode::eDclInputBuiltIn &&
          ir::BuiltIn(it->getOperand(it->getFirstLiteralOperandIndex())) == ir::BuiltIn::ePosition)
        posDecl = it->getDef();
      if (it->getOpCode() == ir::OpCode::eEntryPoint) {
        entryPoint = it->getDef();
        mainFn = SsaDef(it->getOperand(0u));
      }
    }
  }
  if (!cfDecl || !entryPoint) { err = "sun shadow: no float constant buffer or entry point"; return -2; }

  // The loads of the sun's colour: from the float constants, at its register or at a computed address.
  struct Load { SsaDef def; bool dynamic; };
  std::vector<Load> loads;
  const auto fnOf = functionOfOps(b);
  {
    auto [begin, end] = b.getCode();
    for (auto it = begin; it != end; ++it) {
      if (it->getOpCode() != ir::OpCode::eBufferLoad) continue;
      const Op& desc = b.getOpForOperand(*it, 0u);
      if (desc.getOpCode() != ir::OpCode::eDescriptorLoad || SsaDef(desc.getOperand(0u)) != cfDecl) continue;
      const Op& addr = b.getOpForOperand(*it, 1u);
      const bool dynamic = addr.getOpCode() != ir::OpCode::eConstant;
      if (!dynamic && uint32_t(addr.getOperand(0u)) != uint32_t(reg)) continue;
      if (!it->getType().isBasicType() || it->getType().getBaseType(0u) != F4) {
        err = "sun shadow: unexpected constant load type";
        return -2;
      }
      auto f = fnOf.find(it->getDef().getId());
      if (f == fnOf.end() || f->second != mainFn) { err = "sun shadow: colour read outside the entry function"; return -2; }
      loads.push_back({ it->getDef(), dynamic });
    }
  }
  if (loads.empty()) return -1;

  // The entry function's first block: the visibility is computed there (uniform control flow, before any demote).
  SsaDef entryLabel;
  {
    auto [begin, end] = b.getCode();
    for (auto it = begin; it != end; ++it)
      if (it->getDef() == mainFn) {
        ++it;
        if (it != end && it->getOpCode() == ir::OpCode::eLabel) entryLabel = it->getDef();
        break;
      }
  }
  if (!entryLabel) { err = "sun shadow: no entry block"; return -2; }
  // The entry block is split below (the next cascade's filter is a branch): what refers to it as a block must then
  // refer to the block that ends with its original terminator. Its label's construct (the label's operand layout:
  // Label / LabelSelection(merge) / LabelLoop(merge, continue)) moves there too; it cannot be a loop header (nothing
  // branches back to an entry block).
  const ir::Construct entryConstruct = ir::Construct(b.getOp(entryLabel).getFirstLiteralOperandIndex());
  if (entryConstruct == ir::Construct::eStructuredLoop) { err = "sun shadow: entry block is a loop header"; return -2; }
  const SsaDef entryMerge = entryConstruct == ir::Construct::eStructuredSelection
                                ? SsaDef(b.getOp(entryLabel).getOperand(0u)) : SsaDef();
  std::vector<SsaDef> entryPhis;
  {
    std::vector<SsaDef> uses;
    b.getUses(entryLabel, uses);
    for (SsaDef u : uses) {
      const ir::OpCode oc = b.getOp(u).getOpCode();
      if (oc == ir::OpCode::ePhi) entryPhis.push_back(u);
      else if (oc != ir::OpCode::eDebugName) { err = "sun shadow: entry block referenced by a non-phi"; return -2; }
    }
  }

  const SsaDef block = b.add(Op::DclCbv(ir::Type(F4).addArrayDimension(kBlockVec4s), entryPoint, 0u, abi::CbvSunShadow, 1u));
  b.add(Op::DebugName(block, "fnvSunShadow"));
  const SsaDef image = b.add(Op::DclSrv(F, entryPoint, abi::SunRegSpace, 0u, 1u, ir::ResourceKind::eImage2DArray));
  b.add(Op::DebugName(image, "fnvSunCascades"));
  const SsaDef sampler = b.add(Op::DclSampler(entryPoint, abi::SunRegSpace, 0u, 1u));
  b.add(Op::DebugName(sampler, "fnvSunCompare"));
  if (!posDecl) posDecl = b.add(Op::DclInputBuiltIn(F4, entryPoint, ir::BuiltIn::ePosition, ir::InterpolationModes()));

  const SsaDef oldCursor = b.setCursor(entryLabel);
  auto cf = [&](float x) { return b.makeConstant(x); };
  auto cu = [&](uint32_t x) { return b.makeConstant(x); };
  auto add = [&](SsaDef x, SsaDef y) { return b.add(Op::FAdd(F, x, y)); };
  auto sub = [&](SsaDef x, SsaDef y) { return b.add(Op::FSub(F, x, y)); };
  auto mul = [&](SsaDef x, SsaDef y) { return b.add(Op::FMul(F, x, y)); };
  auto div = [&](SsaDef x, SsaDef y) { return b.add(Op::FDiv(F, x, y)); };
  auto sel = [&](SsaDef c, SsaDef x, SsaDef y) { return b.add(Op::Select(F, c, x, y)); };
  auto gt = [&](SsaDef x, SsaDef y) { return b.add(Op::FGt(B, x, y)); };
  auto ex = [&](SsaDef x, uint32_t c) { return b.add(Op::CompositeExtract(F, x, cu(c))); };
  auto lerp = [&](SsaDef x, SsaDef y, SsaDef t) { return add(x, mul(sub(y, x), t)); };

  const SsaDef blockDesc = b.add(Op::DescriptorLoad(ir::ScalarType::eCbv, block, cu(0u)));
  auto vec = [&](SsaDef index) { return b.add(Op::BufferLoad(F4, blockDesc, index, 16u)); };
  const SsaDef s0 = vec(cu(0u)), s1 = vec(cu(1u)), s2 = vec(cu(2u)), s3 = vec(cu(3u)), s4 = vec(cu(4u));
  const SsaDef enabled = ex(s0, 0u), last = ex(s0, 1u), texel = ex(s0, 2u), strength = ex(s0, 3u);

  // The fragment's view-space position (as fnvViewPos).
  const SsaDef fx = b.add(Op::InputLoad(F, posDecl, cu(0u)));
  const SsaDef fy = b.add(Op::InputLoad(F, posDecl, cu(1u)));
  const SsaDef fz = b.add(Op::InputLoad(F, posDecl, cu(2u)));
  const SsaDef zn = ex(s3, 0u), zf = ex(s3, 1u);
  const SsaDef z = div(mul(zn, zf), sub(zf, mul(fz, sub(zf, zn))));
  const SsaDef u = div(sub(fx, ex(s1, 0u)), ex(s1, 2u));
  const SsaDef v = div(sub(fy, ex(s1, 1u)), ex(s1, 3u));
  const SsaDef p[3] = { mul(lerp(ex(s2, 0u), ex(s2, 1u), u), z), mul(lerp(ex(s2, 2u), ex(s2, 3u), v), z), z };

  // The face normal, toward the camera, from the position's screen derivatives.
  SsaDef dx[3], dy[3];
  for (int i = 0; i < 3; i++) {
    dx[i] = b.add(Op::DerivX(F, p[i], ir::DerivativeMode::eDefault));
    dy[i] = b.add(Op::DerivY(F, p[i], ir::DerivativeMode::eDefault));
  }
  const SsaDef n[3] = { sub(mul(dy[1], dx[2]), mul(dy[2], dx[1])), sub(mul(dy[2], dx[0]), mul(dy[0], dx[2])),
                        sub(mul(dy[0], dx[1]), mul(dy[1], dx[0])) };
  const SsaDef len2 = add(add(mul(n[0], n[0]), mul(n[1], n[1])), mul(n[2], n[2]));
  const SsaDef inv = b.add(Op::FRsq(F, b.add(Op::FMax(F, len2, cf(1e-30f)))));
  const SsaDef facing = add(add(mul(n[0], p[0]), mul(n[1], p[1])), mul(n[2], p[2]));
  const SsaDef scale = sel(gt(len2, cf(1e-30f)), mul(inv, sel(gt(facing, cf(0.0f)), cf(-1.0f), cf(1.0f))), cf(0.0f));
  const SsaDef offset = mul(scale, add(cf(1.0f), mul(cf(0.002f), z)));
  SsaDef q[3];
  for (int i = 0; i < 3; i++) q[i] = add(p[i], mul(n[i], offset));

  // The cascade: how many far splits the point is beyond (the splits past the last cascade are huge).
  SsaDef c = cu(0u);
  for (uint32_t i = 0; i < 3; i++)
    c = b.add(Op::IAdd(U, c, b.add(Op::Select(U, gt(q[2], ex(s4, i)), cu(1u), cu(0u)))));
  const SsaDef imageDesc = b.add(Op::DescriptorLoad(ir::ScalarType::eSrv, image, cu(0u)));
  const SsaDef samplerDesc = b.add(Op::DescriptorLoad(ir::ScalarType::eSampler, sampler, cu(0u)));
  // The 3 x 3 compare filter of the point in cascade `cc`.
  auto filter = [&](SsaDef cc) {
    const SsaDef row = b.add(Op::IAdd(U, cu(5u), b.add(Op::IMul(U, cc, cu(4u)))));
    SsaDef clip[3];
    for (uint32_t i = 0; i < 3; i++) {
      const SsaDef r = vec(b.add(Op::IAdd(U, row, cu(i))));
      clip[i] = add(add(add(mul(ex(r, 0u), q[0]), mul(ex(r, 1u), q[1])), mul(ex(r, 2u), q[2])), ex(r, 3u));
    }
    const SsaDef uvx = add(mul(clip[0], cf(0.5f)), cf(0.5f)), uvy = add(mul(clip[1], cf(0.5f)), cf(0.5f));
    const SsaDef layer = b.add(Op::ConvertItoF(F, cc));
    SsaDef sum = cf(0.0f);
    for (int y = -1; y <= 1; y++)
      for (int x = -1; x <= 1; x++) {
        const SsaDef coord = b.add(Op::CompositeConstruct(F2, add(uvx, mul(cf(float(x)), texel)), add(uvy, mul(cf(float(y)), texel))));
        sum = add(sum, b.add(Op::ImageSample(F, imageDesc, samplerDesc, layer, coord, SsaDef(), cf(0.0f), SsaDef(), SsaDef(),
                                             SsaDef(), SsaDef(), clip[2])));
      }
    return mul(sum, cf(1.0f / 9.0f));
  };
  // Over the last 20% of a cascade's depth range the next cascade is blended in, so the change of texel size between
  // cascades is a gradient rather than a line. The last cascade's split is huge: it never blends (it fades out below).
  auto splitOf = [&](SsaDef cc) {
    auto is = [&](uint32_t i) { return b.add(Op::IEq(B, cc, cu(i))); };
    return sel(is(0u), ex(s4, 0u), sel(is(1u), ex(s4, 1u), sel(is(2u), ex(s4, 2u), ex(s4, 3u))));
  };
  const SsaDef split = splitOf(c);
  const SsaDef prevSplit = sel(b.add(Op::IEq(B, c, cu(0u))), cf(0.0f), splitOf(b.add(Op::ISub(U, b.add(Op::UMax(U, c, cu(1u))), cu(1u)))));
  const SsaDef blendStart = sub(split, mul(cf(0.2f), sub(split, prevSplit)));
  const SsaDef blend = b.add(Op::FClamp(F, div(sub(q[2], blendStart), sub(split, blendStart)), cf(0.0f), cf(1.0f)));
  const SsaDef next = b.add(Op::UMin(U, b.add(Op::IAdd(U, c, cu(1u))), cu(3u)));
  // The next cascade only where it is blended in (blend > 0: the last 20% of this cascade's range); at blend 0 the lerp
  // is the first filter exactly, and the next cascade's nine taps (past the last cascade: a clamped layer) are wasted.
  // The entry block ends here in a selection: entry -> (then: the blended filter) -> merge, which continues with the
  // rest of this code and the entry block's original instructions. The taps use an explicit LOD (valid in the branch).
  const SsaDef pcfFirst = filter(c);
  const SsaDef takeNext = gt(blend, cf(0.0f));
  const SsaDef branchDef = b.add(Op::BranchConditional(takeNext, entryLabel, entryLabel));  // targets set below
  const SsaDef thenLabel = b.add(Op::Label());
  const SsaDef pcfBlended = lerp(pcfFirst, filter(next), blend);
  const SsaDef thenEnd = b.add(Op::Branch(entryLabel));  // target set below
  const SsaDef mergeLabel = b.add(entryMerge ? Op::LabelSelection(entryMerge) : Op::Label());
  const SsaDef pcf = b.add(Op::Phi(F).addPhi(entryLabel, pcfFirst).addPhi(thenLabel, pcfBlended));
  b.rewriteOp(branchDef, Op::BranchConditional(takeNext, thenLabel, mergeLabel));
  b.rewriteOp(thenEnd, Op::Branch(mergeLabel));
  b.rewriteOp(entryLabel, Op::LabelSelection(mergeLabel));
  for (SsaDef phiDef : entryPhis) {  // the original successors' phis: their predecessor is now the merge block
    const Op& old = b.getOp(phiDef);
    Op phi = Op::Phi(old.getType());
    for (uint32_t i = 0; i + 1 < old.getOperandCount(); i += 2) {
      const SsaDef blk(old.getOperand(i));
      phi.addPhi(blk == entryLabel ? mergeLabel : blk, SsaDef(old.getOperand(i + 1)));
    }
    b.rewriteOp(phiDef, std::move(phi));
  }
  const SsaDef fadeStart = mul(last, cf(0.85f));
  const SsaDef t = b.add(Op::FClamp(F, div(sub(q[2], fadeStart), sub(last, fadeStart)), cf(0.0f), cf(1.0f)));
  const SsaDef fade = mul(mul(t, t), sub(cf(3.0f), mul(cf(2.0f), t)));
  const SsaDef weighted = sub(cf(1.0f), mul(strength, sub(cf(1.0f), lerp(pcf, cf(1.0f), fade))));
  const SsaDef visibility = sel(b.add(Op::FNe(B, enabled, cf(0.0f))), weighted, cf(1.0f));
  b.setCursor(oldCursor);

  // Scale the colour's rgb at each load (alpha as it is).
  for (const Load& l : loads) {
    const Op loadOp = b.getOp(l.def);
    const SsaDef copy = b.addBefore(l.def, Op(loadOp));
    SsaDef factor = visibility;
    if (l.dynamic) {
      const SsaDef isSun = b.addBefore(l.def, Op::IEq(B, SsaDef(loadOp.getOperand(1u)), cu(uint32_t(reg))));
      factor = b.addBefore(l.def, Op::Select(F, isSun, visibility, cf(1.0f)));
    }
    SsaDef comp[4];
    for (uint32_t i = 0; i < 4; i++) {
      comp[i] = b.addBefore(l.def, Op::CompositeExtract(F, copy, cu(i)));
      if (i < 3) comp[i] = b.addBefore(l.def, Op::FMul(F, comp[i], factor));
    }
    b.rewriteDef(l.def, b.addBefore(l.def, Op::CompositeConstruct(F4, comp[0], comp[1], comp[2], comp[3])));
  }
  return reg;
}
}  // namespace sunshadow

// ---------------------------------------------------------------------------------------------------------------
// The view vector of the game's lighting vertex shaders (for physically based lighting; docs/SHADER_ABI.md "View
// vector").
//
// The game's lighting pixel shaders work in tangent space: each lighting vertex shader turns its light vectors into
// tangent space by dotting them with the vertex's (deformed: skinned, wind-bent) tangent, binormal and normal, and only
// its SPECULAR variants also get the view direction (in their half vector, from EyePosition, c16 of
// ShadowLightShader's vertex constant map, 0x00B7E430). This pass gives every such shader the view direction in the
// same tangent space as one more output, at location kLocation:
//   M * (EyePosition.xyz - P.xyz)
// unnormalised, so that interpolated across a triangle it is the vector to the camera at each pixel (exact on a flat
// surface; normalised at the vertices it is not: up to 15 degrees off on a large triangle near the camera), which the
// pixel shader normalises,
// with M's rows the frame the shader itself dots its light vectors with (the first operand of a dp3 whose operand tree
// reads exactly one of the tangent, binormal and normal inputs; for a shader that dots none and does not deform, the
// inputs themselves), and P the position it transforms for oPos (the operand of oPos.w's dp4). The vector is the
// shader's own, whatever its skinning or wind; nothing else changes. The runtime must have EyePosition uploaded for the
// passes that use it (the game uploads it for specular passes). Checked by tools/shaderdiff --viewvec: a SPECULAR
// variant's half vector with LightData 0 is the same vector.
//
// The same shaders also get the tangent-to-world frame, at kFrameLocation + 0..2: row j is
//   (dot(R_j, M_0), dot(R_j, M_1), dot(R_j, M_2))
// with R_j = c(kFrameRegister + j).xyz, which the plugin sets per draw to the rows of the 3x3 part of the matrix the
// game makes the shader's space with (BSUtilities::GetInverseWorldMatrix of the geometry's world transform and skin:
// world_j = dot(R_j, object) for an object-space direction, up to the transform's scale). A tangent-space direction t
// is then world-space (dot(row_0, t), dot(row_1, t), dot(row_2, t)) up to scale, as the object-space direction M^T t
// is for an orthonormal frame. The pixel shaders normalise it. Checked by tools/shaderdiff --frame: with orthonormal
// frames, the frame applied to the view vector is R (EyePosition - position).
namespace viewvec {
constexpr uint32_t kLocation = 12u;
constexpr uint32_t kFrameLocation = 13u;
constexpr uint32_t kEyeRegister = 16u;
constexpr uint32_t kFrameRegister = 250u;

static bool isFunctionCall(const ir::Op& op, uint32_t args) {
  return op.getOpCode() == ir::OpCode::eFunctionCall && op.getOperandCount() == 1u + args;
}

// 1: the output was added; 0: not a lighting vertex shader of this form; -2 with `err`: an unexpected shape.
static int apply(ir::Builder& b, const std::vector<CtabEntry>& ctab, std::string& err) {
  bool light = false;
  for (const auto& c : ctab) {
    if (c.name == "LightData" && c.set == "float4") light = true;
    if (c.name == "EyePosition" && c.set == "float4" && c.index != kEyeRegister) return 0;  // another constant map
  }
  if (!light) return 0;

  using ir::Op;
  using ir::SsaDef;
  const auto F = ir::ScalarType::eF32;
  const ir::BasicType F4(F, 4u), F3(F, 3u);

  // Declarations: the float constants, the entry point, oPos, and the frame inputs by semantic.
  SsaDef cfDecl, entryPoint, mainFn, posDecl;
  enum { kT, kB, kN, kFrame };
  SsaDef frameDecl[kFrame];
  bool deforms = false;  // blend weights or indices: skinned or wind-bent
  {
    std::map<uint32_t, std::string> semantic;
    auto [begin, end] = b.getDeclarations();
    for (auto it = begin; it != end; ++it)
      if (it->getOpCode() == ir::OpCode::eSemantic && uint32_t(it->getOperand(1u)) == 0u)
        semantic[uint32_t(SsaDef(it->getOperand(0u)).getId())] = it->getLiteralString(2u);
    for (auto it = begin; it != end; ++it) {
      if (it->getOpCode() == ir::OpCode::eDclCbv && uint32_t(it->getOperand(1u)) == 0u &&
          uint32_t(it->getOperand(2u)) == abi::CbvConstFloat)
        cfDecl = it->getDef();
      if (it->getOpCode() == ir::OpCode::eDclOutputBuiltIn &&
          ir::BuiltIn(it->getOperand(it->getFirstLiteralOperandIndex())) == ir::BuiltIn::ePosition)
        posDecl = it->getDef();
      if (it->getOpCode() == ir::OpCode::eDclInput) {
        auto s = semantic.find(uint32_t(it->getDef().getId()));
        if (s == semantic.end()) continue;
        if (s->second == "tangent") frameDecl[kT] = it->getDef();
        if (s->second == "binormal") frameDecl[kB] = it->getDef();
        if (s->second == "normal") frameDecl[kN] = it->getDef();
        if (s->second == "blend" || s->second == "weight") deforms = true;
      }
      if (it->getOpCode() == ir::OpCode::eEntryPoint) {
        entryPoint = it->getDef();
        mainFn = SsaDef(it->getOperand(0u));
      }
    }
  }
  if (!cfDecl || !entryPoint || !posDecl || !frameDecl[kT] || !frameDecl[kB] || !frameDecl[kN]) return 0;

  // Which frame inputs an op's operand tree reads (bit per kT/kB/kN; memoised).
  std::map<uint32_t, uint32_t> reads;
  std::function<uint32_t(SsaDef)> readsOf = [&](SsaDef def) -> uint32_t {
    auto m = reads.find(uint32_t(def.getId()));
    if (m != reads.end()) return m->second;
    reads[uint32_t(def.getId())] = 0u;
    const Op& op = b.getOp(def);
    uint32_t mask = 0u;
    if (op.getOpCode() == ir::OpCode::eInputLoad) {
      for (uint32_t f = 0; f < kFrame; f++)
        if (SsaDef(op.getOperand(0u)) == frameDecl[f]) mask |= 1u << f;
    } else if (op.getOpCode() != ir::OpCode::eFunction && op.getOpCode() != ir::OpCode::eLabel) {
      const uint32_t first = op.getOpCode() == ir::OpCode::eFunctionCall ? 1u : 0u;
      for (uint32_t i = first; i < op.getFirstLiteralOperandIndex(); i++) {
        const SsaDef d = SsaDef(op.getOperand(i));
        if (d) mask |= readsOf(d);
      }
    }
    reads[uint32_t(def.getId())] = mask;
    return mask;
  };

  // The entry function: its return, oPos.w's value, and the frame rows (first operands of dp3 calls on vec3s).
  SsaDef ret, posW, row[kFrame];
  {
    bool inMain = false;
    auto [begin, end] = b.getCode();
    for (auto it = begin; it != end; ++it) {
      if (it->getOpCode() == ir::OpCode::eFunction) inMain = it->getDef() == mainFn;
      if (!inMain) continue;
      if (it->getOpCode() == ir::OpCode::eReturn) ret = it->getDef();
      if (it->getOpCode() == ir::OpCode::eOutputStore && SsaDef(it->getOperand(0u)) == posDecl) {
        const Op& addr = b.getOpForOperand(*it, 1u);
        if (addr.getOpCode() == ir::OpCode::eConstant && uint32_t(addr.getOperand(0u)) == 3u)
          posW = SsaDef(it->getOperand(2u));
      }
      if (isFunctionCall(*it, 2u) && it->getType().isBasicType() && it->getType().getBaseType(0u) == ir::BasicType(F, 1u)) {
        const SsaDef a = SsaDef(it->getOperand(1u)), c = SsaDef(it->getOperand(2u));
        const ir::Type at = b.getOp(a).getType(), ct = b.getOp(c).getType();
        if (!at.isBasicType() || at.getBaseType(0u) != F3 || !ct.isBasicType() || ct.getBaseType(0u) != F3) continue;
        const uint32_t m = readsOf(a);
        for (uint32_t f = 0; f < kFrame; f++)
          if (m == (1u << f) && !readsOf(c) && !row[f]) row[f] = a;  // the frame row (the other operand: a light vector)
      }
    }
  }
  if (!ret || !posW) return 0;
  const Op& w = b.getOp(posW);
  if (!isFunctionCall(w, 2u)) return 0;
  const SsaDef pos = SsaDef(w.getOperand(2u));
  {
    const ir::Type pt = b.getOp(pos).getType();
    if (!pt.isBasicType() || pt.getBaseType(0u) != F4) { err = "view vector: unexpected position type"; return -2; }
  }

  auto cf = [&](float x) { return b.makeConstant(x); };
  auto cu = [&](uint32_t x) { return b.makeConstant(x); };
  auto at = [&](const Op& op) { return b.addBefore(ret, op); };
  auto ex = [&](SsaDef v, uint32_t c) { return at(Op::CompositeExtract(F, v, cu(c))); };
  auto mul = [&](SsaDef x, SsaDef y) { return at(Op::FMul(F, x, y)); };
  auto add = [&](SsaDef x, SsaDef y) { return at(Op::FAdd(F, x, y)); };
  auto dot3 = [&](const SsaDef* x, const SsaDef* y) { return add(add(mul(x[0], y[0]), mul(x[1], y[1])), mul(x[2], y[2])); };

  SsaDef frame[kFrame][3];
  if (row[kT] && row[kB] && row[kN]) {
    for (uint32_t f = 0; f < kFrame; f++)
      for (uint32_t i = 0; i < 3u; i++) frame[f][i] = ex(row[f], i);
  } else if (!row[kT] && !row[kB] && !row[kN] && !deforms) {
    for (uint32_t f = 0; f < kFrame; f++)  // the shader passes its frame on undeformed: the inputs are the frame
      for (uint32_t i = 0; i < 3u; i++) frame[f][i] = at(Op::InputLoad(F, frameDecl[f], cu(i)));
  } else {
    return 0;
  }

  const SsaDef out = b.add(Op::DclOutput(F4, entryPoint, kLocation, 0u));
  b.add(Op::DebugName(out, "oView"));
  const SsaDef desc = at(Op::DescriptorLoad(ir::ScalarType::eCbv, cfDecl, cu(0u)));
  const SsaDef eye = at(Op::BufferLoad(F4, desc, cu(kEyeRegister), 16u));
  SsaDef v[3];
  for (uint32_t i = 0; i < 3u; i++) v[i] = at(Op::FSub(F, ex(eye, i), ex(pos, i)));
  SsaDef t[3];
  for (uint32_t f = 0; f < kFrame; f++) t[f] = dot3(frame[f], v);
  for (uint32_t i = 0; i < 3u; i++) at(Op::OutputStore(out, cu(i), t[i]));
  at(Op::OutputStore(out, cu(3u), cf(0.0f)));
  for (uint32_t j = 0; j < 3u; j++) {
    const SsaDef frameOut = b.add(Op::DclOutput(F4, entryPoint, kFrameLocation + j, 0u));
    b.add(Op::DebugName(frameOut, j == 0 ? "oFrame0" : j == 1 ? "oFrame1" : "oFrame2"));
    const SsaDef r = at(Op::BufferLoad(F4, desc, cu(kFrameRegister + j), 16u));
    SsaDef rj[3];
    for (uint32_t i = 0; i < 3u; i++) rj[i] = ex(r, i);
    for (uint32_t f = 0; f < kFrame; f++) at(Op::OutputStore(frameOut, cu(f), dot3(rj, frame[f])));
    at(Op::OutputStore(frameOut, cu(3u), cf(0.0f)));
  }
  return 1;
}

// Every other vertex shader writes locations 12 to 15 as 0: a pixel shader may read the view vector whatever vertex shader the
// game pairs it with (Vulkan requires the inputs a fragment shader reads to be written), and a zero vector says there is
// none. False if the module has no entry function to write it in.
static bool zero(ir::Builder& b) {
  using ir::Op;
  using ir::SsaDef;
  SsaDef entryPoint, mainFn;
  {
    auto [begin, end] = b.getDeclarations();
    for (auto it = begin; it != end; ++it)
      if (it->getOpCode() == ir::OpCode::eEntryPoint) {
        entryPoint = it->getDef();
        mainFn = SsaDef(it->getOperand(0u));
      }
  }
  if (!entryPoint) return false;
  std::vector<SsaDef> rets;
  {
    bool inMain = false;
    auto [begin, end] = b.getCode();
    for (auto it = begin; it != end; ++it) {
      if (it->getOpCode() == ir::OpCode::eFunction) inMain = it->getDef() == mainFn;
      if (inMain && it->getOpCode() == ir::OpCode::eReturn) rets.push_back(it->getDef());
    }
  }
  if (rets.empty()) return false;
  for (uint32_t l = kLocation; l <= kFrameLocation + 2u; l++) {
    const SsaDef out = b.add(Op::DclOutput(ir::BasicType(ir::ScalarType::eF32, 4u), entryPoint, l, 0u));
    b.add(Op::DebugName(out, l == kLocation ? "oView" : "oFrame"));
    for (SsaDef ret : rets)
      for (uint32_t i = 0; i < 4u; i++) b.addBefore(ret, Op::OutputStore(out, b.makeConstant(i), b.makeConstant(0.0f)));
  }
  return true;
}
}  // namespace viewvec


// The grass pass. TallGrassShader's lit vertex shaders (CTAB: DiffuseDir c0, DiffuseColor c1, AddlParams c7, InstanceData
// c20) light each vertex themselves and hand the pixel shader only the result. For the physically based grass
// (replacements/lighting grass.frag, docs/LIGHTING.md "Grass"), this pass gives the pixel shader the inputs instead, at
// the view vector pass's locations 12..15 (which these shaders otherwise write as 0):
//   12: xyz the blade's normal (the mesh's NORMAL where it has one, which those variants light with: dp3 c0, v1; else
//       the instance's ground normal, which the others light with: frac(InstanceData[i].xyz) * 2 - 1), w the canopy
//       fraction saturate(position.z / canopy height);
//   13: xyz the clump vector position.xyz + (0, 0, canopy height / 2) (Community Shaders' GetClumpNormal: a pivot below the
//       root), w DiffuseDir.x;
//   14: xyz the sun's colour DiffuseColor.rgb x AddlParams.x x the instance's occlusion (frac(InstanceData[i].w) x 0.75 +
//       0.25), w DiffuseDir.y;
//   15: xyz the vertex colour, w DiffuseDir.z.
// i = floor(TEXCOORD1.x), as the shaders compute it (frc, add, mova). The canopy height is c250.x (the plugin sets it per
// draw to the mesh's top, plugin/src/grasslight.cpp; at least 1). Positions are the mesh's (model space: the shaders only
// scale and offset them, no rotation, so model directions are world directions).
namespace grassvec {
constexpr uint32_t kLocation = 12u;
constexpr uint32_t kCanopyRegister = 250u;

// 1: the outputs were added; 0: not a lit grass vertex shader.
static int apply(ir::Builder& b, const std::vector<CtabEntry>& ctab) {
  bool dir = false, color = false, addl = false, inst = false;
  for (const auto& c : ctab) {
    if (c.set != "float4") continue;
    dir |= c.name == "DiffuseDir" && c.index == 0u;
    color |= c.name == "DiffuseColor" && c.index == 1u;
    addl |= c.name == "AddlParams" && c.index == 7u;
    inst |= c.name == "InstanceData" && c.index == 20u;
  }
  if (!dir || !color || !addl || !inst) return 0;

  using ir::Op;
  using ir::SsaDef;
  const auto F = ir::ScalarType::eF32;
  const auto U = ir::ScalarType::eU32;
  const ir::BasicType F4(F, 4u);
  SsaDef cfDecl, entryPoint, mainFn, posIn, colorIn, normalIn, tc1In;
  {
    std::map<uint32_t, std::pair<std::string, uint32_t>> semantic;
    auto [begin, end] = b.getDeclarations();
    for (auto it = begin; it != end; ++it)
      if (it->getOpCode() == ir::OpCode::eSemantic)
        semantic[uint32_t(SsaDef(it->getOperand(0u)).getId())] = { it->getLiteralString(2u), uint32_t(it->getOperand(1u)) };
    for (auto it = begin; it != end; ++it) {
      if (it->getOpCode() == ir::OpCode::eDclCbv && uint32_t(it->getOperand(1u)) == 0u &&
          uint32_t(it->getOperand(2u)) == abi::CbvConstFloat)
        cfDecl = it->getDef();
      if (it->getOpCode() == ir::OpCode::eDclInput) {
        auto sm = semantic.find(uint32_t(it->getDef().getId()));
        if (sm == semantic.end()) continue;
        if (sm->second.first == "position" && sm->second.second == 0u) posIn = it->getDef();
        if (sm->second.first == "color" && sm->second.second == 0u) colorIn = it->getDef();
        if (sm->second.first == "normal" && sm->second.second == 0u) normalIn = it->getDef();
        if (sm->second.first == "texcoord" && sm->second.second == 1u) tc1In = it->getDef();
      }
      if (it->getOpCode() == ir::OpCode::eEntryPoint) {
        entryPoint = it->getDef();
        mainFn = SsaDef(it->getOperand(0u));
      }
    }
  }
  if (!cfDecl || !entryPoint || !posIn || !colorIn || !tc1In) return 0;
  std::vector<SsaDef> rets;
  {
    bool inMain = false;
    auto [begin, end] = b.getCode();
    for (auto it = begin; it != end; ++it) {
      if (it->getOpCode() == ir::OpCode::eFunction) inMain = it->getDef() == mainFn;
      if (inMain && it->getOpCode() == ir::OpCode::eReturn) rets.push_back(it->getDef());
    }
  }
  if (rets.empty()) return 0;

  SsaDef out[4];
  const char* names[4] = { "oGrassNormal", "oGrassClump", "oGrassSun", "oGrassColor" };
  for (uint32_t l = 0; l < 4u; l++) {
    out[l] = b.add(Op::DclOutput(F4, entryPoint, kLocation + l, 0u));
    b.add(Op::DebugName(out[l], names[l]));
  }
  for (SsaDef ret : rets) {
    auto at = [&](const Op& op) { return b.addBefore(ret, op); };
    auto cf = [&](float x) { return b.makeConstant(x); };
    auto cu = [&](uint32_t x) { return b.makeConstant(x); };
    auto ex = [&](SsaDef v, uint32_t c) { return at(Op::CompositeExtract(F, v, cu(c))); };
    auto in = [&](SsaDef decl, uint32_t c) { return at(Op::InputLoad(F, decl, cu(c))); };
    const SsaDef desc = at(Op::DescriptorLoad(ir::ScalarType::eCbv, cfDecl, cu(0u)));
    auto reg = [&](SsaDef address) { return at(Op::BufferLoad(F4, desc, address, 16u)); };
    // The instance.
    const SsaDef idxF = at(Op::FRound(F, in(tc1In, 0u), ir::RoundMode::eNegativeInf));
    const SsaDef idx = at(Op::IAdd(U, at(Op::ConvertFtoI(U, idxF)), cu(20u)));
    const SsaDef instance = reg(idx);
    const SsaDef c0 = reg(cu(0u)), c1 = reg(cu(1u)), c7 = reg(cu(7u)), canopyReg = reg(cu(kCanopyRegister));
    const SsaDef occlusion = at(Op::FAdd(F, at(Op::FMul(F, at(Op::FFract(F, ex(instance, 3u))), cf(0.75f))), cf(0.25f)));
    // 12: the blade's normal and the canopy fraction.
    for (uint32_t i = 0; i < 3u; i++) {
      const SsaDef n = normalIn ? in(normalIn, i)
                                : at(Op::FMul(F, at(Op::FAdd(F, at(Op::FFract(F, ex(instance, i))), cf(-0.5f))), cf(2.0f)));
      at(Op::OutputStore(out[0], cu(i), n));
    }
    const SsaDef h = at(Op::FMax(F, ex(canopyReg, 0u), cf(1.0f)));
    const SsaDef pz = in(posIn, 2u);
    at(Op::OutputStore(out[0], cu(3u), at(Op::FClamp(F, at(Op::FDiv(F, pz, h)), cf(0.0f), cf(1.0f)))));
    // 13: the clump vector.
    at(Op::OutputStore(out[1], cu(0u), in(posIn, 0u)));
    at(Op::OutputStore(out[1], cu(1u), in(posIn, 1u)));
    at(Op::OutputStore(out[1], cu(2u), at(Op::FAdd(F, pz, at(Op::FMul(F, h, cf(0.5f)))))));
    // 14: the sun's colour; 15: the vertex colour; w: the sun's direction.
    const SsaDef scale = at(Op::FMul(F, ex(c7, 0u), occlusion));
    for (uint32_t i = 0; i < 3u; i++) {
      at(Op::OutputStore(out[2], cu(i), at(Op::FMul(F, ex(c1, i), scale))));
      at(Op::OutputStore(out[3], cu(i), in(colorIn, i)));
    }
    at(Op::OutputStore(out[1], cu(3u), ex(c0, 0u)));
    at(Op::OutputStore(out[2], cu(3u), ex(c0, 1u)));
    at(Op::OutputStore(out[3], cu(3u), ex(c0, 2u)));
  }
  return 1;
}
}  // namespace grassvec

static ShaderResult compileShader(const std::string& hash, const std::vector<uint8_t>& code) {
  ShaderResult r;
  r.hash = hash;

  if (code.size() < 8u || code.size() % 4u) {
    r.log.push_back("bad bytecode size");
    return r;
  }

  uint32_t version = 0u;
  std::memcpy(&version, code.data(), 4u);
  uint32_t kind = version >> 16;
  if (kind != 0xFFFEu && kind != 0xFFFFu) {
    r.log.push_back("bad version token");
    return r;
  }
  r.isVertex = kind == 0xFFFEu;
  r.profile = std::string(r.isVertex ? "vs_" : "ps_") + std::to_string((version >> 8) & 0xFFu) + "_" +
              std::to_string(version & 0xFFu);

  readCtab(code, r.ctab);
  readFloatDefs(code, r.defs);

  CaptureLogger logger;
  ir::Builder builder;

  sm3::Converter::Options convOptions = { };
  convOptions.includeDebugNames = true;
  convOptions.fastFloatEmulation = false;  // exact D3D9 multiply-by-zero rules
  // The game never binds depth textures for sampling (counted in game by the backend), so the run-time
  // depth-compare path is left out: no comparison sampling of colour views exists in any module.
  convOptions.emitDepthCompare = false;
  sm3::Converter converter(util::ByteReader(code.data(), code.size()), convOptions);

  if (!converter.convertShader(builder)) {
    r.log = logger.lines;
    r.log.push_back("convertShader failed");
    return r;
  }

  ir::CompileOptions compileOptions = { };
  compileOptions.cseOptions.relocateDescriptorLoad = true;
  compileOptions.descriptorIndexing.optimizeDescriptorIndexing = true;
  // D3D9 `_pp` (partial precision) is a hint; native D3D9 drivers on current GPUs execute it at fp32, which is
  // how the game has always rendered there. Keep min-precision at 32 bits (no shaderFloat16/Int16 requirement).
  compileOptions.min16Options.enableFloat16 = false;
  compileOptions.min16Options.enableInt16 = false;
  ir::legalizeIr(builder, compileOptions);

  collectIo(builder, r.io);
  for (const auto& v : r.io) {
    bool linkedStageVar = (r.isVertex && v.output) || (!r.isVertex && !v.output);
    if (linkedStageVar && !v.builtin && !isFixedFunctionSemantic(v.semantic, v.semanticIndex))
      r.nonFixedIo = true;
  }

  {
    ir::LowerIoPass pass(builder);
    pass.lowerSampleCountToSpecConstant(0u);
    pass.lowerLegacyBuiltInToCbv(ir::BuiltIn::eLegacyConstFloat, 0u, abi::CbvConstFloat);
    pass.lowerLegacyBuiltInToCbv(ir::BuiltIn::eLegacyConstInt, 0u, abi::CbvConstInt);
    pass.lowerLegacyBuiltInToCbv(ir::BuiltIn::eLegacyConstBool, 0u, abi::CbvConstBool);
    pass.lowerLegacyBuiltInToCbv(ir::BuiltIn::eLegacyAlphaTest, 0u, abi::CbvAlphaTest);
    pass.lowerLegacyBuiltInToCbv(ir::BuiltIn::eLegacyFog, 0u, abi::CbvFog);
    pass.lowerLegacyBuiltInToCbv(ir::BuiltIn::eLegacyClipPlanes, 0u, abi::CbvClipPlanes);
    pass.lowerLegacyBuiltInToCbv(ir::BuiltIn::eLegacyPointArgs, 0u, abi::CbvPointArgs);
    pass.lowerLegacyBuiltInToCbv(ir::BuiltIn::eLegacySamplerState, 0u, abi::CbvSamplerState);
    pass.lowerLegacyBuiltInToCbv(ir::BuiltIn::eLegacyTextureStage, 0u, abi::CbvTextureStage);
  }

  if (r.isVertex) {
    std::string err;
    const int v = viewvec::apply(builder, r.ctab, err);
    if (v == -2) {
      r.log = logger.lines;
      r.log.push_back(err);
      return r;
    }
    r.viewVector = v == 1;
    r.grassVector = v == 0 && grassvec::apply(builder, r.ctab) == 1;
    if (v == 0 && !r.grassVector && !viewvec::zero(builder)) {
      r.log = logger.lines;
      r.log.push_back("view vector: no entry point to write location 12 in");
      return r;
    }
  }
  if (!r.isVertex) {
    std::string err;
    r.sunLight = sunshadow::apply(builder, r.ctab, err);
    if (r.sunLight == -2) {
      r.log = logger.lines;
      r.log.push_back(err);
      return r;
    }
  }

  {
    auto [begin, end] = builder.getDeclarations();
    for (auto it = begin; it != end; ++it) {
      if (it->getOpCode() == ir::OpCode::eDclSrv && uint32_t(it->getOperand(1u)) != abi::SunRegSpace)
        r.textures.push_back(TextureDecl { uint32_t(it->getOperand(2u)), uint32_t(it->getOperand(4u)) });
    }
  }

  if (!g_dumpIr.empty()) {
    std::ofstream d(g_dumpIr, std::ios::app);
    d << "=== " << hash << '\n';
    ir::Disassembler::Options o = { };
    o.resolveConstants = true;
    o.showConstants = false;
    ir::Disassembler(builder, o).disassemble(d);
  }

  FnvResourceMapping mapping(r.isVertex ? abi::SetVertex : abi::SetPixel);

  spirv::SpirvBuilder::Options spirvOptions = { };
  spirvOptions.includeDebugNames = true;
  spirvOptions.floatControls2 = false;
  spirvOptions.supportedRoundModesF16 = g_fpCaps.round16;
  spirvOptions.supportedRoundModesF32 = g_fpCaps.round32;
  spirvOptions.supportedRoundModesF64 = g_fpCaps.round64;
  spirvOptions.supportedDenormModesF16 = g_fpCaps.denorm16;
  spirvOptions.supportedDenormModesF32 = g_fpCaps.denorm32;
  spirvOptions.supportedDenormModesF64 = g_fpCaps.denorm64;

  spirv::SpirvBuilder spirvBuilder(builder, mapping, spirvOptions);
  spirvBuilder.buildSpirvBinary();

  size_t size = 0u;
  spirvBuilder.getSpirvBinary(size, nullptr);
  if (!size || size % 4u) {
    r.log = logger.lines;
    r.log.push_back("empty or misaligned SPIR-V");
    return r;
  }
  r.spirvWords.resize(size / 4u);
  spirvBuilder.getSpirvBinary(size, r.spirvWords.data());

  r.bindings = mapping.records;
  r.pushData = mapping.pushDataRequested;
  r.abiViolation = mapping.abiViolation;
  r.log = logger.lines;
  r.ok = !r.abiViolation && !r.pushData;
  if (r.abiViolation) r.log.push_back("binding ABI violation");
  if (r.pushData) r.log.push_back("unexpected push data");
  return r;
}

// ---------------------------------------------------------------------------------------------------------------
static void writeShaderJson(std::ostream& o, const ShaderResult& r, const std::set<std::string>& names) {
  o << "    \"" << r.hash << "\": {\n";
  o << "      \"profile\": \"" << r.profile << "\", \"ok\": " << (r.ok ? "true" : "false")
    << ", \"nonFixedIo\": " << (r.nonFixedIo ? "true" : "false") << ", \"sunLight\": " << r.sunLight
    << ", \"viewVector\": " << (r.viewVector ? "true" : "false") << ", \"grassVector\": " << (r.grassVector ? "true" : "false") << ",\n";

  o << "      \"names\": [";
  bool first = true;
  for (const auto& n : names) { o << (first ? "" : ", ") << "\"" << jsonEscape(n) << "\""; first = false; }
  o << "],\n";

  o << "      \"ctab\": [";
  first = true;
  for (const auto& c : r.ctab) {
    o << (first ? "" : ", ") << "{\"name\": \"" << jsonEscape(c.name) << "\", \"set\": \"" << c.set
      << "\", \"index\": " << c.index << ", \"count\": " << c.count << "}";
    first = false;
  }
  o << "],\n";

  o << "      \"io\": [";
  first = true;
  for (const auto& v : r.io) {
    o << (first ? "" : ", ") << "{\"dir\": \"" << (v.output ? "out" : "in") << "\", "
      << (v.builtin ? "\"builtin\": " : "\"location\": ") << v.location << ", \"components\": " << v.components
      << ", \"semantic\": \"" << jsonEscape(v.semantic) << "\", \"semanticIndex\": " << v.semanticIndex << "}";
    first = false;
  }
  o << "],\n";

  o << "      \"textures\": [";
  first = true;
  for (const auto& t : r.textures) {
    o << (first ? "" : ", ") << "{\"reg\": " << t.reg << ", \"kind\": " << t.kind << "}";
    first = false;
  }
  o << "],\n";

  o << "      \"defs\": [";
  first = true;
  for (const auto& d : r.defs) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "[%u, %.9g, %.9g, %.9g, %.9g]", d.reg, d.v[0], d.v[1], d.v[2], d.v[3]);
    o << (first ? "" : ", ") << buf;
    first = false;
  }
  o << "],\n";

  o << "      \"bindings\": [";
  first = true;
  for (const auto& b : r.bindings) {
    o << (first ? "" : ", ") << "{\"kind\": \"" << b.kind << "\", \"space\": " << b.regSpace << ", \"reg\": "
      << b.regIndex << ", \"set\": " << b.set << ", \"binding\": " << b.binding << "}";
    first = false;
  }
  o << "],\n";

  o << "      \"log\": [";
  first = true;
  for (const auto& l : r.log) { o << (first ? "" : ", ") << "\"" << jsonEscape(l) << "\""; first = false; }
  o << "]\n    }";
}

int main(int argc, char** argv) {
  fs::path shaderDir, outDir;
  std::vector<int> packages;
  std::string only;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--shaders" && i + 1 < argc) shaderDir = argv[++i];
    else if (a == "--out" && i + 1 < argc) outDir = argv[++i];
    else if (a == "--only" && i + 1 < argc) only = argv[++i];
    else if (a == "--dump-ir" && i + 1 < argc) g_dumpIr = argv[++i];
    else if (a == "--fp-caps" && i + 1 < argc) {
      if (!parseFpCaps(argv[++i], g_fpCaps)) { std::cerr << "bad --fp-caps (e.g. rte32,rtz32,ftz32,dp16)\n"; return 2; }
    }
    else if (a == "--packages" && i + 1 < argc) {
      std::stringstream s(argv[++i]);
      std::string tok;
      while (std::getline(s, tok, ',')) packages.push_back(std::stoi(tok));
    } else {
      std::cerr << "usage: sdp2spv --shaders <Data/Shaders> --out <dir> [--packages 2,3,...] [--only NAME]\n"
                   "               [--fp-caps rte32,rtz32,ftz32,dp32,...]  (device float controls; default none)\n";
      return 2;
    }
  }
  if (shaderDir.empty() || outDir.empty()) {
    std::cerr << "--shaders and --out are required\n";
    return 2;
  }

  // Packages the PC build can select (ShaderBuffer_GetCurrentBufferType, 0x00B7FAF0): 2,3,4,6,7,10..17.
  // 9/18/19 are shipped but never selected on PC; they are translated too unless --packages narrows it.
  if (packages.empty()) packages = { 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 };

  fs::create_directories(outDir / "spv");

  std::map<std::string, std::vector<uint8_t>> uniqueCode;             // hash -> bytecode
  std::map<std::string, std::set<std::string>> namesByHash;           // hash -> {"002/SLS1000.vso", ...}
  std::map<int, std::vector<std::pair<std::string, std::string>>> packageIndex;  // pkg -> (name, hash)
  size_t totalBlobs = 0;

  for (int pkg : packages) {
    char fname[64];
    std::snprintf(fname, sizeof(fname), "shaderpackage%03d.sdp", pkg);
    std::vector<PackageEntry> entries;
    std::string err;
    if (!readPackage(shaderDir / fname, entries, err)) {
      std::cerr << "error: " << fname << ": " << err << "\n";
      return 1;
    }
    for (auto& e : entries) {
      if (!only.empty() && e.name != only) continue;
      std::string h = hexDigest(util::md5::Hasher::compute(e.code.data(), e.code.size()));
      char tag[8];
      std::snprintf(tag, sizeof(tag), "%03d/", pkg);
      namesByHash[h].insert(tag + e.name);
      packageIndex[pkg].emplace_back(e.name, h);
      uniqueCode.emplace(h, std::move(e.code));
      totalBlobs++;
    }
  }

  std::cerr << "blobs: " << totalBlobs << ", unique: " << uniqueCode.size() << "\n";

  std::vector<ShaderResult> results;
  results.reserve(uniqueCode.size());
  size_t ok = 0, failed = 0, nonFixed = 0, done = 0;
  std::map<std::string, size_t> byProfile;

  for (const auto& [hash, code] : uniqueCode) {
    ShaderResult r = compileShader(hash, code);
    byProfile[r.profile]++;
    if (r.ok) {
      ok++;
      std::ofstream f(outDir / "spv" / (hash + ".spv"), std::ios::binary | std::ios::trunc);
      f.write(reinterpret_cast<const char*>(r.spirvWords.data()), r.spirvWords.size() * 4u);
      // Keep the original bytecode next to it for the differential test harness.
      std::ofstream b(outDir / "spv" / (hash + ".dxso"), std::ios::binary | std::ios::trunc);
      b.write(reinterpret_cast<const char*>(code.data()), code.size());
    } else {
      failed++;
      std::cerr << "FAIL " << hash << " " << *namesByHash[hash].begin() << " (" << r.profile << ")\n";
      for (const auto& l : r.log) std::cerr << "    " << l << "\n";
    }
    if (r.nonFixedIo) nonFixed++;
    r.spirvWords.clear();
    r.spirvWords.shrink_to_fit();
    results.push_back(std::move(r));
    if (++done % 500 == 0) std::cerr << "  " << done << "/" << uniqueCode.size() << "\n";
  }

  std::ofstream m(outDir / "manifest.json", std::ios::trunc);
  m << "{\n  \"fpCaps\": \"" << g_fpCaps.spec << "\",\n";
  m << "  \"abi\": {\"setVertex\": 0, \"setPixel\": 1, \"cbv\": {\"constFloat\": 0, \"constInt\": 1, "
       "\"constBool\": 2, \"alphaTest\": 3, \"fog\": 4, \"clipPlanes\": 5, \"pointArgs\": 6, \"samplerState\": 7, "
       "\"textureStage\": 8}, \"textureBase\": 16, \"samplerBase\": 32},\n";
  m << "  \"packages\": {\n";
  bool firstPkg = true;
  for (const auto& [pkg, list] : packageIndex) {
    m << (firstPkg ? "" : ",\n") << "    \"" << std::setw(3) << std::setfill('0') << pkg << "\": {";
    bool first = true;
    for (const auto& [name, h] : list) {
      m << (first ? "" : ", ") << "\"" << jsonEscape(name) << "\": \"" << h << "\"";
      first = false;
    }
    m << "}";
    firstPkg = false;
  }
  m << "\n  },\n  \"shaders\": {\n";
  for (size_t i = 0; i < results.size(); i++) {
    writeShaderJson(m, results[i], namesByHash[results[i].hash]);
    m << (i + 1 < results.size() ? ",\n" : "\n");
  }
  m << "  }\n}\n";

  std::cerr << "translated: " << ok << ", failed: " << failed << ", with non-fixed VS/PS interface semantics: "
            << nonFixed << "\n";
  for (const auto& [p, c] : byProfile) std::cerr << "  " << p << ": " << c << "\n";
  return failed ? 1 : 0;
}
