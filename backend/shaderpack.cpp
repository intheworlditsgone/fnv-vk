#include <cstring>

#include "shaderpack.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <bcrypt.h>

#include <algorithm>
#include <cstdio>

namespace fnvvk {

namespace {
// Every pixel shader kills pixels once for the alpha test (the translation's lowered D3DRS_ALPHATESTENABLE, or a
// replacement's fnvAlphaTest): a second discard instruction is the shader's own (D3D9 texkill: all 441 game shaders
// with texkill translate to exactly two, the others to one). FragDepth: the shader writes depth (oDepth). Vertex
// shaders: whether the variable decorated BuiltIn Position is also decorated Invariant.
void spvTraits(PackedShader& s) {
  s.discards = s.writesDepth = s.invariantPosition = false;
  const uint32_t* w = s.spv;
  const size_t n = s.spvBytes / 4;
  uint32_t kills = 0, position = 0;
  std::vector<uint32_t> invariant;
  for (size_t i = 5; i < n;) {
    const uint32_t op = w[i] & 0xFFFF, len = w[i] >> 16;
    if (!len || i + len > n) break;
    if (op == 252 /*OpKill*/ || op == 4416 /*OpTerminateInvocation*/ || op == 5380 /*OpDemoteToHelperInvocation*/) kills++;
    if (op == 71 /*OpDecorate*/ && len >= 4 && w[i + 2] == 11 /*BuiltIn*/ && w[i + 3] == 22 /*FragDepth*/) s.writesDepth = true;
    if (op == 72 /*OpMemberDecorate*/ && len >= 5 && w[i + 3] == 11 && w[i + 4] == 22) s.writesDepth = true;
    if (op == 71 && len >= 4 && w[i + 2] == 11 && w[i + 3] == 0 /*Position*/) position = w[i + 1];
    if (op == 71 && len >= 3 && w[i + 2] == 18 /*Invariant*/) invariant.push_back(w[i + 1]);
    i += len;
  }
  if (s.pixel) {
    s.discards = kills > 1;
  } else {
    s.writesDepth = false;
    s.invariantPosition = position && std::find(invariant.begin(), invariant.end(), position) != invariant.end();
  }
}

template <typename T>
bool take(const uint8_t*& p, const uint8_t* end, T& out) {
  if (size_t(end - p) < sizeof(T)) return false;
  memcpy(&out, p, sizeof(T));
  p += sizeof(T);
  return true;
}
}  // namespace

bool ShaderPack::load(const std::string& path, std::string* error) {
  auto fail = [&](const char* why) {
    if (error) *error = path + ": " + why;
    shaders_.clear();
    blob_.clear();
    return false;
  };
  FILE* f = nullptr;
  if (fopen_s(&f, path.c_str(), "rb") || !f) return fail("cannot open");
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  blob_.resize(size_t(size));
  size_t got = fread(blob_.data(), 1, blob_.size(), f);
  fclose(f);
  if (got != blob_.size()) return fail("short read");

  const uint8_t* p = blob_.data();
  const uint8_t* end = p + blob_.size();
  char magic[4];
  uint32_t version = 0, count = 0;
  if (!take(p, end, magic) || memcmp(magic, "FVKS", 4) || !take(p, end, version) || version != 2 ||
      !take(p, end, count))
    return fail("bad header");
  shaders_.reserve(count);
  for (uint32_t i = 0; i < count; i++) {
    Md5 key;
    uint8_t stage, nDefs, nTex, nIn;
    PackedShader s;
    if (!take(p, end, key.b) || !take(p, end, stage) || !take(p, end, nDefs) || !take(p, end, nTex) ||
        !take(p, end, nIn) || !take(p, end, s.cbvMask) || !take(p, end, s.textureMask) || !take(p, end, s.spvBytes))
      return fail("truncated record");
    s.pixel = stage == 1;
    memcpy(s.md5, key.b.data(), 16);
    s.defs.resize(nDefs);
    for (auto& d : s.defs)
      if (!take(p, end, d.reg) || !take(p, end, d.v)) return fail("truncated defs");
    s.textures.resize(nTex);
    for (auto& t : s.textures)
      if (!take(p, end, t.reg) || !take(p, end, t.kind)) return fail("truncated textures");
    s.inputs.resize(nIn);
    for (auto& in : s.inputs)
      if (!take(p, end, in.usage) || !take(p, end, in.usageIndex) || !take(p, end, in.location) ||
          !take(p, end, in.components))
        return fail("truncated inputs");
    while ((p - blob_.data()) & 3) p++;  // padding to a 4-byte file offset (the vector's storage is 8-aligned)
    if (p > end || size_t(end - p) < s.spvBytes || s.spvBytes % 4)
      return fail("bad spirv");
    s.spv = reinterpret_cast<const uint32_t*>(p);
    p += s.spvBytes;
    s.constAll = !spvConstantRuns(s.spv, s.spvBytes / 4, s.pixel ? 1 : 0, s.pixel ? 224 : 256, s.constRuns);
    spvTraits(s);
    shaders_.emplace(key, std::move(s));
  }
  return true;
}

const PackedShader* ShaderPack::find(const Md5& md5) const {
  auto it = shaders_.find(md5);
  return it == shaders_.end() ? nullptr : &it->second;
}

PackedShader* ShaderPack::findMutable(const Md5& md5) {
  auto it = shaders_.find(md5);
  return it == shaders_.end() ? nullptr : &it->second;
}

// The descriptor bindings a SPIR-V module declares (OpDecorate Binding), by descriptor set, and whether each is a cube
// image (OpVariable -> OpTypePointer -> OpTypeImage with Dim Cube).
struct SpvBinding { uint32_t set, binding; bool cube; };
static void spvBindings(const std::vector<uint32_t>& spv, std::vector<SpvBinding>& out) {
  std::unordered_map<uint32_t, uint32_t> binding, set, varType, pointee, imageDim;
  for (size_t i = 5; i < spv.size();) {
    const uint32_t op = spv[i] & 0xFFFF, len = spv[i] >> 16;
    if (!len || i + len > spv.size()) break;
    if (op == 71 /*OpDecorate*/ && len >= 4) {
      if (spv[i + 2] == 33 /*Binding*/) binding[spv[i + 1]] = spv[i + 3];
      if (spv[i + 2] == 34 /*DescriptorSet*/) set[spv[i + 1]] = spv[i + 3];
    }
    if (op == 25 /*OpTypeImage*/ && len >= 4) imageDim[spv[i + 1]] = spv[i + 3];
    if (op == 32 /*OpTypePointer*/ && len >= 4) pointee[spv[i + 1]] = spv[i + 3];
    if (op == 59 /*OpVariable*/ && len >= 3) varType[spv[i + 2]] = spv[i + 1];
    i += len;
  }
  for (const auto& [id, b] : binding) {
    bool cube = false;
    if (auto v = varType.find(id); v != varType.end())
      if (auto p = pointee.find(v->second); p != pointee.end())
        if (auto d = imageDim.find(p->second); d != imageDim.end()) cube = d->second == 3 /*Cube*/;
    out.push_back({ set.count(id) ? set[id] : 0u, b, cube });
  }
}

bool spvDeclaresInputLocation(const std::vector<uint32_t>& spv, uint32_t location) {
  std::unordered_map<uint32_t, uint32_t> loc;
  std::vector<uint32_t> inputs;
  for (size_t i = 5; i < spv.size();) {
    const uint32_t op = spv[i] & 0xFFFF, len = spv[i] >> 16;
    if (!len || i + len > spv.size()) break;
    if (op == 71 /*OpDecorate*/ && len >= 4 && spv[i + 2] == 30 /*Location*/) loc[spv[i + 1]] = spv[i + 3];
    if (op == 59 /*OpVariable*/ && len >= 4 && spv[i + 3] == 1 /*Input*/) inputs.push_back(spv[i + 2]);
    i += len;
  }
  for (uint32_t id : inputs)
    if (auto l = loc.find(id); l != loc.end() && l->second == location) return true;
  return false;
}

bool spvDepthVariant(const uint32_t* spv, size_t words, std::vector<uint32_t>& out) {
  out.clear();
  if (!spv || words < 5 || spv[0] != 0x07230203u) return false;
  const uint32_t version = spv[1];
  // Pass 1: the Output variables and their pointer types; refuse what the variant cannot drop.
  std::vector<uint32_t> outVars, builtins;
  for (size_t i = 5; i < words;) {
    const uint32_t op = spv[i] & 0xFFFF, len = spv[i] >> 16;
    if (!len || i + len > words) return false;
    if (op == 15 /*OpEntryPoint*/ && spv[i + 1] != 4 /*Fragment*/) return false;
    if (op == 59 /*OpVariable*/ && len >= 4 && spv[i + 3] == 3 /*Output*/) outVars.push_back(spv[i + 2]);
    if (op == 71 /*OpDecorate*/ && len >= 4 && spv[i + 2] == 11 /*BuiltIn*/) builtins.push_back(spv[i + 1]);
    if (op == 72 /*OpMemberDecorate*/ && len >= 5 && spv[i + 3] == 11) builtins.push_back(spv[i + 1]);
    // Side effects other than the colour outputs: storage image writes and atomics, stores through other storage.
    if (op == 99 /*OpImageWrite*/ || (op >= 227 && op <= 242) /*OpAtomic**/ || op == 4421 /*OpAtomicFAddEXT*/) return false;
    i += len;
  }
  if (outVars.empty()) return false;
  for (uint32_t b : builtins)
    if (std::find(outVars.begin(), outVars.end(), b) != outVars.end()) return false;  // FragDepth, SampleMask, ...
  // Pointer types of Output storage (only the outputs and access chains into them use them: no built-in outputs).
  auto isOut = [&](uint32_t id) { return std::find(outVars.begin(), outVars.end(), id) != outVars.end(); };
  out.reserve(words);
  out.insert(out.end(), spv, spv + 5);
  for (size_t i = 5; i < words;) {
    const uint32_t op = spv[i] & 0xFFFF, len = spv[i] >> 16;
    const size_t at = out.size();
    if (op == 71 /*OpDecorate*/ && len >= 3 && isOut(spv[i + 1]) &&
        (spv[i + 2] == 30 /*Location*/ || spv[i + 2] == 31 /*Component*/ || spv[i + 2] == 32 /*Index*/)) {
      i += len;  // interface decorations: not valid on a Private variable
      continue;
    }
    out.insert(out.end(), spv + i, spv + i + len);
    if (op == 15 /*OpEntryPoint*/ && version < 0x00010400u) {
      // Before SPIR-V 1.4 the interface lists only Input and Output variables: the former outputs leave it.
      uint32_t w = 3;  // the name starts at word 3 (execution model, id)
      while (w < len && (spv[i + w] >> 24) != 0) w++;  // the word holding the string's terminating zero byte
      w++;
      std::vector<uint32_t> entry(spv + i, spv + i + w);
      for (uint32_t k = w; k < len; k++)
        if (!isOut(spv[i + k])) entry.push_back(spv[i + k]);
      entry[0] = (uint32_t(entry.size()) << 16) | 15u;
      out.resize(at);
      out.insert(out.end(), entry.begin(), entry.end());
    }
    if (op == 32 /*OpTypePointer*/ && len >= 4 && spv[i + 2] == 3 /*Output*/) out[at + 2] = 6 /*Private*/;
    if (op == 59 /*OpVariable*/ && len >= 4 && spv[i + 3] == 3 /*Output*/) out[at + 3] = 6;
    i += len;
  }
  return true;
}

bool spvColourVariant(const uint32_t* spv, size_t words, std::vector<uint32_t>& out) {
  out.clear();
  if (!spv || words < 5 || spv[0] != 0x07230203u) return false;
  out.reserve(words);
  out.insert(out.end(), spv, spv + 5);
  for (size_t i = 5; i < words;) {
    const uint32_t op = spv[i] & 0xFFFF, len = spv[i] >> 16;
    if (!len || i + len > words) { out.clear(); return false; }
    if (op == 15 /*OpEntryPoint*/ && spv[i + 1] != 4 /*Fragment*/) { out.clear(); return false; }
    // Discards that end the invocation, or a test of whether it was demoted: not removable as an instruction.
    if (op == 252 /*OpKill*/ || op == 4416 /*OpTerminateInvocation*/ || op == 5381 /*OpIsHelperInvocationEXT*/) {
      out.clear();
      return false;
    }
    if (op != 5380 /*OpDemoteToHelperInvocation*/) out.insert(out.end(), spv + i, spv + i + len);
    i += len;
  }
  return true;
}

bool ShaderPack::replace(const Md5& md5, std::vector<uint32_t> spv, const std::vector<PackedShader::Def>& extraDefs) {
  PackedShader* p = findMutable(md5);
  if (!p || spv.empty() || spv[0] != 0x07230203u) return false;
  // What the replacement binds (not what the translation it replaces bound): uniform blocks 0..8, and the 2D or cube
  // textures of sampler registers s (image 16 + s, sampler 32 + s) of the shader's stage set.
  std::vector<SpvBinding> bindings;
  spvBindings(spv, bindings);
  const uint32_t stageSet = p->pixel ? 1 : 0;
  // A pixel replacement may also declare the sun shadow block (9) with the cascades and their sampler (48, 49), as the
  // translated lighting shaders do (docs/SHADER_ABI.md, "Sun shadows"); the runtime binds them by block 9.
  // It may also declare the environment block (10) with the environment cube and its sampler (50, 51; docs/SHADER_ABI.md
  // "Environment").
  uint32_t cbv = 0, tex = 0, cube = 0;
  bool sunImage = false, envImage = false, aoImage = false, puddleImage = false, skinImage = false, skinSampler = false;
  bool skinStore = false;
  for (const auto& [set, b, isCube] : bindings) {
    if (set != stageSet) return false;  // the other stage's set: not this shader's interface
    if (b <= 8) cbv |= 1u << b;
    else if (b == 9 && p->pixel) cbv |= 1u << b;
    else if (b >= 16 && b < 32) { tex |= 1u << (b - 16); if (isCube) cube |= 1u << (b - 16); }
    else if (b >= 32 && b < 48) tex |= 1u << (b - 32);
    else if ((b == 48 || b == 49) && p->pixel) sunImage = true;
    else if (b == 10 && p->pixel) cbv |= 1u << b;
    else if ((b == 50 || b == 51) && p->pixel) envImage = true;
    else if (b == 11 && p->pixel) cbv |= 1u << b;
    else if ((b == 52 || b == 53 || b == 54 || b == 55) && p->pixel) aoImage = true;
    else if (b == 12 && p->pixel) cbv |= 1u << b;
    else if ((b == 56 || b == 57 || b == 58) && p->pixel) puddleImage = true;
    else if (b == 59 && p->pixel) skinImage = true;
    else if (b == 60 && p->pixel) skinSampler = true;
    else if (b == 13 && p->pixel) cbv |= 1u << b;
    else if ((b == 61 || b == 62) && p->pixel) skinStore = true;
    else return false;
  }
  if (aoImage != ((cbv >> 11) & 1u)) return false;  // the AO block and its image come together
  // The cascades come with the sun block (9) or the wet block (12, the rain cover's layer); the puddle map with the wet block.
  if (sunImage != (((cbv >> 9) & 1u) || ((cbv >> 12) & 1u))) return false;
  if (puddleImage != ((cbv >> 12) & 1u)) return false;
  if (envImage != ((cbv >> 10) & 1u)) return false;  // the block and the cube come together
  if (skinImage != skinSampler) return false;  // the detail map comes with its sampler
  if (skinStore != ((cbv >> 13) & 1u)) return false;  // the skin buffer's images come with its block
  p->cbvMask = cbv;
  p->skinDetail = skinImage;
  p->skinStore = skinStore;
  p->textureMask = tex;
  p->textures.clear();
  for (uint32_t s = 0; s < 16; s++)
    if (tex & (1u << s)) p->textures.push_back({ uint8_t(s), uint8_t((cube >> s) & 1u ? 9 /*cube*/ : 5 /*2D*/) });
  replacements_.push_back(std::move(spv));
  p->spv = replacements_.back().data();
  p->spvBytes = uint32_t(replacements_.back().size() * 4);
  p->constAll = !spvConstantRuns(p->spv, p->spvBytes / 4, stageSet, p->pixel ? 224 : 256, p->constRuns);
  spvTraits(*p);
  for (const PackedShader::Def& d : extraDefs) {
    bool found = false;
    for (PackedShader::Def& own : p->defs)
      if (own.reg == d.reg) { own = d; found = true; }
    if (!found) p->defs.push_back(d);
  }
  return true;
}

std::vector<Md5> sdpShaderMd5s(const std::string& shaderDir, const std::string& name, const std::vector<int>& packages) {
  std::vector<Md5> out;
  WIN32_FIND_DATAA fd;
  HANDLE h = FindFirstFileA((shaderDir + "\\shaderpackage*.sdp").c_str(), &fd);
  if (h == INVALID_HANDLE_VALUE) return out;
  do {
    int pkg = -1;
    if (sscanf_s(fd.cFileName, "shaderpackage%d.sdp", &pkg) != 1) continue;
    if (!packages.empty() && std::find(packages.begin(), packages.end(), pkg) == packages.end()) continue;
    FILE* f = nullptr;
    if (fopen_s(&f, (shaderDir + "\\" + fd.cFileName).c_str(), "rb") || !f) continue;
    std::vector<uint8_t> b;
    fseek(f, 0, SEEK_END);
    b.resize(size_t(ftell(f)));
    fseek(f, 0, SEEK_SET);
    const bool read = fread(b.data(), 1, b.size(), f) == b.size();
    fclose(f);
    if (!read || b.size() < 12) continue;
    uint32_t count = 0;
    memcpy(&count, b.data() + 4, 4);
    size_t o = 12;
    for (uint32_t i = 0; i < count && o + 260 <= b.size(); i++) {
      uint32_t size = 0;
      memcpy(&size, b.data() + o + 256, 4);
      if (o + 260 + size > b.size()) break;
      if (strncmp(reinterpret_cast<const char*>(b.data() + o), name.c_str(), 256) == 0) {
        Md5 m;
        if (md5Of(b.data() + o + 260, size, m) && std::find(out.begin(), out.end(), m) == out.end()) out.push_back(m);
      }
      o += 260 + size;
    }
  } while (FindNextFileA(h, &fd));
  FindClose(h);
  return out;
}

bool md5Of(const void* data, size_t size, Md5& out) {
  BCRYPT_ALG_HANDLE alg = nullptr;
  if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_MD5_ALGORITHM, nullptr, 0) != 0) return false;
  NTSTATUS st = BCryptHash(alg, nullptr, 0, static_cast<PUCHAR>(const_cast<void*>(data)), ULONG(size), out.b.data(),
                           ULONG(out.b.size()));
  BCryptCloseAlgorithmProvider(alg, 0);
  return st == 0;
}

}  // namespace fnvvk
