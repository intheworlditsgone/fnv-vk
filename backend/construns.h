#pragma once
// The float-constant registers a SPIR-V module reads from its constant block (uniform binding 0 of its stage's set),
// as a few runs of consecutive registers, so the runtime copies only those into the block (docs/PERFORMANCE.md).
//
// Only reads through OpAccessChain / OpInBoundsAccessChain on the block variable with two constant indices (the
// array member, then the register) are understood. Any other use of the block (a dynamic register index such as
// relative addressing c[a0.x + n], a chain to the whole array, a load or copy of the block, a call taking it, a block
// type other than one array member) makes the answer "every register": spvConstantRuns returns false. An id of the
// block appearing as a literal operand inside a function is also taken as a use (never fewer registers, at worst all).
// Header-only: also used by tools/shaderdiff to check the runs against D3D9 (SHADERDIFF_POISON).

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fnvvk {

struct ConstRun {
  uint16_t first = 0, count = 0;
};

// Registers in [0, regCount) read by `spv` from (set, binding 0). False: all of them (or the module is malformed).
// True with no runs: the block is not read (or not declared).
inline bool spvConstantRuns(const uint32_t* spv, size_t words, uint32_t set, uint32_t regCount,
                            std::vector<ConstRun>& out) {
  out.clear();
  if (words < 5 || spv[0] != 0x07230203u) return false;
  std::unordered_map<uint32_t, uint32_t> binding, descSet, constant, varType, pointee, memberCount, arrayOf;
  std::unordered_set<uint32_t> arrays;
  // Pass 1: declarations.
  for (size_t i = 5; i < words;) {
    const uint32_t op = spv[i] & 0xFFFF, len = spv[i] >> 16;
    if (!len || i + len > words) return false;
    const uint32_t* w = spv + i;
    switch (op) {
      case 71:  // OpDecorate
        if (len >= 4 && w[2] == 33) binding[w[1]] = w[3];
        if (len >= 4 && w[2] == 34) descSet[w[1]] = w[3];
        break;
      case 43:  // OpConstant (32-bit integer or float: only integer-typed ones index; a float id never does)
        if (len == 4) constant[w[2]] = w[3];
        break;
      case 28:  // OpTypeArray
        if (len >= 4) arrays.insert(w[1]);
        break;
      case 30:  // OpTypeStruct
        memberCount[w[1]] = len - 2;
        if (len >= 3) arrayOf[w[1]] = w[2];
        break;
      case 32:  // OpTypePointer
        if (len >= 4) pointee[w[1]] = w[3];
        break;
      case 59:  // OpVariable
        if (len >= 4) varType[w[2]] = w[1];
        break;
      default:
        break;
    }
    i += len;
  }
  uint32_t block = 0;
  for (const auto& [id, b] : binding) {
    if (b != 0 || !varType.count(id)) continue;
    auto s = descSet.find(id);
    if ((s == descSet.end() ? 0u : s->second) != set) continue;
    block = id;
  }
  if (!block) return true;  // no constant block: nothing read
  {
    auto p = pointee.find(varType[block]);
    if (p == pointee.end()) return false;
    auto m = memberCount.find(p->second);
    if (m == memberCount.end() || m->second != 1 || !arrays.count(arrayOf[p->second])) return false;
  }
  // Pass 2: uses inside functions.
  std::vector<bool> used(regCount, false);
  bool inFunction = false;
  for (size_t i = 5; i < words;) {
    const uint32_t op = spv[i] & 0xFFFF, len = spv[i] >> 16;
    const uint32_t* w = spv + i;
    i += len;
    if (op == 54) inFunction = true;        // OpFunction
    if (op == 56) { inFunction = false; continue; }  // OpFunctionEnd
    if (!inFunction) continue;
    if ((op == 65 || op == 66) && len >= 4 && w[3] == block) {  // OpAccessChain / OpInBoundsAccessChain on the block
      if (len < 6) return false;                                  // the whole array
      auto m = constant.find(w[4]), r = constant.find(w[5]);
      if (m == constant.end() || m->second != 0 || r == constant.end()) return false;  // dynamic index
      if (r->second >= regCount) continue;  // out of the block: reads nothing the runtime writes
      used[r->second] = true;
      for (uint32_t k = 6; k < len; k++)
        if (w[k] == block) return false;
      continue;
    }
    for (uint32_t k = 1; k < len; k++)
      if (w[k] == block) return false;  // any other use
  }
  // Runs, joining gaps of up to 2 registers (cheaper to copy than another run); at most 8 runs (else the closest join).
  for (uint32_t r = 0; r < regCount; r++) {
    if (!used[r]) continue;
    if (!out.empty() && r - (out.back().first + out.back().count) <= 2) out.back().count = uint16_t(r + 1 - out.back().first);
    else out.push_back({ uint16_t(r), 1 });
  }
  while (out.size() > 8) {
    size_t best = 0;
    uint32_t gap = ~0u;
    for (size_t k = 0; k + 1 < out.size(); k++) {
      const uint32_t g = out[k + 1].first - (out[k].first + out[k].count);
      if (g < gap) { gap = g; best = k; }
    }
    out[best].count = uint16_t(out[best + 1].first + out[best + 1].count - out[best].first);
    out.erase(out.begin() + best + 1);
  }
  return true;
}

}  // namespace fnvvk
