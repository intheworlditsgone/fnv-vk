// Vertex and index buffer contents from the Gamebryo layer (replaces the observer's Lock/Unlock feed).
//
// Vertex buffers (IDA): every write goes LockVB 0x00E8BFF0 (records {vb, offset, size, flags}; Gamebryo holds its VB
// critical section until the unlock) -> data packed into m_pvTempBuffer -> memcpy(m_pvLockedBuffer, m_pvTempBuffer,
// m_uiLockedBufferSize) -> Unlock. That memcpy is a `call memcpy` (E8 rel32) at three sites: UnlockVB 0x00E8C0DF and
// the inlined copies in PackSkinnedVB 0x00E8CD14 and PackTransformedVB 0x00E8C7D6. They are redirected to vbCommit,
// which performs the same memcpy and captures the bytes.
// Index buffers: NiDX9IndexBufferManager::PackBuffer 0x00E8F200 creates (2*uiMaxLength bytes, when the old buffer
// does not fit) and fills the buffer with pusTriList[0..uiLength) at offset 0.
// During the shadow phase the captures are checked against what D3D9 receives (observer); see logSummary.
#include "gbbuffers.h"

#include <windows.h>

#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "backend.h"
#include "common.h"
#include "shadow.h"

namespace fnvvk::gbbuffers {

namespace {

struct Pending { void* vb = nullptr; uint32_t offset = 0, size = 0, flags = 0; };
thread_local Pending t_lock;

struct Key {
  void* buffer;
  uint32_t offset, size;
  bool operator==(const Key& o) const { return buffer == o.buffer && offset == o.offset && size == o.size; }
};
struct KeyHash {
  size_t operator()(const Key& k) const { return std::hash<uintptr_t>()(uintptr_t(k.buffer) ^ (uintptr_t(k.offset) << 7) ^ k.size); }
};
std::mutex g_m;
// Pending halves of each comparison: whichever side (game-side capture or D3D9 write) arrives first waits here.
std::unordered_map<Key, uint64_t, KeyHash> g_captures, g_observed;

struct Counters {
  std::atomic<uint64_t> created{ 0 }, vbCommits{ 0 }, vbNoLock{ 0 }, ibPacks{ 0 }, ibCreated{ 0 }, checked{ 0 }, match{ 0 }, mismatch{ 0 },
    notCaptured{ 0 };
} g_c;

void compare(uint64_t a, uint64_t b) { (a == b ? g_c.match : g_c.mismatch)++; }
bool g_verify = true;
void capture(void* buffer, uint32_t offset, const void* data, uint32_t size) {
  if (!g_verify) return;
  const uint64_t h = hashBytes(data, size);
  std::lock_guard<std::mutex> lock(g_m);
  auto it = g_observed.find({ buffer, offset, size });
  if (it != g_observed.end()) {
    compare(h, it->second);
    g_observed.erase(it);
  } else {
    g_captures[{ buffer, offset, size }] = h;
  }
}

using LockVBFn = void*(__fastcall*)(void*, void*, void*, uint32_t, uint32_t, uint32_t);
LockVBFn oLockVB;
void* __fastcall hkLockVB(void* self, void* edx, void* vb, uint32_t offset, uint32_t size, uint32_t flags) {
  void* p = oLockVB(self, edx, vb, offset, size, flags);
  t_lock = p ? Pending{ vb, offset, size, flags } : Pending{};
  return p;
}

// Replaces `memcpy(m_pvLockedBuffer, m_pvTempBuffer, m_uiLockedBufferSize)` at the three commit sites.
void* __cdecl vbCommit(void* dst, const void* src, size_t n) {
  memcpy(dst, src, n);
  if (t_lock.vb && n == t_lock.size) {
    capture(t_lock.vb, t_lock.offset, src, uint32_t(n));
    g_c.vbCommits++;
  } else {
    g_c.vbNoLock++;
  }
  t_lock = Pending{};
  return dst;
}

// --- creation (the only callers of CreateVertexBuffer / CreateIndexBuffer, IDA) --------------------------------------
// NiVBSet::CreateBlock 0x00E94580 -> NiVBBlock {+0x08 m_pkVB, +0x0C m_uiSize}; NiUnsharedGeometryGroup::CreateChip
// 0x00E94E30 -> NiVBChip {+0x08 m_pkVB, +0x14 m_uiSize}; NiVBDynamicSet::AllocateBufferSpace 0x00E987C0 replaces
// {+0x10 m_pkVB} (size +0x08 m_uiBlockSize) when the request outgrows it; NiDX9IndexBufferManager::CreateIndexBuffer
// 0x00E8F190 (length, usage, format, pool, desc); PackBuffer creates 2*uiMaxLength bytes when the old IB does not fit.
thread_local int t_inCreator = 0;
struct CreatorScope { CreatorScope() { t_inCreator++; } ~CreatorScope() { t_inCreator--; } };
// Buffers created before the backend exists are queued and registered when it starts (flushCreations).
struct Created { void* buffer; uint32_t size; bool index, index32; };
std::mutex g_pendingMutex;
std::vector<Created> g_pending;
bool g_native = false;  // buffers are gbobjects (they register and upload themselves)
void created(void* buffer, uint32_t size, bool index, bool index32) {
  if (!buffer || g_native) return;
  g_c.created++;
  if (Backend* be = shadow::backendIfReady()) {
    be->createBuffer(BufKey(buffer), size, index, index32);
  } else {
    std::lock_guard<std::mutex> lock(g_pendingMutex);
    g_pending.push_back({ buffer, size, index, index32 });
  }
}

using CreateBlockFn = void*(__fastcall*)(void*, void*, uint32_t);
CreateBlockFn oCreateBlock;
void* __fastcall hkCreateBlock(void* self, void* edx, uint32_t size) {
  void* block;
  {
    CreatorScope scope;
    block = oCreateBlock(self, edx, size);
  }
  if (block) created(*reinterpret_cast<void**>(static_cast<char*>(block) + 0x08), *reinterpret_cast<uint32_t*>(static_cast<char*>(block) + 0x0C), false, false);
  return block;
}
using CreateChipFn = void*(__fastcall*)(void*, void*, void*, uint32_t);
CreateChipFn oCreateChip;
void* __fastcall hkCreateChip(void* self, void* edx, void* buffData, uint32_t stream) {
  void* chip;
  {
    CreatorScope scope;
    chip = oCreateChip(self, edx, buffData, stream);
  }
  if (chip) created(*reinterpret_cast<void**>(static_cast<char*>(chip) + 0x08), *reinterpret_cast<uint32_t*>(static_cast<char*>(chip) + 0x14), false, false);
  return chip;
}
using AllocDynFn = void*(__fastcall*)(void*, void*, uint32_t, uint32_t*, uint32_t*, char, char);
AllocDynFn oAllocDyn;
void* __fastcall hkAllocDyn(void* self, void* edx, uint32_t size, uint32_t* offset, uint32_t* flags, char discard, char swvp) {
  void* before = *reinterpret_cast<void**>(static_cast<char*>(self) + 0x10);
  void* vb;
  {
    CreatorScope scope;
    vb = oAllocDyn(self, edx, size, offset, flags, discard, swvp);
  }
  if (vb && vb != before) created(vb, *reinterpret_cast<uint32_t*>(static_cast<char*>(self) + 0x08), false, false);
  return vb;
}
using CreateIBFn = void*(__fastcall*)(void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, void*);
CreateIBFn oCreateIB;
void* __fastcall hkCreateIB(void* self, void* edx, uint32_t length, uint32_t usage, uint32_t format, uint32_t pool, void* desc) {
  void* ib;
  {
    CreatorScope scope;
    ib = oCreateIB(self, edx, length, usage, format, pool, desc);
  }
  created(ib, length, true, format == 102 /* D3DFMT_INDEX32 */);
  return ib;
}

using PackBufferFn = void*(__fastcall*)(void*, void*, const uint16_t*, uint32_t, uint32_t, void*, uint32_t*, uint32_t, uint32_t);
PackBufferFn oPackBuffer;
void* __fastcall hkPackBuffer(void* self, void* edx, const uint16_t* list, uint32_t length, uint32_t maxLength, void* oldIB,
                              uint32_t* ibSize, uint32_t pool, uint32_t usage) {
  void* ib;
  {
    CreatorScope scope;
    ib = oPackBuffer(self, edx, list, length, maxLength, oldIB, ibSize, pool, usage);
  }
  if (ib && ib != oldIB && !g_native) {
    // Created and filled inside PackBuffer: the D3D9 write reached the backend before this registration, so the
    // contents are written again here, from the game's own index list.
    created(ib, maxLength * 2, true, false);  // D3DFMT_INDEX16, 2 * uiMaxLength bytes
    if (list && length)
      if (Backend* be = shadow::backendIfReady()) be->writeBuffer(BufKey(ib), 0, list, length * 2, 0);
  }
  if (ib && list && length) {
    if (ib != oldIB) g_c.ibCreated++;
    capture(ib, 0, list, length * 2);
    g_c.ibPacks++;
  }
  return ib;
}

bool patchCall(uint32_t site, void* target) {
  uint8_t* p = reinterpret_cast<uint8_t*>(uintptr_t(site));
  if (p[0] != 0xE8) {
    logf("buffer commit patch REFUSED at %08X: opcode %02X, expected call rel32", site, p[0]);
    return false;
  }
  int32_t rel = int32_t(uintptr_t(target) - (uintptr_t(p) + 5));
  DWORD old = 0;
  VirtualProtect(p, 5, PAGE_EXECUTE_READWRITE, &old);
  memcpy(p + 1, &rel, 4);
  VirtualProtect(p, 5, old, &old);
  FlushInstructionCache(GetCurrentProcess(), p, 5);
  return true;
}

}  // namespace

uint64_t hashBytes(const void* data, size_t size) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < size; i++) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

void setNative(bool on) { g_native = on; }
void setVerify(bool on) { g_verify = on; }
void verifyWrite(void* buffer, uint32_t offset, const void* data, uint32_t size) {
  if (!g_verify) return;
  g_c.checked++;
  const uint64_t h = hashBytes(data, size);
  std::lock_guard<std::mutex> lock(g_m);
  auto it = g_captures.find({ buffer, offset, size });
  if (it != g_captures.end()) {
    compare(it->second, h);
    g_captures.erase(it);
  } else {
    g_observed[{ buffer, offset, size }] = h;
  }
}

bool inCreator() { return t_inCreator > 0; }

void flushCreations() {
  Backend* be = shadow::backendIfReady();
  if (!be) return;
  std::lock_guard<std::mutex> lock(g_pendingMutex);
  for (const Created& c : g_pending) be->createBuffer(BufKey(c.buffer), c.size, c.index, c.index32);
  logf("game-side buffers created before the backend started: %zu, registered", g_pending.size());
  g_pending.clear();
}

bool installHooks() {
  hook(0x00E94580, &hkCreateBlock, &oCreateBlock, "NiVBSet::CreateBlock");
  hook(0x00E94E30, &hkCreateChip, &oCreateChip, "NiUnsharedGeometryGroup::CreateChip");
  hook(0x00E987C0, &hkAllocDyn, &oAllocDyn, "NiVBDynamicSet::AllocateBufferSpace");
  hook(0x00E8F190, &hkCreateIB, &oCreateIB, "NiDX9IndexBufferManager::CreateIndexBuffer");
  bool ok = hook(0x00E8BFF0, &hkLockVB, &oLockVB, "NiDX9VertexBufferManager::LockVB");
  ok &= hook(0x00E8F200, &hkPackBuffer, &oPackBuffer, "NiDX9IndexBufferManager::PackBuffer");
  for (uint32_t site : { 0x00E8C0DFu, 0x00E8CD14u, 0x00E8C7D6u }) ok &= patchCall(site, reinterpret_cast<void*>(&vbCommit));
  logf("vertex-buffer commit sites redirected: %s", ok ? "all 3" : "NOT ALL");
  return ok;
}

void logSummary() {
  size_t waitingObserved, waitingCaptured;
  {
    std::lock_guard<std::mutex> lock(g_m);
    waitingObserved = g_observed.size();
    waitingCaptured = g_captures.size();
    g_c.notCaptured = waitingObserved;
  }
  logf("game-side buffer check backlog: %zu D3D9 writes without a game-side capture, %zu captures not seen by D3D9",
       waitingObserved, waitingCaptured);
  logf("game-side buffer creations: %llu", g_c.created.load());
  logf("game-side buffers: VB commits %llu (without a recorded lock %llu), IB packs %llu (new %llu) | D3D9 writes checked "
       "%llu: match %llu, MISMATCH %llu, not captured game-side %llu",
       g_c.vbCommits.load(), g_c.vbNoLock.load(), g_c.ibPacks.load(), g_c.ibCreated.load(), g_c.checked.load(),
       g_c.match.load(), g_c.mismatch.load(), g_c.notCaptured.load());
}

}  // namespace fnvvk::gbbuffers
