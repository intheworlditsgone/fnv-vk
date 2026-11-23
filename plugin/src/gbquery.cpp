// BSOcclusionQuery on Vulkan occlusion queries: exact ports of its six functions (IDA: the only code that touches its
// pOcclusionQuery field). pOcclusionQuery (+0x00) holds the backend query id instead of an IDirect3DQuery9*.
// Users: decals (BGSDecalManager), sun occlusion (BSShaderAccumulator), VATS geometry occlusion.
#include <windows.h>

#include <cstdint>

#include "backend.h"
#include "common.h"
#include "shadow.h"

namespace fnvvk::gbquery {
namespace {

bool& bQueryInProgress = *reinterpret_cast<bool*>(0x012030A4);  // BSOcclusionQuery::bQueryInProgress
uint32_t& idOf(void* self) { return *static_cast<uint32_t*>(self); }

// BSOcclusionQuery::BSOcclusionQuery 0x00C4ECF0
void* __fastcall Ctor(void* self, void*) {
  Backend* be = shadow::backendIfReady();
  idOf(self) = be ? be->createQuery() : 0;
  return self;
}
// ~BSOcclusionQuery 0x00C4ED40
void __fastcall Dtor(void* self, void*) {
  if (idOf(self))
    if (Backend* be = shadow::backendIfReady()) be->destroyQuery(idOf(self));
}
// Begin 0x00C4ED70
char __fastcall Begin(void* self, void*) {
  if (bQueryInProgress) return 0;
  bQueryInProgress = true;
  if (Backend* be = shadow::backendIfReady()) {
    // Queries constructed before the backend existed get their id on first use.
    if (!idOf(self)) idOf(self) = be->createQuery();
    be->beginQuery(idOf(self));
  }
  return 1;
}
// End 0x00C4EDB0
void __fastcall End(void* self, void*) {
  if (!bQueryInProgress) return;
  bQueryInProgress = false;
  if (idOf(self))
    if (Backend* be = shadow::backendIfReady()) be->endQuery(idOf(self));
}
// GetResult 0x00C4EDF0 / GetResultNoWait 0x00C4EE40: D3D9 GetData == S_OK means the DWORD result was written.
bool result(void* self, void* data) {
  uint32_t samples = 0;
  Backend* be = shadow::backendIfReady();
  if (!idOf(self) || !be) {
    if (data) *static_cast<uint32_t*>(data) = 0;
    return true;
  }
  if (!be->queryResult(idOf(self), samples)) return false;
  if (data) *static_cast<uint32_t*>(data) = samples;
  return true;
}
bool __fastcall GetResult(void* self, void*, void* data, uint32_t) { return result(self, data); }
bool __fastcall GetResultNoWait(void* self, void*, void* data, bool) { return result(self, data); }

}  // namespace

bool installHooks() {
  struct H { uintptr_t addr; void* fn; const char* name; };
  const H hooks[] = {
    { 0x00C4ECF0, reinterpret_cast<void*>(&Ctor), "BSOcclusionQuery::BSOcclusionQuery (port)" },
    { 0x00C4ED40, reinterpret_cast<void*>(&Dtor), "~BSOcclusionQuery (port)" },
    { 0x00C4ED70, reinterpret_cast<void*>(&Begin), "BSOcclusionQuery::Begin (port)" },
    { 0x00C4EDB0, reinterpret_cast<void*>(&End), "BSOcclusionQuery::End (port)" },
    { 0x00C4EDF0, reinterpret_cast<void*>(&GetResult), "BSOcclusionQuery::GetResult (port)" },
    { 0x00C4EE40, reinterpret_cast<void*>(&GetResultNoWait), "BSOcclusionQuery::GetResultNoWait (port)" },
  };
  bool ok = true;
  for (const H& h : hooks) {
    void* orig = nullptr;
    ok &= hookFunction(h.addr, h.fn, &orig, h.name);
  }
  return ok;
}

}  // namespace fnvvk::gbquery
