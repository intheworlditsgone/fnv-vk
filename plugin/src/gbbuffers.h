#pragma once
// Vertex/index buffer contents captured from the Gamebryo layer (gbbuffers.cpp).
#include <cstddef>
#include <cstdint>

namespace fnvvk::gbbuffers {

bool installHooks();
// True while a Gamebryo buffer creator runs (its device creation is registered game-side).
bool inCreator();
// Registers the buffers created before the backend existed (call once it is ready).
void flushCreations();
uint64_t hashBytes(const void* data, size_t size);
// Shadow-phase check: a write D3D9 received must equal the last game-side capture for the same range.
// Off in native mode: no hashes of game-side writes (nothing on the D3D9 side to compare with).
void setVerify(bool on);
// Native mode: buffers are Vulkan-backed objects (gbobjects); creation registration here is off.
void setNative(bool on);
void verifyWrite(void* buffer, uint32_t offset, const void* data, uint32_t size);
void logSummary();

}  // namespace fnvvk::gbbuffers
