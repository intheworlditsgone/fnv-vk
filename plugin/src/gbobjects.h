#pragma once
#include <windows.h>

#include <cstdint>

// Vulkan-backed resource objects with the engine's IDirect3D*9 resource ABI (native mode).
struct IDirect3DDevice9;
namespace fnvvk {
class Backend;
}
namespace fnvvk::gbobjects {
bool installBuffers();            // at plugin load, native mode only: buffer creation sites create these objects
void onBackendReady(Backend* be); // registers and uploads the objects created before the backend existed
void logSummary();
bool installPrograms();           // at plugin load, native mode only: shader creation sites create shader objects
// The device's Create* methods (gbdevice.cpp) and gbstate's thunks construct through these.
HRESULT createVertexShader(IDirect3DDevice9* d, const DWORD* code, void** out);
HRESULT createPixelShader(IDirect3DDevice9* d, const DWORD* code, void** out);
HRESULT createVertexDeclaration(IDirect3DDevice9* d, const void* elements, void** out);
HRESULT createVertexBuffer(IDirect3DDevice9* d, uint32_t len, uint32_t usage, uint32_t fvf, uint32_t pool, void** out);
HRESULT createIndexBuffer(IDirect3DDevice9* d, uint32_t len, uint32_t usage, uint32_t fmt, uint32_t pool, void** out);
// `mov r32,[r32+disp]; ...; call r32` -> `...; call rel32 target` (see gbobjects.cpp).
bool patchVirtualCall(uint32_t site, uint32_t disp, void* target, const char* where);
}  // namespace fnvvk::gbobjects
