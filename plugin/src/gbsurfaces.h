#pragma once
// Vulkan-backed textures and surfaces with the engine's IDirect3D*9 ABI (native mode; gbsurfaces.cpp).
#include <windows.h>

#include <d3d9.h>

#include <cstdint>

namespace fnvvk::gbsurfaces {
bool install();  // at plugin load, native mode only
// For the native D3DX placeholders (textures.cpp).
HRESULT createTexture(IDirect3DDevice9* d, uint32_t w, uint32_t h, uint32_t levels, uint32_t usage, uint32_t format,
                      uint32_t pool, void** out);
HRESULT createCubeTexture(IDirect3DDevice9* d, uint32_t edge, uint32_t levels, uint32_t usage, uint32_t format,
                          uint32_t pool, void** out);
// For the plugin device (gbdevice.cpp): standalone surfaces (render targets, depth-stencil, offscreen plain, the implicit
// back buffer) and the copies between plugin objects.
HRESULT createSurface(IDirect3DDevice9* d, uint32_t w, uint32_t h, uint32_t format, uint32_t ms, uint32_t quality,
                      uint32_t usage, uint32_t pool, void** out);
HRESULT updateSurface(void* src, const RECT* srcRect, void* dst, const POINT* dstPoint);
HRESULT updateTexture(void* src, void* dst);
void logSummary();
}  // namespace fnvvk::gbsurfaces
