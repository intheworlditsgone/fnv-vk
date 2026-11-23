#pragma once
// Captures D3D9 device inputs for comparison with the Gamebryo tracker and Vulkan output.
#include <cstdint>

struct IDirect3DDevice9;

namespace fnvvk {
class Backend;
namespace observe {

// Installs the device hooks (once). `dev` is NiDX9Renderer::m_pkD3DDevice9 (renderer +0x288).
// hooks = false (native mode): no D3D9 hooks; only the tracker-issued draw path and the binding maps are used.
bool install(IDirect3DDevice9* dev, Backend* backend, bool drawsEnabled, bool hooks = true);
void logSummary();
// Diagnostics: a frame comparison with a large difference arms a one-frame per-draw probe at its worst pixel.
void enableProbe(bool on);
void onFrameStart();  // call once per frame, before its first draw
// Issues one draw to the backend from the game-side state tracker (called by the draw-site thunks after D3D9 drew).
// `upData`/`upStride`: user-pointer vertices (DrawPrimitiveUP).
void drawFromTracker(uint32_t d3dPrimitiveType, uint32_t primitiveCount, bool indexed, int32_t baseVertex, uint32_t start,
                     const void* upData = nullptr, uint32_t upStride = 0);

// Every D3D9 texture the backend has a texture for (registered by the texture path at upload, removed at destroy).
void registerTexture(void* d3dTexture, uintptr_t texKey);
void unregisterTexture(void* d3dTexture);
// Render-target surfaces (RendererData::m_pkSurface) -> RtKey, maintained by the render-target lifetime hooks.
void registerSurface(void* surface, uintptr_t rtKey);
// Each returns 0 for an object the backend does not know.
uintptr_t rtKeyForSurface(void* surface);
uintptr_t rtKeyForRtTexture(void* texture);  // IDirect3DTexture9* render texture -> RtKey
uintptr_t texKeyForTexture(void* texture);   // IDirect3DBaseTexture9* file/pixel-data texture -> TexKey
void unregisterRtKey(uintptr_t rtKey);
// Render-target textures (NiDX9TextureBufferData::m_pkRenderTexture) -> RtKey, from the Gamebryo object.
void registerRtTexture(void* d3dTexture, uintptr_t rtKey);

// Compares the backend's back buffer with D3D9's (both read back; slow, verification only) and writes both images
// as BMP next to the log when `dumpPrefix` is non-null.
void compareFrame(uintptr_t backBufferKey, const char* dumpPrefix);

}  // namespace observe
}  // namespace fnvvk
