#pragma once
// Vulkan presents into the game window instead of D3D9 (present.cpp).
namespace fnvvk::present {
bool install(bool on, bool forceEx);  // at plugin load, before the renderer is created
bool enabled();
void* window();         // the D3D9 device window (HWND), once the renderer initialised
bool vsync();
bool forceD3D9Ex();  // native mode: NiDX9Renderer::Create gets useD3D9ex = true
}  // namespace fnvvk::present
