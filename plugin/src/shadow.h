#pragma once
// Mirrors Gamebryo render targets and frame calls to Vulkan. D3D9 objects supply the reference for comparison.
#include <cstdint>

namespace fnvvk {
class Backend;
}

namespace fnvvk::shadow {

Backend* backendIfReady();      // the Vulkan backend once initialised, else nullptr
uintptr_t defaultTargetKey();  // buffer 0 of the renderer's default render-target group (the backbuffer)

bool installHooks();                    // lifetime hooks; call from NVSEPlugin_Load (before the renderer exists)
void onRendererCreated(void* renderer); // after NiDX9Renderer::Create returned (not under the loader lock)
void onBeginGroup(void* renderer, void* group);   // before the original BeginUsingRenderTargetGroup
void onEndGroup(void* renderer);                  // after the original EndUsingRenderTargetGroup
// Before DisplayFrame presents (the D3D9 back buffer still holds the frame): frame comparison.
void onBeforeDisplayFrame(void* renderer);
void onDisplayFrame(void* renderer);              // after the original DisplayFrame
// After a leaf draw returned (Gamebryo state cache = the state its draws used). kind: 0 tristrips and the
// other one-pointer leaves, 1 TriShape2, 2 partition, 3 mesh, 4 strip.
void onLeafDraw(void* renderer, int kind);
void logSummary(uint32_t frame);

}  // namespace fnvvk::shadow
