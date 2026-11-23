#pragma once
// Upscaling (upscale.cpp): settings ([Upscaling] in fnv_vk.ini, the menu's Upscaling panel), the game's frame points
// the backend needs (Backend::beginScaledScene ... upscaleScene) and the camera jitter.
#include <cstdint>

namespace fnvvk {
class Backend;
}

namespace fnvvk::upscale {

struct Settings {
  enum Upscaler : int { Off = 0, Fsr3 = 1 } upscaler = Off;
  enum QualityMode : int { NativeAa = 0, Quality = 1, Balanced = 2, Performance = 3, UltraPerformance = 4 } quality = Quality;
  bool sharpen = true;
  float sharpness = 0.5f;
};

void load();               // reads [Upscaling]
Settings& settings();
void apply(Backend* be);   // pushes the settings to the backend (loads fnv_vk_fidelityfx.dll when first needed)
bool save();               // writes [Upscaling]
bool install();            // hooks (native mode)

// Frame points, called from effects.cpp's hooks of the same functions.
void afterDrawScene();                                      // TESMain::DrawWorld_DrawScene returned (world pass done)
void beforeImageSpace(void* source, void* destination);     // BSShaderManager::RenderImageSpaceEffects is about to run

}  // namespace fnvvk::upscale
