#pragma once
// Screenshots (screenshot.cpp): the game's finished frame as the player sees it, without the menu, saved by a worker
// thread as PNG, JPEG or BMP; with HDR output also as an HDR JPEG XR. Settings: [Screenshot] in fnv_vk.ini and the
// menu's Screenshots panel. The hotkey is the menu's (overlay.cpp), which calls request().
#include <cstddef>
#include <cstdint>
#include <string>

#include "imagefile.h"

namespace fnvvk {
class Backend;
}

namespace fnvvk::screenshot {

struct Settings {
  imagefile::Format format = imagefile::Format::Png;
  int quality = 95;       // JPEG and JPEG XR, 1..100 (JPEG XR: 100 is lossless)
  bool hdrImage = true;   // with HDR output: also the HDR image (.jxr, scRGB) as the display shows it
  bool notify = true;     // a HUD message when the file is written
  std::string folder;     // UTF-8; empty: defaultFolder()
};

void load();  // [Screenshot] from fnv_vk.ini
Settings& settings();
bool save();  // writes [Screenshot] (format, quality, HDR image, notify, folder)
std::string defaultFolder();  // Pictures\Fallout New Vegas (UTF-8)
std::string folder();         // the folder in use
void openFolder();            // in Explorer (created first)

void request();  // take one at the end of this frame (any thread)
// At NiDX9Renderer::DisplayFrame, before the backend presents (main thread): takes a requested screenshot of `onscreen`
// (buffer 0 of the default render-target group, width x height) and shows the messages of finished saves.
void frame(Backend* be, uintptr_t onscreen, uint32_t width, uint32_t height);
// IDirect3DDevice9::SetGammaRamp (Renderer::UpdateGamma): SDR output applies it at present, so screenshots do too.
void setGammaRamp(const uint16_t red[256], const uint16_t green[256], const uint16_t blue[256]);

std::string status();  // the last result, for the menu
int pending();         // screenshots taken but not yet written

// B, G, R, A texels -> B, G, R through a per-channel 8-bit table (the gamma ramp: lut[0] red, [1] green, [2] blue).
// The same table the backend's gamma present (backend/builtin/gamma.frag) reads, so the file matches the screen.
inline void toBgr(const uint8_t* bgra, size_t texels, const uint8_t lut[3][256], uint8_t* bgr) {
  for (size_t i = 0; i < texels; i++) {
    bgr[i * 3 + 0] = lut[2][bgra[i * 4 + 0]];
    bgr[i * 3 + 1] = lut[1][bgra[i * 4 + 1]];
    bgr[i * 3 + 2] = lut[0][bgra[i * 4 + 2]];
  }
}

}  // namespace fnvvk::screenshot
