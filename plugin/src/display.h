#pragma once
// HDR output settings ([Display] in fnv_vk.ini, the menu's Display panel) and what Windows reports for the game's
// monitor. The backend does the output (Backend::setHdr); the tonemap effect (effects/tonemap) shapes the highlights.
#include <cstdint>

namespace fnvvk {
class Backend;
}

namespace fnvvk::display {

struct Settings {
  enum Hdr : int { Off = 0, On = 1, Auto = 2 } hdr = Auto;  // Auto: on when Windows HDR is on for the game's monitor
  enum Format : int { Hdr10 = 0, ScRgb = 1 } format = Hdr10;
  float peakNits = 0;        // 0: the display's reported peak
  float paperWhiteNits = 0;  // 0: Windows' SDR content brightness
};

// The game's monitor as Windows reports it (DXGI output colour space and luminance, DisplayConfig SDR white level).
struct Monitor {
  bool known = false;
  bool hdrOn = false;            // Windows HDR is on (the output's colour space is BT.2020 PQ)
  float peakNits = 0;            // DXGI_OUTPUT_DESC1::MaxLuminance
  float sdrWhiteNits = 0;        // the "SDR content brightness" setting (0: not reported)
};

void load(void* window);           // reads [Display] and queries the monitor of `window`
Settings& settings();
const Monitor& monitor();
bool hdrWanted();                  // the settings ask for HDR output now (Auto follows Windows)
float effectivePeak();             // the nits used (setting or display)
float effectivePaperWhite();
void apply(Backend* be);           // pushes the effective settings to the backend
bool save();                       // writes [Display]

}  // namespace fnvvk::display
