#pragma once
// Exterior sun shadows (sunshadows.cpp): cascaded shadow maps of the world from the sun, drawn by the backend from the
// game's own geometry before the world, applied by the game's own lighting to the sun's light (beginLighting) and
// available to effects (fnvShadow). Settings: fnv_vk.ini [Shadows].
#include <string>

#include "backend.h"

namespace fnvvk::sunshadows {

void install();                        // reads [Shadows]
void render(Backend* be);              // at most once per frame, before the world pass draws
void beginLighting(Backend* be);       // the world pass's lighting shadows the sun (when this frame has cascades)
void endLighting(Backend* be);         // ... until here
void fill(Backend::EffectFrame& f);    // this frame's cascades for effects (none indoors, at night or when off)
std::string summary();                 // for the periodic effects summary
// The wet world's rain cover this frame (wetworld.h): a layer of the same maps, drawn from straight above.
struct Cover {
  bool drawn = false;
  uint32_t layer = 0;
  float fromWorld[16] = {};  // camera-relative world -> clip (z 0 at the top, 1 at the bottom)
  float bias = 0;            // clip depth under the top surface still counted as it
  float texel = 0;           // 1 / map size
};
const Cover& cover();

}  // namespace fnvvk::sunshadows
