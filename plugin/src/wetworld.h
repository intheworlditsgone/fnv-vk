#pragma once
// Wet world (wetworld.cpp, docs/WETWORLD.md): in the game's rain, surfaces outdoors get wet, the terrain's depressions
// fill with standing water and the rain ripples it; it dries after the rain. Drawn by the lighting replacement in its
// physically based mode (FNV_WET), with the parameters wet and puddles in the menu (Lighting).
#include <string>

#include "backend.h"

namespace fnvvk::wetworld {

bool enabled();                   // the lighting replacement is on, in its physically based mode, with wet on
bool wantCover();                 // the rain cover is needed this frame (sunshadows draws it)
void update(Backend* be);         // once per frame before the world pass: the weather, the levels, the puddle map
void beginLighting(Backend* be);  // the world pass's lighting gets the wet state
void endLighting(Backend* be);
std::string summary();            // for the periodic effects summary

}  // namespace fnvvk::wetworld
