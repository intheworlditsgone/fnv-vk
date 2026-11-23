#pragma once
// Effects as data (backend/effectlib.h, docs/EFFECTS.md): loaded from Data\NVSE\Plugins\fnv_vk\effects\<id>\ and run at
// fixed points of the game's frame (native mode; effects.cpp). Settings per effect in fnv_vk.ini [Effect.<id>].
#include <vector>

#include "effectlib.h"

namespace fnvvk::effects {

// The loaded effects, in run order. The menu edits `enabled` and parameter values in place; they apply next frame.
std::vector<fx::Effect>& list();
bool save(const fx::Effect& e);  // writes Enable and the parameter values to fnv_vk.ini [Effect.<id>]
void reload();                   // reads and compiles every effect again (keeps current settings by name)
const char* effectsDir();

bool hasPoint(const char* point);  // an effect that loaded runs at `point` (whether enabled or not)
bool pbrLighting();                // the lighting replacement's physically based mode is on (effects with Requires=pbr run)

bool install();  // at plugin load, native mode only
void logSummary();

}  // namespace fnvvk::effects
