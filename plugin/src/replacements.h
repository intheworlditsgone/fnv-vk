#pragma once
// Shader replacements (backend/effectlib.h, docs/REPLACEMENTS.md): GLSL from Data\NVSE\Plugins\fnv_vk\replacements\<id>\
// that replaces game pixel shaders, named as in the game's shader packages. Compiled at plugin load; the enabled ones
// are handed to the backend before it loads its shader pack (enabling or disabling one applies at the next start).
// Parameters apply live. Settings in fnv_vk.ini [Replacement.<id>].
#include <vector>

#include "effectlib.h"

namespace fnvvk {
class Backend;
struct BackendConfig;
}

namespace fnvvk::replacements {

std::vector<fx::Replacement>& list();
bool active(const fx::Replacement& r);           // replacing its shaders in this session
bool readsViewVector();                          // after load(): an enabled replacement reads the view vector (location 12)
bool readsLighting30EyePosition();               // after load(): an enabled Lighting30 (SM3) replacement takes c1 from the game
void load();                                     // at plugin load
void fillConfig(BackendConfig& cfg);             // before backend init
void onBackendReady(Backend* be);                // after backend init: the lighting's skin detail map (docs/SKIN.md)
void applyParams(Backend* be, const fx::Replacement& r);
bool save(const fx::Replacement& r);
const char* dir();

}  // namespace fnvvk::replacements
