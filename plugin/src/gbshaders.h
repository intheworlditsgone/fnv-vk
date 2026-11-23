#pragma once
// Shader programs registered with the backend from the game's own shader creation (see gbshaders.cpp).
#include <cstdint>

namespace fnvvk::gbshaders {

bool installHooks();
// MD5 of the bytecode the game created `handle` from; false if the handle did not come through a hooked path.
bool lookup(void* handle, uint8_t md5[16]);
void logSummary();

}  // namespace fnvvk::gbshaders
