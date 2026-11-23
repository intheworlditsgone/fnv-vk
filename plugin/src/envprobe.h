#pragma once
// Environment probe (envprobe.cpp): a cube map of the scene around the camera, rendered by the game itself (its
// BSCubeMapCamera and scene rendering) one face per frame, for the lighting's ambient term and reflections
// (docs/INDIRECT.md).
#include <cstdint>

namespace fnvvk::envprobe {
bool install();  // at plugin load, after the backend hooks; reads fnv_vk.ini [Probe]
// The last complete probe: the cube texture's backend key (Backend::attachCube), 0 while none is complete. The face
// being drawn is in the other of two cubes, so draws can always sample this one.
uintptr_t current();
}  // namespace fnvvk::envprobe
