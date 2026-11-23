#pragma once
// The view vector of the game's lighting vertex shaders (sdp2spv adds it at location 12, docs/SHADER_ABI.md "View
// vector") is built from EyePosition (c16), which the game uploads only for the pass types whose vertex constant mask
// enables it. When a shader replacement reads the view vector, every lighting pass (one that uploads LightData) must
// upload EyePosition too; this does that in the engine's own per-pass masks.

namespace fnvvk::viewvector {

// `needed`: an active replacement reads the view vector. Installs the hooks only then.
bool install(bool needed);
// The world pass (effects.cpp, with the sun shadows and ambient occlusion): beginWorld at the start of each frame's world
// pass forgets which geometry got its reflection; between begin and end, the first specular pass of each geometry is
// marked (c219.y) to add the environment's reflection in the multi-pass path.
void beginWorld();
void endWorld();

}  // namespace fnvvk::viewvector
