#pragma once
// Loads textures through the game's file system and mip-skip settings, then uploads the d3dxtex result.
// Verification compares each level against NiDX9SourceTextureData::LoadTextureFile (0x00E68A80).
#include <cstdint>
#include <vector>

namespace fnvvk::textures {

bool installHooks();  // lifetime hook (NiDX9SourceTextureData deleting destructor)
void configure(bool verify);
// Call at LoadTextureFile entry (clears the per-thread capture of the game's D3DX input).
void beginLoad();
// D3D9 draws off: file textures become empty D3DPOOL_DEFAULT placeholders, content only in the backend.
void setNative(bool on);
// Called from the LoadTextureFile hook after the original returned. `cw` is the x87 control word at entry.
void onLoadTextureFile(void* data, const char* name, void* file, uint16_t cw, bool loaded);
void logSummary();

}  // namespace fnvvk::textures
