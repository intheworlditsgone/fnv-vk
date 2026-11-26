# Third-party notices

fnv_vk is licensed under the GNU General Public License, version 3 or (at your option) any later version (`LICENSE`).
It is built with the libraries below, each under its own license. Their full license texts are in their folders under
`third_party/`. Each license is compatible with distributing the combined work under GPL-3.0. Binary releases must
include this file and the license texts listed here.

| Library | Use | License | Text |
|---|---|---|---|
| [dxbc-spirv](https://github.com/doitsujin/dxbc-spirv) (Philip Rebohle), with the patches in `patches/dxbc-spirv` | Translating the game's D3D9 shaders to SPIR-V (tools/sdp2spv) | MIT | `third_party/dxbc-spirv/LICENSE` |
| [glslang](https://github.com/KhronosGroup/glslang) (Khronos Group and contributors) | Compiling effects and shader replacements at run time | BSD-3-Clause, with MIT, Apache-2.0 and BSD-2-Clause parts; its GPL-3.0 text applies only to Bison output, which this build does not use | `third_party/glslang/LICENSE.txt`, `third_party/glslang/LICENSES/` |
| [Dear ImGui](https://github.com/ocornut/imgui) (Omar Cornut) | The in-game menu | MIT | `third_party/imgui/LICENSE.txt` |
| [MinHook](https://github.com/TsudaKageyu/minhook) (Tsuda Kageyu), including Hacker Disassembler Engine (Vyacheslav Patkov) | Function hooks | BSD-2-Clause | `third_party/minhook/LICENSE.txt` |
| [DirectXTex](https://github.com/microsoft/DirectXTex) (Microsoft) | DDS texture decoding | MIT | `third_party/DirectXTex/LICENSE` |
| [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) (AMD) | GPU memory allocation | MIT | `third_party/VulkanMemoryAllocator/LICENSE.txt` |
| Vulkan headers (Khronos Group, from the Vulkan SDK) | The Vulkan API declarations | Apache-2.0 or MIT | the Vulkan SDK |

Ported (both GPL-3.0, so the ported code is under GPL-3.0 here as well):

- **VegasShaders** (`src/hlsl/NewVegas/Shaders/Includes/Wetness.hlsl`, the Puddles and RainState notes;
  GPL-3.0-or-later, with additional terms of its own under GPL-3.0 section 7) and **[Community
  Shaders](https://github.com/doodlum/skyrim-community-shaders)**' Wetness Effects (GPL-3.0), which VegasShaders' wetness
  is itself ported from. The wet world's rain ripples (`fnvWetRipples`), reoriented normals, hash, wet albedo, water
  film highlight and reflectance (backend/effectlib.cpp, FNV_WET) are GLSL ports of their HLSL, with their constants; the
  rain amount and the fill and drying schedule (plugin/src/wetworld.cpp) follow their design. Where water stands (the
  flooded terrain heightfield) and the rain cover layer are this project's own.
- **Community Shaders' Skin** (GPL-3.0 with its modding exceptions; `features/Skin/Shaders/Skin/Skin.hlsli`,
  `package/Shaders/Lighting.hlsl` CS_SKIN_SHADING, `Common/BRDF.hlsli`, `Common/Shading.hlsli`, the default profile in
  `src/Features/Skin/Material/Parameters.def`): the skin model in backend/effectlib.cpp (FNV_SKIN) is a GLSL port with
  their constants, and its detail normal map `replacements/lighting/skin_detail_n.dds` is their
  `features/Skin/Shaders/Skin/skin_detail_n.dds`, unchanged.
- **Community Shaders' Hair Specular** (GPL-3.0; `features/Hair Specular/Shaders/Hair/Hair.hlsli`,
  `package/Shaders/Lighting.hlsl` CS_HAIR_SHADING, `Common/LightDisk.hlsli`, the defaults in
  `src/Features/HairSpecular.h`): the hair model in backend/effectlib.cpp (FNV_HAIR) is a GLSL port with their constants,
  and its tangent shift map `replacements/lighting/hair_tangent_shift.dds` is their
  `features/Hair Specular/Shaders/Hair/TangentShift.dds`, unchanged.
- **Community Shaders' Grass Lighting** (GPL-3.0; `features/Grass Lighting/Shaders/GrassLighting/GrassLighting.hlsli`,
  `package/Shaders/RunGrass.hlsl`, `Common/PBRMath.hlsli`, the defaults in `src/Features/GrassLighting.h`): the grass model
  in replacements/lighting/grass.frag and the clump and canopy terms of sdp2spv's grass pass are ports of it.
- **Community Shaders' Subsurface Scattering** (GPL-3.0; `features/Subsurface Scattering/Shaders/SubsurfaceScattering/`,
  `src/Features/SubsurfaceScattering.cpp`): effects/sss ports its separable blur, kernel and albedo handling. The blur is
  Jorge Jimenez and Diego Gutierrez's Separable SSS, whose licence asks binary distributions to carry this notice:
  "Uses Separable SSS. Copyright (C) 2012 by Jorge Jimenez and Diego Gutierrez." (Redistribution and use in source and
  binary forms, with or without modification, are permitted provided that the source keeps the copyright notice and
  binary forms reproduce that disclaimer; provided by the copyright holders "as is", without any express or implied
  warranty; https://www.iryoku.com/separable-sss/.)

Referenced, but no code taken:

- **[DXVK](https://github.com/doitsujin/dxvk)** (zlib). Where the backend matches DXVK's D3D9 behaviour, the code
  comments name the DXVK function. That code is written for this backend, not copied.
- **[Fallout Dynamic Reflections](https://github.com/WallSoGB/Fallout-Dynamic-Reflections)** (LGPL-2.1). Studied for cube map rendering. 
- **[XeGTAO](https://github.com/GameTechDev/XeGTAO)** (MIT) and the published papers named in the shaders. Used as basis for the plugin's GTAO shaders.
