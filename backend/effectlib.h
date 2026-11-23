#pragma once
// Effects as data: folders with a manifest and GLSL passes, compiled to SPIR-V at load (glslang). Game-independent; the
// plugin decides where effects run and feeds the backend (Backend::createEffect / runEffect). Format: docs/EFFECTS.md.
//
// <effects dir>/<id>/effect.ini
//   [Effect]       Name=, Description=, Point=world|tonemap, Order=100, SceneColor=0|1 (passes read the scene colour),
//                  Requires=pbr (runs only while the lighting's physically based mode is on),
//                  Skip=interior,underwater,abovewater,nocascades,<param>=0 (any of them: the effect would leave the
//                  image as it is, so it is not run)
//   [Param.<name>] Label=, Type=float|int|bool|enum, Default=, Min=, Max=, Step=, Options=a,b,c (enum: the choices;
//                  the value is the index)                                       (at most 32, in file order)
//   [Image.<name>] Format=r8|rg8|rgba8|r16f|rg16f|rgba16f|r32f, Scale=1.0 or Size=WxH (fixed),  (at most 8)
//                  Publish=ao|gi|gidepth (for the lighting, docs/EFFECTS.md: the frame's ambient occlusion with its
//                  depth in g; the screen-space indirect light, with gidepth its depth in the same effect)
//   [Pass.<n>]     Shader=<file>, Output=<image name>|scene, Blend=none|alpha|add|multiply   (n = 0, 1, ...),
//                  Skip=pbr (not run while the lighting's physically based mode applies the published AO)
// Every pass is a full-screen fragment shader; the generated header (effectHeader) is placed before its source.
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace fnvvk::fx {

enum class Blend : uint32_t { None, Alpha, Add, Multiply };

struct Param {
  std::string name, label;
  enum class Type { Float, Int, Bool, Enum } type = Type::Float;
  float def = 0, min = 0, max = 1, step = 0;
  std::vector<std::string> options;  // Enum: the choices (the value is the index)
  float value = 0;  // current value (the plugin's settings)
};
struct Image {
  std::string name;
  uint32_t vkFormat = 0;
  float scale = 1.0f;
  std::string publish;  // "ao", "gi", "gidepth" or empty
  uint32_t width = 0, height = 0;  // Size=WxH: a fixed size (Scale unused); 0: Scale of the frame
};
struct Pass {
  std::string shader;
  int output = -1;  // image index, -1 = the scene colour (drawn inside the game's render pass)
  Blend blend = Blend::None;
  bool skipPbr = false;  // Skip=pbr: not run while the lighting applies the published ambient occlusion
  std::vector<uint32_t> spv[2];  // [0] single-sample depth, [1] multisampled depth
};
struct Effect {
  std::string id;  // folder name
  std::string name, description, point = "world";
  int order = 100;
  bool sceneColor = false;
  bool skinBuffer = false;  // SkinBuffer=1: reads the skin buffer (fnvSkinAt; docs/SKIN.md), which the world then fills
  std::string requirement;  // Requires=: "pbr": runs only while the lighting replacement's physically based mode is on
  // Skip=: conditions under which the effect leaves the image as it is (and keeps nothing between runs that a skipped
  // run would change), so the plugin does not run it; any one of them skips.
  enum : uint32_t { SkipInterior = 1, SkipUnderwater = 2, SkipAboveWater = 4, SkipNoCascades = 8 };
  uint32_t skip = 0;
  std::vector<int> skipZero;  // <param>=0: indices of parameters whose value 0 skips
  bool enabled = true;
  std::vector<Param> params;
  std::vector<Image> images;
  std::vector<Pass> passes;
  std::string error;  // empty when the effect loaded and every pass compiled
};

// Shader replacements: GLSL that replaces game pixel shaders, written against the game shaders' own interface (the
// D3D9 constant registers, sampler registers and texcoord inputs of docs/SHADER_ABI.md). docs/REPLACEMENTS.md.
// <dir>/<id>/replace.ini
//   [Replacement]         Name=, Description=
//   [Param.<name>]        as for effects (at most 32); parameter i is float constant c(223 - i), x component
//   [Shader.<game name>]  Source=<file>, Defines=A B C     (game name as in the shader packages, e.g. WATER000.pso)
struct Replacement {
  std::string id, name, description;
  bool enabled = true;
  std::vector<int> packages;  // [Replacement] Packages: the shader packages it is verified for (empty: all)
  std::vector<Param> params;
  struct Target {
    std::string shader, source;
    std::vector<std::string> defines;
    std::vector<uint32_t> spv;
  };
  std::vector<Target> targets;
  std::string error;  // empty when every target compiled
};
constexpr uint32_t kReplacementParamTop = 223;  // parameter i: register kReplacementParamTop - i
std::vector<Replacement> loadReplacements(const std::string& dir, const std::string& cacheDir,
                                          const std::function<void(const char*)>& log);
std::string replacementHeader(const Replacement& r, const std::vector<std::string>& defines);

// Loads every folder under `dir` that has an effect.ini. Compiled SPIR-V is cached in `cacheDir` (keyed by the full
// source); compile errors name the file and line. Effects that fail keep their error and no passes.
std::vector<Effect> loadEffects(const std::string& dir, const std::string& cacheDir,
                                const std::function<void(const char*)>& log);
// The GLSL header placed before every pass of `e` (msDepth: the depth texture is multisampled).
std::string effectHeader(const Effect& e, bool msDepth);

}  // namespace fnvvk::fx
