// Effect loading and GLSL compilation (effectlib.h).
#include "effectlib.h"

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>

#include "glslang/Public/ResourceLimits.h"
#include "glslang/Public/ShaderLang.h"
#include "glslang/SPIRV/GlslangToSpv.h"

namespace fnvvk::fx {
namespace {

namespace fs = std::filesystem;

// VkFormat values (vulkan_core.h), so this file needs no Vulkan headers.
struct FormatName { const char* name; uint32_t vk; };
constexpr FormatName kFormats[] = {
  { "r8", 9 }, { "rg8", 16 }, { "rgba8", 37 }, { "r16f", 76 }, { "rg16f", 83 }, { "rgba16f", 97 }, { "r32f", 100 },
};
constexpr uint32_t kMaxParams = 32, kMaxImages = 8, kMaxPasses = 16;

std::string iniString(const std::string& ini, const char* section, const char* key, const char* def = "") {
  char buf[1024] = "";
  GetPrivateProfileStringA(section, key, def, buf, sizeof(buf), ini.c_str());
  return buf;
}
float iniFloat(const std::string& ini, const char* section, const char* key, float def) {
  const std::string v = iniString(ini, section, key);
  return v.empty() ? def : float(atof(v.c_str()));
}
std::string lower(std::string s) {
  for (char& c : s) c = char(tolower(uint8_t(c)));
  return s;
}
bool validIdentifier(const std::string& s) {
  if (s.empty() || !(isalpha(uint8_t(s[0])) || s[0] == '_')) return false;
  for (char c : s)
    if (!(isalnum(uint8_t(c)) || c == '_')) return false;
  return true;
}
// Section names of an ini file, in file order.
std::vector<std::string> sections(const std::string& ini) {
  std::vector<char> buf(65536);
  const DWORD n = GetPrivateProfileSectionNamesA(buf.data(), DWORD(buf.size()), ini.c_str());
  std::vector<std::string> out;
  for (DWORD i = 0; i < n && buf[i];) {
    out.emplace_back(&buf[i]);
    i += DWORD(out.back().size()) + 1;
  }
  return out;
}
uint64_t fnv1a(const std::string& s) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
  return h;
}

std::once_flag g_glslangInit;

// Compiles one fragment shader: `header` (generated, starts with #version) then `source`. Errors name `file`.
bool compile(const std::string& header, const std::string& source, const std::string& file, std::vector<uint32_t>& spv,
             std::string& error) {
  std::call_once(g_glslangInit, [] { glslang::InitializeProcess(); });
  glslang::TShader shader(EShLangFragment);
  const char* strings[2] = { header.c_str(), source.c_str() };
  const int lengths[2] = { int(header.size()), int(source.size()) };
  const char* names[2] = { "fnv_vk header", file.c_str() };
  shader.setStringsWithLengthsAndNames(strings, lengths, names, 2);
  shader.setEntryPoint("main");
  shader.setEnvInput(glslang::EShSourceGlsl, EShLangFragment, glslang::EShClientVulkan, 100);
  shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_3);
  shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_6);
  const EShMessages msgs = EShMessages(EShMsgSpvRules | EShMsgVulkanRules);
  if (!shader.parse(GetDefaultResources(), 450, false, msgs)) {
    error = shader.getInfoLog();
    return false;
  }
  glslang::TProgram program;
  program.addShader(&shader);
  if (!program.link(msgs)) {
    error = program.getInfoLog();
    return false;
  }
  spv.clear();
  glslang::SpvOptions opt;
  opt.validate = false;
  glslang::GlslangToSpv(*program.getIntermediate(EShLangFragment), spv, &opt);
  return !spv.empty();
}

// One [Param.<name>] section (effects and replacements). False with `error` set.
bool parseParam(const std::string& ini, const std::string& sec, Param& p, std::string& error) {
  p.name = sec.substr(6);
  if (!validIdentifier(p.name)) { error = "parameter name '" + p.name + "' is not a GLSL identifier"; return false; }
  p.label = iniString(ini, sec.c_str(), "Label", p.name.c_str());
  const std::string t = lower(iniString(ini, sec.c_str(), "Type", "float"));
  p.type = t == "int" ? Param::Type::Int : t == "bool" ? Param::Type::Bool : t == "enum" ? Param::Type::Enum : Param::Type::Float;
  if (p.type == Param::Type::Enum) {
    std::stringstream opts(iniString(ini, sec.c_str(), "Options"));
    for (std::string o; std::getline(opts, o, ',');) {
      o.erase(0, o.find_first_not_of(" \t"));
      o.erase(o.find_last_not_of(" \t") + 1);
      if (!o.empty()) p.options.push_back(o);
    }
    if (p.options.empty()) { error = "enum parameter '" + p.name + "' has no Options"; return false; }
  }
  p.def = iniFloat(ini, sec.c_str(), "Default", 0.0f);
  p.min = iniFloat(ini, sec.c_str(), "Min", 0.0f);
  p.max = iniFloat(ini, sec.c_str(), "Max", p.type == Param::Type::Bool ? 1.0f : std::max(1.0f, p.def * 2));
  p.step = iniFloat(ini, sec.c_str(), "Step", 0.0f);
  if (p.type == Param::Type::Enum) {
    p.min = 0;
    p.max = float(p.options.size() - 1);
    p.def = std::clamp(std::floor(p.def), p.min, p.max);
  }
  p.value = p.def;
  return true;
}
// GLSL for parameter `p` stored in `v` (a float expression).
std::string paramDefine(const Param& p, const std::string& v) {
  if (p.type == Param::Type::Int || p.type == Param::Type::Enum) return "#define p_" + p.name + " int(" + v + ")\n";
  if (p.type == Param::Type::Bool) return "#define p_" + p.name + " (" + v + " != 0.0)\n";
  return "#define p_" + p.name + " (" + v + ")\n";
}

bool readText(const fs::path& p, std::string& out) {
  std::ifstream f(p, std::ios::binary);
  if (!f) return false;
  std::stringstream ss;
  ss << f.rdbuf();
  out = ss.str();
  return true;
}

}  // namespace

std::string effectHeader(const Effect& e, bool msDepth) {
  std::string h;
  h += "#version 450\n";
  h += "// fnv_vk effect interface (generated for '" + e.id + "'; docs/EFFECTS.md)\n";
  h += msDepth ? "#define FNV_DEPTH_MS 1\n" : "#define FNV_DEPTH_MS 0\n";
  h += "#if FNV_DEPTH_MS\nlayout(set = 0, binding = 0) uniform texture2DMS fnvDepthTex;\n"
       "#else\nlayout(set = 0, binding = 0) uniform texture2D fnvDepthTex;\n#endif\n";
  h += "layout(set = 0, binding = 1) uniform texture2D fnvColorTex;\n";
  for (size_t i = 0; i < e.images.size(); i++)
    h += "layout(set = 0, binding = " + std::to_string(2 + i) + ") uniform texture2D img_" + e.images[i].name + ";\n";
  h += "layout(set = 0, binding = 10) uniform sampler fnvPoint;\n"
       "layout(set = 0, binding = 11) uniform sampler fnvLinear;\n"
       "layout(set = 0, binding = 12, std140) uniform FnvFrame {\n"
       "  vec4 rect;        // camera viewport in the full-resolution target: x, y, width, height (pixels)\n"
       "  vec4 frustum;     // camera frustum tangents: left, right, top, bottom\n"
       "  vec4 depthRange;  // near, far, submersion, 1 when the lighting applies the published ambient occlusion\n"
       "  vec4 target;      // full-resolution target: width, height, 1/width, 1/height\n"
       "  vec4 time;        // seconds, frame number, seconds since this effect's previous run (0: none), 0\n"
       "  vec4 hdr;         // output (0 SDR, 1 HDR10, 2 scRGB), peak nits, paper white nits, peak / paper white (1 in SDR)\n"
       "  vec4 params[32];\n"
       "  vec4 cameraRows[3]; vec4 waterSun;\n"
       "  layout(row_major) mat4 shadowFromView[4];  // view space -> sun shadow cascade clip space (fnvShadow)\n"
       "  vec4 shadowSplits;  // far view distance of each cascade\n"
       "  vec4 sunView;       // direction toward the sun in view space\n"
       "  vec4 shadowInfo;    // cascades (0: no shadows), texel size, 1 in an interior cell, 0\n"
       "  vec4 sunColor;      // the sun's diffuse light (rgb)\n"
       "  vec4 ambientColor;  // the scene's ambient light (rgb)\n"
       "  vec4 prevCameraRows[3];  // this effect's previous run (one or two frames ago): camera rows,\n"
       "  vec4 prevFrustum;        // frustum tangents,\n"
       "  vec4 prevInfo;           // 1 when there is one, near, far, 0 (fnvPrevPixel)\n"
       "} fnv;\n"
       "layout(push_constant) uniform FnvPass {\n"
       "  vec4 size;  // this pass's output: width, height, 1/width, 1/height\n"
       "  vec4 info;  // output scale relative to the target (0: a fixed-size image, Size=), pass index, 0, 0\n"
       "} fnvPass;\n";
  for (size_t i = 0; i < e.params.size(); i++) h += paramDefine(e.params[i], "fnv.params[" + std::to_string(i) + "].x");
  h += R"GLSL(
// Raw depth (0..1, D3D9-style) of a full-resolution pixel; sample 0 of multisampled depth.
float fnvRawDepth(ivec2 p) {
#if FNV_DEPTH_MS
  return texelFetch(sampler2DMS(fnvDepthTex, fnvPoint), p, 0).r;
#else
  return texelFetch(sampler2D(fnvDepthTex, fnvPoint), p, 0).r;
#endif
}
// View distance of a raw depth value (the game's projection: z = n*f / (f - d*(f-n))).
float fnvLinearZ(float d) { return fnv.depthRange.x * fnv.depthRange.y / (fnv.depthRange.y - d * (fnv.depthRange.y - fnv.depthRange.x)); }
bool fnvIsSky(float d) { return d >= 0.999999; }
bool fnvInsideRect(ivec2 p) { return all(greaterThanEqual(p, ivec2(fnv.rect.xy))) && all(lessThan(p, ivec2(fnv.rect.xy + fnv.rect.zw))); }
ivec2 fnvClampRect(ivec2 p) { return clamp(p, ivec2(fnv.rect.xy), ivec2(fnv.rect.xy + fnv.rect.zw) - 1); }
// View-space position (x right, y up, z forward) of a point at view distance z under a full-resolution pixel position.
vec3 fnvViewPos(vec2 pixel, float z) {
  vec2 uv = (pixel - fnv.rect.xy) / fnv.rect.zw;
  return vec3(mix(fnv.frustum.x, fnv.frustum.y, uv.x) * z, mix(fnv.frustum.z, fnv.frustum.w, uv.y) * z, z);
}
vec3 fnvViewPosAt(ivec2 p) { p = fnvClampRect(p); return fnvViewPos(vec2(p) + 0.5, fnvLinearZ(fnvRawDepth(p))); }
// This fragment's position in full-resolution target pixels (outputs may be scaled).
vec2 fnvPixel() { return gl_FragCoord.xy / fnvPass.info.x; }
// World position of a view-space point (fnvViewPos): game units, z up. Camera rows: world rotation (columns direction,
// up, right) with the origin in w; view -> world: T + R (z, y, x). fnvCameraWorldPos() is the camera's (the origin).
vec3 fnvWorldPos(vec3 v) {
  const vec3 l = vec3(v.z, v.y, v.x);
  return vec3(dot(fnv.cameraRows[0].xyz, l) + fnv.cameraRows[0].w, dot(fnv.cameraRows[1].xyz, l) + fnv.cameraRows[1].w,
              dot(fnv.cameraRows[2].xyz, l) + fnv.cameraRows[2].w);
}
vec3 fnvCameraWorldPos() { return vec3(fnv.cameraRows[0].w, fnv.cameraRows[1].w, fnv.cameraRows[2].w); }
// Where a view-space point (fnvViewPos) was in this effect's previous run: its full-resolution pixel position and its view
// distance then; false without a previous run, behind that camera or outside its view. Camera rows: world rotation
// (columns direction, up, right) with the origin in w; view -> world: T + R (z, y, x).
bool fnvPrevPixel(vec3 v, out vec2 pixel, out float prevZ) {
  pixel = vec2(0.0);
  prevZ = 0.0;
  if (fnv.prevInfo.x == 0.0) return false;
  const vec3 w = fnvWorldPos(v);
  const vec3 d = w - vec3(fnv.prevCameraRows[0].w, fnv.prevCameraRows[1].w, fnv.prevCameraRows[2].w);
  const vec3 pl = d.x * fnv.prevCameraRows[0].xyz + d.y * fnv.prevCameraRows[1].xyz + d.z * fnv.prevCameraRows[2].xyz;
  const vec3 q = vec3(pl.z, pl.y, pl.x);
  if (q.z <= 0.0) return false;
  const vec4 pf = fnv.prevFrustum;
  const vec2 uv = vec2((q.x / q.z - pf.x) / (pf.y - pf.x), (q.y / q.z - pf.z) / (pf.w - pf.z));
  pixel = fnv.rect.xy + uv * fnv.rect.zw;
  prevZ = q.z;
  return all(greaterThanEqual(uv, vec2(0.0))) && all(lessThan(uv, vec2(1.0)));
}
// The skin buffer (docs/SKIN.md "Subsurface scattering"): written by the world's skin in the physically based lighting
// during the opaque half of the world pass (effects at "opaque" see this frame's).
layout(set = 0, binding = 15) uniform texture2D fnvSkinAlbedoTex;
layout(set = 0, binding = 16) uniform utexture2D fnvSkinDepthTex;
// True where full-resolution pixel p shows skin: the nearest skin drawn there is what the depth buffer holds (within
// half a unit or 0.2% of the distance: depth precision and multisampling); albedo its albedo (linear).
bool fnvSkinAt(ivec2 p, out vec3 albedo) {
  albedo = vec3(0.0);
  if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, textureSize(usampler2D(fnvSkinDepthTex, fnvPoint), 0))))
    return false;
  const uint bits = texelFetch(usampler2D(fnvSkinDepthTex, fnvPoint), p, 0).r;
  if (bits == 0xFFFFFFFFu) return false;
  const float d = fnvRawDepth(p);
  if (fnvIsSky(d)) return false;
  const float ls = fnvLinearZ(uintBitsToFloat(bits)), ld = fnvLinearZ(d);
  if (abs(ls - ld) > max(0.5, 0.002 * ld)) return false;
  albedo = pow(texelFetch(sampler2D(fnvSkinAlbedoTex, fnvPoint), p, 0).rgb, vec3(2.2));
  return true;
}
// Sun shadow cascades (exterior shadows; fnv.shadowInfo.x is 0 when there are none this frame).
layout(set = 0, binding = 13) uniform texture2DArray fnvShadowTex;
layout(set = 0, binding = 14) uniform samplerShadow fnvShadowCmp;
// Sun visibility of a view-space point (fnvViewPos): 1 lit, 0 in shadow; 1 without shadows or beyond the last cascade.
// 3 x 3 percentage-closer filtering (each tap is itself a bilinear 2 x 2 compare); over the last 20% of a cascade's
// depth range the next one is blended in (no visible line where the texel size changes); the last cascade fades out.
// The game's lighting computes the same (sdp2spv's sun shadow pass).
float fnvShadowIn(int c, vec3 viewPos) {
  vec4 p = fnv.shadowFromView[c] * vec4(viewPos, 1.0);
  vec2 uv = p.xy * 0.5 + 0.5;
  float sum = 0.0;
  for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++)
      sum += texture(sampler2DArrayShadow(fnvShadowTex, fnvShadowCmp), vec4(uv + vec2(x, y) * fnv.shadowInfo.y, float(c), p.z));
  return sum / 9.0;
}
float fnvShadow(vec3 viewPos) {
  int n = int(fnv.shadowInfo.x);
  if (n == 0 || viewPos.z > fnv.shadowSplits[n - 1]) return 1.0;
  int c = 0;
  while (c < n - 1 && viewPos.z > fnv.shadowSplits[c]) c++;
  float v = fnvShadowIn(c, viewPos);
  if (c < n - 1) {
    float split = fnv.shadowSplits[c], start = split - 0.2 * (split - (c > 0 ? fnv.shadowSplits[c - 1] : 0.0));
    float w = clamp((viewPos.z - start) / (split - start), 0.0, 1.0);
    if (w > 0.0) v = mix(v, fnvShadowIn(c + 1, viewPos), w);
  }
  float last = fnv.shadowSplits[n - 1];
  return mix(v, 1.0, smoothstep(0.85 * last, last, viewPos.z));
}
// The scene colour (requires SceneColor=1 in effect.ini): the frame as it was before this effect ran.
vec4 fnvSceneColor(ivec2 p) { return texelFetch(sampler2D(fnvColorTex, fnvPoint), p, 0); }
)GLSL";
  return h;
}

std::vector<Effect> loadEffects(const std::string& dir, const std::string& cacheDir,
                                const std::function<void(const char*)>& log) {
  std::vector<Effect> out;
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) {
    log(("effects: no effects folder at " + dir).c_str());
    return out;
  }
  fs::create_directories(cacheDir, ec);
  std::vector<fs::path> folders;
  for (const auto& entry : fs::directory_iterator(dir, ec))
    if (entry.is_directory() && fs::exists(entry.path() / "effect.ini")) folders.push_back(entry.path());
  std::sort(folders.begin(), folders.end());

  for (const fs::path& folder : folders) {
    Effect e;
    e.id = folder.filename().string();
    const std::string ini = (folder / "effect.ini").string();
    e.name = iniString(ini, "Effect", "Name", e.id.c_str());
    e.description = iniString(ini, "Effect", "Description");
    e.point = lower(iniString(ini, "Effect", "Point", "world"));
    e.order = GetPrivateProfileIntA("Effect", "Order", 100, ini.c_str());
    e.sceneColor = GetPrivateProfileIntA("Effect", "SceneColor", 0, ini.c_str()) != 0;
    e.skinBuffer = GetPrivateProfileIntA("Effect", "SkinBuffer", 0, ini.c_str()) != 0;
    e.requirement = lower(iniString(ini, "Effect", "Requires", ""));
    if (!e.requirement.empty() && e.requirement != "pbr") e.error = "unknown Requires '" + e.requirement + "'";
    std::vector<std::pair<int, std::string>> passSections;
    for (const std::string& sec : sections(ini)) {
      const std::string ls = lower(sec);
      if (ls.rfind("param.", 0) == 0) {
        Param p;
        if (!parseParam(ini, sec, p, e.error)) break;
        e.params.push_back(p);
      } else if (ls.rfind("image.", 0) == 0) {
        Image im;
        im.name = sec.substr(6);
        if (!validIdentifier(im.name)) { e.error = "image name '" + im.name + "' is not a GLSL identifier"; break; }
        const std::string f = lower(iniString(ini, sec.c_str(), "Format", "rgba8"));
        for (const FormatName& fn : kFormats)
          if (f == fn.name) im.vkFormat = fn.vk;
        if (!im.vkFormat) { e.error = "image '" + im.name + "': unknown format '" + f + "'"; break; }
        im.scale = std::clamp(iniFloat(ini, sec.c_str(), "Scale", 1.0f), 0.0625f, 1.0f);
        // Size=WxH: a fixed size (1..4096 each), for reductions to a few texels (e.g. an average luminance).
        const std::string size = lower(iniString(ini, sec.c_str(), "Size", ""));
        if (!size.empty()) {
          unsigned w = 0, h = 0;
          char extra = 0;
          if (sscanf_s(size.c_str(), "%ux%u%c", &w, &h, &extra, 1) != 2 || w < 1 || h < 1 || w > 4096 || h > 4096) {
            e.error = "image '" + im.name + "': Size must be WxH (1..4096 each), not '" + size + "'";
            break;
          }
          im.width = w;
          im.height = h;
        }
        im.publish = lower(iniString(ini, sec.c_str(), "Publish", ""));
        if (!im.publish.empty() && im.publish != "ao" && im.publish != "gi" && im.publish != "gidepth") {
          e.error = "image '" + im.name + "': unknown Publish '" + im.publish + "'";
          break;
        }
        e.images.push_back(im);
      } else if (ls.rfind("pass.", 0) == 0) {
        passSections.push_back({ atoi(sec.c_str() + 5), sec });
      }
    }
    if (e.error.empty() && e.params.size() > kMaxParams) e.error = "more than 32 parameters";
    if (e.error.empty()) {
      bool gi = false, giDepth = false;
      for (const Image& im : e.images) {
        gi |= im.publish == "gi";
        giDepth |= im.publish == "gidepth";
      }
      if (gi != giDepth) e.error = "Publish=gi and Publish=gidepth go together (the depth validates the light)";
    }
    if (e.error.empty() && e.images.size() > kMaxImages) e.error = "more than 8 images";
    // Skip=: when the effect would leave the image as it is (the plugin then does not run it).
    if (e.error.empty()) {
      std::stringstream list(lower(iniString(ini, "Effect", "Skip", "")));
      for (std::string t; e.error.empty() && std::getline(list, t, ',');) {
        t.erase(0, t.find_first_not_of(" 	"));
        t.erase(t.find_last_not_of(" 	") + 1);
        if (t.empty()) continue;
        if (t == "interior") e.skip |= Effect::SkipInterior;
        else if (t == "underwater") e.skip |= Effect::SkipUnderwater;
        else if (t == "abovewater") e.skip |= Effect::SkipAboveWater;
        else if (t == "nocascades") e.skip |= Effect::SkipNoCascades;
        else if (t.size() > 2 && t.compare(t.size() - 2, 2, "=0") == 0) {
          const std::string name = t.substr(0, t.size() - 2);
          int index = -1;
          for (size_t k = 0; k < e.params.size(); k++)
            if (lower(e.params[k].name) == name) index = int(k);
          if (index < 0) e.error = "Skip: '" + name + "' is not a parameter";
          else e.skipZero.push_back(index);
        } else {
          e.error = "unknown Skip '" + t + "'";
        }
      }
    }
    if (e.error.empty() && (passSections.empty() || passSections.size() > kMaxPasses)) e.error = "needs 1 to 16 [Pass.n] sections";
    std::sort(passSections.begin(), passSections.end());
    for (size_t i = 0; e.error.empty() && i < passSections.size(); i++) {
      const std::string& sec = passSections[i].second;
      Pass p;
      p.shader = iniString(ini, sec.c_str(), "Shader");
      const std::string o = iniString(ini, sec.c_str(), "Output", "scene");
      if (lower(o) != "scene") {
        for (size_t k = 0; k < e.images.size(); k++)
          if (e.images[k].name == o) p.output = int(k);
        if (p.output < 0) { e.error = sec + ": output image '" + o + "' is not declared"; break; }
      }
      const std::string b = lower(iniString(ini, sec.c_str(), "Blend", "none"));
      p.blend = b == "alpha" ? Blend::Alpha : b == "add" ? Blend::Add : b == "multiply" ? Blend::Multiply : Blend::None;
      const std::string skip = lower(iniString(ini, sec.c_str(), "Skip", ""));
      if (skip == "pbr") p.skipPbr = true;
      else if (!skip.empty()) { e.error = sec + ": unknown Skip '" + skip + "'"; break; }
      std::string source;
      if (p.shader.empty() || !readText(folder / p.shader, source)) { e.error = sec + ": cannot read shader '" + p.shader + "'"; break; }
      for (int ms = 0; ms < 2 && e.error.empty(); ms++) {
        const std::string header = effectHeader(e, ms != 0);
        char key[32];
        snprintf(key, sizeof(key), "%016llx", static_cast<unsigned long long>(fnv1a(header + "\n" + source + "\nfnvvk-fx-2")));
        const fs::path cached = fs::path(cacheDir) / (e.id + "_" + std::to_string(i) + "_" + key + ".spv");
        std::string bytes;
        if (readText(cached, bytes) && bytes.size() >= 20 && bytes.size() % 4 == 0) {
          p.spv[ms].resize(bytes.size() / 4);
          memcpy(p.spv[ms].data(), bytes.data(), bytes.size());
          continue;
        }
        std::string err;
        if (!compile(header, source, p.shader, p.spv[ms], err)) {
          e.error = p.shader + (ms ? " (multisampled depth)" : "") + ":\n" + err;
          break;
        }
        std::ofstream f(cached, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(p.spv[ms].data()), std::streamsize(p.spv[ms].size() * 4));
      }
      if (e.error.empty()) e.passes.push_back(std::move(p));
    }
    if (!e.error.empty()) {
      e.passes.clear();
      log(("effect '" + e.id + "' NOT loaded: " + e.error).c_str());
    } else {
      log(("effect '" + e.id + "' (" + e.name + "): " + std::to_string(e.passes.size()) + " passes, " +
           std::to_string(e.params.size()) + " parameters, point '" + e.point + "'").c_str());
    }
    out.push_back(std::move(e));
  }
  std::stable_sort(out.begin(), out.end(), [](const Effect& a, const Effect& b) { return a.order < b.order; });
  return out;
}

std::string replacementHeader(const Replacement& r, const std::vector<std::string>& defines) {
  std::string h;
  h += "#version 450\n";
  h += "#extension GL_EXT_scalar_block_layout : require\n";
  h += "#extension GL_KHR_memory_scope_semantics : require\n";
  h += "#pragma use_vulkan_memory_model\n";  // as the translated game shaders the pipeline links with
  h += "// fnv_vk shader replacement interface (generated for '" + r.id + "'; docs/REPLACEMENTS.md)\n";
  for (const std::string& d : defines) {  // NAME or NAME=VALUE
    const size_t eq = d.find('=');
    h += "#define " + (eq == std::string::npos ? d + " 1" : d.substr(0, eq) + " " + d.substr(eq + 1)) + "\n";
  }
  h += R"GLSL(
// The game's float constants c0..c223 (D3D9 numbering), with the shader's def constants and this replacement's
// parameters overlaid.
layout(set = 1, binding = 0, scalar) uniform FnvConstants { vec4 fnvC[224]; };
layout(set = 1, binding = 3, scalar) uniform FnvAlphaTest { uint fnvAlphaOp; uint fnvAlphaBits; uint fnvAlphaRef; };
// FNV_SAMPLER2D(s, name): the texture bound to sampler register s; sample it with texture(FNV_TEX(name), uv).
#define FNV_SAMPLER2D(s, name) layout(set = 1, binding = 16 + s) uniform texture2D name##_tex; layout(set = 1, binding = 32 + s) uniform sampler name##_smp;
#define FNV_TEX(name) sampler2D(name##_tex, name##_smp)
// FNV_SAMPLERCUBE(s, name): a cube texture at sampler register s; sample it with texture(FNV_TEXCUBE(name), dir).
#define FNV_SAMPLERCUBE(s, name) layout(set = 1, binding = 16 + s) uniform textureCube name##_tex; layout(set = 1, binding = 32 + s) uniform sampler name##_smp;
#define FNV_TEXCUBE(name) samplerCube(name##_tex, name##_smp)
// FNV_TEXCOORD(n) vec4 name;  the vertex shader's TEXCOORDn output.
#define FNV_TEXCOORD(n) layout(location = 1 + n) centroid in
// The vertex shaders' FOG output (every translated pixel shader declares it; the interfaces must match).
layout(location = 11) in float fnvFogInput;
// The pipeline's alpha test (D3DRS_ALPHAFUNC / ALPHAREF), as the translated game shaders apply it.
void fnvAlphaTest(float a) {
  if (fnvAlphaOp == 7u) return;
  float q = clamp(a, 0.0, 1.0) * 255.0;
  if (fnvAlphaBits == 0u) q = floor(q + 0.5);
  float ref = float(fnvAlphaRef);
  bool pass = fnvAlphaOp == 1u ? q < ref : fnvAlphaOp == 2u ? q == ref : fnvAlphaOp == 3u ? q <= ref :
              fnvAlphaOp == 4u ? q > ref : fnvAlphaOp == 5u ? q != ref : fnvAlphaOp == 6u ? q >= ref : false;
  if (!pass) discard;
}
// D3D9's fixed-function fog (D3DRS_FOGENABLE), applied to the colour a pixel shader writes, exactly as the translated
// game shaders apply it (dxbc-spirv sm3 Converter::emitFog): mode 0 blends by the vertex shader's FOG output, 1..3 by
// exp, exp2 or linear fog of the depth (z, or w); the factor is clamped with NaN going to 1 (no fog).
layout(set = 1, binding = 4, scalar) uniform FnvFixedFog {
  uint fnvFogEnable, fnvFogUseZ, fnvFogMode;
  vec3 fnvFogColor;
  float fnvFogScale, fnvFogEnd, fnvFogDensity;
};
float fnvMulLegacy(float a, float b) { return (a == 0.0 || b == 0.0) ? 0.0 : a * b; }
vec3 fnvFixedFog(vec3 color) {
  if (fnvFogEnable == 0u) return color;
  float depth = fnvFogUseZ != 0u ? gl_FragCoord.z : 1.0 / gl_FragCoord.w;
  float f;
  if (fnvFogMode == 3u) f = fnvMulLegacy(fnvFogEnd - depth, fnvFogScale);
  else if (fnvFogMode == 1u || fnvFogMode == 2u) {
    f = fnvMulLegacy(depth, fnvFogDensity);
    if (fnvFogMode == 2u) f = fnvMulLegacy(f, f);
    f = exp2(f * -1.442695041);
  } else f = fnvFogInput;
  f = max(min(f, 1.0), 0.0);
  return fnvFogColor + (color - fnvFogColor) * f;
}
#ifdef FNV_SUN
// Sun shadows (FNV_SUN: the shader replaced lights with the sun; docs/SHADER_ABI.md "Sun shadows"): the same block,
// cascades and computation as the translated lighting shaders (sdp2spv's sun shadow pass). fnvSunVisibility() is the
// sun's visibility at this fragment, 1 when the runtime does not shadow this draw; call it at the top of main (it
// takes screen derivatives) and multiply the sun's light colour by it.
layout(set = 1, binding = 9, scalar) uniform FnvSunShadow { vec4 fnvSun[21]; };
layout(set = 1, binding = 48) uniform texture2DArray fnvSunCascades;
layout(set = 1, binding = 49) uniform samplerShadow fnvSunCompare;
float fnvSunFilter(uint c, vec3 q) {
  uint row = 5u + 4u * c;
  vec3 clip;
  for (uint i = 0u; i < 3u; i++) {
    vec4 r = fnvSun[row + i];
    clip[i] = r.x * q.x + r.y * q.y + r.z * q.z + r.w;
  }
  vec2 uv = clip.xy * 0.5 + 0.5;
  float texel = fnvSun[0].z, sum = 0.0;
  for (int y = -1; y <= 1; y++)
    for (int x = -1; x <= 1; x++)
      sum += textureGrad(sampler2DArrayShadow(fnvSunCascades, fnvSunCompare),
                         vec4(uv + vec2(x, y) * texel, float(c), clip.z), vec2(0.0), vec2(0.0));
  return sum * (1.0 / 9.0);
}
float fnvSunVisibility() {
  vec4 rect = fnvSun[1], fr = fnvSun[2];
  float zn = fnvSun[3].x, zf = fnvSun[3].y;
  float z = zn * zf / (zf - gl_FragCoord.z * (zf - zn));
  float u = (gl_FragCoord.x - rect.x) / rect.z, v = (gl_FragCoord.y - rect.y) / rect.w;
  vec3 p = vec3(mix(fr.x, fr.y, u) * z, mix(fr.z, fr.w, v) * z, z);
  vec3 dx = dFdx(p), dy = dFdy(p);
  vec3 n = vec3(dy.y * dx.z - dy.z * dx.y, dy.z * dx.x - dy.x * dx.z, dy.x * dx.y - dy.y * dx.x);
  float len2 = dot(n, n);
  float scale = len2 > 1e-30 ? inversesqrt(max(len2, 1e-30)) * (dot(n, p) > 0.0 ? -1.0 : 1.0) : 0.0;
  vec3 q = p + n * (scale * (1.0 + 0.002 * z));
  vec4 splits = fnvSun[4];
  uint c = uint(q.z > splits.x) + uint(q.z > splits.y) + uint(q.z > splits.z);
  float split = splits[min(c, 3u)];
  float prev = c == 0u ? 0.0 : splits[c - 1u];
  float start = split - 0.2 * (split - prev);
  float blend = clamp((q.z - start) / (split - start), 0.0, 1.0);
  // The next cascade only where it is blended in (the last 20% of this one's range): at blend 0 the mix is the first
  // filter exactly, and the next cascade's nine taps (past the last cascade: a clamped layer) are wasted. textureGrad
  // keeps the taps valid in this non-uniform branch.
  float pcf = fnvSunFilter(c, q);
  if (blend > 0.0) pcf = mix(pcf, fnvSunFilter(min(c + 1u, 3u), q), blend);
  float last = fnvSun[0].y, fadeStart = last * 0.85;
  float t = clamp((q.z - fadeStart) / (last - fadeStart), 0.0, 1.0);
  float fade = t * t * (3.0 - 2.0 * t);
  float shadowed = mix(pcf, 1.0, fade);
  return fnvSun[0].x != 0.0 ? 1.0 - fnvSun[0].w * (1.0 - shadowed) : 1.0;
}
#endif
#ifdef FNV_ENV
// The environment (FNV_ENV; docs/SHADER_ABI.md "Environment", docs/INDIRECT.md): the probe cube GGX-prefiltered, in
// linear light (sampled with world-space directions), and its irradiance harmonics
// normalised to a mean of 1. fnvEnvValid() is false until a probe exists (the shader then keeps its flat ambient).
layout(set = 1, binding = 10, scalar) uniform FnvEnvironment { vec4 fnvEnv[10]; };
layout(set = 1, binding = 50) uniform textureCube fnvEnvCube;
layout(set = 1, binding = 51) uniform sampler fnvEnvSampler;
bool fnvEnvValid() { return fnvEnv[0].x != 0.0; }
float fnvEnvMaxLod() { return fnvEnv[0].y; }
// Irradiance from direction n (world space, unit) relative to the environment's mean: 1 for a uniform environment.
vec3 fnvEnvIrradiance(vec3 n) {
  float y[9] = float[9](0.282095, 0.488603 * n.y, 0.488603 * n.z, 0.488603 * n.x, 1.092548 * n.x * n.y,
                        1.092548 * n.y * n.z, 0.315392 * (3.0 * n.z * n.z - 1.0), 1.092548 * n.x * n.z,
                        0.546274 * (n.x * n.x - n.y * n.y));
  vec3 e = vec3(0.0);
  for (int i = 0; i < 9; i++) e += fnvEnv[1 + i].xyz * y[i];
  return max(e, vec3(0.0));
}
// The environment's radiance in direction r (world space), linear, GGX-prefiltered for roughness alpha (the cube's
// level l holds alpha (l / fnvEnvMaxLod())^2, so the level is sqrt(alpha) times the last).
vec3 fnvEnvRadiance(vec3 r, float alpha) {
  float lod = sqrt(clamp(alpha, 0.0, 1.0)) * fnvEnvMaxLod();
  return max(textureLod(samplerCube(fnvEnvCube, fnvEnvSampler), r, lod).rgb, vec3(0.0));
}
#endif
#ifdef FNV_AO
// Ambient occlusion and screen-space indirect light (FNV_AO; docs/SHADER_ABI.md "Ambient occlusion"): what effects
// published for the lighting in the previous frame, reprojected to this fragment: its view position from gl_FragCoord in
// the current camera, moved into the publishing run's camera, projected into the images. Each is validated by its own
// published depth: the ambient occlusion image's g (effects/gtao), the indirect light's depth image (effects/ssgi).
// Where that run did not see this surface (its distance there differs by more than about 2%), or outside its view, the
// fragment gets no occlusion and no screen-space light.
layout(set = 1, binding = 11, scalar) uniform FnvAo { vec4 fnvAo[12]; };
layout(set = 1, binding = 52) uniform texture2D fnvAoImage;
layout(set = 1, binding = 53) uniform sampler fnvAoSampler;
layout(set = 1, binding = 54) uniform texture2D fnvGiImage;
layout(set = 1, binding = 55) uniform texture2D fnvGiDepthImage;
// The fragment in the publishing run's view: its normalised viewport position and its view distance there; false
// outside that view.
bool fnvReproject(out vec2 uv, out float dist) {
  uv = vec2(0.0);
  dist = 0.0;
  if (fnvAo[0].x == 0.0) return false;
  vec4 rect = fnvAo[1], fr = fnvAo[2];
  float zn = fnvAo[3].x, zf = fnvAo[3].y;
  float z = zn * zf / (zf - gl_FragCoord.z * (zf - zn));
  float u = (gl_FragCoord.x - rect.x) / rect.z, v = (gl_FragCoord.y - rect.y) / rect.w;
  vec3 p = vec3(mix(fr.x, fr.y, u) * z, mix(fr.z, fr.w, v) * z, z);
  vec3 q = vec3(dot(fnvAo[4].xyz, p) + fnvAo[4].w, dot(fnvAo[5].xyz, p) + fnvAo[5].w, dot(fnvAo[6].xyz, p) + fnvAo[6].w);
  if (q.z <= 0.0) return false;
  vec4 pf = fnvAo[7];
  uv = vec2((q.x / q.z - pf.x) / (pf.y - pf.x), (q.y / q.z - pf.z) / (pf.w - pf.z));
  dist = q.z;
  return all(greaterThanEqual(uv, vec2(0.0))) && all(lessThan(uv, vec2(1.0)));
}
// Published log2 distance at texel c of the ambient occlusion (gi false) or of the indirect light's depth (gi true).
float fnvPublishedDepth(bool gi, ivec2 c) {
  return gi ? texelFetch(sampler2D(fnvGiDepthImage, fnvAoSampler), c, 0).r
            : texelFetch(sampler2D(fnvAoImage, fnvAoSampler), c, 0).g;
}
// The texel of image rect `pr` that shows this surface: the reprojected one when its published distance matches
// (within 0.03 in log2). The camera-only reprojection does not follow objects that moved by themselves (actors): around
// them the publishing run showed something else there (the object where the background is now, or the background where
// the object is). The same surface is usually just beside it: the nearest texel at distances r0..16 r0 texels in 8
// directions whose distance matches. False: none.
bool fnvFindSurface(bool gi, vec4 pr, float r0, vec2 uv, float dist, out ivec2 texel) {
  const float want = log2(dist);
  const vec2 base = pr.xy + uv * pr.zw;
  texel = ivec2(base);
  if (abs(fnvPublishedDepth(gi, texel) - want) <= 0.03) return true;
  const vec2 dirs[8] = vec2[8](vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0), vec2(0.7071, 0.7071),
                               vec2(-0.7071, 0.7071), vec2(0.7071, -0.7071), vec2(-0.7071, -0.7071));
  for (float radius = r0; radius <= 16.0 * r0; radius *= 2.0)
    for (int i = 0; i < 8; i++) {
      const vec2 c = base + dirs[i] * radius;
      if (any(lessThan(c, pr.xy)) || any(greaterThanEqual(c, pr.xy + pr.zw))) continue;
      if (abs(fnvPublishedDepth(gi, ivec2(c)) - want) <= 0.03) {
        texel = ivec2(c);
        return true;
      }
    }
  return false;
}
float fnvAmbientOcclusion() {
  vec2 uv;
  float dist;
  ivec2 t;
  if (fnvAo[0].y == 0.0 || !fnvReproject(uv, dist)) return 1.0;
  // Searched 2..32 texels (the image is at the target's resolution).
  if (!fnvFindSurface(false, fnvAo[8], 2.0, uv, dist, t)) return 1.0;
  return clamp(texelFetch(sampler2D(fnvAoImage, fnvAoSampler), t, 0).r, 0.0, 1.0);
}
// The screen-space indirect light (docs/INDIRECT.md "SSGI"): rgb the light that rays from this surface found on screen
// (irradiance / pi: the units of the game's light colours, to multiply by the albedo), a the fraction of the rays that
// found nothing (the probe's ambient lights that part). False without it.
bool fnvScreenIndirect(out vec4 gi) {
  gi = vec4(0.0, 0.0, 0.0, 1.0);
  vec2 uv;
  float dist;
  ivec2 t;
  if (fnvAo[11].x == 0.0 || !fnvReproject(uv, dist)) return false;
  // Searched 1..16 texels: the image is at half resolution (the same distance on screen as the ambient occlusion's).
  if (!fnvFindSurface(true, fnvAo[10], 1.0, uv, dist, t)) return false;
  gi = texelFetch(sampler2D(fnvGiImage, fnvAoSampler), t, 0);
  gi = vec4(max(gi.rgb, vec3(0.0)), clamp(gi.a, 0.0, 1.0));
  return true;
}
// Specular occlusion from the ambient occlusion (Lagarde and de Rousiers 2014): N.V and perceptual roughness.
float fnvSpecularOcclusion(float ao, float nv, float roughness) {
  return clamp(pow(nv + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao, 0.0, 1.0);
}
#else
float fnvAmbientOcclusion() { return 1.0; }
float fnvSpecularOcclusion(float ao, float nv, float roughness) { return 1.0; }
bool fnvScreenIndirect(out vec4 gi) { gi = vec4(0.0, 0.0, 0.0, 1.0); return false; }
#endif
#ifdef FNV_WET
// Wet world (FNV_WET; docs/WETWORLD.md, docs/SHADER_ABI.md "Wet world"). The material model, the ripples and their
// constants follow VegasShaders' Wetness.hlsl (GPL-3.0), itself after Community Shaders' Wetness Effects (GPL-3.0), after
// Lagarde's "Water drop" articles: a porous surface darkens and gains a glossy water film; standing water is a flat,
// near-mirror film (F0 0.02) that hides the material's bumps and darkens with depth. Here the standing water lies where
// the terrain holds it: the flooded heightfield's basins (plugin/src/puddlemap.h), filled to the weather's level.
// Block 12 (vec4[16]):
//   [0]      x 1 while the world pass has a state, y rain falling 0..1, z surface wetness 0..1, w local puddle level 0..1
//   [1]      the camera's world position, w seconds
//   [2]      x the share of flat open ground local puddles cover at a full level, y 1 with a rain cover, z its layer in
//            the shadow array, w ripple strength
//   [3..5]   camera-relative world -> cover clip rows
//   [6]      x the cover's depth bias, y its texel size
//   [7]      the puddle map's world origin xy, z 1 / texel spacing, w 1 with a map (and the basins' levels, 58)
//   [8]      its size, z 1 while a basin holds water
//   [9]      the camera viewport in target pixels; [10] frustum tangents; [11] near, far
//   [12..14] camera rows (world rotation, columns direction, up, right; origin in w)
layout(set = 1, binding = 12, scalar) uniform FnvWetBlock { vec4 fnvWet[16]; };
#ifndef FNV_SUN
layout(set = 1, binding = 48) uniform texture2DArray fnvSunCascades;
layout(set = 1, binding = 49) uniform samplerShadow fnvSunCompare;
#endif
layout(set = 1, binding = 56) uniform texture2D fnvPuddleMap;
layout(set = 1, binding = 57) uniform sampler fnvPuddleSampler;
layout(set = 1, binding = 58) uniform texture2D fnvPuddleLevels;

bool fnvWetActive() { return fnvWet[0].x != 0.0 && max(max(fnvWet[0].y, fnvWet[0].z), max(fnvWet[0].w, fnvWet[8].z)) > 0.0; }
vec3 fnvWetCamera() { return fnvWet[1].xyz; }
// This fragment's world position (game units, z up), from its depth and the camera.
vec3 fnvWetWorldPos() {
  const vec4 rect = fnvWet[9], fr = fnvWet[10];
  const float zn = fnvWet[11].x, zf = fnvWet[11].y;
  const float z = zn * zf / (zf - gl_FragCoord.z * (zf - zn));
  const float u = (gl_FragCoord.x - rect.x) / rect.z, v = (gl_FragCoord.y - rect.y) / rect.w;
  const vec3 l = vec3(z, mix(fr.z, fr.w, v) * z, mix(fr.x, fr.y, u) * z);  // (forward, up, right)
  return vec3(dot(fnvWet[12].xyz, l), dot(fnvWet[13].xyz, l), dot(fnvWet[14].xyz, l)) + fnvWet[1].xyz;
}
// 1 where the rain reaches the point (camera-relative world), 0 under cover: the cover layer holds the highest surface
// over each spot; 2 x 2 compare taps (each a hardware 2 x 2 filter) soften the drip line. Outside it: open.
float fnvWetOpen(vec3 rel) {
  if (fnvWet[2].y == 0.0) return 1.0;
  const vec3 q = vec3(dot(fnvWet[3].xyz, rel) + fnvWet[3].w, dot(fnvWet[4].xyz, rel) + fnvWet[4].w,
                      dot(fnvWet[5].xyz, rel) + fnvWet[5].w);
  const vec2 uv = q.xy * 0.5 + 0.5;
  if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) || q.z < 0.0 || q.z > 1.0) return 1.0;
  const float t = fnvWet[6].y;
  float sum = 0.0;
  for (int i = 0; i < 4; i++) {
    const vec2 o = vec2(float(i & 1) - 0.5, float(i >> 1) - 0.5) * t;
    sum += texture(sampler2DArrayShadow(fnvSunCascades, fnvSunCompare), vec4(uv + o, fnvWet[2].z, q.z - fnvWet[6].x));
  }
  return sum * 0.25;
}
// "Hash without sine" (Dave Hoskins, MIT), as Wetness.hlsl's WetHash33.
vec3 fnvWetHash33(vec3 p3) {
  p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yxx) * p3.zyx);
}
float fnvWetValueNoise(vec2 p) {
  const vec2 i = floor(p), f = p - i, u = f * f * (3.0 - 2.0 * f);
  const float a = fnvWetHash33(vec3(i, 7.13)).x, b = fnvWetHash33(vec3(i + vec2(1.0, 0.0), 7.13)).x;
  const float c = fnvWetHash33(vec3(i + vec2(0.0, 1.0), 7.13)).x, d = fnvWetHash33(vec3(i + vec2(1.0), 7.13)).x;
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
// Reoriented normal mapping: detail n2 laid onto base n1 (Wetness.hlsl WetReorient, Community Shaders' ReorientNormal).
vec3 fnvWetReorient(vec3 n1, vec3 n2) {
  n1 += vec3(0.0, 0.0, 1.0);
  n2 *= vec3(-1.0, -1.0, 1.0);
  return normalize(n1 * dot(n1, n2) / n1.z - n2);
}
float fnvWetSmoothstepDeriv(float x) { return 6.0 * x * (1.0 - x); }
// Rain drops' ripples (Wetness.hlsl WetRainDrops, Community Shaders' GetRainDrops): in cells of 4 game units, a drop
// each second with chance rain^2; its ring spreads one cell over half a second, half a cell wide, fading as it goes. The
// ripple's normal, z up.
vec3 fnvWetRipples(vec3 wp, vec3 normal, float rain, float strength) {
  const float cellSize = 4.0, interval = 1.0, lifetime = 0.5, radius = 1.0, breadth = 0.5;
  const float t = fnvWet[1].w, chance = rain * rain;
  vec2 gridUV = wp.xy / cellSize + normal.xy;
  const vec2 grid = floor(gridUV);
  gridUV -= grid;
  vec3 rippleNormal = vec3(0.0, 0.0, 1.0);
  for (int i = -1; i <= 1; i++)
    for (int j = -1; j <= 1; j++) {
      const vec2 cell = grid + vec2(i, j);
      const float tOffset = fnvWetHash33(vec3(cell, 1.7)).x;
      float residual = t / interval + tOffset + wp.z * 0.001;
      const float timestep = floor(residual);
      residual -= timestep;
      const vec3 h = fnvWetHash33(vec3(cell, timestep + 0.5));
      const float rippleT = residual * (interval / lifetime);
      if (h.z < chance && rippleT < 1.0) {
        const vec2 toCentre = vec2(i, j) + h.xy - gridUV;
        const float size = mix(0.7, 1.3, fnvWetHash33(h * 53.1).y);
        const float rippleR = radius * size * rippleT;
        const float inner = rippleR - breadth;
        const float band = (length(toCentre) - inner) / breadth;
        if (band > 0.0 && band < 1.0) {
          const float deriv = (band < 0.5 ? fnvWetSmoothstepDeriv(band * 2.0) : -fnvWetSmoothstepDeriv(2.0 - band * 2.0)) *
                              mix(strength, 0.0, rippleT * rippleT);
          const vec3 grad = vec3(normalize(toCentre + 1e-5), -deriv);
          const vec3 bitangent = vec3(-grad.y, grad.x, 0.0);
          rippleNormal = fnvWetReorient(normalize(cross(grad, bitangent)), rippleNormal);
        }
      }
    }
  return rippleNormal;
}
// The depth of the terrain basins' water over the point (game units; negative: above the water), from the puddle map
// (per texel: r its basin, 0 none; g the ground, absolute world z) and the basins' levels (58: basin i's water level at
// (i % 256, i / 256)): the water surface is the texel's basin's level (the ground outside basins), bilinear over the
// map's texels; against the higher of the point and the map's own ground (a terrain triangle between two texels lies a
// little off their bilinear surface, which must not read as water). -1e9 without a map or outside it.
float fnvPuddleWater(vec3 wp) {
  if (fnvWet[7].w == 0.0) return -1.0e9;
  const vec2 sz = fnvWet[8].xy;
  const vec2 t = (wp.xy - fnvWet[7].xy) * fnvWet[7].z;
  if (any(lessThan(t, vec2(0.0))) || any(greaterThan(t, sz - 1.0))) return -1.0e9;
  const ivec2 i0 = ivec2(floor(t));
  const ivec2 i1 = min(i0 + 1, ivec2(sz) - 1);
  const vec2 f = t - vec2(i0);
  float water = 0.0, ground = 0.0;
  for (int k = 0; k < 4; k++) {
    const ivec2 c = ivec2((k & 1) != 0 ? i1.x : i0.x, (k & 2) != 0 ? i1.y : i0.y);
    const vec4 m = texelFetch(sampler2D(fnvPuddleMap, fnvPuddleSampler), c, 0);
    if (m.g < -1.0e8) return -1.0e9;  // no ground there (a cell not loaded)
    const int id = int(m.r + 0.5);
    const float level = id > 0 ? texelFetch(sampler2D(fnvPuddleLevels, fnvPuddleSampler), ivec2(id % 256, id / 256), 0).r : m.g;
    const float w = ((k & 1) != 0 ? f.x : 1.0 - f.x) * ((k & 2) != 0 ? f.y : 1.0 - f.y);
    water += level * w;
    ground += m.g * w;
  }
  return water - max(wp.z, ground);
}

struct FnvWetSurface {
  float darkening;  // how far the albedo darkens (fnvWetAlbedo)
  float gloss;      // how much water film covers the surface 0..1
  float roughness;  // the film's roughness
  float flatten;    // how far standing water hides the material's own normal 0..1
  float murk;       // how deep the standing water is 0..1 (tints what it reflects)
  vec3 normal;      // the film's normal (world)
};
// The water on a surface at world position wp with geometric normal geomN and shading normal n (world, unit); relief:
// the surface's own height detail 0..1 (0.5 its average; lower fills first), negative without one. Wetness.hlsl's three
// zones (GetWetness, its Puddles rework):
// - wet ground: the rain's wetness, on what the rain reaches (faces turned from the sky get at least 0.65 of it): darker,
//   glossy (roughness toward 0.22), the material's bumps kept;
// - standing water, from two sources: the rain settling where it falls (local puddles: where a world-space field, value
//   noise in a 420-unit pattern plus the surface's relief, rises over a threshold that falls as the local level rises,
//   on flat open ground: Wetness.hlsl PuddleField, coverage fnvWet[2].x at a full level, so shallow spots fill first and
//   dry first), and the terrain basins' runoff (fnvPuddleWater): flat, mirror-like, darker with depth;
// - a soaked ring around standing water;
// ripples where the rain falls, at full strength on standing water.
// pool: 1 for surfaces water can stand on (ground, objects), 0 for those it runs off at once (skin, as Community
// Shaders' skin: no puddle effects, 0.95 of the wetness): no standing water, no soaked ring, no ripples.
FnvWetSurface fnvWetness(vec3 wp, vec3 geomN, vec3 n, float relief, float pool) {
  FnvWetSurface s;
  s.darkening = 0.0;
  s.gloss = 0.0;
  s.roughness = 1.0;
  s.flatten = 0.0;
  s.murk = 0.0;
  s.normal = n;
  if (!fnvWetActive()) return s;
  const vec3 rel = wp - fnvWetCamera();
  const float dist = length(rel);
  // Open to the rain: the air just off the face (32 units out sideways, or down from a face turned down), so a wall
  // in the open is wet and a bridge's underside is not.
  const float open = fnvWetOpen(rel + vec3(geomN.xy, min(geomN.z, 0.0)) * 32.0);
  const float up = clamp(geomN.z, 0.0, 1.0);
  // Skin (pool 0) takes 0.95 of the wetness (Community Shaders' Wetness Effects SkinWetness).
  const float rain = clamp(fnvWet[0].z * max(0.65, up) * (pool > 0.0 ? 1.0 : 0.95), 0.0, 1.0) * open;
  float darkening = rain * rain * 0.7;
  float rough = mix(1.0, 0.22, rain);
  float standing = 0.0, deep = 0.0, ring = 0.0;
  if (open > 0.0 && up > 0.5 && pool > 0.0) {
    // The shoreline's wander (0..1): the softness and the rim's width vary with it.
    const float wander = fnvWetValueNoise(wp.xy / 113.0 + 5.1);
    // Local puddles.
    const float level = fnvWet[0].w;
    if (level > 0.001) {
      const vec2 p = wp.xy / 420.0;
      float field = smoothstep(0.2, 0.8, fnvWetValueNoise(p) * 0.62 + fnvWetValueNoise(p * 2.31 + 17.7) * 0.38);
      if (relief >= 0.0) field += 0.6 * (0.5 - relief);
      const float t = 1.0 - level * fnvWet[2].x;
      const float soft = 0.05 * (0.5 + wander);
      const float slope = smoothstep(0.92, 0.97, up);
      const float shore = clamp((field - t) / soft, 0.0, 1.0);
      standing = shore * shore * (3.0 - 2.0 * shore) * slope * open;
      deep = clamp((field - t) / (soft * 5.0), 0.0, 1.0) * standing;
      const float ringW = 0.12 * (0.35 + 1.3 * wander);
      ring = clamp((field - t + ringW) / ringW, 0.0, 1.0) * smoothstep(0.77, 0.92, up) * open * clamp(level * 4.0, 0.0, 1.0);
    }
    // The terrain basins' water.
    const float water = fnvPuddleWater(wp) + (wander - 0.5) * 3.0;
    if (water > -6.0) {
      const float slope = smoothstep(0.6, 0.8, up);
      const float basin = smoothstep(0.0, 1.5, water) * slope * open;
      deep = max(deep, clamp(water / 8.0, 0.0, 1.0) * basin);
      standing = max(standing, basin);
      ring = max(ring, clamp((water + 6.0) / 6.0, 0.0, 1.0) * slope * open);
    }
    ring *= ring * (1.0 - standing);
  }
  s.darkening = mix(max(darkening, ring * 0.85), 1.0 + 0.6 * deep, standing);
  rough = mix(rough, 0.12, ring);
  const float film = max(max(rain, ring), standing);
  vec3 wetN = normalize(mix(n, geomN, standing));
  if (fnvWet[0].y > 0.0 && open > 0.5 && geomN.z > 0.0 && dist < 1000.0 && pool > 0.0) {
    const vec3 ripple = fnvWetRipples(wp, geomN, fnvWet[0].y, fnvWet[2].w);
    const float rippleW = clamp(max(standing, smoothstep(0.5, 1.0, film) * 0.5), 0.0, 1.0);
    wetN = fnvWetReorient(wetN, normalize(mix(vec3(0.0, 0.0, 1.0), ripple, rippleW)));
  }
  s.gloss = clamp(film, 0.0, 1.0);
  s.roughness = mix(max(rough, 0.05), 0.02, standing);
  s.flatten = standing;
  s.murk = deep;
  s.normal = wetN;
  return s;
}
// Wet albedo (Community Shaders): lerp(albedo, albedo^(1 + darkening), 0.5); albedo in linear light.
vec3 fnvWetAlbedo(vec3 albedo, float darkening) { return mix(albedo, pow(max(albedo, 0.0), vec3(1.0 + darkening)), 0.5); }
// The film's highlight from one light of linear colour c (irradiance pi c, the game's convention): GGX, F0 0.02.
vec3 fnvWetSpecular(vec3 N, vec3 V, vec3 L, vec3 c, float roughness) {
  const vec3 H = normalize(L + V);
  const float NdotL = clamp(dot(N, L), 0.0, 1.0), NdotV = clamp(dot(N, V), 0.0, 1.0) + 1e-4;
  const float NdotH = clamp(dot(N, H), 0.0, 1.0), VdotH = clamp(dot(V, H), 0.0, 1.0);
  const float a = roughness * roughness, a2 = a * a;
  const float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
  const float D = a2 / (3.14159265 * d * d);
  const float k = a * 0.5;
  const float vis = 0.25 / ((NdotL * (1.0 - k) + k) * (NdotV * (1.0 - k) + k));
  const float F = 0.02 + 0.98 * pow(1.0 - VdotH, 5.0);
  return c * (3.14159265 * D * vis * F * NdotL);
}
// The film's reflectance (Karis' split-sum fit for F0 0.02), times how much film there is.
float fnvWetReflectance(float NdotV, float roughness, float gloss) {
  const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022), c1 = vec4(1.0, 0.0425, 1.04, -0.04);
  const vec4 r = roughness * c0 + c1;
  const float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
  const vec2 AB = vec2(-1.04, 1.04) * a004 + r.zw;
  return (0.02 * AB.x + AB.y) * gloss;
}
// What standing water tints the light it reflects: clear at the shore, a muddy brown-grey where deep (PuddleMurkTint).
vec3 fnvWetMurk(float murk) { return mix(vec3(1.0), vec3(0.62, 0.55, 0.44), murk); }
#else
bool fnvWetActive() { return false; }
#endif
#ifdef FNV_FRAME
// The tangent-to-world frame of the lighting vertex shaders (sdp2spv; docs/SHADER_ABI.md "View vector"): rows at
// locations 13..15, zero where the vertex shader has none. fnvToWorld(t) is a tangent-space direction in world space
// (unit), or vec3(0) without a frame.
layout(location = 13) in vec4 fnvFrame0;
layout(location = 14) in vec4 fnvFrame1;
layout(location = 15) in vec4 fnvFrame2;
vec3 fnvToWorld(vec3 t) {
  vec3 w = vec3(dot(fnvFrame0.xyz, t), dot(fnvFrame1.xyz, t), dot(fnvFrame2.xyz, t));
  float l = dot(w, w);
  return l > 1e-20 ? w * inversesqrt(l) : vec3(0.0);
}
// A world direction in tangent space (the frame's transpose), unit; vec3(0) without a frame.
vec3 fnvToTangent(vec3 w) {
  vec3 t = fnvFrame0.xyz * w.x + fnvFrame1.xyz * w.y + fnvFrame2.xyz * w.z;
  float l = dot(t, t);
  return l > 1e-20 ? t * inversesqrt(l) : vec3(0.0);
}
#endif
#ifdef FNV_SKIN
// Skin (FNV_SKIN; docs/SKIN.md), ported from Community Shaders' Skin feature (GPL-3.0: features/Skin/Shaders/Skin/
// Skin.hlsli, package/Shaders/Lighting.hlsl's CS_SKIN_SHADING, Common/BRDF.hlsli, Common/Shading.hlsli) with its default
// profile (src/Features/Skin/Material/Parameters.def) and no light disks: two GGX lobes (roughness 0.7 less the gloss
// and 0.35, mixed 0.15; rougher at grazing angles), Burley diffuse, a Charlie sheen (fuzz), reflectance 0.0278, and the
// detail normal map tiled over the skin (strength 0.25, tiling 10). Lights follow the game's convention (a light's
// colour c is the diffuse light on a white surface facing it: irradiance pi c).
layout(set = 1, binding = 59) uniform texture2D fnvSkinDetailMap;
layout(set = 1, binding = 60) uniform sampler fnvSkinDetailSampler;
// The skin buffer (docs/SKIN.md "Subsurface scattering"; Backend::setSkinBuffer): block 13 x 1 while it takes this
// draw's pixels, y z its size; 61 each skin pixel's albedo, 62 the nearest skin depth drawn there (the depth's bits).
layout(set = 1, binding = 13, scalar) uniform FnvSkinBlock { vec4 fnvSkinBuffer; };
layout(set = 1, binding = 61, rgba8) uniform writeonly image2D fnvSkinAlbedoOut;
layout(set = 1, binding = 62, r32ui) uniform uimage2D fnvSkinDepthOut;
// This pixel as skin of albedo `albedo` (gamma): its depth kept if it is the nearest skin drawn there, and with it its
// albedo (two skin fragments racing at one pixel may leave the farther one's albedo with the nearer depth: both skin).
// Call it after any discard.
void fnvSkinStore(vec3 albedo) {
  if (fnvSkinBuffer.x == 0.0) return;
  const ivec2 p = ivec2(gl_FragCoord.xy);
  if (any(greaterThanEqual(p, ivec2(fnvSkinBuffer.yz)))) return;
  const uint z = floatBitsToUint(clamp(gl_FragCoord.z, 0.0, 1.0));
  if (imageAtomicMin(fnvSkinDepthOut, p, z) >= z) imageStore(fnvSkinAlbedoOut, p, vec4(clamp(albedo, 0.0, 1.0), 1.0));
}
const float kSkinRoughness = 0.7, kSkinRoughness2 = 0.35, kSkinLobeMix = 0.15, kSkinF0 = 0.0278, kSkinEdgeRoughness = 0.25;
const float kSkinFuzz = 1.0, kSkinFuzzRoughness = 0.35, kSkinFuzzF0 = 0.045;
const float kSkinDetailStrength = 0.25, kSkinDetailTiling = 10.0;
const float kSkinPi = 3.14159265;
struct FnvSkin {
  float rough, rough2, avg;  // the two lobes' roughness and the visibility term's (their mix)
  float energy;              // multiple-scattering compensation of the specular (GGXEnergyCompensation)
};
float fnvSkinSchlick(float f0, float vh) {
  const float fc = pow(1.0 - vh, 5.0);
  return fc + (1.0 - fc) * f0;
}
vec2 fnvSkinEnvBrdf(float roughness, float nv) {  // Lazarov 2013
  const vec4 r = roughness * vec4(-1.0, -0.0275, -0.572, 0.022) + vec4(1.0, 0.0425, 1.04, -0.04);
  const float a004 = min(r.x * r.x, exp2(-9.28 * nv)) * r.x + r.y;
  return vec2(-1.04, 1.04) * a004 + r.zw;
}
float fnvSkinGgx(float roughness, float nh) {
  const float a = roughness * roughness, a2 = a * a;
  const float d = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / (kSkinPi * d * d);
}
// The material at a pixel; gloss: the game's specular map (0 for a skin without the game's specular flag); nv: N.V.
FnvSkin fnvSkinMaterial(float gloss, float nv) {
  FnvSkin m;
  const float edge = fnvSkinSchlick(0.04, clamp(nv, 0.0, 1.0)) * kSkinEdgeRoughness;
  m.rough = clamp(clamp(kSkinRoughness - gloss, 0.0, 1.0) + edge, 0.02, 1.0);
  m.rough2 = clamp(kSkinRoughness2 + edge, 0.02, 1.0);
  m.avg = mix(m.rough, m.rough2, kSkinLobeMix);
  const float NdotV = clamp(abs(nv) + 1e-5, 0.0, 1.0);
  const float a = m.avg * m.avg;
  const float t = NdotV / (NdotV + 1.103665 * a);
  const float x = 2.0 * t - 1.0;
  const float grazing = 1.424916 + x * (0.341704 + x * (-0.646581 - 0.742605 * x)) - m.avg * (0.847600 + 1.693598 * x);
  const float rough = 1.841562 + 0.210406 * NdotV - 0.899554 * m.avg;
  const float single = clamp(1.0 - t * ((1.0 - t) * grazing + a * a * rough), 0.0, 1.0);
  m.energy = 1.0 + kSkinF0 * (1.0 / single - 1.0);
  return m;
}
// The sheen of one light (Skin::CharlieSpecular): Charlie distribution, Neubelt visibility.
float fnvSkinCharlie(vec3 N, vec3 V, vec3 L, float NdotV) {
  const vec3 H = normalize(V + L);
  const float NdotL = clamp(dot(N, L), 1e-5, 1.0);
  const float invAlpha = pow(kSkinFuzzRoughness, -4.0);
  const float nh = clamp(dot(N, H), 0.0, 1.0);
  const float D = (2.0 + invAlpha) * pow(max(1.0 - nh * nh, 0.0), invAlpha * 0.5) / (2.0 * kSkinPi);
  const float vis = 1.0 / (4.0 * max(NdotL + NdotV - NdotL * NdotV, 1e-6));
  return D * vis * fnvSkinSchlick(kSkinFuzzF0, clamp(dot(V, H), 0.0, 1.0)) * NdotL;
}
// One light (Skin::SkinDirectLightInput): unit direction L, linear colour c; adds its diffuse light (to be multiplied by
// the albedo) and its specular.
void fnvSkinLight(FnvSkin m, vec3 N, vec3 V, vec3 L, vec3 c, inout vec3 diffuse, inout vec3 specular) {
  const vec3 E = c * kSkinPi;
  const vec3 H = normalize(V + L);
  const float diffuseCos = max(clamp(dot(N, L), 0.0, 1.0), 1e-5);
  const float NdotL = clamp(dot(N, L), 1e-5, 1.0);
  const float NdotV = clamp(abs(dot(N, V)) + 1e-5, 0.0, 1.0);
  const float NdotH = clamp(dot(N, H), 0.0, 1.0), VdotH = clamp(dot(V, H), 0.0, 1.0);
  // Burley diffuse (with its 1 / pi).
  const float fd90 = 0.5 + 2.0 * VdotH * VdotH * m.avg;
  const float burley = (1.0 + (fd90 - 1.0) * pow(1.0 - NdotV, 5.0)) * (1.0 + (fd90 - 1.0) * pow(1.0 - diffuseCos, 5.0)) / kSkinPi;
  vec3 d = E * diffuseCos * burley;
  // Two GGX lobes, Smith joint visibility (approximate) at the mixed roughness, Schlick Fresnel.
  const float D = mix(fnvSkinGgx(m.rough, NdotH), fnvSkinGgx(m.rough2, NdotH), kSkinLobeMix);
  const float a = m.avg * m.avg;
  const float G = 0.5 / max(NdotL * (NdotV * (1.0 - a) + a) + NdotV * (NdotL * (1.0 - a) + a), 1e-6);
  const float F = fnvSkinSchlick(kSkinF0, VdotH);
  vec3 s = E * (D * G * F * NdotL * m.energy);
  d *= 1.0 - F;
  // The sheen, energy-conserving against the rest.
  const vec2 fb = fnvSkinEnvBrdf(kSkinFuzzRoughness, NdotV);
  const vec3 fuzz = E * (fnvSkinCharlie(N, V, L, NdotV) * (1.0 + kSkinFuzzF0 * (1.0 / (fb.x + fb.y) - 1.0)));
  const float retained = 1.0 - clamp(fnvSkinSchlick(kSkinFuzzF0, VdotH) * kSkinFuzz, 0.0, 1.0);
  diffuse += d * retained;
  specular += s * retained + fuzz * kSkinFuzz;
}
// The ambient's weights (Skin::SkinIndirectLobeWeights, without its AO): x the diffuse (times the albedo), y the
// specular (times the reflected radiance).
vec2 fnvSkinIndirect(FnvSkin m, float nv) {
  const vec2 ab = fnvSkinEnvBrdf(m.avg, clamp(nv, 0.0, 1.0));
  const float spec = kSkinF0 * ab.x + ab.y;
  return vec2(1.0 - clamp(spec, 0.0, 1.0), spec * (1.0 + kSkinF0 * (1.0 / (ab.x + ab.y) - 1.0)));
}
// The shading normal (tangent space, unit) with the detail normal map reoriented onto it (Reoriented Normal Mapping,
// Barre-Brisebois and Hill 2012); unchanged without a map (the zero dummy).
vec3 fnvSkinDetail(vec3 n, vec2 uv) {
  const vec2 raw = texture(sampler2D(fnvSkinDetailMap, fnvSkinDetailSampler), uv * kSkinDetailTiling).xy;
  if (raw.x + raw.y <= 0.0) return n;
  vec3 d = vec3(raw * 2.0 - 1.0, 0.0);
  d.z = sqrt(clamp(1.0 - dot(d.xy, d.xy), 0.0, 1.0));
  d = normalize(vec3(d.xy * kSkinDetailStrength, max(d.z, 1e-4)));
  const vec3 n1 = n + vec3(0.0, 0.0, 1.0), n2 = d * vec3(-1.0, -1.0, 1.0);
  return normalize(n1 * dot(n1, n2) / n1.z - n2);
}
#endif
#ifdef FNV_HAIR
// Hair (FNV_HAIR; docs/LIGHTING.md "Hair"), ported from Community Shaders' Hair Specular (GPL-3.0: features/Hair
// Specular/Shaders/Hair/Hair.hlsli, package/Shaders/Lighting.hlsl's CS_HAIR_SHADING, Common/LightDisk.hlsli without light
// disks) with its default settings (src/Features/HairSpecular.h): the Marschner model (glossiness 70: roughness 0.3;
// transmission 1; diffuse, specular and their indirect multipliers 1), the tangent shift map, and the screen-space self
// shadow (strength 1, exponent 0.1, scale 2.5). Lights follow the game's convention (a light's colour c is the diffuse
// light on a white surface facing it: irradiance pi c, Community Shaders' PBRLightingCompensation).
layout(set = 1, binding = 63) uniform texture2D fnvHairShiftMap;
layout(set = 1, binding = 64) uniform sampler fnvHairShiftSampler;
const float kHairRoughness = 0.3, kHairTransmission = 1.0, kHairBaseColorMult = 1.5;
const float kHairSelfShadowStrength = 1.0, kHairSelfShadowExponent = 0.1, kHairSelfShadowScale = 2.5;
const float kHairPi = 3.14159265;
const float kHairF0 = ((1.0 - 1.55) / (1.0 + 1.55)) * ((1.0 - 1.55) / (1.0 + 1.55));  // HairF0: index of refraction 1.55
float fnvHairSchlick(float vh) {
  const float fc = pow(1.0 - vh, 5.0);
  return fc + (1.0 - fc) * kHairF0;
}
float fnvHairGauss(float b, float theta) { return exp(-0.5 * theta * theta / (b * b)) / (sqrt(2.0 * kHairPi) * b); }
// The model's base colour from the hair's albedo (gamma, tinted). Community Shaders saturates it (saturation 1), scales
// it by 1.5 and squares it, and shows the model's output as it is (gamma): its colour terms are powers of the base
// (the diffuse scatter sqrt(base), the absorption base^k), so the displayed colour is the albedo's to those powers.
// Here the output is linear (shown through 1 / 2.2): the same base dependence needs the linear colour squared. With the
// linear colour alone the scatter's sqrt halves the colour's saturation (brown hair renders grey).
vec3 fnvHairBaseColor(vec3 albedo) {
  const vec3 lin = pow(clamp(albedo, 0.0, 1.0) * kHairBaseColorMult, vec3(2.2));
  return lin * lin;
}
// The strand direction in tangent space (CS: the mesh's bitangent reoriented to the shading normal n, unit), shifted along
// n by the tangent shift map (centred on 0.5; no shift without the map: the zero dummy).
vec3 fnvHairStrand(vec3 n, vec2 uv) {
  vec3 t = vec3(0.0, 1.0, 0.0) - n * n.y;
  t = dot(t, t) > 1e-8 ? normalize(t) : normalize(vec3(1.0, 0.0, 0.0) - n * n.x);
  const float s = textureLod(sampler2D(fnvHairShiftMap, fnvHairShiftSampler), uv, 0.0).x;
  return s > 0.0 ? normalize(t + n * (s - 0.5)) : t;
}
// Hair::D_Marschner: L, V unit; T the strand (unit); base the linear base colour; area the lobes' widening; backlit the
// transmitted lobe's weight.
vec3 fnvHairMarschner(vec3 L, vec3 V, vec3 T, float roughness, vec3 base, float area, float backlit) {
  const float TdotL = dot(T, L), TdotV = dot(T, V), VdotL = dot(V, L);
  const float cosThetaL = sqrt(max(0.0, 1.0 - TdotL * TdotL));
  const float cosThetaV = sqrt(max(0.0, 1.0 - TdotV * TdotV));
  const float cosThetaD = sqrt((1.0 + cosThetaL * cosThetaV + TdotV * TdotL) / 2.0);
  const vec3 Lp = L - TdotL * T, Vp = V - TdotV * T;
  const float cosPhi = dot(Lp, Vp) * inversesqrt(dot(Lp, Lp) * dot(Vp, Vp) + 1e-6);
  const float cosHalfPhi = sqrt(clamp(0.5 + 0.5 * cosPhi, 0.0, 1.0));
  const float nPrime = 1.19 / cosThetaD + 0.36 * cosThetaD;
  const vec3 alpha = vec3(-0.0499 * 2.0, 0.0499, 0.0499 * 4.0);
  const vec3 B = area + roughness * vec3(1.0, 0.5, 2.0);
  const float thetaH = TdotL + TdotV;
  // R
  float Mp = fnvHairGauss(B.x, thetaH - alpha.x);
  float Np = 0.25 * cosHalfPhi;
  float Fp = fnvHairSchlick(sqrt(clamp(0.5 + 0.5 * VdotL, 0.0, 1.0)));
  const vec3 R = vec3(Mp * Np * Fp * mix(1.0, backlit, clamp(-VdotL, 0.0, 1.0)));
  // TT
  Mp = fnvHairGauss(B.y, thetaH - alpha.y);
  const float a = 1.0 / nPrime;
  const float h = cosHalfPhi * (1.0 + a * (0.6 - 0.8 * cosPhi));
  float f = fnvHairSchlick(cosThetaD * sqrt(clamp(1.0 - h * h, 0.0, 1.0)));
  Fp = (1.0 - f) * (1.0 - f);
  vec3 Tp = pow(abs(base), vec3(0.5 * sqrt(max(1.0 - (h * a) * (h * a), 0.0)) / cosThetaD));
  Np = exp(-3.65 * cosPhi - 3.98);
  const vec3 TT = Mp * Np * Fp * Tp * backlit;
  // TRT
  Mp = fnvHairGauss(B.z, thetaH - alpha.z);
  f = fnvHairSchlick(cosThetaD * 0.5);
  Fp = (1.0 - f) * (1.0 - f) * f;
  Tp = pow(abs(base), vec3(0.8 / cosThetaD));
  Np = exp(17.0 * cosPhi - 16.78);
  const vec3 TRT = Mp * Np * Fp * Tp;
  return max(R + TT + TRT, 0.0);
}
// Hair::GetHairDiffuseAttenuationKajiyaKay: the diffuse scatter (with its 1 / pi), tinted by the base colour where the
// light is shadowed.
vec3 fnvHairKajiya(vec3 T, vec3 V, vec3 L, float shadow, vec3 base) {
  const float TdotL = dot(T, L), TdotV = dot(T, V);
  const float kajiya = 1.0 - abs(TdotL);
  const vec3 fake = V - T * TdotV;
  const vec3 fakeN = dot(fake, fake) > 1e-12 ? normalize(fake) : vec3(0.0);
  const float wrapped = clamp((dot(fakeN, L) + 1.0) * 0.25, 0.0, 1.0);
  const float scatter = (1.0 / kHairPi) * mix(wrapped, kajiya, 0.33);
  const float luma = max(dot(base, vec3(0.2125, 0.7154, 0.0721)), 1e-4);
  const vec3 tint = shadow < 1.0 ? pow(abs(base / luma), vec3(1.0 - shadow)) : vec3(1.0);
  return max(sqrt(base) * scatter * tint, 0.0);
}
// One light (Hair::GetHairDirectLightMarschner): L unit, c its linear colour (unshadowed), shadow its visibility (the
// shadow maps and the self shadow); the light leaving the hair toward V (the base colour is in it).
vec3 fnvHairLight(vec3 T, vec3 V, vec3 L, vec3 c, float shadow, vec3 base) {
  return (fnvHairMarschner(L, V, T, kHairRoughness, base, 0.0, kHairTransmission) + fnvHairKajiya(T, V, L, shadow, base)) *
         (c * kHairPi * shadow);
}
// The ambient's weight (Hair::GetHairIndirectLobeWeights, Marschner): light from the direction across the strand facing
// the view; times the ambient irradiance / pi (the game's ambient colour).
vec3 fnvHairIndirect(vec3 T, vec3 V, vec3 base) {
  const vec3 across = V - T * dot(V, T);
  if (dot(across, across) <= 1e-12) return vec3(0.0);
  const vec3 L = normalize(across);
  return (fnvHairMarschner(L, V, T, kHairRoughness, base, 0.2, 0.0) + fnvHairKajiya(T, V, L, 1.0, base)) * kHairPi;
}
#ifdef FNV_AO
// The self shadow (Hair::HairSelfShadow): 4 steps toward the light over 2.5 cm (game units: 1.428 cm each), from a
// screen-noise start, against the scene's depth. The depth is what the lighting has: the ambient occlusion's published
// depth (the previous frame's, reprojected as fnvAmbientOcclusion does); none while the ambient occlusion is off.
// The light's direction comes in the hair's own space (the shader's light vectors: the model or skin space of its
// position input); its view-space direction is that space's map to the view, recovered from the screen derivatives of the
// fragment's view position and of its position in that space (exact within a triangle: helper lanes interpolate the same
// plane). That map is a rotation and a uniform scale composed with the view's mirror (view x right, y up, z forward from
// the game's right-handed world: fnvWorldPos swaps x and z), so the face normal maps to minus the view-space face normal.
struct FnvHairFrame {
  vec3 p;               // the fragment's view position
  vec3 dpx, dpy;        // its screen derivatives
  vec3 dmx, dmy;        // the screen derivatives of its position in the hair's space
};
// At the top of main (screen derivatives), with the position input in the hair's space.
FnvHairFrame fnvHairFrame(vec3 position) {
  FnvHairFrame f;
  const vec4 rect = fnvAo[1], fr = fnvAo[2];
  const float zn = fnvAo[3].x, zf = fnvAo[3].y;
  const float z = zn * zf / (zf - gl_FragCoord.z * (zf - zn));
  const float u = (gl_FragCoord.x - rect.x) / rect.z, v = (gl_FragCoord.y - rect.y) / rect.w;
  f.p = vec3(mix(fr.x, fr.y, u) * z, mix(fr.z, fr.w, v) * z, z);
  f.dpx = dFdx(f.p);
  f.dpy = dFdy(f.p);
  f.dmx = dFdx(position);
  f.dmy = dFdy(position);
  return f;
}
// A direction in the hair's space, in view space (unit); vec3(0) where the derivatives are degenerate.
vec3 fnvHairToView(FnvHairFrame f, vec3 d) {
  const vec3 nm = cross(f.dmx, f.dmy), nv = cross(f.dpx, f.dpy);
  const float det = dot(nm, nm), nvLen = length(nv);
  if (det < 1e-30 || nvLen < 1e-30) return vec3(0.0);
  const float k = sqrt(nvLen * inversesqrt(det));  // the uniform scale: |nv| = k^2 |nm|
  const float a = dot(cross(f.dmy, nm), d) / det, b = dot(cross(nm, f.dmx), d) / det, c = dot(nm, d) / det;
  const vec3 w = a * f.dpx + b * f.dpy - c * nv / k;
  return dot(w, w) > 1e-30 ? normalize(w) : vec3(0.0);
}
float fnvHairSelfShadow(FnvHairFrame f, vec3 lightDir) {
  if (fnvAo[0].x == 0.0 || fnvAo[0].y == 0.0) return 1.0;
  const vec3 l = fnvHairToView(f, lightDir) * max(kHairSelfShadowScale * 1.428, 0.05);
  if (dot(l, l) == 0.0) return 1.0;
  const float noise = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
  const float stepSize = 0.25;
  vec3 ray = f.p + l * ((noise - 0.5) * 2.0 * stepSize);
  const vec4 pf = fnvAo[7], pr = fnvAo[8];
  int hits = 0;
  for (int i = 0; i < 4; i++) {
    ray += l * stepSize;
    const vec3 q = vec3(dot(fnvAo[4].xyz, ray) + fnvAo[4].w, dot(fnvAo[5].xyz, ray) + fnvAo[5].w,
                        dot(fnvAo[6].xyz, ray) + fnvAo[6].w);
    if (q.z <= 0.0) continue;
    const vec2 uv = vec2((q.x / q.z - pf.x) / (pf.y - pf.x), (q.y / q.z - pf.z) / (pf.w - pf.z));
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThanEqual(uv, vec2(1.0)))) continue;
    // Nearer than the ray (log2 distances; 0.002: the published depth's precision, about 0.14%).
    if (fnvPublishedDepth(false, ivec2(pr.xy + uv * pr.zw)) < log2(q.z) - 0.002) hits++;
  }
  const float shadow = hits > 0 ? 1.0 - pow(float(hits) / 4.0, kHairSelfShadowExponent) : 1.0;
  return mix(1.0, shadow, kHairSelfShadowStrength);
}
#else
struct FnvHairFrame { vec3 p; };
FnvHairFrame fnvHairFrame(vec3 position) { FnvHairFrame f; f.p = vec3(0.0); return f; }
float fnvHairSelfShadow(FnvHairFrame f, vec3 lightDir) { return 1.0; }
#endif
#endif
#if defined(FNV_WET) && defined(FNV_FRAME)
// Wet world on objects (docs/WETWORLD.md), in the object lighting shaders' tangent space with their vertex shaders'
// frame: fnvWetObjectBegin before the lights (it flattens the shading normal n, tangent space, under standing water and
// darkens the albedo, gamma space; relief as fnvWetness), fnvWetObjectFilm over the lit colour (linear) after them.
struct FnvWetObject {
  FnvWetSurface s;
  vec3 wp;
  bool on;
};
FnvWetObject fnvWetObjectBegin(inout vec3 n, inout vec3 albedo, float relief, float pool) {
  FnvWetObject o;
  o.on = false;
  o.wp = vec3(0.0);
  o.s.darkening = 0.0;
  o.s.gloss = 0.0;
  o.s.roughness = 1.0;
  o.s.flatten = 0.0;
  o.s.murk = 0.0;
  o.s.normal = vec3(0.0, 0.0, 1.0);
  const vec3 g = fnvToWorld(vec3(0.0, 0.0, 1.0));
  if (!fnvWetActive() || dot(g, g) == 0.0) return o;
  o.wp = fnvWetWorldPos();
  vec3 nW = fnvToWorld(n);
  if (dot(nW, nW) == 0.0) nW = g;
  o.s = fnvWetness(o.wp, g, nW, relief, pool);
  if (o.s.flatten > 0.0) n = fnvToTangent(normalize(mix(nW, g, o.s.flatten)));
  albedo = pow(fnvWetAlbedo(pow(max(albedo, 0.0), vec3(2.2)), o.s.darkening), vec3(1.0 / 2.2));
  o.on = true;
  return o;
}
// lit: the lit colour (linear); ambient: the ambient colour (gamma; the sky without a probe); sunL: the sun's direction
// (tangent space; vec3(0): no sun); sun: its colour (linear, shadowed).
vec3 fnvWetObjectFilm(FnvWetObject o, vec3 lit, vec3 ambient, vec3 sunL, vec3 sun) {
  if (!o.on || o.s.gloss <= 0.0) return lit;
  const vec3 V = normalize(fnvWetCamera() - o.wp);
  const float R = fnvWetReflectance(clamp(dot(o.s.normal, V), 0.0, 1.0), o.s.roughness, o.s.gloss);
  vec3 sky = pow(max(ambient, 0.0), vec3(2.2));
#ifdef FNV_ENV
  if (fnvEnvValid()) sky = fnvEnvRadiance(reflect(-V, o.s.normal), o.s.roughness * o.s.roughness);
#endif
  const vec3 L = fnvToWorld(sunL);
  const vec3 spec = dot(L, L) > 0.0 ? fnvWetSpecular(o.s.normal, V, L, sun, o.s.roughness) * o.s.gloss : vec3(0.0);
  return lit * (1.0 - R) + (sky * R + spec) * fnvWetMurk(o.s.murk);
}
#endif
)GLSL";
  for (size_t i = 0; i < r.params.size(); i++)
    h += paramDefine(r.params[i], "fnvC[" + std::to_string(kReplacementParamTop - i) + "].x");
  return h;
}

std::vector<Replacement> loadReplacements(const std::string& dir, const std::string& cacheDir,
                                          const std::function<void(const char*)>& log) {
  std::vector<Replacement> out;
  std::error_code ec;
  if (!fs::is_directory(dir, ec)) return out;
  fs::create_directories(cacheDir, ec);
  std::vector<fs::path> folders;
  for (const auto& entry : fs::directory_iterator(dir, ec))
    if (entry.is_directory() && fs::exists(entry.path() / "replace.ini")) folders.push_back(entry.path());
  std::sort(folders.begin(), folders.end());
  for (const fs::path& folder : folders) {
    Replacement r;
    r.id = folder.filename().string();
    const std::string ini = (folder / "replace.ini").string();
    r.name = iniString(ini, "Replacement", "Name", r.id.c_str());
    r.description = iniString(ini, "Replacement", "Description");
    {
      std::string pk = iniString(ini, "Replacement", "Packages");
      for (char& c : pk) if (c == ',') c = ' ';
      std::stringstream ps(pk);
      for (std::string p; ps >> p;) {
        char* end = nullptr;
        const long v = strtol(p.c_str(), &end, 10);
        if (!end || *end || v < 0 || v > 999) { r.error = "Packages: '" + p + "' is not a package number"; break; }
        r.packages.push_back(int(v));
      }
    }
    for (const std::string& sec : sections(ini)) {
      const std::string ls = lower(sec);
      if (ls.rfind("param.", 0) == 0) {
        Param p;
        if (!parseParam(ini, sec, p, r.error)) break;
        r.params.push_back(p);
      } else if (ls.rfind("shader.", 0) == 0) {
        Replacement::Target t;
        t.shader = sec.substr(7);
        t.source = iniString(ini, sec.c_str(), "Source");
        std::string defs = iniString(ini, sec.c_str(), "Defines");
        for (char& c : defs) if (c == ',') c = ' ';
        std::stringstream ds(defs);
        for (std::string d; ds >> d;) {  // NAME or NAME=VALUE (a number or identifier)
          const size_t eq = d.find('=');
          const std::string name = d.substr(0, eq), value = eq == std::string::npos ? "1" : d.substr(eq + 1);
          bool valueOk = !value.empty();
          for (char c : value) valueOk = valueOk && (isalnum(uint8_t(c)) || c == '_' || c == '.');
          if (!validIdentifier(name) || !valueOk) { r.error = sec + ": define '" + d + "' is not NAME or NAME=VALUE"; break; }
          t.defines.push_back(d);
        }
        std::string game = iniString(ini, sec.c_str(), "GameConstants");
        for (char& c : game) if (c == ',') c = ' ';
        std::stringstream gs(game);
        for (std::string g; r.error.empty() && gs >> g;) {  // float register numbers, 0..223
          char* end = nullptr;
          const unsigned long reg = strtoul(g.c_str(), &end, 10);
          if (!end || *end || reg >= 224) r.error = sec + ": GameConstants '" + g + "' is not a register 0..223";
          else t.gameConstants.push_back(uint32_t(reg));
        }
        if (t.source.empty()) r.error = sec + ": no Source";
        if (!r.error.empty()) break;
        r.targets.push_back(t);
      }
    }
    if (r.error.empty() && r.params.size() > kMaxParams) r.error = "more than 32 parameters";
    if (r.error.empty() && r.targets.empty()) r.error = "no [Shader.<name>] sections";
    std::unordered_map<std::string, std::string> sources;
    for (Replacement::Target& t : r.targets) {
      if (!r.error.empty()) break;
      std::string& source = sources[t.source];
      if (source.empty() && !readText(folder / t.source, source)) { r.error = t.shader + ": cannot read '" + t.source + "'"; break; }
      const std::string header = replacementHeader(r, t.defines);
      char key[32];
      snprintf(key, sizeof(key), "%016llx", static_cast<unsigned long long>(fnv1a(header + "\n" + source + "\nfnvvk-repl-1")));
      const fs::path cached = fs::path(cacheDir) / (r.id + "_" + t.shader + "_" + key + ".spv");
      std::string bytes;
      if (readText(cached, bytes) && bytes.size() >= 20 && bytes.size() % 4 == 0) {
        t.spv.resize(bytes.size() / 4);
        memcpy(t.spv.data(), bytes.data(), bytes.size());
        continue;
      }
      std::string err;
      if (!compile(header, source, t.source, t.spv, err)) {
        r.error = t.shader + " (" + t.source + "):\n" + err;
        break;
      }
      std::ofstream f(cached, std::ios::binary | std::ios::trunc);
      f.write(reinterpret_cast<const char*>(t.spv.data()), std::streamsize(t.spv.size() * 4));
    }
    if (!r.error.empty()) {
      for (Replacement::Target& t : r.targets) t.spv.clear();
      log(("replacement '" + r.id + "' NOT loaded: " + r.error).c_str());
    } else {
      log(("replacement '" + r.id + "' (" + r.name + "): " + std::to_string(r.targets.size()) + " shaders, " +
           std::to_string(r.params.size()) + " parameters").c_str());
    }
    out.push_back(std::move(r));
  }
  return out;
}

}  // namespace fnvvk::fx
