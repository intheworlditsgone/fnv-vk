#include "replacements.h"

#include <windows.h>

#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

#include "backend.h"
#include "bc5dds.h"
#include "common.h"
#include "shaderpack.h"

namespace fnvvk::replacements {
namespace {

std::vector<fx::Replacement> g_list;
std::vector<std::vector<std::array<uint8_t, 16>>> g_md5s;  // per replacement: the game shaders it replaced
std::vector<bool> g_active;
std::string g_dir, g_cacheDir, g_ini;

std::string shaderDir() {  // <game>\Data\Shaders
  char exe[MAX_PATH] = "";
  GetModuleFileNameA(nullptr, exe, MAX_PATH);
  std::string d = exe;
  return d.substr(0, d.find_last_of("\\/") + 1) + "Data\\Shaders";
}

std::vector<ShaderReplacement::Constant> constantsOf(const fx::Replacement& r) {
  std::vector<ShaderReplacement::Constant> c;
  for (size_t i = 0; i < r.params.size(); i++)
    c.push_back({ uint32_t(fx::kReplacementParamTop - i), { r.params[i].value, 0.0f, 0.0f, 0.0f } });
  return c;
}

}  // namespace

std::vector<fx::Replacement>& list() { return g_list; }
const char* dir() { return g_dir.c_str(); }
bool active(const fx::Replacement& r) {
  for (size_t i = 0; i < g_list.size(); i++)
    if (&g_list[i] == &r) return g_active[i];
  return false;
}

bool readsViewVector() {
  for (const fx::Replacement& r : g_list)
    if (r.enabled && r.error.empty())  // after load(): the ones fillConfig will install
      for (const fx::Replacement::Target& t : r.targets)
        if (spvDeclaresInputLocation(t.spv, 12) || spvDeclaresInputLocation(t.spv, 13)) return true;
  return false;
}

void load() {
  g_ini = iniPath();
  g_dir = moduleDir() + "fnv_vk\\replacements";
  g_cacheDir = moduleDir() + "fnv_vk\\cache";
  g_list = fx::loadReplacements(g_dir, g_cacheDir, [](const char* s) { logf("%s", s); });
  for (fx::Replacement& r : g_list) {
    const std::string sec = "Replacement." + r.id;
    r.enabled = GetPrivateProfileIntA(sec.c_str(), "Enable", 1, g_ini.c_str()) != 0;
    for (fx::Param& p : r.params) {
      char buf[64] = "";
      GetPrivateProfileStringA(sec.c_str(), p.name.c_str(), "", buf, sizeof(buf), g_ini.c_str());
      p.value = buf[0] ? float(atof(buf)) : p.def;
    }
  }
  g_md5s.assign(g_list.size(), {});
  g_active.assign(g_list.size(), false);
}

// The skin's detail normal map (docs/SKIN.md), shipped with the lighting replacement: uploaded once while the lighting
// replaces the game's shaders; the skin builds sample it (binding 59). Without the file they keep the normal map alone.
void onBackendReady(Backend* be) {
  bool lighting = false;
  for (const fx::Replacement& r : g_list) lighting = lighting || (r.id == "lighting" && r.enabled && r.error.empty());
  if (!lighting) return;
  const std::string path = g_dir + "\\lighting\\skin_detail_n.dds";
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    logf("skin: no detail normal map (%s)", path.c_str());
    return;
  }
  const std::vector<uint8_t> file((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  d3dx::Texture t;
  std::string why;
  constexpr TexKey kSkinDetailKey = TexKey(~uintptr_t(0) - 0x5F);
  if (!bc5DdsTexture(file.data(), file.size(), t, &why)) {
    logf("skin: the detail normal map %s is not usable: %s", path.c_str(), why.c_str());
    return;
  }
  if (!be->createTexture(kSkinDetailKey, t)) {
    logf("skin: the detail normal map (%ux%u, %u levels) was NOT uploaded", t.width, t.height, t.levels);
    return;
  }
  be->setSkinDetail(kSkinDetailKey);
  logf("skin: detail normal map %ux%u, %u levels", t.width, t.height, t.levels);
}

void fillConfig(BackendConfig& cfg) {
  const std::string shaders = shaderDir();
  for (size_t i = 0; i < g_list.size(); i++) {
    const fx::Replacement& r = g_list[i];
    if (!r.enabled || !r.error.empty()) continue;
    const std::vector<ShaderReplacement::Constant> constants = constantsOf(r);
    uint32_t found = 0;
    for (const fx::Replacement::Target& t : r.targets) {
      const std::vector<Md5> md5s = sdpShaderMd5s(shaders, t.shader, r.packages);
      if (md5s.empty()) logf("replacement '%s': no game shader named %s", r.id.c_str(), t.shader.c_str());
      for (const Md5& m : md5s) {
        ShaderReplacement sr;
        memcpy(sr.md5, m.b.data(), 16);
        sr.spv = t.spv;
        sr.constants = constants;
        cfg.shaderReplacements.push_back(std::move(sr));
        std::array<uint8_t, 16> a;
        memcpy(a.data(), m.b.data(), 16);
        g_md5s[i].push_back(a);
      }
      found += md5s.empty() ? 0 : 1;
    }
    g_active[i] = !g_md5s[i].empty();
    logf("replacement '%s': %u of %zu game shaders found (%zu builds)", r.id.c_str(), found, r.targets.size(), g_md5s[i].size());
  }
}

void applyParams(Backend* be, const fx::Replacement& r) {
  if (!be) return;
  for (size_t i = 0; i < g_list.size(); i++) {
    if (&g_list[i] != &r) continue;
    const std::vector<ShaderReplacement::Constant> c = constantsOf(r);
    for (const auto& m : g_md5s[i]) be->setShaderConstants(m.data(), c.data(), uint32_t(c.size()));
  }
}

bool save(const fx::Replacement& r) {
  const std::string sec = "Replacement." + r.id;
  bool ok = WritePrivateProfileStringA(sec.c_str(), "Enable", r.enabled ? "1" : "0", g_ini.c_str()) != 0;
  for (const fx::Param& p : r.params) {
    char v[32];
    if (p.type == fx::Param::Type::Float) snprintf(v, sizeof(v), "%g", p.value);
    else snprintf(v, sizeof(v), "%d", int(p.value));
    ok &= WritePrivateProfileStringA(sec.c_str(), p.name.c_str(), v, g_ini.c_str()) != 0;
  }
  logf("replacement '%s': settings %s fnv_vk.ini", r.id.c_str(), ok ? "saved to" : "could NOT be saved to");
  return ok;
}

}  // namespace fnvvk::replacements
