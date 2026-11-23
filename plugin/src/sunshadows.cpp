// Exterior sun shadows (sunshadows.h), after New Vegas Reloaded's exterior shadow maps (TESReloaded ShadowManager):
// the plugin walks the world's scene graph and draws each lit caster with its own geometry buffers into cascaded
// shadow maps fitted around the camera, before the world is drawn; the game's own lighting then takes the sun's light
// away where the sun does not reach (Backend::setSunLighting, sdp2spv's sun shadow pass).
//
// Game data (FNV 1.4.0.525; offsets read from the decomp, layouts as NVR's GameNi.h):
//   TES* 0x011DEA10: +0x1C NiDirectionalLight* (sun), +0x34 current interior cell (null outdoors), +0x88 worldspace
//     (TES::SetWorldSpace). NiDirectionalLight +0xF0: world direction the light travels (UpdateWorldData copies
//     world-rotation column 0 there).
//   Loaded cells: TES::GetGridCell 0x00457050 (x, y < uGridsToLoad); TESObjectCELL +0xC4 -> 3D node.
//   TESMain::spWorldRoot 0x011DEB7C -> BSSceneGraph: +0xAC camera. NiCamera: world rotation columns are
//     forward (0), up (1), right (2); +0xDC NiFrustum (left, right, top, bottom tangents, near, far, ortho).
//   NiAVObject: +0x20 NiBound* world bound (centre, radius), +0x30 flags (bit 0: app-culled), +0x68 world rotation
//     (row-major 3x3, world = R * (scale * local) + T), +0x8C translation, +0x98 scale. Class by NiRTTI (vtable slot 2:
//     { name, base }).
//   NiNode +0x9C children NiTArray (+0xA0 data, +0xA6 size). BSFadeNode +0xB8 fade alpha.
//   NiGeometry +0x9C property state (6 properties: 0 alpha, 3 shade), +0xB8 NiGeometryData (+0x34 buffer data),
//     +0xBC skin instance.
//   NiGeometryBufferData: +0x10 declaration, +0x20 stream count, +0x24 strides, +0x28 NiVBChip** (+0x08 vertex
//     buffer; bound at offset 0, nid3dshader SetStreamSource), +0x34 index buffer, +0x38 base vertex, +0x3C primitive
//     type, +0x40 triangle count, +0x48 strip count, +0x4C strip lengths (RenderTriShape2 0x00E745A0 draws each strip,
//     start += length).
//   NiAlphaProperty +0x18 flags (0x1 blend, 0x200 test). BSShaderPPLightingProperty +0xAC textures (GetTexture:
//     [set][slot], [0][0] diffuse). NiTexture +0x24 renderer data (the backend's texture key).
// The game's vertex/index buffer and declaration objects are the backend's keys (gbobjects).
#include "cputime.h"
#include "sunshadows.h"

#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "addresses.h"
#include "cascades.h"
#include "common.h"
#include "wetworld.h"

namespace fnvvk::sunshadows {
namespace {

bool g_enabled = true;
uint32_t g_size = 2048, g_cascades = 3;
float g_distance = 8000.0f;   // view distance covered by the cascades (game units)
float g_reach = 8000.0f;      // how far toward the sun casters are gathered beyond a cascade's sphere
float g_strength = 1.0f;      // how much of the sun's light a shadow takes away (0..1)
Backend::SunLighting g_sun{};  // this frame's, once its cascades are drawn
uint64_t g_sunFrame = ~0ull;   // the frame g_sun is for
// The wet world's rain cover (the last layer of the maps; Cover in sunshadows.h): 4096 units each way around the camera
// (2 x 4096 / Resolution units a texel: 4 at 2048), 4000 above and below it; a surface within 24 units under the top
// surface the map holds counts as that surface (open to the sky).
constexpr double kCoverRadius = 4096.0, kCoverAbove = 4000.0, kCoverBelow = 4000.0, kCoverBias = 24.0;
Cover g_cover;
// The far cascade on alternate frames ([Shadows] FarInterval, default 2; 1 every frame): the last cascade is redrawn
// only when the one drawn FarInterval - 1 frames ago no longer covers the current view's far slice (cascades::covers,
// 2 texels in from its sides for the filter) or the maps changed; otherwise its layer keeps the earlier drawing (the
// backend clears only the layers it begins) and its matrices are rebuilt for the current camera from its absolute box.
uint32_t g_farInterval = 2;
struct FarKept {
  bool valid = false;
  uint64_t frame = 0;  // frame it was drawn
  cascades::Cascade c;
  uint32_t size = 0, layers = 0, cascades = 0;
} g_far;
uint64_t g_farReused = 0, g_farDrawn = 0, g_farUncovered = 0;
constexpr double kFarMargin = 2.0;  // texels: the 3 x 3 filter's reach plus its bilinear taps

// Whether the lighting pass being drawn has the sun in its sun-colour slot. BSBatchRenderer::RenderPassImmediately
// (0x00B994F0) keeps the pass it draws at 0x011F91E0 (BSRenderPass: +0x9 light count, +0xC ShadowSceneLight* lights[]);
// ShadowLightShader::InitPass (0x00B78A90) gives light i the colour PSLightColor[i], and a pass's directional light is
// its first. ShadowSceneLight (include/ShadowSceneLight.h): +0xF4 bPointLight, +0xF5 bAmbientLight. Outdoors (the
// only time a state is set) the directional light is the sun's (ShadowSceneNode::SetSunLight, from Sky::GetLight).
bool sunGate(void*) {
  const uint8_t* pass = *reinterpret_cast<const uint8_t* const*>(0x011F91E0);
  if (!pass || pass[0x9] == 0) return false;
  const uint8_t* const* lights = *reinterpret_cast<const uint8_t* const* const*>(pass + 0xC);
  const uint8_t* first = lights ? lights[0] : nullptr;
  return first && first[0xF4] == 0 && first[0xF5] == 0;
}

using cascades::D3;
using cascades::dot;
using cascades::normalize;
D3 readVec(const void* p, uint32_t off) { return { rd<float>(p, off), rd<float>(p, off + 4), rd<float>(p, off + 8) }; }

// Class of a scene-graph object, by its NiRTTI chain (cached per vtable).
enum class Kind : uint8_t { Other, Node, FadeNode, Geometry };
std::unordered_map<const void*, Kind> g_kinds;
std::unordered_map<const void*, uint8_t> g_shadeKinds;  // 0 not a caster, 1 PP lighting (diffuse at +0xAC), 2 other lit casters (no alpha test)
using GetRttiFn = const void*(__thiscall*)(const void*);
bool chainHas(const void* obj, const char* name) {
  const void* rtti = (*reinterpret_cast<GetRttiFn* const*>(obj))[2](obj);
  for (int depth = 0; rtti && depth < 16; depth++) {
    const char* n = rd<const char*>(rtti, 0);
    if (n && strcmp(n, name) == 0) return true;
    rtti = rd<const void*>(rtti, 4);
  }
  return false;
}
// NiObject's own type queries (vtable after ~NiRefObject 0, DeleteThis 1, GetRTTI 2): IsSceneNode 3, IsFadeNode 4,
// IsGeometry 6 (include/NiObject.h).
using IsKindFn = const void*(__thiscall*)(const void*);
Kind kindOf(const void* obj) {
  const void* vt = *reinterpret_cast<const void* const*>(obj);
  auto it = g_kinds.find(vt);
  if (it != g_kinds.end()) return it->second;
  IsKindFn const* v = *reinterpret_cast<IsKindFn const* const*>(obj);
  Kind k = Kind::Other;
  if (v[4](obj)) k = Kind::FadeNode;
  else if (v[3](obj)) k = Kind::Node;
  else if (v[6](obj)) k = Kind::Geometry;  // particles and lines too: their shader properties are not casters
  g_kinds.emplace(vt, k);
  return k;
}
uint8_t shadeKindOf(const void* prop) {
  const void* vt = *reinterpret_cast<const void* const*>(prop);
  auto it = g_shadeKinds.find(vt);
  if (it != g_shadeKinds.end()) return it->second;
  uint8_t k = chainHas(prop, "BSShaderPPLightingProperty") ? 1
              : chainHas(prop, "Lighting30ShaderProperty") || chainHas(prop, "SpeedTreeBranchShaderProperty") ? 2 : 0;
  g_shadeKinds.emplace(vt, k);
  return k;
}

using cascades::Cascade;

struct Caster {
  const void* geom;
  const void* buf;
  float world[12];
  TexKey alpha;
  uint32_t mask;
};

uint64_t g_lastFrame = ~0ull;
const char* g_status = "not run yet";  // why the last frame had no cascades, or "rendered"
uint64_t g_framesRendered = 0;
size_t g_lastCasters = 0;
Backend::EffectFrame g_frame{};  // shadow fields of this frame's cascades
std::vector<Caster> g_casters;

TexKey alphaTextureOf(const void* geom, const void* shade, uint8_t shadeKind) {
  const void* alphaProp = rd<const void*>(geom, 0x9C + 0 * 4);
  if (!alphaProp || shadeKind != 1 || !(rd<uint16_t>(alphaProp, 0x18) & 0x201)) return 0;
  const void* const* set0 = rd<const void* const*>(shade, 0xAC);
  const void* tex = set0 ? set0[0] : nullptr;
  return tex ? TexKey(rd<const void*>(tex, 0x24)) : 0;
}

// Where the walk's objects went (the last frame), for the summary.
struct WalkCounts { uint32_t visited, appCulled, outside, nodes, fading, geometry, other, skinned, notCaster, noBuffers; } g_walk{};

// The loaded exterior cells' 3D nodes, as the game's water depth pass finds them (TES::GetGridCell 0x00457050 over
// uGridsToLoad, setting 0x011C63CC, value +4; TESObjectCELL::Get3D: *(cell + 0xC4)).
std::vector<const void*> loadedCells(const void* tes) {
  using GetGridCellFn = const void* const*(__thiscall*)(const void*, int, int);
  const auto getGridCell = reinterpret_cast<GetGridCellFn>(0x00457050);
  const int grid = *reinterpret_cast<const int*>(0x011C63CC + 4);
  std::vector<const void*> out;
  for (int x = 0; x < grid && grid > 0 && grid < 32; x++)
    for (int y = 0; y < grid; y++) {
      const void* const* slot = getGridCell(tes, x, y);
      const void* cell = slot ? *slot : nullptr;
      const void* const* data = cell ? rd<const void* const*>(cell, 0xC4) : nullptr;
      if (data && *data) out.push_back(*data);
    }
  return out;
}

void gather(const std::vector<const void*>& cells, const D3& camPos, const std::vector<Cascade>& cs) {
  g_casters.clear();
  g_walk = {};
  // A cell's node and its category nodes (0 actors, 1 markers, 2 land, 3 statics, 4 dynamic, 5 multibound) are not
  // culled by bound; below them every object is (as NVR culls per reference). Markers cast nothing, and actors
  // (TESObjectCELL::ActorNode = child 0) are left to the game's own actor shadows: their skinned bodies are not drawn
  // here, so their rigid parts (hair, Pip-Boy, weapons) would cast shadows of their own beside the game's.
  struct Item { const void* o; uint32_t depth; };
  std::vector<Item> stack;
  for (const void* c : cells) stack.push_back({ c, 0 });
  const uint32_t all = (1u << cs.size()) - 1;
  while (!stack.empty()) {
    const void* o = stack.back().o;
    const uint32_t depth = stack.back().depth;
    stack.pop_back();
    if (!o) continue;
    g_walk.visited++;
    if (rd<uint32_t>(o, 0x30) & 1) { g_walk.appCulled++; continue; }
    uint32_t mask = all;
    if (const float* b = depth >= 2 ? rd<const float*>(o, 0x20) : nullptr) {
      const D3 c{ b[0], b[1], b[2] };
      const double r = b[3];
      if (r > 0) {
        mask = 0;
        for (size_t i = 0; i < cs.size(); i++)
          if (cascades::inside(cs[i], c, r)) mask |= 1u << i;
      }
    }
    if (!mask) { g_walk.outside++; continue; }
    const Kind kind = kindOf(o);
    if (kind == Kind::Node || kind == Kind::FadeNode) {
      g_walk.nodes++;
      if (kind == Kind::FadeNode && rd<float>(o, 0xB8) < 0.75f) { g_walk.fading++; continue; }  // fading out
      const void* const* children = rd<const void* const*>(o, 0xA0);
      const uint16_t n = rd<uint16_t>(o, 0xA6);
      if (children)
        for (uint16_t i = 0; i < n; i++)
          if (children[i] && !(depth == 0 && i <= 1)) stack.push_back({ children[i], depth + 1 });  // not actors, markers
      continue;
    }
    if (kind != Kind::Geometry) { g_walk.other++; continue; }
    g_walk.geometry++;
    if (rd<const void*>(o, 0xBC)) { g_walk.skinned++; continue; }  // skinned: not yet
    const void* shade = rd<const void*>(o, 0x9C + 3 * 4);
    const uint8_t shadeKind = shade ? shadeKindOf(shade) : 0;
    if (!shadeKind) { g_walk.notCaster++; continue; }  // sky, water, grass, effects, UI
    const void* data = rd<const void*>(o, 0xB8);
    const void* buf = data ? rd<const void*>(data, 0x34) : nullptr;
    const uint32_t streams = buf ? rd<uint32_t>(buf, 0x20) : 0;
    if (!buf || !rd<const void*>(buf, 0x10) || streams == 0 || streams > 4) { g_walk.noBuffers++; continue; }  // not packed yet, FVF
    Caster c{ o, buf, {}, alphaTextureOf(o, shade, shadeKind), mask };
    const float s = rd<float>(o, 0x98);
    const D3 t = readVec(o, 0x8C) - camPos;
    const double tt[3] = { t.x, t.y, t.z };
    for (int r = 0; r < 3; r++) {
      for (int col = 0; col < 3; col++) c.world[r * 4 + col] = rd<float>(o, 0x68 + (r * 3 + col) * 4) * s;
      c.world[r * 4 + 3] = float(tt[r]);
    }
    g_casters.push_back(c);
  }
}

void submit(Backend* be, const Caster& c) {
  const void* buf = c.buf;
  Backend::ShadowCaster sc;
  sc.streamCount = rd<uint32_t>(buf, 0x20);
  const uint32_t* strides = rd<const uint32_t*>(buf, 0x24);
  const void* const* chips = rd<const void* const*>(buf, 0x28);
  if (!strides || !chips) return;
  for (uint32_t i = 0; i < sc.streamCount; i++) {
    if (!chips[i]) return;
    sc.streams[i] = BufKey(rd<const void*>(chips[i], 0x08));
    sc.strides[i] = strides[i];
  }
  sc.indices = BufKey(rd<const void*>(buf, 0x34));
  sc.decl = DeclKey(rd<const void*>(buf, 0x10));
  sc.d3dPrimitive = rd<uint32_t>(buf, 0x3C);
  sc.baseVertex = int32_t(rd<uint32_t>(buf, 0x38));
  sc.alphaTexture = c.alpha;
  memcpy(sc.world, c.world, sizeof(sc.world));
  const uint32_t strips = rd<uint32_t>(buf, 0x48);
  const uint16_t* lengths = rd<const uint16_t*>(buf, 0x4C);
  if (lengths && strips && sc.indices && sc.d3dPrimitive == 5) {
    // All strips with one setup (the backend draws each, skipping those shorter than 3).
    sc.start = 0;
    sc.stripLengths = lengths;
    sc.stripCount = strips;
    be->drawShadowCaster(sc);
  } else if (lengths && strips) {
    uint32_t start = 0;
    for (uint32_t i = 0; i < strips; i++) {
      if (lengths[i] < 3) { start += lengths[i]; continue; }
      sc.start = start;
      sc.primitiveCount = lengths[i] - 2u;
      be->drawShadowCaster(sc);
      start += lengths[i];
    }
  } else {
    sc.start = 0;
    sc.primitiveCount = rd<uint32_t>(buf, 0x40);
    be->drawShadowCaster(sc);
  }
}

}  // namespace

void install() {
  const std::string ini = iniPath();
  g_enabled = GetPrivateProfileIntA("Shadows", "Enable", 1, ini.c_str()) != 0;
  g_size = std::clamp<uint32_t>(GetPrivateProfileIntA("Shadows", "Resolution", 2048, ini.c_str()), 512, 4096);
  g_cascades = std::clamp<uint32_t>(GetPrivateProfileIntA("Shadows", "Cascades", 3, ini.c_str()), 1, 4);
  char buf[32] = "";
  GetPrivateProfileStringA("Shadows", "Distance", "8000", buf, sizeof(buf), ini.c_str());
  g_distance = std::clamp(float(atof(buf)), 1000.0f, 50000.0f);
  GetPrivateProfileStringA("Shadows", "Strength", "1", buf, sizeof(buf), ini.c_str());
  g_strength = std::clamp(float(atof(buf)), 0.0f, 1.0f);
  g_farInterval = std::clamp<uint32_t>(GetPrivateProfileIntA("Shadows", "FarInterval", 2, ini.c_str()), 1, 4);
}

void render(Backend* be) {
  if (!be) return;
  const uint64_t frame = be->frameIndex();
  if (frame == g_lastFrame) return;
  g_lastFrame = frame;
  g_frame.shadowCascades = 0;
  g_cover.drawn = false;
  // The wet world's rain cover (wetworld.h) is drawn as one more layer of the same maps, from straight above, whether
  // or not the sun casts shadows now (night, overcast, shadows off): wetworld decides when it is needed.
  const bool wantCover = wetworld::wantCover();
  if (!g_enabled && !wantCover) { g_status = "disabled"; return; }
  const void* tes = *reinterpret_cast<const void* const*>(fnv::TES_Singleton);
  if (!tes) { g_status = "no TES"; return; }
  if (rd<const void*>(tes, 0x34)) { g_status = "interior"; return; }
  if (!rd<const void*>(tes, 0x88)) { g_status = "no worldspace"; return; }
  const void* light = rd<const void*>(tes, 0x1C);
  const void* sceneGraph = *reinterpret_cast<const void* const*>(fnv::TESMain_spWorldRoot);
  const void* camera = sceneGraph ? rd<const void*>(sceneGraph, 0xAC) : nullptr;
  if (!camera || rd<bool>(camera, 0xDC + 24)) { g_status = "no perspective camera"; return; }

  const D3 camPos = readVec(camera, 0x8C);
  // The world pass's camera: TESMain::DrawWorld_DrawScene culls and draws with BSSceneGraph::spCamera, and the
  // renderer's view is built from it (NiCamera world rotation columns: forward 0, up 1, right 2; world translation).
  const D3 fwd{ rd<float>(camera, 0x68), rd<float>(camera, 0x74), rd<float>(camera, 0x80) };
  const D3 up{ rd<float>(camera, 0x6C), rd<float>(camera, 0x78), rd<float>(camera, 0x84) };
  const D3 right{ rd<float>(camera, 0x70), rd<float>(camera, 0x7C), rd<float>(camera, 0x88) };
  cascades::Camera cam;
  cam.position = camPos;
  cam.right = right;
  cam.up = up;
  cam.forward = fwd;
  for (int i = 0; i < 4; i++) cam.frustum[i] = rd<float>(camera, 0xDC + 4 * i);
  cam.nearZ = rd<float>(camera, 0xDC + 16);
  cam.farZ = rd<float>(camera, 0xDC + 20);

  // The sun's cascades (at most 3 with the wet world on: its cover takes the fourth layer).
  const uint32_t sunCascades = wetworld::enabled() ? std::min<uint32_t>(g_cascades, 3) : g_cascades;
  std::vector<Cascade> cs;
  D3 L{};
  const char* sunStatus = "rendered";
  if (!g_enabled) sunStatus = "disabled";
  else if (!light) sunStatus = "no sun light";
  else {
    L = normalize(readVec(light, 0xF0) * -1.0);  // toward the sun
    if (L.z < 0.08) sunStatus = "sun below the horizon";
    else {
      cs = cascades::build(cam, L, sunCascades, g_size, g_distance, g_reach);
      if (cs.empty()) sunStatus = "camera range too short";
    }
  }
  const bool sun = !cs.empty();
  if (!sun) g_far.valid = false;
  if (!sun && !wantCover) { g_status = sunStatus; return; }
  const uint32_t layers = sunCascades + (wetworld::enabled() ? 1 : 0);
  // The far cascade drawn recently, if it still covers this view's far slice: kept (see g_farInterval).
  const uint32_t farIdx = sunCascades - 1;
  bool keepFar = false;
  if (sun && sunCascades >= 2 && g_farInterval > 1 && g_far.valid && g_far.size == g_size && g_far.layers == layers &&
      g_far.cascades == sunCascades && frame > g_far.frame && frame - g_far.frame < g_farInterval) {
    if (cascades::covers(g_far.c, cam, cs[farIdx - 1].split, cs[farIdx].split, g_size, kFarMargin)) keepFar = true;
    else g_farUncovered++;
  }
  if (keepFar) {
    const float split = cs[farIdx].split;  // this view's slice (the kept box covers it)
    cs[farIdx] = g_far.c;
    cascades::setMatrices(cs[farIdx], cam);
    cs[farIdx].split = split;
  }
  if (sun) {
    for (uint32_t i = 0; i < sunCascades; i++) {
      memcpy(g_frame.shadowFromView[i], cs[i].fromView, 64);
      g_frame.shadowSplits[i] = cs[i].split;
    }
    // NiLight (include/NiLight.h): +0xC4 dimmer, +0xC8 ambient, +0xD4 diffuse (Sun::Update sets them from the weather).
    const float dimmer = rd<float>(light, 0xC4);
    for (int i = 0; i < 3; i++) {
      g_frame.sunColor[i] = rd<float>(light, 0xD4 + 4 * i) * dimmer;
      g_frame.ambientColor[i] = rd<float>(light, 0xC8 + 4 * i);
    }
    g_frame.sunView[0] = float(dot(L, right));
    g_frame.sunView[1] = float(dot(L, up));
    g_frame.sunView[2] = float(dot(L, fwd));
  }
  // The cover takes the layer (and gather mask bit) after the sun's cascades.
  const uint32_t coverLayer = sunCascades;
  if (wantCover) {
    const Cascade ov = cascades::overhead(cam, kCoverRadius, g_size, kCoverAbove, kCoverBelow);
    while (cs.size() < coverLayer) cs.push_back(Cascade{});  // (no sun: empty boxes that take no casters)
    for (size_t i = 0; i < coverLayer && !sun; i++) cs[i].radius = -1e30, cs[i].minRadius = 1e30f;
    cs.push_back(ov);
  }

  {
    cputime::Scope timed(cputime::SunWalk);
    if (keepFar) {  // the kept far cascade takes no casters (an empty box), so objects only in it are not walked
      std::vector<Cascade> walk = cs;
      walk[farIdx].radius = -1e30;
      walk[farIdx].minRadius = 1e30f;
      gather(loadedCells(tes), camPos, walk);
    } else {
      gather(loadedCells(tes), camPos, cs);
    }
  }
  g_lastCasters = g_casters.size();
  cputime::Scope recordTimed(cputime::SunRecord);
  if (!be->beginShadowMaps(layers, g_size)) { g_status = "shadow map not created"; g_far.valid = false; return; }
  for (uint32_t i = 0; i < uint32_t(cs.size()); i++) {
    if (i < sunCascades && !sun) continue;
    if (keepFar && i == farIdx) continue;  // its layer keeps the earlier drawing
    be->beginShadowCascade(i, cs[i].viewProj);
    for (const Caster& c : g_casters)
      if (c.mask & (1u << i)) submit(be, c);
  }
  be->endShadowMaps();
  if (sun && keepFar) {
    g_farReused++;
  } else if (sun) {
    g_far.valid = true;
    g_far.frame = frame;
    g_far.c = cs[farIdx];
    g_far.size = g_size;
    g_far.layers = layers;
    g_far.cascades = sunCascades;
    g_farDrawn++;
  }
  if (wantCover) {
    g_cover.drawn = true;
    g_cover.layer = coverLayer;
    memcpy(g_cover.fromWorld, cs[coverLayer].viewProj, 64);
    g_cover.bias = float(kCoverBias / (kCoverAbove + kCoverBelow));
    g_cover.texel = 1.0f / float(g_size);
  }
  if (!sun) {
    g_status = sunStatus;
    return;
  }
  g_frame.shadowCascades = sunCascades;
  memcpy(g_sun.frustum, cam.frustum, sizeof(g_sun.frustum));
  g_sun.nearZ = float(cam.nearZ);
  g_sun.farZ = float(cam.farZ);
  for (int i = 0; i < 4; i++) g_sun.port[i] = rd<float>(camera, 0x100 + 4 * i);  // NiCamera NiViewport
  g_sun.cascades = sunCascades;
  memcpy(g_sun.shadowFromView, g_frame.shadowFromView, sizeof(g_sun.shadowFromView));
  memcpy(g_sun.splits, g_frame.shadowSplits, sizeof(g_sun.splits));
  g_sun.strength = g_strength;
  g_sun.gate = &sunGate;
  g_sunFrame = frame;
  g_status = "rendered";
  g_framesRendered++;
}

void beginLighting(Backend* be) {
  if (be && g_sunFrame == be->frameIndex() && g_strength > 0.0f) be->setSunLighting(&g_sun);
}

void endLighting(Backend* be) {
  if (be) be->setSunLighting(nullptr);
}

std::string summary() {
  char b[640];
  snprintf(b, sizeof(b),
           "sun shadows: %s; %llu frames rendered, %zu casters last frame (walk: %u visited, %u app-culled, %u outside the "
           "cascades, %u nodes, %u fading, %u geometry: %u skinned, %u not lit casters, %u without buffers; %u other); far "
           "cascade drawn %llu, kept %llu (redrawn as no longer covering %llu)",
           g_status, static_cast<unsigned long long>(g_framesRendered), g_lastCasters, g_walk.visited, g_walk.appCulled,
           g_walk.outside, g_walk.nodes, g_walk.fading, g_walk.geometry, g_walk.skinned, g_walk.notCaster, g_walk.noBuffers,
           g_walk.other, static_cast<unsigned long long>(g_farDrawn), static_cast<unsigned long long>(g_farReused),
           static_cast<unsigned long long>(g_farUncovered));
  return b;
}

void fill(Backend::EffectFrame& f) {
  f.shadowCascades = g_frame.shadowCascades;
  if (!f.shadowCascades) return;
  memcpy(f.shadowFromView, g_frame.shadowFromView, sizeof(f.shadowFromView));
  memcpy(f.shadowSplits, g_frame.shadowSplits, sizeof(f.shadowSplits));
  memcpy(f.sunView, g_frame.sunView, sizeof(f.sunView));
  memcpy(f.sunColor, g_frame.sunColor, sizeof(f.sunColor));
  memcpy(f.ambientColor, g_frame.ambientColor, sizeof(f.ambientColor));
}

const Cover& cover() { return g_cover; }

}  // namespace fnvvk::sunshadows
