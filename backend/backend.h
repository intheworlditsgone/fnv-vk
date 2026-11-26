#pragma once
// The plugin and offline harness share this interface. Gamebryo objects are identified by opaque keys.
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace fnvvk {

namespace d3dx { struct Texture; }

// Opaque identities of game objects; the backend never dereferences them.
using RtKey = uintptr_t;   // Ni2DBuffer::RendererData*
using TexKey = uintptr_t;  // NiDX9SourceTextureData*
// The vertex/index buffer, shader program and vertex declaration handles the Gamebryo layer stores.
using BufKey = uintptr_t;
using ShaderKey = uintptr_t;
using DeclKey = uintptr_t;

// D3DVERTEXELEMENT9 field values (stream, offset, D3DDECLTYPE, D3DDECLMETHOD, D3DDECLUSAGE, usage index).
struct VertexElement {
  uint16_t stream = 0, offset = 0;
  uint8_t type = 0, method = 0, usage = 0, usageIndex = 0;
};

// Lock flags the game passes when it writes buffer contents (D3DLOCK_*).
enum : uint32_t { kLockNoOverwrite = 0x1000, kLockDiscard = 0x2000 };

// One draw's complete pipeline state, in D3D9 terms (what the game's Gamebryo layer configured). The backend
// translates it; see backend/translate.h for the semantics.
struct DrawSetup {
  const uint32_t* renderStates = nullptr;       // [256], indexed by D3DRENDERSTATETYPE
  const uint32_t (*samplerStates)[14] = nullptr; // [16][14], indexed by sampler then D3DSAMPLERSTATETYPE
  struct Texture {
    enum Kind : uint8_t { None, Tex, RenderTarget } kind = None;
    uintptr_t key = 0;  // TexKey or RtKey
  } textures[16];
  ShaderKey vs = 0, ps = 0;
  DeclKey decl = 0;
  struct Stream {
    BufKey buffer = 0;
    uint32_t offset = 0, stride = 0;
  } streams[8];
  BufKey indices = 0;
  const float* vsConstants = nullptr;  // [256][4]
  const float* psConstants = nullptr;  // [224][4]
  uint64_t vsConstantsVersion = 0, psConstantsVersion = 0;  // changes whenever the arrays change
  // Change whenever renderStates / clipPlanes change (0: unknown, every draw recomputes what depends on them). With the
  // same versions, shaders, declaration, strides, topology and render-target group as the previous draw, the backend
  // reuses that draw's pipeline and render-state blocks.
  uint64_t renderStatesVersion = 0, clipPlanesVersion = 0;
  struct { int32_t x = 0, y = 0; uint32_t width = 0, height = 0; float minZ = 0, maxZ = 1; } viewport;
  int32_t scissor[4] = {};  // RECT
  float clipPlanes[6][4] = {};
};

// D3DFORMAT values the game's render targets use (NiPixelFormat renderer hint).
enum : uint32_t {
  kD3DFMT_A8R8G8B8 = 21,
  kD3DFMT_X8R8G8B8 = 22,
  kD3DFMT_D24S8 = 75,
  kD3DFMT_A16B16G16R16F = 113,
  kD3DFMT_R32F = 114,
  kD3DFMT_R5G6B5 = 23,
  kD3DFMT_D24X8 = 77,
  kD3DFMT_G16R16F = 112,
  kD3DFMT_A32B32G32R32F = 116,
};

// IDirect3DDevice9::Clear flags (the game converts NiRenderer clear modes to these in Do_ClearBuffer 0x00E6F310).
enum : uint32_t { kClearTarget = 1, kClearZBuffer = 2, kClearStencil = 4 };

struct RtDesc {
  uint32_t width = 0, height = 0;
  uint32_t d3dFormat = 0;
  uint32_t samples = 1;  // 1 = no MSAA
  bool depth = false;    // depth-stencil buffer (vs color)
  bool operator==(const RtDesc&) const = default;
};

// A game shader whose translation is replaced (the plugin's shader replacements: GLSL written against the D3D9 shader
// ABI, docs/SHADER_ABI.md). `constants` are overlaid on its float constants as `def`s are (the replacement's
// parameters, in registers the game does not use for that shader).
struct ShaderReplacement {
  uint8_t md5[16] = {};
  std::vector<uint32_t> spv;
  struct Constant { uint32_t reg; float v[4]; };
  std::vector<Constant> constants;
  std::vector<uint32_t> gameConstants;  // float registers the replacement reads as the game set them (no original def)
};

struct BackendConfig {
  std::string shaderPack;     // path of fnv_vk_shaders.bin (tools/packshaders.py); empty = no draws
  // Persistent pipelines: the driver's pipeline-cache data and the keys of every pipeline the game used, loaded at
  // start (the recorded pipelines are created before the first frame) and rewritten while playing. Empty = off.
  std::string pipelineFile;
  bool shadowWindow = true;   // present to a second, never-activated window
  void* presentWindow = nullptr;  // HWND: present into this window instead (the game's own); overrides shadowWindow
  bool vsync = false;             // presentWindow only: FIFO (the game's D3DPRESENT_INTERVAL_ONE) instead of immediate
  bool validation = false;    // enable VK_LAYER_KHRONOS_validation if it is installed for this architecture
  // Wide targets: the game's 8-bit colour targets of the window size (and exactly half or a quarter of it), which hold
  // the image after the game's HDR exposure, are created as RGBA16F so values above 1.0 survive to tonemapping and HDR
  // output. CPU access (readback, writeTarget) still sees the D3D9 layout (8-bit BGRA).
  bool wideTargets = false;
  // Record thread (docs/PERFORMANCE.md "Record thread"): calls are queued and recorded into Vulkan on a worker thread,
  // in the order they were made; calls that return what the recording produced (readbacks, query results, texture
  // creation, stats()) wait for the queue to drain. The caller can be at most one frame ahead of the worker.
  bool recordThread = false;
  // Depth pre-pass (backend_draw.inl "Depth pre-pass"): between beginDepthPrepass and endDepthPrepass the opaque draws
  // are drawn twice, depth only and then in colour with an EQUAL depth test, so each pixel is shaded once.
  bool depthPrepass = true;
  // Draws that can discard (alpha-tested, or a shader with a discard of its own) in the depth pre-pass too: their depth
  // draw runs the shader's discards with the colour work dropped (spvDepthVariant), their colour draw tests EQUAL with
  // no discard (spvColourVariant).
  bool depthPrepassDiscarding = true;
  std::vector<ShaderReplacement> shaderReplacements;  // applied when the shader pack loads, before any pipeline
  std::function<void(const char*)> log;
};

// HDR output (present). The finished frame is gamma 2.2 encoded with 1.0 = SDR white; at present it is decoded and
// shown with SDR white at `paperWhiteNits`, clamped to `peakNits`, as HDR10 (PQ, BT.2020) or scRGB (linear BT.709,
// FP16). Falls back to SDR when the surface does not offer the format.
struct HdrSettings {
  enum Output : uint32_t { Off, Hdr10, ScRgb } output = Off;
  float paperWhiteNits = 200.0f, peakNits = 1000.0f;
};
struct HdrStatus {
  uint32_t active = HdrSettings::Off;  // what the swapchain uses now
  bool hdr10Available = false, scRgbAvailable = false;  // offered by the surface (known once a swapchain exists)
  bool wideTargets = false;
};

// Upscaling (backend/backend_ffx.inl): AMD FSR 3.1 through the FidelityFX API, built for this process as
// fnv_vk_fidelityfx.dll (cmake/fidelityfx.cmake) and loaded when first enabled. The upscaler renders the world at a lower
// resolution: between beginScaledScene and upscaleScene the window-sized render targets (and every group whose first
// colour target is one) render into images of the render size, and the camera's projection is jittered by the caller
// (jitterOffset). upscaleScene upscales the scene colour into the window-sized target; everything after it (the game's
// image space, the interface) runs at the window size.
struct UpscaleSettings {
  enum Upscaler : uint32_t { Off, Fsr3 } upscaler = Off;
  // FSR's quality modes: render size = window size / 1.0, 1.5, 1.7, 2.0, 3.0.
  enum QualityMode : uint32_t { NativeAa, Quality, Balanced, Performance, UltraPerformance } quality = Quality;
  bool sharpen = true;
  float sharpness = 0.5f;  // 0..1 (FSR's RCAS)
  std::string library;     // path of fnv_vk_fidelityfx.dll
};
struct UpscaleStatus {
  bool libraryLoaded = false;
  std::string error;                      // why the last attempt to load or create something failed (empty: none)
  bool upscaling = false;                 // the last world frame was upscaled
  uint32_t renderW = 0, renderH = 0, displayW = 0, displayH = 0;
};
// Camera and timing of the scene upscaleScene reads, as EffectFrame gives the world camera.
struct SceneView {
  float cameraRows[3][4]{};  // world-space rows: rotation (columns direction, up, right) and origin in w
  float frustum[4]{};        // tangents: left, right, top, bottom
  float nearZ = 0, farZ = 0;
  float port[4]{};           // NiViewport left, right, top, bottom (0..1, Gamebryo y up)
  float jitter[2]{};         // the projection's jitter this frame, render pixels (jitterOffset); 0 without an upscaler
  float frameTimeMs = 16.7f;
  bool reset = false;        // the camera cut (loading, teleport): no history
};

struct BackendStats {
  uint64_t frames = 0, groups = 0, clears = 0, resolves = 0;
  uint64_t attached = 0, destroyed = 0, resized = 0, realized = 0;
  uint64_t unknownKeys = 0;        // group bound a renderer-data object that was never attached
  uint64_t sampleMismatches = 0;   // group attachments with different sample counts (Vulkan cannot render that)
  uint64_t sizeMismatches = 0;     // depth attachment smaller than the color render area
  uint64_t unsupportedFormats = 0;
  uint64_t swapchains = 0;         // swapchain (re)creations
  uint64_t texturesCreated = 0, texturesDestroyed = 0, textureFailures = 0, textureBytes = 0;
  // Record thread: the most decoded texture bytes queued for upload at once, and the uploads that waited for the queue
  // to drain first (Recorder::kTexQueueBudget).
  uint64_t textureQueuePeakBytes = 0, textureQueueWaits = 0;
  uint64_t buffersCreated = 0, bufferWrites = 0, bufferRenames = 0, bufferBytesWritten = 0, unknownBuffers = 0;
  uint64_t effectRuns = 0, effectsSkipped = 0;  // effect runs / requested but not possible
  // Of the runs: scene colour copies made and reused (EffectFrame::sceneUnchanged), passes skipped (Skip=pbr), runs
  // skipped for want of skin (EffectDesc::skinBuffer).
  uint64_t fxSceneCopies = 0, fxSceneCopiesReused = 0, fxPassesSkipped = 0, fxNoSkinRuns = 0;
  // Record thread (BackendConfig::recordThread), microseconds: running the queued calls; the callers waiting at present
  // for the previous frame's present, and waiting for calls that return results (recordSyncs of them).
  uint64_t recordBusyUs = 0, recordPresentWaitUs = 0, recordSyncWaitUs = 0, recordSyncs = 0;
  uint64_t overlayFrames = 0;            // presents with the menu drawn
  // Upscaling (backend_ffx.inl): scenes upscaled by FSR, render-size targets filtered up at the end of a scaled scene.
  uint64_t fsrUpscales = 0, fsrTwinBlits = 0;
  uint64_t shadowCasters = 0;            // shadow-map caster draws
  uint64_t sunShadowedDraws = 0;         // game draws whose sun light the cascades shadowed
  uint64_t bufferRecycles = 0;      // renames served by a retired version instead of a new VkBuffer
  uint64_t bufferDirectWrites = 0;  // writes copied by the CPU into a mapped version (no staging copy)
  uint64_t mappedFallbacks = 0;     // mapped version wanted but the dynamic-geometry pool was full
  uint64_t shadersCreated = 0, shadersMissing = 0, pipelines = 0, samplers = 0;
  uint64_t draws = 0, drawsSkipped = 0, drawsNoGroup = 0, drawsNoShader = 0, drawsNoPipeline = 0;
  uint64_t pipelineReuses = 0, stateBlockReuses = 0;  // draws that reused the previous draw's pipeline / state blocks
  // Depth pre-pass: draws held for it, of them drawn depth-only first, and the runs of held draws recorded.
  uint64_t prepassDraws = 0, prepassDepthDraws = 0, prepassFlushes = 0;
  // Of the depth-only draws, ones that can discard (through their shader's depth module); such draws held that could
  // not be (their shader has no depth or colour module: spvDepthVariant / spvColourVariant refused it).
  uint64_t prepassDiscardingDraws = 0, prepassDiscardingRefused = 0;
  uint64_t descriptorSetWrites = 0, descriptorSetCacheHits = 0;  // draw sets written / found written earlier in the frame
  uint64_t missingInputs = 0, unknownTextures = 0, feedbackTextures = 0, textureKindMismatch = 0, tableFogDraws = 0;
  uint64_t depthTextureSamples = 0;  // draws sampling a depth render target (shaders have no depth-compare path)
  uint64_t queriesIssued = 0, queriesUnrecorded = 0;
  uint64_t copies = 0, copiesSkipped = 0, targetWrites = 0, gammaPresents = 0, groupsReopened = 0;
  uint64_t vkErrors = 0, validationMessages = 0;
  uint32_t live = 0, liveRealized = 0, liveTextures = 0, liveBuffers = 0;
};

class Backend {
public:
  Backend();
  ~Backend();
  Backend(const Backend&) = delete;
  Backend& operator=(const Backend&) = delete;

  // Lifetime calls are valid before init(): descriptors are recorded and images are created on first use.
  void attach(RtKey key, const RtDesc& desc);
  bool destroy(RtKey key);  // false if `key` was never attached
  void resize(RtKey key, uint32_t width, uint32_t height);

  // Creates the instance, device and (optionally) the shadow window. Call outside the loader lock.
  bool init(const BackendConfig& cfg, uint32_t windowWidth, uint32_t windowHeight);
  void shutdown();
  bool ready() const;

  // Frame recording, mirroring NiDX9Renderer: BeginUsingRenderTargetGroup / ClearBuffer / EndUsingRenderTargetGroup
  // / DisplayFrame. `rect` is a D3DRECT {x1, y1, x2, y2}; it is clipped to the render area as D3D9 clips it to the
  // viewport, which the game sets to buffer 0's full size before clearing.
  void beginGroup(const RtKey* colors, uint32_t colorCount, RtKey depth);
  void clear(uint32_t d3dClearFlags, uint32_t argb, float z, uint32_t stencil, const int32_t rect[4]);
  void endGroup();
  void present(RtKey onscreen);

  // Verification support (harness / in-game self-checks). Reads back the single-sample image of `key` (resolved
  // data for MSAA targets), as tightly packed texels of the Vulkan format. Waits for the GPU.
  bool readback(RtKey key, std::vector<uint8_t>& texels, uint32_t& vkFormat, uint32_t& texelSize);

  // `tex` is what D3DXCreateTextureFromFileInMemory would produce (d3dxtex). The upload is recorded into the frame's
  // upload command buffer, submitted ahead of its main one, so this may be called from any thread at any time (also
  // inside a render-target group). Replaces an existing texture. With the record thread it is queued, not waited for.
  // False: the format has no Vulkan mapping or there is no data.
  bool createTexture(TexKey key, const d3dx::Texture& tex);
  bool createTexture(TexKey key, d3dx::Texture&& tex);
  // Rendered cube maps (D3D9 cube render textures: NiRenderedCubeMap). The six faces are ordinary render targets
  // (attach, groups, clears, draws), D3D9 face order +X -X +Y -Y +Z -Z. Draws that sample any face key (DrawSetup
  // RenderTarget) get the cube: a cube texture under `cube` (readbackTexture, destroyTexture) that holds the faces'
  // current contents, with `levels` mip levels (0: the full chain) built from them (D3D9 D3DUSAGE_AUTOGENMIPMAP; 1: one
  // level, as the game creates its own). Faces drawn into since the last use are copied in and their mips rebuilt
  // before the next draw that samples the cube (or updateCube). Destroying all six faces destroys the cube.
  void attachCube(TexKey cube, const RtKey faces[6], uint32_t levels);
  bool updateCube(TexKey cube);  // brings the cube up to date now; false if a face is missing or unusable
  // The environment for the replaced lighting shaders (docs/SHADER_ABI.md "Environment"): a rendered cube map
  // (attachCube; given the full mip chain from now on), 0 for none. Pixel replacements that declare the environment
  // block get it, its mips and its irradiance harmonics.
  void setEnvironment(TexKey cube);
  // The environment GGX-prefiltered (linear light, level l = roughness (l / last)^2), as a texture key for
  // readbackTexture (verification).
  static constexpr TexKey kEnvironmentFiltered = TexKey(~uintptr_t(0) - 0x1F);
  // The harmonics of the environment block from six size x size RGBA16F faces (gamma 2.2 encoded, D3D9 face order);
  // returns the mean irradiance's luminance (verification).
  static float projectEnvironment(const uint16_t* texels, uint32_t size, float out[9][4]);
  bool destroyTexture(TexKey key);  // false if `key` has no texture
  // Reads one subresource back (tightly packed, as uploaded). Waits for the GPU. Harness / verification.
  bool readbackTexture(TexKey key, uint32_t face, uint32_t level, std::vector<uint8_t>& data);
  void finishFrame();  // submit and wait without presenting (harness)
  // Diagnostics: the texel at (x, y) of colour attachment 0 of the current render-target group, read mid-group
  // (the render pass is suspended, the work so far submitted and waited for, then resumed with its contents).
  bool probePixel(uint32_t x, uint32_t y, std::vector<uint8_t>& texel, uint32_t& vkFormat);

  // IDirect3DDevice9::StretchRect between render targets (single-sample/resolved surfaces), as NiDX9Renderer::Copy and
  // FastCopy issue it. Rects are {left, top, right, bottom}; null = the whole surface. filter: D3DTEXTUREFILTERTYPE.
  void copyTarget(RtKey src, RtKey dst, const int32_t* srcRect, const int32_t* dstRect, uint32_t filter);

  // A one-level 2D texture from rows of `pitch` bytes (created, or replaced whole). For textures the game streams
  // outside the NiTexture system (Bink movie tiles).
  bool updateTexture2D(TexKey key, uint32_t d3dFormat, uint32_t width, uint32_t height, const void* data, uint32_t pitch);
  // The same without waiting for the record thread (the data is copied; a failure is only logged): data that changes
  // while playing (the wet world's basin levels).
  void uploadTexture2D(TexKey key, uint32_t d3dFormat, uint32_t width, uint32_t height, const void* data, uint32_t pitch);
  // Screen-space textured quads into the open render-target group, as D3D9 draws an XYZRHW|TEX1 strip with stage 0
  // = texture, linear/clamp, no blending, no depth (Bink). Coordinates are D3D9 pretransformed pixel coordinates;
  // `clip` {left, top, right, bottom} is the D3D9 viewport rect. False if no group is open.
  struct ScreenQuad { TexKey tex; float x0, y0, x1, y1; };
  bool drawScreenQuads(const ScreenQuad* quads, uint32_t count, const int32_t clip[4]);

  // Effects (backend/effectlib.h): full-screen passes over the open (or last) render-target group. createEffect takes
  // compiled passes (SPIR-V for single-sample and multisampled depth) and returns an id (0 on failure).
  struct EffectDesc {
    enum Publish : uint8_t { None, Ao, Gi, GiDepth };
    // Publish=ao|gi|gidepth; width, height > 0 (Size=WxH): a fixed size instead of `scale` of the frame.
    struct Image { uint32_t vkFormat; float scale; uint8_t publish = None; uint32_t width = 0, height = 0; };
    struct Pass {
      const uint32_t* spv[2];  // [1]: multisampled depth variant
      size_t spvWords[2];
      int output;              // image index, -1 = the scene colour
      uint32_t blend;          // fx::Blend
      bool skipWhenAoApplied = false;  // Skip=pbr: not run while EffectFrame::aoApplied (the lighting applies the AO)
    };
    std::vector<Image> images;
    std::vector<Pass> passes;
    bool sceneColor = false;   // passes read the scene colour (a copy made before the first pass)
    bool skinBuffer = false;   // passes read the skin buffer: a run with no skin stored since it was activated does
                               // nothing (fnvSkinAt is false everywhere), and is skipped
  };
  uint32_t createEffect(const EffectDesc& d);
  void destroyEffect(uint32_t id);
  // The camera of the group's contents: world camera NiFrustum tangents/near/far and NiViewport (0..1, Gamebryo y up).
  struct EffectFrame {
    float cameraRows[3][4]{}; // world-space rows: forward, up, right coefficients and origin
    float waterSun[4]{};
    float submersion{};
    float frustum[4];
    float nearZ, farZ;
    float port[4];
    float time;
    uint32_t frame;
    // Sun shadows (effects read them with fnvShadow): per cascade, view space (fnvViewPos) to the cascade's clip space
    // (x, y in -1..1, z = depth 0..1), row-major; each cascade's far view distance; the direction toward the sun in view
    // space. shadowCascades 0: no shadows this frame.
    float shadowFromView[4][16];
    float shadowSplits[4];
    float sunView[3];
    uint32_t shadowCascades = 0;
    float sunColor[3];      // the sun's diffuse light (dimmer applied) and the scene's ambient light, as the game's
    float ambientColor[3];  // lighting receives them: effects weigh how much of a pixel's light the sun gives
    bool interior = false;  // the camera is in an interior cell (fnv.shadowInfo.z 1)
    bool aoApplied = false; // the lighting applies the published ambient occlusion (fnv.depthRange.w 1): effects that
                            // publish it do not also darken the image
    bool sceneUnchanged = false;  // nothing was drawn into the group since the previous runEffect call (the effects of
                                  // one point run back to back): that run's scene copy is reused when still current
  };
  // Runs the effect's passes. `params`: one vec4 per parameter (x used). False if the group cannot be used.
  bool runEffect(uint32_t id, const EffectFrame& f, const float* params, uint32_t paramCount);
  // Verification: image `index` of effect `id` as its last run left it, tightly packed (its width and height out).
  bool readbackEffectImage(uint32_t id, uint32_t index, std::vector<uint8_t>& texels, uint32_t& width, uint32_t& height);

  // Sun shadow maps: depth-only cascades of the world seen from the sun, drawn from the casters the caller submits
  // (the game's own buffers and declarations), sampled by effects. beginShadowMaps (size x size texels per cascade),
  // then per cascade beginShadowCascade with its view-projection (row-major, clip = M * (world, 1), z 0..1) and its
  // casters, then endShadowMaps. The open render-target group is suspended meanwhile.
  struct ShadowCaster {
    BufKey streams[4] = {};
    uint32_t strides[4] = {};
    uint32_t streamCount = 0;
    BufKey indices = 0;          // 0: not indexed
    DeclKey decl = 0;            // POSITION0 (and TEXCOORD0 for the alpha test) are read
    uint32_t d3dPrimitive = 4;   // D3DPT_TRIANGLELIST (4) or D3DPT_TRIANGLESTRIP (5)
    int32_t baseVertex = 0;
    uint32_t start = 0, primitiveCount = 0;
    float world[12] = {};        // three rows: world = rows * (object, 1)
    TexKey alphaTexture = 0;     // alpha-tested: no shadow where this texture's alpha < 0.5
    // Strips (d3dPrimitive 5): when stripCount > 0, the caster is that many strips of stripLengths[i] indices each,
    // consecutive from `start` (strips shorter than 3 are skipped), drawn with one setup (start, primitiveCount unused).
    const uint16_t* stripLengths = nullptr;
    uint32_t stripCount = 0;
  };
  bool beginShadowMaps(uint32_t cascades, uint32_t size);
  void beginShadowCascade(uint32_t index, const float viewProj[16]);
  void drawShadowCaster(const ShadowCaster& c);
  void endShadowMaps();

  // Sun shadows in the game's lighting (docs/SHADER_ABI.md, "Sun shadows"): the game's pixel shaders that light with
  // the sun's colour (sdp2spv's sun shadow pass) multiply it by the sun's visibility in the cascades drawn by the last
  // shadow pass. While a state is set, each such draw asks `gate` whether its lighting pass's sun-colour slot holds the
  // sun this time; null (the default): no draw is shadowed, and the shaders compute exactly what the game's do.
  struct SunLighting {
    float frustum[4];               // the world camera, as EffectFrame (the view space the cascades are in)
    float nearZ, farZ;
    float port[4];
    uint32_t cascades = 0;          // of the last shadow pass
    float shadowFromView[4][16];    // as EffectFrame
    float splits[4];
    float strength = 1.0f;          // 0 no shadow .. 1 the sun fully blocked
    bool (*gate)(void* ctx) = nullptr;
    void* ctx = nullptr;
  };
  void setSunLighting(const SunLighting* s);

  // Wet world (docs/WETWORLD.md): while a state is set (the world pass), pixel replacements that declare the wet block
  // (FNV_WET: block 12, the sun shadow array 48 / 49 for the rain cover, the puddle map 56 with its sampler 57, the
  // basins' levels 58) get the weather's wetness, the rain cover (a layer of this frame's shadow maps, drawn from straight
  // above), the local puddles' level and the terrain basins' water (per map texel: its basin and its ground; per basin
  // its water level). null: none.
  struct WetLighting {
    float rain = 0, wetness = 0, puddleLevel = 0;  // 0..1: the rain falling, the surfaces' wetness, the local puddles' level
    float puddleCoverage = 0.25f;   // the share of flat open ground the local puddles cover at a full level
    float time = 0;                 // seconds (the ripples' clock)
    float rippleStrength = 1.0f;
    float cameraRows[3][4];         // the world camera now, as EffectFrame (view -> world; origin in w)
    float frustum[4];
    float nearZ = 0, farZ = 0;
    float port[4];
    bool cover = false;             // the cover layer was drawn this frame
    uint32_t coverLayer = 0;        // its layer in the shadow maps
    float coverFromWorld[16];       // camera-relative world -> cover clip (row-major; z 0..1 downward from the top)
    float coverBias = 0;            // clip depth below the top surface still counted open
    float coverTexel = 0;           // 1 / the cover map's size
    TexKey puddleMap = 0;           // A32B32G32R32F: r the texel's basin (0 none), g the ground (absolute z); 0: no map
    TexKey puddleLevels = 0;        // R32F, 256 wide: the water level (absolute z) of basin i at texel (i % 256, i / 256)
    bool basinWater = false;        // some basin holds water now
    float puddleOrigin[2] = {};     // world x, y of texel (0, 0)
    float puddleSpacing = 0;        // game units between texels
    uint32_t puddleSize[2] = {};
  };
  void setWetLighting(const WetLighting* w);
  // Skin (docs/SKIN.md): the detail normal map pixel replacements that declare it (FNV_SKIN: 59 with its sampler 60)
  // sample, tiled over the skin. 0: none (a flat normal is bound).
  void setSkinDetail(TexKey key);
  // Hair (docs/LIGHTING.md "Hair"): the tangent shift map pixel replacements that declare it (FNV_HAIR: 63 with its
  // sampler 64) sample. 0: none (the zero dummy is bound: no shift).
  void setHairShift(TexKey key);
  // The skin buffer (docs/SKIN.md "Subsurface scattering"): while active (the main world's opaque pass), pixel
  // replacements that declare it (FNV_SKIN: block 13, storage images 61 / 62) write each skin pixel's albedo and the
  // nearest skin depth there; effects read both (fnvSkinAt). Activating sizes it to the current target and clears it.
  void setSkinBuffer(bool active);

  // Ambient occlusion and screen-space indirect light in the replaced lighting (docs/SHADER_ABI.md "Ambient occlusion"):
  // while a state is set (the world pass), pixel replacements that declare the AO block get the images an effect
  // published with Publish=ao (and Publish=gi) in an earlier frame, reprojected from that frame's camera to this one.
  // null: none.
  struct AoLighting {
    float frustum[4];            // the world camera now, as EffectFrame
    float nearZ, farZ;
    float port[4];
    float cameraRows[3][4];
  };
  void setAoLighting(const AoLighting* a);

  // In-game menu (Dear ImGui, drawn onto the swapchain image at present, over the finished frame). The caller owns the
  // ImGui context. overlayBeginFrame starts the renderer's frame and gives the display size (false: no swapchain yet);
  // overlaySubmit(true) draws ImGui::GetDrawData() at the next present.
  bool overlayBeginFrame(uint32_t& width, uint32_t& height);
  void overlaySubmit(bool draw);
  bool inGroup() const;
  // CPU texels written into a (single-sample, colour) render target, rows of `pitch` bytes, ordered with the draws
  // recorded so far (D3D9 LockRect/UnlockRect on a lockable render texture: the canopy shadow masks).
  bool writeTarget(RtKey key, const void* data, uint32_t pitch);
  // IDirect3DDevice9::SetGammaRamp: applied when presenting (identity = plain copy).
  void setGammaRamp(const uint16_t red[256], const uint16_t green[256], const uint16_t blue[256]);
  // HDR output (takes effect at the next present; an output change recreates the swapchain). The gamma ramp applies to
  // SDR output only. Effects see the state in their frame block (fnv.hdr).
  void setHdr(const HdrSettings& s);
  HdrStatus hdrStatus() const;
  // Verification: the HDR present encode of `key` (output Hdr10 or ScRgb, current paper white and peak, no menu layer)
  // rendered into an offscreen image of that output's swapchain format, read back tightly packed. Waits for the GPU.
  bool readbackHdrEncode(RtKey key, uint32_t output, std::vector<uint8_t>& texels, uint32_t& vkFormat);

  // Upscaling (UpscaleSettings above). Settings take effect at the next world frame.
  void setUpscaling(const UpscaleSettings& s);
  UpscaleStatus upscaleStatus() const;  // without waiting (as of the last frame the backend recorded)
  // The render size of `s` for a window of displayW x displayH (FSR's rounding); the window size without an upscaler.
  static void renderSize(const UpscaleSettings& s, uint32_t displayW, uint32_t displayH, uint32_t& w, uint32_t& h);
  // FSR's jitter sequence (Halton 2, 3; ffxFsr3UpscalerGetJitterOffset): the offset of frame `index`, in render pixels.
  // The projection is moved by (2 x / renderW, -2 y / renderH) in clip space.
  static void jitterOffset(uint32_t index, uint32_t renderW, uint32_t displayW, float& x, float& y);
  // The world frame: beginScaledScene before the world is drawn (no effect without an upscaler), captureWorldDepth when
  // the world pass is done (the group's depth, before the first-person view clears it), upscaleScene before the game's
  // image space (`color`, `depth`: the scene texture's colour and depth). endScaledScene ends a scaled scene that
  // reached no upscaleScene (the window-sized targets get their render-size contents, filtered up).
  void beginScaledScene();
  void captureWorldDepth();
  void upscaleScene(RtKey color, RtKey depth, const SceneView& view);
  void endScaledScene();
  // Verification: what the last upscaleScene gave FSR (0: depth, R32F; 1: motion vectors, RG16F UV units) and the world's
  // depth captureWorldDepth copied (2: R32F), tightly packed (their width and height out). Waits for the GPU.
  bool readbackUpscaleInput(uint32_t which, std::vector<uint8_t>& texels, uint32_t& width, uint32_t& height);

  // Occlusion queries (D3DQUERYTYPE_OCCLUSION semantics: samples passing depth/stencil between begin and end, counted
  // precisely). Results never block: 0 = not ready yet, 1 = ready (`samples` set).
  uint32_t createQuery();
  void destroyQuery(uint32_t id);
  void beginQuery(uint32_t id);
  void endQuery(uint32_t id);
  int queryResult(uint32_t id, uint32_t& samples);

  // Index buffers hold 16-bit indices unless `index32`. Writes are ordered with draws as D3D9 orders lock/unlock with
  // draws: a write never changes what an earlier recorded draw reads.
  bool createBuffer(BufKey key, uint32_t size, bool index, bool index32 = false);
  bool destroyBuffer(BufKey key);
  bool writeBuffer(BufKey key, uint32_t offset, const void* data, uint32_t size, uint32_t lockFlags);

  // Shader programs, bound to their translation in the shader pack by the MD5 of the D3D9 bytecode. False if the
  // pack has no such shader.
  bool createShader(ShaderKey key, const uint8_t md5[16]);
  // A replaced shader's constants (its parameters) changed: the new values apply from the next draw.
  bool setShaderConstants(const uint8_t md5[16], const ShaderReplacement::Constant* constants, uint32_t count);
  bool destroyShader(ShaderKey key);
  void createDecl(DeclKey key, const VertexElement* elements, uint32_t count);
  bool destroyDecl(DeclKey key);

  // Records one D3D9-style draw into the current render-target group. `indexed`: DrawIndexedPrimitive
  // (baseVertex, startIndex) else DrawPrimitive (startVertex in `start`).
  void draw(const DrawSetup& state, uint32_t d3dPrimitiveType, uint32_t primitiveCount, bool indexed,
            int32_t baseVertex, uint32_t start);
  // The draws between these two (the world's opaque pass) get a depth pre-pass (BackendConfig::depthPrepass). Queued in
  // order with the other calls.
  void beginDepthPrepass();
  void endDepthPrepass();

  BackendStats stats() const;  // exact: with the record thread, waits for the queued calls
  // Without waiting: the counters as of the last frame the record thread finished (stats() without it), and the number of
  // frames presented so far by the caller (a frame's identity for once-per-frame work; finishFrame counts too).
  BackendStats statsSnapshot() const;
  uint64_t frameIndex() const;
  // What the backend holds in the process's address space that grows with load (any thread, without waiting): command
  // chunks queued for the record thread (and the most at once, the producers that waited for room, the chunk
  // allocations that failed), decoded textures queued, Vulkan memory kept until a frame completes (upload staging,
  // released textures and buffers), and the per-frame host rings (buffer-write staging, uniforms; they keep their size).
  struct MemoryReport {
    uint64_t cmdQueued = 0, cmdQueuedPeak = 0, cmdQueueWaits = 0, cmdAllocFailures = 0;
    uint64_t texQueued = 0, texQueuedPeak = 0;
    uint64_t garbageBytes = 0, hostRingBytes = 0;
  };
  MemoryReport memoryReport() const;

private:
  struct Impl;
  std::unique_ptr<Impl> m;
};

}  // namespace fnvvk
