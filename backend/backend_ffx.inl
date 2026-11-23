// Upscaling (included in Backend::Impl; backend.h UpscaleSettings): AMD FSR 3.1 through the FidelityFX API (ffx-api),
// built for this 32-bit process as fnv_vk_fidelityfx.dll (cmake/fidelityfx.cmake) and loaded when first wanted.
//
// Scaled scene. With an upscaler, the world renders at the render size: between beginScaledScene and upscaleScene every
// window-sized render target resolves to a twin of the render size (lowRts), for any group whose first target is
// window-sized, for sampling and for copies; viewports, scissors, clear and copy rectangles (given by the game in window
// pixels) are scaled to it. Targets of other sizes are untouched (reflections, shadow maps, bloom chains). A group that is
// not window-sized but has the window-sized depth buffer attached (the water reflections) uses the full-size depth.
// upscaleScene upscales the scene texture's twin into the full-size target with FSR; every other twin written during the
// scene is filtered up into its full-size target (linear blit; depth nearest), so the game's image space and anything
// after it sees full-size contents.
//
// Inputs (builtin/fsrinputs.comp): FSR reads the scene colour, a depth and motion vectors at the render size. The game
// clears the depth buffer before drawing the first-person view, so the world's depth is copied at the end of the world
// pass (captureWorldDepth, builtin/fsrdepth.comp) and composited with the first-person view's; the motion vectors are the
// camera's (the game draws none), zero on the first-person view.

// --- scaled scene ---------------------------------------------------------------------------------------------------
bool scaledScene = false;            // between beginScaledScene and upscaleScene / endScaledScene
uint32_t sceneW = 0, sceneH = 0;     // its render size
std::unordered_map<RtKey, Rt> lowRts;  // render-size twins of window-sized targets
std::unordered_set<RtKey> lowDirty;    // twins written in this scaled scene
bool groupLow = false;               // the open (or last) group renders into twins

bool scalable(RtKey key, const Rt& rt) const {
  return scaledScene && rt.desc.width == screenW && rt.desc.height == screenH && !isCubeFace(key);
}
// The render-size twin of `full` (created, or recreated when the render size changed).
Rt* lowTwin(RtKey key, Rt& full) {
  if (!realize(key, full)) return nullptr;  // the twin takes the target's format decision (wide)
  RtDesc d = full.desc;
  d.width = sceneW;
  d.height = sceneH;
  Rt& l = lowRts[key];
  if (!(l.desc == d) || l.twinWide != full.wide) {
    if (l.realized) release(l, true);
    l = Rt{ d };
  }
  l.twin = true;
  l.twinWide = full.wide;
  return realize(key, l) ? &l : nullptr;
}
// The image `key` resolves to: its twin when `low` and it is window-sized in a scaled scene, else its own.
Rt* rtFor(RtKey key, bool low) {
  Rt* rt = lookup(key);
  if (!rt) return nullptr;
  if (low && scalable(key, *rt)) return lowTwin(key, *rt);
  return realize(key, *rt) ? rt : nullptr;
}
// Window pixels -> render pixels.
int32_t toSceneX(int32_t x) const { return int32_t((int64_t(x) * sceneW + screenW / 2) / screenW); }
int32_t toSceneY(int32_t y) const { return int32_t((int64_t(y) * sceneH + screenH / 2) / screenH); }
void dropTwin(RtKey key) {
  auto it = lowRts.find(key);
  if (it != lowRts.end()) {
    release(it->second, true);
    lowRts.erase(it);
  }
  lowDirty.erase(key);
}

// Ends the scaled scene: every twin written in it except `upscaled` (whose full-size target FSR wrote) is filtered up into
// its full-size target.
void finishScaledScene(RtKey upscaled) {
  if (!scaledScene) return;
  if (rendering) doEndGroup();
  for (RtKey key : lowDirty) {
    if (key == upscaled) continue;
    auto t = lowRts.find(key);
    Rt* full = lookup(key);
    if (t == lowRts.end() || !full || !realize(key, *full) || !t->second.realized) continue;
    Rt& low = t->second;
    if (low.desc.depth && !formatSupports(low.format, VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT)) {
      logOnce("fsrdepthblit", "upscaling: depth format %d cannot be blitted: the full-size depth keeps its contents",
              int(low.format));
      continue;
    }
    if (low.desc.depth && low.desc.samples > 1) continue;  // multisampled depth cannot be blitted
    const VkImageAspectFlags a = aspectOf(low);
    barrier(low.image, low.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, a);
    barrier(full->image, full->layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, a);
    VkImageBlit bl{};
    bl.srcSubresource = { a, 0, 0, 1 };
    bl.dstSubresource = bl.srcSubresource;
    bl.srcOffsets[1] = { int32_t(low.desc.width), int32_t(low.desc.height), 1 };
    bl.dstOffsets[1] = { int32_t(full->desc.width), int32_t(full->desc.height), 1 };
    vk.vkCmdBlitImage(slot().cmd, low.image, low.layout, full->image, full->layout, 1, &bl,
                      low.desc.depth ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
    barrier(low.image, low.layout, VK_IMAGE_LAYOUT_GENERAL, a);
    barrier(full->image, full->layout, VK_IMAGE_LAYOUT_GENERAL, a);
    st.fsrTwinBlits++;
  }
  scaledScene = false;
  groupLow = false;
  lowDirty.clear();
}

// --- FidelityFX API -------------------------------------------------------------------------------------------------
UpscaleSettings upSettings;
struct Ffx {
  HMODULE lib = nullptr;
  bool loadTried = false;
  ffxFunctions f{};
  ffxContext upscale = nullptr;  // FSR upscaling, for upW x upH output with upFlags
  uint32_t upW = 0, upH = 0, upFlags = 0;
  bool upReset = true;
} ffx;
mutable std::mutex upStatusMutex;
UpscaleStatus upStatus;  // upStatusMutex

void setUpError(const std::string& e) {
  std::lock_guard<std::mutex> lk(upStatusMutex);
  upStatus.error = e;
}
static void ffxMessage(uint32_t type, const wchar_t* message) {
  // The runtime's messages (debug checking): logged through the backend that is current (one per process).
  if (s_ffxLogSelf && message) {
    char buf[512];
    WideCharToMultiByte(CP_UTF8, 0, message, -1, buf, sizeof(buf), nullptr, nullptr);
    s_ffxLogSelf->logf("FidelityFX %s: %s", type == FFX_API_MESSAGE_TYPE_ERROR ? "error" : "warning", buf);
  }
}
static inline Impl* s_ffxLogSelf = nullptr;

bool ffxLoaded() {
  if (ffx.lib) return true;
  if (ffx.loadTried) return false;
  ffx.loadTried = true;
  s_ffxLogSelf = this;
  ffx.lib = LoadLibraryExA(upSettings.library.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!ffx.lib) {
    const DWORD e = GetLastError();
    logf("upscaling: %s did not load (error %lu)", upSettings.library.c_str(), e);
    setUpError("fnv_vk_fidelityfx.dll did not load (error " + std::to_string(e) + ")");
    return false;
  }
  ffxLoadFunctions(&ffx.f, ffx.lib);
  if (!ffx.f.CreateContext || !ffx.f.DestroyContext || !ffx.f.Configure || !ffx.f.Query || !ffx.f.Dispatch) {
    logf("upscaling: %s lacks the FidelityFX API entry points", upSettings.library.c_str());
    setUpError("fnv_vk_fidelityfx.dll is not the FidelityFX API");
    FreeLibrary(ffx.lib);
    ffx.lib = nullptr;
    return false;
  }
  logf("upscaling: FidelityFX API loaded from %s", upSettings.library.c_str());
  std::lock_guard<std::mutex> lk(upStatusMutex);
  upStatus.libraryLoaded = true;
  return true;
}
ffxCreateBackendVKDesc ffxBackendDesc() {
  ffxCreateBackendVKDesc b{};
  b.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
  b.vkDevice = device;
  b.vkPhysicalDevice = phys;
  b.vkDeviceProcAddr = vk.vkGetDeviceProcAddr;
  return b;
}
static FfxApiResource ffxImage(VkImage image, VkFormat format, uint32_t w, uint32_t h, bool uav, uint32_t state) {
  FfxApiResource r{};
  r.resource = FfxApiHandle(image);
  r.state = state;
  r.description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
  r.description.format = ffxApiGetSurfaceFormatVK(format);
  r.description.width = w;
  r.description.height = h;
  r.description.depth = 1;
  r.description.mipCount = 1;
  r.description.flags = FFX_API_RESOURCE_FLAGS_NONE;
  r.description.usage = uav ? FFX_API_RESOURCE_USAGE_UAV : FFX_API_RESOURCE_USAGE_READ_ONLY;
  return r;
}
// FidelityFX contexts go when the frames that used them have completed. A context's dispatches are recorded into the
// open frame's command buffer, which is submitted only at present, but settings can change before that (the menu is
// drawn before the frame is submitted). Destroying the context then would leave that command buffer using its freed
// pipelines, images and descriptor sets, and the device is lost; waiting for the device does not help, as it waits only
// for submitted work. So a context is retired with the frame being recorded and destroyed by collectGarbage once that
// frame has completed, like the backend's own Garbage.
struct RetiredFfxContext {
  ffxContext ctx;
  uint64_t frame;
};
std::vector<RetiredFfxContext> ffxRetired;
void retireFfxContext(ffxContext& c) {
  if (!c) return;
  ffxRetired.push_back({ c, frameNumber });
  c = nullptr;
}
void collectFfxGarbage(bool all) {  // all: the device is idle (teardown)
  for (size_t i = 0; i < ffxRetired.size();) {
    if (all || ffxRetired[i].frame <= completedFrame) {
      ffx.f.DestroyContext(&ffxRetired[i].ctx, nullptr);
      ffxRetired.erase(ffxRetired.begin() + ptrdiff_t(i));
    } else {
      i++;
    }
  }
}
void destroyUpscaleContext() { retireFfxContext(ffx.upscale); }
bool ensureUpscaleContext(uint32_t w, uint32_t h, uint32_t flags) {
  if (ffx.upscale && ffx.upW == w && ffx.upH == h && ffx.upFlags == flags) return true;
  destroyUpscaleContext();
  ffxCreateBackendVKDesc be = ffxBackendDesc();
  ffxCreateContextDescUpscale cd{};
  cd.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
  cd.header.pNext = &be.header;
  cd.flags = flags;
  cd.maxRenderSize = { w, h };  // Native AA renders at the window size
  cd.maxUpscaleSize = { w, h };
  cd.fpMessage = &ffxMessage;
  const ffxReturnCode_t r = ffx.f.CreateContext(&ffx.upscale, &cd.header, nullptr);
  if (r != FFX_API_RETURN_OK) {
    ffx.upscale = nullptr;
    logOnce("fsrctx" + std::to_string(r), "upscaling: the FSR context was not created (%u)", unsigned(r));
    setUpError("the FSR upscaler could not be created (" + std::to_string(r) + ")");
    return false;
  }
  ffx.upW = w;
  ffx.upH = h;
  ffx.upFlags = flags;
  ffx.upReset = true;
  logf("upscaling: FSR context for %ux%u (flags 0x%X)", w, h, flags);
  return true;
}

// --- FSR inputs: compute passes (builtin/fsrdepth.comp, builtin/fsrinputs.comp) ----------------------------------
VkDescriptorSetLayout fsrSetLayout = VK_NULL_HANDLE;
VkPipelineLayout fsrLayout = VK_NULL_HANDLE;
VkPipeline fsrDepthPipe = VK_NULL_HANDLE, fsrInputsPipe = VK_NULL_HANDLE;
VkSampler fsrPoint = VK_NULL_HANDLE;
VkDescriptorPool fsrPools[kFramesInFlight] = {};
uint64_t fsrPoolFrame[kFramesInFlight] = {};
FxImage worldDepthImg, fsrDepthImg, motionImg, fsrOutImg;
bool worldDepthValid = false;  // worldDepthImg holds this scaled scene's world (captureWorldDepth)
SceneView prevView{};
bool prevViewValid = false;

bool initFsrPasses() {
  if (fsrInputsPipe) return true;
  const VkDescriptorSetLayoutBinding b[5] = {
    { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    { 4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
  };
  VkDescriptorSetLayoutCreateInfo dl{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
  dl.bindingCount = 5;
  dl.pBindings = b;
  if (!check(vk.vkCreateDescriptorSetLayout(device, &dl, nullptr, &fsrSetLayout), "vkCreateDescriptorSetLayout(fsr)"))
    return false;
  VkPipelineLayoutCreateInfo pl{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
  pl.setLayoutCount = 1;
  pl.pSetLayouts = &fsrSetLayout;
  if (!check(vk.vkCreatePipelineLayout(device, &pl, nullptr, &fsrLayout), "vkCreatePipelineLayout(fsr)")) return false;
  auto pipe = [&](const uint32_t* code, size_t bytes, VkPipeline& out) {
    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = bytes;
    smci.pCode = code;
    VkShaderModule m = VK_NULL_HANDLE;
    if (!check(vk.vkCreateShaderModule(device, &smci, nullptr, &m), "vkCreateShaderModule(fsr)")) return false;
    VkComputePipelineCreateInfo ci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    ci.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = m;
    ci.stage.pName = "main";
    ci.layout = fsrLayout;
    const bool ok = check(vk.vkCreateComputePipelines(device, pipelineCache, 1, &ci, nullptr, &out), "vkCreateComputePipelines(fsr)");
    vk.vkDestroyShaderModule(device, m, nullptr);
    return ok;
  };
  if (!pipe(builtin::kFsrDepthComp, sizeof(builtin::kFsrDepthComp), fsrDepthPipe) ||
      !pipe(builtin::kFsrInputsComp, sizeof(builtin::kFsrInputsComp), fsrInputsPipe))
    return false;
  VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
  sci.magFilter = sci.minFilter = VK_FILTER_NEAREST;
  sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  if (!check(vk.vkCreateSampler(device, &sci, nullptr, &fsrPoint), "vkCreateSampler(fsr)")) return false;
  for (VkDescriptorPool& p : fsrPools) {
    const VkDescriptorPoolSize sizes[3] = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16 },
                                            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16 },
                                            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8 } };
    VkDescriptorPoolCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    ci.maxSets = 8;
    ci.poolSizeCount = 3;
    ci.pPoolSizes = sizes;
    if (!check(vk.vkCreateDescriptorPool(device, &ci, nullptr, &p), "vkCreateDescriptorPool(fsr)")) return false;
  }
  return true;
}
VkDescriptorSet fsrSet() {
  const uint32_t s = uint32_t(frameNumber % kFramesInFlight);
  if (fsrPoolFrame[s] != frameNumber) {  // this slot's previous frame has completed (beginFrameIfNeeded waited)
    vk.vkResetDescriptorPool(device, fsrPools[s], 0);
    fsrPoolFrame[s] = frameNumber;
  }
  VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
  ai.descriptorPool = fsrPools[s];
  ai.descriptorSetCount = 1;
  ai.pSetLayouts = &fsrSetLayout;
  VkDescriptorSet set = VK_NULL_HANDLE;
  return check(vk.vkAllocateDescriptorSets(device, &ai, &set), "vkAllocateDescriptorSets(fsr)") ? set : VK_NULL_HANDLE;
}
void writeFsrImage(VkDescriptorSet set, uint32_t binding, VkImageView view, bool storage) {
  VkDescriptorImageInfo ii{ storage ? VK_NULL_HANDLE : fsrPoint, view, VK_IMAGE_LAYOUT_GENERAL };
  VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
  w.dstSet = set;
  w.dstBinding = binding;
  w.descriptorCount = 1;
  w.descriptorType = storage ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  w.pImageInfo = &ii;
  vk.vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
}
bool ensureStorageImage(FxImage& img, uint32_t w, uint32_t h, VkFormat f) {
  if (!ensureFxImage(img, w, h, f, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return false;
  if (img.layout != VK_IMAGE_LAYOUT_GENERAL) barrier(img.image, img.layout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
  return true;
}
void computeBarrier() {  // compute writes -> any later read (the FSR dispatch, blits)
  VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
  mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  mb.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
  mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
  mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
  VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
  di.memoryBarrierCount = 1;
  di.pMemoryBarriers = &mb;
  vk.vkCmdPipelineBarrier2(slot().cmd, &di);
}

void doCaptureWorldDepth() {
  worldDepthValid = false;
  if (!upscalingWanted()) return;
  if (!beginFrameIfNeeded() || !initFsrPasses()) return;
  const RtKey dk = rendering ? groupDepth : lastDepth;
  Rt* d = dk ? rtFor(dk, groupLow) : nullptr;
  if (!d || !d->desc.depth || d->desc.samples > 1 || !d->sampleView) {
    logOnce("fsrworlddepth", "upscaling: the world pass has no single-sample depth buffer: no first-person separation");
    return;
  }
  const bool resume = rendering && !renderingSuspended;
  if (resume) suspendRendering();
  const VkImageLayout old = d->layout;
  if (d->layout != VK_IMAGE_LAYOUT_GENERAL) barrier(d->image, d->layout, VK_IMAGE_LAYOUT_GENERAL, aspectOf(*d));
  const uint32_t w = d->desc.width, h = d->desc.height;
  VkDescriptorSet set = ensureStorageImage(worldDepthImg, w, h, VK_FORMAT_R32_SFLOAT) ? fsrSet() : VK_NULL_HANDLE;
  if (set) {
    writeFsrImage(set, 0, d->sampleView, false);
    writeFsrImage(set, 3, worldDepthImg.view, true);
    VkCommandBuffer cmd = slot().cmd;
    vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fsrDepthPipe);
    vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fsrLayout, 0, 1, &set, 0, nullptr);
    vk.vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
    computeBarrier();
    worldDepthValid = true;
  }
  if (old == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) barrier(d->image, d->layout, old, aspectOf(*d));
  if (resume) resumeRendering();
}

// --- settings ---------------------------------------------------------------------------------------------------------
bool upscalingWanted() const { return upSettings.upscaler != UpscaleSettings::Off; }

void doSetUpscaling(const UpscaleSettings& s) {
  if (s.library != upSettings.library && !ffx.lib) ffx.loadTried = false;  // a new path is tried once more
  upSettings = s;
  if (s.upscaler != UpscaleSettings::Off) {
    ffxLoaded();
    return;
  }
  destroyUpscaleContext();
  for (auto& [k, t] : lowRts) release(t, true);  // no twins without an upscaler
  lowRts.clear();
  std::lock_guard<std::mutex> lk(upStatusMutex);
  upStatus.error.clear();
}

void doBeginScaledScene() {
  if (scaledScene || !upscalingWanted() || !ffxLoaded() || !screenW || !screenH) return;
  if (!beginFrameIfNeeded()) return;
  if (rendering) doEndGroup();  // a group open from before is full-size
  Backend::renderSize(upSettings, screenW, screenH, sceneW, sceneH);
  if (!sceneW || !sceneH) return;
  scaledScene = true;
  worldDepthValid = false;
  lowDirty.clear();
}

void doUpscaleScene(RtKey colorKey, RtKey depthKey, const SceneView& view) {
  if (!beginFrameIfNeeded()) return;
  if (rendering) doEndGroup();
  // doBeginScaledScene ran: an upscaler and its library.
  if (!scaledScene || !initFsrPasses()) {
    finishScaledScene(0);
    return;
  }
  Rt* full = lookup(colorKey);
  Rt* color = rtFor(colorKey, true);
  Rt* depth = rtFor(depthKey, true);
  if (!full || !realize(colorKey, *full) || !color || !depth || !depth->desc.depth || depth->desc.samples > 1 ||
      color->desc.samples > 1 || !depth->sampleView || color == full) {
    logOnce("fsrinputs", "upscaling: the scene texture has no single-sample colour and depth (multisampling is on?): "
            "the scene is not upscaled");
    setUpError("the scene needs single-sample colour and depth (turn the game's antialiasing off)");
    finishScaledScene(0);
    return;
  }
  const uint32_t rw = color->desc.width, rh = color->desc.height;
  if (depth->desc.width < rw || depth->desc.height < rh) {
    finishScaledScene(0);
    return;
  }
  VkCommandBuffer cmd = slot().cmd;
  for (Rt* r : { color, depth })
    if (r->layout != VK_IMAGE_LAYOUT_GENERAL) barrier(r->image, r->layout, VK_IMAGE_LAYOUT_GENERAL, aspectOf(*r));
  // Depth and motion vectors.
  if (!ensureStorageImage(fsrDepthImg, rw, rh, VK_FORMAT_R32_SFLOAT) ||
      !ensureStorageImage(motionImg, rw, rh, VK_FORMAT_R16G16_SFLOAT)) {
    finishScaledScene(0);
    return;
  }
  const bool world = worldDepthValid && worldDepthImg.w == rw && worldDepthImg.h == rh;
  const bool reset = view.reset || !prevViewValid || ffx.upReset;
  struct Cam { float rot[3][4], prevRot[3][4], frustum[4], prevFrustum[4], port[4], info[4]; } cam{};
  const SceneView& pv = prevViewValid ? prevView : view;
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      cam.rot[r][c] = view.cameraRows[r][c];
      cam.prevRot[r][c] = pv.cameraRows[r][c];
    }
    cam.rot[r][3] = float(double(view.cameraRows[r][3]) - double(pv.cameraRows[r][3]));  // T - Tprev
  }
  memcpy(cam.frustum, view.frustum, sizeof(cam.frustum));
  memcpy(cam.prevFrustum, pv.frustum, sizeof(cam.prevFrustum));
  // The camera viewport in the image, as doRunEffect (NiDX9Renderer's NiViewport to D3D viewport conversion).
  const float W = float(rw), H = float(rh);
  cam.port[0] = std::clamp(std::floor(view.port[0] * W), 0.0f, W);
  cam.port[1] = std::clamp(std::floor((1.0f - view.port[2]) * H), 0.0f, H);
  cam.port[2] = std::max(1.0f, std::min(std::floor((view.port[1] - view.port[0]) * W), W - cam.port[0]));
  cam.port[3] = std::max(1.0f, std::min(std::floor((view.port[2] - view.port[3]) * H), H - cam.port[1]));
  cam.info[0] = view.nearZ;
  cam.info[1] = view.farZ;
  cam.info[2] = world ? 1.0f : 0.0f;
  cam.info[3] = reset ? 0.0f : 1.0f;
  HostSlice ub;
  VkDescriptorSet set = fsrSet();
  if (!set || !hostAlloc(drawSlot().uniforms, sizeof(cam), uboAlign, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, ub)) {
    finishScaledScene(0);
    return;
  }
  memcpy(ub.ptr, &cam, sizeof(cam));
  writeFsrImage(set, 0, depth->sampleView, false);
  writeFsrImage(set, 1, world ? worldDepthImg.view : depth->sampleView, false);
  writeFsrImage(set, 2, motionImg.view, true);
  writeFsrImage(set, 3, fsrDepthImg.view, true);
  VkDescriptorBufferInfo bi{ ub.buffer, ub.offset, sizeof(cam) };
  VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
  w.dstSet = set;
  w.dstBinding = 4;
  w.descriptorCount = 1;
  w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  w.pBufferInfo = &bi;
  vk.vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
  vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fsrInputsPipe);
  vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fsrLayout, 0, 1, &set, 0, nullptr);
  vk.vkCmdDispatch(cmd, (rw + 7) / 8, (rh + 7) / 8, 1);
  computeBarrier();
  invalidateDrawState();

  // Camera. Bethesda's units: 70 to the metre.
  const float fovY = std::atan(view.frustum[2]) - std::atan(view.frustum[3]);
  constexpr float kMetresPerUnit = 1.0f / 70.0f;
  const FfxApiResource fDepth =
      ffxImage(fsrDepthImg.image, fsrDepthImg.format, rw, rh, false, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
  const FfxApiResource fMotion =
      ffxImage(motionImg.image, motionImg.format, rw, rh, false, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);

  bool upscaled = false;
  {
    // HDR: the game's HDR scene texture (A16B16G16R16F, linear light before the tone curve); otherwise 8-bit colour.
    const bool hdrScene = full->desc.d3dFormat == kD3DFMT_A16B16G16R16F;
    const uint32_t flags = FFX_UPSCALE_ENABLE_AUTO_EXPOSURE | (hdrScene ? FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE : 0u);
    const uint32_t dw = full->desc.width, dh = full->desc.height;
    if (ensureUpscaleContext(dw, dh, flags)) {
      // The output: the full-size target itself when it can be a storage image, else an RGBA16F image blitted into it.
      const bool direct = full->storage;
      if (direct || ensureStorageImage(fsrOutImg, dw, dh, VK_FORMAT_R16G16B16A16_SFLOAT)) {
        if (full->layout != VK_IMAGE_LAYOUT_GENERAL)
          barrier(full->image, full->layout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
        ffxDispatchDescUpscale d{};
        d.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        d.commandList = cmd;
        d.color = ffxImage(color->image, color->format, rw, rh, false, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        d.depth = fDepth;
        d.motionVectors = fMotion;
        d.output = direct ? ffxImage(full->image, full->format, dw, dh, true, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS)
                          : ffxImage(fsrOutImg.image, fsrOutImg.format, dw, dh, true, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        d.jitterOffset = { view.jitter[0], view.jitter[1] };
        d.motionVectorScale = { W, H };
        d.renderSize = { rw, rh };
        d.upscaleSize = { dw, dh };
        d.enableSharpening = upSettings.sharpen;
        d.sharpness = std::clamp(upSettings.sharpness, 0.0f, 1.0f);
        d.frameTimeDelta = view.frameTimeMs;
        d.preExposure = 1.0f;
        d.reset = reset;
        d.cameraNear = view.nearZ;
        d.cameraFar = view.farZ;
        d.cameraFovAngleVertical = fovY;
        d.viewSpaceToMetersFactor = kMetresPerUnit;
        const ffxReturnCode_t r = [&] {
          return ffx.f.Dispatch(&ffx.upscale, &d.header);
        }();
        invalidateDrawState();
        if (r == FFX_API_RETURN_OK) {
          upscaled = true;
          ffx.upReset = false;
          st.fsrUpscales++;
          if (!direct) {
            barrier(fsrOutImg.image, fsrOutImg.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
            barrier(full->image, full->layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
            VkImageBlit bl{};
            bl.srcSubresource = bl.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            bl.srcOffsets[1] = bl.dstOffsets[1] = { int32_t(dw), int32_t(dh), 1 };
            vk.vkCmdBlitImage(cmd, fsrOutImg.image, fsrOutImg.layout, full->image, full->layout, 1, &bl, VK_FILTER_NEAREST);
            barrier(fsrOutImg.image, fsrOutImg.layout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
            barrier(full->image, full->layout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
          }
        } else {
          logOnce("fsrdispatch" + std::to_string(r), "upscaling: the FSR dispatch failed (%u)", unsigned(r));
          setUpError("the FSR dispatch failed (" + std::to_string(r) + ")");
        }
      }
    }
    finishScaledScene(upscaled ? colorKey : 0);
  }

  prevView = view;
  prevViewValid = true;
  std::lock_guard<std::mutex> lk(upStatusMutex);
  upStatus.upscaling = upscaled;
  upStatus.renderW = rw;
  upStatus.renderH = rh;
  upStatus.displayW = full->desc.width;
  upStatus.displayH = full->desc.height;
}

void teardownFfx() {  // the device is idle
  destroyUpscaleContext();
  collectFfxGarbage(true);
  if (fsrInputsPipe) vk.vkDestroyPipeline(device, fsrInputsPipe, nullptr);
  if (fsrDepthPipe) vk.vkDestroyPipeline(device, fsrDepthPipe, nullptr);
  if (fsrLayout) vk.vkDestroyPipelineLayout(device, fsrLayout, nullptr);
  if (fsrSetLayout) vk.vkDestroyDescriptorSetLayout(device, fsrSetLayout, nullptr);
  if (fsrPoint) vk.vkDestroySampler(device, fsrPoint, nullptr);
  for (VkDescriptorPool& p : fsrPools)
    if (p) vk.vkDestroyDescriptorPool(device, p, nullptr);
  fsrInputsPipe = fsrDepthPipe = VK_NULL_HANDLE;
  fsrLayout = VK_NULL_HANDLE;
  fsrSetLayout = VK_NULL_HANDLE;
  fsrPoint = VK_NULL_HANDLE;
  for (FxImage* i : { &worldDepthImg, &fsrDepthImg, &motionImg, &fsrOutImg }) releaseFxImage(*i);
  for (auto& [k, t] : lowRts) release(t, true);
  lowRts.clear();
}

UpscaleStatus statusSnapshot() const {
  std::lock_guard<std::mutex> lk(upStatusMutex);
  return upStatus;
}
