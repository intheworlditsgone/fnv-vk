// Sun shadow maps (included inside Backend::Impl, after backend_draw.inl).
//
// One depth-only 2D-array image, a layer per cascade. The plugin walks the world and hands over each caster with its
// own vertex/index buffers and declaration (the game's) and its object-to-world transform; the backend draws its
// positions with builtin/shadow.vert (alpha-tested casters also run builtin/shadow.frag, which discards where the
// diffuse texture's alpha is below 0.5). Effects sample the result through fnvShadow() (effectlib.cpp header).
// The game's render-target group is suspended for the whole shadow pass and resumed afterwards.

struct ShadowPush {
  float viewProj[16];  // row-major: clip = M * (world, 1)
  float world[12];     // three rows: world = rows * (object, 1)
};
static_assert(sizeof(ShadowPush) == 112);

VkImage shadowImage = VK_NULL_HANDLE;
VkDeviceMemory shadowMemory = VK_NULL_HANDLE;
VkImageView shadowArrayView = VK_NULL_HANDLE;      // all layers, sampled by effects
std::vector<VkImageView> shadowLayerViews;         // one per cascade, depth attachment
VkImageLayout shadowLayout = VK_IMAGE_LAYOUT_UNDEFINED;
uint32_t shadowSize = 0, shadowLayers = 0;
bool shadowValid = false;      // the image holds this frame's (or the last) cascades
bool shadowPassOpen = false, shadowCascadeOpen = false, shadowResume = false;
ShadowPush shadowPush{};
VkPipelineLayout shadowLayout0 = VK_NULL_HANDLE;   // push constants + the alpha set
VkDescriptorSetLayout shadowAlphaSetLayout = VK_NULL_HANDLE;
std::vector<VkDescriptorPool> shadowPools[kFramesInFlight];
size_t shadowPool[kFramesInFlight] = {};
VkShaderModule shadowVs = VK_NULL_HANDLE, shadowAlphaVs = VK_NULL_HANDLE, shadowPs = VK_NULL_HANDLE;
std::unordered_map<uint64_t, VkPipeline> shadowPipes;  // by declaration layout, topology, alpha
VkSampler shadowCompareSampler = VK_NULL_HANDLE;       // for effects: depth compare, linear (2x2 PCF in hardware)

void destroyShadowImage() {
  for (VkImageView v : shadowLayerViews)
    garbage.push_back({ frameNumber, { VK_NULL_HANDLE, VK_NULL_HANDLE }, { VK_NULL_HANDLE, VK_NULL_HANDLE }, { v, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE } });
  shadowLayerViews.clear();
  if (shadowImage)
    garbage.push_back({ frameNumber, { shadowImage, VK_NULL_HANDLE }, { shadowMemory, VK_NULL_HANDLE }, { shadowArrayView, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE } });
  shadowImage = VK_NULL_HANDLE;
  shadowMemory = VK_NULL_HANDLE;
  shadowArrayView = VK_NULL_HANDLE;
  shadowLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  shadowSize = shadowLayers = 0;
  shadowValid = false;
}

bool ensureShadowImage(uint32_t layers, uint32_t size) {
  if (shadowImage && shadowLayers == layers && shadowSize == size) return true;
  destroyShadowImage();
  VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = VK_FORMAT_D32_SFLOAT;
  ici.extent = { size, size, 1 };
  ici.mipLevels = 1;
  ici.arrayLayers = layers;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  if (!check(vk.vkCreateImage(device, &ici, nullptr, &shadowImage), "vkCreateImage(shadow)")) return false;
  VkMemoryRequirements req{};
  vk.vkGetImageMemoryRequirements(device, shadowImage, &req);
  VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if (mai.memoryTypeIndex == UINT32_MAX || !check(vk.vkAllocateMemory(device, &mai, nullptr, &shadowMemory), "vkAllocateMemory(shadow)") ||
      !check(vk.vkBindImageMemory(device, shadowImage, shadowMemory, 0), "vkBindImageMemory(shadow)")) {
    vk.vkDestroyImage(device, shadowImage, nullptr);
    shadowImage = VK_NULL_HANDLE;
    return false;
  }
  VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
  vci.image = shadowImage;
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  vci.format = VK_FORMAT_D32_SFLOAT;
  vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, layers };
  if (!check(vk.vkCreateImageView(device, &vci, nullptr, &shadowArrayView), "vkCreateImageView(shadow array)")) return false;
  for (uint32_t l = 0; l < layers; l++) {
    VkImageView v = VK_NULL_HANDLE;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, l, 1 };
    if (!check(vk.vkCreateImageView(device, &vci, nullptr, &v), "vkCreateImageView(shadow layer)")) return false;
    shadowLayerViews.push_back(v);
  }
  shadowLayers = layers;
  shadowSize = size;
  shadowLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  return true;
}

bool initShadowObjects() {
  if (shadowLayout0) return true;
  VkDescriptorSetLayoutBinding b[2] = {};
  b[0] = { 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
  b[1] = { 1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
  VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
  lci.bindingCount = 2;
  lci.pBindings = b;
  if (!check(vk.vkCreateDescriptorSetLayout(device, &lci, nullptr, &shadowAlphaSetLayout), "vkCreateDescriptorSetLayout(shadow)")) return false;
  VkPushConstantRange pr{ VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPush) };
  VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &shadowAlphaSetLayout;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pr;
  if (!check(vk.vkCreatePipelineLayout(device, &plci, nullptr, &shadowLayout0), "vkCreatePipelineLayout(shadow)")) return false;
  VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
  smci.codeSize = sizeof(builtin::kShadowVert);
  smci.pCode = builtin::kShadowVert;
  if (!check(vk.vkCreateShaderModule(device, &smci, nullptr, &shadowVs), "vkCreateShaderModule(shadow vs)")) return false;
  smci.codeSize = sizeof(builtin::kShadowAlphaVert);
  smci.pCode = builtin::kShadowAlphaVert;
  if (!check(vk.vkCreateShaderModule(device, &smci, nullptr, &shadowAlphaVs), "vkCreateShaderModule(shadow alpha vs)")) return false;
  smci.codeSize = sizeof(builtin::kShadowFrag);
  smci.pCode = builtin::kShadowFrag;
  if (!check(vk.vkCreateShaderModule(device, &smci, nullptr, &shadowPs), "vkCreateShaderModule(shadow ps)")) return false;
  VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
  sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
  sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  sci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;  // outside a cascade: lit
  sci.compareEnable = VK_TRUE;
  sci.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;            // 1 when the receiver is at or in front of the caster
  sci.maxLod = 0.0f;
  return check(vk.vkCreateSampler(device, &sci, nullptr, &shadowCompareSampler), "vkCreateSampler(shadow compare)");
}

// Before the first shadow pass: a 1-layer map cleared to the far plane (lit), so the cascade bindings are valid. Records
// outside any render pass (suspends the open group).
bool ensureShadowPlaceholder() {
  if (shadowImage) return true;
  if (!initShadowObjects() || !ensureShadowImage(1, 16)) return false;
  const bool resume = rendering && !renderingSuspended;
  if (resume) suspendRendering();
  barrier(shadowImage, shadowLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
  const VkClearDepthStencilValue farPlane{ 1.0f, 0 };
  const VkImageSubresourceRange all{ VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
  vk.vkCmdClearDepthStencilImage(slot().cmd, shadowImage, shadowLayout, &farPlane, 1, &all);
  barrier(shadowImage, shadowLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT);
  if (resume) resumeRendering();
  return true;
}

// Sun shadows in the game's lighting: the state the plugin sets (Backend::setSunLighting) and the block of the sun-lit
// pixel shaders (layout: sdp2spv's sun shadow pass). False (and the block's enable 0) when this draw is not shadowed.
Backend::SunLighting sunLighting{};
bool sunLightingOn = false;
bool sunBlock(float (&blk)[21][4]) {
  const Backend::SunLighting& s = sunLighting;
  if (!sunLightingOn || !shadowValid || !s.cascades || s.cascades > shadowLayers || groupColors.empty()) return false;
  if (sunGateOverride >= 0 ? sunGateOverride == 0 : s.gate && !s.gate(s.ctx)) return false;
  // The camera viewport in the target, as runEffect (NiDX9Renderer's NiViewport to D3D viewport conversion).
  const Rt& color = *rtFor(groupColors[0], groupLow);
  const float W = float(color.desc.width), H = float(color.desc.height);
  const float rx = std::clamp(std::floor(s.port[0] * W), 0.0f, W), ry = std::clamp(std::floor((1.0f - s.port[2]) * H), 0.0f, H);
  const float rw = std::min(std::floor((s.port[1] - s.port[0]) * W), W - rx);
  const float rh = std::min(std::floor((s.port[2] - s.port[3]) * H), H - ry);
  if (rw < 1.0f || rh < 1.0f) return false;
  const uint32_t n = s.cascades;
  const float last = s.splits[n - 1];
  blk[0][0] = 1.0f;
  blk[0][1] = last;
  blk[0][2] = 1.0f / float(shadowSize);
  blk[0][3] = s.strength;
  const float rect[4] = { rx, ry, rw, rh };
  memcpy(blk[1], rect, 16);
  memcpy(blk[2], s.frustum, 16);
  blk[3][0] = s.nearZ;
  blk[3][1] = s.farZ;
  // Far splits: the shader picks the cascade by how many it is beyond, so the ones from the last cascade on are huge.
  for (uint32_t i = 0; i < 4; i++) blk[4][i] = i + 1 < n ? s.splits[i] : 3.0e38f;
  for (uint32_t c = 0; c < n; c++) memcpy(blk[5 + 4 * c], s.shadowFromView[c], 64);
  return true;
}

VkDescriptorSet allocShadowSet() {
  const uint32_t s = uint32_t(frameNumber % kFramesInFlight);
  for (;;) {
    if (shadowPool[s] == shadowPools[s].size()) {
      VkDescriptorPoolSize sizes[] = { { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 256 }, { VK_DESCRIPTOR_TYPE_SAMPLER, 256 } };
      VkDescriptorPoolCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
      ci.maxSets = 256;
      ci.poolSizeCount = 2;
      ci.pPoolSizes = sizes;
      VkDescriptorPool p = VK_NULL_HANDLE;
      if (!check(vk.vkCreateDescriptorPool(device, &ci, nullptr, &p), "vkCreateDescriptorPool(shadow)")) return VK_NULL_HANDLE;
      shadowPools[s].push_back(p);
    }
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = shadowPools[s][shadowPool[s]];
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &shadowAlphaSetLayout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    const VkResult r = vk.vkAllocateDescriptorSets(device, &ai, &set);
    if (r == VK_SUCCESS) return set;
    if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL) return VK_NULL_HANDLE;
    shadowPool[s]++;
  }
}
void resetShadowPools() {  // the slot's previous frame has completed (beginDrawSlot)
  const uint32_t s = uint32_t(frameNumber % kFramesInFlight);
  for (VkDescriptorPool p : shadowPools[s]) vk.vkResetDescriptorPool(device, p, 0);
  shadowPool[s] = 0;
  shadowAlphaSets.clear();
}
std::unordered_map<TexKey, VkDescriptorSet> shadowAlphaSets;  // this frame's, by texture

// The caster's position (POSITION0) and, alpha-tested, texture coordinates (TEXCOORD0) from its declaration.
struct ShadowInputs {
  const VertexElement* pos = nullptr;
  const VertexElement* uv = nullptr;
};
VkPipeline shadowPipelineFor(const ShadowInputs& in, const uint32_t* strides, uint32_t topology, bool alpha) {
  const uint64_t key = (uint64_t(in.pos->stream) << 0) | (uint64_t(in.pos->offset) << 4) | (uint64_t(in.pos->type) << 16) |
                       (uint64_t(strides[in.pos->stream]) << 24) | (uint64_t(topology) << 32) | (uint64_t(alpha) << 36) |
                       (alpha ? (uint64_t(in.uv->stream) << 37) | (uint64_t(in.uv->offset) << 41) | (uint64_t(in.uv->type) << 49) |
                                    (uint64_t(strides[in.uv->stream] & 0xFF) << 54) : 0);
  auto it = shadowPipes.find(key);
  if (it != shadowPipes.end()) return it->second;
  VkVertexInputBindingDescription bind[2] = {};
  VkVertexInputAttributeDescription attr[2] = {};
  uint32_t nb = 0, na = 0;
  auto addBinding = [&](uint32_t stream) {
    for (uint32_t i = 0; i < nb; i++) if (bind[i].binding == stream) return;
    bind[nb++] = { stream, strides[stream], VK_VERTEX_INPUT_RATE_VERTEX };
  };
  addBinding(in.pos->stream);
  attr[na++] = { 0, in.pos->stream, xlat::declType(in.pos->type), in.pos->offset };
  if (alpha) {
    addBinding(in.uv->stream);
    attr[na++] = { 1, in.uv->stream, xlat::declType(in.uv->type), in.uv->offset };
  }
  VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
  vi.vertexBindingDescriptionCount = nb;
  vi.pVertexBindingDescriptions = bind;
  vi.vertexAttributeDescriptionCount = na;
  vi.pVertexAttributeDescriptions = attr;
  VkPipelineShaderStageCreateInfo stages[2] = { { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
                                                { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO } };
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = alpha ? shadowAlphaVs : shadowVs;
  stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = shadowPs;
  stages[1].pName = "main";
  VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
  ia.topology = VkPrimitiveTopology(topology);
  VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
  vp.viewportCount = vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = VK_CULL_MODE_NONE;  // the game's meshes are not closed: both sides cast
  rs.lineWidth = 1.0f;
  rs.depthBiasEnable = VK_TRUE;     // against acne on lit surfaces facing the sun
  rs.depthBiasConstantFactor = 2.0f;
  rs.depthBiasSlopeFactor = 2.5f;
  rs.depthClampEnable = VK_FALSE;
  VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
  ds.depthTestEnable = VK_TRUE;
  ds.depthWriteEnable = VK_TRUE;
  ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
  VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
  VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
  VkPipelineDynamicStateCreateInfo dy{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
  dy.dynamicStateCount = 2;
  dy.pDynamicStates = dyn;
  VkPipelineRenderingCreateInfo rci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
  rci.depthAttachmentFormat = VK_FORMAT_D32_SFLOAT;
  VkGraphicsPipelineCreateInfo pci{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci };
  pci.stageCount = alpha ? 2 : 1;  // depth only: no fragment stage
  pci.pStages = stages;
  pci.pVertexInputState = &vi;
  pci.pInputAssemblyState = &ia;
  pci.pViewportState = &vp;
  pci.pRasterizationState = &rs;
  pci.pMultisampleState = &ms;
  pci.pDepthStencilState = &ds;
  pci.pColorBlendState = &cb;
  pci.pDynamicState = &dy;
  pci.layout = shadowLayout0;
  VkPipeline p = VK_NULL_HANDLE;
  check(vk.vkCreateGraphicsPipelines(device, pipelineCache, 1, &pci, nullptr, &p), "vkCreateGraphicsPipelines(shadow)");
  shadowPipes[key] = p;
  return p;
}

bool doBeginShadowMaps(uint32_t cascades, uint32_t size) {
  if (cascades == 0 || cascades > 4 || size < 256 || !beginFrameIfNeeded() || !initShadowObjects()) return false;
  if (!ensureShadowImage(cascades, size)) return false;
  shadowResume = rendering && !renderingSuspended;
  if (shadowResume) suspendRendering();
  barrier(shadowImage, shadowLayout, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT, VK_REMAINING_ARRAY_LAYERS);
  shadowPassOpen = true;
  shadowCascadeOpen = false;
  shadowValid = false;
  return true;
}

void endShadowCascade() {
  if (!shadowCascadeOpen) return;
  vk.vkCmdEndRendering(slot().cmd);
  shadowCascadeOpen = false;
}

void doBeginShadowCascade(uint32_t index, const float viewProj[16]) {
  if (!shadowPassOpen || index >= shadowLayers) return;
  endShadowCascade();
  VkRenderingAttachmentInfo da{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
  da.imageView = shadowLayerViews[index];
  da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
  da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  da.clearValue.depthStencil = { 1.0f, 0 };
  VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
  ri.renderArea = { { 0, 0 }, { shadowSize, shadowSize } };
  ri.layerCount = 1;
  ri.pDepthAttachment = &da;
  VkCommandBuffer cmd = slot().cmd;
  vk.vkCmdBeginRendering(cmd, &ri);
  const VkViewport v{ 0, 0, float(shadowSize), float(shadowSize), 0.0f, 1.0f };
  const VkRect2D sc{ { 0, 0 }, { shadowSize, shadowSize } };
  vk.vkCmdSetViewport(cmd, 0, 1, &v);
  vk.vkCmdSetScissor(cmd, 0, 1, &sc);
  memcpy(shadowPush.viewProj, viewProj, 64);
  shadowCascadeOpen = true;
}

void doDrawShadowCaster(const Backend::ShadowCaster& c) {
  if (!shadowCascadeOpen || c.streamCount == 0 || c.streamCount > 4 || (c.primitiveCount == 0 && c.stripCount == 0)) return;
  auto dit = decls.find(c.decl);
  if (dit == decls.end()) return;
  ShadowInputs in;
  for (const VertexElement& e : dit->second) {
    if (e.usage == 0 /*POSITION*/ && e.usageIndex == 0 && !in.pos) in.pos = &e;
    if (e.usage == 5 /*TEXCOORD*/ && e.usageIndex == 0 && !in.uv) in.uv = &e;
  }
  if (!in.pos || in.pos->stream >= c.streamCount || !c.strides[in.pos->stream]) return;
  VkImageView alphaView = VK_NULL_HANDLE;
  if (c.alphaTexture && in.uv && in.uv->stream < c.streamCount) {
    auto t = textures.find(c.alphaTexture);
    if (t != textures.end() && t->second.view) alphaView = t->second.view;
  }
  const bool alpha = alphaView != VK_NULL_HANDLE;
  const uint32_t topology = c.d3dPrimitive == 5 ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  if (c.d3dPrimitive != 4 && c.d3dPrimitive != 5) return;
  VkPipeline pipe = shadowPipelineFor(in, c.strides, topology, alpha);
  if (!pipe) return;
  VkCommandBuffer cmd = slot().cmd;
  // Vertex buffers of the streams the pipeline reads.
  uint32_t used[2] = { in.pos->stream, alpha ? uint32_t(in.uv->stream) : uint32_t(in.pos->stream) };
  for (uint32_t i = 0; i < (alpha ? 2u : 1u); i++) {
    auto it = buffers.find(c.streams[used[i]]);
    if (it == buffers.end() || it->second.index) return;
    it->second.lastUse = frameNumber;
    const VkDeviceSize off = 0;
    vk.vkCmdBindVertexBuffers(cmd, used[i], 1, &it->second.cur.buffer, &off);
  }
  vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
  if (alpha) {
    VkDescriptorSet& set = shadowAlphaSets[c.alphaTexture];
    if (!set) {
      set = allocShadowSet();
      if (!set) return;
      uint32_t ss[14] = {};
      ss[D3DSAMP_ADDRESSU] = ss[D3DSAMP_ADDRESSV] = ss[D3DSAMP_ADDRESSW] = D3DTADDRESS_WRAP;
      ss[D3DSAMP_MAGFILTER] = ss[D3DSAMP_MINFILTER] = D3DTEXF_LINEAR;
      ss[D3DSAMP_MIPFILTER] = D3DTEXF_LINEAR;
      ss[D3DSAMP_MAXANISOTROPY] = 1;
      const VkDescriptorImageInfo img{ VK_NULL_HANDLE, alphaView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
      const VkDescriptorImageInfo smp{ samplerFor(ss, false), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
      VkWriteDescriptorSet w[2] = { { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }, { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET } };
      w[0].dstSet = w[1].dstSet = set;
      w[0].dstBinding = 0;
      w[1].dstBinding = 1;
      w[0].descriptorCount = w[1].descriptorCount = 1;
      w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
      w[0].pImageInfo = &img;
      w[1].pImageInfo = &smp;
      vk.vkUpdateDescriptorSets(device, 2, w, 0, nullptr);
    }
    vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowLayout0, 0, 1, &set, 0, nullptr);
  }
  memcpy(shadowPush.world, c.world, 48);
  vk.vkCmdPushConstants(cmd, shadowLayout0, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(ShadowPush), &shadowPush);
  const uint32_t count = c.d3dPrimitive == 5 ? c.primitiveCount + 2 : c.primitiveCount * 3;
  if (c.indices) {
    auto it = buffers.find(c.indices);
    if (it == buffers.end() || !it->second.index) return;
    it->second.lastUse = frameNumber;
    vk.vkCmdBindIndexBuffer(cmd, it->second.cur.buffer, 0, it->second.index32 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
    if (c.stripCount && c.stripLengths && c.d3dPrimitive == 5) {
      // Every strip with the setup above: one draw each.
      uint32_t first = c.start;
      for (uint32_t i = 0; i < c.stripCount; i++) {
        const uint32_t n = c.stripLengths[i];
        if (n >= 3) {
          vk.vkCmdDrawIndexed(cmd, n, 1, first, c.baseVertex, 0);
          st.shadowCasters++;
        }
        first += n;
      }
      return;
    }
    vk.vkCmdDrawIndexed(cmd, count, 1, c.start, c.baseVertex, 0);
  } else {
    vk.vkCmdDraw(cmd, count, 1, c.start, 0);
  }
  st.shadowCasters++;
}

void doEndShadowMaps() {
  if (!shadowPassOpen) return;
  endShadowCascade();
  barrier(shadowImage, shadowLayout, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT, VK_REMAINING_ARRAY_LAYERS);
  shadowPassOpen = false;
  shadowValid = true;
  invalidateDrawState();  // other pipelines, layouts, vertex and index buffers were bound
  if (shadowResume) resumeRendering();
  shadowResume = false;
}

void destroyShadowPath() {
  for (auto& [k, p] : shadowPipes) if (p) vk.vkDestroyPipeline(device, p, nullptr);
  shadowPipes.clear();
  for (auto& pools : shadowPools) {
    for (VkDescriptorPool p : pools) vk.vkDestroyDescriptorPool(device, p, nullptr);
    pools.clear();
  }
  if (shadowVs) vk.vkDestroyShaderModule(device, shadowVs, nullptr);
  if (shadowAlphaVs) vk.vkDestroyShaderModule(device, shadowAlphaVs, nullptr);
  if (shadowPs) vk.vkDestroyShaderModule(device, shadowPs, nullptr);
  if (shadowLayout0) vk.vkDestroyPipelineLayout(device, shadowLayout0, nullptr);
  if (shadowAlphaSetLayout) vk.vkDestroyDescriptorSetLayout(device, shadowAlphaSetLayout, nullptr);
  if (shadowCompareSampler) vk.vkDestroySampler(device, shadowCompareSampler, nullptr);
  shadowVs = shadowAlphaVs = shadowPs = VK_NULL_HANDLE;
  shadowLayout0 = VK_NULL_HANDLE;
  shadowAlphaSetLayout = VK_NULL_HANDLE;
  shadowCompareSampler = VK_NULL_HANDLE;
  destroyShadowImage();  // deferred: freed with the rest of the garbage at teardown
}
