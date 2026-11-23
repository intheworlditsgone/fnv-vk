// Environment light for the replaced lighting shaders (included in Backend::Impl after backend_shadow.inl;
// docs/INDIRECT.md, docs/SHADER_ABI.md "Environment").
//
// The environment is a rendered cube map (attachCube; the plugin's probe) set with setEnvironment. Pixel replacements
// that declare the environment block (binding 10) get it with the cube (binding 50, full mip chain) and a trilinear
// clamp sampler (51). The block:
//   [0]    valid (0: no environment yet), the cube's last mip level, 0, 0
//   [1..9] the cube's irradiance as order-2 spherical harmonics, linear (the cube's gamma 2.2 values decoded), divided
//          by the luminance of its mean over the sphere: E(n) / mean(E) = sum c_i Y_i(n), Y the real SH basis in the
//          order Y00, Y1-1, Y10, Y11, Y2-2, Y2-1, Y20, Y21, Y22 (constants included in the shader's basis).
// Each time the environment cube is assembled with new contents (the probe completes a cube every six frames):
// - the linear cube: the cube decoded to linear light (gamma 2.2), with a mip chain of linear 2x averages (the cube's own
//   mips average the encoded values, which darkens edges between bright and dark);
// - binding 50, the prefiltered cube (builtin/prefilter.frag) from the linear cube: level l holds the split-sum radiance
//   for roughness (l / last)^2 (alpha), level 0 the linear cube itself; up to 128 x 128, down to 4 x 4;
// - the harmonics from the linear cube's 32 x 32 level, copied out and projected on the CPU when that frame has finished
//   on the GPU (a few frames later; never a wait).
static constexpr uint32_t kEnvBlockBinding = 10, kEnvImageBinding = 50, kEnvSamplerBinding = 51;
static constexpr uint32_t kEnvBlockVec4s = 10, kEnvReadbackSize = 32;

TexKey envCube = 0;
VkSampler envSampler = VK_NULL_HANDLE;
float envBlock[kEnvBlockVec4s][4] = {};
struct EnvReadback {
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation alloc = nullptr;
  void* map = nullptr;
  uint32_t size = 0;      // face edge of the copied level
  uint32_t lastLevel = 0; // the cube's last mip level when copied
  uint64_t frame = 0;     // frame whose command buffer holds the copy
  bool pending = false;
} envReadback;
uint64_t envUpdates = 0;
// The prefiltered cube: textures[Backend::kEnvironmentFiltered], with a 2D view per face and level to render into; the
// linear cube, with 2D views of its level 0 faces.
std::vector<VkImageView> envFaceViews;
Tex envLinear;
VkImageView envLinearFaces[6] = {};
VkDescriptorSetLayout envSetLayout = VK_NULL_HANDLE;
VkPipelineLayout envLayout = VK_NULL_HANDLE;
VkPipeline envPipe = VK_NULL_HANDLE;
VkShaderModule envPs = VK_NULL_HANDLE;
VkDescriptorPool envPool = VK_NULL_HANDLE;
static constexpr uint32_t kEnvSetRing = 16;  // two sets per prefilter, reused 8 prefilters later
VkDescriptorSet envSets[kEnvSetRing] = {};
uint32_t envSetNext = 0;

bool initEnvObjects() {
  if (envSampler) return true;
  VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
  sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
  sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
  sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  sci.maxLod = VK_LOD_CLAMP_NONE;
  return check(vk.vkCreateSampler(device, &sci, nullptr, &envSampler), "vkCreateSampler(environment)");
}

bool initPrefilter() {
  if (envPipe) return true;
  VkDescriptorSetLayoutBinding b[2] = {
    { 0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
    { 1, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr },
  };
  VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
  lci.bindingCount = 2;
  lci.pBindings = b;
  if (!check(vk.vkCreateDescriptorSetLayout(device, &lci, nullptr, &envSetLayout), "vkCreateDescriptorSetLayout(prefilter)"))
    return false;
  VkPushConstantRange pc{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, 20 };
  VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &envSetLayout;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pc;
  if (!check(vk.vkCreatePipelineLayout(device, &plci, nullptr, &envLayout), "vkCreatePipelineLayout(prefilter)")) return false;
  VkDescriptorPoolSize sizes[2] = { { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kEnvSetRing }, { VK_DESCRIPTOR_TYPE_SAMPLER, kEnvSetRing } };
  VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
  dpci.maxSets = kEnvSetRing;
  dpci.poolSizeCount = 2;
  dpci.pPoolSizes = sizes;
  if (!check(vk.vkCreateDescriptorPool(device, &dpci, nullptr, &envPool), "vkCreateDescriptorPool(prefilter)")) return false;
  VkDescriptorSetLayout layouts[kEnvSetRing];
  for (auto& l : layouts) l = envSetLayout;
  VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
  ai.descriptorPool = envPool;
  ai.descriptorSetCount = kEnvSetRing;
  ai.pSetLayouts = layouts;
  if (!check(vk.vkAllocateDescriptorSets(device, &ai, envSets), "vkAllocateDescriptorSets(prefilter)")) return false;
  VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
  smci.codeSize = sizeof(builtin::kFullscreenVert);
  smci.pCode = builtin::kFullscreenVert;
  VkShaderModule vs = VK_NULL_HANDLE;
  if (!check(vk.vkCreateShaderModule(device, &smci, nullptr, &vs), "vkCreateShaderModule(prefilter vs)")) return false;
  smci.codeSize = sizeof(builtin::kPrefilterFrag);
  smci.pCode = builtin::kPrefilterFrag;
  if (!check(vk.vkCreateShaderModule(device, &smci, nullptr, &envPs), "vkCreateShaderModule(prefilter)")) {
    vk.vkDestroyShaderModule(device, vs, nullptr);
    return false;
  }
  VkPipelineShaderStageCreateInfo stages[2] = { { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
                                                { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO } };
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = vs;
  stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = envPs;
  stages[1].pName = "main";
  VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
  VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
  vp.viewportCount = vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = VK_CULL_MODE_NONE;
  rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
  ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
  VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
  VkPipelineColorBlendAttachmentState cba{};
  cba.colorWriteMask = 0xF;
  VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
  cb.attachmentCount = 1;
  cb.pAttachments = &cba;
  VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
  VkPipelineDynamicStateCreateInfo dy{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
  dy.dynamicStateCount = 2;
  dy.pDynamicStates = dyn;
  const VkFormat fmt = VK_FORMAT_R16G16B16A16_SFLOAT;
  VkPipelineRenderingCreateInfo rci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
  rci.colorAttachmentCount = 1;
  rci.pColorAttachmentFormats = &fmt;
  VkGraphicsPipelineCreateInfo pci{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci };
  pci.stageCount = 2;
  pci.pStages = stages;
  pci.pVertexInputState = &vi;
  pci.pInputAssemblyState = &ia;
  pci.pViewportState = &vp;
  pci.pRasterizationState = &rs;
  pci.pMultisampleState = &ms;
  pci.pDepthStencilState = &ds;
  pci.pColorBlendState = &cb;
  pci.pDynamicState = &dy;
  pci.layout = envLayout;
  const bool ok = check(vk.vkCreateGraphicsPipelines(device, pipelineCache, 1, &pci, nullptr, &envPipe),
                        "vkCreateGraphicsPipelines(prefilter)");
  vk.vkDestroyShaderModule(device, vs, nullptr);
  return ok;
}

// A cube image (RGBA16F, colour attachment, sampled, transfer) with a cube view; `faceViews` 2D views per level and face
// (levels x 6), or of level 0 only when `allLevels` is false.
bool envCreateCube(uint32_t size, uint32_t levels, Tex& t, VkImageView* faceViews, bool allLevels) {
  VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
  ici.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = VK_FORMAT_R16G16B16A16_SFLOAT;
  ici.extent = { size, size, 1 };
  ici.mipLevels = levels;
  ici.arrayLayers = 6;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
              VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  VmaAllocationCreateInfo iai{};
  iai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  t = Tex{};
  t.format = ici.format;
  t.width = t.height = size;
  t.levels = levels;
  t.layers = 6;
  if (!check(vmaCreateImage(allocator, &ici, &iai, &t.image, &t.alloc, nullptr), "vmaCreateImage(environment cube)")) return false;
  VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
  vci.image = t.image;
  vci.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
  vci.format = t.format;
  vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, 6 };
  if (!check(vk.vkCreateImageView(device, &vci, nullptr, &t.view), "vkCreateImageView(environment cube)")) {
    releaseTexture(t);
    return false;
  }
  vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  for (uint32_t l = 0; l < (allLevels ? levels : 1u); l++)
    for (uint32_t f = 0; f < 6; f++) {
      vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, l, 1, f, 1 };
      check(vk.vkCreateImageView(device, &vci, nullptr, &faceViews[size_t(l) * 6 + f]), "vkCreateImageView(environment face)");
    }
  return true;
}
void envDropViews(VkImageView* views, size_t n) {
  for (size_t i = 0; i < n; i++)
    if (views[i]) addGarbage({ frameNumber, VK_NULL_HANDLE, views[i], VK_NULL_HANDLE, nullptr });
}

// Renders all six faces of one level (`dst` 2D views, COLOR_ATTACHMENT_OPTIMAL) with prefilter.frag from `src` (a cube
// view, SHADER_READ_ONLY).
void envRenderLevel(VkImageView src, const VkImageView* dst, uint32_t edge, float alpha, uint32_t srcSize, bool decode) {
  VkCommandBuffer cmd = slot().cmd;
  const VkDescriptorSet set = envSets[envSetNext++ % kEnvSetRing];
  const VkDescriptorImageInfo img{ VK_NULL_HANDLE, src, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
  const VkDescriptorImageInfo smp{ envSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
  VkWriteDescriptorSet w[2] = { { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }, { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET } };
  w[0].dstSet = set;
  w[0].dstBinding = 0;
  w[0].descriptorCount = 1;
  w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  w[0].pImageInfo = &img;
  w[1] = w[0];
  w[1].dstBinding = 1;
  w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
  w[1].pImageInfo = &smp;
  vk.vkUpdateDescriptorSets(device, 2, w, 0, nullptr);
  vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, envPipe);
  vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, envLayout, 0, 1, &set, 0, nullptr);
  for (uint32_t f = 0; f < 6; f++) {
    VkRenderingAttachmentInfo ca{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    ca.imageView = dst[f];
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea = { { 0, 0 }, { edge, edge } };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    vk.vkCmdBeginRendering(cmd, &ri);
    VkViewport vp{ 0.0f, 0.0f, float(edge), float(edge), 0.0f, 1.0f };
    VkRect2D sc{ { 0, 0 }, { edge, edge } };
    vk.vkCmdSetViewport(cmd, 0, 1, &vp);
    vk.vkCmdSetScissor(cmd, 0, 1, &sc);
    struct { uint32_t face; float alpha, srcSize, dstSize; uint32_t decode; } pc{ f, alpha, float(srcSize), float(edge),
                                                                                 decode ? 1u : 0u };
    vk.vkCmdPushConstants(cmd, envLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
    vk.vkCmdDraw(cmd, 3, 1, 0, 0);
    vk.vkCmdEndRendering(cmd);
  }
}

// The linear and prefiltered cubes from `src` (the assembled environment cube, SHADER_READ_ONLY), outside rendering.
bool envPrefilter(const Tex& src) {
  if (!initPrefilter()) return false;
  // The linear cube: the size of the source, its full chain.
  uint32_t linLevels = 1;
  for (uint32_t s = src.width; s > 1; s >>= 1) linLevels++;
  if (!envLinear.image || envLinear.width != src.width) {
    if (envLinear.image) releaseTexture(envLinear);
    envDropViews(envLinearFaces, 6);
    for (VkImageView& v : envLinearFaces) v = VK_NULL_HANDLE;
    if (!envCreateCube(src.width, linLevels, envLinear, envLinearFaces, false)) return false;
  }
  for (uint32_t f = 0; f < 6; f++)
    subresourceBarrier(envLinear.image, 0, linLevels, f, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  envRenderLevel(src.view, envLinearFaces, src.width, 0.0f, src.width, true);
  VkCommandBuffer cmd = slot().cmd;
  for (uint32_t f = 0; f < 6; f++) {
    subresourceBarrier(envLinear.image, 0, 1, f, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    if (linLevels > 1)
      subresourceBarrier(envLinear.image, 1, linLevels - 1, f, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    for (uint32_t l = 1; l < linLevels; l++) {
      const int32_t s0 = int32_t(std::max(src.width >> (l - 1), 1u)), s1 = int32_t(std::max(src.width >> l, 1u));
      VkImageBlit bl{};
      bl.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, l - 1, f, 1 };
      bl.srcOffsets[1] = { s0, s0, 1 };
      bl.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, l, f, 1 };
      bl.dstOffsets[1] = { s1, s1, 1 };
      vk.vkCmdBlitImage(cmd, envLinear.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, envLinear.image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
      subresourceBarrier(envLinear.image, l, 1, f, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    }
    subresourceBarrier(envLinear.image, 0, linLevels, f, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  }
  const uint32_t size = std::min(src.width, 128u);
  uint32_t levels = 1;
  for (uint32_t s = size; s > 4; s >>= 1) levels++;
  auto it = textures.find(Backend::kEnvironmentFiltered);
  if (it == textures.end() || it->second.width != size || it->second.levels != levels) {
    Tex t;
    envDropViews(envFaceViews.data(), envFaceViews.size());
    envFaceViews.assign(size_t(6) * levels, VK_NULL_HANDLE);
    if (!envCreateCube(size, levels, t, envFaceViews.data(), true)) return false;
    if (it != textures.end()) {
      releaseTexture(it->second);
      it->second = t;
    } else {
      it = textures.emplace(Backend::kEnvironmentFiltered, t).first;
    }
  }
  Tex& dst = it->second;
  for (uint32_t f = 0; f < 6; f++)
    subresourceBarrier(dst.image, 0, levels, f, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  for (uint32_t l = 0; l < levels; l++) {
    const float rough = levels > 1 ? float(l) / float(levels - 1) : 0.0f;
    envRenderLevel(envLinear.view, &envFaceViews[size_t(l) * 6], std::max(size >> l, 1u), rough * rough, src.width, false);
  }
  for (uint32_t f = 0; f < 6; f++)
    subresourceBarrier(dst.image, 0, levels, f, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  invalidateDrawState();
  envBlock[0][1] = float(levels - 1);
  return true;
}

void destroyEnvPath() {
  if (envSampler) vk.vkDestroySampler(device, envSampler, nullptr);
  envSampler = VK_NULL_HANDLE;
  for (VkImageView v : envFaceViews)
    if (v) vk.vkDestroyImageView(device, v, nullptr);
  envFaceViews.clear();
  for (VkImageView& v : envLinearFaces) {
    if (v) vk.vkDestroyImageView(device, v, nullptr);
    v = VK_NULL_HANDLE;
  }
  if (envLinear.view) vk.vkDestroyImageView(device, envLinear.view, nullptr);
  if (envLinear.image) vmaDestroyImage(allocator, envLinear.image, envLinear.alloc);
  envLinear = Tex{};
  if (envPipe) vk.vkDestroyPipeline(device, envPipe, nullptr);
  if (envPs) vk.vkDestroyShaderModule(device, envPs, nullptr);
  if (envLayout) vk.vkDestroyPipelineLayout(device, envLayout, nullptr);
  if (envPool) vk.vkDestroyDescriptorPool(device, envPool, nullptr);
  if (envSetLayout) vk.vkDestroyDescriptorSetLayout(device, envSetLayout, nullptr);
  envPipe = VK_NULL_HANDLE;
  envPs = VK_NULL_HANDLE;
  envLayout = VK_NULL_HANDLE;
  envPool = VK_NULL_HANDLE;
  envSetLayout = VK_NULL_HANDLE;
  if (envReadback.buffer) vmaDestroyBuffer(allocator, envReadback.buffer, envReadback.alloc);
  envReadback = EnvReadback{};
}

// No environment: the derived cubes are freed (they are rebuilt with the next one) and the block is not valid.
void envRelease() {
  auto it = textures.find(Backend::kEnvironmentFiltered);
  if (it != textures.end()) {
    releaseTexture(it->second);
    textures.erase(it);
  }
  envDropViews(envFaceViews.data(), envFaceViews.size());
  envFaceViews.clear();
  envDropViews(envLinearFaces, 6);
  for (VkImageView& v : envLinearFaces) v = VK_NULL_HANDLE;
  if (envLinear.image) releaseTexture(envLinear);
  envBlock[0][0] = 0.0f;
}

void doSetEnvironment(TexKey cube) {
  if (cube == envCube) return;
  envCube = cube;
  if (!cube) {
    envRelease();
    return;
  }
  auto c = cubes.find(cube);
  if (c != cubes.end()) {
    c->second.levels = 0;     // the full chain: rough reflections and the harmonics' level
    c->second.dirty = 0x3F;   // assembled (and so prefiltered) again, even if its faces did not change
  }
}

// Called by assembleCube after `t` (the cube `key`) is up to date, outside rendering.
void envAfterAssemble(TexKey key, const Tex& t) {
  if (key != envCube || t.format != VK_FORMAT_R16G16B16A16_SFLOAT) return;
  if (!envPrefilter(t) || envReadback.pending) return;
  const Tex& lin = envLinear;
  uint32_t level = 0;
  while (level + 1 < lin.levels && std::max(lin.width >> level, 1u) > kEnvReadbackSize) level++;
  const uint32_t size = std::max(lin.width >> level, 1u);
  const VkDeviceSize bytes = VkDeviceSize(6) * size * size * 8;
  if (envReadback.buffer && envReadback.size != size) {
    vmaDestroyBuffer(allocator, envReadback.buffer, envReadback.alloc);
    envReadback.buffer = VK_NULL_HANDLE;
  }
  if (!envReadback.buffer) {
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo bai{};
    bai.usage = VMA_MEMORY_USAGE_AUTO;
    bai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    if (!check(vmaCreateBuffer(allocator, &bci, &bai, &envReadback.buffer, &envReadback.alloc, &info),
               "vmaCreateBuffer(environment readback)"))
      return;
    envReadback.map = info.pMappedData;
  }
  for (uint32_t f = 0; f < 6; f++)
    subresourceBarrier(lin.image, level, 1, f, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
  VkBufferImageCopy r{};
  r.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 6 };
  r.imageExtent = { size, size, 1 };
  vk.vkCmdCopyImageToBuffer(slot().cmd, lin.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, envReadback.buffer, 1, &r);
  for (uint32_t f = 0; f < 6; f++)
    subresourceBarrier(lin.image, level, 1, f, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  envReadback.size = size;
  envReadback.lastLevel = lin.levels - 1;
  envReadback.frame = frameNumber;
  envReadback.pending = true;
}

static float envHalf(uint16_t h) {
  const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
  float v;
  if (e == 0) v = std::ldexp(float(m), -24);
  else if (e == 31) v = m ? 0.0f : 65504.0f;  // NaN as 0, infinity as the largest half
  else v = std::ldexp(float(m | 1024), int(e) - 25);
  return s ? -v : v;
}

// D3D9 cube face directions (texel centre u, v in -1..1, v down): +X -X +Y -Y +Z -Z.
static void envFaceDir(uint32_t face, float u, float v, float d[3]) {
  switch (face) {
    case 0: d[0] = 1; d[1] = -v; d[2] = -u; break;
    case 1: d[0] = -1; d[1] = -v; d[2] = u; break;
    case 2: d[0] = u; d[1] = 1; d[2] = v; break;
    case 3: d[0] = u; d[1] = -1; d[2] = -v; break;
    case 4: d[0] = u; d[1] = -v; d[2] = 1; break;
    default: d[0] = -u; d[1] = -v; d[2] = -1; break;
  }
}

static void envBasis(const float n[3], float y[9]) {
  const float x = n[0], yy = n[1], z = n[2];
  y[0] = 0.282095f;
  y[1] = 0.488603f * yy;
  y[2] = 0.488603f * z;
  y[3] = 0.488603f * x;
  y[4] = 1.092548f * x * yy;
  y[5] = 1.092548f * yy * z;
  y[6] = 0.315392f * (3.0f * z * z - 1.0f);
  y[7] = 1.092548f * x * z;
  y[8] = 0.546274f * (x * x - yy * yy);
}

// The projection itself (static, so tools/backend_test can check it against analytic environments). `texels`: 6 faces
// of size x size RGBA16F, D3D9 face order, linear light. Writes the block's harmonics ([1..9]) and returns the
// luminance of the mean irradiance (0: no light; the block is then not valid).
static float envProject(const uint16_t* texels, uint32_t size, float out[9][4]) {
  double L[9][3] = {};
  double total = 0.0;
  for (uint32_t f = 0; f < 6; f++)
    for (uint32_t j = 0; j < size; j++)
      for (uint32_t i = 0; i < size; i++) {
        const float u = 2.0f * (float(i) + 0.5f) / float(size) - 1.0f, v = 2.0f * (float(j) + 0.5f) / float(size) - 1.0f;
        float d[3];
        envFaceDir(f, u, v, d);
        const float len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        for (float& c : d) c /= len;
        const double w = 1.0 / (double(len) * len * len);  // solid angle of the texel, up to (2 / size)^2
        float y[9];
        envBasis(d, y);
        const uint16_t* t = texels + (size_t(f) * size * size + size_t(j) * size + i) * 4;
        float rgb[3];
        for (int c = 0; c < 3; c++) rgb[c] = std::max(envHalf(t[c]), 0.0f);
        for (int k = 0; k < 9; k++)
          for (int c = 0; c < 3; c++) L[k][c] += w * y[k] * rgb[c];
        total += w;
      }
  const double norm = 4.0 * 3.14159265358979 / total;  // the weights sum to the sphere's solid angle
  const double A[9] = { 3.14159265358979, 2.0943951, 2.0943951, 2.0943951, 0.785398163, 0.785398163, 0.785398163,
                        0.785398163, 0.785398163 };
  double E[9][3];
  for (int k = 0; k < 9; k++)
    for (int c = 0; c < 3; c++) E[k][c] = A[k] * L[k][c] * norm;
  // The mean of E over the sphere is its constant term's value: E00 * Y00.
  const double meanLum = (0.2126 * E[0][0] + 0.7152 * E[0][1] + 0.0722 * E[0][2]) * 0.282095;
  for (int k = 0; k < 9; k++) {
    for (int c = 0; c < 3; c++) out[k][c] = meanLum > 1e-8 ? float(E[k][c] / meanLum) : 0.0f;
    out[k][3] = 0.0f;
  }
  return float(meanLum);
}

// At frame start, after the fences: the harmonics of a finished copy.
void envProcessReadback() {
  if (!envReadback.pending || envReadback.frame > completedFrame) return;
  envReadback.pending = false;
  vmaInvalidateAllocation(allocator, envReadback.alloc, 0, VK_WHOLE_SIZE);
  float sh[9][4];
  const float mean = envProject(static_cast<const uint16_t*>(envReadback.map), envReadback.size, sh);
  envBlock[0][0] = mean > 1e-8f ? 1.0f : 0.0f;
  envBlock[0][2] = envBlock[0][3] = 0.0f;
  memcpy(envBlock[1], sh, sizeof(sh));
  envUpdates++;
}

// The cube's view for binding 50 (assembled now if it changed), or the dummy cube with the block not valid.
VkImageView envView(bool& valid) {
  valid = false;
  if (!envCube) return dummyView[1];
  auto c = cubes.find(envCube);
  if (c == cubes.end()) return dummyView[1];
  for (RtKey k : c->second.faces)  // a face being drawn into: not sampleable in this group
    if (k && (k == groupDepth || std::find(groupColors.begin(), groupColors.end(), k) != groupColors.end()))
      return dummyView[1];
  if (!assembleCube(envCube, c->second)) return dummyView[1];
  auto t = textures.find(Backend::kEnvironmentFiltered);
  if (t == textures.end()) return dummyView[1];
  valid = envBlock[0][0] != 0.0f;
  return t->second.view;
}
