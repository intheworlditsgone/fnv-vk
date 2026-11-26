// D3D9 semantics: backend/translate.h. Shader ABI: docs/SHADER_ABI.md.

// --- host-visible per-frame memory (staging for buffer writes, uniform blocks) ---------------------------------
struct HostChunk {
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation alloc = nullptr;
  uint8_t* ptr = nullptr;
  VkDeviceSize size = 0, used = 0;
};
struct HostRing {
  std::vector<HostChunk> chunks;
  size_t current = 0;
};
struct HostSlice {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
  uint8_t* ptr = nullptr;
};
static constexpr VkDeviceSize kHostChunkSize = 8u << 20;

bool hostAlloc(HostRing& ring, VkDeviceSize size, VkDeviceSize align, VkBufferUsageFlags usage, HostSlice& out) {
  for (;;) {
    if (ring.current < ring.chunks.size()) {
      HostChunk& c = ring.chunks[ring.current];
      VkDeviceSize off = (c.used + align - 1) / align * align;
      if (off + size <= c.size) {
        c.used = off + size;
        out = { c.buffer, off, c.ptr + off };
        return true;
      }
      ring.current++;
      continue;
    }
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = std::max(kHostChunkSize, size);
    bci.usage = usage;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    HostChunk c;
    VmaAllocationInfo info{};
    if (!check(vmaCreateBuffer(allocator, &bci, &aci, &c.buffer, &c.alloc, &info), "vmaCreateBuffer(host ring)"))
      return false;
    c.ptr = static_cast<uint8_t*>(info.pMappedData);
    c.size = bci.size;
    ring.chunks.push_back(c);
    hostRingBytes.fetch_add(c.size, std::memory_order_relaxed);
  }
}
void hostReset(HostRing& ring) {
  for (HostChunk& c : ring.chunks) c.used = 0;
  ring.current = 0;
}
void hostFlush(HostRing& ring) {
  for (size_t i = 0; i <= ring.current && i < ring.chunks.size(); i++)
    if (ring.chunks[i].used) vmaFlushAllocation(allocator, ring.chunks[i].alloc, 0, ring.chunks[i].used);
}
void hostDestroy(HostRing& ring) {
  for (HostChunk& c : ring.chunks) {
    vmaDestroyBuffer(allocator, c.buffer, c.alloc);
    hostRingBytes.fetch_sub(c.size, std::memory_order_relaxed);
  }
  ring.chunks.clear();
  ring.current = 0;
}

struct DrawSlot {
  HostRing staging, uniforms;
  std::vector<VkDescriptorPool> pools;
  size_t pool = 0;
  // The draw sets written in this frame, by their full content (the signature doDraw builds): a later draw with the same
  // content binds the same set instead of allocating and writing one. Cleared with the pools.
  struct CachedSet { VkDescriptorSet set; uint32_t stage, count; size_t at; };
  std::unordered_map<uint64_t, CachedSet> setCache;
  std::vector<uint64_t> sigStore;
} drawSlots[kFramesInFlight];
DrawSlot& drawSlot() { return drawSlots[frameNumber % kFramesInFlight]; }

// --- buffers ---------------------------------------------------------------------------------------------------
// Versions: a write to a buffer the GPU may still read (and that is not NOOVERWRITE) goes to another version, as D3D9
// renames on DISCARD. Retired versions are kept per buffer and reused once the GPU has finished with them, so dynamic
// geometry (particles, UI, NiVBDynamicSet) cycles through a few versions instead of creating and freeing a VkBuffer on
// nearly every write. A buffer renamed repeatedly gets host-visible, mapped versions from dynPool: its writes are
// copied by the CPU straight into the memory the GPU reads (no staging copy, no transfer). dynPool has small blocks and
// a hard block limit, because every mapped block takes address space in this 32-bit process; when it is full the
// buffer keeps device-local versions written through staging.
struct BufVer {
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation alloc = nullptr;
  uint8_t* ptr = nullptr;  // mapped (dynPool) version
};
struct Buf {
  uint32_t size = 0;
  bool index = false, index32 = false;
  BufVer cur;
  uint64_t lastUse = 0;  // last frame that recorded a draw reading `cur`
  uint32_t renames = 0;
  struct Spare {
    BufVer v;
    uint64_t retired;  // reusable once this frame has completed
  };
  std::vector<Spare> spares;
};
static constexpr uint32_t kMaxSpares = 3;              // retired versions kept per buffer
static constexpr uint32_t kMappedAfterRenames = 2;     // renames before a buffer gets mapped versions
static constexpr uint32_t kMaxMappedSize = 4u << 20;   // larger buffers stay device-local
static constexpr VkDeviceSize kDynBlockSize = 8u << 20;
static constexpr size_t kDynMaxBlocks = 16;            // at most 128 MiB of mapped geometry
std::unordered_map<BufKey, Buf> buffers;
std::unordered_set<VkBuffer> uploadTouched;  // buffers written/read by transfers since the last upload barrier
VmaPool dynPool = nullptr;

VkBufferUsageFlags geometryUsage(bool index) {
  return (index ? VK_BUFFER_USAGE_INDEX_BUFFER_BIT : VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
         VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
}
void createDynPool() {
  VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
  bci.size = 65536;
  bci.usage = geometryUsage(false) | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
  VmaAllocationCreateInfo aci{};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  uint32_t type = 0;
  if (vmaFindMemoryTypeIndexForBufferInfo(allocator, &bci, &aci, &type) != VK_SUCCESS) {
    logf("dynamic geometry: no host-visible memory type for vertex/index buffers; staging only");
    return;
  }
  VmaPoolCreateInfo pci{};
  pci.memoryTypeIndex = type;
  pci.blockSize = kDynBlockSize;
  pci.maxBlockCount = kDynMaxBlocks;
  if (!check(vmaCreatePool(allocator, &pci, &dynPool), "vmaCreatePool(dynamic geometry)")) {
    dynPool = nullptr;
    return;
  }
  const VkMemoryPropertyFlags f = memProps.memoryTypes[type].propertyFlags;
  logf("dynamic geometry: mapped versions in memory type %u (%s%s), %u MiB blocks, at most %zu", type,
       (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "device-local, " : "", (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? "coherent" : "non-coherent",
       unsigned(kDynBlockSize >> 20), kDynMaxBlocks);
}

bool newBufVer(const Buf& b, BufVer& v, bool mapped = false) {
  VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
  bci.size = std::max<uint32_t>(b.size, 4);
  bci.usage = geometryUsage(b.index);
  VmaAllocationCreateInfo aci{};
  if (mapped) {
    if (!dynPool) return false;
    aci.pool = dynPool;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(allocator, &bci, &aci, &v.buffer, &v.alloc, &info) != VK_SUCCESS) {  // pool full: expected
      st.mappedFallbacks++;
      v = BufVer{};
      return false;
    }
    v.ptr = static_cast<uint8_t*>(info.pMappedData);
    if (!v.ptr) {
      vmaDestroyBuffer(allocator, v.buffer, v.alloc);
      v = BufVer{};
      return false;
    }
    return true;
  }
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  return check(vmaCreateBuffer(allocator, &bci, &aci, &v.buffer, &v.alloc, nullptr), "vmaCreateBuffer(geometry)");
}
void releaseBufVer(BufVer& v) {
  if (v.buffer) addGarbage({ frameNumber, VK_NULL_HANDLE, VK_NULL_HANDLE, v.buffer, v.alloc });
  v = BufVer{};
}
void releaseBuf(Buf& b) {
  releaseBufVer(b.cur);
  for (Buf::Spare& s : b.spares) releaseBufVer(s.v);
  b.spares.clear();
}
// The version leaving service: kept for reuse (draws of this frame may still read it) or freed when enough are kept.
void retireBufVer(Buf& b, BufVer& v) {
  if (v.buffer && b.spares.size() < kMaxSpares) b.spares.push_back({ v, frameNumber });
  else releaseBufVer(v);
  v = BufVer{};
}
bool takeSpare(Buf& b, bool mapped, BufVer& out) {
  for (size_t i = 0; i < b.spares.size(); i++)
    if (b.spares[i].retired <= completedFrame && (b.spares[i].v.ptr != nullptr) == mapped) {
      out = b.spares[i].v;
      b.spares[i] = b.spares.back();
      b.spares.pop_back();
      return true;
    }
  return false;
}
// Orders transfers in the upload command buffer that touch the same buffer (WAW / RAW between copies).
void uploadHazard(VkCommandBuffer cb, std::initializer_list<VkBuffer> touched) {
  bool hazard = false;
  for (VkBuffer b : touched) hazard |= uploadTouched.count(b) != 0;
  if (hazard) {
    VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    mb.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    di.memoryBarrierCount = 1;
    di.pMemoryBarriers = &mb;
    vk.vkCmdPipelineBarrier2(cb, &di);
    uploadTouched.clear();
  }
  for (VkBuffer b : touched) uploadTouched.insert(b);
}

bool doCreateBuffer(BufKey key, uint32_t size, bool index, bool index32) {
  Buf b;
  b.size = size;
  b.index = index;
  b.index32 = index32;
  if (!newBufVer(b, b.cur)) return false;
  auto it = buffers.find(key);
  if (it != buffers.end()) {
    releaseBuf(it->second);
    it->second = b;
  } else {
    buffers.emplace(key, b);
  }
  st.buffersCreated++;
  return true;
}

bool doWriteBuffer(BufKey key, uint32_t offset, const void* data, uint32_t size, uint32_t flags) {
  auto it = buffers.find(key);
  if (it == buffers.end()) {
    st.unknownBuffers++;
    return false;
  }
  Buf& b = it->second;
  if (!size || offset >= b.size) return true;
  size = std::min(size, b.size - offset);
  if (!beginFrameIfNeeded()) return false;
  const bool inUse = b.lastUse > completedFrame;
  const bool whole = offset == 0 && size == b.size;
  if (inUse && !(flags & kLockNoOverwrite)) {
    // Rename: earlier recorded draws keep reading the old version; the write goes to another one.
    const bool keepOld = !(flags & kLockDiscard) && !whole;
    b.renames++;
    // Mapped only when the old contents are not needed: carrying them over is a GPU copy (mapped memory is often
    // write-combined, far too slow for the CPU to read back).
    bool mapped = b.renames >= kMappedAfterRenames && b.size <= kMaxMappedSize && !keepOld;
    BufVer nv;
    bool reused = takeSpare(b, mapped, nv);
    bool ok = reused || newBufVer(b, nv, mapped);
    if (!ok && mapped) {
      mapped = false;
      reused = takeSpare(b, false, nv);
      ok = reused || newBufVer(b, nv, false);
    }
    if (!ok) return false;
    if (keepOld) {
      VkCommandBuffer cb = uploadCmd();
      uploadHazard(cb, { b.cur.buffer, nv.buffer });
      VkBufferCopy c{ 0, 0, b.size };
      vk.vkCmdCopyBuffer(cb, b.cur.buffer, nv.buffer, 1, &c);
    }
    if (nv.ptr) {  // the buffer moved to mapped versions: device-local spares are no longer reused
      for (size_t i = 0; i < b.spares.size();)
        if (!b.spares[i].v.ptr) {
          releaseBufVer(b.spares[i].v);
          b.spares[i] = b.spares.back();
          b.spares.pop_back();
        } else {
          i++;
        }
    }
    retireBufVer(b, b.cur);
    b.cur = nv;
    b.lastUse = 0;
    st.bufferRenames++;
    if (reused) st.bufferRecycles++;
  }
  if (b.cur.ptr) {
    // Mapped version: the GPU is not reading the written range (renamed above, or NOOVERWRITE), and host writes made
    // before the frame's submit are visible to it.
    memcpy(b.cur.ptr + offset, data, size);
    vmaFlushAllocation(allocator, b.cur.alloc, offset, size);
    st.bufferWrites++;
    st.bufferDirectWrites++;
    st.bufferBytesWritten += size;
    return true;
  }
  VkCommandBuffer cb = uploadCmd();
  HostSlice s;
  if (!hostAlloc(drawSlot().staging, size, 16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, s)) return false;
  memcpy(s.ptr, data, size);
  uploadHazard(cb, { b.cur.buffer });
  VkBufferCopy c{ s.offset, offset, size };
  vk.vkCmdCopyBuffer(cb, s.buffer, b.cur.buffer, 1, &c);
  st.bufferWrites++;
  st.bufferBytesWritten += size;
  return true;
}

// --- shaders, declarations --------------------------------------------------------------------------------------
ShaderPack pack;
struct Shader {
  const PackedShader* packed = nullptr;
  VkShaderModule module = VK_NULL_HANDLE;
};
std::unordered_map<ShaderKey, Shader> shaders;
std::unordered_map<const PackedShader*, VkShaderModule> modules;  // one module per pack entry
std::unordered_map<DeclKey, std::vector<VertexElement>> decls;

VkShaderModule moduleFor(const PackedShader* p) {
  auto it = modules.find(p);
  if (it != modules.end()) return it->second;
  VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
  ci.codeSize = p->spvBytes;
  ci.pCode = p->spv;
  VkShaderModule mod = VK_NULL_HANDLE;
  check(vk.vkCreateShaderModule(device, &ci, nullptr, &mod), "vkCreateShaderModule");
  modules.emplace(p, mod);
  return mod;
}
// The depth pre-pass's module of a pixel shader (spvDepthVariant), VK_NULL_HANDLE when it has none.
std::unordered_map<const PackedShader*, VkShaderModule> depthModules;
VkShaderModule depthModuleFor(const PackedShader* p) {
  auto it = depthModules.find(p);
  if (it != depthModules.end()) return it->second;
  VkShaderModule mod = VK_NULL_HANDLE;
  std::vector<uint32_t> spv;
  if (p->pixel && spvDepthVariant(p->spv, p->spvBytes / 4, spv)) {
    VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    ci.codeSize = spv.size() * 4;
    ci.pCode = spv.data();
    if (!check(vk.vkCreateShaderModule(device, &ci, nullptr, &mod), "vkCreateShaderModule(depth)")) mod = VK_NULL_HANDLE;
  }
  depthModules.emplace(p, mod);
  return mod;
}
// The colour module of a pixel shader (spvColourVariant: no discard), VK_NULL_HANDLE when it has none.
std::unordered_map<const PackedShader*, VkShaderModule> colourModules;
VkShaderModule colourModuleFor(const PackedShader* p) {
  auto it = colourModules.find(p);
  if (it != colourModules.end()) return it->second;
  VkShaderModule mod = VK_NULL_HANDLE;
  std::vector<uint32_t> spv;
  if (p->pixel && spvColourVariant(p->spv, p->spvBytes / 4, spv)) {
    VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    ci.codeSize = spv.size() * 4;
    ci.pCode = spv.data();
    if (!check(vk.vkCreateShaderModule(device, &ci, nullptr, &mod), "vkCreateShaderModule(colour)")) mod = VK_NULL_HANDLE;
  }
  colourModules.emplace(p, mod);
  return mod;
}

// --- layouts and fixed objects ----------------------------------------------------------------------------------
// Set 0 = vertex stage, set 1 = pixel stage. Bindings 0..8 uniform blocks, 16+s images, 32+s samplers (SHADER_ABI).
VkDescriptorSetLayout setLayouts[2] = {};
VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
VkPipelineCache pipelineCache = VK_NULL_HANDLE;
VkImage dummyImage[2] = {};         // 1x1 zero: [0] 2D, [1] cube
VmaAllocation dummyAlloc[2] = {};
VkImageView dummyView[2] = {};
VkBuffer zeroBuffer = VK_NULL_HANDLE;  // 64 zero bytes: unbound vertex inputs and unused uniform blocks
VmaAllocation zeroAlloc = nullptr;
VkDeviceSize uboAlign = 256;
bool nonSeamlessCube = false, anisotropy = false, fragmentStores = false;
float maxAnisotropy = 1.0f;

static constexpr uint32_t kSets = 2, kUboBindings = 10, kTexBindings = 16;
// Sun shadows in the game's lighting (pixel set; docs/SHADER_ABI.md): the block, the cascades and their compare sampler.
static constexpr uint32_t kSunBlockBinding = 9, kSunImageBinding = 48, kSunSamplerBinding = 49;
static constexpr uint32_t kSunBlockVec4s = 21;
// Ambient occlusion in the lighting (pixel set; docs/SHADER_ABI.md "Ambient occlusion").
static constexpr uint32_t kAoBlockBinding = 11, kAoImageBinding = 52, kAoSamplerBinding = 53, kGiImageBinding = 54,
                          kGiDepthBinding = 55;
static constexpr uint32_t kWetBlockBinding = 12, kPuddleImageBinding = 56, kPuddleSamplerBinding = 57, kWetBlockVec4s = 16,
                          kPuddleLevelsBinding = 58;
static constexpr uint32_t kSkinDetailBinding = 59, kSkinDetailSamplerBinding = 60;
static constexpr uint32_t kSkinBlockBinding = 13, kSkinAlbedoBinding = 61, kSkinDepthBinding = 62;
static constexpr uint32_t kHairShiftBinding = 63, kHairShiftSamplerBinding = 64;
static constexpr uint32_t kZeroBinding = 15;  // vertex binding for inputs the declaration does not provide

bool initDrawPath() {
  VkPhysicalDeviceProperties props{};
  vk.vkGetPhysicalDeviceProperties(phys, &props);
  uboAlign = std::max<VkDeviceSize>(props.limits.minUniformBufferOffsetAlignment, 16);
  maxAnisotropy = props.limits.maxSamplerAnisotropy;

  for (uint32_t set = 0; set < kSets; set++) {
    std::vector<VkDescriptorSetLayoutBinding> b;
    VkShaderStageFlags stage = set == 0 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
    // Binding 0 (the float constants, a new block nearly every draw) is a dynamic uniform buffer: its offset is given
    // when the set is bound, so a change of constants rebinds the same set instead of writing a new one.
    for (uint32_t i = 0; i < kUboBindings; i++)
      b.push_back({ i, i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stage, nullptr });
    for (uint32_t i = 0; i < kTexBindings; i++) {
      b.push_back({ 16 + i, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, stage, nullptr });
      b.push_back({ 32 + i, VK_DESCRIPTOR_TYPE_SAMPLER, 1, stage, nullptr });
    }
    if (set == 1) {
      b.push_back({ kSunImageBinding, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, stage, nullptr });
      b.push_back({ kSunSamplerBinding, VK_DESCRIPTOR_TYPE_SAMPLER, 1, stage, nullptr });
      b.push_back({ kEnvBlockBinding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stage, nullptr });
      b.push_back({ kEnvImageBinding, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, stage, nullptr });
      b.push_back({ kEnvSamplerBinding, VK_DESCRIPTOR_TYPE_SAMPLER, 1, stage, nullptr });
      b.push_back({ kAoBlockBinding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stage, nullptr });
      b.push_back({ kAoImageBinding, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, stage, nullptr });
      b.push_back({ kAoSamplerBinding, VK_DESCRIPTOR_TYPE_SAMPLER, 1, stage, nullptr });
      b.push_back({ kGiImageBinding, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, stage, nullptr });
      b.push_back({ kGiDepthBinding, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, stage, nullptr });
      b.push_back({ kWetBlockBinding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stage, nullptr });
      b.push_back({ kPuddleImageBinding, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, stage, nullptr });
      b.push_back({ kPuddleSamplerBinding, VK_DESCRIPTOR_TYPE_SAMPLER, 1, stage, nullptr });
      b.push_back({ kPuddleLevelsBinding, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, stage, nullptr });
      b.push_back({ kSkinDetailBinding, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, stage, nullptr });
      b.push_back({ kSkinDetailSamplerBinding, VK_DESCRIPTOR_TYPE_SAMPLER, 1, stage, nullptr });
      b.push_back({ kSkinBlockBinding, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, stage, nullptr });
      b.push_back({ kSkinAlbedoBinding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, stage, nullptr });
      b.push_back({ kSkinDepthBinding, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, stage, nullptr });
      b.push_back({ kHairShiftBinding, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, stage, nullptr });
      b.push_back({ kHairShiftSamplerBinding, VK_DESCRIPTOR_TYPE_SAMPLER, 1, stage, nullptr });
    }
    VkDescriptorSetLayoutCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    ci.bindingCount = uint32_t(b.size());
    ci.pBindings = b.data();
    if (!check(vk.vkCreateDescriptorSetLayout(device, &ci, nullptr, &setLayouts[set]), "vkCreateDescriptorSetLayout"))
      return false;
  }
  if (!initEnvObjects()) return false;
  VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
  plci.setLayoutCount = kSets;
  plci.pSetLayouts = setLayouts;
  if (!check(vk.vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout), "vkCreatePipelineLayout")) return false;
  VkQueryPoolCreateInfo qci{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
  qci.queryType = VK_QUERY_TYPE_OCCLUSION;
  qci.queryCount = kQuerySlots;
  check(vk.vkCreateQueryPool(device, &qci, nullptr, &queryPool), "vkCreateQueryPool");
  VkPipelineCacheCreateInfo pcci{ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
  std::vector<uint8_t> cacheData;
  loadPipelineFile(props, cacheData);
  pcci.initialDataSize = cacheData.size();
  pcci.pInitialData = cacheData.empty() ? nullptr : cacheData.data();
  if (!check(vk.vkCreatePipelineCache(device, &pcci, nullptr, &pipelineCache), "vkCreatePipelineCache") && !cacheData.empty()) {
    pcci.initialDataSize = 0;  // the driver refused the data: start empty
    pcci.pInitialData = nullptr;
    check(vk.vkCreatePipelineCache(device, &pcci, nullptr, &pipelineCache), "vkCreatePipelineCache(empty)");
  }

  // Dummy images bound where a shader samples a register with no texture (the shader's isNull flag makes the result
  // D3D9's (0,0,0,1); the descriptor only has to be valid and of the declared type).
  for (int i = 0; i < 2; i++) {
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.flags = i ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_B8G8R8A8_UNORM;
    ici.extent = { 1, 1, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = i ? 6 : 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (!check(vmaCreateImage(allocator, &ici, &aci, &dummyImage[i], &dummyAlloc[i], nullptr), "vmaCreateImage(dummy)"))
      return false;
    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = dummyImage[i];
    vci.viewType = i ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
    vci.format = ici.format;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, ici.arrayLayers };
    if (!check(vk.vkCreateImageView(device, &vci, nullptr, &dummyView[i]), "vkCreateImageView(dummy)")) return false;
  }
  VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
  bci.size = 4096;  // also backs unused uniform blocks (the largest is 4 KiB)
  bci.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VmaAllocationCreateInfo aci{};
  aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  if (!check(vmaCreateBuffer(allocator, &bci, &aci, &zeroBuffer, &zeroAlloc, nullptr), "vmaCreateBuffer(zero)")) return false;

  // Initialise the dummies in the first frame's upload commands.
  if (!beginFrameIfNeeded()) return false;
  VkCommandBuffer cb = uploadCmd();
  vk.vkCmdFillBuffer(cb, zeroBuffer, 0, VK_WHOLE_SIZE, 0);
  for (int i = 0; i < 2; i++) {
    VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    b.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = dummyImage[i];
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, i ? 6u : 1u };
    VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vk.vkCmdPipelineBarrier2(cb, &di);
    VkClearColorValue zero{};
    vk.vkCmdClearColorImage(cb, dummyImage[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &b.subresourceRange);
    b.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vk.vkCmdPipelineBarrier2(cb, &di);
  }
  skinBufferUsable = skinBufferFormatsOk();
  if (!skinBufferUsable) logf("skin buffer unavailable (fragmentStoresAndAtomics or rgba8 / r32ui storage): no subsurface scattering");
  if (!ensureSkinBuffer(1, 1)) return false;
  recordSkinBufferClear(cb, VK_IMAGE_LAYOUT_UNDEFINED);
  if (!ensureShadowPlaceholder()) return false;

  if (!cfg.shaderPack.empty()) {
    std::string err;
    if (pack.load(cfg.shaderPack, &err)) logf("shader pack: %zu shaders from %s", pack.size(), cfg.shaderPack.c_str());
    else logf("shader pack not loaded: %s (draws disabled)", err.c_str());
    uint32_t replaced = 0;
    for (const ShaderReplacement& r : cfg.shaderReplacements) {
      Md5 md5;
      memcpy(md5.b.data(), r.md5, 16);
      std::vector<PackedShader::Def> defs;
      for (const ShaderReplacement::Constant& c : r.constants) defs.push_back({ c.reg, { c.v[0], c.v[1], c.v[2], c.v[3] } });
      if (pack.replace(md5, r.spv, defs, r.gameConstants)) replaced++;
    }
    if (!cfg.shaderReplacements.empty())
      logf("shader replacements: %u of %zu applied", replaced, cfg.shaderReplacements.size());
  }
  prebuildPipelines();
  createDynPool();
  return true;
}

void teardownDrawPath() {
  if (pipelineRecords.size() != savedRecords) savePipelineFile();
  for (auto& [k, b] : buffers) {
    if (b.cur.buffer) vmaDestroyBuffer(allocator, b.cur.buffer, b.cur.alloc);
    for (Buf::Spare& s : b.spares) vmaDestroyBuffer(allocator, s.v.buffer, s.v.alloc);
  }
  buffers.clear();
  for (auto& [p, mod] : modules) vk.vkDestroyShaderModule(device, mod, nullptr);
  modules.clear();
  for (auto* variants : { &depthModules, &colourModules }) {
    for (auto& [p, mod] : *variants)
      if (mod) vk.vkDestroyShaderModule(device, mod, nullptr);
    variants->clear();
  }
  shaders.clear();
  for (auto& [k, p] : pipelines) vk.vkDestroyPipeline(device, p, nullptr);
  pipelines.clear();
  pipelineKeys.clear();
  depthPipes.clear();
  depthAlphaPipes.clear();
  colourPipes.clear();
  for (auto& [k, s] : samplerCache) vk.vkDestroySampler(device, s, nullptr);
  samplerCache.clear();
  for (DrawSlot& d : drawSlots) {
    hostDestroy(d.staging);
    hostDestroy(d.uniforms);
    for (VkDescriptorPool p : d.pools) vk.vkDestroyDescriptorPool(device, p, nullptr);
    d.pools.clear();
  }
  for (int i = 0; i < 2; i++) {
    if (dummyView[i]) vk.vkDestroyImageView(device, dummyView[i], nullptr);
    if (dummyImage[i]) vmaDestroyImage(allocator, dummyImage[i], dummyAlloc[i]);
    dummyView[i] = VK_NULL_HANDLE;
    dummyImage[i] = VK_NULL_HANDLE;
  }
  if (zeroBuffer) vmaDestroyBuffer(allocator, zeroBuffer, zeroAlloc);
  zeroBuffer = VK_NULL_HANDLE;
  if (pipelineCache) vk.vkDestroyPipelineCache(device, pipelineCache, nullptr);
  destroyGammaPath();
  destroyHdrPath();
  destroyShadowPath();
  destroyEnvPath();
  destroyQuadPath();
  if (queryPool) vk.vkDestroyQueryPool(device, queryPool, nullptr);
  queryPool = VK_NULL_HANDLE;
  if (pipelineLayout) vk.vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
  for (auto& l : setLayouts)
    if (l) vk.vkDestroyDescriptorSetLayout(device, l, nullptr);
  pipelineCache = VK_NULL_HANDLE;
  pipelineLayout = VK_NULL_HANDLE;
  setLayouts[0] = setLayouts[1] = VK_NULL_HANDLE;
}

// Called when a frame slot is reused (its previous frame has completed).
void beginDrawSlot() {
  DrawSlot& d = drawSlot();
  hostReset(d.staging);
  hostReset(d.uniforms);
  for (VkDescriptorPool p : d.pools) vk.vkResetDescriptorPool(device, p, 0);
  d.pool = 0;
  d.setCache.clear();
  d.sigStore.clear();
  uploadTouched.clear();
  lastVsBlock = lastPsBlock = UniformCache{};
  for (SmallCache& c : smallCache) c.valid = false;  // their ring slices were in this slot's reset ring
  smallGen++;
  resetFxPools();
  resetShadowPools();
  for (int i = 0; i < 2; i++) {
    lastSet[i] = VK_NULL_HANDLE;  // allocated from this slot's reset pools
    lastSigCount[i] = 0;
  }
  invalidateDrawState();
}

// A pixel set (setLayouts[1]) for the backend's own passes that reuse the draw layout (screen quads, gamma, HDR): its
// dynamic binding 0 points at the zero buffer, bound with offset 0 (kZeroOffset).
static constexpr uint32_t kZeroOffset = 0;
VkDescriptorSet allocPassSet() {
  VkDescriptorSet set = allocSet(1);
  if (!set) return VK_NULL_HANDLE;
  const VkDescriptorBufferInfo zero{ zeroBuffer, 0, 64 };
  VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
  w.dstSet = set;
  w.dstBinding = 0;
  w.descriptorCount = 1;
  w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  w.pBufferInfo = &zero;
  vk.vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
  return set;
}
VkDescriptorSet allocSet(uint32_t set) {
  DrawSlot& d = drawSlot();
  for (;;) {
    if (d.pool == d.pools.size()) {
      constexpr uint32_t kSetsPerPool = 2048;
      VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kSetsPerPool * (kUboBindings + 4) },  // + environment, AO, wet, skin blocks
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kSetsPerPool * 2 },                     // the skin buffer
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kSetsPerPool * (kTexBindings + 9) },  // + sun cascades, environment, AO, GI, GI depth, puddles x2, skin detail, hair shift
        { VK_DESCRIPTOR_TYPE_SAMPLER, kSetsPerPool * (kTexBindings + 6) },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, kSetsPerPool },
      };
      VkDescriptorPoolCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
      ci.maxSets = kSetsPerPool;
      ci.poolSizeCount = 5;
      ci.pPoolSizes = sizes;
      VkDescriptorPool p = VK_NULL_HANDLE;
      if (!check(vk.vkCreateDescriptorPool(device, &ci, nullptr, &p), "vkCreateDescriptorPool")) return VK_NULL_HANDLE;
      d.pools.push_back(p);
    }
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = d.pools[d.pool];
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &setLayouts[set];
    VkDescriptorSet s = VK_NULL_HANDLE;
    VkResult r = vk.vkAllocateDescriptorSets(device, &ai, &s);
    if (r == VK_SUCCESS) return s;
    if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL) {
      check(r, "vkAllocateDescriptorSets");
      return VK_NULL_HANDLE;
    }
    d.pool++;
  }
}

// --- samplers ---------------------------------------------------------------------------------------------------
struct SamplerKey {
  uint32_t minF, magF, mipF, u, v, w, aniso, lodBias, maxMip, border, cube;
  bool operator==(const SamplerKey& o) const { return memcmp(this, &o, sizeof(*this)) == 0; }
};
struct BytesHash {
  template <typename T>
  size_t operator()(const T& k) const {
    static_assert(sizeof(T) % 4 == 0, "keys are whole 32-bit words");
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&k);
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < sizeof(T); i += 4) {
      uint32_t w;
      memcpy(&w, p + i, 4);
      h = (h ^ w) * 1099511628211ull;
    }
    return size_t(h ^ (h >> 32));
  }
};
std::unordered_map<SamplerKey, VkSampler, BytesHash> samplerCache;

VkSampler samplerFor(const uint32_t ss[14], bool cube) {
  SamplerKey k{};
  k.minF = ss[D3DSAMP_MINFILTER];
  k.magF = ss[D3DSAMP_MAGFILTER];
  k.mipF = ss[D3DSAMP_MIPFILTER];
  k.u = cube ? D3DTADDRESS_CLAMP : ss[D3DSAMP_ADDRESSU];
  k.v = cube ? D3DTADDRESS_CLAMP : ss[D3DSAMP_ADDRESSV];
  k.w = cube ? D3DTADDRESS_CLAMP : ss[D3DSAMP_ADDRESSW];
  k.aniso = (k.mipF && k.minF == D3DTEXF_ANISOTROPIC) ? ss[D3DSAMP_MAXANISOTROPY] : 0;
  k.lodBias = k.mipF ? ss[D3DSAMP_MIPMAPLODBIAS] : 0;
  k.maxMip = ss[D3DSAMP_MAXMIPLEVEL];
  k.border = (k.u == D3DTADDRESS_BORDER || k.v == D3DTADDRESS_BORDER || k.w == D3DTADDRESS_BORDER) ? ss[D3DSAMP_BORDERCOLOR] : 0;
  k.cube = cube;
  auto it = samplerCache.find(k);
  if (it != samplerCache.end()) return it->second;
  VkSamplerCreateInfo ci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
  VkSamplerCustomBorderColorCreateInfoEXT border{ VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT };
  ci.magFilter = xlat::filter(k.magF);
  ci.minFilter = xlat::filter(k.minF);
  ci.mipmapMode = xlat::mipFilter(k.mipF);
  ci.addressModeU = xlat::addressMode(k.u);
  ci.addressModeV = xlat::addressMode(k.v);
  ci.addressModeW = xlat::addressMode(k.w);
  ci.mipLodBias = xlat::asFloat(k.lodBias);
  ci.anisotropyEnable = anisotropy && k.aniso > 1;
  ci.maxAnisotropy = std::min(float(std::max(k.aniso, 1u)), maxAnisotropy);
  // D3DSAMP_MAXMIPLEVEL is the most detailed level used; MIPFILTER NONE samples only that level.
  ci.minLod = float(k.maxMip);
  ci.maxLod = k.mipF ? 16.0f : float(k.maxMip);
  ci.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  if (k.border) {
    if (customBorder) {
      xlat::d3dColor(k.border, border.customBorderColor.float32);
      border.format = VK_FORMAT_UNDEFINED;
      ci.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
      ci.pNext = &border;
    } else {
      logOnce("border" + std::to_string(k.border), "sampler border colour %08X needs VK_EXT_custom_border_color", k.border);
    }
  }
  if (cube && nonSeamlessCube) ci.flags |= VK_SAMPLER_CREATE_NON_SEAMLESS_CUBE_MAP_BIT_EXT;
  VkSampler s = VK_NULL_HANDLE;
  check(vk.vkCreateSampler(device, &ci, nullptr, &s), "vkCreateSampler");
  samplerCache.emplace(k, s);
  st.samplers++;
  return s;
}
bool customBorder = false;

// --- pipelines --------------------------------------------------------------------------------------------------
struct PipelineKey {
  VkShaderModule vs, ps;
  uint32_t topology;
  uint32_t attrCount;
  struct Attr { uint32_t location, binding, format, offset; } attrs[16];
  uint32_t strides[16];
  uint32_t bindingMask;
  uint32_t blendEnable, srcC, dstC, opC, srcA, dstA, opA;
  uint32_t writeMask[4];
  uint32_t colorFormats[4], colorCount, depthFormat, stencilFormat, samples;
  uint32_t fillMode;
  bool operator==(const PipelineKey& o) const { return memcmp(this, &o, sizeof(*this)) == 0; }
};
std::unordered_map<PipelineKey, VkPipeline, BytesHash> pipelines;
std::unordered_map<VkPipeline, const PipelineKey*> pipelineKeys;  // (the keys in `pipelines`: stable)

VkPipeline pipelineFor(const PipelineKey& k, const PackedShader* vsh = nullptr, const PackedShader* psh = nullptr) {
  auto it = pipelines.find(k);
  if (it != pipelines.end()) return it->second;
  VkPipelineShaderStageCreateInfo stages[2] = { { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
                                                { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO } };
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = k.vs;
  stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = k.ps;
  stages[1].pName = "main";
  VkVertexInputBindingDescription binds[16];
  uint32_t nb = 0;
  for (uint32_t b = 0; b < 16; b++)
    if (k.bindingMask & (1u << b)) binds[nb++] = { b, k.strides[b], VK_VERTEX_INPUT_RATE_VERTEX };
  VkVertexInputAttributeDescription attrs[16];
  for (uint32_t i = 0; i < k.attrCount; i++)
    attrs[i] = { k.attrs[i].location, k.attrs[i].binding, VkFormat(k.attrs[i].format), k.attrs[i].offset };
  VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
  vi.vertexBindingDescriptionCount = nb;
  vi.pVertexBindingDescriptions = binds;
  vi.vertexAttributeDescriptionCount = k.attrCount;
  vi.pVertexAttributeDescriptions = attrs;
  VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
  ia.topology = VkPrimitiveTopology(k.topology);
  VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
  vp.viewportCount = 1;
  vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
  rs.polygonMode = k.fillMode == D3DFILL_WIREFRAME ? VK_POLYGON_MODE_LINE
                   : k.fillMode == D3DFILL_POINT   ? VK_POLYGON_MODE_POINT
                                                   : VK_POLYGON_MODE_FILL;
  rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
  ms.rasterizationSamples = sampleBits(k.samples);
  VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
  VkPipelineColorBlendAttachmentState cba[4] = {};
  for (uint32_t i = 0; i < k.colorCount; i++) {
    cba[i].blendEnable = k.blendEnable;
    cba[i].srcColorBlendFactor = VkBlendFactor(k.srcC);
    cba[i].dstColorBlendFactor = VkBlendFactor(k.dstC);
    cba[i].colorBlendOp = VkBlendOp(k.opC);
    cba[i].srcAlphaBlendFactor = VkBlendFactor(k.srcA);
    cba[i].dstAlphaBlendFactor = VkBlendFactor(k.dstA);
    cba[i].alphaBlendOp = VkBlendOp(k.opA);
    cba[i].colorWriteMask = k.writeMask[i];
  }
  VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
  cb.attachmentCount = k.colorCount;
  cb.pAttachments = cba;
  VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_DEPTH_BIAS,
                           VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK,
                           VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
                           VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE,
                           VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
                           VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP,
                           VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE };
  VkPipelineDynamicStateCreateInfo dy{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
  dy.dynamicStateCount = uint32_t(std::size(dyn));
  dy.pDynamicStates = dyn;
  VkFormat colorFormats[4];
  for (uint32_t i = 0; i < k.colorCount; i++) colorFormats[i] = VkFormat(k.colorFormats[i]);
  VkPipelineRenderingCreateInfo rci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
  rci.colorAttachmentCount = k.colorCount;
  rci.pColorAttachmentFormats = colorFormats;
  rci.depthAttachmentFormat = VkFormat(k.depthFormat);
  rci.stencilAttachmentFormat = VkFormat(k.stencilFormat);
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
  pci.layout = pipelineLayout;
  if (!k.ps) pci.stageCount = 1;  // depth-only (the depth pre-pass): no fragment stage
  VkPipeline p = VK_NULL_HANDLE;
  if (st.pipelines < 8) {
    auto hex = [](const PackedShader* p) {
      std::string h;
      char b[3];
      for (int i = 0; p && i < 16; i++) { snprintf(b, 3, "%02x", p->md5[i]); h += b; }
      return h;
    };
    std::string attrList;
    for (uint32_t i = 0; i < k.attrCount; i++)
      attrList += " " + std::to_string(k.attrs[i].location) + ":" + std::to_string(k.attrs[i].binding) + "/" +
                  std::to_string(k.attrs[i].format) + "@" + std::to_string(k.attrs[i].offset);
    logf("fp environment at entry: x87 %04X mxcsr %08X", g_entryX87, g_entryMxcsr);
    logf("creating pipeline %llu (frame %llu): vs %s ps %s topo %u colors %u fmt0 %u depth %u samples %u blend %u "
         "strides0 %u attrs%s", st.pipelines + 1, frameNumber, hex(vsh).c_str(), hex(psh).c_str(), k.topology,
         k.colorCount, k.colorFormats[0], k.depthFormat, k.samples, k.blendEnable, k.strides[0], attrList.c_str());
  }
  check(vk.vkCreateGraphicsPipelines(device, pipelineCache, 1, &pci, nullptr, &p), "vkCreateGraphicsPipelines");
  if (st.pipelines < 8) logf("created pipeline %llu", st.pipelines + 1);
  auto added = pipelines.emplace(k, p).first;
  if (p) pipelineKeys.emplace(p, &added->first);
  st.pipelines++;
  if (p && vsh && (psh || !k.ps) && !prebuilding) {
    PipelineRecord r;
    memcpy(r.vs, vsh->md5, 16);
    if (psh) memcpy(r.ps, psh->md5, 16);
    else memset(r.ps, 0, 16);  // depth-only
    r.key = k;
    r.key.vs = r.key.ps = VK_NULL_HANDLE;  // modules are per run: the shaders are named by their pack MD5
    pipelineRecords.push_back(r);
  }
  return p;
}

// --- persistent pipelines ---------------------------------------------------------------------------------------
// One file: header, the recorded pipeline keys (shaders by pack MD5), then vkGetPipelineCacheData's blob. The blob is
// passed to the driver only for the same GPU, driver cache UUID and key layout; the keys are rebuilt at start (with the
// driver's data each creation is a cache hit), so pipelines the game used before never compile during play.
struct PipelineRecord {
  uint8_t vs[16], ps[16];
  PipelineKey key;
};
struct PipelineFileHeader {
  uint32_t magic, version, keySize, keyCount, cacheSize, vendorID, deviceID, driverVersion;
  uint8_t uuid[VK_UUID_SIZE];
};
static constexpr uint32_t kPipelineFileMagic = 0x43505646;  // "FVPC"
static constexpr uint32_t kPipelineFileVersion = 1;
std::vector<PipelineRecord> pipelineRecords;
size_t savedRecords = 0;
uint64_t pipelinesSavedAt = 0;
bool prebuilding = false;
VkPhysicalDeviceProperties cacheProps{};

void loadPipelineFile(const VkPhysicalDeviceProperties& props, std::vector<uint8_t>& cacheData) {
  cacheProps = props;
  if (cfg.pipelineFile.empty()) return;
  FILE* f = nullptr;
  if (fopen_s(&f, cfg.pipelineFile.c_str(), "rb") || !f) return;
  PipelineFileHeader h{};
  bool ok = fread(&h, sizeof(h), 1, f) == 1 && h.magic == kPipelineFileMagic && h.version == kPipelineFileVersion &&
            h.keySize == sizeof(PipelineKey) && h.keyCount < 1000000;
  if (ok) {
    pipelineRecords.resize(h.keyCount);
    ok = h.keyCount == 0 || fread(pipelineRecords.data(), sizeof(PipelineRecord), h.keyCount, f) == h.keyCount;
  }
  const bool sameDevice = ok && h.vendorID == props.vendorID && h.deviceID == props.deviceID &&
                          h.driverVersion == props.driverVersion && memcmp(h.uuid, props.pipelineCacheUUID, VK_UUID_SIZE) == 0;
  if (ok && sameDevice && h.cacheSize && h.cacheSize < (256u << 20)) {
    cacheData.resize(h.cacheSize);
    if (fread(cacheData.data(), 1, h.cacheSize, f) != h.cacheSize) cacheData.clear();
  }
  fclose(f);
  if (!ok) {
    pipelineRecords.clear();
    logf("pipeline file %s: unreadable or from another build; starting empty", cfg.pipelineFile.c_str());
    return;
  }
  savedRecords = pipelineRecords.size();
  logf("pipeline file: %u recorded pipelines, driver cache %zu bytes%s", h.keyCount, cacheData.size(),
       sameDevice ? "" : " (other GPU or driver: cache data not used)");
}

void prebuildPipelines() {
  if (pipelineRecords.empty() || !pack.size()) return;
  const uint64_t t0 = GetTickCount64();
  prebuilding = true;
  size_t built = 0, missing = 0;
  for (const PipelineRecord& r : pipelineRecords) {
    Md5 vm, pm;
    memcpy(vm.b.data(), r.vs, 16);
    memcpy(pm.b.data(), r.ps, 16);
    const PackedShader* vsh = pack.find(vm);
    const bool depthOnly = std::all_of(std::begin(r.ps), std::end(r.ps), [](uint8_t b) { return b == 0; });
    const PackedShader* psh = depthOnly ? nullptr : pack.find(pm);
    if (!vsh || (!psh && !depthOnly)) {
      missing++;
      continue;
    }
    PipelineKey k = r.key;
    k.vs = moduleFor(vsh);
    k.ps = psh ? moduleFor(psh) : VK_NULL_HANDLE;
    if (pipelineFor(k, vsh, psh)) built++;
  }
  prebuilding = false;
  logf("pipelines: %zu of %zu recorded pipelines created before the first frame in %llu ms (%zu with shaders not in the pack)",
       built, pipelineRecords.size(), GetTickCount64() - t0, missing);
}

void savePipelineFile() {
  if (cfg.pipelineFile.empty() || !pipelineCache) return;
  size_t size = 0;
  std::vector<uint8_t> data;
  if (vk.vkGetPipelineCacheData(device, pipelineCache, &size, nullptr) == VK_SUCCESS && size) {
    data.resize(size);
    if (vk.vkGetPipelineCacheData(device, pipelineCache, &size, data.data()) != VK_SUCCESS) size = 0;
    data.resize(size);
  }
  PipelineFileHeader h{ kPipelineFileMagic, kPipelineFileVersion, uint32_t(sizeof(PipelineKey)),
                        uint32_t(pipelineRecords.size()), uint32_t(data.size()), cacheProps.vendorID, cacheProps.deviceID,
                        cacheProps.driverVersion, {} };
  memcpy(h.uuid, cacheProps.pipelineCacheUUID, VK_UUID_SIZE);
  const std::string tmp = cfg.pipelineFile + ".tmp";
  FILE* f = nullptr;
  if (fopen_s(&f, tmp.c_str(), "wb") || !f) return;
  bool ok = fwrite(&h, sizeof(h), 1, f) == 1;
  if (ok && !pipelineRecords.empty())
    ok = fwrite(pipelineRecords.data(), sizeof(PipelineRecord), pipelineRecords.size(), f) == pipelineRecords.size();
  if (ok && !data.empty()) ok = fwrite(data.data(), 1, data.size(), f) == data.size();
  ok = fclose(f) == 0 && ok;
  // Written beside and swapped in, so a process killed mid-write (NVTF's fast exit) never leaves a torn file.
  if (ok && MoveFileExA(tmp.c_str(), cfg.pipelineFile.c_str(), MOVEFILE_REPLACE_EXISTING)) savedRecords = pipelineRecords.size();
  else DeleteFileA(tmp.c_str());
}

// After a present: new pipelines are written at most every 10 s (no shutdown hook runs under a fast exit).
void maybeSavePipelines() {
  if (pipelineRecords.size() == savedRecords) return;
  const uint64_t now = GetTickCount64();
  if (now - pipelinesSavedAt < 10000) return;
  pipelinesSavedAt = now;
  savePipelineFile();
}

// --- uniform blocks ---------------------------------------------------------------------------------------------
struct UniformCache {
  uint64_t version = 0;
  const PackedShader* shader = nullptr;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceSize offset = 0;
  bool valid = false;
};
UniformCache lastVsBlock, lastPsBlock;
const bool poisonConstants = getenv("FNVVK_POISON_CONSTANTS") != nullptr;

// The float-constant block: the game's register file with the shader's `def` constants overlaid.
bool constBlock(UniformCache& cache, const float* regs, uint32_t count, uint64_t version, const PackedShader* sh,
                VkDescriptorBufferInfo& out) {
  if (!(cache.valid && cache.version == version && cache.shader == sh)) {
    HostSlice s;
    if (!hostAlloc(drawSlot().uniforms, count * 16, uboAlign, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, s)) return false;
    if (sh->constAll) {
      if (regs) memcpy(s.ptr, regs, count * 16);
      else memset(s.ptr, 0, count * 16);
    } else {
      // Only the registers the shader reads (PackedShader::constRuns); the rest of the block is never read.
      // FNVVK_POISON_CONSTANTS (tests): the rest is NaN, so a register missing from the runs shows in the image.
      if (poisonConstants) {
        const uint32_t nan = 0x7FC00000u;
        for (uint32_t k = 0; k < count * 4; k++) memcpy(s.ptr + k * 4, &nan, 4);
      }
      for (const ConstRun& r : sh->constRuns) {
        const uint32_t n = std::min<uint32_t>(r.count, count > r.first ? count - r.first : 0);
        if (regs) memcpy(s.ptr + r.first * 16, regs + r.first * 4, n * 16);
        else memset(s.ptr + r.first * 16, 0, n * 16);
      }
    }
    for (const PackedShader::Def& d : sh->defs)
      if (d.reg < count) memcpy(s.ptr + d.reg * 16, d.v, 16);
    cache = { version, sh, s.buffer, s.offset, true };
  }
  out = { cache.buffer, cache.offset, VkDeviceSize(count) * 16 };
  return true;
}
bool smallBlock(const void* data, uint32_t size, VkDescriptorBufferInfo& out) {
  HostSlice s;
  if (!hostAlloc(drawSlot().uniforms, size, uboAlign, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, s)) return false;
  memcpy(s.ptr, data, size);
  out = { s.buffer, s.offset, size };
  return true;
}
// The same block, reused while its contents are unchanged (alpha test, fog, clip planes, point size and sampler flags
// rarely change between consecutive draws).
struct SmallCache {
  bool valid = false;
  uint32_t size = 0;
  uint8_t data[512];
  VkDescriptorBufferInfo info{};
} smallCache[10];
bool smallBlockCached(uint32_t id, const void* data, uint32_t size, VkDescriptorBufferInfo& out) {
  SmallCache& c = smallCache[id];
  if (c.valid && c.size == size && memcmp(c.data, data, size) == 0) {
    out = c.info;
    return true;
  }
  if (size > sizeof(c.data) || !smallBlock(data, size, out)) return false;
  c.valid = true;
  c.size = size;
  memcpy(c.data, data, size);
  c.info = out;
  return true;
}

// --- occlusion queries --------------------------------------------------------------------------------------------
static constexpr uint32_t kQuerySlots = 4096;
VkQueryPool queryPool = VK_NULL_HANDLE;
struct Query {
  int32_t slot = -1;          // pool slot of the last issue; -1 = never issued or not recordable
  uint64_t frame = 0;          // frame that recorded it
  bool ended = false;
};
std::unordered_map<uint32_t, Query> queries;
uint32_t nextQueryId = 1, nextSlot = 0, activeQuery = 0;
uint64_t slotFrame[kQuerySlots] = {};
bool hostQueryReset = false;

void endActiveQuery() {
  if (!activeQuery) return;
  auto it = queries.find(activeQuery);
  if (it != queries.end() && it->second.slot >= 0) vk.vkCmdEndQuery(slot().cmd, queryPool, uint32_t(it->second.slot));
  if (it != queries.end()) it->second.ended = true;
  activeQuery = 0;
}
void doBeginQuery(uint32_t id) {
  auto it = queries.find(id);
  if (it == queries.end() || !queryPool) return;
  Query& q = it->second;
  endActiveQuery();
  q = Query{};
  if (!rendering || renderingSuspended || !hostQueryReset) {
    st.queriesUnrecorded++;  // nothing to count into: reported as 0 samples
    return;
  }
  const uint32_t s = nextSlot++ % kQuerySlots;
  if (slotFrame[s] > completedFrame) {
    st.queriesUnrecorded++;  // ring exhausted within the frames in flight
    return;
  }
  vk.vkResetQueryPool(device, queryPool, s, 1);
  vk.vkCmdBeginQuery(slot().cmd, queryPool, s, VK_QUERY_CONTROL_PRECISE_BIT);
  slotFrame[s] = frameNumber;
  q.slot = int32_t(s);
  q.frame = frameNumber;
  activeQuery = id;
  st.queriesIssued++;
}
void doEndQuery(uint32_t id) {
  if (activeQuery == id) endActiveQuery();
}
int doQueryResult(uint32_t id, uint32_t& samples) {
  auto it = queries.find(id);
  if (it == queries.end()) return 0;
  const Query& q = it->second;
  if (q.slot < 0) {
    samples = 0;
    return 1;
  }
  if (q.frame >= frameNumber) return 0;  // not submitted yet
  uint32_t v = 0;
  VkResult r = vk.vkGetQueryPoolResults(device, queryPool, uint32_t(q.slot), 1, sizeof(v), &v, sizeof(v), 0);
  if (r == VK_NOT_READY) return 0;
  if (!check(r, "vkGetQueryPoolResults")) return 0;
  samples = v;
  return 1;
}

// --- built-in screen-space quads (Bink) ---------------------------------------------------------------------------
VkPipelineLayout quadLayout = VK_NULL_HANDLE;
VkShaderModule quadVs = VK_NULL_HANDLE, quadPs = VK_NULL_HANDLE;
std::unordered_map<uint64_t, VkPipeline> quadPipelines;  // (format, samples)

VkPipeline quadPipelineFor(VkFormat fmt, uint32_t samples) {
  const uint64_t key = (uint64_t(fmt) << 8) | samples;
  auto it = quadPipelines.find(key);
  if (it != quadPipelines.end()) return it->second;
  if (!quadLayout) {
    VkPushConstantRange pr{ VK_SHADER_STAGE_VERTEX_BIT, 0, 16 };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = kSets;
    plci.pSetLayouts = setLayouts;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pr;
    if (!check(vk.vkCreatePipelineLayout(device, &plci, nullptr, &quadLayout), "vkCreatePipelineLayout(quad)")) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = sizeof(builtin::kQuadVert);
    smci.pCode = builtin::kQuadVert;
    check(vk.vkCreateShaderModule(device, &smci, nullptr, &quadVs), "vkCreateShaderModule(quad vs)");
    smci.codeSize = sizeof(builtin::kQuadFrag);
    smci.pCode = builtin::kQuadFrag;
    check(vk.vkCreateShaderModule(device, &smci, nullptr, &quadPs), "vkCreateShaderModule(quad ps)");
  }
  VkPipelineShaderStageCreateInfo stages[2] = { { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
                                                { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO } };
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = quadVs;
  stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = quadPs;
  stages[1].pName = "main";
  VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
  VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  VkPipelineViewportStateCreateInfo vp{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
  vp.viewportCount = vp.scissorCount = 1;
  VkPipelineRasterizationStateCreateInfo rs{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
  rs.polygonMode = VK_POLYGON_MODE_FILL;
  rs.cullMode = VK_CULL_MODE_NONE;
  rs.lineWidth = 1.0f;
  VkPipelineMultisampleStateCreateInfo ms{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
  ms.rasterizationSamples = VkSampleCountFlagBits(samples);
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
  VkPipelineRenderingCreateInfo rci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
  rci.colorAttachmentCount = 1;
  rci.pColorAttachmentFormats = &fmt;
  if (groupDepth) {
    // The pipeline must match the open group's attachments.
    const Rt& d = *rtFor(groupDepth, groupLow);
    rci.depthAttachmentFormat = hasDepth(d.format) ? d.format : VK_FORMAT_UNDEFINED;
    rci.stencilAttachmentFormat = hasStencil(d.format) ? d.format : VK_FORMAT_UNDEFINED;
  }
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
  pci.layout = quadLayout;
  VkPipeline p = VK_NULL_HANDLE;
  check(vk.vkCreateGraphicsPipelines(device, pipelineCache, 1, &pci, nullptr, &p), "vkCreateGraphicsPipelines(quad)");
  quadPipelines.emplace(key | (uint64_t(groupDepthFormat) << 40), p);
  return p;
}

bool doDrawScreenQuads(const Backend::ScreenQuad* q, uint32_t n, const int32_t clip[4]) {
  if (!rendering || groupColors.size() != 1) return false;
  const Rt& rt = *rtFor(groupColors[0], groupLow);
  const VkFormat depthFmt = groupDepthFormat;
  const uint64_t pkey = ((uint64_t(rt.format) << 8) | rt.desc.samples) | (uint64_t(depthFmt) << 40);
  auto it = quadPipelines.find(pkey);
  VkPipeline pipe = it != quadPipelines.end() ? it->second : quadPipelineFor(rt.format, rt.desc.samples);
  if (!pipe) return false;
  resumeRendering();
  VkCommandBuffer cmd = slot().cmd;
  vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
  // Its own pipeline (static depth/stencil/cull state), layout and descriptor sets: the next game draw rebinds all.
  invalidateDrawState();
  VkViewport v{ 0, 0, float(renderW), float(renderH), 0, 1 };
  vk.vkCmdSetViewport(cmd, 0, 1, &v);
  // Window pixels; a scaled scene's twins are of the render size (backend_ffx.inl).
  const float qs = groupLow ? float(sceneW) / float(screenW) : 1.0f, qt = groupLow ? float(sceneH) / float(screenH) : 1.0f;
  const int32_t c0 = groupLow ? toSceneX(clip[0]) : clip[0], c1 = groupLow ? toSceneY(clip[1]) : clip[1];
  const int32_t c2 = groupLow ? toSceneX(clip[2]) : clip[2], c3 = groupLow ? toSceneY(clip[3]) : clip[3];
  const int32_t x1 = std::max(c0, 0), y1 = std::max(c1, 0);
  const int32_t x2 = std::min(c2, int32_t(renderW)), y2 = std::min(c3, int32_t(renderH));
  if (x2 <= x1 || y2 <= y1) return true;
  VkRect2D sc{ { x1, y1 }, { uint32_t(x2 - x1), uint32_t(y2 - y1) } };
  vk.vkCmdSetScissor(cmd, 0, 1, &sc);
  uint32_t ss[14] = {};
  ss[D3DSAMP_ADDRESSU] = ss[D3DSAMP_ADDRESSV] = D3DTADDRESS_CLAMP;
  ss[D3DSAMP_ADDRESSW] = D3DTADDRESS_WRAP;
  ss[D3DSAMP_MAGFILTER] = ss[D3DSAMP_MINFILTER] = ss[D3DSAMP_MIPFILTER] = D3DTEXF_LINEAR;
  ss[D3DSAMP_MAXANISOTROPY] = 1;
  const VkSampler smp = samplerFor(ss, false);
  for (uint32_t i = 0; i < n; i++) {
    auto t = textures.find(q[i].tex);
    if (t == textures.end()) { st.unknownTextures++; continue; }
    VkDescriptorSet set = allocPassSet();
    if (!set) return false;
    VkDescriptorImageInfo ii{ VK_NULL_HANDLE, t->second.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo si{ smp, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
    VkWriteDescriptorSet w[2] = { { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }, { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET } };
    w[0].dstSet = w[1].dstSet = set;
    w[0].dstBinding = 16;
    w[0].descriptorCount = 1;
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w[0].pImageInfo = &ii;
    w[1].dstBinding = 32;
    w[1].descriptorCount = 1;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    w[1].pImageInfo = &si;
    vk.vkUpdateDescriptorSets(device, 2, w, 0, nullptr);
    vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, quadLayout, 1, 1, &set, 1, &kZeroOffset);
    // D3D9 pixel centres sit on integer pretransformed coordinates, Vulkan's on +0.5.
    const float sx = 2.0f / float(renderW), sy = 2.0f / float(renderH);
    const float rect[4] = { (q[i].x0 * qs + 0.5f) * sx - 1.0f, (q[i].y0 * qt + 0.5f) * sy - 1.0f,
                            (q[i].x1 * qs + 0.5f) * sx - 1.0f, (q[i].y1 * qt + 0.5f) * sy - 1.0f };
    vk.vkCmdPushConstants(cmd, quadLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, 16, rect);
    vk.vkCmdDraw(cmd, 4, 1, 0, 0);
    st.draws++;
  }
  return true;
}

void destroyQuadPath() {
  for (auto& [k, p] : quadPipelines) vk.vkDestroyPipeline(device, p, nullptr);
  quadPipelines.clear();
  if (quadVs) vk.vkDestroyShaderModule(device, quadVs, nullptr);
  if (quadPs) vk.vkDestroyShaderModule(device, quadPs, nullptr);
  if (quadLayout) vk.vkDestroyPipelineLayout(device, quadLayout, nullptr);
  quadVs = quadPs = VK_NULL_HANDLE;
  quadLayout = VK_NULL_HANDLE;
}

// --- effects (effectlib.h) -----------------------------------------------------------------------------------------
// A loaded effect: full-screen passes run on the open (or last) render-target group, normally the world right after
// TESMain::DrawWorld_DrawScene. Passes write the effect's own images (kept in GENERAL layout; a memory barrier orders
// each pass after the previous one) or the scene colour (drawn inside the group's render pass with the pass's blend,
// so multisampled targets get it per sample). The group's render pass is suspended while depth is read as a texture.
// Descriptor set 0 (shared by all passes of a run): 0 depth, 1 scene colour copy, 2..9 images, 10 point sampler,
// 11 linear sampler, 12 frame uniform block; push constants: the pass's output size and scale (effectHeader()).
struct FxImage {
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
  VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
  uint32_t w = 0, h = 0;
  VkFormat format = VK_FORMAT_UNDEFINED;
  float scale = 1.0f;
  uint32_t fixedW = 0, fixedH = 0;  // Size=WxH: this size whatever the frame's (scale unused)
};
struct FxPass {
  VkShaderModule module[2] = {};  // [1] multisampled depth
  int output = -1;
  uint32_t blend = 0;
  bool skipWhenAo = false;  // EffectDesc::Pass::skipWhenAoApplied
  VkPipeline imagePipe[2] = {};
  std::unordered_map<uint64_t, VkPipeline> scenePipes;  // by group attachments, blend and depth variant
};
struct Fx {
  std::vector<FxImage> images;
  std::vector<FxPass> passes;
  bool sceneColor = false;
  bool skinBuffer = false;  // EffectDesc::skinBuffer: skipped when no skin was stored (skinStoredSinceOn)
  int publishAo = -1;       // image index published as the frame's ambient occlusion (r visibility, g log2 distance)
  int publishGi = -1;       // ... as the frame's screen-space indirect light
  int publishGiDepth = -1;  // ... as the log2 distance the indirect light is for (validates it)
  bool hasPrev = false;  // the camera of the last run, for passes that reproject their own history
  uint64_t prevFrame = 0;
  Backend::EffectFrame prev{};
};
// What an effect published for the lighting (docs/SHADER_ABI.md "Ambient occlusion") and the camera of the run that
// drew it: the ambient occlusion (Publish=ao), and the screen-space indirect light with its depth (Publish=gi, gidepth),
// each from whichever effect publishes it.
struct AoHistory {
  uint32_t effect = 0;         // 0: nothing published
  uint64_t frame = 0;          // backend frame of the run
  float rect[4];               // the camera viewport in the image's pixels
  float imageSize[2];
  Backend::EffectFrame camera;
} aoHistory, giHistory;
bool aoLightingSet = false;
Backend::AoLighting aoLighting{};
// aoBlock's last result and what it was computed from (it changes when the state is set or a run publishes, not per
// draw): the history's frame and effect, the backend frame, the render area.
struct AoBlockKey {
  uint64_t historyFrame = 0, giFrame = 0, frame = 0;
  uint32_t effect = 0, giEffect = 0, w = 0, h = 0;
  bool operator==(const AoBlockKey&) const = default;
} aoBlockKey;
float aoBlockCached[12][4] = {};
VkImageView aoBlockView = VK_NULL_HANDLE, aoBlockGiView = VK_NULL_HANDLE, aoBlockGiDepthView = VK_NULL_HANDLE;
bool aoBlockValid = false;
std::unordered_map<uint32_t, Fx> effects;
uint32_t nextEffect = 1;
FxImage fxColorCopy;
// What fxColorCopy holds: the colour of `color` (group attachment 0) in frame `frame`, still current while `current`
// (no scene pass ran since it was made). A run with EffectFrame::sceneUnchanged reuses it instead of copying again.
struct { uint64_t frame = 0; RtKey color = 0; bool current = false; } fxCopyOf;
VkDescriptorSetLayout fxSetLayout = VK_NULL_HANDLE;
VkPipelineLayout fxLayout = VK_NULL_HANDLE;
VkShaderModule fxVs = VK_NULL_HANDLE;
std::vector<VkDescriptorPool> fxPools[kFramesInFlight];
size_t fxPool[kFramesInFlight] = {};
static constexpr uint32_t kFxImageBindings = 8;
struct FxFrameBlock {
  float rect[4], frustum[4], depthRange[4], target[4], time[4];
  float hdr[4];  // output (HdrSettings::Output active), peak nits, paper white nits, peak / paper white (1 for SDR)
  float params[32][4];
  float cameraRows[3][4], waterSun[4];
  float shadowFromView[4][16];  // row-major (the header declares them row_major)
  float shadowSplits[4];
  float sunView[4];
  float shadowInfo[4];          // cascades, texel size (1 / size), 1 in an interior cell, 0
  float sunColor[4], ambientColor[4];
  float prevCameraRows[3][4];   // the effect's previous run (one or two frames ago): camera rows, frustum,
  float prevFrustum[4];         // and valid, near, far (prevInfo.x 0: no previous run)
  float prevInfo[4];
};

bool initFx() {
  if (fxLayout) return true;
  // 0..9 images, 10/11 samplers, 12 frame block, 13 sun shadow cascades (2D array), 14 its comparison sampler, 15 / 16
  // the skin buffer (albedo, nearest skin depth)
  VkDescriptorSetLayoutBinding b[17] = {};
  for (uint32_t i = 0; i < 17; i++) {
    b[i].binding = i;
    b[i].descriptorCount = 1;
    b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    b[i].descriptorType = i < 10 || i == 13 || i >= 15 ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
                          : i < 12 || i == 14 ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  }
  VkDescriptorSetLayoutCreateInfo lci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
  lci.bindingCount = 17;
  lci.pBindings = b;
  if (!check(vk.vkCreateDescriptorSetLayout(device, &lci, nullptr, &fxSetLayout), "vkCreateDescriptorSetLayout(fx)")) return false;
  VkPushConstantRange pr{ VK_SHADER_STAGE_FRAGMENT_BIT, 0, 32 };
  VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
  plci.setLayoutCount = 1;
  plci.pSetLayouts = &fxSetLayout;
  plci.pushConstantRangeCount = 1;
  plci.pPushConstantRanges = &pr;
  if (!check(vk.vkCreatePipelineLayout(device, &plci, nullptr, &fxLayout), "vkCreatePipelineLayout(fx)")) return false;
  VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
  smci.codeSize = sizeof(builtin::kFullscreenVert);
  smci.pCode = builtin::kFullscreenVert;
  return check(vk.vkCreateShaderModule(device, &smci, nullptr, &fxVs), "vkCreateShaderModule(fx vs)");
}
VkDescriptorSet allocFxSet() {
  const uint32_t s = uint32_t(frameNumber % kFramesInFlight);
  for (;;) {
    if (fxPool[s] == fxPools[s].size()) {
      VkDescriptorPoolSize sizes[] = { { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 64 * 13 }, { VK_DESCRIPTOR_TYPE_SAMPLER, 64 * 3 },
                                       { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64 } };
      VkDescriptorPoolCreateInfo ci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
      ci.maxSets = 64;
      ci.poolSizeCount = 3;
      ci.pPoolSizes = sizes;
      VkDescriptorPool p = VK_NULL_HANDLE;
      if (!check(vk.vkCreateDescriptorPool(device, &ci, nullptr, &p), "vkCreateDescriptorPool(fx)")) return VK_NULL_HANDLE;
      fxPools[s].push_back(p);
    }
    VkDescriptorSetAllocateInfo ai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = fxPools[s][fxPool[s]];
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &fxSetLayout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    const VkResult r = vk.vkAllocateDescriptorSets(device, &ai, &set);
    if (r == VK_SUCCESS) return set;
    if (r != VK_ERROR_OUT_OF_POOL_MEMORY && r != VK_ERROR_FRAGMENTED_POOL) return VK_NULL_HANDLE;
    fxPool[s]++;
  }
}
void resetFxPools() {  // the slot's previous frame has completed (beginDrawSlot)
  const uint32_t s = uint32_t(frameNumber % kFramesInFlight);
  for (VkDescriptorPool p : fxPools[s]) vk.vkResetDescriptorPool(device, p, 0);
  fxPool[s] = 0;
}

VkPipeline fxPipeline(VkShaderModule ps, const VkFormat* colors, uint32_t colorCount, uint32_t samples, VkFormat depth,
                      uint32_t blend) {
  VkPipelineShaderStageCreateInfo stages[2] = { { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
                                                { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO } };
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = fxVs;
  stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = ps;
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
  ms.rasterizationSamples = sampleBits(samples);
  VkPipelineDepthStencilStateCreateInfo ds{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };  // all off
  VkPipelineColorBlendAttachmentState cba[4] = {};
  for (uint32_t i = 0; i < colorCount; i++) cba[i].colorWriteMask = i == 0 ? 0xF : 0;
  switch (fx::Blend(blend)) {
    case fx::Blend::Alpha:  // rgb: src*a + dst*(1-a); alpha kept
      cba[0].blendEnable = VK_TRUE;
      cba[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      cba[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      cba[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
      cba[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      break;
    case fx::Blend::Add:  // rgb: src + dst; alpha kept
      cba[0].blendEnable = VK_TRUE;
      cba[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
      cba[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
      cba[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
      cba[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      break;
    case fx::Blend::Multiply:  // rgb: src * dst; alpha kept
      cba[0].blendEnable = VK_TRUE;
      cba[0].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
      cba[0].dstColorBlendFactor = VK_BLEND_FACTOR_SRC_COLOR;
      cba[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
      cba[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      break;
    default:
      break;
  }
  VkPipelineColorBlendStateCreateInfo cb{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
  cb.attachmentCount = colorCount;
  cb.pAttachments = cba;
  VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
  VkPipelineDynamicStateCreateInfo dy{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
  dy.dynamicStateCount = 2;
  dy.pDynamicStates = dyn;
  VkPipelineRenderingCreateInfo rci{ VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
  rci.colorAttachmentCount = colorCount;
  rci.pColorAttachmentFormats = colors;
  rci.depthAttachmentFormat = hasDepth(depth) ? depth : VK_FORMAT_UNDEFINED;
  rci.stencilAttachmentFormat = hasStencil(depth) ? depth : VK_FORMAT_UNDEFINED;
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
  pci.layout = fxLayout;
  VkPipeline p = VK_NULL_HANDLE;
  check(vk.vkCreateGraphicsPipelines(device, pipelineCache, 1, &pci, nullptr, &p), "vkCreateGraphicsPipelines(fx)");
  return p;
}

void releaseFxImage(FxImage& a) {
  if (a.image) garbage.push_back({ frameNumber, { a.image, VK_NULL_HANDLE }, { a.memory, VK_NULL_HANDLE }, { a.view, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE } });
  a.image = VK_NULL_HANDLE;
  a.memory = VK_NULL_HANDLE;
  a.view = VK_NULL_HANDLE;
  a.layout = VK_IMAGE_LAYOUT_UNDEFINED;
  a.w = a.h = 0;
}
bool ensureFxImage(FxImage& a, uint32_t w, uint32_t h, VkFormat f, VkImageUsageFlags extra = 0) {
  if (a.image && a.w == w && a.h == h && a.format == f) return true;
  releaseFxImage(a);
  const VkImageUsageFlags usage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | extra;  // src: readback
  if (!createImage(f, w, h, VK_SAMPLE_COUNT_1_BIT, usage, VK_IMAGE_ASPECT_COLOR_BIT, a.image, a.memory, a.view)) return false;
  a.w = w;
  a.h = h;
  a.format = f;
  return true;  // layout UNDEFINED: moved to GENERAL outside any render pass (doRunEffect, after suspending the group)
}

uint32_t doCreateEffect(const Backend::EffectDesc& d) {
  if (!initFx() || d.passes.empty() || d.images.size() > kFxImageBindings) return 0;
  Fx fx;
  fx.sceneColor = d.sceneColor;
  fx.skinBuffer = d.skinBuffer;
  for (const Backend::EffectDesc::Image& im : d.images) {
    FxImage f;
    f.format = VkFormat(im.vkFormat);
    f.scale = im.scale;
    f.fixedW = im.width;
    f.fixedH = im.height;
    if (im.publish == Backend::EffectDesc::Ao && fx.publishAo < 0) fx.publishAo = int(fx.images.size());
    if (im.publish == Backend::EffectDesc::Gi && fx.publishGi < 0) fx.publishGi = int(fx.images.size());
    if (im.publish == Backend::EffectDesc::GiDepth && fx.publishGiDepth < 0) fx.publishGiDepth = int(fx.images.size());
    fx.images.push_back(f);
  }
  for (const Backend::EffectDesc::Pass& p : d.passes) {
    FxPass fp;
    fp.output = p.output;
    fp.blend = p.blend;
    fp.skipWhenAo = p.skipWhenAoApplied;
    for (int v = 0; v < 2; v++) {
      VkShaderModuleCreateInfo ci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
      ci.codeSize = p.spvWords[v] * 4;
      ci.pCode = p.spv[v];
      if (!p.spv[v] || !check(vk.vkCreateShaderModule(device, &ci, nullptr, &fp.module[v]), "vkCreateShaderModule(fx)")) {
        for (FxPass& done : fx.passes)
          for (VkShaderModule module : done.module) if (module) vk.vkDestroyShaderModule(device, module, nullptr);
        for (VkShaderModule module : fp.module) if (module) vk.vkDestroyShaderModule(device, module, nullptr);
        return 0;
      }
    }
    if (fp.output >= int(fx.images.size())) return 0;
    fx.passes.push_back(std::move(fp));
  }
  const uint32_t id = nextEffect++;
  effects.emplace(id, std::move(fx));
  return id;
}
bool doReadbackEffectImage(uint32_t id, uint32_t index, std::vector<uint8_t>& out, uint32_t& w, uint32_t& h) {
  auto it = effects.find(id);
  if (it == effects.end() || index >= it->second.images.size()) return false;
  return readbackFxImage(it->second.images[index], out, w, h);
}
// An effect's (or the backend's own) image as its last pass left it (GENERAL), tightly packed. Waits for the GPU.
bool readbackFxImage(FxImage& im, std::vector<uint8_t>& out, uint32_t& w, uint32_t& h) {
  if (!im.image || im.layout != VK_IMAGE_LAYOUT_GENERAL) return false;
  uint32_t texel = 4;
  switch (im.format) {
    case VK_FORMAT_R8_UNORM: texel = 1; break;
    case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16_SFLOAT: texel = 2; break;
    case VK_FORMAT_R16G16B16A16_SFLOAT: texel = 8; break;
    default: break;
  }
  if (rendering) doEndGroup();
  if (!beginFrameIfNeeded()) return false;
  const VkDeviceSize size = VkDeviceSize(im.w) * im.h * texel;
  VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
  bci.size = size;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VmaAllocationCreateInfo bai{};
  bai.usage = VMA_MEMORY_USAGE_AUTO;
  bai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  VkBuffer buf = VK_NULL_HANDLE;
  VmaAllocation alloc = nullptr;
  VmaAllocationInfo info{};
  if (!check(vmaCreateBuffer(allocator, &bci, &bai, &buf, &alloc, &info), "vmaCreateBuffer(fx readback)")) return false;
  barrier(im.image, im.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
  VkBufferImageCopy r{};
  r.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
  r.imageExtent = { im.w, im.h, 1 };
  vk.vkCmdCopyImageToBuffer(slot().cmd, im.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &r);
  barrier(im.image, im.layout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
  const bool ok = submitFrame(VK_NULL_HANDLE, VK_NULL_HANDLE);
  waitIdle();
  if (ok) {
    vmaInvalidateAllocation(allocator, alloc, 0, VK_WHOLE_SIZE);
    out.assign(static_cast<uint8_t*>(info.pMappedData), static_cast<uint8_t*>(info.pMappedData) + size);
    w = im.w;
    h = im.h;
  }
  vmaDestroyBuffer(allocator, buf, alloc);
  return ok;
}
void destroyFx(Fx& fx, bool now) {
  if (now) waitIdle();
  for (FxPass& p : fx.passes) {
    for (VkPipeline pp : p.imagePipe) if (pp) vk.vkDestroyPipeline(device, pp, nullptr);
    for (auto& [k, pp] : p.scenePipes) vk.vkDestroyPipeline(device, pp, nullptr);
    for (VkShaderModule module : p.module) if (module) vk.vkDestroyShaderModule(device, module, nullptr);
  }
  for (FxImage& im : fx.images) releaseFxImage(im);
}
bool doDestroyEffect(uint32_t id) {
  auto it = effects.find(id);
  if (it == effects.end()) return false;
  // Pipelines and modules may be referenced by submitted frames: wait for the device (effects are destroyed rarely:
  // reloads and shutdown).
  if (frameOpen) submitFrame(VK_NULL_HANDLE, VK_NULL_HANDLE);
  destroyFx(it->second, true);
  effects.erase(it);
  if (aoHistory.effect == id) aoHistory = AoHistory{};
  if (giHistory.effect == id) giHistory = AoHistory{};
  invalidateDrawState();
  return true;
}
void destroyAllFx() {  // teardown: the device is idle and the deferred garbage was already freed
  for (auto& [id, fx] : effects) destroyFx(fx, false);
  effects.clear();
  releaseFxImage(fxColorCopy);
  releaseFxImage(skinAlbedoImage);
  releaseFxImage(skinDepthImage);
  for (const Garbage& g : garbage) freeGarbage(g);  // what the releases above just deferred
  garbage.clear();
  for (auto& pools : fxPools) {
    for (VkDescriptorPool p : pools) vk.vkDestroyDescriptorPool(device, p, nullptr);
    pools.clear();
  }
  if (fxVs) vk.vkDestroyShaderModule(device, fxVs, nullptr);
  if (fxLayout) vk.vkDestroyPipelineLayout(device, fxLayout, nullptr);
  if (fxSetLayout) vk.vkDestroyDescriptorSetLayout(device, fxSetLayout, nullptr);
  fxVs = VK_NULL_HANDLE;
  fxLayout = VK_NULL_HANDLE;
  fxSetLayout = VK_NULL_HANDLE;
}

bool doRunEffect(uint32_t id, const Backend::EffectFrame& f, const float* params, uint32_t paramCount) {
  if (!f.sceneUnchanged) fxCopyOf.current = false;  // something may have drawn into the scene since the last copy
  auto it = effects.find(id);
  if (it == effects.end() || !beginFrameIfNeeded()) return false;
  Fx& fx = it->second;
  // No skin stored since the skin buffer was activated: fnvSkinAt is false at every pixel, and an effect reading the
  // skin buffer (effects/sss) leaves the scene as it is. Its images are not published or kept between runs.
  if (fx.skinBuffer && !skinStoredSinceOn) {
    st.fxNoSkinRuns++;
    return true;
  }
  // Passes with Skip=pbr do not run while the lighting applies the published ambient occlusion (effects/gtao's apply:
  // it would multiply the scene by 1).
  auto skipped = [&](const FxPass& p) { return p.skipWhenAo && f.aoApplied; };
  const bool wasRendering = rendering;
  if (!rendering && !reopenLastGroup("effect")) return false;
  auto close = [&](bool result) {
    if (!wasRendering && rendering) doEndGroup();  // the group was closed before: leave it closed
    return result;
  };
  if (groupColors.empty()) return close(false);
  Rt& color = *rtFor(groupColors[0], groupLow);
  Rt* depth = groupDepth ? rtFor(groupDepth, groupLow) : nullptr;
  const bool msDepth = depth && depth->desc.samples > 1;
  const VkImageView depthView = !depth ? VK_NULL_HANDLE : msDepth ? depth->depthReadView : depth->sampleView;
  if (depth && !depthView) return close(false);  // multisampled depth the device cannot sample
  const int variant = msDepth ? 1 : 0;
  const uint32_t W = color.desc.width, H = color.desc.height;
  // The camera viewport in the target, as NiDX9Renderer converts NiViewport (Gamebryo y up) to a D3D viewport.
  float rx = std::clamp(std::floor(f.port[0] * W), 0.0f, float(W)), ry = std::clamp(std::floor((1.0f - f.port[2]) * H), 0.0f, float(H));
  float rw = std::min(std::floor((f.port[1] - f.port[0]) * W), float(W) - rx);
  float rh = std::min(std::floor((f.port[2] - f.port[3]) * H), float(H) - ry);
  if (rw < 16 || rh < 16) return close(false);
  if (depth && (depth->desc.width < rx + rw || depth->desc.height < ry + rh)) return close(false);
  for (FxImage& im : fx.images) {
    const uint32_t iw = im.fixedW ? im.fixedW : std::max(1u, uint32_t(W * im.scale));
    const uint32_t ih = im.fixedH ? im.fixedH : std::max(1u, uint32_t(H * im.scale));
    if (!ensureFxImage(im, iw, ih, im.format)) return close(false);
  }

  // Pipelines: per pass, offscreen ones by depth variant, scene ones also by the group's attachments.
  VkFormat groupFormats[4];
  for (uint32_t i = 0; i < groupColorCount; i++) groupFormats[i] = VkFormat(groupColorFormats[i]);
  uint64_t groupKey = uint64_t(groupColorCount) | (uint64_t(groupSamples) << 4) | (uint64_t(groupDepthFormat) << 12) | (uint64_t(variant) << 24);
  for (uint32_t i = 0; i < groupColorCount; i++) groupKey ^= uint64_t(groupFormats[i]) << (28 + 9 * i);
  for (FxPass& p : fx.passes) {
    if (skipped(p)) continue;
    VkPipeline pipe = VK_NULL_HANDLE;
    if (p.output >= 0) {
      if (!p.imagePipe[variant])
        p.imagePipe[variant] = fxPipeline(p.module[variant], &fx.images[p.output].format, 1, 1, VK_FORMAT_UNDEFINED, p.blend);
      pipe = p.imagePipe[variant];
    } else {
      VkPipeline& sp = p.scenePipes[groupKey];
      if (!sp) sp = fxPipeline(p.module[variant], groupFormats, groupColorCount, groupSamples, groupDepthFormat, p.blend);
      pipe = sp;
    }
    if (!pipe) return close(false);
  }

  // The frame block and the descriptor set shared by every pass of this run.
  HostSlice ub;
  if (!hostAlloc(drawSlot().uniforms, sizeof(FxFrameBlock), uboAlign, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, ub)) return close(false);
  FxFrameBlock fb{};
  const float rectV[4] = { rx, ry, rw, rh };
  memcpy(fb.rect, rectV, 16);
  memcpy(fb.cameraRows, f.cameraRows, sizeof(fb.cameraRows));
  memcpy(fb.waterSun, f.waterSun, sizeof(fb.waterSun));
  memcpy(fb.frustum, f.frustum, 16);
  fb.depthRange[0] = f.nearZ;
  fb.depthRange[1] = f.farZ;
  fb.depthRange[2] = f.submersion;
  fb.shadowInfo[2] = f.interior ? 1.0f : 0.0f;
  fb.depthRange[3] = f.aoApplied ? 1.0f : 0.0f;
  fb.target[0] = float(W);
  fb.target[1] = float(H);
  fb.target[2] = 1.0f / float(W);
  fb.target[3] = 1.0f / float(H);
  fb.time[0] = f.time;
  fb.time[1] = float(f.frame);
  // Seconds since this effect's previous run (one or two frames ago); 0 without one (first run, or it did not run).
  if (fx.hasPrev && frameNumber - fx.prevFrame >= 1 && frameNumber - fx.prevFrame <= 2) fb.time[2] = std::max(0.0f, f.time - fx.prev.time);
  fb.hdr[0] = float(hdrActive);
  fb.hdr[1] = hdr.peakNits;
  fb.hdr[2] = hdr.paperWhiteNits;
  fb.hdr[3] = hdrActive != HdrSettings::Off ? hdr.peakNits / hdr.paperWhiteNits : 1.0f;
  if (shadowValid && f.shadowCascades && f.shadowCascades <= shadowLayers) {
    memcpy(fb.shadowFromView, f.shadowFromView, sizeof(fb.shadowFromView));
    memcpy(fb.shadowSplits, f.shadowSplits, sizeof(fb.shadowSplits));
    memcpy(fb.sunView, f.sunView, 12);
    fb.shadowInfo[0] = float(f.shadowCascades);
    fb.shadowInfo[1] = 1.0f / float(shadowSize);
  }
  memcpy(fb.sunColor, f.sunColor, 12);
  memcpy(fb.ambientColor, f.ambientColor, 12);
  if (fx.hasPrev && frameNumber - fx.prevFrame >= 1 && frameNumber - fx.prevFrame <= 2) {
    memcpy(fb.prevCameraRows, fx.prev.cameraRows, sizeof(fb.prevCameraRows));
    memcpy(fb.prevFrustum, fx.prev.frustum, 16);
    fb.prevInfo[0] = 1.0f;
    fb.prevInfo[1] = fx.prev.nearZ;
    fb.prevInfo[2] = fx.prev.farZ;
  }
  memcpy(fb.params, params, size_t(std::min(paramCount, 32u)) * 16);
  memcpy(ub.ptr, &fb, sizeof(fb));

  suspendRendering();
  const bool copyCurrent = fxCopyOf.current && fxCopyOf.frame == frameNumber && fxCopyOf.color == groupColors[0] &&
                           fxColorCopy.image && fxColorCopy.w == W && fxColorCopy.h == H && fxColorCopy.format == color.format;
  if (fx.sceneColor && copyCurrent) st.fxSceneCopiesReused++;
  if (fx.sceneColor && !copyCurrent) {
    // The scene colour as it is now (resolved when multisampled), for passes that read it.
    if (!ensureFxImage(fxColorCopy, W, H, color.format, VK_IMAGE_USAGE_TRANSFER_DST_BIT)) { resumeRendering(); return close(false); }
    barrier(fxColorCopy.image, fxColorCopy.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    VkCommandBuffer c = slot().cmd;
    if (color.msImage) {
      const VkImageLayout back = color.msLayout;
      barrier(color.msImage, color.msLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
      VkImageResolve r{};
      r.srcSubresource = r.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
      r.extent = { W, H, 1 };
      vk.vkCmdResolveImage(c, color.msImage, color.msLayout, fxColorCopy.image, fxColorCopy.layout, 1, &r);
      barrier(color.msImage, color.msLayout, back, VK_IMAGE_ASPECT_COLOR_BIT);
    } else {
      const VkImageLayout back = color.layout;
      barrier(color.image, color.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
      VkImageCopy cp{};
      cp.srcSubresource = cp.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
      cp.extent = { W, H, 1 };
      vk.vkCmdCopyImage(c, color.image, color.layout, fxColorCopy.image, fxColorCopy.layout, 1, &cp);
      barrier(color.image, color.layout, back, VK_IMAGE_ASPECT_COLOR_BIT);
    }
    barrier(fxColorCopy.image, fxColorCopy.layout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
    fxCopyOf = { frameNumber, groupColors[0], true };
    st.fxSceneCopies++;
  }
  if (depth) barrier(depth->image, depth->layout, VK_IMAGE_LAYOUT_GENERAL, aspectOf(*depth));
  // The skin buffer's writes (the world's skin draws) visible to the passes.
  barrier(skinAlbedoImage.image, skinAlbedoImage.layout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
  barrier(skinDepthImage.image, skinDepthImage.layout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
  ensureShadowPlaceholder();
  for (FxImage& im : fx.images)  // new images start UNDEFINED; layout changes are not allowed inside the render pass
    if (im.layout != VK_IMAGE_LAYOUT_GENERAL) barrier(im.image, im.layout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);

  uint32_t ss[14] = {};
  ss[D3DSAMP_ADDRESSU] = ss[D3DSAMP_ADDRESSV] = ss[D3DSAMP_ADDRESSW] = D3DTADDRESS_CLAMP;
  ss[D3DSAMP_MAGFILTER] = ss[D3DSAMP_MINFILTER] = D3DTEXF_POINT;
  ss[D3DSAMP_MIPFILTER] = D3DTEXF_NONE;
  const VkSampler pointSampler = samplerFor(ss, false);
  ss[D3DSAMP_MAGFILTER] = ss[D3DSAMP_MINFILTER] = D3DTEXF_LINEAR;
  const VkSampler linearSampler = samplerFor(ss, false);
  const VkDescriptorSet set = allocFxSet();
  if (!set) {
    if (depth) barrier(depth->image, depth->layout, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, aspectOf(*depth));
    resumeRendering();
    return close(false);
  }
  {
    // Bindings without a resource get the 1x1 dummy (a valid descriptor; the shader does not read it meaningfully).
    const VkDescriptorImageInfo dummy{ VK_NULL_HANDLE, dummyView[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo imgs[10];
    imgs[0] = depthView ? VkDescriptorImageInfo{ VK_NULL_HANDLE, depthView, VK_IMAGE_LAYOUT_GENERAL } : dummy;
    imgs[1] = fx.sceneColor ? VkDescriptorImageInfo{ VK_NULL_HANDLE, fxColorCopy.view, VK_IMAGE_LAYOUT_GENERAL } : dummy;
    for (uint32_t i = 0; i < kFxImageBindings; i++)
      imgs[2 + i] = i < fx.images.size() ? VkDescriptorImageInfo{ VK_NULL_HANDLE, fx.images[i].view, VK_IMAGE_LAYOUT_GENERAL } : dummy;
    const VkDescriptorImageInfo smp[2] = { { pointSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED },
                                           { linearSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED } };
    const VkDescriptorBufferInfo ubo{ ub.buffer, ub.offset, sizeof(FxFrameBlock) };
    const VkDescriptorImageInfo shadowImg{ VK_NULL_HANDLE, shadowArrayView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    const VkDescriptorImageInfo shadowSmp{ shadowCompareSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
    VkWriteDescriptorSet w[5] = { { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }, { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET },
                                  { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }, { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET },
                                  { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET } };
    w[3].dstSet = w[4].dstSet = set;
    w[3].dstBinding = 13;
    w[4].dstBinding = 14;
    w[3].descriptorCount = w[4].descriptorCount = 1;
    w[3].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    w[3].pImageInfo = &shadowImg;
    w[4].pImageInfo = &shadowSmp;
    w[0].dstSet = set;
    w[0].dstBinding = 0;
    w[0].descriptorCount = 10;  // bindings 0..9: consecutive bindings of one type are written as one array
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w[0].pImageInfo = imgs;
    w[1].dstSet = set;
    w[1].dstBinding = 10;
    w[1].descriptorCount = 2;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    w[1].pImageInfo = smp;
    w[2].dstSet = set;
    w[2].dstBinding = 12;
    w[2].descriptorCount = 1;
    w[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    w[2].pBufferInfo = &ubo;
    vk.vkUpdateDescriptorSets(device, shadowArrayView ? 5 : 3, w, 0, nullptr);
    const VkDescriptorImageInfo skin[2] = { { VK_NULL_HANDLE, skinAlbedoImage.view, VK_IMAGE_LAYOUT_GENERAL },
                                            { VK_NULL_HANDLE, skinDepthImage.view, VK_IMAGE_LAYOUT_GENERAL } };
    VkWriteDescriptorSet ws{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    ws.dstSet = set;
    ws.dstBinding = 15;
    ws.descriptorCount = 2;
    ws.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    ws.pImageInfo = skin;
    vk.vkUpdateDescriptorSets(device, 1, &ws, 0, nullptr);
  }

  VkCommandBuffer cmd = slot().cmd;
  bool inGroup = false;  // the group's render pass is suspended
  for (uint32_t pi = 0; pi < fx.passes.size(); pi++) {
    FxPass& p = fx.passes[pi];
    if (skipped(p)) {
      st.fxPassesSkipped++;
      continue;
    }
    struct { float size[4], info[4]; } pc{};
    if (p.output >= 0) {
      FxImage& out = fx.images[p.output];
      if (inGroup) {
        suspendRendering();
        if (depth) barrier(depth->image, depth->layout, VK_IMAGE_LAYOUT_GENERAL, aspectOf(*depth));
        inGroup = false;
      }
      barrier(out.image, out.layout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);  // after earlier passes
      // A scaled image: the camera viewport's part of it; a fixed-size one: all of it.
      const VkRect2D area = out.fixedW ? VkRect2D{ { 0, 0 }, { out.w, out.h } }
                                       : VkRect2D{ { int32_t(rx * out.scale), int32_t(ry * out.scale) },
                                                   { std::max(1u, uint32_t(rw * out.scale)), std::max(1u, uint32_t(rh * out.scale)) } };
      VkRenderingAttachmentInfo ca{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
      ca.imageView = out.view;
      ca.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
      ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
      ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
      ri.renderArea = area;
      ri.layerCount = 1;
      ri.colorAttachmentCount = 1;
      ri.pColorAttachments = &ca;
      vk.vkCmdBeginRendering(cmd, &ri);
      vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.imagePipe[variant]);
      const VkViewport v{ 0, 0, float(out.w), float(out.h), 0, 1 };
      vk.vkCmdSetViewport(cmd, 0, 1, &v);
      vk.vkCmdSetScissor(cmd, 0, 1, &area);
      pc = { { float(out.w), float(out.h), 1.0f / out.w, 1.0f / out.h }, { out.fixedW ? 0.0f : out.scale, float(pi), 0, 0 } };
      vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fxLayout, 0, 1, &set, 0, nullptr);
      vk.vkCmdPushConstants(cmd, fxLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
      vk.vkCmdDraw(cmd, 3, 1, 0, 0);
      vk.vkCmdEndRendering(cmd);
      barrier(out.image, out.layout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);  // visible to later passes
    } else {
      if (!inGroup) {
        if (depth) barrier(depth->image, depth->layout, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, aspectOf(*depth));
        resumeRendering();
        inGroup = true;
      }
      const VkRect2D area{ { int32_t(rx), int32_t(ry) },
                           { std::min<uint32_t>(uint32_t(rw), renderW > uint32_t(rx) ? renderW - uint32_t(rx) : 0),
                             std::min<uint32_t>(uint32_t(rh), renderH > uint32_t(ry) ? renderH - uint32_t(ry) : 0) } };
      fxCopyOf.current = false;  // the scene changes: a later run copies it again
      vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p.scenePipes[groupKey]);
      const VkViewport v{ 0, 0, float(renderW), float(renderH), 0, 1 };
      vk.vkCmdSetViewport(cmd, 0, 1, &v);
      vk.vkCmdSetScissor(cmd, 0, 1, &area);
      pc = { { float(W), float(H), 1.0f / W, 1.0f / H }, { 1.0f, float(pi), 0, 0 } };
      vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, fxLayout, 0, 1, &set, 0, nullptr);
      vk.vkCmdPushConstants(cmd, fxLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc), &pc);
      vk.vkCmdDraw(cmd, 3, 1, 0, 0);
    }
  }
  if (!inGroup) {
    if (depth) barrier(depth->image, depth->layout, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, aspectOf(*depth));
    resumeRendering();
  }
  invalidateDrawState();  // another pipeline, layout and dynamic state: the next game draw rebinds everything
  fx.hasPrev = true;
  fx.prevFrame = frameNumber;
  fx.prev = f;
  auto publish = [&](AoHistory& h, const FxImage& im) {
    h.effect = id;
    h.frame = frameNumber;
    const float r[4] = { rx * im.scale, ry * im.scale, std::max(1.0f, rw * im.scale), std::max(1.0f, rh * im.scale) };
    memcpy(h.rect, r, sizeof(r));
    h.imageSize[0] = float(im.w);
    h.imageSize[1] = float(im.h);
    h.camera = f;
  };
  if (fx.publishAo >= 0) publish(aoHistory, fx.images[fx.publishAo]);
  if (fx.publishGi >= 0 && fx.publishGiDepth >= 0) publish(giHistory, fx.images[fx.publishGi]);
  return close(true);
}

// The AO block for this draw (docs/SHADER_ABI.md "Ambient occlusion") and the image's view; false: no AO (block 0).
// The reprojection: view space x right, y up, z forward; camera rows R (world rotation, columns: direction, up, right)
// and origin T. view -> world: T + R P v with P v = (v.z, v.y, v.x); world -> previous view: P Rp^T (w - Tp). So
// current view -> previous view: P Rp^T Rc P v + P Rp^T (Tc - Tp).
bool aoBlock(float blk[12][4], VkImageView& view, VkImageView& giView, VkImageView& giDepthView) {
  const AoBlockKey key{ aoHistory.frame, giHistory.frame, frameNumber, aoHistory.effect, giHistory.effect, renderW, renderH };
  if (!(key == aoBlockKey) || !aoLightingSet) {
    aoBlockValid = aoBlockCompute(aoBlockCached, aoBlockView, aoBlockGiView, aoBlockGiDepthView);
    aoBlockKey = aoLightingSet ? key : AoBlockKey{};
  }
  memcpy(blk, aoBlockCached, sizeof(aoBlockCached));
  view = aoBlockView;
  giView = aoBlockGiView;
  giDepthView = aoBlockGiDepthView;
  return aoBlockValid;
}
// A history's image `index` of its effect, if it is from one or two frames ago and readable.
const FxImage* publishedImage(const AoHistory& h, int Fx::*which) {
  if (!h.effect) return nullptr;
  const uint64_t age = frameNumber - h.frame;
  if (age < 1 || age > 2) return nullptr;
  auto it = effects.find(h.effect);
  if (it == effects.end() || it->second.*which < 0) return nullptr;
  const FxImage& im = it->second.images[size_t(it->second.*which)];
  return im.view && im.layout == VK_IMAGE_LAYOUT_GENERAL ? &im : nullptr;
}
bool aoBlockCompute(float blk[12][4], VkImageView& view, VkImageView& giView, VkImageView& giDepthView) {
  memset(blk, 0, sizeof(float) * 48);
  view = giView = giDepthView = VK_NULL_HANDLE;
  if (!aoLightingSet) return false;
  const FxImage* ao = publishedImage(aoHistory, &Fx::publishAo);
  const FxImage* gi = publishedImage(giHistory, &Fx::publishGi);
  const FxImage* giDepth = gi ? publishedImage(giHistory, &Fx::publishGiDepth) : nullptr;
  if (!giDepth) gi = nullptr;
  if (!ao && !gi) return false;
  // One reprojection for both: the camera of the latest of the two; the other is used only when it is from the same
  // run frame (both effects run at the world point of every frame, with the same camera).
  const AoHistory& ref = !ao ? giHistory : !gi ? aoHistory : (giHistory.frame > aoHistory.frame ? giHistory : aoHistory);
  if (ao && aoHistory.frame != ref.frame) ao = nullptr;
  if (gi && giHistory.frame != ref.frame) gi = giDepth = nullptr;
  const Backend::AoLighting& c = aoLighting;
  const Backend::EffectFrame& p = ref.camera;
  if (!(c.nearZ > 0.0f) || !(c.farZ > c.nearZ)) return false;
  // The current viewport in target pixels, as doRunEffect computes it (Gamebryo y up).
  const float tw = float(renderW), th = float(renderH);
  const float rx = std::floor(c.port[0] * tw), ry = std::floor((1.0f - c.port[2]) * th);
  const float rw = std::floor((c.port[1] - c.port[0]) * tw), rh = std::floor((c.port[2] - c.port[3]) * th);
  if (rw < 1.0f || rh < 1.0f) return false;
  double M[3][3], t[3];
  const int P[3] = { 2, 1, 0 };
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      // (P Rp^T Rc P)[i][j] = sum_k Rp[k][P i] Rc[k][P j]
      double s = 0.0;
      for (int k = 0; k < 3; k++) s += double(p.cameraRows[k][P[i]]) * c.cameraRows[k][P[j]];
      M[i][j] = s;
    }
    double s = 0.0;
    for (int k = 0; k < 3; k++) s += double(p.cameraRows[k][P[i]]) * (double(c.cameraRows[k][3]) - p.cameraRows[k][3]);
    t[i] = s;
  }
  blk[0][0] = 1.0f;
  const float rect[4] = { rx, ry, rw, rh };
  memcpy(blk[1], rect, 16);
  memcpy(blk[2], c.frustum, 16);
  blk[3][0] = c.nearZ;
  blk[3][1] = c.farZ;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) blk[4 + i][j] = float(M[i][j]);
    blk[4 + i][3] = float(t[i]);
  }
  memcpy(blk[7], p.frustum, 16);
  if (ao) {
    blk[0][1] = 1.0f;
    memcpy(blk[8], aoHistory.rect, 16);
    blk[9][0] = 1.0f / aoHistory.imageSize[0];
    blk[9][1] = 1.0f / aoHistory.imageSize[1];
    view = ao->view;
  }
  if (gi) {
    memcpy(blk[10], giHistory.rect, 16);
    blk[11][0] = 1.0f;
    giView = gi->view;
    giDepthView = giDepth->view;
  }
  return true;
}

// The wet block (docs/WETWORLD.md, layout in effectlib.cpp FNV_WET) and the puddle map's view. Without a state: zero
// (the shaders then add nothing).
// The skin buffer (Backend::setSkinBuffer; docs/SKIN.md "Subsurface scattering"): rgba8 albedo (gamma) and r32ui
// nearest skin depth (the depth value's bits; ~0: none). Both live in GENERAL; 1 x 1 until first activated.
FxImage skinAlbedoImage, skinDepthImage;
bool skinBufferOn = false, skinBufferUsable = false;
bool skinStoredSinceOn = false;  // a draw took the skin buffer since it was last activated (doSetSkinBuffer(true))
// Records the images' move to GENERAL (from `from`) and the depth's clear to ~0 on cb, outside a render pass.
void recordSkinBufferClear(VkCommandBuffer cb, VkImageLayout from) {
  VkImageMemoryBarrier2 b[2] = { { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 }, { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 } };
  for (int i = 0; i < 2; i++) {
    b[i].srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b[i].srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    b[i].dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b[i].dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b[i].oldLayout = from;
    b[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b[i].srcQueueFamilyIndex = b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b[i].image = i ? skinDepthImage.image : skinAlbedoImage.image;
    b[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
  }
  VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
  di.imageMemoryBarrierCount = 2;
  di.pImageMemoryBarriers = b;
  vk.vkCmdPipelineBarrier2(cb, &di);
  VkClearColorValue none{};
  none.uint32[0] = none.uint32[1] = none.uint32[2] = none.uint32[3] = 0xFFFFFFFFu;
  const VkImageSubresourceRange r{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
  vk.vkCmdClearColorImage(cb, skinDepthImage.image, VK_IMAGE_LAYOUT_GENERAL, &none, 1, &r);
  for (int i = 0; i < 2; i++) b[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
  vk.vkCmdPipelineBarrier2(cb, &di);
  skinAlbedoImage.layout = skinDepthImage.layout = VK_IMAGE_LAYOUT_GENERAL;
}
bool skinBufferFormatsOk() {
  VkFormatProperties a{}, d{};
  vk.vkGetPhysicalDeviceFormatProperties(phys, VK_FORMAT_R8G8B8A8_UNORM, &a);
  vk.vkGetPhysicalDeviceFormatProperties(phys, VK_FORMAT_R32_UINT, &d);
  return fragmentStores && (a.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) &&
         (d.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT);
}
bool ensureSkinBuffer(uint32_t w, uint32_t h) {  // false: unusable (the images keep their old size)
  const VkImageUsageFlags u = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  return ensureFxImage(skinAlbedoImage, w, h, VK_FORMAT_R8G8B8A8_UNORM, u) &&
         ensureFxImage(skinDepthImage, w, h, VK_FORMAT_R32_UINT, u);
}
void doSetSkinBuffer(bool on) {
  skinBufferOn = false;
  if (on) skinStoredSinceOn = false;
  if (!on || !skinBufferUsable || !beginFrameIfNeeded() || !renderW || !renderH) return;
  const bool wasRendering = rendering && !renderingSuspended;
  suspendRendering();
  const VkImageLayout from = skinAlbedoImage.w == renderW && skinAlbedoImage.h == renderH ? VK_IMAGE_LAYOUT_GENERAL
                                                                                           : VK_IMAGE_LAYOUT_UNDEFINED;
  if (ensureSkinBuffer(renderW, renderH)) {
    recordSkinBufferClear(slot().cmd, from);
    skinBufferOn = true;
  }
  if (wasRendering) {
    resumeRendering();
    invalidateDrawState();
  }
}

bool wetLightingSet = false;
Backend::WetLighting wetLighting{};
TexKey skinDetail = 0;  // setSkinDetail
TexKey hairShift = 0;   // setHairShift
void wetBlock(float blk[kWetBlockVec4s][4], VkImageView& puddleView, VkImageView& levelsView) {
  memset(blk, 0, sizeof(float) * 4 * kWetBlockVec4s);
  puddleView = levelsView = VK_NULL_HANDLE;
  if (!wetLightingSet) return;
  const Backend::WetLighting& w = wetLighting;
  if (!(w.nearZ > 0.0f) || !(w.farZ > w.nearZ)) return;
  const float tw = float(renderW), th = float(renderH);
  const float rx = std::floor(w.port[0] * tw), ry = std::floor((1.0f - w.port[2]) * th);
  const float rw = std::floor((w.port[1] - w.port[0]) * tw), rh = std::floor((w.port[2] - w.port[3]) * th);
  if (rw < 1.0f || rh < 1.0f) return;
  blk[0][0] = 1.0f;
  blk[0][1] = w.rain;
  blk[0][2] = w.wetness;
  blk[0][3] = w.puddleLevel;
  for (int i = 0; i < 3; i++) blk[1][i] = w.cameraRows[i][3];
  blk[1][3] = w.time;
  blk[2][0] = w.puddleCoverage;
  const bool cover = w.cover && shadowValid && w.coverLayer < shadowLayers;
  blk[2][1] = cover ? 1.0f : 0.0f;
  blk[2][2] = float(w.coverLayer);
  blk[2][3] = w.rippleStrength;
  for (int r = 0; r < 3; r++) memcpy(blk[3 + r], w.coverFromWorld + 4 * r, 16);
  blk[6][0] = w.coverBias;
  blk[6][1] = w.coverTexel;
  auto it = w.puddleMap ? textures.find(w.puddleMap) : textures.end();
  auto lt = w.puddleLevels ? textures.find(w.puddleLevels) : textures.end();
  if (it != textures.end() && it->second.view && lt != textures.end() && lt->second.view && w.puddleSize[0] &&
      w.puddleSize[1] && w.puddleSpacing > 0.0f) {
    levelsView = lt->second.view;
    blk[7][0] = w.puddleOrigin[0];
    blk[7][1] = w.puddleOrigin[1];
    blk[7][2] = 1.0f / w.puddleSpacing;
    blk[7][3] = 1.0f;
    blk[8][0] = float(w.puddleSize[0]);
    blk[8][1] = float(w.puddleSize[1]);
    blk[8][2] = w.basinWater ? 1.0f : 0.0f;
    puddleView = it->second.view;
  }
  const float rect[4] = { rx, ry, rw, rh };
  memcpy(blk[9], rect, 16);
  memcpy(blk[10], w.frustum, 16);
  blk[11][0] = w.nearZ;
  blk[11][1] = w.farZ;
  memcpy(blk[12], w.cameraRows, sizeof(w.cameraRows));
}

// --- gamma ramp at present ------------------------------------------------------------------------------------------
static constexpr TexKey kGammaLutKey = TexKey(~uintptr_t(0) - 0xF);
bool gammaActive = false;
VkShaderModule gammaPs = VK_NULL_HANDLE;
VkPipeline gammaPipe = VK_NULL_HANDLE;
VkFormat gammaPipeFormat = VK_FORMAT_UNDEFINED;

void doSetGammaRamp(const uint16_t* r, const uint16_t* g, const uint16_t* b) {
  bool identity = true;
  std::vector<uint8_t> lut(256 * 4);
  for (uint32_t i = 0; i < 256; i++) {
    // D3D9 identity is v * 65535 / 255 (+0.5 truncated by the game); compare at 8 bits.
    if (uint32_t(r[i] >> 8) != i || uint32_t(g[i] >> 8) != i || uint32_t(b[i] >> 8) != i) identity = false;
    lut[i * 4 + 0] = uint8_t(b[i] >> 8);  // A8R8G8B8 texel: B, G, R, A
    lut[i * 4 + 1] = uint8_t(g[i] >> 8);
    lut[i * 4 + 2] = uint8_t(r[i] >> 8);
    lut[i * 4 + 3] = 0xFF;
  }
  gammaActive = !identity;
  if (!gammaActive) return;
  d3dx::Texture t;
  t.format = 21;  // D3DFMT_A8R8G8B8
  t.width = 256;
  t.height = 1;
  t.levels = 1;
  t.cube = false;
  d3dx::Level l;
  l.width = 256;
  l.height = 1;
  l.rowPitch = 1024;
  l.rows = 1;
  l.data = std::move(lut);
  t.surfaces.push_back(std::move(l));
  gammaActive = doCreateTexture(kGammaLutKey, t);
}

VkPipeline gammaPipelineFor(VkFormat fmt) {
  if (gammaPipe && gammaPipeFormat == fmt) return gammaPipe;
  if (gammaPipe) {
    vk.vkDeviceWaitIdle(device);
    vk.vkDestroyPipeline(device, gammaPipe, nullptr);
    gammaPipe = VK_NULL_HANDLE;
  }
  if (!quadLayout && !quadPipelineFor(fmt, 1)) return VK_NULL_HANDLE;  // creates the shared layout and vertex shader
  if (!gammaPs) {
    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = sizeof(builtin::kGammaFrag);
    smci.pCode = builtin::kGammaFrag;
    check(vk.vkCreateShaderModule(device, &smci, nullptr, &gammaPs), "vkCreateShaderModule(gamma)");
  }
  VkPipelineShaderStageCreateInfo stages[2] = { { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
                                                { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO } };
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = quadVs;
  stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = gammaPs;
  stages[1].pName = "main";
  VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
  VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
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
  pci.layout = quadLayout;
  check(vk.vkCreateGraphicsPipelines(device, pipelineCache, 1, &pci, nullptr, &gammaPipe), "vkCreateGraphicsPipelines(gamma)");
  gammaPipeFormat = fmt;
  return gammaPipe;
}

// Draws `src` (GENERAL, sampled through `srcView`) over all of `dstView` (COLOR_ATTACHMENT_OPTIMAL) through the LUT.
bool recordGammaPresent(VkCommandBuffer cmd, VkImageView srcView, VkImageView dstView, VkFormat dstFormat, VkExtent2D ext) {
  auto lut = textures.find(kGammaLutKey);
  VkPipeline pipe = lut != textures.end() ? gammaPipelineFor(dstFormat) : VK_NULL_HANDLE;
  if (!pipe) return false;
  VkDescriptorSet set = allocPassSet();
  if (!set) return false;
  uint32_t ss[14] = {};
  ss[D3DSAMP_ADDRESSU] = ss[D3DSAMP_ADDRESSV] = D3DTADDRESS_CLAMP;
  ss[D3DSAMP_ADDRESSW] = D3DTADDRESS_WRAP;
  ss[D3DSAMP_MAGFILTER] = ss[D3DSAMP_MINFILTER] = D3DTEXF_LINEAR;
  ss[D3DSAMP_MIPFILTER] = D3DTEXF_NONE;
  ss[D3DSAMP_MAXANISOTROPY] = 1;
  VkDescriptorImageInfo src{ VK_NULL_HANDLE, srcView, VK_IMAGE_LAYOUT_GENERAL };
  VkDescriptorImageInfo lt{ VK_NULL_HANDLE, lut->second.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
  VkDescriptorImageInfo smp{ samplerFor(ss, false), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
  VkWriteDescriptorSet w[3] = { { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }, { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET },
                                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET } };
  const VkDescriptorImageInfo* infos[3] = { &src, &lt, &smp };
  const uint32_t bindings[3] = { 16, 17, 32 };
  for (int i = 0; i < 3; i++) {
    w[i].dstSet = set;
    w[i].dstBinding = bindings[i];
    w[i].descriptorCount = 1;
    w[i].descriptorType = i == 2 ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w[i].pImageInfo = infos[i];
  }
  vk.vkUpdateDescriptorSets(device, 3, w, 0, nullptr);
  VkRenderingAttachmentInfo ca{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
  ca.imageView = dstView;
  ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
  ri.renderArea = { { 0, 0 }, ext };
  ri.layerCount = 1;
  ri.colorAttachmentCount = 1;
  ri.pColorAttachments = &ca;
  vk.vkCmdBeginRendering(cmd, &ri);
  vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
  VkViewport v{ 0, 0, float(ext.width), float(ext.height), 0, 1 };
  VkRect2D sc{ { 0, 0 }, ext };
  vk.vkCmdSetViewport(cmd, 0, 1, &v);
  vk.vkCmdSetScissor(cmd, 0, 1, &sc);
  vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, quadLayout, 1, 1, &set, 1, &kZeroOffset);
  const float rect[4] = { -1, -1, 1, 1 };
  vk.vkCmdPushConstants(cmd, quadLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, 16, rect);
  vk.vkCmdDraw(cmd, 4, 1, 0, 0);
  vk.vkCmdEndRendering(cmd);
  return true;
}

void destroyGammaPath() {
  if (gammaPipe) vk.vkDestroyPipeline(device, gammaPipe, nullptr);
  if (gammaPs) vk.vkDestroyShaderModule(device, gammaPs, nullptr);
  gammaPipe = VK_NULL_HANDLE;
  gammaPs = VK_NULL_HANDLE;
}

// --- HDR output at present -------------------------------------------------------------------------------------------
// builtin/hdr.frag: the finished frame (gamma 2.2 encoded, 1.0 = SDR white) and the menu layer into HDR10 (PQ, BT.2020)
// or scRGB (linear BT.709, 1.0 = 80 nits), SDR white at the paper white level. Same vertex stage and descriptor set
// layout as the gamma pass; fragment push constants after the vertex stage's rectangle.
FxImage overlayLayer;  // the menu with HDR output (drawOverlayLayer)
VkPipelineLayout hdrLayout = VK_NULL_HANDLE;
VkShaderModule hdrPs = VK_NULL_HANDLE;
std::unordered_map<uint32_t, VkPipeline> hdrPipes;  // by target format

VkPipeline hdrPipelineFor(VkFormat fmt) {
  auto it = hdrPipes.find(uint32_t(fmt));
  if (it != hdrPipes.end()) return it->second;
  if (!quadLayout && !quadPipelineFor(fmt, 1)) return VK_NULL_HANDLE;  // creates the shared vertex shader
  if (!hdrLayout) {
    const VkPushConstantRange pr[2] = { { VK_SHADER_STAGE_VERTEX_BIT, 0, 16 }, { VK_SHADER_STAGE_FRAGMENT_BIT, 16, 16 } };
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = kSets;
    plci.pSetLayouts = setLayouts;
    plci.pushConstantRangeCount = 2;
    plci.pPushConstantRanges = pr;
    if (!check(vk.vkCreatePipelineLayout(device, &plci, nullptr, &hdrLayout), "vkCreatePipelineLayout(hdr)")) return VK_NULL_HANDLE;
  }
  if (!hdrPs) {
    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = sizeof(builtin::kHdrFrag);
    smci.pCode = builtin::kHdrFrag;
    if (!check(vk.vkCreateShaderModule(device, &smci, nullptr, &hdrPs), "vkCreateShaderModule(hdr)")) return VK_NULL_HANDLE;
  }
  VkPipelineShaderStageCreateInfo stages[2] = { { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
                                                { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO } };
  stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  stages[0].module = quadVs;
  stages[0].pName = "main";
  stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  stages[1].module = hdrPs;
  stages[1].pName = "main";
  VkPipelineVertexInputStateCreateInfo vi{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
  VkPipelineInputAssemblyStateCreateInfo ia{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
  ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
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
  pci.layout = hdrLayout;
  VkPipeline pipe = VK_NULL_HANDLE;
  if (!check(vk.vkCreateGraphicsPipelines(device, pipelineCache, 1, &pci, nullptr, &pipe), "vkCreateGraphicsPipelines(hdr)"))
    return VK_NULL_HANDLE;
  hdrPipes[uint32_t(fmt)] = pipe;
  return pipe;
}

// Draws `src` (GENERAL) and the menu layer `menuView` (GENERAL; null = none) over all of `dstView`
// (COLOR_ATTACHMENT_OPTIMAL) in the encoding of `output` (HdrSettings::Hdr10 / ScRgb).
bool recordHdrPresent(VkCommandBuffer cmd, VkImageView srcView, VkImageView menuView, VkImageView dstView, VkFormat dstFormat,
                      VkExtent2D ext, uint32_t output) {
  VkPipeline pipe = hdrPipelineFor(dstFormat);
  if (!pipe) return false;
  VkDescriptorSet set = allocPassSet();
  if (!set) return false;
  uint32_t ss[14] = {};
  ss[D3DSAMP_ADDRESSU] = ss[D3DSAMP_ADDRESSV] = D3DTADDRESS_CLAMP;
  ss[D3DSAMP_ADDRESSW] = D3DTADDRESS_WRAP;
  ss[D3DSAMP_MAGFILTER] = ss[D3DSAMP_MINFILTER] = D3DTEXF_LINEAR;
  ss[D3DSAMP_MIPFILTER] = D3DTEXF_NONE;
  ss[D3DSAMP_MAXANISOTROPY] = 1;
  VkDescriptorImageInfo src{ VK_NULL_HANDLE, srcView, VK_IMAGE_LAYOUT_GENERAL };
  VkDescriptorImageInfo menu = menuView ? VkDescriptorImageInfo{ VK_NULL_HANDLE, menuView, VK_IMAGE_LAYOUT_GENERAL }
                                        : VkDescriptorImageInfo{ VK_NULL_HANDLE, dummyView[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
  VkDescriptorImageInfo smp{ samplerFor(ss, false), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
  VkWriteDescriptorSet w[3] = { { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }, { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET },
                                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET } };
  const VkDescriptorImageInfo* infos[3] = { &src, &menu, &smp };
  const uint32_t bindings[3] = { 16, 17, 32 };
  for (int i = 0; i < 3; i++) {
    w[i].dstSet = set;
    w[i].dstBinding = bindings[i];
    w[i].descriptorCount = 1;
    w[i].descriptorType = i == 2 ? VK_DESCRIPTOR_TYPE_SAMPLER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    w[i].pImageInfo = infos[i];
  }
  vk.vkUpdateDescriptorSets(device, 3, w, 0, nullptr);
  VkRenderingAttachmentInfo ca{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
  ca.imageView = dstView;
  ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
  ri.renderArea = { { 0, 0 }, ext };
  ri.layerCount = 1;
  ri.colorAttachmentCount = 1;
  ri.pColorAttachments = &ca;
  vk.vkCmdBeginRendering(cmd, &ri);
  vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
  VkViewport v{ 0, 0, float(ext.width), float(ext.height), 0, 1 };
  VkRect2D sc{ { 0, 0 }, ext };
  vk.vkCmdSetViewport(cmd, 0, 1, &v);
  vk.vkCmdSetScissor(cmd, 0, 1, &sc);
  vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, hdrLayout, 1, 1, &set, 1, &kZeroOffset);
  const float rect[4] = { -1, -1, 1, 1 };
  vk.vkCmdPushConstants(cmd, hdrLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, 16, rect);
  const float params[4] = { float(output), hdr.paperWhiteNits, hdr.peakNits, menuView ? 1.0f : 0.0f };
  vk.vkCmdPushConstants(cmd, hdrLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 16, 16, params);
  vk.vkCmdDraw(cmd, 4, 1, 0, 0);
  vk.vkCmdEndRendering(cmd);
  return true;
}

// Backend::readbackHdrEncode: the encode into an offscreen image of the output's swapchain format, read back.
bool doReadbackHdrEncode(RtKey key, uint32_t output, std::vector<uint8_t>& out, uint32_t& vkFormat) {
  if (output != HdrSettings::Hdr10 && output != HdrSettings::ScRgb) return false;
  Rt* rt = lookup(key);
  if (!rt || rt->desc.depth || !realize(key, *rt) || !rt->sampleView || !beginFrameIfNeeded()) return false;
  if (rendering) doEndGroup();
  const VkFormat fmt = output == HdrSettings::Hdr10 ? VK_FORMAT_A2B10G10R10_UNORM_PACK32 : VK_FORMAT_R16G16B16A16_SFLOAT;
  const uint32_t ts = output == HdrSettings::Hdr10 ? 4 : 8;
  const uint32_t W = rt->desc.width, H = rt->desc.height;
  FxImage img;
  if (!ensureFxImage(img, W, H, fmt, VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) return false;
  VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
  bci.size = VkDeviceSize(W) * H * ts;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VmaAllocationCreateInfo aci{};
  aci.usage = VMA_MEMORY_USAGE_AUTO;
  aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  VkBuffer buf = VK_NULL_HANDLE;
  VmaAllocation alloc = nullptr;
  VmaAllocationInfo info{};
  if (!check(vmaCreateBuffer(allocator, &bci, &aci, &buf, &alloc, &info), "vmaCreateBuffer(hdr readback)")) {
    releaseFxImage(img);
    return false;
  }
  VkCommandBuffer cmd = slot().cmd;
  barrier(rt->image, rt->layout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
  barrier(img.image, img.layout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
  bool ok = recordHdrPresent(cmd, rt->sampleView, VK_NULL_HANDLE, img.view, fmt, { W, H }, output);
  if (ok) {
    barrier(img.image, img.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    VkBufferImageCopy c{};
    c.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    c.imageExtent = { W, H, 1 };
    vk.vkCmdCopyImageToBuffer(cmd, img.image, img.layout, buf, 1, &c);
  }
  ok = submitFrame(VK_NULL_HANDLE, VK_NULL_HANDLE) && ok;
  waitIdle();
  if (ok) {
    vmaInvalidateAllocation(allocator, alloc, 0, VK_WHOLE_SIZE);
    out.assign(static_cast<uint8_t*>(info.pMappedData), static_cast<uint8_t*>(info.pMappedData) + size_t(W) * H * ts);
    vkFormat = uint32_t(fmt);
  }
  vmaDestroyBuffer(allocator, buf, alloc);
  releaseFxImage(img);
  return ok;
}

void destroyHdrPath() {
  for (auto& [f, pp] : hdrPipes) vk.vkDestroyPipeline(device, pp, nullptr);
  hdrPipes.clear();
  if (hdrPs) vk.vkDestroyShaderModule(device, hdrPs, nullptr);
  if (hdrLayout) vk.vkDestroyPipelineLayout(device, hdrLayout, nullptr);
  hdrPs = VK_NULL_HANDLE;
  hdrLayout = VK_NULL_HANDLE;
  releaseFxImage(overlayLayer);  // deferred: freed with the rest of the garbage at teardown
}

// --- render-target group context used by draws ------------------------------------------------------------------
uint32_t groupColorFormats[4] = {}, groupColorCount = 0, groupSamples = 1;
uint32_t groupX8Mask = 0;  // colour attachments that are D3DFMT_X8R8G8B8 (no alpha channel in D3D9)
VkFormat groupDepthFormat = VK_FORMAT_UNDEFINED;
uint32_t groupRt0Format = 0;
VkPipeline boundPipeline = VK_NULL_HANDLE;

// --- per-draw caches ------------------------------------------------------------------------------------------------
// Consecutive draws mostly repeat their pipeline, dynamic state, samplers and descriptor contents. Each is compared with
// the previous draw's and only what changed is looked up, created, written or recorded.
uint64_t objectGen = 1;  // bumped when shaders, declarations or buffers are destroyed (cached pointers die with them)
struct LookupCache {
  uint64_t gen = 0;
  ShaderKey vsKey = 0, psKey = 0;
  const Shader* vs = nullptr;
  const Shader* ps = nullptr;
  DeclKey declKey = 0;
  const std::vector<VertexElement>* decl = nullptr;
  BufKey bufKey[16] = {};
  Buf* buf[16] = {};
  BufKey ibKey = 0;
  Buf* ib = nullptr;
} lookups;
struct AttrCache {
  uint64_t gen = 0;
  const PackedShader* vsh = nullptr;
  const std::vector<VertexElement>* decl = nullptr;
  uint32_t attrCount = 0, bindingMask = 0, streamMask = 0, missing = 0;
  PipelineKey::Attr attrs[16];
} attrCache;
PipelineKey lastPipeKey;
VkPipeline lastPipe = VK_NULL_HANDLE;
// What the previous draw's pipeline was chosen from (DrawSetup::renderStatesVersion), and its render-state blocks.
struct PipeReuse {
  uint64_t rsVersion = 0, objectGen = 0, groupGen = 0;
  const void* vsh = nullptr;
  const void* psh = nullptr;
  const void* decl = nullptr;
  uint32_t strides[8] = {};
  uint32_t pt = 0;
  bool operator==(const PipeReuse&) const = default;
} lastPipeReuse;
struct StateBlockReuse {
  uint64_t rsVersion = 0, clipVersion = 0, groupGen = 0, smallGen = 0;
  bool operator==(const StateBlockReuse&) const = default;
} lastStateBlocks;
VkDescriptorBufferInfo lastStateUbo[4] = {};  // alpha test, fog, clip planes, point args
bool lastTableFog = false;
uint64_t groupGen = 1, smallGen = 1;
struct SamplerSlot {
  bool valid = false, cube = false;
  uint32_t ss[14];
  VkSampler sampler = VK_NULL_HANDLE;
} samplerSlots[16];
struct DynState {
  bool valid = false;
  VkViewport vp;
  VkRect2D sc;
  uint32_t cull, depthTest, depthWrite, depthOp, stencilTest, sopF[4], sopB[4], cmpMask, writeMask, ref, biasOn;
  float bias, slope, blend[4];
} dynState;
VkDescriptorSet lastSet[2] = {}, boundSets[2] = {};
uint32_t boundOffsets[2] = {};  // the dynamic offsets of binding 0 the bound sets were bound with
uint64_t lastSig[2][112];
uint32_t lastSigCount[2] = {};

// Everything recorded state depends on: called when a command buffer starts, when its slot's pools are reset, and after
// another path binds its own pipeline / layout.
void invalidateDrawState() {
  boundPipeline = VK_NULL_HANDLE;
  dynState.valid = false;
  boundSets[0] = boundSets[1] = VK_NULL_HANDLE;
}

// --- depth pre-pass -------------------------------------------------------------------------------------------------
// The world's opaque pass (Backend::beginDepthPrepass .. endDepthPrepass, around the game's PreResolveDepth half) comes
// in the game's order, by shader and not front to back, so most pixels are shaded several times (4.5 to 6.7 per pixel
// measured at 4K outdoors). Its draws are held while they come and recorded together: first
// every eligible one depth-only (no fragment shader), then all of them in their order, the eligible ones without depth
// writes and with an EQUAL depth test, so each pixel is shaded once, by the surface that ends in front.
// Eligible (doDraw): depth test LESS / LESSEQUAL with depth writes, no blending, no stencil, solid triangles, a pixel
// shader that writes no depth (PackedShader::writesDepth), and a vertex shader whose position is Invariant (the same
// depth in both passes, bit for bit). The others are recorded as they were, among the eligible ones' colour draws; they
// test against the finished opaque depth, which hides only pixels that a later draw covered anyway.
// Draws that can discard (alpha-tested, or a pixel shader with a discard of its own, PackedShader::discards: the
// standard lighting's adts.frag keeps the game's texkill) take part through two modules of their pixel shader
// (BackendConfig::depthPrepassDiscarding; they are the largest part of the opaque pass). The depth draw runs the depth module
// (spvDepthVariant: the shader with its colour outputs made Private, so it discards each pixel as the shader does and
// the driver drops the work that only fed the colours); the colour draw runs the colour module (spvColourVariant: no
// discard) with the EQUAL test. The depth draw alone decides which pixels the draw covers and the colour draw shades
// exactly those, so the two cannot disagree. Their depth draws follow the other ones', which they test against first.
// Where such a draw is coplanar with a surface already drawn (an equal depth) the colour draw also passes where its depth
// draw discarded (as two coplanar opaque draws, the EQUAL test's tie rule).
// Draws are held only inside a render pass instance and outside occlusion queries. Anything else recorded into the
// frame (slot(): barriers, clears, queries, effects, the end of the group) records the held draws first, so the
// command order stays the call order. Everything a held draw records was resolved when it was made (pipeline,
// descriptor sets, buffer versions, dynamic state): nothing it depends on changes before it is recorded.
struct DrawRec {
  VkPipeline pipe = VK_NULL_HANDLE, depthPipe = VK_NULL_HANDLE;
  VkViewport vp{};
  VkRect2D sc{};
  uint32_t cull = 0, depthTest = 0, depthWrite = 0, depthOp = 0, stencilTest = 0, sopF[4] = {}, sopB[4] = {};
  uint32_t cmpMask = 0, writeMask = 0, ref = 0;
  uint32_t biasOn = 0;
  float bias = 0, slope = 0, blend[4] = {};
  VkDescriptorSet sets[2] = {};
  uint32_t dynOffsets[2] = {};
  uint32_t bindingMask = 0;
  VkBuffer vb[16] = {};
  VkDeviceSize vbOffset[16] = {};
  VkBuffer ib = VK_NULL_HANDLE;
  VkIndexType indexType = VK_INDEX_TYPE_UINT16;
  bool indexed = false, prepass = false;
  // prepass of a draw that can discard: its depth draw through the pixel shader's depth module, its colour draw through
  // its colour module (colourPipe).
  bool canDiscard = false;
  VkShaderModule depthModule = VK_NULL_HANDLE, colourModule = VK_NULL_HANDLE;
  VkPipeline colourPipe = VK_NULL_HANDLE;
  uint32_t count = 0, start = 0;
  int32_t baseVertex = 0;
};
bool prepassOn = false;           // between beginDepthPrepass and endDepthPrepass
std::vector<DrawRec> prepassHeld;  // in call order
bool prepassFlushing = false;
std::unordered_map<VkPipeline, VkPipeline> depthPipes;       // a game pipeline -> its depth-only pipeline
std::unordered_map<VkPipeline, VkPipeline> depthAlphaPipes;  // -> its depth pipeline through the depth module
std::unordered_map<VkPipeline, VkPipeline> colourPipes;      // -> itself with the colour module (no discard)
// Whether a draw made now would be held (recordDraw).
bool prepassHolds() const {
  return prepassOn && rendering && !renderingSuspended && !activeQuery && groupDepthFormat != VK_FORMAT_UNDEFINED;
}

void doBeginDepthPrepass() {
  flushPrepass();
  prepassOn = cfg.depthPrepass;
}
void doEndDepthPrepass() {
  flushPrepass();
  prepassOn = false;
}

// The depth-only pipeline of a game pipeline: its vertex stage and input, no colour writes (the group's colour formats
// stay: the same render pass instance), and no fragment shader, or for an alpha-tested draw its pixel shader's depth
// module (`ps`). Shared by the pipelines of one vertex shader and input (and depth module).
VkPipeline depthPipelineFor(VkPipeline pipe, const PackedShader* vsh, VkShaderModule ps) {
  auto& cache = ps ? depthAlphaPipes : depthPipes;
  auto it = cache.find(pipe);
  if (it != cache.end()) return it->second;
  auto k = pipelineKeys.find(pipe);
  if (k == pipelineKeys.end()) return VK_NULL_HANDLE;
  PipelineKey d = *k->second;
  d.ps = ps;
  d.blendEnable = d.srcC = d.dstC = d.opC = d.srcA = d.dstA = d.opA = 0;
  memset(d.writeMask, 0, sizeof(d.writeMask));
  const VkPipeline p = pipelineFor(d, vsh, nullptr);
  cache.emplace(pipe, p);
  return p;
}

// A game pipeline with its pixel shader's colour module instead (the same state and vertex input).
VkPipeline colourPipelineFor(VkPipeline pipe, VkShaderModule ps) {
  auto it = colourPipes.find(pipe);
  if (it != colourPipes.end()) return it->second;
  auto k = pipelineKeys.find(pipe);
  if (k == pipelineKeys.end()) return VK_NULL_HANDLE;
  PipelineKey c = *k->second;
  c.ps = ps;
  const VkPipeline p = pipelineFor(c);
  colourPipes.emplace(pipe, p);
  return p;
}

void recordDraw(DrawRec& r, const PackedShader* vsh) {
  if (!prepassHolds()) {
    r.prepass = false;  // drawn as it is
    emitDraw(slot().cmd, r, false);
    return;
  }
  if (r.prepass) {
    r.depthPipe = depthPipelineFor(r.pipe, vsh, r.canDiscard ? r.depthModule : VK_NULL_HANDLE);
    if (r.canDiscard) r.colourPipe = colourPipelineFor(r.pipe, r.colourModule);
    r.prepass = r.depthPipe != VK_NULL_HANDLE && (!r.canDiscard || r.colourPipe != VK_NULL_HANDLE);
  }
  r.canDiscard = r.canDiscard && r.prepass;
  prepassHeld.push_back(r);
  st.prepassDraws++;
}

// Records the held draws: the eligible ones depth-only (the ones that can discard last), then all of them in colour.
void flushPrepass() {
  if (prepassHeld.empty() || prepassFlushing) return;
  prepassFlushing = true;
  if (rendering && !renderingSuspended) {
    VkCommandBuffer cmd = slotRaw().cmd;
    for (const DrawRec& r : prepassHeld)
      if (r.prepass && !r.canDiscard) {
        emitDraw(cmd, r, true);
        st.prepassDepthDraws++;
      }
    for (const DrawRec& r : prepassHeld)
      if (r.prepass && r.canDiscard) {
        emitDraw(cmd, r, true);
        st.prepassDepthDraws++;
        st.prepassDiscardingDraws++;
      }
    for (const DrawRec& r : prepassHeld) emitDraw(cmd, r, false);
    st.prepassFlushes++;
  } else {
    // (never: every end or suspension of the render pass instance records the held draws first)
    logOnce("prepassorphan", "depth pre-pass: %zu held draws outside a render pass instance were dropped",
            prepassHeld.size());
  }
  prepassHeld.clear();
  prepassFlushing = false;
}

// Records one draw, each piece of state only when it differs from what the command buffer has. depthOnly: the draw's
// pre-pass draw; otherwise a pre-passed draw tests EQUAL without writing depth.
void emitDraw(VkCommandBuffer cmd, const DrawRec& r, bool depthOnly) {
  const VkPipeline pipe = depthOnly ? r.depthPipe : (r.prepass && r.canDiscard) ? r.colourPipe : r.pipe;
  if (pipe != boundPipeline) {
    vk.vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    boundPipeline = pipe;
  }
  DynState& dy = dynState;
  const bool dv = dy.valid;
  if (!dv) {  // constant for every game draw: recorded once per command buffer / after another path's pipeline
    vk.vkCmdSetFrontFace(cmd, VK_FRONT_FACE_CLOCKWISE);
    vk.vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
  }
  if (!dv || memcmp(&r.vp, &dy.vp, sizeof(r.vp)) != 0) {
    vk.vkCmdSetViewport(cmd, 0, 1, &r.vp);
    dy.vp = r.vp;
  }
  if (!dv || memcmp(&r.sc, &dy.sc, sizeof(r.sc)) != 0) {
    vk.vkCmdSetScissor(cmd, 0, 1, &r.sc);
    dy.sc = r.sc;
  }
  auto changed = [dv](uint32_t& slot, uint32_t v) {
    if (dv && slot == v) return false;
    slot = v;
    return true;
  };
  const uint32_t depthWrite = depthOnly ? 1u : r.prepass ? 0u : r.depthWrite;
  const uint32_t depthOp = (r.prepass && !depthOnly) ? uint32_t(VK_COMPARE_OP_EQUAL) : r.depthOp;
  if (changed(dy.cull, r.cull)) vk.vkCmdSetCullMode(cmd, VkCullModeFlags(dy.cull));
  if (changed(dy.depthTest, r.depthTest)) vk.vkCmdSetDepthTestEnable(cmd, dy.depthTest);
  if (changed(dy.depthWrite, depthWrite)) vk.vkCmdSetDepthWriteEnable(cmd, dy.depthWrite);
  if (changed(dy.depthOp, depthOp)) vk.vkCmdSetDepthCompareOp(cmd, VkCompareOp(dy.depthOp));
  if (changed(dy.stencilTest, r.stencilTest)) vk.vkCmdSetStencilTestEnable(cmd, dy.stencilTest);
  if (!dv || memcmp(r.sopF, dy.sopF, sizeof(r.sopF)) != 0) {
    vk.vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_FRONT_BIT, VkStencilOp(r.sopF[0]), VkStencilOp(r.sopF[1]),
                         VkStencilOp(r.sopF[2]), VkCompareOp(r.sopF[3]));
    memcpy(dy.sopF, r.sopF, sizeof(r.sopF));
  }
  if (!dv || memcmp(r.sopB, dy.sopB, sizeof(r.sopB)) != 0) {
    vk.vkCmdSetStencilOp(cmd, VK_STENCIL_FACE_BACK_BIT, VkStencilOp(r.sopB[0]), VkStencilOp(r.sopB[1]),
                         VkStencilOp(r.sopB[2]), VkCompareOp(r.sopB[3]));
    memcpy(dy.sopB, r.sopB, sizeof(r.sopB));
  }
  if (changed(dy.cmpMask, r.cmpMask)) vk.vkCmdSetStencilCompareMask(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, dy.cmpMask);
  if (changed(dy.writeMask, r.writeMask)) vk.vkCmdSetStencilWriteMask(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, dy.writeMask);
  if (changed(dy.ref, r.ref)) vk.vkCmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK, dy.ref);
  if (changed(dy.biasOn, r.biasOn)) vk.vkCmdSetDepthBiasEnable(cmd, dy.biasOn);
  if (r.biasOn && (!dv || dy.bias != r.bias || dy.slope != r.slope)) {
    vk.vkCmdSetDepthBias(cmd, r.bias, 0.0f, r.slope);
    dy.bias = r.bias;
    dy.slope = r.slope;
  } else if (!r.biasOn) {
    dy.bias = dy.slope = -1.0f;  // forces the next enabled bias to be recorded
  }
  if (!dv || memcmp(r.blend, dy.blend, sizeof(r.blend)) != 0) {
    vk.vkCmdSetBlendConstants(cmd, r.blend);
    memcpy(dy.blend, r.blend, sizeof(r.blend));
  }
  dy.valid = true;
  if (r.sets[0] != boundSets[0] || r.sets[1] != boundSets[1] || r.dynOffsets[0] != boundOffsets[0] ||
      r.dynOffsets[1] != boundOffsets[1]) {
    vk.vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, kSets, r.sets, kSets,
                               r.dynOffsets);
    boundSets[0] = r.sets[0];
    boundSets[1] = r.sets[1];
    boundOffsets[0] = r.dynOffsets[0];
    boundOffsets[1] = r.dynOffsets[1];
  }
  for (uint32_t b = 0; b < 16; b++)
    if (r.bindingMask & (1u << b)) vk.vkCmdBindVertexBuffers(cmd, b, 1, &r.vb[b], &r.vbOffset[b]);
  if (r.indexed) {
    vk.vkCmdBindIndexBuffer(cmd, r.ib, 0, r.indexType);
    vk.vkCmdDrawIndexed(cmd, r.count, 1, r.start, r.baseVertex, 0);
  } else {
    vk.vkCmdDraw(cmd, r.count, 1, r.start, 0);
  }
}

// Texture view for one sampler register. Returns false when nothing usable is bound.
bool viewFor(const DrawSetup::Texture& t, VkImageView& view, VkImageLayout& layout, bool& cube) {
  if (t.kind == DrawSetup::Texture::Tex) {
    auto it = textures.find(t.key);
    if (it == textures.end()) {
      st.unknownTextures++;
      return false;
    }
    view = it->second.view;
    layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    cube = it->second.layers == 6;
    return true;
  }
  if (t.kind == DrawSetup::Texture::RenderTarget && isCubeFace(t.key)) {
    // A rendered cube map's face: the cube, unless one of its faces is being drawn into (feedback, as below).
    auto f = cubeFaces.find(t.key);
    auto c = cubes.find(f->second.first);
    if (c != cubes.end())
      for (RtKey k : c->second.faces)
        if (k && (k == groupDepth || std::find(groupColors.begin(), groupColors.end(), k) != groupColors.end())) {
          st.feedbackTextures++;
          return false;
        }
    const Tex* cubeTex = cubeForFace(t.key);
    if (!cubeTex) {
      st.unknownTextures++;
      return false;
    }
    view = cubeTex->view;
    layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    cube = true;
    return true;
  }
  if (t.kind == DrawSetup::Texture::RenderTarget) {
    auto it = rts.find(t.key);
    if (it == rts.end() || !it->second.realized || !it->second.sampleView) {
      st.unknownTextures++;
      return false;
    }
    // In a scaled scene a window-sized target holds this frame's contents in its render-size twin (backend_ffx.inl).
    Rt* tp = scalable(t.key, it->second) ? lowTwin(t.key, it->second) : &it->second;
    if (!tp) {
      st.unknownTextures++;
      return false;
    }
    Rt& target = *tp;
    if (target.desc.depth) {
      // The shaders are translated without the depth-compare path (tools/sdp2spv: the game is not known to sample
      // depth textures). Count it so a game path that does is visible.
      st.depthTextureSamples++;
    }
    for (RtKey k : groupColors)
      if (k == t.key) { st.feedbackTextures++; return false; }
    if (groupDepth == t.key) { st.feedbackTextures++; return false; }
    if (!target.sampleView) {
      st.unknownTextures++;
      return false;
    }
    if (target.layout != VK_IMAGE_LAYOUT_GENERAL) {
      // Never rendered or still in a transfer layout: make it sampleable (outside the render pass).
      suspendRendering();
      barrier(target.image, target.layout, VK_IMAGE_LAYOUT_GENERAL, aspectOf(target));
      resumeRendering();
    }
    view = target.sampleView;
    layout = VK_IMAGE_LAYOUT_GENERAL;
    cube = false;
    return true;
  }
  return false;
}

void doDraw(const DrawSetup& ds, uint32_t pt, uint32_t prims, bool indexed, int32_t baseVertex, uint32_t start) {
  if (!rendering && !reopenLastGroup("draw")) { st.drawsNoGroup++; return; }
  if (!prims || !pack.size()) { st.drawsSkipped++; return; }
  if (lookups.gen != objectGen) {
    lookups = LookupCache{};
    lookups.gen = objectGen;
  }
  if (ds.vs != lookups.vsKey || !lookups.vs) {
    auto it = shaders.find(ds.vs);
    lookups.vsKey = ds.vs;
    lookups.vs = it == shaders.end() ? nullptr : &it->second;
  }
  if (ds.ps != lookups.psKey || !lookups.ps) {
    auto it = shaders.find(ds.ps);
    lookups.psKey = ds.ps;
    lookups.ps = it == shaders.end() ? nullptr : &it->second;
  }
  if (!lookups.vs || !lookups.ps || !lookups.vs->packed || !lookups.ps->packed) {
    st.drawsNoShader++;
    return;
  }
  const PackedShader* vsh = lookups.vs->packed;
  const PackedShader* psh = lookups.ps->packed;
  if (ds.decl != lookups.declKey || !lookups.decl) {
    auto it = decls.find(ds.decl);
    lookups.declKey = ds.decl;
    lookups.decl = it == decls.end() ? nullptr : &it->second;
  }
  if (!lookups.decl) { st.drawsSkipped++; return; }
  const uint32_t* rsv = ds.renderStates;

  PipeReuse reuse;
  reuse.rsVersion = ds.renderStatesVersion;
  reuse.objectGen = objectGen;
  reuse.groupGen = groupGen;
  reuse.vsh = vsh;
  reuse.psh = psh;
  reuse.decl = lookups.decl;
  for (uint32_t b = 0; b < 8; b++) reuse.strides[b] = ds.streams[b].stride;
  reuse.pt = pt;
  VkPipeline pipe;
  if (ds.renderStatesVersion && lastPipe && reuse == lastPipeReuse) {
    // Nothing the pipeline depends on changed since the previous draw.
    pipe = lastPipe;
    st.missingInputs += attrCache.missing;
    st.pipelineReuses++;
  } else {
  PipelineKey k;
  memset(&k, 0, sizeof(k));
  k.vs = moduleFor(vsh);
  k.ps = moduleFor(psh);
  k.topology = xlat::topology(pt);
  // Vertex inputs: the shader's inputs matched to the declaration's elements, once per (shader, declaration).
  if (attrCache.gen != objectGen || attrCache.vsh != vsh || attrCache.decl != lookups.decl) {
    attrCache = AttrCache{};
    attrCache.gen = objectGen;
    attrCache.vsh = vsh;
    attrCache.decl = lookups.decl;
    for (const PackedShader::Input& in : vsh->inputs) {
      if (attrCache.attrCount == 16) break;
      const VertexElement* found = nullptr;
      for (const VertexElement& e : *lookups.decl)
        if (e.usage == in.usage && e.usageIndex == in.usageIndex && e.type != D3DDECLTYPE_UNUSED && e.stream < 8) {
          found = &e;
          break;
        }
      PipelineKey::Attr& a = attrCache.attrs[attrCache.attrCount++];
      a.location = in.location;
      if (found && xlat::declType(found->type) != VK_FORMAT_UNDEFINED) {
        a.binding = found->stream;
        a.format = xlat::declType(found->type);
        a.offset = found->offset;
        attrCache.streamMask |= 1u << a.binding;
      } else {
        attrCache.missing++;
        a.binding = kZeroBinding;
        a.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        a.offset = 0;
      }
      attrCache.bindingMask |= 1u << a.binding;
    }
  }
  k.attrCount = attrCache.attrCount;
  memcpy(k.attrs, attrCache.attrs, sizeof(PipelineKey::Attr) * attrCache.attrCount);
  k.bindingMask = attrCache.bindingMask;
  for (uint32_t b = 0; b < 8; b++)
    if (attrCache.streamMask & (1u << b)) k.strides[b] = ds.streams[b].stride;
  st.missingInputs += attrCache.missing;
  // Blend (DXVK BindBlendState), with D3D9's missing alpha channel of X8 targets reading as 1.
  uint32_t src = rsv[D3DRS_SRCBLEND], dst = rsv[D3DRS_DESTBLEND];
  xlat::fixupBlend(src, dst);
  uint32_t srcA = src, dstA = dst, opA = rsv[D3DRS_BLENDOP];
  if (rsv[D3DRS_SEPARATEALPHABLENDENABLE]) {
    srcA = rsv[D3DRS_SRCBLENDALPHA];
    dstA = rsv[D3DRS_DESTBLENDALPHA];
    opA = rsv[D3DRS_BLENDOPALPHA];
    xlat::fixupBlend(srcA, dstA);
  }
  k.blendEnable = rsv[D3DRS_ALPHABLENDENABLE] ? 1 : 0;
  k.srcC = xlat::blendFactor(src, false);
  k.dstC = xlat::blendFactor(dst, false);
  k.opC = xlat::blendOp(rsv[D3DRS_BLENDOP]);
  k.srcA = xlat::blendFactor(srcA, true);
  k.dstA = xlat::blendFactor(dstA, true);
  k.opA = xlat::blendOp(opA);
  if (groupX8Mask & 1) {
    auto norm = [](uint32_t& f) {
      if (f == VK_BLEND_FACTOR_DST_ALPHA) f = VK_BLEND_FACTOR_ONE;
      else if (f == VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA) f = VK_BLEND_FACTOR_ZERO;
    };
    norm(k.srcC);
    norm(k.dstC);
  }
  static constexpr D3DRENDERSTATETYPE kWrite[4] = { D3DRS_COLORWRITEENABLE, D3DRS_COLORWRITEENABLE1,
                                                    D3DRS_COLORWRITEENABLE2, D3DRS_COLORWRITEENABLE3 };
  k.colorCount = groupColorCount;
  for (uint32_t i = 0; i < groupColorCount; i++) {
    k.writeMask[i] = rsv[kWrite[i]] & 0xF;  // D3DCOLORWRITEENABLE_RED..ALPHA = VK R,G,B,A bits
    k.colorFormats[i] = groupColorFormats[i];
  }
  k.depthFormat = groupDepthFormat;
  k.stencilFormat = hasStencil(groupDepthFormat) ? groupDepthFormat : VK_FORMAT_UNDEFINED;
  k.samples = groupSamples;
  k.fillMode = rsv[D3DRS_FILLMODE];
  if (lastPipe && memcmp(&k, &lastPipeKey, sizeof(k)) == 0) {
    pipe = lastPipe;
  } else {
    pipe = pipelineFor(k, vsh, psh);
    lastPipeKey = k;
    lastPipe = pipe;
  }
  lastPipeReuse = reuse;
  }
  if (!pipe) { st.drawsNoPipeline++; return; }

  // Textures first: binding an RT that needs a layout change may restart the render pass.
  VkDescriptorImageInfo img[kSets][kTexBindings] = {};
  VkDescriptorImageInfo smp[kSets][kTexBindings] = {};
  struct SamplerStateEntry { uint32_t type, useDref, useProj, isNull, useGather, drefClamp; float drefScale; };
  SamplerStateEntry ssBlock[16] = {};
  for (uint32_t s = 0; s < 16; s++) ssBlock[s] = { 2, 0, 0, 1, 0, 0, 1.0f };
  const PackedShader* stageSh[kSets] = { vsh, psh };
  for (uint32_t set = 0; set < kSets; set++) {
    for (const PackedShader::Tex& t : stageSh[set]->textures) {
      const uint32_t s = t.reg & 15;
      const bool wantCube = t.kind == 9;
      VkImageView view = VK_NULL_HANDLE;
      VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
      bool cube = false;
      bool ok = set == 1 && viewFor(ds.textures[s], view, layout, cube);
      if (ok && cube != wantCube) {
        st.textureKindMismatch++;
        ok = false;
      }
      if (!ok) {
        view = dummyView[wantCube ? 1 : 0];
        layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      }
      img[set][s] = { VK_NULL_HANDLE, view, layout };
      SamplerSlot& sl = samplerSlots[s];
      if (!sl.valid || sl.cube != wantCube || memcmp(sl.ss, ds.samplerStates[s], sizeof(sl.ss)) != 0) {
        sl.valid = true;
        sl.cube = wantCube;
        memcpy(sl.ss, ds.samplerStates[s], sizeof(sl.ss));
        sl.sampler = samplerFor(ds.samplerStates[s], wantCube);
      }
      smp[set][s] = { sl.sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
      if (set == 1) ssBlock[s] = { wantCube ? 3u : 2u, 0, 0, ok ? 0u : 1u, 0, 0, 1.0f };
    }
  }

  // The environment cube, resolved with the textures and before this draw binds anything: bringing it up to date may
  // record the backend's own passes (the cube's copies, the GGX prefilter with its own pipeline and descriptor set),
  // which invalidate the draw state; the draw then binds its pipeline, sets and dynamic state after them.
  const bool envLit = (psh->cbvMask & (1u << kEnvBlockBinding)) != 0;
  bool envValid = false;
  VkDescriptorImageInfo envImage{ VK_NULL_HANDLE, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
  if (envLit) envImage.imageView = envView(envValid);

  // What the draw records, resolved now; recorded at once, or held for the depth pre-pass (recordDraw).
  DrawRec r;
  r.pipe = pipe;

  // Viewport and scissor (DXVK BindViewportAndScissor: half-pixel offset, y flip, MinZ bias). The game gives them in window
  // pixels: a scaled scene's twins are of the render size (backend_ffx.inl).
  auto v = ds.viewport;
  int32_t dsc[4] = { ds.scissor[0], ds.scissor[1], ds.scissor[2], ds.scissor[3] };
  if (groupLow) {
    const int32_t x1 = toSceneX(v.x + int32_t(v.width)), y1 = toSceneY(v.y + int32_t(v.height));
    v.x = toSceneX(v.x);
    v.y = toSceneY(v.y);
    v.width = uint32_t(std::max(x1 - v.x, 0));
    v.height = uint32_t(std::max(y1 - v.y, 0));
    for (int i = 0; i < 4; i++) dsc[i] = (i & 1) ? toSceneY(dsc[i]) : toSceneX(dsc[i]);
  }
  float zBias = v.minZ >= 0.5f ? 0.0f : 0.001f;
  VkViewport vp{ float(v.x) + 0.5f, float(int64_t(v.height) + v.y) + 0.5f, float(v.width), -float(v.height),
                 std::clamp(v.minZ, 0.0f, 1.0f), std::clamp(std::max(v.maxZ, v.minZ + zBias), 0.0f, 1.0f) };
  r.vp = vp;
  VkRect2D sc;
  if (rsv[D3DRS_SCISSORTESTENABLE]) {
    int32_t ax = std::max<int32_t>(std::max<int32_t>(0, dsc[0]), v.x);
    int32_t ay = std::max<int32_t>(std::max<int32_t>(0, dsc[1]), v.y);
    int32_t bx = std::min<int32_t>(std::max<int32_t>(ax, dsc[2]), v.x + int32_t(v.width));
    int32_t by = std::min<int32_t>(std::max<int32_t>(ay, dsc[3]), v.y + int32_t(v.height));
    sc = { { ax, ay }, { uint32_t(std::max(bx - ax, 0)), uint32_t(std::max(by - ay, 0)) } };
  } else {
    sc = { { std::max(v.x, 0), std::max(v.y, 0) }, { v.width, v.height } };
  }
  r.sc = sc;
  r.cull = uint32_t(xlat::cullMode(rsv[D3DRS_CULLMODE]));
  const bool hasDs = groupDepthFormat != VK_FORMAT_UNDEFINED;
  r.depthTest = hasDs && rsv[D3DRS_ZENABLE];
  r.depthWrite = hasDs && rsv[D3DRS_ZWRITEENABLE];
  r.depthOp = uint32_t(xlat::compareOp(rsv[D3DRS_ZFUNC]));
  const bool stencil = hasStencil(groupDepthFormat) && rsv[D3DRS_STENCILENABLE];
  r.stencilTest = stencil;
  {
    // Front = clockwise; D3DRS_CCW_* apply to counter-clockwise (back) faces in two-sided mode.
    const bool twoSided = stencil && rsv[D3DRS_TWOSIDEDSTENCILMODE];
    const uint32_t f[4] = { uint32_t(xlat::stencilOp(rsv[D3DRS_STENCILFAIL])), uint32_t(xlat::stencilOp(rsv[D3DRS_STENCILPASS])),
                            uint32_t(xlat::stencilOp(rsv[D3DRS_STENCILZFAIL])), uint32_t(xlat::compareOp(rsv[D3DRS_STENCILFUNC])) };
    const uint32_t b[4] = { twoSided ? uint32_t(xlat::stencilOp(rsv[D3DRS_CCW_STENCILFAIL])) : f[0],
                            twoSided ? uint32_t(xlat::stencilOp(rsv[D3DRS_CCW_STENCILPASS])) : f[1],
                            twoSided ? uint32_t(xlat::stencilOp(rsv[D3DRS_CCW_STENCILZFAIL])) : f[2],
                            twoSided ? uint32_t(xlat::compareOp(rsv[D3DRS_CCW_STENCILFUNC])) : f[3] };
    memcpy(r.sopF, f, sizeof(f));
    memcpy(r.sopB, b, sizeof(b));
    r.cmpMask = rsv[D3DRS_STENCILMASK] & 0xFF;
    r.writeMask = rsv[D3DRS_STENCILWRITEMASK] & 0xFF;
    r.ref = rsv[D3DRS_STENCILREF] & 0xFF;
  }
  {
    // D3DRS_DEPTHBIAS is in depth-range units; Vulkan's constant factor is in units of the format's r (DXVK).
    const float bias = xlat::asFloat(rsv[D3DRS_DEPTHBIAS]), slope = xlat::asFloat(rsv[D3DRS_SLOPESCALEDEPTHBIAS]);
    const float scale = groupDepthFormat == VK_FORMAT_D24_UNORM_S8_UINT ? 16777216.0f : 8388608.0f;
    r.biasOn = bias != 0.0f || slope != 0.0f;
    r.bias = bias * scale;
    r.slope = slope;
  }
  xlat::d3dColor(rsv[D3DRS_BLENDFACTOR], r.blend);

  // Uniform blocks (SHADER_ABI struct layouts).
  VkDescriptorBufferInfo ubo[kSets][kUboBindings];
  const VkDescriptorBufferInfo zeroInfo{ zeroBuffer, 0, 4096 };
  for (auto& set : ubo)
    for (auto& b : set) b = zeroInfo;
  if (!constBlock(lastVsBlock, ds.vsConstants, 256, ds.vsConstantsVersion, vsh, ubo[0][0])) return;
  if (!constBlock(lastPsBlock, ds.psConstants, 224, ds.psConstantsVersion, psh, ubo[1][0])) return;
  const StateBlockReuse blocks{ ds.renderStatesVersion, ds.clipPlanesVersion, groupGen, smallGen };
  if (ds.renderStatesVersion && ds.clipPlanesVersion && blocks == lastStateBlocks) {
    // Alpha test, fog, clip planes and point args depend only on the render states, the clip planes and the group.
    ubo[1][3] = lastStateUbo[0];
    ubo[1][4] = lastStateUbo[1];
    ubo[0][5] = lastStateUbo[2];
    ubo[0][6] = ubo[1][6] = lastStateUbo[3];
    if (lastTableFog) st.tableFogDraws++;
    st.stateBlockReuses++;
  } else {
  {
    struct { uint32_t compareOp, precision, ref; } at;
    at.compareOp = rsv[D3DRS_ALPHATESTENABLE] ? uint32_t(xlat::compareOp(rsv[D3DRS_ALPHAFUNC])) : 7u;
    at.precision = at.compareOp == 7 ? 0u
                   : groupRt0Format == kD3DFMT_A16B16G16R16F ? 0x7u
                   : groupRt0Format == kD3DFMT_R32F          ? 0xFu
                                                             : 0u;
    at.ref = rsv[D3DRS_ALPHAREF] & 0xFF;
    if (!smallBlockCached(0, &at, sizeof(at), ubo[1][3])) return;
  }
  {
    struct { uint32_t enable, useZ, mode; float color[3], scale, end, density; } fog;
    const bool enabled = rsv[D3DRS_FOGENABLE] != 0;
    const bool pixelFog = enabled && rsv[D3DRS_FOGTABLEMODE] != D3DFOG_NONE;
    fog.enable = enabled;
    fog.useZ = 1;
    fog.mode = pixelFog ? rsv[D3DRS_FOGTABLEMODE] : 0u;  // 0: the vertex shader's oFog is the factor
    if (pixelFog) st.tableFogDraws++;
    float c[4];
    xlat::d3dColor(rsv[D3DRS_FOGCOLOR], c);
    fog.color[0] = c[0];
    fog.color[1] = c[1];
    fog.color[2] = c[2];
    float fs = xlat::asFloat(rsv[D3DRS_FOGSTART]), fe = xlat::asFloat(rsv[D3DRS_FOGEND]);
    fog.scale = fe != fs ? 1.0f / (fe - fs) : 0.0f;
    fog.end = fe;
    fog.density = xlat::asFloat(rsv[D3DRS_FOGDENSITY]);
    if (!smallBlockCached(1, &fog, sizeof(fog), ubo[1][4])) return;
  }
  {
    struct { uint32_t count; float planes[6][4]; } cp{};
    for (uint32_t i = 0; i < 6; i++)
      if (rsv[D3DRS_CLIPPLANEENABLE] & (1u << i)) memcpy(cp.planes[cp.count++], ds.clipPlanes[i], 16);
    if (!smallBlockCached(2, &cp, sizeof(cp), ubo[0][5])) return;
  }
  {
    struct { uint32_t sprite; float size, min, max; } pa{ rsv[D3DRS_POINTSPRITEENABLE] ? 1u : 0u,
                                                        xlat::asFloat(rsv[D3DRS_POINTSIZE]),
                                                        xlat::asFloat(rsv[D3DRS_POINTSIZE_MIN]),
                                                        xlat::asFloat(rsv[D3DRS_POINTSIZE_MAX]) };
    if (!smallBlockCached(3, &pa, sizeof(pa), ubo[0][6])) return;
    ubo[1][6] = ubo[0][6];
  }
  lastStateUbo[0] = ubo[1][3];
  lastStateUbo[1] = ubo[1][4];
  lastStateUbo[2] = ubo[0][5];
  lastStateUbo[3] = ubo[0][6];
  lastTableFog = rsv[D3DRS_FOGENABLE] && rsv[D3DRS_FOGTABLEMODE] != D3DFOG_NONE;
  lastStateBlocks = blocks;
  }
  if (!smallBlockCached(4, ssBlock, sizeof(ssBlock), ubo[1][7])) return;
  const bool wetLit = (psh->cbvMask & (1u << kWetBlockBinding)) != 0;
  // The sun shadow array: for the sun's cascades (block 9) and the wet world's rain cover (block 12).
  const bool sunLit = (psh->cbvMask & ((1u << kSunBlockBinding) | (1u << kWetBlockBinding))) != 0;
  VkDescriptorBufferInfo wetUbo{};
  VkDescriptorImageInfo puddleImage{ VK_NULL_HANDLE, dummyView[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
  VkDescriptorImageInfo puddleLevels = puddleImage;
  VkSampler puddleSampler = VK_NULL_HANDLE;
  if (wetLit) {
    float blk[kWetBlockVec4s][4];
    VkImageView pv = VK_NULL_HANDLE, lv = VK_NULL_HANDLE;
    wetBlock(blk, pv, lv);
    if (pv) puddleImage = { VK_NULL_HANDLE, pv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    if (lv) puddleLevels = { VK_NULL_HANDLE, lv, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    if (!smallBlockCached(8, blk, sizeof(blk), wetUbo)) return;
    uint32_t pss[14] = {};
    pss[D3DSAMP_ADDRESSU] = pss[D3DSAMP_ADDRESSV] = pss[D3DSAMP_ADDRESSW] = D3DTADDRESS_CLAMP;
    pss[D3DSAMP_MAGFILTER] = pss[D3DSAMP_MINFILTER] = D3DTEXF_POINT;
    pss[D3DSAMP_MIPFILTER] = D3DTEXF_NONE;
    puddleSampler = samplerFor(pss, false);
  }
  if (psh->cbvMask & (1u << kSunBlockBinding)) {
    float blk[kSunBlockVec4s][4] = {};
    if (sunBlock(blk)) st.sunShadowedDraws++;
    if (!smallBlockCached(5, blk, sizeof(blk), ubo[1][kSunBlockBinding])) return;
  }
  VkDescriptorBufferInfo envUbo{};
  if (envLit) {
    float blk[kEnvBlockVec4s][4];
    memcpy(blk, envBlock, sizeof(blk));
    blk[0][0] = envValid ? 1.0f : 0.0f;
    if (!smallBlockCached(6, blk, sizeof(blk), envUbo)) return;
  }
  const VkDescriptorImageInfo envSamplerInfo{ envSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
  const VkDescriptorImageInfo puddleSamplerInfo{ puddleSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
  // The skin buffer: block 13 x 1 while it takes this draw's pixels (active, sized to this target), y z its size.
  const bool skinStore = psh->skinStore;
  VkDescriptorBufferInfo skinUbo{};
  const VkDescriptorImageInfo skinAlbedoInfo{ VK_NULL_HANDLE, skinAlbedoImage.view, VK_IMAGE_LAYOUT_GENERAL };
  const VkDescriptorImageInfo skinDepthInfo{ VK_NULL_HANDLE, skinDepthImage.view, VK_IMAGE_LAYOUT_GENERAL };
  if (skinStore) {
    const bool takes = skinBufferOn && skinAlbedoImage.w == renderW && skinAlbedoImage.h == renderH;
    skinStoredSinceOn |= takes;
    const float blk[4] = { takes ? 1.0f : 0.0f, float(skinAlbedoImage.w), float(skinAlbedoImage.h), 0.0f };
    if (!smallBlockCached(9, blk, sizeof(blk), skinUbo)) return;
  }
  // The skin's detail normal map: tiled (wrap), trilinear, anisotropic as the game's own maps; without one, the dummy
  // (the shader reads it as flat).
  const bool skinLit = psh->skinDetail;
  VkDescriptorImageInfo skinImage{ VK_NULL_HANDLE, dummyView[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
  VkDescriptorImageInfo skinSamplerInfo{};
  if (skinLit) {
    auto detail = skinDetail ? textures.find(skinDetail) : textures.end();
    if (detail != textures.end() && detail->second.view) skinImage.imageView = detail->second.view;
    uint32_t kss[14] = {};
    kss[D3DSAMP_ADDRESSU] = kss[D3DSAMP_ADDRESSV] = kss[D3DSAMP_ADDRESSW] = D3DTADDRESS_WRAP;
    kss[D3DSAMP_MAGFILTER] = kss[D3DSAMP_MINFILTER] = kss[D3DSAMP_MIPFILTER] = D3DTEXF_LINEAR;
    skinSamplerInfo = { samplerFor(kss, false), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
  }
  // The hair's tangent shift map: tiled (wrap), linear; without one, the dummy (the shader reads it as no shift).
  const bool hairLit = psh->hairShift;
  VkDescriptorImageInfo hairImage{ VK_NULL_HANDLE, dummyView[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
  VkDescriptorImageInfo hairSamplerInfo{};
  if (hairLit) {
    auto shift = hairShift ? textures.find(hairShift) : textures.end();
    if (shift != textures.end() && shift->second.view) hairImage.imageView = shift->second.view;
    uint32_t kss[14] = {};
    kss[D3DSAMP_ADDRESSU] = kss[D3DSAMP_ADDRESSV] = kss[D3DSAMP_ADDRESSW] = D3DTADDRESS_WRAP;
    kss[D3DSAMP_MAGFILTER] = kss[D3DSAMP_MINFILTER] = kss[D3DSAMP_MIPFILTER] = D3DTEXF_LINEAR;
    hairSamplerInfo = { samplerFor(kss, false), VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
  }
  const bool aoLit = (psh->cbvMask & (1u << kAoBlockBinding)) != 0;
  VkDescriptorBufferInfo aoUbo{};
  VkDescriptorImageInfo aoImage{ VK_NULL_HANDLE, dummyView[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
  VkDescriptorImageInfo giImage = aoImage, giDepthImage = aoImage;
  if (aoLit) {
    float blk[12][4];
    VkImageView av = VK_NULL_HANDLE, gv = VK_NULL_HANDLE, gdv = VK_NULL_HANDLE;
    if (aoBlock(blk, av, gv, gdv)) {
      if (av) aoImage = { VK_NULL_HANDLE, av, VK_IMAGE_LAYOUT_GENERAL };
      if (gv) giImage = { VK_NULL_HANDLE, gv, VK_IMAGE_LAYOUT_GENERAL };
      if (gdv) giDepthImage = { VK_NULL_HANDLE, gdv, VK_IMAGE_LAYOUT_GENERAL };
    }
    if (!smallBlockCached(7, blk, sizeof(blk), aoUbo)) return;
  }
  const VkDescriptorImageInfo aoSamplerInfo{ envSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };
  const VkDescriptorImageInfo sunImage{ VK_NULL_HANDLE, shadowArrayView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
  const VkDescriptorImageInfo sunSampler{ shadowCompareSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED };

  // Depth pre-pass eligibility (see "depth pre-pass" above).
  const VkCompareOp zf = VkCompareOp(r.depthOp);
  const bool alphaTest = rsv[D3DRS_ALPHATESTENABLE] && rsv[D3DRS_ALPHAFUNC] != D3DCMP_ALWAYS;
  const bool depthDecides = r.depthTest && r.depthWrite && (zf == VK_COMPARE_OP_LESS || zf == VK_COMPARE_OP_LESS_OR_EQUAL) &&
                            !rsv[D3DRS_ALPHABLENDENABLE] && !stencil && rsv[D3DRS_FILLMODE] == D3DFILL_SOLID &&
                            !psh->writesDepth && vsh->invariantPosition &&
                            xlat::topology(pt) >= VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST &&
                            xlat::topology(pt) <= VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
  const bool mayDiscard = alphaTest || psh->discards;
  r.prepass = depthDecides && !mayDiscard;
  if (depthDecides && mayDiscard && cfg.depthPrepassDiscarding && prepassHolds()) {
    r.depthModule = depthModuleFor(psh);
    r.colourModule = r.depthModule ? colourModuleFor(psh) : VK_NULL_HANDLE;
    r.canDiscard = r.prepass = r.depthModule && r.colourModule;
    if (!r.canDiscard) st.prepassDiscardingRefused++;
  }

  VkDescriptorSet sets[kSets];
  VkWriteDescriptorSet writes[kSets * (kUboBindings + 2 * kTexBindings) + 21];
  uint32_t nw = 0;
  // The constant blocks as written into the sets: offset 0, the slice's offset is the dynamic offset at bind time.
  const VkDescriptorBufferInfo constInfo[kSets] = { { ubo[0][0].buffer, 0, ubo[0][0].range },
                                                    { ubo[1][0].buffer, 0, ubo[1][0].range } };
  for (uint32_t set = 0; set < kSets; set++) {
    const PackedShader* sh = stageSh[set];
    // The set's whole content: which bindings the shader declares and what each one points at. A set identical to the
    // previous draw's for this stage is reused as it is.
    uint64_t sig[112];
    uint32_t n = 0;
    sig[n++] = uint64_t(reinterpret_cast<uintptr_t>(sh));
    // Binding 0: always written (its dynamic offset is bound for every set); its buffer and range only, not its offset.
    sig[n++] = uint64_t(ubo[set][0].buffer);
    sig[n++] = ubo[set][0].range;
    for (uint32_t b = 1; b < kUboBindings; b++)
      if (sh->cbvMask & (1u << b)) {
        sig[n++] = uint64_t(ubo[set][b].buffer);
        sig[n++] = ubo[set][b].offset;
        sig[n++] = ubo[set][b].range;
      }
    for (const PackedShader::Tex& t : sh->textures) {
      const uint32_t s = t.reg & 15;
      if (n + 3 > 112) break;
      sig[n++] = uint64_t(img[set][s].imageView);
      sig[n++] = uint64_t(img[set][s].imageLayout) | (uint64_t(s) << 32);
      sig[n++] = uint64_t(smp[set][s].sampler);
    }
    if (set == 1 && sunLit) sig[n++] = uint64_t(shadowArrayView);
    if (set == 1 && aoLit) {
      sig[n++] = uint64_t(aoUbo.buffer) ^ (uint64_t(aoUbo.offset) << 40);
      sig[n++] = uint64_t(aoImage.imageView);
      sig[n++] = uint64_t(giImage.imageView);
      sig[n++] = uint64_t(giDepthImage.imageView);
    }
    if (set == 1 && wetLit) {
      sig[n++] = uint64_t(wetUbo.buffer) ^ (uint64_t(wetUbo.offset) << 40);
      sig[n++] = uint64_t(puddleImage.imageView);
      sig[n++] = uint64_t(puddleLevels.imageView);
    }
    if (set == 1 && skinLit) sig[n++] = uint64_t(skinImage.imageView);
    if (set == 1 && hairLit) sig[n++] = uint64_t(hairImage.imageView);
    if (set == 1 && skinStore) {
      sig[n++] = uint64_t(skinUbo.buffer) ^ (uint64_t(skinUbo.offset) << 40);
      sig[n++] = uint64_t(skinAlbedoInfo.imageView);
      sig[n++] = uint64_t(skinDepthInfo.imageView);
    }
    if (set == 1 && envLit) {
      sig[n++] = uint64_t(envUbo.buffer);
      sig[n++] = envUbo.offset;
      sig[n++] = uint64_t(envImage.imageView);
    }
    if (lastSet[set] && n == lastSigCount[set] && memcmp(sig, lastSig[set], n * sizeof(uint64_t)) == 0) {
      sets[set] = lastSet[set];
      continue;
    }
    // An earlier set of this frame with the same content.
    DrawSlot& slotSets = drawSlot();
    uint64_t h = 0x9E3779B97F4A7C15ull ^ set;
    for (uint32_t i = 0; i < n; i++) h = (h ^ sig[i]) * 0x100000001B3ull + (h >> 29);
    auto cached = slotSets.setCache.find(h);
    if (cached != slotSets.setCache.end() && cached->second.stage == set && cached->second.count == n &&
        memcmp(&slotSets.sigStore[cached->second.at], sig, n * sizeof(uint64_t)) == 0) {
      sets[set] = cached->second.set;
      lastSet[set] = sets[set];
      lastSigCount[set] = n;
      memcpy(lastSig[set], sig, n * sizeof(uint64_t));
      st.descriptorSetCacheHits++;
      continue;
    }
    sets[set] = allocSet(set);
    if (!sets[set]) return;
    slotSets.setCache[h] = { sets[set], set, n, slotSets.sigStore.size() };
    slotSets.sigStore.insert(slotSets.sigStore.end(), sig, sig + n);
    st.descriptorSetWrites++;
    lastSet[set] = sets[set];
    lastSigCount[set] = n;
    memcpy(lastSig[set], sig, n * sizeof(uint64_t));
    for (uint32_t b = 0; b < kUboBindings; b++) {
      if (b != 0 && !(sh->cbvMask & (1u << b))) continue;
      VkWriteDescriptorSet& w = writes[nw++];
      w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      w.dstSet = sets[set];
      w.dstBinding = b;
      w.descriptorCount = 1;
      w.descriptorType = b == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      w.pBufferInfo = b == 0 ? &constInfo[set] : &ubo[set][b];
    }
    for (const PackedShader::Tex& t : sh->textures) {
      const uint32_t s = t.reg & 15;
      VkWriteDescriptorSet& wi = writes[nw++];
      wi = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      wi.dstSet = sets[set];
      wi.dstBinding = 16 + s;
      wi.descriptorCount = 1;
      wi.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      wi.pImageInfo = &img[set][s];
      VkWriteDescriptorSet& ws = writes[nw++];
      ws = wi;
      ws.dstBinding = 32 + s;
      ws.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
      ws.pImageInfo = &smp[set][s];
    }
    if (set == 1 && sunLit) {
      VkWriteDescriptorSet& wi = writes[nw++];
      wi = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      wi.dstSet = sets[set];
      wi.dstBinding = kSunImageBinding;
      wi.descriptorCount = 1;
      wi.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      wi.pImageInfo = &sunImage;
      VkWriteDescriptorSet& ws = writes[nw++];
      ws = wi;
      ws.dstBinding = kSunSamplerBinding;
      ws.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
      ws.pImageInfo = &sunSampler;
    }
    if (set == 1 && wetLit) {
      VkWriteDescriptorSet& wb = writes[nw++];
      wb = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      wb.dstSet = sets[set];
      wb.dstBinding = kWetBlockBinding;
      wb.descriptorCount = 1;
      wb.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      wb.pBufferInfo = &wetUbo;
      VkWriteDescriptorSet& wi = writes[nw++];
      wi = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      wi.dstSet = sets[set];
      wi.dstBinding = kPuddleImageBinding;
      wi.descriptorCount = 1;
      wi.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      wi.pImageInfo = &puddleImage;
      VkWriteDescriptorSet& ws = writes[nw++];
      ws = wi;
      ws.dstBinding = kPuddleSamplerBinding;
      ws.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
      ws.pImageInfo = &puddleSamplerInfo;
      VkWriteDescriptorSet& wl = writes[nw++];
      wl = wi;
      wl.dstBinding = kPuddleLevelsBinding;
      wl.pImageInfo = &puddleLevels;
    }
    if (set == 1 && skinLit) {
      VkWriteDescriptorSet& wi = writes[nw++];
      wi = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      wi.dstSet = sets[set];
      wi.dstBinding = kSkinDetailBinding;
      wi.descriptorCount = 1;
      wi.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      wi.pImageInfo = &skinImage;
      VkWriteDescriptorSet& ws = writes[nw++];
      ws = wi;
      ws.dstBinding = kSkinDetailSamplerBinding;
      ws.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
      ws.pImageInfo = &skinSamplerInfo;
    }
    if (set == 1 && hairLit) {
      VkWriteDescriptorSet& wi = writes[nw++];
      wi = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      wi.dstSet = sets[set];
      wi.dstBinding = kHairShiftBinding;
      wi.descriptorCount = 1;
      wi.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      wi.pImageInfo = &hairImage;
      VkWriteDescriptorSet& ws = writes[nw++];
      ws = wi;
      ws.dstBinding = kHairShiftSamplerBinding;
      ws.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
      ws.pImageInfo = &hairSamplerInfo;
    }
    if (set == 1 && skinStore) {
      VkWriteDescriptorSet& wb = writes[nw++];
      wb = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      wb.dstSet = sets[set];
      wb.dstBinding = kSkinBlockBinding;
      wb.descriptorCount = 1;
      wb.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      wb.pBufferInfo = &skinUbo;
      VkWriteDescriptorSet& wa = writes[nw++];
      wa = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      wa.dstSet = sets[set];
      wa.dstBinding = kSkinAlbedoBinding;
      wa.descriptorCount = 1;
      wa.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      wa.pImageInfo = &skinAlbedoInfo;
      VkWriteDescriptorSet& wd = writes[nw++];
      wd = wa;
      wd.dstBinding = kSkinDepthBinding;
      wd.pImageInfo = &skinDepthInfo;
    }
    if (set == 1 && envLit) {
      VkWriteDescriptorSet& wb = writes[nw++];
      wb = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      wb.dstSet = sets[set];
      wb.dstBinding = kEnvBlockBinding;
      wb.descriptorCount = 1;
      wb.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      wb.pBufferInfo = &envUbo;
      VkWriteDescriptorSet& wi = writes[nw++];
      wi = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      wi.dstSet = sets[set];
      wi.dstBinding = kEnvImageBinding;
      wi.descriptorCount = 1;
      wi.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      wi.pImageInfo = &envImage;
      VkWriteDescriptorSet& ws = writes[nw++];
      ws = wi;
      ws.dstBinding = kEnvSamplerBinding;
      ws.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
      ws.pImageInfo = &envSamplerInfo;
    }
    if (set == 1 && aoLit) {
      VkWriteDescriptorSet& wb = writes[nw++];
      wb = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      wb.dstSet = sets[set];
      wb.dstBinding = kAoBlockBinding;
      wb.descriptorCount = 1;
      wb.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      wb.pBufferInfo = &aoUbo;
      VkWriteDescriptorSet& wi = writes[nw++];
      wi = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      wi.dstSet = sets[set];
      wi.dstBinding = kAoImageBinding;
      wi.descriptorCount = 1;
      wi.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      wi.pImageInfo = &aoImage;
      VkWriteDescriptorSet& ws = writes[nw++];
      ws = wi;
      ws.dstBinding = kAoSamplerBinding;
      ws.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
      ws.pImageInfo = &aoSamplerInfo;
      VkWriteDescriptorSet& wg = writes[nw++];
      wg = wi;
      wg.dstBinding = kGiImageBinding;
      wg.pImageInfo = &giImage;
      VkWriteDescriptorSet& wd = writes[nw++];
      wd = wi;
      wd.dstBinding = kGiDepthBinding;
      wd.pImageInfo = &giDepthImage;
    }
  }
  if (nw) vk.vkUpdateDescriptorSets(device, nw, writes, 0, nullptr);
  const uint32_t dynOffsets[kSets] = { uint32_t(ubo[0][0].offset), uint32_t(ubo[1][0].offset) };
  r.sets[0] = sets[0];
  r.sets[1] = sets[1];
  r.dynOffsets[0] = dynOffsets[0];
  r.dynOffsets[1] = dynOffsets[1];

  for (uint32_t b = 0; b < 16; b++) {
    if (!(attrCache.bindingMask & (1u << b))) continue;
    VkBuffer vb = zeroBuffer;
    VkDeviceSize off = 0;
    if (b != kZeroBinding) {
      if (lookups.bufKey[b] != ds.streams[b].buffer || !lookups.buf[b]) {
        auto it = buffers.find(ds.streams[b].buffer);
        lookups.bufKey[b] = ds.streams[b].buffer;
        lookups.buf[b] = it == buffers.end() ? nullptr : &it->second;
      }
      Buf* bp = lookups.buf[b];
      if (!bp) {
        st.unknownBuffers++;
        st.drawsSkipped++;
        return;
      }
      bp->lastUse = frameNumber;
      vb = bp->cur.buffer;
      off = ds.streams[b].offset;
    }
    r.vb[b] = vb;
    r.vbOffset[b] = off;
  }
  r.bindingMask = attrCache.bindingMask;
  r.count = xlat::vertexCount(pt, prims);
  r.start = start;
  r.baseVertex = baseVertex;
  r.indexed = indexed;
  if (indexed) {
    if (lookups.ibKey != ds.indices || !lookups.ib) {
      auto it = buffers.find(ds.indices);
      lookups.ibKey = ds.indices;
      lookups.ib = it == buffers.end() ? nullptr : &it->second;
    }
    Buf* ib = lookups.ib;
    if (!ib || !ib->index) {
      st.unknownBuffers++;
      st.drawsSkipped++;
      return;
    }
    ib->lastUse = frameNumber;
    r.ib = ib->cur.buffer;
    r.indexType = ib->index32 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16;
  }
  recordDraw(r, vsh);
  st.draws++;
}
