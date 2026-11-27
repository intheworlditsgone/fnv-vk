#include "builtin_spv.h"
#include "backend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <deque>
#include <future>
#include <mutex>
#include <set>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <xmmintrin.h>

#include "d3dxtex/texture.h"
#include "shaderpack.h"
#include "translate.h"
#include "vma.h"
// After vkfn.h (via vma.h / translate.h): Vulkan is included with the Win32 platform and without prototypes.
#include "effectlib.h"
#include "imgui.h"
#include "imgui_impl_vulkan.h"
// FSR 3.1 (backend_ffx.inl): the FidelityFX API's headers as patched for this build (cmake/fidelityfx.cmake).
#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_api_loader.h>
#include <ffx_api/ffx_upscale.h>
#include <ffx_api/vk/ffx_api_vk.h>

namespace fnvvk {

namespace {

constexpr uint32_t kFramesInFlight = 2;
constexpr uint32_t kMaxLoggedValidation = 64;

const char* vkResultName(VkResult r) {
  switch (r) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_SUBOPTIMAL_KHR: return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_OUT_OF_DATE_KHR: return "VK_ERROR_OUT_OF_DATE_KHR";
    case VK_ERROR_SURFACE_LOST_KHR: return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    default: return "VkResult(other)";
  }
}

bool hasDepth(VkFormat f) {
  return f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT || f == VK_FORMAT_D32_SFLOAT ||
         f == VK_FORMAT_D16_UNORM;
}
bool hasStencil(VkFormat f) { return f == VK_FORMAT_D24_UNORM_S8_UINT || f == VK_FORMAT_D32_SFLOAT_S8_UINT; }

// IEEE half <-> float (wide targets are read and written by the CPU in their D3D9 8-bit layout).
float halfToFloat(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000) << 16, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
  uint32_t bits;
  if (exp == 0) {
    if (!man) {
      bits = sign;
    } else {  // subnormal
      int e = -1;
      uint32_t m = man;
      do { e++; m <<= 1; } while (!(m & 0x400));
      bits = sign | uint32_t(127 - 15 - e) << 23 | (m & 0x3FF) << 13;
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | man << 13;
  } else {
    bits = sign | (exp + 127 - 15) << 23 | man << 13;
  }
  float f;
  memcpy(&f, &bits, 4);
  return f;
}
uint16_t floatToHalf(float f) {  // inputs here are 0..1 (8-bit unorm values): normal range, round to nearest
  uint32_t bits;
  memcpy(&bits, &f, 4);
  const uint32_t sign = (bits >> 16) & 0x8000;
  const int exp = int((bits >> 23) & 0xFF) - 127 + 15;
  if (exp <= 0) return uint16_t(sign);  // below half's normal range (the smallest unorm8 step, 1/255, is far above it)
  if (exp >= 31) return uint16_t(sign | 0x7C00);
  uint32_t h = sign | uint32_t(exp) << 10 | ((bits >> 13) & 0x3FF);
  if (bits & 0x1000) h++;  // round on the dropped bits
  return uint16_t(h);
}

uint32_t texelSizeOf(VkFormat f) {
  switch (f) {
    case VK_FORMAT_B8G8R8A8_UNORM: return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return 8;
    case VK_FORMAT_R32_SFLOAT: return 4;
    case VK_FORMAT_R5G6B5_UNORM_PACK16: return 2;
    case VK_FORMAT_R16G16_SFLOAT: return 4;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    case VK_FORMAT_D24_UNORM_S8_UINT: return 4;  // depth aspect copy: 24-bit value in a 32-bit texel
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return 4; // depth aspect copy: 32-bit float
    default: return 0;
  }
}

VkSampleCountFlagBits sampleBits(uint32_t samples) {
  switch (samples) {
    case 2: return VK_SAMPLE_COUNT_2_BIT;
    case 4: return VK_SAMPLE_COUNT_4_BIT;
    case 8: return VK_SAMPLE_COUNT_8_BIT;
    default: return VK_SAMPLE_COUNT_1_BIT;
  }
}

}  // namespace

// The game's main thread runs with the x87 control word D3D9 leaves (24-bit precision: the device is created without
// FPU_PRESERVE). Driver code, notably the shader compiler behind vkCreateGraphicsPipelines, expects the default
// environment and crashes under the game's. Every backend entry point runs under the default control word and
// restores the caller's.
// The caller's x87 control word and MXCSR at the last guarded entry (diagnostics).
uint16_t g_entryX87 = 0;
uint32_t g_entryMxcsr = 0;
struct FpuGuard {
#if defined(_M_IX86)
  uint16_t x87 = 0;
  uint32_t mxcsr = 0;
  FpuGuard() {
    uint16_t cw = 0;
    __asm fnstcw cw;
    x87 = cw;
    mxcsr = _mm_getcsr();
    g_entryX87 = x87;
    g_entryMxcsr = mxcsr;
    uint16_t def = 0x027F;  // MSVC default: all exceptions masked, 53-bit precision, round to nearest
    __asm fldcw def;
    _mm_setcsr(0x1F80);     // default: all SSE exceptions masked, round to nearest, no FTZ/DAZ
  }
  ~FpuGuard() {
    uint16_t v = x87;
    __asm fnclex;
    __asm fldcw v;
    _mm_setcsr(mxcsr);
  }
#else
  FpuGuard() {}  // x64 (the test harness) has no x87 state to guard
#endif
};

struct Backend::Impl {
  // --- lifetime ------------------------------------------------------------------------------------------------
  struct Rt {
    RtDesc desc;
    bool realized = false;
    VkFormat format = VK_FORMAT_UNDEFINED;
    // Single-sample image: the color target (resolve destination when MSAA), or the depth image.
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    // Multisampled color attachment (D3D9 m_pkSurface2); depth MSAA lives in `image` itself.
    VkImage msImage = VK_NULL_HANDLE;
    VkDeviceMemory msMemory = VK_NULL_HANDLE;
    VkImageView msView = VK_NULL_HANDLE;
    VkImageLayout msLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // View for sampling `image` as a texture: depth aspect only for depth targets, alpha forced to 1 for X8.
    VkImageView sampleView = VK_NULL_HANDLE;
    // Multisampled depth only: a depth-aspect view of `image` for the backend's own passes (ambient occlusion reads
    // sample 0). Never bound for game draws (they use sampleView, which multisampled depth does not have).
    VkImageView depthReadView = VK_NULL_HANDLE;
    bool wide = false;  // an 8-bit D3D9 target created as RGBA16F (BackendConfig::wideTargets)
    // A render-size twin of a window-sized target (backend_ffx.inl): takes that target's wide decision (twinWide), as
    // its own size would not be recognised as the window's.
    bool twin = false, twinWide = false;
    bool storage = false;  // also a storage image (window-sized RGBA16F colour: FSR writes its output into it)
  };
  struct Garbage {
    uint64_t frame;  // free once this frame's submission has completed
    VkImage images[2];
    VkDeviceMemory memories[2];
    VkImageView views[4];
  };

  // One image per Gamebryo texture renderer-data object, sub-allocated with VMA.
  struct Tex {
    VkImage image = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0, levels = 0, layers = 0;
    uint64_t bytes = 0;
  };
  struct VmaGarbage {
    uint64_t frame;
    VkImage image;
    VkImageView view;
    VkBuffer buffer;
    VmaAllocation alloc;
    uint64_t bytes = 0;  // the allocation's size (MemoryReport::garbageBytes)
  };
  // Memory held for the GPU until a frame completes, and the per-frame host rings' size (MemoryReport; any thread).
  std::atomic<uint64_t> garbageBytes{ 0 }, hostRingBytes{ 0 };
  void addGarbage(VmaGarbage g) {
    if (g.alloc) {
      VmaAllocationInfo info{};
      vmaGetAllocationInfo(allocator, g.alloc, &info);
      g.bytes = info.size;
      garbageBytes.fetch_add(g.bytes, std::memory_order_relaxed);
    }
    vmaGarbage.push_back(g);
  }

  mutable std::mutex mutex;
  BackendConfig cfg;
  std::unordered_map<RtKey, Rt> rts;
  std::deque<Garbage> garbage;
  std::unordered_map<TexKey, Tex> textures;
  // Rendered cube maps (attachCube): six face targets, presented to draws as the cube texture textures[cube].
  struct RenderedCube {
    RtKey faces[6]{};
    uint32_t levels = 1;  // requested; 0 = the full chain
    uint8_t dirty = 0x3F;  // faces whose contents the cube does not hold yet
  };
  std::unordered_map<TexKey, RenderedCube> cubes;
  std::unordered_map<RtKey, std::pair<TexKey, uint32_t>> cubeFaces;  // face target -> (cube, face index)
  std::deque<VmaGarbage> vmaGarbage;
  VmaAllocator allocator = nullptr;
  BackendStats st;
  std::set<std::string> loggedOnce;

  // --- Vulkan --------------------------------------------------------------------------------------------------
  VkFn vk;
  VkInstance instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
  VkPhysicalDevice phys = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties memProps{};
  VkDevice device = VK_NULL_HANDLE;
  uint32_t queueFamily = UINT32_MAX;
  VkQueue queue = VK_NULL_HANDLE;
  VkFormat depthFormat = VK_FORMAT_UNDEFINED;
  bool initialized = false;

  struct FrameSlot {
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBuffer upload = VK_NULL_HANDLE;  // texture uploads; submitted ahead of `cmd`
    bool uploadOpen = false;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore acquired = VK_NULL_HANDLE;
    uint64_t submittedFrame = 0;  // frame number whose work this slot's fence guards
  } slots[kFramesInFlight];
  uint64_t frameNumber = 1;       // frame currently being recorded
  uint64_t completedFrame = 0;    // all frames <= this have finished on the GPU
  bool frameOpen = false;

  bool rendering = false;
  std::vector<RtKey> groupColors;
  RtKey groupDepth = 0;
  uint32_t renderW = 0, renderH = 0;

  // --- shadow window -------------------------------------------------------------------------------------------
  std::thread windowThread;
  HWND hwnd = nullptr;
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  VkSwapchainKHR swapchain = VK_NULL_HANDLE;
  VkFormat swapFormat = VK_FORMAT_UNDEFINED;
  VkExtent2D swapExtent{};
  std::vector<VkImage> swapImages;
  std::vector<VkImageView> swapViews;  // when the swapchain supports colour attachment (gamma pass)
  std::vector<VkSemaphore> renderDone;  // one per swapchain image
  bool swapchainDirty = false;
  // HDR output (backend.h HdrSettings). hdrActive is what the current swapchain uses.
  HdrSettings hdr;
  uint32_t hdrActive = HdrSettings::Off;
  bool hdr10Available = false, scRgbAvailable = false, colorspaceExt = false;
  uint32_t screenW = 0, screenH = 0;  // window size at init: the size of the targets that are widened

  void logf(const char* fmt, ...) {
    if (!cfg.log) return;
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    cfg.log(buf);
  }
  void logOnce(const std::string& key, const char* fmt, ...) {
    if (!loggedOnce.insert(key).second || !cfg.log) return;
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    cfg.log(buf);
  }
  bool check(VkResult r, const char* what) {
    if (r == VK_SUCCESS) return true;
    st.vkErrors++;
    logf("vk error: %s -> %s (%d)", what, vkResultName(r), int(r));
    return false;
  }

  static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                      VkDebugUtilsMessageTypeFlagsEXT,
                                                      const VkDebugUtilsMessengerCallbackDataEXT* data, void* user) {
    auto* self = static_cast<Impl*>(user);
    if (severity < VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) return VK_FALSE;
    uint64_t n = ++self->st.validationMessages;
    if (n <= kMaxLoggedValidation)
      self->logf("validation[%llu]: %s", static_cast<unsigned long long>(n), data->pMessage ? data->pMessage : "");
    return VK_FALSE;
  }

  // --- formats -------------------------------------------------------------------------------------------------
  VkFormat vkFormatFor(const RtDesc& d) {
    switch (d.d3dFormat) {
      case kD3DFMT_A8R8G8B8:
      case kD3DFMT_X8R8G8B8: return VK_FORMAT_B8G8R8A8_UNORM;  // X8: alpha forced to 1 when sampled (later slices)
      case kD3DFMT_A16B16G16R16F: return VK_FORMAT_R16G16B16A16_SFLOAT;
      case kD3DFMT_R32F: return VK_FORMAT_R32_SFLOAT;
      case kD3DFMT_R5G6B5: return VK_FORMAT_R5G6B5_UNORM_PACK16;   // same bit layout (R in 15:11)
      case kD3DFMT_G16R16F: return VK_FORMAT_R16G16_SFLOAT;        // R in the low half, as D3D
      case kD3DFMT_A32B32G32R32F: return VK_FORMAT_R32G32B32A32_SFLOAT;
      case kD3DFMT_D24S8:
      case kD3DFMT_D24X8: return depthFormat;
      default: return VK_FORMAT_UNDEFINED;
    }
  }

  bool formatSupports(VkFormat f, VkFormatFeatureFlags need) {
    VkFormatProperties p{};
    vk.vkGetPhysicalDeviceFormatProperties(phys, f, &p);
    return (p.optimalTilingFeatures & need) == need;
  }

  uint32_t memoryType(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
  }

  bool createImage(VkFormat format, uint32_t w, uint32_t h, VkSampleCountFlagBits samples, VkImageUsageFlags usage,
                   VkImageAspectFlags aspect, VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = { w, h, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = samples;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!check(vk.vkCreateImage(device, &ici, nullptr, &img), "vkCreateImage")) return false;
    VkMemoryRequirements req{};
    vk.vkGetImageMemoryRequirements(device, img, &req);
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // Render targets are few (tens); one dedicated allocation each is well inside maxMemoryAllocationCount.
    if (mai.memoryTypeIndex == UINT32_MAX || !check(vk.vkAllocateMemory(device, &mai, nullptr, &mem), "vkAllocateMemory") ||
        !check(vk.vkBindImageMemory(device, img, mem, 0), "vkBindImageMemory")) {
      vk.vkDestroyImage(device, img, nullptr);
      img = VK_NULL_HANDLE;
      return false;
    }
    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = img;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = { aspect, 0, 1, 0, 1 };
    return check(vk.vkCreateImageView(device, &vci, nullptr, &view), "vkCreateImageView");
  }

  // BackendConfig::wideTargets: the 8-bit colour targets of the window size (and exactly a half or a quarter of it) and
  // the X8R8G8B8 back buffer. They hold the image after the game's HDR exposure (ISHDRBLENDINSHADERCIN: exposure, bloom,
  // cinematic, no tone curve), which 8 bits clip at 1.0.
  bool widen(const RtDesc& d) const {
    if (!cfg.wideTargets || d.depth) return false;
    if (d.d3dFormat == kD3DFMT_X8R8G8B8) return true;
    if (d.d3dFormat != kD3DFMT_A8R8G8B8 || !screenW || !screenH) return false;
    for (uint32_t div : { 1u, 2u, 4u })
      if (d.width == screenW / div && d.height == screenH / div) return true;
    return false;
  }

  bool realize(RtKey key, Rt& rt) {
    if (rt.realized) return true;
    rt.wide = rt.twin ? rt.twinWide : widen(rt.desc);
    rt.format = rt.wide ? VK_FORMAT_R16G16B16A16_SFLOAT : vkFormatFor(rt.desc);
    if (rt.format == VK_FORMAT_UNDEFINED) {
      st.unsupportedFormats++;
      logOnce("fmt" + std::to_string(rt.desc.d3dFormat), "unsupported render-target D3DFORMAT %u (key %p)",
              rt.desc.d3dFormat, reinterpret_cast<void*>(key));
      return false;
    }
    if (rt.desc.width == 0 || rt.desc.height == 0) return false;
    VkSampleCountFlagBits samples = sampleBits(rt.desc.samples);
    bool ok;
    if (rt.desc.depth) {
      VkImageAspectFlags aspect = VK_IMAGE_ASPECT_DEPTH_BIT | (hasStencil(rt.format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
      VkImageUsageFlags usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      VkPhysicalDeviceProperties pp{};
      vk.vkGetPhysicalDeviceProperties(phys, &pp);
      // Multisampled depth is sampleable too where the device allows it (ambient occlusion reads it).
      const bool msSampled = samples != VK_SAMPLE_COUNT_1_BIT && (pp.limits.sampledImageDepthSampleCounts & samples);
      if (samples == VK_SAMPLE_COUNT_1_BIT || msSampled) usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
      ok = createImage(rt.format, rt.desc.width, rt.desc.height, samples, usage, aspect, rt.image, rt.memory, rt.view);
      if (ok && msSampled) {
        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image = rt.image;
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = rt.format;
        vci.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
        ok = check(vk.vkCreateImageView(device, &vci, nullptr, &rt.depthReadView), "vkCreateImageView(depth read)");
      }
    } else {
      VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      // Window-sized RGBA16F colour can be FSR's output (backend_ffx.inl): a storage image too.
      rt.storage = !rt.twin && rt.format == VK_FORMAT_R16G16B16A16_SFLOAT && rt.desc.width == screenW &&
                   rt.desc.height == screenH && formatSupports(rt.format, VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT);
      if (rt.storage) usage |= VK_IMAGE_USAGE_STORAGE_BIT;
      ok = createImage(rt.format, rt.desc.width, rt.desc.height, VK_SAMPLE_COUNT_1_BIT, usage, VK_IMAGE_ASPECT_COLOR_BIT,
                       rt.image, rt.memory, rt.view);
      if (ok && samples != VK_SAMPLE_COUNT_1_BIT)
        ok = createImage(rt.format, rt.desc.width, rt.desc.height, samples,
                         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT,
                         rt.msImage, rt.msMemory, rt.msView);
    }
    rt.layout = rt.msLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (ok && (rt.desc.samples <= 1 || !rt.desc.depth)) {
      VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
      vci.image = rt.image;
      vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
      vci.format = rt.format;
      if (rt.desc.d3dFormat == kD3DFMT_X8R8G8B8) vci.components.a = VK_COMPONENT_SWIZZLE_ONE;
      vci.subresourceRange = { VkImageAspectFlags(rt.desc.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT),
                               0, 1, 0, 1 };
      ok = check(vk.vkCreateImageView(device, &vci, nullptr, &rt.sampleView), "vkCreateImageView(sample)");
    }
    if (!ok) {
      release(rt, false);
      return false;
    }
    rt.realized = true;
    st.realized++;
    return true;
  }

  // Queues the Vulkan objects of `rt` for destruction once the GPU can no longer be using them.
  void release(Rt& rt, bool deferred) {
    Garbage g{ frameNumber, { rt.image, rt.msImage }, { rt.memory, rt.msMemory },
               { rt.view, rt.msView, rt.sampleView, rt.depthReadView } };
    const bool twin = rt.twin, twinWide = rt.twinWide;
    rt = Rt{ rt.desc };
    rt.twin = twin;
    rt.twinWide = twinWide;
    if (!device) return;
    if (deferred) garbage.push_back(g);
    else freeGarbage(g);
  }
  void freeGarbage(const Garbage& g) {
    if (g.views[2]) vk.vkDestroyImageView(device, g.views[2], nullptr);
    if (g.views[3]) vk.vkDestroyImageView(device, g.views[3], nullptr);
    for (int i = 0; i < 2; i++) {
      if (g.views[i]) vk.vkDestroyImageView(device, g.views[i], nullptr);
      if (g.images[i]) vk.vkDestroyImage(device, g.images[i], nullptr);
      if (g.memories[i]) vk.vkFreeMemory(device, g.memories[i], nullptr);
    }
  }
  void freeVmaGarbage(const VmaGarbage& g) {
    garbageBytes.fetch_sub(g.bytes, std::memory_order_relaxed);
    if (g.view) vk.vkDestroyImageView(device, g.view, nullptr);
    if (g.image) vmaDestroyImage(allocator, g.image, g.alloc);
    else if (g.buffer) vmaDestroyBuffer(allocator, g.buffer, g.alloc);
  }
  void collectGarbage() {
    collectFfxGarbage(false);  // FidelityFX contexts retired by completed frames (backend_ffx.inl)
    while (!garbage.empty() && garbage.front().frame <= completedFrame) {
      freeGarbage(garbage.front());
      garbage.pop_front();
    }
    while (!vmaGarbage.empty() && vmaGarbage.front().frame <= completedFrame) {
      freeVmaGarbage(vmaGarbage.front());
      vmaGarbage.pop_front();
    }
  }

  // --- frames --------------------------------------------------------------------------------------------------
  // The frame's command buffer for recording: draws held for the depth pre-pass are recorded first (anything else
  // recorded comes after them, in call order). slotRaw: without that (the pre-pass's own recording).
  FrameSlot& slot() {
    flushPrepass();
    return slots[frameNumber % kFramesInFlight];
  }
  FrameSlot& slotRaw() { return slots[frameNumber % kFramesInFlight]; }

  bool beginFrameIfNeeded() {
    if (frameOpen) return true;
    if (!initialized) return false;
    FrameSlot& s = slot();
    if (s.submittedFrame) {
      check(vk.vkWaitForFences(device, 1, &s.fence, VK_TRUE, UINT64_MAX), "vkWaitForFences");
      completedFrame = std::max(completedFrame, s.submittedFrame);
    }
    collectGarbage();
    envProcessReadback();
    beginDrawSlot();  // also invalidates the per-draw state caches (new command buffer)
    check(vk.vkResetCommandPool(device, s.pool, 0), "vkResetCommandPool");
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (!check(vk.vkBeginCommandBuffer(s.cmd, &bi), "vkBeginCommandBuffer")) return false;
    frameOpen = true;
    return true;
  }

  // `layers`: array layers from 0 (VK_REMAINING_ARRAY_LAYERS: all; the sun shadow cascades).
  void barrier(VkImage img, VkImageLayout& cur, VkImageLayout to, VkImageAspectFlags aspect, uint32_t layers = 1) {
    VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.oldLayout = cur;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = { aspect, 0, 1, 0, layers };
    VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vk.vkCmdPipelineBarrier2(slot().cmd, &di);
    cur = to;
  }
  VkImageAspectFlags aspectOf(const Rt& rt) {
    if (!rt.desc.depth) return VK_IMAGE_ASPECT_COLOR_BIT;
    return VK_IMAGE_ASPECT_DEPTH_BIT | (hasStencil(rt.format) ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
  }

  Rt* lookup(RtKey key) {
    auto it = rts.find(key);
    if (it == rts.end()) {
      st.unknownKeys++;
      logOnce("unknown" + std::to_string(key), "group binds renderer data %p that was never attached",
              reinterpret_cast<void*>(key));
      return nullptr;
    }
    return &it->second;
  }

  void doBeginGroup(const RtKey* colors, uint32_t colorCount, RtKey depth) {
    if (rendering) doEndGroup();
    groupGen++;  // the attachments (formats, samples) that pipelines and the alpha-test block depend on may differ
    if (!beginFrameIfNeeded()) return;
    st.groups++;
    groupColors.clear();
    groupDepth = 0;

    // A scaled scene (backend_ffx.inl): a group whose first target is window-sized renders into the render-size twins.
    groupLow = false;
    if (scaledScene) {
      RtKey first = depth;
      for (uint32_t i = 0; i < colorCount; i++)
        if (colors[i]) { first = colors[i]; break; }
      auto f = first ? rts.find(first) : rts.end();
      groupLow = f != rts.end() && scalable(first, f->second);
    }
    std::vector<Rt*> cs;
    uint32_t samples = 0;
    renderW = UINT32_MAX;
    renderH = UINT32_MAX;
    for (uint32_t i = 0; i < colorCount; i++) {
      if (!colors[i]) continue;
      Rt* rt = rtFor(colors[i], groupLow);
      if (!rt) continue;
      if (rt->twin) lowDirty.insert(colors[i]);
      if (samples && rt->desc.samples != samples) {
        st.sampleMismatches++;
        logOnce("msc" + std::to_string(colors[i]), "group color %u has %u samples, color 0 has %u: dropped", i,
                rt->desc.samples, samples);
        continue;
      }
      samples = rt->desc.samples;
      markCubeFace(colors[i]);
      renderW = std::min(renderW, rt->desc.width);
      renderH = std::min(renderH, rt->desc.height);
      cs.push_back(rt);
      groupColors.push_back(colors[i]);
    }
    Rt* ds = nullptr;
    if (depth) {
      ds = rtFor(depth, groupLow);
      if (ds && ds->twin) lowDirty.insert(depth);
      if (ds && samples && ds->desc.samples != samples) {
        st.sampleMismatches++;
        logOnce("msd" + std::to_string(depth) + "_" + std::to_string(samples),
                "group depth %p has %u samples, colors have %u: depth dropped for this group",
                reinterpret_cast<void*>(depth), ds->desc.samples, samples);
        ds = nullptr;
      }
      if (ds && cs.empty()) {
        renderW = ds->desc.width;
        renderH = ds->desc.height;
      }
      if (ds && (ds->desc.width < renderW || ds->desc.height < renderH)) {
        st.sizeMismatches++;
        logOnce("dsz" + std::to_string(depth), "group depth %ux%u smaller than render area %ux%u: depth dropped",
                ds->desc.width, ds->desc.height, renderW, renderH);
        ds = nullptr;
      }
      if (ds) groupDepth = depth;
    }
    if (cs.empty() && !ds) {
      renderW = renderH = 0;
      return;  // nothing renderable; clears in this group are ignored
    }

    std::vector<VkRenderingAttachmentInfo> ca(cs.size(), { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO });
    for (size_t i = 0; i < cs.size(); i++) {
      Rt& rt = *cs[i];
      bool ms = rt.msImage != VK_NULL_HANDLE;
      barrier(ms ? rt.msImage : rt.image, ms ? rt.msLayout : rt.layout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
              VK_IMAGE_ASPECT_COLOR_BIT);
      ca[i].imageView = ms ? rt.msView : rt.view;
      ca[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;  // D3D9 keeps contents; clears are explicit
      ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    }
    VkRenderingAttachmentInfo da{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    if (ds) {
      barrier(ds->image, ds->layout, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, aspectOf(*ds));
      da.imageView = ds->view;
      da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      da.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
      da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    }
    groupColorCount = uint32_t(std::min<size_t>(cs.size(), 4));
    groupX8Mask = 0;
    for (uint32_t i = 0; i < groupColorCount; i++) {
      groupColorFormats[i] = cs[i]->format;
      if (cs[i]->desc.d3dFormat == kD3DFMT_X8R8G8B8) groupX8Mask |= 1u << i;
    }
    groupRt0Format = cs.empty() ? 0 : cs[0]->desc.d3dFormat;
    groupSamples = cs.empty() ? (ds ? ds->desc.samples : 1) : cs[0]->desc.samples;
    if (!groupSamples) groupSamples = 1;
    groupDepthFormat = ds ? ds->format : VK_FORMAT_UNDEFINED;
    groupColorAtt.assign(ca.begin(), ca.end());
    groupDepthAtt = da;
    groupHasDepth = ds != nullptr;
    groupHasStencil = ds && hasStencil(ds->format);
    rendering = true;
    beginRenderingInstance();
  }

  // The render pass instance of the current group. Draws that must change an image layout (an RT sampled before
  // it was ever rendered) end it, record the barrier and start it again; attachments load their contents.
  std::vector<VkRenderingAttachmentInfo> groupColorAtt;
  VkRenderingAttachmentInfo groupDepthAtt{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
  bool groupHasDepth = false, groupHasStencil = false, renderingSuspended = false;
  void beginRenderingInstance() {
    VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea = { { 0, 0 }, { renderW, renderH } };
    ri.layerCount = 1;
    ri.colorAttachmentCount = uint32_t(groupColorAtt.size());
    ri.pColorAttachments = groupColorAtt.data();
    ri.pDepthAttachment = groupHasDepth ? &groupDepthAtt : nullptr;
    ri.pStencilAttachment = groupHasStencil ? &groupDepthAtt : nullptr;
    vk.vkCmdBeginRendering(slot().cmd, &ri);
  }
  void suspendRendering() {
    if (!rendering || renderingSuspended) return;
    endActiveQuery();
    vk.vkCmdEndRendering(slot().cmd);
    renderingSuspended = true;
  }
  void resumeRendering() {
    if (!renderingSuspended) return;
    renderingSuspended = false;
    beginRenderingInstance();
  }

  void doClear(uint32_t flags, uint32_t argb, float z, uint32_t stencil, const int32_t rect[4]) {
    if (!rendering && !reopenLastGroup("clear")) return;
    st.clears++;
    // D3D9 clips the rect to the viewport, which Do_ClearBuffer sets to buffer 0's size (= render area).
    int32_t r[4] = { rect[0], rect[1], rect[2], rect[3] };
    if (groupLow) {  // window pixels -> render pixels
      r[0] = toSceneX(r[0]);
      r[1] = toSceneY(r[1]);
      r[2] = toSceneX(r[2]);
      r[3] = toSceneY(r[3]);
    }
    int32_t x1 = std::max(r[0], 0), y1 = std::max(r[1], 0);
    int32_t x2 = std::min(r[2], int32_t(renderW)), y2 = std::min(r[3], int32_t(renderH));
    if (x2 <= x1 || y2 <= y1) return;
    VkClearRect cr{ { { x1, y1 }, { uint32_t(x2 - x1), uint32_t(y2 - y1) } }, 0, 1 };

    std::vector<VkClearAttachment> atts;
    if (flags & kClearTarget) {
      // D3DCOLOR -> float as the D3D9 reference computes it: c * (1/255) in single precision. Verified against
      // real D3D9 in tools/backend_test (R32F exposes it: 0xC0 clears to 0x3F40C0C2, where 192/255 rounds to ...C1).
      constexpr float kInv255 = 1.0f / 255.0f;
      VkClearColorValue c{};
      c.float32[0] = float((argb >> 16) & 0xFF) * kInv255;
      c.float32[1] = float((argb >> 8) & 0xFF) * kInv255;
      c.float32[2] = float(argb & 0xFF) * kInv255;
      c.float32[3] = float((argb >> 24) & 0xFF) * kInv255;
      for (uint32_t i = 0; i < groupColors.size(); i++) atts.push_back({ VK_IMAGE_ASPECT_COLOR_BIT, i, { c } });
    }
    if (groupDepth) {
      const Rt& ds = *rtFor(groupDepth, groupLow);
      VkImageAspectFlags a = 0;
      if ((flags & kClearZBuffer) && hasDepth(ds.format)) a |= VK_IMAGE_ASPECT_DEPTH_BIT;
      if ((flags & kClearStencil) && hasStencil(ds.format)) a |= VK_IMAGE_ASPECT_STENCIL_BIT;
      if (a) {
        VkClearAttachment da{ a, 0, {} };
        da.clearValue.depthStencil = { std::clamp(z, 0.0f, 1.0f), stencil & 0xFF };
        atts.push_back(da);
      }
    }
    if (!atts.empty()) vk.vkCmdClearAttachments(slot().cmd, uint32_t(atts.size()), atts.data(), 1, &cr);
  }

  void doEndGroup() {
    if (!rendering) return;
    if (!renderingSuspended) endActiveQuery();
    VkCommandBuffer cmd = slot().cmd;
    if (!renderingSuspended) vk.vkCmdEndRendering(cmd);
    renderingSuspended = false;
    rendering = false;
    // NiDX9Renderer::Do_EndUsingRenderTargetGroup (0x00E74DC0) resolves every bound buffer whose MSAA level is
    // nonzero (StretchRect m_pkSurface2 -> m_pkSurface).
    for (RtKey key : groupColors) {
      Rt& rt = *rtFor(key, groupLow);
      if (!rt.msImage) continue;
      barrier(rt.msImage, rt.msLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
      barrier(rt.image, rt.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
      VkImageResolve r{};
      r.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
      r.dstSubresource = r.srcSubresource;
      r.extent = { rt.desc.width, rt.desc.height, 1 };
      vk.vkCmdResolveImage(cmd, rt.msImage, rt.msLayout, rt.image, rt.layout, 1, &r);
      st.resolves++;
    }
    // Leave every attachment in GENERAL: later groups may sample it (barrier() is a full memory dependency).
    for (RtKey key : groupColors) {
      Rt& rt = *rtFor(key, groupLow);
      if (rt.layout != VK_IMAGE_LAYOUT_GENERAL) barrier(rt.image, rt.layout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
    }
    if (groupDepth) {
      Rt& rt = *rtFor(groupDepth, groupLow);
      if (rt.desc.samples <= 1 && rt.layout != VK_IMAGE_LAYOUT_GENERAL)
        barrier(rt.image, rt.layout, VK_IMAGE_LAYOUT_GENERAL, aspectOf(rt));
    }
    // D3D9 keeps the render targets bound after EndUsingRenderTargetGroup (and the backend ends groups itself for
    // submissions): a later draw or clear outside a group reopens these (reopenLastGroup).
    lastColors = groupColors;
    lastDepth = groupDepth;
    groupColors.clear();
    groupDepth = 0;
  }
  std::vector<RtKey> lastColors;
  RtKey lastDepth = 0;
  bool reopenLastGroup(const char* what) {
    if (rendering) return true;
    if (lastColors.empty() && !lastDepth) return false;
    const std::vector<RtKey> c = lastColors;
    const RtKey d = lastDepth;
    doBeginGroup(c.data(), uint32_t(c.size()), d);
    st.groupsReopened++;
    logOnce(std::string("reopen_") + what, "%s outside a render-target group: the last group's targets (still bound in "
            "D3D9) were reopened", what);
    return rendering;
  }

  // Ends and submits the open frame. `waitAcquire` / `signal` are the swapchain semaphores when presenting.
  bool submitFrame(VkSemaphore waitAcquire, VkSemaphore signal) {
    if (!frameOpen) return true;
    if (rendering) doEndGroup();
    FrameSlot& s = slot();
    if (!check(vk.vkEndCommandBuffer(s.cmd), "vkEndCommandBuffer")) return false;
    check(vk.vkResetFences(device, 1, &s.fence), "vkResetFences");
    // Uploads first: their barriers order the copies before every use in the frame's main command buffer
    // (same queue, submission order).
    VkCommandBufferSubmitInfo cbis[2] = { { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO },
                                          { VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO } };
    uint32_t cbCount = 0;
    if (s.uploadOpen) {
      // Buffer writes (copies) become visible to every later use in the frame.
      VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
      mb.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
      mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      mb.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
      mb.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
      VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
      di.memoryBarrierCount = 1;
      di.pMemoryBarriers = &mb;
      vk.vkCmdPipelineBarrier2(s.upload, &di);
      hostFlush(drawSlot().staging);
      if (!check(vk.vkEndCommandBuffer(s.upload), "vkEndCommandBuffer(upload)")) return false;
      cbis[cbCount++].commandBuffer = s.upload;
      s.uploadOpen = false;
    }
    cbis[cbCount++].commandBuffer = s.cmd;
    hostFlush(drawSlot().uniforms);
    VkSemaphoreSubmitInfo wi{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    wi.semaphore = waitAcquire;
    wi.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSemaphoreSubmitInfo si{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    si.semaphore = signal;
    si.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    VkSubmitInfo2 sub{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    sub.commandBufferInfoCount = cbCount;
    sub.pCommandBufferInfos = cbis;
    sub.waitSemaphoreInfoCount = waitAcquire ? 1 : 0;
    sub.pWaitSemaphoreInfos = &wi;
    sub.signalSemaphoreInfoCount = signal ? 1 : 0;
    sub.pSignalSemaphoreInfos = &si;
    bool ok;
    {
      ok = check(vk.vkQueueSubmit2(queue, 1, &sub, s.fence), "vkQueueSubmit2");
    }
    s.submittedFrame = frameNumber;
    frameNumber++;
    frameOpen = false;
    st.frames++;
    return ok;
  }

  void waitIdle() {
    if (!device) return;
    check(vk.vkDeviceWaitIdle(device), "vkDeviceWaitIdle");
    completedFrame = frameNumber - 1;
    collectGarbage();
  }

  // --- swapchain -----------------------------------------------------------------------------------------------
  void destroySwapchain() {
    for (VkSemaphore s : renderDone) vk.vkDestroySemaphore(device, s, nullptr);
    renderDone.clear();
    swapImages.clear();
    for (VkImageView v : swapViews)
      if (v) vk.vkDestroyImageView(device, v, nullptr);
    swapViews.clear();
    if (swapchain) vk.vkDestroySwapchainKHR(device, swapchain, nullptr);
    swapchain = VK_NULL_HANDLE;
    publishOverlay();
  }

  bool createSwapchain() {
    VkSurfaceCapabilitiesKHR caps{};
    if (!check(vk.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface, &caps), "surface caps")) return false;
    if (caps.currentExtent.width == 0 || caps.currentExtent.height == 0) return false;  // minimized
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) {
      logf("shadow window: swapchain images cannot be blit targets; present disabled");
      return false;
    }
    uint32_t n = 0;
    vk.vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(n);
    vk.vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &n, fmts.data());
    VkSurfaceFormatKHR fmt = fmts.empty() ? VkSurfaceFormatKHR{} : fmts[0];
    for (auto& f : fmts)
      if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) fmt = f;
    // HDR: HDR10 (10-bit PQ, BT.2020) or scRGB (FP16 linear BT.709), drawn by the encode pass (colour attachment).
    VkSurfaceFormatKHR hdr10{}, scrgb{};
    hdr10Available = scRgbAvailable = false;
    if (colorspaceExt && (caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT))
      for (auto& f : fmts) {
        if (f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT &&
            (f.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || f.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32)) {
          if (!hdr10Available || f.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32) hdr10 = f;
          hdr10Available = true;
        }
        if (f.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT && f.format == VK_FORMAT_R16G16B16A16_SFLOAT) {
          scrgb = f;
          scRgbAvailable = true;
        }
      }
    // The requested HDR output, else the other HDR one, else SDR.
    hdrActive = HdrSettings::Off;
    if (hdr.output != HdrSettings::Off) {
      const bool wantPq = hdr.output == HdrSettings::Hdr10;
      if (wantPq ? hdr10Available : scRgbAvailable) hdrActive = hdr.output;
      else if (wantPq ? scRgbAvailable : hdr10Available) hdrActive = wantPq ? HdrSettings::ScRgb : HdrSettings::Hdr10;
      if (hdrActive != HdrSettings::Off) fmt = hdrActive == HdrSettings::Hdr10 ? hdr10 : scrgb;
    }
    vk.vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &n, nullptr);
    std::vector<VkPresentModeKHR> modes(n);
    vk.vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &n, modes.data());
    // The shadow window never throttles the game; the game window follows the game's presentation interval.
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    if (!(cfg.presentWindow && cfg.vsync)) {
      for (VkPresentModeKHR pm : modes) if (pm == VK_PRESENT_MODE_MAILBOX_KHR) mode = pm;
      for (VkPresentModeKHR pm : modes) if (pm == VK_PRESENT_MODE_IMMEDIATE_KHR) mode = pm;
    }

    VkSwapchainCreateInfoKHR sci{ VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR };
    sci.surface = surface;
    sci.minImageCount = std::max(caps.minImageCount, 3u);
    if (caps.maxImageCount) sci.minImageCount = std::min(sci.minImageCount, caps.maxImageCount);
    sci.imageFormat = fmt.format;
    sci.imageColorSpace = fmt.colorSpace;
    sci.imageExtent = caps.currentExtent;
    sci.imageArrayLayers = 1;
    sci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     (caps.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);  // gamma pass
    sci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    sci.preTransform = caps.currentTransform;
    sci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    sci.presentMode = mode;
    sci.clipped = VK_TRUE;
    sci.oldSwapchain = swapchain;
    VkSwapchainKHR sc = VK_NULL_HANDLE;
    if (!check(vk.vkCreateSwapchainKHR(device, &sci, nullptr, &sc), "vkCreateSwapchainKHR")) return false;
    destroySwapchain();
    swapchain = sc;
    swapFormat = fmt.format;
    swapExtent = caps.currentExtent;
    vk.vkGetSwapchainImagesKHR(device, swapchain, &n, nullptr);
    swapImages.resize(n);
    vk.vkGetSwapchainImagesKHR(device, swapchain, &n, swapImages.data());
    swapViews.assign(n, VK_NULL_HANDLE);
    if (sci.imageUsage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
      for (uint32_t i = 0; i < n; i++) {
        VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
        vci.image = swapImages[i];
        vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vci.format = fmt.format;
        vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        check(vk.vkCreateImageView(device, &vci, nullptr, &swapViews[i]), "vkCreateImageView(swapchain)");
      }
    VkSemaphoreCreateInfo sem{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    renderDone.resize(n);
    for (auto& s : renderDone) check(vk.vkCreateSemaphore(device, &sem, nullptr, &s), "vkCreateSemaphore");
    logOnce("swapfmt" + std::to_string(fmt.format) + "_" + std::to_string(fmt.colorSpace) + "_" + std::to_string(mode),
            "swapchain: %ux%u format %d colour space %d (%s) present mode %d, %u images; HDR10 %s, scRGB %s",
            swapExtent.width, swapExtent.height, int(fmt.format), int(fmt.colorSpace),
            hdrActive == HdrSettings::Hdr10 ? "HDR10" : hdrActive == HdrSettings::ScRgb ? "scRGB" : "SDR", int(mode), n,
            hdr10Available ? "offered" : "not offered", scRgbAvailable ? "offered" : "not offered");
    swapchainDirty = false;
    st.swapchains++;
    publishOverlay();
    publishHdr();
    return true;
  }

  void doPresent(RtKey onscreen) {
    if (!beginFrameIfNeeded()) return;
    if (scaledScene) finishScaledScene(0);  // a scaled scene the caller did not end
    if (rendering) doEndGroup();
    Rt* rt = onscreen ? lookup(onscreen) : nullptr;
    if (rt && !realize(onscreen, *rt)) rt = nullptr;
    bool canPresent = surface && rt && (swapchain || createSwapchain());
    if (canPresent && swapchainDirty) {
      // Recreating the swapchain replaces objects the open command buffer may already reference: finish it first.
      if (!submitFrame(VK_NULL_HANDLE, VK_NULL_HANDLE)) return;
      waitIdle();
      canPresent = createSwapchain();
      if (!beginFrameIfNeeded()) return;
    }
    if (!canPresent) {
      submitFrame(VK_NULL_HANDLE, VK_NULL_HANDLE);
      return;
    }
    FrameSlot& s = slot();
    uint32_t index = 0;
    VkResult ar;
    {
      ar = vk.vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, s.acquired, VK_NULL_HANDLE, &index);
    }
    if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
      swapchainDirty = true;
      submitFrame(VK_NULL_HANDLE, VK_NULL_HANDLE);
      return;
    }
    if (ar == VK_SUBOPTIMAL_KHR) swapchainDirty = true;
    else if (!check(ar, "vkAcquireNextImageKHR")) {
      submitFrame(VK_NULL_HANDLE, VK_NULL_HANDLE);
      return;
    }
    VkCommandBuffer cmd = s.cmd;
    VkImage dst = swapImages[index];
    VkImageLayout dstLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    const bool hdrPass = hdrActive != HdrSettings::Off && index < swapViews.size() && swapViews[index] && rt->sampleView;
    if (hdrPass) {
      // The menu goes into its own layer, composited at paper white by the encode (an HDR swapchain value is not an
      // SDR colour).
      bool menu;
      {
        menu = drawOverlayLayer(cmd);
      }
      barrier(rt->image, rt->layout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
      barrier(dst, dstLayout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
      recordHdrPresent(cmd, rt->sampleView, menu ? overlayLayer.view : VK_NULL_HANDLE, swapViews[index], swapFormat,
                       swapExtent, hdrActive);
    } else {
    {
    barrier(rt->image, rt->layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    barrier(dst, dstLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    const bool gammaPass = gammaActive && index < swapViews.size() && swapViews[index] && rt->sampleView;
    if (gammaPass) {
      barrier(rt->image, rt->layout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
      barrier(dst, dstLayout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    }
    if (gammaPass && recordGammaPresent(cmd, rt->sampleView, swapViews[index], swapFormat, swapExtent)) {
      st.gammaPresents++;
    } else {
    if (gammaPass) {
      barrier(rt->image, rt->layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
      barrier(dst, dstLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    }
    VkImageBlit blit{};
    blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blit.srcOffsets[1] = { int32_t(rt->desc.width), int32_t(rt->desc.height), 1 };
    blit.dstSubresource = blit.srcSubresource;
    blit.dstOffsets[1] = { int32_t(swapExtent.width), int32_t(swapExtent.height), 1 };
    bool same = swapExtent.width == rt->desc.width && swapExtent.height == rt->desc.height;
    vk.vkCmdBlitImage(cmd, rt->image, rt->layout, dst, dstLayout, 1, &blit, same ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
    }
    }
    if (overlayPending && overlayUsable() && overlayFrame && index < swapViews.size() && swapViews[index]) {
      // The menu, over the finished frame (after the gamma pass: it is not the game's image).
      if (ImDrawData* dd = &overlayFrame->dd) {
        barrier(dst, dstLayout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
        VkRenderingAttachmentInfo ca{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
        ca.imageView = swapViews[index];
        ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
        ri.renderArea = { { 0, 0 }, swapExtent };
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &ca;
        vk.vkCmdBeginRendering(cmd, &ri);
        ImGui_ImplVulkan_RenderDrawData(dd, cmd);
        vk.vkCmdEndRendering(cmd);
        st.overlayFrames++;
      }
    }
    }  // SDR
    overlayPending = false;
    overlayFrame.reset();
    barrier(dst, dstLayout, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_ASPECT_COLOR_BIT);
    VkSemaphore done = renderDone[index];
    if (!submitFrame(s.acquired, done)) return;
    VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR };
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &done;
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain;
    pi.pImageIndices = &index;
    VkResult pr;
    {
      pr = vk.vkQueuePresentKHR(queue, &pi);
    }
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) swapchainDirty = true;
    else check(pr, "vkQueuePresentKHR");
    maybeSavePipelines();
  }

  // --- in-game menu (Dear ImGui) --------------------------------------------------------------------------------
  // ImGui's Vulkan renderer (imgui_impl_vulkan) on this device, queue and swapchain format, with dynamic rendering.
  // Its own submissions (font atlas uploads) run on this thread, under the backend lock, like every other submit.
  bool overlayReady = false, overlayPending = false;
  VkFormat overlayFormat = VK_FORMAT_UNDEFINED;
  // The menu frame to draw at the next present: a copy of ImGui's draw data made by overlaySubmit on the caller's thread,
  // so the present (on the record thread) never reads the caller's ImGui context while the caller builds its next frame.
  // `textures`: ImGui's textures with updates pending (the font atlas), applied by the draw; the caller then waits for
  // that present before its next ImGui frame (overlaySyncNext).
  struct OverlayFrame {
    ImDrawData dd;
    ImVector<ImTextureData*> textures;
    ~OverlayFrame() {
      for (ImDrawList* l : dd.CmdLists) IM_DELETE(l);
    }
  };
  std::shared_ptr<OverlayFrame> overlayFrame;  // mutex
  // For the caller without waiting for the record thread: the swapchain size and whether the renderer is ready for
  // its format (publishOverlay), and the HDR status (publishHdr).
  std::atomic<uint32_t> pubSwapW{ 0 }, pubSwapH{ 0 };
  std::atomic<bool> pubOverlayReady{ false };
  bool overlaySyncNext = true;  // the caller's: its next overlayBeginFrame waits for the queue
  mutable std::mutex pubHdrMutex;
  HdrStatus pubHdr;  // pubHdrMutex
  bool overlayUsable() const { return overlayReady && overlayFormat == overlayTargetFormat(); }
  void publishOverlay() {
    pubSwapW = swapchain ? swapExtent.width : 0u;
    pubSwapH = swapchain ? swapExtent.height : 0u;
    pubOverlayReady = swapchain && overlayUsable() && uint32_t(swapImages.size()) == overlayImageCount;
  }
  void publishHdr() {
    std::lock_guard<std::mutex> lk(pubHdrMutex);
    pubHdr.active = hdrActive;
    pubHdr.hdr10Available = hdr10Available;
    pubHdr.scRgbAvailable = scRgbAvailable;
    pubHdr.wideTargets = cfg.wideTargets;
  }
  // HDR output: the menu is drawn into this layer (8-bit, cleared to transparent: premultiplied colour, coverage alpha).
  VkFormat overlayTargetFormat() const { return hdrActive != HdrSettings::Off ? VK_FORMAT_R8G8B8A8_UNORM : swapFormat; }
  bool drawOverlayLayer(VkCommandBuffer cmd) {
    if (!overlayPending || !overlayUsable() || overlayFormat != VK_FORMAT_R8G8B8A8_UNORM || !overlayFrame) return false;
    ImDrawData* dd = &overlayFrame->dd;
    if (!ensureFxImage(overlayLayer, swapExtent.width, swapExtent.height, VK_FORMAT_R8G8B8A8_UNORM)) return false;
    barrier(overlayLayer.image, overlayLayer.layout, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    VkRenderingAttachmentInfo ca{ VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO };
    ca.imageView = overlayLayer.view;
    ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ca.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    VkRenderingInfo ri{ VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea = { { 0, 0 }, swapExtent };
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &ca;
    vk.vkCmdBeginRendering(cmd, &ri);
    ImGui_ImplVulkan_RenderDrawData(dd, cmd);
    vk.vkCmdEndRendering(cmd);
    barrier(overlayLayer.image, overlayLayer.layout, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_ASPECT_COLOR_BIT);
    st.overlayFrames++;
    return true;
  }
  uint32_t overlayImageCount = 0;
  static PFN_vkVoidFunction overlayLoader(const char* name, void* user) {
    auto* self = static_cast<Impl*>(user);
    return self->vk.vkGetInstanceProcAddr(self->instance, name);
  }
  bool initOverlay() {
    if (!ImGui::GetCurrentContext() || !swapchain || swapViews.empty() || !swapViews[0]) return false;
    const uint32_t images = uint32_t(swapImages.size());
    const VkFormat target = overlayTargetFormat();
    if (overlayReady && overlayFormat == target) {
      if (images != overlayImageCount) {
        ImGui_ImplVulkan_SetMinImageCount(std::max(2u, images));
        overlayImageCount = images;
      }
      return true;
    }
    if (overlayReady) {  // the target format changed (swapchain format, HDR layer): the pipeline is rebuilt
      ImGui_ImplVulkan_Shutdown();
      overlayReady = false;
    }
    if (!ImGui_ImplVulkan_LoadFunctions(VK_API_VERSION_1_3, &overlayLoader, this)) {
      logOnce("imgui_load", "menu: ImGui Vulkan functions not loaded");
      return false;
    }
    overlayFormat = target;
    ImGui_ImplVulkan_InitInfo ii{};
    ii.ApiVersion = VK_API_VERSION_1_3;
    ii.Instance = instance;
    ii.PhysicalDevice = phys;
    ii.Device = device;
    ii.QueueFamily = queueFamily;
    ii.Queue = queue;
    ii.DescriptorPoolSize = 64;  // the backend's own pool (font atlas and any user textures)
    ii.MinImageCount = std::max(2u, images);
    ii.ImageCount = std::max(ii.MinImageCount, images);
    ii.PipelineCache = pipelineCache;
    ii.UseDynamicRendering = true;
    ii.PipelineInfoMain.PipelineRenderingCreateInfo = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    ii.PipelineInfoMain.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    ii.PipelineInfoMain.PipelineRenderingCreateInfo.pColorAttachmentFormats = &overlayFormat;
    ii.MinAllocationSize = 1024 * 1024;
    if (!ImGui_ImplVulkan_Init(&ii)) {
      logOnce("imgui_init", "menu: ImGui Vulkan renderer did not initialise");
      return false;
    }
    overlayReady = true;
    overlayImageCount = images;
    publishOverlay();
    logf("menu: ImGui renderer ready (swapchain %ux%u, format %d, %u images)", swapExtent.width, swapExtent.height,
         int(overlayFormat), images);
    return true;
  }
  void shutdownOverlay() {
    if (!overlayReady) return;
    overlayFrame.reset();
    ImGui_ImplVulkan_Shutdown();
    overlayReady = false;
    publishOverlay();
  }

  // --- window --------------------------------------------------------------------------------------------------
  static constexpr UINT kMsgDestroy = WM_APP + 1;
  static LRESULT CALLBACK wndProc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == kMsgDestroy) { DestroyWindow(h); return 0; }           // runs on the owning thread
    if (msg == WM_CLOSE) { ShowWindow(h, SW_MINIMIZE); return 0; }  // closing is owned by the plugin
    if (msg == WM_DESTROY) { PostQuitMessage(0); return 0; }
    if (msg == WM_MOUSEACTIVATE) return MA_NOACTIVATE;
    return DefWindowProcA(h, msg, wp, lp);
  }

  // The window lives on its own thread so its messages never go through (or stall on) the game's message loop.
  // It is never activated: the game pauses itself whenever its own window loses activation (WinMain 0x0086A850).
  bool createWindow(uint32_t w, uint32_t h) {
    std::promise<HWND> made;
    auto fut = made.get_future();
    windowThread = std::thread([w, h, &made] {
      WNDCLASSA wc{};
      wc.lpfnWndProc = wndProc;
      wc.hInstance = GetModuleHandleA(nullptr);
      wc.lpszClassName = "fnv_vk_shadow";
      wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
      RegisterClassA(&wc);
      RECT r{ 0, 0, LONG(w), LONG(h) };
      DWORD style = WS_OVERLAPPEDWINDOW;
      AdjustWindowRect(&r, style, FALSE);
      HWND wnd = CreateWindowExA(WS_EX_NOACTIVATE, wc.lpszClassName, "fnv_vk shadow (Vulkan)", style, CW_USEDEFAULT,
                                 CW_USEDEFAULT, r.right - r.left, r.bottom - r.top, nullptr, nullptr, wc.hInstance,
                                 nullptr);
      if (wnd) ShowWindow(wnd, SW_SHOWNOACTIVATE);
      made.set_value(wnd);
      if (!wnd) return;
      MSG msg;
      while (GetMessageA(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
      }
    });
    hwnd = fut.get();
    if (!hwnd) {
      windowThread.join();
      return false;
    }
    return true;
  }
  void destroyWindow() {
    if (!hwnd) return;
    // DestroyWindow must run on the owning thread: WM_DESTROY then posts WM_QUIT and the thread's loop ends.
    PostMessageA(hwnd, kMsgDestroy, 0, 0);
    if (windowThread.joinable()) windowThread.join();
    hwnd = nullptr;
  }

  // --- init ----------------------------------------------------------------------------------------------------
  bool initVulkan(uint32_t w, uint32_t h) {
    if (!vk.loadGlobal()) { logf("vulkan-1.dll or global entry points missing"); return false; }

    std::vector<const char*> layers, exts = { VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME };
    bool debugUtils = false;
    {
      uint32_t n = 0;
      vk.vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
      std::vector<VkExtensionProperties> props(n);
      vk.vkEnumerateInstanceExtensionProperties(nullptr, &n, props.data());
      for (auto& p : props) if (!strcmp(p.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) debugUtils = true;
    }
    if (cfg.validation) {
      uint32_t n = 0;
      vk.vkEnumerateInstanceLayerProperties(&n, nullptr);
      std::vector<VkLayerProperties> props(n);
      vk.vkEnumerateInstanceLayerProperties(&n, props.data());
      bool found = false;
      for (auto& p : props) if (!strcmp(p.layerName, "VK_LAYER_KHRONOS_validation")) found = true;
      if (found) layers.push_back("VK_LAYER_KHRONOS_validation");
      logf("validation layer %s", found ? "enabled" : "requested but not installed for this architecture");
    }
    if (debugUtils) exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    {  // HDR colour spaces (HDR10 ST2084, extended sRGB linear) for the swapchain
      uint32_t n = 0;
      vk.vkEnumerateInstanceExtensionProperties(nullptr, &n, nullptr);
      std::vector<VkExtensionProperties> props(n);
      vk.vkEnumerateInstanceExtensionProperties(nullptr, &n, props.data());
      for (auto& p : props) if (!strcmp(p.extensionName, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME)) colorspaceExt = true;
      if (colorspaceExt) exts.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
    }

    VkApplicationInfo app{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.pApplicationName = "fnv_vk";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    ici.enabledLayerCount = uint32_t(layers.size());
    ici.ppEnabledLayerNames = layers.data();
    ici.enabledExtensionCount = uint32_t(exts.size());
    ici.ppEnabledExtensionNames = exts.data();
    if (!check(vk.vkCreateInstance(&ici, nullptr, &instance), "vkCreateInstance")) return false;
    vk.loadInstance(instance);
    if (debugUtils && vk.vkCreateDebugUtilsMessengerEXT) {
      VkDebugUtilsMessengerCreateInfoEXT mci{ VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
      mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
      mci.pfnUserCallback = debugCallback;
      mci.pUserData = this;
      check(vk.vkCreateDebugUtilsMessengerEXT(instance, &mci, nullptr, &messenger), "vkCreateDebugUtilsMessengerEXT");
    }

    if (cfg.presentWindow) {
      VkWin32SurfaceCreateInfoKHR sci{ VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
      sci.hinstance = GetModuleHandleA(nullptr);
      sci.hwnd = static_cast<HWND>(cfg.presentWindow);
      if (!check(vk.vkCreateWin32SurfaceKHR(instance, &sci, nullptr, &surface), "vkCreateWin32SurfaceKHR(game window)"))
        surface = VK_NULL_HANDLE;
      else
        logf("presenting into the game window %p (vsync %d)", cfg.presentWindow, int(cfg.vsync));
    } else if (cfg.shadowWindow) {
      if (createWindow(w, h)) {
        VkWin32SurfaceCreateInfoKHR sci{ VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR };
        sci.hinstance = GetModuleHandleA(nullptr);
        sci.hwnd = hwnd;
        if (!check(vk.vkCreateWin32SurfaceKHR(instance, &sci, nullptr, &surface), "vkCreateWin32SurfaceKHR"))
          surface = VK_NULL_HANDLE;
      } else {
        logf("shadow window creation failed (%lu); continuing without present", GetLastError());
      }
    }

    // Device: the first Vulkan 1.3 GPU with a graphics queue that can present to the shadow window.
    uint32_t n = 0;
    vk.vkEnumeratePhysicalDevices(instance, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vk.vkEnumeratePhysicalDevices(instance, &n, pds.data());
    for (VkPhysicalDevice pd : pds) {
      VkPhysicalDeviceProperties p{};
      vk.vkGetPhysicalDeviceProperties(pd, &p);
      if (p.apiVersion < VK_API_VERSION_1_3) continue;
      uint32_t qn = 0;
      vk.vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
      std::vector<VkQueueFamilyProperties> qf(qn);
      vk.vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qf.data());
      for (uint32_t i = 0; i < qn; i++) {
        if (!(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) continue;
        VkBool32 present = VK_TRUE;
        if (surface) vk.vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &present);
        if (!present) continue;
        phys = pd;
        queueFamily = i;
        logf("backend device: %s (Vulkan %u.%u), queue family %u", p.deviceName, VK_API_VERSION_MAJOR(p.apiVersion),
             VK_API_VERSION_MINOR(p.apiVersion), i);
        break;
      }
      if (phys) break;
    }
    if (!phys) { logf("no Vulkan 1.3 device with a graphics+present queue"); return false; }
    vk.vkGetPhysicalDeviceMemoryProperties(phys, &memProps);

    std::set<std::string> devExtAvail;
    {
      uint32_t ne = 0;
      vk.vkEnumerateDeviceExtensionProperties(phys, nullptr, &ne, nullptr);
      std::vector<VkExtensionProperties> props(ne);
      vk.vkEnumerateDeviceExtensionProperties(phys, nullptr, &ne, props.data());
      for (auto& p : props) devExtAvail.insert(p.extensionName);
    }
    VkPhysicalDeviceNonSeamlessCubeMapFeaturesEXT fNsc{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NON_SEAMLESS_CUBE_MAP_FEATURES_EXT };
    VkPhysicalDeviceCustomBorderColorFeaturesEXT fCbc{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT, &fNsc };
    VkPhysicalDeviceVulkan13Features f13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, &fCbc };
    VkPhysicalDeviceVulkan12Features f12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13 };
    VkPhysicalDeviceVulkan11Features f11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &f12 };
    VkPhysicalDeviceFeatures2 f2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f11 };
    vk.vkGetPhysicalDeviceFeatures2(phys, &f2);
    if (!f2.features.textureCompressionBC) {
      logf("device lacks textureCompressionBC (the game's textures are DXT1/3/5)");
      return false;
    }
    if (!f13.dynamicRendering || !f13.synchronization2) {
      logf("device lacks dynamicRendering/synchronization2");
      return false;
    }
    // Required by the translated game shaders (docs/SHADER_ABI.md).
    if (!f12.scalarBlockLayout || !f12.vulkanMemoryModel || !f12.bufferDeviceAddress ||
        !f13.shaderDemoteToHelperInvocation) {
      logf("device lacks a feature the translated shaders require (scalarBlockLayout %u vulkanMemoryModel %u "
           "bufferDeviceAddress %u demote %u)", f12.scalarBlockLayout, f12.vulkanMemoryModel, f12.bufferDeviceAddress,
           f13.shaderDemoteToHelperInvocation);
      return false;
    }
    nonSeamlessCube = devExtAvail.count(VK_EXT_NON_SEAMLESS_CUBE_MAP_EXTENSION_NAME) && fNsc.nonSeamlessCubeMap;
    customBorder = devExtAvail.count(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME) && fCbc.customBorderColors &&
                   fCbc.customBorderColorWithoutFormat;
    anisotropy = f2.features.samplerAnisotropy;
    fragmentStores = f2.features.fragmentStoresAndAtomics;
    if (!nonSeamlessCube) logf("VK_EXT_non_seamless_cube_map unavailable: cube maps filter across faces (D3D9 does not)");
    VkPhysicalDeviceNonSeamlessCubeMapFeaturesEXT eNsc{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NON_SEAMLESS_CUBE_MAP_FEATURES_EXT };
    eNsc.nonSeamlessCubeMap = nonSeamlessCube;
    VkPhysicalDeviceCustomBorderColorFeaturesEXT eCbc{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT };
    eCbc.customBorderColors = customBorder;
    eCbc.customBorderColorWithoutFormat = customBorder;
    void* extChain = nullptr;
    if (customBorder) { eCbc.pNext = extChain; extChain = &eCbc; }
    if (nonSeamlessCube) { eNsc.pNext = extChain; extChain = &eNsc; }
    VkPhysicalDeviceVulkan13Features e13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, extChain };
    e13.dynamicRendering = VK_TRUE;
    e13.synchronization2 = VK_TRUE;
    e13.shaderDemoteToHelperInvocation = VK_TRUE;
    VkPhysicalDeviceVulkan12Features e12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &e13 };
    e12.scalarBlockLayout = VK_TRUE;
    e12.vulkanMemoryModel = VK_TRUE;
    e12.vulkanMemoryModelDeviceScope = f12.vulkanMemoryModelDeviceScope;
    e12.bufferDeviceAddress = VK_TRUE;
    e12.hostQueryReset = f12.hostQueryReset;
    hostQueryReset = f12.hostQueryReset;
    // FSR 3.1 (backend_ffx.inl): what its shaders use, where the device has it. The SDK picks its fp16 and wave64 shader
    // variants from what the physical device supports, so those are enabled whenever supported.
    e12.shaderFloat16 = f12.shaderFloat16;
    e13.subgroupSizeControl = f13.subgroupSizeControl;
    e13.computeFullSubgroups = f13.computeFullSubgroups;
    VkPhysicalDeviceVulkan11Features e11{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES, &e12 };
    e11.storageBuffer16BitAccess = f11.storageBuffer16BitAccess;
    e11.uniformAndStorageBuffer16BitAccess = f11.uniformAndStorageBuffer16BitAccess;
    const float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo qci{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = queueFamily;
    qci.queueCount = 1;
    qci.pQueuePriorities = &queuePriority;
    std::vector<const char*> devExts;
    if (surface) devExts.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    if (nonSeamlessCube) devExts.push_back(VK_EXT_NON_SEAMLESS_CUBE_MAP_EXTENSION_NAME);
    if (customBorder) devExts.push_back(VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
    VkPhysicalDeviceFeatures2 e2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &e11 };
    e2.features.textureCompressionBC = VK_TRUE;
    e2.features.samplerAnisotropy = f2.features.samplerAnisotropy;
    e2.features.fragmentStoresAndAtomics = f2.features.fragmentStoresAndAtomics;  // the skin buffer
    e2.features.shaderClipDistance = f2.features.shaderClipDistance;
    e2.features.independentBlend = f2.features.independentBlend;
    e2.features.fillModeNonSolid = f2.features.fillModeNonSolid;
    e2.features.occlusionQueryPrecise = f2.features.occlusionQueryPrecise;
    e2.features.shaderInt16 = f2.features.shaderInt16;  // FSR 3.1's 16-bit shader variants
    e2.features.shaderStorageImageReadWithoutFormat = f2.features.shaderStorageImageReadWithoutFormat;
    e2.features.shaderStorageImageWriteWithoutFormat = f2.features.shaderStorageImageWriteWithoutFormat;
    if (!f2.features.shaderClipDistance || !f2.features.independentBlend)
      logf("device lacks shaderClipDistance or independentBlend (needed by the translated shaders / D3D9 write masks)");
    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &e2 };
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(devExts.size());
    dci.ppEnabledExtensionNames = devExts.data();
    if (!check(vk.vkCreateDevice(phys, &dci, nullptr, &device), "vkCreateDevice")) return false;
    if (!vk.loadDevice(device)) {
      // Swapchain entry points are null without the extension; everything else is core 1.3.
      if (surface) { logf("missing device entry points"); return false; }
    }
    vk.vkGetDeviceQueue(device, queueFamily, 0, &queue);

    VmaVulkanFunctions vf{};
    vf.vkGetInstanceProcAddr = vk.vkGetInstanceProcAddr;
    vf.vkGetDeviceProcAddr = vk.vkGetDeviceProcAddr;
    VmaAllocatorCreateInfo aci{};
    aci.physicalDevice = phys;
    aci.device = device;
    aci.instance = instance;
    aci.vulkanApiVersion = VK_API_VERSION_1_3;
    aci.pVulkanFunctions = &vf;
    // The game is a 32-bit process without LARGEADDRESSAWARE: 2 GiB of address space, most of it used by the game.
    // VMA's default 256 MiB blocks, mapped whole for host-visible memory, would take a large share of it (and left
    // the driver unable to allocate its 2 MiB compiler arena: it crashed dereferencing the failed allocation).
    aci.preferredLargeHeapBlockSize = 16ull << 20;
    if (!check(vmaCreateAllocator(&aci, &allocator), "vmaCreateAllocator")) return false;

    // D3DFMT_D24S8 -> D24_UNORM_S8_UINT where supported (exact match), else D32_SFLOAT_S8_UINT.
    VkFormatFeatureFlags dsNeed = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
    depthFormat = formatSupports(VK_FORMAT_D24_UNORM_S8_UINT, dsNeed) ? VK_FORMAT_D24_UNORM_S8_UINT
                                                                       : VK_FORMAT_D32_SFLOAT_S8_UINT;
    logf("D3DFMT_D24S8 -> %s", depthFormat == VK_FORMAT_D24_UNORM_S8_UINT ? "VK_FORMAT_D24_UNORM_S8_UINT"
                                                                          : "VK_FORMAT_D32_SFLOAT_S8_UINT");

    for (FrameSlot& s : slots) {
      VkCommandPoolCreateInfo pci{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
      pci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
      pci.queueFamilyIndex = queueFamily;
      if (!check(vk.vkCreateCommandPool(device, &pci, nullptr, &s.pool), "vkCreateCommandPool")) return false;
      VkCommandBufferAllocateInfo ai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
      ai.commandPool = s.pool;
      ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      ai.commandBufferCount = 2;
      VkCommandBuffer cbs[2] = {};
      if (!check(vk.vkAllocateCommandBuffers(device, &ai, cbs), "vkAllocateCommandBuffers")) return false;
      s.cmd = cbs[0];
      s.upload = cbs[1];
      VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
      fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
      if (!check(vk.vkCreateFence(device, &fci, nullptr, &s.fence), "vkCreateFence")) return false;
      VkSemaphoreCreateInfo sem{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
      if (!check(vk.vkCreateSemaphore(device, &sem, nullptr, &s.acquired), "vkCreateSemaphore")) return false;
    }
    initialized = true;  // initDrawPath records into the first frame
    return initDrawPath();
  }

  void teardown() {
    if (device) {
      if (frameOpen) submitFrame(VK_NULL_HANDLE, VK_NULL_HANDLE);
      waitIdle();
      shutdownOverlay();
      teardownFfx();
      for (auto& [key, rt] : rts) release(rt, false);
      if (allocator) teardownDrawPath();
      for (const Garbage& g : garbage) freeGarbage(g);  // the device is idle: every deferred object is unused
      garbage.clear();
      if (allocator) {
        for (auto& [key, t] : textures) freeVmaGarbage({ 0, t.image, t.view, VK_NULL_HANDLE, t.alloc });
        textures.clear();
        for (const VmaGarbage& g : vmaGarbage) freeVmaGarbage(g);
        vmaGarbage.clear();
        destroyAllFx();
        if (dynPool) vmaDestroyPool(allocator, dynPool);  // after every buffer from it is freed
        dynPool = nullptr;
        vmaDestroyAllocator(allocator);
        allocator = nullptr;
      }
      destroySwapchain();
      for (FrameSlot& s : slots) {
        if (s.acquired) vk.vkDestroySemaphore(device, s.acquired, nullptr);
        if (s.fence) vk.vkDestroyFence(device, s.fence, nullptr);
        if (s.pool) vk.vkDestroyCommandPool(device, s.pool, nullptr);
        s = FrameSlot{};
      }
      vk.vkDestroyDevice(device, nullptr);
      device = VK_NULL_HANDLE;
    }
    if (surface) vk.vkDestroySurfaceKHR(instance, surface, nullptr);
    surface = VK_NULL_HANDLE;
    destroyWindow();
    if (messenger) vk.vkDestroyDebugUtilsMessengerEXT(instance, messenger, nullptr);
    messenger = VK_NULL_HANDLE;
    if (instance) vk.vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
    initialized = false;
  }

  // --- textures ------------------------------------------------------------------------------------------------
  struct TexFormat {
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkComponentMapping swizzle{ VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
  };
  // One- and two-channel block compression as FOURCCs (DDS files written for D3D9 drivers' BC4 / BC5 support; the game
  // uses none).
  static constexpr uint32_t kFourccBC4U = 0x55344342, kFourccATI1 = 0x31495441;
  static constexpr uint32_t kFourccBC5U = 0x55354342, kFourccATI2 = 0x32495441;
  // D3DFORMAT -> Vulkan format + view swizzle reproducing what D3D9 samples.
  static TexFormat texFormatFor(uint32_t d3dFmt) {
    TexFormat f;
    switch (d3dFmt) {
      case d3dx::kFmtDXT1: f.format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK; break;  // 3-colour blocks: index 3 = (0,0,0,0)
      // DXT2/DXT4 are DXT3/DXT5 blocks whose colour is premultiplied by alpha: D3D9 samples the stored values as they
      // are, so they decode identically (the plugin's device reports them as texture formats).
      case d3dx::kFmtDXT2:
      case d3dx::kFmtDXT3: f.format = VK_FORMAT_BC2_UNORM_BLOCK; break;
      case d3dx::kFmtDXT4:
      case d3dx::kFmtDXT5: f.format = VK_FORMAT_BC3_UNORM_BLOCK; break;
      case kFourccBC5U:
      case kFourccATI2: f.format = VK_FORMAT_BC5_UNORM_BLOCK; break;  // two-channel normal maps (the skin's detail)
      case kFourccBC4U:
      case kFourccATI1: f.format = VK_FORMAT_BC4_UNORM_BLOCK; break;  // one-channel data (the hair's tangent shift)
      case d3dx::kFmtA8R8G8B8: f.format = VK_FORMAT_B8G8R8A8_UNORM; break;
      case d3dx::kFmtX8R8G8B8:
        f.format = VK_FORMAT_B8G8R8A8_UNORM;
        f.swizzle.a = VK_COMPONENT_SWIZZLE_ONE;  // D3D9 samples X8 formats with alpha 1
        break;
      case kD3DFMT_R5G6B5: f.format = VK_FORMAT_R5G6B5_UNORM_PACK16; break;  // same bit layout (R in 15:11)
      case kD3DFMT_A16B16G16R16F: f.format = VK_FORMAT_R16G16B16A16_SFLOAT; break;
      case kD3DFMT_A32B32G32R32F: f.format = VK_FORMAT_R32G32B32A32_SFLOAT; break;  // data maps (wet world's puddles)
      case kD3DFMT_R32F: f.format = VK_FORMAT_R32_SFLOAT; break;                     // data (wet world's basin levels)
      case d3dx::kFmtA4R4G4B4:
        // D3D A4R4G4B4 bits [15:12]=A [11:8]=R [7:4]=G [3:0]=B; Vulkan B4G4R4A4_UNORM_PACK16 names those fields
        // B, G, R, A respectively, so read r<-G, g<-R, b<-A, a<-B.
        f.format = VK_FORMAT_B4G4R4A4_UNORM_PACK16;
        f.swizzle = { VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_A, VK_COMPONENT_SWIZZLE_B };
        break;
    }
    return f;
  }

  VkCommandBuffer uploadCmd() {
    FrameSlot& s = slot();
    if (!s.uploadOpen) {
      VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
      bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      if (!check(vk.vkBeginCommandBuffer(s.upload, &bi), "vkBeginCommandBuffer(upload)")) return VK_NULL_HANDLE;
      s.uploadOpen = true;
    }
    return s.upload;
  }

  void releaseTexture(Tex& t) {
    addGarbage({ frameNumber, t.image, t.view, VK_NULL_HANDLE, t.alloc });
    t = Tex{};
  }

  // --- rendered cube maps --------------------------------------------------------------------------------------
  void markCubeFace(RtKey k) {
    auto it = cubeFaces.find(k);
    if (it == cubeFaces.end()) return;
    auto c = cubes.find(it->second.first);
    if (c != cubes.end()) c->second.dirty |= uint8_t(1u << it->second.second);
  }
  void subresourceBarrier(VkImage img, uint32_t level, uint32_t levels, uint32_t layer, VkImageLayout from,
                          VkImageLayout to) {
    VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, level, levels, layer, 1 };
    VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vk.vkCmdPipelineBarrier2(slot().cmd, &di);
  }
  // Brings textures[key] up to date with the faces: (re)created when the faces' size or format changes, then each
  // dirty face copied into its layer and that layer's mips rebuilt by successive linear 2x downsamples (D3D9
  // autogeneration filters linearly by default, SetAutoGenFilterType). Outside any rendering (suspended meanwhile).
  bool assembleCube(TexKey key, RenderedCube& c) {
    // Nothing drawn into the faces, and none attached, resized or destroyed, since the last assembly (each of those
    // marks the face dirty): the cube is current. Draws sampling a cube call this for every draw.
    if (!c.dirty) {
      auto it = textures.find(key);
      if (it != textures.end() && it->second.image) return true;
    }
    Rt* f[6];
    for (uint32_t i = 0; i < 6; i++) {
      auto it = c.faces[i] ? rts.find(c.faces[i]) : rts.end();
      if (it == rts.end() || !realize(c.faces[i], it->second)) return false;
      f[i] = &it->second;
      const RtDesc& d = f[i]->desc;
      if (d.depth || d.samples > 1 || d.width != d.height || !(d == f[0]->desc) || f[i]->format != f[0]->format)
        return false;
    }
    const uint32_t size = f[0]->desc.width;
    const VkFormat format = f[0]->format;
    uint32_t full = 1;
    for (uint32_t s = size; s > 1; s >>= 1) full++;
    uint32_t levels = c.levels ? std::min(c.levels, full) : full;
    if (levels > 1 && !formatSupports(format, VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
                                                  VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT)) {
      logOnce("cubemips" + std::to_string(format), "rendered cube map format %d cannot be blitted: one mip level",
              int(format));
      levels = 1;
    }
    auto it = textures.find(key);
    bool fresh = false;
    if (it == textures.end() || it->second.format != format || it->second.width != size || it->second.levels != levels) {
      VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
      ici.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
      ici.imageType = VK_IMAGE_TYPE_2D;
      ici.format = format;
      ici.extent = { size, size, 1 };
      ici.mipLevels = levels;
      ici.arrayLayers = 6;
      ici.samples = VK_SAMPLE_COUNT_1_BIT;
      ici.tiling = VK_IMAGE_TILING_OPTIMAL;
      ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
      VmaAllocationCreateInfo iai{};
      iai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
      Tex t;
      t.format = format;
      t.width = t.height = size;
      t.levels = levels;
      t.layers = 6;
      t.bytes = 0;
      if (!check(vmaCreateImage(allocator, &ici, &iai, &t.image, &t.alloc, nullptr), "vmaCreateImage(cube)")) return false;
      VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
      vci.image = t.image;
      vci.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
      vci.format = format;
      vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, 6 };
      if (!check(vk.vkCreateImageView(device, &vci, nullptr, &t.view), "vkCreateImageView(cube)")) {
        releaseTexture(t);
        return false;
      }
      if (it != textures.end()) {
        releaseTexture(it->second);
        it->second = t;
      } else {
        it = textures.emplace(key, t).first;
      }
      fresh = true;
      c.dirty = 0x3F;
    }
    if (!c.dirty) return true;
    Tex& t = it->second;
    suspendRendering();
    for (uint32_t i = 0; i < 6; i++) {
      if (!(c.dirty & (1u << i))) continue;
      Rt& r = *f[i];
      barrier(r.image, r.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
      subresourceBarrier(t.image, 0, levels, i, fresh ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      VkImageCopy cp{};
      cp.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
      cp.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, i, 1 };
      cp.extent = { size, size, 1 };
      vk.vkCmdCopyImage(slot().cmd, r.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.image,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &cp);
      for (uint32_t l = 1; l < levels; l++) {
        subresourceBarrier(t.image, l - 1, 1, i, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        const int32_t s0 = int32_t(std::max(size >> (l - 1), 1u)), s1 = int32_t(std::max(size >> l, 1u));
        VkImageBlit bl{};
        bl.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, l - 1, i, 1 };
        bl.srcOffsets[1] = { s0, s0, 1 };
        bl.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, l, i, 1 };
        bl.dstOffsets[1] = { s1, s1, 1 };
        vk.vkCmdBlitImage(slot().cmd, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.image,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
      }
      if (levels > 1)
        subresourceBarrier(t.image, 0, levels - 1, i, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      subresourceBarrier(t.image, levels - 1, 1, i, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    c.dirty = 0;
    envAfterAssemble(key, t);
    resumeRendering();
    return true;
  }
  // The cube a face target belongs to, assembled for sampling; null if `face` is not a face or the cube is unusable.
  const Tex* cubeForFace(RtKey face) {
    auto it = cubeFaces.find(face);
    if (it == cubeFaces.end()) return nullptr;
    auto c = cubes.find(it->second.first);
    if (c == cubes.end() || !assembleCube(c->first, c->second)) return nullptr;
    auto t = textures.find(c->first);
    return t == textures.end() ? nullptr : &t->second;
  }
  bool isCubeFace(RtKey k) const { return cubeFaces.count(k) != 0; }
  void detachCubeFace(RtKey face) {
    auto it = cubeFaces.find(face);
    if (it == cubeFaces.end()) return;
    auto c = cubes.find(it->second.first);
    cubeFaces.erase(it);
    if (c == cubes.end()) return;
    bool any = false;
    for (RtKey& k : c->second.faces) {
      if (k == face) k = 0;
      any |= k != 0;
    }
    if (any) return;
    auto t = textures.find(c->first);
    if (t != textures.end()) {
      releaseTexture(t->second);
      textures.erase(t);
    }
    cubes.erase(c);
  }

  bool doCreateTexture(TexKey key, const d3dx::Texture& src) {
    if (!beginFrameIfNeeded()) return false;
    TexFormat tf = texFormatFor(src.format);
    if (tf.format == VK_FORMAT_UNDEFINED || src.surfaces.empty()) {
      st.textureFailures++;
      logOnce("texfmt" + std::to_string(src.format), "texture D3DFORMAT %u has no Vulkan mapping", src.format);
      return false;
    }
    const uint32_t layers = src.cube ? 6 : 1;
    // Staging: every subresource, offsets aligned to 16 (texel-block and copy-offset alignment).
    VkDeviceSize total = 0;
    for (const d3dx::Level& l : src.surfaces) total = ((total + 15) & ~VkDeviceSize(15)) + l.data.size();
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = total;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo bai{};
    bai.usage = VMA_MEMORY_USAGE_AUTO;
    bai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = nullptr;
    VmaAllocationInfo stagingInfo{};
    if (!check(vmaCreateBuffer(allocator, &bci, &bai, &staging, &stagingAlloc, &stagingInfo), "vmaCreateBuffer(staging)")) {
      st.textureFailures++;
      return false;
    }
    std::vector<VkBufferImageCopy> regions;
    regions.reserve(src.surfaces.size());
    VkDeviceSize off = 0;
    for (uint32_t f = 0; f < layers; f++)
      for (uint32_t l = 0; l < src.levels; l++) {
        const d3dx::Level& lv = src.surfaces[size_t(f) * src.levels + l];
        off = (off + 15) & ~VkDeviceSize(15);
        memcpy(static_cast<uint8_t*>(stagingInfo.pMappedData) + off, lv.data.data(), lv.data.size());
        VkBufferImageCopy r{};
        r.bufferOffset = off;
        r.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, l, f, 1 };
        r.imageExtent = { lv.width, lv.height, 1 };
        regions.push_back(r);
        off += lv.data.size();
      }
    vmaFlushAllocation(allocator, stagingAlloc, 0, VK_WHOLE_SIZE);

    VkImageCreateInfo ici{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.flags = src.cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = tf.format;
    ici.extent = { src.width, src.height, 1 };
    ici.mipLevels = src.levels;
    ici.arrayLayers = layers;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo iai{};
    iai.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    Tex t;
    t.format = tf.format;
    t.width = src.width;
    t.height = src.height;
    t.levels = src.levels;
    t.layers = layers;
    t.bytes = total;
    if (!check(vmaCreateImage(allocator, &ici, &iai, &t.image, &t.alloc, nullptr), "vmaCreateImage(texture)")) {
      addGarbage({ frameNumber, VK_NULL_HANDLE, VK_NULL_HANDLE, staging, stagingAlloc });
      st.textureFailures++;
      return false;
    }
    VkImageViewCreateInfo vci{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = t.image;
    vci.viewType = src.cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
    vci.format = tf.format;
    vci.components = tf.swizzle;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, src.levels, 0, layers };
    check(vk.vkCreateImageView(device, &vci, nullptr, &t.view), "vkCreateImageView(texture)");

    VkCommandBuffer cb = uploadCmd();
    VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    b.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    b.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = t.image;
    b.subresourceRange = vci.subresourceRange;
    VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    vk.vkCmdPipelineBarrier2(cb, &di);
    vk.vkCmdCopyBufferToImage(cb, staging, t.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, uint32_t(regions.size()),
                              regions.data());
    b.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vk.vkCmdPipelineBarrier2(cb, &di);
    addGarbage({ frameNumber, VK_NULL_HANDLE, VK_NULL_HANDLE, staging, stagingAlloc });

    auto it = textures.find(key);
    if (it != textures.end()) {
      st.textureBytes -= it->second.bytes;
      releaseTexture(it->second);
      it->second = t;
    } else {
      textures.emplace(key, t);
    }
    st.texturesCreated++;
    st.textureBytes += t.bytes;
    return true;
  }

  bool doReadbackTexture(TexKey key, uint32_t face, uint32_t level, std::vector<uint8_t>& out) {
    auto it = textures.find(key);
    if (it == textures.end() || face >= it->second.layers || level >= it->second.levels) return false;
    Tex& t = it->second;
    // Make sure the upload has executed, then copy the subresource out.
    submitFrame(VK_NULL_HANDLE, VK_NULL_HANDLE);
    waitIdle();
    if (!beginFrameIfNeeded()) return false;
    uint32_t w = std::max(t.width >> level, 1u), h = std::max(t.height >> level, 1u);
    bool bc = t.format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK || t.format == VK_FORMAT_BC2_UNORM_BLOCK ||
              t.format == VK_FORMAT_BC3_UNORM_BLOCK || t.format == VK_FORMAT_BC4_UNORM_BLOCK ||
              t.format == VK_FORMAT_BC5_UNORM_BLOCK;
    uint32_t texel = 4;
    switch (t.format) {
      case VK_FORMAT_B4G4R4A4_UNORM_PACK16: case VK_FORMAT_R5G6B5_UNORM_PACK16: texel = 2; break;
      case VK_FORMAT_R16G16B16A16_SFLOAT: texel = 8; break;
      case VK_FORMAT_R32G32B32A32_SFLOAT: texel = 16; break;
      default: break;
    }
    VkDeviceSize size = bc ? VkDeviceSize((w + 3) / 4) * ((h + 3) / 4) *
                                 (t.format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK || t.format == VK_FORMAT_BC4_UNORM_BLOCK ? 8 : 16)
                           : VkDeviceSize(w) * h * texel;
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo bai{};
    bai.usage = VMA_MEMORY_USAGE_AUTO;
    bai.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VmaAllocationInfo info{};
    if (!check(vmaCreateBuffer(allocator, &bci, &bai, &buf, &alloc, &info), "vmaCreateBuffer(readback)")) return false;
    VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    b.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    b.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = t.image;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, level, 1, face, 1 };
    VkDependencyInfo di{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    di.imageMemoryBarrierCount = 1;
    di.pImageMemoryBarriers = &b;
    VkCommandBuffer cb = slot().cmd;
    vk.vkCmdPipelineBarrier2(cb, &di);
    VkBufferImageCopy r{};
    r.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, face, 1 };
    r.imageExtent = { w, h, 1 };
    vk.vkCmdCopyImageToBuffer(cb, t.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &r);
    b.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    b.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vk.vkCmdPipelineBarrier2(cb, &di);
    bool ok = submitFrame(VK_NULL_HANDLE, VK_NULL_HANDLE);
    waitIdle();
    if (ok) {
      vmaInvalidateAllocation(allocator, alloc, 0, VK_WHOLE_SIZE);
      out.assign(static_cast<uint8_t*>(info.pMappedData), static_cast<uint8_t*>(info.pMappedData) + size);
    }
    vmaDestroyBuffer(allocator, buf, alloc);
    return ok;
  }

#include "backend_draw.inl"
#include "backend_shadow.inl"
#include "backend_env.inl"
#include "backend_ffx.inl"
#include "backend_record.inl"

  // --- CPU writes into a render target ---------------------------------------------------------------------------
  bool doWriteTarget(RtKey key, const void* data, uint32_t pitch) {
    if (!beginFrameIfNeeded()) return false;
    markCubeFace(key);
    Rt* rt = lookup(key);
    if (!rt || rt->desc.depth || !realize(key, *rt)) return false;
    const uint32_t ts = texelSizeOf(rt->format);
    if (!ts) return false;
    const uint32_t row = rt->desc.width * ts;
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = VkDeviceSize(row) * rt->desc.height;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VmaAllocationInfo info{};
    if (!check(vmaCreateBuffer(allocator, &bci, &aci, &buf, &alloc, &info), "vmaCreateBuffer(target write)")) return false;
    for (uint32_t y = 0; y < rt->desc.height; y++) {
      uint8_t* dstRow = static_cast<uint8_t*>(info.pMappedData) + size_t(y) * row;
      const uint8_t* srcRow = static_cast<const uint8_t*>(data) + size_t(y) * pitch;
      if (!rt->wide) {
        memcpy(dstRow, srcRow, row);
        continue;
      }
      // The caller writes the D3D9 layout (A8R8G8B8 / X8R8G8B8: B, G, R, A bytes); the image is RGBA16F.
      uint16_t* h = reinterpret_cast<uint16_t*>(dstRow);
      for (uint32_t x = 0; x < rt->desc.width; x++) {
        const uint8_t* t = srcRow + x * 4;
        h[x * 4 + 0] = floatToHalf(t[2] / 255.0f);
        h[x * 4 + 1] = floatToHalf(t[1] / 255.0f);
        h[x * 4 + 2] = floatToHalf(t[0] / 255.0f);
        h[x * 4 + 3] = floatToHalf(rt->desc.d3dFormat == kD3DFMT_X8R8G8B8 ? 1.0f : t[3] / 255.0f);
      }
    }
    vmaFlushAllocation(allocator, alloc, 0, VK_WHOLE_SIZE);
    const bool resume = rendering && !renderingSuspended;
    if (resume) suspendRendering();
    const VkImageLayout old = rt->layout;
    barrier(rt->image, rt->layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    VkBufferImageCopy c{};
    c.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    c.imageExtent = { rt->desc.width, rt->desc.height, 1 };
    vk.vkCmdCopyBufferToImage(slot().cmd, buf, rt->image, rt->layout, 1, &c);
    barrier(rt->image, rt->layout, old == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : old, VK_IMAGE_ASPECT_COLOR_BIT);
    addGarbage({ frameNumber, VK_NULL_HANDLE, VK_NULL_HANDLE, buf, alloc });
    st.targetWrites++;
    if (resume) resumeRendering();
    return true;
  }

  // --- StretchRect between render targets ------------------------------------------------------------------------
  void doCopyTarget(RtKey srcK, RtKey dstK, const int32_t* sr, const int32_t* dr, uint32_t filter) {
    markCubeFace(dstK);
    if (!beginFrameIfNeeded()) return;
    // In a scaled scene window-sized targets are their render-size twins; the rects are in window pixels.
    Rt* s = rtFor(srcK, true);
    Rt* d = rtFor(dstK, true);
    if (!s || !d || s == d || s->desc.depth != d->desc.depth) {
      st.copiesSkipped++;
      return;
    }
    if (d->twin) lowDirty.insert(dstK);
    int32_t a[4] = { 0, 0, int32_t(s->desc.width), int32_t(s->desc.height) };
    int32_t b[4] = { 0, 0, int32_t(d->desc.width), int32_t(d->desc.height) };
    auto toScene = [&](int32_t r[4]) {
      r[0] = toSceneX(r[0]);
      r[1] = toSceneY(r[1]);
      r[2] = toSceneX(r[2]);
      r[3] = toSceneY(r[3]);
    };
    if (sr) {
      memcpy(a, sr, sizeof(a));
      if (s->twin) toScene(a);
    }
    if (dr) {
      memcpy(b, dr, sizeof(b));
      if (d->twin) toScene(b);
    }
    // D3D9 fails a StretchRect whose rects leave the surfaces.
    auto inside = [](const int32_t r[4], const Rt& t) {
      return r[0] >= 0 && r[1] >= 0 && r[2] > r[0] && r[3] > r[1] && r[2] <= int32_t(t.desc.width) &&
             r[3] <= int32_t(t.desc.height);
    };
    if (!inside(a, *s) || !inside(b, *d)) {
      st.copiesSkipped++;
      return;
    }
    const bool resume = rendering && !renderingSuspended;
    if (resume) suspendRendering();
    VkCommandBuffer cmd = slot().cmd;
    const VkImageAspectFlags as = aspectOf(*s), ad = aspectOf(*d);
    const VkImageLayout sOld = s->layout, dOld = d->layout;
    barrier(s->image, s->layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, as);
    barrier(d->image, d->layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, ad);
    const int32_t w = a[2] - a[0], h = a[3] - a[1];
    if (s->format == d->format && w == b[2] - b[0] && h == b[3] - b[1]) {
      VkImageCopy c{};
      c.srcSubresource = { as, 0, 0, 1 };
      c.dstSubresource = { ad, 0, 0, 1 };
      c.srcOffset = { a[0], a[1], 0 };
      c.dstOffset = { b[0], b[1], 0 };
      c.extent = { uint32_t(w), uint32_t(h), 1 };
      vk.vkCmdCopyImage(cmd, s->image, s->layout, d->image, d->layout, 1, &c);
    } else {
      VkImageBlit bl{};
      bl.srcSubresource = { as, 0, 0, 1 };
      bl.dstSubresource = { ad, 0, 0, 1 };
      bl.srcOffsets[0] = { a[0], a[1], 0 };
      bl.srcOffsets[1] = { a[2], a[3], 1 };
      bl.dstOffsets[0] = { b[0], b[1], 0 };
      bl.dstOffsets[1] = { b[2], b[3], 1 };
      // D3DTEXF_LINEAR (2) filters; NONE/POINT sample the nearest texel. Depth blits must be nearest.
      const VkFilter f = filter == 2 && !s->desc.depth ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
      vk.vkCmdBlitImage(cmd, s->image, s->layout, d->image, d->layout, 1, &bl, f);
    }
    // Back to where they were (attachment layouts while a group is suspended), GENERAL if they were fresh.
    barrier(s->image, s->layout, sOld == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : sOld, as);
    barrier(d->image, d->layout, dOld == VK_IMAGE_LAYOUT_UNDEFINED ? VK_IMAGE_LAYOUT_GENERAL : dOld, ad);
    st.copies++;
    if (resume) resumeRendering();
  }

  // --- pixel probe (diagnostics) -------------------------------------------------------------------------------
  bool doProbePixel(uint32_t x, uint32_t y, std::vector<uint8_t>& texel, uint32_t& vkFormat) {
    if (!rendering || groupColors.empty() || !frameOpen) return false;
    Rt& rt = *rtFor(groupColors[0], groupLow);
    const uint32_t ts = texelSizeOf(rt.format);
    if (!ts || x >= rt.desc.width || y >= rt.desc.height) return false;
    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = 16;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer buf = VK_NULL_HANDLE;
    VmaAllocation alloc = nullptr;
    VmaAllocationInfo info{};
    if (!check(vmaCreateBuffer(allocator, &bci, &aci, &buf, &alloc, &info), "vmaCreateBuffer(probe)")) return false;
    suspendRendering();
    VkCommandBuffer cmd = slot().cmd;
    const bool ms = rt.msImage != VK_NULL_HANDLE;
    const VkImageLayout attachLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    if (ms) {
      // 1x1 resolve of the multisampled attachment into the same texel of the single-sample image (the whole image is
      // resolved again at the end of the group).
      barrier(rt.msImage, rt.msLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
      barrier(rt.image, rt.layout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
      VkImageResolve r{};
      r.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
      r.dstSubresource = r.srcSubresource;
      r.srcOffset = r.dstOffset = { int32_t(x), int32_t(y), 0 };
      r.extent = { 1, 1, 1 };
      vk.vkCmdResolveImage(cmd, rt.msImage, rt.msLayout, rt.image, rt.layout, 1, &r);
    }
    barrier(rt.image, rt.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT);
    VkBufferImageCopy c{};
    c.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    c.imageOffset = { int32_t(x), int32_t(y), 0 };
    c.imageExtent = { 1, 1, 1 };
    vk.vkCmdCopyImageToBuffer(cmd, rt.image, rt.layout, buf, 1, &c);
    // Back to the attachment layouts, then submit and wait; the next command buffer resumes the render pass.
    if (ms) barrier(rt.msImage, rt.msLayout, attachLayout, VK_IMAGE_ASPECT_COLOR_BIT);
    barrier(rt.image, rt.layout, ms ? VK_IMAGE_LAYOUT_GENERAL : attachLayout, VK_IMAGE_ASPECT_COLOR_BIT);
    // The group continues after the probe: hide it from submitFrame (which would end it), resume it afterwards.
    rendering = false;
    renderingSuspended = false;
    const bool ok = submitFrame(VK_NULL_HANDLE, VK_NULL_HANDLE);
    waitIdle();
    if (ok) {
      vmaInvalidateAllocation(allocator, alloc, 0, VK_WHOLE_SIZE);
      texel.assign(static_cast<uint8_t*>(info.pMappedData), static_cast<uint8_t*>(info.pMappedData) + ts);
      vkFormat = uint32_t(rt.format);
    }
    vmaDestroyBuffer(allocator, buf, alloc);
    // submitFrame ended the frame: start the next command buffer inside the same group.
    rendering = beginFrameIfNeeded();
    if (rendering) beginRenderingInstance();
    return ok;
  }

  // --- readback ------------------------------------------------------------------------------------------------
  bool doReadback(RtKey key, std::vector<uint8_t>& out, uint32_t& vkFormat, uint32_t& texelSize) {
    auto it = rts.find(key);
    if (it == rts.end() || !it->second.realized || !initialized) return false;
    Rt& rt = it->second;
    if (rt.desc.depth && rt.desc.samples > 1) return false;  // MSAA depth has no single-sample copy
    texelSize = texelSizeOf(rt.format);
    vkFormat = uint32_t(rt.format);
    if (!texelSize) return false;
    VkDeviceSize size = VkDeviceSize(rt.desc.width) * rt.desc.height * texelSize;

    VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkBuffer buf = VK_NULL_HANDLE;
    if (!check(vk.vkCreateBuffer(device, &bci, nullptr, &buf), "vkCreateBuffer")) return false;
    VkMemoryRequirements req{};
    vk.vkGetBufferMemoryRequirements(device, buf, &req);
    VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = memoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkDeviceMemory mem = VK_NULL_HANDLE;
    bool ok = mai.memoryTypeIndex != UINT32_MAX && check(vk.vkAllocateMemory(device, &mai, nullptr, &mem), "vkAllocateMemory") &&
              check(vk.vkBindBufferMemory(device, buf, mem, 0), "vkBindBufferMemory");
    if (ok) {
      // Recorded into the current frame, which is then submitted and waited for.
      if (!beginFrameIfNeeded()) ok = false;
    }
    if (ok) {
      if (rendering) doEndGroup();
      barrier(rt.image, rt.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, aspectOf(rt));
      VkBufferImageCopy c{};
      c.imageSubresource = { VkImageAspectFlags(rt.desc.depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1 };
      c.imageExtent = { rt.desc.width, rt.desc.height, 1 };
      vk.vkCmdCopyImageToBuffer(slot().cmd, rt.image, rt.layout, buf, 1, &c);
      ok = submitFrame(VK_NULL_HANDLE, VK_NULL_HANDLE);
      waitIdle();
    }
    if (ok) {
      void* p = nullptr;
      ok = check(vk.vkMapMemory(device, mem, 0, size, 0, &p), "vkMapMemory");
      if (ok) {
        if (!rt.wide) {
          out.assign(static_cast<uint8_t*>(p), static_cast<uint8_t*>(p) + size);
        } else {
          // Wide target: returned in its D3D9 layout (8-bit B, G, R, A), values clamped to 0..1 as the D3D9 surface holds.
          const size_t texels = size_t(rt.desc.width) * rt.desc.height;
          out.resize(texels * 4);
          const uint16_t* h = static_cast<const uint16_t*>(p);
          auto unorm = [](float v) { return uint8_t(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f); };
          for (size_t i = 0; i < texels; i++) {
            out[i * 4 + 0] = unorm(halfToFloat(h[i * 4 + 2]));
            out[i * 4 + 1] = unorm(halfToFloat(h[i * 4 + 1]));
            out[i * 4 + 2] = unorm(halfToFloat(h[i * 4 + 0]));
            out[i * 4 + 3] = rt.desc.d3dFormat == kD3DFMT_X8R8G8B8 ? 0xFF : unorm(halfToFloat(h[i * 4 + 3]));
          }
          texelSize = 4;
          vkFormat = uint32_t(VK_FORMAT_B8G8R8A8_UNORM);
        }
        vk.vkUnmapMemory(device, mem);
      }
    }
    if (mem) vk.vkFreeMemory(device, mem, nullptr);
    vk.vkDestroyBuffer(device, buf, nullptr);
    return ok;
  }
};

Backend::Backend() : m(std::make_unique<Impl>()) {}
Backend::~Backend() { shutdown(); }

// With the record thread (BackendConfig::recordThread) every call below goes through m->post (queued, run in order on
// the worker) or m->call (run on the worker after the queue drains, the caller waits); without it both run the call now.
// Queued calls copy what their arguments point to. Their results are what the caller can know: bool results nobody
// reads are true, and beginShadowMaps is true (a shadow pass that cannot begin leaves no valid shadow on the worker:
// the lighting is unshadowed, as when it fails without the thread).

void Backend::attach(RtKey key, const RtDesc& desc) {
  m->post([this, key, desc] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    m->st.attached++;
    auto it = m->rts.find(key);
    if (it != m->rts.end()) {
      if (it->second.desc == desc) return;
      m->release(it->second, true);
      it->second.desc = desc;
      m->markCubeFace(key);
      return;
    }
    m->rts.emplace(key, Impl::Rt{ desc });
  });
}

bool Backend::destroy(RtKey key) {
  auto body = [this, key] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    auto it = m->rts.find(key);
    if (it == m->rts.end()) return false;
    m->st.destroyed++;
    m->detachCubeFace(key);
    m->release(it->second, true);
    m->dropTwin(key);
    // A destroyed target can no longer be reopened as "still bound".
    m->lastColors.erase(std::remove(m->lastColors.begin(), m->lastColors.end(), key), m->lastColors.end());
    if (m->lastDepth == key) m->lastDepth = 0;
    m->rts.erase(it);
    return true;
  };
  if (!m->rec) return body();
  m->post([body] { body(); });
  return true;
}

void Backend::resize(RtKey key, uint32_t width, uint32_t height) {
  m->post([this, key, width, height] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    auto it = m->rts.find(key);
    if (it == m->rts.end()) return;
    m->st.resized++;
    if (it->second.desc.width == width && it->second.desc.height == height) return;
    RtDesc d = it->second.desc;
    d.width = width;
    d.height = height;
    m->release(it->second, true);
    it->second.desc = d;
    m->markCubeFace(key);
  });
}

bool Backend::init(const BackendConfig& cfg, uint32_t w, uint32_t h) {
  {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) return true;
    m->cfg = cfg;
    m->screenW = w;
    m->screenH = h;
    if (!m->initVulkan(w, h)) {
      m->teardown();
      return false;
    }
    m->initialized = true;
    m->readyFlag = true;
  }
  // Calls made before init (attach) ran directly; from here on they are queued.
  if (cfg.recordThread) {
    m->rec = std::make_unique<Impl::Recorder>(*m);
    m->rec->publishStats();
    m->rec->start();
  }
  return true;
}

void Backend::shutdown() {
  if (!m) return;
  if (m->rec) {
    m->rec->stop();  // runs everything queued
    m->rec.reset();
  }
  std::lock_guard<std::mutex> lock(m->mutex);
  m->readyFlag = false;
  m->teardown();
}

bool Backend::ready() const { return m->readyFlag.load(std::memory_order_acquire); }

void Backend::beginGroup(const RtKey* colors, uint32_t colorCount, RtKey depth) {
  std::array<RtKey, 8> c{};
  const uint32_t n = std::min<uint32_t>(colorCount, 8);
  for (uint32_t i = 0; i < n; i++) c[i] = colors[i];
  m->post([this, c, n, depth] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    m->doBeginGroup(c.data(), n, depth);
  });
}

void Backend::clear(uint32_t flags, uint32_t argb, float z, uint32_t stencil, const int32_t rect[4]) {
  std::array<int32_t, 4> r{};
  const bool hasRect = rect != nullptr;
  if (hasRect) memcpy(r.data(), rect, sizeof(r));
  m->post([this, flags, argb, z, stencil, r, hasRect] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    m->doClear(flags, argb, z, stencil, hasRect ? r.data() : nullptr);
  });
}

void Backend::endGroup() {
  m->post([this] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    m->doEndGroup();
  });
}

void Backend::beginDepthPrepass() {
  m->post([this] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) m->doBeginDepthPrepass();
  });
}
void Backend::endDepthPrepass() {
  m->post([this] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) m->doEndDepthPrepass();
  });
}

void Backend::present(RtKey onscreen) {
  if (m->rec) {
    m->rec->present(onscreen);
    return;
  }
  std::lock_guard<std::mutex> lock(m->mutex);
  FpuGuard fpu;
  m->doPresent(onscreen);
}

void Backend::finishFrame() {
  if (m->rec) m->rec->frames.fetch_add(1, std::memory_order_relaxed);
  m->call([this] {
    {
      std::lock_guard<std::mutex> lock(m->mutex);
      FpuGuard fpu;
      m->submitFrame(VK_NULL_HANDLE, VK_NULL_HANDLE);
      m->waitIdle();
    }
    if (m->rec) m->rec->pollQueries();  // everything has completed: publish the queries' results
  });
}

bool Backend::readback(RtKey key, std::vector<uint8_t>& texels, uint32_t& vkFormat, uint32_t& texelSize) {
  return m->call([&] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    return m->doReadback(key, texels, vkFormat, texelSize);
  });
}

bool Backend::createTexture(TexKey key, const d3dx::Texture& tex) {
  if (!m->rec) {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    return m->initialized && m->doCreateTexture(key, tex);
  }
  return createTexture(key, d3dx::Texture(tex));
}

// With the record thread the upload is queued, not waited for: waiting would stall the game's main thread for up to a
// frame per texture while it streams a cell. Draws queued after this call see the texture (queue order). The result
// is what doCreateTexture decides before touching the device (texFormatFor is pure); an allocation failure on the
// worker is only counted in BackendStats::textureFailures, and draws then see the texture as unknown.
// The decoded data queued is bounded (Recorder::kTexQueueBudget): past it the caller waits for the queue to drain. Each
// upload is handed to the worker at once (not held for kPublishEvery commands), so its data is not kept longer than needed.
bool Backend::createTexture(TexKey key, d3dx::Texture&& tex) {
  if (!m->rec) return createTexture(key, static_cast<const d3dx::Texture&>(tex));
  auto& r = *m->rec;
  const bool ok = Impl::texFormatFor(tex.format).format != VK_FORMAT_UNDEFINED && !tex.surfaces.empty();
  uint64_t bytes = 0;
  for (const d3dx::Level& l : tex.surfaces) bytes += l.data.size();
  const uint64_t queued = r.texQueued.load(std::memory_order_acquire);
  if (queued && queued + bytes > r.kTexQueueBudget) {
    r.texWaits.fetch_add(1, std::memory_order_relaxed);
    m->call([] {});  // every upload queued before this has run and freed its decoded data
  }
  std::lock_guard<std::mutex> lk(r.pmu);
  if (ok) r.textures.insert(key);
  const uint64_t now = r.texQueued.fetch_add(bytes, std::memory_order_acq_rel) + bytes;
  for (uint64_t peak = r.texQueuedPeak.load(std::memory_order_relaxed);
       now > peak && !r.texQueuedPeak.compare_exchange_weak(peak, now, std::memory_order_relaxed);) {
  }
  r.postLocked(0, [](uint8_t*) {}, [this, key, bytes, t = std::move(tex)](const uint8_t*) mutable {
    {
      std::lock_guard<std::mutex> lock(m->mutex);
      FpuGuard fpu;
      if (m->initialized) m->doCreateTexture(key, t);  // (an unmapped format is logged and counted there)
    }
    t = d3dx::Texture{};  // the decoded data is freed before the queue stops counting it
    m->rec->texQueued.fetch_sub(bytes, std::memory_order_acq_rel);
  });
  r.publishLocked();
  return ok;
}

void Backend::attachCube(TexKey cube, const RtKey faces[6], uint32_t levels) {
  std::array<RtKey, 6> f{};
  for (uint32_t i = 0; i < 6; i++) f[i] = faces[i];
  m->post([this, cube, f, levels] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    auto old = m->cubes.find(cube);
    if (old != m->cubes.end())
      for (RtKey k : old->second.faces)
        if (k) m->cubeFaces.erase(k);
    Impl::RenderedCube& c = m->cubes[cube];
    c.levels = levels;
    c.dirty = 0x3F;
    for (uint32_t i = 0; i < 6; i++) {
      c.faces[i] = f[i];
      if (f[i]) m->cubeFaces[f[i]] = { cube, i };
    }
  });
}

bool Backend::updateCube(TexKey cube) {
  return m->call([&] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    auto it = m->cubes.find(cube);
    return m->initialized && it != m->cubes.end() && m->beginFrameIfNeeded() && m->assembleCube(cube, it->second);
  });
}

bool Backend::readbackEffectImage(uint32_t id, uint32_t index, std::vector<uint8_t>& texels, uint32_t& w, uint32_t& h) {
  return m->call([&] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    return m->initialized && m->doReadbackEffectImage(id, index, texels, w, h);
  });
}

void Backend::setAoLighting(const AoLighting* a) {
  const bool set = a != nullptr;
  AoLighting copy{};
  if (a) copy = *a;
  m->post([this, set, copy] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    m->aoLightingSet = set;
    if (set) m->aoLighting = copy;
    m->aoBlockKey = {};  // recomputed by the next AO-lit draw
  });
}

void Backend::setEnvironment(TexKey cube) {
  m->post([this, cube] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    m->doSetEnvironment(cube);
  });
}

float Backend::projectEnvironment(const uint16_t* texels, uint32_t size, float out[9][4]) {
  return Impl::envProject(texels, size, out);
}

bool Backend::destroyTexture(TexKey key) {
  auto body = [this, key] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    auto it = m->textures.find(key);
    if (it == m->textures.end()) return false;
    m->st.texturesDestroyed++;
    m->st.textureBytes -= it->second.bytes;
    m->releaseTexture(it->second);
    m->textures.erase(it);
    return true;
  };
  if (!m->rec) return body();
  bool existed;
  {
    std::lock_guard<std::mutex> lk(m->rec->pmu);
    existed = m->rec->textures.erase(key) != 0;
    m->rec->postLocked(0, [](uint8_t*) {}, [body](const uint8_t*) { body(); });
  }
  return existed;
}

bool Backend::readbackTexture(TexKey key, uint32_t face, uint32_t level, std::vector<uint8_t>& data) {
  return m->call([&] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    return m->initialized && m->doReadbackTexture(key, face, level, data);
  });
}

bool Backend::createBuffer(BufKey key, uint32_t size, bool index, bool index32) {
  auto body = [this, key, size, index, index32] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    return m->initialized && m->doCreateBuffer(key, size, index, index32);
  };
  if (!m->rec) return body();
  m->post([body] { body(); });
  return true;
}

bool Backend::destroyBuffer(BufKey key) {
  auto body = [this, key] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    auto it = m->buffers.find(key);
    if (it == m->buffers.end()) return false;
    m->releaseBuf(it->second);
    m->buffers.erase(it);
    m->objectGen++;
    return true;
  };
  if (!m->rec) return body();
  m->post([body] { body(); });
  return true;
}

bool Backend::writeBuffer(BufKey key, uint32_t offset, const void* data, uint32_t size, uint32_t lockFlags) {
  if (!m->rec) {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    return m->initialized && m->doWriteBuffer(key, offset, data, size, lockFlags);
  }
  const bool hasData = data != nullptr;
  m->rec->postWith(
      hasData ? size : 0, [&](uint8_t* p) { if (hasData) memcpy(p, data, size); },
      [this, key, offset, size, lockFlags, hasData](const uint8_t* p) {
        std::lock_guard<std::mutex> lock(m->mutex);
        FpuGuard fpu;
        if (m->initialized) m->doWriteBuffer(key, offset, hasData ? p : nullptr, size, lockFlags);
      });
  return true;
}

bool Backend::setShaderConstants(const uint8_t md5[16], const ShaderReplacement::Constant* constants, uint32_t count) {
  Md5 h;
  memcpy(h.b.data(), md5, 16);
  std::vector<ShaderReplacement::Constant> c(constants, constants + count);
  auto body = [this, h, c] {
    std::lock_guard<std::mutex> lock(m->mutex);
    PackedShader* p = m->pack.findMutable(h);
    if (!p) return false;
    for (const ShaderReplacement::Constant& k : c) {
      bool found = false;
      for (PackedShader::Def& d : p->defs)
        if (d.reg == k.reg) { memcpy(d.v, k.v, 16); found = true; }
      if (!found) p->defs.push_back({ k.reg, { k.v[0], k.v[1], k.v[2], k.v[3] } });
    }
    m->lastVsBlock.valid = m->lastPsBlock.valid = false;  // the cached constant blocks hold the old values
    return true;
  };
  if (!m->rec) return body();
  const bool known = m->pack.find(h) != nullptr;  // the pack's entries never change (only their defs, on the worker)
  m->post([body] { body(); });
  return known;
}

bool Backend::createShader(ShaderKey key, const uint8_t md5[16]) {
  Md5 h;
  memcpy(h.b.data(), md5, 16);
  auto body = [this, key, h] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    const PackedShader* p = m->pack.find(h);
    m->shaders[key] = Impl::Shader{ p, VK_NULL_HANDLE };
    m->objectGen++;  // a re-created key may name another pack entry
    if (p) m->st.shadersCreated++;
    else m->st.shadersMissing++;
    return p != nullptr;
  };
  if (!m->rec) return body();
  const PackedShader* p = m->pack.find(h);
  {
    std::lock_guard<std::mutex> lk(m->rec->pmu);
    m->rec->shaders[key] = p;
    m->rec->forgetShaders();
    m->rec->postLocked(0, [](uint8_t*) {}, [body](const uint8_t*) { body(); });
  }
  return p != nullptr;
}

bool Backend::destroyShader(ShaderKey key) {
  auto body = [this, key] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    m->objectGen++;
    return m->shaders.erase(key) != 0;
  };
  if (!m->rec) return body();
  bool existed;
  {
    std::lock_guard<std::mutex> lk(m->rec->pmu);
    existed = m->rec->shaders.erase(key) != 0;
    m->rec->forgetShaders();
    m->rec->postLocked(0, [](uint8_t*) {}, [body](const uint8_t*) { body(); });
  }
  return existed;
}

void Backend::createDecl(DeclKey key, const VertexElement* elements, uint32_t count) {
  std::vector<VertexElement> e(elements, elements + count);
  m->post([this, key, e] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    m->objectGen++;  // the attribute matching of a re-created declaration is redone
    m->decls[key] = e;
  });
}

bool Backend::destroyDecl(DeclKey key) {
  auto body = [this, key] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    m->objectGen++;
    return m->decls.erase(key) != 0;
  };
  if (!m->rec) return body();
  m->post([body] { body(); });
  return true;
}

void Backend::draw(const DrawSetup& state, uint32_t pt, uint32_t prims, bool indexed, int32_t baseVertex,
                   uint32_t start) {
  if (m->rec) {
    m->rec->draw(state, pt, prims, indexed, baseVertex, start);
    return;
  }
  std::lock_guard<std::mutex> lock(m->mutex);
  FpuGuard fpu;
  if (m->initialized) m->doDraw(state, pt, prims, indexed, baseVertex, start);
}

bool Backend::probePixel(uint32_t x, uint32_t y, std::vector<uint8_t>& texel, uint32_t& vkFormat) {
  return m->call([&] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    return m->initialized && m->doProbePixel(x, y, texel, vkFormat);
  });
}

uint32_t Backend::createQuery() {
  if (!m->rec) {
    std::lock_guard<std::mutex> lock(m->mutex);
    uint32_t id = m->nextQueryId++;
    m->queries[id] = Impl::Query{};
    return id;
  }
  std::lock_guard<std::mutex> lk(m->rec->pmu);
  const uint32_t id = m->rec->nextQueryId++;
  m->rec->queryEnds[id] = 0;
  m->rec->postLocked(0, [](uint8_t*) {}, [this, id](const uint8_t*) {
    std::lock_guard<std::mutex> lock(m->mutex);
    m->queries[id] = Impl::Query{};
  });
  return id;
}
void Backend::destroyQuery(uint32_t id) {
  auto body = [this, id] {
    std::lock_guard<std::mutex> lock(m->mutex);
    if (m->activeQuery == id) m->endActiveQuery();
    m->queries.erase(id);
  };
  if (!m->rec) return body();
  std::lock_guard<std::mutex> lk(m->rec->pmu);
  m->rec->queryEnds.erase(id);
  m->rec->postLocked(0, [](uint8_t*) {}, [this, body, id](const uint8_t*) {
    body();
    Impl::Recorder& r = *m->rec;
    for (size_t i = 0; i < r.pending.size();)
      if (r.pending[i].id == id) { r.pending[i] = r.pending.back(); r.pending.pop_back(); }
      else i++;
    std::lock_guard<std::mutex> pl(r.pubmu);
    r.results.erase(id);
  });
}
void Backend::beginQuery(uint32_t id) {
  m->post([this, id] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) m->doBeginQuery(id);
  });
}
void Backend::endQuery(uint32_t id) {
  if (!m->rec) {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) m->doEndQuery(id);
    return;
  }
  std::lock_guard<std::mutex> lk(m->rec->pmu);
  auto e = m->rec->queryEnds.find(id);
  if (e == m->rec->queryEnds.end()) return;  // not created (as doEndQuery ignores it)
  const uint64_t seq = ++e->second;
  m->rec->postLocked(0, [](uint8_t*) {}, [this, id, seq](const uint8_t*) {
    {
      std::lock_guard<std::mutex> lock(m->mutex);
      FpuGuard fpu;
      if (m->initialized) m->doEndQuery(id);
    }
    // Its result is published when the GPU has it (Recorder::pollQueries, after every chunk).
    Impl::Recorder& r = *m->rec;
    for (auto& p : r.pending)
      if (p.id == id) { p.seq = seq; return; }
    r.pending.push_back({ id, seq });
  });
}
int Backend::queryResult(uint32_t id, uint32_t& samples) {
  if (!m->rec) {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    return m->initialized ? m->doQueryResult(id, samples) : 0;
  }
  uint64_t seq;
  {
    std::lock_guard<std::mutex> lk(m->rec->pmu);
    auto e = m->rec->queryEnds.find(id);
    if (e == m->rec->queryEnds.end()) return 0;  // unknown query
    seq = e->second;
  }
  if (seq == 0) {  // never ended: as a query that was never issued
    samples = 0;
    return 1;
  }
  {
    std::lock_guard<std::mutex> pl(m->rec->pubmu);
    auto r = m->rec->results.find(id);
    if (r != m->rec->results.end() && r->second.seq == seq) {
      samples = r->second.samples;
      return 1;
    }
  }
  // Not yet: its end may still be in the chunk being filled; hand that to the worker so the result can come.
  std::lock_guard<std::mutex> lk(m->rec->pmu);
  m->rec->publishLocked();
  return 0;
}

void Backend::copyTarget(RtKey src, RtKey dst, const int32_t* srcRect, const int32_t* dstRect, uint32_t filter) {
  std::array<int32_t, 4> a{}, b{};
  const bool hasA = srcRect != nullptr, hasB = dstRect != nullptr;
  if (hasA) memcpy(a.data(), srcRect, sizeof(a));
  if (hasB) memcpy(b.data(), dstRect, sizeof(b));
  m->post([this, src, dst, a, b, hasA, hasB, filter] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) m->doCopyTarget(src, dst, hasA ? a.data() : nullptr, hasB ? b.data() : nullptr, filter);
  });
}

bool Backend::updateTexture2D(TexKey key, uint32_t d3dFormat, uint32_t width, uint32_t height, const void* data,
                              uint32_t pitch) {
  if (!data || !width || !height) return false;
  d3dx::Texture t;
  t.format = d3dFormat;
  t.width = width;
  t.height = height;
  t.levels = 1;
  t.cube = false;
  d3dx::Level l;
  l.width = width;
  l.height = height;
  const uint32_t bpp = d3dFormat == 116 ? 16 : d3dFormat == 113 ? 8 : 4;  // A32B32G32R32F, A16B16G16R16F, else 32-bit
  l.rowPitch = width * bpp;
  l.rows = height;
  l.data.resize(size_t(l.rowPitch) * height);
  for (uint32_t y = 0; y < height; y++)
    memcpy(l.data.data() + size_t(y) * l.rowPitch, static_cast<const uint8_t*>(data) + size_t(y) * pitch, l.rowPitch);
  t.surfaces.push_back(std::move(l));
  return createTexture(key, t);
}
void Backend::uploadTexture2D(TexKey key, uint32_t d3dFormat, uint32_t width, uint32_t height, const void* data,
                              uint32_t pitch) {
  if (!data || !width || !height) return;
  d3dx::Texture t;
  t.format = d3dFormat;
  t.width = width;
  t.height = height;
  t.levels = 1;
  t.cube = false;
  d3dx::Level l;
  l.width = width;
  l.height = height;
  const uint32_t bpp = d3dFormat == 116 ? 16 : d3dFormat == 113 ? 8 : 4;
  l.rowPitch = width * bpp;
  l.rows = height;
  l.data.resize(size_t(l.rowPitch) * height);
  for (uint32_t y = 0; y < height; y++)
    memcpy(l.data.data() + size_t(y) * l.rowPitch, static_cast<const uint8_t*>(data) + size_t(y) * pitch, l.rowPitch);
  t.surfaces.push_back(std::move(l));
  m->post([this, key, t = std::move(t)] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) m->doCreateTexture(key, t);
  });
}
bool Backend::overlayBeginFrame(uint32_t& width, uint32_t& height) {
  // The menu's draw data goes to the present as a copy (overlaySubmit), so the caller does not wait for the record
  // thread. It waits only when the renderer is not ready for the swapchain's format (initOverlay creates or rebuilds
  // it, which ImGui's frame must not overlap) or the last menu frame carried texture updates for the present to apply
  // (overlaySyncNext).
  if (m->rec && !m->overlaySyncNext && m->pubOverlayReady.load()) {
    width = m->pubSwapW.load();
    height = m->pubSwapH.load();
    if (width && height) return true;
  }
  const bool ok = m->call([&] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (!m->initialized || !m->swapchain || !m->initOverlay()) return false;
    ImGui_ImplVulkan_NewFrame();
    m->publishOverlay();
    width = m->swapExtent.width;
    height = m->swapExtent.height;
    return true;
  });
  if (ok) m->overlaySyncNext = false;
  return ok;
}

void Backend::overlaySubmit(bool draw) {
  // A copy of the frame ImGui::Render made (the caller's context), drawn by the next present.
  std::shared_ptr<Impl::OverlayFrame> fr;
  ImDrawData* dd = draw && ImGui::GetCurrentContext() ? ImGui::GetDrawData() : nullptr;
  if (dd && dd->Valid) {
    fr = std::make_shared<Impl::OverlayFrame>();
    fr->dd.Valid = true;
    fr->dd.DisplayPos = dd->DisplayPos;
    fr->dd.DisplaySize = dd->DisplaySize;
    fr->dd.FramebufferScale = dd->FramebufferScale;
    for (ImDrawList* l : dd->CmdLists) fr->dd.CmdLists.push_back(l->CloneOutput());
    fr->dd.CmdListsCount = fr->dd.CmdLists.Size;
    fr->dd.TotalVtxCount = dd->TotalVtxCount;
    fr->dd.TotalIdxCount = dd->TotalIdxCount;
    fr->dd.Textures = nullptr;
    if (dd->Textures)
      for (ImTextureData* t : *dd->Textures)
        if (t->Status != ImTextureStatus_OK) fr->textures.push_back(t);
    if (!fr->textures.empty()) {
      // Pending texture updates (the font atlas): applied by the draw from ImGui's own texture objects, which the
      // caller's next ImGui frame may change, so that frame waits for this present.
      fr->dd.Textures = &fr->textures;
      m->overlaySyncNext = true;
    }
  }
  m->post([this, draw, fr] {
    std::lock_guard<std::mutex> lock(m->mutex);
    m->overlayFrame = fr;
    m->overlayPending = draw && fr && m->overlayReady;
  });
}

uint32_t Backend::createEffect(const EffectDesc& d) {
  return m->call([&] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    return m->initialized ? m->doCreateEffect(d) : 0u;
  });
}

void Backend::destroyEffect(uint32_t id) {
  m->post([this, id] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) m->doDestroyEffect(id);
  });
}

bool Backend::runEffect(uint32_t id, const EffectFrame& f, const float* params, uint32_t paramCount) {
  auto body = [this, id](const EffectFrame& fr, const float* p, uint32_t n) {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    const bool ok = m->initialized && m->doRunEffect(id, fr, p, n);
    if (ok) m->st.effectRuns++;
    else m->st.effectsSkipped++;
    return ok;
  };
  if (!m->rec) return body(f, params, paramCount);
  const bool hasParams = params != nullptr;
  const size_t bytes = hasParams ? size_t(paramCount) * 16 : 0;
  m->rec->postWith(bytes, [&](uint8_t* p) { if (hasParams) memcpy(p, params, bytes); },
                   [body, f, paramCount, hasParams](const uint8_t* p) {
                     body(f, hasParams ? reinterpret_cast<const float*>(p) : nullptr, paramCount);
                   });
  return true;
}

bool Backend::drawScreenQuads(const ScreenQuad* quads, uint32_t count, const int32_t clip[4]) {
  if (!m->rec) {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    return m->initialized && m->doDrawScreenQuads(quads, count, clip);
  }
  std::array<int32_t, 4> c{};
  const bool hasClip = clip != nullptr;
  if (hasClip) memcpy(c.data(), clip, sizeof(c));
  const size_t bytes = size_t(count) * sizeof(ScreenQuad);
  m->rec->postWith(bytes, [&](uint8_t* p) { if (bytes) memcpy(p, quads, bytes); },
                   [this, count, c, hasClip](const uint8_t* p) {
                     std::lock_guard<std::mutex> lock(m->mutex);
                     FpuGuard fpu;
                     if (m->initialized)
                       m->doDrawScreenQuads(reinterpret_cast<const ScreenQuad*>(p), count, hasClip ? c.data() : nullptr);
                   });
  return true;
}
bool Backend::writeTarget(RtKey key, const void* data, uint32_t pitch) {
  return m->call([&] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    return m->initialized && data && m->doWriteTarget(key, data, pitch);
  });
}
void Backend::setGammaRamp(const uint16_t red[256], const uint16_t green[256], const uint16_t blue[256]) {
  std::array<uint16_t, 768> ramp{};
  memcpy(ramp.data(), red, 512);
  memcpy(ramp.data() + 256, green, 512);
  memcpy(ramp.data() + 512, blue, 512);
  m->post([this, ramp] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) m->doSetGammaRamp(ramp.data(), ramp.data() + 256, ramp.data() + 512);
  });
}
void Backend::setHdr(const HdrSettings& s) {
  m->post([this, s] {
    std::lock_guard<std::mutex> lock(m->mutex);
    if (s.output != m->hdr.output) m->swapchainDirty = true;  // another format and colour space
    m->hdr = s;
    m->hdr.paperWhiteNits = std::clamp(s.paperWhiteNits, 40.0f, 1000.0f);
    m->hdr.peakNits = std::clamp(s.peakNits, m->hdr.paperWhiteNits, 10000.0f);
  });
}
HdrStatus Backend::hdrStatus() const {
  // As the current swapchain has it (published when it is created): no wait for the record thread.
  std::lock_guard<std::mutex> lk(m->pubHdrMutex);
  HdrStatus st = m->pubHdr;
  st.wideTargets = m->cfg.wideTargets;
  return st;
}
bool Backend::readbackHdrEncode(RtKey key, uint32_t output, std::vector<uint8_t>& texels, uint32_t& vkFormat) {
  return m->call([&] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    return m->initialized && m->doReadbackHdrEncode(key, output, texels, vkFormat);
  });
}
bool Backend::beginShadowMaps(uint32_t cascades, uint32_t size) {
  auto body = [this, cascades, size] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    return m->initialized && m->doBeginShadowMaps(cascades, size);
  };
  if (!m->rec) return body();
  m->post([body] { body(); });
  return true;
}
void Backend::beginShadowCascade(uint32_t index, const float viewProj[16]) {
  std::array<float, 16> vp{};
  memcpy(vp.data(), viewProj, sizeof(vp));
  m->post([this, index, vp] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) m->doBeginShadowCascade(index, vp.data());
  });
}
void Backend::drawShadowCaster(const ShadowCaster& c) {
  if (!m->rec) {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) m->doDrawShadowCaster(c);
    return;
  }
  const bool strips = c.stripLengths && c.stripCount;
  const size_t bytes = strips ? size_t(c.stripCount) * sizeof(uint16_t) : 0;
  m->rec->postWith(bytes, [&](uint8_t* p) { if (bytes) memcpy(p, c.stripLengths, bytes); },
                   [this, c, strips](const uint8_t* p) {
                     ShadowCaster k = c;
                     k.stripLengths = strips ? reinterpret_cast<const uint16_t*>(p) : nullptr;
                     std::lock_guard<std::mutex> lock(m->mutex);
                     FpuGuard fpu;
                     if (m->initialized) m->doDrawShadowCaster(k);
                   });
}
void Backend::endShadowMaps() {
  m->post([this] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) m->doEndShadowMaps();
  });
}
void Backend::setSkinBuffer(bool active) {
  m->post([this, active] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) m->doSetSkinBuffer(active);
  });
}
void Backend::setSkinDetail(TexKey key) {
  m->post([this, key] {
    std::lock_guard<std::mutex> lock(m->mutex);
    m->skinDetail = key;
  });
}
void Backend::setHairShift(TexKey key) {
  m->post([this, key] {
    std::lock_guard<std::mutex> lock(m->mutex);
    m->hairShift = key;
  });
}
void Backend::setWetLighting(const WetLighting* w) {
  const bool on = w != nullptr;
  WetLighting copy{};
  if (w) copy = *w;
  m->post([this, on, copy] {
    std::lock_guard<std::mutex> lock(m->mutex);
    m->wetLightingSet = on;
    if (on) m->wetLighting = copy;
  });
}
void Backend::setSunLighting(const SunLighting* s) {
  const bool on = s != nullptr;
  SunLighting copy{};
  if (s) copy = *s;
  if (m->rec) {
    // The gate reads the game's state at the draw: the caller asks it when it queues each draw (Recorder::draw).
    std::lock_guard<std::mutex> lk(m->rec->pmu);
    m->rec->sunOn = on;
    m->rec->gate = on ? copy.gate : nullptr;
    m->rec->gateCtx = on ? copy.ctx : nullptr;
    copy.gate = nullptr;
    copy.ctx = nullptr;
    m->rec->postLocked(0, [](uint8_t*) {}, [this, on, copy](const uint8_t*) {
      std::lock_guard<std::mutex> lock(m->mutex);
      m->sunLightingOn = on;
      if (on) m->sunLighting = copy;
    });
    return;
  }
  std::lock_guard<std::mutex> lock(m->mutex);
  m->sunLightingOn = on;
  if (on) m->sunLighting = copy;
}
bool Backend::inGroup() const {
  return m->call([&] {
    std::lock_guard<std::mutex> lock(m->mutex);
    return m->rendering;
  });
}

BackendStats Backend::stats() const {
  return m->call([&] {
    std::lock_guard<std::mutex> lock(m->mutex);
    return m->collectStats();
  });
}

BackendStats Backend::statsSnapshot() const {
  if (!m->rec) return stats();
  std::lock_guard<std::mutex> lk(m->rec->pubmu);
  return m->rec->snapshot;
}

Backend::MemoryReport Backend::memoryReport() const {
  MemoryReport r;
  r.garbageBytes = m->garbageBytes.load(std::memory_order_relaxed);
  r.hostRingBytes = m->hostRingBytes.load(std::memory_order_relaxed);
  if (m->rec) {
    r.cmdQueued = m->rec->queued.load(std::memory_order_relaxed);
    r.cmdQueuedPeak = m->rec->queuedPeak.load(std::memory_order_relaxed);
    r.cmdQueueWaits = m->rec->queueWaits.load(std::memory_order_relaxed);
    r.cmdAllocFailures = m->rec->chunkAllocFailures.load(std::memory_order_relaxed);
    r.texQueued = m->rec->texQueued.load(std::memory_order_relaxed);
    r.texQueuedPeak = m->rec->texQueuedPeak.load(std::memory_order_relaxed);
  }
  return r;
}

uint64_t Backend::frameIndex() const {
  if (m->rec) return m->rec->frames.load(std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(m->mutex);
  return m->st.frames;
}


// --- upscaling (backend_ffx.inl) ---------------------------------------------------------------------------------
void Backend::setUpscaling(const UpscaleSettings& s) {
  // Waits for the queue: the library is loaded (or not) when this returns, so upscaleStatus tells the caller.
  m->call([this, s] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) m->doSetUpscaling(s);
  });
}

UpscaleStatus Backend::upscaleStatus() const { return m->statusSnapshot(); }

void Backend::renderSize(const UpscaleSettings& s, uint32_t displayW, uint32_t displayH, uint32_t& w, uint32_t& h) {
  // ffxFsr3UpscalerGetRenderResolutionFromQualityMode: each dimension divided by the mode's ratio, truncated.
  static constexpr float kRatio[] = { 1.0f, 1.5f, 1.7f, 2.0f, 3.0f };
  const float ratio = s.upscaler == UpscaleSettings::Off ? 1.0f : kRatio[std::min<uint32_t>(s.quality, 4)];
  w = uint32_t(float(displayW) / ratio);
  h = uint32_t(float(displayH) / ratio);
}

void Backend::jitterOffset(uint32_t index, uint32_t renderW, uint32_t displayW, float& x, float& y) {
  // ffxFsr3UpscalerGetJitterPhaseCount and ffxFsr3UpscalerGetJitterOffset (Halton bases 2 and 3, from index 1).
  x = y = 0.0f;
  if (!renderW || !displayW) return;
  const int32_t phases = int32_t(8.0f * std::pow(float(displayW) / float(renderW), 2.0f));
  if (phases <= 0) return;
  auto halton = [](int32_t index, int32_t base) {
    float f = 1.0f, result = 0.0f;
    for (int32_t i = index; i > 0;) {
      f /= float(base);
      result = result + f * float(i % base);
      i = int32_t(std::floor(float(i) / float(base)));
    }
    return result;
  };
  const int32_t i = int32_t(index % uint32_t(phases)) + 1;
  x = halton(i, 2) - 0.5f;
  y = halton(i, 3) - 0.5f;
}

void Backend::beginScaledScene() {
  m->post([this] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) m->doBeginScaledScene();
  });
}

void Backend::captureWorldDepth() {
  m->post([this] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) m->doCaptureWorldDepth();
  });
}

void Backend::upscaleScene(RtKey color, RtKey depth, const SceneView& view) {
  m->post([this, color, depth, view] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized) m->doUpscaleScene(color, depth, view);
  });
}

void Backend::endScaledScene() {
  m->post([this] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (m->initialized && m->scaledScene && m->beginFrameIfNeeded()) m->finishScaledScene(0);
  });
}

bool Backend::readbackUpscaleInput(uint32_t which, std::vector<uint8_t>& texels, uint32_t& w, uint32_t& h) {
  return m->call([&] {
    std::lock_guard<std::mutex> lock(m->mutex);
    FpuGuard fpu;
    if (!m->initialized || which > 2) return false;
    Impl::FxImage& im = which == 0 ? m->fsrDepthImg : which == 1 ? m->motionImg : m->worldDepthImg;
    return m->readbackFxImage(im, texels, w, h);
  });
}

}  // namespace fnvvk
