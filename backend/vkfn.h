#pragma once
// Vulkan entry points, loaded at runtime from vulkan-1.dll (the SDK ships no 32-bit import library, and the plugin
// must not have a load-time dependency on Vulkan).
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#include <windows.h>
#include <vulkan/vulkan.h>

#define FNVVK_GLOBAL_FUNCS(X) \
  X(vkCreateInstance)         \
  X(vkEnumerateInstanceLayerProperties) \
  X(vkEnumerateInstanceExtensionProperties)

#define FNVVK_INSTANCE_FUNCS(X)                     \
  X(vkDestroyInstance)                              \
  X(vkEnumeratePhysicalDevices)                     \
  X(vkGetPhysicalDeviceProperties)                  \
  X(vkGetPhysicalDeviceProperties2)                 \
  X(vkGetPhysicalDeviceFeatures2)                   \
  X(vkGetPhysicalDeviceQueueFamilyProperties)       \
  X(vkGetPhysicalDeviceMemoryProperties)            \
  X(vkGetPhysicalDeviceFormatProperties)            \
  X(vkGetPhysicalDeviceImageFormatProperties)       \
  X(vkEnumerateDeviceExtensionProperties)           \
  X(vkCreateDevice)                                 \
  X(vkGetDeviceProcAddr)                            \
  X(vkCreateWin32SurfaceKHR)                        \
  X(vkDestroySurfaceKHR)                            \
  X(vkGetPhysicalDeviceSurfaceSupportKHR)           \
  X(vkGetPhysicalDeviceSurfaceCapabilitiesKHR)      \
  X(vkGetPhysicalDeviceSurfaceFormatsKHR)           \
  X(vkGetPhysicalDeviceSurfacePresentModesKHR)      \
  X(vkCreateDebugUtilsMessengerEXT)                 \
  X(vkDestroyDebugUtilsMessengerEXT)

#define FNVVK_DEVICE_FUNCS(X)          \
  X(vkDestroyDevice)                   \
  X(vkGetDeviceQueue)                  \
  X(vkDeviceWaitIdle)                  \
  X(vkQueueSubmit2)                    \
  X(vkQueueWaitIdle)                   \
  X(vkQueuePresentKHR)                 \
  X(vkCreateSwapchainKHR)              \
  X(vkDestroySwapchainKHR)             \
  X(vkGetSwapchainImagesKHR)           \
  X(vkAcquireNextImageKHR)             \
  X(vkCreateImage)                     \
  X(vkDestroyImage)                    \
  X(vkGetImageMemoryRequirements)      \
  X(vkCreateImageView)                 \
  X(vkDestroyImageView)                \
  X(vkCreateBuffer)                    \
  X(vkDestroyBuffer)                   \
  X(vkGetBufferMemoryRequirements)     \
  X(vkBindBufferMemory)                \
  X(vkAllocateMemory)                  \
  X(vkFreeMemory)                      \
  X(vkBindImageMemory)                 \
  X(vkMapMemory)                       \
  X(vkUnmapMemory)                     \
  X(vkInvalidateMappedMemoryRanges)    \
  X(vkCreateCommandPool)               \
  X(vkDestroyCommandPool)              \
  X(vkResetCommandPool)                \
  X(vkAllocateCommandBuffers)          \
  X(vkBeginCommandBuffer)              \
  X(vkEndCommandBuffer)                \
  X(vkCreateFence)                     \
  X(vkDestroyFence)                    \
  X(vkWaitForFences)                   \
  X(vkResetFences)                     \
  X(vkCreateSemaphore)                 \
  X(vkDestroySemaphore)                \
  X(vkCmdPipelineBarrier2)             \
  X(vkCmdBeginRendering)               \
  X(vkCmdEndRendering)                 \
  X(vkCmdClearAttachments)             \
  X(vkCmdBlitImage)                    \
  X(vkCmdResolveImage)                 \
  X(vkCmdCopyImageToBuffer)        \
  X(vkCmdCopyBufferToImage)        \
  X(vkCmdCopyBuffer)               \
  X(vkFlushMappedMemoryRanges)     \
  X(vkCreateShaderModule)          \
  X(vkDestroyShaderModule)         \
  X(vkCreateGraphicsPipelines)     \
  X(vkDestroyPipeline)             \
  X(vkCreatePipelineLayout)        \
  X(vkDestroyPipelineLayout)       \
  X(vkCreateDescriptorSetLayout)   \
  X(vkDestroyDescriptorSetLayout)  \
  X(vkCreateDescriptorPool)        \
  X(vkDestroyDescriptorPool)       \
  X(vkResetDescriptorPool)         \
  X(vkAllocateDescriptorSets)      \
  X(vkUpdateDescriptorSets)        \
  X(vkCreateSampler)               \
  X(vkDestroySampler)              \
  X(vkCreatePipelineCache)         \
  X(vkDestroyPipelineCache)        \
  X(vkGetPipelineCacheData)        \
  X(vkCmdBindPipeline)             \
  X(vkCmdBindDescriptorSets)       \
  X(vkCmdBindVertexBuffers)        \
  X(vkCmdBindIndexBuffer)          \
  X(vkCmdDraw)                     \
  X(vkCmdDrawIndexed)              \
  X(vkCmdSetViewport)              \
  X(vkCmdSetScissor)               \
  X(vkCmdSetDepthBias)             \
  X(vkCmdSetBlendConstants)        \
  X(vkCmdSetStencilCompareMask)    \
  X(vkCmdSetStencilWriteMask)      \
  X(vkCmdSetStencilReference)      \
  X(vkCmdSetCullMode)              \
  X(vkCmdSetFrontFace)             \
  X(vkCmdSetPrimitiveTopology)     \
  X(vkCmdSetDepthTestEnable)       \
  X(vkCmdSetDepthWriteEnable)      \
  X(vkCmdSetDepthCompareOp)        \
  X(vkCmdSetStencilTestEnable)     \
  X(vkCmdSetStencilOp)             \
  X(vkCmdSetDepthBiasEnable)       \
  X(vkCmdSetDepthBoundsTestEnable) \
  X(vkCmdFillBuffer)                \
  X(vkCmdClearColorImage)          \
  X(vkCmdClearDepthStencilImage)   \
  X(vkCmdCopyImage)                \
  X(vkCmdPushConstants)            \
  X(vkCreateQueryPool)            \
  X(vkDestroyQueryPool)           \
  X(vkResetQueryPool)             \
  X(vkCmdBeginQuery)              \
  X(vkCmdEndQuery)                \
  X(vkGetQueryPoolResults)        \
  X(vkCreateComputePipelines)     \
  X(vkCmdDispatch)

namespace fnvvk {

struct VkFn {
  HMODULE lib = nullptr;
  PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
#define FNVVK_DECL(n) PFN_##n n = nullptr;
  FNVVK_GLOBAL_FUNCS(FNVVK_DECL)
  FNVVK_INSTANCE_FUNCS(FNVVK_DECL)
  FNVVK_DEVICE_FUNCS(FNVVK_DECL)
#undef FNVVK_DECL

  bool loadGlobal();                 // false if vulkan-1.dll or a global entry point is missing
  void loadInstance(VkInstance i);   // extension entry points stay null if the extension is not enabled
  bool loadDevice(VkDevice d);       // false if a required device entry point is missing
};

}  // namespace fnvvk
