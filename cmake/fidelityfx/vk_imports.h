// Force-included into every source of fnv_vk_fidelityfx.dll (cmake/fidelityfx/CMakeLists.txt). The SDK calls the
// Vulkan entry points below directly, as if linked against vulkan-1.lib; the Vulkan SDK has no 32-bit import library.
// Here each is a function pointer of the same name, taken from the vulkan-1.dll the process has already loaded (the
// fnv_vk backend loads it before it loads this DLL). vulkan-1.dll exports each of them: the loader's trampolines dispatch
// on the handle passed (VkDevice, VkPhysicalDevice, VkQueue, VkCommandBuffer). A call the SDK makes that is not listed
// here fails to compile (VK_NO_PROTOTYPES), so the list is complete by construction.
#pragma once
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#ifdef __cplusplus
#include <vulkan/vulkan.h>

#define FNVVK_FFX_VK_IMPORTS(X)                                                                                         \
  X(vkAcquireNextImageKHR) X(vkAllocateCommandBuffers) X(vkAllocateDescriptorSets) X(vkAllocateMemory)                 \
  X(vkBeginCommandBuffer) X(vkBindImageMemory) X(vkCmdBeginRenderPass) X(vkCmdBindDescriptorSets) X(vkCmdBindPipeline) \
  X(vkCmdCopyImage) X(vkCmdDraw) X(vkCmdEndRenderPass) X(vkCmdFillBuffer) X(vkCmdPipelineBarrier)                      \
  X(vkCmdPushConstants) X(vkCmdSetScissor) X(vkCmdSetViewport) X(vkCreateBuffer) X(vkCreateCommandPool)                \
  X(vkCreateDescriptorPool) X(vkCreateDescriptorSetLayout) X(vkCreateFramebuffer) X(vkCreateGraphicsPipelines)         \
  X(vkCreateImage) X(vkCreateImageView) X(vkCreatePipelineLayout) X(vkCreateRenderPass) X(vkCreateSemaphore)           \
  X(vkCreateShaderModule) X(vkCreateSwapchainKHR) X(vkDestroyCommandPool) X(vkDestroyDescriptorPool)                   \
  X(vkDestroyDescriptorSetLayout) X(vkDestroyFramebuffer) X(vkDestroyImage) X(vkDestroyImageView) X(vkDestroyPipeline) \
  X(vkDestroyPipelineLayout) X(vkDestroyRenderPass) X(vkDestroySemaphore) X(vkDestroyShaderModule)                     \
  X(vkDestroySwapchainKHR) X(vkDeviceWaitIdle) X(vkEndCommandBuffer) X(vkEnumerateDeviceExtensionProperties)           \
  X(vkFreeCommandBuffers) X(vkFreeDescriptorSets) X(vkFreeMemory) X(vkGetDeviceProcAddr)                               \
  X(vkGetImageMemoryRequirements) X(vkGetPhysicalDeviceFeatures) X(vkGetPhysicalDeviceFeatures2)                       \
  X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceProperties2)            \
  X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceSurfaceSupportKHR) X(vkGetSemaphoreCounterValue)    \
  X(vkGetSwapchainImagesKHR) X(vkQueuePresentKHR) X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkResetCommandBuffer)          \
  X(vkResetCommandPool) X(vkUpdateDescriptorSets) X(vkWaitSemaphores)

#define FNVVK_FFX_DECLARE(name) extern PFN_##name name;
FNVVK_FFX_VK_IMPORTS(FNVVK_FFX_DECLARE)
#undef FNVVK_FFX_DECLARE

// True when every entry point above was found (checked before a context is created: ffx_api.cpp, patches/fidelityfx).
bool fnvvkFfxVulkanImportsResolved();
#endif
