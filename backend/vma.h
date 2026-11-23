#pragma once
// Vulkan Memory Allocator (third_party/VulkanMemoryAllocator, v3.5.0, MIT), configured for runtime-loaded Vulkan:
// no prototypes are linked; VMA fetches every entry point through the vkGetInstanceProcAddr/vkGetDeviceProcAddr the
// backend loaded from vulkan-1.dll.
#include "vkfn.h"

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include "vk_mem_alloc.h"
