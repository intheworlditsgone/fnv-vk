// The Vulkan entry points of vk_imports.h, from the vulkan-1.dll already loaded in the process. Resolved on first use of
// the FidelityFX API (ffxCreateContext), not in DllMain.
#include "vk_imports.h"

#include <windows.h>

#define FNVVK_FFX_DEFINE(name) PFN_##name name = nullptr;
FNVVK_FFX_VK_IMPORTS(FNVVK_FFX_DEFINE)
#undef FNVVK_FFX_DEFINE

bool fnvvkFfxVulkanImportsResolved() {
  static const bool resolved = [] {
    HMODULE vk = GetModuleHandleW(L"vulkan-1.dll");
    if (!vk) return false;
    bool all = true;
#define FNVVK_FFX_RESOLVE(name)                                                                \
  name = reinterpret_cast<PFN_##name>(reinterpret_cast<void*>(GetProcAddress(vk, #name)));    \
  all &= name != nullptr;
    FNVVK_FFX_VK_IMPORTS(FNVVK_FFX_RESOLVE)
#undef FNVVK_FFX_RESOLVE
    return all;
  }();
  return resolved;
}
