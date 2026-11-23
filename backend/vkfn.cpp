#include "vkfn.h"

namespace fnvvk {

bool VkFn::loadGlobal() {
  lib = LoadLibraryA("vulkan-1.dll");
  if (!lib) return false;
  vkGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(GetProcAddress(lib, "vkGetInstanceProcAddr"));
  if (!vkGetInstanceProcAddr) return false;
  bool ok = true;
#define FNVVK_LOAD(n) ok &= (n = reinterpret_cast<PFN_##n>(vkGetInstanceProcAddr(nullptr, #n))) != nullptr;
  FNVVK_GLOBAL_FUNCS(FNVVK_LOAD)
#undef FNVVK_LOAD
  return ok;
}

void VkFn::loadInstance(VkInstance i) {
#define FNVVK_LOAD(n) n = reinterpret_cast<PFN_##n>(vkGetInstanceProcAddr(i, #n));
  FNVVK_INSTANCE_FUNCS(FNVVK_LOAD)
#undef FNVVK_LOAD
}

bool VkFn::loadDevice(VkDevice d) {
  bool ok = true;
#define FNVVK_LOAD(n) ok &= (n = reinterpret_cast<PFN_##n>(vkGetDeviceProcAddr(d, #n))) != nullptr;
  FNVVK_DEVICE_FUNCS(FNVVK_LOAD)
#undef FNVVK_LOAD
  return ok;
}

}  // namespace fnvvk
