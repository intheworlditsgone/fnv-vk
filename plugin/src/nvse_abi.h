#pragma once
// The few xNVSE plugin ABI declarations this plugin needs (layouts from xNVSE nvse/PluginAPI.h).
#include <cstdint>

namespace nvse {

struct PluginInfo {
  enum { kInfoVersion = 1 };
  uint32_t infoVersion;
  const char* name;
  uint32_t version;
};

struct Interface {
  uint32_t nvseVersion;
  uint32_t runtimeVersion;
  uint32_t editorVersion;
  uint32_t isEditor;
  // Further members are not used.
};

}  // namespace nvse
