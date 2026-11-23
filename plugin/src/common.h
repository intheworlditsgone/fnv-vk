#pragma once
#include <cstdint>
#include <string>

namespace fnvvk {

void logf(const char* fmt, ...);
std::string moduleDir();  // directory of fnv_vk.dll, with trailing separator
inline std::string iniPath() { return moduleDir() + "fnv_vk.ini"; }

// Read / write a T at byte offset `off` of a game object.
template <typename T> T rd(const void* p, uint32_t off) {
  return *reinterpret_cast<const T*>(static_cast<const char*>(p) + off);
}
template <typename T> void wr(void* p, uint32_t off, T v) { *reinterpret_cast<T*>(static_cast<char*>(p) + off) = v; }

// MinHook detour; logs the result. `original` receives the trampoline.
bool hookFunction(uintptr_t addr, void* detour, void** original, const char* name);
// The same with the detour's signature checked against the original's.
template <typename Fn> bool hook(uintptr_t addr, Fn detour, Fn* original, const char* name) {
  return hookFunction(addr, reinterpret_cast<void*>(detour), reinterpret_cast<void**>(original), name);
}

// Vectored first-chance fault logger (crashlog.cpp). Call outside the loader lock.
void installCrashLogger();

}  // namespace fnvvk
