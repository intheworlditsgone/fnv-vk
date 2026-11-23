// Shader programs from the Gamebryo/BSShader layer (replaces the observer's IDirect3DDevice9::Create*Shader feed).
//
// Every shader handle the game creates comes from one of these (IDA, all callers of the device's Create*Shader):
//  - BSShader::CompileVertexShaderHLSL 0x00BE0FE0 / CompilePixelShaderHLSL 0x00BE1750 (__thiscall: this, path, macros,
//    version, filename) -> NiD3DVertexShader* (handle +0x34) / NiD3DPixelShader* (handle +0x2C). The bytecode is the
//    shader package record ShaderBuffer::Get(BSShaderManager::pShaderBufferLoader 0x011F9498, filename) 0x00B7FAC0:
//    { char name[256]; u32 size @+256; code @+260 } (.sdp format, docs/FEASIBILITY.md).
//  - NiD3DShaderProgramCreator::CreateVertexShader 0x00E98890 / CreatePixelShader 0x00E988D0 (__stdcall, code first)
//    -> handle; the code size is the token stream's (D3DXGetShaderSize).
// The observer still sees the device calls and checks each handle against what this module registered.
#include "gbshaders.h"

#include <windows.h>

#include <atomic>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "backend.h"
#include "common.h"
#include "shaderpack.h"
#include "shadow.h"

namespace fnvvk::gbshaders {

namespace {

struct Counters {
  std::atomic<uint64_t> fromPackage{ 0 }, fromCreator{ 0 }, noRecord{ 0 }, notInPack{ 0 };
} g_c;
std::mutex g_m;
std::unordered_map<void*, Md5> g_byHandle;  // shader handle -> MD5 of the bytecode the game created it from

using GetShaderSizeFn = UINT(WINAPI*)(const DWORD*);
GetShaderSizeFn g_getShaderSize = nullptr;

void registerShader(void* handle, const void* code, uint32_t size, std::atomic<uint64_t>& source) {
  if (!handle || !code || !size) return;
  Md5 m;
  if (!md5Of(code, size, m)) return;
  {
    std::lock_guard<std::mutex> lock(g_m);
    g_byHandle[handle] = m;
  }
  source++;
  if (Backend* be = shadow::backendIfReady())
    if (!be->createShader(ShaderKey(handle), m.b.data())) g_c.notInPack++;
}

// The package record for `filename`, or null (the HLSL-compile / ShaderBufferCreator paths have none).
const uint8_t* packageRecord(const char* filename) {
  void* loader = *reinterpret_cast<void**>(0x011F9498);
  if (!loader || !filename) return nullptr;
  using GetFn = const uint8_t*(__fastcall*)(void*, void*, const char*);
  return reinterpret_cast<GetFn>(0x00B7FAC0)(loader, nullptr, filename);
}

using CompileFn = void*(__fastcall*)(void*, void*, char*, const void*, const char*, const char*);
CompileFn oCompileVS, oCompilePS;
void* __fastcall hkCompileVS(void* self, void* edx, char* path, const void* macro, const char* ver, const char* file) {
  void* obj = oCompileVS(self, edx, path, macro, ver, file);
  if (obj) {
    void* handle = *reinterpret_cast<void**>(static_cast<char*>(obj) + 0x34);
    if (const uint8_t* rec = packageRecord(file)) registerShader(handle, rec + 260, *reinterpret_cast<const uint32_t*>(rec + 256), g_c.fromPackage);
    else g_c.noRecord++;
  }
  return obj;
}
void* __fastcall hkCompilePS(void* self, void* edx, char* path, const void* macro, const char* ver, const char* file) {
  void* obj = oCompilePS(self, edx, path, macro, ver, file);
  if (obj) {
    void* handle = *reinterpret_cast<void**>(static_cast<char*>(obj) + 0x2C);
    if (const uint8_t* rec = packageRecord(file)) registerShader(handle, rec + 260, *reinterpret_cast<const uint32_t*>(rec + 256), g_c.fromPackage);
    else g_c.noRecord++;
  }
  return obj;
}

using CreatorVSFn = void*(__stdcall*)(const DWORD*, void*, unsigned, void*, unsigned, bool);
using CreatorPSFn = void*(__stdcall*)(const DWORD*);
CreatorVSFn oCreatorVS;
CreatorPSFn oCreatorPS;
void* __stdcall hkCreatorVS(const DWORD* code, void* decl, unsigned a3, void* a4, unsigned a5, bool a6) {
  void* h = oCreatorVS(code, decl, a3, a4, a5, a6);
  if (h && code && g_getShaderSize) registerShader(h, code, g_getShaderSize(code), g_c.fromCreator);
  return h;
}
void* __stdcall hkCreatorPS(const DWORD* code) {
  void* h = oCreatorPS(code);
  if (h && code && g_getShaderSize) registerShader(h, code, g_getShaderSize(code), g_c.fromCreator);
  return h;
}

}  // namespace

bool installHooks() {
  if (HMODULE dx = GetModuleHandleA("d3dx9_38.dll"))
    g_getShaderSize = reinterpret_cast<GetShaderSizeFn>(GetProcAddress(dx, "D3DXGetShaderSize"));
  bool ok = hook(0x00BE0FE0, &hkCompileVS, &oCompileVS, "BSShader::CompileVertexShaderHLSL");
  ok &= hook(0x00BE1750, &hkCompilePS, &oCompilePS, "BSShader::CompilePixelShaderHLSL");
  ok &= hook(0x00E98890, &hkCreatorVS, &oCreatorVS, "NiD3DShaderProgramCreator::CreateVertexShader");
  ok &= hook(0x00E988D0, &hkCreatorPS, &oCreatorPS, "NiD3DShaderProgramCreator::CreatePixelShader");
  return ok;
}

bool lookup(void* handle, uint8_t md5[16]) {
  std::lock_guard<std::mutex> lock(g_m);
  auto it = g_byHandle.find(handle);
  if (it == g_byHandle.end()) return false;
  memcpy(md5, it->second.b.data(), 16);
  return true;
}

void logSummary() {
  logf("game-sourced shaders: from packages %llu, from creators %llu, no package record %llu, not in pack %llu",
       g_c.fromPackage.load(), g_c.fromCreator.load(), g_c.noRecord.load(), g_c.notInPack.load());
}

}  // namespace fnvvk::gbshaders
