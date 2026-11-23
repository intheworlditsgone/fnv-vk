// First-chance exception logger: access violations and other faults raised in fnv_vk, the Vulkan driver/loader or
// layers are logged with the faulting address and a symbolised scan of return addresses on the faulting stack, then
// passed on unchanged. The game's own first-chance exceptions (it uses SEH internally) are logged only once per
// address, and only for fault codes, so normal operation adds nothing to the log.
#include <windows.h>
#include <dbghelp.h>
#include <tlhelp32.h>

#include <cstdio>
#include <mutex>
#include <set>
#include <string>

#include "common.h"

namespace fnvvk {
namespace {

std::mutex g_m;
std::set<DWORD> g_seen;
bool g_symInit = false;

std::string describe(DWORD64 a) {
  HMODULE mod = nullptr;
  char name[MAX_PATH] = "?";
  if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         reinterpret_cast<LPCSTR>(uintptr_t(a)), &mod) && mod) {
    GetModuleFileNameA(mod, name, sizeof(name));
    const char* base = strrchr(name, '\\');
    std::string out = base ? base + 1 : name;
    char buf[sizeof(SYMBOL_INFO) + 256] = {};
    auto* sym = reinterpret_cast<SYMBOL_INFO*>(buf);
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 255;
    DWORD64 disp = 0;
    char tmp[512];
    if (g_symInit && SymFromAddr(GetCurrentProcess(), a, &disp, sym))
      snprintf(tmp, sizeof(tmp), "%s!%s+0x%llX", out.c_str(), sym->Name, disp);
    else
      snprintf(tmp, sizeof(tmp), "%s+0x%llX", out.c_str(), a - reinterpret_cast<uintptr_t>(mod));
    return tmp;
  }
  return "?";
}

bool interesting(DWORD64 a) {
  std::string d = describe(a);
  return d.find("fnv_vk") != std::string::npos || d.find("nvoglv") != std::string::npos ||
         d.find("vulkan") != std::string::npos || d.find("VkLayer") != std::string::npos ||
         d.find("amdvlk") != std::string::npos || d.find("igvk") != std::string::npos;
}

// What every other thread is executing: EIP plus the first return addresses on its stack. Collected with the threads
// suspended into fixed storage (no allocation, logging or symbol work while anything is frozen), resolved after.
void logOtherThreads() {
  struct T { DWORD tid; uint32_t eip; uint32_t ret[12]; int n; };
  static T ts[128];
  int count = 0;
  HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
  if (snap == INVALID_HANDLE_VALUE) return;
  THREADENTRY32 te{ sizeof(te) };
  const DWORD self = GetCurrentThreadId(), pid = GetCurrentProcessId();
  HANDLE hs[128];
  int nh = 0;
  for (BOOL ok = Thread32First(snap, &te); ok && nh < 128; ok = Thread32Next(snap, &te)) {
    if (te.th32OwnerProcessID != pid || te.th32ThreadID == self) continue;
    HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, FALSE, te.th32ThreadID);
    if (!h) continue;
    if (SuspendThread(h) == DWORD(-1)) { CloseHandle(h); continue; }
    hs[nh] = h;
    CONTEXT c{};
    c.ContextFlags = CONTEXT_CONTROL;
    T& t = ts[count];
    t = T{ te.th32ThreadID, 0, {}, 0 };
    if (GetThreadContext(h, &c)) {
      t.eip = c.Eip;
      const uint32_t* sp = reinterpret_cast<const uint32_t*>(uintptr_t(c.Esp));
      for (int i = 0; i < 2048 && t.n < 12; i++) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (!VirtualQuery(sp + i, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT) break;
        uint32_t v = sp[i];
        if (v < 0x10000) continue;
        if (!VirtualQuery(reinterpret_cast<void*>(uintptr_t(v)), &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT ||
            !(mbi.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)))
          continue;
        t.ret[t.n++] = v;
      }
    }
    count++;
    nh++;
  }
  CloseHandle(snap);
  for (int i = 0; i < nh; i++) {
    ResumeThread(hs[i]);
    CloseHandle(hs[i]);
  }
  for (int i = 0; i < count; i++) {
    std::string line = "  other thread " + std::to_string(ts[i].tid) + ": " + describe(ts[i].eip);
    bool driver = line.find("nv") != std::string::npos;
    for (int k = 0; k < ts[i].n; k++) {
      std::string d = describe(ts[i].ret[k]);
      if (d.find("nv") != std::string::npos || d.find("d3d9") != std::string::npos || d.find("fnv_vk") != std::string::npos)
        driver = true;
      if (k < 8) line += " <- " + d;
    }
    if (driver) logf("%s", line.c_str());
  }
}

LONG CALLBACK handler(EXCEPTION_POINTERS* ep) {
  const DWORD code = ep->ExceptionRecord->ExceptionCode;
  const bool fault = code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_ILLEGAL_INSTRUCTION ||
                     code == EXCEPTION_INT_DIVIDE_BY_ZERO || code == EXCEPTION_STACK_OVERFLOW ||
                     code == 0xE06D7363 /* C++ exception */;
  if (!fault) return EXCEPTION_CONTINUE_SEARCH;
  const DWORD64 at = reinterpret_cast<uintptr_t>(ep->ExceptionRecord->ExceptionAddress);
  std::lock_guard<std::mutex> lock(g_m);
  // C++ exceptions are raised from KERNELBASE; judge them by the throwing frames below instead.
  const CONTEXT* c = ep->ContextRecord;
  bool ours = interesting(at);
  const uint32_t* sp = reinterpret_cast<const uint32_t*>(uintptr_t(c->Esp));
  std::string frames;
  int shown = 0;
  for (int i = 0; i < 65536 && shown < 160; i++) {
    MEMORY_BASIC_INFORMATION mbi{};
    if (!VirtualQuery(sp + i, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT) break;
    uint32_t v = sp[i];
    if (v < 0x10000) continue;
    const uint8_t* code7 = reinterpret_cast<const uint8_t*>(uintptr_t(v)) - 7;
    if (!VirtualQuery(code7, &mbi, sizeof(mbi)) || mbi.State != MEM_COMMIT ||
        !(mbi.Protect & (PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY)))
      continue;
    bool afterCall = code7[2] == 0xE8 || (code7[5] == 0xFF && (code7[6] & 0x38) == 0x10) ||
                     (code7[4] == 0xFF && (code7[5] & 0x38) == 0x10) || (code7[1] == 0xFF && (code7[2] & 0x38) == 0x10);
    if (!afterCall) continue;
    std::string d = describe(v);
    if (d == "?") continue;
    // Past the first frames, only the modules that tell whose call this was.
    if (shown >= 40 && !interesting(v)) continue;
    if (shown < 10 && interesting(v)) ours = true;
    char line[600];
    snprintf(line, sizeof(line), "\n    [esp+%04X] %s", unsigned(i * 4), d.c_str());
    frames += line;
    shown++;
  }
  if (!ours) {
    if (!g_seen.insert(DWORD(at)).second || g_seen.size() > 64) return EXCEPTION_CONTINUE_SEARCH;
  } else if (g_seen.size() > 256) {
    return EXCEPTION_CONTINUE_SEARCH;
  } else {
    g_seen.insert(DWORD(at));
  }
  char head[512];
  snprintf(head, sizeof(head), "EXCEPTION %08lX at %s (first chance, thread %lu) params %08lX %08lX | eax %08lX ebx %08lX "
           "ecx %08lX edx %08lX esi %08lX edi %08lX ebp %08lX esp %08lX",
           code, describe(at).c_str(), GetCurrentThreadId(),
           ep->ExceptionRecord->NumberParameters > 0 ? ULONG(ep->ExceptionRecord->ExceptionInformation[0]) : 0,
           ep->ExceptionRecord->NumberParameters > 1 ? ULONG(ep->ExceptionRecord->ExceptionInformation[1]) : 0,
           c->Eax, c->Ebx, c->Ecx, c->Edx, c->Esi, c->Edi, c->Ebp, c->Esp);
  logf("%s%s", head, frames.c_str());
  logf("  registers: ecx %s | esi %s | ebx %s", describe(c->Ecx).c_str(), describe(c->Esi).c_str(), describe(c->Ebx).c_str());
  if (describe(at).find("nvoglv") != std::string::npos) logOtherThreads();
  return EXCEPTION_CONTINUE_SEARCH;
}

}  // namespace

void installCrashLogger() {
  SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
  std::string dir = moduleDir();
  g_symInit = SymInitialize(GetCurrentProcess(), dir.c_str(), TRUE) != FALSE;
  AddVectoredExceptionHandler(1, handler);
  logf("crash logger installed (symbols %s)", g_symInit ? "loaded" : "unavailable");
}

}  // namespace fnvvk
