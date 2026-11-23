#include "cputime.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <cstdio>

namespace fnvvk::cputime {

namespace {
uint64_t g_tsc0 = 0;
LARGE_INTEGER g_qpc0{};
}  // namespace

std::string summary(uint32_t frames) {
  const uint64_t tsc = __rdtsc();
  LARGE_INTEGER qpc, freq;
  QueryPerformanceCounter(&qpc);
  QueryPerformanceFrequency(&freq);
  const Totals t = g_totals;
  g_totals = Totals{};
  const bool first = g_tsc0 == 0;
  const uint64_t dtsc = tsc - g_tsc0;
  const double ms = double(qpc.QuadPart - g_qpc0.QuadPart) * 1000.0 / double(freq.QuadPart);
  g_tsc0 = tsc;
  g_qpc0 = qpc;
  if (first || !frames || !dtsc || ms <= 0.0) return std::string();
  const double msPerTick = ms / double(dtsc);
  auto per = [&](Section s) { return double(t.ticks[s]) * msPerTick / frames; };
  auto n = [&](Section s) { return double(t.calls[s]) / frames; };
  char b[768];
  snprintf(b, sizeof(b),
           "cpu ms per frame over %u frames: frame %.2f | game draws %.2f (%.0f draws: backend recording %.2f, plugin %.2f) | "
           "world pass opaque %.2f, blended %.2f (the game's work with its draws) | sun shadows: walk %.2f, recording %.2f | "
           "probe %.2f (%.2f faces) | effects %.2f | present %.2f",
           frames, ms / frames, per(Draw), n(Draw), per(DrawBackend), per(Draw) - per(DrawBackend), per(WorldOpaque),
           per(WorldBlended), per(SunWalk), per(SunRecord), per(Probe), n(Probe), per(Effects), per(Present));
  return b;
}

}  // namespace fnvvk::cputime
