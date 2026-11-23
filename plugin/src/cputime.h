#pragma once
// CPU time of the plugin's sections of a frame, summed over the summary window (main.cpp kSummaryEvery frames) and
// logged once per window as milliseconds per frame (docs/PERFORMANCE.md). Read with the time stamp counter (about
// 10 ns a read), converted with QueryPerformanceCounter over the window. Every section runs on the game's main thread,
// so the sums are plain counters. Sections nest: a draw of the probe counts in Probe and in Draw.
#include <cstdint>
#include <string>

#include <intrin.h>

namespace fnvvk::cputime {

enum Section : uint32_t {
  Draw,          // observe::drawFromTracker: the plugin's translation of a game draw plus the backend's recording
  DrawBackend,   // of that, Backend::draw
  WorldOpaque,   // BSShaderAccumulator::FinishAccumulating PreResolveDepth of the world pass (the game's work and its draws)
  WorldBlended,  // the same, PostResolveDepth
  SunWalk,       // sun shadows: the walk of the loaded cells for casters
  SunRecord,     // sun shadows: recording the casters into the cascades
  Probe,         // the environment probe's face (the game's culling and drawing of the scene into it)
  Effects,       // the effect passes
  Present,       // Backend::present (submission, acquire, present)
  kCount
};

struct Totals {
  uint64_t ticks[kCount] = {};
  uint64_t calls[kCount] = {};
};
inline Totals g_totals;

struct Scope {
  Section s;
  uint64_t t0;
  explicit Scope(Section section) : s(section), t0(__rdtsc()) {}
  ~Scope() {
    g_totals.ticks[s] += __rdtsc() - t0;
    g_totals.calls[s]++;
  }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;
};

// The summary line for the window since the previous call (its first call only starts the window), then a new window.
std::string summary(uint32_t frames);

}  // namespace fnvvk::cputime
