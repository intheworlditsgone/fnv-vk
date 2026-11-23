#pragma once
// Wet world: where rain water stands on the terrain (docs/WETWORLD.md). Game-independent, so tools/backend_test runs
// exactly this code.
//
// Input: the terrain's height on a regular grid (row-major, y * nx + x), with the samples that have no height (cells
// not loaded) marked invalid. Water runs off the edges of the grid and into the unloaded cells: those are the drains.
//
// Priority-Flood (Barnes, Lehman, Mulla 2014): starting from the drains, the lowest unvisited cell is taken next; each
// neighbour's spill height is the higher of its own height and the spill height of the cell it was reached from (its
// parent: the way its water runs out). The spill height is the lowest level at which water on a cell could drain away;
// above the ground it marks a depression, filled to its brim.
//
// Basins: the connected cells (4-neighbourhood) standing below their spill height, which share one water surface.
// Runoff: water on a cell runs to its parent, and on along the parents to a drain; it stops in the first basin on that
// way. A basin's catchment is every cell whose water reaches it that way (its own cells included).
// Filling (fill and spill, without the overflow of a full basin into the next one): after rain that would stand
// `depth` units deep on flat ground, a basin holds its own cells' rain plus `runoff` of the rain on the rest of its
// catchment (the rest soaks in); its water level is the height where the volume under it in the basin equals that,
// at most the spill height. Every cell of a basin has the same level.
#include <algorithm>
#include <cstdint>
#include <queue>
#include <vector>

namespace fnvvk::puddles {

struct Map {
  uint32_t nx = 0, ny = 0;
  std::vector<float> spill;    // spill height (= the ground outside basins); kNoGround where invalid
  std::vector<uint32_t> basin; // 1-based basin of the cell, 0 outside basins
  // Per basin (index basin - 1): its cells' heights ascending, its bottom and spill heights, its catchment (cells).
  struct Basin {
    std::vector<float> heights;
    float bottom = 0, spill = 0;
    uint32_t catchment = 0;
  };
  std::vector<Basin> basins;
};
constexpr float kNoGround = -1.0e9f;
// Water deeper than this over the ground counts as a basin (game units; below it, the sampling's own noise).
constexpr float kMinDepth = 0.5f;

inline Map flood(const std::vector<float>& h, const std::vector<uint8_t>& valid, uint32_t nx, uint32_t ny) {
  Map m;
  m.nx = nx;
  m.ny = ny;
  const size_t n = size_t(nx) * ny;
  m.spill.assign(n, kNoGround);
  m.basin.assign(n, 0);
  if (nx == 0 || ny == 0 || h.size() != n || valid.size() != n) return m;
  struct Item {
    float level;
    uint32_t i;
    bool operator>(const Item& o) const { return level > o.level || (level == o.level && i > o.i); }
  };
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> open;
  std::vector<uint8_t> seen(n, 0);
  std::vector<uint32_t> parent(n, UINT32_MAX), order;
  order.reserve(n);
  auto isDrain = [&](uint32_t x, uint32_t y) {
    if (x == 0 || y == 0 || x + 1 == nx || y + 1 == ny) return true;
    return !valid[size_t(y) * nx + x - 1] || !valid[size_t(y) * nx + x + 1] || !valid[size_t(y - 1) * nx + x] ||
           !valid[size_t(y + 1) * nx + x];
  };
  for (uint32_t y = 0; y < ny; y++)
    for (uint32_t x = 0; x < nx; x++) {
      const uint32_t i = y * nx + x;
      if (!valid[i]) { seen[i] = 1; continue; }
      if (isDrain(x, y)) {
        seen[i] = 1;
        m.spill[i] = h[i];
        open.push({ h[i], i });
      }
    }
  while (!open.empty()) {
    const Item c = open.top();
    open.pop();
    order.push_back(c.i);
    const uint32_t x = c.i % nx, y = c.i / nx;
    const uint32_t nb[4] = { x > 0 ? c.i - 1 : UINT32_MAX, x + 1 < nx ? c.i + 1 : UINT32_MAX,
                             y > 0 ? c.i - nx : UINT32_MAX, y + 1 < ny ? c.i + nx : UINT32_MAX };
    for (uint32_t k : nb) {
      if (k == UINT32_MAX || seen[k]) continue;
      seen[k] = 1;
      parent[k] = c.i;
      const float level = h[k] > c.level ? h[k] : c.level;
      m.spill[k] = level;
      open.push({ level, k });
    }
  }
  std::vector<uint32_t> stack;
  for (uint32_t s = 0; s < n; s++) {
    if (!valid[s] || m.basin[s] || m.spill[s] - h[s] < kMinDepth) continue;
    m.basins.emplace_back();
    const uint32_t id = uint32_t(m.basins.size());
    Map::Basin& b = m.basins.back();
    b.spill = m.spill[s];
    stack.push_back(s);
    m.basin[s] = id;
    while (!stack.empty()) {
      const uint32_t c = stack.back();
      stack.pop_back();
      b.heights.push_back(h[c]);
      const uint32_t x = c % nx, y = c / nx;
      const uint32_t nb[4] = { x > 0 ? c - 1 : UINT32_MAX, x + 1 < nx ? c + 1 : UINT32_MAX,
                               y > 0 ? c - nx : UINT32_MAX, y + 1 < ny ? c + nx : UINT32_MAX };
      for (uint32_t k : nb)
        if (k != UINT32_MAX && valid[k] && !m.basin[k] && m.spill[k] - h[k] >= kMinDepth && m.spill[k] == b.spill) {
          m.basin[k] = id;
          stack.push_back(k);
        }
    }
    std::sort(b.heights.begin(), b.heights.end());
    b.bottom = b.heights.front();
  }
  // Catchments: in flooding order a cell's parent comes before it, so the basin a cell's water reaches is its own, or
  // (outside basins) the one its parent's water reaches.
  std::vector<uint32_t> reaches(n, 0);
  for (uint32_t c : order) {
    reaches[c] = m.basin[c] ? m.basin[c] : parent[c] != UINT32_MAX ? reaches[parent[c]] : 0;
    if (reaches[c]) m.basins[reaches[c] - 1].catchment++;
  }
  return m;
}

// A basin's water level after rain `depth` units deep (on flat ground), with `runoff` of the rain on the rest of its
// catchment reaching it. Volumes in cell areas: the volume under level L is the sum over its cells of max(0, L - h).
inline float basinLevel(const Map::Basin& b, float depth, float runoff) {
  if (b.heights.empty() || depth <= 0.0f) return b.bottom;
  const float cells = float(b.heights.size());
  const double volume = double(depth) * (cells + runoff * (float(b.catchment) - cells));
  // Walk up the sorted heights: between heights k-1 and k, k cells are under water.
  double below = 0.0;  // the sum of the heights of the cells under water
  for (size_t k = 1; k <= b.heights.size(); k++) {
    below += b.heights[k - 1];
    const double level = (volume + below) / double(k);  // k * level - below = volume
    const double next = k < b.heights.size() ? b.heights[k] : 1e30;
    if (level <= next) return float(std::min<double>(level, b.spill));
  }
  return b.spill;
}

}  // namespace fnvvk::puddles
