// Motion blur, pass 3(one texel per tile): the longest half-vector of the tiles around this one (McGuire et al.
// "NeighborMax"). The paper's 3 x 3 neighbourhood reaches blurs as long as a tile; here the neighbourhood reaches the
// longest blur allowed (p_maxBlur): any pixel whose streak can cover this tile's pixels is in it.
layout(location = 0) out vec2 outMax;

void main() {
  const ivec2 tiles = textureSize(sampler2D(img_tiles, fnvPoint), 0);
  const float tilePx = fnv.target.y / float(tiles.y);
  const int r = clamp(int(ceil(0.01 * p_maxBlur * fnv.rect.w / tilePx)), 1, 16);
  const ivec2 t = ivec2(gl_FragCoord.xy);
  vec2 best = vec2(0.0);
  float bestLen = 0.0;
  for (int y = -r; y <= r; y++)
    for (int x = -r; x <= r; x++) {
      const ivec2 n = t + ivec2(x, y);
      if (any(lessThan(n, ivec2(0))) || any(greaterThanEqual(n, tiles))) continue;
      const vec2 v = texelFetch(sampler2D(img_tiles, fnvPoint), n, 0).xy;
      const float l = dot(v, v);
      // Only a tile whose streak reaches this one (its length at least the gap between the tiles, in pixels).
      const float gap = max(float(max(abs(x), abs(y))) - 1.0, 0.0) * tilePx;
      if (l > bestLen && l >= gap * gap) {
        bestLen = l;
        best = v;
      }
    }
  outMax = best;
}
