// Motion blur, pass 2(one texel per 16 x 16 pixels): the longest blur half-vector in the tile (McGuire et al.
// "TileMax"). Tile t covers full-resolution pixels [t * size / tiles, (t + 1) * size / tiles).
layout(location = 0) out vec2 outMax;

void main() {
  const ivec2 full = textureSize(sampler2D(img_vel, fnvPoint), 0);
  const ivec2 tiles = ivec2(fnvPass.size.xy);  // (this pass's output, img_tiles)
  const ivec2 t = ivec2(gl_FragCoord.xy);
  const ivec2 lo = (t * full) / tiles, hi = ((t + 1) * full) / tiles;
  vec2 best = vec2(0.0);
  float bestLen = 0.0;
  for (int y = lo.y; y < hi.y; y++)
    for (int x = lo.x; x < hi.x; x++) {
      const vec2 v = texelFetch(sampler2D(img_vel, fnvPoint), ivec2(x, y), 0).xy;
      const float l = dot(v, v);
      if (l > bestLen) {
        bestLen = l;
        best = v;
      }
    }
  outMax = best;
}
