// Depth of field, pass 5(one texel per tile): the largest near blur radius of any tile whose blur reaches this tile
// (the near layer spreads in front of what is behind it, also over in-focus pixels: its gather radius must come from
// the neighbours). The neighbourhood reaches the largest radius allowed (p_maxBlur).
layout(location = 0) out float outNear;

void main() {
  const ivec2 tiles = textureSize(sampler2D(img_tiles, fnvPoint), 0);
  const float tilePx = fnv.target.y / float(tiles.y);
  const int r = clamp(int(ceil(0.01 * p_maxBlur * fnv.rect.w / tilePx)), 1, 16);
  const ivec2 t = ivec2(gl_FragCoord.xy);
  float near = 0.0;
  for (int y = -r; y <= r; y++)
    for (int x = -r; x <= r; x++) {
      const ivec2 n = t + ivec2(x, y);
      if (any(lessThan(n, ivec2(0))) || any(greaterThanEqual(n, tiles))) continue;
      const float v = texelFetch(sampler2D(img_tiles, fnvPoint), n, 0).r;
      const float gap = max(float(max(abs(x), abs(y))) - 1.0, 0.0) * tilePx;  // pixels between the two tiles
      if (v >= gap) near = max(near, v);
    }
  outNear = near;
}
