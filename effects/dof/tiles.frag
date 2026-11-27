// Depth of field, pass 4(one texel per 16 x 16 pixels): the largest blur radius in front of the focus in the tile
// (full-resolution pixels; 0 with none). Tile t covers full-resolution pixels [t * size / tiles, (t + 1) * size /
// tiles), half-resolution texels half of that.
layout(location = 0) out float outNear;

void main() {
  const ivec2 halfSize = textureSize(sampler2D(img_half, fnvPoint), 0);
  const ivec2 tiles = ivec2(fnvPass.size.xy);  // (this pass's output, img_tiles)
  const ivec2 t = ivec2(gl_FragCoord.xy);
  const ivec2 lo = (t * halfSize) / tiles, hi = max(((t + 1) * halfSize) / tiles, lo + 1);
  float near = 0.0;
  for (int y = lo.y; y < hi.y; y++)
    for (int x = lo.x; x < hi.x; x++) near = max(near, -texelFetch(sampler2D(img_half, fnvPoint), ivec2(x, y), 0).a);
  outNear = near;
}
