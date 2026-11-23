// Sun shafts, pass 2: the half-resolution light brought to full resolution: the four nearest half-resolution texels,
// bilinear weights times view-distance similarity, so edges stay sharp.
layout(location = 0) out float outScatter;

void main() {
  outScatter = 0.0;
  const ivec2 pix = ivec2(gl_FragCoord.xy);
  if (!fnvInsideRect(pix) || fnv.shadowInfo.x == 0.0) return;
  const float d = fnvRawDepth(pix);
  const float dist = min(fnvIsSky(d) ? 1.0e9 : fnvLinearZ(d), 60000.0);  // as stored by march.frag
  const ivec2 size = textureSize(sampler2D(img_smooth, fnvPoint), 0);
  const vec2 h = (vec2(pix) + 0.5) * vec2(size) / fnv.target.xy - 0.5;
  const ivec2 h0 = ivec2(floor(h));
  const vec2 f = h - vec2(h0);
  float sum = 0.0, weights = 0.0;
  for (int y = 0; y <= 1; y++)
    for (int x = 0; x <= 1; x++) {
      const vec2 s = texelFetch(sampler2D(img_smooth, fnvPoint), clamp(h0 + ivec2(x, y), ivec2(0), size - 1), 0).rg;
      const float bilinear = (x == 0 ? 1.0 - f.x : f.x) * (y == 0 ? 1.0 - f.y : f.y);
      const float w = bilinear * max(1e-3, 1.0 - abs(s.g - dist) / (0.05 * dist + 1.0));
      sum += s.r * w;
      weights += w;
    }
  outScatter = weights > 0.0 ? sum / weights : 0.0;
}
