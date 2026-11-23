// Auto-exposure, pass 1 (16 x 16): each texel the weighted mean of its block of aelum, and the sum of the weights. The
// weight is 1, blended by the centre weighting toward a Gaussian around the image centre (sigma a quarter of the
// height), so the subject in the middle counts more than the sky at the edges.
layout(location = 0) out vec2 outBlock;

void main() {
  const ivec2 size = textureSize(sampler2D(img_aelum, fnvPoint), 0);
  const ivec2 t = ivec2(gl_FragCoord.xy);
  const ivec2 lo = (t * size) / 16, hi = ((t + 1) * size) / 16;
  const float aspect = float(size.x) / float(size.y);
  float sum = 0.0, weights = 0.0;
  for (int y = lo.y; y < hi.y; y++)
    for (int x = lo.x; x < hi.x; x++) {
      const vec2 uv = (vec2(x, y) + 0.5) / vec2(size) - 0.5;
      const vec2 d = vec2(uv.x * aspect, uv.y);
      const float w = mix(1.0, exp(-dot(d, d) / (2.0 * 0.25 * 0.25)), clamp(p_aeCentre, 0.0, 1.0));
      sum += texelFetch(sampler2D(img_aelum, fnvPoint), ivec2(x, y), 0).r * w;
      weights += w;
    }
  outBlock = vec2(weights > 0.0 ? sum / weights : 0.0, weights);
}
