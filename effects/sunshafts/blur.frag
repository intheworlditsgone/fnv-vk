// Sun shafts, pass 1 (half resolution): removes the march's jitter noise. A 5 x 5 neighbourhood weighted by view
// distance similarity, so the light of the sky and of a nearer surface do not bleed into each other.
layout(location = 0) out vec2 outScatter;

void main() {
  const ivec2 pix = ivec2(gl_FragCoord.xy);
  const ivec2 size = textureSize(sampler2D(img_scatter, fnvPoint), 0);
  const vec2 centre = texelFetch(sampler2D(img_scatter, fnvPoint), pix, 0).rg;
  float sum = 0.0, weights = 0.0;
  for (int y = -2; y <= 2; y++)
    for (int x = -2; x <= 2; x++) {
      const vec2 s = texelFetch(sampler2D(img_scatter, fnvPoint), clamp(pix + ivec2(x, y), ivec2(0), size - 1), 0).rg;
      const float w = max(0.0, 1.0 - abs(s.g - centre.g) / (0.05 * centre.g + 1.0));
      sum += s.r * w;
      weights += w;
    }
  outScatter = vec2(weights > 0.0 ? sum / weights : centre.r, centre.g);
}
