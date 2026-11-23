// Auto-exposure, pass 0 (one texel per 16 x 16 pixels): the mean log2 luminance of its block, from a 4 x 4 grid of
// bilinear taps (each averages 2 x 2 pixels), in linear light (the colours are gamma 2.2 encoded).
layout(location = 0) out float outLogLum;

const vec3 kLuma = vec3(0.2126, 0.7152, 0.0722);  // BT.709

void main() {
  const vec2 centre = fnvPixel();  // the block's centre in full-resolution pixels
  float sum = 0.0;
  for (int y = 0; y < 4; y++)
    for (int x = 0; x < 4; x++) {
      const vec2 p = clamp(centre + vec2(float(x) * 4.0 - 6.0, float(y) * 4.0 - 6.0), vec2(1.0), fnv.target.xy - 1.0);
      const vec3 c = texture(sampler2D(fnvColorTex, fnvLinear), p * fnv.target.zw).rgb;
      sum += log2(max(dot(pow(max(c, 0.0), vec3(2.2)), kLuma), 1.0 / 16384.0));
    }
  outLogLum = sum / 16.0;
}
