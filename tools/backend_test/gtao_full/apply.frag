// Multiplies the denoised visibility into the scene colour (effect.ini: Output=scene, Blend=multiply), unless the
// lighting applies it to the ambient light itself (fnv.depthRange.w 1: the lighting's physically based mode).
layout(location = 0) out vec4 color;

void main() {
  const float v = fnv.depthRange.w != 0.0 ? 1.0 : texelFetch(sampler2D(img_blurred, fnvPoint), ivec2(gl_FragCoord.xy), 0).r;
  color = vec4(v, v, v, 1.0);
}
