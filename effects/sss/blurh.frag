// Subsurface scattering, pass 2: the horizontal blur of the skin's light (img_irr), Community Shaders' SSSSBlurCS.
// Off skin the light passes unchanged.
layout(location = 0) out vec4 outBlur;

// The blur along dir (viewport uv) at skin pixel p of light colorM, from src (a: 1 on skin). The width: the kernel's
// offsets times the blur radius over the pixel's view distance, through the projection (1 / tan(fov / 2)); each tap
// turned by a per-pixel angle (interleaved gradient noise, 16 frames), so the two passes do not band along the axes.
// A tap at another depth (by more than 1 / (thickness x the projection's scale) units) or off the skin counts as the
// pixel's own light (Community Shaders blends depth steps back the same way; off-skin taps are this port's: the scene
// there has no albedo removed).
float ign(vec2 px, uint frame) {
  const float step = float(frame % 16u) * 0.0625;
  px += vec2(step * 4.7526, step * 3.1914);
  return fract(52.9829189 * fract(dot(px, vec2(0.06711056, 0.00583715))));
}
vec3 sssBlur(texture2D src, ivec2 p, vec2 dir, vec3 colorM) {
  const float depthM = fnvLinearZ(fnvRawDepth(p));
  const float proj = 2.0 / (fnv.frustum.z - fnv.frustum.w);
  const vec2 finalStep = proj / depthM * dir * p_radius;
  const float a = ign(vec2(p), uint(fnv.time.y)) * 6.2831853;
  const float c = cos(a), s = sin(a);
  const vec2 uv = (vec2(p) + 0.5 - fnv.rect.xy) / fnv.rect.zw;
  vec3 sum = colorM * texelFetch(sampler2D(img_kernel, fnvPoint), ivec2(0, 0), 0).rgb;
  for (int i = 1; i < 21; i++) {
    const vec4 k = texelFetch(sampler2D(img_kernel, fnvPoint), ivec2(i, 0), 0);
    vec2 off = k.a * finalStep;
    off = vec2(off.x * c - off.y * s, off.x * s + off.y * c);
    const ivec2 q = fnvClampRect(ivec2(floor(fnv.rect.xy + (uv + off) * fnv.rect.zw)));
    const vec4 t = texelFetch(sampler2D(src, fnvPoint), q, 0);
    float r = clamp(p_thickness * proj * abs(depthM - fnvLinearZ(fnvRawDepth(q))), 0.0, 1.0);
    if (t.a == 0.0) r = 1.0;
    sum += k.rgb * mix(t.rgb, colorM, r * r);
  }
  return sum;
}

void main() {
  const ivec2 p = ivec2(gl_FragCoord.xy);
  const vec4 m = texelFetch(sampler2D(img_irr, fnvPoint), p, 0);
  outBlur = m.a == 0.0 || p_radius <= 0.0 ? m : vec4(sssBlur(img_irr, p, vec2(1.0, 0.0), m.rgb), 1.0);
}
