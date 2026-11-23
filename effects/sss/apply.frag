// Subsurface scattering, pass 3: the vertical blur (as blurh.frag's) of the horizontally blurred light (img_blur), the
// albedo's square root put back (Community Shaders' SSSApplyAlbedo: exact without blur), back to the scene's encoding.
// Into img_final: alpha 1 on skin (the result, which composite.frag puts in the scene), 0 elsewhere.
layout(location = 0) out vec4 outColor;

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
  outColor = vec4(0.0);
  const ivec2 p = ivec2(gl_FragCoord.xy);
  const vec4 m = texelFetch(sampler2D(img_blur, fnvPoint), p, 0);
  vec3 albedo;
  if (m.a == 0.0 || p_radius <= 0.0 || !fnvSkinAt(p, albedo)) return;
  const vec3 irradiance = sssBlur(img_blur, p, vec2(0.0, 1.0), m.rgb);
  const float t = pow(1e-3, 2.2);
  const vec3 participation = smoothstep(vec3(t), vec3(4.0 * t), albedo);
  const vec3 original = pow(max(fnvSceneColor(p).rgb, 0.0), vec3(2.2));
  const vec3 c = irradiance * sqrt(albedo) * participation + original * (1.0 - participation * participation);
  outColor = vec4(pow(max(c, 0.0), vec3(1.0 / 2.2)), 1.0);
}
