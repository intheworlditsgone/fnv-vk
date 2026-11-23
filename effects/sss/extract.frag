// Subsurface scattering, pass 1: the light to blur, in linear light. On skin (fnvSkinAt) the scene's colour with the
// albedo partly removed (Community Shaders' pre-and-post scatter: divided by the albedo's square root; its
// participation keeps near-black albedo out), a = 1; elsewhere the colour as it is, a = 0.
layout(location = 0) out vec4 outIrr;

void main() {
  const ivec2 p = ivec2(gl_FragCoord.xy);
  const vec3 c = pow(max(fnvSceneColor(p).rgb, 0.0), vec3(2.2));
  vec3 albedo;
  if (!fnvInsideRect(p) || !fnvSkinAt(p, albedo)) {
    outIrr = vec4(c, 0.0);
    return;
  }
  const float t = pow(1e-3, 2.2);
  const vec3 participation = smoothstep(vec3(t), vec3(4.0 * t), albedo);
  outIrr = vec4(c * participation / max(sqrt(albedo), vec3(1e-6)), 1.0);
}
