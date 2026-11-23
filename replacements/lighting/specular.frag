// lighting_2x_p_Specular_p.hlsl: the game's separate specular pass for objects, reconstructed from its compiled variants
// SLS2047..SLS2056 (tools/lightingtable.py; verified with tools/replacediff.py). Defines: POINT (point lights, with
// falloff), NUM_PT_LIGHTS (2 or 3 lights in one pass, with POINT), HAIR (the strand highlight), PROJ_SHADOW (the
// game's actor shadows, on the single light), PARALLAX (ParallaxShader: the normal map at coordinates shifted along the
// view vector t7 by HeightMap, s2).
//
// Light i: direction t1, t2, t5 (w: radius with POINT), half vector t3, t4, t6, colour PSLightColor[i] (c3 + i);
// falloff 1 - |direction / radius|^2. The game's Blinn-Phong (power Toggles.z, gloss in the normal map's alpha), damped
// where the light grazes the surface; HAIR: the highlight along the tangent bent from the normal (power 30, gloss 0.7).
// Output: one light, the saturated colour with alpha its sum; several lights (not HAIR or PARALLAX), the saturated sum
// of the lights' full colours (alpha included).
//
// Physically based mode (parameter pbr; off: exactly the original): as every lighting shader (adts.frag,
// docs/LIGHTING.md): GGX specular of the pass's lights (the game adds this pass for materials with the specular
// flag), its roughness from the power and scaled by the gloss map and the parameter specular; the alpha the game's;
// hair keeps the game's highlight.

#ifndef NUM_PT_LIGHTS
#define NUM_PT_LIGHTS 1
#endif

FNV_SAMPLER2D(0, NormalMap)
#ifdef PARALLAX
FNV_SAMPLER2D(2, HeightMap)
#endif
#ifdef PROJ_SHADOW
FNV_SAMPLER2D(4, ShadowMap)
FNV_SAMPLER2D(5, ShadowMaskMap)
#endif

layout(location = 1) in vec4 t0;
layout(location = 2) centroid in vec4 t1;
layout(location = 4) centroid in vec4 t3;
#if NUM_PT_LIGHTS >= 2
layout(location = 3) centroid in vec4 t2;
layout(location = 5) centroid in vec4 t4;
#endif
#if NUM_PT_LIGHTS >= 3
layout(location = 6) centroid in vec4 t5;
layout(location = 7) centroid in vec4 t6;
#endif
#ifdef PROJ_SHADOW
layout(location = 7) in vec4 t6;
#endif
#if defined(PARALLAX) || defined(FACEGENBLEND)
layout(location = 8) centroid in vec4 t7;  // the view vector (tangent space)
#define VIEW t7
#endif
#ifndef HAIR
layout(location = 12) in vec4 fnvView;  // the direction to the camera, tangent space; 0: none
#endif
layout(location = 0) out vec4 oC0;

#define Toggles fnvC[27]

vec3 nrm(vec3 v) {
  float r = inversesqrt(dot(v, v));
  return vec3(fnvMulLegacy(v.x, r), fnvMulLegacy(v.y, r), fnvMulLegacy(v.z, r));
}
float powLegacy(float x, float y) { return exp2(fnvMulLegacy(y, log2(abs(x)))); }

#ifdef PARALLAX
// The parallax texture coordinates: t0 shifted along the normalised view vector by the height (0.04 h - 0.02).
vec2 parallaxUv() {
  float h = texture(FNV_TEX(HeightMap), t0.xy).x * 0.04 - 0.02;
  return VIEW.xy * inversesqrt(dot(VIEW.xyz, VIEW.xyz)) * h + t0.xy;
}
#endif
#ifdef FACEGENBLEND
// Skin: the back light through the edges, (1 - N.E)^2 sat(E.-L).
float skinBack(vec3 n, vec3 l, vec3 e) {
  float f = 1.0 - clamp(dot(n, e), 0.0, 1.0);
  return clamp(dot(e, -l), 0.0, 1.0) * (f * f);
}
// Skin: a point light (unit direction l), its diffuse, back light and the wrap of light past the terminator tinted.
vec3 skinPoint(vec3 n, vec3 l, vec3 e, vec3 c, vec3 tint) {
  float d = dot(n, l);
  float ds = clamp(d, 0.0, 1.0);
  float w = clamp((d + 0.3) * 0.769230783, 0.0, 1.0);
  float wrap = clamp((w * -2.0 + 3.0) * (w * w) - (ds * -2.0 + 3.0) * (ds * ds), 0.0, 1.0);
  return c * ds + skinBack(n, l, e) * mix(c, tint, 0.5) + wrap * tint;
}
#endif

#ifndef HAIR
const float kPi = 3.14159265;
vec3 toLinear(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
vec3 toGamma(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }
// The specular of one light (unit direction l, linear colour c) on a dielectric (reflectance 0.04) of GGX roughness
// alpha: Smith height-correlated visibility, Schlick Fresnel. The game's light colour is the diffuse light on a white
// surface facing it, so the irradiance is pi times it.
vec3 pbrSpecular(vec3 n, vec3 v, vec3 l, vec3 c, float alpha) {
  float nl = clamp(dot(n, l), 0.0, 1.0);
  if (nl <= 0.0) return vec3(0.0);
  vec3 h = normalize(l + v);
  float nv = max(dot(n, v), 1e-4), nh = clamp(dot(n, h), 0.0, 1.0), vh = clamp(dot(v, h), 0.0, 1.0);
  float a2 = alpha * alpha;
  float f = 0.04 + 0.96 * pow(1.0 - vh, 5.0);
  float d = nh * nh * (a2 - 1.0) + 1.0;
  float D = a2 / (kPi * d * d);
  float vis = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2));
  return c * (nl * kPi * D * vis * f);
}
// A specular material's roughness from the game's Blinn-Phong power, and its specular's scale: the gloss map (the
// game's specular intensity; 0.5 and above a full dielectric), the parameter specular, and the game's specular LOD fade.
float pbrRoughness(float power) { return clamp(sqrt(2.0 / (max(power, 0.0) + 2.0)), 0.03, 1.0); }
float pbrSpecularScale(float gloss, float lodFade) { return clamp(2.0 * gloss, 0.0, 1.0) * p_specular * clamp(lodFade, 0.0, 1.0); }
// The environment's reflection (docs/INDIRECT.md "Reflections"), as adts.frag's: in the multi-pass path only the first
// specular pass of a geometry in the world pass adds it (c219.y, the plugin), and none for a material with its own
// environment-map pass (c219.x).
#if defined(FNV_ENV) && defined(FNV_FRAME)
vec3 pbrReflection(vec3 n, vec3 v, float alpha) {
  if (!fnvEnvValid() || fnvC[219].x != 0.0 || fnvC[219].y == 0.0) return vec3(0.0);
  vec3 rt = reflect(-v, n);
  vec3 r = fnvToWorld(rt);
  if (dot(r, r) == 0.0) return vec3(0.0);
  float nv = clamp(dot(n, v), 0.0, 1.0), rough = sqrt(alpha);
  vec4 q = rough * vec4(-1.0, -0.0275, -0.572, 0.022) + vec4(1.0, 0.0425, 1.04, -0.04);
  float a004 = min(q.x * q.x, exp2(-9.28 * nv)) * q.x + q.y;
  vec2 ab = vec2(-1.04, 1.04) * a004 + q.zw;
  float horizon = clamp(1.0 + rt.z, 0.0, 1.0);
  return fnvEnvRadiance(r, alpha) * ((0.04 * ab.x + ab.y) * horizon * horizon) *
         fnvSpecularOcclusion(fnvAmbientOcclusion(), nv, rough);
}
#else
vec3 pbrReflection(vec3 n, vec3 v, float alpha) { return vec3(0.0); }
#endif
// Light direction l (w: radius with POINT, the game's falloff 1 - |l / radius|^2).
vec3 pbrSpecularLight(vec3 n, vec3 v, vec4 l, vec3 c, float alpha) {
#ifdef POINT
  vec3 d = l.xyz * (1.0 / l.w);
  c *= 1.0 - clamp(dot(d, d), 0.0, 1.0);
#endif
  return pbrSpecular(n, v, nrm(l.xyz), c, alpha);
}
#endif

float specular(vec3 n, float gloss, vec4 l, vec3 h) {
  vec3 ln = nrm(l.xyz);
#ifdef HAIR
  vec3 tangent = nrm(n * 0.5 + vec3(0.0, 0.0, 1.0));
  float x = clamp(abs(dot(tangent, ln) - dot(tangent, nrm(h))), 0.0, 1.0);
  float s = powLegacy(1.0 - x, 30.0) * (gloss * 0.7);
#else
  float s = powLegacy(clamp(dot(n, nrm(h)), 0.0, 1.0), Toggles.z) * gloss;
#endif
  float nl = dot(n, ln);
  s = 0.2 - nl >= 0.0 ? s * clamp(nl + 0.5, 0.0, 1.0) : s;
#ifdef POINT
  vec3 d = l.xyz * (1.0 / l.w);
  s *= 1.0 - clamp(dot(d, d), 0.0, 1.0);
#endif
  return s;
}

void main() {
#ifdef FNV_SUN
  const float sunVisibility = fnvSunVisibility();
#else
  const float sunVisibility = 1.0;
#endif
#ifdef PARALLAX
  vec4 nmap = texture(FNV_TEX(NormalMap), parallaxUv());
#else
  vec4 nmap = texture(FNV_TEX(NormalMap), t0.xy);
#endif
  vec3 n = nrm((nmap.xyz - 0.5) * 2.0);
  vec4 c0 = vec4(fnvC[3].xyz * sunVisibility, fnvC[3].w);
  vec4 sum = specular(n, nmap.w, t1, t3.xyz) * c0;
#if NUM_PT_LIGHTS >= 2
  sum += specular(n, nmap.w, t2, t4.xyz) * fnvC[4];
#endif
#if NUM_PT_LIGHTS >= 3
  sum += specular(n, nmap.w, t5, t6.xyz) * fnvC[5];
#endif
#if NUM_PT_LIGHTS >= 2 && !defined(HAIR) && !defined(PARALLAX)
  oC0 = clamp(sum, 0.0, 1.0);
#else
  vec3 color = sum.xyz;
#ifdef PROJ_SHADOW
  vec4 sm = texture(FNV_TEX(ShadowMap), t6.xy);
  float mask = texture(FNV_TEX(ShadowMaskMap), t6.zw).x;
  color *= mask * (sm.xyz - 1.0) + 1.0;
#endif
  oC0 = vec4(clamp(color, 0.0, 1.0), dot(color, vec3(1.0)));
#endif
#ifndef HAIR
  if (p_pbr && dot(fnvView.xyz, fnvView.xyz) > 0.0) {
    vec3 v = normalize(fnvView.xyz);
    float alpha = pbrRoughness(Toggles.z), scale = pbrSpecularScale(nmap.w, 1.0);
    vec3 spec = pbrSpecularLight(n, v, t1, toLinear(c0.xyz), alpha);
#if NUM_PT_LIGHTS >= 2
    spec += pbrSpecularLight(n, v, t2, toLinear(fnvC[4].xyz), alpha);
#endif
#if NUM_PT_LIGHTS >= 3
    spec += pbrSpecularLight(n, v, t5, toLinear(fnvC[5].xyz), alpha);
#endif
#ifdef PROJ_SHADOW
    spec *= mask * (sm.xyz - 1.0) + 1.0;  // the shadow on the lights, not on the environment's reflection
#endif
    spec = (spec + pbrReflection(n, v, alpha)) * scale;
    oC0.xyz = clamp(toGamma(spec), 0.0, 1.0);
  }
#endif
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
