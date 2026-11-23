// lighting_2x_p_landlighting_p.hlsl: the game's extra landscape light passes, reconstructed from its compiled variants
// SLS2062..SLS2069 (tools/lightingtable.py; verified with tools/replacediff.py). Defines: DIFF with PTLIGHT (three
// point lights, diffuse), SPECULAR (the light's specular), PROJ_SHADOW (the game's projected shadows on the specular),
// LANDALPHA (alpha: the layer weights v0, v1 dotted with two colours: AmbientColor and EmittanceColor with DIFF,
// PSLightColor[1] and [2] otherwise).
//
// DIFF: point light i's tangent space vector is t2, t3, t4 (w: radius), colour PSLightColor[i] (c3 + i), falloff
// 1 - |vector| / radius; the light times the base map times t1. SPECULAR: the game's Blinn-Phong with the half vector t3
// and the light's direction t2 (both encoded 0..1), power Toggles.z, gloss in the normal map's alpha, times
// PSLightColor[0] and t1.
//
// Physically based mode (parameter pbr; off: exactly the original): as every lighting shader (adts.frag,
// docs/LIGHTING.md): DIFF: the point lights summed in linear space; SPECULAR: GGX specular, its roughness from
// the power and scaled by the gloss map and the parameter specular, from the view vector (location 12).

FNV_SAMPLER2D(1, NormalMap)
#ifdef DIFF
FNV_SAMPLER2D(0, BaseMap)
#endif
#ifdef PROJ_SHADOW
FNV_SAMPLER2D(2, ShadowMap)
FNV_SAMPLER2D(3, ShadowMaskMap)
#endif

layout(location = 1) in vec4 t0;
layout(location = 2) centroid in vec4 t1;
layout(location = 3) centroid in vec4 t2;
layout(location = 4) centroid in vec4 t3;
#ifdef DIFF
layout(location = 5) centroid in vec4 t4;
#elif defined(PROJ_SHADOW)
layout(location = 5) in vec4 t4;
#endif
#ifdef LANDALPHA
layout(location = 9) in vec4 v0;
layout(location = 10) in vec4 v1;
#endif
#ifndef DIFF
layout(location = 12) in vec4 fnvView;  // the direction to the camera, tangent space; 0: none
#endif
layout(location = 0) out vec4 oC0;

#define Toggles fnvC[27]

vec3 nrm(vec3 v) {
  float r = inversesqrt(dot(v, v));
  return vec3(fnvMulLegacy(v.x, r), fnvMulLegacy(v.y, r), fnvMulLegacy(v.z, r));
}
float powLegacy(float x, float y) { return exp2(fnvMulLegacy(y, log2(abs(x)))); }

#ifdef DIFF
vec3 toLinear(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
vec3 toGamma(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }
#else
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
#endif

#ifdef DIFF
vec3 pointLight(vec4 v, vec3 color, vec3 n) {
  float len = 1.0 / inversesqrt(dot(v.xyz, v.xyz));
  float att = 1.0 - clamp(len * (1.0 / v.w), 0.0, 1.0);
  return clamp(dot(n, nrm(v.xyz)), 0.0, 1.0) * att * color;
}
#endif

void main() {
#ifdef FNV_SUN
  const float sunVisibility = fnvSunVisibility();
#else
  const float sunVisibility = 1.0;
#endif
  vec4 nmap = texture(FNV_TEX(NormalMap), t0.xy);
  vec3 n = nrm((nmap.xyz - 0.5) * 2.0);
  vec3 c0 = fnvC[3].xyz * sunVisibility;
#ifdef DIFF
  vec3 light = pointLight(t2, c0, n) + pointLight(t3, fnvC[4].xyz, n) + pointLight(t4, fnvC[5].xyz, n);
  vec3 color = light * (texture(FNV_TEX(BaseMap), t0.xy).xyz * t1.xyz);
  if (p_pbr) {
    vec3 sum = pointLight(t2, toLinear(c0), n) + pointLight(t3, toLinear(fnvC[4].xyz), n) + pointLight(t4, toLinear(fnvC[5].xyz), n);
    color = texture(FNV_TEX(BaseMap), t0.xy).xyz * t1.xyz * toGamma(sum);
  }
#else
  float s = powLegacy(clamp(dot(n, (t3.xyz - 0.5) * 2.0), 0.0, 1.0), Toggles.z) * nmap.w;
  float nl = dot(n, (t2.xyz - 0.5) * 2.0);
  s = 0.2 - nl >= 0.0 ? s * clamp(nl + 0.5, 0.0, 1.0) : s;
  vec3 color = s * c0 * t1.xyz;
#ifdef PROJ_SHADOW
  vec4 sm = texture(FNV_TEX(ShadowMap), t4.xy);
  float mask = texture(FNV_TEX(ShadowMaskMap), t4.zw).x;
  color *= mask * (sm.xyz - 1.0) + 1.0;
#endif
  if (p_pbr && dot(fnvView.xyz, fnvView.xyz) > 0.0) {
    vec3 spec = pbrSpecular(n, normalize(fnvView.xyz), normalize((t2.xyz - 0.5) * 2.0), toLinear(c0 * t1.xyz),
                            pbrRoughness(Toggles.z)) * pbrSpecularScale(nmap.w, 1.0);
#ifdef PROJ_SHADOW
    spec *= mask * (sm.xyz - 1.0) + 1.0;
#endif
    color = toGamma(spec);
  }
#endif

#if defined(LANDALPHA) && defined(DIFF)
  float alpha = dot(fnvC[1], v0) + dot(fnvC[2], v1);
#elif defined(LANDALPHA)
  float alpha = dot(fnvC[4], v0) + dot(fnvC[5], v1);
#else
  float alpha = 1.0;
#endif
  oC0 = vec4(color, alpha);
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
