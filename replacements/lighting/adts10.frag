// lighting_2x_p_ADTS10_p.hlsl: the game's per-pixel object lighting with several lights in one pass, reconstructed
// from its compiled variants SLS2029..SLS2036 (tools/lightingtable.py; verified with tools/replacediff.py).
// Defines: LIGHTS (4 or 9), SPECULAR, SI (compiled to the same code as without it), OPT, FACEGENBLEND (SkinShader,
// SKIN2010..SKIN2012: the view vector takes t2 and the light directions move up one interpolant, so one light fewer
// with LIGHTS 9 or SPECULAR; point lights get the skin terms tinted (0.3, 0, 0); the face tint FaceGenMap0/1 at s2, s3).
//
// Light i has the colour PSLightColor[i] (c3 + i) and is lit only while the pass's light count is above i. Without OPT
// light 0 is the sun (direction t2, no falloff) and the count is EmittanceColor.w; with OPT (interiors) every light is a
// point light and the count is PSLightColor[0].w. A point light's direction is the interpolant after the previous
// light's, its falloff 1 - |(PSLightPosition - position) / radius|^2 (position: t1). The variant handles as many lights
// as the interpolants hold: 3 with SPECULAR (directions t2..t4, half vectors t5..t7), else 4 (LIGHTS 4) or 6 (LIGHTS 9).
// Constants: c1 AmbientColor, c2 EmittanceColor, c3.. PSLightColor, c19.. PSLightPosition (xyz, w radius), c27 Toggles
// (x vertex colour, y fog, z specular power, w alpha cut-off; with OPT the power is PSLightColor[1].w).
//
// Physically based mode (parameter pbr): as adts.frag's, over the pass's lights (skin keeps the game's shading).

#ifdef FACEGENBLEND
#define DIR0 3  // the interpolant of light 0's direction
#else
#define DIR0 2
#endif
#ifdef SPECULAR
#define NLIGHTS (5 - DIR0)  // directions up to t4, half vectors from t5
#elif LIGHTS == 9
#define NLIGHTS (8 - DIR0)
#else
#define NLIGHTS 4
#endif

FNV_SAMPLER2D(0, BaseMap)
FNV_SAMPLER2D(1, NormalMap)
#ifdef FACEGENBLEND
FNV_SAMPLER2D(2, FaceGenMap0)
FNV_SAMPLER2D(3, FaceGenMap1)
#endif

layout(location = 1) in vec4 t0;
layout(location = 2) in vec4 t1;
layout(location = 3) in vec4 t2;
layout(location = 4) in vec4 t3;
layout(location = 5) in vec4 t4;
layout(location = 6) in vec4 t5;
#if DIR0 + NLIGHTS > 6 || defined(SPECULAR)
layout(location = 7) in vec4 t6;
#endif
#if DIR0 + NLIGHTS > 7 || (defined(SPECULAR) && NLIGHTS >= 3)
layout(location = 8) in vec4 t7;
#endif
layout(location = 9) in vec4 v0;
layout(location = 10) in vec4 v1;
layout(location = 0) out vec4 oC0;

#define AmbientColor fnvC[1]
#define EmittanceColor fnvC[2]
#define Toggles fnvC[27]

vec3 nrm(vec3 v) {
  float r = inversesqrt(dot(v, v));
  return vec3(fnvMulLegacy(v.x, r), fnvMulLegacy(v.y, r), fnvMulLegacy(v.z, r));
}

vec3 interp(int k) {
  return k == 2 ? t2.xyz : k == 3 ? t3.xyz : k == 4 ? t4.xyz : k == 5 ? t5.xyz
#if DIR0 + NLIGHTS > 6 || defined(SPECULAR)
       : k == 6 ? t6.xyz
#endif
#if DIR0 + NLIGHTS > 7 || (defined(SPECULAR) && NLIGHTS >= 3)
       : k == 7 ? t7.xyz
#endif
       : vec3(0.0);
}
vec3 lightDir(int i) { return interp(DIR0 + i); }

#ifdef FACEGENBLEND
// Skin: a point light (unit direction l), its diffuse, back light through the edges and the wrap of light past the
// terminator tinted.
vec3 skinPoint(vec3 n, vec3 l, vec3 e, vec3 c, vec3 tint) {
  float d = dot(n, l);
  float ds = clamp(d, 0.0, 1.0);
  float w = clamp((d + 0.3) * 0.769230783, 0.0, 1.0);
  float wrap = clamp((w * -2.0 + 3.0) * (w * w) - (ds * -2.0 + 3.0) * (ds * ds), 0.0, 1.0);
  float f = 1.0 - clamp(dot(n, e), 0.0, 1.0);
  float back = clamp(dot(e, -l), 0.0, 1.0) * (f * f);
  return c * ds + back * mix(c, tint, 0.5) + wrap * tint;
}
#endif

#ifdef OPT
#define LIGHT_COUNT fnvC[3].w
#define FIRST_POINT 0  // light 0 is a point light too
#else
#define LIGHT_COUNT EmittanceColor.w
#define FIRST_POINT 1
#endif

#define PBR 1
layout(location = 12) in vec4 fnvView;  // the direction to the camera, tangent space; 0: none
const float kPi = 3.14159265;
vec3 toLinear(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
vec3 toGamma(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }
#ifdef FNV_FRAME
vec3 fnvWorldNormal(vec3 n) { return fnvToWorld(n); }
#else
vec3 fnvWorldNormal(vec3 n) { return vec3(0.0); }
#endif
// Normalize probe irradiance by its mean to preserve the game's ambient intensity.
#ifdef FNV_ENV
vec3 pbrAmbient(vec3 ambient, vec3 nWorld) {
  vec3 a = toLinear(ambient);
  if (!fnvEnvValid() || dot(nWorld, nWorld) == 0.0) return a;
  return a * clamp(dot(fnvEnvIrradiance(nWorld), vec3(0.2126, 0.7152, 0.0722)), 0.0, 4.0);
}
#else
vec3 pbrAmbient(vec3 ambient, vec3 nWorld) { return toLinear(ambient); }
#endif
// SSGI alpha is the fraction of rays that missed and still need probe lighting.
vec3 pbrIndirect(vec3 ambient, vec3 nWorld, float ao) {
  vec3 probe = pbrAmbient(ambient, nWorld) * ao;
  vec4 gi;
  return fnvScreenIndirect(gi) ? gi.rgb + gi.a * probe : probe;
}
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
// The environment's reflection (docs/INDIRECT.md "Reflections") on a dielectric (reflectance 0.04) of GGX roughness
// alpha: the probe prefiltered for alpha in the mirror direction (world space through the vertex shader's frame),
// times the split-sum's directional albedo (Karis 2014's analytic fit of the GGX DFG term), faded where the mirror
// direction goes under the surface (horizon occlusion: the normal map can point the reflection below the geometry).
// Zero without a probe or a frame, and for a material with its own environment-map pass (c219.x, the plugin: the game
// draws that reflection itself).
#if defined(FNV_ENV) && defined(FNV_FRAME)
vec3 pbrReflection(vec3 n, vec3 v, float alpha) {
  if (!fnvEnvValid() || fnvC[219].x != 0.0) return vec3(0.0);
  vec3 rt = reflect(-v, n);
  vec3 r = fnvToWorld(rt);
  if (dot(r, r) == 0.0) return vec3(0.0);
  float nv = clamp(dot(n, v), 0.0, 1.0), rough = sqrt(alpha);
  vec4 q = rough * vec4(-1.0, -0.0275, -0.572, 0.022) + vec4(1.0, 0.0425, 1.04, -0.04);
  float a004 = min(q.x * q.x, exp2(-9.28 * nv)) * q.x + q.y;
  vec2 ab = vec2(-1.04, 1.04) * a004 + q.zw;
  float horizon = clamp(1.0 + rt.z, 0.0, 1.0);
  return fnvEnvRadiance(r, alpha) * ((0.04 * ab.x + ab.y) * horizon * horizon);
}
#else
vec3 pbrReflection(vec3 n, vec3 v, float alpha) { return vec3(0.0); }
#endif

// Falloff of light i (1 for the sun).
float falloff(int i) {
  if (i < FIRST_POINT) return 1.0;
  vec4 p = fnvC[19 + i - FIRST_POINT];
  vec3 d = (p.xyz - t1.xyz) * (1.0 / p.w);
  return 1.0 - clamp(dot(d, d), 0.0, 1.0);
}

void main() {
#ifdef FNV_SUN
  const float sunVisibility = fnvSunVisibility();  // light 0's (the sun's outdoors); before any discard
#else
  const float sunVisibility = 1.0;
#endif
  vec4 base = texture(FNV_TEX(BaseMap), t0.xy);
#ifndef OPT
  if ((AmbientColor.w - 1.0 >= 0.0 ? 0.0 : 1.0) * (base.w - Toggles.w) < 0.0) discard;
#endif
  vec4 nmap = texture(FNV_TEX(NormalMap), t0.xy);
  vec3 n = nrm((nmap.xyz - 0.5) * 2.0);

  vec3 light = vec3(0.0);
#ifdef SPECULAR
  vec3 spec = vec3(0.0);
#ifdef OPT
  float power = fnvC[4].w;
#else
  float power = Toggles.z;
#endif
#endif
  for (int i = 0; i < NLIGHTS; i++) {
    if (!(LIGHT_COUNT - float(i) > 0.0) && (i > 0 || FIRST_POINT == 0)) continue;  // not lit this pass
    vec3 l = lightDir(i);
    float att = falloff(i);
    vec3 dir = i < FIRST_POINT ? l : nrm(l);
    vec3 color = fnvC[3 + i].xyz * (i == 0 ? sunVisibility : 1.0);
#ifdef FACEGENBLEND
    if (i >= FIRST_POINT) light += skinPoint(n, dir, nrm(t2.xyz), color, vec3(0.3, 0.0, 0.0)) * att;
    else light += color * clamp(dot(n, dir), 0.0, 1.0) * att;
#else
    light += color * clamp(dot(n, dir), 0.0, 1.0) * att;
#endif
#ifdef SPECULAR
    vec3 h = interp(5 + i);
    float s = exp2(fnvMulLegacy(log2(clamp(dot(n, nrm(h)), 0.0, 1.0)), power)) * nmap.w * att;
    float nl = dot(n, l);
    s = 0.2 - nl >= 0.0 ? s * clamp(nl + 0.5, 0.0, 1.0) : s;
    spec += s * color;
#endif
  }
  light = max(light + AmbientColor.xyz, 0.0);

  vec3 baseColor = base.xyz;
#ifdef FACEGENBLEND
  baseColor = ((texture(FNV_TEX(FaceGenMap0), t0.xy).xyz - 0.5) * 2.0 + baseColor) *
              (texture(FNV_TEX(FaceGenMap1), t0.xy).xyz * 2.0) * 2.0;
#endif
#ifdef OPT
  vec3 albedo = baseColor * v0.xyz;
#else
  vec3 albedo = -Toggles.x >= 0.0 ? baseColor : baseColor * v0.xyz;
#endif
  vec3 color = light * albedo;
#ifdef SPECULAR
  color += spec * t1.w;
#endif
#ifdef PBR
  if (p_pbr) {
#ifdef FACEGENBLEND
    // Skin (docs/SKIN.md), as adts.frag: Community Shaders' skin model on each light of the pass; the game's own light
    // through thin skin kept for the point lights (back light and wrap, tinted (0.3, 0, 0)), in linear light. The gloss
    // of a skin with the game's specular flag: its normal map's alpha, faded as the game's specular.
    const vec3 e = nrm(t2.xyz);
    const vec3 v = dot(fnvView.xyz, fnvView.xyz) > 0.0 ? normalize(fnvView.xyz) : e;
    vec3 ns = fnvSkinDetail(n, t0.xy);
#ifdef FNV_WET
    FnvWetObject wet = fnvWetObjectBegin(ns, albedo, -1.0, 0.0);  // rain on skin, as adts.frag
#endif
#ifdef SPECULAR
    const float gloss = nmap.w * clamp(t1.w, 0.0, 1.0);
#else
    const float gloss = 0.0;
#endif
    const FnvSkin m = fnvSkinMaterial(gloss, dot(ns, v));
    vec3 diffuse = vec3(0.0), spec = vec3(0.0), through = vec3(0.0);
    const vec3 tint = toLinear(vec3(0.3, 0.0, 0.0));
    for (int i = 0; i < NLIGHTS; i++) {
      if (!(LIGHT_COUNT - float(i) > 0.0) && (i > 0 || FIRST_POINT == 0)) continue;
      const float att = falloff(i);
      const vec3 c = toLinear(fnvC[3 + i].xyz) * (i == 0 ? sunVisibility : 1.0);
      const vec3 l = normalize(lightDir(i));
      fnvSkinLight(m, ns, v, l, c * att, diffuse, spec);
      if (i >= FIRST_POINT) {
        const float d = dot(ns, l), ds = clamp(d, 0.0, 1.0);
        const float w = clamp((d + 0.3) * 0.769230783, 0.0, 1.0);
        const float wrap = clamp((w * -2.0 + 3.0) * (w * w) - (ds * -2.0 + 3.0) * (ds * ds), 0.0, 1.0);
        const float f = 1.0 - clamp(dot(ns, e), 0.0, 1.0);
        through += (clamp(dot(e, -l), 0.0, 1.0) * (f * f) * mix(c, tint, 0.5) + wrap * tint) * att;
      }
    }
    // The ambient (Skin::SkinIndirectLobeWeights): the probe's diffuse light by the lobes' remainder, and the reflection
    // (the probe along the mirror direction at the mixed roughness; the ambient colour without one), faded under the
    // horizon of the geometry, occluded by the ambient occlusion.
    const float ao = fnvAmbientOcclusion();
    const float nv = clamp(dot(ns, v), 0.0, 1.0);
    const vec2 lobes = fnvSkinIndirect(m, nv);
    vec3 lit = toLinear(albedo) * (diffuse + through + lobes.x * pbrIndirect(AmbientColor.xyz, fnvWorldNormal(ns), ao)) + spec;
    const vec3 r = reflect(-v, ns);
    const float horizon = clamp(1.0 + r.z, 0.0, 1.0);
    vec3 env = toLinear(AmbientColor.xyz);
#ifdef FNV_ENV
    const vec3 rw = fnvToWorld(r);
    if (fnvEnvValid() && dot(rw, rw) > 0.0) env = fnvEnvRadiance(rw, m.avg * m.avg);
#endif
    lit += env * (lobes.y * horizon * horizon * fnvSpecularOcclusion(ao, nv, m.avg));
#ifdef FNV_WET
    lit = FIRST_POINT > 0 ? fnvWetObjectFilm(wet, lit, AmbientColor.xyz, normalize(lightDir(0)), toLinear(fnvC[3].xyz) * sunVisibility)
                          : fnvWetObjectFilm(wet, lit, AmbientColor.xyz, vec3(0.0), vec3(0.0));
#endif
    color = toGamma(lit);
#else
#ifdef FNV_WET
    // Wet world (docs/WETWORLD.md), as adts.frag.
    FnvWetObject wet = fnvWetObjectBegin(n, albedo, clamp(0.5 + 0.6 * (dot(baseColor, vec3(0.3, 0.59, 0.11)) -
        dot(textureLod(FNV_TEX(BaseMap), t0.xy, 14.0).xyz, vec3(0.3, 0.59, 0.11))), 0.0, 1.0), 1.0);
#endif
    vec3 diffuse = vec3(0.0);
#ifdef SPECULAR
    vec3 spec = vec3(0.0);
    const bool viewOk = dot(fnvView.xyz, fnvView.xyz) > 0.0;
    vec3 v = viewOk ? normalize(fnvView.xyz) : vec3(0.0, 0.0, 1.0);
    float alpha = pbrRoughness(power), scale = pbrSpecularScale(nmap.w, t1.w);
#endif
    for (int i = 0; i < NLIGHTS; i++) {
      if (!(LIGHT_COUNT - float(i) > 0.0) && (i > 0 || FIRST_POINT == 0)) continue;
      vec3 c = toLinear(fnvC[3 + i].xyz) * (i == 0 ? sunVisibility : 1.0) * falloff(i);
      vec3 l = normalize(lightDir(i));
      diffuse += c * clamp(dot(n, l), 0.0, 1.0);
#ifdef SPECULAR
      if (viewOk) spec += pbrSpecular(n, v, l, c, alpha) * scale;
#endif
    }
    float ao = fnvAmbientOcclusion();
    vec3 lit = toLinear(albedo) * (diffuse + pbrIndirect(AmbientColor.xyz, fnvWorldNormal(n), ao));
#ifdef SPECULAR
    if (viewOk) spec += pbrReflection(n, v, alpha) * (scale * fnvSpecularOcclusion(ao, clamp(dot(n, v), 0.0, 1.0), sqrt(alpha)));
    lit += spec;
#endif
#ifdef FNV_WET
    lit = FIRST_POINT > 0 ? fnvWetObjectFilm(wet, lit, AmbientColor.xyz, normalize(lightDir(0)),
                                             toLinear(fnvC[3].xyz) * sunVisibility)
                          : fnvWetObjectFilm(wet, lit, AmbientColor.xyz, vec3(0.0), vec3(0.0));
#endif
    color = toGamma(lit);
#endif  // FACEGENBLEND
  }
#endif
#ifdef OPT
  vec3 rgb = mix(color, v1.xyz, v1.w);
#else
  vec3 rgb = -Toggles.y >= 0.0 ? color : mix(color, v1.xyz, v1.w);
#endif
  oC0 = vec4(rgb, base.w * AmbientColor.w);
  fnvAlphaTest(oC0.w);
#if defined(FNV_SKIN) && defined(FACEGENBLEND)
  if (p_pbr) fnvSkinStore(albedo);  // the skin buffer for the subsurface scattering (docs/SKIN.md)
#endif
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
