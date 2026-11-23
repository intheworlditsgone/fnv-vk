// lighting_2x_p_AD_p.hlsl: the game's ambient and diffuse light for objects (the light only, which a later pass
// multiplies with the texture), reconstructed from its compiled variants SLS2037..SLS2044 (tools/lightingtable.py;
// verified with tools/replacediff.py). Defines: LIGHTS (2 or 3: the sun and 1 or 2 point lights), SI (the glow map
// times EmittanceColor added), PROJ_SHADOW (the game's actor shadows, on the sun), PARALLAX (ParallaxShader: the
// normal and base maps at coordinates shifted along the view vector t7 by HeightMap, s2), FACEGENBLEND (SkinShader:
// skin lighting with the view vector t7, the glow map as the skin's tint).
//
// The sun: direction t1 (tangent space), colour PSLightColor[0] (c3). Point light i: direction t2, t3, colour c4, c5,
// attenuation from AttenuationMap (s4) at t4, t5 (1 - a(xy) - a(zw)). Constants: c1 AmbientColor, c2 EmittanceColor.
// Alpha: the base map's.
//
// Physically based mode (parameter pbr; off: exactly the original): as every lighting shader (adts.frag,
// docs/LIGHTING.md): the lights, ambient and glow summed in linear space (the game's Texture pass then multiplies
// by the albedo, which commutes with the conversion back). Skin (FACEGENBLEND, docs/SKIN.md): the skin model of
// adts.frag; its terms that do not scale with the albedo (the specular lobes, the sheen, the reflection) are divided by
// the base map's colour, which the Texture pass multiplies back (on a head the face tint, not in this pass, scales them).

FNV_SAMPLER2D(0, BaseMap)
FNV_SAMPLER2D(1, NormalMap)
#if defined(SI) || defined(FACEGENBLEND)
FNV_SAMPLER2D(3, GlowMap)
#endif
#ifdef PARALLAX
FNV_SAMPLER2D(2, HeightMap)
#endif
FNV_SAMPLER2D(4, AttenuationMap)
#ifdef PROJ_SHADOW
FNV_SAMPLER2D(5, ShadowMap)
FNV_SAMPLER2D(6, ShadowMaskMap)
#endif

layout(location = 1) in vec4 t0;
layout(location = 2) centroid in vec4 t1;
layout(location = 3) centroid in vec4 t2;
layout(location = 5) in vec4 t4;
#if LIGHTS >= 3
layout(location = 4) centroid in vec4 t3;
layout(location = 6) in vec4 t5;
#endif
#ifdef PROJ_SHADOW
layout(location = 7) in vec4 t6;
#endif
#if defined(PARALLAX) || defined(FACEGENBLEND)
layout(location = 8) centroid in vec4 t7;  // the view vector (tangent space)
#define VIEW t7
#endif
#ifdef FACEGENBLEND
layout(location = 12) in vec4 fnvView;  // the direction to the camera, tangent space; 0: none
#endif
layout(location = 0) out vec4 oC0;

#define AmbientColor fnvC[1]
#define EmittanceColor fnvC[2]

vec3 nrm(vec3 v) {
  float r = inversesqrt(dot(v, v));
  return vec3(fnvMulLegacy(v.x, r), fnvMulLegacy(v.y, r), fnvMulLegacy(v.z, r));
}

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

float attenuation(vec4 coords) {
  return clamp((1.0 - texture(FNV_TEX(AttenuationMap), coords.xy).x) - texture(FNV_TEX(AttenuationMap), coords.zw).x,
               0.0, 1.0);
}

void main() {
#ifdef FNV_SUN
  const float sunVisibility = fnvSunVisibility();
#else
  const float sunVisibility = 1.0;
#endif
#ifdef PARALLAX
  const vec2 uv = parallaxUv();
#else
  const vec2 uv = t0.xy;
#endif
  vec3 n = nrm((texture(FNV_TEX(NormalMap), uv).xyz - 0.5) * 2.0);
#ifdef FNV_SKIN
  vec3 skinStoreAlbedo = vec3(-1.0);  // the skin buffer's albedo (gamma), after the alpha test
#endif
  vec3 sunColor = fnvC[3].xyz * sunVisibility;
  vec3 light = clamp(dot(n, t1.xyz), 0.0, 1.0) * sunColor;
#ifdef FACEGENBLEND
  vec3 e = nrm(t7.xyz);
  vec3 tint = texture(FNV_TEX(GlowMap), t0.xy).xyz;
  light += sunColor * skinBack(n, t1.xyz, e) * 0.5;
#endif
#ifdef PROJ_SHADOW
  vec4 sm = texture(FNV_TEX(ShadowMap), t6.xy);
  float mask = texture(FNV_TEX(ShadowMaskMap), t6.zw).x;
  light *= mask * (sm.xyz - 1.0) + 1.0;
#endif
#ifdef FACEGENBLEND
  light += skinPoint(n, nrm(t2.xyz), e, fnvC[4].xyz, tint) * attenuation(t4);
#if LIGHTS >= 3
  light += skinPoint(n, nrm(t3.xyz), e, fnvC[5].xyz, tint) * attenuation(t5);
#endif
#else
  light += clamp(dot(n, nrm(t2.xyz)), 0.0, 1.0) * fnvC[4].xyz * attenuation(t4);
#if LIGHTS >= 3
  light += clamp(dot(n, nrm(t3.xyz)), 0.0, 1.0) * fnvC[5].xyz * attenuation(t5);
#endif
#endif
  light += AmbientColor.xyz;
#ifdef SI
  light += EmittanceColor.xyz * texture(FNV_TEX(GlowMap), t0.xy).xyz;
#endif
#ifndef FACEGENBLEND
  if (p_pbr) {
    vec3 sum = toLinear(fnvC[3].xyz) * sunVisibility * clamp(dot(n, t1.xyz), 0.0, 1.0);
#ifdef PROJ_SHADOW
    sum *= mask * (sm.xyz - 1.0) + 1.0;
#endif
    sum += toLinear(fnvC[4].xyz) * clamp(dot(n, nrm(t2.xyz)), 0.0, 1.0) * attenuation(t4);
#if LIGHTS >= 3
    sum += toLinear(fnvC[5].xyz) * clamp(dot(n, nrm(t3.xyz)), 0.0, 1.0) * attenuation(t5);
#endif
    sum += pbrIndirect(AmbientColor.xyz, fnvWorldNormal(n), fnvAmbientOcclusion());
#ifdef SI
    sum += toLinear(EmittanceColor.xyz * texture(FNV_TEX(GlowMap), t0.xy).xyz);
#endif
    light = toGamma(sum);
  }
#else
  if (p_pbr) {
    const vec3 v = dot(fnvView.xyz, fnvView.xyz) > 0.0 ? normalize(fnvView.xyz) : e;
    vec3 ns = fnvSkinDetail(n, uv);
    const vec3 albedo = max(toLinear(texture(FNV_TEX(BaseMap), uv).xyz), vec3(0.02));
#ifdef FNV_WET
    // Rain on skin, as adts.frag: the darkening as a factor on the light the Texture pass multiplies by the albedo.
    vec3 wetAlbedo = toGamma(albedo);
    FnvWetObject wet = fnvWetObjectBegin(ns, wetAlbedo, -1.0, 0.0);
    const vec3 darken = toLinear(wetAlbedo) / albedo;
#else
    const vec3 darken = vec3(1.0);
#endif
    const FnvSkin m = fnvSkinMaterial(0.0, dot(ns, v));
    vec3 sunC = toLinear(fnvC[3].xyz) * sunVisibility;
#ifdef PROJ_SHADOW
    sunC *= mask * (sm.xyz - 1.0) + 1.0;
#endif
    const vec3 sunDir = normalize(t1.xyz);
    vec3 diffuse = vec3(0.0), spec = vec3(0.0);
    fnvSkinLight(m, ns, v, sunDir, sunC, diffuse, spec);
    vec3 through = sunC * (skinBack(ns, sunDir, e) * 0.5);
    const vec3 tintL = toLinear(tint);
    for (int i = 0; i < LIGHTS - 1; i++) {
#if LIGHTS >= 3
      const vec3 dir = i == 0 ? t2.xyz : t3.xyz;
      const float att = attenuation(i == 0 ? t4 : t5);
#else
      const vec3 dir = t2.xyz;
      const float att = attenuation(t4);
#endif
      const vec3 c = toLinear(fnvC[4 + i].xyz), l = nrm(dir);
      fnvSkinLight(m, ns, v, l, c * att, diffuse, spec);
      const float d = dot(ns, l), ds = clamp(d, 0.0, 1.0);
      const float w = clamp((d + 0.3) * 0.769230783, 0.0, 1.0);
      const float wrap = clamp((w * -2.0 + 3.0) * (w * w) - (ds * -2.0 + 3.0) * (ds * ds), 0.0, 1.0);
      through += (skinBack(ns, l, e) * mix(c, tintL, 0.5) + wrap * tintL) * att;
    }
    const float ao = fnvAmbientOcclusion();
    const float nv = clamp(dot(ns, v), 0.0, 1.0);
    const vec2 lobes = fnvSkinIndirect(m, nv);
    const vec3 r = reflect(-v, ns);
    const float horizon = clamp(1.0 + r.z, 0.0, 1.0);
    vec3 env = toLinear(AmbientColor.xyz);
#ifdef FNV_ENV
    const vec3 rw = fnvToWorld(r);
    if (fnvEnvValid() && dot(rw, rw) > 0.0) env = fnvEnvRadiance(rw, m.avg * m.avg);
#endif
    const vec3 unscaled = spec + env * (lobes.y * horizon * horizon * fnvSpecularOcclusion(ao, nv, m.avg));
    vec3 scaled = (diffuse + through + lobes.x * pbrIndirect(AmbientColor.xyz, fnvWorldNormal(ns), ao)) * darken;
    vec3 added = unscaled;
#ifdef FNV_WET
    // The film: lit (1 - R) + its own light; R from the film over 0 and over 1.
    const vec3 film0 = fnvWetObjectFilm(wet, vec3(0.0), AmbientColor.xyz, sunDir, sunC);
    const vec3 keep = fnvWetObjectFilm(wet, vec3(1.0), AmbientColor.xyz, sunDir, sunC) - film0;
    scaled *= keep;
    added = added * keep + film0;
#endif
    light = toGamma(scaled + added / albedo);
    skinStoreAlbedo = toGamma(albedo);  // the base map (the face tint is in the Texture pass)
  }
#endif
  oC0 = vec4(light, texture(FNV_TEX(BaseMap), uv).w);
  fnvAlphaTest(oC0.w);
#ifdef FNV_SKIN
  if (skinStoreAlbedo.x >= 0.0) fnvSkinStore(skinStoreAlbedo);
#endif
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
