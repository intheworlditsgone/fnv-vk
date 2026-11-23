// lighting_2x_p_LandO_p.hlsl: the game's landscape lighting (exterior terrain), reconstructed from its compiled variants
// SLS2092..SLS2148 (tools/lightingtable.py; verified with tools/replacediff.py). Defines: TEXCOUNT (1..7 blended
// layers), PTLIGHTS (three point lights), IGNORE_NORMALS (no normal maps: the vertex normal), PROJ_SHADOW (the game's
// projected shadows), FOG and HIGHQUAL (every shipped variant), SILHOUETTE (the ambient colour fogged, nothing else).
//
// Layer i has BaseMap[i] (s0 + i) and NormalMap[i] (s7 + i) and the blend weight v0.x, v0.y, v0.z, v1.x, v1.y, v1.z,
// v1.w. The sun's direction PSLightDir (c18) is in world space; t3..t5 are the rows taking world to tangent space.
// Point light i: position and radius PSLightPosition[i] (c19 + i), colour PSLightColor[1 + i] (c4 + i), the pixel's
// world position t2, falloff 1 - |(position - t2) / radius|^2. The lit colour is multiplied by t1 and fogged with t7
// (rgb colour, w amount). Constants: c1 AmbientColor, c3 the sun's PSLightColor[0].
//
// Physically based mode (parameter pbr; off: exactly the original): the lights and ambient summed in linear space, as
// every lighting shader's (adts.frag, docs/LIGHTING.md); the terrain has no specular material (the game's land specular
// is its own pass, landlighting.frag), so diffuse only. With FNV_WET and the parameter wet, the rain's wetness, standing
// water and ripples (docs/WETWORLD.md).

#ifndef SILHOUETTE
FNV_SAMPLER2D(0, BaseMap0)
#if TEXCOUNT >= 2
FNV_SAMPLER2D(1, BaseMap1)
#endif
#if TEXCOUNT >= 3
FNV_SAMPLER2D(2, BaseMap2)
#endif
#if TEXCOUNT >= 4
FNV_SAMPLER2D(3, BaseMap3)
#endif
#if TEXCOUNT >= 5
FNV_SAMPLER2D(4, BaseMap4)
#endif
#if TEXCOUNT >= 6
FNV_SAMPLER2D(5, BaseMap5)
#endif
#if TEXCOUNT >= 7
FNV_SAMPLER2D(6, BaseMap6)
#endif
#ifndef IGNORE_NORMALS
FNV_SAMPLER2D(7, NormalMap0)
#if TEXCOUNT >= 2
FNV_SAMPLER2D(8, NormalMap1)
#endif
#if TEXCOUNT >= 3
FNV_SAMPLER2D(9, NormalMap2)
#endif
#if TEXCOUNT >= 4
FNV_SAMPLER2D(10, NormalMap3)
#endif
#if TEXCOUNT >= 5
FNV_SAMPLER2D(11, NormalMap4)
#endif
#if TEXCOUNT >= 6
FNV_SAMPLER2D(12, NormalMap5)
#endif
#if TEXCOUNT >= 7
FNV_SAMPLER2D(13, NormalMap6)
#endif
#endif
#ifdef PROJ_SHADOW
FNV_SAMPLER2D(14, ShadowMap)
FNV_SAMPLER2D(15, ShadowMaskMap)
#endif
#endif

#ifndef SILHOUETTE
layout(location = 1) in vec4 t0;
layout(location = 2) centroid in vec4 t1;
#ifdef PTLIGHTS
layout(location = 3) centroid in vec4 t2;
#endif
layout(location = 4) centroid in vec4 t3;
layout(location = 5) centroid in vec4 t4;
layout(location = 6) centroid in vec4 t5;
#ifdef PROJ_SHADOW
layout(location = 7) centroid in vec4 t6;
#endif
layout(location = 9) in vec4 v0;
#if TEXCOUNT >= 4
layout(location = 10) in vec4 v1;
#endif
#endif
layout(location = 8) centroid in vec4 t7;
layout(location = 0) out vec4 oC0;

#define AmbientColor fnvC[1]
#define PSLightDir fnvC[18]

vec3 nrm(vec3 v) {
  float r = inversesqrt(dot(v, v));
  return vec3(fnvMulLegacy(v.x, r), fnvMulLegacy(v.y, r), fnvMulLegacy(v.z, r));
}

#ifndef SILHOUETTE
float weight(int i) {
#if TEXCOUNT >= 4
  return i == 0 ? v0.x : i == 1 ? v0.y : i == 2 ? v0.z : i == 3 ? v1.x : i == 4 ? v1.y : i == 5 ? v1.z : v1.w;
#else
  return i == 0 ? v0.x : i == 1 ? v0.y : v0.z;
#endif
}

vec3 baseLayer(int i) {
  return i == 0 ? texture(FNV_TEX(BaseMap0), t0.xy).xyz
#if TEXCOUNT >= 2
       : i == 1 ? texture(FNV_TEX(BaseMap1), t0.xy).xyz
#endif
#if TEXCOUNT >= 3
       : i == 2 ? texture(FNV_TEX(BaseMap2), t0.xy).xyz
#endif
#if TEXCOUNT >= 4
       : i == 3 ? texture(FNV_TEX(BaseMap3), t0.xy).xyz
#endif
#if TEXCOUNT >= 5
       : i == 4 ? texture(FNV_TEX(BaseMap4), t0.xy).xyz
#endif
#if TEXCOUNT >= 6
       : i == 5 ? texture(FNV_TEX(BaseMap5), t0.xy).xyz
#endif
#if TEXCOUNT >= 7
       : i == 6 ? texture(FNV_TEX(BaseMap6), t0.xy).xyz
#endif
       : vec3(0.0);
}
#ifdef FNV_WET
// Layer i's average colour (its last mip; the wet world's relief).
vec3 baseLayerMean(int i) {
  return i == 0 ? textureLod(FNV_TEX(BaseMap0), t0.xy, 14.0).xyz
#if TEXCOUNT >= 2
       : i == 1 ? textureLod(FNV_TEX(BaseMap1), t0.xy, 14.0).xyz
#endif
#if TEXCOUNT >= 3
       : i == 2 ? textureLod(FNV_TEX(BaseMap2), t0.xy, 14.0).xyz
#endif
#if TEXCOUNT >= 4
       : i == 3 ? textureLod(FNV_TEX(BaseMap3), t0.xy, 14.0).xyz
#endif
#if TEXCOUNT >= 5
       : i == 4 ? textureLod(FNV_TEX(BaseMap4), t0.xy, 14.0).xyz
#endif
#if TEXCOUNT >= 6
       : i == 5 ? textureLod(FNV_TEX(BaseMap5), t0.xy, 14.0).xyz
#endif
#if TEXCOUNT >= 7
       : i == 6 ? textureLod(FNV_TEX(BaseMap6), t0.xy, 14.0).xyz
#endif
       : vec3(0.0);
}
#endif

#ifndef IGNORE_NORMALS
vec3 normalLayer(int i) {
  return i == 0 ? texture(FNV_TEX(NormalMap0), t0.xy).xyz
#if TEXCOUNT >= 2
       : i == 1 ? texture(FNV_TEX(NormalMap1), t0.xy).xyz
#endif
#if TEXCOUNT >= 3
       : i == 2 ? texture(FNV_TEX(NormalMap2), t0.xy).xyz
#endif
#if TEXCOUNT >= 4
       : i == 3 ? texture(FNV_TEX(NormalMap3), t0.xy).xyz
#endif
#if TEXCOUNT >= 5
       : i == 4 ? texture(FNV_TEX(NormalMap4), t0.xy).xyz
#endif
#if TEXCOUNT >= 6
       : i == 5 ? texture(FNV_TEX(NormalMap5), t0.xy).xyz
#endif
#if TEXCOUNT >= 7
       : i == 6 ? texture(FNV_TEX(NormalMap6), t0.xy).xyz
#endif
       : vec3(0.0);
}
#endif

// World to tangent space (the rows normalised per pixel).
vec3 toTangent(vec3 b0, vec3 b1, vec3 b2, vec3 v) { return vec3(dot(b0, v), dot(b1, v), dot(b2, v)); }

vec3 toLinear(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
vec3 toGamma(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }
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

#ifdef PTLIGHTS
// Point light i (0..2): its diffuse term with falloff.
vec3 pointLight(int i, vec3 n, vec3 b0, vec3 b1, vec3 b2) {
  vec4 p = fnvC[19 + i];
  vec3 d = p.xyz - t2.xyz;
  vec3 s = d * (1.0 / p.w);
  float att = 1.0 - clamp(dot(s, s), 0.0, 1.0);
  return clamp(dot(n, nrm(toTangent(b0, b1, b2, d))), 0.0, 1.0) * att * fnvC[4 + i].xyz;
}
#endif
#endif

void main() {
#ifdef SILHOUETTE
  oC0 = vec4(mix(AmbientColor.xyz, t7.xyz, t7.w), 1.0);
#else
#ifdef FNV_SUN
  const float sunVisibility = fnvSunVisibility();
#else
  const float sunVisibility = 1.0;
#endif
  vec3 b0 = nrm(t3.xyz), b1 = nrm(t4.xyz), b2 = nrm(t5.xyz);
#ifdef IGNORE_NORMALS
  vec3 n = b2;
#else
  vec3 nsum = vec3(0.0);
  for (int i = 0; i < TEXCOUNT; i++) nsum += (normalLayer(i) - 0.5) * weight(i) * 2.0;
  vec3 n = nrm(nsum);
#endif
  vec3 albedo = vec3(0.0);
  for (int i = 0; i < TEXCOUNT; i++) albedo += baseLayer(i) * weight(i);

  vec3 light = clamp(dot(n, toTangent(b0, b1, b2, PSLightDir.xyz)), 0.0, 1.0) * (fnvC[3].xyz * sunVisibility);
#ifdef PTLIGHTS
  light += pointLight(0, n, b0, b1, b2) + pointLight(1, n, b0, b1, b2) + pointLight(2, n, b0, b1, b2);
#endif
#ifdef PROJ_SHADOW
  vec4 sm = texture(FNV_TEX(ShadowMap), t6.xy);
  float mask = texture(FNV_TEX(ShadowMaskMap), t6.zw).x;
  light *= mask * (sm.xyz - 1.0) + 1.0;  // with PTLIGHTS the point lights too, as the original
#endif
  vec3 color = (light + AmbientColor.xyz) * albedo * t1.xyz;
  if (p_pbr) {
#ifdef FNV_WET
    // Wet world (docs/WETWORLD.md): standing water flattens the normal the lights see, the wet ground darkens, and its
    // water film (the sun's highlight and the reflection) is laid over the lit colour below.
    FnvWetSurface wet;
    wet.gloss = 0.0;
    vec3 wp = vec3(0.0);
    if (fnvWetActive()) {
      wp = fnvWetWorldPos();
      const vec3 nWorld = nrm(b0 * n.x + b1 * n.y + b2 * n.z);
      // The ground's relief for the local puddles: each layer's brightness against its own average (its last mip), so
      // its dark gaps and ruts fill first (Wetness.hlsl ObjectPuddleHeight).
      float detail = 0.0;
      for (int i = 0; i < TEXCOUNT; i++) detail += (dot(baseLayer(i), vec3(0.3, 0.59, 0.11)) - dot(baseLayerMean(i), vec3(0.3, 0.59, 0.11))) * weight(i);
      wet = fnvWetness(wp, b2, nWorld, clamp(0.5 + 0.6 * detail, 0.0, 1.0), 1.0);
      if (wet.flatten > 0.0) n = nrm(toTangent(b0, b1, b2, normalize(mix(nWorld, b2, wet.flatten))));
      albedo = toGamma(fnvWetAlbedo(toLinear(albedo), wet.darkening));
    }
#endif
    vec3 diffuse = toLinear(fnvC[3].xyz) * sunVisibility * clamp(dot(n, toTangent(b0, b1, b2, PSLightDir.xyz)), 0.0, 1.0);
#ifdef PTLIGHTS
    for (int i = 0; i < 3; i++) {
      vec4 p = fnvC[19 + i];
      vec3 d = p.xyz - t2.xyz;
      vec3 s = d * (1.0 / p.w);
      float att = 1.0 - clamp(dot(s, s), 0.0, 1.0);
      diffuse += toLinear(fnvC[4 + i].xyz) * att * clamp(dot(n, nrm(toTangent(b0, b1, b2, d))), 0.0, 1.0);
    }
#endif
#ifdef PROJ_SHADOW
    diffuse *= mask * (sm.xyz - 1.0) + 1.0;  // the shadow on every light, as the original
#endif
    color = albedo * t1.xyz * toGamma(diffuse + pbrIndirect(AmbientColor.xyz, nrm(b0 * n.x + b1 * n.y + b2 * n.z),
                                                            fnvAmbientOcclusion()));
#ifdef FNV_WET
    if (wet.gloss > 0.0) {
      // The film over the lit ground, in linear light: what it reflects (the environment probe along the reflected view,
      // at the film's roughness; the sky's ambient colour without a probe) by its reflectance, plus the sun's highlight
      // (shadowed as the sun's light is); what passes the film is the lit ground. Deep water tints both (murk).
      const vec3 V = normalize(fnvWetCamera() - wp);
      const vec3 L = normalize(PSLightDir.xyz);
      const float R = fnvWetReflectance(clamp(dot(wet.normal, V), 0.0, 1.0), wet.roughness, wet.gloss);
      vec3 sky = toLinear(AmbientColor.xyz);
#ifdef FNV_ENV
      if (fnvEnvValid()) sky = fnvEnvRadiance(reflect(-V, wet.normal), wet.roughness * wet.roughness);
#endif
      vec3 spec = fnvWetSpecular(wet.normal, V, L, toLinear(fnvC[3].xyz) * sunVisibility, wet.roughness) * wet.gloss;
#ifdef PROJ_SHADOW
      spec *= mask * (sm.xyz - 1.0) + 1.0;
#endif
      color = toGamma(toLinear(color) * (1.0 - R) + (sky * R + spec) * fnvWetMurk(wet.murk));
    }
#endif
  }
  oC0 = vec4(mix(color, t7.xyz, t7.w), 1.0);
#endif
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
