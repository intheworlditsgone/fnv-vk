// lighting_2x_p_DiffusePt_p.hlsl: the game's diffuse light of 2 or 3 point lights for objects (the light only),
// reconstructed from its compiled variants SLS2045, SLS2046 (tools/lightingtable.py; verified with tools/replacediff.py).
// Defines: LIGHTS (2 or 3), PARALLAX (ParallaxShader: the normal map at coordinates shifted along the view vector t7
// by HeightMap, s2), FACEGENBLEND (SkinShader: skin lighting with the view vector t7, tinted (0.3, 0, 0)).
//
// Light i: direction t1, t2, t3 (tangent space), colour PSLightColor[i] (c3 + i), attenuation from AttenuationMap (s3)
// at t4, t5, t6 (1 - a(xy) - a(zw)). Alpha 1.
//
// Physically based mode (parameter pbr; off: exactly the original): as every lighting shader (adts.frag,
// docs/LIGHTING.md): the pass's lights summed in linear space. Skin (FACEGENBLEND, docs/SKIN.md): the skin model's
// diffuse and the game's light through thin skin; its specular lobes and sheen do not scale with the albedo the Texture
// pass multiplies by, and this pass has no base map to divide by, so they are left to the first pass's lights.

FNV_SAMPLER2D(0, NormalMap)
FNV_SAMPLER2D(3, AttenuationMap)
#ifdef PARALLAX
FNV_SAMPLER2D(2, HeightMap)
#endif

layout(location = 1) in vec4 t0;
layout(location = 2) centroid in vec4 t1;
layout(location = 3) centroid in vec4 t2;
layout(location = 5) in vec4 t4;
layout(location = 6) in vec4 t5;
#if LIGHTS >= 3
layout(location = 4) centroid in vec4 t3;
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

vec3 pointLight(vec3 n, vec3 dir, vec3 color, vec4 coords) {
  float att = clamp((1.0 - texture(FNV_TEX(AttenuationMap), coords.xy).x) -
                    texture(FNV_TEX(AttenuationMap), coords.zw).x, 0.0, 1.0);
#ifdef FACEGENBLEND
  return skinPoint(n, nrm(dir), nrm(t7.xyz), color, vec3(0.3, 0.0, 0.0)) * att;
#else
  return clamp(dot(n, nrm(dir)), 0.0, 1.0) * color * att;
#endif
}

void main() {
#ifdef FNV_SUN
  const float sunVisibility = fnvSunVisibility();
#else
  const float sunVisibility = 1.0;
#endif
#ifdef PARALLAX
  vec3 n = nrm((texture(FNV_TEX(NormalMap), parallaxUv()).xyz - 0.5) * 2.0);
#else
  vec3 n = nrm((texture(FNV_TEX(NormalMap), t0.xy).xyz - 0.5) * 2.0);
#endif
  vec3 light = pointLight(n, t1.xyz, fnvC[3].xyz * sunVisibility, t4) + pointLight(n, t2.xyz, fnvC[4].xyz, t5);
#if LIGHTS >= 3
  light += pointLight(n, t3.xyz, fnvC[5].xyz, t6);
#endif
#ifndef FACEGENBLEND
  if (p_pbr) {
    vec3 sum = pointLight(n, t1.xyz, toLinear(fnvC[3].xyz) * sunVisibility, t4) + pointLight(n, t2.xyz, toLinear(fnvC[4].xyz), t5);
#if LIGHTS >= 3
    sum += pointLight(n, t3.xyz, toLinear(fnvC[5].xyz), t6);
#endif
    light = toGamma(sum);
  }
#else
  if (p_pbr) {
    const vec3 e = nrm(t7.xyz);
    const vec3 v = dot(fnvView.xyz, fnvView.xyz) > 0.0 ? normalize(fnvView.xyz) : e;
    const vec3 ns = fnvSkinDetail(n, t0.xy);
    const FnvSkin m = fnvSkinMaterial(0.0, dot(ns, v));
    const vec3 tintL = toLinear(vec3(0.3, 0.0, 0.0));
    vec3 diffuse = vec3(0.0), spec = vec3(0.0), through = vec3(0.0);
    for (int i = 0; i < LIGHTS; i++) {
      const vec3 dir = i == 0 ? t1.xyz : i == 1 ? t2.xyz
#if LIGHTS >= 3
                     : t3.xyz;
#else
                     : t2.xyz;
#endif
      const vec4 coords = i == 0 ? t4 : i == 1 ? t5
#if LIGHTS >= 3
                        : t6;
#else
                        : t5;
#endif
      const float att = clamp((1.0 - texture(FNV_TEX(AttenuationMap), coords.xy).x) -
                              texture(FNV_TEX(AttenuationMap), coords.zw).x, 0.0, 1.0);
      const vec3 c = toLinear(fnvC[3 + i].xyz) * (i == 0 ? sunVisibility : 1.0), l = nrm(dir);
      fnvSkinLight(m, ns, v, l, c * att, diffuse, spec);
      const float d = dot(ns, l), ds = clamp(d, 0.0, 1.0);
      const float w = clamp((d + 0.3) * 0.769230783, 0.0, 1.0);
      const float wrap = clamp((w * -2.0 + 3.0) * (w * w) - (ds * -2.0 + 3.0) * (ds * ds), 0.0, 1.0);
      const float f = 1.0 - clamp(dot(ns, e), 0.0, 1.0);
      through += (clamp(dot(e, -l), 0.0, 1.0) * (f * f) * mix(c, tintL, 0.5) + wrap * tintL) * att;
    }
    light = toGamma(diffuse + through);
  }
#endif
  oC0 = vec4(light, 1.0);
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
