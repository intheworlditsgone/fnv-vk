// lighting_2x_p_land_p.hlsl and lighting_1x_p_land_p.hlsl: the game's single-texture landscape shaders, reconstructed
// from their compiled variants (SLS2060..SLS2086, SLS1040..SLS1045; tools/lightingtable.py; verified with
// tools/replacediff.py). Defines: SM1X (the 1.x source: light saturated, inputs not centroid), PTLIGHT (a point light's
// pass: no ambient), SI (the glow map times the vertex colour, unlit), LANDALPHA (alpha: the layer weights dotted with
// PSLightColor[1] and [2]), PROJ_SHADOW (the game's projected shadows on the light), LANDLO with FOG (the low-detail
// land: a noise map darkens the base, fogged with t5, alpha t4.x).
//
// The light's direction t3 and the normal map are encoded 0..1 and used without normalising; t2 is the vertex colour.
// Constants: c1 AmbientColor, c3 PSLightColor[0] (the light of this pass).
//
// Physically based mode (parameter pbr; off: exactly the original): as every lighting shader (adts.frag,
// docs/LIGHTING.md): the light and ambient summed in linear space (the glow pass unchanged).

#ifdef SM1X
#define CENTROID
#else
#define CENTROID centroid
#endif

FNV_SAMPLER2D(0, BaseMap)
#ifdef SI
FNV_SAMPLER2D(1, GlowMap)
#else
FNV_SAMPLER2D(1, NormalMap)
#endif
#ifdef LANDLO
FNV_SAMPLER2D(2, LODLandNoise)
#endif
#ifdef PROJ_SHADOW
FNV_SAMPLER2D(2, ShadowMap)
FNV_SAMPLER2D(3, ShadowMaskMap)
#endif

layout(location = 1) in vec4 t0;
#ifdef LANDLO
layout(location = 4) centroid in vec4 t3;
layout(location = 5) centroid in vec4 t4;
layout(location = 6) centroid in vec4 t5;
#else
layout(location = 2) in vec4 t1;
layout(location = 3) CENTROID in vec4 t2;
#ifndef SI
layout(location = 4) CENTROID in vec4 t3;
#endif
#ifdef PROJ_SHADOW
layout(location = 5) in vec4 t4;
#endif
#endif
#ifdef LANDALPHA
layout(location = 9) in vec4 v0;
layout(location = 10) in vec4 v1;
#endif
layout(location = 0) out vec4 oC0;

#define AmbientColor fnvC[1]

vec3 toLinear(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
vec3 toGamma(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }

void main() {
#if defined(FNV_SUN) && !defined(SI)
  const float sunVisibility = fnvSunVisibility();
#else
  const float sunVisibility = 1.0;
#endif
  vec3 base = texture(FNV_TEX(BaseMap), t0.xy).xyz;
#ifdef SI
  vec3 color = base * t2.xyz * texture(FNV_TEX(GlowMap), t1.xy).xyz;
#elif defined(LANDLO)
  float noise = texture(FNV_TEX(LODLandNoise), t0.xy * 1.75).x * 0.8 + 0.55;
  vec3 n = (texture(FNV_TEX(NormalMap), t0.xy).xyz - 0.5) * 2.0;
  vec3 light = clamp(dot(n, t3.xyz), 0.0, 1.0) * (fnvC[3].xyz * sunVisibility) + AmbientColor.xyz;
  if (p_pbr) light = toGamma(toLinear(fnvC[3].xyz) * sunVisibility * clamp(dot(n, t3.xyz), 0.0, 1.0) + toLinear(AmbientColor.xyz));
  vec3 color = mix(base * noise * light, t5.xyz, t5.w);
#else
  vec3 n = (texture(FNV_TEX(NormalMap), t1.xy).xyz - 0.5) * 2.0;
  vec3 light = clamp(dot(n, (t3.xyz - 0.5) * 2.0), 0.0, 1.0) * (fnvC[3].xyz * sunVisibility);
#ifdef PROJ_SHADOW
  vec4 sm = texture(FNV_TEX(ShadowMap), t4.xy);
  float mask = texture(FNV_TEX(ShadowMaskMap), t4.zw).x;
  light *= mask * (sm.xyz - 1.0) + 1.0;
#endif
#ifndef PTLIGHT
  light += AmbientColor.xyz;
#endif
#ifdef SM1X
  light = clamp(light, 0.0, 1.0);
#endif
  if (p_pbr) {
    vec3 sum = toLinear(fnvC[3].xyz) * sunVisibility * clamp(dot(n, (t3.xyz - 0.5) * 2.0), 0.0, 1.0);
#ifdef PROJ_SHADOW
    sum *= mask * (sm.xyz - 1.0) + 1.0;
#endif
#ifndef PTLIGHT
    sum += toLinear(AmbientColor.xyz);
#endif
    light = toGamma(sum);
#ifdef SM1X
    light = clamp(light, 0.0, 1.0);
#endif
  }
  vec3 color = light * (base * t2.xyz);
#endif

#ifdef LANDALPHA
  float alpha = dot(fnvC[4], v0) + dot(fnvC[5], v1);
#elif defined(LANDLO)
  float alpha = t4.x;
#else
  float alpha = 1.0;
#endif
  oC0 = vec4(color, alpha);
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
