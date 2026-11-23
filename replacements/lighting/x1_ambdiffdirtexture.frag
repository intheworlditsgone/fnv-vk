// lighting_1x_p_ambDiffDirTexture_p.hlsl: the game's 1.x object lighting (ambient, the sun's diffuse, texture),
// reconstructed from its compiled variants SLS1009..SLS1028 (tools/lightingtable.py; verified with
// tools/replacediff.py). Defines: VC (vertex colour v0), ALPHA (alpha times AmbientColor.w), FOG (fogged with v0, or v1
// with VC), SI (glow map at t2 added to the light), FACEGENBLEND (face tint: FaceGenMap at t2 as a detail, FaceGenMap2
// at t3 as a 2x tint; the light's direction is v0, or v1 with FOG).
//
// The light: saturate(saturate(N.L) PSLightColor[0] + AmbientColor [+ glow]); N the normal map at t1, L the encoded
// direction t3, neither normalised. Alpha: the base map's.

FNV_SAMPLER2D(0, BaseMap)
FNV_SAMPLER2D(1, NormalMap)
#ifdef SI
FNV_SAMPLER2D(2, GlowMap)
#endif
#ifdef FACEGENBLEND
FNV_SAMPLER2D(2, FaceGenMap)
FNV_SAMPLER2D(3, FaceGenMap2)
#endif

layout(location = 1) in vec4 t0;
layout(location = 2) in vec4 t1;
#if defined(SI) || defined(FACEGENBLEND)
layout(location = 3) in vec4 t2;
#endif
layout(location = 4) in vec4 t3;
#if defined(VC) || defined(FOG) || defined(FACEGENBLEND)
layout(location = 9) in vec4 v0;
#endif
#if (defined(VC) || defined(FACEGENBLEND)) && defined(FOG)
layout(location = 10) in vec4 v1;
#endif
layout(location = 0) out vec4 oC0;

#define AmbientColor fnvC[1]

void main() {
#ifdef FNV_SUN
  const float sunVisibility = fnvSunVisibility();
#else
  const float sunVisibility = 1.0;
#endif
  vec4 base = texture(FNV_TEX(BaseMap), t0.xy);
  vec3 n = (texture(FNV_TEX(NormalMap), t1.xy).xyz - 0.5) * 2.0;
#if defined(FACEGENBLEND) && defined(FOG)
  vec3 l = (v1.xyz - 0.5) * 2.0;
#elif defined(FACEGENBLEND)
  vec3 l = (v0.xyz - 0.5) * 2.0;
#else
  vec3 l = (t3.xyz - 0.5) * 2.0;
#endif
  vec3 light = AmbientColor.xyz;
#ifdef SI
  light += texture(FNV_TEX(GlowMap), t2.xy).xyz;
#endif
  light = clamp(clamp(dot(n, l), 0.0, 1.0) * (fnvC[3].xyz * sunVisibility) + light, 0.0, 1.0);

#ifdef FACEGENBLEND
  vec3 detail = (texture(FNV_TEX(FaceGenMap), t2.xy).xyz - 0.5) * 2.0 + base.xyz;
  vec3 albedo = texture(FNV_TEX(FaceGenMap2), t3.xy).xyz * 2.0 * detail * 2.0;
#elif defined(VC)
  vec3 albedo = base.xyz * v0.xyz;
#else
  vec3 albedo = base.xyz;
#endif
  vec3 color = light * albedo;
#if defined(FOG) && (defined(VC) || defined(FACEGENBLEND))
  #if defined(VC)
  color = mix(color, v1.xyz, v1.w);
  #else
  color = mix(color, v0.xyz, v0.w);
  #endif
#elif defined(FOG)
  color = mix(color, v0.xyz, v0.w);
#endif
#ifdef ALPHA
  float alpha = base.w * AmbientColor.w;
#else
  float alpha = base.w;
#endif
  oC0 = vec4(color, alpha);
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
