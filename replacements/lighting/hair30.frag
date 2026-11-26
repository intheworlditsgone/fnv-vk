// lighting\3x\SM3Lighting.p.hlsl's hair builds (Lighting30Shader, which draws hair, beards and brows: the property's
// shader is switched to it by BSShaderPPLightingProperty::ClarifyShader 0x00B68880 on GPUs DetectGpuVendor 0x00B4F710
// rates for shader model 3), reconstructed from the compiled SM3002 / SM3003 / SM3014 / SM3015 (MAXLIGHTS=4) and
// SM3LL002 / 003 / 014 / 015 (MAXLIGHTS=2) of packages 010..013 (Lighting30Shader::LoadPixelShaders 0x00BBA5D0 rows HAIR,
// HAIR SPECULAR, HAIR PROJSHADOW, HAIR SPECULAR PROJSHADOW; verified against them on D3D9 with tools/replacediff.py).
// Defines: SPECULAR (the strand highlights: AnisoMap s4, EyePosition), PROJ_SHADOW (the game's actor shadows on the
// sun: ShadowMap s2, ShadowMask s3 at t1), MAXLIGHTS (lights in the pass, the sun included).
//
// The lighting is in tangent space: t3, t4, t5 the tangent, binormal and normal and t6 the position, all in the space of
// the vertex shader's position (model, or skinned), as the lights (LightData) and EyePosition are. ToggleNumLights: x > 0
// the sun is light 0 (colour LightData[0], direction LightData[1]); point light i: colour LightData[2i], position
// LightData[2i + 1] (w its radius), i from (x > 0) while i < trunc(min(x + y, MAXLIGHTS)); w the sun's highlight scale.
//
// Physically based mode (parameter pbr; off: exactly the original): Community Shaders' hair (FNV_HAIR; docs/LIGHTING.md
// "Hair"): the Marschner model on the same lights, its ambient weight on the ambient colour (with the ambient occlusion
// and the screen-space indirect light), the tangent shift map and the self shadow; the albedo is the original's (the hair
// tint, the layer map), the fog the original's.

#ifndef MAXLIGHTS
#define MAXLIGHTS 4
#endif

FNV_SAMPLER2D(0, BaseMap)
FNV_SAMPLER2D(1, NormalMap)
#ifdef PROJ_SHADOW
FNV_SAMPLER2D(2, ShadowMap)
FNV_SAMPLER2D(3, ShadowMaskMap)
#endif
#ifdef SPECULAR
FNV_SAMPLER2D(4, AnisoMap)
#endif
FNV_SAMPLER2D(5, LayerMap)

layout(location = 1) in vec4 t0;
#ifdef PROJ_SHADOW
layout(location = 2) in vec4 t1;
#endif
layout(location = 4) centroid in vec4 t3;
layout(location = 5) centroid in vec4 t4;
layout(location = 6) centroid in vec4 t5;
layout(location = 7) centroid in vec4 t6;
layout(location = 8) centroid in vec4 t7;
layout(location = 9) in vec4 v0;
layout(location = 0) out vec4 oC0;

#define AmbientColor fnvC[0]
#define EyePosition fnvC[1]
#define HairTint fnvC[2]
#define MatAlpha fnvC[3]
#define ToggleNumLights fnvC[20]
#define LightData(i) fnvC[21 + (i)]

// D3D9 nrm: v * rsq(dot(v, v)) with the legacy multiply (a zero vector stays zero).
vec3 nrm(vec3 v) {
  float r = inversesqrt(dot(v, v));
  return vec3(fnvMulLegacy(v.x, r), fnvMulLegacy(v.y, r), fnvMulLegacy(v.z, r));
}
// D3D9 pow: exp2(y * log2(|x|)), the legacy multiply making pow(0, 0) = 1.
float powLegacy(float x, float y) { return exp2(fnvMulLegacy(y, log2(abs(x)))); }

vec3 toLinear(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
vec3 toGamma(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }

void main() {
#ifdef FNV_SUN
  const float sunVisibility = fnvSunVisibility();  // before any discard: it takes screen derivatives
#else
  const float sunVisibility = 1.0;
#endif
#ifdef FNV_HAIR
  // The self shadow's frame (screen derivatives).
  const FnvHairFrame frame = fnvHairFrame(t6.xyz);
#endif
  // Every map sampled before the alpha cut-off's discard (it ends the helper lanes the implicit mip levels need).
  const vec4 base = texture(FNV_TEX(BaseMap), t0.xy);
  const vec4 nmap = texture(FNV_TEX(NormalMap), t0.xy);
  const vec4 layer = texture(FNV_TEX(LayerMap), t0.xy);
#ifdef PROJ_SHADOW
  const vec3 shadowMap = texture(FNV_TEX(ShadowMap), t1.xy).xyz;
  const float shadowMask = texture(FNV_TEX(ShadowMaskMap), t1.zw).x;
#endif
  if ((MatAlpha.x - 1.0 >= 0.0 ? 0.0 : 1.0) * (base.w - MatAlpha.y) < 0.0) discard;
  const vec3 n = nrm((nmap.xyz - 0.5) * 2.0);
  // The tangent frame (rows), to which the lights and the eye are taken.
  const vec3 T = nrm(t3.xyz), B = nrm(t4.xyz), N = nrm(t5.xyz);
#define TANGENT(v) vec3(dot(T, (v)), dot(B, (v)), dot(N, (v)))

  const bool sun = ToggleNumLights.x > 0.0;
  const int first = sun ? 1 : 0;
  const float count = trunc(min(ToggleNumLights.x + ToggleNumLights.y, float(MAXLIGHTS)));
  const vec3 sunDir = TANGENT(LightData(1).xyz);  // not normalised (as the original)
  vec3 shadow = vec3(1.0);  // the game's actor shadows on the sun (diffuse and highlight)
#ifdef PROJ_SHADOW
  shadow = mix(vec3(1.0), shadowMap, shadowMask);
#endif
  // Light i's colour: light 0 is the sun's slot, shadowed (the sun shadow gate passes only when it holds the sun).
#define LIGHT_COLOUR(i) (LightData(2 * (i)).xyz * ((i) == 0 ? sunVisibility : 1.0))

  const vec3 tint = ((HairTint.xyz - 0.5) * v0.y + 0.5) * 2.0;
  const vec3 albedo = tint * mix(base.xyz, layer.xyz, layer.w);

  vec3 diffuse = sun ? max(dot(n, sunDir), 0.0) * LIGHT_COLOUR(0) * shadow : vec3(0.0);
#ifdef SPECULAR
  const vec3 view = TANGENT(EyePosition.xyz - t6.xyz);
  const vec3 viewN = view * inversesqrt(dot(view, view));
  // Two strand directions bent from the normal; the sun's highlight is the anisotropy map at (T.L, T.H) of each.
  const vec3 bentA = n * 0.5 + vec3(0.0, 0.0, 1.0);
  const vec3 strand1 = bentA * inversesqrt(dot(bentA, bentA));
  const vec3 strand2 = nrm(strand1 + vec3(-0.05, 0.0, 0.0));
  const vec3 sunHalf = nrm(viewN + sunDir);
  const float aniso2 = texture(FNV_TEX(AnisoMap), vec2(dot(strand2, sunDir), dot(strand2, sunHalf))).w;
  const float aniso1 = texture(FNV_TEX(AnisoMap), vec2(dot(strand1, sunDir), dot(strand1, sunHalf))).w;
  vec3 spec = sun ? (tint * aniso2 + aniso1 * 0.7) * LIGHT_COLOUR(0) * max(sunDir.z, 0.0) * ToggleNumLights.w * shadow
                  : vec3(0.0);
#endif
  for (int k = 0; k < MAXLIGHTS; k++) {
    const int i = first + k;
    if (float(i) - count >= 0.0 || i > 3) break;
    const vec4 pos = LightData(2 * i + 1);
    const vec3 d = pos.xyz - t6.xyz;
    const float r = clamp(fnvMulLegacy(1.0 / pos.w, sqrt(dot(d, d))), 0.0, 1.0);
    const float att = 1.0 - r * r;
    const vec3 L = nrm(TANGENT(d));
    diffuse += LIGHT_COLOUR(i) * max(clamp(dot(n, L), 0.0, 1.0) * att, 0.0);
#ifdef SPECULAR
    // A point light's highlight: along the first strand direction, power 30.
    const float x = clamp(abs(dot(strand1, L) - dot(strand1, nrm(viewN + L))), 0.0, 1.0);
    spec += att * LIGHT_COLOUR(i) * (powLegacy(1.0 - x, 30.0) * 0.7) * max(L.z, 0.0);
#endif
  }
  diffuse += AmbientColor.xyz;
  vec3 lit = diffuse * albedo;
#ifdef SPECULAR
  lit += spec * v0.y * (nmap.w * LightData(1).w);
#endif

#ifdef FNV_HAIR
  if (p_pbr) {
    // Community Shaders' hair (docs/LIGHTING.md "Hair"): each light's Marschner lobes and diffuse scatter, shadowed by the
    // sun shadows, the game's actor shadows and the self shadow; the ambient by its weight across the strand.
    const vec3 eye = TANGENT(EyePosition.xyz - t6.xyz);
    const vec3 V = dot(eye, eye) > 0.0 ? normalize(eye) : vec3(0.0, 0.0, 1.0);
    const vec3 baseLin = fnvHairBaseColor(albedo);
    const vec3 strand = fnvHairStrand(n, t0.xy);
    vec3 hair = vec3(0.0);
    if (sun && dot(sunDir, sunDir) > 0.0) {
      const float vis = sunVisibility * dot(shadow, vec3(1.0 / 3.0)) * fnvHairSelfShadow(frame, LightData(1).xyz);
      hair += fnvHairLight(strand, V, normalize(sunDir), toLinear(LightData(0).xyz), vis, baseLin);
    }
    for (int k = 0; k < MAXLIGHTS; k++) {
      const int i = first + k;
      if (float(i) - count >= 0.0 || i > 3) break;
      const vec4 pos = LightData(2 * i + 1);
      const vec3 d = pos.xyz - t6.xyz;
      const float r = clamp(fnvMulLegacy(1.0 / pos.w, sqrt(dot(d, d))), 0.0, 1.0);
      const vec3 L = nrm(TANGENT(d));
      if (dot(L, L) == 0.0) continue;
      const float vis = (i == 0 ? sunVisibility : 1.0) * fnvHairSelfShadow(frame, d);
      hair += fnvHairLight(strand, V, L, toLinear(LightData(2 * i).xyz) * (1.0 - r * r), vis, baseLin);
    }
    vec3 ambient = toLinear(AmbientColor.xyz) * fnvAmbientOcclusion();
    vec4 gi;
    if (fnvScreenIndirect(gi)) ambient = gi.rgb + gi.a * ambient;
    hair += fnvHairIndirect(strand, V, baseLin) * ambient;
    lit = toGamma(hair);
  }
#endif

  // The fog, and the pass's blend modes (MatAlpha z: fog to black, w: fog to white).
  const float f = t7.w;
  const vec3 fogged = mix(lit, t7.xyz, f);
  const vec3 toBlack = lit - f * lit;
  const vec3 toWhite = mix(lit, vec3(1.0), f);
  oC0 = vec4(mix(mix(fogged, toBlack, MatAlpha.z), toWhite, MatAlpha.w), base.w * MatAlpha.x);
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
