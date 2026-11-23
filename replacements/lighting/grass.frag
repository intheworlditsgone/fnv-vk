// tallgrass\1x\p\highDetail.p.hlsl and highDetail_1pt.p.hlsl: the game's grass (TallGrassShader, GRASS2nnn and the
// ps_3_0 GRASS23xnnn, each also TMS), reconstructed from their compiled variants (tools/ida_shadertables.py,
// tools/lightingtable.py; verified with tools/replacediff.py). Defines: PT (the _1pt source: a point light), PROJ_SHADOW
// (the game's projected shadows on the sun), TRANSPARENCYMS (alpha to coverage: alpha saturate(1.75 a) instead of the
// cut at AlphaTestRef), PS_2_0 (every variant). The shadow-map rows (SHADOWMAP: GRASS2004..2006) stay translated.
//
// Lit per vertex: t5 the sun's diffuse (w: fade), t4 the ambient; the point light 0.4 PointLightColor (c2) times the
// attenuation from AttMap (s1) at t1 (1 - a(xy) - a(zw)); the shadow ShadowMap at t2, ShadowMaskMap at t3, by t3.z.
// Fogged with v0 (rgb colour, w amount). Alpha: 0 where the diffuse map's alpha is at or below AlphaTestRef (c3.x),
// else 1, times t5.w.
//
// Physically based mode (parameter pbr; off: exactly the original; docs/LIGHTING.md "Grass"): Community Shaders' Grass
// Lighting (GPL-3.0; RunGrass.hlsl, GrassLighting.hlsli, PBRMath.hlsli's thin surfaces) with its defaults (subsurface
// scattering 1, balance 1, canopy occlusion 0.5, clump shading 0.5; FNV's grass has no normal maps, so no specular): a
// thin sheet reflecting half its albedo toward the sun and passing the light through from behind, the blade's normal
// bent halfway toward its tuft's dome (the clump), and the light occluded toward the tuft's base (the canopy). The
// inputs come from the grass vertex shaders (sdp2spv's grass pass, locations 12..15): the blade's normal, the canopy
// fraction, the clump vector, the sun's direction and colour (with the instance's occlusion) and the vertex colour.
// The ambient (t4: the game's, with the instance's occlusion) reaches both faces; the point light keeps the game's term.

#ifdef PT
#define SHADOW_REG 2
#else
#define SHADOW_REG 1
#endif

FNV_SAMPLER2D(0, DiffuseMap)
#ifdef PT
FNV_SAMPLER2D(1, AttMap)
layout(location = 2) in vec4 t1;
#endif
#ifdef PROJ_SHADOW
FNV_SAMPLER2D(SHADOW_REG, ShadowMap)
FNV_SAMPLER2D(SHADOW_REG + 1, ShadowMaskMap)
layout(location = 3) centroid in vec4 t2;
layout(location = 4) centroid in vec4 t3;
#endif
layout(location = 1) in vec4 t0;
layout(location = 5) centroid in vec4 t4;
layout(location = 6) centroid in vec4 t5;
layout(location = 9) in vec4 v0;
layout(location = 12) in vec4 fnvGrass0;  // xyz the blade's normal (world), w the canopy fraction; 0: none
layout(location = 13) in vec4 fnvGrass1;  // xyz the clump vector, w the sun's direction x
layout(location = 14) in vec4 fnvGrass2;  // xyz the sun's colour (gamma), w its direction y
layout(location = 15) in vec4 fnvGrass3;  // xyz the vertex colour (gamma), w the sun's direction z
layout(location = 0) out vec4 oC0;

vec3 toLinear(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
vec3 toGamma(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }

// Community Shaders' Grass Lighting with its defaults, in linear light (the game's light colour c is the diffuse light
// on a white surface facing it). shadow: the sun's (the game's projected shadows); base: the diffuse map (gamma).
vec3 grassLight(vec3 base, vec3 shadow) {
  const float kClumpShading = 0.5, kCanopyOcclusion = 0.5;
  vec3 n = normalize(fnvGrass0.xyz);
  if (!gl_FrontFacing) n = -n;  // the blade's lit side follows its winding
  const vec3 clump = normalize(fnvGrass1.xyz);
  n = normalize(n + (clump - n * dot(n, clump)) * kClumpShading);  // GetClumpBlendedNormal
  const vec3 L = normalize(vec3(fnvGrass1.w, fnvGrass2.w, fnvGrass3.w));
  const float canopy = 1.0 - kCanopyOcclusion * (1.0 - clamp(fnvGrass0.w, 0.0, 1.0));
  const float clumpVis = mix(1.0, clamp(dot(clump, L) * 0.5 + 0.5, 0.0, 1.0), kClumpShading);
  // The vertex colour: its brightest channel is occlusion, the rest a tint (CS: vertexColor / vertexAO).
  const vec3 vc = toLinear(fnvGrass3.rgb);
  const float vao = toLinear(vec3(max(max(fnvGrass3.r, fnvGrass3.g), fnvGrass3.b))).x;
  const vec3 albedo = toLinear(base) * (vc / max(vao, 1e-6));
  // Subsurface 1, balance 1 (GetThinSurfaceSubsurfaceAlbedos): the lit face reflects half, the rest passes to the back.
  const vec3 reflectionAlbedo = 0.5 * albedo, backAlbedo = 0.5 * albedo;
  const vec3 transmissionAlbedo = clamp(albedo * 0.5 + backAlbedo, 0.0, 1.0);  // shine-through at opacity 0.5
  const vec3 E = toLinear(fnvGrass2.rgb) * shadow * (canopy * clumpVis);
  const float nl = dot(n, L);
  vec3 c = reflectionAlbedo * clamp(nl, 0.0, 1.0) * E + transmissionAlbedo * clamp(-nl, 0.0, 1.0) * E;
  // The ambient on both faces: the reflection's share and the back's, by the canopy.
  c += (reflectionAlbedo + backAlbedo) * toLinear(t4.xyz) * canopy;
#ifdef PT
  const float att = clamp((1.0 - texture(FNV_TEX(AttMap), t1.xy).x) - texture(FNV_TEX(AttMap), t1.zw).x, 0.0, 1.0);
  c += albedo * toLinear(0.4 * fnvC[2].xyz) * att;
#endif
  return toGamma(c);
}

void main() {
  vec3 sun = t5.xyz;
#ifdef PROJ_SHADOW
  vec3 sm = texture(FNV_TEX(ShadowMap), t2.xy).xyz;
  float mask = texture(FNV_TEX(ShadowMaskMap), t3.xy).x;
  sun *= t3.z * (mask * (sm - 1.0)) + 1.0;
#endif
  vec3 light = sun + t4.xyz;
#ifdef PT
  float att = clamp((1.0 - texture(FNV_TEX(AttMap), t1.xy).x) - texture(FNV_TEX(AttMap), t1.zw).x, 0.0, 1.0);
  light += 0.4 * fnvC[2].xyz * att;
#endif
  vec4 base = texture(FNV_TEX(DiffuseMap), t0.xy);
#ifdef TRANSPARENCYMS
  float a = clamp(base.w * 1.75, 0.0, 1.0);
#else
  float a = fnvC[3].x - base.w >= 0.0 ? 0.0 : 1.0;
#endif
  vec3 lit = light * base.xyz;
  if (p_pbr && dot(fnvGrass0.xyz, fnvGrass0.xyz) > 0.0) {
#ifdef PROJ_SHADOW
    lit = grassLight(base.xyz, t3.z * (mask * (sm - 1.0)) + 1.0);
#else
    lit = grassLight(base.xyz, vec3(1.0));
#endif
  }
  oC0 = vec4(mix(lit, v0.xyz, v0.w), a * t5.w);
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
