// lighting_1x_p_base_p.hlsl: the game's 1.x unlit passes, reconstructed from its compiled variants (SLS1000, SLS1003..
// SLS1006, SLS1030, SLS1031, SLS1039, SLS1047, SLS2150; tools/lightingtable.py; verified with tools/replacediff.py).
// Defines: TEX (the diffuse map's colour; without it AmbientColor), VC (times the vertex colour v0), FACEGENBLEND (face
// tint: DecalMap at t1 as a detail, Decal2Map at t2 as a 2x tint), HAIR (LayerMap over the base, tinted by
// EmittanceColor by v0.y), FOG (the vertex's fog v0 as the colour; with TEX alpha times v0.w), FOGANDAMBIENT
// (AmbientColor fogged with v0), CONSTALPHA (colour (1, AmbientColor.w, AmbientColor.z); alpha 1, or with
// AmbientColor.z 0 a cut at diffuse alpha 0.2).

#if !(defined(FOG) && !defined(TEX))
FNV_SAMPLER2D(0, DiffuseMap)
layout(location = 1) in vec4 t0;
#endif
#ifdef FACEGENBLEND
FNV_SAMPLER2D(1, DecalMap)
FNV_SAMPLER2D(2, Decal2Map)
layout(location = 2) in vec4 t1;
layout(location = 3) in vec4 t2;
#endif
#ifdef HAIR
FNV_SAMPLER2D(1, LayerMap)
#endif
#if defined(VC) || defined(HAIR) || defined(FOG)
layout(location = 9) in vec4 v0;
#endif
layout(location = 0) out vec4 oC0;

#define AmbientColor fnvC[1]
#define EmittanceColor fnvC[2]

void main() {
#if defined(FOGANDAMBIENT)
  oC0 = vec4(mix(AmbientColor.xyz, v0.xyz, v0.w), 1.0);
#elif defined(FOG) && defined(TEX)
  oC0 = vec4(v0.xyz, texture(FNV_TEX(DiffuseMap), t0.xy).w * v0.w);
#elif defined(FOG)
  oC0 = v0;
#elif defined(CONSTALPHA)
  float cut = 0.2 - texture(FNV_TEX(DiffuseMap), t0.xy).w >= 0.0 ? 0.0 : 1.0;
  oC0 = vec4(1.0, AmbientColor.w, AmbientColor.z, -(AmbientColor.z * AmbientColor.z) >= 0.0 ? cut : 1.0);
#else
  vec4 base = texture(FNV_TEX(DiffuseMap), t0.xy);
#if defined(FACEGENBLEND)
  vec3 detail = (texture(FNV_TEX(DecalMap), t1.xy).xyz - 0.5) * 2.0 + base.xyz;
  oC0 = vec4(texture(FNV_TEX(Decal2Map), t2.xy).xyz * 2.0 * detail * 2.0, base.w);
#elif defined(HAIR)
  vec4 layer = texture(FNV_TEX(LayerMap), t0.xy);
  oC0 = vec4(((EmittanceColor.xyz - 0.5) * v0.y + 0.5) * 2.0 * mix(base.xyz, layer.xyz, layer.w), base.w);
#elif defined(VC)
  oC0 = vec4(base.xyz * v0.xyz, base.w);
#elif defined(TEX)
  oC0 = base;
#else
  oC0 = vec4(AmbientColor.xyz, base.w);
#endif
#endif
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
