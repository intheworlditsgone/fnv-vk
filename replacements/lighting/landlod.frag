// lighting_2x_p_landlod_p.hlsl: the game's distant (LOD) landscape, reconstructed from its compiled variants SLS2003,
// SLS2006 and SLS2149 (tools/lightingtable.py; verified with tools/replacediff.py). Defines: SILHOUETTE (the ambient
// colour fogged, nothing else). The table's other defines (MODELSPACENORM, LODLANDCLIP, LODLANDNOISE, PROJ_SHADOW) do
// not change the shipped builds: SLS2003 and SLS2006 are the same program, with the noise and without shadows.
//
// The block's base and normal maps fade into its parent block's (LODParentTex s4, LODParentNormals s6; LODTexParams
// c31: xy the block's offset in the parent, w the fade, 1 or more meaning no parent); a noise map (s7) varies the
// brightness. The light: PSLightColor[0] (c3) along t1 (model space), ambient c1, fogged with v1 (rgb colour, w amount).
//
// Physically based mode (parameter pbr): the light and ambient summed in linear space, as the terrain's
// (lando.frag).

#ifndef SILHOUETTE
FNV_SAMPLER2D(0, BaseMap)
FNV_SAMPLER2D(1, NormalMap)
FNV_SAMPLER2D(4, LODParentTex)
FNV_SAMPLER2D(6, LODParentNormals)
FNV_SAMPLER2D(7, LODLandNoise)
layout(location = 1) in vec4 t0;
layout(location = 2) centroid in vec4 t1;
#endif
layout(location = 10) in vec4 v1;
layout(location = 0) out vec4 oC0;

#define AmbientColor fnvC[1]
#define LODTexParams fnvC[31]

void main() {
#ifdef SILHOUETTE
  oC0 = vec4(mix(AmbientColor.xyz, v1.xyz, v1.w), 1.0);
#else
#ifdef FNV_SUN
  const float sunVisibility = fnvSunVisibility();
#else
  const float sunVisibility = 1.0;
#endif
  bool noParent = LODTexParams.w - 1.0 >= 0.0;
  // Base: the block's texture inset by half a texel of 128, the parent's at the block's quarter.
  vec2 uv = t0.xy * 0.9921875 + 0.00390625;
  vec2 parentUv = uv * 0.5 + mix(LODTexParams.xy, vec2(0.25), 0.0078125);
  vec3 base = texture(FNV_TEX(BaseMap), uv).xyz;
  base = noParent ? base : mix(texture(FNV_TEX(LODParentTex), parentUv).xyz, base, LODTexParams.w);
  base *= texture(FNV_TEX(LODLandNoise), t0.xy * 1.75).x * 0.8 + 0.55;
  vec3 nm = texture(FNV_TEX(NormalMap), t0.xy).xyz;
  nm = noParent ? nm : mix(texture(FNV_TEX(LODParentNormals), t0.xy * 0.5 + LODTexParams.xy).xyz, nm, LODTexParams.w);
  vec3 n = (nm - 0.5) * 2.0;
  vec3 light = max(fnvC[3].xyz * sunVisibility * clamp(dot(n, t1.xyz), 0.0, 1.0) + AmbientColor.xyz, 0.0);
  vec3 color = base * light;
  if (p_pbr) {
    vec3 sun = pow(max(fnvC[3].xyz, 0.0), vec3(2.2)) * sunVisibility * clamp(dot(n, t1.xyz), 0.0, 1.0);
    color = base * pow(sun + pow(max(AmbientColor.xyz, 0.0), vec3(2.2)), vec3(1.0 / 2.2));
  }
  oC0 = vec4(mix(color, v1.xyz, v1.w), 1.0);
#endif
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
