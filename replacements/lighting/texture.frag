// lighting_2x_p_Texture_p.hlsl: the game's texture-only pass (PAR2023, with PARALLAX), reconstructed from its compiled
// variant (tools/lightingtable.py; verified with tools/replacediff.py). The base map at coordinates shifted along the
// view vector t6 by HeightMap (s2), times the vertex colour when Toggles.x is set; alpha 1.

FNV_SAMPLER2D(0, BaseMap)
#ifdef PARALLAX
FNV_SAMPLER2D(2, HeightMap)
layout(location = 7) centroid in vec4 t6;  // the view vector (tangent space)
#define VIEW t6
#endif
layout(location = 1) in vec4 t0;
layout(location = 9) in vec4 v0;
layout(location = 0) out vec4 oC0;

#ifdef PARALLAX
vec2 parallaxUv() {
  float h = texture(FNV_TEX(HeightMap), t0.xy).x * 0.04 - 0.02;
  return VIEW.xy * inversesqrt(dot(VIEW.xyz, VIEW.xyz)) * h + t0.xy;
}
#endif

void main() {
#ifdef PARALLAX
  vec3 base = texture(FNV_TEX(BaseMap), parallaxUv()).xyz;
#else
  vec3 base = texture(FNV_TEX(BaseMap), t0.xy).xyz;
#endif
  oC0 = vec4(-fnvC[27].x >= 0.0 ? base : base * v0.xyz, 1.0);
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
