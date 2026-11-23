// lighting_1x_p_vertDir_p.hlsl and lighting_1x_p_vertDirAndPt_p.hlsl: the game's per-vertex lit 1.x pass,
// reconstructed from their compiled variants SLS1034, SLS1035 (tools/lightingtable.py; verified with
// tools/replacediff.py). Defines: AND_PT (the second source: a point light too). The light: v0 times PSLightColor[0]
// plus AmbientColor (plus v1 times PSLightColor[1]), times the diffuse map; alpha the diffuse map's.

FNV_SAMPLER2D(0, DiffuseMap)
layout(location = 1) in vec4 t0;
layout(location = 9) in vec4 v0;
#ifdef AND_PT
layout(location = 10) in vec4 v1;
#endif
layout(location = 0) out vec4 oC0;

void main() {
#ifdef FNV_SUN
  const float sunVisibility = fnvSunVisibility();
#else
  const float sunVisibility = 1.0;
#endif
  vec3 light = v0.xyz * (fnvC[3].xyz * sunVisibility) + fnvC[1].xyz;
#ifdef AND_PT
  light += v1.xyz * fnvC[4].xyz;
#endif
  vec4 base = texture(FNV_TEX(DiffuseMap), t0.xy);
  oC0 = vec4(light * base.xyz, base.w);
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
