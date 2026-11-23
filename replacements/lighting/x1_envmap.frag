// lighting_1x_p_envmap_p.hlsl: the game's 1.x environment map reflections, reconstructed from its compiled variants
// SLS1032, SLS1033 (tools/lightingtable.py; verified with tools/replacediff.py). Defines: VC (times the vertex colour).
// As lighting_2x_p_EnvMap_p (envmap.frag) without normalising: t1..t3 the rows taking tangent to world space, their w the
// view vector; strength the normal map's alpha or the CustomEnvMask (s2) by Toggles.w, times Toggles.z and
// AmbientColor.w; EnvironmentCubeMap at s3. Alpha 1.

FNV_SAMPLER2D(0, NormalMap)
FNV_SAMPLER2D(2, CustomEnvMask)
FNV_SAMPLERCUBE(3, EnvironmentCubeMap)
layout(location = 1) in vec4 t0;
layout(location = 2) in vec4 t1;
layout(location = 3) in vec4 t2;
layout(location = 4) in vec4 t3;
#ifdef VC
layout(location = 9) in vec4 v0;
#endif
layout(location = 0) out vec4 oC0;

void main() {
  vec3 e = vec3(t1.w, t2.w, t3.w);
  vec4 nmap = texture(FNV_TEX(NormalMap), t0.xy);
  vec3 nt = nmap.xyz * 2.0 - 1.0;
  vec3 n = vec3(dot(nt, t1.xyz), dot(nt, t2.xyz), dot(nt, t3.xyz));
  vec3 r = 2.0 * dot(n, e) * n - e * dot(n, n);
  float strength = mix(nmap.w, texture(FNV_TEX(CustomEnvMask), t0.xy).x, fnvC[27].w) * fnvC[27].z;
  vec3 env = texture(FNV_TEXCUBE(EnvironmentCubeMap), r).xyz * strength * fnvC[1].w;
#ifdef VC
  env *= v0.xyz;
#endif
  oC0 = vec4(env, 1.0);
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
