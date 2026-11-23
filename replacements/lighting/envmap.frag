// lighting_2x_p_EnvMap_p.hlsl: the game's environment map reflections for objects, reconstructed from its compiled
// variants SLS2057..SLS2059 (tools/lightingtable.py; verified with tools/replacediff.py). Defines: WINDOW (the view
// vector reversed), EYE (eyes: the normal from t1, no normal map or mask, lit by PSLightColor[0]).
//
// The reflection of the view vector about the world space normal looks up EnvironmentCubeMap (s1). Without EYE: t1..t3
// are the rows taking tangent to world space, their w the view vector; the strength is the normal map's alpha, or the
// CustomEnvMask (s3) by EnvToggles.w, times EnvToggles.z. EYE: t1 the encoded normal (w: a factor), t2 the view vector.
// Constants: c1.w AmbientColor's alpha (a factor), c27 EnvToggles (x: vertex colour). The result times t4.x; alpha 1.
//
// Physically based mode (parameter pbr, with a probe; docs/INDIRECT.md "Reflections"): the reflected direction looks up
// the environment probe (the scene around the camera, rendered live) instead of the material's static cube map, with
// the material's own mask and strength. The probe is linear light; it is encoded back (gamma 2.2) as the game's cube
// maps are, since this pass adds into the target as the game's does. Eyes keep their cube map.

FNV_SAMPLERCUBE(1, EnvironmentCubeMap)
#ifndef EYE
FNV_SAMPLER2D(0, NormalMap)
FNV_SAMPLER2D(3, CustomEnvMask)
layout(location = 1) in vec4 t0;
layout(location = 4) centroid in vec4 t3;
#endif
layout(location = 2) centroid in vec4 t1;
layout(location = 3) centroid in vec4 t2;
layout(location = 5) in vec4 t4;
layout(location = 9) in vec4 v0;
layout(location = 0) out vec4 oC0;

#define EnvToggles fnvC[27]

vec3 nrm(vec3 v) {
  float r = inversesqrt(dot(v, v));
  return vec3(fnvMulLegacy(v.x, r), fnvMulLegacy(v.y, r), fnvMulLegacy(v.z, r));
}

// The view vector e reflected about n (n not assumed unit length, as the original).
vec3 reflectView(vec3 n, vec3 e) { return 2.0 * dot(n, e) * n - e * dot(n, n); }

void main() {
#ifdef FNV_SUN
  const float sunVisibility = fnvSunVisibility();
#else
  const float sunVisibility = 1.0;
#endif
#ifdef EYE
  vec3 n = nrm((t1.xyz - 0.5) * 2.0);
  vec3 env = texture(FNV_TEXCUBE(EnvironmentCubeMap), reflectView(n, nrm(t2.xyz))).xyz * fnvC[1].w;
#else
#ifdef WINDOW
  vec3 e = nrm(-vec3(t1.w, t2.w, t3.w));
#else
  vec3 e = nrm(vec3(t1.w, t2.w, t3.w));
#endif
  vec4 nmap = texture(FNV_TEX(NormalMap), t0.xy);
  vec3 nt = nrm((nmap.xyz - 0.5) * 2.0);
  vec3 n = nrm(vec3(dot(nt, t1.xyz), dot(nt, t2.xyz), dot(nt, t3.xyz)));
  float strength = mix(nmap.w, texture(FNV_TEX(CustomEnvMask), t0.xy).x, EnvToggles.w) * EnvToggles.z;
  vec3 dir = reflectView(n, e);
  vec3 cube = texture(FNV_TEXCUBE(EnvironmentCubeMap), dir).xyz;
#ifdef FNV_ENV
  if (p_pbr && fnvEnvValid()) cube = pow(fnvEnvRadiance(dir, 0.0), vec3(1.0 / 2.2));
#endif
  vec3 env = cube * strength * fnvC[1].w;
#endif
  env = -EnvToggles.x >= 0.0 ? env : env * v0.xyz;
#ifdef EYE
  env = env * (fnvC[3].xyz * sunVisibility) * t1.w;
#endif
  oC0 = vec4(env * t4.x, 1.0);
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
