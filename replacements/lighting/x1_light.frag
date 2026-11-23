// The game's 1.x per-light passes, one file for five sources with the same inputs, reconstructed from their compiled
// variants (tools/lightingtable.py; verified with tools/replacediff.py). The define names the source:
//   DIFFUSE_PT  lighting_1x_p_diffusePt_p.hlsl (SLS1001): a point light's diffuse.
//   DIFFUSE_DIR lighting_1x_p_diffuseDir_p.hlsl (SLS1002): a directional light's diffuse.
//   SPECULAR_PT lighting_1x_p_specularPt_p.hlsl (SLS1007): a point light's specular (power 4).
//   SPECULAR_DIR lighting_1x_p_specularDir_p.hlsl (SLS1008): a directional light's specular (power 4 and 8).
//   AMB_DIFF_DIR_AND_PT lighting_1x_p_ambDiffDirAndPt_p.hlsl (SLS1029): ambient, the sun (direction v0) and a point
//                         light's diffuse in one.
// (SPECMULT=toneA, the specular variants' define, is the same in both.) N: the normal map at t0 (alpha: gloss); a point
// light's direction is the normalisation cube map (s3) at t3, its attenuation AttMapXY (t1) times AttMapZ (t2); a
// directional light's direction the encoded t1. None is normalised beyond the cube map. PSLightColor[0] is c3.

FNV_SAMPLER2D(0, NormalMap)
#if defined(DIFFUSE_PT) || defined(SPECULAR_PT) || defined(AMB_DIFF_DIR_AND_PT)
#define POINT_LIGHT 1
FNV_SAMPLER2D(1, AttMapXY)
FNV_SAMPLER2D(2, AttMapZ)
FNV_SAMPLERCUBE(3, NormalCubeMap)
layout(location = 3) in vec4 t2;
layout(location = 4) in vec4 t3;
#endif
layout(location = 1) in vec4 t0;
layout(location = 2) in vec4 t1;
#ifdef AMB_DIFF_DIR_AND_PT
layout(location = 9) in vec4 v0;
#endif
layout(location = 0) out vec4 oC0;

void main() {
#ifdef FNV_SUN
  const float sunVisibility = fnvSunVisibility();
#else
  const float sunVisibility = 1.0;
#endif
  vec4 c3 = vec4(fnvC[3].xyz * sunVisibility, fnvC[3].w);
  vec4 nmap = texture(FNV_TEX(NormalMap), t0.xy);
  vec3 n = (nmap.xyz - 0.5) * 2.0;
#ifdef POINT_LIGHT
  vec3 l = (texture(FNV_TEXCUBE(NormalCubeMap), t3.xyz).xyz - 0.5) * 2.0;
  vec3 att = texture(FNV_TEX(AttMapZ), t2.xy).xyz * texture(FNV_TEX(AttMapXY), t1.xy).xyz;
#else
  vec3 l = (t1.xyz - 0.5) * 2.0;
#endif
  float d = clamp(dot(n, l), 0.0, 1.0);
#if defined(DIFFUSE_PT)
  oC0 = vec4(d * c3.xyz * att, c3.w);
#elif defined(DIFFUSE_DIR)
  oC0 = vec4(d * c3.xyz, c3.w);
#elif defined(SPECULAR_PT)
  float s = d * d;
  s = s * s;
  vec3 spec = nmap.w * (s * c3.xyz);
  oC0 = vec4(spec * att, nmap.w * spec.x);
#elif defined(SPECULAR_DIR)
  float s = d * d;
  s = s * s;
  oC0 = vec4(vec3(nmap.w * (s * s * c3.x)), nmap.w * s);
#else
  float sun = clamp(dot(n, (v0.xyz - 0.5) * 2.0), 0.0, 1.0);
  oC0 = vec4(clamp(att * d * fnvC[4].xyz + (sun * c3.xyz + fnvC[1].xyz), 0.0, 1.0), fnvC[1].w);
#endif
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
