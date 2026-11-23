// lighting_1x_p_texEffect_p.hlsl: the game's fill and rim texture effect (SLS2089, HQ), reconstructed from its compiled
// variant (tools/lightingtable.py; verified with tools/replacediff.py). The source texture (s1, t1) plus FillColor (c0),
// scaled by its alpha times FillColor.w, plus RimColor (c1) by the rim term saturate(1 - N.t3)^fVars.x (N the normal map
// at t0, not normalised); fogged with v1. Alpha: the source's times FillColor.w plus RimColor.w by the rim term.

FNV_SAMPLER2D(0, NormalMap)
FNV_SAMPLER2D(1, SourceTexture)
layout(location = 1) in vec4 t0;
layout(location = 2) in vec4 t1;
layout(location = 4) centroid in vec4 t3;
layout(location = 10) in vec4 v1;
layout(location = 0) out vec4 oC0;

void main() {
  vec4 src = texture(FNV_TEX(SourceTexture), t1.xy);
  vec3 n = (texture(FNV_TEX(NormalMap), t0.xy).xyz - 0.5) * 2.0;
  float rim = exp2(fnvMulLegacy(fnvC[2].x, log2(abs(clamp(1.0 - dot(n, t3.xyz), 0.0, 1.0)))));
  float a = src.w * fnvC[0].w;
  vec4 c = rim * fnvC[1] + vec4((src.xyz + fnvC[0].xyz) * a, a);
  oC0 = vec4(mix(c.xyz, v1.xyz, v1.w), c.w);
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
