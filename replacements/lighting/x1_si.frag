// lighting_1x_p_SI_p.hlsl: the game's 1.x glow passes, reconstructed from its compiled variants SLS1036..SLS1038
// (tools/lightingtable.py; verified with tools/replacediff.py). Defines: AMBIENT (AmbientColor plus the glow map at t1
// times EmittanceColor, alpha the diffuse map's), without it (the glow map, at s0, as alpha: red times
// EmittanceColor.x, colour black), EMMITMULT (the value toneB in every variant), SIBLOCK (black, alpha 0).

#ifdef AMBIENT
FNV_SAMPLER2D(0, DiffuseMap)
FNV_SAMPLER2D(1, GlowMap)
layout(location = 1) in vec4 t0;
#elif !defined(SIBLOCK)
FNV_SAMPLER2D(0, GlowMap)
#endif
#ifndef SIBLOCK
layout(location = 2) in vec4 t1;
#endif
layout(location = 0) out vec4 oC0;

void main() {
#if defined(SIBLOCK)
  oC0 = vec4(0.0);
#elif defined(AMBIENT)
  vec3 glow = texture(FNV_TEX(GlowMap), t1.xy).xyz;
  oC0 = vec4(fnvC[2].xyz * glow + fnvC[1].xyz, texture(FNV_TEX(DiffuseMap), t0.xy).w);
#else
  oC0 = vec4(0.0, 0.0, 0.0, texture(FNV_TEX(GlowMap), t1.xy).x * fnvC[2].x);
#endif
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
