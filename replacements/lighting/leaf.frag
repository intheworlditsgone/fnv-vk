// speedtree\leaf.p.hlsl: the game's SpeedTree leaves (STLEAF2nnn, each also TMS), reconstructed from their compiled
// variants (tools/ida_shadertables.py, tools/lightingtable.py; verified with tools/replacediff.py). Lit per vertex: the
// diffuse map times t1; FOG: fogged with t2 (rgb colour, w amount); TRANSPARENCYMS: alpha doubled (alpha to coverage).

FNV_SAMPLER2D(0, DiffuseMap)
layout(location = 1) in vec4 t0;
layout(location = 2) in vec4 t1;
#ifdef FOG
layout(location = 3) in vec4 t2;
#endif
layout(location = 0) out vec4 oC0;

void main() {
  vec4 base = texture(FNV_TEX(DiffuseMap), t0.xy);
  vec3 color = base.xyz * t1.xyz;
#ifdef FOG
  color = mix(color, t2.xyz, t2.w);
#endif
#ifdef TRANSPARENCYMS
  oC0 = vec4(color, base.w * 2.0);
#else
  oC0 = vec4(color, base.w);
#endif
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
