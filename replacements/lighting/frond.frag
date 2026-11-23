// speedtree\frond.p.hlsl: the game's SpeedTree fronds (STFROND2nnn), reconstructed from their compiled variants
// (tools/ida_shadertables.py, tools/lightingtable.py; verified with tools/replacediff.py). Lit per vertex: the diffuse
// map times v0 plus v1 dimmed by SunlightDimmer (c0.x); FOG: fogged with t1 (rgb colour, w amount). Alpha: the map's.

FNV_SAMPLER2D(0, DiffuseMap)
layout(location = 1) in vec4 t0;
#ifdef FOG
layout(location = 2) in vec4 t1;
#endif
layout(location = 9) in vec4 v0;
layout(location = 10) in vec4 v1;
layout(location = 0) out vec4 oC0;

void main() {
  vec4 base = texture(FNV_TEX(DiffuseMap), t0.xy);
  vec3 color = base.xyz * (v1.xyz * fnvC[0].x + v0.xyz);
#ifdef FOG
  color = mix(color, t1.xyz, t1.w);
#endif
  oC0 = vec4(color, base.w);
  fnvAlphaTest(oC0.w);
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
