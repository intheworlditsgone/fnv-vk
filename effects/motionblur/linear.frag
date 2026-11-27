// Motion blur, pass 1 (full resolution): the scene in linear light (its colours are gamma 2.2 encoded), so the
// reconstruction can filter it bilinearly at fractional positions along the streaks (filtering the encoded colours
// would not be linear).
layout(location = 0) out vec4 outLinear;

void main() {
  const vec4 s = fnvSceneColor(ivec2(gl_FragCoord.xy));
  outLinear = vec4(pow(max(s.rgb, 0.0), vec3(2.2)), s.a);
}
