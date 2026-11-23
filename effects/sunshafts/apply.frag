// Sun shafts, pass 3: the light added to the scene in the sun's colour (effect.ini: Output=scene, Blend=add).
layout(location = 0) out vec4 color;

void main() {
  const float scatter = texelFetch(sampler2D(img_full, fnvPoint), ivec2(gl_FragCoord.xy), 0).r;
  if (scatter == 0.0) discard;  // adds nothing: no blend read and write of the (multisampled) target
  color = vec4(fnv.sunColor.rgb * (p_intensity * scatter), 0.0);
}
