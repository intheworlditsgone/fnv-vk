// Depth of field, pass 8 (the scene): the sharp image, the far layer where the pixel lies behind the focus (blended in
// over the first 0.5 .. 2 pixels of blur radius), and the near layer over both by its coverage (premultiplied). The
// colours are gamma encoded (the game's content is made for a gamma 2.2 display); the layers are in linear light.
layout(location = 0) out vec4 color;

void main() {
  const ivec2 p = ivec2(gl_FragCoord.xy);
  const vec4 s = fnvSceneColor(p);
  if (!fnvInsideRect(p)) {
    color = s;
    return;
  }
  const float r = texelFetch(sampler2D(img_coc, fnvPoint), p, 0).r;
  const vec2 uv = (vec2(p) + 0.5) / (2.0 * vec2(textureSize(sampler2D(img_half, fnvPoint), 0)));
  vec3 c = pow(max(s.rgb, 0.0), vec3(2.2));
  if (r > 0.5) c = mix(c, texture(sampler2D(img_farblur, fnvLinear), uv).rgb, smoothstep(0.5, 2.0, r));
  const vec4 near = texture(sampler2D(img_nearblur, fnvLinear), uv);
  c = c * (1.0 - near.a) + near.rgb;
  color = vec4(pow(c, vec3(1.0 / 2.2)), s.a);
}
