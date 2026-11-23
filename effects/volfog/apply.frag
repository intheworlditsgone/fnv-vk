// Volumetric fog, pass 2: the scene seen through the fog, in linear light (the game's colours, and its light colours,
// are gamma 2.2 encoded): scene * T + sun's in-scattered light * sunColor + sky light * (1 - T). The sky light is the
// game's ambient light colour, which is the radiance of a uniform sky (a white surface under it shows that colour);
// a medium of albedo 1 scatters (1 - T) of it toward the camera. Every sample gets the pixel's value: with MSAA the
// scene colour is the resolved one, so the later resolve gives the same image.
layout(location = 0) out vec4 color;

void main() {
  const ivec2 pix = ivec2(gl_FragCoord.xy);
  const vec4 s = fnvSceneColor(pix);
  const vec2 fog = texelFetch(sampler2D(img_full, fnvPoint), pix, 0).rg;
  if (fog.y >= 1.0 && fog.x <= 0.0) { color = s; return; }  // no fog here (indoors, under water, off)
  const vec3 sun = pow(max(fnv.sunColor.rgb, 0.0), vec3(2.2));
  const vec3 sky = pow(max(fnv.ambientColor.rgb, 0.0), vec3(2.2));
  const vec3 lit = pow(max(s.rgb, 0.0), vec3(2.2)) * fog.y + sun * fog.x + sky * (p_ambient * (1.0 - fog.y));
  color = vec4(pow(lit, vec3(1.0 / 2.2)), s.a);
}
