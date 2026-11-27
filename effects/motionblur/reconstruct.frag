// Motion blur, pass 4 (the scene): McGuire et al.'s reconstruction (I3D 2012, "Reconstruct"). Samples along the
// neighbourhood's dominant half-vector vN, each weighted by how much it can contribute to this pixel X:
//   a sample Y in front of X blurs over X if its own streak reaches it (cone of |V(Y)|),
//   a sample behind X is seen through X's own blur (cone of |V(X)|),
//   and both being blurred and reaching each other (the cylinders) covers the case of the same moving surface.
// X itself weighs 1 / |V(X)|. Depths compare softly (1% of the distance). The colours are gamma encoded (the game's
// content is made for a gamma 2.2 display); they are averaged in linear light (img_lin, linear.frag) and encoded again.
layout(location = 0) out vec4 color;

// Y in front of X (1), behind (0), within the soft range between.
float front(float zY, float zX) { return clamp(1.0 - (zY - zX) / max(1.0, 0.01 * min(zX, zY)), 0.0, 1.0); }
float cone(float d, float v) { return clamp(1.0 - d / max(v, 1e-3), 0.0, 1.0); }
float cylinder(float d, float v) { return 1.0 - smoothstep(0.95 * v, 1.05 * v, d); }

void main() {
  const ivec2 p = ivec2(gl_FragCoord.xy);
  const vec4 sx = fnvSceneColor(p);
  const ivec2 full = textureSize(sampler2D(img_vel, fnvPoint), 0);
  const ivec2 tiles = textureSize(sampler2D(img_neighbor, fnvPoint), 0);
  const vec2 vN = texelFetch(sampler2D(img_neighbor, fnvPoint), min((p * tiles) / full, tiles - 1), 0).xy;
  const float lenN = length(vN);
  if (lenN < 0.5 || !fnvInsideRect(p)) {
    color = sx;
    return;
  }
  const vec4 vx = texelFetch(sampler2D(img_vel, fnvPoint), p, 0);
  const float lenX = length(vx.xy), zX = vx.z;
  const float wX = 1.0 / max(lenX, 0.5);
  vec3 sum = texelFetch(sampler2D(img_lin, fnvPoint), p, 0).rgb * wX;
  float weights = wX;
  // Interleaved gradient noise (Jimenez 2014): offsets the samples by up to half a step per pixel.
  const float j = fract(52.9829189 * fract(dot(vec2(p), vec2(0.06711056, 0.00583715)))) - 0.5;
  const int n = clamp(p_samples, 2, 64);
  for (int i = 0; i < n; i++) {
    const float t = mix(-1.0, 1.0, (float(i) + j + 1.0) / float(n + 1));
    // The colour bilinear at the exact position along the streak; velocity and depth of the pixel there.
    const vec2 yf = clamp(vec2(p) + 0.5 + t * vN, fnv.rect.xy + 0.5, fnv.rect.xy + fnv.rect.zw - 0.5);
    const ivec2 y = ivec2(floor(yf));
    const float d = length(yf - (vec2(p) + 0.5));
    if (d < 0.5) continue;
    const vec4 vy = texelFetch(sampler2D(img_vel, fnvPoint), y, 0);
    const float lenY = length(vy.xy), zY = vy.z;
    const float f = front(zY, zX), b = front(zX, zY);
    const float a = f * cone(d, lenY) + b * cone(d, lenX) + cylinder(d, lenY) * cylinder(d, lenX) * 2.0;
    sum += texture(sampler2D(img_lin, fnvLinear), yf * fnv.target.zw).rgb * a;
    weights += a;
  }
  color = vec4(pow(sum / weights, vec3(1.0 / 2.2)), sx.a);
}
