// Depth of field, pass 3 (half resolution): colour in linear light and signed blur radius per 2 x 2 pixels.
// The radius is the nearest pixel's (the smallest signed radius: it grows with distance), so an in-focus edge in front
// of a blurred background stays in focus and the background's blur does not take it in; the colour is the mean of the
// pixels, each weighted by how close its own radius is to that one (less of the background's colour mixed into the
// edge).
layout(location = 0) out vec4 outHalf;

vec3 lin(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }

void main() {
  const ivec2 h = ivec2(gl_FragCoord.xy);
  vec3 c[4];
  float r[4];
  float rs = 1e9;
  for (int i = 0; i < 4; i++) {
    const ivec2 p = fnvClampRect(2 * h + ivec2(i & 1, i >> 1));
    c[i] = lin(fnvSceneColor(p).rgb);
    r[i] = texelFetch(sampler2D(img_coc, fnvPoint), p, 0).r;
    rs = min(rs, r[i]);
  }
  vec3 sum = vec3(0.0);
  float weights = 0.0;
  for (int i = 0; i < 4; i++) {
    const float w = 1.0 / (1.0 + abs(r[i] - rs));
    sum += c[i] * w;
    weights += w;
  }
  outHalf = vec4(sum / weights, rs);
}
