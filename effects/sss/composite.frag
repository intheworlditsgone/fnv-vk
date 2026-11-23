// Subsurface scattering, pass 4: the scattered skin (img_final) over the scene: alpha 1 on skin replaces it, alpha 0
// elsewhere (and outside the camera's viewport, which the image passes do not write) leaves it as it is.
layout(location = 0) out vec4 outColor;

void main() {
  const ivec2 p = ivec2(gl_FragCoord.xy);
  outColor = fnvInsideRect(p) ? texelFetch(sampler2D(img_final, fnvPoint), p, 0) : vec4(0.0);  // (written inside only)
  if (outColor.a == 0.0) discard;  // not skin: leaves the scene as it is, without the blend's read and write
}
