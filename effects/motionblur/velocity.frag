// Motion blur, pass 0 (full resolution): each pixel's blur half-vector (hv) and view distance.
//
// The surface under the pixel (its depth with this run's camera) is projected with the previous run's camera
// (fnvPrevPixel); the pixel moved by `pixel - prev` over one frame. The shutter is open for p_shutter of the frame
// time, centred on this frame's instant, so the surface's streak spans `motion * shutter`: half of it each way
// (McGuire et al.: V is the half-vector). Its length is limited to the longest blur. A position behind the previous
// camera (no projection), no previous run, or a camera that moved farther than p_cutDistance (a cut): no blur.
layout(location = 0) out vec4 outVel;

void main() {
  const ivec2 p = ivec2(gl_FragCoord.xy);
  const float d = fnvRawDepth(fnvClampRect(p));
  const float z = fnvLinearZ(d);
  vec2 hv = vec2(0.0);
  if (fnvInsideRect(p) && fnv.prevInfo.x != 0.0) {
    const vec3 prevCamera = vec3(fnv.prevCameraRows[0].w, fnv.prevCameraRows[1].w, fnv.prevCameraRows[2].w);
    if (distance(fnvCameraWorldPos(), prevCamera) <= p_cutDistance) {
      const vec2 pixel = vec2(p) + 0.5;
      vec2 prev;
      float prevZ;
      fnvPrevPixel(fnvViewPos(pixel, z), prev, prevZ);  // (false also when it was outside the previous view: still used)
      if (prevZ > 0.0) {
        hv = (pixel - prev) * (0.5 * clamp(p_shutter, 0.0, 1.0));
        const float maxLen = 0.01 * p_maxBlur * fnv.rect.w;
        const float len = length(hv);
        if (len > maxLen) hv *= maxLen / len;
        if (!(len < 1e9)) hv = vec2(0.0);  // not a number (degenerate depth)
      }
    }
  }
  outVel = vec4(hv, z, 0.0);
}
