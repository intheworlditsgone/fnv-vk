// Depth of field, pass 2 (full resolution): each pixel's signed blur radius (half the thin lens's circle of confusion)
// in full-resolution pixels, negative in front of the focus. The later passes read it instead of the depth (the scene
// pass cannot: the depth buffer is the game's attachment there).
layout(location = 0) out float outCoc;

const float kMmPerUnit = 14.2875;  // 64 units per yard

void main() {
  const ivec2 p = fnvClampRect(ivec2(gl_FragCoord.xy));
  const float focus = texelFetch(sampler2D(img_focus, fnvPoint), ivec2(0), 0).r;
  const float z = fnvLinearZ(fnvRawDepth(p));
  const float tanHalf = 0.5 * (fnv.frustum.y - fnv.frustum.x);
  const float f = p_focalLength > 0.0 ? p_focalLength : 18.0 / max(tanHalf, 1e-3);  // mm: a 36 mm sensor's width
  const float aperture = f / max(p_fNumber, 0.5);
  const float zf = max(focus * kMmPerUnit, 1.5 * f), zm = max(z * kMmPerUnit, 1e-3);
  const float c = aperture * f * (zm - zf) / (zm * (zf - f));  // mm on the sensor
  const float rMax = 0.01 * p_maxBlur * fnv.rect.w;
  const float r = clamp(0.5 * c / 36.0 * fnv.rect.z, -rMax, rMax);
  outCoc = p_nearBlur ? r : max(r, 0.0);
}
