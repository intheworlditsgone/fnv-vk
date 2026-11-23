// The distance the indirect light is for: log2 of the view distance at the full-resolution pixel each half-resolution
// texel of gi was computed from (ssgi.frag, gitemporal.frag: fnvPixel()). The next frame's gitemporal.frag validates its
// history with it, and the lighting its reprojection (published as gidepth). Outside the camera viewport: 0.
layout(location = 0) out float outDepth;

void main() {
  const ivec2 pix = ivec2(fnvPixel());
  outDepth = fnvInsideRect(pix) ? log2(fnvLinearZ(fnvRawDepth(pix))) : 0.0;
}
