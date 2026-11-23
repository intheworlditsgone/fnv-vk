// Keeps this frame's indirect light as the next frame's history (gitemporal.frag reads it; a pass cannot read the image
// it writes).
layout(location = 0) out vec4 outGi;

void main() { outGi = texelFetch(sampler2D(img_gi, fnvPoint), ivec2(gl_FragCoord.xy), 0); }
