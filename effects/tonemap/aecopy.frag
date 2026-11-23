// Auto-exposure, pass 3: keeps this frame's adapted value as the next frame's (aeadapt.frag reads it; a pass cannot
// read the image it writes).
layout(location = 0) out float outPrev;

void main() { outPrev = texelFetch(sampler2D(img_aeadapt, fnvPoint), ivec2(0), 0).r; }
