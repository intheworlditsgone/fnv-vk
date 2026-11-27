// Depth of field, pass 1: keeps this run's focus distance for the next run (focus.frag reads it; a pass cannot read the
// image it writes).
layout(location = 0) out float outPrev;

void main() { outPrev = texelFetch(sampler2D(img_focus, fnvPoint), ivec2(0), 0).r; }
