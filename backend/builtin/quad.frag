#version 450
// Fixed-function stage 0 as Bink's Setup_Renderstate sets it: COLOROP MODULATE(texture, diffuse = white),
// ALPHAOP SELECTARG1(texture).
layout(set = 1, binding = 16) uniform texture2D tex;
layout(set = 1, binding = 32) uniform sampler smp;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() { color = texture(sampler2D(tex, smp), uv); }
