#version 450
// Alpha-tested sun shadow casters (foliage, fences): no shadow where the diffuse texture is transparent.
layout(set = 0, binding = 0) uniform texture2D diffuse;
layout(set = 0, binding = 1) uniform sampler smp;
layout(location = 0) in vec2 uv;
void main() {
  if (texture(sampler2D(diffuse, smp), uv).a < 0.5) discard;
}
