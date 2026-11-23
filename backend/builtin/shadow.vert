#version 450
// Sun shadow map caster (backend_shadow.inl): the game's vertex positions through the caster's object-to-world rows
// and the cascade's view-projection. Texture coordinates for alpha-tested casters.
layout(push_constant) uniform Shadow {
  layout(row_major) mat4 viewProj;
  vec4 world[3];
} pc;
layout(location = 0) in vec4 inPosition;  // missing components read as (0, 0, 0, 1)
#ifdef ALPHA
layout(location = 1) in vec4 inTexcoord;
layout(location = 0) out vec2 uv;
#endif
void main() {
  vec4 p = vec4(inPosition.xyz, 1.0);
  vec3 w = vec3(dot(pc.world[0], p), dot(pc.world[1], p), dot(pc.world[2], p));
  gl_Position = pc.viewProj * vec4(w, 1.0);
#ifdef ALPHA
  uv = inTexcoord.xy;
#endif
}
