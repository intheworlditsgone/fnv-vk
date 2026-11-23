#version 450
// Vulkan-side vertex shader for the pixel-shader differential test. Writes the interface the translated D3D9
// pixel shaders expect (docs/SHADER_ABI.md): TEXCOORD0-7 at locations 1-8, COLOR0/1 at 9/10, FOG at 11, and the
// lighting vertex shaders' view vector at 12 (a varying unit vector here, for replacements that read it) and their
// tangent-to-world frame at 13..15.
layout(location = 0) in vec4 inPos;
layout(location = 1) in vec4 inT[8];
layout(location = 9) in vec4 inC0;
layout(location = 10) in vec4 inC1;

layout(location = 1) out vec4 outT[8];
layout(location = 9) out vec4 outC0;
layout(location = 10) out vec4 outC1;
layout(location = 11) out float outFog;
layout(location = 12) out vec4 outView;
layout(location = 13) out vec4 outFrame[3];  // the tangent-to-world frame: a fixed rotation

void main() {
  gl_Position = inPos;
  for (int i = 0; i < 8; i++) outT[i] = inT[i];
  outC0 = inC0;
  outC1 = inC1;
  outFog = 1.0;
  outView = vec4(normalize(inT[2].yzx - vec3(0.25, -0.5, 0.3)), 0.0);
  outFrame[0] = vec4(0.0, 0.0, 1.0, 0.0);
  outFrame[1] = vec4(1.0, 0.0, 0.0, 0.0);
  outFrame[2] = vec4(0.0, 1.0, 0.0, 0.0);
}
