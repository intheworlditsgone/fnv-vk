#version 450
// GGX prefiltering of the environment cube (backend_env.inl): one face of one level, in linear light. decode 1: the
// output is the source's level 0 decoded from gamma 2.2 (the linear cube). Otherwise the source is the linear cube (with
// its mips of linear averages) and each output texel is the
// split-sum radiance term for roughness alpha with N = V = R = the texel's direction:
//   sum L(l) N.l / sum N.l over l = reflect(-V, h), h GGX-distributed (Karis 2013),
// each sample read from the source mip whose texel solid angle matches the sample's (filtered importance sampling,
// GPU Gems 3 ch. 20). alpha 0: the source itself.
layout(set = 0, binding = 0) uniform textureCube srcCube;
layout(set = 0, binding = 1) uniform sampler srcSampler;
layout(push_constant) uniform P {
  uint face;
  float alpha;
  float srcSize;   // source level-0 edge
  float dstSize;   // this level's edge
  uint decode;     // 1: the source is gamma 2.2 encoded (read level 0 only)
} p;
layout(location = 0) out vec4 outColor;

const float kPi = 3.14159265;
const uint kSamples = 128u;

// D3D9 cube face direction of a texel centre (u, v in -1..1, v down): +X -X +Y -Y +Z -Z.
vec3 faceDir(uint f, vec2 uv) {
  float u = uv.x, v = uv.y;
  if (f == 0u) return vec3(1.0, -v, -u);
  if (f == 1u) return vec3(-1.0, -v, u);
  if (f == 2u) return vec3(u, 1.0, v);
  if (f == 3u) return vec3(u, -1.0, -v);
  if (f == 4u) return vec3(u, -v, 1.0);
  return vec3(-u, -v, -1.0);
}
vec3 decode(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); }
vec2 hammersley(uint i, uint n) {
  uint b = i;
  b = (b << 16u) | (b >> 16u);
  b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
  b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
  b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
  b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
  return vec2(float(i) / float(n), float(b) * 2.3283064365386963e-10);
}

void main() {
  vec2 uv = gl_FragCoord.xy / p.dstSize * 2.0 - 1.0;
  vec3 n = normalize(faceDir(p.face, uv));
  if (p.decode != 0u) {
    outColor = vec4(decode(textureLod(samplerCube(srcCube, srcSampler), n, 0.0).rgb), 1.0);
    return;
  }
  if (p.alpha <= 0.0) {
    outColor = vec4(textureLod(samplerCube(srcCube, srcSampler), n, 0.0).rgb, 1.0);
    return;
  }
  vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
  vec3 tx = normalize(cross(up, n)), ty = cross(n, tx);
  float a2 = p.alpha * p.alpha;
  float texelSolidAngle = 4.0 * kPi / (6.0 * p.srcSize * p.srcSize);
  float maxLod = log2(p.srcSize);
  vec3 sum = vec3(0.0);
  float weight = 0.0;
  for (uint i = 0u; i < kSamples; i++) {
    vec2 xi = hammersley(i, kSamples);
    float phi = 2.0 * kPi * xi.x;
    float cosT = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
    float sinT = sqrt(max(1.0 - cosT * cosT, 0.0));
    vec3 h = tx * (sinT * cos(phi)) + ty * (sinT * sin(phi)) + n * cosT;
    vec3 l = 2.0 * dot(n, h) * h - n;
    float nl = dot(n, l);
    if (nl <= 0.0) continue;
    float nh = max(dot(n, h), 0.0);
    float d = nh * nh * (a2 - 1.0) + 1.0;
    float D = a2 / (kPi * d * d);
    float pdf = D * 0.25;  // D N.h / (4 V.h) with N = V
    float sampleSolidAngle = 1.0 / (float(kSamples) * pdf + 1e-6);
    float lod = clamp(0.5 * log2(sampleSolidAngle / texelSolidAngle) + 1.0, 0.0, maxLod);
    sum += textureLod(samplerCube(srcCube, srcSampler), l, lod).rgb * nl;
    weight += nl;
  }
  outColor = vec4(sum / max(weight, 1e-6), 1.0);
}
