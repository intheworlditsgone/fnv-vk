// Realistic water: replaces the game's water pixel shaders (WATER000-034.pso, water.p.hlsl) with the same inputs.
//
// What the game provides (read from WATER000.pso, WaterShader::CreateConstantMaps / SetupGeometryConstants 0x00B864D0
// and the water depth pass SLS2074.pso):
//   TEXCOORD0  world position (xyz, the space of EyePos), w: world scale (object -> world units)
//   TEXCOORD1  object-space position
//   TEXCOORD2-5 projection rows (0.5 * MVP rows + 0.5 * row 3, row 3): projective screen coordinates
//   TEXCOORD6  xy: displacement-map coordinates (wading), w: alpha
//   TEXCOORD7  xy: noise (normal) map coordinates
//   c1 EyePos, c2 ShallowColor, c3 DeepColor, c4 ReflectionColor, c5 FresnelRI (x: Fresnel R0, y: fog power,
//   w: reflectivity), c6 BlendRadius (w: wading ripple strength), c8 VarAmounts (x: sun specular power, y: reflection
//   amount, z: opacity without depth, w: distortion), c9 FogParam (x: far, y: far - near, z/w: opacity by depth),
//   c10 FogColor (w: opacity), c11 DepthFalloff (x: start, y: end), c12 SunDir (w: specular magnitude), c13 SunColor.
//   s0 reflection (planar, mirrored), s1 refraction (the scene behind the water, fogged), s2 noise normal map
//   (animated by the game), s3 wading displacement, s4 water depth: saturate(depth of the ground below the water
//   plane / scale) in every channel, the units of DepthFalloff.
// Per-channel absorption follows Pope & Fry (1997), blended with the water type's tint.

FNV_TEXCOORD(0) vec4 vWorld;
FNV_TEXCOORD(1) vec4 vObject;
FNV_TEXCOORD(2) vec4 vProjX;
FNV_TEXCOORD(3) vec4 vProjY;
FNV_TEXCOORD(4) vec4 vProjZ;
FNV_TEXCOORD(5) vec4 vProjW;
FNV_TEXCOORD(6) vec4 vMisc;
FNV_TEXCOORD(7) vec4 vNoiseUv;
layout(location = 0) out vec4 outColor;

#ifdef REFLECTIONS
FNV_SAMPLER2D(0, ReflectionMap)
#endif
#ifdef REFRACTIONS
FNV_SAMPLER2D(1, RefractionMap)
#endif
#ifndef LOD
FNV_SAMPLER2D(2, NoiseMap)
#endif
#ifdef WADING
FNV_SAMPLER2D(3, DisplacementMap)
#endif
#ifdef DEPTH
FNV_SAMPLER2D(4, DepthMap)
#endif

#define EyePos fnvC[1]
#define ShallowColor fnvC[2]
#define DeepColor fnvC[3]
#define ReflectionColor fnvC[4]
#define FresnelRI fnvC[5]
#define BlendRadius fnvC[6]
#define VarAmounts fnvC[8]
#define FogParam fnvC[9]
#define FogColor fnvC[10]
#define DepthFalloff fnvC[11]
#define SunDir fnvC[12]
#define SunColor fnvC[13]

const float PI = 3.14159265;

vec4 project(vec4 p) { return vec4(dot(vProjX, p), dot(vProjY, p), dot(vProjZ, p), dot(vProjW, p)); }
vec2 screenUv(vec4 c) { return vec2(c.x, c.w - c.y) / c.w; }  // refraction and depth maps
vec2 mirrorUv(vec4 c) { return c.xy / c.w; }                  // the planar reflection (mirrored vertically)

// Unpolarised Fresnel reflectance of a dielectric with relative index eta, for the cosine of the incident angle.
float fresnel(float cosI, float eta) {
  float g2 = eta * eta - 1.0 + cosI * cosI;
  if (g2 <= 0.0) return 1.0;
  float g = sqrt(g2);
  float a = (g - cosI) / (g + cosI);
  float b = (cosI * (g + cosI) - 1.0) / (cosI * (g - cosI) + 1.0);
  return 0.5 * a * a * (1.0 + b * b);
}

float fogAmount(float d) {
  float vis = clamp((FogParam.x - d) / max(FogParam.y, 1e-3), 0.0, 1.0);
  return pow(1.0 - vis, max(FresnelRI.y, 1e-3));
}

void main() {
  vec3 toEye = EyePos.xyz - vWorld.xyz;
  float dist = max(length(toEye), 1e-3);
  vec3 V = toEye / dist;
  float hdist = length(toEye.xy);
  float fade = clamp(2.0 - hdist / 4096.0, 0.0, 1.0);  // the game's: full detail to 4096 units, none past 8192
  vec4 base = vec4(vObject.xyz, 1.0);

  // Water depth (0 at the shore, 1 at the depth map's scale) and how far it is into DepthFalloff.
#ifdef DEPTH
  float rawDepth = texture(FNV_TEX(DepthMap), screenUv(project(base))).y;
  float depth = mix(1.0, rawDepth, fade);  // toward deep with distance, as the game does
#else
  float rawDepth = 0.5 * (DepthFalloff.x + DepthFalloff.y);
  float depth = rawDepth;
#endif
  float falloff = clamp((rawDepth - DepthFalloff.x) / max(DepthFalloff.y - DepthFalloff.x, 1e-3), 0.0, 1.0);

  // Surface normal (z up).
  vec3 N = vec3(0.0, 0.0, 1.0);
  float variance = 0.0;  // slope variance lost to fading, added to the roughness
#ifndef LOD
  vec2 uv = vNoiseUv.xy;
  const mat2 rotA = mat2(0.8, 0.6, -0.6, 0.8), rotB = mat2(0.6, -0.8, 0.8, 0.6);
  vec2 n0 = texture(FNV_TEX(NoiseMap), uv).xy * 2.0 - 1.0;
  vec2 n1 = texture(FNV_TEX(NoiseMap), rotA * uv * 0.37 + vec2(0.31, 0.17)).xy * 2.0 - 1.0;
  vec2 n2 = texture(FNV_TEX(NoiseMap), rotB * uv * 2.3 + vec2(0.71, 0.43)).xy * 2.0 - 1.0;
  float microFade = clamp(1.0 - dist / 1500.0, 0.0, 1.0);
  vec2 slope = n0 + p_detail * (0.55 * n1 + 0.35 * microFade * n2);
  float calm = mix(0.3, 1.0, falloff);  // smaller waves in the shallows
  N = normalize(vec3(slope * p_wave * calm * fade, 1.0));
  variance = 0.05 * p_wave * (1.0 - fade) + 0.01 * p_detail * (1.0 - microFade);
#endif
#ifdef WADING
  vec2 ripple = (texture(FNV_TEX(DisplacementMap), vMisc.xy).zw - 0.5) * BlendRadius.w;
  vec3 rn = normalize(vec3(ripple, 0.04));
  N = normalize(mix(rn, N, clamp(rn.z, 0.0, 1.0)));
#endif

  float cosV = max(dot(N, V), 1e-3);
  float s = sqrt(clamp(FresnelRI.x, 0.005, 0.25));
  float eta = (1.0 + s) / (1.0 - s);  // the index of refraction whose normal-incidence reflectance is R0
  float F = fresnel(cosV, eta);

  float distortion = mix(4.0, VarAmounts.w, clamp(hdist * 0.0002, 0.0, 1.0));  // world units, as the game
  vec3 reflection = ReflectionColor.rgb;
#ifdef REFLECTIONS
  vec4 reflPos = vec4(vObject.xy + N.xy * distortion * p_reflDistortion / vWorld.w, vObject.z, 1.0);
  reflection = mix(ReflectionColor.rgb, texture(FNV_TEX(ReflectionMap), mirrorUv(project(reflPos))).rgb, VarAmounts.y);
#endif
  reflection *= FresnelRI.w;

  // What is behind the surface, without the fog of its distance (the refraction texture is the fogged scene).
#ifdef REFRACTIONS
  vec4 refrPos = vec4(vObject.xy + N.xy * distortion * falloff * p_refrDistortion / vWorld.w, vObject.z, 1.0);
  vec3 behind = texture(FNV_TEX(RefractionMap), screenUv(project(refrPos))).rgb;
  float fogBehind = fogAmount(dist + depth);
  behind = max((behind - fogBehind * FogColor.rgb) / (1.0 - fogBehind + 1e-4), 0.0);
#else
  vec3 behind = ShallowColor.rgb;
#endif

  float cosT = sqrt(max(1.0 - (1.0 - cosV * cosV) / (eta * eta), 0.0));
  float path = depth / max(cosT, 0.2);
  vec3 tint = -log(clamp(ShallowColor.rgb, 0.02, 0.98));
  tint /= max(dot(tint, vec3(1.0 / 3.0)), 1e-3);
  const vec3 pureWater = vec3(2.51, 0.42, 0.07);  // Pope & Fry absorption at 650 / 550 / 450 nm, mean 1
  vec3 absorption = mix(pureWater, tint, p_tint) * (3.0 * p_density / max(DepthFalloff.y, 0.05));
  vec3 T = exp(-absorption * path);
#ifdef DEPTH
  float authored = falloff * (1.0 - clamp(FogParam.z * (1.0 - rawDepth) / max(FogParam.w, 1e-3), 0.0, 1.0)) * FogColor.w;
#else
  float authored = clamp(VarAmounts.z, 0.0, 1.0);
#endif
  T *= 1.0 - p_turbidity * authored;  // the water type's own opacity, as extra turbidity

#ifdef SUN
  vec3 L = normalize(SunDir.xyz);
  float light = mix(0.35, 1.0, clamp(L.z, 0.0, 1.0));
#else
  float light = 1.0;
#endif
  vec3 scattered = mix(ShallowColor.rgb, DeepColor.rgb, falloff) * light * p_scatter;
  vec3 body = behind * T + scattered * (1.0 - T);
  vec3 color = mix(body, reflection, F);

#ifdef SUN
  float alpha = clamp(sqrt(2.0 / (max(VarAmounts.x, 0.0) + 2.0)) * p_roughness, 0.02, 1.0);
  float a2 = min(alpha * alpha + variance, 1.0);
  float NdotL = dot(N, L);
  if (NdotL > 0.0) {
    vec3 H = normalize(V + L);
    float NdotH = max(dot(N, H), 0.0), VdotH = max(dot(V, H), 0.0);
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / (PI * d * d);
    float vis = 0.5 / max(cosV * sqrt(a2 + (1.0 - a2) * NdotL * NdotL) + NdotL * sqrt(a2 + (1.0 - a2) * cosV * cosV), 1e-5);
    float spec = D * vis * fresnel(VdotH, eta) * NdotL * PI;
    color += min(spec, 64.0) * SunColor.rgb * SunDir.w * p_sun;
  }
#endif

  color = mix(color, FogColor.rgb, fogAmount(dist));
  fnvAlphaTest(vMisc.w);
  outColor = vec4(color, vMisc.w);
}
