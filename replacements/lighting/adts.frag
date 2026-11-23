// lighting_2x_p_ADTS_p.hlsl: the game's per-pixel object lighting (ambient, diffuse, texture, specular), the sun and up
// to one point light, reconstructed from its compiled variants SLS2000..SLS2027 (permutations: tools/lightingtable.py;
// verified against them on D3D9 with tools/replacediff.py). Defines: LIGHTS (2: a point light), SPECULAR, SI (glow
// map), HAIR, PROJ_SHADOW (the game's actor shadows), STBB (SpeedTree branches), OPT (no alpha cut-off or toggles), LOD,
// SILHOUETTE (with LIGHTS=0: the ambient colour fogged, with the base map's alpha), PARALLAX (ParallaxShader, PAR2nnn:
// the colour and normal maps at texture coordinates shifted along the view vector t6 by HeightMap, s3), FACEGENBLEND
// (SkinShader, SKIN2nnn: skin lighting with the view vector t6, and the face tint FaceGenMap0/1 at s2, s3).
//
// Physically based mode (parameter pbr; off: exactly the original), the same in every lighting shader
// (docs/LIGHTING.md): the lights and ambient summed in linear space (2.2), so the diffuse is
// albedo * gamma(sum lin(colour) N.L attenuation shadow + lin(ambient)); and only for a material with the game's
// specular flag (the SPECULAR variants) GGX specular, its roughness from the game's Blinn-Phong power
// (alpha = sqrt(2 / (power + 2))), scaled by the gloss map, the parameter specular and the game's specular LOD fade
// (LightData[0].w, t1.w here, GetSpecularLODFade 0x00B66B80): at the end of the fade it is the non-specular variants'
// diffuse, where the game switches to them. The view vector: the lighting vertex shaders' (location 12). Hair and skin
// keep the game's shading (their own models).
//
// Constants (the game's fixed registers): c1 AmbientColor (w: material alpha), c2 EmittanceColor (HAIR: hair tint),
// c3.. PSLightColor[] (sun first), c27 Toggles (x: vertex colour, y: fog, z: specular power, w: alpha cut-off).
// Inputs: t0 uv; t1 the sun's direction in tangent space (w: specular fade); v0 vertex colour; v1 fog (rgb, amount);
// LIGHTS=2: t2 the point light's direction, attenuation coordinates in t4 (t5 with SPECULAR); SPECULAR: t3 the sun's
// half vector (t4 the point light's); PROJ_SHADOW: t7 shadow coordinates.

#ifndef LIGHTS
#define LIGHTS 1
#endif
#if defined(OPT) || defined(LOD)
#define FIXED_PATH 1  // no alpha cut-off, vertex colour and fog always
#endif

FNV_SAMPLER2D(0, BaseMap)
#ifndef SILHOUETTE
FNV_SAMPLER2D(1, NormalMap)
#endif
#if defined(SI) || defined(HAIR) || (defined(FACEGENBLEND) && LIGHTS >= 2)
FNV_SAMPLER2D(4, GlowMap)
#endif
#ifdef PARALLAX
FNV_SAMPLER2D(3, HeightMap)
#endif
#ifdef FACEGENBLEND
FNV_SAMPLER2D(2, FaceGenMap0)
FNV_SAMPLER2D(3, FaceGenMap1)
#endif
#if defined(PARALLAX) || defined(FACEGENBLEND)
layout(location = 7) centroid in vec4 t6;  // the view vector (tangent space)
#define VIEW t6
#endif
#if LIGHTS >= 2
FNV_SAMPLER2D(5, AttenuationMap)
#endif
#ifdef PROJ_SHADOW
FNV_SAMPLER2D(6, ShadowMap)
FNV_SAMPLER2D(7, ShadowMaskMap)
#endif

layout(location = 1) in vec4 t0;
#ifndef SILHOUETTE
layout(location = 2) centroid in vec4 t1;
#endif
#if LIGHTS >= 2
layout(location = 3) centroid in vec4 t2;
#endif
#ifdef SPECULAR
layout(location = 4) centroid in vec4 t3;
#endif
#if LIGHTS >= 2 && defined(SPECULAR)
layout(location = 5) centroid in vec4 t4;
layout(location = 6) in vec4 t5;
#define ATTEN_COORDS t5
#elif LIGHTS >= 2
layout(location = 5) in vec4 t4;
#define ATTEN_COORDS t4
#endif
#ifdef PROJ_SHADOW
layout(location = 8) in vec4 t7;
#endif
#ifndef SILHOUETTE
layout(location = 9) in vec4 v0;
#endif
layout(location = 10) in vec4 v1;
layout(location = 0) out vec4 oC0;

#if !defined(HAIR) && !defined(SILHOUETTE)
#define PBR 1
layout(location = 12) in vec4 fnvView;  // the direction to the camera, tangent space; 0: none
#endif

#define PSLightColor_(i) (fnvC[3 + (i)].xyz * ((i) == 0 ? sunVisibility : 1.0))  // light 0: the sun, shadowed
#define AmbientColor fnvC[1]
#define EmittanceColor fnvC[2]
#define Toggles fnvC[27]

// D3D9 nrm: v * rsq(dot(v, v)) with the legacy multiply (a zero vector stays zero).
vec3 nrm(vec3 v) {
  float r = inversesqrt(dot(v, v));
  return vec3(fnvMulLegacy(v.x, r), fnvMulLegacy(v.y, r), fnvMulLegacy(v.z, r));
}
// D3D9 pow: exp2(y * log2(|x|)), the legacy multiply making pow(0, 0) = 1.
float powLegacy(float x, float y) { return exp2(fnvMulLegacy(y, log2(abs(x)))); }

#ifdef PARALLAX
// The parallax texture coordinates: t0 shifted along the normalised view vector by the height (0.04 h - 0.02).
vec2 parallaxUv() {
  float h = texture(FNV_TEX(HeightMap), t0.xy).x * 0.04 - 0.02;
  return t6.xy * inversesqrt(dot(t6.xyz, t6.xyz)) * h + t0.xy;
}
#endif
#ifdef FACEGENBLEND
// Skin: the back light through the edges, (1 - N.E)^2 sat(E.-L).
float skinBack(vec3 n, vec3 l, vec3 e) {
  float f = 1.0 - clamp(dot(n, e), 0.0, 1.0);
  return clamp(dot(e, -l), 0.0, 1.0) * (f * f);
}
// Skin: a point light (unit direction l), its diffuse, back light and the wrap of light past the terminator tinted.
vec3 skinPoint(vec3 n, vec3 l, vec3 e, vec3 c, vec3 tint) {
  float d = dot(n, l);
  float ds = clamp(d, 0.0, 1.0);
  float w = clamp((d + 0.3) * 0.769230783, 0.0, 1.0);
  float wrap = clamp((w * -2.0 + 3.0) * (w * w) - (ds * -2.0 + 3.0) * (ds * ds), 0.0, 1.0);
  return c * ds + skinBack(n, l, e) * mix(c, tint, 0.5) + wrap * tint;
}
#endif

#ifdef PBR
const float kPi = 3.14159265;
vec3 toLinear(vec3 c) { return pow(max(c, 0.0), vec3(2.2)); }
vec3 toGamma(vec3 c) { return pow(max(c, 0.0), vec3(1.0 / 2.2)); }
#ifdef FNV_FRAME
vec3 fnvWorldNormal(vec3 n) { return fnvToWorld(n); }
#else
vec3 fnvWorldNormal(vec3 n) { return vec3(0.0); }
#endif
// Normalize probe irradiance by its mean to preserve the game's ambient intensity.
#ifdef FNV_ENV
vec3 pbrAmbient(vec3 ambient, vec3 nWorld) {
  vec3 a = toLinear(ambient);
  if (!fnvEnvValid() || dot(nWorld, nWorld) == 0.0) return a;
  return a * clamp(dot(fnvEnvIrradiance(nWorld), vec3(0.2126, 0.7152, 0.0722)), 0.0, 4.0);
}
#else
vec3 pbrAmbient(vec3 ambient, vec3 nWorld) { return toLinear(ambient); }
#endif
// SSGI alpha is the fraction of rays that missed and still need probe lighting.
vec3 pbrIndirect(vec3 ambient, vec3 nWorld, float ao) {
  vec3 probe = pbrAmbient(ambient, nWorld) * ao;
  vec4 gi;
  return fnvScreenIndirect(gi) ? gi.rgb + gi.a * probe : probe;
}
// The specular of one light (unit direction l, linear colour c) on a dielectric (reflectance 0.04) of GGX roughness
// alpha: Smith height-correlated visibility, Schlick Fresnel. The game's light colour is the diffuse light on a white
// surface facing it, so the irradiance is pi times it.
vec3 pbrSpecular(vec3 n, vec3 v, vec3 l, vec3 c, float alpha) {
  float nl = clamp(dot(n, l), 0.0, 1.0);
  if (nl <= 0.0) return vec3(0.0);
  vec3 h = normalize(l + v);
  float nv = max(dot(n, v), 1e-4), nh = clamp(dot(n, h), 0.0, 1.0), vh = clamp(dot(v, h), 0.0, 1.0);
  float a2 = alpha * alpha;
  float f = 0.04 + 0.96 * pow(1.0 - vh, 5.0);
  float d = nh * nh * (a2 - 1.0) + 1.0;
  float D = a2 / (kPi * d * d);
  float vis = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2));
  return c * (nl * kPi * D * vis * f);
}
// A specular material's roughness from the game's Blinn-Phong power, and its specular's scale: the gloss map (the
// game's specular intensity; 0.5 and above a full dielectric), the parameter specular, and the game's specular LOD fade.
float pbrRoughness(float power) { return clamp(sqrt(2.0 / (max(power, 0.0) + 2.0)), 0.03, 1.0); }
float pbrSpecularScale(float gloss, float lodFade) { return clamp(2.0 * gloss, 0.0, 1.0) * p_specular * clamp(lodFade, 0.0, 1.0); }
// The environment's reflection (docs/INDIRECT.md "Reflections") on a dielectric (reflectance 0.04) of GGX roughness
// alpha: the probe prefiltered for alpha in the mirror direction (world space through the vertex shader's frame),
// times the split-sum's directional albedo (Karis 2014's analytic fit of the GGX DFG term), faded where the mirror
// direction goes under the surface (horizon occlusion: the normal map can point the reflection below the geometry).
// Zero without a probe or a frame, and for a material with its own environment-map pass (c219.x, the plugin: the game
// draws that reflection itself).
#if defined(FNV_ENV) && defined(FNV_FRAME)
vec3 pbrReflection(vec3 n, vec3 v, float alpha) {
  if (!fnvEnvValid() || fnvC[219].x != 0.0) return vec3(0.0);
  vec3 rt = reflect(-v, n);
  vec3 r = fnvToWorld(rt);
  if (dot(r, r) == 0.0) return vec3(0.0);
  float nv = clamp(dot(n, v), 0.0, 1.0), rough = sqrt(alpha);
  vec4 q = rough * vec4(-1.0, -0.0275, -0.572, 0.022) + vec4(1.0, 0.0425, 1.04, -0.04);
  float a004 = min(q.x * q.x, exp2(-9.28 * nv)) * q.x + q.y;
  vec2 ab = vec2(-1.04, 1.04) * a004 + q.zw;
  float horizon = clamp(1.0 + rt.z, 0.0, 1.0);
  return fnvEnvRadiance(r, alpha) * ((0.04 * ab.x + ab.y) * horizon * horizon);
}
#else
vec3 pbrReflection(vec3 n, vec3 v, float alpha) { return vec3(0.0); }
#endif
#endif

// Specular of one light (the game's Blinn-Phong, damped where the light grazes the surface).
float specular(vec3 n, vec3 l, vec3 h, float gloss) {
  float s = powLegacy(clamp(dot(n, nrm(h)), 0.0, 1.0), Toggles.z) * gloss;
  float nl = dot(n, l);
  return 0.2 - nl >= 0.0 ? s * clamp(nl + 0.5, 0.0, 1.0) : s;
}

void main() {
#ifdef FNV_SUN
  const float sunVisibility = fnvSunVisibility();  // before any discard: it takes screen derivatives
#else
  const float sunVisibility = 1.0;
#endif
  vec4 base = texture(FNV_TEX(BaseMap), t0.xy);
#ifndef FIXED_PATH
  if ((AmbientColor.w - 1.0 >= 0.0 ? 0.0 : 1.0) * (base.w - Toggles.w) < 0.0) discard;
#endif
#ifdef SILHOUETTE
  oC0 = vec4(mix(AmbientColor.xyz, v1.xyz, v1.w), base.w * AmbientColor.w);
#else
#ifdef PARALLAX
  const vec2 uv = parallaxUv();
  vec3 baseColor = texture(FNV_TEX(BaseMap), uv).xyz;  // the alpha stays the unshifted map's
#else
  const vec2 uv = t0.xy;
  vec3 baseColor = base.xyz;
#endif
  vec4 nmap = texture(FNV_TEX(NormalMap), uv);
  vec3 n = nrm((nmap.xyz - 0.5) * 2.0);

  vec3 sun = PSLightColor_(0) * clamp(dot(n, t1.xyz), 0.0, 1.0);
#ifdef FACEGENBLEND
  vec3 e = nrm(t6.xyz);
  sun += PSLightColor_(0) * skinBack(n, t1.xyz, e) * 0.5;
#endif
#ifdef STBB
  sun *= 0.85;
#endif
  vec3 shadow = vec3(1.0);  // the game's actor shadows on the sun (diffuse and specular)
#ifdef PROJ_SHADOW
  vec4 sm = texture(FNV_TEX(ShadowMap), t7.xy);
  float mask = texture(FNV_TEX(ShadowMaskMap), t7.zw).x;
  shadow = mask * (sm.xyz - 1.0) + 1.0;
  sun *= shadow;
#endif
  vec3 light = sun;
#if LIGHTS >= 2
  float att = clamp((1.0 - texture(FNV_TEX(AttenuationMap), ATTEN_COORDS.xy).x) -
                    texture(FNV_TEX(AttenuationMap), ATTEN_COORDS.zw).x, 0.0, 1.0);
#ifdef FACEGENBLEND
  light += skinPoint(n, nrm(t2.xyz), e, PSLightColor_(1), texture(FNV_TEX(GlowMap), t0.xy).xyz) * att;
#else
  light += PSLightColor_(1) * clamp(dot(n, nrm(t2.xyz)), 0.0, 1.0) * att;
#endif
#endif
#if defined(SI) && !defined(HAIR)
  light += EmittanceColor.xyz * texture(FNV_TEX(GlowMap), t0.xy).xyz;
#endif
  light = max(light + AmbientColor.xyz, 0.0);

#ifdef HAIR
  vec4 layer = texture(FNV_TEX(GlowMap), t0.xy);
  vec3 albedo = (((EmittanceColor.xyz - 0.5) * v0.y + 0.5) * 2.0) * mix(base.xyz, layer.xyz, layer.w);
#else
#ifdef FACEGENBLEND
  // The face tint: FaceGenMap0 a signed detail on the base, FaceGenMap1 a 2x multiply, the result doubled.
  baseColor = ((texture(FNV_TEX(FaceGenMap0), t0.xy).xyz - 0.5) * 2.0 + baseColor) *
              (texture(FNV_TEX(FaceGenMap1), t0.xy).xyz * 2.0) * 2.0;
#endif
#ifdef FIXED_PATH
  vec3 albedo = baseColor * v0.xyz;
#else
  vec3 albedo = -Toggles.x >= 0.0 ? baseColor : baseColor * v0.xyz;
#endif
#endif
  vec3 color = light * albedo;

#ifdef SPECULAR
#ifdef HAIR
  // Hair: a strand highlight along the tangent bent from the normal.
  vec3 h = nrm(t3.xyz);
  vec3 tangent = nrm(n * 0.5 + vec3(0.0, 0.0, 1.0));
  float x = clamp(abs(dot(tangent, t1.xyz) - dot(tangent, h)), 0.0, 1.0);
  float s = powLegacy(1.0 - x, 30.0) * (nmap.w * 0.7);
  float nl = dot(n, t1.xyz);
  s = 0.2 - nl >= 0.0 ? s * clamp(nl + 0.5, 0.0, 1.0) : s;
  vec3 spec = clamp(s * PSLightColor_(0) * t1.w, 0.0, 1.0) * shadow;
#else
  vec3 spec = clamp(specular(n, t1.xyz, t3.xyz, nmap.w) * PSLightColor_(0) * t1.w, 0.0, 1.0) * shadow;
#endif
#if LIGHTS >= 2
  spec += clamp(specular(n, nrm(t2.xyz), t4.xyz, nmap.w) * PSLightColor_(1) * att, 0.0, 1.0);
#endif
  color += spec;
#endif

#ifdef PBR
  if (p_pbr) {
#ifdef FACEGENBLEND
    // Skin (docs/SKIN.md): Community Shaders' skin model (FNV_SKIN) on the game's lights, the detail normal map over the
    // normal map. The game's own light through thin skin is kept (in linear light): the sun's back light at the
    // silhouette, and a point light's back light and wrap past the terminator tinted by the glow map.
    const vec3 v = dot(fnvView.xyz, fnvView.xyz) > 0.0 ? normalize(fnvView.xyz) : e;
    vec3 ns = fnvSkinDetail(n, uv);
#ifdef FNV_WET
    // Rain on skin (docs/SKIN.md): the wet world's darkening and water film; water runs off skin (no standing water).
    FnvWetObject wet = fnvWetObjectBegin(ns, albedo, -1.0, 0.0);
#endif
    const FnvSkin m = fnvSkinMaterial(0.0, dot(ns, v));
    const vec3 sunColor = toLinear(fnvC[3].xyz) * sunVisibility * shadow, sunDir = normalize(t1.xyz);
    vec3 diffuse = vec3(0.0), spec = vec3(0.0);
    fnvSkinLight(m, ns, v, sunDir, sunColor, diffuse, spec);
    vec3 through = sunColor * (skinBack(ns, sunDir, e) * 0.5);
#if LIGHTS >= 2
    const vec3 pointColor = toLinear(fnvC[4].xyz), pointDir = nrm(t2.xyz);
    fnvSkinLight(m, ns, v, pointDir, pointColor * att, diffuse, spec);
    const vec3 tint = toLinear(texture(FNV_TEX(GlowMap), t0.xy).xyz);
    const float pd = dot(ns, pointDir), pds = clamp(pd, 0.0, 1.0);
    const float pw = clamp((pd + 0.3) * 0.769230783, 0.0, 1.0);
    const float wrap = clamp((pw * -2.0 + 3.0) * (pw * pw) - (pds * -2.0 + 3.0) * (pds * pds), 0.0, 1.0);
    through += (skinBack(ns, pointDir, e) * mix(pointColor, tint, 0.5) + wrap * tint) * att;
#endif
    // The ambient (Skin::SkinIndirectLobeWeights): the probe's diffuse light by the lobes' remainder, and the reflection
    // (the probe along the mirror direction at the mixed roughness; the ambient colour without one), faded under the
    // horizon of the geometry, occluded by the ambient occlusion.
    const float ao = fnvAmbientOcclusion();
    const float nv = clamp(dot(ns, v), 0.0, 1.0);
    const vec2 lobes = fnvSkinIndirect(m, nv);
    vec3 lit = toLinear(albedo) * (diffuse + through + lobes.x * pbrIndirect(AmbientColor.xyz, fnvWorldNormal(ns), ao)) + spec;
    const vec3 r = reflect(-v, ns);
    const float horizon = clamp(1.0 + r.z, 0.0, 1.0);
    vec3 env = toLinear(AmbientColor.xyz);
#ifdef FNV_ENV
    const vec3 rw = fnvToWorld(r);
    if (fnvEnvValid() && dot(rw, rw) > 0.0) env = fnvEnvRadiance(rw, m.avg * m.avg);
#endif
    lit += env * (lobes.y * horizon * horizon * fnvSpecularOcclusion(ao, nv, m.avg));
#ifdef FNV_WET
    lit = fnvWetObjectFilm(wet, lit, AmbientColor.xyz, sunDir, sunColor);
#endif
    color = toGamma(lit);
#else
#ifdef FNV_WET
    // Wet world (docs/WETWORLD.md): the rain on the object (its relief for the local puddles: the base map's brightness
    // against its own average, its last mip).
    FnvWetObject wet = fnvWetObjectBegin(n, albedo, clamp(0.5 + 0.6 * (dot(baseColor, vec3(0.3, 0.59, 0.11)) -
        dot(textureLod(FNV_TEX(BaseMap), uv, 14.0).xyz, vec3(0.3, 0.59, 0.11))), 0.0, 1.0), 1.0);
#endif
    vec3 sunColor = toLinear(fnvC[3].xyz) * sunVisibility * shadow;
#ifdef STBB
    sunColor *= 0.85;
#endif
    vec3 sunDir = normalize(t1.xyz);
    vec3 diffuse = sunColor * clamp(dot(n, sunDir), 0.0, 1.0);
#if LIGHTS >= 2
    vec3 pointColor = toLinear(fnvC[4].xyz) * att, pointDir = nrm(t2.xyz);
    diffuse += pointColor * clamp(dot(n, pointDir), 0.0, 1.0);
#endif
    // Indirect light: the ambient, given direction by the probe and occluded by the ambient occlusion; the glow map's
    // emittance is the surface's own light (neither).
    float ao = fnvAmbientOcclusion();
    vec3 indirect = pbrIndirect(AmbientColor.xyz, fnvWorldNormal(n), ao);
#if defined(SI)
    indirect += toLinear(EmittanceColor.xyz * texture(FNV_TEX(GlowMap), t0.xy).xyz);
#endif
    vec3 lit = toLinear(albedo) * (diffuse + indirect);
#ifdef SPECULAR
    if (dot(fnvView.xyz, fnvView.xyz) > 0.0) {
      vec3 v = normalize(fnvView.xyz);
      float alpha = pbrRoughness(Toggles.z), scale = pbrSpecularScale(nmap.w, t1.w);
      lit += pbrSpecular(n, v, sunDir, sunColor, alpha) * scale;
#if LIGHTS >= 2
      lit += pbrSpecular(n, v, pointDir, pointColor, alpha) * scale;
#endif
      lit += pbrReflection(n, v, alpha) * (scale * fnvSpecularOcclusion(ao, clamp(dot(n, v), 0.0, 1.0), sqrt(alpha)));
    }
#endif
#ifdef FNV_WET
    lit = fnvWetObjectFilm(wet, lit, AmbientColor.xyz, sunDir, sunColor);
#endif
    color = toGamma(lit);
#endif  // FACEGENBLEND
  }
#endif

#ifdef FIXED_PATH
  vec3 rgb = mix(color, v1.xyz, v1.w);
#else
  vec3 rgb = -Toggles.y >= 0.0 ? color : mix(color, v1.xyz, v1.w);
#endif
  oC0 = vec4(rgb, base.w * AmbientColor.w);
#endif
  fnvAlphaTest(oC0.w);
#if defined(FNV_SKIN) && defined(FACEGENBLEND)
  if (p_pbr) fnvSkinStore(albedo);  // the skin buffer for the subsurface scattering (docs/SKIN.md)
#endif
  oC0.xyz = fnvFixedFog(oC0.xyz);
}
