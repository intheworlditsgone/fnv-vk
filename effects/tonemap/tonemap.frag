// Tonemapping at the "tonemap" point: after the game's image space, before the HUD.
//
// The game's HDR pass (ISHDRBLENDINSHADERCIN) applies exposure (HDRParam.x / adapted luminance), adds bloom and the
// cinematic saturation, tint, brightness, contrast and fade, but no tone curve: values above 1.0 were clipped by its
// 8-bit targets. With wide targets they arrive here intact. The colours are gamma encoded (the game's content is made
// for a gamma 2.2 display), so they are decoded to linear light, mapped by the operator, and encoded again.
//
// Grading, in linear light before the operator (so the curve compresses whatever it produces): contrast scales the
// distance from mid grey (0.18) in log2 space, saturation scales each colour's distance from its BT.709 luminance.
// At 1.0 both are skipped (the vanilla operator stays bit-exact).
//
// Interiors: with p_interior on, an interior cell (fnv.shadowInfo.z 1, set by the plugin at this point too) uses the
// p_i* parameters instead (settings()).
//
// SDR output: the operator's result (0..1) is the image.
// HDR output: each operator keeps its look in shadows and midtones and only the highlights are extended toward the
// display's peak. The operator's result is scaled by the luminance ratio of two neutral curves: both are the identity
// below `knee`, above it one rolls off to SDR white (1.0) and the other to peak / paper white. Below the knee the ratio
// is 1 (the SDR image exactly); where the operator reaches white, the ratio reaches the peak.

layout(location = 0) out vec4 color;

const vec3 kLuma = vec3(0.2126, 0.7152, 0.0722);  // BT.709

vec3 reinhard(vec3 x) { return x / (1.0 + x); }
vec3 reinhardExtended(vec3 x, float w) { return min(x * (1.0 + x / (w * w)) / (1.0 + x), 1.0); }  // 1.0 at x = w

// Krzysztof Narkowicz, "ACES Filmic Tone Mapping Curve" (2016), with his 0.6 input scale.
vec3 acesNarkowicz(vec3 x) {
  x *= 0.6;
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

// Stephen Hill's fit of the ACES RRT + sRGB ODT (BakingLab, ACES.hlsl).
const mat3 kAcesIn = mat3(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383, 0.04823, 0.01566, 0.83777);
const mat3 kAcesOut = mat3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276, -0.07367, -0.00605, 1.07602);
vec3 acesHill(vec3 x) {
  vec3 v = kAcesIn * x;
  vec3 a = v * (v + 0.0245786) - 0.000090537;
  vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
  return clamp(kAcesOut * (a / b), 0.0, 1.0);
}

// AgX (Troy Sobotka), the minimal approximation by Benjamin Wrensch: inset matrix, log2 encoding over
// [-12.47393, 4.026069] EV, 6th order polynomial fit of the sigmoid, outset matrix, display decode (2.2).
const mat3 kAgxIn = mat3(0.842479062253094, 0.0423282422610123, 0.0423756549057051,
                         0.0784335999999992, 0.878468636469772, 0.0784336,
                         0.0792237451477643, 0.0791661274605434, 0.879142973793104);
const mat3 kAgxOut = mat3(1.19687900512017, -0.0528968517574562, -0.0529716355144438,
                          -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
                          -0.0990297440797205, -0.0989611768448433, 1.15107367264116);
vec3 agx(vec3 x, bool punchy) {
  const float minEv = -12.47393, maxEv = 4.026069;
  vec3 v = kAgxIn * x;
  v = (clamp(log2(max(v, 1e-10)), minEv, maxEv) - minEv) / (maxEv - minEv);
  vec3 v2 = v * v, v4 = v2 * v2;
  v = 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v - 0.00232;
  if (punchy) {  // the "Punchy" look: power 1.35, saturation 1.4
    v = pow(max(v, 0.0), vec3(1.35));
    float l = dot(v, kLuma);
    v = l + 1.4 * (v - l);
  }
  v = kAgxOut * v;
  return clamp(pow(max(v, 0.0), vec3(2.2)), 0.0, 1.0);
}

// John Hable's Uncharted 2 curve (GDC 2010), exposure bias 2, white point 11.2.
vec3 hableCurve(vec3 x) {
  const float A = 0.15, B = 0.50, C = 0.10, D = 0.20, E = 0.02, F = 0.30;
  return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}
vec3 hable(vec3 x) { return clamp(hableCurve(2.0 * x) / hableCurve(vec3(11.2)), 0.0, 1.0); }

// Hajime Uchimura, "HDR theory and practice" (Gran Turismo, CEDEC 2017); defaults P=1, a=1, m=0.22, l=0.4, c=1.33, b=0.
vec3 uchimura(vec3 x) {
  const float P = 1.0, a = 1.0, m = 0.22, l = 0.4, c = 1.33, b = 0.0;
  float l0 = ((P - m) * l) / a;
  float S0 = m + l0, S1 = m + a * l0;
  float C2 = (a * P) / (P - S1), CP = -C2 / P;
  vec3 w0 = 1.0 - smoothstep(0.0, m, x);
  vec3 w2 = step(m + l0, x);
  vec3 w1 = 1.0 - w0 - w2;
  vec3 T = m * pow(x / m, vec3(c)) + b;
  vec3 S = P - (P - S1) * exp(CP * (x - S0));
  vec3 L = m + a * (x - m);
  return clamp(T * w0 + L * w1 + S * w2, 0.0, 1.0);
}

// Timothy Lottes, "Advanced Techniques and Optimization of HDR Color Pipelines" (GDC 2016): contrast 1.6, shoulder
// 0.977, hdrMax 8, mid grey 0.18 -> 0.267.
vec3 lottes(vec3 x) {
  const float a = 1.6, d = 0.977, hdrMax = 8.0, midIn = 0.18, midOut = 0.267;
  float b = (-pow(midIn, a) + pow(hdrMax, a) * midOut) / ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
  float c = (pow(hdrMax, a * d) * pow(midIn, a) - pow(hdrMax, a) * pow(midIn, a * d) * midOut) /
            ((pow(hdrMax, a * d) - pow(midIn, a * d)) * midOut);
  return clamp(pow(x, vec3(a)) / (pow(x, vec3(a * d)) * b + c), 0.0, 1.0);
}

// The settings in effect: the interior ones in an interior cell (fnv.shadowInfo.z 1) while "Interiors: own settings" is
// on, the general ones otherwise.
struct Settings {
  int op;
  float exposure, contrast, saturation, white;
  bool autoExposure;
  float aeTarget, aeMin, aeMax;
};
Settings settings() {
  if (p_interior && fnv.shadowInfo.z != 0.0)
    return Settings(p_iOp, p_iExposure, p_iContrast, p_iSaturation, p_iWhite, p_iAutoExposure, p_iAeTarget, p_iAeMin, p_iAeMax);
  return Settings(p_op, p_exposure, p_contrast, p_saturation, p_white, p_autoExposure, p_aeTarget, p_aeMin, p_aeMax);
}

vec3 tonemap(vec3 x, int op, float white) {
  switch (op) {
    case 1: return reinhard(x);
    case 2: return reinhardExtended(x, white);
    case 3: return acesNarkowicz(x);
    case 4: return acesHill(x);
    case 5: return agx(x, false);
    case 6: return agx(x, true);
    case 7: return hable(x);
    case 8: return uchimura(x);
    case 9: return lottes(x);
    default: return clamp(x, 0.0, 1.0);  // vanilla: the clip the game's 8-bit targets did
  }
}

// Identity below k, then a Reinhard shoulder (slope 1 at k) approaching m.
float rolloff(float y, float k, float m) {
  if (y <= k) return y;
  float t = (y - k) / (m - k);
  return k + (m - k) * t / (1.0 + t);
}

vec3 grade(vec3 x, float contrast, float saturation) {
  if (contrast != 1.0) x = 0.18 * pow(max(x, 0.0) / 0.18, vec3(contrast));
  if (saturation != 1.0) {
    float l = dot(x, kLuma);
    x = max(l + saturation * (x - l), 0.0);
  }
  return x;
}

void main() {
  ivec2 p = ivec2(fnvPixel());
  vec4 s = fnvSceneColor(p);
  const Settings c = settings();
  // Auto-exposure (aeadapt.frag): the correction that brings the adapted log-average luminance to the target (mid grey
  // 0.18 times 2^target), within the allowed range; off: none (the manual exposure alone, as before).
  float ev = c.exposure;
  if (c.autoExposure) {
    const float adapted = texelFetch(sampler2D(img_aeadapt, fnvPoint), ivec2(0), 0).r;
    ev += clamp(log2(0.18) + c.aeTarget - adapted, c.aeMin, c.aeMax);
  }
  vec3 x = grade(pow(max(s.rgb, 0.0), vec3(2.2)) * exp2(ev), c.contrast, c.saturation);
  vec3 t = tonemap(x, c.op, c.white);
  if (fnv.hdr.x != 0.0) {
    float y = dot(x, kLuma);
    float k = p_knee;
    if (y > k) t *= rolloff(y, k, max(fnv.hdr.w, 1.0 + 1e-3)) / rolloff(y, k, 1.0);
  }
  color = vec4(pow(t, vec3(1.0 / 2.2)), s.a);
}
