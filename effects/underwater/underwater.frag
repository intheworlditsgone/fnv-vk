layout(location = 0) out vec4 color;

const float waterIndex = 1.333;
const vec2 waveK[4] = vec2[](vec2(.037,.014), vec2(-.018,.052),
                            vec2(.076,-.031), vec2(.023,.098));
const vec4 amplitude = vec4(2.3,1.2,.65,.31);
const vec4 speed = vec4(1.21,1.43,1.76,1.98);

vec2 waveSlope(vec2 xy) {
  vec2 slope = vec2(0);
  for (int i=0; i<4; ++i)
    slope += amplitude[i]*waveK[i]*cos(dot(waveK[i],xy)-speed[i]*fnv.time.x);
  return slope;
}

// Horizontal landing position of a solar ray at depth h.
vec2 landingOffset(vec2 surface, float h, vec3 sun) {
  vec3 normal = normalize(vec3(-waveSlope(surface),1));
  vec3 light = refract(-sun,normal,1.0/waterIndex);
  return light.xy * (h / max(-light.z,.15));
}

float concentration(vec2 receiver, float h, vec3 sun) {
  vec3 flatRay = refract(-sun,vec3(0,0,1),1.0/waterIndex);
  vec2 entry = receiver - h*flatRay.xy / max(-flatRay.z,.15);
  entry = receiver - landingOffset(entry,h,sun);
  vec2 center = landingOffset(entry,h,sun);
  // Area of a small surface patch after projection to the receiver plane.
  vec2 dx = vec2(1,0) + (landingOffset(entry+vec2(2,0),h,sun)-center)*.5;
  vec2 dy = vec2(0,1) + (landingOffset(entry+vec2(0,2),h,sun)-center)*.5;
  float area = abs(determinant(mat2(dx,dy)));
  return inversesqrt(area*area+.16) * sqrt(1.16);
}

vec3 worldDirection(vec3 view) {
  return vec3(dot(fnv.cameraRows[0].xyz,view.zyx),
              dot(fnv.cameraRows[1].xyz,view.zyx),
              dot(fnv.cameraRows[2].xyz,view.zyx));
}

vec3 lookupDirection(vec3 dir, vec3 fallback) {
  vec3 axes = dir.x*fnv.cameraRows[0].xyz + dir.y*fnv.cameraRows[1].xyz + dir.z*fnv.cameraRows[2].xyz;
  if (axes.x <= .001) return fallback;
  vec2 tanXY = axes.zy / axes.x;
  vec2 uv = (tanXY-fnv.frustum.xz)/(fnv.frustum.yw-fnv.frustum.xz);
  if (any(lessThanEqual(uv,vec2(0))) || any(greaterThanEqual(uv,vec2(1)))) return fallback;
  vec2 pixel = fnv.rect.xy+uv*fnv.rect.zw;
  vec3 sampleColor = textureLod(sampler2D(fnvColorTex,fnvLinear),pixel*fnv.target.zw,0).rgb;
  vec2 edge = min(uv,1.0-uv);
  return mix(fallback,pow(max(sampleColor,vec3(0)),vec3(2.2)),smoothstep(0.0,.06,min(edge.x,edge.y)));
}

// Average of an exponentially attenuated cosine over one view-ray segment.
// End-point weights include both camera extinction and sunlight loss with depth.
float integratedMode(float phase, float cycles, float decay, float startWeight, float endWeight) {
  float norm = decay*decay + cycles*cycles;
  if (norm < 0.0001)
    return 0.5*(startWeight + endWeight)*cos(phase + 0.5*cycles);
  float endPhase = phase + cycles;
  return (endWeight*(decay*cos(endPhase) + cycles*sin(endPhase))
        - startWeight*(decay*cos(phase) + cycles*sin(phase))) / norm;
}

float shaftLight(vec3 origin, vec3 ray, vec3 sun, float immersion, float path, float extinction) {
  vec3 down = refract(-sun,vec3(0,0,1),1.0/waterIndex);
  vec2 slope = down.xy/down.z;
  // Project every point on the submerged segment back to the mean water plane.
  vec2 entry = origin.xy + immersion*slope;
  vec2 travel = (ray.xy - ray.z*slope)*path;
  float sunLoss = 1.0/(550.0*max(-down.z,.01));
  float startWeight = exp(-immersion*sunLoss);
  float endWeight = exp(-max(immersion-ray.z*path,0.0)*sunLoss - extinction*path);
  float decay = path*(ray.z*sunLoss-extinction);
  float mean = integratedMode(0.0,0.0,decay,startWeight,endWeight);
  // Low-frequency wave curvature modulates direct light. Integrating each mode
  // removes view-step banding even when a long ray crosses many wave crests.
  float irradiance = mean;
  for (int i=0; i<4; ++i) {
    vec2 k = waveK[i]*0.36;
    float phase = dot(k,entry)-speed[i]*0.6*fnv.time.x;
    irradiance += 0.23*integratedMode(phase,dot(k,travel),decay,startWeight,endWeight);
  }
  float forward = max(dot(ray,-down),0.0);
  return max(irradiance,0.0)*path*extinction*(0.22+0.78*forward*forward);
}

void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  color = fnvSceneColor(px);
  if (!fnvInsideRect(px)) return;
  vec3 origin = vec3(fnv.cameraRows[0].w,fnv.cameraRows[1].w,fnv.cameraRows[2].w);
  float immersion = fnv.depthRange.z;
  vec3 view = fnvViewPos(gl_FragCoord.xy,1);
  vec3 ray = normalize(worldDirection(view));
  float depth = fnvRawDepth(px);
  float hit = fnvLinearZ(depth)*length(view);
  float surface = ray.z > 0.0 ? immersion/max(ray.z,1e-7) : 1e15;
  float path = min(hit,surface);
  vec3 pos = origin + path*ray;
  vec3 scene = pow(max(color.rgb,vec3(0)),vec3(2.2));
  vec3 sun = fnv.waterSun.xyz / max(length(fnv.waterSun.xyz),.001);
  float day = clamp(fnv.waterSun.w,0,1)*smoothstep(.02,.28,sun.z);
  float h = max(immersion-path*ray.z,0);
  vec3 dx = dFdx(pos), dy = dFdy(pos);
  if (surface <= hit+1.0) {
    vec3 normal = normalize(vec3(-waveSlope(pos.xy),1));
    float cosine = clamp(dot(ray,normal),0,1);
    vec3 air = refract(ray,-normal,waterIndex);
    vec3 reflected = lookupDirection(reflect(ray,normal),scene);
    if (dot(air,air) < .5) scene = reflected;
    else {
      float transmitted = max(dot(air,normal),0);
      float rs = (waterIndex*cosine-transmitted)/max(waterIndex*cosine+transmitted,.001);
      float rp = (cosine-waterIndex*transmitted)/max(cosine+waterIndex*transmitted,.001);
      scene = mix(lookupDirection(air,scene),reflected,.5*(rs*rs+rp*rp));
    }
  } else if (!fnvIsSky(depth) && day > 0.0 && h < 900.0) {
    vec3 n = cross(dx,dy);
    n *= inversesqrt(max(dot(n,n),1e-12));
    if (dot(n,ray)>0) n=-n;
    float resolved = 1.0-smoothstep(4,20,max(length(dx),length(dy)));
    float direct = day*max(dot(n,sun),0)*resolved*exp(-h/280.0);
    scene *= 1.0+direct*(concentration(pos.xy,h,sun)-1.0);
  }

  // Slider controls inverse visibility distance. Half-distance in game units at density one.
  vec3 sigma = log(2.0)*max(p_density,0.0)/vec3(3200,8600,12500);
  vec3 transmission = exp(-sigma*path);
  vec3 veil = vec3(.012,.040,.048)*(0.25+.75*day);
  vec3 radiance = scene*transmission + veil*(1.0-transmission);
  if (day > 0.0 && path > 0.0) {
    float shafts = shaftLight(origin,ray,sun,immersion,path,sigma.g);
    radiance += day*shafts*vec3(.18,.24,.20);
  }
  color.rgb = pow(max(radiance,vec3(0)),vec3(1.0/2.2));
}
