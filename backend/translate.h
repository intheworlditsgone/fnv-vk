#pragma once
// D3D9 state values -> Vulkan. Pure functions; the D3D9 semantics follow DXVK's d3d9 front end (zlib licence,
// src/d3d9/d3d9_util.{h,cpp} and d3d9_device.cpp, v2.7.1), which is the reference implementation of D3D9 on Vulkan.
// Only enum values are used from <d3d9types.h>; nothing here touches a D3D9 interface.
#include "vkfn.h"  // windows.h + vulkan.h (prototype-free)
#include <d3d9types.h>

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace fnvvk::xlat {

inline VkCompareOp compareOp(uint32_t f) {
  switch (f) {
    case D3DCMP_NEVER: return VK_COMPARE_OP_NEVER;
    case D3DCMP_LESS: return VK_COMPARE_OP_LESS;
    case D3DCMP_EQUAL: return VK_COMPARE_OP_EQUAL;
    case D3DCMP_LESSEQUAL: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case D3DCMP_GREATER: return VK_COMPARE_OP_GREATER;
    case D3DCMP_NOTEQUAL: return VK_COMPARE_OP_NOT_EQUAL;
    case D3DCMP_GREATEREQUAL: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    case D3DCMP_ALWAYS: return VK_COMPARE_OP_ALWAYS;
    default: return VK_COMPARE_OP_NEVER;
  }
}

inline VkStencilOp stencilOp(uint32_t op) {
  switch (op) {
    case D3DSTENCILOP_KEEP: return VK_STENCIL_OP_KEEP;
    case D3DSTENCILOP_ZERO: return VK_STENCIL_OP_ZERO;
    case D3DSTENCILOP_REPLACE: return VK_STENCIL_OP_REPLACE;
    case D3DSTENCILOP_INCRSAT: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
    case D3DSTENCILOP_DECRSAT: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
    case D3DSTENCILOP_INVERT: return VK_STENCIL_OP_INVERT;
    case D3DSTENCILOP_INCR: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
    case D3DSTENCILOP_DECR: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
    default: return VK_STENCIL_OP_KEEP;
  }
}

// D3D9 front faces are clockwise; with the y-flipped viewport (see viewport()) screen orientation is preserved.
inline VkCullModeFlags cullMode(uint32_t m) {
  switch (m) {
    case D3DCULL_CW: return VK_CULL_MODE_FRONT_BIT;
    case D3DCULL_CCW: return VK_CULL_MODE_BACK_BIT;
    default: return VK_CULL_MODE_NONE;
  }
}

inline VkBlendFactor blendFactor(uint32_t b, bool alpha) {
  switch (b) {
    case D3DBLEND_ZERO: return VK_BLEND_FACTOR_ZERO;
    case D3DBLEND_ONE: return VK_BLEND_FACTOR_ONE;
    case D3DBLEND_SRCCOLOR: return VK_BLEND_FACTOR_SRC_COLOR;
    case D3DBLEND_INVSRCCOLOR: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case D3DBLEND_SRCALPHA: return VK_BLEND_FACTOR_SRC_ALPHA;
    case D3DBLEND_INVSRCALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case D3DBLEND_DESTALPHA: return VK_BLEND_FACTOR_DST_ALPHA;
    case D3DBLEND_INVDESTALPHA: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case D3DBLEND_DESTCOLOR: return VK_BLEND_FACTOR_DST_COLOR;
    case D3DBLEND_INVDESTCOLOR: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case D3DBLEND_SRCALPHASAT: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case D3DBLEND_BOTHSRCALPHA: return VK_BLEND_FACTOR_SRC_ALPHA;
    case D3DBLEND_BOTHINVSRCALPHA: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case D3DBLEND_BLENDFACTOR: return alpha ? VK_BLEND_FACTOR_CONSTANT_ALPHA : VK_BLEND_FACTOR_CONSTANT_COLOR;
    case D3DBLEND_INVBLENDFACTOR:
      return alpha ? VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA : VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    default: return VK_BLEND_FACTOR_ZERO;
  }
}

inline VkBlendOp blendOp(uint32_t op) {
  switch (op) {
    case D3DBLENDOP_SUBTRACT: return VK_BLEND_OP_SUBTRACT;
    case D3DBLENDOP_REVSUBTRACT: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case D3DBLENDOP_MIN: return VK_BLEND_OP_MIN;
    case D3DBLENDOP_MAX: return VK_BLEND_OP_MAX;
    default: return VK_BLEND_OP_ADD;
  }
}

// D3D9 BOTHSRCALPHA / BOTHINVSRCALPHA set the destination factor too (DXVK FixupBlendState).
inline void fixupBlend(uint32_t& src, uint32_t& dst) {
  if (src == D3DBLEND_BOTHSRCALPHA) { src = D3DBLEND_SRCALPHA; dst = D3DBLEND_INVSRCALPHA; }
  else if (src == D3DBLEND_BOTHINVSRCALPHA) { src = D3DBLEND_INVSRCALPHA; dst = D3DBLEND_SRCALPHA; }
}

inline VkFormat declType(uint32_t t) {
  switch (t) {
    case D3DDECLTYPE_FLOAT1: return VK_FORMAT_R32_SFLOAT;
    case D3DDECLTYPE_FLOAT2: return VK_FORMAT_R32G32_SFLOAT;
    case D3DDECLTYPE_FLOAT3: return VK_FORMAT_R32G32B32_SFLOAT;
    case D3DDECLTYPE_FLOAT4: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case D3DDECLTYPE_D3DCOLOR: return VK_FORMAT_B8G8R8A8_UNORM;
    case D3DDECLTYPE_UBYTE4: return VK_FORMAT_R8G8B8A8_USCALED;
    case D3DDECLTYPE_SHORT2: return VK_FORMAT_R16G16_SSCALED;
    case D3DDECLTYPE_SHORT4: return VK_FORMAT_R16G16B16A16_SSCALED;
    case D3DDECLTYPE_UBYTE4N: return VK_FORMAT_R8G8B8A8_UNORM;
    case D3DDECLTYPE_SHORT2N: return VK_FORMAT_R16G16_SNORM;
    case D3DDECLTYPE_SHORT4N: return VK_FORMAT_R16G16B16A16_SNORM;
    case D3DDECLTYPE_USHORT2N: return VK_FORMAT_R16G16_UNORM;
    case D3DDECLTYPE_USHORT4N: return VK_FORMAT_R16G16B16A16_UNORM;
    case D3DDECLTYPE_UDEC3: return VK_FORMAT_A2B10G10R10_USCALED_PACK32;
    case D3DDECLTYPE_DEC3N: return VK_FORMAT_A2B10G10R10_SNORM_PACK32;
    case D3DDECLTYPE_FLOAT16_2: return VK_FORMAT_R16G16_SFLOAT;
    case D3DDECLTYPE_FLOAT16_4: return VK_FORMAT_R16G16B16A16_SFLOAT;
    default: return VK_FORMAT_UNDEFINED;
  }
}

inline VkPrimitiveTopology topology(uint32_t pt) {
  switch (pt) {
    case D3DPT_POINTLIST: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case D3DPT_LINELIST: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case D3DPT_LINESTRIP: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case D3DPT_TRIANGLESTRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    case D3DPT_TRIANGLEFAN: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  }
}

inline uint32_t vertexCount(uint32_t pt, uint32_t prims) {
  switch (pt) {
    case D3DPT_POINTLIST: return prims;
    case D3DPT_LINELIST: return prims * 2;
    case D3DPT_LINESTRIP: return prims + 1;
    case D3DPT_TRIANGLESTRIP:
    case D3DPT_TRIANGLEFAN: return prims + 2;
    default: return prims * 3;
  }
}

inline VkFilter filter(uint32_t f) { return f > D3DTEXF_POINT ? VK_FILTER_LINEAR : VK_FILTER_NEAREST; }
inline VkSamplerMipmapMode mipFilter(uint32_t f) {
  return f > D3DTEXF_POINT ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
}
inline VkSamplerAddressMode addressMode(uint32_t m) {
  switch (m) {
    case D3DTADDRESS_MIRROR: return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    case D3DTADDRESS_CLAMP: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case D3DTADDRESS_BORDER: return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
    case D3DTADDRESS_MIRRORONCE: return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
    default: return VK_SAMPLER_ADDRESS_MODE_REPEAT;
  }
}

// D3DCOLOR (A8R8G8B8) -> rgba floats, as the D3D9 reference converts it (c * 1/255 in single precision; verified
// for clears against real D3D9 in tools/backend_test).
inline void d3dColor(uint32_t argb, float out[4]) {
  constexpr float k = 1.0f / 255.0f;
  out[0] = float((argb >> 16) & 0xFF) * k;
  out[1] = float((argb >> 8) & 0xFF) * k;
  out[2] = float(argb & 0xFF) * k;
  out[3] = float((argb >> 24) & 0xFF) * k;
}

inline float asFloat(uint32_t v) {
  float f;
  memcpy(&f, &v, 4);
  return f;
}

}  // namespace fnvvk::xlat
