// shaderdiff: differential test of translated pixel shaders.
//
// For every job (one original ps_2_x/ps_3_0 blob plus its SPIR-V translation from sdp2spv) it renders a 64x64
// RGBA32F quad twice with identical seeded inputs:
//   * reference: the original bytecode on the system D3D9 runtime (hardware device);
//   * candidate: the SPIR-V on Vulkan 1.3, bound with the ABI in docs/SHADER_ABI.md.
// and compares the two images per pixel.
//
// Input alignment between the APIs:
//   * D3D9 samples pixels at integer window coordinates, Vulkan at +0.5. The D3D9 quad is shifted by -0.5 px so
//     the interpolants at each pixel are identical.
//   * D3D9 clip +Y is the top row; Vulkan uses a negative-height viewport to match.
//   * Both targets are cleared to a sentinel so discard (texkill) behaviour is compared too.
//
// Usage: shaderdiff <jobs.txt> <spv-dir> <results.csv> [--no-validation]
// jobs.txt lines: <hash> <profile> <cbv bindings, comma separated> <textures "reg:kind,..." or "-"> <defs or "-">
//                 [<sun light colour register or -1>]   (kind: 5 = 2D, 9 = cube; ir::ResourceKind)
//
// Sun shadows (sdp2spv's sun shadow pass, docs/SHADER_ABI.md): shaders with a sun register also get the sun block
// (binding 9) and a 2-layer D32 cascade image with a compare sampler (48, 49). SHADERDIFF_SUN selects the test:
//   off (default)  the block is zero: the shader must give exactly the original's result;
//   shadow         cascade 1 is fully occluded and the block selects it for every pixel: the result must be the
//                  original's with the sun colour's rgb times (1 - SHADERDIFF_SUN_STRENGTH, default 1);
//   blend          cascade 0 lit, cascade 1 fully occluded, and every fragment inside cascade 0's blend band (its
//                  last 20%): the result must be the original's with the sun colour's rgb times 1 - the blend weight;
//   splitx/splity  cascade 1's left half is occluded and maps view-space x (splitx) or y (splity) across it: each pixel
//                  must be the shadowed result on the side the camera geometry puts in shadow and the original's on
//                  the other, except a band of 6 pixels where the 3x3 filter mixes the two.

#include <windows.h>
#include <d3d9.h>
#include <d3dcompiler.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <stdexcept>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "passthrough_vert.h"
#include "../../backend/construns.h"

static constexpr uint32_t W = 64, H = 64, TEX = 32;
static constexpr float kSentinel = -7777.0f;

#define CHECK_HR(x) do { HRESULT hr_ = (x); if (FAILED(hr_)) { char b_[256]; std::snprintf(b_, sizeof(b_), "%s failed: 0x%08lx", #x, (unsigned long)hr_); throw std::runtime_error(b_); } } while (0)
#define CHECK_VK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { char b_[256]; std::snprintf(b_, sizeof(b_), "%s failed: %d", #x, (int)r_); throw std::runtime_error(b_); } } while (0)

// ---------------------------------------------------------------------------------------------------------------
struct Job {
  std::string hash, profile;
  std::vector<uint32_t> cbvs;
  std::vector<std::pair<uint32_t, uint32_t>> textures;  // reg, kind
  std::string defs;                                     // shader-defined float constants, "reg/x/y/z/w,..."
  int sun = -1;                                         // sun light colour register (sun shadow pass), -1: none
  bool isPs3() const { return profile.rfind("ps_3", 0) == 0; }
};

static void applyDefOverlay(const std::string& defStr, std::vector<float>& constants);
// SHADERDIFF_POISON: as the runtime with FNVVK_POISON_CONSTANTS, only the registers of the shader's constant runs
// (backend/construns.h) keep their values and the rest of the block is NaN, then the defs are overlaid again (the
// runtime overlays them after the copy). A register the shader reads but the runs miss changes the result.
// SHADERDIFF_POISON=drop: the negative control (see below).
static void poisonUnread(const uint32_t* spv, size_t words, uint32_t set, std::vector<float>& c, const std::string& defStr) {
  if (!std::getenv("SHADERDIFF_POISON")) return;
  std::vector<fnvvk::ConstRun> runs;
  if (!fnvvk::spvConstantRuns(spv, words, set, uint32_t(c.size() / 4), runs)) return;  // the whole block is copied
  // SHADERDIFF_POISON=drop (negative control): the first run is left out too, so the shaders that read it must differ.
  if (std::string(std::getenv("SHADERDIFF_POISON")) == "drop" && !runs.empty()) runs.erase(runs.begin());
  std::vector<float> out(c.size(), std::numeric_limits<float>::quiet_NaN());
  for (const auto& r : runs)
    for (uint32_t k = r.first * 4u; k < (r.first + r.count) * 4u && k < c.size(); k++) out[k] = c[k];
  c.swap(out);
  applyDefOverlay(defStr, c);
}

struct Inputs {
  std::array<float, 4 * 11 * 4> vertices;  // 4 vertices x 11 float4 attributes (pos, t0..t7, c0, c1)
  std::array<float, 224 * 4> constants;
  // One BGRA8 texture per sampler register; cube textures have 6 faces.
  std::map<uint32_t, std::vector<uint32_t>> texels;
};

static std::vector<uint32_t> smoothTexture(std::mt19937& rng, uint32_t faces) {
  // Low-frequency content so bilinear filtering differences between APIs stay tiny.
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::vector<uint32_t> out(TEX * TEX * faces);
  for (uint32_t f = 0; f < faces; f++) {
    float p[4][3];
    for (auto& c : p) for (auto& v : c) v = u(rng) * 6.2831853f;
    float fr[4] = { 1.0f + u(rng) * 2.0f, 1.0f + u(rng) * 2.0f, 1.0f + u(rng) * 2.0f, 1.0f + u(rng) * 2.0f };
    for (uint32_t y = 0; y < TEX; y++) {
      for (uint32_t x = 0; x < TEX; x++) {
        float fx = float(x) / TEX * 6.2831853f, fy = float(y) / TEX * 6.2831853f;
        uint32_t ch[4];
        for (int c = 0; c < 4; c++) {
          float v = 0.5f + 0.25f * std::sin(fx * fr[c] + p[c][0]) + 0.25f * std::cos(fy * fr[(c + 1) & 3] + p[c][1]);
          ch[c] = uint32_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
        }
        // A8R8G8B8 in memory = B, G, R, A bytes = VK_FORMAT_B8G8R8A8_UNORM.
        out[(f * TEX + y) * TEX + x] = (ch[3] << 24) | (ch[0] << 16) | (ch[1] << 8) | ch[2];
      }
    }
  }
  return out;
}

static Inputs makeInputs(const Job& job) {
  std::seed_seq seq(job.hash.begin(), job.hash.end());
  std::mt19937 rng(seq);
  std::uniform_real_distribution<float> tc(-1.0f, 2.0f), col(0.0f, 1.0f), k(-1.0f, 1.0f);

  Inputs in;
  const float pos[4][2] = { { -1, 1 }, { 1, 1 }, { -1, -1 }, { 1, -1 } };  // triangle strip, clip space
  for (int v = 0; v < 4; v++) {
    float* a = &in.vertices[v * 44];
    a[0] = pos[v][0]; a[1] = pos[v][1]; a[2] = 0.5f; a[3] = 1.0f;
    for (int t = 0; t < 8; t++) for (int c = 0; c < 4; c++) a[4 + t * 4 + c] = tc(rng);
    for (int c = 0; c < 8; c++) a[36 + c] = col(rng);
  }
  for (auto& c : in.constants) c = k(rng);
  // SHADERDIFF_CONST="reg=x,y,z,w;reg=...": float constants pinned in both APIs (e.g. a replacement's settings).
  if (const char* pin = std::getenv("SHADERDIFF_CONST")) {
    std::stringstream ps(pin);
    std::string item;
    while (std::getline(ps, item, ';')) {
      const size_t eq = item.find('=');
      if (eq == std::string::npos) continue;
      const uint32_t reg = uint32_t(std::stoul(item.substr(0, eq)));
      std::stringstream vs(item.substr(eq + 1));
      std::string v;
      for (uint32_t c = 0; c < 4 && std::getline(vs, v, ','); c++)
        if (reg < 224) in.constants[reg * 4 + c] = std::stof(v);
    }
  }
  for (auto [reg, kind] : job.textures) in.texels[reg] = smoothTexture(rng, kind == 9 ? 6 : 1);
  return in;
}

// The sun test (SHADERDIFF_SUN): what the Vulkan side binds for a shader with a sun register.
enum class SunMode { Off, Shadow, Blend, SplitX, SplitY };
static constexpr float kBlendSplit = 0.36f;  // blend mode: cascade 0's far split (fragments at view depth 0.3307)
// The sun visibility the shader must compute in the blend mode, from its own formula: the quad at depth 0.5 is at view
// distance 4/3, moved 1 + 0.002 * z toward the camera along its normal; cascade 0 spans 0 .. kBlendSplit.
static float blendVisibility() {
  const float z = 1.0f * 2.0f / (2.0f - 0.5f * (2.0f - 1.0f));
  const float qz = z - (1.0f + 0.002f * z);
  const float start = kBlendSplit - 0.2f * (kBlendSplit - 0.0f);
  const float w = std::clamp((qz - start) / (kBlendSplit - start), 0.0f, 1.0f);
  return 1.0f + (0.0f - 1.0f) * w;  // lerp(cascade 0 = 1, cascade 1 = 0, w)
}
static constexpr uint32_t kSunMap = 64;  // cascade texels per side
struct SunSetup {
  float block[21][4] = { };
  std::vector<float> depth = std::vector<float>(kSunMap * kSunMap * 2, 1.0f);  // 2 layers, all lit
};
// Screen setup shared by every mode: viewport rect = the 64x64 target, frustum tangents (-1, 1, 1, -1), near 1, far 2
// (the harness quad's depth 0.5 is at view distance 4/3), cascade 0 everywhere lit, the first split at 0 so every
// fragment takes cascade 1 (rows 9..11), whose depth reference is 0.5.
static SunSetup makeSun(SunMode mode, float strength) {
  SunSetup u;
  if (mode == SunMode::Off) return u;
  const float block0[4] = { 1.0f, 1.0e6f, 1.0f / kSunMap, strength };
  const float rect[4] = { 0.0f, 0.0f, float(W), float(H) };
  const float frustum[4] = { -1.0f, 1.0f, 1.0f, -1.0f };
  const float range[4] = { 1.0f, 2.0f, 0.0f, 0.0f };
  const float splits[4] = { mode == SunMode::Blend ? kBlendSplit : 0.0f, 3.0e38f, 3.0e38f, 3.0e38f };
  std::memcpy(u.block[0], block0, 16);
  std::memcpy(u.block[1], rect, 16);
  std::memcpy(u.block[2], frustum, 16);
  std::memcpy(u.block[3], range, 16);
  std::memcpy(u.block[4], splits, 16);
  if (mode == SunMode::SplitX) u.block[9][0] = 0.6f;  // clip x = 0.6 * view x
  if (mode == SunMode::SplitY) u.block[9][1] = 0.6f;  // clip x = 0.6 * view y
  u.block[11][3] = 0.5f;                              // depth reference
  for (uint32_t y = 0; y < kSunMap; y++)
    for (uint32_t x = 0; x < kSunMap; x++)
      u.depth[kSunMap * kSunMap + y * kSunMap + x] = mode == SunMode::Shadow || mode == SunMode::Blend || x < kSunMap / 2 ? 0.0f : 1.0f;
  return u;
}

static std::vector<uint8_t> readFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// ---------------------------------------------------------------------------------------------------------------
// D3D9 reference
class D3D9Ref {
public:
  D3D9Ref() {
    WNDCLASSA wc = { };
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.lpszClassName = "shaderdiff";
    RegisterClassA(&wc);
    m_wnd = CreateWindowA("shaderdiff", "shaderdiff", WS_OVERLAPPEDWINDOW, 0, 0, W, H, nullptr, nullptr, wc.hInstance, nullptr);

    m_d3d = Direct3DCreate9(D3D_SDK_VERSION);
    if (!m_d3d) throw std::runtime_error("Direct3DCreate9 failed");
    D3DPRESENT_PARAMETERS pp = { };
    pp.Windowed = TRUE;
    pp.SwapEffect = D3DSWAPEFFECT_DISCARD;
    pp.BackBufferWidth = W;
    pp.BackBufferHeight = H;
    pp.BackBufferFormat = D3DFMT_X8R8G8B8;
    pp.hDeviceWindow = m_wnd;
    CHECK_HR(m_d3d->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, m_wnd,
      D3DCREATE_HARDWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE, &pp, &m_dev));

    D3DADAPTER_IDENTIFIER9 id = { };
    m_d3d->GetAdapterIdentifier(D3DADAPTER_DEFAULT, 0, &id);
    adapter = id.Description;

    CHECK_HR(m_dev->CreateTexture(W, H, 1, D3DUSAGE_RENDERTARGET, D3DFMT_A32B32G32R32F, D3DPOOL_DEFAULT, &m_rt, nullptr));
    CHECK_HR(m_rt->GetSurfaceLevel(0, &m_rtSurf));
    CHECK_HR(m_dev->CreateOffscreenPlainSurface(W, H, D3DFMT_A32B32G32R32F, D3DPOOL_SYSTEMMEM, &m_readback, nullptr));

    const D3DVERTEXELEMENT9 decl[] = {
      { 0, 0, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITION, 0 },
      { 0, 16, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 0 },
      { 0, 32, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 1 },
      { 0, 48, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 2 },
      { 0, 64, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 3 },
      { 0, 80, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 4 },
      { 0, 96, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 5 },
      { 0, 112, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 6 },
      { 0, 128, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, 7 },
      { 0, 144, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 0 },
      { 0, 160, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 1 },
      D3DDECL_END()
    };
    CHECK_HR(m_dev->CreateVertexDeclaration(decl, &m_decl));

    // Pass-through vertex shaders, one per pixel shader model (D3D9 does not mix SM3 with SM2).
    // The -0.5 px shift aligns D3D9's integer pixel centres with Vulkan's half-integer ones.
    const char* hlsl =
      "struct I { float4 p:POSITION; float4 t0:TEXCOORD0; float4 t1:TEXCOORD1; float4 t2:TEXCOORD2; float4 t3:TEXCOORD3;"
      "  float4 t4:TEXCOORD4; float4 t5:TEXCOORD5; float4 t6:TEXCOORD6; float4 t7:TEXCOORD7; float4 c0:COLOR0; float4 c1:COLOR1; };\n"
      "struct O { float4 p:POSITION; float4 t0:TEXCOORD0; float4 t1:TEXCOORD1; float4 t2:TEXCOORD2; float4 t3:TEXCOORD3;"
      "  float4 t4:TEXCOORD4; float4 t5:TEXCOORD5; float4 t6:TEXCOORD6; float4 t7:TEXCOORD7; float4 c0:COLOR0; float4 c1:COLOR1; };\n"
      "float2 halfPixel : register(c0);\n"
      "O main(I i) { O o = (O)i; o.p.xy += halfPixel * o.p.w; return o; }\n";
    m_vs2 = compileVs(hlsl, "vs_2_0");
    m_vs3 = compileVs(hlsl, "vs_3_0");
  }

  ~D3D9Ref() {
    for (auto* p : { (IUnknown*)m_sentinelSurf, (IUnknown*)m_sentinelSys, (IUnknown*)m_vs2, (IUnknown*)m_vs3, (IUnknown*)m_decl, (IUnknown*)m_readback, (IUnknown*)m_rtSurf,
                     (IUnknown*)m_rt, (IUnknown*)m_dev, (IUnknown*)m_d3d })
      if (p) p->Release();
    if (m_wnd) DestroyWindow(m_wnd);
  }

  std::vector<float> run(const Job& job, const std::vector<uint8_t>& psCode, const Inputs& in) {
    IDirect3DPixelShader9* ps = nullptr;
    CHECK_HR(m_dev->CreatePixelShader(reinterpret_cast<const DWORD*>(psCode.data()), &ps));

    std::vector<IDirect3DBaseTexture9*> textures;
    for (const auto& [reg, texels] : in.texels) {
      bool cube = texels.size() == size_t(TEX) * TEX * 6;
      IDirect3DBaseTexture9* base = nullptr;
      if (cube) {
        IDirect3DCubeTexture9* t = nullptr;
        CHECK_HR(m_dev->CreateCubeTexture(TEX, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &t, nullptr));
        for (uint32_t f = 0; f < 6; f++) upload(t, D3DCUBEMAP_FACES(f), &texels[f * TEX * TEX]);
        base = t;
      } else {
        IDirect3DTexture9* t = nullptr;
        CHECK_HR(m_dev->CreateTexture(TEX, TEX, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &t, nullptr));
        D3DLOCKED_RECT lr;
        CHECK_HR(t->LockRect(0, &lr, nullptr, 0));
        for (uint32_t y = 0; y < TEX; y++) std::memcpy((uint8_t*)lr.pBits + y * lr.Pitch, &texels[y * TEX], TEX * 4);
        t->UnlockRect(0);
        base = t;
      }
      textures.push_back(base);
      m_dev->SetTexture(reg, base);
      m_dev->SetSamplerState(reg, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
      m_dev->SetSamplerState(reg, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
      m_dev->SetSamplerState(reg, D3DSAMP_MIPFILTER, D3DTEXF_NONE);
      m_dev->SetSamplerState(reg, D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP);
      m_dev->SetSamplerState(reg, D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP);
      m_dev->SetSamplerState(reg, D3DSAMP_ADDRESSW, D3DTADDRESS_WRAP);
      m_dev->SetSamplerState(reg, D3DSAMP_SRGBTEXTURE, FALSE);
    }

    CHECK_HR(m_dev->SetRenderTarget(0, m_rtSurf));
    D3DVIEWPORT9 vp = { 0, 0, W, H, 0.0f, 1.0f };
    m_dev->SetViewport(&vp);
    m_dev->SetRenderState(D3DRS_ZENABLE, FALSE);
    m_dev->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
    m_dev->SetRenderState(D3DRS_ALPHATESTENABLE, FALSE);
    m_dev->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
    m_dev->SetRenderState(D3DRS_FOGENABLE, FALSE);
    m_dev->SetRenderState(D3DRS_LIGHTING, FALSE);
    m_dev->SetRenderState(D3DRS_SRGBWRITEENABLE, FALSE);
    m_dev->SetRenderState(D3DRS_COLORWRITEENABLE, 0xF);

    const float halfPixel[4] = { -1.0f / W, 1.0f / H, 0.0f, 0.0f };
    m_dev->SetVertexShaderConstantF(0, halfPixel, 1);
    m_dev->SetPixelShaderConstantF(0, in.constants.data(), job.isPs3() ? 224 : 32);
    m_dev->SetVertexDeclaration(m_decl);
    m_dev->SetVertexShader(job.isPs3() ? m_vs3 : m_vs2);
    m_dev->SetPixelShader(ps);

    // Clear()/ColorFill take a D3DCOLOR and cannot express the negative float sentinel; copy it in instead.
    fillSentinel();
    CHECK_HR(m_dev->BeginScene());
    CHECK_HR(m_dev->DrawPrimitiveUP(D3DPT_TRIANGLESTRIP, 2, in.vertices.data(), 44 * sizeof(float)));
    CHECK_HR(m_dev->EndScene());

    CHECK_HR(m_dev->GetRenderTargetData(m_rtSurf, m_readback));
    std::vector<float> out(W * H * 4);
    D3DLOCKED_RECT lr;
    CHECK_HR(m_readback->LockRect(&lr, nullptr, D3DLOCK_READONLY));
    for (uint32_t y = 0; y < H; y++) std::memcpy(&out[y * W * 4], (uint8_t*)lr.pBits + y * lr.Pitch, W * 16);
    m_readback->UnlockRect();

    for (uint32_t i = 0; i < 16; i++) m_dev->SetTexture(i, nullptr);
    for (auto* t : textures) t->Release();
    m_dev->SetPixelShader(nullptr);
    ps->Release();
    return out;
  }

  std::string adapter;

private:
  IDirect3DVertexShader9* compileVs(const char* src, const char* profile) {
    ID3DBlob* code = nullptr;
    ID3DBlob* err = nullptr;
    if (FAILED(D3DCompile(src, std::strlen(src), nullptr, nullptr, nullptr, "main", profile, 0, 0, &code, &err)))
      throw std::runtime_error(std::string("D3DCompile: ") + (err ? (const char*)err->GetBufferPointer() : "?"));
    IDirect3DVertexShader9* vs = nullptr;
    CHECK_HR(m_dev->CreateVertexShader((const DWORD*)code->GetBufferPointer(), &vs));
    code->Release();
    return vs;
  }

  void upload(IDirect3DCubeTexture9* t, D3DCUBEMAP_FACES face, const uint32_t* texels) {
    D3DLOCKED_RECT lr;
    CHECK_HR(t->LockRect(face, 0, &lr, nullptr, 0));
    for (uint32_t y = 0; y < TEX; y++) std::memcpy((uint8_t*)lr.pBits + y * lr.Pitch, &texels[y * TEX], TEX * 4);
    t->UnlockRect(face, 0);
  }

  void fillSentinel() {
    if (!m_sentinelSurf) {
      CHECK_HR(m_dev->CreateOffscreenPlainSurface(W, H, D3DFMT_A32B32G32R32F, D3DPOOL_SYSTEMMEM, &m_sentinelSys, nullptr));
      D3DLOCKED_RECT lr;
      CHECK_HR(m_sentinelSys->LockRect(&lr, nullptr, 0));
      for (uint32_t y = 0; y < H; y++) {
        float* row = (float*)((uint8_t*)lr.pBits + y * lr.Pitch);
        for (uint32_t i = 0; i < W * 4; i++) row[i] = kSentinel;
      }
      m_sentinelSys->UnlockRect();
      // A default-pool off-screen plain surface is always a valid StretchRect source to a render-target texture.
      CHECK_HR(m_dev->CreateOffscreenPlainSurface(W, H, D3DFMT_A32B32G32R32F, D3DPOOL_DEFAULT, &m_sentinelSurf, nullptr));
      CHECK_HR(m_dev->UpdateSurface(m_sentinelSys, nullptr, m_sentinelSurf, nullptr));
    }
    CHECK_HR(m_dev->StretchRect(m_sentinelSurf, nullptr, m_rtSurf, nullptr, D3DTEXF_NONE));
  }

  HWND m_wnd = nullptr;
  IDirect3D9* m_d3d = nullptr;
  IDirect3DDevice9* m_dev = nullptr;
  IDirect3DTexture9* m_rt = nullptr;
  IDirect3DSurface9* m_rtSurf = nullptr;
  IDirect3DSurface9* m_readback = nullptr;
  IDirect3DVertexDeclaration9* m_decl = nullptr;
  IDirect3DVertexShader9* m_vs2 = nullptr;
  IDirect3DVertexShader9* m_vs3 = nullptr;
  IDirect3DSurface9* m_sentinelSurf = nullptr;
  IDirect3DSurface9* m_sentinelSys = nullptr;
};

// ---------------------------------------------------------------------------------------------------------------
// Vulkan candidate
static uint32_t g_validationMessages = 0;
static std::map<std::string, std::string> g_jobMessages;  // VUID -> first message text, cleared per job

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT sev,
    VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
  if (sev >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    g_validationMessages++;
    std::string id = data->pMessageIdName ? data->pMessageIdName : "?";
    std::string msg = data->pMessage ? data->pMessage : "";
    // The layer appends a "reported N times" notice once a VUID hits its duplicate limit; keep the first text.
    g_jobMessages.emplace(id, msg);
  }
  return VK_FALSE;
}

class VkCand {
public:
  explicit VkCand(bool validation) {
    VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    app.apiVersion = VK_API_VERSION_1_3;
    app.pApplicationName = "shaderdiff";
    const char* layers[] = { "VK_LAYER_KHRONOS_validation" };
    const char* exts[] = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &app;
    if (validation) {
      ici.enabledLayerCount = 1; ici.ppEnabledLayerNames = layers;
      ici.enabledExtensionCount = 1; ici.ppEnabledExtensionNames = exts;
    }
    CHECK_VK(vkCreateInstance(&ici, nullptr, &m_instance));
    if (validation) {
      auto create = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
      VkDebugUtilsMessengerCreateInfoEXT mci = { VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT };
      mci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
      mci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
      mci.pfnUserCallback = debugCallback;
      CHECK_VK(create(m_instance, &mci, nullptr, &m_messenger));
    }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(m_instance, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(m_instance, &n, pds.data());
    for (auto pd : pds) {
      VkPhysicalDeviceProperties p;
      vkGetPhysicalDeviceProperties(pd, &p);
      if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU || !m_pd) { m_pd = pd; adapter = p.deviceName; }
    }
    if (!m_pd) throw std::runtime_error("no Vulkan device");

    checkRequirements();

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(m_pd, &qn, qf.data());
    for (uint32_t i = 0; i < qn; i++) if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { m_qfi = i; break; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = m_qfi; qci.queueCount = 1; qci.pQueuePriorities = &prio;

    // Optional: VK_EXT_non_seamless_cube_map for D3D9 cube filtering semantics.
    {
      VkPhysicalDeviceNonSeamlessCubeMapFeaturesEXT ns = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NON_SEAMLESS_CUBE_MAP_FEATURES_EXT };
      VkPhysicalDeviceFeatures2 q = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &ns };
      uint32_t en = 0;
      vkEnumerateDeviceExtensionProperties(m_pd, nullptr, &en, nullptr);
      std::vector<VkExtensionProperties> ext(en);
      vkEnumerateDeviceExtensionProperties(m_pd, nullptr, &en, ext.data());
      bool has = false, hasXfb = false;
      for (auto& e : ext) {
        has |= std::strcmp(e.extensionName, VK_EXT_NON_SEAMLESS_CUBE_MAP_EXTENSION_NAME) == 0;
        hasXfb |= std::strcmp(e.extensionName, VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME) == 0;
      }
      VkPhysicalDeviceTransformFeedbackFeaturesEXT xf = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT };
      ns.pNext = &xf;
      vkGetPhysicalDeviceFeatures2(m_pd, &q);
      if (has) m_nonSeamlessCube = ns.nonSeamlessCubeMap && !std::getenv("SHADERDIFF_SEAMLESS");
      // Transform feedback captures translated vertex-shader outputs for the VS differential test.
      if (hasXfb) m_xfb = xf.transformFeedback == VK_TRUE;
      requirementReport += m_nonSeamlessCube ? " | nonSeamlessCubeMap=on" : " | nonSeamlessCubeMap=off";
      requirementReport += m_xfb ? " | transformFeedback=on" : " | transformFeedback=off";
    }

    VkPhysicalDeviceTransformFeedbackFeaturesEXT fxf = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT };
    fxf.transformFeedback = VK_TRUE;
    VkPhysicalDeviceNonSeamlessCubeMapFeaturesEXT fns = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_NON_SEAMLESS_CUBE_MAP_FEATURES_EXT };
    fns.nonSeamlessCubeMap = VK_TRUE;
    void* chain = nullptr;
    std::vector<const char*> devExts;
    if (m_xfb) { fxf.pNext = chain; chain = &fxf; devExts.push_back(VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME); }
    if (m_nonSeamlessCube) { fns.pNext = chain; chain = &fns; devExts.push_back(VK_EXT_NON_SEAMLESS_CUBE_MAP_EXTENSION_NAME); }
    VkPhysicalDeviceVulkan13Features f13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, chain };
    f13.dynamicRendering = VK_TRUE; f13.synchronization2 = VK_TRUE; f13.shaderDemoteToHelperInvocation = VK_TRUE;
    VkPhysicalDeviceVulkan12Features f12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13 };
    f12.scalarBlockLayout = VK_TRUE; f12.vulkanMemoryModel = VK_TRUE; f12.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12 };
    f2.features.shaderClipDistance = VK_TRUE;
    f2.features.fragmentStoresAndAtomics = VK_TRUE;  // the skin buffer's stores (FNV_SKIN)

    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f2 };
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = uint32_t(devExts.size()); dci.ppEnabledExtensionNames = devExts.data();
    CHECK_VK(vkCreateDevice(m_pd, &dci, nullptr, &m_dev));
    vkGetDeviceQueue(m_dev, m_qfi, 0, &m_queue);
    if (m_xfb) {
      m_cmdBindXfb = (PFN_vkCmdBindTransformFeedbackBuffersEXT)vkGetDeviceProcAddr(m_dev, "vkCmdBindTransformFeedbackBuffersEXT");
      m_cmdBeginXfb = (PFN_vkCmdBeginTransformFeedbackEXT)vkGetDeviceProcAddr(m_dev, "vkCmdBeginTransformFeedbackEXT");
      m_cmdEndXfb = (PFN_vkCmdEndTransformFeedbackEXT)vkGetDeviceProcAddr(m_dev, "vkCmdEndTransformFeedbackEXT");
    }

    VkCommandPoolCreateInfo cpi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpi.queueFamilyIndex = m_qfi;
    CHECK_VK(vkCreateCommandPool(m_dev, &cpi, nullptr, &m_pool));
    VkCommandBufferAllocateInfo cai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cai.commandPool = m_pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    CHECK_VK(vkAllocateCommandBuffers(m_dev, &cai, &m_cmd));
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    CHECK_VK(vkCreateFence(m_dev, &fci, nullptr, &m_fence));

    m_rt = createImage(VK_FORMAT_R32G32B32A32_SFLOAT, W, H, 1,
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, false);
    m_rt1 = createImage(VK_FORMAT_R32G32B32A32_SFLOAT, W, H, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, false);
    m_readback = createBuffer(W * H * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    m_vbo = createBuffer(sizeof(Inputs::vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    m_staging = createBuffer(TEX * TEX * 4 * 6 * 16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

    // Legacy-state constant buffers (docs/SHADER_ABI.md), fixed for the test: alpha test ALWAYS, fog off,
    // no point sprites, sampler state per bound texture (filled per job).
    m_cbFloat = createBuffer(224 * 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    m_cbAlpha = createBuffer(16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    m_cbFog = createBuffer(48, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    m_cbPoint = createBuffer(16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    m_cbSampler = createBuffer(16 * 28, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    m_cbDummy = createBuffer(256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    m_cbSun = createBuffer(sizeof(SunSetup::block), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    m_cbEnv = createBuffer(160, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    m_cbAo = createBuffer(192, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    std::memset(m_cbAo.map, 0, 192);
    m_cbWet = createBuffer(256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    std::memset(m_cbWet.map, 0, 256);
    { uint32_t a[4] = { 7u, 0u, 0u, 0u }; std::memcpy(m_cbAlpha.map, a, sizeof(a)); }
    std::memset(m_cbFog.map, 0, 48);
    { struct { uint32_t sprite; float size, mn, mx; } p = { 0u, 1.0f, 1.0f, 1.0f }; std::memcpy(m_cbPoint.map, &p, sizeof(p)); }
    std::memset(m_cbDummy.map, 0, 256);

    VkSamplerCreateInfo si = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    si.magFilter = si.minFilter = VK_FILTER_LINEAR;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.maxLod = 0.0f;
    CHECK_VK(vkCreateSampler(m_dev, &si, nullptr, &m_sampler));
    // D3D9 cube maps: each face is addressed on its own with clamp-to-edge, no filtering across face edges
    // (the D3D9 address mode is ignored for cubes). Vulkan matches with CLAMP_TO_EDGE plus
    // VK_EXT_non_seamless_cube_map -- the same setup DXVK uses (d3d9_device.cpp: cube -> CLAMP_TO_EDGE,
    // setLegacyCubeFilter).
    si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (m_nonSeamlessCube) si.flags |= VK_SAMPLER_CREATE_NON_SEAMLESS_CUBE_MAP_BIT_EXT;
    CHECK_VK(vkCreateSampler(m_dev, &si, nullptr, &m_samplerCube));
    {
      // The runtime's cascade sampler (backend_shadow.inl): linear compare, outside the map lit.
      VkSamplerCreateInfo ci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
      ci.magFilter = ci.minFilter = VK_FILTER_LINEAR;
      ci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
      ci.addressModeU = ci.addressModeV = ci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
      ci.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
      ci.compareEnable = VK_TRUE;
      ci.compareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
      ci.maxLod = 0.0f;
      CHECK_VK(vkCreateSampler(m_dev, &ci, nullptr, &m_sunSampler));
    }

    VkShaderModuleCreateInfo smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smi.codeSize = sizeof(kPassthroughVertSpv); smi.pCode = kPassthroughVertSpv;
    CHECK_VK(vkCreateShaderModule(m_dev, &smi, nullptr, &m_vs));

    VkDescriptorSetLayoutCreateInfo empty = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    CHECK_VK(vkCreateDescriptorSetLayout(m_dev, &empty, nullptr, &m_emptySetLayout));

    VkDescriptorPoolSize sizes[] = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16 }, { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 26 },
                                     { VK_DESCRIPTOR_TYPE_SAMPLER, 24 },
                                     { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 } };
    VkDescriptorPoolCreateInfo dpi = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT; dpi.maxSets = 2; dpi.poolSizeCount = 4; dpi.pPoolSizes = sizes;
    CHECK_VK(vkCreateDescriptorPool(m_dev, &dpi, nullptr, &m_descPool));
  }

  ~VkCand() {
    if (!m_dev) return;
    vkDeviceWaitIdle(m_dev);
    vkDestroyDescriptorPool(m_dev, m_descPool, nullptr);
    vkDestroyDescriptorSetLayout(m_dev, m_emptySetLayout, nullptr);
    vkDestroyShaderModule(m_dev, m_vs, nullptr);
    vkDestroySampler(m_dev, m_sampler, nullptr);
    vkDestroySampler(m_dev, m_samplerCube, nullptr);
    vkDestroySampler(m_dev, m_sunSampler, nullptr);
    for (auto* b : { &m_readback, &m_vbo, &m_staging, &m_cbFloat, &m_cbAlpha, &m_cbFog, &m_cbPoint, &m_cbSampler, &m_cbDummy,
                     &m_cbSun, &m_cbEnv, &m_cbAo, &m_cbWet })
      destroyBuffer(*b);
    destroyImage(m_rt);
    destroyImage(m_rt1);
    vkDestroyFence(m_dev, m_fence, nullptr);
    vkDestroyCommandPool(m_dev, m_pool, nullptr);
    vkDestroyDevice(m_dev, nullptr);
    if (m_messenger) {
      auto destroy = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT");
      destroy(m_instance, m_messenger, nullptr);
    }
    vkDestroyInstance(m_instance, nullptr);
  }

  std::vector<float> run(const Job& job, const std::vector<uint8_t>& spv, const Inputs& in, const SunSetup& sun = SunSetup()) {
    std::vector<Image> textures;
    Image sunMap;
    // The cascades: the sun shadow test's (a sun light colour) or, for the wet world's rain cover, the same placeholder.
    const bool cascadesLit = job.sun >= 0 || std::find(job.cbvs.begin(), job.cbvs.end(), 12u) != job.cbvs.end();
    if (cascadesLit) {
      sunMap = createDepthArray(kSunMap, 2, sun.depth);
      std::memcpy(m_cbSun.map, sun.block, sizeof(sun.block));
    }
    // The environment (replacements with FNV_ENV, binding 10 with the cube 50 and sampler 51; docs/SHADER_ABI.md
    // "Environment"): a mid-grey cube, and the block from SHADERDIFF_ENV: unset, not valid (the shader keeps its flat
    // ambient); "uniform", valid with the harmonics of a uniform environment (irradiance ratio 1 everywhere); "sky", valid
    // with 1 + z (2 up, 0 down).
    const bool envLit = std::find(job.cbvs.begin(), job.cbvs.end(), 10u) != job.cbvs.end();
    Image envCube;
    if (envLit) {
      envCube = createImage(VK_FORMAT_B8G8R8A8_UNORM, TEX, TEX, 6, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, true);
      std::vector<uint32_t> grey(size_t(TEX) * TEX * 6, 0xFF808080u);
      std::memcpy(m_staging.map, grey.data(), grey.size() * 4);
      uploadImage(envCube, 6);
      float blk[10][4] = {};
      const char* mode = std::getenv("SHADERDIFF_ENV");
      if (mode) {
        blk[0][0] = 1.0f;
        for (int c = 0; c < 3; c++) blk[1][c] = 1.0f / 0.282095f;
        if (std::string(mode) == "sky")
          for (int c = 0; c < 3; c++) blk[3][c] = 1.0f / 0.488603f;  // Y10 (z)
      }
      std::memcpy(m_cbEnv.map, blk, sizeof(blk));
    }
    // Ambient occlusion (replacements with FNV_AO, block 11 with the image 52 and sampler 53): the block 0 (no
    // occlusion; the backend test checks the reprojection), a grey image.
    const bool aoLit = std::find(job.cbvs.begin(), job.cbvs.end(), 11u) != job.cbvs.end();
    // Wet world (replacements with FNV_WET, block 12 with the puddle map 56 and its sampler 57): the block zero, so the
    // shader sees no wet state (docs/WETWORLD.md); the map a grey placeholder.
    const bool wetLit = std::find(job.cbvs.begin(), job.cbvs.end(), 12u) != job.cbvs.end();
    Image wetImage{};
    if (wetLit) {
      wetImage = createImage(VK_FORMAT_B8G8R8A8_UNORM, TEX, TEX, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, false);
      std::vector<uint32_t> grey(size_t(TEX) * TEX, 0xFF808080u);
      std::memcpy(m_staging.map, grey.data(), grey.size() * 4);
      uploadImage(wetImage, 1);
    }
    // Skin (replacements with FNV_SKIN: the detail normal map 59 and its sampler 60): a zero image, which the shader
    // reads as no detail map (docs/SKIN.md).
    const bool skinLit = std::find(job.cbvs.begin(), job.cbvs.end(), 59u) != job.cbvs.end();
    Image skinImage{};
    if (skinLit) {
      skinImage = createImage(VK_FORMAT_B8G8R8A8_UNORM, TEX, TEX, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, false);
      std::vector<uint32_t> zero(size_t(TEX) * TEX, 0u);
      std::memcpy(m_staging.map, zero.data(), zero.size() * 4);
      uploadImage(skinImage, 1);
    }
    // Hair (replacements with FNV_HAIR: the tangent shift map 63 and its sampler 64): a zero image, which the shader
    // reads as no shift (docs/LIGHTING.md "Hair").
    const bool hairLit = std::find(job.cbvs.begin(), job.cbvs.end(), 63u) != job.cbvs.end();
    Image hairImage{};
    if (hairLit) {
      hairImage = createImage(VK_FORMAT_B8G8R8A8_UNORM, TEX, TEX, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, false);
      std::vector<uint32_t> zero(size_t(TEX) * TEX, 0u);
      std::memcpy(m_staging.map, zero.data(), zero.size() * 4);
      uploadImage(hairImage, 1);
    }
    // The skin buffer (FNV_SKIN: block 13 with the storage images 61 / 62): the block zero, so nothing is written; the
    // images real storage images of the shader's formats.
    const bool skinStore = std::find(job.cbvs.begin(), job.cbvs.end(), 13u) != job.cbvs.end();
    Image skinAlbedoImg{}, skinDepthImg{};
    if (skinStore) {
      skinAlbedoImg = createImage(VK_FORMAT_R8G8B8A8_UNORM, TEX, TEX, 1, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, false);
      skinDepthImg = createImage(VK_FORMAT_R32_UINT, TEX, TEX, 1, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, false);
      std::memset(m_staging.map, 0xFF, size_t(TEX) * TEX * 4);
      uploadStorageImage(skinAlbedoImg);
      uploadStorageImage(skinDepthImg);
    }
    Image aoImage;
    if (aoLit) {
      aoImage = createImage(VK_FORMAT_B8G8R8A8_UNORM, TEX, TEX, 1, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, false);
      std::vector<uint32_t> grey(size_t(TEX) * TEX, 0xFF808080u);
      std::memcpy(m_staging.map, grey.data(), grey.size() * 4);
      uploadImage(aoImage, 1);
    }
    std::map<uint32_t, size_t> texByReg;
    for (const auto& [reg, texels] : in.texels) {
      bool cube = texels.size() == size_t(TEX) * TEX * 6;
      textures.push_back(createImage(VK_FORMAT_B8G8R8A8_UNORM, TEX, TEX, cube ? 6 : 1,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, cube));
      texByReg[reg] = textures.size() - 1;
      std::memcpy(m_staging.map, texels.data(), texels.size() * 4);
      uploadImage(textures.back(), cube ? 6 : 1);
    }

    {
      // Runtime ABI: shader-defined constants overlay the API constants (see applyDefOverlay).
      std::vector<float> c(in.constants.begin(), in.constants.end());
      applyDefOverlay(job.defs, c);
      poisonUnread(reinterpret_cast<const uint32_t*>(spv.data()), spv.size() / 4, 1, c, job.defs);
      std::memcpy(m_cbFloat.map, c.data(), c.size() * 4);
    }
    std::memcpy(m_vbo.map, in.vertices.data(), sizeof(in.vertices));
    {
      struct S { uint32_t type, dref, proj, isNull, gather, drefClamp; float drefScale; };
      static_assert(sizeof(S) == 28);
      S s[16] = { };
      for (auto& e : s) { e.type = 2u; e.isNull = 1u; e.drefScale = 1.0f; }
      for (auto [reg, kind] : job.textures) { s[reg].type = kind == 9 ? 3u : 2u; s[reg].isNull = 0u; }
      std::memcpy(m_cbSampler.map, s, sizeof(s));
    }

    // Descriptor set 1 (pixel stage) from the shader's reflected bindings.
    std::vector<VkDescriptorSetLayoutBinding> lb;
    for (uint32_t b : job.cbvs)  // the blocks (59 and 63 in the list are the skin's detail map and the hair's shift map, below)
      if (b < 16) lb.push_back({ b, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
    for (auto [reg, kind] : job.textures) {
      lb.push_back({ 16 + reg, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
      lb.push_back({ 32 + reg, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
    }
    if (cascadesLit) {
      lb.push_back({ 48, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
      lb.push_back({ 49, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
    }
    if (envLit) {
      lb.push_back({ 50, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
      lb.push_back({ 51, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
    }
    if (aoLit) {
      lb.push_back({ 52, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
      lb.push_back({ 53, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
      lb.push_back({ 54, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
      lb.push_back({ 55, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
    }
    if (wetLit) {
      lb.push_back({ 56, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
      lb.push_back({ 57, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
      lb.push_back({ 58, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
    }
    if (skinStore) {
      lb.push_back({ 61, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
      lb.push_back({ 62, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
    }
    if (skinLit) {
      lb.push_back({ 59, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
      lb.push_back({ 60, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
    }
    if (hairLit) {
      lb.push_back({ 63, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
      lb.push_back({ 64, VK_DESCRIPTOR_TYPE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr });
    }
    VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dli.bindingCount = uint32_t(lb.size()); dli.pBindings = lb.data();
    VkDescriptorSetLayout psLayout;
    CHECK_VK(vkCreateDescriptorSetLayout(m_dev, &dli, nullptr, &psLayout));

    VkDescriptorSetLayout layouts[2] = { m_emptySetLayout, psLayout };
    VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.setLayoutCount = 2; pli.pSetLayouts = layouts;
    VkPipelineLayout pipeLayout;
    CHECK_VK(vkCreatePipelineLayout(m_dev, &pli, nullptr, &pipeLayout));

    VkDescriptorSetAllocateInfo dai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = m_descPool; dai.descriptorSetCount = 1; dai.pSetLayouts = &psLayout;
    VkDescriptorSet set;
    CHECK_VK(vkAllocateDescriptorSets(m_dev, &dai, &set));

    std::vector<VkDescriptorBufferInfo> bufInfos;
    std::vector<VkDescriptorImageInfo> imgInfos;
    bufInfos.reserve(16); imgInfos.reserve(64);
    std::vector<VkWriteDescriptorSet> writes;
    for (uint32_t b : job.cbvs) {
      if (b >= 16) continue;  // not a block: the skin's detail map
      Buffer* buf = b == 0 ? &m_cbFloat : b == 3 ? &m_cbAlpha : b == 4 ? &m_cbFog : b == 6 ? &m_cbPoint
                  : b == 7 ? &m_cbSampler : b == 9 ? &m_cbSun : b == 10 ? &m_cbEnv : b == 11 ? &m_cbAo
                  : b == 12 || b == 13 ? &m_cbWet : &m_cbDummy;  // 13, the skin buffer's block: zero (no writes)
      bufInfos.push_back({ buf->buf, 0, VK_WHOLE_SIZE });
      VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      w.dstSet = set; w.dstBinding = b; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      w.pBufferInfo = &bufInfos.back();
      writes.push_back(w);
    }
    for (auto [reg, kind] : job.textures) {
      imgInfos.push_back({ VK_NULL_HANDLE, textures[texByReg[reg]].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL });
      VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      w.dstSet = set; w.dstBinding = 16 + reg; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
      imgInfos.push_back({ kind == 9 ? m_samplerCube : m_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED });
      w.dstBinding = 32 + reg; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
    }
    if (cascadesLit) {
      imgInfos.push_back({ VK_NULL_HANDLE, sunMap.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL });
      VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      w.dstSet = set; w.dstBinding = 48; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
      imgInfos.push_back({ m_sunSampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED });
      w.dstBinding = 49; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
    }
    if (aoLit) {
      imgInfos.push_back({ VK_NULL_HANDLE, aoImage.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL });
      VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      w.dstSet = set; w.dstBinding = 52; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
      imgInfos.push_back({ m_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED });
      w.dstBinding = 53; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
      imgInfos.push_back({ VK_NULL_HANDLE, aoImage.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL });
      w.dstBinding = 54; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
      imgInfos.push_back({ VK_NULL_HANDLE, aoImage.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL });
      w.dstBinding = 55; w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
    }
    if (wetLit) {
      imgInfos.push_back({ VK_NULL_HANDLE, wetImage.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL });
      VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      w.dstSet = set; w.dstBinding = 56; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
      imgInfos.push_back({ m_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED });
      w.dstBinding = 57; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
      imgInfos.push_back({ VK_NULL_HANDLE, wetImage.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL });
      w.dstBinding = 58; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE; w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
    }
    if (skinStore) {
      imgInfos.push_back({ VK_NULL_HANDLE, skinAlbedoImg.view, VK_IMAGE_LAYOUT_GENERAL });
      VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      w.dstSet = set; w.dstBinding = 61; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
      imgInfos.push_back({ VK_NULL_HANDLE, skinDepthImg.view, VK_IMAGE_LAYOUT_GENERAL });
      w.dstBinding = 62; w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
    }
    if (skinLit) {
      imgInfos.push_back({ VK_NULL_HANDLE, skinImage.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL });
      VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      w.dstSet = set; w.dstBinding = 59; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
      imgInfos.push_back({ m_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED });
      w.dstBinding = 60; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
    }
    if (hairLit) {
      imgInfos.push_back({ VK_NULL_HANDLE, hairImage.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL });
      VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      w.dstSet = set; w.dstBinding = 63; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
      imgInfos.push_back({ m_sampler, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED });
      w.dstBinding = 64; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
    }
    if (envLit) {
      imgInfos.push_back({ VK_NULL_HANDLE, envCube.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL });
      VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      w.dstSet = set; w.dstBinding = 50; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
      imgInfos.push_back({ m_samplerCube, VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED });
      w.dstBinding = 51; w.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER; w.pImageInfo = &imgInfos.back();
      writes.push_back(w);
    }
    vkUpdateDescriptorSets(m_dev, uint32_t(writes.size()), writes.data(), 0, nullptr);

    VkShaderModuleCreateInfo smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smi.codeSize = spv.size(); smi.pCode = reinterpret_cast<const uint32_t*>(spv.data());
    VkShaderModule fs;
    CHECK_VK(vkCreateShaderModule(m_dev, &smi, nullptr, &fs));
    VkPipeline pipe = createPipeline(pipeLayout, fs);

    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK(vkBeginCommandBuffer(m_cmd, &bi));
    barrier(m_rt.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    barrier(m_rt1.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    // Attachment 1 exists so MRT shaders (oC1) have a target, as the game provides; only oC0 is compared.
    VkRenderingAttachmentInfo ca[2] = { { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO },
                                        { VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO } };
    for (int i = 0; i < 2; i++) {
      ca[i].imageView = i ? m_rt1.view : m_rt.view; ca[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      ca[i].clearValue.color = { { kSentinel, kSentinel, kSentinel, kSentinel } };
    }
    VkRenderingInfo ri = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea = { { 0, 0 }, { W, H } }; ri.layerCount = 1; ri.colorAttachmentCount = 2; ri.pColorAttachments = ca;
    vkCmdBeginRendering(m_cmd, &ri);
    vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(m_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 1, 1, &set, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindVertexBuffers(m_cmd, 0, 1, &m_vbo.buf, &off);
    VkViewport vp = { 0.0f, float(H), float(W), -float(H), 0.0f, 1.0f };  // D3D9 orientation
    VkRect2D sc = { { 0, 0 }, { W, H } };
    vkCmdSetViewport(m_cmd, 0, 1, &vp);
    vkCmdSetScissor(m_cmd, 0, 1, &sc);
    vkCmdDraw(m_cmd, 4, 1, 0, 0);
    vkCmdEndRendering(m_cmd);
    barrier(m_rt.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy copy = { };
    copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.imageExtent = { W, H, 1 };
    vkCmdCopyImageToBuffer(m_cmd, m_rt.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_readback.buf, 1, &copy);
    CHECK_VK(vkEndCommandBuffer(m_cmd));
    submitAndWait();

    std::vector<float> out(W * H * 4);
    std::memcpy(out.data(), m_readback.map, out.size() * 4);

    vkDestroyPipeline(m_dev, pipe, nullptr);
    vkDestroyShaderModule(m_dev, fs, nullptr);
    vkFreeDescriptorSets(m_dev, m_descPool, 1, &set);
    vkDestroyPipelineLayout(m_dev, pipeLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_dev, psLayout, nullptr);
    for (auto& t : textures) destroyImage(t);
    if (envLit) destroyImage(envCube);
    if (aoLit) destroyImage(aoImage);
    if (wetLit) destroyImage(wetImage);
    if (skinLit) destroyImage(skinImage);
    if (hairLit) destroyImage(hairImage);
    if (skinStore) { destroyImage(skinAlbedoImg); destroyImage(skinDepthImg); }
    destroyImage(sunMap);
    return out;
  }

  std::string adapter;
  std::string requirementReport;

private:
  struct Buffer { VkBuffer buf = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; void* map = nullptr; };
  struct Image { VkImage image = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; };

  void checkRequirements() {
    VkPhysicalDeviceVulkan13Features f13 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    VkPhysicalDeviceVulkan12Features f12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES, &f13 };
    VkPhysicalDeviceFeatures2 f2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &f12 };
    vkGetPhysicalDeviceFeatures2(m_pd, &f2);
    VkPhysicalDeviceVulkan12Properties p12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES };
    VkPhysicalDeviceProperties2 p2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &p12 };
    vkGetPhysicalDeviceProperties2(m_pd, &p2);

    struct Req { const char* name; bool ok; } reqs[] = {
      { "apiVersion>=1.3", p2.properties.apiVersion >= VK_API_VERSION_1_3 },
      { "scalarBlockLayout", bool(f12.scalarBlockLayout) },
      { "vulkanMemoryModel", bool(f12.vulkanMemoryModel) },
      { "bufferDeviceAddress", bool(f12.bufferDeviceAddress) },
      { "shaderDemoteToHelperInvocation", bool(f13.shaderDemoteToHelperInvocation) },
      { "dynamicRendering", bool(f13.dynamicRendering) },
      { "shaderClipDistance", bool(f2.features.shaderClipDistance) },
    };
    // Float-control execution modes depend on the profile the cache was built for (manifest "fpCaps"); a module
    // declaring a mode this device lacks is reported per shader by the validation layer.
    std::stringstream s;
    bool all = true;
    for (auto& r : reqs) { s << r.name << "=" << (r.ok ? "yes" : "NO") << " "; all &= r.ok; }
    s << "| fp32: rte=" << bool(p12.shaderRoundingModeRTEFloat32) << " rtz=" << bool(p12.shaderRoundingModeRTZFloat32)
      << " ftz=" << bool(p12.shaderDenormFlushToZeroFloat32) << " dp=" << bool(p12.shaderDenormPreserveFloat32);
    requirementReport = s.str();
    if (!all) throw std::runtime_error("device lacks requirements: " + requirementReport);
  }

  uint32_t memType(uint32_t bits, VkMemoryPropertyFlags flags) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(m_pd, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & flags) == flags) return i;
    throw std::runtime_error("no memory type");
  }

  Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage) {
    Buffer b;
    VkBufferCreateInfo ci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    ci.size = size; ci.usage = usage;
    CHECK_VK(vkCreateBuffer(m_dev, &ci, nullptr, &b.buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(m_dev, b.buf, &mr);
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = mr.size;
    ai.memoryTypeIndex = memType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    CHECK_VK(vkAllocateMemory(m_dev, &ai, nullptr, &b.mem));
    CHECK_VK(vkBindBufferMemory(m_dev, b.buf, b.mem, 0));
    CHECK_VK(vkMapMemory(m_dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
    return b;
  }

  void destroyBuffer(Buffer& b) {
    if (b.buf) vkDestroyBuffer(m_dev, b.buf, nullptr);
    if (b.mem) vkFreeMemory(m_dev, b.mem, nullptr);
    b = { };
  }

  Image createImage(VkFormat fmt, uint32_t w, uint32_t h, uint32_t layers, VkImageUsageFlags usage, bool cube) {
    Image im;
    VkImageCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    ci.imageType = VK_IMAGE_TYPE_2D; ci.format = fmt; ci.extent = { w, h, 1 }; ci.mipLevels = 1; ci.arrayLayers = layers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT; ci.tiling = VK_IMAGE_TILING_OPTIMAL; ci.usage = usage;
    CHECK_VK(vkCreateImage(m_dev, &ci, nullptr, &im.image));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(m_dev, im.image, &mr);
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = mr.size; ai.memoryTypeIndex = memType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK_VK(vkAllocateMemory(m_dev, &ai, nullptr, &im.mem));
    CHECK_VK(vkBindImageMemory(m_dev, im.image, im.mem, 0));
    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = im.image; vi.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D; vi.format = fmt;
    vi.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers };
    CHECK_VK(vkCreateImageView(m_dev, &vi, nullptr, &im.view));
    return im;
  }

  // A D32 2D array of `layers` size x size layers holding `depth` (layer-major), ready to sample.
  Image createDepthArray(uint32_t size, uint32_t layers, const std::vector<float>& depth) {
    Image im;
    VkImageCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D; ci.format = VK_FORMAT_D32_SFLOAT; ci.extent = { size, size, 1 }; ci.mipLevels = 1;
    ci.arrayLayers = layers; ci.samples = VK_SAMPLE_COUNT_1_BIT; ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    CHECK_VK(vkCreateImage(m_dev, &ci, nullptr, &im.image));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(m_dev, im.image, &mr);
    VkMemoryAllocateInfo ai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    ai.allocationSize = mr.size; ai.memoryTypeIndex = memType(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK_VK(vkAllocateMemory(m_dev, &ai, nullptr, &im.mem));
    CHECK_VK(vkBindImageMemory(m_dev, im.image, im.mem, 0));
    VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vi.image = im.image; vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY; vi.format = ci.format;
    vi.subresourceRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, layers };
    CHECK_VK(vkCreateImageView(m_dev, &vi, nullptr, &im.view));

    if (depth.size() * 4 > TEX * TEX * 4 * 6 * 16) throw std::runtime_error("cascade data larger than the staging buffer");
    std::memcpy(m_staging.map, depth.data(), depth.size() * 4);
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK(vkBeginCommandBuffer(m_cmd, &bi));
    VkImageMemoryBarrier2 b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    b.srcStageMask = VK_PIPELINE_STAGE_2_NONE; b.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    b.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.image = im.image;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange = vi.subresourceRange;
    VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(m_cmd, &di);
    VkBufferImageCopy c = { };
    c.imageSubresource = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, layers };
    c.imageExtent = { size, size, 1 };
    vkCmdCopyBufferToImage(m_cmd, m_staging.buf, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    b.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT; b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    b.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT; b.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    vkCmdPipelineBarrier2(m_cmd, &di);
    CHECK_VK(vkEndCommandBuffer(m_cmd));
    submitAndWait();
    return im;
  }

  void destroyImage(Image& im) {
    if (im.view) vkDestroyImageView(m_dev, im.view, nullptr);
    if (im.image) vkDestroyImage(m_dev, im.image, nullptr);
    if (im.mem) vkFreeMemory(m_dev, im.mem, nullptr);
    im = { };
  }

  void barrier(VkImage image, VkImageLayout from, VkImageLayout to, uint32_t layers,
               VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess, VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {
    VkImageMemoryBarrier2 b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    b.srcStageMask = srcStage; b.srcAccessMask = srcAccess; b.dstStageMask = dstStage; b.dstAccessMask = dstAccess;
    b.oldLayout = from; b.newLayout = to; b.image = image;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers };
    VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    di.imageMemoryBarrierCount = 1; di.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(m_cmd, &di);
  }

  void submitAndWait() {
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1; si.pCommandBuffers = &m_cmd;
    CHECK_VK(vkQueueSubmit(m_queue, 1, &si, m_fence));
    CHECK_VK(vkWaitForFences(m_dev, 1, &m_fence, VK_TRUE, UINT64_MAX));
    CHECK_VK(vkResetFences(m_dev, 1, &m_fence));
    CHECK_VK(vkResetCommandBuffer(m_cmd, 0));
  }

  // An image's whole content from the staging buffer, left in GENERAL (storage images).
  void uploadStorageImage(Image& im) {
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK(vkBeginCommandBuffer(m_cmd, &bi));
    barrier(im.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
            VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy c = { };
    c.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    c.imageExtent = { TEX, TEX, 1 };
    vkCmdCopyBufferToImage(m_cmd, m_staging.buf, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    barrier(im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, 1,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    CHECK_VK(vkEndCommandBuffer(m_cmd));
    submitAndWait();
  }
  void uploadImage(Image& im, uint32_t layers) {
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK(vkBeginCommandBuffer(m_cmd, &bi));
    barrier(im.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, layers,
            VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkBufferImageCopy c = { };
    c.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers };
    c.imageExtent = { TEX, TEX, 1 };
    vkCmdCopyBufferToImage(m_cmd, m_staging.buf, im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
    barrier(im.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, layers,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    CHECK_VK(vkEndCommandBuffer(m_cmd));
    submitAndWait();
  }

  VkPipeline createPipeline(VkPipelineLayout layout, VkShaderModule fs) {
    VkPipelineShaderStageCreateInfo st[2] = { { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO },
                                               { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO } };
    st[0].stage = VK_SHADER_STAGE_VERTEX_BIT; st[0].module = m_vs; st[0].pName = "main";
    st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; st[1].module = fs; st[1].pName = "main";

    VkVertexInputBindingDescription vb = { 0, 44 * sizeof(float), VK_VERTEX_INPUT_RATE_VERTEX };
    VkVertexInputAttributeDescription va[11];
    for (uint32_t i = 0; i < 11; i++) va[i] = { i, 0, VK_FORMAT_R32G32B32A32_SFLOAT, i * 16 };
    VkPipelineVertexInputStateCreateInfo vi = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vb;
    vi.vertexAttributeDescriptionCount = 11; vi.pVertexAttributeDescriptions = va;
    VkPipelineInputAssemblyStateCreateInfo ia = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    VkPipelineViewportStateCreateInfo vps = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vps.viewportCount = 1; vps.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_CLOCKWISE; rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds = { VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
    VkPipelineColorBlendAttachmentState cba[2] = { };
    cba[0].colorWriteMask = cba[1].colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 2; cb.pAttachments = cba;
    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dy.dynamicStateCount = 2; dy.pDynamicStates = dyn;
    VkFormat fmt[2] = { VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT };
    VkPipelineRenderingCreateInfo rci = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    rci.colorAttachmentCount = 2; rci.pColorAttachmentFormats = fmt;

    VkGraphicsPipelineCreateInfo gp = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci };
    gp.stageCount = 2; gp.pStages = st; gp.pVertexInputState = &vi; gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vps; gp.pRasterizationState = &rs; gp.pMultisampleState = &ms; gp.pDepthStencilState = &ds;
    gp.pColorBlendState = &cb; gp.pDynamicState = &dy; gp.layout = layout;
    VkPipeline p;
    CHECK_VK(vkCreateGraphicsPipelines(m_dev, VK_NULL_HANDLE, 1, &gp, nullptr, &p));
    return p;
  }

  VkInstance m_instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT m_messenger = VK_NULL_HANDLE;
  VkPhysicalDevice m_pd = VK_NULL_HANDLE;
  VkDevice m_dev = VK_NULL_HANDLE;
  uint32_t m_qfi = 0;
  VkQueue m_queue = VK_NULL_HANDLE;
  VkCommandPool m_pool = VK_NULL_HANDLE;
  VkCommandBuffer m_cmd = VK_NULL_HANDLE;
  VkFence m_fence = VK_NULL_HANDLE;
  Image m_rt, m_rt1;
  Buffer m_readback, m_vbo, m_staging, m_cbFloat, m_cbAlpha, m_cbFog, m_cbPoint, m_cbSampler, m_cbDummy, m_cbSun, m_cbEnv, m_cbAo, m_cbWet;
  VkSampler m_sampler = VK_NULL_HANDLE;
  VkSampler m_sunSampler = VK_NULL_HANDLE;
  VkSampler m_samplerCube = VK_NULL_HANDLE;
  VkShaderModule m_vs = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_emptySetLayout = VK_NULL_HANDLE;
  VkDescriptorPool m_descPool = VK_NULL_HANDLE;
  bool m_nonSeamlessCube = false;
  bool m_xfb = false;
  PFN_vkCmdBindTransformFeedbackBuffersEXT m_cmdBindXfb = nullptr;
  PFN_vkCmdBeginTransformFeedbackEXT m_cmdBeginXfb = nullptr;
  PFN_vkCmdEndTransformFeedbackEXT m_cmdEndXfb = nullptr;

public:
  // ------------------------------------------------------------------------------------------------------------
  // Vertex-shader test: run the translated VS over `count` vertices with transform feedback capturing
  // Position (offset 0) and locations 0..15 (offset 16*(L+1); 12: sdp2spv's view vector, 13..15: its tangent-to-world
  // frame), stride kXfbStride.
  static constexpr uint32_t kXfbStride = 16 * 17;

  std::vector<float> runVs(const std::vector<uint32_t>& spvIn, const std::vector<uint32_t>& cbvs,
                           const std::vector<std::pair<uint32_t, uint32_t>>& attribs,  // (location, float4 slot)
                           const std::vector<float>& vertexData, uint32_t vertexStride, uint32_t count,
                           const std::vector<float>& constants) {
    if (!m_xfb) throw std::runtime_error("transform feedback unavailable");
    std::vector<uint32_t> spv = patchXfb(spvIn);

    Buffer vbo = createBuffer(vertexData.size() * 4, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    std::memcpy(vbo.map, vertexData.data(), vertexData.size() * 4);
    Buffer xfb = createBuffer(VkDeviceSize(count) * kXfbStride, VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT);
    std::memset(xfb.map, 0, size_t(count) * kXfbStride);
    Buffer cbFloat = createBuffer(256 * 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    std::memcpy(cbFloat.map, constants.data(), 256 * 16);
    Buffer cbClip = createBuffer(128, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    std::memset(cbClip.map, 0, 128);  // plane count 0
    Buffer cbPoint = createBuffer(16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    { struct { uint32_t sprite; float size, mn, mx; } p = { 0u, 1.0f, 1.0f, 1.0f }; std::memcpy(cbPoint.map, &p, sizeof(p)); }

    std::vector<VkDescriptorSetLayoutBinding> lb;
    for (uint32_t b : cbvs) lb.push_back({ b, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr });
    VkDescriptorSetLayoutCreateInfo dli = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dli.bindingCount = uint32_t(lb.size()); dli.pBindings = lb.data();
    VkDescriptorSetLayout vsLayout;
    CHECK_VK(vkCreateDescriptorSetLayout(m_dev, &dli, nullptr, &vsLayout));
    VkDescriptorSetLayout layouts[2] = { vsLayout, m_emptySetLayout };
    VkPipelineLayoutCreateInfo pli = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pli.setLayoutCount = 2; pli.pSetLayouts = layouts;
    VkPipelineLayout pipeLayout;
    CHECK_VK(vkCreatePipelineLayout(m_dev, &pli, nullptr, &pipeLayout));
    VkDescriptorSetAllocateInfo dai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = m_descPool; dai.descriptorSetCount = 1; dai.pSetLayouts = &vsLayout;
    VkDescriptorSet set;
    CHECK_VK(vkAllocateDescriptorSets(m_dev, &dai, &set));
    std::vector<VkDescriptorBufferInfo> bi; bi.reserve(8);
    std::vector<VkWriteDescriptorSet> writes;
    for (uint32_t b : cbvs) {
      Buffer* buf = b == 0 ? &cbFloat : b == 5 ? &cbClip : b == 6 ? &cbPoint : &m_cbDummy;
      bi.push_back({ buf->buf, 0, VK_WHOLE_SIZE });
      VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
      w.dstSet = set; w.dstBinding = b; w.descriptorCount = 1; w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      w.pBufferInfo = &bi.back();
      writes.push_back(w);
    }
    vkUpdateDescriptorSets(m_dev, uint32_t(writes.size()), writes.data(), 0, nullptr);

    VkShaderModuleCreateInfo smi = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smi.codeSize = spv.size() * 4; smi.pCode = spv.data();
    VkShaderModule vs;
    CHECK_VK(vkCreateShaderModule(m_dev, &smi, nullptr, &vs));

    VkPipelineShaderStageCreateInfo st = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    st.stage = VK_SHADER_STAGE_VERTEX_BIT; st.module = vs; st.pName = "main";
    VkVertexInputBindingDescription vb = { 0, vertexStride, VK_VERTEX_INPUT_RATE_VERTEX };
    std::vector<VkVertexInputAttributeDescription> va;
    for (auto [loc, slot] : attribs) va.push_back({ loc, 0, VK_FORMAT_R32G32B32A32_SFLOAT, slot * 16 });
    VkPipelineVertexInputStateCreateInfo vi = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    vi.vertexBindingDescriptionCount = 1; vi.pVertexBindingDescriptions = &vb;
    vi.vertexAttributeDescriptionCount = uint32_t(va.size()); vi.pVertexAttributeDescriptions = va.data();
    VkPipelineInputAssemblyStateCreateInfo ia = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    VkPipelineViewportStateCreateInfo vps = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vps.viewportCount = 1; vps.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.rasterizerDiscardEnable = VK_TRUE; rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkDynamicState dyn[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo dy = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    dy.dynamicStateCount = 2; dy.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo rci = { VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO };
    VkGraphicsPipelineCreateInfo gp = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, &rci };
    gp.stageCount = 1; gp.pStages = &st; gp.pVertexInputState = &vi; gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vps; gp.pRasterizationState = &rs; gp.pMultisampleState = &ms; gp.pDynamicState = &dy;
    gp.layout = pipeLayout;
    VkPipeline pipe;
    CHECK_VK(vkCreateGraphicsPipelines(m_dev, VK_NULL_HANDLE, 1, &gp, nullptr, &pipe));

    VkCommandBufferBeginInfo cbi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    cbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK(vkBeginCommandBuffer(m_cmd, &cbi));
    VkRenderingInfo ri = { VK_STRUCTURE_TYPE_RENDERING_INFO };
    ri.renderArea = { { 0, 0 }, { W, H } }; ri.layerCount = 1;
    vkCmdBeginRendering(m_cmd, &ri);
    vkCmdBindPipeline(m_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(m_cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 0, 1, &set, 0, nullptr);
    VkDeviceSize zero = 0, xfbSize = VkDeviceSize(count) * kXfbStride;
    vkCmdBindVertexBuffers(m_cmd, 0, 1, &vbo.buf, &zero);
    VkViewport vp = { 0.0f, 0.0f, float(W), float(H), 0.0f, 1.0f };
    VkRect2D sc = { { 0, 0 }, { W, H } };
    vkCmdSetViewport(m_cmd, 0, 1, &vp);
    vkCmdSetScissor(m_cmd, 0, 1, &sc);
    m_cmdBindXfb(m_cmd, 0, 1, &xfb.buf, &zero, &xfbSize);
    m_cmdBeginXfb(m_cmd, 0, 0, nullptr, nullptr);
    vkCmdDraw(m_cmd, count, 1, 0, 0);
    m_cmdEndXfb(m_cmd, 0, 0, nullptr, nullptr);
    vkCmdEndRendering(m_cmd);
    VkMemoryBarrier2 mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    mb.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFORM_FEEDBACK_WRITE_BIT_EXT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT; mb.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo di = { VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    di.memoryBarrierCount = 1; di.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(m_cmd, &di);
    CHECK_VK(vkEndCommandBuffer(m_cmd));
    submitAndWait();

    std::vector<float> out(size_t(count) * kXfbStride / 4);
    std::memcpy(out.data(), xfb.map, out.size() * 4);

    vkDestroyPipeline(m_dev, pipe, nullptr);
    vkDestroyShaderModule(m_dev, vs, nullptr);
    vkFreeDescriptorSets(m_dev, m_descPool, 1, &set);
    vkDestroyPipelineLayout(m_dev, pipeLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_dev, vsLayout, nullptr);
    for (auto* b : { &vbo, &xfb, &cbFloat, &cbClip, &cbPoint }) destroyBuffer(*b);
    return out;
  }

private:
  // Adds transform-feedback decorations to a vertex shader at the SPIR-V word level:
  //   OpCapability TransformFeedback, OpExecutionMode Xfb, and XfbBuffer/XfbStride/Offset on the Position
  //   built-in and every Location 0..11 output.
  static std::vector<uint32_t> patchXfb(const std::vector<uint32_t>& in) {
    enum : uint32_t { OpCapability = 17, OpEntryPoint = 15, OpExecutionMode = 16, OpDecorate = 71, OpVariable = 59 };
    enum : uint32_t { CapTransformFeedback = 53, ModeXfb = 11, DecBuiltIn = 11, DecLocation = 30, DecOffset = 35,
                      DecXfbBuffer = 36, DecXfbStride = 37, BuiltInPosition = 0, StorageOutput = 3 };
    if (in.size() < 5 || in[0] != 0x07230203u) throw std::runtime_error("not SPIR-V");

    std::map<uint32_t, uint32_t> location, builtin;  // id -> value
    std::set<uint32_t> outputs;
    uint32_t entry = 0;
    size_t firstCap = 0, afterEntry = 0, firstDecorate = 0;
    for (size_t i = 5; i < in.size();) {
      uint32_t op = in[i] & 0xFFFFu, wc = in[i] >> 16;
      if (!wc || i + wc > in.size()) throw std::runtime_error("bad SPIR-V");
      if (op == OpCapability && !firstCap) firstCap = i;
      if (op == OpEntryPoint) { entry = in[i + 2]; afterEntry = i + wc; }
      if (op == OpDecorate && !firstDecorate) firstDecorate = i;
      if (op == OpDecorate && wc >= 4 && in[i + 2] == DecLocation) location[in[i + 1]] = in[i + 3];
      if (op == OpDecorate && wc >= 4 && in[i + 2] == DecBuiltIn) builtin[in[i + 1]] = in[i + 3];
      if (op == OpVariable && wc >= 4 && in[i + 3] == StorageOutput) outputs.insert(in[i + 2]);
      i += wc;
    }
    if (!firstCap || !afterEntry || !firstDecorate) throw std::runtime_error("unexpected SPIR-V layout");

    std::vector<uint32_t> decos;
    auto deco = [&](uint32_t id, uint32_t d, uint32_t v) { decos.insert(decos.end(), { (4u << 16) | OpDecorate, id, d, v }); };
    for (uint32_t id : outputs) {
      uint32_t offset;
      if (builtin.count(id) && builtin[id] == BuiltInPosition) offset = 0;
      else if (location.count(id) && location[id] < 16) offset = 16 * (location[id] + 1);
      else continue;  // ClipDistance, PointSize: not captured
      deco(id, DecXfbBuffer, 0);
      deco(id, DecXfbStride, kXfbStride);
      deco(id, DecOffset, offset);
    }

    // Insert in descending position order so earlier offsets stay valid.
    std::vector<uint32_t> out(in);
    out.insert(out.begin() + firstDecorate, decos.begin(), decos.end());
    const uint32_t mode[] = { (3u << 16) | OpExecutionMode, entry, ModeXfb };
    out.insert(out.begin() + afterEntry, std::begin(mode), std::end(mode));
    const uint32_t cap[] = { (2u << 16) | OpCapability, CapTransformFeedback };
    out.insert(out.begin() + firstCap, std::begin(cap), std::end(cap));
    return out;
  }
};

// ---------------------------------------------------------------------------------------------------------------
struct Diff {
  uint32_t mismatched = 0;    // pixels with any channel outside tolerance
  uint32_t discardDiff = 0;   // pixels written by one API and discarded by the other
  uint32_t nonFinite = 0;     // differing pixels where either side holds inf/NaN
  uint32_t robust = 0;        // mismatching pixels where the D3D9 reference is stable under input perturbation
  double maxAbs = 0.0, sumAbs = 0.0;
  uint32_t compared = 0;
  int firstX = -1, firstY = -1;
  float ref[4] = { }, got[4] = { };
};

static bool closeEnough(float a, float b) {
  if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
  if (std::isinf(a) || std::isinf(b)) return a == b;
  double d = std::fabs(double(a) - double(b));
  return d <= 2e-3 + 2e-3 * std::fabs(double(a));
}

static Diff compare(const std::vector<float>& ref, const std::vector<float>& got) {
  Diff d;
  for (uint32_t p = 0; p < W * H; p++) {
    const float* r = &ref[p * 4];
    const float* g = &got[p * 4];
    bool rDisc = r[0] == kSentinel && r[1] == kSentinel && r[2] == kSentinel && r[3] == kSentinel;
    bool gDisc = g[0] == kSentinel && g[1] == kSentinel && g[2] == kSentinel && g[3] == kSentinel;
    bool bad = false;
    if (rDisc != gDisc) { d.discardDiff++; bad = true; }
    else if (!rDisc) {
      for (int c = 0; c < 4; c++) {
        if (!closeEnough(r[c], g[c])) bad = true;
        if (std::isfinite(r[c]) && std::isfinite(g[c])) {
          double e = std::fabs(double(r[c]) - double(g[c]));
          d.maxAbs = std::max(d.maxAbs, e);
          d.sumAbs += e;
          d.compared++;
        }
      }
    }
    bool nonFinite = false;
    for (int c = 0; c < 4; c++) nonFinite |= !std::isfinite(r[c]) || !std::isfinite(g[c]);
    if (bad && nonFinite && rDisc == gDisc) {
      // Both APIs produced a value but at least one is inf/NaN: IEEE special-value territory (e.g. pow(0,-x),
      // inf-inf) where the D3D9 driver's fast-math reassociation decides the result. Counted separately.
      d.nonFinite++;
      bad = false;
    }
    if (bad) {
      if (d.mismatched == 0) {
        d.firstX = int(p % W); d.firstY = int(p / W);
        std::memcpy(d.ref, r, 16); std::memcpy(d.got, g, 16);
      }
      d.mismatched++;
    }
  }
  return d;
}

static Job parseJob(const std::string& line) {
  std::istringstream s(line);
  Job j;
  std::string cbvs, tex;
  s >> j.hash >> j.profile >> cbvs >> tex >> j.defs;
  if (!(s >> j.sun)) j.sun = -1;
  if (cbvs != "-") {
    std::stringstream cs(cbvs);
    std::string t;
    while (std::getline(cs, t, ',')) j.cbvs.push_back(uint32_t(std::stoul(t)));
  }
  if (tex != "-") {
    std::stringstream ts(tex);
    std::string t;
    while (std::getline(ts, t, ',')) {
      auto c = t.find(':');
      j.textures.emplace_back(uint32_t(std::stoul(t.substr(0, c))), uint32_t(std::stoul(t.substr(c + 1))));
    }
  }
  return j;
}

// Overlays shader-defined float constants ("reg/x/y/z/w,..." or "-") onto a float4 constant array.
static void applyDefOverlay(const std::string& defStr, std::vector<float>& constants) {
  if (defStr.empty() || defStr == "-") return;
  std::stringstream ds(defStr);
  std::string t;
  while (std::getline(ds, t, ',')) {
    std::stringstream ps(t);
    std::string f;
    std::vector<float> vals;
    while (std::getline(ps, f, '/')) vals.push_back(std::strtof(f.c_str(), nullptr));
    if (vals.size() != 5) throw std::runtime_error("bad def entry " + t);
    size_t reg = size_t(vals[0]);
    if (reg * 4 + 4 <= constants.size()) std::memcpy(&constants[reg * 4], &vals[1], 16);
  }
}

// ---------------------------------------------------------------------------------------------------------------
// Vertex-shader reference: D3D9 software vertex processing (the D3D9 runtime's own CPU implementation) via
// IDirect3DDevice9::ProcessVertices. Output layout matches VkCand::runVs (position at 0, location L at 16*(L+1)).
struct VsInput { std::string semantic; uint32_t index = 0, location = 0; };

class D3D9VsRef {
public:
  D3D9VsRef() {
    WNDCLASSA wc = { };
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandleA(nullptr);
    wc.lpszClassName = "shaderdiffvs";
    RegisterClassA(&wc);
    m_wnd = CreateWindowA("shaderdiffvs", "shaderdiffvs", WS_OVERLAPPEDWINDOW, 0, 0, W, H, nullptr, nullptr, wc.hInstance, nullptr);
    m_d3d = Direct3DCreate9(D3D_SDK_VERSION);
    if (!m_d3d) throw std::runtime_error("Direct3DCreate9 failed");
    D3DPRESENT_PARAMETERS pp = { };
    pp.Windowed = TRUE; pp.SwapEffect = D3DSWAPEFFECT_DISCARD; pp.BackBufferWidth = W; pp.BackBufferHeight = H;
    pp.BackBufferFormat = D3DFMT_X8R8G8B8; pp.hDeviceWindow = m_wnd;
    CHECK_HR(m_d3d->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, m_wnd,
      D3DCREATE_SOFTWARE_VERTEXPROCESSING | D3DCREATE_FPU_PRESERVE, &pp, &m_dev));

    // ProcessVertices requires a destination buffer created with an FVF; outputs follow the FVF layout:
    //   XYZRHW @0 (16 B), DIFFUSE @16 (D3DCOLOR), SPECULAR @20 (D3DCOLOR, alpha = fog), TEX0..7 float4 @24+16i.
    // So oD0/oD1/oFog come back as 8-bit values; texcoords and position at full precision.
    std::vector<D3DVERTEXELEMENT9> out = { { 0, 0, D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_POSITIONT, 0 },
      { 0, 16, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 0 },
      { 0, 20, D3DDECLTYPE_D3DCOLOR, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_COLOR, 1 } };
    for (BYTE t = 0; t < 8; t++) out.push_back({ 0, WORD(24 + 16 * t), D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, D3DDECLUSAGE_TEXCOORD, t });
    out.push_back(D3DDECL_END());
    CHECK_HR(m_dev->CreateVertexDeclaration(out.data(), &m_outDecl));
  }

  static constexpr DWORD kDstFvf = D3DFVF_XYZRHW | D3DFVF_DIFFUSE | D3DFVF_SPECULAR | D3DFVF_TEX8 |
    D3DFVF_TEXCOORDSIZE4(0) | D3DFVF_TEXCOORDSIZE4(1) | D3DFVF_TEXCOORDSIZE4(2) | D3DFVF_TEXCOORDSIZE4(3) |
    D3DFVF_TEXCOORDSIZE4(4) | D3DFVF_TEXCOORDSIZE4(5) | D3DFVF_TEXCOORDSIZE4(6) | D3DFVF_TEXCOORDSIZE4(7);
  static constexpr uint32_t kDstStride = 24 + 16 * 8;

  ~D3D9VsRef() {
    if (m_outDecl) m_outDecl->Release();
    if (m_dev) m_dev->Release();
    if (m_d3d) m_d3d->Release();
    if (m_wnd) DestroyWindow(m_wnd);
  }

  // Returns count * 52 floats. Position is converted back from screen space (viewport 2x2 at the origin,
  // depth 0..1) to clip space: x = (xs - 1) w, y = (1 - ys) w, z = zs w, w = 1/rhw.
  std::vector<float> run(const std::vector<uint8_t>& code, const std::vector<VsInput>& inputs,
                         const std::vector<float>& vertexData, uint32_t stride, uint32_t count,
                         const std::vector<float>& constants) {
    IDirect3DVertexShader9* vs = nullptr;
    CHECK_HR(m_dev->CreateVertexShader(reinterpret_cast<const DWORD*>(code.data()), &vs));

    static const std::map<std::string, BYTE> usage = {
      { "position", D3DDECLUSAGE_POSITION }, { "weight", D3DDECLUSAGE_BLENDWEIGHT }, { "blend", D3DDECLUSAGE_BLENDINDICES },
      { "normal", D3DDECLUSAGE_NORMAL }, { "psize", D3DDECLUSAGE_PSIZE }, { "texcoord", D3DDECLUSAGE_TEXCOORD },
      { "tangent", D3DDECLUSAGE_TANGENT }, { "binormal", D3DDECLUSAGE_BINORMAL }, { "color", D3DDECLUSAGE_COLOR },
      { "fog", D3DDECLUSAGE_FOG }, { "depth", D3DDECLUSAGE_DEPTH }, { "sample", D3DDECLUSAGE_SAMPLE } };
    std::vector<D3DVERTEXELEMENT9> in;
    for (size_t k = 0; k < inputs.size(); k++) {
      auto u = usage.find(inputs[k].semantic);
      if (u == usage.end()) throw std::runtime_error("unknown input semantic " + inputs[k].semantic);
      in.push_back({ 0, WORD(16 * k), D3DDECLTYPE_FLOAT4, D3DDECLMETHOD_DEFAULT, u->second, BYTE(inputs[k].index) });
    }
    in.push_back(D3DDECL_END());
    IDirect3DVertexDeclaration9* inDecl = nullptr;
    CHECK_HR(m_dev->CreateVertexDeclaration(in.data(), &inDecl));

    IDirect3DVertexBuffer9* src = nullptr;
    IDirect3DVertexBuffer9* dst = nullptr;
    CHECK_HR(m_dev->CreateVertexBuffer(UINT(vertexData.size() * 4), 0, 0, D3DPOOL_SYSTEMMEM, &src, nullptr));
    CHECK_HR(m_dev->CreateVertexBuffer(count * kDstStride, 0, kDstFvf, D3DPOOL_SYSTEMMEM, &dst, nullptr));
    void* p = nullptr;
    CHECK_HR(src->Lock(0, 0, &p, 0)); std::memcpy(p, vertexData.data(), vertexData.size() * 4); src->Unlock();
    CHECK_HR(dst->Lock(0, 0, &p, 0)); std::memset(p, 0, count * kDstStride); dst->Unlock();

    D3DVIEWPORT9 vp = { 0, 0, 2, 2, 0.0f, 1.0f };
    m_dev->SetViewport(&vp);
    m_dev->SetRenderState(D3DRS_CLIPPING, FALSE);
    m_dev->SetRenderState(D3DRS_CLIPPLANEENABLE, 0);
    m_dev->SetRenderState(D3DRS_FOGENABLE, FALSE);
    m_dev->SetVertexShaderConstantF(0, constants.data(), 256);
    m_dev->SetStreamSource(0, src, 0, stride);
    m_dev->SetVertexDeclaration(inDecl);
    m_dev->SetVertexShader(vs);
    // The output declaration is required for vs_3_0 (semantic-matched outputs); older models write the
    // destination FVF layout from the fixed output registers.
    const bool sm3 = code.size() >= 4 && ((code[1] & 0xFF) >= 3);
    CHECK_HR(m_dev->ProcessVertices(0, 0, count, dst, sm3 ? m_outDecl : nullptr, 0));

    // Repack into the transform-feedback layout (position @0, location L @ 16*(L+1); color0 = L9,
    // color1 = L10, fog = L11), converting D3DCOLOR (B,G,R,A bytes) to floats.
    std::vector<float> out(size_t(count) * 52, 0.0f);
    CHECK_HR(dst->Lock(0, 0, &p, D3DLOCK_READONLY));
    const uint8_t* bytes = static_cast<const uint8_t*>(p);
    for (uint32_t v = 0; v < count; v++) {
      const uint8_t* src8 = bytes + size_t(v) * kDstStride;
      float* o = &out[v * 52];
      float pos[4];
      std::memcpy(pos, src8, 16);
      float w = 1.0f / pos[3];
      o[0] = (pos[0] - 1.0f) * w; o[1] = (1.0f - pos[1]) * w; o[2] = pos[2] * w; o[3] = w;
      auto color = [&](const uint8_t* c, float* dstv) { dstv[0] = c[2] / 255.0f; dstv[1] = c[1] / 255.0f; dstv[2] = c[0] / 255.0f; dstv[3] = c[3] / 255.0f; };
      color(src8 + 16, o + 4 * 10);  // COLOR0 -> location 9
      color(src8 + 20, o + 4 * 11);  // COLOR1 -> location 10
      o[4 * 12] = src8[23] / 255.0f;  // fog = specular alpha -> location 11
      std::memcpy(o + 4 * 2, src8 + 24, 16 * 8);  // TEXCOORD0..7 -> locations 1..8
    }
    dst->Unlock();

    m_dev->SetVertexShader(nullptr);
    m_dev->SetStreamSource(0, nullptr, 0, 0);
    for (auto* u : { (IUnknown*)dst, (IUnknown*)src, (IUnknown*)inDecl, (IUnknown*)vs }) u->Release();
    return out;
  }

private:
  HWND m_wnd = nullptr;
  IDirect3D9* m_d3d = nullptr;
  IDirect3DDevice9* m_dev = nullptr;
  IDirect3DVertexDeclaration9* m_outDecl = nullptr;
};

static constexpr uint32_t kGotFloats = 17 * 4;  // floats per vertex in VkCand::runVs output (kXfbStride / 4)

// VS job line: <hash> <profile> <cbvs> <inputs semantic:index:location,...> <outputs: locations,... (P = position)>
static int runVsMode(const char* jobsPath, const std::string& spvDir, const char* outPath, bool validation) {
  D3D9VsRef d3d;
  VkCand vk(validation);
  std::cerr << "Vulkan: " << vk.adapter << "\n" << vk.requirementReport << "\n";
  constexpr uint32_t N = 256;

  std::ifstream jobs(jobsPath);
  std::ofstream out(outPath, std::ios::trunc);
  out << "hash,profile,status,mismatchedVerts,skippedW,maxAbs,validation,firstVertex,firstOutput,ref,got,error\n";
  std::string line;
  size_t n = 0;
  while (std::getline(jobs, line)) {
    if (line.empty()) continue;
    std::istringstream s(line);
    std::string hash, profile, cbvStr, inStr, outStr, defStr;
    s >> hash >> profile >> cbvStr >> inStr >> outStr >> defStr;
    std::vector<uint32_t> cbvs;
    { std::stringstream cs(cbvStr); std::string t; while (cbvStr != "-" && std::getline(cs, t, ',')) cbvs.push_back(std::stoul(t)); }
    std::vector<VsInput> inputs;
    { std::stringstream is(inStr); std::string t;
      while (std::getline(is, t, ',')) {
        auto a = t.find(':'), b = t.rfind(':');
        inputs.push_back({ t.substr(0, a), uint32_t(std::stoul(t.substr(a + 1, b - a - 1))), uint32_t(std::stoul(t.substr(b + 1))) });
      } }
    // Outputs as "<P|location>:<written mask>"; only written components are compared.
    std::vector<std::pair<int, uint32_t>> outputs;  // (-1 = position | location, component bit mask)
    { std::stringstream os(outStr); std::string t;
      while (outStr != "-" && std::getline(os, t, ',')) {
        auto c = t.find(':');
        std::string key = t.substr(0, c), mask = c == std::string::npos ? "xyzw" : t.substr(c + 1);
        uint32_t bits = 0;
        for (char ch : mask) bits |= ch == 'x' ? 1u : ch == 'y' ? 2u : ch == 'z' ? 4u : ch == 'w' ? 8u : 0u;
        outputs.emplace_back(key == "P" ? -1 : std::stoi(key), bits);
      } }

    std::string status = "ok", error;
    uint32_t mism = 0, skippedW = 0;
    double maxAbs = 0.0;
    int firstVertex = -1, firstOutput = -2;
    float refv[4] = { }, gotv[4] = { };
    uint32_t valBefore = g_validationMessages;
    g_jobMessages.clear();
    try {
      // Inputs: one float4 per declared input, in declaration order.
      std::seed_seq seq(hash.begin(), hash.end());
      std::mt19937 rng(seq);
      std::uniform_real_distribution<float> sym(-1.0f, 1.0f), unit(0.0f, 1.0f), tc(-1.0f, 2.0f);
      std::uniform_int_distribution<int> bone(0, 17);
      const uint32_t stride = uint32_t(inputs.size()) * 16;
      std::vector<float> vtx(size_t(N) * inputs.size() * 4);
      for (uint32_t v = 0; v < N; v++) {
        for (size_t k = 0; k < inputs.size(); k++) {
          float* a = &vtx[(v * inputs.size() + k) * 4];
          const std::string& sem = inputs[k].semantic;
          for (int c = 0; c < 4; c++) {
            if (sem == "position") a[c] = c == 3 ? 1.0f : sym(rng);
            else if (sem == "color" || sem == "weight") a[c] = unit(rng);
            else if (sem == "texcoord") a[c] = tc(rng);
            // BLENDINDICES arrive as D3DCOLOR-normalised bone indices (k/255) and are scaled by 765.01 to
            // register offsets 3k; keep them within the 18-bone partition the shaders are built for.
            else if (sem == "blend") a[c] = float(bone(rng)) / 255.0f;
            else a[c] = sym(rng);
          }
        }
      }
      std::vector<float> constants(256 * 4);
      for (auto& c : constants) c = sym(rng);
      // SHADERDIFF_FRAME: an orthonormal tangent frame per vertex (a random rotation's rows as tangent, binormal and
      // normal) and an orthonormal world rotation in c250..c252 (the rows the plugin uploads), for the frame check below.
      const bool frameTest = std::getenv("SHADERDIFF_FRAME") != nullptr;
      auto randomRotation = [&](float m[9]) {
        // Gram-Schmidt on random vectors: rows m[0..2], m[3..5], m[6..8], right-handed.
        float a[3], b[3];
        for (int c = 0; c < 3; c++) { a[c] = sym(rng); b[c] = sym(rng); }
        auto norm = [](float* x) { const float l = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]); for (int c = 0; c < 3; c++) x[c] /= l; };
        norm(a);
        const float d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        for (int c = 0; c < 3; c++) b[c] -= d * a[c];
        norm(b);
        const float n[3] = { a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] };
        for (int c = 0; c < 3; c++) { m[c] = a[c]; m[3 + c] = b[c]; m[6 + c] = n[c]; }
      };
      bool deformed = false;
      for (const VsInput& in : inputs) deformed |= in.semantic == "blend" || in.semantic == "weight";
      float worldRot[9];
      if (frameTest) {
        randomRotation(worldRot);
        for (int j = 0; j < 3; j++)
          for (int c = 0; c < 4; c++) constants[(250 + j) * 4 + c] = c < 3 ? worldRot[3 * j + c] : 0.0f;
        for (uint32_t v = 0; v < N; v++) {
          float f[9];
          randomRotation(f);
          for (size_t k = 0; k < inputs.size(); k++) {
            const std::string& sem = inputs[k].semantic;
            const int row = sem == "tangent" ? 0 : sem == "binormal" ? 1 : sem == "normal" ? 2 : -1;
            if (row < 0) continue;
            float* a = &vtx[(v * inputs.size() + k) * 4];
            for (int c = 0; c < 3; c++) a[c] = f[3 * row + c];
            a[3] = 0.0f;
          }
        }
      }
      // SHADERDIFF_VIEWVEC: LightData (c25) xyz 0 in both APIs, so a SPECULAR shader's half vector is its view vector.
      const bool viewvecTest = std::getenv("SHADERDIFF_VIEWVEC") != nullptr;
      if (viewvecTest) constants[25 * 4 + 0] = constants[25 * 4 + 1] = constants[25 * 4 + 2] = 0.0f;
      // Runtime ABI: the backend overlays the shader's own `def` values onto the uploaded constants (D3D9
      // returns def values for relative reads). D3D9 applies defs itself, so only the Vulkan copy gets them.
      std::vector<float> vkConstants = constants;
      applyDefOverlay(defStr, vkConstants);

      auto dxso = readFile(spvDir + "/" + hash + ".dxso");
      auto spvBytes = readFile(spvDir + "/" + hash + ".spv");
      std::vector<uint32_t> spv(spvBytes.size() / 4);
      std::memcpy(spv.data(), spvBytes.data(), spv.size() * 4);
      poisonUnread(spv.data(), spv.size(), 0, vkConstants, defStr);
      std::vector<std::pair<uint32_t, uint32_t>> attribs;
      for (size_t k = 0; k < inputs.size(); k++) attribs.emplace_back(inputs[k].location, uint32_t(k));

      const bool trace = std::getenv("SHADERDIFF_TRACE") != nullptr;
      if (trace) std::cerr << hash << ": d3d9 ProcessVertices\n";
      auto ref = d3d.run(dxso, inputs, vtx, stride, N, constants);
      if (trace) std::cerr << hash << ": vulkan xfb\n";
      auto got = vk.runVs(spv, cbvs, attribs, vtx, stride, N, vkConstants);
      if (trace) std::cerr << hash << ": compare\n";

      for (uint32_t v = 0; v < N; v++) {
        const float* r = &ref[v * 52];
        const float* g = &got[v * kGotFloats];
        bool bad = false;
        int badOut = -2;
        float rv[4] = { }, gv[4] = { };
        for (auto [o, maskBits] : outputs) {
          if (o == -1) {
            // Compare what rasterisation consumes: NDC and 1/w. Skip near-singular projections.
            if (std::fabs(r[3]) < 1e-3f || std::fabs(g[3]) < 1e-3f) { skippedW++; continue; }
            for (int c = 0; c < 4; c++) {
              float a = c < 3 ? r[c] / r[3] : 1.0f / r[3];
              float b = c < 3 ? g[c] / g[3] : 1.0f / g[3];
              maxAbs = std::max(maxAbs, double(std::fabs(a - b)));
              if (!closeEnough(a, b) && !bad) { bad = true; badOut = -1; std::memcpy(rv, r, 16); std::memcpy(gv, g, 16); }
            }
          } else if (o >= 1 && o <= 8) {
            const float* ro = r + 4 * (o + 1);
            const float* go = g + 4 * (o + 1);
            for (int c = 0; c < 4; c++) {
              if (!(maskBits & (1u << c))) continue;
              if (std::isfinite(ro[c]) && std::isfinite(go[c])) maxAbs = std::max(maxAbs, double(std::fabs(ro[c] - go[c])));
              if (!closeEnough(ro[c], go[c]) && !bad) { bad = true; badOut = o; std::memcpy(rv, ro, 16); std::memcpy(gv, go, 16); }
            }
          } else if (o >= 9 && o <= 11) {
            // The D3D9 reference only reports these as 8-bit D3DCOLOR (FVF destination): clamp the candidate
            // to [0,1] and allow one 8-bit step. COLOR1 alpha carries fog in the FVF, so skip it.
            int comps = o == 11 ? 1 : (o == 10 ? 3 : 4);
            const float* ro = r + 4 * (o + 1);
            const float* go = g + 4 * (o + 1);
            for (int c = 0; c < comps; c++) {
              if (o != 11 && !(maskBits & (1u << c))) continue;
              float q = std::isnan(go[c]) ? 0.0f : std::clamp(go[c], 0.0f, 1.0f);
              if (std::fabs(q - ro[c]) > 1.01f / 255.0f && !bad) { bad = true; badOut = o; std::memcpy(rv, ro, 16); std::memcpy(gv, go, 16); }
            }
          }
        }
        if (bad) {
          if (!mism) { firstVertex = int(v); firstOutput = badOut; std::memcpy(refv, rv, 16); std::memcpy(gotv, gv, 16); }
          mism++;
        }
      }
      if (mism) status = "mismatch";
      // The view vector (location 12) must have the direction of one of D3D9's texture coordinate outputs (xyz) at
      // every vertex: the half vector the game computes from EyePosition, with no light added (some shaders leave it
      // unnormalised, the land shaders store it encoded as 0.5 x + 0.5).
      // Point-light shaders take LightData as positions: a second setup puts every light (LightData[0..9]) at
      // EyePosition, where each light vector, and so each half vector, is the view vector.
      // The tangent-to-world frame (locations 13..15) applied to the view vector (location 12, M (E - P)) is the world
      // rotation applied to E - P (EyePosition c16 minus the position input) when the frame is orthonormal. Only for
      // undeformed shaders (the position the shader transforms is then its input).
      if (frameTest && !mism && !deformed) {
        int posInput = -1;
        for (size_t k = 0; k < inputs.size(); k++)
          if (inputs[k].semantic == "position") posInput = int(k);
        uint32_t frameBad = 0;
        for (uint32_t v = 0; v < N && posInput >= 0; v++) {
          const float* g = &got[v * kGotFloats];
          const float* view = g + 4 * 13;
          float w[3], e[3];
          for (int j = 0; j < 3; j++) {
            const float* row = g + 4 * (14 + j);
            w[j] = row[0] * view[0] + row[1] * view[1] + row[2] * view[2];
          }
          const float* p = &vtx[(v * inputs.size() + posInput) * 4];
          float d[3];
          for (int c = 0; c < 3; c++) d[c] = vkConstants[16 * 4 + c] - p[c];
          for (int j = 0; j < 3; j++) e[j] = worldRot[3 * j] * d[0] + worldRot[3 * j + 1] * d[1] + worldRot[3 * j + 2] * d[2];
          bool same = true;
          for (int c = 0; c < 3; c++) same &= std::fabs(w[c] - e[c]) <= 1e-4f * (1.0f + std::fabs(e[c]));
          if (!same && !frameBad++) { firstVertex = int(v); std::memcpy(gotv, w, 12); std::memcpy(refv, e, 12); }
        }
        if (posInput < 0) error = "frame: no position input";
        else if (frameBad) { status = "frame"; firstOutput = 13; mism = frameBad; }
        else error = "frame ok";
      }
      if (viewvecTest && !mism) {
        auto matchIn = [&](const std::vector<float>& r, const std::vector<float>& g) {
          for (int o = 1; o <= 8; o++)
            for (int encoded = 0; encoded < 2; encoded++) {
              bool all = true;
              for (uint32_t v = 0; v < N && all; v++) {
                float d[3];
                for (int c = 0; c < 3; c++) {
                  const float x = r[v * 52 + 4 * (o + 1) + c];
                  d[c] = encoded ? x * 2.0f - 1.0f : x;
                }
                const float len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                const float* e = &g[v * kGotFloats + 4 * 13];  // the view vector, unnormalised
                const float elen = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
                all = len > 1e-6f && elen > 1e-6f;
                for (int c = 0; c < 3 && all; c++) all = closeEnough(d[c] / len, e[c] / elen);
              }
              if (all) return o + (encoded ? 100 : 0);
            }
          return -1;
        };
        int found = matchIn(ref, got);
        if (found < 0) {
          std::vector<float> atEye = constants;
          for (int l = 25; l < 35; l++)
            for (int c = 0; c < 3; c++) atEye[l * 4 + c] = constants[16 * 4 + c];
          atEye[16 * 4 + 3] = 4.0f;  // EyePosition.w: the light count of the interior (OPT) shaders, all lit
          std::vector<float> vkAtEye = atEye;
          applyDefOverlay(defStr, vkAtEye);
          poisonUnread(spv.data(), spv.size(), 0, vkAtEye, defStr);
          const auto ref2 = d3d.run(dxso, inputs, vtx, stride, N, atEye);
          const auto got2 = vk.runVs(spv, cbvs, attribs, vtx, stride, N, vkAtEye);
          found = matchIn(ref2, got2);
          if (found >= 0) found += 1000;
        }
        if (found < 0) {
          status = "viewvec";
          firstVertex = 0;
          firstOutput = 12;
          std::memcpy(gotv, &got[4 * 13], 16);
        } else {
          error = "view vector = TEXCOORD" + std::to_string(found % 100 - 1) + (found % 1000 >= 100 ? " (encoded)" : "") +
                  (found >= 1000 ? " (lights at the eye)" : "");
        }
      }
    } catch (const std::exception& e) {
      status = "error";
      error = e.what();
    }
    uint32_t val = g_validationMessages - valBefore;
    if (val && status == "ok") status = "validation";
    for (const auto& [id, msg] : g_jobMessages) error += (error.empty() ? "" : " | ") + id + ": " + msg.substr(0, 200);
    for (auto& ch : error) if (ch == ',' || ch == '\n' || ch == '"') ch = ' ';
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%s,%s,%s,%u,%u,%.6g,%u,%d,%d,%g %g %g %g,%g %g %g %g,", hash.c_str(), profile.c_str(),
      status.c_str(), mism, skippedW, maxAbs, val, firstVertex, firstOutput, refv[0], refv[1], refv[2], refv[3],
      gotv[0], gotv[1], gotv[2], gotv[3]);
    out << buf << error << "\n";
    out.flush();
    if (++n % 100 == 0) std::cerr << "  " << n << "\n";
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 5 && std::string(argv[1]) == "--vs") {
    bool validation = !(argc > 5 && std::string(argv[5]) == "--no-validation");
    try { return runVsMode(argv[2], argv[3], argv[4], validation); }
    catch (const std::exception& e) { std::cerr << "fatal: " << e.what() << "\n"; return 1; }
  }
  if (argc < 4) {
    std::cerr << "usage: shaderdiff <jobs.txt> <spv-dir> <results.csv> [--no-validation]\n"
                 "       shaderdiff --vs <vs-jobs.txt> <spv-dir> <results.csv> [--no-validation]\n";
    return 2;
  }
  bool validation = !(argc > 4 && std::string(argv[4]) == "--no-validation");
  SunMode sunMode = SunMode::Off;
  if (const char* m = std::getenv("SHADERDIFF_SUN")) {
    const std::string v = m;
    sunMode = v == "shadow" ? SunMode::Shadow : v == "blend" ? SunMode::Blend : v == "splitx" ? SunMode::SplitX
            : v == "splity" ? SunMode::SplitY : SunMode::Off;
    if (sunMode == SunMode::Off && v != "off") { std::cerr << "SHADERDIFF_SUN: off, shadow, blend, splitx or splity\n"; return 2; }
  }
  const char* strengthEnv = std::getenv("SHADERDIFF_SUN_STRENGTH");
  const float sunStrength = sunMode == SunMode::Shadow && strengthEnv ? std::strtof(strengthEnv, nullptr) : 1.0f;
  const SunSetup sunSetup = makeSun(sunMode, sunStrength);
  try {
    D3D9Ref d3d;
    VkCand vk(validation);
    std::cerr << "D3D9: " << d3d.adapter << "\nVulkan: " << vk.adapter << "\n" << vk.requirementReport << "\n";

    std::ifstream jobs(argv[1]);
    std::ofstream out(argv[3], std::ios::trunc);
    out << "hash,profile,status,mismatched,discardDiff,maxAbs,meanAbs,validation,firstX,firstY,ref,got,nonFinite,robust,error\n";
    std::string line;
    size_t n = 0;
    while (std::getline(jobs, line)) {
      if (line.empty()) continue;
      Job job = parseJob(line);
      if (sunMode != SunMode::Off && job.sun < 0) continue;  // the sun modes test the shaders with a sun light colour
      std::string spvDir = argv[2];
      std::string status = "ok", error;
      Diff d;
      uint32_t valBefore = g_validationMessages;
      g_jobMessages.clear();
      try {
        auto dxso = readFile(spvDir + "/" + job.hash + ".dxso");
        auto spv = readFile(spvDir + "/" + job.hash + ".spv");
        Inputs in = makeInputs(job);
        auto ref = d3d.run(job, dxso, in);
        auto got = vk.run(job, spv, in, sunSetup);
        if (sunMode != SunMode::Off) {
          // The shadowed result: the original shader with the sun colour's rgb scaled as the visibility scales it.
          Inputs dark = in;
          const float vis = sunMode == SunMode::Blend ? blendVisibility() : 1.0f - sunStrength;
          for (int c = 0; c < 3; c++) dark.constants[job.sun * 4 + c] *= vis;
          const auto shadowed = d3d.run(job, dxso, dark);
          for (uint32_t p = 0; p < W * H; p++) {
            const uint32_t x = p % W, y = p / W;
            // splitx: columns left of the centre are in shadow; splity: rows below it (view y points up).
            const uint32_t t = sunMode == SunMode::SplitX ? x : sunMode == SunMode::SplitY ? (H - 1 - y) : 0;
            const bool whole = sunMode == SunMode::Shadow || sunMode == SunMode::Blend;
            const bool band = !whole && t >= 29 && t <= 34;
            const bool dim = whole || t < 29;
            const float* src = band ? &got[p * 4] : dim ? &shadowed[p * 4] : &ref[p * 4];
            std::memcpy(&ref[p * 4], src, 16);
          }
        }
        d = compare(ref, got);
        if (const char* dump = std::getenv("SHADERDIFF_DUMP_DIR")) {
          // Raw 64x64x4 float32 images for offline analysis.
          std::ofstream(std::string(dump) + "/" + job.hash + ".ref.f32", std::ios::binary)
            .write(reinterpret_cast<const char*>(ref.data()), ref.size() * 4);
          std::ofstream(std::string(dump) + "/" + job.hash + ".got.f32", std::ios::binary)
            .write(reinterpret_cast<const char*>(got.data()), got.size() * 4);
          std::ofstream(std::string(dump) + "/" + job.hash + ".vtx.f32", std::ios::binary)
            .write(reinterpret_cast<const char*>(in.vertices.data()), sizeof(in.vertices));
          std::ofstream(std::string(dump) + "/" + job.hash + ".c.f32", std::ios::binary)
            .write(reinterpret_cast<const char*>(in.constants.data()), sizeof(in.constants));
          for (const auto& [reg, texels] : in.texels)
            std::ofstream(std::string(dump) + "/" + job.hash + ".s" + std::to_string(reg) + ".bgra", std::ios::binary)
              .write(reinterpret_cast<const char*>(texels.data()), texels.size() * 4);
        }
        if (d.mismatched) {
          status = "mismatch";
          // Sensitivity oracle: re-render the D3D9 reference with every interpolant perturbed by +-1e-6
          // (relative). A mismatching pixel whose reference value stays within tolerance under both
          // perturbations is "robust": the shader is well-conditioned there, so the difference cannot be
          // explained by rounding and points at a translation problem. Mismatches where D3D9 itself moves are
          // ill-conditioned (data-dependent branches, loop exits, near-singular divisions).
          auto perturbed = [&](float scale) {
            Inputs p = in;
            for (int v = 0; v < 4; v++)
              for (int a = 4; a < 44; a++) p.vertices[v * 44 + a] *= scale;
            return d3d.run(job, dxso, p);
          };
          const char* epsEnv = std::getenv("SHADERDIFF_EPS");
          const float eps = epsEnv ? std::strtof(epsEnv, nullptr) : 1e-6f;
          auto refUp = perturbed(1.0f + eps);
          auto refDown = perturbed(1.0f - eps);
          for (uint32_t p = 0; p < W * H; p++) {
            bool bad = false, stable = true;
            for (int c = 0; c < 4; c++) {
              size_t i = p * 4 + c;
              bad |= !closeEnough(ref[i], got[i]);
              stable &= closeEnough(ref[i], refUp[i]) && closeEnough(ref[i], refDown[i]);
            }
            if (bad && stable) d.robust++;
          }
        }
      } catch (const std::exception& e) {
        status = "error";
        error = e.what();
      }
      uint32_t val = g_validationMessages - valBefore;
      if (val && status == "ok") status = "validation";
      // One entry per distinct VUID: "<VUID>: <first 200 chars>" joined with " | ".
      for (const auto& [id, msg] : g_jobMessages)
        error += (error.empty() ? "" : " | ") + id + ": " + msg.substr(0, 200);
      for (auto& ch : error) if (ch == ',' || ch == '\n' || ch == '"') ch = ' ';
      char buf[1024];
      std::snprintf(buf, sizeof(buf), "%s,%s,%s,%u,%u,%.6g,%.6g,%u,%d,%d,%g %g %g %g,%g %g %g %g,%u,%u,",
        job.hash.c_str(), job.profile.c_str(), status.c_str(), d.mismatched, d.discardDiff, d.maxAbs,
        d.compared ? d.sumAbs / d.compared : 0.0, val, d.firstX, d.firstY,
        d.ref[0], d.ref[1], d.ref[2], d.ref[3], d.got[0], d.got[1], d.got[2], d.got[3], d.nonFinite, d.robust);
      out << buf << error << "\n";
      out.flush();
      if (++n % 100 == 0) std::cerr << "  " << n << "\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "fatal: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
