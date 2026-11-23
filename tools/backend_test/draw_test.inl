// Draw oracle: the same game shaders, geometry, texture, constants and render/sampler states drawn
// by real D3D9 and by the backend; render targets compared texel by texel. The D3D9 device's own state (read back
// with GetRenderState/GetSamplerState) is what the backend receives, so defaults cannot drift between the two.

namespace drawtest {

struct Vtx {
  float x, y, z;
  float u, v;
  uint32_t color;
};

static bool readFile(const std::string& path, std::vector<uint8_t>& out) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  out.resize(size_t(ftell(f)));
  fseek(f, 0, SEEK_SET);
  bool ok = fread(out.data(), 1, out.size(), f) == out.size();
  fclose(f);
  return ok;
}

static bool hexMd5(const std::string& hex, uint8_t out[16]) {
  if (hex.size() != 32) return false;
  for (int i = 0; i < 16; i++) out[i] = uint8_t(std::stoul(hex.substr(i * 2, 2), nullptr, 16));
  return true;
}

struct Case {
  const char* name;
  uint32_t rtFormat = kD3DFMT_A8R8G8B8;
  bool depth = false;
  std::vector<std::pair<D3DRENDERSTATETYPE, DWORD>> rs;
  std::vector<std::pair<D3DSAMPLERSTATETYPE, DWORD>> ss;
  std::vector<Vtx> verts;
  std::vector<uint16_t> indices;  // empty: DrawPrimitive
  D3DPRIMITIVETYPE prim = D3DPT_TRIANGLELIST;
  bool tile1000 = false;
  bool mipTexture = false;  // bind the 256x256 mip-chain texture (distinct colour per level)  // TILE1000 shaders, 20-byte position+texcoord vertices (first in-game pipeline)
  RECT scissor{ 0, 0, 0, 0 };
  D3DVIEWPORT9 viewport{ 0, 0, 0, 0, 0.0f, 1.0f };  // width 0 = full target
  std::vector<std::pair<uint32_t, std::array<float, 4>>> vsc, psc;
};

static const uint32_t kW = 96, kH = 64;
static float drawtestHalf(uint16_t h) {  // IEEE half -> float
  const uint32_t sign = uint32_t(h >> 15) << 31, exp = (h >> 10) & 31, man = h & 1023;
  uint32_t bits;
  if (exp == 0) {
    if (!man) bits = sign;
    else {
      int e = -1;
      uint32_t m = man;
      do { m <<= 1; e++; } while (!(m & 1024));
      bits = sign | uint32_t(127 - 15 - e) << 23 | (m & 1023) << 13;
    }
  } else if (exp == 31) bits = sign | 0x7F800000u | man << 13;
  else bits = sign | (exp + 112) << 23 | man << 13;
  float f;
  memcpy(&f, &bits, 4);
  return f;
}

static uint32_t strideOf(const Case& c) { return c.tile1000 ? 20u : uint32_t(sizeof(Vtx)); }
static std::vector<uint8_t> vertexBytes(const Case& c) {
  std::vector<uint8_t> out(c.verts.size() * strideOf(c));
  for (size_t i = 0; i < c.verts.size(); i++) memcpy(&out[i * strideOf(c)], &c.verts[i], strideOf(c));
  return out;
}

static std::vector<uint32_t> makeTexture(uint32_t w, uint32_t h) {
  std::vector<uint32_t> t(size_t(w) * h);
  for (uint32_t y = 0; y < h; y++)
    for (uint32_t x = 0; x < w; x++) {
      uint32_t a = ((x ^ y) & 4) ? 0xFF : 0x40 + ((x * 13 + y * 7) & 0x7F);
      uint32_t r = (x * 255) / (w - 1), g = (y * 255) / (h - 1), b = ((x + y) * 37) & 0xFF;
      t[size_t(y) * w + x] = (a << 24) | (r << 16) | (g << 8) | b;
    }
  return t;
}

// D3D9 side ------------------------------------------------------------------------------------------------------
struct D3DDraw {
  IDirect3DDevice9* dev;
  IDirect3DTexture9* tex = nullptr;
  IDirect3DTexture9* mipTex = nullptr;
  std::vector<uint32_t> texels;
  std::vector<std::vector<uint32_t>> mipLevels;  // 256x256 .. 1x1
  uint32_t tw = 16, th = 16;
  DWORD lastQuery = 0;  // occlusion-query result of the last run

  bool init() {
    texels = makeTexture(tw, th);
    if (FAILED(dev->CreateTexture(tw, th, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &tex, nullptr))) return false;
    D3DLOCKED_RECT lr{};
    if (FAILED(tex->LockRect(0, &lr, nullptr, 0))) return false;
    for (uint32_t y = 0; y < th; y++) memcpy(static_cast<uint8_t*>(lr.pBits) + y * lr.Pitch, &texels[size_t(y) * tw], tw * 4);
    tex->UnlockRect(0);
    // Mip chain: a checker in a colour unique to each level, so a different level or filter shows up immediately.
    if (FAILED(dev->CreateTexture(256, 256, 9, 0, D3DFMT_A8R8G8B8, D3DPOOL_MANAGED, &mipTex, nullptr))) return false;
    static const uint32_t kLevelColor[9] = { 0xFFFF4040, 0xFF40FF40, 0xFF4040FF, 0xFFFFFF40, 0xFF40FFFF,
                                             0xFFFF40FF, 0xFFC0C0C0, 0xFF806020, 0xFF208060 };
    for (uint32_t l = 0; l < 9; l++) {
      uint32_t s = 256 >> l;
      std::vector<uint32_t> px(size_t(s) * s);
      for (uint32_t y = 0; y < s; y++)
        for (uint32_t x = 0; x < s; x++) px[size_t(y) * s + x] = (((x >> 2) ^ (y >> 2)) & 1) ? kLevelColor[l] : 0xFF101010;
      mipLevels.push_back(px);
      if (FAILED(mipTex->LockRect(l, &lr, nullptr, 0))) return false;
      for (uint32_t y = 0; y < s; y++) memcpy(static_cast<uint8_t*>(lr.pBits) + y * lr.Pitch, &px[size_t(y) * s], s * 4);
      mipTex->UnlockRect(l);
    }
    return true;
  }

  bool run(const Case& c, const std::vector<uint8_t>& vsCode, const std::vector<uint8_t>& psCode,
           std::vector<uint8_t>& out, uint32_t rsOut[256], uint32_t ssOut[16][14]) {
    IDirect3DSurface9 *rt = nullptr, *ds = nullptr, *sys = nullptr;
    IDirect3DVertexShader9* vs = nullptr;
    IDirect3DPixelShader9* ps = nullptr;
    IDirect3DVertexDeclaration9* decl = nullptr;
    IDirect3DVertexBuffer9* vb = nullptr;
    IDirect3DIndexBuffer9* ib = nullptr;
    bool ok = SUCCEEDED(dev->CreateRenderTarget(kW, kH, D3DFORMAT(c.rtFormat), D3DMULTISAMPLE_NONE, 0, FALSE, &rt, nullptr)) &&
              SUCCEEDED(dev->CreateOffscreenPlainSurface(kW, kH, D3DFORMAT(c.rtFormat), D3DPOOL_SYSTEMMEM, &sys, nullptr)) &&
              (!c.depth || SUCCEEDED(dev->CreateDepthStencilSurface(kW, kH, D3DFMT_D24S8, D3DMULTISAMPLE_NONE, 0, FALSE, &ds, nullptr))) &&
              SUCCEEDED(dev->CreateVertexShader(reinterpret_cast<const DWORD*>(vsCode.data()), &vs)) &&
              SUCCEEDED(dev->CreatePixelShader(reinterpret_cast<const DWORD*>(psCode.data()), &ps));
    D3DVERTEXELEMENT9 elems[] = { { 0, 0, D3DDECLTYPE_FLOAT3, 0, D3DDECLUSAGE_POSITION, 0 },
                                  { 0, 12, D3DDECLTYPE_FLOAT2, 0, D3DDECLUSAGE_TEXCOORD, 0 },
                                  { 0, 20, D3DDECLTYPE_D3DCOLOR, 0, D3DDECLUSAGE_COLOR, 0 },
                                  D3DDECL_END() };
    if (c.tile1000) elems[2] = D3DDECL_END();
    const std::vector<uint8_t> vbytes = vertexBytes(c);
    ok = ok && SUCCEEDED(dev->CreateVertexDeclaration(elems, &decl));
    ok = ok && SUCCEEDED(dev->CreateVertexBuffer(UINT(vbytes.size()), D3DUSAGE_WRITEONLY, 0, D3DPOOL_DEFAULT, &vb, nullptr));
    if (ok) {
      void* p = nullptr;
      ok = SUCCEEDED(vb->Lock(0, 0, &p, 0));
      if (ok) { memcpy(p, vbytes.data(), vbytes.size()); vb->Unlock(); }
    }
    if (ok && !c.indices.empty()) {
      ok = SUCCEEDED(dev->CreateIndexBuffer(UINT(c.indices.size() * 2), D3DUSAGE_WRITEONLY, D3DFMT_INDEX16, D3DPOOL_DEFAULT, &ib, nullptr));
      void* p = nullptr;
      if (ok && SUCCEEDED(ib->Lock(0, 0, &p, 0))) { memcpy(p, c.indices.data(), c.indices.size() * 2); ib->Unlock(); }
    }
    if (ok) {
      // A clean, known device state per case: the D3D9 defaults, then the case's states.
      dev->SetRenderTarget(0, rt);
      dev->SetDepthStencilSurface(ds);
      D3DVIEWPORT9 full{ 0, 0, kW, kH, 0.0f, 1.0f };
      dev->SetViewport(&full);
      dev->Clear(0, nullptr, D3DCLEAR_TARGET | (ds ? D3DCLEAR_ZBUFFER | D3DCLEAR_STENCIL : 0), 0xFF203040, 1.0f, 0);
      D3DVIEWPORT9 vp = c.viewport.Width ? c.viewport : full;
      dev->SetViewport(&vp);
      for (auto& [s, v] : c.rs) dev->SetRenderState(s, v);
      for (auto& [s, v] : c.ss) dev->SetSamplerState(0, s, v);
      if (c.scissor.right) dev->SetScissorRect(&c.scissor);
      dev->SetVertexShader(vs);
      dev->SetPixelShader(ps);
      dev->SetVertexDeclaration(decl);
      dev->SetStreamSource(0, vb, 0, strideOf(c));
      dev->SetIndices(ib);
      dev->SetTexture(0, c.mipTexture ? mipTex : tex);
      dev->SetTexture(1, tex);
      for (auto& [r, v] : c.vsc) dev->SetVertexShaderConstantF(r, v.data(), 1);
      for (auto& [r, v] : c.psc) dev->SetPixelShaderConstantF(r, v.data(), 1);
      for (uint32_t i = 0; i < 256; i++) { DWORD v = 0; dev->GetRenderState(D3DRENDERSTATETYPE(i), &v); rsOut[i] = v; }
      for (uint32_t s = 0; s < 16; s++)
        for (uint32_t i = 0; i < 14; i++) { DWORD v = 0; if (i) dev->GetSamplerState(s, D3DSAMPLERSTATETYPE(i), &v); ssOut[s][i] = v; }
      dev->BeginScene();
      uint32_t prims = c.prim == D3DPT_TRIANGLELIST ? uint32_t((c.indices.empty() ? c.verts.size() : c.indices.size()) / 3)
                                                    : uint32_t((c.indices.empty() ? c.verts.size() : c.indices.size()) - 2);
      IDirect3DQuery9* q = nullptr;
      dev->CreateQuery(D3DQUERYTYPE_OCCLUSION, &q);
      if (q) q->Issue(D3DISSUE_BEGIN);
      if (ib) ok = SUCCEEDED(dev->DrawIndexedPrimitive(c.prim, 0, 0, UINT(c.verts.size()), 0, prims));
      else ok = SUCCEEDED(dev->DrawPrimitive(c.prim, 0, prims));
      if (q) q->Issue(D3DISSUE_END);
      dev->EndScene();
      lastQuery = ~0u;
      if (q) {
        while (q->GetData(&lastQuery, sizeof(lastQuery), D3DGETDATA_FLUSH) == S_FALSE) {}
        q->Release();
      }
      ok = ok && SUCCEEDED(dev->GetRenderTargetData(rt, sys));
      D3DLOCKED_RECT lr{};
      if (ok && SUCCEEDED(sys->LockRect(&lr, nullptr, D3DLOCK_READONLY))) {
        out.resize(size_t(kW) * kH * 4);
        for (uint32_t y = 0; y < kH; y++) memcpy(out.data() + size_t(y) * kW * 4, static_cast<uint8_t*>(lr.pBits) + size_t(y) * lr.Pitch, kW * 4);
        sys->UnlockRect();
      } else {
        ok = false;
      }
    }
    dev->SetStreamSource(0, nullptr, 0, 0);
    dev->SetIndices(nullptr);
    dev->SetVertexShader(nullptr);
    dev->SetPixelShader(nullptr);
    dev->SetVertexDeclaration(nullptr);
    if (ib) ib->Release();
    if (vb) vb->Release();
    if (decl) decl->Release();
    if (ps) ps->Release();
    if (vs) vs->Release();
    if (ds) ds->Release();
    if (sys) sys->Release();
    if (rt) rt->Release();
    return ok;
  }
};

static void run(Backend& be, D3DRef& ref, const std::string& cacheDir) {
  // A fresh D3D9 device per test run keeps its state at the documented defaults; each case restores what it changed
  // by recreating the device state through a state block captured at the start.
  IDirect3DStateBlock9* defaults = nullptr;
  ref.dev->CreateStateBlock(D3DSBT_ALL, &defaults);
  D3DDraw d3d{ ref.dev };
  if (!d3d.init()) { fail("draw test: D3D9 texture"); return; }

  d3dx::Texture tex;
  tex.format = d3dx::kFmtA8R8G8B8;
  tex.width = d3d.tw;
  tex.height = d3d.th;
  tex.levels = 1;
  tex.cube = false;
  d3dx::Level lv;
  lv.width = d3d.tw;
  lv.height = d3d.th;
  lv.rowPitch = d3d.tw * 4;
  lv.rows = d3d.th;
  lv.data.resize(d3d.texels.size() * 4);
  memcpy(lv.data.data(), d3d.texels.data(), lv.data.size());
  tex.surfaces.push_back(lv);
  const TexKey texKey = 0x7E000;
  if (!be.createTexture(texKey, tex)) { fail("draw test: backend texture"); return; }
  const TexKey mipKey = 0x7E040;
  {
    d3dx::Texture mt;
    mt.format = d3dx::kFmtA8R8G8B8;
    mt.width = mt.height = 256;
    mt.levels = 9;
    mt.cube = false;
    for (uint32_t l = 0; l < 9; l++) {
      uint32_t s = 256 >> l;
      d3dx::Level ml;
      ml.width = ml.height = s;
      ml.rowPitch = s * 4;
      ml.rows = s;
      ml.data.resize(size_t(s) * s * 4);
      memcpy(ml.data.data(), d3d.mipLevels[l].data(), ml.data.size());
      mt.surfaces.push_back(ml);
    }
    if (!be.createTexture(mipKey, mt)) { fail("draw test: backend mip texture"); return; }
  }
  const DeclKey declKey = 0x7D000;
  const VertexElement ve[] = { { 0, 0, D3DDECLTYPE_FLOAT3, 0, D3DDECLUSAGE_POSITION, 0 },
                               { 0, 12, D3DDECLTYPE_FLOAT2, 0, D3DDECLUSAGE_TEXCOORD, 0 },
                               { 0, 20, D3DDECLTYPE_D3DCOLOR, 0, D3DDECLUSAGE_COLOR, 0 } };
  be.createDecl(declKey, ve, 3);
  const DeclKey declKey2 = 0x7D040;
  be.createDecl(declKey2, ve, 2);

  std::vector<Case> cases;
  // Screen-space mapping for ModelViewProj (c0..c3), applied by the shader as dp4 rows.
  const float sx = 2.0f / kW, sy = -2.0f / kH;
  auto mvp = [&](Case& c) {
    c.vsc.push_back({ 0, { sx, 0, 0, -1 } });
    c.vsc.push_back({ 1, { 0, sy, 0, 1 } });
    c.vsc.push_back({ 2, { 0, 0, 1, 0 } });
    c.vsc.push_back({ 3, { 0, 0, 0, 1 } });
  };
  auto tri = [](std::vector<Vtx>& v, float x0, float y0, float x1, float y1, float x2, float y2, float z, uint32_t col,
                float uvScale = 1.0f) {
    v.push_back({ x0, y0, z, x0 / 16 * uvScale, y0 / 16 * uvScale, col });
    v.push_back({ x1, y1, z, x1 / 16 * uvScale, y1 / 16 * uvScale, col });
    v.push_back({ x2, y2, z, x2 / 16 * uvScale, y2 / 16 * uvScale, col });
  };
  auto quad = [&](std::vector<Vtx>& v, float x0, float y0, float x1, float y1, float z, uint32_t col, float uvs = 1.0f) {
    tri(v, x0, y0, x1, y0, x0, y1, z, col, uvs);  // clockwise on screen (y down)
    tri(v, x1, y0, x1, y1, x0, y1, z, col, uvs);
  };
  auto base = [&](const char* name) {
    Case c;
    c.name = name;
    c.rs = { { D3DRS_CULLMODE, D3DCULL_NONE }, { D3DRS_ZENABLE, FALSE }, { D3DRS_LIGHTING, FALSE } };
    mvp(c);
    c.psc.push_back({ 0, { 1.0f, 0.8f, 0.6f, 0.9f } });
    return c;
  };
  {
    Case c = base("tile point sampling");
    quad(c.verts, 4.3f, 3.7f, 60.2f, 50.9f, 0.5f, 0xFFFFFFFF);
    tri(c.verts, 50, 10, 90, 20, 70, 60, 0.5f, 0xC08040FF);
    cases.push_back(c);
  }
  {
    Case c = base("tile linear wrap blend");
    c.ss = { { D3DSAMP_MINFILTER, D3DTEXF_LINEAR }, { D3DSAMP_MAGFILTER, D3DTEXF_LINEAR }, { D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP },
             { D3DSAMP_ADDRESSV, D3DTADDRESS_MIRROR } };
    c.rs.push_back({ D3DRS_ALPHABLENDENABLE, TRUE });
    c.rs.push_back({ D3DRS_SRCBLEND, D3DBLEND_SRCALPHA });
    c.rs.push_back({ D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA });
    quad(c.verts, 2, 2, 94, 62, 0.5f, 0xE0FFFFFF, 3.3f);
    cases.push_back(c);
  }
  {
    Case c = base("tile alpha test + clamp");
    c.ss = { { D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP }, { D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP } };
    c.rs.push_back({ D3DRS_ALPHATESTENABLE, TRUE });
    c.rs.push_back({ D3DRS_ALPHAFUNC, D3DCMP_GREATER });
    c.rs.push_back({ D3DRS_ALPHAREF, 0x80 });
    quad(c.verts, 0, 0, 96, 64, 0.5f, 0xFFFFFFFF, 1.7f);
    cases.push_back(c);
  }
  {
    Case c = base("cull ccw, mixed winding");
    c.rs[0] = { D3DRS_CULLMODE, D3DCULL_CCW };
    tri(c.verts, 5, 5, 40, 5, 5, 40, 0.5f, 0xFFFF0000);     // clockwise on screen: kept
    tri(c.verts, 50, 5, 50, 40, 90, 5, 0.5f, 0xFF00FF00);   // counter-clockwise: culled
    cases.push_back(c);
  }
  {
    Case c = base("depth less + write, D24S8");
    c.depth = true;
    c.rs[1] = { D3DRS_ZENABLE, TRUE };
    c.rs.push_back({ D3DRS_ZFUNC, D3DCMP_LESS });
    c.rs.push_back({ D3DRS_ZWRITEENABLE, TRUE });
    quad(c.verts, 10, 10, 70, 50, 0.6f, 0xFFFF0000);
    quad(c.verts, 30, 20, 90, 60, 0.3f, 0xFF00FF00);
    quad(c.verts, 0, 0, 50, 30, 0.45f, 0xFF0000FF);
    cases.push_back(c);
  }
  {
    Case c = base("scissor + colour write mask");
    c.rs.push_back({ D3DRS_SCISSORTESTENABLE, TRUE });
    c.rs.push_back({ D3DRS_COLORWRITEENABLE, D3DCOLORWRITEENABLE_RED | D3DCOLORWRITEENABLE_ALPHA });
    c.scissor = { 13, 7, 77, 51 };
    quad(c.verts, 0, 0, 96, 64, 0.5f, 0xFFFFFFFF);
    cases.push_back(c);
  }
  {
    Case c = base("viewport offset, X8 target, dest-alpha blend");
    c.rtFormat = kD3DFMT_X8R8G8B8;
    c.viewport = { 8, 4, 64, 48, 0.0f, 1.0f };
    c.rs.push_back({ D3DRS_ALPHABLENDENABLE, TRUE });
    c.rs.push_back({ D3DRS_SRCBLEND, D3DBLEND_DESTALPHA });
    c.rs.push_back({ D3DRS_DESTBLEND, D3DBLEND_INVDESTALPHA });
    quad(c.verts, 0, 0, 96, 64, 0.5f, 0x80FFFFFF);
    cases.push_back(c);
  }
  {
    Case c = base("TILE1000 + D24S8 + blend (first in-game pipeline)");
    c.tile1000 = true;
    c.depth = true;
    c.rs.push_back({ D3DRS_ALPHABLENDENABLE, TRUE });
    c.rs.push_back({ D3DRS_SRCBLEND, D3DBLEND_SRCALPHA });
    c.rs.push_back({ D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA });
    quad(c.verts, 4, 4, 92, 60, 0.5f, 0xFFFFFFFF);
    cases.push_back(c);
  }
  {
    D3DCAPS9 caps{};
    ref.dev->GetDeviceCaps(&caps);
    printf("D3D9 caps: TextureFilterCaps %08lX (MINFANISOTROPIC %d, MAGFANISOTROPIC %d) MaxAnisotropy %lu\n",
           caps.TextureFilterCaps, (caps.TextureFilterCaps & D3DPTFILTERCAPS_MINFANISOTROPIC) != 0,
           (caps.TextureFilterCaps & D3DPTFILTERCAPS_MAGFANISOTROPIC) != 0, caps.MaxAnisotropy);
  }
  for (int aniso : { 0, 15, 16, 8, -15 }) {
    static char names[8][64];
    static int ni = 0;
    char* name = names[ni++ & 7];
    const bool magAniso = aniso < 0;
    aniso = aniso < 0 ? -aniso : aniso;
    snprintf(name, 64, aniso ? (magAniso ? "floor, anisotropic %d, mag anisotropic" : "floor, anisotropic %d") : "floor, trilinear (control)", aniso);
    Case c = base(name);
    c.mipTexture = true;
    // Perspective floor: oPos = (dp4 c0..c3) of (x, y, z, 1) -> clip (f x, f y, a z + b, z).
    const float f = 1.5f, n = 0.5f, fr = 100.0f, a = fr / (fr - n), b = -n * a;
    c.vsc = { { 0, { f, 0, 0, 0 } }, { 1, { 0, f, 0, 0 } }, { 2, { 0, 0, a, b } }, { 3, { 0, 0, 1, 0 } } };
    c.ss = { { D3DSAMP_MINFILTER, DWORD(aniso ? D3DTEXF_ANISOTROPIC : D3DTEXF_LINEAR) }, { D3DSAMP_MAGFILTER, DWORD(magAniso ? D3DTEXF_ANISOTROPIC : D3DTEXF_LINEAR) },
             { D3DSAMP_MIPFILTER, D3DTEXF_LINEAR }, { D3DSAMP_MAXANISOTROPY, DWORD(aniso ? aniso : 1) },
             { D3DSAMP_ADDRESSU, D3DTADDRESS_WRAP }, { D3DSAMP_ADDRESSV, D3DTADDRESS_WRAP } };
    const float x0 = -2, x1 = 2, z0 = 1, z1 = 40, y = -1, uvs = 0.25f;
    c.verts = { { x0, y, z0, x0 * uvs, z0 * uvs, 0xFFFFFFFF }, { x1, y, z0, x1 * uvs, z0 * uvs, 0xFFFFFFFF },
                { x0, y, z1, x0 * uvs, z1 * uvs, 0xFFFFFFFF }, { x1, y, z0, x1 * uvs, z0 * uvs, 0xFFFFFFFF },
                { x1, y, z1, x1 * uvs, z1 * uvs, 0xFFFFFFFF }, { x0, y, z1, x0 * uvs, z1 * uvs, 0xFFFFFFFF } };
    cases.push_back(c);
  }
  {
    Case c = base("indexed strip");
    c.prim = D3DPT_TRIANGLESTRIP;
    c.verts = { { 10, 10, 0.5f, 0, 0, 0xFFFFFFFF }, { 80, 12, 0.5f, 2, 0, 0xFF00FFFF },
                { 12, 55, 0.5f, 0, 2, 0xFFFF00FF }, { 85, 58, 0.5f, 2, 2, 0xFFFFFF00 } };
    c.indices = { 0, 1, 2, 3 };
    cases.push_back(c);
  }

  // Resolve shader md5s by name from the manifest's package index (package 002).
  std::vector<uint8_t> manifest;
  if (!readFile(cacheDir + "/manifest.json", manifest)) { fail("draw test: manifest"); return; }
  std::string mf(manifest.begin(), manifest.end());
  auto md5ByName = [&](const char* name) -> std::string {
    std::string key = std::string("\"") + name + "\": \"";
    size_t p = mf.find(key);
    return p == std::string::npos ? std::string() : mf.substr(p + key.size(), 32);
  };
  const std::string vs1002 = md5ByName("TILE1002.vso"), ps1002 = md5ByName("TILE1002.pso");
  const std::string vs1000 = md5ByName("TILE1000.vso"), ps1000 = md5ByName("TILE1000.pso");
  if (vs1002.empty() || ps1002.empty() || vs1000.empty() || ps1000.empty()) { fail("draw test: TILE shaders not in manifest"); return; }

  int idx = 0, passed = 0, queriesMatched = 0;
  for (Case& c : cases) {
    idx++;
    const std::string& tileVs = c.tile1000 ? vs1000 : vs1002;
    const std::string& tilePs = c.tile1000 ? ps1000 : ps1002;
    std::vector<uint8_t> vsCode, psCode;
    if (!readFile(cacheDir + "/spv/" + tileVs + ".dxso", vsCode) || !readFile(cacheDir + "/spv/" + tilePs + ".dxso", psCode)) {
      fail("draw test: bytecode files");
      return;
    }
    defaults->Apply();
    std::vector<uint8_t> a;
    uint32_t rs[256] = {};
    uint32_t ss[16][14] = {};
    if (!d3d.run(c, vsCode, psCode, a, rs, ss)) { fail("draw test '%s': D3D9 run failed", c.name); continue; }

    uint8_t vmd5[16], pmd5[16];
    hexMd5(tileVs, vmd5);
    hexMd5(tilePs, pmd5);
    const ShaderKey vsKey = 0x7C000 + idx * 0x40, psKey = 0x7B000 + idx * 0x40;
    if (!be.createShader(vsKey, vmd5) || !be.createShader(psKey, pmd5)) { fail("draw test: shader not in pack"); return; }
    const BufKey vbKey = 0x7A000 + idx * 0x40, ibKey = 0x79000 + idx * 0x40;
    const std::vector<uint8_t> vbytes = vertexBytes(c);
    be.createBuffer(vbKey, uint32_t(vbytes.size()), false);
    be.writeBuffer(vbKey, 0, vbytes.data(), uint32_t(vbytes.size()), 0);
    if (!c.indices.empty()) {
      be.createBuffer(ibKey, uint32_t(c.indices.size() * 2), true);
      be.writeBuffer(ibKey, 0, c.indices.data(), uint32_t(c.indices.size() * 2), 0);
    }
    RtKey rt = newKey(), dsk = c.depth ? newKey() : 0;
    be.attach(rt, { kW, kH, c.rtFormat, 1, false });
    if (dsk) be.attach(dsk, { kW, kH, kD3DFMT_D24S8, 1, true });
    static float vsc[256][4], psc[224][4];
    memset(vsc, 0, sizeof(vsc));
    memset(psc, 0, sizeof(psc));
    for (auto& [r, v] : c.vsc) memcpy(vsc[r], v.data(), 16);
    for (auto& [r, v] : c.psc) memcpy(psc[r], v.data(), 16);
    DrawSetup s;
    s.renderStates = rs;
    s.samplerStates = ss;
    s.textures[0] = { DrawSetup::Texture::Tex, c.mipTexture ? mipKey : texKey };
    s.textures[1] = { DrawSetup::Texture::Tex, texKey };
    s.vs = vsKey;
    s.ps = psKey;
    s.decl = c.tile1000 ? declKey2 : declKey;
    s.streams[0] = { vbKey, 0, strideOf(c) };
    s.indices = c.indices.empty() ? 0 : ibKey;
    s.vsConstants = &vsc[0][0];
    s.psConstants = &psc[0][0];
    s.vsConstantsVersion = s.psConstantsVersion = uint64_t(idx);
    D3DVIEWPORT9 vp = c.viewport.Width ? c.viewport : D3DVIEWPORT9{ 0, 0, kW, kH, 0.0f, 1.0f };
    s.viewport = { int32_t(vp.X), int32_t(vp.Y), vp.Width, vp.Height, vp.MinZ, vp.MaxZ };
    s.scissor[0] = c.scissor.left;
    s.scissor[1] = c.scissor.top;
    s.scissor[2] = c.scissor.right;
    s.scissor[3] = c.scissor.bottom;
    be.beginGroup(&rt, 1, dsk);
    int32_t full[4] = { 0, 0, int32_t(kW), int32_t(kH) };
    be.clear(kClearTarget | (dsk ? kClearZBuffer | kClearStencil : 0), 0xFF203040, 1.0f, 0, full);
    uint32_t prims = c.prim == D3DPT_TRIANGLELIST ? uint32_t((c.indices.empty() ? c.verts.size() : c.indices.size()) / 3)
                                                  : uint32_t((c.indices.empty() ? c.verts.size() : c.indices.size()) - 2);
    const uint32_t qid = be.createQuery();
    be.beginQuery(qid);
    be.draw(s, c.prim, prims, !c.indices.empty(), 0, 0);
    be.endQuery(qid);
    be.endGroup();
    std::vector<uint8_t> b;
    uint32_t vf = 0, ts = 0;
    const bool rbOk = be.readback(rt, b, vf, ts);
    {
      uint32_t samples = ~0u;
      be.finishFrame();
      if (be.queryResult(qid, samples) != 1) fail("draw test '%s': occlusion query not ready after the frame", c.name);
      else if (samples != d3d.lastQuery) fail("draw test '%s': occlusion query d3d %lu vk %u", c.name, d3d.lastQuery, samples);
      else queriesMatched++;
      be.destroyQuery(qid);
    }
    if (!rbOk || b.size() != a.size()) { fail("draw test '%s': backend readback", c.name); continue; }
    if (const char* dir = getenv("FNV_DUMP_DIR")) {
      // Raw BGRA dumps of both images (kW x kH) for inspection.
      for (int k = 0; k < 2; k++) {
        std::string path = std::string(dir) + "/case" + std::to_string(idx) + (k ? "_vk.raw" : "_d3d.raw");
        if (FILE* f = fopen(path.c_str(), "wb")) { fwrite((k ? b : a).data(), 1, a.size(), f); fclose(f); }
      }
    }
    // Compare (X8: the X byte is undefined in D3D9).
    size_t diff = 0, covered = 0;
    int maxd = 0, firstX = -1, firstY = -1;
    for (size_t i = 0; i < a.size(); i += 4) {
      bool changed = false;
      for (int ch = 0; ch < 4; ch++) {
        if (c.rtFormat == kD3DFMT_X8R8G8B8 && ch == 3) continue;
        int d = std::abs(int(a[i + ch]) - int(b[i + ch]));
        if (d) {
          changed = true;
          maxd = std::max(maxd, d);
        }
      }
      uint32_t ap;
      memcpy(&ap, &a[i], 4);
      if ((ap & 0xFFFFFF) != 0x203040) covered++;
      if (changed) {
        if (!diff) { firstX = int((i / 4) % kW); firstY = int((i / 4) / kW); }
        diff++;
      }
    }
    if (!covered) fail("draw test '%s': D3D9 drew nothing (case is not testing anything)", c.name);
    if (diff) {
      uint32_t pa, pb;
      memcpy(&pa, &a[(size_t(firstY) * kW + firstX) * 4], 4);
      memcpy(&pb, &b[(size_t(firstY) * kW + firstX) * 4], 4);
      fail("draw test '%s': %zu of %u texels differ (max channel diff %d; first (%d,%d) d3d %08X vk %08X), %zu covered",
           c.name, diff, kW * kH, maxd, firstX, firstY, pa, pb, covered);
    } else {
      passed++;
      printf("draw test '%s': identical (%zu texels covered)\n", c.name, covered);
    }
    be.destroy(rt);
    if (dsk) be.destroy(dsk);
    be.destroyBuffer(vbKey);
    if (!c.indices.empty()) be.destroyBuffer(ibKey);
    be.destroyShader(vsKey);
    be.destroyShader(psKey);
  }
  printf("occlusion queries: %d match D3D9\n", queriesMatched);

  // Dynamic geometry: one buffer rewritten between draws of the same frame, as the game's dynamic vertex sets are.
  // DISCARD renames (recycled versions, mapped after repeated renames), NOOVERWRITE appends into a version in use, and
  // a partial write to a version in use (the old contents carried over by a GPU copy). Every draw must render D3D9's
  // image of the same case.
  {
    Case& c = cases[0];
    std::vector<uint8_t> vsCode, psCode, a;
    readFile(cacheDir + "/spv/" + vs1002 + ".dxso", vsCode);
    readFile(cacheDir + "/spv/" + ps1002 + ".dxso", psCode);
    defaults->Apply();
    uint32_t rs[256] = {};
    uint32_t ss[16][14] = {};
    if (c.tile1000 || !c.indices.empty() || !d3d.run(c, vsCode, psCode, a, rs, ss)) {
      fail("dynamic geometry test: reference case unusable");
    } else {
      uint8_t vmd5[16], pmd5[16];
      hexMd5(vs1002, vmd5);
      hexMd5(ps1002, pmd5);
      const ShaderKey vsKey = 0x7C800, psKey = 0x7B800;
      be.createShader(vsKey, vmd5);
      be.createShader(psKey, pmd5);
      const std::vector<uint8_t> v = vertexBytes(c);
      const uint32_t vsz = uint32_t(v.size()), total = vsz * 4;
      const BufKey dyn = 0x78800;
      be.createBuffer(dyn, total, false);
      const std::vector<uint8_t> junk(total, 0xCD);
      static float vsc[256][4], psc[224][4];
      memset(vsc, 0, sizeof(vsc));
      memset(psc, 0, sizeof(psc));
      for (auto& [r, val] : c.vsc) memcpy(vsc[r], val.data(), 16);
      for (auto& [r, val] : c.psc) memcpy(psc[r], val.data(), 16);
      DrawSetup s;
      s.renderStates = rs;
      s.samplerStates = ss;
      s.textures[0] = { DrawSetup::Texture::Tex, c.mipTexture ? mipKey : texKey };
      s.textures[1] = { DrawSetup::Texture::Tex, texKey };
      s.vs = vsKey;
      s.ps = psKey;
      s.decl = declKey;
      s.vsConstants = &vsc[0][0];
      s.psConstants = &psc[0][0];
      s.vsConstantsVersion = s.psConstantsVersion = 1000;
      D3DVIEWPORT9 vp = c.viewport.Width ? c.viewport : D3DVIEWPORT9{ 0, 0, kW, kH, 0.0f, 1.0f };
      s.viewport = { int32_t(vp.X), int32_t(vp.Y), vp.Width, vp.Height, vp.MinZ, vp.MaxZ };
      memcpy(s.scissor, &c.scissor, sizeof(s.scissor));
      const uint32_t prims = c.prim == D3DPT_TRIANGLELIST ? uint32_t(c.verts.size() / 3) : uint32_t(c.verts.size() - 2);
      const BackendStats before = be.stats();
      int drawsChecked = 0, drawsWrong = 0;
      for (uint32_t f = 0; f < 12; f++) {
        RtKey rts[4];
        auto drawInto = [&](int i, uint32_t off) {
          rts[i] = newKey();
          be.attach(rts[i], { kW, kH, c.rtFormat, 1, false });
          s.streams[0] = { dyn, off, strideOf(c) };
          be.beginGroup(&rts[i], 1, 0);
          int32_t full[4] = { 0, 0, int32_t(kW), int32_t(kH) };
          be.clear(kClearTarget, 0xFF203040, 1.0f, 0, full);
          be.draw(s, c.prim, prims, false, 0, 0);
          be.endGroup();
        };
        be.writeBuffer(dyn, 0, junk.data(), total, kLockDiscard);
        be.writeBuffer(dyn, 0, v.data(), vsz, kLockNoOverwrite);
        drawInto(0, 0);
        be.writeBuffer(dyn, 0, junk.data(), total, kLockDiscard);       // in use: renamed
        be.writeBuffer(dyn, vsz, v.data(), vsz, kLockNoOverwrite);
        drawInto(1, vsz);
        be.writeBuffer(dyn, 2 * vsz, v.data(), vsz, kLockNoOverwrite);  // append into the version in use
        drawInto(2, 2 * vsz);
        be.writeBuffer(dyn, 3 * vsz, v.data(), vsz, 0);                 // partial, in use: renamed, contents kept
        drawInto(3, vsz);                                               // reads the carried-over range
        for (int i = 0; i < 4; i++) {
          std::vector<uint8_t> b;
          uint32_t vf = 0, ts = 0;
          drawsChecked++;
          if (!be.readback(rts[i], b, vf, ts) || b.size() != a.size()) {
            drawsWrong++;
          } else {
            size_t diff = 0;
            for (size_t p = 0; p < a.size(); p++)
              if (!(c.rtFormat == kD3DFMT_X8R8G8B8 && p % 4 == 3) && a[p] != b[p]) diff++;
            if (diff) {
              drawsWrong++;
              if (drawsWrong <= 4) printf("dynamic geometry: frame %u draw %d: %zu bytes differ from D3D9\n", f, i, diff);
            }
          }
          be.destroy(rts[i]);
        }
      }
      const BackendStats after = be.stats();
      const uint64_t renames = after.bufferRenames - before.bufferRenames, recycled = after.bufferRecycles - before.bufferRecycles,
                     direct = after.bufferDirectWrites - before.bufferDirectWrites, created = after.buffersCreated - before.buffersCreated;
      printf("dynamic geometry: %d/%d draws identical to D3D9 | renames %llu (recycled %llu), direct writes %llu, pool full %llu, "
             "buffers created %llu\n", drawsChecked - drawsWrong, drawsChecked, renames, recycled, direct,
             after.mappedFallbacks - before.mappedFallbacks, created);
      if (drawsWrong) fail("dynamic geometry: %d of %d draws differ from D3D9", drawsWrong, drawsChecked);
      if (!renames || !recycled) fail("dynamic geometry: renames %llu, recycled %llu (both expected)", renames, recycled);
      if (!direct) printf("dynamic geometry: no direct writes (no host-visible pool on this device?)\n");
      be.destroyBuffer(dyn);
      be.destroyShader(vsKey);
      be.destroyShader(psKey);
    }
  }
  // Effects from data (effects/ in the repository, as the plugin loads them): manifest, generated header, glslang, backend.
  std::vector<fx::Effect> loaded = fx::loadEffects("effects", "build/fx_cache_test", [](const char* s) { printf("[fx] %s\n", s); });
  const fx::Effect* gtaoFx = nullptr;
  for (const fx::Effect& e : loaded)
    if (e.id == "gtao" && e.error.empty()) gtaoFx = &e;
  if (!gtaoFx) fail("effects: effects/gtao did not load");
  uint32_t gtaoId = 0;
  if (gtaoFx) {
    Backend::EffectDesc d;
    d.sceneColor = gtaoFx->sceneColor;
    for (const fx::Image& im : gtaoFx->images) d.images.push_back({ im.vkFormat, im.scale });
    for (const fx::Pass& p : gtaoFx->passes)
      d.passes.push_back({ { p.spv[0].data(), p.spv[1].data() }, { p.spv[0].size(), p.spv[1].size() }, p.output, uint32_t(p.blend) });
    gtaoId = be.createEffect(d);
    if (!gtaoId) fail("effects: the backend did not create gtao");
  }
  // The full-resolution GTAO this effect replaced (tools/backend_test/gtao_full: its passes before the half-resolution
  // version), to measure how far the half-resolution AO is from it on the same depth.
  std::vector<fx::Effect> fullFx = fx::loadEffects("tools/backend_test", "build/fx_cache_test", [](const char*) {});
  const fx::Effect* gtaoFull = nullptr;
  for (const fx::Effect& e : fullFx)
    if (e.id == "gtao_full" && e.error.empty()) gtaoFull = &e;
  uint32_t gtaoFullId = 0;
  if (!gtaoFull) fail("effects: tools/backend_test/gtao_full did not load");
  else {
    Backend::EffectDesc d;
    for (const fx::Image& im : gtaoFull->images) d.images.push_back({ im.vkFormat, im.scale });
    for (const fx::Pass& p : gtaoFull->passes)
      d.passes.push_back({ { p.spv[0].data(), p.spv[1].data() }, { p.spv[0].size(), p.spv[1].size() }, p.output, uint32_t(p.blend) });
    gtaoFullId = be.createEffect(d);
  }
  auto imageIndex = [](const fx::Effect* e, const char* name) {
    for (size_t i = 0; e && i < e->images.size(); i++)
      if (e->images[i].name == name) return uint32_t(i);
    return ~0u;
  };
  // A shader error must name the file and line.
  {
    CreateDirectoryA("build\\fx_broken", nullptr);
    CreateDirectoryA("build\\fx_broken\\bad", nullptr);
    if (FILE* f = fopen("build/fx_broken/bad/effect.ini", "wb")) {
      fputs("[Effect]\nName=Broken\n[Pass.0]\nShader=bad.frag\nOutput=scene\n", f);
      fclose(f);
    }
    if (FILE* f = fopen("build/fx_broken/bad/bad.frag", "wb")) {
      fputs("layout(location = 0) out vec4 c;\nvoid main() {\n  c = vec4(undefinedThing);\n}\n", f);
      fclose(f);
    }
    std::vector<fx::Effect> bad = fx::loadEffects("build/fx_broken", "build/fx_cache_test", [](const char*) {});
    const bool named = bad.size() == 1 && bad[0].error.find("bad.frag:3") != std::string::npos;
    printf("effects: broken shader reported as: %s\n", bad.empty() ? "(none)" : bad[0].error.substr(0, 120).c_str());
    if (!named) fail("effects: compile error does not name bad.frag:3");
  }

  // GTAO on a known depth scene (colour cleared to white, geometry writes depth only, so the result is the visibility):
  // x 0..48 a concave V crease (two view-space planes, deepest at x 24), x 48..56 nothing (cleared depth = sky),
  // x 56..96 a plane facing the camera. GTAO's integral is exactly 1 for a flat unoccluded surface; the crease must
  // darken toward its centre; the sky must stay untouched. Run without and with 4x MSAA.
  for (uint32_t samples : { 1u, 4u }) {
    if (!gtaoId) break;
    Case c = cases[0];
    c.name = samples > 1 ? "gtao 4x msaa" : "gtao";
    c.depth = true;
    c.rs = { { D3DRS_CULLMODE, D3DCULL_NONE }, { D3DRS_ZENABLE, TRUE }, { D3DRS_ZWRITEENABLE, TRUE },
             { D3DRS_ZFUNC, D3DCMP_LESSEQUAL }, { D3DRS_COLORWRITEENABLE, 0 }, { D3DRS_LIGHTING, FALSE } };
    c.verts.clear();
    auto vtx = [&](float x, float y, float z) { c.verts.push_back({ x, y, z, 0.0f, 0.0f, 0xFFFFFFFF }); };
    auto plane = [&](float x0, float z0, float x1, float z1) {  // full height, depth linear in x
      vtx(x0, 0, z0); vtx(x1, 0, z1); vtx(x0, float(kH), z0);
      vtx(x1, 0, z1); vtx(x1, float(kH), z1); vtx(x0, float(kH), z0);
    };
    plane(0, 0.30f, 24, 0.70f);
    plane(24, 0.70f, 48, 0.30f);
    plane(56, 0.50f, 96, 0.50f);
    std::vector<uint8_t> vsCode, psCode, a;
    readFile(cacheDir + "/spv/" + vs1002 + ".dxso", vsCode);
    readFile(cacheDir + "/spv/" + ps1002 + ".dxso", psCode);
    defaults->Apply();
    uint32_t rs[256] = {};
    uint32_t ss[16][14] = {};
    if (!d3d.run(c, vsCode, psCode, a, rs, ss)) { fail("gtao test: D3D9 state capture failed"); continue; }
    uint8_t vmd5[16], pmd5[16];
    hexMd5(vs1002, vmd5);
    hexMd5(ps1002, pmd5);
    const ShaderKey vsKey = 0x7CC00 + samples, psKey = 0x7BC00 + samples;
    be.createShader(vsKey, vmd5);
    be.createShader(psKey, pmd5);
    const std::vector<uint8_t> v = vertexBytes(c);
    const BufKey vb = 0x78C00 + samples;
    be.createBuffer(vb, uint32_t(v.size()), false);
    be.writeBuffer(vb, 0, v.data(), uint32_t(v.size()), 0);
    static float vsc[256][4], psc[224][4];
    memset(vsc, 0, sizeof(vsc));
    memset(psc, 0, sizeof(psc));
    for (auto& [r, val] : c.vsc) memcpy(vsc[r], val.data(), 16);
    for (auto& [r, val] : c.psc) memcpy(psc[r], val.data(), 16);
    DrawSetup s;
    s.renderStates = rs;
    s.samplerStates = ss;
    s.textures[0] = { DrawSetup::Texture::Tex, texKey };
    s.vs = vsKey;
    s.ps = psKey;
    s.decl = declKey;
    s.streams[0] = { vb, 0, strideOf(c) };
    s.vsConstants = &vsc[0][0];
    s.psConstants = &psc[0][0];
    s.vsConstantsVersion = s.psConstantsVersion = 2000 + samples;
    s.viewport = { 0, 0, kW, kH, 0.0f, 1.0f };
    RtKey rt = newKey(), dsk = newKey();
    be.attach(rt, { kW, kH, kD3DFMT_A8R8G8B8, samples, false });
    be.attach(dsk, { kW, kH, kD3DFMT_D24S8, samples, true });
    be.beginGroup(&rt, 1, dsk);
    int32_t full[4] = { 0, 0, int32_t(kW), int32_t(kH) };
    be.clear(kClearTarget | kClearZBuffer | kClearStencil, 0xFFFFFFFF, 1.0f, 0, full);
    be.draw(s, D3DPT_TRIANGLELIST, uint32_t(c.verts.size() / 3), false, 0, 0);
    Backend::EffectFrame fr{};
    const float aspect = float(kH) / float(kW);
    fr.frustum[0] = -1.0f; fr.frustum[1] = 1.0f; fr.frustum[2] = aspect; fr.frustum[3] = -aspect;
    fr.nearZ = 1.0f;
    fr.farZ = 100.0f;
    fr.port[0] = 0.0f; fr.port[1] = 1.0f; fr.port[2] = 1.0f; fr.port[3] = 0.0f;
    // gtao parameters by name: radius 1, intensity 1, 4 slices, 8 steps, fade 1000..2000.
    float params[32][4] = {};
    const std::pair<const char*, float> values[] = { { "radius", 1.0f }, { "intensity", 1.0f }, { "slices", 4.0f },
                                                     { "steps", 8.0f }, { "fadeStart", 1000.0f }, { "fadeEnd", 2000.0f } };
    for (size_t k = 0; k < gtaoFx->params.size(); k++) {
      params[k][0] = gtaoFx->params[k].def;
      for (const auto& [name, v] : values)
        if (gtaoFx->params[k].name == name) params[k][0] = v;
    }
    const bool applied = be.runEffect(gtaoId, fr, &params[0][0], uint32_t(gtaoFx->params.size()));
    be.endGroup();
    std::vector<uint8_t> b;
    uint32_t vf = 0, ts = 0;
    if (!applied) {
      fail("%s: not applied", c.name);
    } else if (!be.readback(rt, b, vf, ts) || ts != 4 || b.size() < size_t(kW) * kH * 4) {
      fail("%s: readback", c.name);
    } else {
      auto vis = [&](uint32_t x, uint32_t y) { return b[(size_t(y) * kW + x) * 4 + 2] / 255.0f; };  // BGRA: R
      float creaseMax = 0, flatMin = 1, skyMin = 1, rimMean = 0;
      for (uint32_t y = 8; y < kH - 8; y++) {
        for (uint32_t x = 22; x <= 26; x++) creaseMax = std::max(creaseMax, vis(x, y));
        for (uint32_t x = 78; x <= 90; x++) flatMin = std::min(flatMin, vis(x, y));  // beyond one radius (24 px) of the crease
        for (uint32_t x = 50; x <= 54; x++) skyMin = std::min(skyMin, vis(x, y));
        rimMean += vis(4, y);
      }
      rimMean /= float(kH - 16);
      printf("%s: crease centre visibility <= %.3f, crease rim %.3f, flat >= %.3f, sky >= %.3f\n", c.name, creaseMax, rimMean,
             flatMin, skyMin);
      if (skyMin < 1.0f) fail("%s: sky changed (%.3f)", c.name, skyMin);
      if (flatMin < 0.985f) fail("%s: flat plane occluded (%.3f)", c.name, flatMin);  // GTAO integrates to 1 there (8-bit storage)
      if (creaseMax > 0.85f) fail("%s: crease not darkened (%.3f)", c.name, creaseMax);
      if (creaseMax >= rimMean) fail("%s: crease centre not darker than its rim", c.name);
      // The published AO (full resolution in both) against the full-resolution GTAO's, on the same depth: the
      // half-resolution computation changes the image this much.
      be.beginGroup(&rt, 1, dsk);
      const bool ranFull = gtaoFullId && be.runEffect(gtaoFullId, fr, &params[0][0], uint32_t(gtaoFx->params.size()));
      be.endGroup();
      std::vector<uint8_t> half, full;
      uint32_t w1 = 0, h1 = 0, w2 = 0, h2 = 0;
      if (!ranFull || !be.readbackEffectImage(gtaoId, imageIndex(gtaoFx, "blurred"), half, w1, h1) ||
          !be.readbackEffectImage(gtaoFullId, imageIndex(gtaoFull, "blurred"), full, w2, h2) || w1 != kW || h1 != kH ||
          w2 != kW || h2 != kH || half.size() != full.size()) {
        fail("%s: AO images for the comparison", c.name);
      } else {
        std::vector<float> diffs;
        double sum = 0, region[3] = {}, regionMax[3] = {};
        int regionN[3] = {}, worstX = 0, worstY = 0;
        float worstHalf = 0, worstFull = 0;
        for (size_t i = 0; i < size_t(kW) * kH; i++) {
          const float a = drawtestHalf(reinterpret_cast<const uint16_t*>(half.data())[i * 2]);
          const float b2 = drawtestHalf(reinterpret_cast<const uint16_t*>(full.data())[i * 2]);
          const float dd = std::fabs(a - b2);
          if (dd > (diffs.empty() ? -1.0f : *std::max_element(diffs.begin(), diffs.end()))) {
            worstX = int(i % kW);
            worstY = int(i / kW);
            worstHalf = a;
            worstFull = b2;
          }
          diffs.push_back(dd);
          sum += dd;
          const uint32_t x = uint32_t(i % kW);
          const int r = x < 48 ? 0 : x < 56 ? 1 : 2;  // the crease, the sky gap, the flat plane
          region[r] += dd;
          regionN[r]++;
          regionMax[r] = std::max(regionMax[r], double(dd));
        }
        std::sort(diffs.begin(), diffs.end());
        const float p99 = diffs[size_t(diffs.size() * 0.99)], mx = diffs.back();
        const double mean = sum / diffs.size();
        printf("%s: half resolution vs full resolution AO: mean difference %.4f, 99th percentile %.4f, max %.4f at (%d, %d) "
               "(%.3f vs %.3f); by region mean / max: crease %.4f / %.4f, sky gap %.4f / %.4f, flat %.4f / %.4f\n", c.name,
               mean, p99, mx, worstX, worstY, worstHalf, worstFull, region[0] / regionN[0], regionMax[0],
               region[1] / regionN[1], regionMax[1], region[2] / regionN[2], regionMax[2]);
        // Measured when the half resolution came in (2026-11-24): mean 0.014; crease mean 0.027, max 0.235 (its
        // one-pixel bottom line: the half-resolution texels stand for the pixels beside it, never on it); flat and sky
        // under 0.02. A regression guard on those numbers:
        if (mean > 0.02 || region[0] / regionN[0] > 0.04 || regionMax[0] > 0.3 || regionMax[1] > 0.03 || regionMax[2] > 0.03)
          fail("%s: half-resolution AO further from the full-resolution one than measured", c.name);
      }
    }
    be.destroy(rt);
    be.destroy(dsk);
    be.destroyBuffer(vb);
    be.destroyShader(vsKey);
    be.destroyShader(psKey);
  }

  if (gtaoId) be.destroyEffect(gtaoId);
  if (gtaoFullId) be.destroyEffect(gtaoFullId);
  defaults->Apply();
  defaults->Release();
  be.destroyTexture(texKey);
  be.destroyTexture(mipKey);
  be.destroyDecl(declKey);
  be.destroyDecl(declKey2);
  printf("draw oracle: %d/%zu cases identical to D3D9\n", passed, cases.size());
}

}  // namespace drawtest
