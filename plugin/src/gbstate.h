#pragma once
// Tracks D3D9 state at the Gamebryo setters.
#include <cstdint>
#include <vector>

struct IDirect3DDevice9;

namespace fnvvk::gbstate {

struct State {
  uint32_t rs[256] = {};
  uint32_t ss[16][14] = {};
  void* tex[16] = {};
  void* vs = nullptr;
  void* ps = nullptr;
  void* decl = nullptr;
  uint32_t fvf = 0;
  struct Stream { void* vb; uint32_t offset, stride; } streams[8] = {};
  void* ib = nullptr;
  struct { uint32_t x, y, w, h; float minZ, maxZ; } viewport = {};
  int32_t scissor[4] = {};
  float clip[6][4] = {};
  float vsc[256][4] = {};
  float psc[224][4] = {};
  uint64_t vscVersion = 1, pscVersion = 1;
  uint64_t rsVersion = 1, clipVersion = 1;  // bumped when a render state / clip plane changes value
};

// The tracked state (main thread).
State& state();

// The device facade every port calls instead of IDirect3DDevice9: tracks, then forwards while D3D9 exists.
namespace dev {
void setRenderState(IDirect3DDevice9* d, uint32_t s, uint32_t v);
void setSamplerState(IDirect3DDevice9* d, uint32_t stage, uint32_t s, uint32_t v);
void setTexture(IDirect3DDevice9* d, uint32_t stage, void* tex);
void setTextureStageState(IDirect3DDevice9* d, uint32_t stage, uint32_t s, uint32_t v);
void setVertexShader(IDirect3DDevice9* d, void* vs);
void setPixelShader(IDirect3DDevice9* d, void* ps);
void setFVF(IDirect3DDevice9* d, uint32_t fvf);
void setDeclaration(IDirect3DDevice9* d, void* decl);
long setVSConstF(IDirect3DDevice9* d, uint32_t start, const float* v, uint32_t n);
long setPSConstF(IDirect3DDevice9* d, uint32_t start, const float* v, uint32_t n);
long setVSConstI(IDirect3DDevice9* d, uint32_t start, const int* v, uint32_t n);
long setPSConstI(IDirect3DDevice9* d, uint32_t start, const int* v, uint32_t n);
long setVSConstB(IDirect3DDevice9* d, uint32_t start, const int* v, uint32_t n);
long setPSConstB(IDirect3DDevice9* d, uint32_t start, const int* v, uint32_t n);
}  // namespace dev

// Rewrites the confirmed direct device-call sites in game and shader code (IDA) so they call the facade.
// Applied at plugin load, before the renderer creates anything.
bool patchCallSites();
// The D3DVERTEXELEMENT9 bytes a declaration handle was created from (game-side registration).
bool declarationElements(void* handle, std::vector<uint8_t>& out);
// True while a draw-site thunk is calling D3D9 (the draw came through a patched site).
bool inDrawThunk();
// True while the declaration thunk is creating a declaration (D3D9 receives the game-side element array).
bool inDeclThunk();
// Registers the declarations created before the backend existed (call once it is ready).
void flushDeclarations();

// Starts tracking from the device's current state (called once the device exists) and installs the ports.
void initFromDevice(IDirect3DDevice9* d);
// false: draws issued at the patched sites reach only the backend (D3D9 keeps the state, does no draw work).
void setForwardDraws(bool on);
bool forwardDraws();
// No D3D9 at all (the plugin's device, gbdevice.cpp): nothing is forwarded; declarations are plugin objects.
void setNoD3D9();

// The tracker's device-method entry points with the IDirect3DDevice9 semantics (the plugin device's methods; the
// patched game sites call the same code).
namespace api {
long setRenderState(IDirect3DDevice9* d, uint32_t s, uint32_t v);
long setSamplerState(IDirect3DDevice9* d, uint32_t stage, uint32_t s, uint32_t v);
long setViewport(IDirect3DDevice9* d, const void* viewport);  // D3DVIEWPORT9*
long setClipPlane(IDirect3DDevice9* d, uint32_t i, const float* plane);
long setScissorRect(IDirect3DDevice9* d, const void* rect);   // RECT*
long setVertexShader(IDirect3DDevice9* d, void* vs);
long setPixelShader(IDirect3DDevice9* d, void* ps);
long setVertexDeclaration(IDirect3DDevice9* d, void* decl);
long setStreamSource(IDirect3DDevice9* d, uint32_t n, void* vb, uint32_t offset, uint32_t stride);
long setIndices(IDirect3DDevice9* d, void* ib);
long createVertexDeclaration(IDirect3DDevice9* d, const void* elements, void** out);
long drawPrimitive(IDirect3DDevice9* d, uint32_t pt, uint32_t start, uint32_t prims);
long drawIndexedPrimitive(IDirect3DDevice9* d, uint32_t pt, int32_t base, uint32_t minIdx, uint32_t nv, uint32_t startIdx,
                          uint32_t prims);
long drawPrimitiveUP(IDirect3DDevice9* d, uint32_t pt, uint32_t prims, const void* data, uint32_t stride);
long getRenderTargetData(IDirect3DDevice9* d, void* src, void* dst);
}  // namespace api
bool installHooks();

}  // namespace fnvvk::gbstate
