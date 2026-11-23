#pragma once
// FalloutNV.exe 1.4.0.525 (Steam). Addresses checked against the byte-matching decomp's units.json and the vtable
// slots against the exe's .rdata; parameter counts agree with each function's stack cleanup.
#include <cstdint>

namespace fnv {

constexpr uint32_t kRuntimeVersion = 0x040020D0;  // MAKE_NEW_VEGAS_VERSION(4, 0, 525), as xNVSE reports it

// static GraphicsRenderer* __cdecl Create(15 x unsigned int)
constexpr uintptr_t NiDX9Renderer_Create = 0x00E76210;

// NiDX9Renderer vtable (115 slots); the bodies are hooked directly.
constexpr uintptr_t NiDX9Renderer_vtbl = 0x010EE4BC;
constexpr uintptr_t Do_BeginFrame = 0x00E74F40;                   // slot 95, bool __thiscall ()
constexpr uintptr_t Do_EndFrame = 0x00E69660;                     // slot 96, bool __thiscall ()
constexpr uintptr_t Do_DisplayFrame = 0x00E75000;                 // slot 97, bool __thiscall ()
constexpr uintptr_t Do_BeginUsingRenderTargetGroup = 0x00E6EEA0;  // slot 101, bool __thiscall (target*, uint clearMode)
constexpr uintptr_t Do_EndUsingRenderTargetGroup = 0x00E74DC0;    // slot 102, bool __thiscall ()

// Draw paths
constexpr uintptr_t DrawIndexedPrimitive_TriList = 0x00E698A0;  // int __thiscall (int, int vertexCount, int, int primCount)
constexpr uintptr_t RenderTriShape2 = 0x00E745A0;               // void __thiscall (src*)
constexpr uintptr_t Do_DrawPartition = 0x00E6D310;              // void __thiscall (stripSet*, geometry*, int)
constexpr uintptr_t DrawGeometry = 0x00E74840;                  // void __thiscall (Geom*)
constexpr uintptr_t DrawMesh = 0x00E715A0;                      // void __thiscall (Mesh*, uint, uint, void*, void*, Batch*)
constexpr uintptr_t DrawStrip = 0x00E718A0;                     // void __thiscall (Geometry*, Flags*, Shape*, void*, Color*)
constexpr uintptr_t RenderParticles = 0x00E70140;               // void __thiscall (SceneArg*)
constexpr uintptr_t RenderLines = 0x00E70720;                   // void __thiscall (StripGeometry*)
constexpr uintptr_t RenderScreenTexture = 0x00E70BE0;           // void __thiscall (LineGeometry*)
constexpr uintptr_t RenderSlot107 = 0x00E744B0;                 // vtable slot 107, void __thiscall (Src*)
constexpr uintptr_t RenderSlot114 = 0x00E750E0;                 // vtable slot 114, void __thiscall (Input*)
constexpr uintptr_t BSBatchRenderer_RenderPassImmediately = 0x00B994F0;  // void __cdecl (pass*, uint pass, bool, bool, bool)

// Frame / content
constexpr uintptr_t Main_Swap = 0x0086FF70;          // void __thiscall ()
constexpr uintptr_t ShaderBuffer_Load = 0x00B7FE40;  // bool __thiscall (int packageIndex)

// Render-target classes (vtables from the exe's RTTI names; layouts next to main.cpp's describeBuffer)
constexpr uintptr_t vtbl_NiRenderTargetGroup = 0x011010EC;
constexpr uintptr_t vtbl_Ni2DBuffer = 0x0109F4DC;
constexpr uintptr_t vtbl_NiDepthStencilBuffer = 0x010A2084;
constexpr uintptr_t vtbl_NiDX92DBufferData = 0x010EEED4;
constexpr uintptr_t vtbl_NiDX9TextureBufferData = 0x010EEFBC;
constexpr uintptr_t vtbl_NiDX9ImplicitBufferData = 0x010EF06C;
constexpr uintptr_t vtbl_NiDX9ImplicitDepthStencilBufferData = 0x010EF114;
constexpr uintptr_t vtbl_NiDX9AdditionalDepthStencilBufferData = 0x010EF184;

// Render-target lifetime
constexpr uintptr_t Ni2DBuffer_SetRendererData = 0x00A8F140;  // void __thiscall (RendererData*)       retn 4
constexpr uintptr_t Ni2DBuffer_ResetDimensions = 0x00A8F080;  // void __thiscall (uint w, uint h)      retn 8
// Deleting destructors, all void* __thiscall (char) retn 4.
constexpr uintptr_t NiDX92DBufferData_DeletingDtor = 0x00E7D620;  // also ImplicitDepthStencilBufferData's slot 0
constexpr uintptr_t NiDX9TextureBufferData_DeletingDtor = 0x00E7D230;
constexpr uintptr_t NiDX9ImplicitBufferData_DeletingDtor = 0x00E7D690;
constexpr uintptr_t NiDX9AdditionalDepthStencilBufferData_DeletingDtor = 0x00E7DC40;
// bool __thiscall (IDirect3DDevice9*) retn 4
constexpr uintptr_t NiDX9ImplicitBufferData_Recreate = 0x00E7D0A0;
constexpr uintptr_t NiDX9ImplicitDepthStencilBufferData_Recreate = 0x00E7CE30;
constexpr uintptr_t NiDX9AdditionalDepthStencilBufferData_Recreate = 0x00E7CF50;
constexpr uintptr_t Do_ClearBuffer = 0x00E6F310;  // slot 98, void __thiscall (NiRect<float>*, uint mode) retn 8

// Textures
constexpr uintptr_t NiDX9SourceTextureData_LoadTextureFile = 0x00E68A80;  // bool __thiscall (const char*, NiDX9Renderer*, FormatPrefs*, NiFile*) retn 10h
constexpr uintptr_t NiDX9SourceTextureData_DeletingDtor = 0x00E68EA0;    // vtable 0x010ED37C slot 0, void* __thiscall (uint)
constexpr uint32_t NiDX9TextureData_D3DTexture = 0x64;                    // D3DBaseTexturePtr m_pkD3DTexture
constexpr uintptr_t NiDX9SourceTextureData_uiMipMapSkip = 0x0126EFC4;     // unsigned (set by TESRenderer::Init)
constexpr uintptr_t NiDX9SourceTextureData_uiMipMapMinimum = 0x0126EFC8;  // unsigned
constexpr uintptr_t NiPath_Standardize = 0x00AA2A00;                      // bool __cdecl (char*), in place: '/' -> '\', collapses repeats
constexpr uintptr_t NiFile_GetFile = 0x00AA1A20;                          // NiFile* __cdecl (const char*, OpenMode, uint bufSize)
constexpr uintptr_t NiBinaryStream_Read = 0x00462D80;                     // uint __thiscall (void*, uint); NiFile vtbl: +0 delete, +4 Bool, +0x1C GetFileSize

// d3dx9_38.dll (the game's import), identified by its PE header; RVAs from Microsoft's public PDB for this build.
constexpr uint32_t d3dx9_38_TimeDateStamp = 0x4833557D;
constexpr uint32_t d3dx9_38_SizeOfImage = 0x003C8000;
constexpr uintptr_t d3dx9_38_EncodeDXT1_RVA = 0x00229C16;  // D3DXTex::D3DXEncodeDXT1(uchar*, D3DXCOLOR const*, int), stdcall
constexpr uintptr_t d3dx9_38_EncodeDXT3_RVA = 0x00229D52;
constexpr uintptr_t d3dx9_38_EncodeDXT5_RVA = 0x00229E94;

constexpr uintptr_t NiRenderer_ms_pRenderer = 0x011F4748;  // NiRenderer* (.data)
constexpr uint32_t NiDX9Renderer_D3DDeviceOffset = 0x288;   // IDirect3DDevice9* member

constexpr uintptr_t TES_Singleton = 0x011DEA10;        // TES*
constexpr uintptr_t TESMain_spWorldRoot = 0x011DEB7C;  // BSSceneGraph* (+0xAC: spCamera)

}  // namespace fnv
