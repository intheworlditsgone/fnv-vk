#pragma once
// The plugin's IDirect3D9Ex / IDirect3DDevice9Ex (native mode; gbdevice.cpp): D3D9 is never created.
namespace fnvvk::gbdevice {
bool install();  // at plugin load, before the renderer is created
void logSummary();
}  // namespace fnvvk::gbdevice
