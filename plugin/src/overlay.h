#pragma once
// In-game menu (Dear ImGui; overlay.cpp). Native mode.
namespace fnvvk {
class Backend;
namespace overlay {
void init(void* gameWindow);  // after the backend exists, on the main thread (not under the loader lock)
void frame(Backend* be);      // once per frame, before the backend presents
}  // namespace overlay
}  // namespace fnvvk
