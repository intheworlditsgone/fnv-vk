// In-game menu: Dear ImGui, drawn by the backend over the finished frame at present (Backend::overlayBeginFrame /
// overlaySubmit). The menu is generated from the loaded effects (effects.h): nothing in it knows a particular effect.
//
// Input: the game polls DirectInput once per frame (Controls::UpdateValues) into its own keyboard and mouse buffers.
// That poll is hooked: the hotkeys (a DIK_ scancode plus Ctrl / Shift / Alt; the menu's toggle, default Insert, and the
// screenshot, default F11; bound in the menu's Screenshots panel, [Menu] and [Screenshot] Key / Modifiers in
// fnv_vk.ini) and the menu's keys come from the polled keyboard state, and while the menu is open both buffers are
// cleared, so the game sees no input.
// Buffered keystrokes (GetDeviceData) are withheld while open; text comes from WM_CHAR (window procedure subclass). The
// cursor and mouse buttons come from the menu's own non-exclusive mouse device.
#include "overlay.h"

#include <windows.h>

#define DIRECTINPUT_VERSION 0x0800
#include <dinput.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "backend.h"
#include "common.h"
#include "display.h"
#include "effects.h"
#include "replacements.h"
#include "screenshot.h"
#include "upscale.h"
#include "imgui.h"

namespace fnvvk::overlay {
namespace {

std::atomic<bool> g_open{ false };
bool g_ready = false;
HWND g_wnd = nullptr;
WNDPROC g_oldProc = nullptr;
std::string g_imguiIni;

// Input events from the DirectInput hooks (any thread) to the frame (main thread).
struct Event {
  enum Type : uint8_t { Key, MouseMove, MouseButton, Wheel, Char } type;
  int a = 0, b = 0;
};
std::mutex g_inputMutex;
std::vector<Event> g_events;
void push(Event e) {
  std::lock_guard<std::mutex> lock(g_inputMutex);
  if (g_events.size() < 4096) g_events.push_back(e);
}

// --- hotkeys -----------------------------------------------------------------------------------------------------------
// A hotkey is a key (DIK_ scancode, 0: none) and the modifiers that must be held with it. It fires when its key goes
// down with at least its modifiers held (more may be held: Shift for running does not block a plain key); when two
// hotkeys of the same key both match, only the one with more modifiers fires.
enum Mod : uint8_t { ModCtrl = 1, ModShift = 2, ModAlt = 4 };
enum Action : int { ActMenu, ActScreenshot, ActCount };
const char* const kActionName[ActCount] = { "Open / close this menu", "Take a screenshot" };
const char* const kActionSection[ActCount] = { "Menu", "Screenshot" };  // fnv_vk.ini: Key, Modifiers
constexpr uint16_t kActionDefault[ActCount] = { DIK_INSERT, DIK_F11 };
std::atomic<uint16_t> g_hotkey[ActCount] = { kActionDefault[ActMenu], kActionDefault[ActScreenshot] };  // key | mods << 8
uint8_t hkKey(uint16_t h) { return uint8_t(h & 0xFF); }
uint8_t hkMods(uint16_t h) { return uint8_t(h >> 8); }
int modCount(uint8_t m) { return (m & 1) + ((m >> 1) & 1) + ((m >> 2) & 1); }
bool isModifierKey(int k) {
  return k == DIK_LCONTROL || k == DIK_RCONTROL || k == DIK_LSHIFT || k == DIK_RSHIFT || k == DIK_LMENU || k == DIK_RMENU;
}
uint8_t modsHeld(const uint8_t* keys) {
  return uint8_t((((keys[DIK_LCONTROL] | keys[DIK_RCONTROL]) & 0x80) ? ModCtrl : 0) |
                 (((keys[DIK_LSHIFT] | keys[DIK_RSHIFT]) & 0x80) ? ModShift : 0) |
                 (((keys[DIK_LMENU] | keys[DIK_RMENU]) & 0x80) ? ModAlt : 0));
}

// Binding from the menu: the next key pressed (with the modifiers held) becomes the action's hotkey; Esc cancels.
std::atomic<int> g_binding{ -1 };             // the action being bound
std::atomic<int> g_bindConflict{ -1 };        // the last binding was refused: this action has that hotkey
std::atomic<bool> g_hotkeysChanged{ false };  // bound since the last save
std::atomic<bool> g_textInput{ false };       // a menu text field has the keyboard: no screenshot hotkey

// --- game input --------------------------------------------------------------------------------------------------------
// Controls::UpdateValues 0x00A23010 (__thiscall, no arguments) is the game's input poll. Each call copies the previous
// keyboard state to this+0x19F8 and reads the current one into this+0x18F8 (GetDeviceState(256)), and copies the
// previous mouse state to this+0x1B38 and reads the current one into this+0x1B24 (GetDeviceState(20): DIMOUSESTATE2;
// Controls::GetMouseAxis 0x00A239E0 returns its lX/lY/lZ at +0x1B24/+0x1B28/+0x1B2C). Everything in the game reads
// those buffers, so after the poll they carry the menu's keys and, while the menu is open, are cleared: the game sees no
// keys held and no mouse movement, wheel or buttons.
constexpr uint32_t kKeysOffset = 0x18F8, kMouseOffset = 0x1B24, kMouseStateSize = 20;
uint8_t g_prevKeys[256] = {};

using UpdateValuesFn = void(__fastcall*)(void* self, void* edx);
UpdateValuesFn oUpdateValues = nullptr;
void __fastcall hkUpdateValues(void* self, void* edx) {
  oUpdateValues(self, edx);
  uint8_t* keys = static_cast<uint8_t*>(self) + kKeysOffset;
  auto pressed = [&](int k) { return k && (keys[k] & 0x80) && !(g_prevKeys[k] & 0x80); };
  const uint8_t mods = modsHeld(keys);
  int consumed = -1;  // a key press that bound or fired a hotkey (not given to the menu)
  const int binding = g_binding;
  if (binding >= 0 && binding < ActCount && g_open) {
    for (int k = 1; k < 256 && consumed < 0; k++) {
      if (!pressed(k) || isModifierKey(k)) continue;
      consumed = k;
      g_binding = -1;
      if (k == DIK_ESCAPE) break;  // cancelled
      const uint16_t h = uint16_t(k | (mods << 8));
      int conflict = -1;
      for (int a = 0; a < ActCount; a++)
        if (a != binding && g_hotkey[a] == h) conflict = a;
      g_bindConflict = conflict;
      if (conflict < 0) {
        g_hotkey[binding] = h;
        g_hotkeysChanged = true;
      }
    }
  } else {
    int fire[ActCount];
    for (int a = 0; a < ActCount; a++) {
      const uint16_t h = g_hotkey[a];
      fire[a] = pressed(hkKey(h)) && (mods & hkMods(h)) == hkMods(h) ? modCount(hkMods(h)) : -1;
    }
    for (int a = 0; a < ActCount; a++) {
      if (fire[a] < 0) continue;
      bool beaten = false;  // the same key bound with more modifiers, also held
      for (int b = 0; b < ActCount; b++)
        if (b != a && fire[b] > fire[a] && hkKey(g_hotkey[b]) == hkKey(g_hotkey[a])) beaten = true;
      if (beaten) continue;
      if (a == ActMenu) {
        g_open = !g_open;
        consumed = hkKey(g_hotkey[a]);
      } else if (a == ActScreenshot && !(g_open && g_textInput)) {
        screenshot::request();
        if (g_open) consumed = hkKey(g_hotkey[a]);
      }
    }
  }
  if (g_open) {
    const int menuKey = hkKey(g_hotkey[ActMenu]);
    for (int k = 0; k < 256; k++)
      if (k != menuKey && k != consumed && ((keys[k] ^ g_prevKeys[k]) & 0x80)) push({ Event::Key, k, (keys[k] & 0x80) ? 1 : 0 });
  }
  memcpy(g_prevKeys, keys, 256);
  if (g_open) {
    memset(keys, 0, 256);
    memset(static_cast<uint8_t*>(self) + kMouseOffset, 0, kMouseStateSize);
  }
}

// Buffered keyboard data (IDirectInputDevice8::GetDeviceData, the game's keystroke queue): withheld while the menu is
// open, so typing in the menu does not reach the game. The menu's keys come from the polled state above.
using GetDataFn = HRESULT(STDMETHODCALLTYPE*)(void*, DWORD, LPDIDEVICEOBJECTDATA, LPDWORD, DWORD);
GetDataFn oGetData = nullptr;
HRESULT STDMETHODCALLTYPE hkGetData(void* self, DWORD cbObj, LPDIDEVICEOBJECTDATA rg, LPDWORD inOut, DWORD flags) {
  const HRESULT hr = oGetData(self, cbObj, rg, inOut, flags);
  if (SUCCEEDED(hr) && rg && inOut && !(flags & DIGDD_PEEK) && g_open) *inOut = 0;
  return hr;
}

using CreateDIFn = HRESULT(WINAPI*)(HINSTANCE, DWORD, REFIID, LPVOID*, LPUNKNOWN);
CreateDIFn directInputCreate() {
  HMODULE di = LoadLibraryA("dinput8.dll");
  return di ? reinterpret_cast<CreateDIFn>(GetProcAddress(di, "DirectInput8Create")) : nullptr;
}

bool hookInput() {
  bool ok = hook(0x00A23010, &hkUpdateValues, &oUpdateValues, "Controls::UpdateValues (menu)");
  // GetDeviceData of dinput8's device implementation, found through a throwaway keyboard device (the implementation is
  // shared by every device of the A and W interfaces).
  CreateDIFn create = directInputCreate();
  IDirectInput8A* dinput = nullptr;
  IDirectInputDevice8A* dev = nullptr;
  if (create && SUCCEEDED(create(GetModuleHandleA(nullptr), DIRECTINPUT_VERSION, IID_IDirectInput8A, reinterpret_cast<void**>(&dinput), nullptr)) &&
      dinput && SUCCEEDED(dinput->CreateDevice(GUID_SysKeyboard, &dev, nullptr)) && dev) {
    void* getData = (*reinterpret_cast<void***>(dev))[10];
    ok &= hook(uintptr_t(getData), &hkGetData, &oGetData, "IDirectInputDevice8::GetDeviceData (menu)");
  } else {
    ok = false;
  }
  if (dev) dev->Release();
  if (dinput) dinput->Release();
  return ok;
}

// The menu's own mouse device: non-exclusive and background (DirectInput always permits non-exclusive access next to
// the game's exclusive one). Its relative movement and buttons drive the menu's cursor.
IDirectInputDevice8A* g_menuMouse = nullptr;
void createMenuMouse(HWND wnd) {
  CreateDIFn create = directInputCreate();
  IDirectInput8A* dinput = nullptr;
  if (!create || FAILED(create(GetModuleHandleA(nullptr), DIRECTINPUT_VERSION, IID_IDirectInput8A, reinterpret_cast<void**>(&dinput), nullptr)) || !dinput)
    return;
  HRESULT hr = dinput->CreateDevice(GUID_SysMouse, &g_menuMouse, nullptr);
  if (SUCCEEDED(hr)) hr = g_menuMouse->SetDataFormat(&c_dfDIMouse2);
  if (SUCCEEDED(hr)) hr = g_menuMouse->SetCooperativeLevel(wnd, DISCL_NONEXCLUSIVE | DISCL_BACKGROUND);
  if (FAILED(hr) && g_menuMouse) {
    g_menuMouse->Release();
    g_menuMouse = nullptr;
  }
  dinput->Release();  // the device holds its own reference
}
uint8_t g_menuButtons[8] = {};
bool readMenuMouse(DIMOUSESTATE2& m) {
  m = {};
  return g_menuMouse && SUCCEEDED(g_menuMouse->Acquire()) && SUCCEEDED(g_menuMouse->GetDeviceState(sizeof(m), &m));
}
void pollMenuMouse() {
  DIMOUSESTATE2 m;
  if (!readMenuMouse(m)) return;
  if (m.lX || m.lY) push({ Event::MouseMove, int(m.lX), int(m.lY) });
  if (m.lZ) push({ Event::Wheel, int(m.lZ), 0 });
  for (int b = 0; b < 8; b++) {
    if ((m.rgbButtons[b] ^ g_menuButtons[b]) & 0x80) push({ Event::MouseButton, b, (m.rgbButtons[b] & 0x80) ? 1 : 0 });
    g_menuButtons[b] = m.rgbButtons[b];
  }
}

LRESULT CALLBACK wndProc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
  if (g_open && msg == WM_CHAR) {
    push({ Event::Char, int(wp), 0 });
    return 0;
  }
  return CallWindowProcW(g_oldProc, h, msg, wp, lp);
}

// DIK_ scancode -> ImGuiKey (the keys a menu uses: navigation, editing, shortcuts, digits and letters).
ImGuiKey keyFor(int dik) {
  switch (dik) {
    case DIK_TAB: return ImGuiKey_Tab;
    case DIK_LEFT: return ImGuiKey_LeftArrow;
    case DIK_RIGHT: return ImGuiKey_RightArrow;
    case DIK_UP: return ImGuiKey_UpArrow;
    case DIK_DOWN: return ImGuiKey_DownArrow;
    case DIK_PRIOR: return ImGuiKey_PageUp;
    case DIK_NEXT: return ImGuiKey_PageDown;
    case DIK_HOME: return ImGuiKey_Home;
    case DIK_END: return ImGuiKey_End;
    case DIK_INSERT: return ImGuiKey_Insert;
    case DIK_DELETE: return ImGuiKey_Delete;
    case DIK_BACK: return ImGuiKey_Backspace;
    case DIK_SPACE: return ImGuiKey_Space;
    case DIK_RETURN: return ImGuiKey_Enter;
    case DIK_NUMPADENTER: return ImGuiKey_KeypadEnter;
    case DIK_ESCAPE: return ImGuiKey_Escape;
    case DIK_LCONTROL: return ImGuiKey_LeftCtrl;
    case DIK_RCONTROL: return ImGuiKey_RightCtrl;
    case DIK_LSHIFT: return ImGuiKey_LeftShift;
    case DIK_RSHIFT: return ImGuiKey_RightShift;
    case DIK_LMENU: return ImGuiKey_LeftAlt;
    case DIK_RMENU: return ImGuiKey_RightAlt;
    case DIK_A: return ImGuiKey_A;
    case DIK_C: return ImGuiKey_C;
    case DIK_V: return ImGuiKey_V;
    case DIK_X: return ImGuiKey_X;
    case DIK_Y: return ImGuiKey_Y;
    case DIK_Z: return ImGuiKey_Z;
    case DIK_MINUS: return ImGuiKey_Minus;
    case DIK_PERIOD: return ImGuiKey_Period;
    default:
      if (dik >= DIK_1 && dik <= DIK_9) return ImGuiKey(ImGuiKey_1 + (dik - DIK_1));
      if (dik == DIK_0) return ImGuiKey_0;
      return ImGuiKey_None;
  }
}

// --- frame ------------------------------------------------------------------------------------------------------------
float g_mouseX = 0, g_mouseY = 0, g_scale = 1.0f;
bool g_scaled = false, g_wasOpen = false;
LARGE_INTEGER g_freq{}, g_last{};
float g_frameMs[180] = {};
int g_frameIdx = 0;
BackendStats g_prevStats{};
float g_drawsPerFrame = 0;

void applyInput(ImGuiIO& io, float dispW, float dispH) {
  std::vector<Event> events;
  {
    std::lock_guard<std::mutex> lock(g_inputMutex);
    events.swap(g_events);
  }
  bool ctrl = false, shift = false, alt = false;
  for (const Event& e : events) {
    switch (e.type) {
      case Event::Key: {
        if (e.a == DIK_ESCAPE && e.b) g_open = false;  // Esc closes the menu (the game does not see it)
        const ImGuiKey k = keyFor(e.a);
        if (k != ImGuiKey_None) io.AddKeyEvent(k, e.b != 0);
        if (e.a == DIK_LCONTROL || e.a == DIK_RCONTROL) ctrl = true;
        if (e.a == DIK_LSHIFT || e.a == DIK_RSHIFT) shift = true;
        if (e.a == DIK_LMENU || e.a == DIK_RMENU) alt = true;
        break;
      }
      case Event::MouseMove:
        g_mouseX = std::clamp(g_mouseX + float(e.a) * g_scale, 0.0f, dispW - 1);
        g_mouseY = std::clamp(g_mouseY + float(e.b) * g_scale, 0.0f, dispH - 1);
        io.AddMousePosEvent(g_mouseX, g_mouseY);
        break;
      case Event::MouseButton:
        if (e.a < 5) io.AddMouseButtonEvent(e.a, e.b != 0);
        break;
      case Event::Wheel:
        io.AddMouseWheelEvent(0.0f, float(e.a) / 120.0f);
        break;
      case Event::Char:
        io.AddInputCharacterUTF16(ImWchar16(e.a));
        break;
    }
  }
  if (ctrl || shift || alt) {  // modifier state from the current keyboard state
    io.AddKeyEvent(ImGuiMod_Ctrl, (g_prevKeys[DIK_LCONTROL] | g_prevKeys[DIK_RCONTROL]) & 0x80);
    io.AddKeyEvent(ImGuiMod_Shift, (g_prevKeys[DIK_LSHIFT] | g_prevKeys[DIK_RSHIFT]) & 0x80);
    io.AddKeyEvent(ImGuiMod_Alt, (g_prevKeys[DIK_LMENU] | g_prevKeys[DIK_RMENU]) & 0x80);
  }
}

std::string keyName(uint8_t dik) {
  char buf[64];
  LONG lp = LONG(dik & 0x7F) << 16;
  if (dik & 0x80) lp |= 1 << 24;  // extended keys (Insert, Home, arrows...)
  if (!GetKeyNameTextA(lp, buf, sizeof(buf))) snprintf(buf, sizeof(buf), "0x%02X", dik);
  return buf;
}
std::string hotkeyName(uint16_t h) {
  if (!hkKey(h)) return "(none)";
  std::string s;
  if (hkMods(h) & ModCtrl) s += "Ctrl+";
  if (hkMods(h) & ModShift) s += "Shift+";
  if (hkMods(h) & ModAlt) s += "Alt+";
  return s + keyName(hkKey(h));
}

void loadHotkeys() {
  const std::string ini = iniPath();
  for (int a = 0; a < ActCount; a++) {
    char v[32] = "";
    uint16_t h = kActionDefault[a];
    GetPrivateProfileStringA(kActionSection[a], "Key", "", v, sizeof(v), ini.c_str());
    if (v[0]) {  // decimal or 0x hex; 0: none (not for the menu's own key)
      const unsigned long k = strtoul(v, nullptr, 0);
      if (k < 256 && (k > 0 || a != ActMenu)) h = uint16_t(k);
    }
    const unsigned mods = GetPrivateProfileIntA(kActionSection[a], "Modifiers", 0, ini.c_str()) & 7;
    if (hkKey(h)) h = uint16_t(h | (mods << 8));
    g_hotkey[a] = h;
  }
}
bool saveHotkeys() {
  const std::string ini = iniPath();
  bool ok = true;
  for (int a = 0; a < ActCount; a++) {
    char v[16];
    snprintf(v, sizeof(v), "0x%02X", hkKey(g_hotkey[a]));
    ok &= WritePrivateProfileStringA(kActionSection[a], "Key", v, ini.c_str()) != 0;
    snprintf(v, sizeof(v), "%u", unsigned(hkMods(g_hotkey[a])));
    ok &= WritePrivateProfileStringA(kActionSection[a], "Modifiers", v, ini.c_str()) != 0;
  }
  if (ok) g_hotkeysChanged = false;
  logf("menu: hotkeys (menu %s, screenshot %s) %s fnv_vk.ini", hotkeyName(g_hotkey[ActMenu]).c_str(),
       hotkeyName(g_hotkey[ActScreenshot]).c_str(), ok ? "saved to" : "could NOT be saved to");
  return ok;
}

// One parameter's widget (effects and replacements); true when the value changed.
bool paramWidget(fx::Param& p) {
  const float before = p.value;
  switch (p.type) {
    case fx::Param::Type::Float:
      ImGui::SliderFloat(p.label.c_str(), &p.value, p.min, p.max, "%.3g");
      break;
    case fx::Param::Type::Int: {
      int v = int(p.value);
      if (ImGui::SliderInt(p.label.c_str(), &v, int(p.min), int(p.max))) p.value = float(v);
      break;
    }
    case fx::Param::Type::Bool: {
      bool v = p.value != 0.0f;
      if (ImGui::Checkbox(p.label.c_str(), &v)) p.value = v ? 1.0f : 0.0f;
      break;
    }
    case fx::Param::Type::Enum: {
      const int cur = std::clamp(int(p.value), 0, int(p.options.size()) - 1);
      if (ImGui::BeginCombo(p.label.c_str(), p.options[cur].c_str())) {
        for (int i = 0; i < int(p.options.size()); i++)
          if (ImGui::Selectable(p.options[i].c_str(), i == cur)) p.value = float(i);
        ImGui::EndCombo();
      }
      break;
    }
  }
  return p.value != before;
}

void effectsPanel() {
  std::vector<fx::Effect>& list = effects::list();
  if (ImGui::Button("Reload effects")) effects::reload();
  ImGui::SameLine();
  ImGui::TextDisabled("%s", effects::effectsDir());
  if (list.empty()) ImGui::TextWrapped("No effects found. Each effect is a folder with an effect.ini (docs/EFFECTS.md).");
  for (fx::Effect& e : list) {
    ImGui::PushID(e.id.c_str());
    ImGui::Separator();
    if (!e.error.empty()) {
      ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.4f, 1.0f), "%s: failed to load", e.name.c_str());
      ImGui::TextWrapped("%s", e.error.c_str());
      ImGui::PopID();
      continue;
    }
    ImGui::Checkbox(e.name.c_str(), &e.enabled);
    if (!e.description.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", e.description.c_str());
    ImGui::SameLine();
    ImGui::TextDisabled("(%s)", e.point.c_str());
    if (e.requirement == "pbr" && !effects::pbrLighting()) {
      ImGui::SameLine();
      ImGui::TextDisabled("needs Lighting: Physically based lighting");
    }
    if (!e.params.empty() && ImGui::TreeNode("Parameters")) {
      for (fx::Param& p : e.params) paramWidget(p);
      if (ImGui::Button("Defaults"))
        for (fx::Param& p : e.params) p.value = p.def;
      ImGui::SameLine();
      if (ImGui::Button("Save")) effects::save(e);
      ImGui::TreePop();
    } else if (e.params.empty()) {
      ImGui::SameLine();
      if (ImGui::SmallButton("Save")) effects::save(e);
    }
    ImGui::PopID();
  }
}

void displayPanel(Backend* be) {
  display::Settings& s = display::settings();
  const display::Monitor& mon = display::monitor();
  const HdrStatus st = be->hdrStatus();
  if (mon.known)
    ImGui::Text("Windows HDR: %s   Display peak: %.0f nits   SDR content brightness: %.0f nits", mon.hdrOn ? "on" : "off",
                mon.peakNits, mon.sdrWhiteNits);
  else
    ImGui::TextDisabled("Windows did not report this monitor's HDR state");
  ImGui::Text("Output: %s", st.active == HdrSettings::Hdr10 ? "HDR10 (PQ)" : st.active == HdrSettings::ScRgb ? "scRGB (FP16)" : "SDR");
  bool changed = false;
  static const char* kHdr[] = { "Off", "On", "Auto (follow Windows)" };
  int hdr = int(s.hdr);
  if (ImGui::Combo("HDR output", &hdr, kHdr, 3)) {
    s.hdr = display::Settings::Hdr(hdr);
    changed = true;
  }
  static const char* kFormat[] = { "HDR10 (PQ, 10-bit)", "scRGB (FP16)" };
  int fmt = int(s.format);
  if (ImGui::Combo("HDR format", &fmt, kFormat, 2)) {
    s.format = display::Settings::Format(fmt);
    changed = true;
  }
  changed |= ImGui::SliderFloat("Peak brightness (nits)", &s.peakNits, 0.0f, 4000.0f,
                                s.peakNits > 0 ? "%.0f" : "display's value");
  changed |= ImGui::SliderFloat("Paper white (nits)", &s.paperWhiteNits, 0.0f, 500.0f,
                                s.paperWhiteNits > 0 ? "%.0f" : "Windows' value");
  ImGui::TextDisabled("In use: peak %.0f nits, paper white %.0f nits (0 on a slider: the value Windows reports)",
                      display::effectivePeak(), display::effectivePaperWhite());
  if (display::hdrWanted() && st.active == HdrSettings::Off && !(st.hdr10Available || st.scRgbAvailable))
    ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "The display path offers no HDR format (is Windows HDR on?)");
  if (!st.wideTargets)
    ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f),
                       "Highlights are clipped: no tonemap effect was loaded and HDR was off at start (restart to widen)");
  if (changed) display::apply(be);
  if (ImGui::Button("Save##display")) display::save();
}

void upscalingPanel(Backend* be) {
  upscale::Settings& s = upscale::settings();
  const UpscaleStatus st = be->upscaleStatus();
  bool changed = false;
  static const char* kUpscaler[] = { "Off (native resolution)", "AMD FSR 3.1" };
  int up = int(s.upscaler);
  if (ImGui::Combo("Upscaler", &up, kUpscaler, 2)) {
    s.upscaler = upscale::Settings::Upscaler(up);
    changed = true;
  }
  if (s.upscaler != upscale::Settings::Off) {
    static const char* kQuality[] = { "Native AA (1.0x)", "Quality (1.5x)", "Balanced (1.7x)", "Performance (2.0x)",
                                      "Ultra Performance (3.0x)" };
    int q = int(s.quality);
    if (ImGui::Combo("Scaling mode", &q, kQuality, 5)) {
      s.quality = upscale::Settings::QualityMode(q);
      changed = true;
    }
    changed |= ImGui::Checkbox("Sharpening", &s.sharpen);
    if (s.sharpen) changed |= ImGui::SliderFloat("Sharpness", &s.sharpness, 0.0f, 1.0f, "%.2f");
    if (st.displayW && st.renderW)
      ImGui::Text("Rendering %ux%u, output %ux%u%s", st.renderW, st.renderH, st.displayW, st.displayH,
                  st.upscaling ? "" : " (not upscaled last frame)");
  }
  if (!st.error.empty()) ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.3f, 1.0f), "%s", st.error.c_str());
  else if (s.upscaler != upscale::Settings::Off && !st.libraryLoaded)
    ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "fnv_vk_fidelityfx.dll is not loaded");
  if (changed) upscale::apply(be);
  if (ImGui::Button("Save##upscaling")) upscale::save();
}

void replacementsPanel(Backend* be) {
  std::vector<fx::Replacement>& list = replacements::list();
  ImGui::TextDisabled("%s", replacements::dir());
  if (list.empty()) ImGui::TextWrapped("No shader replacements found (docs/REPLACEMENTS.md).");
  for (fx::Replacement& r : list) {
    ImGui::PushID(r.id.c_str());
    ImGui::Separator();
    if (!r.error.empty()) {
      ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.4f, 1.0f), "%s: failed to load", r.name.c_str());
      ImGui::TextWrapped("%s", r.error.c_str());
      ImGui::PopID();
      continue;
    }
    ImGui::Checkbox(r.name.c_str(), &r.enabled);
    if (!r.description.empty() && ImGui::IsItemHovered()) ImGui::SetTooltip("%s", r.description.c_str());
    const bool on = replacements::active(r);
    ImGui::SameLine();
    if (r.enabled != on) ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "(save, then restart to %s)", r.enabled ? "enable" : "disable");
    else ImGui::TextDisabled(on ? "(active)" : "(off)");
    if (!r.params.empty() && ImGui::TreeNode("Parameters")) {
      bool changed = false;
      for (fx::Param& p : r.params) changed |= paramWidget(p);
      if (ImGui::Button("Defaults")) {
        for (fx::Param& p : r.params) p.value = p.def;
        changed = true;
      }
      if (changed && on) replacements::applyParams(be, r);
      ImGui::SameLine();
      if (ImGui::Button("Save")) replacements::save(r);
      ImGui::TreePop();
    } else if (r.params.empty()) {
      ImGui::SameLine();
      if (ImGui::SmallButton("Save")) replacements::save(r);
    }
    ImGui::PopID();
  }
}

void performancePanel(Backend* be) {
  float sum = 0, worst = 0;
  for (float ms : g_frameMs) {
    sum += ms;
    worst = std::max(worst, ms);
  }
  const float avg = sum / float(std::size(g_frameMs));
  ImGui::Text("%.1f fps   %.2f ms average, %.2f ms worst (last %zu frames)", avg > 0 ? 1000.0f / avg : 0.0f, avg, worst,
              std::size(g_frameMs));
  ImGui::PlotLines("##frametime", g_frameMs, int(std::size(g_frameMs)), g_frameIdx, nullptr, 0.0f, std::max(33.3f, worst),
                   ImVec2(-1, 60 * g_scale));
  const BackendStats s = be->statsSnapshot();  // without waiting for the record thread
  ImGui::Text("Draws per frame: %.0f", g_drawsPerFrame);
  ImGui::Text("Pipelines: %llu   Samplers: %llu", s.pipelines, s.samplers);
  ImGui::Text("Buffer renames: %llu (recycled %llu)   direct writes: %llu", s.bufferRenames, s.bufferRecycles,
              s.bufferDirectWrites);
  ImGui::Text("Effect passes run: %llu   not possible: %llu", s.effectRuns, s.effectsSkipped);
}

void screenshotsPanel() {
  screenshot::Settings& s = screenshot::settings();
  if (ImGui::Button("Take screenshot")) screenshot::request();
  if (ImGui::IsItemHovered()) ImGui::SetTooltip("The game's frame as shown on screen, without this menu");
  ImGui::SameLine();
  const int pending = screenshot::pending();
  if (pending > 0) ImGui::TextDisabled("Saving %d...", pending);
  else ImGui::TextWrapped("%s", screenshot::status().c_str());

  ImGui::SeparatorText("Hotkeys");
  const int binding = g_binding;
  for (int a = 0; a < ActCount; a++) {
    ImGui::PushID(a);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(kActionName[a]);
    ImGui::SameLine(230 * g_scale);
    const std::string label = binding == a ? "Press a key... (Esc: cancel)" : hotkeyName(g_hotkey[a]);
    if (ImGui::Button((label + "###bind").c_str(), ImVec2(230 * g_scale, 0))) {
      g_bindConflict = -1;
      g_binding = binding == a ? -1 : a;
    }
    if (ImGui::IsItemHovered() && binding != a)
      ImGui::SetTooltip("Click, then press the key (hold Ctrl, Shift or Alt with it for a combination)");
    if (a != ActMenu && hkKey(g_hotkey[a])) {
      ImGui::SameLine();
      if (ImGui::SmallButton("Clear")) {
        g_hotkey[a] = 0;
        g_hotkeysChanged = true;
      }
    }
    ImGui::PopID();
  }
  const int conflict = g_bindConflict;
  if (conflict >= 0 && conflict < ActCount)
    ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "That key is already used for \"%s\"", kActionName[conflict]);
  ImGui::TextDisabled("With the menu closed the game also receives these keys: pick keys it does not use.");

  ImGui::SeparatorText("Files");
  static const char* kFormats[] = { "PNG (lossless)", "JPEG", "BMP" };
  int fmt = int(s.format);
  if (ImGui::Combo("Format", &fmt, kFormats, 3)) s.format = imagefile::Format(fmt);
  ImGui::Checkbox("With HDR output, also save the HDR image (.jxr)", &s.hdrImage);
  if (ImGui::IsItemHovered())
    ImGui::SetTooltip("JPEG XR in scRGB, as Windows saves HDR screenshots: exactly what the HDR display shows.\n"
                      "The other file then holds the SDR image (highlights clipped at paper white).");
  if (s.format == imagefile::Format::Jpeg || s.hdrImage)
    ImGui::SliderInt("Quality (JPEG, JPEG XR)", &s.quality, 1, 100, s.quality == 100 ? "%d (JPEG XR: lossless)" : "%d");
  ImGui::Checkbox("Show a message in the HUD when saved", &s.notify);
  static char folder[1024] = "";
  static bool folderInit = false;
  if (!folderInit) {
    snprintf(folder, sizeof(folder), "%s", s.folder.c_str());
    folderInit = true;
  }
  if (ImGui::InputTextWithHint("Folder", screenshot::defaultFolder().c_str(), folder, sizeof(folder))) s.folder = folder;
  if (ImGui::Button("Open folder")) screenshot::openFolder();
  ImGui::SameLine();
  if (ImGui::Button("Save##screenshots")) {
    screenshot::save();
    saveHotkeys();
  }
  if (g_hotkeysChanged) {
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.3f, 1.0f), "(hotkeys changed: Save keeps them)");
  }
}

}  // namespace

void init(void* gameWindow) {
  if (g_ready) return;
  loadHotkeys();
  screenshot::load();
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  g_imguiIni = moduleDir() + "fnv_vk_menu.ini";  // window positions
  io.IniFilename = g_imguiIni.c_str();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  io.BackendPlatformName = "fnv_vk (DirectInput)";
  ImGui::StyleColorsDark();
  char fonts[MAX_PATH] = "";
  GetWindowsDirectoryA(fonts, MAX_PATH);
  const std::string segoe = std::string(fonts) + "\\Fonts\\segoeui.ttf";
  if (GetFileAttributesA(segoe.c_str()) != INVALID_FILE_ATTRIBUTES) io.Fonts->AddFontFromFileTTF(segoe.c_str(), 17.0f);
  QueryPerformanceFrequency(&g_freq);
  QueryPerformanceCounter(&g_last);
  const bool di = hookInput();
  g_wnd = static_cast<HWND>(gameWindow);
  if (g_wnd) g_oldProc = reinterpret_cast<WNDPROC>(SetWindowLongPtrW(g_wnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&wndProc)));
  if (g_wnd) createMenuMouse(g_wnd);
  g_ready = true;
  logf("menu: ready (toggle %s = DIK 0x%02X, screenshot %s; input hooks %s)", hotkeyName(g_hotkey[ActMenu]).c_str(),
       hkKey(g_hotkey[ActMenu]), hotkeyName(g_hotkey[ActScreenshot]).c_str(), di ? "installed" : "FAILED");
}

void frame(Backend* be) {
  if (!g_ready || !be) return;
  LARGE_INTEGER now{};
  QueryPerformanceCounter(&now);
  const float dt = float(double(now.QuadPart - g_last.QuadPart) / double(g_freq.QuadPart));
  g_last = now;
  g_frameMs[g_frameIdx] = dt * 1000.0f;
  g_frameIdx = (g_frameIdx + 1) % int(std::size(g_frameMs));
  const BackendStats s = be->statsSnapshot();  // without waiting for the record thread
  if (s.frames > g_prevStats.frames) g_drawsPerFrame = float(s.draws - g_prevStats.draws) / float(s.frames - g_prevStats.frames);
  g_prevStats = s;

  const bool open = g_open;
  if (!open) {
    g_binding = -1;
    g_textInput = false;
    if (g_wasOpen) {
      std::lock_guard<std::mutex> lock(g_inputMutex);
      g_events.clear();
    }
    g_wasOpen = false;
    be->overlaySubmit(false);
    return;
  }
  uint32_t w = 0, h = 0;
  if (!be->overlayBeginFrame(w, h)) return;
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(float(w), float(h));
  io.DeltaTime = std::max(dt, 1e-4f);
  if (!g_scaled) {  // readable at any resolution: sizes relative to 1080 lines
    g_scale = std::max(1.0f, float(h) / 1080.0f);
    ImGui::GetStyle().ScaleAllSizes(g_scale);
    ImGui::GetStyle().FontScaleMain = g_scale;
    g_scaled = true;
  }
  if (!g_wasOpen) {  // cursor in the middle of the screen when the menu opens
    g_mouseX = float(w) * 0.5f;
    g_mouseY = float(h) * 0.5f;
    io.AddMousePosEvent(g_mouseX, g_mouseY);
    DIMOUSESTATE2 m;  // a fresh reading: movement accumulated while closed is discarded, held buttons stay held
    if (readMenuMouse(m)) memcpy(g_menuButtons, m.rgbButtons, 8);
    g_wasOpen = true;
  }
  pollMenuMouse();
  io.MouseDrawCursor = true;
  applyInput(io, float(w), float(h));

  ImGui::NewFrame();
  ImGui::SetNextWindowPos(ImVec2(40 * g_scale, 40 * g_scale), ImGuiCond_FirstUseEver);
  ImGui::SetNextWindowSize(ImVec2(560 * g_scale, 620 * g_scale), ImGuiCond_FirstUseEver);
  if (ImGui::Begin("fnv_vk")) {
    if (ImGui::CollapsingHeader("Performance", ImGuiTreeNodeFlags_DefaultOpen)) performancePanel(be);
    if (ImGui::CollapsingHeader("Display", ImGuiTreeNodeFlags_DefaultOpen)) displayPanel(be);
    if (ImGui::CollapsingHeader("Upscaling", ImGuiTreeNodeFlags_DefaultOpen)) upscalingPanel(be);
    if (ImGui::CollapsingHeader("Effects", ImGuiTreeNodeFlags_DefaultOpen)) effectsPanel();
    if (ImGui::CollapsingHeader("Shader replacements", ImGuiTreeNodeFlags_DefaultOpen)) replacementsPanel(be);
    if (ImGui::CollapsingHeader("Screenshots and hotkeys", ImGuiTreeNodeFlags_DefaultOpen)) screenshotsPanel();
    ImGui::Separator();
    ImGui::TextDisabled("%s or Esc: close   (the game receives no input while this menu is open)",
                        hotkeyName(g_hotkey[ActMenu]).c_str());
  }
  ImGui::End();
  ImGui::Render();
  g_textInput = io.WantTextInput;
  be->overlaySubmit(true);
}

}  // namespace fnvvk::overlay
