// Screenshots (screenshot.h).
//
// What is captured: the game's finished frame, buffer 0 of the renderer's default render-target group at DisplayFrame
// (the HUD and the game's menus included, fnv_vk's menu not: it is drawn onto the swapchain image at present). It is
// read with Backend::readback, the path the game's own ScreenShot::TakeScreenShot uses (shadow.cpp screenshotLock):
// 8-bit B, G, R, A, a wide (RGBA16F) target clamped to 0..1 as the D3D9 surface holds it.
//   - SDR output shows that image through the game's gamma ramp (backend gamma.frag: channel value v -> ramp[v] >> 8),
//     so the same table is applied here and the file holds what is on screen.
//   - HDR output shows the frame through the HDR encode (paper white, highlights up to the peak). The SDR file is the
//     clamped image (the gamma ramp does not apply to HDR output); the HDR file is the encode for an scRGB swapchain
//     (Backend::readbackHdrEncode), which is exactly what the display shows, saved as JPEG XR.
// The readbacks wait for the GPU on the main thread (as the game's own screenshot does); converting and encoding run on
// a worker thread. Finished saves are announced with the game's own HUD message (Interface::ShowMessage, as
// ScreenShot::TakeScreenShot does).
#include "screenshot.h"

#include <windows.h>
#include <objbase.h>
#include <shellapi.h>
#include <shlobj.h>

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <new>
#include <thread>
#include <vector>

#include "backend.h"
#include "common.h"
#include "gbstate.h"

namespace fnvvk::screenshot {
namespace {

constexpr uint32_t kVkB8G8R8A8Unorm = 44;      // VK_FORMAT_B8G8R8A8_UNORM
constexpr uint32_t kVkR16G16B16A16Sfloat = 97;  // VK_FORMAT_R16G16B16A16_SFLOAT
constexpr int kMaxPending = 2;  // frames held for the worker at once (a 4K frame with its HDR copy is ~100 MB)

// Interface::ShowMessage 0x007052F0 (pdb params apString, aeEmotion, apIconFilename, apSoundName, afShowSeconds,
// abMoveToHead): queues a HUD message if the HUD exists. ScreenShot::TakeScreenShot calls it (msg, 0, 0, 0, 2.0f, 0).
using ShowMessageFn = bool(__cdecl*)(const char*, int, const char*, const char*, float, bool);
const ShowMessageFn ShowMessage = reinterpret_cast<ShowMessageFn>(0x007052F0);

Settings g_settings;
std::string g_ini;
std::atomic<bool> g_requested{ false };

std::mutex g_gammaMutex;
uint8_t g_gamma[3][256];
bool g_gammaInit = [] {
  for (int c = 0; c < 3; c++)
    for (int i = 0; i < 256; i++) g_gamma[c][i] = uint8_t(i);
  return true;
}();

std::wstring widen(const std::string& s) {
  if (s.empty()) return {};
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
  std::wstring w(size_t(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), w.data(), n);
  return w;
}
std::string narrow(const std::wstring& w) {
  if (w.empty()) return {};
  const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), nullptr, 0, nullptr, nullptr);
  std::string s(size_t(n), '\0');
  WideCharToMultiByte(CP_UTF8, 0, w.data(), int(w.size()), s.data(), n, nullptr, nullptr);
  return s;
}

// --- worker ------------------------------------------------------------------------------------------------------------
struct Job {
  uint32_t width = 0, height = 0;
  std::vector<uint8_t> sdr;  // B, G, R, A
  std::vector<uint8_t> hdr;  // R, G, B, A half floats (scRGB); empty: none
  uint8_t lut[3][256];
  std::wstring folder, name;  // name without extension
  imagefile::Format format;
  int quality;
  bool notify;
};
// Heap-allocated and never freed: the detached worker may still run while the process exits.
struct Queue {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<Job> jobs;
  std::vector<std::string> messages;  // finished saves, shown by the main thread
  std::string status = "No screenshot taken yet";
  int pending = 0;
  bool started = false;
};
Queue& q() {
  static Queue* p = new Queue;
  return *p;
}

void finish(const std::string& message, bool notify) {
  logf("screenshot: %s", message.c_str());
  std::lock_guard<std::mutex> lk(q().mu);
  q().status = message;
  if (notify) q().messages.push_back(message);
  q().pending--;
}

bool ensureFolder(const std::wstring& folder, std::string& error) {
  const int r = SHCreateDirectoryExW(nullptr, folder.c_str(), nullptr);
  if (r == ERROR_SUCCESS || r == ERROR_ALREADY_EXISTS || r == ERROR_FILE_EXISTS) return true;
  char b[96];
  snprintf(b, sizeof(b), "the folder could not be created (error %d)", r);
  error = b;
  return false;
}

std::string process(Job& j) {
  std::string error;
  if (!ensureFolder(j.folder, error)) return "Screenshot NOT saved: " + error + " (" + narrow(j.folder) + ")";
  const std::wstring base = j.folder + L"\\" + j.name;
  const std::wstring sdrPath = base + imagefile::extension(j.format);
  {
    std::vector<uint8_t> bgr(size_t(j.width) * j.height * 3);
    toBgr(j.sdr.data(), size_t(j.width) * j.height, j.lut, bgr.data());
    std::vector<uint8_t>().swap(j.sdr);
    if (!imagefile::writeSdr(sdrPath, j.format, bgr.data(), j.width, j.height, j.quality, error))
      return "Screenshot NOT saved: " + error + " (" + narrow(sdrPath) + ")";
  }
  std::string written = narrow(j.name + imagefile::extension(j.format));
  if (!j.hdr.empty()) {
    uint16_t* h = reinterpret_cast<uint16_t*>(j.hdr.data());
    for (size_t i = 0, n = size_t(j.width) * j.height; i < n; i++) h[i * 4 + 3] = 0x3C00;  // opaque (1.0)
    const std::wstring hdrPath = base + L".jxr";
    if (!imagefile::writeScRgbJxr(hdrPath, h, j.width, j.height, j.quality, error))
      return "Screenshot saved: " + written + "; the HDR image NOT: " + error;
    written += " + .jxr";
  }
  return "Screenshot saved: " + written;
}

void worker() {
  CoInitializeEx(nullptr, COINIT_MULTITHREADED);  // for the thread's lifetime (it runs until the process exits)
  for (;;) {
    Job j;
    {
      std::unique_lock<std::mutex> lk(q().mu);
      q().cv.wait(lk, [] { return !q().jobs.empty(); });
      j = std::move(q().jobs.front());
      q().jobs.pop_front();
    }
    std::string message;
    try {
      message = process(j);
    } catch (const std::bad_alloc&) {
      message = "Screenshot NOT saved: out of memory";
    }
    const bool notify = j.notify;
    j = Job{};
    finish(message, notify);
  }
}

void submit(Job&& j) {
  std::lock_guard<std::mutex> lk(q().mu);
  q().jobs.push_back(std::move(j));
  if (!q().started) {
    std::thread(worker).detach();
    q().started = true;
  }
  q().cv.notify_one();
}

std::wstring fileName() {
  SYSTEMTIME t;
  GetLocalTime(&t);
  wchar_t b[64];
  swprintf(b, 64, L"FalloutNV_%04u-%02u-%02u_%02u-%02u-%02u-%03u", t.wYear, t.wMonth, t.wDay, t.wHour, t.wMinute,
           t.wSecond, t.wMilliseconds);
  return b;
}

void notifyNow(const std::string& message) {
  std::lock_guard<std::mutex> lk(q().mu);
  q().status = message;
  if (g_settings.notify) q().messages.push_back(message);
}

void take(Backend* be, uintptr_t onscreen, uint32_t width, uint32_t height) {
  if (gbstate::forwardDraws()) {
    notifyNow("Screenshot NOT taken: the frame is drawn by D3D9 (use the game's own screenshot)");
    return;
  }
  if (!be || !onscreen || !width || !height) {
    notifyNow("Screenshot NOT taken: no frame");
    return;
  }
  {
    std::lock_guard<std::mutex> lk(q().mu);
    if (q().pending >= kMaxPending) {
      q().status = "Screenshot skipped: still saving the previous ones";
      logf("screenshot: skipped (%d still saving)", q().pending);
      return;
    }
  }
  Job j;
  j.width = width;
  j.height = height;
  const size_t texels = size_t(width) * height;
  const HdrStatus hdr = be->hdrStatus();
  const bool wantHdr = hdr.active != HdrSettings::Off && g_settings.hdrImage;
  try {  // the readbacks fill these on the record thread, which must not allocate them
    j.sdr.reserve(texels * 4);
    if (wantHdr) j.hdr.reserve(texels * 8);
  } catch (const std::bad_alloc&) {
    notifyNow("Screenshot NOT taken: out of memory");
    return;
  }
  uint32_t vf = 0, ts = 0;
  if (!be->readback(onscreen, j.sdr, vf, ts) || vf != kVkB8G8R8A8Unorm || ts != 4 || j.sdr.size() != texels * 4) {
    logf("screenshot: readback of %ux%u failed (format %u, texel %u, %zu bytes)", width, height, vf, ts, j.sdr.size());
    notifyNow("Screenshot NOT taken: the frame could not be read");
    return;
  }
  if (hdr.active == HdrSettings::Off) {  // SDR output: through the gamma ramp, as presented
    std::lock_guard<std::mutex> lk(g_gammaMutex);
    memcpy(j.lut, g_gamma, sizeof(j.lut));
  } else {
    for (int c = 0; c < 3; c++)
      for (int i = 0; i < 256; i++) j.lut[c][i] = uint8_t(i);
  }
  if (wantHdr && (!be->readbackHdrEncode(onscreen, HdrSettings::ScRgb, j.hdr, vf) || vf != kVkR16G16B16A16Sfloat ||
                  j.hdr.size() != texels * 8)) {
    logf("screenshot: HDR readback failed (format %u, %zu bytes); saving the SDR image only", vf, j.hdr.size());
    std::vector<uint8_t>().swap(j.hdr);
  }
  j.folder = widen(folder());
  j.name = fileName();
  j.format = g_settings.format;
  j.quality = g_settings.quality;
  j.notify = g_settings.notify;
  logf("screenshot: %ux%u taken (%s output%s) -> %s", width, height, hdr.active == HdrSettings::Off ? "SDR" : "HDR",
       j.hdr.empty() ? "" : ", with HDR image", narrow(j.folder + L"\\" + j.name).c_str());
  {
    std::lock_guard<std::mutex> lk(q().mu);
    q().pending++;
    q().status = "Saving...";
  }
  submit(std::move(j));
}

const char* formatName(imagefile::Format f) {
  return f == imagefile::Format::Jpeg ? "jpg" : f == imagefile::Format::Bmp ? "bmp" : "png";
}

}  // namespace

void load() {
  g_ini = iniPath();
  char v[1024] = "";
  GetPrivateProfileStringA("Screenshot", "Format", "png", v, sizeof(v), g_ini.c_str());
  g_settings.format = !_stricmp(v, "jpg") || !_stricmp(v, "jpeg") ? imagefile::Format::Jpeg
                      : !_stricmp(v, "bmp")                       ? imagefile::Format::Bmp
                                                                  : imagefile::Format::Png;
  const int quality = int(GetPrivateProfileIntA("Screenshot", "Quality", 95, g_ini.c_str()));
  g_settings.quality = quality < 1 ? 1 : quality > 100 ? 100 : quality;
  g_settings.hdrImage = GetPrivateProfileIntA("Screenshot", "HDRImage", 1, g_ini.c_str()) != 0;
  g_settings.notify = GetPrivateProfileIntA("Screenshot", "Notify", 1, g_ini.c_str()) != 0;
  GetPrivateProfileStringA("Screenshot", "Folder", "", v, sizeof(v), g_ini.c_str());  // UTF-8 bytes, as written
  g_settings.folder = v;
  logf("screenshot: %s, quality %d, HDR image %s, folder %s", formatName(g_settings.format), g_settings.quality,
       g_settings.hdrImage ? "on" : "off", folder().c_str());
}

Settings& settings() { return g_settings; }

bool save() {
  char v[16];
  bool ok = WritePrivateProfileStringA("Screenshot", "Format", formatName(g_settings.format), g_ini.c_str()) != 0;
  snprintf(v, sizeof(v), "%d", g_settings.quality);
  ok &= WritePrivateProfileStringA("Screenshot", "Quality", v, g_ini.c_str()) != 0;
  ok &= WritePrivateProfileStringA("Screenshot", "HDRImage", g_settings.hdrImage ? "1" : "0", g_ini.c_str()) != 0;
  ok &= WritePrivateProfileStringA("Screenshot", "Notify", g_settings.notify ? "1" : "0", g_ini.c_str()) != 0;
  ok &= WritePrivateProfileStringA("Screenshot", "Folder", g_settings.folder.c_str(), g_ini.c_str()) != 0;
  logf("screenshot: settings %s fnv_vk.ini", ok ? "saved to" : "could NOT be saved to");
  return ok;
}

std::string defaultFolder() {
  std::wstring dir;
  PWSTR pictures = nullptr;
  if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Pictures, KF_FLAG_DEFAULT, nullptr, &pictures)) && pictures)
    dir = pictures;
  CoTaskMemFree(pictures);
  if (dir.empty()) {  // no Pictures folder: next to the game, where its own screenshots go
    wchar_t exe[MAX_PATH] = L"";
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    dir = exe;
    dir.resize(dir.find_last_of(L"\\/") == std::wstring::npos ? 0 : dir.find_last_of(L"\\/"));
    return narrow(dir + L"\\Screenshots");
  }
  return narrow(dir + L"\\Fallout New Vegas");
}

std::string folder() {
  std::string f = g_settings.folder;
  while (!f.empty() && (f.back() == '\\' || f.back() == '/' || f.back() == ' ')) f.pop_back();
  return f.empty() ? defaultFolder() : f;
}

void openFolder() {
  const std::wstring f = widen(folder());
  std::string error;
  if (ensureFolder(f, error)) ShellExecuteW(nullptr, L"open", f.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
  else notifyNow("Cannot open the screenshot folder: " + error);
}

void request() { g_requested = true; }

void frame(Backend* be, uintptr_t onscreen, uint32_t width, uint32_t height) {
  if (g_requested.exchange(false)) take(be, onscreen, width, height);
  std::vector<std::string> messages;
  {
    std::lock_guard<std::mutex> lk(q().mu);
    messages.swap(q().messages);
  }
  for (const std::string& m : messages) ShowMessage(m.c_str(), 0, nullptr, nullptr, 2.0f, false);
}

void setGammaRamp(const uint16_t red[256], const uint16_t green[256], const uint16_t blue[256]) {
  std::lock_guard<std::mutex> lk(g_gammaMutex);
  for (int i = 0; i < 256; i++) {  // the backend's table (doSetGammaRamp): the high byte of each entry
    g_gamma[0][i] = uint8_t(red[i] >> 8);
    g_gamma[1][i] = uint8_t(green[i] >> 8);
    g_gamma[2][i] = uint8_t(blue[i] >> 8);
  }
}

std::string status() {
  std::lock_guard<std::mutex> lk(q().mu);
  return q().status;
}

int pending() {
  std::lock_guard<std::mutex> lk(q().mu);
  return q().pending;
}

}  // namespace fnvvk::screenshot
