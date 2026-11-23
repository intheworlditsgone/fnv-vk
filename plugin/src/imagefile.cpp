// Image files through WIC (imagefile.h). The pixels are handed to WIC as a bitmap in their own pixel format; when the
// encoder negotiates another one (SetPixelFormat changes the GUID it is given), WICConvertBitmapSource converts.
#include "imagefile.h"

#include <windows.h>
#include <wincodec.h>
#include <ocidl.h>

#include <cstdio>

namespace fnvvk::imagefile {
namespace {

template <typename T> struct Com {
  T* p = nullptr;
  ~Com() {
    if (p) p->Release();
  }
  T** operator&() { return &p; }
  T* operator->() const { return p; }
  explicit operator bool() const { return p != nullptr; }
};

bool fail(std::string& error, const char* what, HRESULT hr) {
  char b[160];
  snprintf(b, sizeof(b), "%s failed (HRESULT 0x%08lX)", what, static_cast<unsigned long>(hr));
  error = b;
  return false;
}

bool setOption(IPropertyBag2* bag, const wchar_t* name, VARIANT& v) {
  PROPBAG2 opt{};
  opt.pstrName = const_cast<LPOLESTR>(name);
  return SUCCEEDED(bag->Write(1, &opt, &v));
}

// One frame: `pixels` (rows of `stride` bytes, pixel format `srcFormat`) written as `wantFormat` (or what the encoder
// negotiates from it). `quality` < 0: no ImageQuality option. `fullChroma`: JPEG with 4:4:4 sampling (WIC's default is
// 4:2:0, which blurs colour edges -- the HUD's text, sharp coloured detail -- at any quality).
bool encode(const std::wstring& path, REFGUID container, const uint8_t* pixels, uint32_t width, uint32_t height,
            uint32_t stride, REFGUID srcFormat, REFGUID wantFormat, float quality, bool fullChroma, std::string& error) {
  Com<IWICImagingFactory> factory;
  HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
  if (FAILED(hr)) return fail(error, "WIC factory", hr);
  Com<IWICBitmap> bitmap;
  hr = factory->CreateBitmapFromMemory(width, height, srcFormat, stride, stride * height, const_cast<BYTE*>(pixels), &bitmap);
  if (FAILED(hr)) return fail(error, "CreateBitmapFromMemory", hr);
  Com<IWICStream> stream;
  hr = factory->CreateStream(&stream);
  if (SUCCEEDED(hr)) hr = stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE);
  if (FAILED(hr)) return fail(error, "opening the file for writing", hr);
  Com<IWICBitmapEncoder> encoder;
  hr = factory->CreateEncoder(container, nullptr, &encoder);
  if (SUCCEEDED(hr)) hr = encoder->Initialize(stream.p, WICBitmapEncoderNoCache);
  if (FAILED(hr)) return fail(error, "creating the encoder", hr);
  Com<IWICBitmapFrameEncode> frame;
  Com<IPropertyBag2> options;
  hr = encoder->CreateNewFrame(&frame, &options);
  if (FAILED(hr)) return fail(error, "CreateNewFrame", hr);
  if (quality >= 0 && options) {
    VARIANT v;
    VariantInit(&v);
    v.vt = VT_R4;
    v.fltVal = quality;
    if (!setOption(options.p, L"ImageQuality", v)) return fail(error, "setting the image quality", E_FAIL);
  }
  if (fullChroma && options) {
    VARIANT v;
    VariantInit(&v);
    v.vt = VT_UI1;
    v.bVal = WICJpegYCrCbSubsampling444;
    if (!setOption(options.p, L"JpegYCrCbSubsampling", v)) return fail(error, "setting 4:4:4 chroma sampling", E_FAIL);
  }
  hr = frame->Initialize(options.p);
  if (SUCCEEDED(hr)) hr = frame->SetSize(width, height);
  WICPixelFormatGUID fmt = wantFormat;
  if (SUCCEEDED(hr)) hr = frame->SetPixelFormat(&fmt);
  if (FAILED(hr)) return fail(error, "configuring the frame", hr);
  Com<IWICBitmapSource> converted;
  IWICBitmapSource* source = bitmap.p;
  if (!IsEqualGUID(fmt, srcFormat)) {
    hr = WICConvertBitmapSource(fmt, bitmap.p, &converted);
    if (FAILED(hr)) return fail(error, "pixel format conversion", hr);
    source = converted.p;
  }
  hr = frame->WriteSource(source, nullptr);
  if (SUCCEEDED(hr)) hr = frame->Commit();
  if (SUCCEEDED(hr)) hr = encoder->Commit();
  if (FAILED(hr)) return fail(error, "writing the image", hr);
  return true;
}

}  // namespace

const wchar_t* extension(Format f) {
  switch (f) {
    case Format::Jpeg: return L".jpg";
    case Format::Bmp: return L".bmp";
    default: return L".png";
  }
}

bool writeSdr(const std::wstring& path, Format f, const uint8_t* bgr, uint32_t width, uint32_t height, int quality,
              std::string& error) {
  const GUID& container = f == Format::Jpeg ? GUID_ContainerFormatJpeg : f == Format::Bmp ? GUID_ContainerFormatBmp
                                                                                          : GUID_ContainerFormatPng;
  const float q = f == Format::Jpeg ? float(quality < 1 ? 1 : quality > 100 ? 100 : quality) / 100.0f : -1.0f;
  const bool ok = encode(path, container, bgr, width, height, width * 3, GUID_WICPixelFormat24bppBGR,
                         GUID_WICPixelFormat24bppBGR, q, f == Format::Jpeg, error);
  if (!ok) DeleteFileW(path.c_str());
  return ok;
}

bool writeScRgbJxr(const std::wstring& path, const uint16_t* rgbaHalf, uint32_t width, uint32_t height, int quality,
                   std::string& error) {
  const float q = float(quality < 1 ? 1 : quality > 100 ? 100 : quality) / 100.0f;
  const bool ok = encode(path, GUID_ContainerFormatWmp, reinterpret_cast<const uint8_t*>(rgbaHalf), width, height,
                         width * 8, GUID_WICPixelFormat64bppRGBAHalf, GUID_WICPixelFormat64bppRGBAHalf, q, false, error);
  if (!ok) DeleteFileW(path.c_str());
  return ok;
}

bool read(const std::wstring& path, bool hdr, std::vector<uint8_t>& texels, uint32_t& width, uint32_t& height,
          std::string& error) {
  Com<IWICImagingFactory> factory;
  HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
  if (FAILED(hr)) return fail(error, "WIC factory", hr);
  Com<IWICBitmapDecoder> decoder;
  hr = factory->CreateDecoderFromFilename(path.c_str(), nullptr, GENERIC_READ, WICDecodeMetadataCacheOnDemand, &decoder);
  if (FAILED(hr)) return fail(error, "opening the image", hr);
  Com<IWICBitmapFrameDecode> frame;
  hr = decoder->GetFrame(0, &frame);
  if (FAILED(hr)) return fail(error, "GetFrame", hr);
  Com<IWICBitmapSource> converted;
  const GUID& want = hdr ? GUID_WICPixelFormat64bppRGBAHalf : GUID_WICPixelFormat32bppBGRA;
  hr = WICConvertBitmapSource(want, frame.p, &converted);
  if (SUCCEEDED(hr)) hr = converted->GetSize(&width, &height);
  if (FAILED(hr)) return fail(error, "converting the image", hr);
  const uint32_t stride = width * (hdr ? 8 : 4);
  texels.resize(size_t(stride) * height);
  hr = converted->CopyPixels(nullptr, stride, UINT(texels.size()), texels.data());
  if (FAILED(hr)) return fail(error, "CopyPixels", hr);
  return true;
}

}  // namespace fnvvk::imagefile
