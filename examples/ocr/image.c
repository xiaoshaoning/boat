// image.c - Image loading and preprocessing using Windows WIC
#include "image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <wincodec.h>
#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")

// WIC helper: convert UTF-8 to wide char
static wchar_t* utf8_to_wchar(const char* utf8) {
    int wlen = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
    wchar_t* wstr = (wchar_t*)malloc(wlen * sizeof(wchar_t));
    if (wstr) MultiByteToWideChar(CP_UTF8, 0, utf8, -1, wstr, wlen);
    return wstr;
}

// Get image dimensions without loading pixel data
int ocr_image_get_dimensions(const char* filename, int* out_w, int* out_h) {
    wchar_t* wpath = utf8_to_wchar(filename);
    if (!wpath) return 0;

    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) { free(wpath); return 0; }

    IWICImagingFactory* factory = NULL;
    IWICBitmapDecoder* decoder = NULL;
    IWICBitmapFrameDecode* frame = NULL;
    int ok = 0;

    hr = CoCreateInstance(&CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER,
                          &IID_IWICImagingFactory, (void**)&factory);
    if (FAILED(hr)) goto cleanup;

    hr = factory->lpVtbl->CreateDecoderFromFilename(factory, wpath, NULL, GENERIC_READ,
                                                     WICDecodeMetadataCacheOnLoad, &decoder);
    if (FAILED(hr)) goto cleanup;

    hr = decoder->lpVtbl->GetFrame(decoder, 0, &frame);
    if (FAILED(hr)) goto cleanup;

    UINT w, h;
    hr = frame->lpVtbl->GetSize(frame, &w, &h);
    if (SUCCEEDED(hr)) {
        *out_w = (int)w;
        *out_h = (int)h;
        ok = 1;
    }

cleanup:
    if (frame) frame->lpVtbl->Release(frame);
    if (decoder) decoder->lpVtbl->Release(decoder);
    if (factory) factory->lpVtbl->Release(factory);
    CoUninitialize();
    free(wpath);
    return ok;
}

void ocr_compute_target_size(int orig_w, int orig_h, int* out_w, int* out_h) {
    // Start by rounding each dimension to the nearest multiple of 28 (patch_size * spatial_merge)
    // so that H/14 and W/14 are both even for a clean 2x2 downsample conv.
    int w = ((orig_w + 14) / 28) * 28;
    int h = ((orig_h + 14) / 28) * 28;

    // Cap the number of patches to prevent OOM in the ViT attention.
    // Max patches (~3456 = 6 tiles worth at 336^2) keeps attention ~800MB.
    int pw = w / 14, ph = h / 14;
    const int MAX_PATCHES = 5000;
    if (pw * ph > MAX_PATCHES) {
        float scale = sqrtf((float)MAX_PATCHES / (float)(pw * ph));
        pw = (int)(pw * scale);
        ph = (int)(ph * scale);
        // Ensure both are even for clean downsample
        if (pw % 2) pw++;
        if (ph % 2) ph++;
        w = pw * 14;
        h = ph * 14;
    }

    *out_w = w;
    *out_h = h;
}

static unsigned char* load_image_wic(const wchar_t* wpath, int* out_w, int* out_h, int* out_ch) {
    HRESULT hr;
    IWICImagingFactory* factory = NULL;
    IWICBitmapDecoder* decoder = NULL;
    IWICBitmapFrameDecode* frame = NULL;
    IWICFormatConverter* converter = NULL;
    unsigned char* pixels = NULL;

    hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) return NULL;

    hr = CoCreateInstance(&CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER,
                          &IID_IWICImagingFactory, (void**)&factory);
    if (FAILED(hr)) { CoUninitialize(); return NULL; }

    hr = factory->lpVtbl->CreateDecoderFromFilename(factory, wpath, NULL, GENERIC_READ,
                                                     WICDecodeMetadataCacheOnLoad, &decoder);
    if (FAILED(hr)) { factory->lpVtbl->Release(factory); CoUninitialize(); return NULL; }

    hr = decoder->lpVtbl->GetFrame(decoder, 0, &frame);
    if (FAILED(hr)) { decoder->lpVtbl->Release(decoder); factory->lpVtbl->Release(factory); CoUninitialize(); return NULL; }

    UINT w, h;
    frame->lpVtbl->GetSize(frame, &w, &h);

    hr = factory->lpVtbl->CreateFormatConverter(factory, &converter);
    if (FAILED(hr)) { frame->lpVtbl->Release(frame); decoder->lpVtbl->Release(decoder); factory->lpVtbl->Release(factory); CoUninitialize(); return NULL; }

    hr = converter->lpVtbl->Initialize(converter, (IWICBitmapSource*)frame, &GUID_WICPixelFormat32bppRGBA,
                                        WICBitmapDitherTypeNone, NULL, 0.0f, WICBitmapPaletteTypeCustom);
    if (FAILED(hr)) { converter->lpVtbl->Release(converter); frame->lpVtbl->Release(frame); decoder->lpVtbl->Release(decoder); factory->lpVtbl->Release(factory); CoUninitialize(); return NULL; }

    UINT stride = w * 4;
    UINT buf_size = stride * h;
    pixels = (unsigned char*)malloc(buf_size);
    if (!pixels) { converter->lpVtbl->Release(converter); frame->lpVtbl->Release(frame); decoder->lpVtbl->Release(decoder); factory->lpVtbl->Release(factory); CoUninitialize(); return NULL; }

    hr = converter->lpVtbl->CopyPixels(converter, NULL, stride, buf_size, pixels);

    converter->lpVtbl->Release(converter);
    frame->lpVtbl->Release(frame);
    decoder->lpVtbl->Release(decoder);
    factory->lpVtbl->Release(factory);
    CoUninitialize();

    if (FAILED(hr)) { free(pixels); return NULL; }

    *out_w = (int)w;
    *out_h = (int)h;
    *out_ch = 4;
    return pixels;
}


// Bilinear resize
static void resize_bilinear(const float* src, int sw, int sh, int sc,
                             float* dst, int dw, int dh) {
    for (int y = 0; y < dh; y++) {
        float sy = (float)y * (sh - 1) / (dh - 1);
        int sy0 = (int)sy;
        int sy1 = sy0 < sh - 1 ? sy0 + 1 : sy0;
        float fy = sy - sy0;
        for (int x = 0; x < dw; x++) {
            float sx = (float)x * (sw - 1) / (dw - 1);
            int sx0 = (int)sx;
            int sx1 = sx0 < sw - 1 ? sx0 + 1 : sx0;
            float fx = sx - sx0;
            for (int c = 0; c < sc; c++) {
                float v00 = src[(sy0 * sw + sx0) * sc + c];
                float v10 = src[(sy0 * sw + sx1) * sc + c];
                float v01 = src[(sy1 * sw + sx0) * sc + c];
                float v11 = src[(sy1 * sw + sx1) * sc + c];
                float v0 = v00 + fx * (v10 - v00);
                float v1 = v01 + fx * (v11 - v01);
                dst[(y * dw + x) * sc + c] = v0 + fy * (v1 - v0);
            }
        }
    }
}

boat_tensor_t* ocr_image_load(const char* filename, int target_w, int target_h,
                               const float mean[3], const float std[3]) {
    unsigned char* img_data = NULL;
    int w = 0, h = 0, channels = 0;

#ifdef _WIN32
    // Try WIC first
    int wlen = MultiByteToWideChar(CP_UTF8, 0, filename, -1, NULL, 0);
    wchar_t* wpath = (wchar_t*)malloc(wlen * sizeof(wchar_t));
    MultiByteToWideChar(CP_UTF8, 0, filename, -1, wpath, wlen);
    img_data = load_image_wic(wpath, &w, &h, &channels);
    free(wpath);
#endif

    if (!img_data) {
        fprintf(stderr, "[ERROR] Cannot load image: %s\n", filename);
        return NULL;
    }

    // Convert RGBA to RGB
    int sc = 3;
    float* src_float = (float*)malloc(w * h * sc * sizeof(float));
    if (!src_float) { free(img_data); return NULL; }

    for (int i = 0; i < w * h; i++) {
        src_float[i * 3 + 0] = img_data[i * channels + 0] / 255.0f;
        src_float[i * 3 + 1] = img_data[i * channels + 1] / 255.0f;
        src_float[i * 3 + 2] = img_data[i * channels + 2] / 255.0f;
    }
    free(img_data);

    // Resize to target_w x target_h
    float* resized = (float*)malloc(target_w * target_h * sc * sizeof(float));
    resize_bilinear(src_float, w, h, sc, resized, target_w, target_h);
    free(src_float);

    // Create boat tensor in CHW format [1, 3, H, W]
    int64_t shape[] = { 1, 3, target_h, target_w };
    boat_tensor_t* tensor = boat_tensor_create(shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!tensor) { free(resized); return NULL; }

    float* tdata = (float*)boat_tensor_data(tensor);
    for (int y = 0; y < target_h; y++) {
        for (int x = 0; x < target_w; x++) {
            for (int c = 0; c < 3; c++) {
                float val = resized[(y * target_w + x) * 3 + c];
                tdata[c * target_h * target_w + y * target_w + x] = (val - mean[c]) / std[c];
            }
        }
    }
    free(resized);

    return tensor;
}
