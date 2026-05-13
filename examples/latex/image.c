// image.c - Minimal BMP/PPM image loader implementation
#include "image.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// 24-bit BMP loader (Windows BMP format)
// ---------------------------------------------------------------------------
#pragma pack(push, 1)
typedef struct {
    uint16_t type;      // 'BM'
    uint32_t size;      // file size
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t offset;    // pixel data offset
} bmp_header_t;

typedef struct {
    uint32_t header_size;
    int32_t  width;
    int32_t  height;
    uint16_t planes;
    uint16_t bpp;
    uint32_t compression;
    uint32_t image_size;
    int32_t  x_pels_per_meter;
    int32_t  y_pels_per_meter;
    uint32_t clr_used;
    uint32_t clr_important;
} bmp_info_t;
#pragma pack(pop)

static uint8_t* load_bmp(const char* path, int* w, int* h) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    bmp_header_t bh;
    bmp_info_t bi;
    if (fread(&bh, sizeof(bh), 1, f) != 1 || fread(&bi, sizeof(bi), 1, f) != 1) {
        fclose(f); return NULL;
    }
    if (bh.type != 0x4D42 || bi.bpp != 24 || bi.compression != 0) {
        fclose(f); return NULL;
    }

    int width = bi.width;
    int height = abs((int)bi.height);
    int row_stride = ((width * 24 + 31) / 32) * 4; // BMP rows are 4-byte aligned

    uint8_t* pixels = (uint8_t*)malloc((size_t)width * (size_t)height * 3);
    if (!pixels) { fclose(f); return NULL; }

    uint8_t* row = (uint8_t*)malloc((size_t)row_stride);
    if (!row) { free(pixels); fclose(f); return NULL; }

    // BMP stores rows bottom-to-top if height > 0
    int top_down = bi.height < 0;

    fseek(f, (long)bh.offset, SEEK_SET);
    for (int y = 0; y < height; y++) {
        if (fread(row, 1, (size_t)row_stride, f) != (size_t)row_stride) {
            free(row); free(pixels); fclose(f); return NULL;
        }
        int dst_y = top_down ? y : (height - 1 - y);
        for (int x = 0; x < width; x++) {
            // BMP: BGR -> RGB
            pixels[(size_t)(dst_y * width + x) * 3 + 0] = row[x * 3 + 2];
            pixels[(size_t)(dst_y * width + x) * 3 + 1] = row[x * 3 + 1];
            pixels[(size_t)(dst_y * width + x) * 3 + 2] = row[x * 3 + 0];
        }
    }

    free(row);
    fclose(f);
    *w = width;
    *h = height;
    return pixels;
}

// ---------------------------------------------------------------------------
// PPM P6 loader
// ---------------------------------------------------------------------------
static uint8_t* load_ppm(const char* path, int* w, int* h) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;

    // Read header
    char magic[3];
    int width, height, max_val;
    if (fscanf(f, "%2s %d %d %d", magic, &width, &height, &max_val) != 4) {
        fclose(f); return NULL;
    }
    if (magic[0] != 'P' || magic[1] != '6' || max_val != 255) {
        fclose(f); return NULL;
    }
    // Skip single whitespace after max_val
    fgetc(f);

    uint8_t* pixels = (uint8_t*)malloc((size_t)width * (size_t)height * 3);
    if (!pixels) { fclose(f); return NULL; }

    if (fread(pixels, 1, (size_t)width * (size_t)height * 3, f) !=
        (size_t)width * (size_t)height * 3) {
        free(pixels); fclose(f); return NULL;
    }

    fclose(f);
    *w = width;
    *h = height;
    return pixels;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
uint8_t* nougat_load_image(const char* path, int* out_w, int* out_h) {
    // Try BMP first, then PPM
    uint8_t* pixels = load_bmp(path, out_w, out_h);
    if (pixels) return pixels;
    return load_ppm(path, out_w, out_h);
}

boat_tensor_t* nougat_image_to_tensor(const uint8_t* pixels, int H, int W, boat_device_t device) {
    // Output: [1, 3, H, W]
    int64_t shape[] = { 1, 3, H, W };
    boat_tensor_t* t = boat_tensor_create(shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!t) return NULL;

    float* data = (float*)boat_tensor_data(t);
    // Normalize: pixel = (pixel / 255.0f - 0.5f) / 0.5f = pixel * 2.0f/255.0f - 1.0f
    const float scale = 2.0f / 255.0f;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            size_t src_idx = (size_t)(y * W + x) * 3;
            // CHW layout: channel c, position (y, x)
            size_t dst_r = (size_t)0 * H * W + (size_t)y * W + x;  // R channel
            size_t dst_g = (size_t)1 * H * W + (size_t)y * W + x;  // G channel
            size_t dst_b = (size_t)2 * H * W + (size_t)y * W + x;  // B channel
            data[dst_r] = (float)pixels[src_idx + 0] * scale - 1.0f;
            data[dst_g] = (float)pixels[src_idx + 1] * scale - 1.0f;
            data[dst_b] = (float)pixels[src_idx + 2] * scale - 1.0f;
        }
    }

    // Move to device if needed
    if (device != BOAT_DEVICE_CPU) {
        boat_tensor_t* d = boat_tensor_to_device(t, device);
        boat_tensor_unref(t);
        t = d;
    }

    return t;
}
