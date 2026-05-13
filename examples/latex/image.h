// image.h - Minimal BMP/PPM image loader (no external dependencies)
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef NOUGAT_IMAGE_H
#define NOUGAT_IMAGE_H

#include <boat/tensor.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Load an image from a BMP (24-bit) or PPM (P6) file.
// Returns uint8_t RGB pixel data [H, W, 3] — caller must free().
uint8_t* nougat_load_image(const char* path, int* out_w, int* out_h);

// Convert uint8_t RGB [H,W,3] to float32 CHW [3,H,W] tensor,
// normalized to [-1, 1] using mean=0.5, std=0.5 per channel.
// Output tensor shape: [batch=1, C=3, H, W], dtype=float32, device=CPU.
boat_tensor_t* nougat_image_to_tensor(const uint8_t* pixels, int H, int W, boat_device_t device);

#ifdef __cplusplus
}
#endif

#endif // NOUGAT_IMAGE_H
