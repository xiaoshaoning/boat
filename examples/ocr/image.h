// image.h - Image loading and preprocessing for GLM-OCR
#ifndef BOAT_OCR_IMAGE_H
#define BOAT_OCR_IMAGE_H

#include <boat/tensor.h>

// Get image dimensions without loading pixel data.
// Returns 1 on success, 0 on failure.
int ocr_image_get_dimensions(const char* filename, int* out_w, int* out_h);

// Compute target dimensions for OCR image preprocessing:
// Rounds each dimension to the nearest multiple of 28 (patch_size * spatial_merge)
// to produce a clean grid for the CogViT downsample conv.
// Ensures both H/14 and W/14 are even numbers of patches.
void ocr_compute_target_size(int orig_w, int orig_h, int* out_w, int* out_h);

// Load an image from file, resize to target_w x target_h,
// normalize with given mean/std, and return as a FP32 tensor.
// Output shape: [1, 3, target_h, target_w] (CHW format)
// Returns NULL on failure.
boat_tensor_t* ocr_image_load(const char* filename, int target_w, int target_h,
                               const float mean[3], const float std[3]);

#endif // BOAT_OCR_IMAGE_H
