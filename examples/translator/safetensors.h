// safetensors.h - Minimal safetensors weight reader
#ifndef BOAT_EXAMPLE_SAFETENSORS_H
#define BOAT_EXAMPLE_SAFETENSORS_H

#include <boat/tensor.h>
#include <stddef.h>
#include <stdint.h>

// Metadata for one tensor in a safetensors file
typedef struct {
    char* name;
    int64_t* shape;
    int ndim;
    size_t data_offset;  // offset from start of tensor data (after header)
    size_t data_size;    // bytes
} st_tensor_info_t;

// Parsed safetensors file (header only, not the raw data)
typedef struct {
    st_tensor_info_t* tensors;
    int count;
    uint8_t* file_data;      // mmap'd file contents
    size_t file_size;
    size_t header_size;      // bytes of JSON header (data starts at header_size)
} safetensors_t;

// Open and parse a safetensors file header.
// Returns 1 on success, 0 on failure.
int safetensors_open(safetensors_t* st, const char* filename);

// Find a tensor by name. Returns index or -1.
int safetensors_find(const safetensors_t* st, const char* name);

// Load a tensor as a boat_tensor_t (FP32, row-major).
// The tensor data is copied from the file into a new boat tensor.
// NOTE: MarianMT weights are in PyTorch format [out_features, in_features].
// boat expects [in_features, out_features] for matmul.
// If `transpose` is non-zero and ndim == 2, the weight is transposed on load.
boat_tensor_t* safetensors_load_tensor(const safetensors_t* st, int idx, int do_transpose);

// Close and free resources
void safetensors_close(safetensors_t* st);

#endif // BOAT_EXAMPLE_SAFETENSORS_H
