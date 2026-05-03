// safetensors.h - Minimal safetensors weight reader with BF16 support
#ifndef BOAT_EXAMPLE_SAFETENSORS_H
#define BOAT_EXAMPLE_SAFETENSORS_H

#include <boat/tensor.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
    char* name;
    int64_t* shape;
    int ndim;
    char dtype[16];
    size_t data_offset;
    size_t data_size;
} st_tensor_info_t;

typedef struct {
    st_tensor_info_t* tensors;
    int count;
    uint8_t* file_data;
    size_t file_size;
    size_t header_size;
} safetensors_t;

int safetensors_open(safetensors_t* st, const char* filename);
int safetensors_find(const safetensors_t* st, const char* name);
boat_tensor_t* safetensors_load_tensor(const safetensors_t* st, int idx, int do_transpose);
void safetensors_close(safetensors_t* st);

#endif // BOAT_EXAMPLE_SAFETENSORS_H
