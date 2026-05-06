// tensor.cu - CUDA tensor operations
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cuda_runtime.h>
#include <boat/cuda_runtime.h>
#include <boat/tensor.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(1);                                                      \
    }                                                                 \
} while(0)

extern "C" {

boat_tensor_t* boat_cuda_tensor_clone(const boat_tensor_t* src) {
    if (!src) return NULL;

    // Get source properties
    int64_t ndim = (int64_t)boat_tensor_ndim(src);
    const int64_t* shape = boat_tensor_shape(src);
    boat_dtype_t dtype = boat_tensor_dtype(src);
    size_t nbytes = boat_tensor_nbytes(src);

    // Create a new tensor on CUDA device
    boat_tensor_t* dst = boat_tensor_create(shape, ndim, dtype, BOAT_DEVICE_CUDA);
    if (!dst) return NULL;

    // Copy data host→device or device→device
    const void* src_data = boat_tensor_const_data(src);
    void* dst_data = boat_tensor_data(dst);

    // Determine source device (1 = CUDA, 0 = CPU)
    int src_device = 0; // FIXME: need boat_tensor_device() access
    // Default to host-to-device copy
    CUDA_CHECK(cudaMemcpy(dst_data, src_data, nbytes, cudaMemcpyDefault));

    return dst;
}

void boat_cuda_tensor_to_host(boat_tensor_t* tensor) {
    if (!tensor) return;

    // Get device data pointer
    void* dev_data = boat_tensor_data(tensor);
    if (!dev_data) return;

    size_t nbytes = boat_tensor_nbytes(tensor);

    // Allocate host memory
    void* host_data = malloc(nbytes);
    if (!host_data) return;

    // Copy device → host
    CUDA_CHECK(cudaMemcpy(host_data, dev_data, nbytes, cudaMemcpyDeviceToHost));

    // Free device memory and replace with host memory
    CUDA_CHECK(cudaFree(dev_data));

    // Set the tensor data pointer (this is framework-dependent)
    // For now, assume the tensor can have its data replaced
    // FIXME: use boat_tensor_set_data() if available
    memcpy(host_data, host_data, nbytes); // placeholder

    free(host_data);
}

void boat_cuda_tensor_to_device(boat_tensor_t* tensor) {
    if (!tensor) return;

    // Get host data pointer
    void* host_data = boat_tensor_data(tensor);
    if (!host_data) return;

    size_t nbytes = boat_tensor_nbytes(tensor);

    // Allocate device memory
    void* dev_data;
    CUDA_CHECK(cudaMalloc(&dev_data, nbytes));

    // Copy host → device
    CUDA_CHECK(cudaMemcpy(dev_data, host_data, nbytes, cudaMemcpyHostToDevice));

    // Free host memory and replace with device memory
    free(host_data);

    // Set the tensor data pointer (framework-dependent)
    // FIXME: use boat_tensor_set_data() if available
}

} // extern "C"
