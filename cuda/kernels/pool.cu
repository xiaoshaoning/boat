// pool.cu - CUDA pooling kernels
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cuda_runtime.h>
#include <boat/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(1);                                                      \
    }                                                                 \
} while(0)

// ---------------------------------------------------------------------------
// MaxPool2d forward
// Input:  [N, C, H, W]
// Output: [N, C, H_out, W_out]  where H_out = (H - KH)/stride + 1, etc.
// ---------------------------------------------------------------------------
__global__ void maxpool2d_fwd_f32_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t* __restrict__ indices,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t KH, int64_t KW,
    int64_t pad_h, int64_t pad_w,
    int64_t stride_h, int64_t stride_w,
    int64_t H_out, int64_t W_out)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H_out * W_out;
    if (idx >= total) return;

    int64_t w_out = idx % W_out;
    int64_t h_out = (idx / W_out) % H_out;
    int64_t c = (idx / (W_out * H_out)) % C;
    int64_t n = idx / (W_out * H_out * C);

    int64_t h_start = h_out * stride_h - pad_h;
    int64_t w_start = w_out * stride_w - pad_w;
    int64_t h_end = min(h_start + KH, H);
    int64_t w_end = min(w_start + KW, W);
    h_start = max(h_start, (int64_t)0);
    w_start = max(w_start, (int64_t)0);

    float max_val = -FLT_MAX;
    int64_t max_idx = 0;

    for (int64_t kh = h_start; kh < h_end; kh++) {
        for (int64_t kw = w_start; kw < w_end; kw++) {
            int64_t in_idx = ((n * C + c) * H + kh) * W + kw;
            float val = input[in_idx];
            if (val > max_val) {
                max_val = val;
                max_idx = in_idx;
            }
        }
    }

    output[idx] = max_val;
    if (indices) indices[idx] = max_idx;
}

// ---------------------------------------------------------------------------
// MaxPool2d backward: scatter grad_output back to grad_input
// Uses indices from forward pass for simple scatter
// ---------------------------------------------------------------------------
__global__ void maxpool2d_bwd_f32_kernel(
    const float* __restrict__ grad_output,
    const int64_t* __restrict__ indices,
    float* __restrict__ grad_input,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = N * C * H_out * W_out;
    if (idx >= total) return;

    int64_t out_idx = indices[idx];
    atomicAdd(&grad_input[out_idx], grad_output[idx]);
}

extern "C" {

void boat_cuda_maxpool2d_forward_f32(
    const float* input, float* output, int64_t* indices,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t KH, int64_t KW,
    int64_t pad_h, int64_t pad_w,
    int64_t stride_h, int64_t stride_w,
    int64_t H_out, int64_t W_out)
{
    int64_t total = N * C * H_out * W_out;
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((total + block - 1) / block);
    maxpool2d_fwd_f32_kernel<<<grid, block>>>(
        input, output, indices,
        N, C, H, W, KH, KW,
        pad_h, pad_w, stride_h, stride_w,
        H_out, W_out);
    CUDA_CHECK(cudaGetLastError());
}

void boat_cuda_maxpool2d_backward_f32(
    const float* grad_output, const int64_t* indices,
    float* grad_input,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t H_out, int64_t W_out)
{
    int64_t total = N * C * H_out * W_out;
    const unsigned int block = 256;
    const unsigned int grid = (unsigned int)((total + block - 1) / block);
    maxpool2d_bwd_f32_kernel<<<grid, block>>>(
        grad_output, indices, grad_input,
        N, C, H, W, H_out, W_out);
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"
