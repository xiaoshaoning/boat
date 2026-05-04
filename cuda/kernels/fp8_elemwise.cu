// fp8_elemwise.cu — FP8 element-wise CUDA kernels (add, mul, relu, residual)
// Dequantize FP8 → FP32 → compute → quantize FP32 → FP8
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <boat/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(1);                                                      \
    }                                                                 \
} while(0)

// ---------------------------------------------------------------------------
// FP8 ↔ FP32 conversion device helpers (same as fp8_conversion.cu)
// ---------------------------------------------------------------------------
__device__ inline float fp8e4m3_to_float(unsigned char v) {
    if (((v >> 3) & 0x0F) == 0) {
        unsigned int mant = (unsigned int)(v & 0x07);
        if (mant == 0) return 0.0f;
        unsigned int sign = (unsigned int)(v >> 7) << 31;
        unsigned int biased, mant_bits;
        if (mant >= 4) { biased = 120; mant_bits = (mant - 4) << 21; }
        else if (mant >= 2) { biased = 119; mant_bits = (mant - 2) << 22; }
        else { biased = 118; mant_bits = 0; }
        unsigned int fp32_bits = sign | (biased << 23) | mant_bits;
        float result;
        memcpy(&result, &fp32_bits, sizeof(result));
        return result;
    }
    unsigned int sign = (unsigned int)(v >> 7) << 31;
    int exp = (int)((v >> 3) & 0x0F);
    unsigned int mant = (unsigned int)(v & 0x07);
    if (exp == 15 && mant == 7) {
        unsigned int inf_bits = sign | 0x7F800000u;
        float result;
        memcpy(&result, &inf_bits, sizeof(result));
        return result;
    }
    unsigned int biased = (unsigned int)(exp + 120);
    unsigned int fp32_bits = sign | (biased << 23) | (mant << 20);
    float result;
    memcpy(&result, &fp32_bits, sizeof(result));
    return result;
}

__device__ inline unsigned char fp8e4m3_from_float(float x) {
    unsigned int bits = __float_as_int(x);
    unsigned int fp8_sign = (bits >> 24) & 0x80;
    int exp_biased = (int)((bits >> 23) & 0xFF);
    unsigned int mant_fp32 = bits & 0x007FFFFF;

    // NaN / Inf
    if (exp_biased == 255) return 0x7F | fp8_sign;
    // Zero
    if (exp_biased == 0 && mant_fp32 == 0) return 0;

    // Clamp to max finite: 448.0 = 0x43E00000
    if ((bits & 0x7FFFFFFF) > 0x43E00000u) return fp8_sign | 0x7E;

    int exp_unbias = exp_biased - 127;

    // Underflow: flush to zero for values below E4M3 subnormal range (< 2^(-9))
    if (exp_unbias < -9) return 0;

    // E4M3 subnormals: encode as E=0, M = round(x * 512)
    // Value range: [2^(-9), 2^(-6)), maps to M in [1, 7]
    if (exp_unbias < -6) {
        unsigned int full = (1u << 23) | mant_fp32;
        int shift = exp_unbias + 9;
        full = full << shift;
        unsigned int fp8_mant = (full >> 23) & 0x07;
        unsigned int round_bit = (full >> 22) & 1;
        unsigned int sticky = (full & 0x003FFFFFu) != 0 ? 1 : 0;

        if (round_bit && (sticky || (fp8_mant & 1))) {
            fp8_mant++;
            if (fp8_mant == 8) return fp8_sign | 0x08;   // rounded up to min normal
        }
        if (fp8_mant == 0) return 0;
        return fp8_sign | fp8_mant;
    }

    // Normal: e4m3_exp = exp_unbias + 7, range [1, 15]
    int e4m3_exp = exp_unbias + 7;

    unsigned int fp8_mant = (mant_fp32 >> 20) & 0x07;
    unsigned int round_bit = (mant_fp32 >> 19) & 1;
    unsigned int sticky = (mant_fp32 & 0x0007FFFF) != 0 ? 1 : 0;

    if (round_bit && (sticky || (fp8_mant & 1))) {
        fp8_mant++;
        if (fp8_mant == 8) { fp8_mant = 0; e4m3_exp++; }
    }

    // Saturate on overflow
    if (e4m3_exp > 15) return fp8_sign | (15 << 3) | 6;
    if (e4m3_exp == 15 && fp8_mant > 6) fp8_mant = 6;

    return fp8_sign | (unsigned char)(e4m3_exp << 3) | fp8_mant;
}

// ---------------------------------------------------------------------------
// FP8 add: out[i] = a[i] + b[i]
// ---------------------------------------------------------------------------
__global__ void fp8_add_kernel(const unsigned char* __restrict__ a,
                                const unsigned char* __restrict__ b,
                                unsigned char* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float fa = fp8e4m3_to_float(a[i]);
    float fb = fp8e4m3_to_float(b[i]);
    out[i] = fp8e4m3_from_float(fa + fb);
}

void boat_cuda_fp8_add(const void* a, const void* b, void* out, int n) {
    const int block = 256;
    unsigned int grid = (unsigned int)((n + block - 1) / block);
    fp8_add_kernel<<<grid, block>>>((const unsigned char*)a,
                                     (const unsigned char*)b,
                                     (unsigned char*)out, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// FP8 multiply: out[i] = a[i] * b[i]
// ---------------------------------------------------------------------------
__global__ void fp8_mul_kernel(const unsigned char* __restrict__ a,
                                const unsigned char* __restrict__ b,
                                unsigned char* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float fa = fp8e4m3_to_float(a[i]);
    float fb = fp8e4m3_to_float(b[i]);
    out[i] = fp8e4m3_from_float(fa * fb);
}

void boat_cuda_fp8_mul(const void* a, const void* b, void* out, int n) {
    const int block = 256;
    unsigned int grid = (unsigned int)((n + block - 1) / block);
    fp8_mul_kernel<<<grid, block>>>((const unsigned char*)a,
                                     (const unsigned char*)b,
                                     (unsigned char*)out, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// FP8 ReLU: out[i] = max(a[i], 0)
// ---------------------------------------------------------------------------
__global__ void fp8_relu_kernel(const unsigned char* __restrict__ a,
                                 unsigned char* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float fa = fp8e4m3_to_float(a[i]);
    out[i] = (fa > 0.0f) ? a[i] : 0;
}

void boat_cuda_fp8_relu(const void* a, void* out, int n) {
    const int block = 256;
    unsigned int grid = (unsigned int)((n + block - 1) / block);
    fp8_relu_kernel<<<grid, block>>>((const unsigned char*)a,
                                      (unsigned char*)out, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// FP8 residual add: y[i] += x[i]  (in-place accumulation, FP8 domain)
// ---------------------------------------------------------------------------
__global__ void fp8_residual_add_kernel(unsigned char* __restrict__ y,
                                         const unsigned char* __restrict__ x,
                                         int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float fy = fp8e4m3_to_float(y[i]);
    float fx = fp8e4m3_to_float(x[i]);
    y[i] = fp8e4m3_from_float(fy + fx);
}

void boat_cuda_fp8_residual_add(void* y, const void* x, int n) {
    const int block = 256;
    unsigned int grid = (unsigned int)((n + block - 1) / block);
    fp8_residual_add_kernel<<<grid, block>>>((unsigned char*)y,
                                              (const unsigned char*)x, n);
    CUDA_CHECK(cudaGetLastError());
}
