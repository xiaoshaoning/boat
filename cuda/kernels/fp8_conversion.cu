// fp8_conversion.cu — FP32 ↔ FP8 (E4M3) conversion CUDA kernels
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
// FP32 → FP8 (E4M3) — pure bit manipulation (portable, no math lib calls)
// E4M3: sign=1, exp=4 (bias=7), mant=3
//   Normal:   (-1)^S * 2^(E-7) * (1 + M/8), E in [1, 15], M in [0, 6]
//   Subnormal: (-1)^S * 2^(-6) * M/8,       E=0, M>0
//   Zero:      E=0, M=0
//   NaN:       E=15, M=7
// Max finite: E=15, M=6 => 2^8 * 1.75 = 448
// ---------------------------------------------------------------------------
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
        // M = round(x * 512); compute via fixed-point: x * 2^23 * 2^(exp_unbias+9) / 2^23
        unsigned int full = (1u << 23) | mant_fp32;          // 24-bit significand
        int shift = exp_unbias + 9;                           // shift in [0, 2]
        full = full << shift;                                 // now full = M * 2^23 + fractional
        unsigned int fp8_mant = (full >> 23) & 0x07;
        unsigned int round_bit = (full >> 22) & 1;
        unsigned int sticky = (full & 0x003FFFFFu) != 0 ? 1 : 0;

        if (round_bit && (sticky || (fp8_mant & 1))) {
            fp8_mant++;
            if (fp8_mant == 8) return 0x08;                   // rounded up to min normal (positive)
        }
        if (fp8_mant == 0) return 0;
        return fp8_sign | fp8_mant;
    }

    // Normal: e4m3_exp = exp_unbias + 7, range [1, 15]
    int e4m3_exp = exp_unbias + 7;

    // Extract 3 mantissa bits from FP32 (bits 22..20) with RNE on bit 19
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

__global__ void fp32_to_fp8_kernel(const float* __restrict__ in,
                                    unsigned char* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        out[i] = fp8e4m3_from_float(in[i]);
}

void boat_cuda_fp32_to_fp8(const float* in, void* out, int n) {
    const int block = 256;
    unsigned int grid = (unsigned int)((n + block - 1) / block);
    fp32_to_fp8_kernel<<<grid, block>>>(in, (unsigned char*)out, n);
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// FP8 (E4M3) → FP32 — manual bit manipulation
// Format: S(1) E(4) M(3), bias=7
// Normal:  value = (-1)^S * 2^(E-7) * (1 + M/8)
// Subnormal (E==0, M>0): value = (-1)^S * 2^(-6) * (M/8)
// Zero (E==0, M==0): 0
// E==15: NaN (M>0) or Inf (M==0)
// ---------------------------------------------------------------------------
__device__ inline float fp8e4m3_to_float(unsigned char v) {
    unsigned int sign = (unsigned int)(v >> 7) << 31;
    int exp = (int)((v >> 3) & 0x0F);
    unsigned int mant = (unsigned int)(v & 0x07);

    if (exp == 0) {
        if (mant == 0) return 0.0f;
        unsigned int sign = (unsigned int)(v >> 7) << 31;
        // Subnormal: value = (-1)^S * 2^(-6) * mant/8 = (-1)^S * mant * 2^(-9)
        // Renormalize into FP32 normal form
        unsigned int biased, mant_bits;
        if (mant >= 4) { biased = 120; mant_bits = (mant - 4) << 21; }
        else if (mant >= 2) { biased = 119; mant_bits = (mant - 2) << 22; }
        else { biased = 118; mant_bits = 0; }
        unsigned int fp32_bits = sign | (biased << 23) | mant_bits;
        float result;
        memcpy(&result, &fp32_bits, sizeof(result));
        return result;
    }
    if (exp == 15 && mant == 7) {
        // NaN or Inf — treat as Inf with sign
        unsigned int inf_bits = sign | 0x7F800000u;
        float result;
        memcpy(&result, &inf_bits, sizeof(result));
        return result;
    }

    // Normal: (-1)^S * 2^(exp-7) * (1 + mant/8)
    // FP32 exponent = exp - 7 + 127 = exp + 120
    unsigned int biased = (unsigned int)(exp + 120);
    unsigned int fp32_bits = sign | (biased << 23) | (mant << 20);
    float result;
    memcpy(&result, &fp32_bits, sizeof(result));
    return result;
}

__global__ void fp8_to_fp32_kernel(const unsigned char* __restrict__ in,
                                    float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fp8e4m3_to_float(in[i]);
}

void boat_cuda_fp8_to_fp32(const void* in, float* out, int n) {
    const int block = 256;
    unsigned int grid = (unsigned int)((n + block - 1) / block);
    fp8_to_fp32_kernel<<<grid, block>>>((const unsigned char*)in, out, n);
    CUDA_CHECK(cudaGetLastError());
}
