// kernels.cuh - CUDA kernel declarations for Qwen3-ASR (FP32)
#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// cuBLAS wrappers for linear projections
// C[M,N] = A[M,K] @ B[K,N]  (row-major)
// ---------------------------------------------------------------------------
void matmul_f32_cuda(cublasHandle_t handle,
                     const float* A, const float* B, float* C,
                     int M, int K, int N);

// ---------------------------------------------------------------------------
// RoPE 1D (first-half/second-half split, in-place on Q and K)
// Q: [T, NH*HD], K: [T, NKV*HD]
// cos/sin: precomputed tables of size [max_pos, HD/2]
// ---------------------------------------------------------------------------
void rope_1d_f32_cuda(float* d_Q, float* d_K,
                      int T, int NH, int NKV, int HD,
                      const float* d_cos, const float* d_sin,
                      int pos_offset, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Sinusoidal positional encoding
// pe[T, D]: T positions, D = encoder d_model (896)
// ---------------------------------------------------------------------------
void sinusoidal_pe_f32_cuda(float* d_pe, int T, int D, float base,
                            cudaStream_t stream);

// ---------------------------------------------------------------------------
// Fused encoder MHA attention (full, no causal mask)
// Grid: (T, NH), Block: HD (64)
// Q,K,V: [T, NH*HD] interleaved, O: [T, NH*HD]
// ---------------------------------------------------------------------------
void fused_enc_attn_f32_cuda(const float* d_Q, const float* d_K,
                              const float* d_V, float* d_O,
                              int T, int NH, int HD, float scale,
                              cudaStream_t stream);

// ---------------------------------------------------------------------------
// Fused GQA prefill attention (with causal mask)
// Grid: (T, NH), Block: HD (128)
// Q: [T, NH*HD], K,V: [T, NKV*HD], O: [T, NH*HD]
// ---------------------------------------------------------------------------
void fused_gqa_prefill_attn_f32_cuda(const float* d_Q, const float* d_K,
                                      const float* d_V, float* d_O,
                                      int T, int NH, int NKV, int HD,
                                      float scale, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Fused GQA decode attention (single query position, online softmax)
// Grid: NH, Block: HD (128)
// Q: [NH*HD] (single position), K_cache,V_cache: [kv_len, NKV*HD]
// O: [NH*HD] output
// ---------------------------------------------------------------------------
void fused_gqa_decode_attn_f32_cuda(const float* d_Q,
                                     const float* d_K_cache,
                                     const float* d_V_cache,
                                     float* d_O,
                                     int kv_len, int NH, int NKV, int HD,
                                     cudaStream_t stream);

// ---------------------------------------------------------------------------
// Embedding gather: out[i,:] = table[tokens[i], :]
// ---------------------------------------------------------------------------
void embed_gather_f32_cuda(const float* d_table, const int* d_tokens,
                           float* d_out, int num_tokens, int hidden_size,
                           cudaStream_t stream);

// ---------------------------------------------------------------------------
// Residual add: y[i] += x[i]
// ---------------------------------------------------------------------------
void residual_add_f32_cuda(float* d_y, const float* d_x, int N,
                           cudaStream_t stream);

// ---------------------------------------------------------------------------
// SiLU activation (in-place): y[i] = y[i] / (1 + exp(-y[i]))
// ---------------------------------------------------------------------------
void silu_inplace_f32_cuda(float* d_data, int N, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Element-wise multiply: z[i] = x[i] * y[i]
// ---------------------------------------------------------------------------
void mul_f32_cuda(float* d_z, const float* d_x, const float* d_y, int N,
                  cudaStream_t stream);

#ifdef __cplusplus
}
#endif
