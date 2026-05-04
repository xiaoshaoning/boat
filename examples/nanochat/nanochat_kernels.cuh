// nanochat_kernels.cuh - CUDA kernel declarations for NanoChat
#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// cuBLAS transposed matmul: C[M,N] = A[M,K] @ W[N,K]^T
// ---------------------------------------------------------------------------
void matmul_bt_cuda(cublasHandle_t handle,
                    const float* A, const float* W, float* C,
                    int M, int K, int N);

// ---------------------------------------------------------------------------
// 1D standard RoPE (in-place on Q and K)
// Q: [seq_len, num_heads * head_dim], K: [seq_len, num_kv_heads * head_dim]
// ---------------------------------------------------------------------------
void apply_rope_cuda(float* d_q, float* d_k,
                     int seq_len, int num_heads, int num_kv_heads,
                     int head_dim, float theta,
                     const int* d_pos,
                     cudaStream_t stream);

// ---------------------------------------------------------------------------
// ReLU² activation: y = relu(x)^2 = (x > 0 ? x : 0)^2
// ---------------------------------------------------------------------------
void relu2_cuda(float* d_data, int N, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Logit softcap: y = cap * tanh(x / cap)
// ---------------------------------------------------------------------------
void softcap_cuda(float* d_data, int N, float cap, cudaStream_t stream);

// ---------------------------------------------------------------------------
// RMSNorm without learnable weight: y_i = x_i * rsqrt(mean_sq + eps)
// ---------------------------------------------------------------------------
void rmsnorm_nw_cuda(const float* d_x, float* d_y,
                     int rows, int cols, float eps,
                     cudaStream_t stream);

// ---------------------------------------------------------------------------
// Fused MHA prefill attention — no cuBLAS, pure CUDA
// Grid: (seq_len, num_heads), Block: head_dim threads
// ---------------------------------------------------------------------------
void fused_prefill_attention_cuda(
    const float* d_q, const float* d_k, const float* d_v,
    float* d_ctx,
    int seq_len, int num_heads, int head_dim,
    float scale, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Fused MHA decode attention — no cuBLAS, pure CUDA
// Grid: num_heads, Block: head_dim threads
// ---------------------------------------------------------------------------
void fused_decode_attention_cuda(
    const float* d_q,
    const float* d_k_cache, const float* d_v_cache,
    float* d_ctx,
    int kv_len, int num_heads, int head_dim,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Embedding gather: out[i, :] = table[tokens[i], :]
// ---------------------------------------------------------------------------
void embed_gather_cuda(const float* d_table, const int* d_tokens,
                       float* d_out, int num_tokens, int hidden_size,
                       cudaStream_t stream);

// ---------------------------------------------------------------------------
// Residual add: y[i] += x[i]
// ---------------------------------------------------------------------------
void residual_add_cuda(float* d_y, const float* d_x, int N, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Simple GEMM: C[M,N] = A[M,K] @ W[N,K]^T (naive kernel for verification)
// ---------------------------------------------------------------------------
void gemm_naive_cuda(const float* d_a, const float* d_w, float* d_c,
                     int M, int K, int N, cudaStream_t stream);

// ---------------------------------------------------------------------------
// Max element-wise difference between two buffers
// ---------------------------------------------------------------------------
void max_diff_cuda(const float* d_a, const float* d_b, int N,
                   float* h_result, cudaStream_t stream);

// ---------------------------------------------------------------------------
// NaN scan utility
// ---------------------------------------------------------------------------
void nan_scan_cuda(const float* d_buf, int N, const char* label,
                   cudaStream_t stream);

#ifdef __cplusplus
}
#endif
