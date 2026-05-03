// ocr_cuda_kernels.cuh - CUDA kernel declarations for OCR operations
#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <boat/cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------------------------------------------------------------
// cuBLAS transposed matmul: C[M,N] = A[M,K] @ W[N,K]^T
// W is stored as [N,K] (out_features, in_features) row-major.
// ----------------------------------------------------------------------------
void matmul_bt_cuda(cublasHandle_t handle,
                    const float* A, const float* W, float* C,
                    int M, int K, int N);

// ----------------------------------------------------------------------------
// Fused matmul_bt + bias: C[M,N] = A[M,K] @ W[N,K]^T + bias[N]
// ----------------------------------------------------------------------------
void matmul_bt_bias_cuda(cublasHandle_t handle,
                          const float* A, const float* W, const float* bias,
                          float* C, int M, int K, int N);

// ----------------------------------------------------------------------------
// CogViT patch embedding (custom conv2d with temporal merge)
// Weight: [1024, 6, 14, 14] (temporal_patch_size=2 merged into channels)
// Input:  [1, 3, H, W] in CHW layout
// Output: [1024, H/14, W/14] in CHW layout (channel-first)
// ----------------------------------------------------------------------------
void patch_embed_cuda(const float* d_input, const float* d_weight,
                       const float* d_bias, float* d_output,
                       int H, int W, int C_out, int patch_size,
                       cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
// Patch reorder: row-major -> spatial-block order
// Input:  [C, patch_h, patch_w] (channel-first after patch embed)
// Output: [num_patches, C] where num_patches = patch_h * patch_w,
//         reordered into 2x2 spatial-block order
// ----------------------------------------------------------------------------
void patch_reorder_cuda(const float* d_input, float* d_output,
                         int C, int patch_h, int patch_w,
                         cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
// CogViT 2D RoPE (precompute cos/sin on GPU)
// Output: cos_buf[N, head_dim], sin_buf[N, head_dim] on device
// head_dim = 64, with layout: [h0..h15, w0..w15, h0..h15, w0..w15] per position
// ----------------------------------------------------------------------------
void rope_2d_compute_cuda(float* d_cos, float* d_sin,
                           int patch_h, int patch_w, int head_dim,
                           float theta_base,
                           cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
// Apply 2D RoPE to Q and K buffers (in-place)
// Q/K are [N, num_heads * head_dim] with stride_qk = total dims per token
// ----------------------------------------------------------------------------
void apply_rope_2d_cuda(float* d_q, float* d_k, int N, int num_heads,
                          int head_dim, int stride_qk,
                          const float* d_cos, const float* d_sin,
                          cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
// M-RoPE for GLM: 3D rotary embeddings with temporal/height/width sections
// Q: [seq_len, num_heads * head_dim], K: [seq_len, num_kv_heads * head_dim]
// Applied in-place. head_dim=128, sections: [16(t), 24(h), 24(w)] × 2 repeats
// ----------------------------------------------------------------------------
void apply_mrope_cuda(float* d_q, float* d_k,
                       int seq_len, int num_heads, int num_kv_heads,
                       int head_dim, float theta,
                       const int* d_pos_t, const int* d_pos_h, const int* d_pos_w,
                       cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
// Standard 1D RoPE for GLM decoder (matches CPU apply_rope_glm)
// Q: [seq_len, num_heads * head_dim], K: [seq_len, num_kv_heads * head_dim]
// Rotates consecutive dimension pairs (d, d+1) using a single position per token
// ----------------------------------------------------------------------------
void glm_rope_cuda(float* d_q, float* d_k,
                    int seq_len, int num_heads, int num_kv_heads,
                    int head_dim, float theta,
                    const int* d_pos,
                    cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
// CogViT downsample: group 4 consecutive patches -> 1 output token
// Weight: [1536, 1024, 2, 2], Bias: [1536]
// Input:  [num_patches, 1024]  (in spatial-block order, groups of 4)
// Output: [num_groups, 1536] where num_groups = num_patches / 4
// ----------------------------------------------------------------------------
void downsample_cuda(const float* d_input, const float* d_weight,
                      const float* d_bias, float* d_output,
                      int num_patches, int hidden_size, int out_hidden_size,
                      cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
// SiLU gate activation for interleaved layout (CogViT blocks)
// d is [N, 2*ff_dim] interleaved row-major: [gate_row|up_row] per row
// Output: d[0..N*ff_dim) = silu(gate) * up
// ----------------------------------------------------------------------------
void silu_gate_cuda(float* d_data, int N, int ff_dim,
                     cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
// SiLU gate activation for contiguous layout (merger)
// gate at d[0..M*ff), up at d[M*ff..2M*ff) — contiguous blocks
// Output: d[0..M*ff) = silu(gate) * up (in-place in the gate block)
// ----------------------------------------------------------------------------
void merger_silu_gate_cuda(float* d_data, int M, int ff_dim,
                            cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
// LayerNorm with bias + GELU (fused for merger):
//   y = GELU(LayerNorm(x, weight, bias))
// ----------------------------------------------------------------------------
void layernorm_gelu_cuda(const float* d_x, float* d_y,
                          const float* d_weight, const float* d_bias,
                          int M, int D, float eps,
                          cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
// CogViT batch attention: QK^T -> softmax -> PV for all heads
// Q, K, V: [N, num_heads * head_dim] with stride_qkv = total dims per token
// Output: context [N, num_heads * head_dim]
// ----------------------------------------------------------------------------
void batched_attention_cuda(cublasHandle_t handle,
                             const float* d_q, const float* d_k, const float* d_v,
                             float* d_context, int N, int num_heads, int head_dim,
                             int stride_qkv, float scale,
                             cudaStream_t stream = 0);

// ----------------------------------------------------------------------------
// NaN scan utility: checks entire buffer for NaN/Inf, prints first occurrence
// ----------------------------------------------------------------------------
void nan_scan_cuda(const float* d_buf, int N, const char* label,
                    cudaStream_t stream = 0);

#ifdef __cplusplus
}
#endif
