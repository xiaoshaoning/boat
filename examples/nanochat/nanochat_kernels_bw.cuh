// nanochat_kernels_bw.cuh — Backward CUDA kernel declarations for NanoChat training
#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// RMSNorm backward (no learnable weight) — BF16 input, FP32 grad I/O
// x: [rows, cols] BF16 — saved input to RMSNorm
// d_y: [rows, cols] FP32 — upstream gradient
// d_x: [rows, cols] FP32 — output gradient
// ---------------------------------------------------------------------------
void rmsnorm_nw_bf16_bw_cuda(
    const __nv_bfloat16* d_x_saved, const float* d_y,
    float* d_x, int rows, int cols, float eps,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// ReLU² backward — BF16 input, FP32 grad I/O
// d_input[i] = (x[i] > 0) ? 2 * x[i] * d_output[i] : 0
// ---------------------------------------------------------------------------
void relu2_bf16_bw_cuda(
    const __nv_bfloat16* d_saved, const float* d_output,
    float* d_input, int N, cudaStream_t stream);

// ---------------------------------------------------------------------------
// RoPE backward (transpose rotation) — BF16 saved Q/K, FP32 grad I/O
// Forward: v0' = v0*cos + v1*sin,  v1' = v1*cos - v0*sin
// Backward: dv0 = dv0'*cos - dv1'*sin,  dv1 = dv1'*cos + dv0'*sin
// ---------------------------------------------------------------------------
void rope_bf16_bw_cuda(
    const __nv_bfloat16* d_q_rope,      // saved Q after RoPE [seq, n_heads*head_dim]
    const __nv_bfloat16* d_k_rope,      // saved K after RoPE [seq, n_kv_heads*head_dim]
    const float* d_q_out,               // FP32 grad wrt Q after RoPE
    const float* d_k_out,               // FP32 grad wrt K after RoPE
    float* d_q_in,                      // FP32 grad wrt Q before RoPE (output)
    float* d_k_in,                      // FP32 grad wrt K before RoPE (output)
    int seq_len, int num_heads, int num_kv_heads,
    int head_dim, float theta,
    const int* d_pos,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Softmax backward: dS = P * (dP - sum(P * dP, keepdim=True))
// d_prob: [rows, cols] BF16 — saved softmax probabilities P
// d_out:  [rows, cols] FP32 — upstream gradient dP
// d_score:[rows, cols] FP32 — output gradient dS
// ---------------------------------------------------------------------------
void softmax_bw_bf16_cuda(
    const __nv_bfloat16* d_prob, const float* d_out,
    float* d_score, int rows, int cols,
    cudaStream_t stream);

// FP32 variant of softmax backward — for training attention path
void softmax_bw_f32_cuda(
    const float* d_prob, const float* d_out,
    float* d_score, int rows, int cols,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Causal mask: set upper triangle of attention scores to -inf
// d_scores: [num_heads, seq_len, seq_len] FP32 in-place
// ---------------------------------------------------------------------------
void causal_mask_f32_cuda(float* d_scores, int seq_len, int num_heads,
                           cudaStream_t stream);

// ---------------------------------------------------------------------------
// Fused cross-entropy loss + softcap backward
// d_logits: [B, V] FP32 — logits after softcap
// d_targets:[B] int32 — target token IDs
// h_loss:   host output for scalar loss value
// d_raw_grad:[B, V] FP32 — gradient wrt pre-softcap logits (output)
// Optionally pass pre-computed d_logits_raw for softcap derivative reuse
// ---------------------------------------------------------------------------
void cross_entropy_softcap_bw_cuda(
    const float* d_logits_capped,
    const int* d_targets,
    int B, int V, float softcap,
    float* h_loss,
    float* d_raw_grad,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Strided-batched Q @ K^T for attention scores: S[H,T,T] = Q[H,T,d] @ K[H,T,d]^T
// BF16 Q/K input, FP32 S output
// ---------------------------------------------------------------------------
void attn_scores_bf16_cuda(cublasHandle_t handle,
    const __nv_bfloat16* d_q,       // [seq, num_heads*head_dim] BF16
    const __nv_bfloat16* d_k,       // [seq, num_heads*head_dim] BF16
    float* d_scores,                 // [num_heads, seq, seq] FP32 output
    int seq_len, int head_dim, int num_heads,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Strided-batched matmul P @ V: O[H,T,d] = P[H,T,T] @ V[H,T,d]
// FP32 P input, BF16 V input, BF16 O output
// ---------------------------------------------------------------------------
void attn_apply_pv_bf16_cuda(cublasHandle_t handle,
    const float* d_p,               // [num_heads, seq, seq] FP32
    const __nv_bfloat16* d_v,        // [seq, num_heads*head_dim] BF16
    __nv_bfloat16* d_out,            // [seq, num_heads*head_dim] BF16 output
    int seq_len, int head_dim, int num_heads,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Training attention forward (materialized scores, causal mask, softmax)
// Replaces fused kernel during training; saves P for backward
// d_p_saved: [num_heads, seq, seq] FP32 — saved softmax probabilities (output)
// ---------------------------------------------------------------------------
void training_attention_bf16_fwd_cuda(cublasHandle_t handle,
    const __nv_bfloat16* d_q,       // [seq, n_heads*head_dim] BF16 (post-RoPE+QK-norm)
    const __nv_bfloat16* d_k,       // [seq, n_kv_heads*head_dim] BF16
    const __nv_bfloat16* d_v,       // [seq, n_heads*head_dim] BF16
    __nv_bfloat16* d_out,           // [seq, n_heads*head_dim] BF16 output
    float* d_scores,                 // workspace [n_heads, seq, seq] FP32
    float* d_p_saved,               // [n_heads, seq, seq] FP32 — saved P (may alias scores)
    int seq_len, int head_dim, int num_heads, float scale,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Training attention backward
// dQ, dK, dV: [seq, num_heads*head_dim] FP32 output gradients
// ---------------------------------------------------------------------------
void training_attention_bf16_bw_cuda(cublasHandle_t handle,
    const float* d_p_saved,         // [n_heads, seq, seq] FP32 — saved P from forward
    const __nv_bfloat16* d_q,       // [seq, n_heads*head_dim] BF16 — Q from forward (pre-RMSNorm, post-RoPE)
    const __nv_bfloat16* d_k,       // [seq, n_kv_heads*head_dim] BF16 — K from forward
    const __nv_bfloat16* d_v,       // [seq, n_heads*head_dim] BF16 — V from forward
    const float* d_out,             // [seq, n_heads*head_dim] FP32 — grad wrt output
    float* d_q_grad,                // [seq, n_heads*head_dim] FP32 output
    float* d_k_grad,                // [seq, n_heads*head_dim] FP32 output
    float* d_v_grad,                // [seq, n_heads*head_dim] FP32 output
    float* d_workspace,             // [n_heads, seq, seq] FP32 temp
    int seq_len, int head_dim, int num_heads,
    cudaStream_t stream);

// ---------------------------------------------------------------------------
// Mixed-type GemmEx wrapper: C = A @ B^T  (supports mixed BF16/FP32 types)
// type_a/type_b: CUDA_R_16BF or CUDA_R_32F
// ---------------------------------------------------------------------------
void mixed_gemm_nt_cuda(cublasHandle_t handle,
    const void* A, cudaDataType typeA,
    const void* B, cudaDataType typeB,
    void* C, cudaDataType typeC,
    int M, int K, int N,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
