// training.cu — NanoChat full training step: forward, backward, optimizer
#include "training.h"
#include "model.h"
#include "config.h"
#include "nanochat_kernels.cuh"
#include "nanochat_kernels_bw.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <boat/cuda_runtime.h>

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "[CUDA-TRAIN] %s:%d: error %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define CUBLAS_CHECK(call) do {                                         \
    cublasStatus_t stat = call;                                         \
    if (stat != CUBLAS_STATUS_SUCCESS) {                                \
        fprintf(stderr, "[cuBLAS-TRAIN] %s:%d: error %d\n",            \
                __FILE__, __LINE__, (int)stat);                         \
        exit(1);                                                        \
    }                                                                   \
} while(0)

extern "C" cublasHandle_t boat_cuda_get_cublas_handle(void);

// ============================================================================
// FP32 gradient helpers for cuBLAS (matching forward layout C = A @ W^T)
// ============================================================================

// dW[N,K] = dC[M,N]^T @ A[M,K] — weight gradient
// A is BF16, dC/dW FP32
static void grad_weight_bt_bf16(cublasHandle_t handle,
    const __nv_bfloat16* A, int M, int K,
    const float* dC, int N,
    float* dW, cudaStream_t stream) {
    float alpha = 1.0f, beta = 0.0f;
    // dW[N,K]_row = dC[M,N]^T @ A[M,K]
    // col-major: dW_col[K,N] = A_col[K,M](OP_N) @ dC_col[N,M](OP_T)
    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_T, K, N, M, &alpha,
        A, CUDA_R_16BF, K,
        dC, CUDA_R_32F, N,
        &beta, dW, CUDA_R_32F, K,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// dA[M,K] = dC[M,N] @ W[N,K] — input gradient
static void grad_input_bt_bf16(cublasHandle_t handle,
    const float* dC, int M, int N,
    const __nv_bfloat16* W, int K,
    float* dA, cudaStream_t stream) {
    float alpha = 1.0f, beta = 0.0f;
    // dA[M,K]_row = dC[M,N] @ W[N,K]
    // col-major: dA_col[K,M] = W_col[K,N](OP_N) @ dC_col[N,M](OP_N)
    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha,
        W, CUDA_R_16BF, K,
        dC, CUDA_R_32F, N,
        &beta, dA, CUDA_R_32F, K,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// FP32 gradient helpers (both inputs FP32)
static void grad_weight_f32(cublasHandle_t handle,
    const float* A, int M, int K,
    const float* dC, int N,
    float* dW, cudaStream_t stream) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_T, K, N, M, &alpha,
        A, CUDA_R_32F, K,
        dC, CUDA_R_32F, N,
        &beta, dW, CUDA_R_32F, K,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

static void grad_input_f32(cublasHandle_t handle,
    const float* dC, int M, int N,
    const float* W, int K,
    float* dA, cudaStream_t stream) {
    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha,
        W, CUDA_R_32F, K,
        dC, CUDA_R_32F, N,
        &beta, dA, CUDA_R_32F, K,
        CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// FP32 element-wise add: y[i] += x[i]
static __global__ void add_f32_kernel(float* y, const float* x, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] += x[i];
}
static void add_f32_cuda(float* y, const float* x, int N, cudaStream_t stream) {
    const int block = 256;
    unsigned int grid = (unsigned int)((N + block - 1) / block);
    add_f32_kernel<<<grid, block, 0, stream>>>(y, x, N);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Allocate training buffers
// ============================================================================
int nanochat_cuda_model_alloc_train(nanochat_cuda_model_t* model, int max_T) {
    if (!model || max_T < 2) return 0;
    model->act_capacity = max_T;
    model->f32_capacity = max_T;

    size_t H = (size_t)model->hidden_size;
    size_t V = (size_t)model->vocab_size;
    size_t nH = (size_t)model->num_heads;
    int L = model->n_layers;

    // Gradient + state buffer size (same layout)
    size_t grad_total = TRAIN_GRAD_TOTAL(model);

    // Activation storage (BF16 per-layer saved activations)
    size_t act_total = ACT_BUF_TOTAL(model, max_T);

    // FP32 workspace: logits [T,V] + P [H,T,T] + flow [T,H] + attn_ws [H,T,T]
    size_t f32_total = (size_t)max_T * V + 2ULL * nH * (size_t)max_T * (size_t)max_T
                       + (size_t)max_T * H;

    // BF16 training temp: T*(5*H+FF) for forward (QKV temps + FF + saved embed)
    size_t tmp_elements = (size_t)max_T * (5ULL * (size_t)model->hidden_size
                            + (size_t)model->intermediate_size);

    // Allocate
    CUDA_CHECK(cudaMalloc(&model->d_grad_buf, grad_total * sizeof(float)));
    CUDA_CHECK(cudaMemset(model->d_grad_buf, 0, grad_total * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&model->d_m_buf, grad_total * sizeof(float)));
    CUDA_CHECK(cudaMemset(model->d_m_buf, 0, grad_total * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&model->d_v_buf, grad_total * sizeof(float)));
    CUDA_CHECK(cudaMemset(model->d_v_buf, 0, grad_total * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&model->d_act_buf, act_total * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemset(model->d_act_buf, 0, act_total * sizeof(__nv_bfloat16)));

    CUDA_CHECK(cudaMalloc(&model->d_f32_buf, f32_total * sizeof(float)));
    CUDA_CHECK(cudaMemset(model->d_f32_buf, 0, f32_total * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&model->d_train_tmp_bf16, tmp_elements * sizeof(__nv_bfloat16)));

    fprintf(stderr, "[NanoChat-Train] Allocated: grad+m+v=%.1fMB act=%.1fMB f32=%.1fMB tmp=%.1fMB\n",
            grad_total * 12e-6f, act_total * 2e-6f, f32_total * 4e-6f, tmp_elements * 2e-6f);
    return 1;
}

void nanochat_cuda_model_free_train(nanochat_cuda_model_t* model) {
    if (!model) return;
    #define FREE_T(p) do { if (p) { cudaFree(p); (p) = NULL; } } while(0)
    FREE_T(model->d_grad_buf);
    FREE_T(model->d_m_buf);
    FREE_T(model->d_v_buf);
    FREE_T(model->d_act_buf);
    FREE_T(model->d_f32_buf);
    FREE_T(model->d_train_tmp_bf16);
    #undef FREE_T
}

// ============================================================================
// Training forward: embed → 34 layers (saving activations) → lm_head → softcap
// ============================================================================
static void train_forward(nanochat_cuda_model_t* model,
    const int* d_tokens, int T,
    __nv_bfloat16* d_act, float* d_f32,
    __nv_bfloat16* d_tmp_bf16, cudaStream_t stream)
{
    cublasHandle_t handle = boat_cuda_get_cublas_handle();
    int H = model->hidden_size;
    int V = model->vocab_size;
    int FF = model->intermediate_size;
    int L = model->n_layers;
    float scale = 1.0f / sqrtf((float)model->head_dim);
    size_t TH = (size_t)T * H;
    size_t TFF = (size_t)T * FF;

    // Set RoPE positions (0, 1, ..., T-1)
    int* h_pos = (int*)malloc((size_t)T * sizeof(int));
    for (int i = 0; i < T; i++) h_pos[i] = i;
    CUDA_CHECK(cudaMemcpyAsync(model->d_pos, h_pos, (size_t)T * sizeof(int),
                                cudaMemcpyHostToDevice, stream));
    free(h_pos);

    // BF16 temp sub-buffers (uses d_tmp_bf16, sized T*(4*H+FF) = T*17408 for H=2176, FF=8704)
    // Layout (freed after use allows reuse):
    //   d_n_attn [T,H] — offset 0, freed after QKV proj
    //   d_q_buf  [T,H] — offset TH, freed after QK-norm save, reused for o_tmp
    //   d_k_buf  [T,H] — offset 2*TH, freed after QK-norm save, reused for n_mlp
    //   d_v_buf  [T,H] — offset 3*TH, freed after v_save
    //   d_ff_buf [T,FF] — offset 4*TH, used during MLP
    __nv_bfloat16* d_n_attn = d_tmp_bf16;                      // [T,H]
    __nv_bfloat16* d_q_buf  = d_tmp_bf16 + TH;                  // [T,H]
    __nv_bfloat16* d_k_buf  = d_tmp_bf16 + 2 * TH;              // [T,H]
    __nv_bfloat16* d_v_buf  = d_tmp_bf16 + 3 * TH;              // [T,H]
    __nv_bfloat16* d_ff_buf = d_tmp_bf16 + 4 * TH;              // [T,FF]

    // Embed tokens → d_hidden
    __nv_bfloat16* d_hidden = d_act;  // h_in[0]
    embed_gather_bf16_cuda(model->d_embed_tokens, d_tokens, d_hidden, T, H, stream);

    // Save embedding output for pre-layers RMSNorm backward
    // Stored in d_tmp_bf16 extra space (allocated T*(5*H+FF))
    __nv_bfloat16* d_embed_saved = d_tmp_bf16 + 4 * TH + TFF;  // [T,H]
    CUDA_CHECK(cudaMemcpyAsync(d_embed_saved, d_hidden, TH * sizeof(__nv_bfloat16),
                                cudaMemcpyDeviceToDevice, stream));

    // Pre-layers RMSNorm
    rmsnorm_nw_bf16_cuda(d_hidden, d_hidden, T, H, NANOCHAT_RMS_EPS, stream);

    // Run layers
    for (int l = 0; l < L; l++) {
        __nv_bfloat16* h_in   = d_act + ACT_OFFSET_HIN(model, l, T);
        __nv_bfloat16* q_save = d_act + ACT_OFFSET_Q(model, l, T);
        __nv_bfloat16* k_save = d_act + ACT_OFFSET_K(model, l, T);
        __nv_bfloat16* v_save = d_act + ACT_OFFSET_V(model, l, T);
        __nv_bfloat16* a_save = d_act + ACT_OFFSET_ATTN_OUT(model, l, T);
        __nv_bfloat16* f_save = d_act + ACT_OFFSET_FF(model, l, T);

        // Save layer input (except for layer 0 where it's already at h_in[0])
        if (l > 0)
            CUDA_CHECK(cudaMemcpyAsync(h_in, d_hidden, TH * sizeof(__nv_bfloat16),
                                        cudaMemcpyDeviceToDevice, stream));

        // Pre-attention RMSNorm → d_n_attn (separate from d_q_buf to preserve norm for Q/K/V)
        rmsnorm_nw_bf16_cuda(d_hidden, d_n_attn, T, H, NANOCHAT_RMS_EPS, stream);

        // QKV projections: all read from d_n_attn
        matmul_bt_bf16_cuda(handle, d_n_attn, model->d_q_proj[l], d_q_buf, T, H, H);
        matmul_bt_bf16_cuda(handle, d_n_attn, model->d_k_proj[l], d_k_buf, T, H, H);
        matmul_bt_bf16_cuda(handle, d_n_attn, model->d_v_proj[l], d_v_buf, T, H, H);

        // RoPE (in-place)
        apply_rope_bf16_cuda(d_q_buf, d_k_buf, T, model->num_heads, model->num_kv_heads,
                        model->head_dim, NANOCHAT_ROPE_THETA, model->d_pos, stream);

        // QK norm → save
        rmsnorm_nw_bf16_cuda(d_q_buf, q_save, T * model->num_heads, model->head_dim,
                             NANOCHAT_RMS_EPS, stream);
        rmsnorm_nw_bf16_cuda(d_k_buf, k_save, T * model->num_heads, model->head_dim,
                             NANOCHAT_RMS_EPS, stream);
        CUDA_CHECK(cudaMemcpyAsync(v_save, d_v_buf, TH * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToDevice, stream));

        // Training attention forward: uses d_f32[T*V:] for scores/P workspace
        float* d_scores = d_f32 + (size_t)T * model->vocab_size;  // [H,T,T] region
        training_attention_bf16_fwd_cuda(handle,
            q_save, k_save, v_save, a_save,
            d_scores, d_scores, T, model->head_dim, model->num_heads, scale, stream);

        // Output projection: o = attn_out @ W_o^T → d_o_buf
        // d_q_buf is free after RoPE+QK-norm, reuse as o_buf
        __nv_bfloat16* d_o_buf2 = d_q_buf;  // [T,H]
        matmul_bt_bf16_cuda(handle, a_save, model->d_o_proj[l], d_o_buf2, T, H, H);

        // Residual: d_hidden += o
        residual_add_bf16_cuda(d_hidden, d_o_buf2, (int)TH, stream);

        // Pre-MLP RMSNorm: n_mlp = rmsnorm(d_hidden) → d_k_buf (free after QK norm)
        __nv_bfloat16* n_mlp_buf = d_k_buf;  // [T,H]
        rmsnorm_nw_bf16_cuda(d_hidden, n_mlp_buf, T, H, NANOCHAT_RMS_EPS, stream);

        // FC1 + ReLU²
        matmul_bt_bf16_cuda(handle, n_mlp_buf, model->d_fc1[l], d_ff_buf, T, H, FF);
        relu2_bf16_cuda(d_ff_buf, (int)TFF, stream);
        // Save f1_act
        CUDA_CHECK(cudaMemcpyAsync(f_save, d_ff_buf, TFF * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToDevice, stream));

        // FC2 → d_o_buf (=d_q_buf=free)
        matmul_bt_bf16_cuda(handle, d_ff_buf, model->d_fc2[l], d_o_buf2, T, FF, H);

        // Residual: d_hidden += mlp_out
        residual_add_bf16_cuda(d_hidden, d_o_buf2, (int)TH, stream);
    }

    // Final RMSNorm
    rmsnorm_nw_bf16_cuda(d_hidden, d_hidden, T, H, NANOCHAT_RMS_EPS, stream);

    // LM head: logits[T,V] = hidden[T,H] @ lm_head[V,H]^T
    float* d_logits = d_f32;  // [T,V] at start of d_f32
    matmul_bt_bf16_out_f32_cuda(handle, d_hidden, model->d_lm_head, d_logits, T, H, V);

    // Logit softcap
    softcap_cuda(d_logits, T * V, NANOCHAT_SOFTCAP, stream);
}

// ============================================================================
// Embedding backward: scatter-add gradient from each position to the embed row
// ============================================================================
__global__ void embed_gather_bw_f32_kernel(float* d_table, const int* d_tokens,
    const float* d_grad, int T, int H) {
    int i = blockIdx.x;
    int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (i < T && j < H) {
        int token = d_tokens[i];
        atomicAdd(&d_table[(size_t)token * H + j], d_grad[(size_t)i * H + j]);
    }
}

static void embed_gather_bf16_bw_cuda(float* d_table_grad,
    const int* d_tokens, const float* d_embed_grad, int T, int H,
    cudaStream_t stream) {
    dim3 grid((unsigned int)T, (unsigned int)((H + 255) / 256));
    dim3 block(256);
    embed_gather_bw_f32_kernel<<<grid, block, 0, stream>>>(d_table_grad, d_tokens, d_embed_grad, T, H);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Backward pass: lm_head grad → per-layer backprop → weight gradients
// ============================================================================
static void train_backward(nanochat_cuda_model_t* model,
    const int* d_tokens, int T,
    __nv_bfloat16* d_act, float* d_f32,
    __nv_bfloat16* d_tmp_bf16, cudaStream_t stream)
{
    cublasHandle_t handle = boat_cuda_get_cublas_handle();
    int H = model->hidden_size;
    int V = model->vocab_size;
    int FF = model->intermediate_size;
    int L = model->n_layers;
    int nH = model->num_heads;
    int hd = model->head_dim;
    float scale = 1.0f / sqrtf((float)hd);
    size_t TH = (size_t)T * H;
    size_t TFF = (size_t)T * FF;
    int B = T - 1;  // valid batch positions for next-token prediction

    // ------------------------------------------------------------------
    // FP32 workspace layout in d_f32 (total: T*V + 2*H*T*T + T*H):
    //   [0 : T*V)          → logits_grad (Phase 1), then scratch (Phase 2)
    //   [T*V : T*V+H*T*T)  → P_workspace (recompute P per layer)
    //   [T*V+H*T*T : +T*H) → d_flow: gradient flowing through layers
    //   [T*V+H*T*T+T*H :)  → attn_bw_workspace [H*T*T]
    // ------------------------------------------------------------------
    float* d_logits_grad = d_f32;               // [T,V]
    float* d_p_ws        = d_f32 + (size_t)T * V;   // [H,T,T]
    float* d_flow        = d_p_ws + (size_t)nH * T * T;  // [T,H]
    float* d_attn_ws     = d_flow + TH;          // [H,T,T]

    // Scratch region within d_logits_grad (first T*V = 536 MB, available after lm_head grad):
    float* d_scratch = d_f32;  // same as d_logits_grad, reused

    // Hidden state used for lm_head: final RMSNorm output = d_act (h_in[0] region)
    __nv_bfloat16* d_final_hidden = d_act;  // [T,H]

    // ------------------------------------------------------------------
    // 1. CE backward has already been called before this function.
    //    d_logits_grad [T,V] contains the CE+softcap gradient.
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // 2. LM head weight gradient
    //    d_lm_head_grad[V,H] = d_logits_grad[B,V]^T @ d_final_hidden[B,H]
    // ------------------------------------------------------------------
    grad_weight_bt_bf16(handle, d_final_hidden, B, H, d_logits_grad, V,
        TRAIN_GRAD_PTR(model, TRAIN_OFFSET_LMHEAD(model)), stream);

    // ------------------------------------------------------------------
    // 3. Gradient through lm_head to hidden
    //    d_hidden_grad[B,H] = d_logits_grad[B,V] @ lm_head[V,H]
    // ------------------------------------------------------------------
    grad_input_bt_bf16(handle, d_logits_grad, B, V, model->d_lm_head, H, d_flow, stream);
    // Zero out gradient for position B (last position, no loss contribution)
    if (B < T)
        CUDA_CHECK(cudaMemsetAsync(d_flow + (size_t)B * H, 0,
                                    (size_t)(T - B) * H * sizeof(float), stream));

    // ------------------------------------------------------------------
    // 4. Final RMSNorm backward
    //    Need saved input = d_final_hidden. Output written to d_flow.
    // ------------------------------------------------------------------
    rmsnorm_nw_bf16_bw_cuda(d_final_hidden, d_flow, d_flow, T, H,
                            NANOCHAT_RMS_EPS, stream);

    // ------------------------------------------------------------------
    // 5. Per-layer backward loop (reverse order)
    // ------------------------------------------------------------------
    for (int l = L - 1; l >= 0; l--) {
        // Saved activations for this layer (all BF16)
        __nv_bfloat16* h_in     = d_act + ACT_OFFSET_HIN(model, l, T);
        __nv_bfloat16* q_save   = d_act + ACT_OFFSET_Q(model, l, T);
        __nv_bfloat16* k_save   = d_act + ACT_OFFSET_K(model, l, T);
        __nv_bfloat16* v_save   = d_act + ACT_OFFSET_V(model, l, T);
        __nv_bfloat16* a_save   = d_act + ACT_OFFSET_ATTN_OUT(model, l, T);
        __nv_bfloat16* f_save   = d_act + ACT_OFFSET_FF(model, l, T);

        // Gradient pointers for this layer's weights (FP32)
        float* dW_q  = TRAIN_GRAD_PTR(model, TRAIN_OFFSET_Q(model, l));
        float* dW_k  = TRAIN_GRAD_PTR(model, TRAIN_OFFSET_K(model, l));
        float* dW_v  = TRAIN_GRAD_PTR(model, TRAIN_OFFSET_V(model, l));
        float* dW_o  = TRAIN_GRAD_PTR(model, TRAIN_OFFSET_O(model, l));
        float* dW_f1 = TRAIN_GRAD_PTR(model, TRAIN_OFFSET_FC1(model, l));
        float* dW_f2 = TRAIN_GRAD_PTR(model, TRAIN_OFFSET_FC2(model, l));

        // === Phase A: MLP backward ===
        // d_flow [T,H] = gradient d_h_out from upstream.
        // h_out = h_after_attn + f2. So d_h_after_attn_residual = d_flow.

        // A1: FC2 weight grad: dW_f2[H,FF] = f_save[T,FF]^T @ d_flow[T,H]
        grad_weight_bt_bf16(handle, f_save, T, FF, d_flow, H, dW_f2, stream);

        // A2: d_f1_act = d_flow @ W_fc2  [T,FF] = [T,H] @ [H,FF]
        float* d_f1_act = d_scratch;  // T*V ≥ T*FF ✓
        grad_input_bt_bf16(handle, d_flow, T, H, model->d_fc2[l], FF, d_f1_act, stream);

        // A3: ReLU² backward (in-place on d_f1_act)
        float* d_f1 = d_f1_act;  // reuse, same size [T,FF]
        relu2_bf16_bw_cuda(f_save, d_f1_act, d_f1, (int)TFF, stream);

        // A4: Recompute n_mlp = rmsnorm(h_after_attn) for FC1 weight grad.
        //     h_after_attn = h_in + a_save @ W_o^T
        __nv_bfloat16* d_o_tmp = d_tmp_bf16;  // [T,H]
        __nv_bfloat16* d_ha_tmp = d_o_tmp + TH;  // [T,H]
        matmul_bt_bf16_cuda(handle, a_save, model->d_o_proj[l], d_o_tmp, T, H, H);
        CUDA_CHECK(cudaMemcpyAsync(d_ha_tmp, h_in, TH * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToDevice, stream));
        residual_add_bf16_cuda(d_ha_tmp, d_o_tmp, (int)TH, stream);
        rmsnorm_nw_bf16_cuda(d_ha_tmp, d_o_tmp, T, H, NANOCHAT_RMS_EPS, stream);

        // A5: FC1 weight grad: dW_f1[FF,H] = n_mlp[T,H]^T @ d_f1[T,FF]
        grad_weight_bt_bf16(handle, d_o_tmp, T, H, d_f1, FF, dW_f1, stream);

        // A6: d_n_mlp = d_f1 @ W_fc1  [T,H] = [T,FF] @ [FF,H]
        // Use d_scratch + TFF to avoid aliasing with d_f1 (also at d_scratch)
        float* d_n_mlp = d_scratch + TFF;  // [T,H]
        grad_input_bt_bf16(handle, d_f1, T, FF, model->d_fc1[l], H, d_n_mlp, stream);

        // A7: Pre-MLP RMSNorm backward
        // d_h_after_attn += rmsnorm_bw(h_after_attn, d_n_mlp)
        // Write result back to d_n_mlp (in-place)
        rmsnorm_nw_bf16_bw_cuda(d_ha_tmp, d_n_mlp, d_n_mlp, T, H,
                                NANOCHAT_RMS_EPS, stream);
        // Now d_n_mlp = d_h_after_attn gradient from MLP path

        // A8: Add MLP residual: d_h_after_attn_total = d_flow + d_n_mlp
        // Write result to d_flow (which is now the combined gradient for h_after_attn)
        add_f32_cuda(d_flow, d_n_mlp, (int)TH, stream);
        // d_flow now = d_h_after_attn_total (gradient wrt h_after_attn, the input to MLP)

        // === Phase B: O proj backward ===
        // d_h_after_attn = d_flow (gradient flowing through O proj + residual path)
        // h_after_attn = h_in + o, where o = attn_out @ W_o^T
        // dW_o = attn_out^T @ d_h_after_attn  (weight grad)
        // d_a = d_h_after_attn @ W_o^T? NO: d_o = d_h_after_attn (since o = attn_out @ W_o, d_o = d_h_after_attn)
        // Actually: h_after_attn = h_in + attn_out @ W_o^T
        // d_h_in_residual = d_h_after_attn (direct path)
        // d_attn_out = d_h_after_attn @ W_o  (grad through O proj)

        // B1: O proj weight grad: dW_o[H,H] = a_save[T,H]^T @ d_flow[T,H]
        grad_weight_bt_bf16(handle, a_save, T, H, d_flow, H, dW_o, stream);

        // B2: d_a = d_flow @ W_o  [T,H] = [T,H] @ [H,H]
        // In forward: o = a_save @ W_o^T so d_a = d_flow @ W_o
        // Use grad_input: dC[M,N] @ W[N,K] where M=T, N=H, K=H
        float* d_a = d_scratch;  // [T,H], reuse scratch (d_n_mlp consumed)
        grad_input_bt_bf16(handle, d_flow, T, H, model->d_o_proj[l], H, d_a, stream);

        // === Phase C: Attention backward ===
        // Recompute P = softmax(q_save @ k_save^T * scale) via training attention.
        // training_attention_bf16_fwd_cuda computes P and P@V; we keep P and discard P@V.
        training_attention_bf16_fwd_cuda(handle,
            q_save, k_save, v_save, d_o_tmp,
            d_p_ws, d_p_ws, T, hd, nH, scale, stream);
        // Now d_p_ws contains P = softmax(q_save @ k_save^T * scale)

        // C2: Attention backward
        // d_a = gradient at attention output. q_save/k_save/v_save = BF16 forward activations.
        // d_q_grad, d_k_grad, d_v_grad from d_scratch region [3*T,H]
        float* d_q_grad = d_scratch;                  // [T,H]
        float* d_k_grad = d_scratch + TH;              // [T,H]
        float* d_v_grad = d_scratch + 2 * TH;           // [T,H]
        // 3*TH = 3*T*2176*4 = T*26112 bytes = ~53 MB at T=2048, fits in T*V=536 MB ✓

        training_attention_bf16_bw_cuda(handle,
            d_p_ws,              // P (recomputed)
            q_save, k_save, v_save,  // BF16 forward activations
            d_a,                 // gradient of attention output
            d_q_grad, d_k_grad, d_v_grad,
            d_attn_ws,           // workspace [H,T,T]
            T, hd, nH, stream);

        // === Phase D: QK-norm backward + RoPE backward ===
        // QK norm is per-head RMSNorm applied to q_save and k_save.
        // RMSNorm backward for Q: reshape [T*nH, hd]
        rmsnorm_nw_bf16_bw_cuda(q_save, d_q_grad, d_q_grad,
                                T * nH, hd, NANOCHAT_RMS_EPS, stream);
        rmsnorm_nw_bf16_bw_cuda(k_save, d_k_grad, d_k_grad,
                                T * nH, hd, NANOCHAT_RMS_EPS, stream);

        // RoPE backward (transpose rotation, in-place on d_q_grad, d_k_grad)
        // rope_bf16_bw_cuda takes q_rope (unused), d_q_out, writes d_q_in
        rope_bf16_bw_cuda(
            q_save, k_save,        // saved forward Q/K (after RoPE+QKnorm — unused params)
            d_q_grad, d_k_grad,    // FP32 gradients from attention backward
            d_q_grad, d_k_grad,    // FP32 output (after RoPE backward, in-place)
            T, nH, nH, hd, NANOCHAT_ROPE_THETA, model->d_pos, stream);

        // === Phase E: QKV projection weight gradients ===
        // Need n_attn = rmsnorm(h_in). Recompute at d_tmp_bf16 + 2*TH (free after Phase A).
        __nv_bfloat16* d_n_attn_tmp = d_tmp_bf16 + 2 * TH;  // [T,H]
        rmsnorm_nw_bf16_cuda(h_in, d_n_attn_tmp, T, H, NANOCHAT_RMS_EPS, stream);

        // dW_q[H,H] = n_attn[T,H]^T @ d_q_grad[T,H]
        grad_weight_bt_bf16(handle, d_n_attn_tmp, T, H, d_q_grad, H, dW_q, stream);
        grad_weight_bt_bf16(handle, d_n_attn_tmp, T, H, d_k_grad, H, dW_k, stream);
        grad_weight_bt_bf16(handle, d_n_attn_tmp, T, H, d_v_grad, H, dW_v, stream);

        // === Phase F: Gradient through QKV to RMSNorm input ===
        // d_n_attn = d_q_grad @ W_q + d_k_grad @ W_k + d_v_grad @ W_v
        // Use d_scratch + 3*TH (after d_q/k/v_grad regions) to avoid aliasing.
        float* d_n_attn_grad = d_scratch + 3 * TH;  // [T,H]
        grad_input_bt_bf16(handle, d_q_grad, T, H, model->d_q_proj[l], H, d_n_attn_grad, stream);
        float* d_k_proj_grad_tmp = d_scratch + 4 * TH;  // [T,H]
        grad_input_bt_bf16(handle, d_k_grad, T, H, model->d_k_proj[l], H, d_k_proj_grad_tmp, stream);
        add_f32_cuda(d_n_attn_grad, d_k_proj_grad_tmp, (int)TH, stream);
        float* d_v_proj_grad_tmp = d_scratch + 5 * TH;  // [T,H]
        grad_input_bt_bf16(handle, d_v_grad, T, H, model->d_v_proj[l], H, d_v_proj_grad_tmp, stream);
        add_f32_cuda(d_n_attn_grad, d_v_proj_grad_tmp, (int)TH, stream);

        // === Phase G: RMSNorm backward → combine with residual → pass upstream ===
        // d_h_in = rmsnorm_bw(h_in, d_n_attn_grad) + d_flow (residual from attention path)
        rmsnorm_nw_bf16_bw_cuda(h_in, d_n_attn_grad, d_n_attn_grad, T, H,
                                NANOCHAT_RMS_EPS, stream);
        add_f32_cuda(d_n_attn_grad, d_flow, (int)TH, stream);
        CUDA_CHECK(cudaMemcpyAsync(d_flow, d_n_attn_grad, TH * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));
    }

    // ------------------------------------------------------------------
    // 6. Pre-layers RMSNorm backward
    // ------------------------------------------------------------------
    __nv_bfloat16* d_embed_saved = d_tmp_bf16 + 4 * TH + TFF;  // saved before RMSNorm in forward
    rmsnorm_nw_bf16_bw_cuda(d_embed_saved, d_flow, d_flow, T, H,
                            NANOCHAT_RMS_EPS, stream);

    // ------------------------------------------------------------------
    // 7. Embedding table gradient: scatter d_flow to embed table rows
    // ------------------------------------------------------------------
    embed_gather_bf16_bw_cuda(
        TRAIN_GRAD_PTR(model, TRAIN_OFFSET_EMBED(model)),
        d_tokens, d_flow, T, H, stream);
}

// ============================================================================
// Optimizer step: BF16 Adam update on all parameters
// ============================================================================
static void train_optimizer_step(nanochat_cuda_model_t* model,
    float lr, int step, cudaStream_t stream)
{
    // Adam hyperparameters (standard)
    float beta1 = 0.9f, beta2 = 0.999f, eps = 1e-8f;
    float beta1_t = powf(beta1, (float)(step + 1));
    float beta2_t = powf(beta2, (float)(step + 1));

    size_t num_hidden = (size_t)model->vocab_size * model->hidden_size;
    size_t num_hidden_sq = (size_t)model->hidden_size * model->hidden_size;
    size_t num_ff_hidden = (size_t)model->intermediate_size * model->hidden_size;

    // Embed tokens and lm_head
    boat_cuda_adam_update_bf16(
        model->d_embed_tokens,
        TRAIN_GRAD_PTR(model, TRAIN_OFFSET_EMBED(model)),
        TRAIN_M_PTR(model, TRAIN_OFFSET_EMBED(model)),
        TRAIN_V_PTR(model, TRAIN_OFFSET_EMBED(model)),
        lr, beta1, beta2, beta1_t, beta2_t, eps, num_hidden);

    boat_cuda_adam_update_bf16(
        model->d_lm_head,
        TRAIN_GRAD_PTR(model, TRAIN_OFFSET_LMHEAD(model)),
        TRAIN_M_PTR(model, TRAIN_OFFSET_LMHEAD(model)),
        TRAIN_V_PTR(model, TRAIN_OFFSET_LMHEAD(model)),
        lr, beta1, beta2, beta1_t, beta2_t, eps, num_hidden);

    // Per-layer weights
    for (int l = 0; l < model->n_layers; l++) {
        boat_cuda_adam_update_bf16(model->d_q_proj[l],
            TRAIN_GRAD_PTR(model, TRAIN_OFFSET_Q(model, l)),
            TRAIN_M_PTR(model, TRAIN_OFFSET_Q(model, l)),
            TRAIN_V_PTR(model, TRAIN_OFFSET_Q(model, l)),
            lr, beta1, beta2, beta1_t, beta2_t, eps, num_hidden_sq);
        boat_cuda_adam_update_bf16(model->d_k_proj[l],
            TRAIN_GRAD_PTR(model, TRAIN_OFFSET_K(model, l)),
            TRAIN_M_PTR(model, TRAIN_OFFSET_K(model, l)),
            TRAIN_V_PTR(model, TRAIN_OFFSET_K(model, l)),
            lr, beta1, beta2, beta1_t, beta2_t, eps, num_hidden_sq);
        boat_cuda_adam_update_bf16(model->d_v_proj[l],
            TRAIN_GRAD_PTR(model, TRAIN_OFFSET_V(model, l)),
            TRAIN_M_PTR(model, TRAIN_OFFSET_V(model, l)),
            TRAIN_V_PTR(model, TRAIN_OFFSET_V(model, l)),
            lr, beta1, beta2, beta1_t, beta2_t, eps, num_hidden_sq);
        boat_cuda_adam_update_bf16(model->d_o_proj[l],
            TRAIN_GRAD_PTR(model, TRAIN_OFFSET_O(model, l)),
            TRAIN_M_PTR(model, TRAIN_OFFSET_O(model, l)),
            TRAIN_V_PTR(model, TRAIN_OFFSET_O(model, l)),
            lr, beta1, beta2, beta1_t, beta2_t, eps, num_hidden_sq);
        boat_cuda_adam_update_bf16(model->d_fc1[l],
            TRAIN_GRAD_PTR(model, TRAIN_OFFSET_FC1(model, l)),
            TRAIN_M_PTR(model, TRAIN_OFFSET_FC1(model, l)),
            TRAIN_V_PTR(model, TRAIN_OFFSET_FC1(model, l)),
            lr, beta1, beta2, beta1_t, beta2_t, eps, num_ff_hidden);
        boat_cuda_adam_update_bf16(model->d_fc2[l],
            TRAIN_GRAD_PTR(model, TRAIN_OFFSET_FC2(model, l)),
            TRAIN_M_PTR(model, TRAIN_OFFSET_FC2(model, l)),
            TRAIN_V_PTR(model, TRAIN_OFFSET_FC2(model, l)),
            lr, beta1, beta2, beta1_t, beta2_t, eps, num_ff_hidden);
    }

    // Zero out gradients for next step
    size_t grad_total = TRAIN_GRAD_TOTAL(model);
    CUDA_CHECK(cudaMemsetAsync(model->d_grad_buf, 0, grad_total * sizeof(float), stream));
}

// ============================================================================
// Public API: single training step
// ============================================================================
void nanochat_cuda_train_step(nanochat_cuda_model_t* model,
                               const int* d_tokens, int seq_len,
                               float lr, float* h_loss)
{
    if (!model || !model->d_grad_buf || seq_len < 2) {
        fprintf(stderr, "[NanoChat-Train] invalid args or training not allocated\n");
        return;
    }
    if (seq_len > model->act_capacity || seq_len > model->f32_capacity) {
        fprintf(stderr, "[NanoChat-Train] seq_len %d exceeds capacity %d\n",
                seq_len, model->act_capacity);
        return;
    }

    cudaStream_t stream = 0;

    // Phase 1: Forward (embed → layers → lm_head → softcap)
    train_forward(model, d_tokens, seq_len,
                  model->d_act_buf, model->d_f32_buf,
                  model->d_train_tmp_bf16, stream);

    // Phase 2: CE loss + backward (combines softmax, CE, softcap backward)
    // d_logits = d_f32_buf [T,V] — currently contains capped logits
    // targets = d_tokens+1 [T-1] — next-token prediction
    float* d_logits = model->d_f32_buf;
    const int* d_targets = d_tokens + 1;
    int B = seq_len - 1;
    cross_entropy_softcap_bw_cuda(d_logits, d_targets, B, model->vocab_size,
                                   NANOCHAT_SOFTCAP, h_loss, d_logits, stream);

    // Phase 3: Backward (lm_head grad + per-layer backprop)
    train_backward(model, d_tokens, seq_len,
                   model->d_act_buf, model->d_f32_buf,
                   model->d_train_tmp_bf16, stream);

    // Phase 4: Optimizer step (BF16 Adam update + zero grad)
    static int global_step = 0;
    train_optimizer_step(model, lr, global_step++, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
}
