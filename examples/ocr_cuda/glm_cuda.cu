// glm_cuda.cu - CUDA-accelerated GLM decoder
#include "glm_cuda.cuh"
#include "ocr_cuda_kernels.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h>

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                        \
    }                                                                   \
} while(0)

extern "C" cublasHandle_t boat_cuda_get_cublas_handle(void);

#define CUBLAS_CHECK(call) do {                                         \
    cublasStatus_t stat = call;                                         \
    if (stat != CUBLAS_STATUS_SUCCESS) {                                \
        fprintf(stderr, "[cuBLAS] %s:%d: error %d\n",                  \
                __FILE__, __LINE__, (int)stat);                         \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define KV_SIZE (GLM_NUM_KV_HEADS * GLM_HEAD_DIM)

// ----------------------------------------------------------------------------
// Element-wise add: y[i] += x[i]
// ----------------------------------------------------------------------------
__global__ void residual_add_kernel(float* y, const float* x, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) y[idx] += x[idx];
}

// ----------------------------------------------------------------------------
// Copy logits from device to host + return as boat_tensor
// ----------------------------------------------------------------------------
static boat_tensor_t* make_logits_tensor(const float* d_logits, int vocab_size) {
    int64_t shape[] = { 1, vocab_size };
    boat_tensor_t* t = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (t) {
        CUDA_CHECK(cudaMemcpy(boat_tensor_data(t), d_logits,
                               (size_t)vocab_size * sizeof(float),
                               cudaMemcpyDeviceToHost));
    }
    return t;
}

// ----------------------------------------------------------------------------
// Helper: load weight to GPU
// ----------------------------------------------------------------------------
static float* load_w_gpu(safetensors_t* st, const char* name) {
    int idx = safetensors_find(st, name);
    if (idx < 0) { fprintf(stderr, "[GLM-CUDA] Weight not found: %s\n", name); return NULL; }
    boat_tensor_t* t = safetensors_load_tensor(st, idx, 0);
    if (!t) { fprintf(stderr, "[GLM-CUDA] Failed to load: %s\n", name); return NULL; }
    // Debug: print shape info
    int ndim = (int)boat_tensor_ndim(t);
    const int64_t* sh = boat_tensor_shape(t);
    fprintf(stderr, "[GLM-CUDA] load %s shape=[", name);
    for (int i = 0; i < ndim; i++) fprintf(stderr, "%s%lld", i?",":"", (long long)sh[i]);
    fprintf(stderr, "]\n");
    size_t nbytes = boat_tensor_nbytes(t);
    float* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, nbytes));
    CUDA_CHECK(cudaMemcpy(d_ptr, boat_tensor_data(t), nbytes, cudaMemcpyHostToDevice));
    boat_tensor_unref(t);
    return d_ptr;
}

// ----------------------------------------------------------------------------
// Helper kernels for head extraction/writing (GQA prefill)
// ----------------------------------------------------------------------------
__global__ void extract_head_kernel(float* dst, const float* src, int rows, int cols, int src_stride, int src_off) {
    int r = blockIdx.x;
    int c = threadIdx.x;
    if (r >= rows || c >= cols) return;
    dst[r * cols + c] = src[r * src_stride + src_off + c];
}

__global__ void write_head_kernel(float* dst, const float* src, int rows, int cols, int dst_stride, int dst_off) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    int r = idx / cols;
    int c = idx % cols;
    dst[r * dst_stride + dst_off + c] = src[idx];
}

__global__ void causal_softmax_kernel(float* scores, int seq_len) {
    int i = blockIdx.x;
    if (i >= seq_len) return;
    float* row = scores + i * seq_len;
    for (int j = i + 1; j < seq_len; j++) row[j] = -INFINITY;
    float mx = row[0];
    for (int j = 1; j <= i; j++) if (row[j] > mx) mx = row[j];
    float sum = 0.0f;
    for (int j = 0; j <= i; j++) { row[j] = expf(row[j] - mx); sum += row[j]; }
    float inv = 1.0f / sum;
    for (int j = 0; j <= i; j++) row[j] *= inv;
}

// ----------------------------------------------------------------------------
// PV multiply kernel: context[q][d] = sum_{t <= q} scores[q][t] * V[t][d]
// scores stored column-major [seq_len, seq_len] from QK^T after causal softmax:
//   scores[q*seq_len + t] = prob(query q attends key t) = softmax_t(K_t · Q_q)
// V stored row-major [seq_len, kv_size]
// ----------------------------------------------------------------------------
__global__ void gqa_pv_kernel(const float* __restrict__ scores,
                               const float* __restrict__ v,
                               float* __restrict__ context,
                               int seq_len, int head_dim,
                               int q_off, int kv_off, int q_size, int kv_size) {
    int q = blockIdx.x;
    int d = threadIdx.x;
    if (q >= seq_len || d >= head_dim) return;
    float sum = 0.0f;
    for (int t = 0; t <= q; t++)
        sum += scores[q * seq_len + t] * v[t * kv_size + kv_off + d];
    context[q * q_size + q_off + d] = sum;
}

// ----------------------------------------------------------------------------
// GQA prefill attention
// Per-head extract -> cuBLAS QK^T -> causal softmax -> custom PV -> write back
// ----------------------------------------------------------------------------
static void gqa_prefill_attention_cuda(cublasHandle_t handle,
                                        const float* d_q, const float* d_k, const float* d_v,
                                        float* d_context,
                                        int seq_len, int num_heads, int num_kv_heads, int head_dim,
                                        float scale, cudaStream_t stream) {
    int groups = num_heads / num_kv_heads;
    int q_size = num_heads * head_dim;
    int kv_size = num_kv_heads * head_dim;

    float *d_q_h, *d_k_h, *d_s_h;
    CUDA_CHECK(cudaMalloc(&d_q_h, (size_t)seq_len * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k_h, (size_t)seq_len * head_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_s_h, (size_t)seq_len * seq_len * sizeof(float)));

    for (int h = 0; h < num_heads; h++) {
        int kv_h = h / groups;

        // Extract Q_h from strided buffer
        extract_head_kernel<<<seq_len, head_dim, 0, stream>>>(d_q_h, d_q, seq_len, head_dim, q_size, h * head_dim);
        // Extract K_{kv_h}
        extract_head_kernel<<<seq_len, head_dim, 0, stream>>>(d_k_h, d_k, seq_len, head_dim, kv_size, kv_h * head_dim);
        CUDA_CHECK(cudaGetLastError());

        // scores = Q_h @ K_h^T * scale (cuBLAS column-major output: scores[r + c*seq_len] = K_r · Q_c)
        float alpha = scale, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    seq_len, seq_len, head_dim,
                    &alpha, d_k_h, head_dim, d_q_h, head_dim,
                    &beta, d_s_h, seq_len));

        // Causal mask + softmax
        causal_softmax_kernel<<<seq_len, 1, 0, stream>>>(d_s_h, seq_len);
        CUDA_CHECK(cudaGetLastError());

        // context_h[q][d] = sum_{t <= q} scores[q*seq_len + t] * V[t][d]
        gqa_pv_kernel<<<seq_len, head_dim, 0, stream>>>(d_s_h, d_v, d_context,
                                                         seq_len, head_dim,
                                                         h * head_dim, kv_h * head_dim,
                                                         q_size, kv_size);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(d_q_h)); CUDA_CHECK(cudaFree(d_k_h));
    CUDA_CHECK(cudaFree(d_s_h));
}

// ----------------------------------------------------------------------------
// Fused GQA decode attention with KV cache
// Q: [1, q_size], K/V_cache: [kv_len, kv_size]
// Per head: score[j] = Q_h @ K_{kv_h}[j] / sqrt(hd), softmax, context = sum score[j] * V_{kv_h}[j]
// ----------------------------------------------------------------------------
__global__ void gqa_decode_fused_kernel(const float* __restrict__ q,
                                          const float* __restrict__ k_cache,
                                          const float* __restrict__ v_cache,
                                          float* __restrict__ ctx,
                                          int kv_len, int nh, int nkv, int hd, int groups) {
    int h = blockIdx.x;
    if (h >= nh) return;
    int kv_h = h / groups;
    int q_off = h * hd;
    int kv_off = kv_h * hd;
    int kvs = nkv * hd;
    float scale = 1.0f / sqrtf((float)hd);

    extern __shared__ float sh[];
    float* scores = sh;

    int tid = threadIdx.x;
    for (int j = tid; j < kv_len; j += blockDim.x) {
        float sum = 0.0f;
        for (int d = 0; d < hd; d++)
            sum += q[q_off + d] * k_cache[j * kvs + kv_off + d];
        scores[j] = sum * scale;
    }
    __syncthreads();

    if (tid == 0) {
        float mx = -INFINITY;
        for (int j = 0; j < kv_len; j++) if (scores[j] > mx) mx = scores[j];
        float sum = 0.0f;
        for (int j = 0; j < kv_len; j++) { scores[j] = expf(scores[j] - mx); sum += scores[j]; }
        float inv = 1.0f / sum;
        for (int j = 0; j < kv_len; j++) scores[j] *= inv;
    }
    __syncthreads();

    if (tid < hd) {
        float sum = 0.0f;
        for (int j = 0; j < kv_len; j++)
            sum += scores[j] * v_cache[j * kvs + kv_off + tid];
        ctx[q_off + tid] = sum;
    }
}

// ============================================================================
// GLM decoder layer forward (prefill mode)
// ============================================================================
static void glm_layer_prefill_cuda(cublasHandle_t handle,
                                    float* d_hidden,
                                    int seq_len, int hidden_size,
                                    int num_heads, int num_kv_heads, int head_dim,
                                    const float* d_in_ln_w,
                                    const float* d_q_w, const float* d_k_w,
                                    const float* d_v_w, const float* d_o_w,
                                    const float* d_psa_ln_w,
                                    const float* d_pa_ln_w,
                                    const float* d_gate_up_w, const float* d_down_w,
                                    const float* d_pmlp_ln_w,
                                    float* d_k_cache, float* d_v_cache, int* kv_len,
                                    const int* d_pos_t,
                                    const int* d_pos_h,
                                    const int* d_pos_w,
                                    float* d_tmp,
                                    cudaStream_t stream) {
    int q_size = num_heads * head_dim;
    int kv_size = num_kv_heads * head_dim;
    int ff_dim = GLM_INTERMEDIATE_SIZE;

    // Buffer layout in d_tmp:
    float* d_normed = d_tmp;
    float* d_q      = d_tmp + seq_len * hidden_size;
    float* d_k      = d_tmp + seq_len * (hidden_size + q_size);
    float* d_v      = d_tmp + seq_len * (hidden_size + q_size + kv_size);
    // d_attn_out: place AFTER d_v to avoid overwriting d_q on long sequences
    float* d_attn_out = d_tmp + seq_len * (hidden_size + q_size + kv_size + kv_size);
    // d_ff placed after d_q (reuses d_q's space after attention is done)
    float* d_ff = d_tmp + seq_len * q_size;
    // d_mlp_out placed AFTER d_ff's max extent (gate_up writes 2*ff_dim)
    float* d_mlp_out = d_tmp + seq_len * (q_size + 2 * ff_dim);

    // 1. Pre-attention RMSNorm
    boat_cuda_rmsnorm_forward_f32(d_hidden, d_in_ln_w, d_normed, seq_len, hidden_size, 1e-5f);

    // 2. QKV projections
    matmul_bt_cuda(handle, d_normed, d_q_w, d_q, seq_len, hidden_size, q_size);
    matmul_bt_cuda(handle, d_normed, d_k_w, d_k, seq_len, hidden_size, kv_size);
    matmul_bt_cuda(handle, d_normed, d_v_w, d_v, seq_len, hidden_size, kv_size);

    // 3. M-RoPE (matches CPU apply_rope_mrope)
    apply_mrope_cuda(d_q, d_k, seq_len, num_heads, num_kv_heads, head_dim,
                     GLM_ROPE_THETA, d_pos_t, d_pos_h, d_pos_w, stream);

    // 4. Store K,V in KV cache
    CUDA_CHECK(cudaMemcpyAsync(d_k_cache, d_k, (size_t)seq_len * kv_size * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v_cache, d_v, (size_t)seq_len * kv_size * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
    *kv_len = seq_len;

    // 5. GQA attention (prefill)
    float scale = 1.0f / sqrtf((float)head_dim);
    gqa_prefill_attention_cuda(handle, d_q, d_k, d_v, d_attn_out,
                                seq_len, num_heads, num_kv_heads, head_dim,
                                scale, stream);
    CUDA_CHECK(cudaGetLastError());

    // 6. Output projection
    matmul_bt_cuda(handle, d_attn_out, d_o_w, d_ff, seq_len, q_size, hidden_size);

    // 7. Post-self-attention RMSNorm + residual
    boat_cuda_rmsnorm_forward_f32(d_ff, d_psa_ln_w, d_normed, seq_len, hidden_size, 1e-5f);
    {
        int total = seq_len * hidden_size;
        const int block = 256;
        unsigned int grid = (unsigned int)((total + block - 1) / block);
        residual_add_kernel<<<grid, block, 0, stream>>>(d_hidden, d_normed, total);
        CUDA_CHECK(cudaGetLastError());
    }

    // 8. Pre-MLP RMSNorm
    boat_cuda_rmsnorm_forward_f32(d_hidden, d_pa_ln_w, d_normed, seq_len, hidden_size, 1e-5f);

    // 9. SiLU FFN
    matmul_bt_cuda(handle, d_normed, d_gate_up_w, d_ff, seq_len, hidden_size, 2 * ff_dim);
    silu_gate_cuda(d_ff, seq_len, ff_dim, stream);
    matmul_bt_cuda(handle, d_ff, d_down_w, d_mlp_out, seq_len, ff_dim, hidden_size);

    // 10. Post-MLP RMSNorm + residual
    boat_cuda_rmsnorm_forward_f32(d_mlp_out, d_pmlp_ln_w, d_normed, seq_len, hidden_size, 1e-5f);
    {
        int total = seq_len * hidden_size;
        const int block = 256;
        unsigned int grid = (unsigned int)((total + block - 1) / block);
        residual_add_kernel<<<grid, block, 0, stream>>>(d_hidden, d_normed, total);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// GLM decoder layer forward (decode mode, single token)
// ============================================================================
static void glm_layer_decode_cuda(cublasHandle_t handle,
                                   float* d_hidden,
                                   int hidden_size,
                                   int num_heads, int num_kv_heads, int head_dim,
                                   const float* d_in_ln_w,
                                   const float* d_q_w, const float* d_k_w,
                                   const float* d_v_w, const float* d_o_w,
                                   const float* d_psa_ln_w,
                                   const float* d_pa_ln_w,
                                   const float* d_gate_up_w, const float* d_down_w,
                                   const float* d_pmlp_ln_w,
                                   float* d_k_cache, float* d_v_cache, int* h_kv_len,
                                   const int* d_pos_t,
                                   const int* d_pos_h,
                                   const int* d_pos_w,
                                   float* d_tmp,
                                   cudaStream_t stream) {
    int q_size = num_heads * head_dim;
    int kv_size = num_kv_heads * head_dim;
    int ff_dim = GLM_INTERMEDIATE_SIZE;
    int kv_len = *h_kv_len;

    // Buffer layout
    float* d_normed = d_tmp;
    float* d_q_tmp = d_tmp + hidden_size;
    float* d_k_tmp = d_tmp + hidden_size + q_size;
    float* d_v_tmp = d_tmp + hidden_size + q_size + kv_size;
    float* d_attn_out = d_normed;

    // 1. Pre-attention RMSNorm
    boat_cuda_rmsnorm_forward_f32(d_hidden, d_in_ln_w, d_normed, 1, hidden_size, 1e-5f);

    // 2. QKV projections
    matmul_bt_cuda(handle, d_normed, d_q_w, d_q_tmp, 1, hidden_size, q_size);
    matmul_bt_cuda(handle, d_normed, d_k_w, d_k_tmp, 1, hidden_size, kv_size);
    matmul_bt_cuda(handle, d_normed, d_v_w, d_v_tmp, 1, hidden_size, kv_size);

    // 3. M-RoPE (matches CPU decode: apply_rope_mrope with abs_pos for T/H/W)
    apply_mrope_cuda(d_q_tmp, d_k_tmp, 1, num_heads, num_kv_heads, head_dim,
                     GLM_ROPE_THETA, d_pos_t, d_pos_h, d_pos_w, stream);

    // 4. Append to KV cache
    CUDA_CHECK(cudaMemcpyAsync(d_k_cache + kv_len * kv_size, d_k_tmp,
                                (size_t)kv_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_v_cache + kv_len * kv_size, d_v_tmp,
                                (size_t)kv_size * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    (*h_kv_len)++;
    int new_kv_len = kv_len + 1;

    // 5. GQA decode attention (fused kernel)
    int groups = num_heads / num_kv_heads;
    float* d_context;
    CUDA_CHECK(cudaMalloc(&d_context, (size_t)q_size * sizeof(float)));

    int shmem = (new_kv_len + head_dim) * sizeof(float);
    int nthreads = 256;

    gqa_decode_fused_kernel<<<num_heads, nthreads, shmem, stream>>>(
        d_q_tmp, d_k_cache, d_v_cache, d_context,
        new_kv_len, num_heads, num_kv_heads, head_dim, groups);
    CUDA_CHECK(cudaGetLastError());

    // 6. Output projection
    matmul_bt_cuda(handle, d_context, d_o_w, d_attn_out, 1, q_size, hidden_size);

    CUDA_CHECK(cudaFree(d_context));

    // 7. Post-self-attention RMSNorm + residual
    boat_cuda_rmsnorm_forward_f32(d_attn_out, d_psa_ln_w, d_normed, 1, hidden_size, 1e-5f);
    {
        const int block = 256;
        unsigned int grid = (unsigned int)((hidden_size + block - 1) / block);
        residual_add_kernel<<<grid, block, 0, stream>>>(d_hidden, d_normed, hidden_size);
        CUDA_CHECK(cudaGetLastError());
    }

    // 8. Pre-MLP RMSNorm
    boat_cuda_rmsnorm_forward_f32(d_hidden, d_pa_ln_w, d_normed, 1, hidden_size, 1e-5f);

    // 9. SiLU FFN
    matmul_bt_cuda(handle, d_normed, d_gate_up_w, d_k_tmp, 1, hidden_size, 2 * ff_dim);
    silu_gate_cuda(d_k_tmp, 1, ff_dim, stream);
    matmul_bt_cuda(handle, d_k_tmp, d_down_w, d_q_tmp, 1, ff_dim, hidden_size);

    // 10. Post-MLP RMSNorm + residual
    boat_cuda_rmsnorm_forward_f32(d_q_tmp, d_pmlp_ln_w, d_normed, 1, hidden_size, 1e-5f);
    {
        const int block = 256;
        unsigned int grid = (unsigned int)((hidden_size + block - 1) / block);
        residual_add_kernel<<<grid, block, 0, stream>>>(d_hidden, d_normed, hidden_size);
        CUDA_CHECK(cudaGetLastError());
    }
}

// ============================================================================
// GLM forward (prefill full sequence)
// ============================================================================
boat_tensor_t* glm_cuda_forward(glm_cuda_model_t* model,
                                 const float* d_input_hidden,
                                 int seq_len,
                                 int prefill_pos_end,
                                 int gen_count,
                                 const int* h_pos_t,
                                 const int* h_pos_h,
                                 const int* h_pos_w) {
    (void)prefill_pos_end;
    (void)gen_count;

    cublasHandle_t handle = boat_cuda_get_cublas_handle();
    cudaStream_t stream = 0;

    // Upload M-RoPE positions to GPU (computed by caller via glm_compute_rope_positions)
    glm_cuda_ensure_pos_bufs(model, seq_len);
    CUDA_CHECK(cudaMemcpy(model->d_pos_t, h_pos_t, (size_t)seq_len * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(model->d_pos_h, h_pos_h, (size_t)seq_len * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(model->d_pos_w, h_pos_w, (size_t)seq_len * sizeof(int), cudaMemcpyHostToDevice));

    int q_size = GLM_NUM_HEADS * GLM_HEAD_DIM;
    int ff_dim = GLM_INTERMEDIATE_SIZE;
    size_t tmp_size = (size_t)seq_len * (q_size + 2 * ff_dim + GLM_HIDDEN_SIZE) * sizeof(float);
    float* d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_size));

    float* d_hidden = (float*)d_input_hidden;

    for (int l = 0; l < GLM_NUM_LAYERS; l++) {
        fprintf(stderr, "[GLM-CUDA] Layer %d/%d prefill\n", l + 1, GLM_NUM_LAYERS);
        glm_layer_prefill_cuda(handle,
                                d_hidden,
                                seq_len, GLM_HIDDEN_SIZE,
                                GLM_NUM_HEADS, GLM_NUM_KV_HEADS, GLM_HEAD_DIM,
                                model->layers[l].d_input_layernorm_weight,
                                model->layers[l].d_q_proj_weight,
                                model->layers[l].d_k_proj_weight,
                                model->layers[l].d_v_proj_weight,
                                model->layers[l].d_o_proj_weight,
                                model->layers[l].d_post_self_attn_layernorm_weight,
                                model->layers[l].d_post_attention_layernorm_weight,
                                model->layers[l].d_gate_up_proj_weight,
                                model->layers[l].d_down_proj_weight,
                                model->layers[l].d_post_mlp_layernorm_weight,
                                model->layers[l].d_k_cache,
                                model->layers[l].d_v_cache,
                                &model->layers[l].seq_len,
                                model->d_pos_t, model->d_pos_h, model->d_pos_w,
                                d_tmp, stream);
        // NaN check after each layer
        {
            float check[2];
            cudaMemcpy(check, d_hidden + (seq_len - 1) * GLM_HIDDEN_SIZE, sizeof(check), cudaMemcpyDeviceToHost);
            if (isnan(check[0]) || isinf(check[0]) || isnan(check[1]) || isinf(check[1])) {
                fprintf(stderr, "[GLM-CUDA] LAYER %d: NaN/Inf detected in hidden state!\n", l + 1);
                break;
            }
        }
    }

    // Final RMSNorm
    boat_cuda_rmsnorm_forward_f32(d_hidden, model->d_norm_weight, d_hidden, seq_len, GLM_HIDDEN_SIZE, 1e-5f);

    // LM head: logits[1, vocab] = hidden[last_pos] @ lm_head^T
    int last_pos = seq_len - 1;
    float* d_logits;
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)GLM_VOCAB_SIZE * sizeof(float)));
    const float* d_last_hidden = d_hidden + last_pos * GLM_HIDDEN_SIZE;
    matmul_bt_cuda(handle, d_last_hidden, model->d_lm_head_weight,
                   d_logits, 1, GLM_HIDDEN_SIZE, GLM_VOCAB_SIZE);

    boat_tensor_t* result = make_logits_tensor(d_logits, GLM_VOCAB_SIZE);

    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return result;
}

// ============================================================================
// GLM decode step (single token)
// ============================================================================
boat_tensor_t* glm_cuda_decode_step(glm_cuda_model_t* model,
                                      const float* d_embed,
                                      int abs_pos) {
    cublasHandle_t handle = boat_cuda_get_cublas_handle();
    cudaStream_t stream = 0;

    int q_size = GLM_NUM_HEADS * GLM_HEAD_DIM;
    int kv_size = GLM_NUM_KV_HEADS * GLM_HEAD_DIM;
    int ff_dim = GLM_INTERMEDIATE_SIZE;

    // M-RoPE: text tokens use position = abs_pos for all 3 dims (matching CPU)
    int h_positions[3] = { abs_pos, abs_pos, abs_pos };
    CUDA_CHECK(cudaMemcpy(model->d_pos_t, h_positions, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(model->d_pos_h, h_positions + 1, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(model->d_pos_w, h_positions + 2, sizeof(int), cudaMemcpyHostToDevice));

    size_t tmp_size = (size_t)(GLM_HIDDEN_SIZE + q_size + kv_size + ff_dim + GLM_HIDDEN_SIZE + q_size) * sizeof(float);
    float* d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_size));

    float* d_hidden;
    CUDA_CHECK(cudaMalloc(&d_hidden, (size_t)GLM_HIDDEN_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_hidden, d_embed, (size_t)GLM_HIDDEN_SIZE * sizeof(float),
                           cudaMemcpyDeviceToDevice));

    for (int l = 0; l < GLM_NUM_LAYERS; l++) {
        glm_layer_decode_cuda(handle, d_hidden, GLM_HIDDEN_SIZE,
                               GLM_NUM_HEADS, GLM_NUM_KV_HEADS, GLM_HEAD_DIM,
                               model->layers[l].d_input_layernorm_weight,
                               model->layers[l].d_q_proj_weight,
                               model->layers[l].d_k_proj_weight,
                               model->layers[l].d_v_proj_weight,
                               model->layers[l].d_o_proj_weight,
                               model->layers[l].d_post_self_attn_layernorm_weight,
                               model->layers[l].d_post_attention_layernorm_weight,
                               model->layers[l].d_gate_up_proj_weight,
                               model->layers[l].d_down_proj_weight,
                               model->layers[l].d_post_mlp_layernorm_weight,
                               model->layers[l].d_k_cache,
                               model->layers[l].d_v_cache,
                               &model->layers[l].seq_len,
                               model->d_pos_t, model->d_pos_h, model->d_pos_w,
                               d_tmp, stream);
    }

    // Final RMSNorm
    boat_cuda_rmsnorm_forward_f32(d_hidden, model->d_norm_weight, d_hidden, 1, GLM_HIDDEN_SIZE, 1e-5f);

    // LM head
    float* d_logits;
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)GLM_VOCAB_SIZE * sizeof(float)));
    matmul_bt_cuda(handle, d_hidden, model->d_lm_head_weight,
                   d_logits, 1, GLM_HIDDEN_SIZE, GLM_VOCAB_SIZE);

    boat_tensor_t* result = make_logits_tensor(d_logits, GLM_VOCAB_SIZE);

    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return result;
}

// ============================================================================
// M-RoPE position computation (host side)
// ============================================================================
void glm_compute_rope_positions(int* pos_t, int* pos_h, int* pos_w,
                                 int total_prefill, int vis_start,
                                 int num_vis_tokens, int vis_grid_h, int vis_grid_w) {
    int cur = 0, ti = 0;
    for (; ti < vis_start; ti++)
        pos_t[ti] = pos_h[ti] = pos_w[ti] = cur++;
    for (; ti < vis_start + num_vis_tokens; ti++) {
        int img_idx = ti - vis_start;
        int row = img_idx / vis_grid_w;
        int col = img_idx % vis_grid_w;
        pos_t[ti] = cur;
        pos_h[ti] = cur + row;
        pos_w[ti] = cur + col;
    }
    cur += vis_grid_h > vis_grid_w ? vis_grid_h : vis_grid_w;
    for (; ti < total_prefill; ti++)
        pos_t[ti] = pos_h[ti] = pos_w[ti] = cur++;
}

// ============================================================================
// Model management
// ============================================================================
int glm_cuda_load(glm_cuda_model_t* model, safetensors_t* st) {
    memset(model, 0, sizeof(*model));

    // Embed tokens not loaded on GPU (lookups done on CPU in main program)
    model->d_embed_tokens_weight = NULL;
    model->d_norm_weight = load_w_gpu(st, "model.language_model.norm.weight");
    model->d_lm_head_weight = load_w_gpu(st, "lm_head.weight");
    if (!model->d_lm_head_weight) {
        fprintf(stderr, "[GLM-CUDA] lm_head.weight not found, trying embed_tokens.weight (weight tying)...\n");
        model->d_lm_head_weight = load_w_gpu(st, "model.language_model.embed_tokens.weight");
    }

    if (!model->d_norm_weight) {
        fprintf(stderr, "[GLM-CUDA] Missing norm.weight\n");
        return 0;
    }

    size_t cache_bytes = (size_t)GLM_MAX_SEQ_LEN * KV_SIZE * sizeof(float);

    for (int l = 0; l < GLM_NUM_LAYERS; l++) {
        char name[256];

        snprintf(name, sizeof(name), "model.language_model.layers.%d.input_layernorm.weight", l);
        model->layers[l].d_input_layernorm_weight = load_w_gpu(st, name);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.q_proj.weight", l);
        model->layers[l].d_q_proj_weight = load_w_gpu(st, name);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.k_proj.weight", l);
        model->layers[l].d_k_proj_weight = load_w_gpu(st, name);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.v_proj.weight", l);
        model->layers[l].d_v_proj_weight = load_w_gpu(st, name);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.self_attn.o_proj.weight", l);
        model->layers[l].d_o_proj_weight = load_w_gpu(st, name);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.post_self_attn_layernorm.weight", l);
        model->layers[l].d_post_self_attn_layernorm_weight = load_w_gpu(st, name);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.post_attention_layernorm.weight", l);
        model->layers[l].d_post_attention_layernorm_weight = load_w_gpu(st, name);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.gate_up_proj.weight", l);
        model->layers[l].d_gate_up_proj_weight = load_w_gpu(st, name);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.mlp.down_proj.weight", l);
        model->layers[l].d_down_proj_weight = load_w_gpu(st, name);

        snprintf(name, sizeof(name), "model.language_model.layers.%d.post_mlp_layernorm.weight", l);
        model->layers[l].d_post_mlp_layernorm_weight = load_w_gpu(st, name);

        if (!model->layers[l].d_q_proj_weight || !model->layers[l].d_k_proj_weight ||
            !model->layers[l].d_v_proj_weight || !model->layers[l].d_o_proj_weight) {
            fprintf(stderr, "[GLM-CUDA] Missing attn weights layer %d\n", l);
            return 0;
        }

        CUDA_CHECK(cudaMalloc(&model->layers[l].d_k_cache, cache_bytes));
        CUDA_CHECK(cudaMalloc(&model->layers[l].d_v_cache, cache_bytes));
        CUDA_CHECK(cudaMemset(model->layers[l].d_k_cache, 0, cache_bytes));
        CUDA_CHECK(cudaMemset(model->layers[l].d_v_cache, 0, cache_bytes));
        model->layers[l].seq_len = 0;
    }

    if (!model->d_lm_head_weight) {
        fprintf(stderr, "[GLM-CUDA] Missing lm_head.weight (no fallback found)\n");
        return 0;
    }

    model->pos_buf_size = GLM_MAX_SEQ_LEN;
    CUDA_CHECK(cudaMalloc(&model->d_pos_t, GLM_MAX_SEQ_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&model->d_pos_h, GLM_MAX_SEQ_LEN * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&model->d_pos_w, GLM_MAX_SEQ_LEN * sizeof(int)));

    fprintf(stderr, "[GLM-CUDA] Loaded: %d layers, hidden=%d, heads=%d/%d (all on GPU)\n",
            GLM_NUM_LAYERS, GLM_HIDDEN_SIZE, GLM_NUM_HEADS, GLM_NUM_KV_HEADS);
    return 1;
}

void glm_cuda_kv_cache_reset(glm_cuda_model_t* model) {
    size_t cache_bytes = (size_t)GLM_MAX_SEQ_LEN * KV_SIZE * sizeof(float);
    for (int l = 0; l < GLM_NUM_LAYERS; l++) {
        model->layers[l].seq_len = 0;
        CUDA_CHECK(cudaMemset(model->layers[l].d_k_cache, 0, cache_bytes));
        CUDA_CHECK(cudaMemset(model->layers[l].d_v_cache, 0, cache_bytes));
    }
}

void glm_cuda_ensure_pos_bufs(glm_cuda_model_t* model, int max_seq_len) {
    if (model->pos_buf_size >= max_seq_len) return;
    if (model->d_pos_t) cudaFree(model->d_pos_t);
    if (model->d_pos_h) cudaFree(model->d_pos_h);
    if (model->d_pos_w) cudaFree(model->d_pos_w);
    model->pos_buf_size = max_seq_len;
    CUDA_CHECK(cudaMalloc(&model->d_pos_t, max_seq_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&model->d_pos_h, max_seq_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&model->d_pos_w, max_seq_len * sizeof(int)));
}

static void free_gpu(void* p) { if (p) cudaFree(p); }

void glm_cuda_free(glm_cuda_model_t* model) {
    if (!model) return;
    free_gpu(model->d_embed_tokens_weight);
    free_gpu(model->d_norm_weight);
    free_gpu(model->d_lm_head_weight);
    for (int l = 0; l < GLM_NUM_LAYERS; l++) {
        free_gpu(model->layers[l].d_input_layernorm_weight);
        free_gpu(model->layers[l].d_q_proj_weight);
        free_gpu(model->layers[l].d_k_proj_weight);
        free_gpu(model->layers[l].d_v_proj_weight);
        free_gpu(model->layers[l].d_o_proj_weight);
        free_gpu(model->layers[l].d_post_self_attn_layernorm_weight);
        free_gpu(model->layers[l].d_post_attention_layernorm_weight);
        free_gpu(model->layers[l].d_gate_up_proj_weight);
        free_gpu(model->layers[l].d_down_proj_weight);
        free_gpu(model->layers[l].d_post_mlp_layernorm_weight);
        free_gpu(model->layers[l].d_k_cache);
        free_gpu(model->layers[l].d_v_cache);
    }
    free_gpu(model->d_pos_t);
    free_gpu(model->d_pos_h);
    free_gpu(model->d_pos_w);
    memset(model, 0, sizeof(*model));
}
