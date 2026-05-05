// model.cu — Qwen3-ASR GPU model: weight upload, encoder, decoder
#include "model.h"
#include "kernels.cuh"
#include "../qwen3-asr/config.h"
#include "../qwen3-asr/weights.h"
#include <boat/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                        \
    }                                                                   \
} while(0)

#define CUBLAS_CHECK(call) do {                                         \
    cublasStatus_t stat = call;                                         \
    if (stat != CUBLAS_STATUS_SUCCESS) {                                \
        fprintf(stderr, "[cuBLAS] %s:%d: error %d\n",                  \
                __FILE__, __LINE__, (int)stat);                         \
        exit(1);                                                        \
    }                                                                   \
} while(0)

extern "C" cublasHandle_t boat_cuda_get_cublas_handle(void);

// ---------------------------------------------------------------------------
// Upload CPU float array to GPU
// ---------------------------------------------------------------------------
static float* upload_to_gpu(const float* src, size_t n) {
    float* d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_ptr, src, n * sizeof(float), cudaMemcpyHostToDevice));
    return d_ptr;
}

// ---------------------------------------------------------------------------
// Permute conv3 output: [C, H, W] -> [W, C*H]  (C=480, H=16)
// ---------------------------------------------------------------------------
__global__ void permute_conv_out_kernel(float* out, const float* in,
                                         int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H * W;
    if (idx >= total) return;
    int t = idx % W;
    int hw = idx / W;
    int h = hw % H;
    int f = hw / H;
    out[(size_t)t * (C * H) + (size_t)f * H + h] = in[idx];
}

// ---------------------------------------------------------------------------
// Precompute RoPE cos/sin tables on CPU, upload to GPU
// ---------------------------------------------------------------------------
static void precompute_rope_tables(float** d_cos, float** d_sin,
                                    int max_pos, int head_dim, float theta) {
    int half = head_dim / 2;
    float* h_cos = (float*)malloc((size_t)max_pos * half * sizeof(float));
    float* h_sin = (float*)malloc((size_t)max_pos * half * sizeof(float));
    if (!h_cos || !h_sin) { fprintf(stderr, "OOM for RoPE tables\n"); exit(1); }

    for (int pos = 0; pos < max_pos; pos++) {
        for (int i = 0; i < half; i++) {
            float freq = powf(theta, -2.0f * i / (float)head_dim);
            h_cos[pos * half + i] = cosf(pos * freq);
            h_sin[pos * half + i] = sinf(pos * freq);
        }
    }

    *d_cos = upload_to_gpu(h_cos, (size_t)max_pos * half);
    *d_sin = upload_to_gpu(h_sin, (size_t)max_pos * half);
    free(h_cos);
    free(h_sin);
}

// ============================================================================
// Initialize model: upload all weights to GPU
// ============================================================================
int qwen3asr_cuda_model_init(qwen3asr_cuda_model_t* model,
                              const qwen3asr_weights_t* w) {
    memset(model, 0, sizeof(*model));
    model->enc_num_layers = QWEN3ASR_ENCODER_NUM_LAYERS;
    model->dec_num_layers = QWEN3ASR_DECODER_NUM_LAYERS;

    // ---- Encoder conv weights ----
    model->d_conv1_w = upload_to_gpu(w->conv1_w, 480 * 1 * 3 * 3);
    model->d_conv1_b = upload_to_gpu(w->conv1_b, 480);
    model->d_conv2_w = upload_to_gpu(w->conv2_w, 480 * 480 * 3 * 3);
    model->d_conv2_b = upload_to_gpu(w->conv2_b, 480);
    model->d_conv3_w = upload_to_gpu(w->conv3_w, 480 * 480 * 3 * 3);
    model->d_conv3_b = upload_to_gpu(w->conv3_b, 480);
    model->d_conv_out_w = upload_to_gpu(w->conv_out_w, 7680 * 896);

    // ---- Encoder post-projection ----
    model->d_ln_post_w = upload_to_gpu(w->ln_post_w, 896);
    model->d_ln_post_b = upload_to_gpu(w->ln_post_b, 896);
    model->d_proj1_w = upload_to_gpu(w->proj1_w, 896 * 896);
    model->d_proj1_b = upload_to_gpu(w->proj1_b, 896);
    model->d_proj2_w = upload_to_gpu(w->proj2_w, 1024 * 896);
    model->d_proj2_b = upload_to_gpu(w->proj2_b, 1024);

    // ---- Encoder layers ----
    for (int l = 0; l < QWEN3ASR_ENCODER_NUM_LAYERS; l++) {
        const qwen3asr_encoder_layer_weights_t* lw = &w->encoder_layers[l];
        model->d_enc_q_proj[l] = upload_to_gpu(lw->q_proj, 896 * 896);
        model->d_enc_k_proj[l] = upload_to_gpu(lw->k_proj, 896 * 896);
        model->d_enc_v_proj[l] = upload_to_gpu(lw->v_proj, 896 * 896);
        model->d_enc_o_proj[l] = upload_to_gpu(lw->o_proj, 896 * 896);
        model->d_enc_q_bias[l] = upload_to_gpu(lw->q_bias, 896);
        model->d_enc_k_bias[l] = upload_to_gpu(lw->k_bias, 896);
        model->d_enc_v_bias[l] = upload_to_gpu(lw->v_bias, 896);
        model->d_enc_o_bias[l] = upload_to_gpu(lw->o_bias, 896);
        model->d_enc_attn_ln_w[l] = upload_to_gpu(lw->attn_ln_w, 896);
        model->d_enc_attn_ln_b[l] = upload_to_gpu(lw->attn_ln_b, 896);
        model->d_enc_fc1_w[l] = upload_to_gpu(lw->fc1_w, 896 * 3584);
        model->d_enc_fc1_b[l] = upload_to_gpu(lw->fc1_b, 3584);
        model->d_enc_fc2_w[l] = upload_to_gpu(lw->fc2_w, 3584 * 896);
        model->d_enc_fc2_b[l] = upload_to_gpu(lw->fc2_b, 896);
        model->d_enc_final_ln_w[l] = upload_to_gpu(lw->final_ln_w, 896);
        model->d_enc_final_ln_b[l] = upload_to_gpu(lw->final_ln_b, 896);
    }

    // ---- Decoder weights ----
    model->d_embed_tokens = upload_to_gpu(w->embed_tokens,
        (size_t)QWEN3ASR_VOCAB_SIZE * QWEN3ASR_DECODER_HIDDEN_SIZE);
    model->d_norm_w = upload_to_gpu(w->norm_w, QWEN3ASR_DECODER_HIDDEN_SIZE);
    model->d_lm_head = upload_to_gpu(w->lm_head,
        (size_t)QWEN3ASR_VOCAB_SIZE * QWEN3ASR_DECODER_HIDDEN_SIZE);

    // ---- Decoder layers ----
    for (int l = 0; l < QWEN3ASR_DECODER_NUM_LAYERS; l++) {
        const qwen3asr_decoder_layer_weights_t* lw = &w->decoder_layers[l];
        model->d_dec_q_proj[l] = upload_to_gpu(lw->q_proj,
            (size_t)QWEN3ASR_DECODER_HIDDEN_SIZE * QWEN3ASR_DECODER_NUM_HEADS * QWEN3ASR_DECODER_HEAD_DIM);
        model->d_dec_k_proj[l] = upload_to_gpu(lw->k_proj,
            (size_t)QWEN3ASR_DECODER_HIDDEN_SIZE * QWEN3ASR_DECODER_NUM_KV_HEADS * QWEN3ASR_DECODER_HEAD_DIM);
        model->d_dec_v_proj[l] = upload_to_gpu(lw->v_proj,
            (size_t)QWEN3ASR_DECODER_HIDDEN_SIZE * QWEN3ASR_DECODER_NUM_KV_HEADS * QWEN3ASR_DECODER_HEAD_DIM);
        model->d_dec_o_proj[l] = upload_to_gpu(lw->o_proj,
            (size_t)QWEN3ASR_DECODER_HIDDEN_SIZE * QWEN3ASR_DECODER_NUM_HEADS * QWEN3ASR_DECODER_HEAD_DIM);
        model->d_dec_q_norm[l] = upload_to_gpu(lw->q_norm, QWEN3ASR_DECODER_HEAD_DIM);
        model->d_dec_k_norm[l] = upload_to_gpu(lw->k_norm, QWEN3ASR_DECODER_HEAD_DIM);
        model->d_dec_gate_proj[l] = upload_to_gpu(lw->gate_proj,
            (size_t)QWEN3ASR_DECODER_HIDDEN_SIZE * QWEN3ASR_DECODER_INTERMEDIATE);
        model->d_dec_up_proj[l] = upload_to_gpu(lw->up_proj,
            (size_t)QWEN3ASR_DECODER_HIDDEN_SIZE * QWEN3ASR_DECODER_INTERMEDIATE);
        model->d_dec_down_proj[l] = upload_to_gpu(lw->down_proj,
            (size_t)QWEN3ASR_DECODER_INTERMEDIATE * QWEN3ASR_DECODER_HIDDEN_SIZE);
        model->d_dec_input_ln[l] = upload_to_gpu(lw->input_ln, QWEN3ASR_DECODER_HIDDEN_SIZE);
        model->d_dec_post_attn_ln[l] = upload_to_gpu(lw->post_attn_ln, QWEN3ASR_DECODER_HIDDEN_SIZE);
    }

    // ---- KV cache ----
    int kv_dim = QWEN3ASR_DECODER_NUM_KV_HEADS * QWEN3ASR_DECODER_HEAD_DIM;
    size_t cache_bytes = (size_t)QWEN3ASR_MAX_SEQ_LEN * kv_dim * sizeof(float);
    for (int l = 0; l < QWEN3ASR_DECODER_NUM_LAYERS; l++) {
        CUDA_CHECK(cudaMalloc(&model->d_k_cache[l], cache_bytes));
        CUDA_CHECK(cudaMalloc(&model->d_v_cache[l], cache_bytes));
        CUDA_CHECK(cudaMemset(model->d_k_cache[l], 0, cache_bytes));
        CUDA_CHECK(cudaMemset(model->d_v_cache[l], 0, cache_bytes));
        model->kv_len[l] = 0;
    }

    // ---- RoPE tables ----
    precompute_rope_tables(&model->d_rope_cos, &model->d_rope_sin,
        QWEN3ASR_MAX_SEQ_LEN, QWEN3ASR_DECODER_HEAD_DIM, QWEN3ASR_ROPE_THETA);

    // ---- Pre-allocate temp buffers ----
    // Single token decode buffer [1, 1024]
    CUDA_CHECK(cudaMalloc(&model->d_single,
        (size_t)QWEN3ASR_DECODER_HIDDEN_SIZE * sizeof(float)));

    fprintf(stderr, "[Qwen3-ASR-CUDA] Model loaded: %d encoder layers, %d decoder layers\n",
            QWEN3ASR_ENCODER_NUM_LAYERS, QWEN3ASR_DECODER_NUM_LAYERS);
    return 1;
}

// ============================================================================
// Free all GPU memory
// ============================================================================
void qwen3asr_cuda_model_free(qwen3asr_cuda_model_t* model) {
    if (!model) return;
#define FREE(p) do { if (p) { cudaFree(p); (p) = NULL; } } while(0)
    FREE(model->d_conv1_w); FREE(model->d_conv1_b);
    FREE(model->d_conv2_w); FREE(model->d_conv2_b);
    FREE(model->d_conv3_w); FREE(model->d_conv3_b);
    FREE(model->d_conv_out_w);
    FREE(model->d_ln_post_w); FREE(model->d_ln_post_b);
    FREE(model->d_proj1_w); FREE(model->d_proj1_b);
    FREE(model->d_proj2_w); FREE(model->d_proj2_b);

    for (int l = 0; l < model->enc_num_layers; l++) {
        FREE(model->d_enc_q_proj[l]); FREE(model->d_enc_q_bias[l]);
        FREE(model->d_enc_k_proj[l]); FREE(model->d_enc_k_bias[l]);
        FREE(model->d_enc_v_proj[l]); FREE(model->d_enc_v_bias[l]);
        FREE(model->d_enc_o_proj[l]); FREE(model->d_enc_o_bias[l]);
        FREE(model->d_enc_attn_ln_w[l]); FREE(model->d_enc_attn_ln_b[l]);
        FREE(model->d_enc_fc1_w[l]); FREE(model->d_enc_fc1_b[l]);
        FREE(model->d_enc_fc2_w[l]); FREE(model->d_enc_fc2_b[l]);
        FREE(model->d_enc_final_ln_w[l]); FREE(model->d_enc_final_ln_b[l]);
    }

    FREE(model->d_embed_tokens);
    FREE(model->d_norm_w);
    FREE(model->d_lm_head);

    for (int l = 0; l < model->dec_num_layers; l++) {
        FREE(model->d_dec_q_proj[l]);
        FREE(model->d_dec_k_proj[l]);
        FREE(model->d_dec_v_proj[l]);
        FREE(model->d_dec_o_proj[l]);
        FREE(model->d_dec_q_norm[l]);
        FREE(model->d_dec_k_norm[l]);
        FREE(model->d_dec_gate_proj[l]);
        FREE(model->d_dec_up_proj[l]);
        FREE(model->d_dec_down_proj[l]);
        FREE(model->d_dec_input_ln[l]);
        FREE(model->d_dec_post_attn_ln[l]);
        FREE(model->d_k_cache[l]);
        FREE(model->d_v_cache[l]);
        model->kv_len[l] = 0;
    }

    FREE(model->d_rope_cos);
    FREE(model->d_rope_sin);
    FREE(model->d_enc_tmp);
    FREE(model->d_dec_tmp);
    FREE(model->d_single);
#undef FREE
}

// ============================================================================
// Reset KV cache
// ============================================================================
void qwen3asr_cuda_model_reset_kv(qwen3asr_cuda_model_t* model) {
    int kv_dim = QWEN3ASR_DECODER_NUM_KV_HEADS * QWEN3ASR_DECODER_HEAD_DIM;
    size_t cache_bytes = (size_t)QWEN3ASR_MAX_SEQ_LEN * kv_dim * sizeof(float);
    for (int l = 0; l < QWEN3ASR_DECODER_NUM_LAYERS; l++) {
        model->kv_len[l] = 0;
        CUDA_CHECK(cudaMemset(model->d_k_cache[l], 0, cache_bytes));
        CUDA_CHECK(cudaMemset(model->d_v_cache[l], 0, cache_bytes));
    }
}

// ============================================================================
// Add bias to 2D matrix (in-place): out[i,j] += bias[j]
// ============================================================================
__global__ void add_bias_f32_kernel(float* data, const float* bias,
                                     int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;
    data[idx] += bias[idx % cols];
}

static void add_bias_gpu(float* d_data, const float* d_bias, int rows, int cols) {
    int n = rows * cols;
    const int block = 256;
    unsigned int grid = (unsigned int)((n + block - 1) / block);
    add_bias_f32_kernel<<<grid, block>>>(d_data, d_bias, rows, cols);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// RMSNorm for 2D tensor (with learned weight) — wraps framework kernel
// ============================================================================
static void rmsnorm_gpu(float* d_out, const float* d_x,
                         int rows, int cols, const float* d_weight, float eps) {
    boat_cuda_rmsnorm_forward_f32(d_x, d_weight, d_out, rows, cols, eps);
}

// LayerNorm wrapper
static void layernorm_gpu(float* d_out, const float* d_x,
                           int rows, int cols,
                           const float* d_weight, const float* d_bias, float eps) {
    boat_cuda_layernorm_forward_f32(d_x, d_weight, d_bias, d_out, rows, cols, eps);
}

// ============================================================================
// Encoder forward pass (with chunked conv frontend, matching Python/C reference)
// ============================================================================
float* qwen3asr_cuda_encoder_forward(qwen3asr_cuda_model_t* model,
                                      const float* d_mel, int T_mel) {
    cublasHandle_t handle = boat_cuda_get_cublas_handle();
    cudaStream_t stream = 0;

    int D = QWEN3ASR_ENCODER_D_MODEL;      // 896
    int NH = QWEN3ASR_ENCODER_NUM_HEADS;    // 14
    int HD = QWEN3ASR_ENCODER_HEAD_DIM;     // 64
    int FF = QWEN3ASR_ENCODER_FFN_DIM;      // 3584
    int CHUNK = QWEN3ASR_CONV_CHUNK_SIZE;   // 100

    // Compute total valid frames from chunking
    int num_chunks = (T_mel + CHUNK - 1) / CHUNK;
    int total_valid = 0;
    int chunk_valids[256];
    for (int c = 0; c < num_chunks; c++) {
        int start = c * CHUNK;
        int input_len = (T_mel - start < CHUNK) ? (T_mel - start) : CHUNK;
        int fl = (input_len - 1) / 2 + 1;
        int t2 = (fl - 1) / 2 + 1;
        int v = (t2 - 1) / 2 + 1;
        chunk_valids[c] = v;
        total_valid += v;
    }

    // Allocate full encoder output buffer
    float *d_enc_input;
    CUDA_CHECK(cudaMalloc(&d_enc_input, (size_t)total_valid * D * sizeof(float)));

    // Compute pad value: mean of first mel column
    float h_pad_val = 0.0f;
    {
        float *h_first_col = (float*)malloc(128 * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_first_col, d_mel, 128 * sizeof(float),
                               cudaMemcpyDeviceToHost));
        for (int i = 0; i < 128; i++) h_pad_val += h_first_col[i];
        h_pad_val /= 128.0f;
        free(h_first_col);
    }

    // Per-chunk conv temp buffers (reused)
    int feat_dim = 480 * 16;  // 7680
    float *d_chunk_buf = NULL;  // [128, CHUNK] padded input
    float *d_c1 = NULL, *d_c2 = NULL, *d_c3 = NULL;
    float *d_reshaped = NULL, *d_proj = NULL, *d_pe = NULL;

    int out_offset = 0;
    for (int c = 0; c < num_chunks; c++) {
        int start = c * CHUNK;
        int input_len = (T_mel - start < CHUNK) ? (T_mel - start) : CHUNK;
        int actual_len = (input_len < CHUNK) ? CHUNK : input_len;

        int W1 = (actual_len + 1) / 2;
        int W2 = (W1 + 1) / 2;
        int W3 = (W2 + 1) / 2;
        int valid = chunk_valids[c];

        // Lazily allocate temp buffers (max size across chunks)
        if (!d_chunk_buf)
            CUDA_CHECK(cudaMalloc(&d_chunk_buf, (size_t)128 * CHUNK * sizeof(float)));
        if (!d_c1)
            CUDA_CHECK(cudaMalloc(&d_c1, (size_t)480 * 64 * W1 * sizeof(float)));
        if (!d_c2)
            CUDA_CHECK(cudaMalloc(&d_c2, (size_t)480 * 32 * W2 * sizeof(float)));
        if (!d_c3)
            CUDA_CHECK(cudaMalloc(&d_c3, (size_t)480 * 16 * W3 * sizeof(float)));
        if (!d_reshaped)
            CUDA_CHECK(cudaMalloc(&d_reshaped, (size_t)W3 * feat_dim * sizeof(float)));
        if (!d_proj)
            CUDA_CHECK(cudaMalloc(&d_proj, (size_t)W3 * D * sizeof(float)));
        if (!d_pe)
            CUDA_CHECK(cudaMalloc(&d_pe, (size_t)valid * D * sizeof(float)));

        // Extract chunk from d_mel [128, T_mel] -> d_chunk_buf [128, actual_len]
        // For non-padded chunks: copy input_len columns
        // For padded chunk: copy input_len columns + fill padding
        for (int r = 0; r < 128; r++) {
            CUDA_CHECK(cudaMemcpyAsync(
                d_chunk_buf + (size_t)r * actual_len,
                d_mel + (size_t)r * T_mel + start,
                (size_t)input_len * sizeof(float),
                cudaMemcpyDeviceToDevice, stream));
        }
        if (input_len < CHUNK) {
            // Pad remaining columns with h_pad_val
            // Fill each row's padding region
            for (int r = 0; r < 128; r++)
                boat_cuda_fill_f32(d_chunk_buf + (size_t)r * CHUNK + input_len,
                                   h_pad_val, CHUNK - input_len);
        }

        // Conv1: [1, 1, 128, actual_len] -> [1, 480, 64, W1]
        boat_cuda_conv2d_forward_f32(d_chunk_buf, model->d_conv1_w, model->d_conv1_b,
            d_c1, 1, 1, 128, (size_t)actual_len, 480, 3, 3, 1, 2, 1);
        boat_cuda_gelu_f32(d_c1, d_c1, 480ULL * 64 * W1);

        // Conv2: [1, 480, 64, W1] -> [1, 480, 32, W2]
        boat_cuda_conv2d_forward_f32(d_c1, model->d_conv2_w, model->d_conv2_b,
            d_c2, 1, 480, 64, (size_t)W1, 480, 3, 3, 1, 2, 1);
        boat_cuda_gelu_f32(d_c2, d_c2, 480ULL * 32 * W2);

        // Conv3: [1, 480, 32, W2] -> [1, 480, 16, W3]
        boat_cuda_conv2d_forward_f32(d_c2, model->d_conv3_w, model->d_conv3_b,
            d_c3, 1, 480, 32, (size_t)W2, 480, 3, 3, 1, 2, 1);
        boat_cuda_gelu_f32(d_c3, d_c3, 480ULL * 16 * W3);

        // Permute: [480, 16, W3] -> [W3, 7680]
        {
            const int block = 256;
            unsigned int grid = (unsigned int)((480ULL * 16 * W3 + block - 1) / block);
            permute_conv_out_kernel<<<grid, block>>>(
                d_reshaped, d_c3, 480, 16, W3);
            CUDA_CHECK(cudaGetLastError());
        }

        // Linear: [W3, 7680] @ [7680, 896] -> [W3, 896]
        matmul_f32_cuda(handle, d_reshaped, model->d_conv_out_w, d_proj,
                        W3, feat_dim, D);

        // Chunk-local sinusoidal PE (positions 0..valid-1)
        sinusoidal_pe_f32_cuda(d_pe, valid, D, 10000.0f, stream);

        // Copy trimmed proj to output, then add PE
        CUDA_CHECK(cudaMemcpyAsync(
            d_enc_input + (size_t)out_offset * D,
            d_proj, (size_t)valid * D * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
        residual_add_f32_cuda(d_enc_input + (size_t)out_offset * D,
                               d_pe, valid * D, stream);

        out_offset += valid;
    }

    // Free conv temp buffers
    CUDA_CHECK(cudaFree(d_chunk_buf));
    CUDA_CHECK(cudaFree(d_c1));
    CUDA_CHECK(cudaFree(d_c2));
    CUDA_CHECK(cudaFree(d_c3));
    CUDA_CHECK(cudaFree(d_reshaped));
    CUDA_CHECK(cudaFree(d_proj));
    CUDA_CHECK(cudaFree(d_pe));

    int T = total_valid;

    // ---- Step 2: Encoder transformer layers ----
    // Allocate temp buffer for one layer's compute
    // Max needed: normed[T,D] + Q[T,D] + K[T,D] + V[T,D] + residual[T,D] + ffn_norm[T,D] + fc1[T,FF]
    // = T * (6*D + FF) floats
    size_t layer_tmp_size = (size_t)T * (6 * D + FF);
    float *d_layer_tmp;
    CUDA_CHECK(cudaMalloc(&d_layer_tmp, layer_tmp_size * sizeof(float)));

    // Temp buffer pointers
    float *d_normed   = d_layer_tmp;
    float *d_Q        = d_layer_tmp + (size_t)T * D;
    float *d_K        = d_Q + (size_t)T * D;
    float *d_V        = d_K + (size_t)T * D;
    float *d_residual = d_V + (size_t)T * D;
    float *d_ffn_norm = d_residual + (size_t)T * D;
    float *d_fc1      = d_ffn_norm + (size_t)T * D;

    float *d_hidden = d_enc_input;  // will be modified in-place

    for (int l = 0; l < QWEN3ASR_ENCODER_NUM_LAYERS; l++) {
        // Pre-attention LayerNorm
        layernorm_gpu(d_normed, d_hidden, T, D,
                      model->d_enc_attn_ln_w[l],
                      model->d_enc_attn_ln_b[l], 1e-5f);

        // QKV projections
        matmul_f32_cuda(handle, d_normed, model->d_enc_q_proj[l], d_Q, T, D, D);
        matmul_f32_cuda(handle, d_normed, model->d_enc_k_proj[l], d_K, T, D, D);
        matmul_f32_cuda(handle, d_normed, model->d_enc_v_proj[l], d_V, T, D, D);

        // Add biases
        add_bias_gpu(d_Q, model->d_enc_q_bias[l], T, D);
        add_bias_gpu(d_K, model->d_enc_k_bias[l], T, D);
        add_bias_gpu(d_V, model->d_enc_v_bias[l], T, D);

        // Fused MHA attention (no causal mask)
        float scale = 1.0f / sqrtf((float)HD);
        fused_enc_attn_f32_cuda(d_Q, d_K, d_V, d_Q, T, NH, HD, scale, stream);

        // O projection
        matmul_f32_cuda(handle, d_Q, model->d_enc_o_proj[l], d_normed, T, D, D);
        add_bias_gpu(d_normed, model->d_enc_o_bias[l], T, D);

        // Residual: hidden = hidden + attn_out
        residual_add_f32_cuda(d_hidden, d_normed, T * D, stream);

        // Pre-FFN LayerNorm
        layernorm_gpu(d_ffn_norm, d_hidden, T, D,
                      model->d_enc_final_ln_w[l],
                      model->d_enc_final_ln_b[l], 1e-5f);

        // FC1 -> GELU -> FC2
        matmul_f32_cuda(handle, d_ffn_norm, model->d_enc_fc1_w[l], d_fc1, T, D, FF);
        add_bias_gpu(d_fc1, model->d_enc_fc1_b[l], T, FF);
        boat_cuda_gelu_f32(d_fc1, d_fc1, (size_t)T * FF);

        matmul_f32_cuda(handle, d_fc1, model->d_enc_fc2_w[l], d_normed, T, FF, D);
        add_bias_gpu(d_normed, model->d_enc_fc2_b[l], T, D);

        // Residual
        residual_add_f32_cuda(d_hidden, d_normed, T * D, stream);
    }
    CUDA_CHECK(cudaFree(d_layer_tmp));

    // ---- Step 3: Post-projection ----
    float *d_post_norm;
    CUDA_CHECK(cudaMalloc(&d_post_norm, (size_t)T * D * sizeof(float)));
    layernorm_gpu(d_post_norm, d_hidden, T, D,
                  model->d_ln_post_w, model->d_ln_post_b, 1e-5f);
    CUDA_CHECK(cudaFree(d_hidden));

    // Proj1 -> GELU -> Proj2: [T,896] -> [T,896] -> [T,1024]
    float *d_p1;
    CUDA_CHECK(cudaMalloc(&d_p1, (size_t)T * D * sizeof(float)));
    matmul_f32_cuda(handle, d_post_norm, model->d_proj1_w, d_p1, T, D, D);
    add_bias_gpu(d_p1, model->d_proj1_b, T, D);
    boat_cuda_gelu_f32(d_p1, d_p1, (size_t)T * D);
    CUDA_CHECK(cudaFree(d_post_norm));

    int out_dim = QWEN3ASR_ENCODER_OUTPUT_DIM;  // 1024
    float *d_result;
    CUDA_CHECK(cudaMalloc(&d_result, (size_t)T * out_dim * sizeof(float)));
    matmul_f32_cuda(handle, d_p1, model->d_proj2_w, d_result, T, D, out_dim);
    add_bias_gpu(d_result, model->d_proj2_b, T, out_dim);
    CUDA_CHECK(cudaFree(d_p1));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return d_result;  // [T, 1024], caller must cudaFree
}

// ============================================================================
// Decoder layer (shared between prefill and decode)
// hidden: [T, 1024] in-place (output overwrites input)
// For prefill (T>1): fills KV cache from offset 0
// For decode (T==1): appends to KV cache at cache_offset
// ============================================================================
static int decoder_layer_gpu(qwen3asr_cuda_model_t* model,
    float* d_hidden, int T, int layer_idx, int cache_offset,
    float* d_tmp) {
    cublasHandle_t handle = boat_cuda_get_cublas_handle();
    cudaStream_t stream = 0;

    int H = QWEN3ASR_DECODER_HIDDEN_SIZE;       // 1024
    int NH = QWEN3ASR_DECODER_NUM_HEADS;         // 16
    int NKV = QWEN3ASR_DECODER_NUM_KV_HEADS;     // 8
    int HD = QWEN3ASR_DECODER_HEAD_DIM;          // 128
    int q_dim = NH * HD;                         // 2048
    int kv_dim = NKV * HD;                       // 1024
    int FF = QWEN3ASR_DECODER_INTERMEDIATE;      // 3072

    // Temp buffer layout (all offsets from d_tmp):
    // normed[T,H], Q[T,q_dim], K[T,kv_dim], V[T,kv_dim],
    // post_norm[T,H], gate[T,FF], up[T,FF]
    float *d_normed   = d_tmp;
    float *d_Q        = d_tmp + (size_t)T * H;
    float *d_K        = d_Q + (size_t)T * q_dim;
    float *d_V        = d_K + (size_t)T * kv_dim;
    float *d_post_norm = d_V + (size_t)T * kv_dim;
    float *d_gate     = d_post_norm + (size_t)T * H;
    float *d_up       = d_gate + (size_t)T * FF;

    // 1. Pre-attention RMSNorm
    rmsnorm_gpu(d_normed, d_hidden, T, H, model->d_dec_input_ln[layer_idx],
                QWEN3ASR_RMS_EPS);

    // 2. QKV projections
    matmul_f32_cuda(handle, d_normed, model->d_dec_q_proj[layer_idx], d_Q, T, H, q_dim);
    matmul_f32_cuda(handle, d_normed, model->d_dec_k_proj[layer_idx], d_K, T, H, kv_dim);
    matmul_f32_cuda(handle, d_normed, model->d_dec_v_proj[layer_idx], d_V, T, H, kv_dim);

    // 3. Per-head Q/K RMSNorm
    rmsnorm_gpu(d_Q, d_Q, T * NH, HD, model->d_dec_q_norm[layer_idx], QWEN3ASR_RMS_EPS);
    rmsnorm_gpu(d_K, d_K, T * NKV, HD, model->d_dec_k_norm[layer_idx], QWEN3ASR_RMS_EPS);

    // 4. RoPE
    rope_1d_f32_cuda(d_Q, d_K, T, NH, NKV, HD,
                     model->d_rope_cos, model->d_rope_sin,
                     cache_offset, stream);

    // 5. KV cache store
    for (int t = 0; t < T; t++) {
        CUDA_CHECK(cudaMemcpyAsync(
            model->d_k_cache[layer_idx] + (size_t)(cache_offset + t) * kv_dim,
            d_K + (size_t)t * kv_dim,
            (size_t)kv_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            model->d_v_cache[layer_idx] + (size_t)(cache_offset + t) * kv_dim,
            d_V + (size_t)t * kv_dim,
            (size_t)kv_dim * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }

    // 6. Attention
    float scale = 1.0f / sqrtf((float)HD);
    if (T > 1) {
        // Prefill: fused GQA attention with causal mask
        fused_gqa_prefill_attn_f32_cuda(d_Q, d_K, d_V, d_Q,
                                         T, NH, NKV, HD, scale, stream);
    } else {
        // Decode: fused GQA decode attention (reads from KV cache)
        int total_cache = cache_offset + 1;
        fused_gqa_decode_attn_f32_cuda(d_Q,
            model->d_k_cache[layer_idx], model->d_v_cache[layer_idx],
            d_Q, total_cache, NH, NKV, HD, stream);
    }

    // 7. O projection
    matmul_f32_cuda(handle, d_Q, model->d_dec_o_proj[layer_idx], d_normed, T, q_dim, H);

    // 8. Residual
    residual_add_f32_cuda(d_hidden, d_normed, T * H, stream);

    // 9. Post-attention RMSNorm
    rmsnorm_gpu(d_post_norm, d_hidden, T, H,
                model->d_dec_post_attn_ln[layer_idx], QWEN3ASR_RMS_EPS);

    // 10. SiLU-gated MLP
    matmul_f32_cuda(handle, d_post_norm, model->d_dec_gate_proj[layer_idx], d_gate, T, H, FF);
    matmul_f32_cuda(handle, d_post_norm, model->d_dec_up_proj[layer_idx], d_up, T, H, FF);
    silu_inplace_f32_cuda(d_gate, T * FF, stream);
    mul_f32_cuda(d_gate, d_gate, d_up, T * FF, stream);

    // 11. Down projection
    matmul_f32_cuda(handle, d_gate, model->d_dec_down_proj[layer_idx], d_normed, T, FF, H);

    // 12. Residual
    residual_add_f32_cuda(d_hidden, d_normed, T * H, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}

// ============================================================================
// Decoder prefill forward
// ============================================================================
float* qwen3asr_cuda_decoder_forward(qwen3asr_cuda_model_t* model,
                                      const float* d_merged, int T) {
    cublasHandle_t handle = boat_cuda_get_cublas_handle();
    cudaStream_t stream = 0;
    int H = QWEN3ASR_DECODER_HIDDEN_SIZE;
    int NH = QWEN3ASR_DECODER_NUM_HEADS;
    int NKV = QWEN3ASR_DECODER_NUM_KV_HEADS;
    int HD = QWEN3ASR_DECODER_HEAD_DIM;
    int q_dim = NH * HD;
    int kv_dim = NKV * HD;
    int FF = QWEN3ASR_DECODER_INTERMEDIATE;

    // Allocate decoder temp buffer
    // Max: normed[T,H] + Q[T,q_dim] + K[T,kv_dim] + V[T,kv_dim] + post_norm[T,H] + gate[T,FF] + up[T,FF]
    size_t tmp_size = (size_t)T * (H + q_dim + kv_dim + kv_dim + H + FF + FF);
    float *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_size * sizeof(float)));

    // Copy merged input to working buffer
    float *d_hidden;
    CUDA_CHECK(cudaMalloc(&d_hidden, (size_t)T * H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_hidden, d_merged, (size_t)T * H * sizeof(float),
                           cudaMemcpyDeviceToDevice));

    // Run all decoder layers
    for (int l = 0; l < QWEN3ASR_DECODER_NUM_LAYERS; l++) {
        if (decoder_layer_gpu(model, d_hidden, T, l, 0, d_tmp) != 0) {
            CUDA_CHECK(cudaFree(d_tmp));
            CUDA_CHECK(cudaFree(d_hidden));
            return NULL;
        }
    }
    CUDA_CHECK(cudaFree(d_tmp));

    // Update kv_len for all layers
    for (int l = 0; l < QWEN3ASR_DECODER_NUM_LAYERS; l++)
        model->kv_len[l] = T;

    // Final RMSNorm
    rmsnorm_gpu(d_hidden, d_hidden, T, H, model->d_norm_w, QWEN3ASR_RMS_EPS);

    // LM head: logits = hidden[last_pos] @ lm_head
    // lm_head stored as [vocab_size, H] (via LOAD_WEIGHT_2D: [H, V] stored)
    // Wait, let me check: lm_head is loaded as LOAD_WEIGHT_2D(&w->lm_head, tensors, "lm_head.weight", {V, H})
    // So it transposes from [V, H] to [H, V] = [K=H, N=V]
    // matmul: [1, H] @ [H, V] = [1, V]
    int last_pos = T - 1;
    int V = QWEN3ASR_VOCAB_SIZE;
    float *d_logits;
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)V * sizeof(float)));
    matmul_f32_cuda(handle, d_hidden + (size_t)last_pos * H,
                    model->d_lm_head, d_logits, 1, H, V);
    CUDA_CHECK(cudaFree(d_hidden));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return d_logits;
}

// ============================================================================
// Decoder decode step (single token)
// ============================================================================
float* qwen3asr_cuda_decoder_step(qwen3asr_cuda_model_t* model,
                                   const float* d_embed, int pos) {
    cublasHandle_t handle = boat_cuda_get_cublas_handle();
    cudaStream_t stream = 0;
    int H = QWEN3ASR_DECODER_HIDDEN_SIZE;
    int NH = QWEN3ASR_DECODER_NUM_HEADS;
    int NKV = QWEN3ASR_DECODER_NUM_KV_HEADS;
    int HD = QWEN3ASR_DECODER_HEAD_DIM;
    int q_dim = NH * HD;
    int kv_dim = NKV * HD;
    int FF = QWEN3ASR_DECODER_INTERMEDIATE;

    // Allocate temp buffer for single token
    size_t tmp_size = (size_t)(H + q_dim + kv_dim + kv_dim + H + FF + FF);
    float *d_tmp;
    CUDA_CHECK(cudaMalloc(&d_tmp, tmp_size * sizeof(float)));

    // Copy embedding to working buffer
    float *d_hidden;
    CUDA_CHECK(cudaMalloc(&d_hidden, (size_t)H * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_hidden, d_embed, (size_t)H * sizeof(float),
                           cudaMemcpyDeviceToDevice));

    // Run all decoder layers
    for (int l = 0; l < QWEN3ASR_DECODER_NUM_LAYERS; l++) {
        if (decoder_layer_gpu(model, d_hidden, 1, l, pos, d_tmp) != 0) {
            CUDA_CHECK(cudaFree(d_tmp));
            CUDA_CHECK(cudaFree(d_hidden));
            return NULL;
        }
    }
    CUDA_CHECK(cudaFree(d_tmp));

    // Update kv_len (set to pos+1 for all layers)
    for (int l = 0; l < QWEN3ASR_DECODER_NUM_LAYERS; l++)
        model->kv_len[l] = pos + 1;

    // Final RMSNorm
    rmsnorm_gpu(d_hidden, d_hidden, 1, H, model->d_norm_w, QWEN3ASR_RMS_EPS);

    // LM head
    int V = QWEN3ASR_VOCAB_SIZE;
    float *d_logits;
    CUDA_CHECK(cudaMalloc(&d_logits, (size_t)V * sizeof(float)));
    matmul_f32_cuda(handle, d_hidden, model->d_lm_head, d_logits, 1, H, V);
    CUDA_CHECK(cudaFree(d_hidden));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return d_logits;
}
