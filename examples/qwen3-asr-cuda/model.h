// model.h — Qwen3-ASR CUDA GPU model struct and API
// All weights and compute in FP32
#pragma once
#include "../qwen3-asr/config.h"
#include "../qwen3-asr/weights.h"
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qwen3asr_cuda_model_s {
    // ---- Encoder weights (GPU FP32) ----
    float *d_conv1_w, *d_conv1_b;   // [480,1,3,3] / [480]
    float *d_conv2_w, *d_conv2_b;   // [480,480,3,3] / [480]
    float *d_conv3_w, *d_conv3_b;   // [480,480,3,3] / [480]
    float *d_conv_out_w;            // [896, 7680]

    // Encoder per-layer weights
    float *d_enc_q_proj[QWEN3ASR_ENCODER_NUM_LAYERS];
    float *d_enc_k_proj[QWEN3ASR_ENCODER_NUM_LAYERS];
    float *d_enc_v_proj[QWEN3ASR_ENCODER_NUM_LAYERS];
    float *d_enc_o_proj[QWEN3ASR_ENCODER_NUM_LAYERS];
    float *d_enc_q_bias[QWEN3ASR_ENCODER_NUM_LAYERS];
    float *d_enc_k_bias[QWEN3ASR_ENCODER_NUM_LAYERS];
    float *d_enc_v_bias[QWEN3ASR_ENCODER_NUM_LAYERS];
    float *d_enc_o_bias[QWEN3ASR_ENCODER_NUM_LAYERS];
    float *d_enc_attn_ln_w[QWEN3ASR_ENCODER_NUM_LAYERS];
    float *d_enc_attn_ln_b[QWEN3ASR_ENCODER_NUM_LAYERS];
    float *d_enc_fc1_w[QWEN3ASR_ENCODER_NUM_LAYERS];
    float *d_enc_fc1_b[QWEN3ASR_ENCODER_NUM_LAYERS];
    float *d_enc_fc2_w[QWEN3ASR_ENCODER_NUM_LAYERS];
    float *d_enc_fc2_b[QWEN3ASR_ENCODER_NUM_LAYERS];
    float *d_enc_final_ln_w[QWEN3ASR_ENCODER_NUM_LAYERS];
    float *d_enc_final_ln_b[QWEN3ASR_ENCODER_NUM_LAYERS];

    // Encoder post-projection
    float *d_ln_post_w, *d_ln_post_b;   // [896]
    float *d_proj1_w, *d_proj1_b;       // [896,896] / [896]
    float *d_proj2_w, *d_proj2_b;       // [1024,896] / [1024]

    // ---- Decoder weights (GPU FP32) ----
    float *d_embed_tokens;  // [151936, 1024]
    float *d_norm_w;        // [1024]
    float *d_lm_head;       // [151936, 1024]

    // Decoder per-layer weights
    float *d_dec_q_proj[QWEN3ASR_DECODER_NUM_LAYERS];       // [1024, 2048]
    float *d_dec_k_proj[QWEN3ASR_DECODER_NUM_LAYERS];       // [1024, 1024]
    float *d_dec_v_proj[QWEN3ASR_DECODER_NUM_LAYERS];       // [1024, 1024]
    float *d_dec_o_proj[QWEN3ASR_DECODER_NUM_LAYERS];       // [1024, 2048]
    float *d_dec_q_norm[QWEN3ASR_DECODER_NUM_LAYERS];       // [128]
    float *d_dec_k_norm[QWEN3ASR_DECODER_NUM_LAYERS];       // [128]
    float *d_dec_gate_proj[QWEN3ASR_DECODER_NUM_LAYERS];    // [1024, 3072]
    float *d_dec_up_proj[QWEN3ASR_DECODER_NUM_LAYERS];      // [1024, 3072]
    float *d_dec_down_proj[QWEN3ASR_DECODER_NUM_LAYERS];    // [3072, 1024]
    float *d_dec_input_ln[QWEN3ASR_DECODER_NUM_LAYERS];     // [1024]
    float *d_dec_post_attn_ln[QWEN3ASR_DECODER_NUM_LAYERS]; // [1024]

    // KV cache (GPU FP32)
    float *d_k_cache[QWEN3ASR_DECODER_NUM_LAYERS];  // [max_seq_len, NKV*HD]
    float *d_v_cache[QWEN3ASR_DECODER_NUM_LAYERS];
    int kv_len[QWEN3ASR_DECODER_NUM_LAYERS];

    // RoPE cos/sin tables (GPU, precomputed)
    float *d_rope_cos;  // [max_seq_len, head_dim/2]
    float *d_rope_sin;

    // Pre-allocated temp buffers (GPU FP32)
    float *d_enc_tmp;    // encoder temp: sized for max(T*896*6 + T*3584)
    float *d_dec_tmp;    // decoder temp: sized for max(T*1024*6 + T*3072)
    float *d_single;     // single-token decode: [1, 1024] embedding buffer
    int   enc_tmp_bytes;
    int   dec_tmp_bytes;

    // Model config copied from weights
    int enc_num_layers;
    int dec_num_layers;
} qwen3asr_cuda_model_t;

// Initialize model: upload all weights from CPU to GPU
// Returns 1 on success, 0 on failure
int qwen3asr_cuda_model_init(qwen3asr_cuda_model_t* model,
                              const qwen3asr_weights_t* w);

// Free all GPU memory
void qwen3asr_cuda_model_free(qwen3asr_cuda_model_t* model);

// Reset KV cache
void qwen3asr_cuda_model_reset_kv(qwen3asr_cuda_model_t* model);

// ---- Encoder ----
// Encode mel spectrogram on GPU.
// d_mel: [128, T] on GPU
// Returns d_audio: [T_out, 1024] on GPU (caller must cudaFree)
float* qwen3asr_cuda_encoder_forward(qwen3asr_cuda_model_t* model,
                                      const float* d_mel, int T_mel);

// ---- Decoder prefill ----
// Process merged embeddings, fill KV cache, return last logits.
// d_merged: [T, 1024] on GPU
// Returns logits: [1, vocab_size] on GPU (caller must cudaFree)
float* qwen3asr_cuda_decoder_forward(qwen3asr_cuda_model_t* model,
                                      const float* d_merged, int T);

// ---- Decoder step ----
// Single-token decode with KV cache append.
// d_embed: [1, 1024] on GPU, pos: absolute position
// Returns logits: [1, vocab_size] on GPU (caller must cudaFree)
float* qwen3asr_cuda_decoder_step(qwen3asr_cuda_model_t* model,
                                   const float* d_embed, int pos);

#ifdef __cplusplus
}
#endif
