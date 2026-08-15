// weights.h - NanoChat weight loader from HuggingFace safetensors
#pragma once

#include "config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float* embed_tokens; // [vocab_size, hidden_size]
    float* lm_head;      // [vocab_size, hidden_size]
    float* q_proj[34];   // [hidden_size, hidden_size] each
    float* k_proj[34];
    float* v_proj[34];
    float* o_proj[34];
    float* fc1[34]; // [intermediate_size, hidden_size] each
    float* fc2[34]; // [hidden_size, intermediate_size] each
    int n_layers;
    int vocab_size;
    int hidden_size;
    int intermediate_size;
} nanochat_weights_t;

// Load weights from model.safetensors in the given directory.
// Returns NULL on failure. All weights are CPU FP32 arrays.
nanochat_weights_t* nanochat_weights_load(const char* model_dir);

// Free all weight arrays.
void nanochat_weights_free(nanochat_weights_t* w);

#ifdef __cplusplus
}
#endif
