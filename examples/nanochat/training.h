// training.h — NanoChat training API
#pragma once
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif
#include "model.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Single training step for a batch of 1 sequence
// model: initialized model with training buffers allocated
// d_tokens: [seq_len] int32 token IDs on GPU (input sequence)
// seq_len: number of tokens in the sequence (must be >= 2)
// lr: learning rate for Adam optimizer
// h_loss: output scalar loss value (on host)
// ---------------------------------------------------------------------------
void nanochat_cuda_train_step(nanochat_cuda_model_t* model,
                               const int* d_tokens, int seq_len,
                               float lr, float* h_loss);

// ---------------------------------------------------------------------------
// Cosine LR schedule helpers (caller-managed, not used internally)
// ---------------------------------------------------------------------------
static inline float nanochat_cosine_lr(int step, int warmup, int cooldown,
                                        float peak_lr, float min_lr) {
    if (step < warmup)
        return min_lr + (peak_lr - min_lr) * (float)step / (float)warmup;
    if (step >= cooldown)
        return min_lr;
    float progress = (float)(step - warmup) / (float)(cooldown - warmup);
    float cosine = 0.5f * (1.0f + cosf((float)M_PI * progress));
    return min_lr + (peak_lr - min_lr) * cosine;
}

#ifdef __cplusplus
}
#endif
