// encoder.h — Qwen3-ASR audio encoder API
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef QWEN3ASR_ENCODER_H
#define QWEN3ASR_ENCODER_H

#include "config.h"
#include "weights.h"
#include <boat/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qwen3asr_encoder_t qwen3asr_encoder_t;

// Create encoder from loaded weights
qwen3asr_encoder_t* qwen3asr_encoder_create(const qwen3asr_weights_t *w);

// Run encoder forward pass
// mel: [128, T] float32 log-mel spectrogram
// Returns: [T_out, 1024] float32 audio features
boat_tensor_t* qwen3asr_encoder_forward(qwen3asr_encoder_t *enc, const boat_tensor_t *mel);

// Free encoder
void qwen3asr_encoder_free(qwen3asr_encoder_t *enc);

#ifdef __cplusplus
}
#endif

#endif // QWEN3ASR_ENCODER_H
