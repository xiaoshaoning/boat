// decoder.h — Qwen3-ASR text decoder API
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef QWEN3ASR_DECODER_H
#define QWEN3ASR_DECODER_H

#include "config.h"
#include "weights.h"
#include <boat/tensor.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qwen3asr_decoder_t qwen3asr_decoder_t;

// Standard ASR prompt tokens (hardcoded — matches decoder.py build_prompt_tokens())
// Format: <|im_start|>transcribe\n<|audio_placeholder|>\n<|im_end|>\n<|im_start|>output\n
// 'trans'=1458, 'cribe'=3114, 'output'=3006
#define QWEN3ASR_PROMPT_TOKENS {151644, 1458, 3114, 198, 151676, 198, 151645, 198, 151644, 3006, 198}
#define QWEN3ASR_PROMPT_LEN 11
#define QWEN3ASR_PROMPT_PLACEHOLDER_POS 4  // index of 151676 in prompt

// Create decoder from loaded weights
qwen3asr_decoder_t* qwen3asr_decoder_create(const qwen3asr_weights_t *w);

// Free decoder (including KV cache)
void qwen3asr_decoder_free(qwen3asr_decoder_t *dec);

// Reset KV cache (for new sequence)
void qwen3asr_decoder_reset_kv(qwen3asr_decoder_t *dec);

// Prefix forward pass: process merged embeddings, fill KV cache
// merged: [T, 1024] float32 — combined text embeddings + audio features
// Returns logits for the last position [1, vocab_size] float32
boat_tensor_t* qwen3asr_decoder_forward(qwen3asr_decoder_t *dec, const boat_tensor_t *merged);

// Single-token decode step (autoregressive)
// token_embed: [1, 1024] embedding of current token
// pos: absolute position in the sequence
// Returns logits [1, vocab_size] float32
boat_tensor_t* qwen3asr_decoder_step(qwen3asr_decoder_t *dec, const boat_tensor_t *token_embed, int pos);

// Build prompt embeddings: embed text tokens and merge audio features
// embed_weight: [vocab_size, 1024] embedding table
// audio_features: [T_audio, 1024] encoder output
// Returns: merged embeddings [T_total, 1024] — caller frees
boat_tensor_t* qwen3asr_build_prompt(const boat_tensor_t *embed_weight,
                                       const boat_tensor_t *audio_features);

#ifdef __cplusplus
}
#endif

#endif // QWEN3ASR_DECODER_H
