// sampling.h - Token sampling utilities for autoregressive generation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_SAMPLING_H
#define BOAT_SAMPLING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Sample a token from logits using top-k filtering and temperature scaling.
// logits:     [vocab_size] float32 array of raw logits
// vocab_size: number of tokens in vocabulary
// top_k:      restrict to top-k highest logits (0 = use full vocabulary)
// temperature: scaling factor (0.0 = greedy argmax, >0 = sample from scaled distribution)
// Returns:    sampled token ID
BOAT_API int boat_sample_token(const float* logits, int vocab_size, int top_k, float temperature);

#ifdef __cplusplus
}
#endif

#endif // BOAT_SAMPLING_H
