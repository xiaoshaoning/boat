// nanochat.h - Public API for NanoChat CUDA inference
#pragma once
#include "config.h"
#include "tokenizer.h"
#include "engine.h"

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------
int nanochat_sample_token(const float* logits, int vocab_size,
                           int top_k, float temp);
