// sampling.h - Token sampling for autoregressive generation
#pragma once

// Sample a token from logits using top-k + temperature.
// Returns the sampled token ID.
int minimind_sample_token(const float* logits, int vocab_size, int top_k, float temperature);
