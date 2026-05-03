// sampling.h - Top-k sampling for NanoChat
#pragma once

// Sample a token from logits using top-k + temperature.
// If temp <= 0, returns argmax (greedy).
int nanochat_sample_token(const float* logits, int vocab_size,
                           int top_k, float temp);
