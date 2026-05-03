// sampling.h - Text generation sampling strategies
#ifndef BOAT_OCR_SAMPLING_H
#define BOAT_OCR_SAMPLING_H

#include <stddef.h>

// Sample the next token using greedy (argmax) strategy
int sample_greedy(const float* logits, int n_vocab);

// Sample the next token using top-k strategy
int sample_topk(const float* logits, int n_vocab, int k, float temp);

// Sample the next token using top-p (nucleus) strategy
int sample_topp(const float* logits, int n_vocab, float p, float temp);

#endif // BOAT_OCR_SAMPLING_H
