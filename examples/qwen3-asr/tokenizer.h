// tokenizer.h — Qwen2 BPE token decoder
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef QWEN3ASR_TOKENIZER_H
#define QWEN3ASR_TOKENIZER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Decode token IDs to text using GPT-2 bytes_to_unicode BPE decoding.
// model_dir: directory containing vocab.json
// tokens: array of token IDs
// n_tokens: number of tokens
// Returns malloc'd UTF-8 string — caller frees with free()
char* qwen3asr_decode_tokens(const char *model_dir, const int *tokens, int n_tokens);

#ifdef __cplusplus
}
#endif

#endif // QWEN3ASR_TOKENIZER_H
