// engine.h - NanoChat inference engine
#pragma once
#include "config.h"
#include "tokenizer.h"
#include "model.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    nanochat_cuda_model_t* model;
    nanochat_tokenizer_t* tokenizer;
    int max_seq_len;
} nanochat_engine_t;

// Create inference engine: load model and tokenizer
nanochat_engine_t* nanochat_engine_create(const char* model_dir);

// Free engine
void nanochat_engine_free(nanochat_engine_t* eng);

// Generate text from a prompt. Returns malloc'd string (caller must free).
char* nanochat_generate(nanochat_engine_t* eng,
                         const char* prompt,
                         int max_tokens,
                         float temperature,
                         int top_k);

#ifdef __cplusplus
}
#endif
