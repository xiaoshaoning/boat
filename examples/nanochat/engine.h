// engine.h - NanoChat inference engine
#pragma once
#include "config.h"
#include "tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration — full struct in model.h (CUDA-compiled only) */
typedef struct nanochat_cuda_model_s nanochat_cuda_model_t;

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

// Streaming generate — calls on_text(new_text, user_data) for each fragment.
// Returns full generated text (caller must free). on_text can be NULL.
typedef void (*nanochat_stream_fn)(const char* text, void* user_data);
char* nanochat_generate_stream(nanochat_engine_t* eng,
                                const char* prompt,
                                int max_tokens,
                                float temperature,
                                int top_k,
                                nanochat_stream_fn on_text,
                                void* user_data);

// Interactive chat REPL — runs until user types exit/quit.
void nanochat_chat(nanochat_engine_t* eng,
                    int max_new_tokens,
                    float temperature,
                    int top_k);

#ifdef __cplusplus
}
#endif
