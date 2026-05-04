// engine.cu - NanoChat inference engine (prefill + decode loop + chat)
#include "engine.h"
#include "nanochat.h"
#include "tokenizer.h"
#include "weights.h"
#include "model.h"
#include "nanochat_kernels.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                           \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                    \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                        \
    }                                                                   \
} while(0)

// ---------------------------------------------------------------------------
// Create inference engine
// ---------------------------------------------------------------------------
nanochat_engine_t* nanochat_engine_create(const char* model_dir) {
    nanochat_engine_t* eng = (nanochat_engine_t*)calloc(1, sizeof(nanochat_engine_t));
    if (!eng) return NULL;
    eng->max_seq_len = NANOCHAT_MAX_SEQ_LEN;

    // Build paths
    size_t dir_len = strlen(model_dir);
    char* tok_path = (char*)malloc(dir_len + 32);
    char* weights_path = (char*)malloc(dir_len + 32);
    if (!tok_path || !weights_path) {
        free(tok_path); free(weights_path); free(eng);
        return NULL;
    }
    snprintf(tok_path, dir_len + 32, "%s/tokenizer.json", model_dir);
    snprintf(weights_path, dir_len + 32, "%s/model.safetensors", model_dir);

    // 1. Load tokenizer
    eng->tokenizer = (nanochat_tokenizer_t*)malloc(sizeof(nanochat_tokenizer_t));
    if (!eng->tokenizer || !nanochat_tokenizer_init(eng->tokenizer, tok_path)) {
        fprintf(stderr, "[NanoChat] Failed to load tokenizer from %s\n", tok_path);
        free(tok_path); free(weights_path);
        nanochat_tokenizer_free(eng->tokenizer); free(eng->tokenizer); free(eng);
        return NULL;
    }
    fprintf(stderr, "[NanoChat] Tokenizer loaded from %s\n", tok_path);

    // 2. Load weights (CPU-side)
    nanochat_weights_t* weights = nanochat_weights_load(model_dir);
    if (!weights) {
        fprintf(stderr, "[NanoChat] Failed to load weights from %s\n", weights_path);
        free(tok_path); free(weights_path);
        nanochat_tokenizer_free(eng->tokenizer); free(eng->tokenizer); free(eng);
        return NULL;
    }
    fprintf(stderr, "[NanoChat] Weights loaded from %s\n", weights_path);
    free(tok_path); free(weights_path);

    // 3. Upload to GPU
    eng->model = (nanochat_cuda_model_t*)calloc(1, sizeof(nanochat_cuda_model_t));
    if (!eng->model || !nanochat_cuda_model_init(eng->model, weights)) {
        fprintf(stderr, "[NanoChat] Failed to upload weights to GPU\n");
        nanochat_weights_free(weights);
        nanochat_cuda_model_free(eng->model); free(eng->model);
        nanochat_tokenizer_free(eng->tokenizer); free(eng->tokenizer); free(eng);
        return NULL;
    }

    // Free CPU weights (no longer needed)
    nanochat_weights_free(weights);

    fprintf(stderr, "[NanoChat] Engine ready (max_seq_len=%d)\n", eng->max_seq_len);
    return eng;
}

// ---------------------------------------------------------------------------
// Free engine
// ---------------------------------------------------------------------------
void nanochat_engine_free(nanochat_engine_t* eng) {
    if (!eng) return;
    if (eng->model) {
        nanochat_cuda_model_free(eng->model);
        free(eng->model);
    }
    if (eng->tokenizer) {
        nanochat_tokenizer_free(eng->tokenizer);
        free(eng->tokenizer);
    }
    memset(eng, 0, sizeof(*eng));
    free(eng);
}

// ============================================================================
// Core generation from pre-tokenized input
// Takes ownership of tokens (will realloc/free it internally).
// ============================================================================
static char* generate_from_tokens(nanochat_engine_t* eng,
                                   int* tokens, int prompt_len,
                                   int max_new_tokens,
                                   float temperature, int top_k,
                                   nanochat_stream_fn on_text, void* user_data) {
    nanochat_tokenizer_t* tok = eng->tokenizer;
    int max_total = prompt_len + max_new_tokens;
    tokens = (int*)realloc(tokens, (size_t)max_total * sizeof(int));
    int total_tokens = prompt_len;

    // Reset KV cache for fresh prefill
    nanochat_cuda_model_reset_kv_cache(eng->model);

    // Embed prompt tokens on GPU (BF16)
    int* d_tokens;
    __nv_bfloat16* d_embed;
    CUDA_CHECK(cudaMalloc(&d_tokens, (size_t)prompt_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_embed, (size_t)prompt_len * eng->model->hidden_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_tokens, tokens, (size_t)prompt_len * sizeof(int), cudaMemcpyHostToDevice));
    embed_gather_bf16_cuda(eng->model->d_embed_tokens, d_tokens, d_embed, prompt_len, eng->model->hidden_size, 0);
    CUDA_CHECK(cudaFree(d_tokens));

    // Run prefill
    float* d_logits = nanochat_cuda_model_forward(eng->model, d_embed, prompt_len);
    CUDA_CHECK(cudaFree(d_embed));

    // Prefill logits → host
    float* h_logits = (float*)malloc((size_t)eng->model->vocab_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_logits, d_logits, (size_t)eng->model->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_logits));

    // Result buffer
    size_t result_cap = 1024;
    char* result = (char*)malloc(result_cap);
    size_t result_len = 0;
    result[0] = '\0';

    int hidden_size = eng->model->hidden_size;
    __nv_bfloat16* d_single_embed = NULL;

    // Decode loop
    for (int gen_i = 0; gen_i < max_new_tokens; gen_i++) {
        int next;
        if (gen_i == 0) {
            next = nanochat_sample_token(h_logits, eng->model->vocab_size, top_k, temperature);
        } else {
            if (!d_single_embed)
                CUDA_CHECK(cudaMalloc(&d_single_embed, (size_t)hidden_size * sizeof(__nv_bfloat16)));
            CUDA_CHECK(cudaMemcpy(d_single_embed,
                eng->model->d_embed_tokens + (size_t)next * hidden_size,
                (size_t)hidden_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
            d_logits = nanochat_cuda_model_decode(eng->model, d_single_embed, total_tokens - 1);
            CUDA_CHECK(cudaMemcpy(h_logits, d_logits,
                (size_t)eng->model->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaFree(d_logits));
            next = nanochat_sample_token(h_logits, eng->model->vocab_size, top_k, temperature);
        }

        // Check stop tokens
        if (next == NANOCHAT_TOKEN_EOS || next == NANOCHAT_TOKEN_ASSISTANT_END) break;

        tokens[total_tokens++] = next;

        // Emit non-special tokens via callback
        if (!nanochat_token_is_special(next)) {
            char* tok_text = nanochat_tokenizer_decode(tok, &next, 1);
            if (tok_text) {
                size_t tlen = strlen(tok_text);
                if (result_len + tlen + 1 > result_cap) {
                    result_cap = result_cap * 2 + tlen;
                    result = (char*)realloc(result, result_cap);
                }
                memcpy(result + result_len, tok_text, tlen);
                result_len += tlen;
                result[result_len] = '\0';
                if (on_text) on_text(tok_text, user_data);
                free(tok_text);
            }
        }
    }

    if (d_single_embed) CUDA_CHECK(cudaFree(d_single_embed));
    free(h_logits);
    free(tokens);
    return result;
}

// ============================================================================
// Generate from string prompt (public API)
// ============================================================================
char* nanochat_generate_stream(nanochat_engine_t* eng,
                                const char* prompt,
                                int max_new_tokens,
                                float temperature,
                                int top_k,
                                nanochat_stream_fn on_text,
                                void* user_data) {
    if (!eng || !eng->model || !eng->tokenizer) return NULL;

    size_t text_len = strlen(prompt);
    int prompt_len;
    int* tokens = nanochat_tokenizer_encode(eng->tokenizer, prompt, text_len, &prompt_len);
    if (!tokens || prompt_len == 0) { free(tokens); return NULL; }

    return generate_from_tokens(eng, tokens, prompt_len,
                                max_new_tokens, temperature, top_k,
                                on_text, user_data);
}

// Non-streaming variant
char* nanochat_generate(nanochat_engine_t* eng,
                         const char* prompt,
                         int max_tokens,
                         float temperature,
                         int top_k) {
    return nanochat_generate_stream(eng, prompt, max_tokens, temperature, top_k, NULL, NULL);
}

// ============================================================================
// Interactive chat — multi-turn conversation with chat template
// ============================================================================

#define MAX_HISTORY_TURNS 64

typedef struct {
    int* user_ids[MAX_HISTORY_TURNS];
    int   user_len[MAX_HISTORY_TURNS];
    int* assistant_ids[MAX_HISTORY_TURNS];
    int   assistant_len[MAX_HISTORY_TURNS];
    int num_turns;
} chat_history_t;

static void chat_history_free(chat_history_t* hist) {
    for (int i = 0; i < hist->num_turns; i++) {
        free(hist->user_ids[i]);
        free(hist->assistant_ids[i]);
    }
    hist->num_turns = 0;
}

static void chat_history_append(chat_history_t* hist,
                                 const int* user_ids, int user_len,
                                 const int* assistant_ids, int assistant_len) {
    if (hist->num_turns >= MAX_HISTORY_TURNS) {
        free(hist->user_ids[0]);
        free(hist->assistant_ids[0]);
        memmove(&hist->user_ids[0], &hist->user_ids[1], (hist->num_turns - 1) * sizeof(int*));
        memmove(&hist->user_len[0], &hist->user_len[1], (hist->num_turns - 1) * sizeof(int));
        memmove(&hist->assistant_ids[0], &hist->assistant_ids[1], (hist->num_turns - 1) * sizeof(int*));
        memmove(&hist->assistant_len[0], &hist->assistant_len[1], (hist->num_turns - 1) * sizeof(int));
        hist->num_turns--;
    }
    int i = hist->num_turns;
    hist->user_ids[i] = (int*)malloc((size_t)user_len * sizeof(int));
    memcpy(hist->user_ids[i], user_ids, (size_t)user_len * sizeof(int));
    hist->user_len[i] = user_len;
    if (assistant_ids && assistant_len > 0) {
        hist->assistant_ids[i] = (int*)malloc((size_t)assistant_len * sizeof(int));
        memcpy(hist->assistant_ids[i], assistant_ids, (size_t)assistant_len * sizeof(int));
        hist->assistant_len[i] = assistant_len;
    } else {
        hist->assistant_ids[i] = NULL;
        hist->assistant_len[i] = 0;
    }
    hist->num_turns++;
}

// Build token array for the full chat prompt.
// Format: <|user_start|>u1<|user_end|><|assistant_start|>a1<|assistant_end|>
//         <|user_start|>u2<|user_end|><|assistant_start|>
// Allocates and returns token array, sets *out_len. Caller must free.
static int* build_chat_tokens(chat_history_t* hist,
                               const int* user_input_ids, int user_input_len,
                               int* out_len) {
    // First pass: count total tokens needed
    int total = 0;
    for (int i = 0; i < hist->num_turns; i++) {
        total += 1 + hist->user_len[i] + 1 + 1 + hist->assistant_len[i] + 1;
        // USER_START + user + USER_END + ASSISTANT_START + assistant + ASSISTANT_END
    }
    total += 1 + user_input_len + 1 + 1;
    // USER_START + input + USER_END + ASSISTANT_START

    int* tokens = (int*)malloc((size_t)total * sizeof(int));
    int pos = 0;

    for (int i = 0; i < hist->num_turns; i++) {
        tokens[pos++] = NANOCHAT_TOKEN_USER_START;
        memcpy(tokens + pos, hist->user_ids[i], (size_t)hist->user_len[i] * sizeof(int));
        pos += hist->user_len[i];
        tokens[pos++] = NANOCHAT_TOKEN_USER_END;
        tokens[pos++] = NANOCHAT_TOKEN_ASSISTANT_START;
        if (hist->assistant_len[i] > 0) {
            memcpy(tokens + pos, hist->assistant_ids[i], (size_t)hist->assistant_len[i] * sizeof(int));
            pos += hist->assistant_len[i];
        }
        tokens[pos++] = NANOCHAT_TOKEN_ASSISTANT_END;
    }

    tokens[pos++] = NANOCHAT_TOKEN_USER_START;
    memcpy(tokens + pos, user_input_ids, (size_t)user_input_len * sizeof(int));
    pos += user_input_len;
    tokens[pos++] = NANOCHAT_TOKEN_USER_END;
    tokens[pos++] = NANOCHAT_TOKEN_ASSISTANT_START;

    *out_len = pos;
    return tokens;
}

// Streaming callback that filters special token markers from output
// and prints to stdout with color prefix
static void chat_stream_callback(const char* text, void* user_data) {
    (void)user_data;
    // Filter out any <|...|> special token markers that might leak through
    // (These appear when the model generates special token text as byte tokens)
    const char* p = text;
    while (*p) {
        if (p[0] == '<' && p[1] == '|') {
            // Skip until we find |> or end of string
            p += 2;
            while (*p && !(p[0] == '|' && p[1] == '>')) p++;
            if (*p) p += 2; // skip the |>
        } else {
            putchar(*p);
            p++;
        }
    }
    fflush(stdout);
}

void nanochat_chat(nanochat_engine_t* eng,
                    int max_new_tokens,
                    float temperature,
                    int top_k) {
    if (!eng || !eng->model || !eng->tokenizer) return;

    nanochat_tokenizer_t* tok = eng->tokenizer;
    chat_history_t hist = {0};
    char input_buf[4096];

    fprintf(stderr, "\n[NanoChat] Interactive chat \xe2\x80\x94 type 'exit' or 'quit' to end\n");
    fprintf(stderr, "[NanoChat] Commands: /reset to clear history\n\n");

    while (1) {
        fprintf(stdout, "\033[32m>>> \033[0m");
        fflush(stdout);

        if (!fgets(input_buf, sizeof(input_buf), stdin)) break;
        size_t len = strlen(input_buf);
        while (len > 0 && (input_buf[len - 1] == '\n' || input_buf[len - 1] == '\r'))
            input_buf[--len] = '\0';
        if (len == 0) continue;

        if (strcmp(input_buf, "exit") == 0 || strcmp(input_buf, "quit") == 0) {
            fprintf(stdout, "\033[33mBye!\033[0m\n");
            break;
        }
        if (strcmp(input_buf, "/reset") == 0) {
            chat_history_free(&hist);
            nanochat_cuda_model_reset_kv_cache(eng->model);
            fprintf(stderr, "[NanoChat] History cleared.\n\n");
            continue;
        }

        // Tokenize user input
        int user_len;
        int* user_ids = nanochat_tokenizer_encode(tok, input_buf, len, &user_len);
        if (!user_ids || user_len == 0) { free(user_ids); continue; }

        // Build full prompt token array with proper special token IDs
        int prompt_len;
        int* prompt_tokens = build_chat_tokens(&hist, user_ids, user_len, &prompt_len);

        // Truncate oldest turns if prompt exceeds max context
        int reserve = max_new_tokens < 128 ? 128 : max_new_tokens;
        int max_prompt = eng->max_seq_len - reserve;
        while (prompt_len > max_prompt && hist.num_turns > 0) {
            // Drop oldest history turn
            free(hist.user_ids[0]);
            free(hist.assistant_ids[0]);
            memmove(&hist.user_ids[0], &hist.user_ids[1], (hist.num_turns - 1) * sizeof(int*));
            memmove(&hist.user_len[0], &hist.user_len[1], (hist.num_turns - 1) * sizeof(int));
            memmove(&hist.assistant_ids[0], &hist.assistant_ids[1], (hist.num_turns - 1) * sizeof(int*));
            memmove(&hist.assistant_len[0], &hist.assistant_len[1], (hist.num_turns - 1) * sizeof(int));
            hist.num_turns--;
            free(prompt_tokens);
            prompt_tokens = build_chat_tokens(&hist, user_ids, user_len, &prompt_len);
        }

        fprintf(stdout, "\033[36mAI: \033[0m");
        fflush(stdout);

        char* reply = generate_from_tokens(eng, prompt_tokens, prompt_len,
                                            max_new_tokens, temperature, top_k,
                                            chat_stream_callback, NULL);

        if (reply) {
            fprintf(stdout, "\n");
            fflush(stdout);

            // Tokenize assistant reply for history storage
            size_t reply_sz = strlen(reply);
            int reply_tok_len;
            int* reply_ids = nanochat_tokenizer_encode(tok, reply, reply_sz, &reply_tok_len);
            if (reply_ids) {
                chat_history_append(&hist, user_ids, user_len, reply_ids, reply_tok_len);
                free(reply_ids);
            } else {
                chat_history_append(&hist, user_ids, user_len, NULL, 0);
            }
            free(reply);
        } else {
            fprintf(stdout, "\033[31m[Error: generation failed]\033[0m\n");
            fflush(stdout);
        }

        free(user_ids);
    }

    chat_history_free(&hist);
    nanochat_cuda_model_reset_kv_cache(eng->model);
}
