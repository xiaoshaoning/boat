// engine.c - NanoChat inference engine (prefill + decode loop)
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

// ---------------------------------------------------------------------------
// Generate text from a prompt
// ---------------------------------------------------------------------------
char* nanochat_generate(nanochat_engine_t* eng,
                         const char* prompt,
                         int max_tokens,
                         float temperature,
                         int top_k) {
    if (!eng || !eng->model || !eng->tokenizer) return NULL;

    // 1. Tokenize prompt (no BOS — model was trained without it)
    size_t text_len = strlen(prompt);
    int prompt_len;
    int* tokens = nanochat_tokenizer_encode(eng->tokenizer, prompt, text_len, &prompt_len);
    if (!tokens || prompt_len == 0) { free(tokens); return NULL; }

    tokens = (int*)realloc(tokens, (size_t)(prompt_len + max_tokens) * sizeof(int));
    int total_tokens = prompt_len;

    // 2. Embed all prompt tokens on GPU
    int* d_tokens; float* d_embed;
    CUDA_CHECK(cudaMalloc(&d_tokens, (size_t)prompt_len * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_embed, (size_t)prompt_len * eng->model->hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_tokens, tokens, (size_t)prompt_len * sizeof(int), cudaMemcpyHostToDevice));
    embed_gather_cuda(eng->model->d_embed_tokens, d_tokens, d_embed, prompt_len, eng->model->hidden_size, 0);
    CUDA_CHECK(cudaFree(d_tokens));

    // 3. Run prefill (populates KV cache, returns logits for last position)
    float* d_logits = nanochat_cuda_model_forward(eng->model, d_embed, prompt_len);
    CUDA_CHECK(cudaFree(d_embed));

    // 4. Sample first generated token
    float* h_logits = (float*)malloc((size_t)eng->model->vocab_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_logits, d_logits, (size_t)eng->model->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_logits));
    int next = nanochat_sample_token(h_logits, eng->model->vocab_size, top_k, temperature);
    tokens[total_tokens++] = next;

    // 5. Decode loop
    int hidden_size = eng->model->hidden_size;
    float* d_single_embed = NULL;
    while (total_tokens < prompt_len + max_tokens) {
        if (!d_single_embed) CUDA_CHECK(cudaMalloc(&d_single_embed, (size_t)hidden_size * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_single_embed, eng->model->d_embed_tokens + (size_t)next * hidden_size,
                               (size_t)hidden_size * sizeof(float), cudaMemcpyDeviceToDevice));
        d_logits = nanochat_cuda_model_decode(eng->model, d_single_embed, total_tokens - 1);
        CUDA_CHECK(cudaMemcpy(h_logits, d_logits, (size_t)eng->model->vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_logits));
        next = nanochat_sample_token(h_logits, eng->model->vocab_size, top_k, temperature);
        tokens[total_tokens++] = next;

        // Stop on EOS token (token 1 = eos_token_id)
        if (next == 1) break;
    }
    if (d_single_embed) CUDA_CHECK(cudaFree(d_single_embed));
    free(h_logits);

    // 6. Decode tokens to text
    char* result = nanochat_tokenizer_decode(eng->tokenizer, tokens, total_tokens);
    free(tokens);
    return result;
}
