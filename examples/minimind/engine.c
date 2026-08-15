// engine.c - MiniMind generation engine
#include "engine.h"
#include "tokenizer.h"
#include "sampling.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Chat template for MiniMind (ChatML style)
// <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
char* minimind_format_chat_prompt(const char* user_input) {
    size_t len = strlen(user_input);
    // MiniMind chat template (full, with think block)
    char* buf = (char*)malloc(len + 256);
    if (!buf) return NULL;
    snprintf(buf, len + 256,
             "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
             user_input);
    return buf;
}

static minimind_tokenizer_t* g_tokenizer = NULL;

char* minimind_generate(minimind_model_t* m, const char* prompt, int max_tokens, float temperature,
                        int top_k) {
    if (!g_tokenizer) {
        fprintf(stderr, "Tokenizer not loaded. Call minimind_engine_set_tokenizer() first.\n");
        return NULL;
    }

    int VS = m->config.vocab_size;

    // Tokenize prompt
    int* prompt_tokens = (int*)malloc(4096 * sizeof(int));
    int n_prompt = minimind_tokenizer_encode(g_tokenizer, prompt, prompt_tokens, 4096);
    if (n_prompt <= 0) {
        free(prompt_tokens);
        return NULL;
    }

    // Allocate output buffer
    int max_out = n_prompt + max_tokens;
    int* all_tokens = (int*)malloc((size_t)max_out * sizeof(int));
    memcpy(all_tokens, prompt_tokens, (size_t)n_prompt * sizeof(int));
    int total = n_prompt;

    // Prefill
    float* logits = (float*)malloc((size_t)VS * sizeof(float));
    minimind_prefill(m, all_tokens, total, logits);

    // Sample first token
    int next = minimind_sample_token(logits, VS, top_k, temperature);
    all_tokens[total++] = next;

    // Decode loop
    while (total < max_out) {
        if (next == MINIMIND_EOS_TOKEN_ID) break;
        minimind_decode(m, next, logits);
        next = minimind_sample_token(logits, VS, top_k, temperature);
        all_tokens[total++] = next;
    }

    // Decode only the generated part (skip prompt)
    int* gen_tokens = all_tokens + n_prompt;
    int n_gen = total - n_prompt;
    char* result = minimind_tokenizer_decode(g_tokenizer, gen_tokens, n_gen);

    free(logits);
    free(prompt_tokens);
    free(all_tokens);
    return result;
}

// Global tokenizer access for engine
void minimind_engine_set_tokenizer(minimind_tokenizer_t* tok) {
    g_tokenizer = tok;
}
