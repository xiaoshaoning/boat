// minimind_gen.c - Read tokens from gen_input.bin, generate, write to gen_output.bin
// Used for tokenizer-bypass testing.
#include "model.h"
#include "sampling.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    const char* model_dir = (argc > 1) ? argv[1] : "./weights";

    // Load model
    minimind_model_t m;
    if (minimind_model_init(&m, model_dir) != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Read input
    FILE* f = fopen("gen_input.bin", "rb");
    if (!f) { fprintf(stderr, "Cannot open gen_input.bin\n"); return 1; }
    int n_prompt, max_tokens;
    fread(&n_prompt, sizeof(int), 1, f);
    int* tokens = (int*)malloc(((size_t)n_prompt + 4096) * sizeof(int));
    for (int i = 0; i < n_prompt; i++) fread(&tokens[i], sizeof(int), 1, f);
    fread(&max_tokens, sizeof(int), 1, f);
    fclose(f);

    printf("Generating from %d prompt tokens, max %d new tokens\n", n_prompt, max_tokens);

    int VS = m.config.vocab_size;
    float* logits = (float*)malloc((size_t)VS * sizeof(float));
    int total = n_prompt;

    // Prefill
    minimind_prefill(&m, tokens, total, logits);

    // Generate loop
    float temp = 0.85f;
    int top_k = 50;
    int next = minimind_sample_token(logits, VS, top_k, temp);
    tokens[total++] = next;
    printf("Generated %d tokens so far\n", total - n_prompt);

    while (total - n_prompt < max_tokens) {
        if (next == MINIMIND_EOS_TOKEN_ID) break;
        minimind_decode(&m, next, logits);
        next = minimind_sample_token(logits, VS, top_k, temp);
        tokens[total++] = next;
    }

    // Write output tokens
    int n_gen = total - n_prompt;
    f = fopen("gen_output.bin", "wb");
    fwrite(&n_gen, sizeof(int), 1, f);
    for (int i = n_prompt; i < total; i++) fwrite(&tokens[i], sizeof(int), 1, f);
    fclose(f);

    printf("Wrote %d generated tokens to gen_output.bin\n", n_gen);

    free(logits);
    free(tokens);
    minimind_model_free(&m);
    return 0;
}
