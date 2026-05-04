// nanochat_cli.c - NanoChat CLI entry point
#include "nanochat.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: nanochat_cli <model_dir> <prompt> [max_tokens=256] [temperature=0.7] [top_k=40]\n");
        return 1;
    }

    const char* model_dir = argv[1];
    const char* prompt = argv[2];
    int max_tokens = (argc > 3) ? atoi(argv[3]) : 256;
    float temperature = (argc > 4) ? (float)atof(argv[4]) : 0.7f;
    int top_k = (argc > 5) ? atoi(argv[5]) : 40;

    if (max_tokens <= 0) max_tokens = 256;
    if (max_tokens > 1024) max_tokens = 1024;
    if (temperature < 0.0f) temperature = 0.0f;
    if (top_k <= 0) top_k = 1;

    fprintf(stderr, "[NanoChat] Initializing engine from: %s\n", model_dir);

    nanochat_engine_t* eng = nanochat_engine_create(model_dir);
    if (!eng) {
        fprintf(stderr, "[NanoChat] Failed to create engine\n");
        return 1;
    }

    fprintf(stderr, "[NanoChat] Generating...\n");
    fprintf(stderr, "----------------------------------------\n");

    char* result = nanochat_generate(eng, prompt, max_tokens, temperature, top_k);

    if (result) {
        printf("%s\n", result);
        free(result);
    } else {
        fprintf(stderr, "[NanoChat] Generation failed\n");
    }

    nanochat_engine_free(eng);
    return 0;
}
