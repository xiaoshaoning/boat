// minimind_cli.c - MiniMind CLI chat demo
#include "model.h"
#include "engine.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <locale.h>

#ifdef _WIN32
#include <windows.h>
static char* ansi_to_utf8(const char* ansi) {
    int wlen = MultiByteToWideChar(CP_ACP, 0, ansi, -1, NULL, 0);
    if (wlen <= 0) return strdup(ansi);
    wchar_t* wbuf = (wchar_t*)malloc(wlen * sizeof(wchar_t));
    MultiByteToWideChar(CP_ACP, 0, ansi, -1, wbuf, wlen);
    int ulen = WideCharToMultiByte(CP_UTF8, 0, wbuf, -1, NULL, 0, NULL, NULL);
    char* utf8 = (char*)malloc(ulen + 1);
    WideCharToMultiByte(CP_UTF8, 0, wbuf, -1, utf8, ulen, NULL, NULL);
    utf8[ulen] = '\0';
    free(wbuf);
    return utf8;
}
#endif

extern void minimind_engine_set_tokenizer(minimind_tokenizer_t* tok);

int main(int argc, char** argv) {
    setlocale(LC_ALL, ".UTF8");

    if (argc < 2) {
        printf("Usage: %s <model_dir> [prompt] [max_tokens] [temp] [top_k]\n", argv[0]);
        printf("  model_dir:  directory with model.bin, model_meta.json, tokenizer.json\n");
        printf("  prompt:     single-shot generation (omit for interactive mode)\n");
        printf("  max_tokens: max new tokens (default 256)\n");
        printf("  temp:       temperature, 0=greedy (default 0.85)\n");
        printf("  top_k:      top-k filtering (default 50)\n");
        return 1;
    }

    const char* model_dir = argv[1];
    #ifdef _WIN32
    char* utf8_prompt = NULL;
    if (argc > 2) utf8_prompt = ansi_to_utf8(argv[2]);
    const char* prompt = utf8_prompt ? utf8_prompt : (argc > 2 ? argv[2] : NULL);
    #else
    const char* prompt = (argc > 2) ? argv[2] : NULL;
    #endif
    int max_tokens = (argc > 3) ? atoi(argv[3]) : 256;
    float temp = (argc > 4) ? (float)atof(argv[4]) : 0.85f;
    int top_k = (argc > 5) ? atoi(argv[5]) : 50;

    srand((unsigned)time(NULL));

    printf("Loading model from %s ...\n", model_dir);
    minimind_model_t model;
    if (minimind_model_init(&model, model_dir) != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    printf("Loading tokenizer...\n");
    minimind_tokenizer_t* tok = minimind_tokenizer_load(model_dir);
    if (!tok) {
        fprintf(stderr, "Failed to load tokenizer\n");
        minimind_model_free(&model);
        return 1;
    }
    minimind_engine_set_tokenizer(tok);

    if (prompt) {
        char* chat_prompt = minimind_format_chat_prompt(prompt);
        printf("Q: %s\n", prompt);
        printf("A: "); fflush(stdout);

        clock_t st = clock();
        char* reply = minimind_generate(&model, chat_prompt, max_tokens, temp, top_k);
        clock_t et = clock();

        if (reply) {
            printf("%s\n", reply);
            printf("[%.2f sec]\n", (double)(et - st) / CLOCKS_PER_SEC);
            free(reply);
        }
        free(chat_prompt);
    } else {
        printf("MiniMind Chat (type 'quit' to exit)\n\n");
        char input[4096];
        while (1) {
            printf("You: "); fflush(stdout);
            if (!fgets(input, sizeof(input), stdin)) break;
            input[strcspn(input, "\n")] = '\0';
            if (strcmp(input, "quit") == 0 || strcmp(input, "exit") == 0) break;
            if (input[0] == '\0') continue;

            char* chat_prompt = minimind_format_chat_prompt(input);
            printf("MiniMind: "); fflush(stdout);

            clock_t st = clock();
            char* reply = minimind_generate(&model, chat_prompt, max_tokens, temp, top_k);
            clock_t et = clock();

            if (reply) {
                printf("%s\n", reply);
                printf("[%.2f sec]\n", (double)(et - st) / CLOCKS_PER_SEC);
                free(reply);
            }
            free(chat_prompt);
        }
    }

    #ifdef _WIN32
    free(utf8_prompt);
    #endif
    minimind_tokenizer_free(tok);
    minimind_model_free(&model);
    printf("Done.\n");
    return 0;
}
