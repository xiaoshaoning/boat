// test_tokenizer.c - Debug BPE tokenizer
#include "tokenizer.h"
#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    minimind_tokenizer_t* tok = minimind_tokenizer_load("./weights");

    // Test encoding "你好"
    const char* text = "你好";
    int tokens[128];
    int n = minimind_tokenizer_encode(tok, text, tokens, 128);

    printf("Encoded %d tokens for \"%s\":\n", n, text);
    for (int i = 0; i < n; i++) {
        printf("  [%d] id=%d", i, tokens[i]);
        if (tokens[i] < tok->vocab_size && tok->vocab[tokens[i]][0]) {
            // Print hex of token bytes
            printf(" token=\"");
            for (char* p = tok->vocab[tokens[i]]; *p; p++)
                printf("\\x%02x", (unsigned char)*p);
            printf("\"");
        }
        printf("\n");
    }

    // Also test: "user\n"
    printf("\n--- user\\n ---\n");
    n = minimind_tokenizer_encode(tok, "user\n", tokens, 128);
    printf("Encoded %d tokens:\n", n);
    for (int i = 0; i < n; i++) {
        printf("  [%d] id=%d -> \"", i, tokens[i]);
        for (char* p = tok->vocab[tokens[i]]; *p; p++)
            printf("\\x%02x", (unsigned char)*p);
        printf("\"\n");
    }

    minimind_tokenizer_free(tok);
    return 0;
}
