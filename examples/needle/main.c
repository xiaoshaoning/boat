// main.c - Needle 2 inference via Boat: a C reimplementation of the Simple
// Attention Network (SAN) decode path, reading the self-contained .cact
// deployment blob produced by the needle repo.
//
// Usage:
//   needle2 <model.cact> [--prompt "text"] [--max-new-tokens N] [--temperature T]
//   needle2 --selftest        (tokenizer self-test, no model file needed)
//
// Mirrors `needle run --checkpoint <ckpt> --prompt ...`.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cact.h"
#include "san.h"
#include "tokenizer.h"

static int selftest(void) {
    // Build a RAW tokenizer blob with a complete vocab (4 specials + a few
    // normal/user pieces + all 256 byte pieces) and check round-trip encoding
    // plus marker detection. Exercises the parser without any model file.
    enum { N_SPECIAL = 4, N_EXTRA = 4, N_PIECES = N_SPECIAL + N_EXTRA + 256 };
    const char* special[] = {"<pad>", "</s>", "<s>", "<unk>"};
    const char* extra[] = {"hello", "world",
                           "\xE2\x96\x81"
                           "hello",
                           "<tool_call>"};
    unsigned char etypes[] = {0, 0, 0, 3};
    // Max blob size: header + N_PIECES * (7 + longest surface).
    size_t cap = 24 + (size_t)N_PIECES * (7 + 12);
    unsigned char* blob = (unsigned char*)calloc(1, cap);
    if (!blob) return 1;
    unsigned char* p = blob;
    unsigned int v = N_PIECES;
    memcpy(p, &v, 4);
    p += 4;
    v = 0;
    memcpy(p, &v, 4);
    p += 4; // pad
    v = 1;
    memcpy(p, &v, 4);
    p += 4; // eos
    v = 2;
    memcpy(p, &v, 4);
    p += 4; // bos
    v = 3;
    memcpy(p, &v, 4);
    p += 4; // unk
    p[0] = 1;
    p[1] = 1; // add_dummy_prefix, byte_fallback
    p += 4;   // 2 flags + 2-byte pad

    int id_marker = -1;
    for (int i = 0; i < N_PIECES; i++) {
        char surf[16];
        unsigned char type;
        float score = -100.0f + (float)i;
        if (i < N_SPECIAL) {
            snprintf(surf, sizeof(surf), "%s", special[i]);
            type = 2; // CONTROL
        } else if (i < N_SPECIAL + N_EXTRA) {
            int k = i - N_SPECIAL;
            snprintf(surf, sizeof(surf), "%s", extra[k]);
            type = etypes[k];
            if (k == 3) id_marker = i; // <tool_call>
        } else {
            snprintf(surf, sizeof(surf), "<0x%02X>", i - (N_SPECIAL + N_EXTRA));
            type = 4; // BYTE
        }
        unsigned int len = (unsigned int)strlen(surf);
        memcpy(p, &score, 4);
        p += 4;
        *p++ = type;
        memcpy(p, &len, 2);
        p += 2;
        memcpy(p, surf, len);
        p += len;
    }
    needle_tokenizer_t tok;
    if (needle_tokenizer_init(&tok, blob, (size_t)(p - blob)) != 0) {
        fprintf(stderr, "selftest: tokenizer init failed\n");
        free(blob);
        return 1;
    }
    free(blob);
    int ids[1024];
    int n = needle_tokenizer_encode(&tok, "hello world", ids, 1024);
    char* dec = needle_tokenizer_decode(&tok, ids, (size_t)(n < 0 ? 0 : n));
    if (n <= 0 || !dec || strcmp(dec, "hello world") != 0) {
        fprintf(stderr, "selftest: round-trip failed (n=%d, dec=%s)\n", n, dec ? dec : "(null)");
        free(dec);
        needle_tokenizer_free(&tok);
        return 1;
    }
    free(dec);
    if (id_marker < 0) {
        fprintf(stderr, "selftest: marker piece missing\n");
        needle_tokenizer_free(&tok);
        return 1;
    }
    n = needle_tokenizer_encode(&tok, "a <tool_call> b", ids, 1024);
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (ids[i] == id_marker) found = 1;
    }
    if (!found) {
        fprintf(stderr, "selftest: marker not emitted\n");
        needle_tokenizer_free(&tok);
        return 1;
    }
    needle_tokenizer_free(&tok);
    printf("selftest OK (tokenizer round-trip + marker detection)\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: needle2 <model.cact> [--prompt \"...\"] [--max-new-tokens N] "
                        "[--temperature T]\n"
                        "       needle2 --selftest\n");
        return 1;
    }
    if (strcmp(argv[1], "--selftest") == 0) {
        return selftest();
    }

    const char* cact_path = argv[1];
    const char* prompt = "The most surprising thing about";
    int max_new = 64;
    float temperature = 0.0f;
    int print_tokens = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--max-new-tokens") == 0 && i + 1 < argc) {
            max_new = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--print-tokens") == 0) {
            print_tokens = 1;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 1;
        }
    }

    needle_cact_t cact;
    if (needle_cact_open(&cact, cact_path) != 0) return 1;
    needle_model_t* model = needle_model_load(&cact);
    if (!model) {
        fprintf(stderr, "failed to load model from %s\n", cact_path);
        needle_cact_close(&cact);
        return 1;
    }
    uint32_t tok_idx = needle_cact_index_tokenizer(&cact);
    if (tok_idx >= cact.hdr.num_tensors) {
        fprintf(stderr, "no tokenizer found in the .cact blob\n");
        needle_model_free(model);
        needle_cact_close(&cact);
        return 1;
    }
    const uint8_t* tok_blob;
    size_t tok_nbytes;
    needle_cact_tensor_raw(&cact, tok_idx, &tok_blob, &tok_nbytes);
    needle_tokenizer_t tok;
    if (needle_tokenizer_init(&tok, tok_blob, tok_nbytes) != 0) {
        fprintf(stderr, "failed to parse tokenizer\n");
        needle_model_free(model);
        needle_cact_close(&cact);
        return 1;
    }

    printf("model: %s\n", needle_model_engine_version());
    printf("prompt: %s\n", prompt);
    fflush(stdout);

    if (print_tokens) {
        size_t n = 0;
        int32_t* ids = needle_model_generate_ids(model, &tok, prompt, max_new, temperature, &n);
        if (!ids) {
            fprintf(stderr, "generation failed\n");
            needle_tokenizer_free(&tok);
            needle_model_free(model);
            needle_cact_close(&cact);
            return 1;
        }
        for (size_t i = 0; i < n; i++)
            printf("%s%d", i ? " " : "", ids[i]);
        printf("\n");
        free(ids);
        needle_tokenizer_free(&tok);
        needle_model_free(model);
        needle_cact_close(&cact);
        return 0;
    }

    char* text = needle_model_generate(model, &tok, prompt, max_new, temperature, stdout);
    printf("\n");
    if (!text) {
        fprintf(stderr, "generation failed\n");
        needle_tokenizer_free(&tok);
        needle_model_free(model);
        needle_cact_close(&cact);
        return 1;
    }
    printf("---\n%s\n", text);
    free(text);

    needle_tokenizer_free(&tok);
    needle_model_free(model);
    needle_cact_close(&cact);
    return 0;
}
