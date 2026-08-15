// test_forward.c - Compare C forward pass logits against Python reference
#include "model.h"
#include "weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <model_dir> <test_name>\n", argv[0]);
        printf("  e.g. %s ./weights test_18tokens\n", argv[0]);
        return 1;
    }

    minimind_model_t m;
    if (minimind_model_init(&m, argv[1]) != 0) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    // Read input tokens
    char path[512];
    snprintf(path, sizeof(path), "forward_test/%s_input.bin", argv[2]);
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        return 1;
    }
    int n_tokens;
    fread(&n_tokens, sizeof(int), 1, f);
    int* tokens = (int*)malloc((size_t)n_tokens * sizeof(int));
    for (int i = 0; i < n_tokens; i++)
        fread(&tokens[i], sizeof(int), 1, f);
    fclose(f);

    // Run forward pass
    int VS = m.config.vocab_size;
    float* logits = (float*)malloc((size_t)VS * sizeof(float));
    minimind_prefill(&m, tokens, n_tokens, logits);

    // Read expected logits
    snprintf(path, sizeof(path), "forward_test/%s_logits.bin", argv[2]);
    f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path);
        return 1;
    }
    float* expected = (float*)malloc((size_t)VS * sizeof(float));
    fread(expected, sizeof(float), (size_t)VS, f);
    fclose(f);

    // Compare
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;
    int top5_c[5] = {0}, top5_py[5] = {0};
    float top5_c_val[5] = {0}, top5_py_val[5] = {0};

    for (int i = 0; i < VS; i++) {
        float err = fabsf(logits[i] - expected[i]);
        if (err > max_abs_err) max_abs_err = err;

        float rel = (fabsf(expected[i]) > 1e-6f) ? err / fabsf(expected[i]) : 0.0f;
        if (rel > max_rel_err) max_rel_err = rel;

        // Track top-5 for C and Python
        float cv = logits[i], pv = expected[i];
        for (int k = 0; k < 5; k++) {
            if (cv > top5_c_val[k]) {
                for (int j = 4; j > k; j--) {
                    top5_c[j] = top5_c[j - 1];
                    top5_c_val[j] = top5_c_val[j - 1];
                }
                top5_c[k] = i;
                top5_c_val[k] = cv;
                break;
            }
            if (pv > top5_py_val[k]) {
                for (int j = 4; j > k; j--) {
                    top5_py[j] = top5_py[j - 1];
                    top5_py_val[j] = top5_py_val[j - 1];
                }
                top5_py[k] = i;
                top5_py_val[k] = pv;
                break;
            }
        }
    }

    printf("Test: %s (%d tokens)\n", argv[2], n_tokens);
    printf("Max abs error: %e\n", max_abs_err);
    printf("Max rel error: %e\n", max_rel_err);
    printf("C   top-5: ");
    for (int k = 0; k < 5; k++)
        printf("[id=%d val=%.4f] ", top5_c[k], top5_c_val[k]);
    printf("\nPy  top-5: ");
    for (int k = 0; k < 5; k++)
        printf("[id=%d val=%.4f] ", top5_py[k], top5_py_val[k]);
    printf("\n");

    if (max_abs_err < 1e-3f && max_rel_err < 1e-4f) {
        printf("PASS: Forward pass matches Python reference.\n");
    } else if (max_abs_err < 1.0f) {
        printf("WARN: Forward pass has moderate errors - check numerical precision.\n");
    } else {
        printf("FAIL: Forward pass has large errors - possible bug in implementation.\n");
    }

    free(tokens);
    free(logits);
    free(expected);
    minimind_model_free(&m);
    return 0;
}
