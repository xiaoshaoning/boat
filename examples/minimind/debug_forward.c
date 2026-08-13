// debug_forward.c - Compare layer-by-layer hidden states
#include "model.h"
#include "weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Forward decls from model.c
void layer_forward(minimind_model_t* m, int layer_idx, float* hidden, int n_tokens, int start_pos);
void rmsnorm(float* x, const float* weight, int n_rows, int dim, float eps);

int main(int argc, char** argv) {
    minimind_model_t m;
    if (minimind_model_init(&m, "./weights") != 0) { return 1; }

    // Read input tokens
    FILE* f = fopen("debug_data/input.bin", "rb");
    int n_tokens;
    fread(&n_tokens, sizeof(int), 1, f);
    int tokens[32];
    for (int i = 0; i < n_tokens; i++) fread(&tokens[i], sizeof(int), 1, f);
    fclose(f);

    int HS = m.config.hidden_size;
    int VS = m.config.vocab_size;
    float eps = m.config.rms_eps;

    // Reset KV cache
    minimind_model_reset_kv_cache(&m);

    // Embedding lookup
    for (int t = 0; t < n_tokens; t++) {
        int tid = tokens[t];
        memcpy(m.hidden + t * HS, m.embed_tokens + (size_t)tid * HS, HS * sizeof(float));
    }

    // Forward through layers, saving hidden state after each layer
    for (int l = 0; l < m.config.num_layers; l++) {
        layer_forward(&m, l, m.hidden, n_tokens, 0);

        // Compare with Python
        char path[128];
        snprintf(path, sizeof(path), "debug_data/layer%d_hidden.bin", l);
        f = fopen(path, "rb");
        if (f) {
            float* py_hidden = (float*)malloc((size_t)n_tokens * HS * sizeof(float));
            fread(py_hidden, sizeof(float), (size_t)n_tokens * HS, f);
            fclose(f);

            float max_err = 0.0f;
            for (int i = 0; i < n_tokens * HS; i++) {
                float err = fabsf(m.hidden[i] - py_hidden[i]);
                if (err > max_err) max_err = err;
            }
            printf("Layer %d: max abs error = %e", l, max_err);
            if (max_err < 1e-4f) printf(" OK\n");
            else if (max_err < 1e-2f) printf(" WARN\n");
            else printf(" FAIL\n");
            free(py_hidden);
        }
    }
    m.kv_len = n_tokens;

    // Final RMSNorm
    rmsnorm(m.hidden, m.final_norm_weight, n_tokens, HS, eps);

    // LM Head
    float* logits = (float*)malloc((size_t)VS * sizeof(float));
    const float* last_hidden = m.hidden + (n_tokens - 1) * HS;
    for (int v = 0; v < VS; v++) {
        const float* emb_row = m.lm_head + (size_t)v * HS;
        float dot = 0.0f;
        for (int i = 0; i < HS; i++) dot += last_hidden[i] * emb_row[i];
        logits[v] = dot;
    }

    // Compare logits
    f = fopen("debug_data/logits.bin", "rb");
    float* py_logits = (float*)malloc((size_t)VS * sizeof(float));
    fread(py_logits, sizeof(float), (size_t)VS, f);
    fclose(f);
    float max_err = 0.0f;
    for (int i = 0; i < VS; i++) {
        float err = fabsf(logits[i] - py_logits[i]);
        if (err > max_err) max_err = err;
    }
    printf("Logits: max abs error = %e\n", max_err);

    free(logits);
    free(py_logits);
    minimind_model_free(&m);
    return 0;
}
