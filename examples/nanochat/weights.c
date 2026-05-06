// weights.c - Load NanoChat weights from HuggingFace safetensors
#include "weights.h"
#include "../common/safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

nanochat_weights_t* nanochat_weights_load(const char* model_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);

    safetensors_t st;
    if (!safetensors_open(&st, path)) {
        fprintf(stderr, "[NanoChat] Failed to open %s\n", path);
        return NULL;
    }

    nanochat_weights_t* w = (nanochat_weights_t*)calloc(1, sizeof(nanochat_weights_t));
    if (!w) { safetensors_close(&st); return NULL; }

    w->n_layers = NANOCHAT_NUM_LAYERS;
    w->vocab_size = NANOCHAT_VOCAB_SIZE;
    w->hidden_size = NANOCHAT_HIDDEN_SIZE;
    w->intermediate_size = NANOCHAT_INTERMEDIATE_SIZE;

#define LOAD_WEIGHT(dst, name) do { \
    int idx = safetensors_find(&st, name); \
    if (idx < 0) { fprintf(stderr, "[NanoChat] Missing: %s\n", name); \
                   nanochat_weights_free(w); safetensors_close(&st); return NULL; } \
    boat_tensor_t* t = safetensors_load_tensor(&st, idx, 0); \
    if (!t) { fprintf(stderr, "[NanoChat] Failed to load: %s\n", name); \
              nanochat_weights_free(w); safetensors_close(&st); return NULL; } \
    size_t n = boat_tensor_nelements(t); \
    dst = (float*)malloc(n * sizeof(float)); \
    memcpy(dst, boat_tensor_data(t), n * sizeof(float)); \
    boat_tensor_unref(t); \
} while(0)

    LOAD_WEIGHT(w->embed_tokens, "model.embed_tokens.weight");
    LOAD_WEIGHT(w->lm_head, "lm_head.weight");

    for (int l = 0; l < NANOCHAT_NUM_LAYERS; l++) {
        char key[256];
        snprintf(key, sizeof(key), "model.layers.%d.self_attn.q_proj.weight", l);
        LOAD_WEIGHT(w->q_proj[l], key);
        snprintf(key, sizeof(key), "model.layers.%d.self_attn.k_proj.weight", l);
        LOAD_WEIGHT(w->k_proj[l], key);
        snprintf(key, sizeof(key), "model.layers.%d.self_attn.v_proj.weight", l);
        LOAD_WEIGHT(w->v_proj[l], key);
        snprintf(key, sizeof(key), "model.layers.%d.self_attn.o_proj.weight", l);
        LOAD_WEIGHT(w->o_proj[l], key);
        snprintf(key, sizeof(key), "model.layers.%d.mlp.fc1.weight", l);
        LOAD_WEIGHT(w->fc1[l], key);
        snprintf(key, sizeof(key), "model.layers.%d.mlp.fc2.weight", l);
        LOAD_WEIGHT(w->fc2[l], key);
    }

    safetensors_close(&st);
    fprintf(stderr, "[NanoChat] Loaded %d layers from %s\n", NANOCHAT_NUM_LAYERS, path);
    return w;
}

void nanochat_weights_free(nanochat_weights_t* w) {
    if (!w) return;
    free(w->embed_tokens);
    free(w->lm_head);
    for (int l = 0; l < w->n_layers && l < NANOCHAT_NUM_LAYERS; l++) {
        free(w->q_proj[l]); free(w->k_proj[l]); free(w->v_proj[l]); free(w->o_proj[l]);
        free(w->fc1[l]); free(w->fc2[l]);
    }
    memset(w, 0, sizeof(*w));
    free(w);
}
