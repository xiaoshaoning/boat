// weights.c — Qwen3-ASR safetensors weight loader
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include "weights.h"
#include "config.h"
#include "../common/safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Helper: load a 2D weight with transpose, allocate + memcpy
#define LOAD_WEIGHT_2D(dst, rows, cols, st, name) do { \
    int _idx = safetensors_find(st, name); \
    if (_idx < 0) { fprintf(stderr, "ERROR: missing weight: %s\n", name); goto fail; } \
    boat_tensor_t *_t = safetensors_load_tensor(st, _idx, 1); \
    if (!_t) { fprintf(stderr, "ERROR: load failed: %s\n", name); goto fail; } \
    dst = (float*)malloc((size_t)(rows)*(cols)*sizeof(float)); \
    if (!dst) { boat_tensor_unref(_t); goto fail; } \
    memcpy(dst, boat_tensor_data(_t), (size_t)(rows)*(cols)*sizeof(float)); \
    boat_tensor_unref(_t); \
} while(0)

// Helper: load a 1D weight
#define LOAD_WEIGHT_1D(dst, n, st, name) do { \
    int _idx = safetensors_find(st, name); \
    if (_idx < 0) { fprintf(stderr, "ERROR: missing weight: %s\n", name); goto fail; } \
    boat_tensor_t *_t = safetensors_load_tensor(st, _idx, 0); \
    if (!_t) { fprintf(stderr, "ERROR: load failed: %s\n", name); goto fail; } \
    dst = (float*)malloc((size_t)(n)*sizeof(float)); \
    if (!dst) { boat_tensor_unref(_t); goto fail; } \
    memcpy(dst, boat_tensor_data(_t), (size_t)(n)*sizeof(float)); \
    boat_tensor_unref(_t); \
} while(0)

// Helper: load conv weight (4D, no transpose)
#define LOAD_WEIGHT_CONV(dst, bytes, st, name) do { \
    int _idx = safetensors_find(st, name); \
    if (_idx < 0) { fprintf(stderr, "ERROR: missing weight: %s\n", name); goto fail; } \
    boat_tensor_t *_t = safetensors_load_tensor(st, _idx, 0); \
    if (!_t) { fprintf(stderr, "ERROR: load failed: %s\n", name); goto fail; } \
    dst = (float*)malloc((size_t)(bytes)); \
    if (!dst) { boat_tensor_unref(_t); goto fail; } \
    memcpy(dst, boat_tensor_data(_t), (size_t)(bytes)); \
    boat_tensor_unref(_t); \
} while(0)

// Helper: load a 2D weight WITHOUT transpose (for lookup tables like embed_tokens)
#define LOAD_WEIGHT_2D_NT(dst, rows, cols, st, name) do { \
    int _idx = safetensors_find(st, name); \
    if (_idx < 0) { fprintf(stderr, "ERROR: missing weight: %s\n", name); goto fail; } \
    boat_tensor_t *_t = safetensors_load_tensor(st, _idx, 0); \
    if (!_t) { fprintf(stderr, "ERROR: load failed: %s\n", name); goto fail; } \
    dst = (float*)malloc((size_t)(rows)*(cols)*sizeof(float)); \
    if (!dst) { boat_tensor_unref(_t); goto fail; } \
    memcpy(dst, boat_tensor_data(_t), (size_t)(rows)*(cols)*sizeof(float)); \
    boat_tensor_unref(_t); \
} while(0)

qwen3asr_weights_t* qwen3asr_weights_load(const char *model_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/qwen3_asr_weights.safetensors", model_dir);

    safetensors_t st;
    if (!safetensors_open(&st, path)) {
        fprintf(stderr, "ERROR: cannot open safetensors: %s\n", path);
        return NULL;
    }



    qwen3asr_weights_t *w = (qwen3asr_weights_t*)calloc(1, sizeof(qwen3asr_weights_t));
    if (!w) {
        safetensors_close(&st);
        return NULL;
    }

    // ---- Encoder: conv frontend ----
    LOAD_WEIGHT_CONV(w->conv1_w, 480 * 1 * 3 * 3 * 4, &st, "audio_tower.conv2d1.weight");
    LOAD_WEIGHT_1D(w->conv1_b, 480, &st, "audio_tower.conv2d1.bias");
    LOAD_WEIGHT_CONV(w->conv2_w, 480 * 480 * 3 * 3 * 4, &st, "audio_tower.conv2d2.weight");
    LOAD_WEIGHT_1D(w->conv2_b, 480, &st, "audio_tower.conv2d2.bias");
    LOAD_WEIGHT_CONV(w->conv3_w, 480 * 480 * 3 * 3 * 4, &st, "audio_tower.conv2d3.weight");
    LOAD_WEIGHT_1D(w->conv3_b, 480, &st, "audio_tower.conv2d3.bias");
    LOAD_WEIGHT_2D(w->conv_out_w, 896, 7680, &st, "audio_tower.conv_out.weight");

    // ---- Encoder: 18 layers ----
    for (int l = 0; l < QWEN3ASR_ENCODER_NUM_LAYERS; l++) {
        char key[256];
        qwen3asr_encoder_layer_weights_t *lw = &w->encoder_layers[l];

        snprintf(key, sizeof(key), "audio_tower.layers.%d.self_attn.q_proj.weight", l);
        LOAD_WEIGHT_2D(lw->q_proj, 896, 896, &st, key);
        snprintf(key, sizeof(key), "audio_tower.layers.%d.self_attn.q_proj.bias", l);
        LOAD_WEIGHT_1D(lw->q_bias, 896, &st, key);

        snprintf(key, sizeof(key), "audio_tower.layers.%d.self_attn.k_proj.weight", l);
        LOAD_WEIGHT_2D(lw->k_proj, 896, 896, &st, key);
        snprintf(key, sizeof(key), "audio_tower.layers.%d.self_attn.k_proj.bias", l);
        LOAD_WEIGHT_1D(lw->k_bias, 896, &st, key);

        snprintf(key, sizeof(key), "audio_tower.layers.%d.self_attn.v_proj.weight", l);
        LOAD_WEIGHT_2D(lw->v_proj, 896, 896, &st, key);
        snprintf(key, sizeof(key), "audio_tower.layers.%d.self_attn.v_proj.bias", l);
        LOAD_WEIGHT_1D(lw->v_bias, 896, &st, key);

        snprintf(key, sizeof(key), "audio_tower.layers.%d.self_attn.out_proj.weight", l);
        LOAD_WEIGHT_2D(lw->o_proj, 896, 896, &st, key);
        snprintf(key, sizeof(key), "audio_tower.layers.%d.self_attn.out_proj.bias", l);
        LOAD_WEIGHT_1D(lw->o_bias, 896, &st, key);

        snprintf(key, sizeof(key), "audio_tower.layers.%d.self_attn_layer_norm.weight", l);
        LOAD_WEIGHT_1D(lw->attn_ln_w, 896, &st, key);
        snprintf(key, sizeof(key), "audio_tower.layers.%d.self_attn_layer_norm.bias", l);
        LOAD_WEIGHT_1D(lw->attn_ln_b, 896, &st, key);

        snprintf(key, sizeof(key), "audio_tower.layers.%d.fc1.weight", l);
        LOAD_WEIGHT_2D(lw->fc1_w, 3584, 896, &st, key);
        snprintf(key, sizeof(key), "audio_tower.layers.%d.fc1.bias", l);
        LOAD_WEIGHT_1D(lw->fc1_b, 3584, &st, key);

        snprintf(key, sizeof(key), "audio_tower.layers.%d.fc2.weight", l);
        LOAD_WEIGHT_2D(lw->fc2_w, 896, 3584, &st, key);
        snprintf(key, sizeof(key), "audio_tower.layers.%d.fc2.bias", l);
        LOAD_WEIGHT_1D(lw->fc2_b, 896, &st, key);

        snprintf(key, sizeof(key), "audio_tower.layers.%d.final_layer_norm.weight", l);
        LOAD_WEIGHT_1D(lw->final_ln_w, 896, &st, key);
        snprintf(key, sizeof(key), "audio_tower.layers.%d.final_layer_norm.bias", l);
        LOAD_WEIGHT_1D(lw->final_ln_b, 896, &st, key);
    }

    // ---- Encoder: post-projection ----
    LOAD_WEIGHT_1D(w->ln_post_w, 896, &st, "audio_tower.ln_post.weight");
    LOAD_WEIGHT_1D(w->ln_post_b, 896, &st, "audio_tower.ln_post.bias");
    LOAD_WEIGHT_2D(w->proj1_w, 896, 896, &st, "audio_tower.proj1.weight");
    LOAD_WEIGHT_1D(w->proj1_b, 896, &st, "audio_tower.proj1.bias");
    LOAD_WEIGHT_2D(w->proj2_w, 1024, 896, &st, "audio_tower.proj2.weight");
    LOAD_WEIGHT_1D(w->proj2_b, 1024, &st, "audio_tower.proj2.bias");

    // ---- Decoder: embed + norm + lm_head ----
    LOAD_WEIGHT_2D_NT(w->embed_tokens, 151936, 1024, &st, "model.embed_tokens.weight");
    LOAD_WEIGHT_1D(w->norm_w, 1024, &st, "model.norm.weight");
    LOAD_WEIGHT_2D(w->lm_head, 151936, 1024, &st, "lm_head.weight");

    // ---- Decoder: 28 layers ----
    for (int l = 0; l < QWEN3ASR_DECODER_NUM_LAYERS; l++) {
        char key[256];
        qwen3asr_decoder_layer_weights_t *lw = &w->decoder_layers[l];

        snprintf(key, sizeof(key), "model.layers.%d.self_attn.q_proj.weight", l);
        LOAD_WEIGHT_2D(lw->q_proj, 2048, 1024, &st, key);
        snprintf(key, sizeof(key), "model.layers.%d.self_attn.k_proj.weight", l);
        LOAD_WEIGHT_2D(lw->k_proj, 1024, 1024, &st, key);
        snprintf(key, sizeof(key), "model.layers.%d.self_attn.v_proj.weight", l);
        LOAD_WEIGHT_2D(lw->v_proj, 1024, 1024, &st, key);
        snprintf(key, sizeof(key), "model.layers.%d.self_attn.o_proj.weight", l);
        LOAD_WEIGHT_2D(lw->o_proj, 2048, 1024, &st, key);

        snprintf(key, sizeof(key), "model.layers.%d.self_attn.q_norm.weight", l);
        LOAD_WEIGHT_1D(lw->q_norm, 128, &st, key);
        snprintf(key, sizeof(key), "model.layers.%d.self_attn.k_norm.weight", l);
        LOAD_WEIGHT_1D(lw->k_norm, 128, &st, key);

        snprintf(key, sizeof(key), "model.layers.%d.mlp.gate_proj.weight", l);
        LOAD_WEIGHT_2D(lw->gate_proj, 3072, 1024, &st, key);
        snprintf(key, sizeof(key), "model.layers.%d.mlp.up_proj.weight", l);
        LOAD_WEIGHT_2D(lw->up_proj, 3072, 1024, &st, key);
        snprintf(key, sizeof(key), "model.layers.%d.mlp.down_proj.weight", l);
        LOAD_WEIGHT_2D(lw->down_proj, 3072, 1024, &st, key);

        snprintf(key, sizeof(key), "model.layers.%d.input_layernorm.weight", l);
        LOAD_WEIGHT_1D(lw->input_ln, 1024, &st, key);
        snprintf(key, sizeof(key), "model.layers.%d.post_attention_layernorm.weight", l);
        LOAD_WEIGHT_1D(lw->post_attn_ln, 1024, &st, key);
    }

    safetensors_close(&st);
    printf("Weights loaded: %d encoder layers, %d decoder layers, %.0f MB total\n",
           QWEN3ASR_ENCODER_NUM_LAYERS, QWEN3ASR_DECODER_NUM_LAYERS,
           /* rough estimate skipped */ 0.0);
    return w;

fail:
    safetensors_close(&st);
    qwen3asr_weights_free(w);
    return NULL;
}

static void free_encoder_layer(qwen3asr_encoder_layer_weights_t *lw) {
    free(lw->q_proj); free(lw->k_proj); free(lw->v_proj); free(lw->o_proj);
    free(lw->q_bias); free(lw->k_bias); free(lw->v_bias); free(lw->o_bias);
    free(lw->attn_ln_w); free(lw->attn_ln_b);
    free(lw->fc1_w); free(lw->fc1_b);
    free(lw->fc2_w); free(lw->fc2_b);
    free(lw->final_ln_w); free(lw->final_ln_b);
    memset(lw, 0, sizeof(*lw));
}

static void free_decoder_layer(qwen3asr_decoder_layer_weights_t *lw) {
    free(lw->q_proj); free(lw->k_proj); free(lw->v_proj); free(lw->o_proj);
    free(lw->q_norm); free(lw->k_norm);
    free(lw->gate_proj); free(lw->up_proj); free(lw->down_proj);
    free(lw->input_ln); free(lw->post_attn_ln);
    memset(lw, 0, sizeof(*lw));
}

void qwen3asr_weights_free(qwen3asr_weights_t *w) {
    if (!w) return;

    // Free encoder conv
    free(w->conv1_w); free(w->conv1_b);
    free(w->conv2_w); free(w->conv2_b);
    free(w->conv3_w); free(w->conv3_b);
    free(w->conv_out_w);

    // Free encoder layers
    for (int i = 0; i < QWEN3ASR_ENCODER_NUM_LAYERS; i++)
        free_encoder_layer(&w->encoder_layers[i]);

    // Free encoder post-projection
    free(w->ln_post_w); free(w->ln_post_b);
    free(w->proj1_w); free(w->proj1_b);
    free(w->proj2_w); free(w->proj2_b);

    // Free decoder globals
    free(w->embed_tokens);
    free(w->norm_w);
    free(w->lm_head);

    // Free decoder layers
    for (int i = 0; i < QWEN3ASR_DECODER_NUM_LAYERS; i++)
        free_decoder_layer(&w->decoder_layers[i]);

    memset(w, 0, sizeof(*w));
    free(w);
}
