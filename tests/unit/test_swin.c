// test_swin.c - Swin Transformer encoder forward test
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0
//
// Exercises boat_swin_forward with a small synthetic config + weights, checks
// the output shape, and frees everything (valgrind verifies no leaks).

#include <boat/layers/swin.h>
#include <boat/tensor.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>

static boat_tensor_t* ft(const int64_t* shape, size_t ndim) {
    boat_tensor_t* t = boat_tensor_create(shape, ndim, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    assert(t != NULL);
    float* d = (float*)boat_tensor_data(t);
    for (size_t i = 0; i < boat_tensor_nelements(t); i++) d[i] = 0.1f * (float)((i * 7) % 11);
    return t;
}

static void fill_block(boat_swin_block_weights_t* b, int dim) {
    int hidden = 4 * dim;
    int64_t v1[] = {dim}, h1[] = {hidden};
    int64_t m2[] = {dim, dim};
    int64_t fc1[] = {dim, hidden}, fc2[] = {hidden, dim};
    b->norm1_weight = ft(v1, 1); b->norm1_bias = ft(v1, 1);
    b->query_weight = ft(m2, 2); b->query_bias = ft(v1, 1);
    b->key_weight   = ft(m2, 2); b->key_bias   = ft(v1, 1);
    b->value_weight = ft(m2, 2); b->value_bias = ft(v1, 1);
    b->proj_weight  = ft(m2, 2); b->proj_bias  = ft(v1, 1);
    b->norm2_weight = ft(v1, 1); b->norm2_bias = ft(v1, 1);
    b->mlp_fc1_weight = ft(fc1, 2); b->mlp_fc1_bias = ft(h1, 1);
    b->mlp_fc2_weight = ft(fc2, 2); b->mlp_fc2_bias = ft(v1, 1);
    b->rel_pos_bias_table = NULL;
    b->rel_pos_index = NULL;
}

static void free_block(boat_swin_block_weights_t* b) {
    if (!b) return;
    boat_tensor_free(b->norm1_weight); boat_tensor_free(b->norm1_bias);
    boat_tensor_free(b->query_weight); boat_tensor_free(b->query_bias);
    boat_tensor_free(b->key_weight);   boat_tensor_free(b->key_bias);
    boat_tensor_free(b->value_weight); boat_tensor_free(b->value_bias);
    boat_tensor_free(b->proj_weight);  boat_tensor_free(b->proj_bias);
    boat_tensor_free(b->norm2_weight); boat_tensor_free(b->norm2_bias);
    boat_tensor_free(b->mlp_fc1_weight); boat_tensor_free(b->mlp_fc1_bias);
    boat_tensor_free(b->mlp_fc2_weight); boat_tensor_free(b->mlp_fc2_bias);
    boat_tensor_free(b->rel_pos_bias_table);
    boat_tensor_free(b->rel_pos_index);
}

static void fill_downsample(boat_swin_downsample_weights_t* d, int dim) {
    int64_t d4[] = {4 * dim}, d2[] = {2 * dim};
    int64_t red[] = {4 * dim, 2 * dim};
    d->norm_weight = ft(d4, 1); d->norm_bias = ft(d4, 1);
    d->reduction_weight = ft(red, 2); d->reduction_bias = ft(d2, 1);
}

static void free_downsample(boat_swin_downsample_weights_t* d) {
    if (!d) return;
    boat_tensor_free(d->norm_weight); boat_tensor_free(d->norm_bias);
    boat_tensor_free(d->reduction_weight); boat_tensor_free(d->reduction_bias);
}

static void test_swin_forward(void) {
    printf("Testing Swin Transformer forward...\n");

    boat_swin_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.embed_dim = 8;
    for (int i = 0; i < 4; i++) { cfg.depths[i] = 1; cfg.num_heads[i] = 2; }
    cfg.window_size = 2;
    cfg.patch_size = 4;
    cfg.num_channels = 3;
    cfg.mlp_ratio = 4.0f;
    cfg.qkv_bias = true;
    cfg.layer_norm_eps = 1e-5f;

    boat_swin_weights_t w;
    memset(&w, 0, sizeof(w));

    // Patch embed: proj_weight [embed_dim, C, ps, ps].
    int64_t pew[] = {8, 3, 4, 4}, pe8[] = {8};
    w.patch_embed.proj_weight = ft(pew, 4);
    w.patch_embed.proj_bias = ft(pe8, 1);
    w.patch_embed.norm_weight = ft(pe8, 1);
    w.patch_embed.norm_bias = ft(pe8, 1);

    // Stages.
    int dim = cfg.embed_dim;
    for (int s = 0; s < 4; s++) {
        w.stages[s].blocks = (boat_swin_block_weights_t*)malloc(sizeof(boat_swin_block_weights_t));
        memset(w.stages[s].blocks, 0, sizeof(boat_swin_block_weights_t));
        fill_block(w.stages[s].blocks, dim);
        if (s < 3) {
            w.stages[s].downsample = (boat_swin_downsample_weights_t*)malloc(sizeof(boat_swin_downsample_weights_t));
            memset(w.stages[s].downsample, 0, sizeof(boat_swin_downsample_weights_t));
            fill_downsample(w.stages[s].downsample, dim);
            dim *= 2;
        }
    }

    int64_t ish[] = {1, 3, 64, 64};
    boat_tensor_t* input = boat_tensor_create(ish, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    assert(input != NULL);
    float* id = (float*)boat_tensor_data(input);
    for (size_t i = 0; i < boat_tensor_nelements(input); i++) id[i] = 0.05f * (float)((i * 3) % 13);

    boat_tensor_t* out = boat_swin_forward(&cfg, &w, input);
    assert(out != NULL);
    // [1, num_patches, final_dim] = [1, 4, 64]
    assert(boat_tensor_ndim(out) == 3);
    assert(boat_tensor_shape(out)[0] == 1);
    assert(boat_tensor_shape(out)[1] == 4);
    assert(boat_tensor_shape(out)[2] == 64);
    // Sanity: output has no NaNs.
    float* od = (float*)boat_tensor_data(out);
    for (size_t i = 0; i < boat_tensor_nelements(out); i++) assert(od[i] == od[i]);

    boat_tensor_free(out);
    boat_tensor_free(input);

    // Free all weights.
    boat_tensor_free(w.patch_embed.proj_weight);
    boat_tensor_free(w.patch_embed.proj_bias);
    boat_tensor_free(w.patch_embed.norm_weight);
    boat_tensor_free(w.patch_embed.norm_bias);
    for (int s = 0; s < 4; s++) {
        free_block(w.stages[s].blocks);
        free(w.stages[s].blocks);
        if (s < 3) {
            free_downsample(w.stages[s].downsample);
            free(w.stages[s].downsample);
        }
    }

    printf("  OK\n");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== Swin Transformer Test ===\n\n");
    test_swin_forward();
    printf("\n=== Swin test passed ===\n");
    return 0;
}
