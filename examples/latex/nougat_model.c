// nougat_model.c - Load Nougat-LaTeX weights from HuggingFace safetensors
#include "nougat_model.h"
#include "../common/safetensors.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Helpers: load a weight tensor, optionally transposing 2D linear weights
// ---------------------------------------------------------------------------
static boat_tensor_t* load_weight(const safetensors_t* st, const char* name, int do_transpose) {
    int idx = safetensors_find(st, name);
    if (idx < 0) {
        fprintf(stderr, "[Nougat] Missing weight: %s\n", name);
        if (st->count > 0) {
            fprintf(stderr, "[Nougat] First tensor in table: '%s'\n", st->tensors[0].name);
            fprintf(stderr, "[Nougat] Last tensor in table: '%s'\n",
                    st->tensors[st->count - 1].name);
        }
        return NULL;
    }
    boat_tensor_t* t = safetensors_load_tensor(st, idx, do_transpose);
    if (!t) {
        fprintf(stderr, "[Nougat] Failed to load: %s\n", name);
        return NULL;
    }
    return t;
}

// Load a required weight; exits on failure.
// NOTE: 'st' must be a safetensors_t* pointer in the calling scope.
#define LOAD(dst, name, transpose)                                                                 \
    do {                                                                                           \
        dst = load_weight(st, name, transpose);                                                    \
        if (!dst) {                                                                                \
            nougat_model_free(model);                                                              \
            safetensors_close(st);                                                                 \
            return NULL;                                                                           \
        }                                                                                          \
    } while (0)

// Load an optional weight (may be NULL).
#define LOAD_OPT(dst, name, transpose)                                                             \
    do {                                                                                           \
        dst = load_weight(st, name, transpose);                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// Swin block weight loading
// ---------------------------------------------------------------------------
static int load_swin_block(safetensors_t* st, nougat_model_t* model,
                           boat_swin_block_weights_t* block, const char* prefix, int dim,
                           int num_heads, int ws) {
    char key[512];

    snprintf(key, sizeof(key), "%s.attention.self.query.weight", prefix);
    LOAD(block->query_weight, key, 1);
    snprintf(key, sizeof(key), "%s.attention.self.query.bias", prefix);
    LOAD(block->query_bias, key, 0);

    snprintf(key, sizeof(key), "%s.attention.self.key.weight", prefix);
    LOAD(block->key_weight, key, 1);
    snprintf(key, sizeof(key), "%s.attention.self.key.bias", prefix);
    LOAD(block->key_bias, key, 0);

    snprintf(key, sizeof(key), "%s.attention.self.value.weight", prefix);
    LOAD(block->value_weight, key, 1);
    snprintf(key, sizeof(key), "%s.attention.self.value.bias", prefix);
    LOAD(block->value_bias, key, 0);

    snprintf(key, sizeof(key), "%s.attention.output.dense.weight", prefix);
    LOAD(block->proj_weight, key, 1);
    snprintf(key, sizeof(key), "%s.attention.output.dense.bias", prefix);
    LOAD(block->proj_bias, key, 0);

    snprintf(key, sizeof(key), "%s.layernorm_before.weight", prefix);
    LOAD(block->norm1_weight, key, 0);
    snprintf(key, sizeof(key), "%s.layernorm_before.bias", prefix);
    LOAD(block->norm1_bias, key, 0);

    snprintf(key, sizeof(key), "%s.layernorm_after.weight", prefix);
    LOAD(block->norm2_weight, key, 0);
    snprintf(key, sizeof(key), "%s.layernorm_after.bias", prefix);
    LOAD(block->norm2_bias, key, 0);

    snprintf(key, sizeof(key), "%s.intermediate.dense.weight", prefix);
    LOAD(block->mlp_fc1_weight, key, 1);
    snprintf(key, sizeof(key), "%s.intermediate.dense.bias", prefix);
    LOAD(block->mlp_fc1_bias, key, 0);

    snprintf(key, sizeof(key), "%s.output.dense.weight", prefix);
    LOAD(block->mlp_fc2_weight, key, 1);
    snprintf(key, sizeof(key), "%s.output.dense.bias", prefix);
    LOAD(block->mlp_fc2_bias, key, 0);

    snprintf(key, sizeof(key), "%s.attention.self.relative_position_bias_table", prefix);
    LOAD_OPT(block->rel_pos_bias_table, key, 0);

    snprintf(key, sizeof(key), "%s.attention.self.relative_position_index", prefix);
    LOAD_OPT(block->rel_pos_index, key, 0);

    return 1;
}

// ---------------------------------------------------------------------------
// Decoder layer weight loading
// ---------------------------------------------------------------------------
static int load_decoder_layer(safetensors_t* st, nougat_model_t* model,
                              boat_decoder_layer_weights_t* layer, const char* prefix) {
    char key[512];

    snprintf(key, sizeof(key), "%s.self_attn.q_proj.weight", prefix);
    LOAD(layer->self_q_weight, key, 1);
    snprintf(key, sizeof(key), "%s.self_attn.q_proj.bias", prefix);
    LOAD(layer->self_q_bias, key, 0);

    snprintf(key, sizeof(key), "%s.self_attn.k_proj.weight", prefix);
    LOAD(layer->self_k_weight, key, 1);
    snprintf(key, sizeof(key), "%s.self_attn.k_proj.bias", prefix);
    LOAD(layer->self_k_bias, key, 0);

    snprintf(key, sizeof(key), "%s.self_attn.v_proj.weight", prefix);
    LOAD(layer->self_v_weight, key, 1);
    snprintf(key, sizeof(key), "%s.self_attn.v_proj.bias", prefix);
    LOAD(layer->self_v_bias, key, 0);

    snprintf(key, sizeof(key), "%s.self_attn.out_proj.weight", prefix);
    LOAD(layer->self_o_weight, key, 1);
    snprintf(key, sizeof(key), "%s.self_attn.out_proj.bias", prefix);
    LOAD(layer->self_o_bias, key, 0);

    snprintf(key, sizeof(key), "%s.self_attn_layer_norm.weight", prefix);
    LOAD(layer->self_ln_weight, key, 0);
    snprintf(key, sizeof(key), "%s.self_attn_layer_norm.bias", prefix);
    LOAD(layer->self_ln_bias, key, 0);

    snprintf(key, sizeof(key), "%s.encoder_attn.q_proj.weight", prefix);
    LOAD(layer->cross_q_weight, key, 1);
    snprintf(key, sizeof(key), "%s.encoder_attn.q_proj.bias", prefix);
    LOAD(layer->cross_q_bias, key, 0);

    snprintf(key, sizeof(key), "%s.encoder_attn.k_proj.weight", prefix);
    LOAD(layer->cross_k_weight, key, 1);
    snprintf(key, sizeof(key), "%s.encoder_attn.k_proj.bias", prefix);
    LOAD(layer->cross_k_bias, key, 0);

    snprintf(key, sizeof(key), "%s.encoder_attn.v_proj.weight", prefix);
    LOAD(layer->cross_v_weight, key, 1);
    snprintf(key, sizeof(key), "%s.encoder_attn.v_proj.bias", prefix);
    LOAD(layer->cross_v_bias, key, 0);

    snprintf(key, sizeof(key), "%s.encoder_attn.out_proj.weight", prefix);
    LOAD(layer->cross_o_weight, key, 1);
    snprintf(key, sizeof(key), "%s.encoder_attn.out_proj.bias", prefix);
    LOAD(layer->cross_o_bias, key, 0);

    snprintf(key, sizeof(key), "%s.encoder_attn_layer_norm.weight", prefix);
    LOAD(layer->cross_ln_weight, key, 0);
    snprintf(key, sizeof(key), "%s.encoder_attn_layer_norm.bias", prefix);
    LOAD(layer->cross_ln_bias, key, 0);

    snprintf(key, sizeof(key), "%s.fc1.weight", prefix);
    LOAD(layer->fc1_weight, key, 1);
    snprintf(key, sizeof(key), "%s.fc1.bias", prefix);
    LOAD(layer->fc1_bias, key, 0);

    snprintf(key, sizeof(key), "%s.fc2.weight", prefix);
    LOAD(layer->fc2_weight, key, 1);
    snprintf(key, sizeof(key), "%s.fc2.bias", prefix);
    LOAD(layer->fc2_bias, key, 0);

    snprintf(key, sizeof(key), "%s.final_layer_norm.weight", prefix);
    LOAD(layer->ffn_ln_weight, key, 0);
    snprintf(key, sizeof(key), "%s.final_layer_norm.bias", prefix);
    LOAD(layer->ffn_ln_bias, key, 0);

    return 1;
}

// ---------------------------------------------------------------------------
// Main loader
// ---------------------------------------------------------------------------

#define NUM_DECODER_LAYERS 10
#define SWIN_DEPTHS {2, 2, 14, 2}
#define SWIN_HEADS {4, 8, 16, 32}

nougat_model_t* nougat_model_create(const char* model_dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/model.safetensors", model_dir);

    safetensors_t st_data;
    safetensors_t* st = &st_data;
    if (!safetensors_open(st, path)) {
        fprintf(stderr, "[Nougat] Failed to open %s\n", path);
        return NULL;
    }

    nougat_model_t* model = (nougat_model_t*)calloc(1, sizeof(nougat_model_t));
    if (!model) {
        safetensors_close(st);
        return NULL;
    }

    // ---- Config ----
    model->num_decoder_layers = NUM_DECODER_LAYERS;

    int swin_depths[4] = SWIN_DEPTHS;
    int swin_heads[4] = SWIN_HEADS;
    memcpy(model->swin_config.depths, swin_depths, sizeof(swin_depths));
    memcpy(model->swin_config.num_heads, swin_heads, sizeof(swin_heads));
    model->swin_config.embed_dim = 128;
    model->swin_config.window_size = 7;
    model->swin_config.patch_size = 4;
    model->swin_config.num_channels = 3;
    model->swin_config.mlp_ratio = 4.0f;
    model->swin_config.qkv_bias = true;
    model->swin_config.layer_norm_eps = 1e-5f;

    model->decoder_config.d_model = 1024;
    model->decoder_config.num_heads = 16;
    model->decoder_config.d_ff = 4096;
    model->decoder_config.layer_norm_eps = 1e-5f;
    model->decoder_config.pre_norm = true;
    model->decoder_config.activation = "gelu";

    // ---- Swin Encoder allocation ----
    model->encoder = (boat_swin_weights_t*)calloc(1, sizeof(boat_swin_weights_t));
    if (!model->encoder) {
        nougat_model_free(model);
        safetensors_close(st);
        return NULL;
    }

    // ---- Patch Embed ----
    boat_swin_patch_embed_weights_t* pe = &model->encoder->patch_embed;
    LOAD(pe->proj_weight, "encoder.embeddings.patch_embeddings.projection.weight", 0);
    LOAD(pe->proj_bias, "encoder.embeddings.patch_embeddings.projection.bias", 0);
    LOAD(pe->norm_weight, "encoder.embeddings.norm.weight", 0);
    LOAD(pe->norm_bias, "encoder.embeddings.norm.bias", 0);

    // ---- Encoder Stages ----
    for (int s = 0; s < 4; s++) {
        int dim = model->swin_config.embed_dim * (1 << s); // 128,256,512,1024
        int heads = swin_heads[s];
        int num_blocks = swin_depths[s];

        model->encoder->stages[s].blocks =
            (boat_swin_block_weights_t*)calloc(num_blocks, sizeof(boat_swin_block_weights_t));

        for (int b = 0; b < num_blocks; b++) {
            char prefix[256];
            snprintf(prefix, sizeof(prefix), "encoder.encoder.layers.%d.blocks.%d", s, b);
            if (!load_swin_block(st, model, &model->encoder->stages[s].blocks[b], prefix, dim,
                                 heads, model->swin_config.window_size)) {
                nougat_model_free(model);
                safetensors_close(st);
                return NULL;
            }
        }

        // Downsample (stages 0-2 only)
        if (s < 3) {
            model->encoder->stages[s].downsample =
                (boat_swin_downsample_weights_t*)calloc(1, sizeof(boat_swin_downsample_weights_t));

            char prefix[256];
            snprintf(prefix, sizeof(prefix), "encoder.encoder.layers.%d.downsample", s);
            char key[512];

            snprintf(key, sizeof(key), "%s.norm.weight", prefix);
            LOAD(model->encoder->stages[s].downsample->norm_weight, key, 0);
            snprintf(key, sizeof(key), "%s.norm.bias", prefix);
            LOAD(model->encoder->stages[s].downsample->norm_bias, key, 0);
            snprintf(key, sizeof(key), "%s.reduction.weight", prefix);
            LOAD(model->encoder->stages[s].downsample->reduction_weight, key, 1);
            // DonutSwin has no reduction_bias
            model->encoder->stages[s].downsample->reduction_bias = NULL;
        } else {
            model->encoder->stages[s].downsample = NULL;
        }
    }

    // ---- Decoder Layers ----
    model->decoder_layers = (boat_decoder_layer_weights_t**)calloc(
        NUM_DECODER_LAYERS, sizeof(boat_decoder_layer_weights_t*));

    for (int l = 0; l < NUM_DECODER_LAYERS; l++) {
        model->decoder_layers[l] =
            (boat_decoder_layer_weights_t*)calloc(1, sizeof(boat_decoder_layer_weights_t));
        char prefix[256];
        snprintf(prefix, sizeof(prefix), "decoder.model.decoder.layers.%d", l);
        if (!load_decoder_layer(st, model, model->decoder_layers[l], prefix)) {
            nougat_model_free(model);
            safetensors_close(st);
            return NULL;
        }
    }

    // ---- Decoder Embedding + Final ----
    LOAD(model->embed_tokens_weight, "decoder.model.decoder.embed_tokens.weight", 0);
    LOAD(model->embed_positions_weight, "decoder.model.decoder.embed_positions.weight", 0);
    LOAD(model->layernorm_embedding_weight, "decoder.model.decoder.layernorm_embedding.weight", 0);
    LOAD(model->layernorm_embedding_bias, "decoder.model.decoder.layernorm_embedding.bias", 0);
    LOAD(model->final_layer_norm_weight, "decoder.model.decoder.layer_norm.weight", 0);
    LOAD(model->final_layer_norm_bias, "decoder.model.decoder.layer_norm.bias", 0);
    LOAD(model->lm_head_weight, "decoder.lm_head.weight", 1);

    safetensors_close(st);
    fprintf(stderr, "[Nougat] Loaded model from %s/model.safetensors\n", model_dir);
    return model;
}

#undef LOAD
#undef LOAD_OPT

// ---------------------------------------------------------------------------
// Free helpers
// ---------------------------------------------------------------------------
static void free_swin_block(boat_swin_block_weights_t* block) {
    if (!block) return;
    if (block->query_weight) boat_tensor_unref(block->query_weight);
    if (block->query_bias) boat_tensor_unref(block->query_bias);
    if (block->key_weight) boat_tensor_unref(block->key_weight);
    if (block->key_bias) boat_tensor_unref(block->key_bias);
    if (block->value_weight) boat_tensor_unref(block->value_weight);
    if (block->value_bias) boat_tensor_unref(block->value_bias);
    if (block->proj_weight) boat_tensor_unref(block->proj_weight);
    if (block->proj_bias) boat_tensor_unref(block->proj_bias);
    if (block->norm1_weight) boat_tensor_unref(block->norm1_weight);
    if (block->norm1_bias) boat_tensor_unref(block->norm1_bias);
    if (block->norm2_weight) boat_tensor_unref(block->norm2_weight);
    if (block->norm2_bias) boat_tensor_unref(block->norm2_bias);
    if (block->mlp_fc1_weight) boat_tensor_unref(block->mlp_fc1_weight);
    if (block->mlp_fc1_bias) boat_tensor_unref(block->mlp_fc1_bias);
    if (block->mlp_fc2_weight) boat_tensor_unref(block->mlp_fc2_weight);
    if (block->mlp_fc2_bias) boat_tensor_unref(block->mlp_fc2_bias);
    if (block->rel_pos_bias_table) boat_tensor_unref(block->rel_pos_bias_table);
    if (block->rel_pos_index) boat_tensor_unref(block->rel_pos_index);
}

static void free_decoder_layer(boat_decoder_layer_weights_t* layer) {
    if (!layer) return;
    if (layer->self_q_weight) boat_tensor_unref(layer->self_q_weight);
    if (layer->self_q_bias) boat_tensor_unref(layer->self_q_bias);
    if (layer->self_k_weight) boat_tensor_unref(layer->self_k_weight);
    if (layer->self_k_bias) boat_tensor_unref(layer->self_k_bias);
    if (layer->self_v_weight) boat_tensor_unref(layer->self_v_weight);
    if (layer->self_v_bias) boat_tensor_unref(layer->self_v_bias);
    if (layer->self_o_weight) boat_tensor_unref(layer->self_o_weight);
    if (layer->self_o_bias) boat_tensor_unref(layer->self_o_bias);
    if (layer->self_ln_weight) boat_tensor_unref(layer->self_ln_weight);
    if (layer->self_ln_bias) boat_tensor_unref(layer->self_ln_bias);
    if (layer->cross_q_weight) boat_tensor_unref(layer->cross_q_weight);
    if (layer->cross_q_bias) boat_tensor_unref(layer->cross_q_bias);
    if (layer->cross_k_weight) boat_tensor_unref(layer->cross_k_weight);
    if (layer->cross_k_bias) boat_tensor_unref(layer->cross_k_bias);
    if (layer->cross_v_weight) boat_tensor_unref(layer->cross_v_weight);
    if (layer->cross_v_bias) boat_tensor_unref(layer->cross_v_bias);
    if (layer->cross_o_weight) boat_tensor_unref(layer->cross_o_weight);
    if (layer->cross_o_bias) boat_tensor_unref(layer->cross_o_bias);
    if (layer->cross_ln_weight) boat_tensor_unref(layer->cross_ln_weight);
    if (layer->cross_ln_bias) boat_tensor_unref(layer->cross_ln_bias);
    if (layer->fc1_weight) boat_tensor_unref(layer->fc1_weight);
    if (layer->fc1_bias) boat_tensor_unref(layer->fc1_bias);
    if (layer->fc2_weight) boat_tensor_unref(layer->fc2_weight);
    if (layer->fc2_bias) boat_tensor_unref(layer->fc2_bias);
    if (layer->ffn_ln_weight) boat_tensor_unref(layer->ffn_ln_weight);
    if (layer->ffn_ln_bias) boat_tensor_unref(layer->ffn_ln_bias);
}

void nougat_model_free(nougat_model_t* model) {
    if (!model) return;

    // Free encoder
    if (model->encoder) {
        // Free patch embed
        boat_swin_patch_embed_weights_t* pe = &model->encoder->patch_embed;
        if (pe->proj_weight) boat_tensor_unref(pe->proj_weight);
        if (pe->proj_bias) boat_tensor_unref(pe->proj_bias);
        if (pe->norm_weight) boat_tensor_unref(pe->norm_weight);
        if (pe->norm_bias) boat_tensor_unref(pe->norm_bias);

        // Free encoder stages
        for (int s = 0; s < 4; s++) {
            for (int b = 0; b < model->swin_config.depths[s]; b++) {
                free_swin_block(&model->encoder->stages[s].blocks[b]);
            }
            free(model->encoder->stages[s].blocks);
            if (model->encoder->stages[s].downsample) {
                boat_swin_downsample_weights_t* ds = model->encoder->stages[s].downsample;
                if (ds->norm_weight) boat_tensor_unref(ds->norm_weight);
                if (ds->norm_bias) boat_tensor_unref(ds->norm_bias);
                if (ds->reduction_weight) boat_tensor_unref(ds->reduction_weight);
                if (ds->reduction_bias) boat_tensor_unref(ds->reduction_bias);
                free(ds);
            }
        }
        free(model->encoder);
    }

    // Free decoder layers
    for (int l = 0; l < model->num_decoder_layers; l++) {
        if (model->decoder_layers[l]) {
            free_decoder_layer(model->decoder_layers[l]);
            free(model->decoder_layers[l]);
        }
    }
    free(model->decoder_layers);

    // Free embedding/final tensors
    if (model->embed_tokens_weight) boat_tensor_unref(model->embed_tokens_weight);
    if (model->embed_positions_weight) boat_tensor_unref(model->embed_positions_weight);
    if (model->layernorm_embedding_weight) boat_tensor_unref(model->layernorm_embedding_weight);
    if (model->layernorm_embedding_bias) boat_tensor_unref(model->layernorm_embedding_bias);
    if (model->final_layer_norm_weight) boat_tensor_unref(model->final_layer_norm_weight);
    if (model->final_layer_norm_bias) boat_tensor_unref(model->final_layer_norm_bias);
    if (model->lm_head_weight) boat_tensor_unref(model->lm_head_weight);

    free(model);
}

// ---------------------------------------------------------------------------
// Device transfer
// ---------------------------------------------------------------------------
int nougat_model_to_device(nougat_model_t* model, boat_device_t device) {
    if (!model || device == BOAT_DEVICE_CPU) return 1;

    boat_tensor_t* t;

    // Helper macro to transfer a tensor
#define TO_DEVICE(tptr)                                                                            \
    do {                                                                                           \
        if (*(tptr)) {                                                                             \
            t = boat_tensor_to_device(*(tptr), device);                                            \
            if (!t) {                                                                              \
                fprintf(stderr, "[Nougat] Failed to transfer tensor to device\n");                 \
                return 0;                                                                          \
            }                                                                                      \
            boat_tensor_unref(*(tptr));                                                            \
            *(tptr) = t;                                                                           \
        }                                                                                          \
    } while (0)

    // Patch embed
    TO_DEVICE(&model->encoder->patch_embed.proj_weight);
    TO_DEVICE(&model->encoder->patch_embed.proj_bias);
    TO_DEVICE(&model->encoder->patch_embed.norm_weight);
    TO_DEVICE(&model->encoder->patch_embed.norm_bias);

    // Encoder stages
    for (int s = 0; s < 4; s++) {
        for (int b = 0; b < model->swin_config.depths[s]; b++) {
            boat_swin_block_weights_t* blk = &model->encoder->stages[s].blocks[b];
            TO_DEVICE(&blk->query_weight);
            TO_DEVICE(&blk->query_bias);
            TO_DEVICE(&blk->key_weight);
            TO_DEVICE(&blk->key_bias);
            TO_DEVICE(&blk->value_weight);
            TO_DEVICE(&blk->value_bias);
            TO_DEVICE(&blk->proj_weight);
            TO_DEVICE(&blk->proj_bias);
            TO_DEVICE(&blk->norm1_weight);
            TO_DEVICE(&blk->norm1_bias);
            TO_DEVICE(&blk->norm2_weight);
            TO_DEVICE(&blk->norm2_bias);
            TO_DEVICE(&blk->mlp_fc1_weight);
            TO_DEVICE(&blk->mlp_fc1_bias);
            TO_DEVICE(&blk->mlp_fc2_weight);
            TO_DEVICE(&blk->mlp_fc2_bias);
            TO_DEVICE(&blk->rel_pos_bias_table);
            // rel_pos_index stays on CPU (used as index lookup)
        }
        if (model->encoder->stages[s].downsample) {
            boat_swin_downsample_weights_t* ds = model->encoder->stages[s].downsample;
            TO_DEVICE(&ds->norm_weight);
            TO_DEVICE(&ds->norm_bias);
            TO_DEVICE(&ds->reduction_weight);
            TO_DEVICE(&ds->reduction_bias);
        }
    }

    // Decoder layers
    for (int l = 0; l < model->num_decoder_layers; l++) {
        boat_decoder_layer_weights_t* layer = model->decoder_layers[l];
        TO_DEVICE(&layer->self_q_weight);
        TO_DEVICE(&layer->self_q_bias);
        TO_DEVICE(&layer->self_k_weight);
        TO_DEVICE(&layer->self_k_bias);
        TO_DEVICE(&layer->self_v_weight);
        TO_DEVICE(&layer->self_v_bias);
        TO_DEVICE(&layer->self_o_weight);
        TO_DEVICE(&layer->self_o_bias);
        TO_DEVICE(&layer->self_ln_weight);
        TO_DEVICE(&layer->self_ln_bias);
        TO_DEVICE(&layer->cross_q_weight);
        TO_DEVICE(&layer->cross_q_bias);
        TO_DEVICE(&layer->cross_k_weight);
        TO_DEVICE(&layer->cross_k_bias);
        TO_DEVICE(&layer->cross_v_weight);
        TO_DEVICE(&layer->cross_v_bias);
        TO_DEVICE(&layer->cross_o_weight);
        TO_DEVICE(&layer->cross_o_bias);
        TO_DEVICE(&layer->cross_ln_weight);
        TO_DEVICE(&layer->cross_ln_bias);
        TO_DEVICE(&layer->fc1_weight);
        TO_DEVICE(&layer->fc1_bias);
        TO_DEVICE(&layer->fc2_weight);
        TO_DEVICE(&layer->fc2_bias);
        TO_DEVICE(&layer->ffn_ln_weight);
        TO_DEVICE(&layer->ffn_ln_bias);
    }

    // Embedding / final
    TO_DEVICE(&model->embed_tokens_weight);
    TO_DEVICE(&model->embed_positions_weight);
    TO_DEVICE(&model->layernorm_embedding_weight);
    TO_DEVICE(&model->layernorm_embedding_bias);
    TO_DEVICE(&model->final_layer_norm_weight);
    TO_DEVICE(&model->final_layer_norm_bias);
    TO_DEVICE(&model->lm_head_weight);

#undef TO_DEVICE

    fprintf(stderr, "[Nougat] Moved weights to device %d\n", (int)device);
    return 1;
}
