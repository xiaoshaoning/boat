// config.h - MiniMind model hyperparameters
// Generated from minimind-3 config (768 hidden, 8 layers, 8 heads, GQA 2:1)
#pragma once

#define MINIMIND_VOCAB_SIZE 6400
#define MINIMIND_HIDDEN_SIZE 768
#define MINIMIND_NUM_HEADS 8
#define MINIMIND_NUM_KV_HEADS 4
#define MINIMIND_HEAD_DIM 96
#define MINIMIND_NUM_LAYERS 8
#define MINIMIND_INTERMEDIATE_SIZE 2432
#define MINIMIND_MAX_SEQ_LEN 2048
#define MINIMIND_ROPE_THETA 1000000.0f
#define MINIMIND_RMS_EPS 1e-6f
#define MINIMIND_EOS_TOKEN_ID 2
#define MINIMIND_BOS_TOKEN_ID 1

typedef struct {
    int vocab_size;
    int hidden_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int num_layers;
    int intermediate_size;
    int max_seq_len;
    float rope_theta;
    float rms_eps;
} minimind_config_t;

static inline minimind_config_t minimind_default_config(void) {
    minimind_config_t c;
    c.vocab_size = MINIMIND_VOCAB_SIZE;
    c.hidden_size = MINIMIND_HIDDEN_SIZE;
    c.num_heads = MINIMIND_NUM_HEADS;
    c.num_kv_heads = MINIMIND_NUM_KV_HEADS;
    c.head_dim = MINIMIND_HEAD_DIM;
    c.num_layers = MINIMIND_NUM_LAYERS;
    c.intermediate_size = MINIMIND_INTERMEDIATE_SIZE;
    c.max_seq_len = MINIMIND_MAX_SEQ_LEN;
    c.rope_theta = MINIMIND_ROPE_THETA;
    c.rms_eps = MINIMIND_RMS_EPS;
    return c;
}
