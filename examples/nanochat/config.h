// config.h - NanoChat model configuration
#pragma once

#define NANOCHAT_VOCAB_SIZE      65536
#define NANOCHAT_HIDDEN_SIZE     2176
#define NANOCHAT_NUM_HEADS       17
#define NANOCHAT_NUM_KV_HEADS    17  // MHA (n_kv == n_head)
#define NANOCHAT_HEAD_DIM        128 // hidden_size / num_heads
#define NANOCHAT_NUM_LAYERS      34
#define NANOCHAT_INTERMEDIATE_SIZE 8704
#define NANOCHAT_MAX_SEQ_LEN     2048
#define NANOCHAT_ROPE_THETA      10000.0f
#define NANOCHAT_RMS_EPS         1e-6f
#define NANOCHAT_SOFTCAP         15.0f

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
    float softcap;
} nanochat_config_t;

static inline nanochat_config_t nanochat_get_config(void) {
    nanochat_config_t c;
    c.vocab_size = NANOCHAT_VOCAB_SIZE;
    c.hidden_size = NANOCHAT_HIDDEN_SIZE;
    c.num_heads = NANOCHAT_NUM_HEADS;
    c.num_kv_heads = NANOCHAT_NUM_KV_HEADS;
    c.head_dim = NANOCHAT_HEAD_DIM;
    c.num_layers = NANOCHAT_NUM_LAYERS;
    c.intermediate_size = NANOCHAT_INTERMEDIATE_SIZE;
    c.max_seq_len = NANOCHAT_MAX_SEQ_LEN;
    c.rope_theta = NANOCHAT_ROPE_THETA;
    c.rms_eps = NANOCHAT_RMS_EPS;
    c.softcap = NANOCHAT_SOFTCAP;
    return c;
}
