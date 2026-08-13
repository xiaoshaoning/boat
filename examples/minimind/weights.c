// weights.c - MiniMind weight loader
// Reads model.bin + model_meta.json, transposes 2D weights,
// precomputes RoPE tables, allocates KV caches and working buffers.
#include "weights.h"
#include "../common/json.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

// --- File I/O helpers ---

static char* read_file(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open: %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)malloc(sz + 1);
    if (!buf) { fclose(f); return NULL; }
    size_t nread = fread(buf, 1, sz, f);
    fclose(f);
    buf[nread] = '\0';
    *out_len = nread;
    return buf;
}

// --- Weight tensor metadata ---
typedef struct {
    char* name;
    int offset;
    int ndim;
    int shape[4];
} wt_entry_t;

#define MAX_WEIGHTS 128

// --- JSON helpers ---

static int json_parse_array_int(json_ctx_t* ctx, int* out, int max_n) {
    if (json_next(ctx) != '[') return -1;
    int n = 0;
    while (n < max_n) {
        json_skip_ws(ctx);
        if (json_peek(ctx) == ']') { ctx->pos++; break; }
        out[n++] = (int)json_parse_int(ctx);
        json_skip_ws(ctx);
        if (json_peek(ctx) == ',') ctx->pos++;
    }
    return n;
}

static int parse_meta(const char* json_str, size_t len, wt_entry_t* entries, int* n_out) {
    json_ctx_t ctx;
    json_init(&ctx, json_str, len);
    // Expect: { "name": { "offset": N, "shape": [...] }, ... }
    if (json_next(&ctx) != '{') return -1;

    int num = 0;
    while (num < MAX_WEIGHTS) {
        json_skip_ws(&ctx);
        if (json_peek(&ctx) == '}') { ctx.pos++; break; }
        if (json_peek(&ctx) == ',') ctx.pos++;

        char* name = json_parse_string(&ctx);
        if (!name) break;
        if (json_next(&ctx) != ':') { free(name); break; }
        if (json_next(&ctx) != '{') { free(name); break; }

        wt_entry_t* e = &entries[num];
        e->name = name;
        e->offset = 0;
        e->ndim = 0;

        while (1) {
            json_skip_ws(&ctx);
            if (json_peek(&ctx) == '}') { ctx.pos++; break; }
            if (json_peek(&ctx) == ',') ctx.pos++;
            char* key = json_parse_string(&ctx);
            if (!key) { free(name); break; }
            json_expect(&ctx, ':');

            if (strcmp(key, "offset") == 0) {
                e->offset = (int)json_parse_int(&ctx);
            } else if (strcmp(key, "shape") == 0) {
                e->ndim = json_parse_array_int(&ctx, e->shape, 4);
            }
            free(key);
        }
        num++;
    }
    *n_out = num;
    return 0;
}

static wt_entry_t* find_entry(wt_entry_t* entries, int n, const char* name) {
    for (int i = 0; i < n; i++) {
        if (strcmp(entries[i].name, name) == 0) return &entries[i];
    }
    return NULL;
}

// --- Transpose and copy helper ---
// PyTorch weights are [out, in]; we transpose to [in, out] for our C matmul.
// For 1D tensors (norms): just copy.
static void load_weight_transpose(float** dst, const float* src, int dim0, int dim1) {
    float* w = (float*)malloc(dim0 * dim1 * sizeof(float));
    for (int i = 0; i < dim0; i++) {
        for (int j = 0; j < dim1; j++) {
            w[j * dim0 + i] = src[i * dim1 + j];
        }
    }
    *dst = w;
}

static void load_weight_copy(float** dst, const float* src, int n) {
    float* w = (float*)malloc(n * sizeof(float));
    memcpy(w, src, n * sizeof(float));
    *dst = w;
}

// --- RoPE table precomputation ---
static void precompute_rope(float* cos_table, float* sin_table,
                             int max_seq_len, int head_dim, float theta) {
    int half_dim = head_dim / 2;
    float* freqs = (float*)malloc(half_dim * sizeof(float));
    // Python: freqs = 1.0 / (theta ** (arange(0, dim, 2)[:dim//2].float() / dim))
    for (int i = 0; i < half_dim; i++) {
        freqs[i] = 1.0f / powf(theta, 2.0f * i / (float)head_dim);
    }
    // Python: freqs = outer(arange(max_seq_len), freqs) -> [max_seq_len, half_dim]
    // Then: cos = cat([cos(freqs), cos(freqs)], dim=-1) -> [max_seq_len, head_dim]
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float angle = (float)pos * freqs[i];
            float c = cosf(angle);
            float s = sinf(angle);
            cos_table[pos * head_dim + i] = c;
            cos_table[pos * head_dim + i + half_dim] = c;
            sin_table[pos * head_dim + i] = s;
            sin_table[pos * head_dim + i + half_dim] = s;
        }
    }
    free(freqs);
}

// --- Main loader ---
int minimind_weights_load(minimind_model_t* m, const char* model_dir) {
    memset(m, 0, sizeof(*m));
    minimind_config_t cfg = minimind_default_config();
    m->config = cfg;
    m->max_seq_len = cfg.max_seq_len;

    // 1. Read metadata JSON
    char meta_path[512];
    snprintf(meta_path, sizeof(meta_path), "%s/model_meta.json", model_dir);
    size_t json_len;
    char* json_str = read_file(meta_path, &json_len);
    if (!json_str) return -1;

    wt_entry_t entries[MAX_WEIGHTS];
    int n_entries = 0;
    if (parse_meta(json_str, json_len, entries, &n_entries) != 0) {
        fprintf(stderr, "Failed to parse metadata\n");
        free(json_str);
        return -1;
    }
    free(json_str);

    // 2. Read binary weight data
    char bin_path[512];
    snprintf(bin_path, sizeof(bin_path), "%s/model.bin", model_dir);
    size_t bin_len;
    char* bin_data_char = read_file(bin_path, &bin_len);
    if (!bin_data_char) {
        for (int i = 0; i < n_entries; i++) free(entries[i].name);
        return -1;
    }
    const float* bin_data = (const float*)bin_data_char;
    m->block_data = bin_data_char;
    m->block_size = bin_len;

    // 3. Helper to get tensor data
#define GET_TENSOR(name) find_entry(entries, n_entries, name)
// Load a 2D weight with transpose (for linear layers: PyTorch [out,in] -> C [in,out])
#define LOAD_WEIGHT_T(dst, name) do { \
    wt_entry_t* _e = GET_TENSOR(name); \
    if (!_e) { fprintf(stderr, "Missing: %s\n", name); return -1; } \
    int _d0 = _e->shape[0], _d1 = (_e->ndim > 1) ? _e->shape[1] : 0; \
    if (_e->ndim == 2 && _d0 > 1 && _d1 > 1) \
        load_weight_transpose(&(dst), bin_data + _e->offset / 4, _d0, _d1); \
    else \
        load_weight_copy(&(dst), bin_data + _e->offset / 4, _d0); \
} while(0)
// Load without transpose (for embeddings: keep [vocab_size, hidden_size] layout)
#define LOAD_WEIGHT(dst, name) do { \
    wt_entry_t* _e = GET_TENSOR(name); \
    if (!_e) { fprintf(stderr, "Missing: %s\n", name); return -1; } \
    int _d0 = _e->shape[0], _d1 = (_e->ndim > 1) ? _e->shape[1] : 0; \
    if (_e->ndim == 2 && _d0 > 1 && _d1 > 1) \
        load_weight_copy(&(dst), bin_data + _e->offset / 4, _d0 * _d1); \
    else \
        load_weight_copy(&(dst), bin_data + _e->offset / 4, _d0); \
} while(0)

    // 4. Load weights by name
    // Embeddings: NO transpose (lookup table, [vocab_size, hidden_size])
    LOAD_WEIGHT(m->embed_tokens, "model.embed_tokens.weight");
    m->lm_head = m->embed_tokens; // weight tying

    char name[256];
    for (int l = 0; l < cfg.num_layers; l++) {
        // Linear projections: transpose from PyTorch [out,in] to C [in,out]
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_proj.weight", l);
        LOAD_WEIGHT_T(m->q_proj[l], name);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_proj.weight", l);
        LOAD_WEIGHT_T(m->k_proj[l], name);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.v_proj.weight", l);
        LOAD_WEIGHT_T(m->v_proj[l], name);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.o_proj.weight", l);
        LOAD_WEIGHT_T(m->o_proj[l], name);

        // Norm weights: 1D, no transpose needed
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.q_norm.weight", l);
        LOAD_WEIGHT(m->q_norm_weight[l], name);
        snprintf(name, sizeof(name), "model.layers.%d.self_attn.k_norm.weight", l);
        LOAD_WEIGHT(m->k_norm_weight[l], name);

        snprintf(name, sizeof(name), "model.layers.%d.input_layernorm.weight", l);
        LOAD_WEIGHT(m->input_layernorm_weight[l], name);
        snprintf(name, sizeof(name), "model.layers.%d.post_attention_layernorm.weight", l);
        LOAD_WEIGHT(m->post_attention_layernorm_weight[l], name);

        snprintf(name, sizeof(name), "model.layers.%d.mlp.gate_proj.weight", l);
        LOAD_WEIGHT_T(m->gate_proj[l], name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.down_proj.weight", l);
        LOAD_WEIGHT_T(m->down_proj[l], name);
        snprintf(name, sizeof(name), "model.layers.%d.mlp.up_proj.weight", l);
        LOAD_WEIGHT_T(m->up_proj[l], name);
    }
    LOAD_WEIGHT(m->final_norm_weight, "model.norm.weight");

    // Skip lm_head.weight since it's weight-tied (already loaded as embed_tokens)

    // 5. Precompute RoPE tables
    int rope_sz = cfg.max_seq_len * cfg.head_dim;
    m->cos_table = (float*)malloc(rope_sz * sizeof(float));
    m->sin_table = (float*)malloc(rope_sz * sizeof(float));
    precompute_rope(m->cos_table, m->sin_table, cfg.max_seq_len, cfg.head_dim, cfg.rope_theta);

    // 6. Allocate KV caches
    int kv_dim = cfg.num_kv_heads * cfg.head_dim; // 4 * 96 = 384
    size_t kv_per_layer = (size_t)cfg.max_seq_len * kv_dim;
    for (int l = 0; l < cfg.num_layers; l++) {
        m->k_cache[l] = (float*)calloc(kv_per_layer, sizeof(float));
        m->v_cache[l] = (float*)calloc(kv_per_layer, sizeof(float));
    }
    m->kv_len = 0;

    // 7. Allocate working buffers
    size_t hidden_sz  = (size_t)cfg.max_seq_len * cfg.hidden_size;
    size_t q_sz       = (size_t)cfg.max_seq_len * cfg.num_heads * cfg.head_dim;
    size_t kv_sz      = (size_t)cfg.max_seq_len * cfg.num_kv_heads * cfg.head_dim;
    size_t ffn_sz     = (size_t)cfg.max_seq_len * cfg.intermediate_size;

    m->hidden   = (float*)calloc(hidden_sz, sizeof(float));
    m->hidden2  = (float*)calloc(hidden_sz, sizeof(float));
    m->q_buf    = (float*)calloc(q_sz, sizeof(float));
    m->k_buf    = (float*)calloc(kv_sz, sizeof(float));
    m->v_buf    = (float*)calloc(kv_sz, sizeof(float));
    m->attn_out = (float*)calloc(hidden_sz, sizeof(float));
    m->ffn_gate = (float*)calloc(ffn_sz, sizeof(float));
    m->ffn_up   = (float*)calloc(ffn_sz, sizeof(float));

    // Cleanup metadata entries
    for (int i = 0; i < n_entries; i++) free(entries[i].name);

    printf("MiniMind weights loaded: %d layers, %.1f MB weights\n",
           cfg.num_layers, (double)bin_len / (1024.0 * 1024.0));
    return 0;
#undef GET_TENSOR
#undef LOAD_WEIGHT
}

void minimind_weights_free(minimind_model_t* m) {
    free(m->block_data);
    free(m->cos_table);
    free(m->sin_table);
    for (int l = 0; l < MINIMIND_NUM_LAYERS; l++) {
        free(m->k_cache[l]);
        free(m->v_cache[l]);
        // Free individual weight arrays
        free(m->q_proj[l]); free(m->k_proj[l]); free(m->v_proj[l]); free(m->o_proj[l]);
        free(m->q_norm_weight[l]); free(m->k_norm_weight[l]);
        free(m->input_layernorm_weight[l]); free(m->post_attention_layernorm_weight[l]);
        free(m->gate_proj[l]); free(m->down_proj[l]); free(m->up_proj[l]);
    }
    free(m->embed_tokens);  // frees both embed_tokens and lm_head (same pointer)
    free(m->final_norm_weight);
    free(m->hidden); free(m->hidden2);
    free(m->q_buf); free(m->k_buf); free(m->v_buf);
    free(m->attn_out);
    free(m->ffn_gate); free(m->ffn_up);
}
