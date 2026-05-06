// transformer.c - Character-level Transformer for text generation
// Demonstrates building, training, and running inference with a transformer
// using the Boat deep learning framework.
//
// Architecture:
//   Embedding + PositionalEncoding -> N x TransformerBlock -> Dense -> Softmax
//   Each block: Self-Attention -> residual -> LayerNorm -> FFN -> residual -> LayerNorm
//
// Data: character-level next-token prediction on embedded text corpus

#include <boat.h>
#include <boat/tensor.h>
#include <boat/layers.h>
#include <boat/layers/attention.h>
#include <boat/ops.h>
#include <boat/optimizers.h>
#include <boat/loss.h>
#include <boat/memory.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>

// ============================================================
// Hyperparameters
// ============================================================
#define D_MODEL      64
#define N_HEADS      4
#define N_LAYERS     3
#define D_FF         256
#define MAX_SEQ_LEN  48
#define BATCH_SIZE   8
#define N_EPOCHS     60
#define LR           0.003f
#define VOCAB_SIZE   55

// Special token IDs
#define TOK_PAD 0
#define TOK_SOS 1
#define TOK_EOS 2
#define TOK_UNK 3
#define TOK_START 4  // First real character token

// ============================================================
// Character Vocabulary
// ============================================================
// Token layout: 0=PAD, 1=SOS, 2=EOS, 3=UNK, 4..4+36=chars
static const char VOCAB_CHARS[] = "abcdefghijklmnopqrstuvwxyz .,!?'-:;\n";
#define NUM_VOCAB_CHARS ((int)(sizeof(VOCAB_CHARS) - 1))

static int char_to_id(char c) {
    if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
    for (int i = 0; i < NUM_VOCAB_CHARS; i++) {
        if (VOCAB_CHARS[i] == c) return TOK_START + i;
    }
    return TOK_UNK;
}

static char id_to_char(int id) {
    if (id >= TOK_START && id < TOK_START + NUM_VOCAB_CHARS)
        return VOCAB_CHARS[id - TOK_START];
    return '?';
}

// ============================================================
// Embedded Training Corpus
// ============================================================
static const char* TRAINING_SENTENCES[] = {
    "the quick brown fox jumps over the lazy dog",
    "machine learning is transforming the world",
    "deep neural networks can learn complex patterns",
    "transformers are powerful models for sequence tasks",
    "attention allows models to focus on important parts",
    "the sun rises in the east and sets in the west",
    "natural language processing enables computers to understand text",
    "artificial intelligence is the future of technology",
    "python and c are programming languages",
    "data is the new oil in the modern economy",
};
#define NUM_SENTENCES ((int)(sizeof(TRAINING_SENTENCES) / sizeof(TRAINING_SENTENCES[0])))

static int build_corpus(int* corpus, int max_len) {
    int pos = 0;
    for (int i = 0; i < NUM_SENTENCES && pos < max_len - 2; i++) {
        if (pos > 0) corpus[pos++] = ' ';
        const char* s = TRAINING_SENTENCES[i];
        while (*s && pos < max_len - 1) {
            corpus[pos++] = char_to_id(*s);
            s++;
        }
        corpus[pos++] = '.'; // end of sentence
    }
    corpus[pos] = TOK_EOS;
    return pos + 1;
}

static void create_training_batch(
    int* input_buf, int* target_buf,
    const int* corpus, int corpus_len, int batch_size, int seq_len, int epoch) {
    srand(epoch + 42);
    int max_start = corpus_len - seq_len - 1;
    if (max_start < 1) max_start = 1;

    for (int b = 0; b < batch_size; b++) {
        int start = rand() % max_start;
        for (int t = 0; t < seq_len; t++) {
            int idx = b * seq_len + t;
            input_buf[idx] = (start + t < corpus_len) ? corpus[start + t] : TOK_PAD;
            target_buf[idx] = (start + t + 1 < corpus_len) ? corpus[start + t + 1] : TOK_PAD;
        }
    }
}

// ============================================================
// Sinusoidal Positional Encoding
// ============================================================
static boat_tensor_t* create_positional_encoding(int max_seq_len, int d_model) {
    int64_t shape[] = {max_seq_len, d_model};
    boat_tensor_t* pe = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!pe) return NULL;

    float* data = (float*)boat_tensor_data(pe);
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < d_model; i++) {
            float angle = (float)pos / powf(10000.0f, (2.0f * (float)(i / 2)) / (float)d_model);
            if (i % 2 == 0)
                data[pos * d_model + i] = sinf(angle);
            else
                data[pos * d_model + i] = cosf(angle);
        }
    }
    return pe;
}

// ============================================================
// Layer Normalization (manual, since framework norm needs more work)
// ============================================================
typedef struct {
    boat_tensor_t* gamma;
    boat_tensor_t* beta;
    boat_tensor_t* grad_gamma;
    boat_tensor_t* grad_beta;
    float eps;
    boat_tensor_t* cache_input;
    boat_tensor_t* cache_mean;
    boat_tensor_t* cache_var;
} manual_ln_t;

static manual_ln_t* manual_ln_create(int normalized_shape, float eps) {
    manual_ln_t* ln = (manual_ln_t*)boat_malloc(sizeof(manual_ln_t), BOAT_DEVICE_CPU);
    if (!ln) return NULL;

    int64_t shape[] = {normalized_shape};
    ln->gamma = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    ln->beta  = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    ln->grad_gamma = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    ln->grad_beta  = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

    float* gd = (float*)boat_tensor_data(ln->gamma);
    float* bd = (float*)boat_tensor_data(ln->beta);
    for (int i = 0; i < normalized_shape; i++) {
        gd[i] = 1.0f;
        bd[i] = 0.0f;
    }

    ln->eps = eps;
    ln->cache_input = NULL;
    ln->cache_mean = NULL;
    ln->cache_var = NULL;
    return ln;
}

static void manual_ln_free(manual_ln_t* ln) {
    if (!ln) return;
    if (ln->gamma) boat_tensor_unref(ln->gamma);
    if (ln->beta) boat_tensor_unref(ln->beta);
    if (ln->grad_gamma) boat_tensor_unref(ln->grad_gamma);
    if (ln->grad_beta) boat_tensor_unref(ln->grad_beta);
    if (ln->cache_input) boat_tensor_unref(ln->cache_input);
    if (ln->cache_mean) boat_tensor_unref(ln->cache_mean);
    if (ln->cache_var) boat_tensor_unref(ln->cache_var);
    boat_free(ln);
}

static boat_tensor_t* manual_ln_forward(manual_ln_t* ln, const boat_tensor_t* input) {
    if (!ln || !input) return NULL;

    int64_t ndim = boat_tensor_ndim(input);
    const int64_t* shape = boat_tensor_shape(input);
    int64_t d_model = shape[ndim - 1];

    boat_tensor_t* output = boat_tensor_create_like(input);
    if (!output) return NULL;

    int64_t outer = 1;
    for (int64_t i = 0; i < ndim - 1; i++) outer *= shape[i];

    size_t n = (size_t)d_model;
    const float* x = (const float*)boat_tensor_const_data(input);
    float* y = (float*)boat_tensor_data(output);
    const float* g = (const float*)boat_tensor_const_data(ln->gamma);
    const float* b = (const float*)boat_tensor_const_data(ln->beta);
    float eps = ln->eps;

    float* mean = (float*)boat_malloc((size_t)outer * sizeof(float), BOAT_DEVICE_CPU);
    float* var = (float*)boat_malloc((size_t)outer * sizeof(float), BOAT_DEVICE_CPU);

    for (int64_t i = 0; i < outer; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < n; j++) sum += x[i * n + j];
        float m = sum / (float)n;

        float vsum = 0.0f;
        for (size_t j = 0; j < n; j++) {
            float diff = x[i * n + j] - m;
            vsum += diff * diff;
        }
        float v = vsum / (float)n;

        mean[i] = m;
        var[i] = v;

        float inv_std = 1.0f / sqrtf(v + eps);
        for (size_t j = 0; j < n; j++) {
            y[i * n + j] = (x[i * n + j] - m) * inv_std * g[j] + b[j];
        }
    }

    if (ln->cache_input) boat_tensor_unref(ln->cache_input);
    ln->cache_input = boat_tensor_create_like(input);
    memcpy(boat_tensor_data(ln->cache_input), x, boat_tensor_nbytes(input));

    if (ln->cache_mean) boat_tensor_unref(ln->cache_mean);
    ln->cache_mean = boat_tensor_from_data((int64_t[]){outer}, 1, BOAT_DTYPE_FLOAT32, mean);

    if (ln->cache_var) boat_tensor_unref(ln->cache_var);
    ln->cache_var = boat_tensor_from_data((int64_t[]){outer}, 1, BOAT_DTYPE_FLOAT32, var);

    boat_free(mean);
    boat_free(var);
    return output;
}

static boat_tensor_t* manual_ln_backward(manual_ln_t* ln, const boat_tensor_t* grad_output) {
    if (!ln || !grad_output || !ln->cache_input) return NULL;

    const boat_tensor_t* input = ln->cache_input;
    int64_t ndim = boat_tensor_ndim(input);
    const int64_t* shape = boat_tensor_shape(input);
    int64_t d_model = shape[ndim - 1];

    int64_t outer = 1;
    for (int64_t i = 0; i < ndim - 1; i++) outer *= shape[i];
    size_t n = (size_t)d_model;

    const float* x = (const float*)boat_tensor_const_data(input);
    const float* mean = (const float*)boat_tensor_const_data(ln->cache_mean);
    const float* var = (const float*)boat_tensor_const_data(ln->cache_var);
    const float* grad_out = (const float*)boat_tensor_const_data(grad_output);
    const float* g = (const float*)boat_tensor_const_data(ln->gamma);
    float eps = ln->eps;

    boat_tensor_t* grad_input = boat_tensor_create_like(input);
    float* dx = (float*)boat_tensor_data(grad_input);

    float* dg = (float*)boat_tensor_data(ln->grad_gamma);
    float* db = (float*)boat_tensor_data(ln->grad_beta);

    memset(dg, 0, n * sizeof(float));
    memset(db, 0, n * sizeof(float));

    for (int64_t i = 0; i < outer; i++) {
        float m = mean[i];
        float v = var[i];
        float inv_std = 1.0f / sqrtf(v + eps);

        float dx_hat[256];
        if (n > 256) return NULL;

        for (size_t j = 0; j < n; j++) {
            dx_hat[j] = grad_out[i * n + j] * g[j];
            dg[j] += dx_hat[j] * (x[i * n + j] - m) * inv_std;
            db[j] += grad_out[i * n + j];
        }

        float sum_dx_hat = 0.0f;
        float sum_dx_hat_x_hat = 0.0f;
        for (size_t j = 0; j < n; j++) {
            float x_hat = (x[i * n + j] - m) * inv_std;
            sum_dx_hat += dx_hat[j];
            sum_dx_hat_x_hat += dx_hat[j] * x_hat;
        }
        float inv_n = 1.0f / (float)n;
        for (size_t j = 0; j < n; j++) {
            float x_hat = (x[i * n + j] - m) * inv_std;
            dx[i * n + j] = inv_std * (dx_hat[j] - inv_n * (sum_dx_hat + x_hat * sum_dx_hat_x_hat));
        }
    }

    return grad_input;
}

static void manual_ln_zero_grad(manual_ln_t* ln) {
    if (!ln) return;
    memset(boat_tensor_data(ln->grad_gamma), 0, boat_tensor_nbytes(ln->grad_gamma));
    memset(boat_tensor_data(ln->grad_beta), 0, boat_tensor_nbytes(ln->grad_beta));
}

// ============================================================
// Forward pass for a single transformer block
// ============================================================
// Pre-LN architecture:
//   x1 = x + attention(layernorm(x))
//   x2 = x1 + ffn(layernorm(x1))

static boat_tensor_t* forward_block(
    boat_tensor_t* input,
    manual_ln_t* ln1, manual_ln_t* ln2,
    boat_attention_layer_t* attn,
    boat_dense_layer_t* ffn1, boat_dense_layer_t* ffn2,
    boat_relu_layer_t* relu,
    int batch_size, int seq_len, int d_model, int d_ff,
    boat_tensor_t** out_attn, boat_tensor_t** out_ffn,
    boat_tensor_t** ln1_out, boat_tensor_t** ln2_out,
    boat_tensor_t** residual1_out, boat_tensor_t** relu_out) {

    *ln1_out = manual_ln_forward(ln1, input);
    if (!*ln1_out) return NULL;

    *out_attn = boat_attention_layer_forward(attn, *ln1_out, *ln1_out, *ln1_out, NULL);
    if (!*out_attn) return NULL;

    *residual1_out = boat_add(input, *out_attn);
    if (!*residual1_out) return NULL;

    *ln2_out = manual_ln_forward(ln2, *residual1_out);
    if (!*ln2_out) return NULL;

    int64_t r2d[] = {batch_size * seq_len, d_model};
    boat_tensor_t* l2_2d = boat_tensor_reshape(*ln2_out, r2d, 2);
    if (!l2_2d) return NULL;

    boat_tensor_t* f1 = boat_dense_layer_forward(ffn1, l2_2d);
    boat_tensor_unref(l2_2d);
    if (!f1) return NULL;

    *relu_out = boat_relu_layer_forward(relu, f1);
    boat_tensor_unref(f1);
    if (!*relu_out) return NULL;

    int64_t rff[] = {batch_size * seq_len, d_ff};
    boat_tensor_t* r_2d = boat_tensor_reshape(*relu_out, rff, 2);
    if (!r_2d) return NULL;

    boat_tensor_t* f2 = boat_dense_layer_forward(ffn2, r_2d);
    boat_tensor_unref(r_2d);
    if (!f2) return NULL;

    int64_t r3d[] = {batch_size, seq_len, d_model};
    *out_ffn = boat_tensor_reshape(f2, r3d, 3);
    boat_tensor_unref(f2);
    if (!*out_ffn) return NULL;

    boat_tensor_t* output = boat_add(*residual1_out, *out_ffn);
    if (!output) return NULL;

    return output;
}

// ============================================================
// Backward pass for a single transformer block
// ============================================================
static boat_tensor_t* backward_block(
    boat_tensor_t* grad_output,
    manual_ln_t* ln1, manual_ln_t* ln2,
    boat_attention_layer_t* attn,
    boat_dense_layer_t* ffn1, boat_dense_layer_t* ffn2,
    boat_relu_layer_t* relu,
    int batch_size, int seq_len, int d_model, int d_ff,
    boat_tensor_t* cached_input, boat_tensor_t* cached_attn_out,
    boat_tensor_t* cached_ffn_out, boat_tensor_t* cached_ln1_out,
    boat_tensor_t* cached_ln2_out, boat_tensor_t* cached_residual1,
    boat_tensor_t* cached_relu_out) {

    (void)cached_input; (void)cached_attn_out; (void)cached_ffn_out;
    (void)cached_ln1_out; (void)cached_ln2_out;
    (void)cached_residual1; (void)cached_relu_out;

    // Core backward flow:
    //   output = residual1 + ffn(layernorm2(residual1))
    //   residual1 = input + attn(layernorm1(input))
    //
    // Strategy: compute grad through FFN, merge at second residual,
    // then continue through attn, merge at first residual.

    int64_t bsd[] = {batch_size * seq_len, d_model};
    int64_t bsf[] = {batch_size * seq_len, d_ff};
    int64_t b3d[] = {batch_size, seq_len, d_model};

    // --- Backward through FFN (need 2D reshape for dense) ---
    boat_tensor_t* g2d = boat_tensor_reshape(grad_output, bsd, 2);
    if (!g2d) return NULL;

    boat_tensor_t* g_relu = boat_dense_layer_backward(ffn2, g2d);
    boat_tensor_unref(g2d);
    if (!g_relu) return NULL;

    boat_tensor_t* g_f1 = boat_relu_layer_backward(relu, g_relu);
    boat_tensor_unref(g_relu);
    if (!g_f1) return NULL;

    boat_tensor_t* g_l2_2d = boat_dense_layer_backward(ffn1, g_f1);
    boat_tensor_unref(g_f1);
    if (!g_l2_2d) return NULL;

    // Reshape FFN gradient back to 3D
    boat_tensor_t* g_ffn = boat_tensor_reshape(g_l2_2d, b3d, 3);
    boat_tensor_unref(g_l2_2d);
    if (!g_ffn) return NULL;

    // Second residual merge: grad through FFN + identity path
    boat_tensor_t* g_res1 = boat_add(g_ffn, grad_output);
    boat_tensor_unref(g_ffn);
    if (!g_res1) return NULL;

    // --- Backward through LayerNorm2 ---
    boat_tensor_t* g_attn = manual_ln_backward(ln2, g_res1);
    if (!g_attn) { boat_tensor_unref(g_res1); return NULL; }

    // --- Backward through self-attention ---
    boat_tensor_t* g_ln1 = boat_attention_layer_backward(attn, g_attn);
    boat_tensor_unref(g_attn);
    if (!g_ln1) { boat_tensor_unref(g_res1); return NULL; }

    // --- Backward through LayerNorm1 ---
    boat_tensor_t* g_attn_path = manual_ln_backward(ln1, g_ln1);
    boat_tensor_unref(g_ln1);
    if (!g_attn_path) { boat_tensor_unref(g_res1); return NULL; }

    // First residual merge: input = (input) + attn_path + (input) identity
    // g_res1 = gradient w.r.t. residual1 = gradient through identity path of first residual
    // g_attn_path = gradient through ln1 -> attn path
    // So grad_input = g_attn_path + g_res1
    boat_tensor_t* grad_input = boat_add(g_attn_path, g_res1);
    boat_tensor_unref(g_attn_path);
    boat_tensor_unref(g_res1);
    if (!grad_input) return NULL;

    return grad_input;
}

// ============================================================
// Main
// ============================================================
int main(void) {
    setbuf(stdout, NULL);
    srand((unsigned)time(NULL));

    int d_model = D_MODEL;
    int d_ff = D_FF;
    int n_layers = N_LAYERS;
    int n_heads = N_HEADS;
    int batch_size = BATCH_SIZE;
    int seq_len = MAX_SEQ_LEN;
    int vocab_size = VOCAB_SIZE;
    int n_epochs = N_EPOCHS;
    float lr = LR;

    printf("=== Transformer Character-Level Language Model ===\n\n");
    printf("Hyperparameters:\n");
    printf("  d_model=%d, n_heads=%d, n_layers=%d, d_ff=%d\n", d_model, n_heads, n_layers, d_ff);
    printf("  vocab=%d, seq_len=%d, batch=%d, lr=%.4f, epochs=%d\n\n",
           vocab_size, seq_len, batch_size, lr, n_epochs);

    // ---- Build corpus ----
    int corpus[4096];
    int corpus_len = build_corpus(corpus, 4096);
    printf("Corpus: %d tokens from %d sentences\n\n", corpus_len, NUM_SENTENCES);

    // ---- Create model components ----

    // Positional encoding
    boat_tensor_t* pos_encoding = create_positional_encoding(seq_len, d_model);
    if (!pos_encoding) { fprintf(stderr, "Failed to create positional encoding\n"); return 1; }

    // Embedding weight
    int64_t emb_shape[] = {vocab_size, d_model};
    boat_tensor_t* embed_weight = boat_tensor_create(emb_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* embed_grad = boat_tensor_create(emb_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

    float* ew = (float*)boat_tensor_data(embed_weight);
    for (int i = 0; i < vocab_size * d_model; i++) ew[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    // Transformer blocks
    manual_ln_t** ln1 = (manual_ln_t**)boat_malloc(n_layers * sizeof(manual_ln_t*), BOAT_DEVICE_CPU);
    manual_ln_t** ln2 = (manual_ln_t**)boat_malloc(n_layers * sizeof(manual_ln_t*), BOAT_DEVICE_CPU);
    boat_attention_layer_t** attn_layers = (boat_attention_layer_t**)boat_malloc(n_layers * sizeof(boat_attention_layer_t*), BOAT_DEVICE_CPU);
    boat_dense_layer_t** ffn1_layers = (boat_dense_layer_t**)boat_malloc(n_layers * sizeof(boat_dense_layer_t*), BOAT_DEVICE_CPU);
    boat_dense_layer_t** ffn2_layers = (boat_dense_layer_t**)boat_malloc(n_layers * sizeof(boat_dense_layer_t*), BOAT_DEVICE_CPU);
    boat_relu_layer_t** relu_layers = (boat_relu_layer_t**)boat_malloc(n_layers * sizeof(boat_relu_layer_t*), BOAT_DEVICE_CPU);

    for (int i = 0; i < n_layers; i++) {
        ln1[i] = manual_ln_create(d_model, 1e-5f);
        ln2[i] = manual_ln_create(d_model, 1e-5f);
        attn_layers[i] = boat_attention_layer_create((size_t)d_model, (size_t)n_heads, (size_t)n_heads, 0.0f, true);
        ffn1_layers[i] = boat_dense_layer_create((size_t)d_model, (size_t)d_ff, true);
        ffn2_layers[i] = boat_dense_layer_create((size_t)d_ff, (size_t)d_model, true);
        relu_layers[i] = boat_relu_layer_create();
        if (!ln1[i] || !ln2[i] || !attn_layers[i] || !ffn1_layers[i] || !ffn2_layers[i] || !relu_layers[i]) {
            fprintf(stderr, "Failed to create layer %d\n", i);
            return 1;
        }
    }

    // Output projection
    boat_dense_layer_t* out_proj = boat_dense_layer_create((size_t)d_model, (size_t)vocab_size, true);
    if (!out_proj) { fprintf(stderr, "Failed to create output projection\n"); return 1; }

    // ---- Adam Optimizer ----
    boat_optimizer_t* optimizer = boat_adam_optimizer_create(lr, 0.9f, 0.999f, 1e-8f);
    if (!optimizer) { fprintf(stderr, "Failed to create optimizer\n"); return 1; }

    // Register embedding
    boat_optimizer_add_parameter(optimizer, embed_weight, embed_grad);

    // Register attention parameters (8 per layer: 4 weights + 4 biases)
    for (int i = 0; i < n_layers; i++) {
        boat_attention_t* attn = (boat_attention_t*)attn_layers[i];
        struct { boat_tensor_t* w; boat_tensor_t* g; } ap[] = {
            {boat_attention_get_weight_q(attn), boat_attention_get_grad_weight_q(attn)},
            {boat_attention_get_weight_k(attn), boat_attention_get_grad_weight_k(attn)},
            {boat_attention_get_weight_v(attn), boat_attention_get_grad_weight_v(attn)},
            {boat_attention_get_weight_o(attn), boat_attention_get_grad_weight_o(attn)},
            {boat_attention_get_bias_q(attn), boat_attention_get_grad_bias_q(attn)},
            {boat_attention_get_bias_k(attn), boat_attention_get_grad_bias_k(attn)},
            {boat_attention_get_bias_v(attn), boat_attention_get_grad_bias_v(attn)},
            {boat_attention_get_bias_o(attn), boat_attention_get_grad_bias_o(attn)},
        };
        for (int j = 0; j < 8; j++) {
            if (ap[j].w && ap[j].g) boat_optimizer_add_parameter(optimizer, ap[j].w, ap[j].g);
        }
    }

    // Register output projection
    boat_tensor_t* ow = boat_dense_layer_get_weight(out_proj);
    boat_tensor_t* ob = boat_dense_layer_get_bias(out_proj);
    boat_tensor_t* ogw = boat_dense_layer_get_grad_weight(out_proj);
    boat_tensor_t* ogb = boat_dense_layer_get_grad_bias(out_proj);
    if (ow && ogw) boat_optimizer_add_parameter(optimizer, ow, ogw);
    if (ob && ogb) boat_optimizer_add_parameter(optimizer, ob, ogb);

    // Register FFN parameters
    for (int i = 0; i < n_layers; i++) {
        boat_tensor_t* w1 = boat_dense_layer_get_weight(ffn1_layers[i]);
        boat_tensor_t* b1 = boat_dense_layer_get_bias(ffn1_layers[i]);
        boat_tensor_t* gw1 = boat_dense_layer_get_grad_weight(ffn1_layers[i]);
        boat_tensor_t* gb1 = boat_dense_layer_get_grad_bias(ffn1_layers[i]);
        if (w1 && gw1) boat_optimizer_add_parameter(optimizer, w1, gw1);
        if (b1 && gb1) boat_optimizer_add_parameter(optimizer, b1, gb1);

        boat_tensor_t* w2 = boat_dense_layer_get_weight(ffn2_layers[i]);
        boat_tensor_t* b2 = boat_dense_layer_get_bias(ffn2_layers[i]);
        boat_tensor_t* gw2 = boat_dense_layer_get_grad_weight(ffn2_layers[i]);
        boat_tensor_t* gb2 = boat_dense_layer_get_grad_bias(ffn2_layers[i]);
        if (w2 && gw2) boat_optimizer_add_parameter(optimizer, w2, gw2);
        if (b2 && gb2) boat_optimizer_add_parameter(optimizer, b2, gb2);
    }

    printf("All parameters registered with Adam optimizer.\n\n");

    // ---- Training Loop ----
    int input_buf[BATCH_SIZE * MAX_SEQ_LEN];
    int target_buf[BATCH_SIZE * MAX_SEQ_LEN];

    boat_loss_t* loss_fn = boat_cross_entropy_loss_create();
    int64_t target_shape[] = {BATCH_SIZE, MAX_SEQ_LEN, VOCAB_SIZE};
    boat_tensor_t* targets_onehot = boat_tensor_create(target_shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

    printf("Starting training...\n\n");

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        create_training_batch(input_buf, target_buf, corpus, corpus_len, batch_size, seq_len, epoch);

        float* toh = (float*)boat_tensor_data(targets_onehot);
        memset(toh, 0, boat_tensor_nbytes(targets_onehot));
        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < seq_len; t++) {
                int id = target_buf[b * seq_len + t];
                if (id >= 0 && id < vocab_size)
                    toh[(b * seq_len + t) * vocab_size + id] = 1.0f;
            }
        }

        // ---- Forward pass ----

        int64_t eshape[] = {batch_size, seq_len, d_model};
        boat_tensor_t* x = boat_tensor_create(eshape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (!x) { fprintf(stderr, "Failed to create embedding output\n"); return 1; }
        float* xd = (float*)boat_tensor_data(x);
        const float* ed = (const float*)boat_tensor_const_data(embed_weight);

        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < seq_len; t++) {
                int id = input_buf[b * seq_len + t];
                if (id < 0 || id >= vocab_size) id = TOK_UNK;
                memcpy(&xd[(b * seq_len + t) * d_model], &ed[id * d_model], (size_t)d_model * sizeof(float));
            }
        }

        const float* pe = (const float*)boat_tensor_const_data(pos_encoding);
        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < seq_len; t++) {
                for (int d = 0; d < d_model; d++) {
                    xd[(b * seq_len + t) * d_model + d] += pe[t * d_model + d];
                }
            }
        }

        // Transformer blocks forward
        boat_tensor_t* block_inputs[16];
        boat_tensor_t* attn_outs[16];
        boat_tensor_t* ffn_outs[16];
        boat_tensor_t* ln1_outs[16];
        boat_tensor_t* ln2_outs[16];
        boat_tensor_t* residual1_outs[16];
        boat_tensor_t* relu_outs[16];

        boat_tensor_t* block_out = x;
        boat_tensor_ref(block_out);

        for (int i = 0; i < n_layers; i++) {
            block_inputs[i] = block_out;

            boat_tensor_t* out = forward_block(
                block_out, ln1[i], ln2[i], attn_layers[i],
                ffn1_layers[i], ffn2_layers[i], relu_layers[i],
                batch_size, seq_len, d_model, d_ff,
                &attn_outs[i], &ffn_outs[i],
                &ln1_outs[i], &ln2_outs[i],
                &residual1_outs[i], &relu_outs[i]);

            if (!out) { fprintf(stderr, "Block %d forward failed\n", i); return 1; }
            block_out = out;
        }

        // Output projection
        int64_t o2d[] = {batch_size * seq_len, d_model};
        int64_t o3d[] = {batch_size, seq_len, d_model};
        int64_t l2d[] = {batch_size * seq_len, vocab_size};
        int64_t l3d[] = {batch_size, seq_len, vocab_size};

        boat_tensor_t* bo_2d = boat_tensor_reshape(block_out, o2d, 2);
        if (!bo_2d) return 1;

        boat_tensor_t* logits_2d = boat_dense_layer_forward(out_proj, bo_2d);
        boat_tensor_unref(bo_2d);
        if (!logits_2d) return 1;

        boat_tensor_t* logits = boat_tensor_reshape(logits_2d, l3d, 3);
        boat_tensor_unref(logits_2d);
        if (!logits) return 1;

        // Softmax + cross-entropy gradient
        boat_tensor_t* probs = boat_softmax(logits, 2);
        if (!probs) return 1;

        float loss = boat_loss_compute(loss_fn, probs, targets_onehot);

        // Gradient at logit level: (probs - targets) / (B*S)
        boat_tensor_t* grad_logits = boat_sub(probs, targets_onehot);
        float scale = 1.0f / (float)(batch_size * seq_len);
        boat_mul_scalar_(grad_logits, scale);

        boat_tensor_unref(probs);

        // ---- Backward pass ----

        // Output projection backward
        boat_tensor_t* gl_2d = boat_tensor_reshape(grad_logits, l2d, 2);
        boat_tensor_unref(grad_logits);
        if (!gl_2d) return 1;

        boat_tensor_t* go_2d = boat_dense_layer_backward(out_proj, gl_2d);
        boat_tensor_unref(gl_2d);
        if (!go_2d) return 1;

        boat_tensor_t* grad = boat_tensor_reshape(go_2d, o3d, 3);
        boat_tensor_unref(go_2d);
        if (!grad) return 1;

        // Backward through each block (reverse order)
        for (int i = n_layers - 1; i >= 0; i--) {
            boat_tensor_t* new_grad = backward_block(
                grad, ln1[i], ln2[i], attn_layers[i],
                ffn1_layers[i], ffn2_layers[i], relu_layers[i],
                batch_size, seq_len, d_model, d_ff,
                block_inputs[i], attn_outs[i], ffn_outs[i],
                ln1_outs[i], ln2_outs[i], residual1_outs[i], relu_outs[i]);

            boat_tensor_unref(grad);
            if (!new_grad) { fprintf(stderr, "Block %d backward failed\n", i); return 1; }
            grad = new_grad;
        }

        // Embedding backward
        float* gd = (float*)boat_tensor_data(embed_grad);
        memset(gd, 0, boat_tensor_nbytes(embed_grad));
        for (int b = 0; b < batch_size; b++) {
            for (int t = 0; t < seq_len; t++) {
                int id = input_buf[b * seq_len + t];
                if (id < 0 || id >= vocab_size) id = TOK_UNK;
                for (int d = 0; d < d_model; d++) {
                    gd[id * d_model + d] += ((float*)boat_tensor_data(grad))[(b * seq_len + t) * d_model + d];
                }
            }
        }
        boat_tensor_unref(grad);

        // ---- Optimizer Step + Zero Grad ----
        boat_optimizer_step(optimizer);
        boat_optimizer_zero_grad(optimizer);
        for (int i = 0; i < n_layers; i++) {
            manual_ln_zero_grad(ln1[i]);
            manual_ln_zero_grad(ln2[i]);
        }

        // Free cached forward tensors
        for (int i = 0; i < n_layers; i++) {
            if (attn_outs[i]) boat_tensor_unref(attn_outs[i]);
            if (ffn_outs[i]) boat_tensor_unref(ffn_outs[i]);
            if (ln1_outs[i]) boat_tensor_unref(ln1_outs[i]);
            if (ln2_outs[i]) boat_tensor_unref(ln2_outs[i]);
            if (residual1_outs[i]) boat_tensor_unref(residual1_outs[i]);
            if (relu_outs[i]) boat_tensor_unref(relu_outs[i]);
            if (block_inputs[i]) boat_tensor_unref(block_inputs[i]);
        }
        if (block_out) boat_tensor_unref(block_out);

        if (epoch % 5 == 0 || epoch == n_epochs - 1) {
            printf("Epoch %4d/%d  Loss: %.6f  Perplexity: %.2f\n",
                   epoch + 1, n_epochs, loss, expf(loss));
        }
    }

    printf("\nTraining complete!\n\n");

    // ---- Inference: Generate text ----
    printf("=== Inference: Generating text ===\n\n");

    const float* inf_ed = (const float*)boat_tensor_const_data(embed_weight);
    const float* inf_pe = (const float*)boat_tensor_const_data(pos_encoding);

    int gen_seq[MAX_SEQ_LEN];
    int gen_len = 1;
    gen_seq[0] = '.';

    for (int step = 0; step < 80; step++) {
        int input_ids[MAX_SEQ_LEN];
        int pad = seq_len - gen_len;
        if (pad < 0) { pad = 0; gen_len = seq_len; }
        for (int i = 0; i < pad; i++) input_ids[i] = TOK_PAD;
        for (int i = 0; i < gen_len; i++) input_ids[pad + i] = gen_seq[i];

        int64_t ishape[] = {1, seq_len, d_model};
        boat_tensor_t* inf_x = boat_tensor_create(ishape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        float* ixd = (float*)boat_tensor_data(inf_x);
        for (int t = 0; t < seq_len; t++) {
            int id = input_ids[t];
            if (id < 0 || id >= vocab_size) id = TOK_UNK;
            memcpy(&ixd[t * d_model], &inf_ed[id * d_model], (size_t)d_model * sizeof(float));
            for (int d = 0; d < d_model; d++)
                ixd[t * d_model + d] += inf_pe[t * d_model + d];
        }

        // Inference forward through all blocks
        for (int i = 0; i < n_layers; i++) {
            boat_tensor_t* il1 = manual_ln_forward(ln1[i], inf_x);
            if (!il1) break;

            boat_tensor_t* ia = boat_attention_layer_forward(attn_layers[i], il1, il1, il1, NULL);
            boat_tensor_unref(il1);
            if (!ia) break;

            boat_tensor_t* ir1 = boat_add(inf_x, ia);
            boat_tensor_unref(ia);
            if (!ir1) break;

            boat_tensor_t* il2 = manual_ln_forward(ln2[i], ir1);
            if (!il2) { boat_tensor_unref(ir1); break; }

            int64_t i2d[] = {seq_len, d_model};
            boat_tensor_t* il2_2d = boat_tensor_reshape(il2, i2d, 2);
            boat_tensor_unref(il2);
            if (!il2_2d) { boat_tensor_unref(ir1); break; }

            boat_tensor_t* if1 = boat_dense_layer_forward(ffn1_layers[i], il2_2d);
            boat_tensor_unref(il2_2d);
            if (!if1) { boat_tensor_unref(ir1); break; }

            boat_tensor_t* ir = boat_relu_layer_forward(relu_layers[i], if1);
            boat_tensor_unref(if1);
            if (!ir) { boat_tensor_unref(ir1); break; }

            int64_t ird[] = {seq_len, d_ff};
            boat_tensor_t* ir_2d = boat_tensor_reshape(ir, ird, 2);
            boat_tensor_unref(ir);
            if (!ir_2d) { boat_tensor_unref(ir1); break; }

            boat_tensor_t* if2 = boat_dense_layer_forward(ffn2_layers[i], ir_2d);
            boat_tensor_unref(ir_2d);
            if (!if2) { boat_tensor_unref(ir1); break; }

            int64_t i3d[] = {1, seq_len, d_model};
            boat_tensor_t* ivo = boat_tensor_reshape(if2, i3d, 3);
            boat_tensor_unref(if2);
            if (!ivo) { boat_tensor_unref(ir1); break; }

            boat_tensor_t* io = boat_add(ir1, ivo);
            boat_tensor_unref(ivo);
            boat_tensor_unref(ir1);
            boat_tensor_unref(inf_x);
            inf_x = io;
        }

        // Output projection
        int64_t of2d[] = {seq_len, d_model};
        boat_tensor_t* ix2d = boat_tensor_reshape(inf_x, of2d, 2);
        if (!ix2d) break;

        boat_tensor_t* il2d = boat_dense_layer_forward(out_proj, ix2d);
        boat_tensor_unref(ix2d);
        if (!il2d) break;

        int64_t of3d[] = {1, seq_len, vocab_size};
        boat_tensor_t* il3d = boat_tensor_reshape(il2d, of3d, 3);
        boat_tensor_unref(il2d);
        if (!il3d) break;

        boat_tensor_t* ip = boat_softmax(il3d, 2);
        boat_tensor_unref(il3d);
        if (!ip) break;

        // Sample with temperature
        float temp = 0.8f;
        int last_pos = (gen_len < seq_len) ? gen_len + pad - 1 : seq_len - 1;
        const float* pd = (const float*)boat_tensor_const_data(ip);
        const float* lp = &pd[last_pos * vocab_size];

        float ps[VOCAB_SIZE], sum_ps = 0.0f;
        for (int i = TOK_START; i < vocab_size; i++) {
            ps[i] = powf(fmaxf(lp[i], 1e-10f), 1.0f / temp);
            sum_ps += ps[i];
        }
        if (sum_ps > 0) {
            float r = (float)rand() / (float)RAND_MAX * sum_ps;
            float cum = 0.0f;
            for (int i = TOK_START; i < vocab_size; i++) {
                cum += ps[i];
                if (r <= cum) {
                    if (gen_len < MAX_SEQ_LEN) gen_seq[gen_len++] = i;
                    break;
                }
            }
        }

        boat_tensor_unref(ip);
        boat_tensor_unref(inf_x);
    }

    printf("Generated: ");
    for (int i = 0; i < gen_len; i++) putchar(id_to_char(gen_seq[i]));
    printf("\n\n");

    // ---- Cleanup ----
    boat_tensor_unref(pos_encoding);
    boat_tensor_unref(embed_weight);
    boat_tensor_unref(embed_grad);
    boat_tensor_unref(targets_onehot);

    for (int i = 0; i < n_layers; i++) {
        manual_ln_free(ln1[i]);
        manual_ln_free(ln2[i]);
        boat_attention_layer_free(attn_layers[i]);
        boat_dense_layer_free(ffn1_layers[i]);
        boat_dense_layer_free(ffn2_layers[i]);
        boat_relu_layer_free(relu_layers[i]);
    }

    boat_free(ln1);
    boat_free(ln2);
    boat_free(attn_layers);
    boat_free(ffn1_layers);
    boat_free(ffn2_layers);
    boat_free(relu_layers);

    boat_dense_layer_free(out_proj);
    boat_optimizer_free(optimizer);
    boat_loss_free(loss_fn);

    printf("All done!\n");
    return 0;
}
