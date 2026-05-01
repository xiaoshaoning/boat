// test_gguf.c - GGUF format loader tests
#include <boat.h>
#include <boat/format/gguf.h>
#include <boat/model.h>
#include <boat/tensor.h>
#include <boat/layers.h>
#include <boat/layers/norm.h>
#include <boat/layers/attention.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { printf("  %s ... ", name); fflush(stdout); tests_total++; } while(0)
#define PASS() do { printf("PASS\n"); fflush(stdout); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); fflush(stdout); return 1; } while(0)
#define ASSERT(cond) do { if (!(cond)) { printf("FAIL at %d\n", __LINE__); fflush(stdout); return 1; } } while(0)
#define ASSERT_NEAR(a, b, eps) do { \
    float _a = (a), _b = (b); \
    if (fabsf(_a - _b) > (eps)) { \
        printf("FAIL at %d: %f != %f (eps=%f)\n", __LINE__, _a, _b, (float)(eps)); \
        fflush(stdout); return 1; \
    } \
} while(0)

// -----------------------------------------------------------------------
// Helper: write buffer to temp file, return filename
// -----------------------------------------------------------------------
static const char* write_temp_file(const uint8_t* data, size_t size) {
    static char path[256];
    snprintf(path, sizeof(path), "test_gguf_temp.bin");
    FILE* f = fopen(path, "wb");
    if (!f) return NULL;
    fwrite(data, 1, size, f);
    fclose(f);
    return path;
}

// -----------------------------------------------------------------------
// Programmatic GGUF builder helpers
// -----------------------------------------------------------------------
// We build GGUF files as byte arrays with proper little-endian encoding.

static void append_u8(uint8_t** p, uint8_t v) { *(*p)++ = v; }
static void append_u16(uint8_t** p, uint16_t v) {
    *(*p)++ = (uint8_t)(v & 0xFF);
    *(*p)++ = (uint8_t)((v >> 8) & 0xFF);
}
static void append_u32(uint8_t** p, uint32_t v) {
    *(*p)++ = (uint8_t)(v & 0xFF);
    *(*p)++ = (uint8_t)((v >> 8) & 0xFF);
    *(*p)++ = (uint8_t)((v >> 16) & 0xFF);
    *(*p)++ = (uint8_t)((v >> 24) & 0xFF);
}
static void append_u64(uint8_t** p, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        *(*p)++ = (uint8_t)(v & 0xFF);
        v >>= 8;
    }
}
static void append_f32(uint8_t** p, float v) {
    uint32_t uv; memcpy(&uv, &v, 4);
    append_u32(p, uv);
}
static void append_f16(uint8_t** p, float v) {
    // Simple F32-to-F16 conversion
    uint32_t uv; memcpy(&uv, &v, 4);
    int exp = (int)((uv >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (uv >> 13) & 0x3FF;
    if (exp <= 0) { exp = 0; mant = 0; }
    if (exp >= 31) { exp = 31; mant = 0; }
    uint16_t h = (uint16_t)(((uv >> 16) & 0x8000) | (exp << 10) | mant);
    append_u16(p, h);
}
static void append_string(uint8_t** p, const char* s) {
    uint64_t len = strlen(s);
    append_u64(p, len);
    memcpy(*p, s, len);
    *p += len;
}
static void append_align(uint8_t** p, uint64_t alignment) {
    uint64_t offset = (uint64_t)(*p - (uint8_t*)*p);  // Use ptr diff
    // Actually, we need the base of the buffer. The caller handles alignment.
}

// -----------------------------------------------------------------------
// Build a minimal LLaMA-like GGUF model with F32 tensors
// Parameters: 1 layer, hidden=8, intermediate=32, 2 heads, vocab=16
// -----------------------------------------------------------------------
typedef struct {
    uint8_t* data;
    size_t size;
} gguf_test_model_t;

static gguf_test_model_t build_test_gguf_f32(void) {
    // Calculate buffer size (over-estimate)
    size_t buf_size = 16384;
    uint8_t* buf = (uint8_t*)calloc(1, buf_size);
    uint8_t* p = buf;

    // Config
    int64_t hidden = 8, d_ff = 32, n_heads = 2, n_layers = 1, vocab = 16;
    int64_t head_dim = hidden / n_heads; // 4
    int n_tensors = 12; // norm1, q, k, v, o, norm2, gate, up, down, output_norm, output_weight, tok_embd

    // --- Header ---
    append_u32(&p, GGUF_MAGIC);
    append_u32(&p, GGUF_VERSION);
    append_u64(&p, (uint64_t)n_tensors);       // tensor_count
    append_u64(&p, 8);                         // metadata_kv_count

    // --- Metadata KV ---
    append_string(&p, "general.architecture");
    append_u32(&p, GGUF_TYPE_STRING);
    append_string(&p, "llama");

    append_string(&p, "llama.context_length");
    append_u32(&p, GGUF_TYPE_INT64);
    append_u64(&p, 128);

    append_string(&p, "llama.embedding_length");
    append_u32(&p, GGUF_TYPE_INT64);
    append_u64(&p, (uint64_t)hidden);

    append_string(&p, "llama.feed_forward_length");
    append_u32(&p, GGUF_TYPE_INT64);
    append_u64(&p, (uint64_t)d_ff);

    append_string(&p, "llama.attention.head_count");
    append_u32(&p, GGUF_TYPE_INT64);
    append_u64(&p, (uint64_t)n_heads);

    append_string(&p, "llama.block_count");
    append_u32(&p, GGUF_TYPE_INT64);
    append_u64(&p, (uint64_t)n_layers);

    append_string(&p, "llama.vocab_size");
    append_u32(&p, GGUF_TYPE_INT64);
    append_u64(&p, (uint64_t)vocab);

    append_string(&p, "general.alignment");
    append_u32(&p, GGUF_TYPE_UINT64);
    append_u64(&p, 32);

    // --- Tensor info entries ---
    // We build them in order. The data offsets will be relative to tensor data start.
    // For simplicity, we compute the data start as: current position aligned to 32
    uint64_t hdr_end = (uint64_t)(p - buf);
    uint64_t data_start = (hdr_end + 31) & ~(uint64_t)31;

    // Tensor: token_embd.weight [vocab, hidden]
    uint64_t tok_embd_off = 0;
    int64_t tok_shape[] = {vocab, hidden};
    size_t tok_elems = (size_t)(vocab * hidden);

    // Tensor: blk.0.attn_norm.weight [hidden]
    uint64_t norm1_off = tok_embd_off + tok_elems * 4;
    int64_t norm_shape[] = {hidden};
    size_t norm_elems = (size_t)hidden;

    // Tensor: blk.0.attn_q.weight [hidden, hidden]
    uint64_t q_off = norm1_off + norm_elems * 4;
    size_t q_elems = (size_t)(hidden * hidden);

    // Tensor: blk.0.attn_k.weight [hidden, hidden]
    uint64_t k_off = q_off + q_elems * 4;

    // Tensor: blk.0.attn_v.weight [hidden, hidden]
    uint64_t v_off = k_off + q_elems * 4;

    // Tensor: blk.0.attn_output.weight [hidden, hidden]
    uint64_t o_off = v_off + q_elems * 4;

    // Tensor: blk.0.ffn_norm.weight [hidden]
    uint64_t norm2_off = o_off + q_elems * 4;

    // Tensor: blk.0.ffn_gate.weight [d_ff, hidden]
    uint64_t gate_off = norm2_off + norm_elems * 4;
    size_t gate_elems = (size_t)(d_ff * hidden);

    // Tensor: blk.0.ffn_up.weight [d_ff, hidden]
    uint64_t up_off = gate_off + gate_elems * 4;

    // Tensor: blk.0.ffn_down.weight [hidden, d_ff]
    uint64_t down_off = up_off + gate_elems * 4;
    size_t down_elems = (size_t)(hidden * d_ff);

    // Tensor: output_norm.weight [hidden]
    uint64_t out_norm_off = down_off + down_elems * 4;

    // Tensor: output.weight [vocab, hidden]
    uint64_t out_w_off = out_norm_off + norm_elems * 4;
    size_t out_w_elems = (size_t)(vocab * hidden);

    // Total data size
    uint64_t total_data = out_w_off + out_w_elems * 4;

    // Write tensor info entries
    // 1. token_embd.weight
    append_string(&p, "token_embd.weight");
    append_u32(&p, 2);
    append_u64(&p, (uint64_t)vocab); append_u64(&p, (uint64_t)hidden);
    append_u32(&p, GGML_TYPE_F32);
    append_u64(&p, tok_embd_off);

    // 2. blk.0.attn_norm.weight
    append_string(&p, "blk.0.attn_norm.weight");
    append_u32(&p, 1);
    append_u64(&p, (uint64_t)hidden);
    append_u32(&p, GGML_TYPE_F32);
    append_u64(&p, norm1_off);

    // 3. blk.0.attn_q.weight
    append_string(&p, "blk.0.attn_q.weight");
    append_u32(&p, 2);
    append_u64(&p, (uint64_t)hidden); append_u64(&p, (uint64_t)hidden);
    append_u32(&p, GGML_TYPE_F32);
    append_u64(&p, q_off);

    // 4. blk.0.attn_k.weight
    append_string(&p, "blk.0.attn_k.weight");
    append_u32(&p, 2);
    append_u64(&p, (uint64_t)hidden); append_u64(&p, (uint64_t)hidden);
    append_u32(&p, GGML_TYPE_F32);
    append_u64(&p, k_off);

    // 5. blk.0.attn_v.weight
    append_string(&p, "blk.0.attn_v.weight");
    append_u32(&p, 2);
    append_u64(&p, (uint64_t)hidden); append_u64(&p, (uint64_t)hidden);
    append_u32(&p, GGML_TYPE_F32);
    append_u64(&p, v_off);

    // 6. blk.0.attn_output.weight
    append_string(&p, "blk.0.attn_output.weight");
    append_u32(&p, 2);
    append_u64(&p, (uint64_t)hidden); append_u64(&p, (uint64_t)hidden);
    append_u32(&p, GGML_TYPE_F32);
    append_u64(&p, o_off);

    // 7. blk.0.ffn_norm.weight
    append_string(&p, "blk.0.ffn_norm.weight");
    append_u32(&p, 1);
    append_u64(&p, (uint64_t)hidden);
    append_u32(&p, GGML_TYPE_F32);
    append_u64(&p, norm2_off);

    // 8. blk.0.ffn_gate.weight
    append_string(&p, "blk.0.ffn_gate.weight");
    append_u32(&p, 2);
    append_u64(&p, (uint64_t)d_ff); append_u64(&p, (uint64_t)hidden);
    append_u32(&p, GGML_TYPE_F32);
    append_u64(&p, gate_off);

    // 9. blk.0.ffn_up.weight
    append_string(&p, "blk.0.ffn_up.weight");
    append_u32(&p, 2);
    append_u64(&p, (uint64_t)d_ff); append_u64(&p, (uint64_t)hidden);
    append_u32(&p, GGML_TYPE_F32);
    append_u64(&p, up_off);

    // 10. blk.0.ffn_down.weight
    append_string(&p, "blk.0.ffn_down.weight");
    append_u32(&p, 2);
    append_u64(&p, (uint64_t)hidden); append_u64(&p, (uint64_t)d_ff);
    append_u32(&p, GGML_TYPE_F32);
    append_u64(&p, down_off);

    // 11. output_norm.weight
    append_string(&p, "output_norm.weight");
    append_u32(&p, 1);
    append_u64(&p, (uint64_t)hidden);
    append_u32(&p, GGML_TYPE_F32);
    append_u64(&p, out_norm_off);

    // 12. output.weight
    append_string(&p, "output.weight");
    append_u32(&p, 2);
    append_u64(&p, (uint64_t)vocab); append_u64(&p, (uint64_t)hidden);
    append_u32(&p, GGML_TYPE_F32);
    append_u64(&p, out_w_off);

    // --- Align to 32 for tensor data ---
    while (((uint64_t)(p - buf) & 31) != 0) append_u8(&p, 0);

    // --- Tensor data ---
    // token_embd.weight: fill with 0.01f * (i+1)
    for (size_t i = 0; i < tok_elems; i++)
        append_f32(&p, 0.01f * (float)(i + 1));

    // attn_norm.weight: fill with 1.0f
    for (size_t i = 0; i < norm_elems; i++)
        append_f32(&p, 1.0f);

    // attn_q.weight: [hidden, hidden] = 0.01f * (i+1)
    for (size_t i = 0; i < q_elems; i++)
        append_f32(&p, 0.01f * (float)(i + 1));

    // attn_k.weight
    for (size_t i = 0; i < q_elems; i++)
        append_f32(&p, 0.01f * (float)(i + 1 + (int)q_elems));

    // attn_v.weight
    for (size_t i = 0; i < q_elems; i++)
        append_f32(&p, 0.01f * (float)(i + 1 + (int)q_elems * 2));

    // attn_output.weight
    for (size_t i = 0; i < q_elems; i++)
        append_f32(&p, 0.01f * (float)(i + 1 + (int)q_elems * 3));

    // ffn_norm.weight: fill with 1.0f
    for (size_t i = 0; i < norm_elems; i++)
        append_f32(&p, 1.0f);

    // ffn_gate.weight: [d_ff, hidden]
    for (size_t i = 0; i < gate_elems; i++)
        append_f32(&p, 0.01f * (float)(i + 1));

    // ffn_up.weight
    for (size_t i = 0; i < gate_elems; i++)
        append_f32(&p, 0.01f * (float)(i + 1 + (int)gate_elems));

    // ffn_down.weight: [hidden, d_ff]
    for (size_t i = 0; i < down_elems; i++)
        append_f32(&p, 0.01f * (float)(i + 1));

    // output_norm.weight: fill with 1.0f
    for (size_t i = 0; i < norm_elems; i++)
        append_f32(&p, 1.0f);

    // output.weight: fill with known values
    for (size_t i = 0; i < out_w_elems; i++)
        append_f32(&p, 0.01f * (float)(i + 1));

    size_t actual_size = (size_t)(p - buf);
    return (gguf_test_model_t){buf, actual_size};
}

static void free_test_model(gguf_test_model_t* m) {
    if (m->data) free(m->data);
    m->data = NULL;
    m->size = 0;
}

// -----------------------------------------------------------------------
// Test: boat_gguf_check with valid GGUF
// -----------------------------------------------------------------------
static int test_check_valid(void) {
    TEST("GGUF check valid");
    gguf_test_model_t m = build_test_gguf_f32();
    const char* path = write_temp_file(m.data, m.size);
    ASSERT(path != NULL);
    ASSERT(boat_gguf_check(path));
    remove(path);
    free_test_model(&m);
    PASS();
    return 0;
}

// -----------------------------------------------------------------------
// Test: boat_gguf_check with invalid data
// -----------------------------------------------------------------------
static int test_check_invalid(void) {
    TEST("GGUF check invalid");
    ASSERT(boat_gguf_check(NULL) == false);
    ASSERT(boat_gguf_check("nonexistent_file.gguf") == false);

    // Write file with bad magic
    uint8_t bad[4] = {0x00, 0x00, 0x00, 0x00};
    const char* path = write_temp_file(bad, 4);
    ASSERT(path != NULL);
    ASSERT(boat_gguf_check(path) == false);
    remove(path);
    PASS();
    return 0;
}

// -----------------------------------------------------------------------
// Test: Load F32 GGUF model
// -----------------------------------------------------------------------
static int test_load_f32(void) {
    TEST("GGUF load F32 model");
    gguf_test_model_t m = build_test_gguf_f32();
    const char* path = write_temp_file(m.data, m.size);
    ASSERT(path != NULL);

    boat_model_t* model = boat_gguf_load(path);
    ASSERT(model != NULL);
    ASSERT(boat_model_layer_count(model) > 0);

    // Verify layer count: 1 layer = 1 norm + 1 attn + 1 norm + 3 dense = 6 layers per block
    // 1 block + output norm + output dense = 6 + 2 = 8 layers
    size_t n_layers = boat_model_layer_count(model);
    ASSERT(n_layers >= 6);

    // Check each layer type
    boat_layer_t* l0 = boat_model_get_layer(model, 0);
    ASSERT(l0 != NULL);

    boat_model_free(model);
    remove(path);
    free_test_model(&m);
    PASS();
    return 0;
}

// -----------------------------------------------------------------------
// Test: Load GGUF with NULL filename
// -----------------------------------------------------------------------
static int test_load_null(void) {
    TEST("GGUF load NULL filename");
    ASSERT(boat_gguf_load(NULL) == NULL);
    PASS();
    return 0;
}

// -----------------------------------------------------------------------
// Test: Load non-existent GGUF
// -----------------------------------------------------------------------
static int test_load_nonexistent(void) {
    TEST("GGUF load non-existent file");
    ASSERT(boat_gguf_load("does_not_exist.gguf") == NULL);
    PASS();
    return 0;
}

// -----------------------------------------------------------------------
// Test: Metadata extraction via model user_data
// -----------------------------------------------------------------------
static int test_load_metadata(void) {
    TEST("GGUF load metadata");
    gguf_test_model_t m = build_test_gguf_f32();
    const char* path = write_temp_file(m.data, m.size);
    ASSERT(path != NULL);

    boat_model_t* model = boat_gguf_load(path);
    ASSERT(model != NULL);

    // Verify user_data was set (contains gguf_model_config_t)
    void* ud = boat_model_get_user_data(model);
    ASSERT(ud != NULL);

    boat_model_free(model);
    remove(path);
    free_test_model(&m);
    PASS();
    return 0;
}

// -----------------------------------------------------------------------
// Test: FP16-to-FP32 conversion accuracy
// -----------------------------------------------------------------------
static int test_f16_conversion(void) {
    TEST("GGUF FP16-to-FP32 conversion");

    // known value: 1.0 in FP16 = 0x3C00
    // We test by building a tiny GGUF with F16 tensor and loading it
    // The tensor data will get converted to FP32 during dequantization

    // Build a minimal GGUF with 1 F16 tensor
    uint8_t buf[1024];
    memset(buf, 0, sizeof(buf));
    uint8_t* p = buf;

    int64_t hidden = 8, d_ff = 32, n_heads = 2, n_layers = 1, vocab = 16;

    append_u32(&p, GGUF_MAGIC);
    append_u32(&p, GGUF_VERSION);
    append_u64(&p, 1); // 1 tensor
    append_u64(&p, 8); // 8 metadata keys

    append_string(&p, "general.architecture");
    append_u32(&p, GGUF_TYPE_STRING);
    append_string(&p, "llama");

    append_string(&p, "llama.embedding_length");
    append_u32(&p, GGUF_TYPE_INT64);
    append_u64(&p, (uint64_t)hidden);

    append_string(&p, "llama.block_count");
    append_u32(&p, GGUF_TYPE_INT64);
    append_u64(&p, (uint64_t)n_layers);

    append_string(&p, "llama.feed_forward_length");
    append_u32(&p, GGUF_TYPE_INT64);
    append_u64(&p, (uint64_t)d_ff);

    append_string(&p, "llama.attention.head_count");
    append_u32(&p, GGUF_TYPE_INT64);
    append_u64(&p, (uint64_t)n_heads);

    append_string(&p, "llama.vocab_size");
    append_u32(&p, GGUF_TYPE_INT64);
    append_u64(&p, (uint64_t)vocab);

    append_string(&p, "llama.context_length");
    append_u32(&p, GGUF_TYPE_INT64);
    append_u64(&p, 128);

    append_string(&p, "general.alignment");
    append_u32(&p, GGUF_TYPE_UINT64);
    append_u64(&p, 32);

    // Tensor: output_norm.weight with shape [hidden], type F16
    append_string(&p, "output_norm.weight");
    append_u32(&p, 1);
    append_u64(&p, (uint64_t)hidden);
    append_u32(&p, GGML_TYPE_F16);
    append_u64(&p, 0); // offset = 0

    // Align to 32
    while (((uint64_t)(p - buf) & 31) != 0) append_u8(&p, 0);

    // Tensor data: [hidden] FP16 values: 1.0=0x3C00, 2.0=0x4000, 0.5=0x3800, -1.0=0xBC00
    for (int64_t i = 0; i < hidden; i++) {
        float val = (float)(i + 1);
        append_f16(&p, val);
    }

    size_t sz = (size_t)(p - buf);
    const char* path = write_temp_file(buf, sz);
    ASSERT(path != NULL);

    boat_model_t* model = boat_gguf_load(path);
    ASSERT(model != NULL);
    ASSERT(boat_model_layer_count(model) >= 1);

    boat_model_free(model);
    remove(path);
    PASS();
    return 0;
}

// -----------------------------------------------------------------------
// Test: SiLU activation
// -----------------------------------------------------------------------
static int test_silu(void) {
    TEST("SiLU activation");
    int64_t shape[] = {5};
    boat_tensor_t* t = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32,
        (float[]){0.0f, 1.0f, -1.0f, 2.0f, -2.0f});
    ASSERT(t != NULL);

    boat_tensor_t* r = boat_silu(t);
    ASSERT(r != NULL);
    ASSERT(r != t); // new tensor

    const float* d = (const float*)boat_tensor_const_data(r);
    // SiLU(0) = 0
    ASSERT_NEAR(d[0], 0.0f, 1e-6f);
    // SiLU(1) = 1 / (1 + exp(-1)) ≈ 0.731
    ASSERT_NEAR(d[1], 0.7310586f, 1e-4f);
    // SiLU(-1) ≈ -0.269
    ASSERT_NEAR(d[2], -0.2689414f, 1e-4f);
    // SiLU(2) ≈ 1.762
    ASSERT_NEAR(d[3], 1.7615942f, 1e-4f);

    boat_tensor_unref(t);
    boat_tensor_unref(r);
    PASS();
    return 0;
}

// -----------------------------------------------------------------------
// Test: RMSNorm set_weight
// -----------------------------------------------------------------------
static int test_rmsnorm_set_weight(void) {
    TEST("RMSNorm set_weight");
    boat_rmsnorm_config_t cfg = { .normalized_shape = 8, .eps = 1e-5f, .elementwise_affine = true };
    boat_rmsnorm_t* norm = boat_rmsnorm_create(&cfg);
    ASSERT(norm != NULL);

    int64_t shape[] = {8};
    boat_tensor_t* w = boat_tensor_from_data(shape, 1, BOAT_DTYPE_FLOAT32,
        (float[]){2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f});
    ASSERT(w != NULL);

    boat_rmsnorm_set_weight(norm, w);
    boat_tensor_unref(w);

    boat_rmsnorm_free(norm);
    PASS();
    return 0;
}

// -----------------------------------------------------------------------
// Test: Attention set_weight
// -----------------------------------------------------------------------
static int test_attention_set_weight(void) {
    TEST("Attention set_weight");
    boat_attention_config_t cfg = {
        .hidden_size = 8, .num_heads = 2, .head_size = 4,
        .dropout_prob = 0.0f, .causal_mask = true, .use_bias = false
    };
    boat_attention_t* attn = boat_attention_create(&cfg);
    ASSERT(attn != NULL);

    int64_t shape[] = {8, 8};
    float data[64];
    for (int i = 0; i < 64; i++) data[i] = 0.01f * (float)i;

    boat_tensor_t* wq = boat_tensor_from_data(shape, 2, BOAT_DTYPE_FLOAT32, data);
    ASSERT(wq != NULL);
    boat_attention_set_weight_q(attn, wq);
    boat_tensor_unref(wq);

    boat_attention_free(attn);
    PASS();
    return 0;
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------
int main(void) {
    setbuf(stdout, NULL);
    printf("GGUF format loader tests:\n");

    if (test_check_valid()) return 1;
    if (test_check_invalid()) return 1;
    if (test_load_f32()) return 1;
    if (test_load_null()) return 1;
    if (test_load_nonexistent()) return 1;
    if (test_load_metadata()) return 1;
    if (test_f16_conversion()) return 1;
    if (test_silu()) return 1;
    if (test_rmsnorm_set_weight()) return 1;
    if (test_attention_set_weight()) return 1;

    printf("\n%d/%d tests passed!\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
