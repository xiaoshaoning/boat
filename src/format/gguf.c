// gguf.c - GGUF model format loader
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/format/gguf.h>
#include <boat.h>
#include <boat/model.h>
#include <boat/layers.h>
#include <boat/layers/attention.h>
#include <boat/layers/norm.h>
#include <boat/memory.h>
#include <boat/tensor.h>
#include <boat/ops.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Maximum dimensions per tensor
#define GGUF_MAX_DIMS 8
#define GGUF_ARCH_NAME_LEN 64

// Block size constants for quantized types
#define GGUF_BLOCK_Q4 32
#define GGUF_BLOCK_Q5 32
#define GGUF_BLOCK_Q8 32

// Block byte sizes
#define GGUF_BLOCK_BYTES_Q4_0 18   // 2 (f16 d) + 16 (nibbles)
#define GGUF_BLOCK_BYTES_Q4_1 20   // 2+2 (f16 d, f16 m) + 16
#define GGUF_BLOCK_BYTES_Q5_0 22   // 2 (f16 d) + 4 (qh) + 16 (qs)
#define GGUF_BLOCK_BYTES_Q5_1 24   // 2+2 (f16 d, f16 m) + 4 (qh) + 16 (qs)
#define GGUF_BLOCK_BYTES_Q8_0 34   // 2 (f16 d) + 32 (int8[])

// -----------------------------------------------------------------------
// GGUF in-memory representation
// -----------------------------------------------------------------------

typedef struct {
    char* key;
    gguf_type_t type;
    union {
        uint8_t     uint8_val;
        int8_t      int8_val;
        uint16_t    uint16_val;
        int16_t     int16_val;
        uint32_t    uint32_val;
        int32_t     int32_val;
        float       float32_val;
        bool        bool_val;
        uint64_t    uint64_val;
        int64_t     int64_val;
        double      float64_val;
        struct {
            char*   data;
            uint64_t len;
        } string;
        struct {
            gguf_type_t elem_type;
            uint64_t    count;
            void*       elems;
        } array;
    } value;
} gguf_kv_t;

typedef struct {
    char*       name;
    uint32_t    n_dimensions;
    uint64_t    dimensions[GGUF_MAX_DIMS];
    ggml_type_t type;
    uint64_t    offset;
    size_t      n_elements;
} gguf_tensor_info_t;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_kv_count;
    gguf_kv_t* metadata;
    gguf_tensor_info_t* tensors;
    const uint8_t* tensor_data;
    size_t file_size;
    uint64_t alignment;
} gguf_context_t;

// Architecture config extracted from metadata
typedef struct {
    char arch[GGUF_ARCH_NAME_LEN];
    int64_t hidden_size;
    int64_t num_heads;
    int64_t num_kv_heads;
    int64_t num_layers;
    int64_t intermediate_size;
    int64_t vocab_size;
    float   norm_eps;
} gguf_model_config_t;

// -----------------------------------------------------------------------
// Binary reading helpers (little-endian)
// -----------------------------------------------------------------------

static uint8_t  read_u8(const uint8_t** p)  { uint8_t v; memcpy(&v, *p, 1); *p += 1; return v; }
static int8_t   read_i8(const uint8_t** p)  { int8_t v; memcpy(&v, *p, 1); *p += 1; return v; }
static uint16_t read_u16(const uint8_t** p) { uint16_t v; memcpy(&v, *p, 2); *p += 2; return v; }
static int16_t  read_i16(const uint8_t** p) { int16_t v; memcpy(&v, *p, 2); *p += 2; return v; }
static uint32_t read_u32(const uint8_t** p) { uint32_t v; memcpy(&v, *p, 4); *p += 4; return v; }
static int32_t  read_i32(const uint8_t** p) { int32_t v; memcpy(&v, *p, 4); *p += 4; return v; }
static uint64_t read_u64(const uint8_t** p) { uint64_t v; memcpy(&v, *p, 8); *p += 8; return v; }
static int64_t  read_i64(const uint8_t** p) { int64_t v; memcpy(&v, *p, 8); *p += 8; return v; }
static float    read_f32(const uint8_t** p) { float v; memcpy(&v, *p, 4); *p += 4; return v; }
static double   read_f64(const uint8_t** p) { double v; memcpy(&v, *p, 8); *p += 8; return v; }
static bool     read_bool(const uint8_t** p) { return read_u8(p) != 0; }

static char* read_string(const uint8_t** p) {
    uint64_t len = read_u64(p);
    char* s = (char*)boat_malloc((size_t)len + 1, BOAT_DEVICE_CPU);
    if (!s) return NULL;
    if (len > 0) memcpy(s, *p, (size_t)len);
    s[len] = '\0';
    *p += len;
    return s;
}

// -----------------------------------------------------------------------
// FP16 to FP32 conversion
// -----------------------------------------------------------------------

static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    int exp = (int)((h >> 10) & 0x1F);
    uint32_t mant = (uint32_t)(h & 0x3FF);
    if (exp == 0) {
        if (mant == 0) { uint32_t r = sign; float f; memcpy(&f, &r, 4); return f; }
        while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
        exp++;
        mant &= 0x3FF;
    } else if (exp == 31) {
        uint32_t r = sign | 0x7F800000 | (mant << 13);
        float f; memcpy(&f, &r, 4); return f;
    }
    uint32_t r = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &r, 4); return f;
}

// -----------------------------------------------------------------------
// Block dequantizers
// -----------------------------------------------------------------------

// Q4_0: block = { f16 d; u8 qs[16] }  — 32 4-bit values, packed as nibbles
static void dequant_q4_0_block(const uint8_t* block, float* out) {
    uint16_t d_bits; memcpy(&d_bits, block, 2);
    float d = f16_to_f32(d_bits);
    const uint8_t* qs = block + 2;
    for (int i = 0; i < 32; i++) {
        uint8_t nibble = (qs[i / 2] >> (4 * (i % 2))) & 0xF;
        out[i] = ((float)(int8_t)(nibble - 8)) * d;
    }
}

// Q4_1: block = { f16 d; f16 m; u8 qs[16] }
static void dequant_q4_1_block(const uint8_t* block, float* out) {
    uint16_t d_bits, m_bits;
    memcpy(&d_bits, block, 2); memcpy(&m_bits, block + 2, 2);
    float d = f16_to_f32(d_bits);
    float m = f16_to_f32(m_bits);
    const uint8_t* qs = block + 4;
    for (int i = 0; i < 32; i++) {
        uint8_t nibble = (qs[i / 2] >> (4 * (i % 2))) & 0xF;
        out[i] = (float)nibble * d + m;
    }
}

// Q5_0: block = { f16 d; u8 qh[4]; u8 qs[16] }
static void dequant_q5_0_block(const uint8_t* block, float* out) {
    uint16_t d_bits; memcpy(&d_bits, block, 2);
    float d = f16_to_f32(d_bits);
    const uint8_t* qh = block + 2;
    const uint8_t* qs = block + 6;
    for (int i = 0; i < 32; i++) {
        uint8_t low = (qs[i / 2] >> (4 * (i % 2))) & 0xF;
        uint8_t high = (qh[i / 8] >> (i % 8)) & 1;
        uint8_t val = low | (high << 4);
        out[i] = ((float)val - 16.0f) * d;
    }
}

// Q5_1: block = { f16 d; f16 m; u8 qh[4]; u8 qs[16] }
static void dequant_q5_1_block(const uint8_t* block, float* out) {
    uint16_t d_bits, m_bits;
    memcpy(&d_bits, block, 2); memcpy(&m_bits, block + 2, 2);
    float d = f16_to_f32(d_bits);
    float m = f16_to_f32(m_bits);
    const uint8_t* qh = block + 4;
    const uint8_t* qs = block + 8;
    for (int i = 0; i < 32; i++) {
        uint8_t low = (qs[i / 2] >> (4 * (i % 2))) & 0xF;
        uint8_t high = (qh[i / 8] >> (i % 8)) & 1;
        uint8_t val = low | (high << 4);
        out[i] = (float)val * d + m;
    }
}

// Q8_0: block = { f16 d; i8 qs[32] }
static void dequant_q8_0_block(const uint8_t* block, float* out) {
    uint16_t d_bits; memcpy(&d_bits, block, 2);
    float d = f16_to_f32(d_bits);
    const int8_t* qs = (const int8_t*)(block + 2);
    for (int i = 0; i < 32; i++)
        out[i] = (float)qs[i] * d;
}

// -----------------------------------------------------------------------
// Tensor dequantization dispatcher
// -----------------------------------------------------------------------

static bool dequantize_tensor(const uint8_t* src, float* dst,
                               const gguf_tensor_info_t* info) {
    size_t n = info->n_elements;
    if (n == 0) return true;

    switch (info->type) {
    case GGML_TYPE_F32:
        memcpy(dst, src, n * 4);
        return true;

    case GGML_TYPE_F16: {
        for (size_t i = 0; i < n; i++) {
            uint16_t h; memcpy(&h, src + i * 2, 2);
            dst[i] = f16_to_f32(h);
        }
        return true;
    }

    case GGML_TYPE_Q4_0: {
        size_t nb = (n + GGUF_BLOCK_Q4 - 1) / GGUF_BLOCK_Q4;
        for (size_t b = 0; b < nb; b++) {
            dequant_q4_0_block(src + b * GGUF_BLOCK_BYTES_Q4_0, dst + b * GGUF_BLOCK_Q4);
        }
        return true;
    }

    case GGML_TYPE_Q4_1: {
        size_t nb = (n + GGUF_BLOCK_Q4 - 1) / GGUF_BLOCK_Q4;
        for (size_t b = 0; b < nb; b++) {
            dequant_q4_1_block(src + b * GGUF_BLOCK_BYTES_Q4_1, dst + b * GGUF_BLOCK_Q4);
        }
        return true;
    }

    case GGML_TYPE_Q5_0: {
        size_t nb = (n + GGUF_BLOCK_Q5 - 1) / GGUF_BLOCK_Q5;
        for (size_t b = 0; b < nb; b++) {
            dequant_q5_0_block(src + b * GGUF_BLOCK_BYTES_Q5_0, dst + b * GGUF_BLOCK_Q5);
        }
        return true;
    }

    case GGML_TYPE_Q5_1: {
        size_t nb = (n + GGUF_BLOCK_Q5 - 1) / GGUF_BLOCK_Q5;
        for (size_t b = 0; b < nb; b++) {
            dequant_q5_1_block(src + b * GGUF_BLOCK_BYTES_Q5_1, dst + b * GGUF_BLOCK_Q5);
        }
        return true;
    }

    case GGML_TYPE_Q8_0: {
        size_t nb = (n + GGUF_BLOCK_Q8 - 1) / GGUF_BLOCK_Q8;
        for (size_t b = 0; b < nb; b++) {
            dequant_q8_0_block(src + b * GGUF_BLOCK_BYTES_Q8_0, dst + b * GGUF_BLOCK_Q8);
        }
        return true;
    }

    default:
        boat_set_errorf(BOAT_ERROR_FORMAT, "[GGUF] Unsupported tensor type %d\n", info->type);
        return false;
    }
}

// -----------------------------------------------------------------------
// GGUF context management
// -----------------------------------------------------------------------

static void gguf_context_free(gguf_context_t* ctx) {
    if (!ctx) return;
    for (uint64_t i = 0; i < ctx->metadata_kv_count; i++) {
        boat_free(ctx->metadata[i].key);
        if (ctx->metadata[i].type == GGUF_TYPE_STRING) {
            boat_free(ctx->metadata[i].value.string.data);
        } else if (ctx->metadata[i].type == GGUF_TYPE_ARRAY) {
            boat_free(ctx->metadata[i].value.array.elems);
        }
    }
    boat_free(ctx->metadata);
    for (uint64_t i = 0; i < ctx->tensor_count; i++) {
        boat_free(ctx->tensors[i].name);
    }
    boat_free(ctx->tensors);
    boat_free(ctx);
}

// -----------------------------------------------------------------------
// GGUF parser
// -----------------------------------------------------------------------

static gguf_context_t* gguf_parse(const uint8_t* data, size_t size) {
    // Minimum header: 4+4+8+8 = 24 bytes
    if (size < 24) {
        boat_set_errorf(BOAT_ERROR_FORMAT, "[GGUF] File too small\n");
        return NULL;
    }

    const uint8_t* p = data;

    uint32_t magic = read_u32(&p);
    if (magic != GGUF_MAGIC) {
        boat_set_errorf(BOAT_ERROR_FORMAT, "[GGUF] Invalid magic: 0x%08X\n", magic);
        return NULL;
    }

    uint32_t version = read_u32(&p);
    if (version < 2 || version > GGUF_VERSION) {
        boat_set_errorf(BOAT_ERROR_FORMAT, "[GGUF] Unsupported version %u\n", version);
        return NULL;
    }

    uint64_t tensor_count = read_u64(&p);
    uint64_t metadata_kv_count = read_u64(&p);

    gguf_context_t* ctx = (gguf_context_t*)boat_malloc(sizeof(gguf_context_t), BOAT_DEVICE_CPU);
    if (!ctx) return NULL;
    memset(ctx, 0, sizeof(gguf_context_t));
    ctx->magic = magic;
    ctx->version = version;
    ctx->tensor_count = tensor_count;
    ctx->metadata_kv_count = metadata_kv_count;
    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;
    ctx->file_size = size;

    // Allocate metadata arrays
    ctx->metadata = NULL;
    if (metadata_kv_count > 0) {
        ctx->metadata = (gguf_kv_t*)boat_malloc((size_t)metadata_kv_count * sizeof(gguf_kv_t), BOAT_DEVICE_CPU);
        if (!ctx->metadata) { boat_free(ctx); return NULL; }
        memset(ctx->metadata, 0, (size_t)metadata_kv_count * sizeof(gguf_kv_t));
    }

    // Parse metadata KV pairs
    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        ctx->metadata[i].key = read_string(&p);
        if (!ctx->metadata[i].key) { gguf_context_free(ctx); return NULL; }

        ctx->metadata[i].type = (gguf_type_t)read_u32(&p);

        switch (ctx->metadata[i].type) {
        case GGUF_TYPE_UINT8:   ctx->metadata[i].value.uint8_val = read_u8(&p); break;
        case GGUF_TYPE_INT8:    ctx->metadata[i].value.int8_val = read_i8(&p); break;
        case GGUF_TYPE_UINT16:  ctx->metadata[i].value.uint16_val = read_u16(&p); break;
        case GGUF_TYPE_INT16:   ctx->metadata[i].value.int16_val = read_i16(&p); break;
        case GGUF_TYPE_UINT32:  ctx->metadata[i].value.uint32_val = read_u32(&p); break;
        case GGUF_TYPE_INT32:   ctx->metadata[i].value.int32_val = read_i32(&p); break;
        case GGUF_TYPE_FLOAT32: ctx->metadata[i].value.float32_val = read_f32(&p); break;
        case GGUF_TYPE_BOOL:    ctx->metadata[i].value.bool_val = read_bool(&p); break;
        case GGUF_TYPE_UINT64:  ctx->metadata[i].value.uint64_val = read_u64(&p); break;
        case GGUF_TYPE_INT64:   ctx->metadata[i].value.int64_val = read_i64(&p); break;
        case GGUF_TYPE_FLOAT64: ctx->metadata[i].value.float64_val = read_f64(&p); break;
        case GGUF_TYPE_STRING:
            ctx->metadata[i].value.string.data = read_string(&p);
            ctx->metadata[i].value.string.len = strlen(ctx->metadata[i].value.string.data);
            break;
        case GGUF_TYPE_ARRAY: {
            ctx->metadata[i].value.array.elem_type = (gguf_type_t)read_u32(&p);
            ctx->metadata[i].value.array.count = read_u64(&p);
            size_t elem_size = 0;
            switch (ctx->metadata[i].value.array.elem_type) {
                case GGUF_TYPE_UINT8: case GGUF_TYPE_INT8: case GGUF_TYPE_BOOL:
                    elem_size = 1; break;
                case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16:
                    elem_size = 2; break;
                case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32:
                    elem_size = 4; break;
                case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64:
                    elem_size = 8; break;
                case GGUF_TYPE_STRING:
                    elem_size = 0; break; // special handling
                default: break;
            }
            if (elem_size > 0) {
                ctx->metadata[i].value.array.elems = boat_malloc(
                    (size_t)ctx->metadata[i].value.array.count * elem_size, BOAT_DEVICE_CPU);
                if (!ctx->metadata[i].value.array.elems) { gguf_context_free(ctx); return NULL; }
                memcpy(ctx->metadata[i].value.array.elems, p,
                       (size_t)ctx->metadata[i].value.array.count * elem_size);
                p += (size_t)ctx->metadata[i].value.array.count * elem_size;
            } else {
                // String arrays not fully handled; skip elements
                for (uint64_t j = 0; j < ctx->metadata[i].value.array.count; j++) {
                    uint64_t slen = read_u64(&p);
                    p += slen;
                }
            }
            break;
        }
        default:
            boat_set_errorf(BOAT_ERROR_FORMAT, "[GGUF] Unknown metadata type %d\n", ctx->metadata[i].type);
            gguf_context_free(ctx);
            return NULL;
        }

        // Check alignment metadata value
        if (strcmp(ctx->metadata[i].key, "general.alignment") == 0 &&
            ctx->metadata[i].type == GGUF_TYPE_UINT64) {
            ctx->alignment = ctx->metadata[i].value.uint64_val;
        }
    }

    // Parse tensor info entries
    ctx->tensors = NULL;
    if (tensor_count > 0) {
        ctx->tensors = (gguf_tensor_info_t*)boat_malloc(
            (size_t)tensor_count * sizeof(gguf_tensor_info_t), BOAT_DEVICE_CPU);
        if (!ctx->tensors) { gguf_context_free(ctx); return NULL; }
        memset(ctx->tensors, 0, (size_t)tensor_count * sizeof(gguf_tensor_info_t));
    }

    for (uint64_t i = 0; i < tensor_count; i++) {
        ctx->tensors[i].name = read_string(&p);
        if (!ctx->tensors[i].name) { gguf_context_free(ctx); return NULL; }

        ctx->tensors[i].n_dimensions = read_u32(&p);
        if (ctx->tensors[i].n_dimensions > GGUF_MAX_DIMS) {
            boat_set_errorf(BOAT_ERROR_FORMAT, "[GGUF] Tensor %s has %u dimensions (max %d)\n",
                           ctx->tensors[i].name, ctx->tensors[i].n_dimensions, GGUF_MAX_DIMS);
            gguf_context_free(ctx);
            return NULL;
        }
        for (uint32_t d = 0; d < ctx->tensors[i].n_dimensions; d++) {
            ctx->tensors[i].dimensions[d] = read_u64(&p);
        }
        ctx->tensors[i].type = (ggml_type_t)read_u32(&p);
        ctx->tensors[i].offset = read_u64(&p);

        // Compute element count
        ctx->tensors[i].n_elements = 1;
        for (uint32_t d = 0; d < ctx->tensors[i].n_dimensions; d++) {
            ctx->tensors[i].n_elements *= (size_t)ctx->tensors[i].dimensions[d];
        }
    }

    // Tensor data starts at current position (already past header + metadata + tensor info)
    // Align to ctx->alignment
    uint64_t data_offset = (uint64_t)(p - data);
    uint64_t aligned = (data_offset + ctx->alignment - 1) & ~(ctx->alignment - 1);
    ctx->tensor_data = data + aligned;

    return ctx;
}

// -----------------------------------------------------------------------
// Metadata extraction helper
// -----------------------------------------------------------------------

static int64_t find_metadata_int(const gguf_context_t* ctx, const char* key, int64_t def) {
    for (uint64_t i = 0; i < ctx->metadata_kv_count; i++) {
        if (strcmp(ctx->metadata[i].key, key) == 0) {
            switch (ctx->metadata[i].type) {
            case GGUF_TYPE_INT64:  return ctx->metadata[i].value.int64_val;
            case GGUF_TYPE_UINT64: return (int64_t)ctx->metadata[i].value.uint64_val;
            case GGUF_TYPE_INT32:  return ctx->metadata[i].value.int32_val;
            case GGUF_TYPE_UINT32: return ctx->metadata[i].value.uint32_val;
            default: break;
            }
        }
    }
    return def;
}

static float find_metadata_float(const gguf_context_t* ctx, const char* key, float def) {
    for (uint64_t i = 0; i < ctx->metadata_kv_count; i++) {
        if (strcmp(ctx->metadata[i].key, key) == 0) {
            if (ctx->metadata[i].type == GGUF_TYPE_FLOAT32)
                return ctx->metadata[i].value.float32_val;
            if (ctx->metadata[i].type == GGUF_TYPE_FLOAT64)
                return (float)ctx->metadata[i].value.float64_val;
        }
    }
    return def;
}

static const char* find_metadata_string(const gguf_context_t* ctx, const char* key) {
    for (uint64_t i = 0; i < ctx->metadata_kv_count; i++) {
        if (strcmp(ctx->metadata[i].key, key) == 0 &&
            ctx->metadata[i].type == GGUF_TYPE_STRING) {
            return ctx->metadata[i].value.string.data;
        }
    }
    return NULL;
}

// -----------------------------------------------------------------------
// Tensor lookup by name
// -----------------------------------------------------------------------

static const gguf_tensor_info_t* find_tensor(const gguf_context_t* ctx, const char* name) {
    for (uint64_t i = 0; i < ctx->tensor_count; i++) {
        if (strcmp(ctx->tensors[i].name, name) == 0)
            return &ctx->tensors[i];
    }
    return NULL;
}

// -----------------------------------------------------------------------
// Dequantize + transpose helper
// -----------------------------------------------------------------------
// GGUF stores weights in shape [out_features, in_features] (PyTorch convention).
// Boat stores as [in_features, out_features]. Non-square weights need transpose.
// Returns a new boat_tensor_t with ref_count=1. Caller owns it.

static boat_tensor_t* dequantize_and_transpose(const gguf_context_t* ctx,
    const gguf_tensor_info_t* info, int64_t in_features, int64_t out_features) {
    if (!info) return NULL;

    // Check if GGUF shape matches [out, in]
    bool needs_transpose = true;
    if (info->n_dimensions >= 2 &&
        (int64_t)info->dimensions[0] == in_features &&
        (int64_t)info->dimensions[1] == out_features) {
        // Shape already matches Boat convention
        needs_transpose = false;
    }

    // Dequantize to temporary buffer
    float* temp = (float*)boat_malloc(info->n_elements * sizeof(float), BOAT_DEVICE_CPU);
    if (!temp) return NULL;
    if (!dequantize_tensor(ctx->tensor_data + info->offset, temp, info)) {
        boat_free(temp);
        return NULL;
    }

    int64_t boat_shape[2] = {in_features, out_features};
    boat_tensor_t* t = boat_tensor_create(boat_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!t) { boat_free(temp); return NULL; }

    float* dst = (float*)boat_tensor_data(t);

    if (needs_transpose) {
        // GGUF has shape [out, in], Boat needs [in, out]
        for (int64_t i = 0; i < in_features; i++) {
            for (int64_t j = 0; j < out_features; j++) {
                dst[i * out_features + j] = temp[j * in_features + i];
            }
        }
    } else {
        memcpy(dst, temp, info->n_elements * sizeof(float));
    }

    boat_free(temp);
    return t;
}

// Variant: dequantize to 1D (for norm weights, biases)
static boat_tensor_t* dequantize_1d(const gguf_context_t* ctx,
    const gguf_tensor_info_t* info, int64_t n) {
    if (!info) return NULL;

    int64_t shape[] = {n};
    boat_tensor_t* t = boat_tensor_create(shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!t) return NULL;

    float* dst = (float*)boat_tensor_data(t);
    if (!dequantize_tensor(ctx->tensor_data + info->offset, dst, info)) {
        boat_tensor_unref(t);
        return NULL;
    }
    return t;
}

// -----------------------------------------------------------------------
// Extract model configuration from GGUF metadata
// -----------------------------------------------------------------------

static bool extract_config(const gguf_context_t* ctx, gguf_model_config_t* cfg) {
    memset(cfg, 0, sizeof(gguf_model_config_t));

    // Read architecture name
    const char* arch = find_metadata_string(ctx, "general.architecture");
    if (!arch) {
        boat_set_errorf(BOAT_ERROR_FORMAT, "[GGUF] Missing general.architecture metadata\n");
        return false;
    }
    strncpy(cfg->arch, arch, GGUF_ARCH_NAME_LEN - 1);
    cfg->arch[GGUF_ARCH_NAME_LEN - 1] = '\0';

    // Helper: try both "<arch>.<key>" and bare "<key>"
    #define READ_CFG_INT(key, field, def) do { \
        char buf[128]; \
        snprintf(buf, sizeof(buf), "%s.%s", cfg->arch, key); \
        cfg->field = find_metadata_int(ctx, buf, def); \
        if (cfg->field == (int64_t)def) \
            cfg->field = find_metadata_int(ctx, key, def); \
    } while(0)

    #define READ_CFG_FLOAT(key, field, def) do { \
        char buf[128]; \
        snprintf(buf, sizeof(buf), "%s.%s", cfg->arch, key); \
        cfg->field = find_metadata_float(ctx, buf, def); \
        if (cfg->field == (float)def - 1.0f) \
            cfg->field = find_metadata_float(ctx, key, def); \
    } while(0)

    // Map common metadata keys to our config
    // LLaMA naming: embedding_length, block_count, feed_forward_length, attention.head_count,
    //               attention.head_count_kv, context_length, vocab_size, layer_norm_rms_epsilon
    // GPT-2 naming: n_embd, n_layer, n_ff, n_head, n_ctx, n_vocab

    READ_CFG_INT("embedding_length", hidden_size, 0);
    if (cfg->hidden_size == 0) cfg->hidden_size = find_metadata_int(ctx, "n_embd", 0);

    READ_CFG_INT("block_count", num_layers, 0);
    if (cfg->num_layers == 0) cfg->num_layers = find_metadata_int(ctx, "n_layer", 0);

    READ_CFG_INT("feed_forward_length", intermediate_size, 0);
    if (cfg->intermediate_size == 0) cfg->intermediate_size = find_metadata_int(ctx, "n_ff", 0);

    READ_CFG_INT("attention.head_count", num_heads, 0);
    if (cfg->num_heads == 0) cfg->num_heads = find_metadata_int(ctx, "n_head", 0);

    READ_CFG_INT("attention.head_count_kv", num_kv_heads, cfg->num_heads);

    READ_CFG_INT("vocab_size", vocab_size, 0);
    if (cfg->vocab_size == 0) cfg->vocab_size = find_metadata_int(ctx, "n_vocab", 0);

    READ_CFG_FLOAT("layer_norm_rms_epsilon", norm_eps, 1e-5f);

    // Validate essential fields
    if (cfg->hidden_size == 0 || cfg->num_layers == 0 || cfg->num_heads == 0) {
        boat_set_errorf(BOAT_ERROR_FORMAT, "[GGUF] Missing essential model config fields\n");
        return false;
    }

    return true;
}

// -----------------------------------------------------------------------
// Model builder
// -----------------------------------------------------------------------

static boat_model_t* build_llama_model(const gguf_context_t* ctx,
                                        const gguf_model_config_t* cfg) {
    boat_model_t* model = boat_model_create();
    if (!model) return NULL;

    int64_t hidden = cfg->hidden_size;
    int64_t n_heads = cfg->num_heads;
    int64_t n_layers = cfg->num_layers;
    int64_t d_ff = cfg->intermediate_size;
    int64_t head_dim = hidden / n_heads;

    // --- Create layers for each transformer block ---
    for (int64_t i = 0; i < n_layers; i++) {
        char tname[128];

        // 1. Pre-attention RMSNorm
        snprintf(tname, sizeof(tname), "blk.%lld.attn_norm.weight", (long long)i);
        const gguf_tensor_info_t* norm1_info = find_tensor(ctx, tname);
        if (!norm1_info) {
            // Try GPT-2 style: "gpt2" or "model.h.attn_norm.weight"
            snprintf(tname, sizeof(tname), "h.%lld.attn_norm.weight", (long long)i);
            norm1_info = find_tensor(ctx, tname);
        }
        if (norm1_info) {
            boat_rmsnorm_config_t rcfg = {
                .normalized_shape = (size_t)hidden,
                .eps = cfg->norm_eps,
                .elementwise_affine = true
            };
            boat_rmsnorm_t* rms = boat_rmsnorm_create(&rcfg);
            if (!rms) { boat_model_free(model); return NULL; }
            boat_tensor_t* w = dequantize_1d(ctx, norm1_info, hidden);
            if (!w) { boat_rmsnorm_free(rms); boat_model_free(model); return NULL; }
            boat_rmsnorm_set_weight(rms, w);
            boat_tensor_unref(w);

            boat_layer_t* layer = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            layer->data = rms;
            layer->type = BOAT_LAYER_TYPE_RMSNORM;
            layer->ops = NULL; // set_layer_ops in model_add_layer will set it
            boat_model_add_layer(model, layer);
        }

        // 2. Self-attention
        const gguf_tensor_info_t* q_info = NULL;
        const gguf_tensor_info_t* k_info = NULL;
        const gguf_tensor_info_t* v_info = NULL;
        const gguf_tensor_info_t* o_info = NULL;

        snprintf(tname, sizeof(tname), "blk.%lld.attn_q.weight", (long long)i);
        q_info = find_tensor(ctx, tname);
        snprintf(tname, sizeof(tname), "blk.%lld.attn_k.weight", (long long)i);
        k_info = find_tensor(ctx, tname);
        snprintf(tname, sizeof(tname), "blk.%lld.attn_v.weight", (long long)i);
        v_info = find_tensor(ctx, tname);
        snprintf(tname, sizeof(tname), "blk.%lld.attn_output.weight", (long long)i);
        o_info = find_tensor(ctx, tname);

        // GPT-2 style: h.{i}.attn.c_attn.weight (merged QKV)
        if (!q_info) {
            snprintf(tname, sizeof(tname), "h.%lld.attn.c_attn.weight", (long long)i);
            q_info = find_tensor(ctx, tname);
            // For merged QKV, we skip individual K/V loading
        }

        // Try GPT-2 style separate Q/K/V/O
        if (!q_info) {
            snprintf(tname, sizeof(tname), "h.%lld.attn.q_proj.weight", (long long)i);
            q_info = find_tensor(ctx, tname);
            snprintf(tname, sizeof(tname), "h.%lld.attn.k_proj.weight", (long long)i);
            k_info = find_tensor(ctx, tname);
            snprintf(tname, sizeof(tname), "h.%lld.attn.v_proj.weight", (long long)i);
            v_info = find_tensor(ctx, tname);
            snprintf(tname, sizeof(tname), "h.%lld.attn.o_proj.weight", (long long)i);
            o_info = find_tensor(ctx, tname);
        }

        if (q_info && o_info) {
            boat_attention_config_t acfg = {
                .hidden_size  = (size_t)hidden,
                .num_heads    = (size_t)n_heads,
                .head_size    = (size_t)head_dim,
                .dropout_prob = 0.0f,
                .causal_mask  = true,
                .use_bias     = false,
                .use_rotary   = true,
                .rotary_theta = 10000.0f
            };
            boat_attention_t* attn = boat_attention_create(&acfg);
            if (!attn) { boat_model_free(model); return NULL; }

            // Load Q weight
            boat_tensor_t* wq = dequantize_and_transpose(ctx, q_info, hidden, hidden);
            if (wq) { boat_attention_set_weight_q(attn, wq); boat_tensor_unref(wq); }

            // Load K weight (if separate)
            if (k_info) {
                boat_tensor_t* wk = dequantize_and_transpose(ctx, k_info, hidden, hidden);
                if (wk) { boat_attention_set_weight_k(attn, wk); boat_tensor_unref(wk); }
            }

            // Load V weight (if separate)
            if (v_info) {
                boat_tensor_t* wv = dequantize_and_transpose(ctx, v_info, hidden, hidden);
                if (wv) { boat_attention_set_weight_v(attn, wv); boat_tensor_unref(wv); }
            }

            // Load O weight
            boat_tensor_t* wo = dequantize_and_transpose(ctx, o_info, hidden, hidden);
            if (wo) { boat_attention_set_weight_o(attn, wo); boat_tensor_unref(wo); }

            boat_layer_t* layer = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            layer->data = attn;
            layer->type = BOAT_LAYER_TYPE_ATTENTION;
            layer->ops = NULL;
            boat_model_add_layer(model, layer);
        }

        // 3. Pre-FFN RMSNorm
        snprintf(tname, sizeof(tname), "blk.%lld.ffn_norm.weight", (long long)i);
        const gguf_tensor_info_t* norm2_info = find_tensor(ctx, tname);
        if (!norm2_info) {
            snprintf(tname, sizeof(tname), "h.%lld.ffn_norm.weight", (long long)i);
            norm2_info = find_tensor(ctx, tname);
        }
        if (norm2_info) {
            boat_rmsnorm_config_t rcfg = {
                .normalized_shape = (size_t)hidden,
                .eps = cfg->norm_eps,
                .elementwise_affine = true
            };
            boat_rmsnorm_t* rms = boat_rmsnorm_create(&rcfg);
            if (!rms) { boat_model_free(model); return NULL; }
            boat_tensor_t* w = dequantize_1d(ctx, norm2_info, hidden);
            if (!w) { boat_rmsnorm_free(rms); boat_model_free(model); return NULL; }
            boat_rmsnorm_set_weight(rms, w);
            boat_tensor_unref(w);

            boat_layer_t* layer = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            layer->data = rms;
            layer->type = BOAT_LAYER_TYPE_RMSNORM;
            layer->ops = NULL;
            boat_model_add_layer(model, layer);
        }

        // 4. FFN gate projection (SwiGLU: gate = silu(x @ W_gate))
        snprintf(tname, sizeof(tname), "blk.%lld.ffn_gate.weight", (long long)i);
        const gguf_tensor_info_t* gate_info = find_tensor(ctx, tname);
        // Also try GPT-2 naming
        if (!gate_info) {
            snprintf(tname, sizeof(tname), "h.%lld.ffn.gate.weight", (long long)i);
            gate_info = find_tensor(ctx, tname);
            if (!gate_info) {
                snprintf(tname, sizeof(tname), "h.%lld.ffn.gate_proj.weight", (long long)i);
                gate_info = find_tensor(ctx, tname);
            }
        }
        if (gate_info) {
            boat_dense_layer_t* gate = boat_dense_layer_create(
                (size_t)hidden, (size_t)d_ff, false);
            if (!gate) { boat_model_free(model); return NULL; }
            boat_tensor_t* w = dequantize_and_transpose(ctx, gate_info, hidden, d_ff);
            if (w) { boat_dense_layer_set_weight(gate, w); boat_tensor_unref(w); }
            boat_layer_t* layer = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            layer->data = gate;
            layer->type = BOAT_LAYER_TYPE_DENSE;
            layer->ops = NULL;
            boat_model_add_layer(model, layer);
        }

        // 5. FFN up projection
        snprintf(tname, sizeof(tname), "blk.%lld.ffn_up.weight", (long long)i);
        const gguf_tensor_info_t* up_info = find_tensor(ctx, tname);
        if (!up_info) {
            snprintf(tname, sizeof(tname), "h.%lld.ffn.up_proj.weight", (long long)i);
            up_info = find_tensor(ctx, tname);
        }
        if (up_info) {
            boat_dense_layer_t* up = boat_dense_layer_create(
                (size_t)hidden, (size_t)d_ff, false);
            if (!up) { boat_model_free(model); return NULL; }
            boat_tensor_t* w = dequantize_and_transpose(ctx, up_info, hidden, d_ff);
            if (w) { boat_dense_layer_set_weight(up, w); boat_tensor_unref(w); }
            boat_layer_t* layer = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            layer->data = up;
            layer->type = BOAT_LAYER_TYPE_DENSE;
            layer->ops = NULL;
            boat_model_add_layer(model, layer);
        }

        // 5b. FFN down projection
        snprintf(tname, sizeof(tname), "blk.%lld.ffn_down.weight", (long long)i);
        const gguf_tensor_info_t* down_info = find_tensor(ctx, tname);
        if (!down_info) {
            snprintf(tname, sizeof(tname), "h.%lld.ffn.down_proj.weight", (long long)i);
            down_info = find_tensor(ctx, tname);
        }
        if (down_info) {
            boat_dense_layer_t* down = boat_dense_layer_create(
                (size_t)d_ff, (size_t)hidden, false);
            if (!down) { boat_model_free(model); return NULL; }
            // Note: down_proj shape is [hidden, d_ff] in GGUF, Boat expects [d_ff, hidden]
            boat_tensor_t* w = dequantize_and_transpose(ctx, down_info, d_ff, hidden);
            if (w) { boat_dense_layer_set_weight(down, w); boat_tensor_unref(w); }
            boat_layer_t* layer = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            layer->data = down;
            layer->type = BOAT_LAYER_TYPE_DENSE;
            layer->ops = NULL;
            boat_model_add_layer(model, layer);
        }
    }

    // --- Output RMSNorm ---
    const gguf_tensor_info_t* out_norm = find_tensor(ctx, "output_norm.weight");
    if (!out_norm) out_norm = find_tensor(ctx, "model.norm.weight"); // GPT-2
    if (out_norm) {
        boat_rmsnorm_config_t rcfg = {
            .normalized_shape = (size_t)hidden,
            .eps = cfg->norm_eps,
            .elementwise_affine = true
        };
        boat_rmsnorm_t* rms = boat_rmsnorm_create(&rcfg);
        if (rms) {
            boat_tensor_t* w = dequantize_1d(ctx, out_norm, hidden);
            if (w) { boat_rmsnorm_set_weight(rms, w); boat_tensor_unref(w); }
            boat_layer_t* layer = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            layer->data = rms;
            layer->type = BOAT_LAYER_TYPE_RMSNORM;
            layer->ops = NULL;
            boat_model_add_layer(model, layer);
        }
    }

    // --- Output projection ---
    const gguf_tensor_info_t* out_proj = find_tensor(ctx, "output.weight");
    if (!out_proj) out_proj = find_tensor(ctx, "model.lm_head.weight"); // GPT-2
    if (out_proj && cfg->vocab_size > 0) {
        boat_dense_layer_t* out = boat_dense_layer_create(
            (size_t)hidden, (size_t)cfg->vocab_size, false);
        if (out) {
            boat_tensor_t* w = dequantize_and_transpose(ctx, out_proj, hidden, cfg->vocab_size);
            if (w) { boat_dense_layer_set_weight(out, w); boat_tensor_unref(w); }
            boat_layer_t* layer = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            layer->data = out;
            layer->type = BOAT_LAYER_TYPE_DENSE;
            layer->ops = NULL;
            boat_model_add_layer(model, layer);
        }
    }

    // Store config in model user_data for downstream use
    gguf_model_config_t* cfg_copy = (gguf_model_config_t*)boat_malloc(
        sizeof(gguf_model_config_t), BOAT_DEVICE_CPU);
    if (cfg_copy) {
        memcpy(cfg_copy, cfg, sizeof(gguf_model_config_t));
        boat_model_set_user_data(model, cfg_copy, boat_memory_free);
    }

    return model;
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

BOAT_API bool boat_gguf_check(const char* filename) {
    if (!filename) return false;

    FILE* f = fopen(filename, "rb");
    if (!f) return false;

    uint32_t magic = 0;
    fread(&magic, 4, 1, f);
    fclose(f);

    return magic == GGUF_MAGIC;
}

BOAT_API boat_model_t* boat_gguf_load(const char* filename) {
    if (!filename) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[GGUF] filename is NULL\n");
        return NULL;
    }

    // Read file
    FILE* f = fopen(filename, "rb");
    if (!f) {
        boat_set_errorf(BOAT_ERROR_FILE_IO, "[GGUF] Cannot open %s\n", filename);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    if (fsize <= 0) {
        fclose(f);
        boat_set_errorf(BOAT_ERROR_FILE_IO, "[GGUF] Empty file\n");
        return NULL;
    }
    rewind(f);

    uint8_t* file_data = (uint8_t*)boat_malloc((size_t)fsize, BOAT_DEVICE_CPU);
    if (!file_data) { fclose(f); return NULL; }

    size_t bytes_read = fread(file_data, 1, (size_t)fsize, f);
    fclose(f);

    if (bytes_read != (size_t)fsize) {
        boat_free(file_data);
        boat_set_errorf(BOAT_ERROR_FILE_IO, "[GGUF] Failed to read file\n");
        return NULL;
    }

    // Parse GGUF
    gguf_context_t* ctx = gguf_parse(file_data, (size_t)fsize);
    if (!ctx) {
        boat_free(file_data);
        return NULL;
    }

    // Extract config
    gguf_model_config_t cfg;
    if (!extract_config(ctx, &cfg)) {
        gguf_context_free(ctx);
        boat_free(file_data);
        return NULL;
    }

    // Build model
    boat_model_t* model = build_llama_model(ctx, &cfg);

    gguf_context_free(ctx);
    boat_free(file_data);

    return model;
}
