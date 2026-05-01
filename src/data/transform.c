// transform.c - Data transform implementations
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#define BOAT_BUILDING_DLL
#include <boat/data.h>
#include <boat.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ---------------------------------------------------------------------------
// Transform chain
// ---------------------------------------------------------------------------

struct boat_transform_chain_t {
    boat_transform_func_t* fns;
    void** contexts;
    size_t count;
    size_t capacity;
};

boat_transform_chain_t* boat_transform_chain_create(void) {
    boat_transform_chain_t* chain = boat_malloc(sizeof(boat_transform_chain_t), BOAT_DEVICE_CPU);
    if (!chain) return NULL;

    chain->count = 0;
    chain->capacity = 4;
    chain->fns = boat_malloc(sizeof(boat_transform_func_t) * chain->capacity, BOAT_DEVICE_CPU);
    chain->contexts = boat_malloc(sizeof(void*) * chain->capacity, BOAT_DEVICE_CPU);

    if (!chain->fns || !chain->contexts) {
        boat_free(chain->fns);
        boat_free(chain->contexts);
        boat_free(chain);
        return NULL;
    }

    return chain;
}

void boat_transform_chain_free(boat_transform_chain_t* chain) {
    if (!chain) return;
    boat_free(chain->fns);
    boat_free(chain->contexts);
    boat_free(chain);
}

void boat_transform_chain_add(boat_transform_chain_t* chain,
                              boat_transform_func_t fn, void* context) {
    if (!chain || !fn) return;

    if (chain->count >= chain->capacity) {
        size_t new_cap = chain->capacity * 2;
        boat_transform_func_t* new_fns = boat_realloc(
            chain->fns, sizeof(boat_transform_func_t) * new_cap, BOAT_DEVICE_CPU);
        void** new_ctx = boat_realloc(
            chain->contexts, sizeof(void*) * new_cap, BOAT_DEVICE_CPU);
        if (!new_fns || !new_ctx) {
            boat_free(new_fns);
            boat_free(new_ctx);
            return;
        }
        chain->fns = new_fns;
        chain->contexts = new_ctx;
        chain->capacity = new_cap;
    }

    chain->fns[chain->count] = fn;
    chain->contexts[chain->count] = context;
    chain->count++;
}

boat_tensor_t* boat_transform_chain_apply(boat_transform_chain_t* chain,
                                          boat_tensor_t* sample) {
    if (!chain || !sample) return NULL;

    boat_tensor_t* current = sample;
    for (size_t i = 0; i < chain->count; i++) {
        boat_tensor_t* next = chain->fns[i](current, chain->contexts[i]);
        // If the transform returned a different tensor, free the old one
        if (next != current) {
            boat_tensor_unref(current);
            current = next;
        }
        if (!current) return NULL;
    }

    return current;
}

// ---------------------------------------------------------------------------
// Simple deterministic hash-based RNG (no dependency on rand/srand)
// ---------------------------------------------------------------------------
static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static float frand(uint32_t* state) {
    return (float)(xorshift32(state) & 0x7FFFFFFFu) / 2147483648.0f;
}

// ---------------------------------------------------------------------------
// Normalize
// ---------------------------------------------------------------------------
boat_tensor_t* boat_transform_normalize(boat_tensor_t* sample, void* context) {
    if (!sample) return NULL;

    boat_dtype_t dt = boat_tensor_dtype(sample);
    if (dt != BOAT_DTYPE_FLOAT32) {
        boat_set_errorf(BOAT_ERROR_INVALID_OPERATION,
            "[Transform] Normalize requires FLOAT32, got %s\n", boat_dtype_name(dt));
        return sample;  // pass through
    }

    float mean = 0.0f;
    float std = 1.0f;
    if (context) {
        float* params = (float*)context;
        mean = params[0];
        std = params[1];
    }

    if (std == 0.0f) std = 1.0f;

    float* data = (float*)boat_tensor_data(sample);
    size_t n = boat_tensor_nelements(sample);
    for (size_t i = 0; i < n; i++) {
        data[i] = (data[i] - mean) / std;
    }

    return sample;
}

// ---------------------------------------------------------------------------
// Random horizontal flip (50%, in-place on [C,H,W] float32)
// ---------------------------------------------------------------------------
boat_tensor_t* boat_transform_random_hflip(boat_tensor_t* sample, void* context) {
    (void)context;
    if (!sample) return NULL;

    if (boat_tensor_ndim(sample) != 3 || boat_tensor_dtype(sample) != BOAT_DTYPE_FLOAT32) {
        return sample;  // pass through
    }

    // Simple counter-based RNG state kept in a static var for simplicity.
    // Each call advances the state so successive calls get different flips.
    static uint32_t rng_state = 0;
    if (rng_state == 0) rng_state = (uint32_t)time(NULL);

    if (frand(&rng_state) < 0.5f) {
        const int64_t* shape = boat_tensor_shape(sample);
        int64_t C = shape[0], H = shape[1], W = shape[2];
        float* data = (float*)boat_tensor_data(sample);

        for (int64_t c = 0; c < C; c++) {
            for (int64_t h = 0; h < H; h++) {
                float* row = data + (c * H + h) * W;
                for (int64_t w = 0; w < W / 2; w++) {
                    float tmp = row[w];
                    row[w] = row[W - 1 - w];
                    row[W - 1 - w] = tmp;
                }
            }
        }
    }

    return sample;
}

// ---------------------------------------------------------------------------
// Random crop (in-place on [C,H,W] float32)
// ---------------------------------------------------------------------------
boat_tensor_t* boat_transform_random_crop(boat_tensor_t* sample, void* context) {
    if (!sample) return NULL;

    if (boat_tensor_ndim(sample) != 3 || boat_tensor_dtype(sample) != BOAT_DTYPE_FLOAT32) {
        return sample;  // pass through
    }

    const int64_t* shape = boat_tensor_shape(sample);
    int64_t C = shape[0], H = shape[1], W = shape[2];

    int64_t crop_h = H - 2;
    int64_t crop_w = W - 2;
    if (context) {
        size_t* params = (size_t*)context;
        crop_h = (int64_t)params[0];
        crop_w = (int64_t)params[1];
    }

    if (crop_h > H || crop_w > W || crop_h <= 0 || crop_w <= 0) {
        return sample;  // invalid crop size, pass through
    }

    if (crop_h == H && crop_w == W) return sample;

    static uint32_t rng_state = 0;
    if (rng_state == 0) rng_state = (uint32_t)time(NULL);
    if (rng_state == 0) rng_state = 1;

    int64_t top = (int64_t)(frand(&rng_state) * (float)(H - crop_h + 1));
    int64_t left = (int64_t)(frand(&rng_state) * (float)(W - crop_w + 1));

    if (top < 0) top = 0;
    if (left < 0) left = 0;
    if (top + crop_h > H) top = H - crop_h;
    if (left + crop_w > W) left = W - crop_w;

    // Allocate output tensor for the crop
    int64_t out_shape[] = {C, crop_h, crop_w};
    boat_tensor_t* out = boat_tensor_create(out_shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!out) return NULL;

    float* src = (float*)boat_tensor_data(sample);
    float* dst = (float*)boat_tensor_data(out);

    for (int64_t c = 0; c < C; c++) {
        for (int64_t h = 0; h < crop_h; h++) {
            const float* src_row = src + (c * H + top + h) * W + left;
            float* dst_row = dst + (c * crop_h + h) * crop_w;
            memcpy(dst_row, src_row, (size_t)crop_w * sizeof(float));
        }
    }

    return out;
}
