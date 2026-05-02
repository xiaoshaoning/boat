// prune.c - Weight pruning implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#define BOAT_BUILDING_DLL
#include <boat/prune.h>
#include <boat/tensor.h>
#include <boat/layers.h>
#include <boat/model.h>
#include <boat/quantize.h>
#include <boat.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Internal structures
// ---------------------------------------------------------------------------

typedef struct {
    size_t layer_index;
    boat_tensor_t* mask;   // FP32, same shape as weight, values 1.0 (keep) or 0.0 (pruned)
} boat_prune_mask_entry_t;

struct boat_prune_context_t {
    boat_model_t* model;
    boat_prune_mask_entry_t* entries;
    size_t num_entries;
    size_t capacity;
};

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

static bool is_layer_prunable(boat_layer_type_t type) {
    return type == BOAT_LAYER_TYPE_DENSE || type == BOAT_LAYER_TYPE_CONV2D;
}

// Get weight tensor for a layer
static boat_tensor_t* get_layer_weight(void* layer_data, boat_layer_type_t type) {
    switch (type) {
        case BOAT_LAYER_TYPE_DENSE:
            return boat_dense_layer_get_weight((const boat_dense_layer_t*)layer_data);
        case BOAT_LAYER_TYPE_CONV2D:
            return boat_conv_layer_get_weight((const boat_conv_layer_t*)layer_data);
        default:
            return NULL;
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

BOAT_API boat_prune_config_t boat_prune_config_default(void) {
    boat_prune_config_t cfg;
    cfg.sparsity = 0.5f;
    cfg.structured = false;
    cfg.threshold = 0.0f;
    cfg.iterative_steps = 1;
    cfg.prune_dim = 0;
    cfg.min_keep_ratio = 0.1f;
    return cfg;
}

// ---------------------------------------------------------------------------
// Context management
// ---------------------------------------------------------------------------

BOAT_API boat_prune_context_t* boat_prune_context_create(boat_model_t* model) {
    if (!model) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Prune] NULL model\n");
        return NULL;
    }

    boat_prune_context_t* ctx = (boat_prune_context_t*)boat_malloc(
        sizeof(boat_prune_context_t), BOAT_DEVICE_CPU);
    if (!ctx) return NULL;

    ctx->model = model;
    ctx->entries = NULL;
    ctx->num_entries = 0;
    ctx->capacity = 0;

    // Count prunable layers
    size_t n = boat_model_layer_count(model);
    for (size_t i = 0; i < n; i++) {
        boat_layer_t* layer = boat_model_get_layer(model, i);
        if (layer && is_layer_prunable(layer->type)) {
            ctx->num_entries++;
        }
    }

    if (ctx->num_entries == 0) {
        // No prunable layers, but context is valid (empty)
        return ctx;
    }

    ctx->entries = (boat_prune_mask_entry_t*)boat_malloc(
        sizeof(boat_prune_mask_entry_t) * ctx->num_entries, BOAT_DEVICE_CPU);
    if (!ctx->entries) {
        boat_free(ctx);
        return NULL;
    }
    ctx->capacity = ctx->num_entries;

    // Initialize entries with masks = all ones
    size_t idx = 0;
    for (size_t i = 0; i < n; i++) {
        boat_layer_t* layer = boat_model_get_layer(model, i);
        if (!layer || !is_layer_prunable(layer->type)) continue;

        ctx->entries[idx].layer_index = i;
        ctx->entries[idx].mask = NULL;
        idx++;
    }

    return ctx;
}

BOAT_API void boat_prune_context_free(boat_prune_context_t* ctx) {
    if (!ctx) return;
    for (size_t i = 0; i < ctx->num_entries; i++) {
        if (ctx->entries[i].mask) {
            boat_tensor_unref(ctx->entries[i].mask);
        }
    }
    boat_free(ctx->entries);
    boat_free(ctx);
}

BOAT_API boat_tensor_t* boat_prune_get_mask(const boat_prune_context_t* ctx, size_t layer_index) {
    if (!ctx) return NULL;
    for (size_t i = 0; i < ctx->num_entries; i++) {
        if (ctx->entries[i].layer_index == layer_index) {
            return ctx->entries[i].mask;
        }
    }
    return NULL;
}

// ---------------------------------------------------------------------------
// Threshold computation
// ---------------------------------------------------------------------------

static int compare_float_asc(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

BOAT_API float boat_compute_prune_threshold(const boat_tensor_t* weight, float sparsity) {
    if (!weight) return 0.0f;
    if (sparsity <= 0.0f) return -FLT_MAX;
    if (sparsity >= 1.0f) return FLT_MAX;

    size_t n = boat_tensor_nelements(weight);
    if (n == 0) return 0.0f;

    const float* data = (const float*)boat_tensor_const_data(weight);

    // Compute absolute values and sort
    float* abs_vals = (float*)boat_malloc(sizeof(float) * n, BOAT_DEVICE_CPU);
    if (!abs_vals) return 0.0f;

    for (size_t i = 0; i < n; i++) {
        abs_vals[i] = fabsf(data[i]);
    }

    qsort(abs_vals, n, sizeof(float), compare_float_asc);

    // Pick value at sparsity percentile
    size_t idx = (size_t)(sparsity * (float)(n - 1));
    if (idx >= n) idx = n - 1;
    float threshold = abs_vals[idx];

    boat_free(abs_vals);
    return threshold;
}

// ---------------------------------------------------------------------------
// Mask creation
// ---------------------------------------------------------------------------

BOAT_API boat_tensor_t* boat_create_magnitude_mask(const boat_tensor_t* weight, float threshold) {
    if (!weight) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Prune] NULL weight in create_magnitude_mask\n");
        return NULL;
    }

    size_t n = boat_tensor_nelements(weight);
    const int64_t* shape = boat_tensor_shape(weight);
    size_t ndim = boat_tensor_ndim(weight);

    boat_tensor_t* mask = boat_tensor_create(shape, ndim, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!mask) return NULL;

    const float* w = (const float*)boat_tensor_const_data(weight);
    float* m = (float*)boat_tensor_data(mask);

    for (size_t i = 0; i < n; i++) {
        m[i] = (fabsf(w[i]) <= threshold) ? 0.0f : 1.0f;
    }

    return mask;
}

BOAT_API boat_tensor_t* boat_create_structured_mask(const boat_tensor_t* weight, size_t dim,
                                                     float threshold, float min_keep_ratio) {
    if (!weight) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Prune] NULL weight in create_structured_mask\n");
        return NULL;
    }

    size_t ndim = boat_tensor_ndim(weight);
    const int64_t* shape = boat_tensor_shape(weight);

    if (dim >= ndim) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[Prune] dim %zu out of range (ndim=%zu)\n", dim, ndim);
        return NULL;
    }

    size_t n_channels = (size_t)shape[dim];
    const float* w = (const float*)boat_tensor_const_data(weight);

    // Compute per-slice L2 norms
    // stride for dim: total elements from dim onward
    size_t slice_stride = 1;
    for (size_t i = dim; i < ndim; i++) {
        slice_stride *= (size_t)shape[i];
    }
    size_t outer_stride = slice_stride; // elements per slice along dim
    size_t inner_stride = slice_stride / n_channels; // elements per channel within a slice

    float* l2_norms = (float*)boat_malloc(sizeof(float) * n_channels, BOAT_DEVICE_CPU);
    if (!l2_norms) return NULL;

    // Compute outer count (number of groups before dim)
    size_t outer_count = 1;
    for (size_t i = 0; i < dim; i++) {
        outer_count *= (size_t)shape[i];
    }

    for (size_t c = 0; c < n_channels; c++) {
        float sum_sq = 0.0f;
        for (size_t o = 0; o < outer_count; o++) {
            size_t base = o * outer_stride + c * inner_stride;
            for (size_t j = 0; j < inner_stride; j++) {
                float v = w[base + j];
                sum_sq += v * v;
            }
        }
        l2_norms[c] = sqrtf(sum_sq);
    }

    // Determine threshold: sort L2 norms, pick at sparsity percentile
    float* sorted = (float*)boat_malloc(sizeof(float) * n_channels, BOAT_DEVICE_CPU);
    if (!sorted) {
        boat_free(l2_norms);
        return NULL;
    }
    memcpy(sorted, l2_norms, sizeof(float) * n_channels);
    qsort(sorted, n_channels, sizeof(float), compare_float_asc);

    // If threshold <= 0, compute from min_keep_ratio
    float effective_threshold = threshold;
    if (effective_threshold <= 0.0f) {
        size_t keep_idx = (size_t)((1.0f - min_keep_ratio) * (float)(n_channels - 1));
        if (keep_idx >= n_channels) keep_idx = n_channels - 1;
        effective_threshold = sorted[keep_idx];
    } else {
        // threshold given directly: clamp min_keep
        size_t max_prune = (size_t)((1.0f - min_keep_ratio) * (float)n_channels);
        if (max_prune >= n_channels) max_prune = n_channels - 1;
        float min_threshold = sorted[max_prune];
        if (effective_threshold < min_threshold) {
            effective_threshold = min_threshold;
        }
    }

    boat_free(sorted);

    // Create mask
    boat_tensor_t* mask = boat_tensor_create(shape, ndim, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!mask) {
        boat_free(l2_norms);
        return NULL;
    }
    float* m = (float*)boat_tensor_data(mask);

    for (size_t c = 0; c < n_channels; c++) {
        float val = (l2_norms[c] < effective_threshold) ? 0.0f : 1.0f;
        for (size_t o = 0; o < outer_count; o++) {
            size_t base = o * outer_stride + c * inner_stride;
            for (size_t j = 0; j < inner_stride; j++) {
                m[base + j] = val;
            }
        }
    }

    boat_free(l2_norms);
    return mask;
}

// ---------------------------------------------------------------------------
// Mask application
// ---------------------------------------------------------------------------

BOAT_API bool boat_apply_mask(boat_tensor_t* weight, const boat_tensor_t* mask) {
    if (!weight || !mask) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Prune] NULL argument in apply_mask\n");
        return false;
    }
    if (boat_tensor_dtype(weight) != BOAT_DTYPE_FLOAT32 ||
        boat_tensor_dtype(mask) != BOAT_DTYPE_FLOAT32) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Prune] apply_mask requires FP32 tensors\n");
        return false;
    }
    if (boat_tensor_nelements(weight) != boat_tensor_nelements(mask)) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Prune] Shape mismatch in apply_mask\n");
        return false;
    }

    size_t n = boat_tensor_nelements(weight);
    float* w = (float*)boat_tensor_data(weight);
    const float* m = (const float*)boat_tensor_const_data(mask);

    for (size_t i = 0; i < n; i++) {
        w[i] *= m[i];
    }

    return true;
}

// ---------------------------------------------------------------------------
// Per-layer pruning
// ---------------------------------------------------------------------------

BOAT_API bool boat_prune_layer(boat_prune_context_t* ctx, size_t layer_index,
                                const boat_prune_config_t* config) {
    if (!ctx || !config) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Prune] NULL argument\n");
        return false;
    }

    boat_layer_t* layer = boat_model_get_layer(ctx->model, layer_index);
    if (!layer || !is_layer_prunable(layer->type)) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Prune] Layer %zu not prunable\n", layer_index);
        return false;
    }

    boat_tensor_t* weight = get_layer_weight(layer->data, layer->type);
    if (!weight) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Prune] Layer %zu has no weight\n", layer_index);
        return false;
    }
    if (boat_tensor_dtype(weight) != BOAT_DTYPE_FLOAT32) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Prune] Weight must be FP32\n");
        return false;
    }

    // Determine threshold
    float thr = config->threshold;
    if (thr <= 0.0f) {
        thr = boat_compute_prune_threshold(weight, config->sparsity);
    }

    // Create mask
    boat_tensor_t* mask;
    if (config->structured) {
        size_t dim = config->prune_dim;
        if (layer->type == BOAT_LAYER_TYPE_DENSE && dim == 0) {
            // Dense default: output_features is dim=1
            dim = 1;
        }
        mask = boat_create_structured_mask(weight, dim, thr, config->min_keep_ratio);
    } else {
        mask = boat_create_magnitude_mask(weight, thr);
    }

    if (!mask) return false;

    // Apply mask
    if (!boat_apply_mask(weight, mask)) {
        boat_tensor_unref(mask);
        return false;
    }

    // Store mask in context (replace existing if any)
    for (size_t i = 0; i < ctx->num_entries; i++) {
        if (ctx->entries[i].layer_index == layer_index) {
            if (ctx->entries[i].mask) {
                boat_tensor_unref(ctx->entries[i].mask);
            }
            ctx->entries[i].mask = mask;
            boat_tensor_ref(mask); // context owns a ref
            boat_tensor_unref(mask); // drop our local ref (context has it)
            return true;
        }
    }

    boat_tensor_unref(mask);
    return true;
}

BOAT_API bool boat_prune_model(boat_prune_context_t* ctx, const boat_prune_config_t* config) {
    if (!ctx || !config) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Prune] NULL argument\n");
        return false;
    }

    size_t n = boat_model_layer_count(ctx->model);
    for (size_t i = 0; i < n; i++) {
        boat_layer_t* layer = boat_model_get_layer(ctx->model, i);
        if (!layer || !is_layer_prunable(layer->type)) continue;
        if (!boat_prune_layer(ctx, i, config)) return false;
    }
    return true;
}

BOAT_API bool boat_prune_apply_masks(const boat_prune_context_t* ctx) {
    if (!ctx) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Prune] NULL context\n");
        return false;
    }

    for (size_t i = 0; i < ctx->num_entries; i++) {
        if (!ctx->entries[i].mask) continue;

        boat_layer_t* layer = boat_model_get_layer(ctx->model, ctx->entries[i].layer_index);
        if (!layer) continue;

        boat_tensor_t* weight = get_layer_weight(layer->data, layer->type);
        if (!weight) continue;

        // Raw loop for performance (identical shapes guaranteed)
        size_t n = boat_tensor_nelements(weight);
        float* w = (float*)boat_tensor_data(weight);
        const float* m = (const float*)boat_tensor_const_data(ctx->entries[i].mask);

        for (size_t j = 0; j < n; j++) {
            w[j] *= m[j];
        }
    }
    return true;
}

BOAT_API bool boat_prune_remove_mask(boat_prune_context_t* ctx, size_t layer_index) {
    if (!ctx) return false;

    for (size_t i = 0; i < ctx->num_entries; i++) {
        if (ctx->entries[i].layer_index == layer_index) {
            if (ctx->entries[i].mask) {
                boat_tensor_unref(ctx->entries[i].mask);
                ctx->entries[i].mask = NULL;
            }
            return true;
        }
    }
    return false;
}

BOAT_API void boat_prune_remove_all_masks(boat_prune_context_t* ctx) {
    if (!ctx) return;
    for (size_t i = 0; i < ctx->num_entries; i++) {
        if (ctx->entries[i].mask) {
            boat_tensor_unref(ctx->entries[i].mask);
            ctx->entries[i].mask = NULL;
        }
    }
}

// ---------------------------------------------------------------------------
// QAT fine-tuning after pruning
// ---------------------------------------------------------------------------

BOAT_API bool boat_prune_fake_quantize_model(const boat_prune_context_t* ctx,
                                              const boat_quant_config_t* quant_config) {
    if (!ctx || !quant_config) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Prune] NULL argument\n");
        return false;
    }

    size_t n = boat_model_layer_count(ctx->model);
    for (size_t i = 0; i < n; i++) {
        boat_layer_t* layer = boat_model_get_layer(ctx->model, i);
        if (!layer || !is_layer_prunable(layer->type)) continue;

        boat_tensor_t* weight = get_layer_weight(layer->data, layer->type);
        if (!weight) continue;
        if (boat_tensor_dtype(weight) != BOAT_DTYPE_FLOAT32) continue;

        if (!boat_fake_quantize(weight, quant_config)) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Sparsity computation
// ---------------------------------------------------------------------------

BOAT_API float boat_compute_sparsity(const boat_tensor_t* weight) {
    if (!weight) return 0.0f;

    size_t n = boat_tensor_nelements(weight);
    if (n == 0) return 0.0f;

    const float* d = (const float*)boat_tensor_const_data(weight);
    size_t zero_count = 0;
    for (size_t i = 0; i < n; i++) {
        if (d[i] == 0.0f) zero_count++;
    }

    return (float)zero_count / (float)n;
}

BOAT_API float boat_compute_structured_sparsity(const boat_tensor_t* weight, size_t dim) {
    if (!weight) return 0.0f;

    size_t ndim = boat_tensor_ndim(weight);
    const int64_t* shape = boat_tensor_shape(weight);

    if (dim >= ndim) return 0.0f;

    size_t n_channels = (size_t)shape[dim];
    size_t n = boat_tensor_nelements(weight);
    const float* d = (const float*)boat_tensor_const_data(weight);

    size_t outer_count = 1;
    for (size_t i = 0; i < dim; i++) {
        outer_count *= (size_t)shape[i];
    }

    size_t inner_stride = 1;
    for (size_t i = dim + 1; i < ndim; i++) {
        inner_stride *= (size_t)shape[i];
    }

    size_t outer_stride = n_channels * inner_stride;
    size_t zero_slices = 0;

    for (size_t c = 0; c < n_channels; c++) {
        bool all_zero = true;
        for (size_t o = 0; o < outer_count && all_zero; o++) {
            size_t base = o * outer_stride + c * inner_stride;
            for (size_t j = 0; j < inner_stride; j++) {
                if (d[base + j] != 0.0f) {
                    all_zero = false;
                    break;
                }
            }
        }
        if (all_zero) zero_slices++;
    }

    return (float)zero_slices / (float)n_channels;
}
