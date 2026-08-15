// softmax_cross_entropy.c - Softmax cross-entropy loss function
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/loss.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <math.h>
#include <stdlib.h>

// Softmax cross-entropy loss structure
typedef struct {
    boat_loss_type_t type; // Always BOAT_LOSS_SOFTMAX_CROSS_ENTROPY
} softmax_cross_entropy_loss_t;

// Forward declaration for dispatch
float softmax_cross_entropy_loss_compute(boat_loss_t* loss, const void* predictions,
                                         const void* targets);

// Create softmax cross-entropy loss function
BOAT_API boat_loss_t* boat_softmax_cross_entropy_loss_create() {
    softmax_cross_entropy_loss_t* loss = (softmax_cross_entropy_loss_t*)boat_malloc(
        sizeof(softmax_cross_entropy_loss_t), BOAT_DEVICE_CPU);
    if (!loss) {
        return NULL;
    }

    loss->type = BOAT_LOSS_SOFTMAX_CROSS_ENTROPY;

    return (boat_loss_t*)loss;
}

// Numerically stable softmax over a single row of logits.
static void softmax_row(const float* logits, size_t n, float* out) {
    float max_val = logits[0];
    for (size_t i = 1; i < n; i++) {
        if (logits[i] > max_val) max_val = logits[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        out[i] = expf(logits[i] - max_val);
        sum += out[i];
    }
    if (sum > 0.0f) {
        for (size_t i = 0; i < n; i++)
            out[i] /= sum;
    }
}

// Read an integer class label from the targets tensor (INT32 or INT64).
static int64_t read_label(const boat_tensor_t* targets, size_t idx) {
    boat_dtype_t dtype = boat_tensor_dtype(targets);
    if (dtype == BOAT_DTYPE_INT64) {
        return ((const int64_t*)boat_tensor_const_data(targets))[idx];
    }
    return (int64_t)((const int32_t*)boat_tensor_const_data(targets))[idx];
}

// Compute softmax cross-entropy: -mean(log(softmax(logits)[target])).
float softmax_cross_entropy_loss_compute(boat_loss_t* loss, const void* predictions,
                                         const void* targets) {
    (void)loss;
    if (!predictions || !targets) return 0.0f;

    const boat_tensor_t* logits = (const boat_tensor_t*)predictions;
    const boat_tensor_t* target = (const boat_tensor_t*)targets;

    if (boat_tensor_ndim(logits) != 2 || boat_tensor_ndim(target) != 1) return 0.0f;
    if (boat_tensor_dtype(logits) != BOAT_DTYPE_FLOAT32) return 0.0f;
    boat_dtype_t tdtype = boat_tensor_dtype(target);
    if (tdtype != BOAT_DTYPE_INT32 && tdtype != BOAT_DTYPE_INT64) return 0.0f;

    const int64_t* logits_shape = boat_tensor_shape(logits);
    size_t batch = (size_t)logits_shape[0];
    size_t num_classes = (size_t)logits_shape[1];
    if (num_classes == 0) return 0.0f;
    if (boat_tensor_nelements(target) != batch) return 0.0f;

    const float* l = (const float*)boat_tensor_const_data(logits);

    float* sm = (float*)malloc(num_classes * sizeof(float));
    if (!sm) return 0.0f;

    float total = 0.0f;
    for (size_t b = 0; b < batch; b++) {
        softmax_row(l + b * num_classes, num_classes, sm);
        int64_t label = read_label(target, b);
        if (label < 0 || label >= (int64_t)num_classes) {
            free(sm);
            return 0.0f;
        }
        float p = sm[label];
        if (p < 1e-12f) p = 1e-12f;
        total += -logf(p);
    }
    free(sm);

    return total / (float)batch;
}

// Backward: grad[i][j] = (softmax(logits[i])[j] - (j == target[i])) / batch.
boat_tensor_t* softmax_cross_entropy_loss_backward(boat_loss_t* loss, const void* predictions,
                                                   const void* targets) {
    (void)loss;
    if (!predictions || !targets) return NULL;

    const boat_tensor_t* logits = (const boat_tensor_t*)predictions;
    const boat_tensor_t* target = (const boat_tensor_t*)targets;

    if (boat_tensor_ndim(logits) != 2 || boat_tensor_ndim(target) != 1) return NULL;
    if (boat_tensor_dtype(logits) != BOAT_DTYPE_FLOAT32) return NULL;
    boat_dtype_t tdtype = boat_tensor_dtype(target);
    if (tdtype != BOAT_DTYPE_INT32 && tdtype != BOAT_DTYPE_INT64) return NULL;

    const int64_t* logits_shape = boat_tensor_shape(logits);
    size_t batch = (size_t)logits_shape[0];
    size_t num_classes = (size_t)logits_shape[1];
    if (num_classes == 0) return NULL;
    if (boat_tensor_nelements(target) != batch) return NULL;

    boat_tensor_t* grad =
        boat_tensor_create(logits_shape, 2, BOAT_DTYPE_FLOAT32, boat_tensor_device(logits));
    if (!grad) return NULL;

    const float* l = (const float*)boat_tensor_const_data(logits);
    float* g = (float*)boat_tensor_data(grad);
    float inv_batch = 1.0f / (float)batch;

    float* sm = (float*)malloc(num_classes * sizeof(float));
    if (!sm) {
        boat_tensor_unref(grad);
        return NULL;
    }

    for (size_t b = 0; b < batch; b++) {
        softmax_row(l + b * num_classes, num_classes, sm);
        int64_t label = read_label(target, b);
        if (label < 0 || label >= (int64_t)num_classes) {
            free(sm);
            boat_tensor_unref(grad);
            return NULL;
        }
        for (size_t j = 0; j < num_classes; j++) {
            float p = sm[j];
            g[b * num_classes + j] = (p - (j == (size_t)label ? 1.0f : 0.0f)) * inv_batch;
        }
    }
    free(sm);

    return grad;
}
