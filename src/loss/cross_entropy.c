// cross_entropy.c - Cross Entropy loss function
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/loss.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Forward declaration for dispatch
float cross_entropy_loss_compute(boat_loss_t* loss_ptr, const void* predictions_ptr, const void* targets_ptr);

// Cross entropy loss structure
typedef struct {
    boat_loss_type_t type;  // Always BOAT_LOSS_CROSS_ENTROPY
    float sum;              // Accumulated sum for batch averaging
    int count;              // Number of accumulated elements
} cross_entropy_loss_t;

// Create cross entropy loss function
boat_loss_t* boat_cross_entropy_loss_create() {
    cross_entropy_loss_t* loss = (cross_entropy_loss_t*)boat_malloc(sizeof(cross_entropy_loss_t), BOAT_DEVICE_CPU);
    if (!loss) {
        return NULL;
    }

    loss->type = BOAT_LOSS_CROSS_ENTROPY;
    loss->sum = 0.0f;
    loss->count = 0;

    return (boat_loss_t*)loss;
}

// Helper function: clip value to avoid log(0)
static float clip_for_log(float value, float epsilon) {
    if (value < epsilon) {
        return epsilon;
    }
    if (value > 1.0f - epsilon) {
        return 1.0f - epsilon;
    }
    return value;
}

// Compute cross entropy loss between predictions and targets
float cross_entropy_loss_compute(boat_loss_t* loss_ptr, const void* predictions_ptr, const void* targets_ptr) {
    if (!loss_ptr || !predictions_ptr || !targets_ptr) {
        return 0.0f;
    }

    cross_entropy_loss_t* loss = (cross_entropy_loss_t*)loss_ptr;
    const boat_tensor_t* predictions = (const boat_tensor_t*)predictions_ptr;
    const boat_tensor_t* targets = (const boat_tensor_t*)targets_ptr;

    // Verify tensors have same shape and dtype
    if (boat_tensor_ndim(predictions) != boat_tensor_ndim(targets)) {
        return 0.0f;
    }

    size_t ndim = boat_tensor_ndim(predictions);
    const int64_t* pred_shape = boat_tensor_shape(predictions);
    const int64_t* target_shape = boat_tensor_shape(targets);

    for (size_t i = 0; i < ndim; i++) {
        if (pred_shape[i] != target_shape[i]) {
            return 0.0f;
        }
    }

    if (boat_tensor_dtype(predictions) != boat_tensor_dtype(targets)) {
        return 0.0f;
    }

    // Only support float32 for now
    if (boat_tensor_dtype(predictions) != BOAT_DTYPE_FLOAT32) {
        return 0.0f;
    }

    // For simplicity, assume predictions are logits (not probabilities)
    // and targets are one-hot encoded
    const float* pred_data = (const float*)boat_tensor_data(predictions);
    const float* target_data = (const float*)boat_tensor_data(targets);
    size_t num_elements = boat_tensor_nbytes(predictions) / sizeof(float);

    // Simple cross entropy: -sum(target * log(softmax(pred)))
    // For now, use a simplified version: -sum(target * log(clip(pred)))
    float sum_loss = 0.0f;
    float epsilon = 1e-7f;

    for (size_t i = 0; i < num_elements; i++) {
        float pred_clipped = clip_for_log(pred_data[i], epsilon);
        sum_loss += target_data[i] * logf(pred_clipped);
    }

    float cross_entropy = -sum_loss / num_elements;

    // Update accumulated stats
    loss->sum += sum_loss;
    loss->count += num_elements;

    return cross_entropy;
}

// Note: This function has a different name to avoid conflict with boat_loss_compute
// We need to implement a dispatch mechanism. For now, we'll create a wrapper.
// Actually, we need to modify the loss.h API or create a unified dispatch.
// Let's create a simple dispatch in a separate common file later.

