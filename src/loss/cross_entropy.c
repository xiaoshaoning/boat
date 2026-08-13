// cross_entropy.c - Cross Entropy loss function
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/loss.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifdef BOAT_WITH_CUDA
#include <boat/cuda_runtime.h>
#endif

// Forward declaration for dispatch
float cross_entropy_loss_compute(boat_loss_t* loss_ptr, const void* predictions_ptr, const void* targets_ptr);

// Cross entropy loss structure
typedef struct {
    boat_loss_type_t type;  // Always BOAT_LOSS_CROSS_ENTROPY
} cross_entropy_loss_t;

// Create cross entropy loss function
BOAT_API boat_loss_t* boat_cross_entropy_loss_create() {
    cross_entropy_loss_t* loss = (cross_entropy_loss_t*)boat_malloc(sizeof(cross_entropy_loss_t), BOAT_DEVICE_CPU);
    if (!loss) {
        return NULL;
    }

    loss->type = BOAT_LOSS_CROSS_ENTROPY;

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

    // Predictions are expected to be probabilities (already softmaxed), and
    // targets are one-hot encoded. Cross entropy = -mean(target * log(pred)).
    const float* pred_data = (const float*)boat_tensor_data(predictions);
    const float* target_data = (const float*)boat_tensor_data(targets);
    size_t num_elements = boat_tensor_nelements(predictions);
    if (num_elements == 0) {
        return 0.0f;
    }

    float sum_loss = 0.0f;
    float epsilon = 1e-7f;

    for (size_t i = 0; i < num_elements; i++) {
        float pred_clipped = clip_for_log(pred_data[i], epsilon);
        sum_loss += target_data[i] * logf(pred_clipped);
    }

    float cross_entropy = -sum_loss / (float)num_elements;

    return cross_entropy;
}

// Compute cross entropy backward gradient
boat_tensor_t* cross_entropy_loss_backward(boat_loss_t* loss_ptr, const void* predictions_ptr, const void* targets_ptr) {
    if (!loss_ptr || !predictions_ptr || !targets_ptr) {
        return NULL;
    }

    const boat_tensor_t* predictions = (const boat_tensor_t*)predictions_ptr;
    const boat_tensor_t* targets = (const boat_tensor_t*)targets_ptr;

    // Verify tensors have same shape and dtype
    if (boat_tensor_ndim(predictions) != boat_tensor_ndim(targets)) {
        return NULL;
    }

    size_t ndim = boat_tensor_ndim(predictions);
    const int64_t* pred_shape = boat_tensor_shape(predictions);
    const int64_t* target_shape = boat_tensor_shape(targets);

    for (size_t i = 0; i < ndim; i++) {
        if (pred_shape[i] != target_shape[i]) {
            return NULL;
        }
    }

    if (boat_tensor_dtype(predictions) != boat_tensor_dtype(targets)) {
        return NULL;
    }

    if (boat_tensor_dtype(predictions) != BOAT_DTYPE_FLOAT32) {
        return NULL;
    }

    size_t num_elements = boat_tensor_nelements(predictions);
    const float* pred_data = (const float*)boat_tensor_data(predictions);
    const float* target_data = (const float*)boat_tensor_data(targets);

    boat_tensor_t* grad = boat_tensor_create_like(predictions);
    if (!grad) {
        return NULL;
    }

    float* grad_data = (float*)boat_tensor_data(grad);
    float epsilon = 1e-7f;
    float inv_n = 1.0f / (float)num_elements;

    for (size_t i = 0; i < num_elements; i++) {
        float pred_clipped = clip_for_log(pred_data[i], epsilon);
        grad_data[i] = -inv_n * target_data[i] / pred_clipped;
    }

    return grad;
}

// Note: This function has a different name to avoid conflict with boat_loss_compute
// We need to implement a dispatch mechanism. For now, we'll create a wrapper.
// Actually, we need to modify the loss.h API or create a unified dispatch.
// Let's create a simple dispatch in a separate common file later.

