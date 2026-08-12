// huber.c - Huber loss function
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/loss.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Huber loss structure
typedef struct {
    boat_loss_type_t type;  // Always BOAT_LOSS_HUBER
    float delta;            // Huber delta parameter
    float sum;              // Accumulated sum for batch averaging
    int count;              // Number of accumulated elements
} huber_loss_t;

// Forward declaration for dispatch
float huber_loss_compute(boat_loss_t* loss_ptr, const void* predictions_ptr, const void* targets_ptr);

// Create Huber loss function with specified delta
BOAT_API boat_loss_t* BOAT_API boat_huber_loss_create(float delta) {
    if (delta <= 0.0f) {
        return NULL;
    }

    huber_loss_t* loss = (huber_loss_t*)boat_malloc(sizeof(huber_loss_t), BOAT_DEVICE_CPU);
    if (!loss) {
        return NULL;
    }

    loss->type = BOAT_LOSS_HUBER;
    loss->delta = delta;
    loss->sum = 0.0f;
    loss->count = 0;

    return (boat_loss_t*)loss;
}

// Compute Huber loss between predictions and targets
float huber_loss_compute(boat_loss_t* loss_ptr, const void* predictions_ptr, const void* targets_ptr) {
    if (!loss_ptr || !predictions_ptr || !targets_ptr) {
        return 0.0f;
    }

    huber_loss_t* loss = (huber_loss_t*)loss_ptr;
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

    const float* pred_data = (const float*)boat_tensor_data(predictions);
    const float* target_data = (const float*)boat_tensor_data(targets);
    size_t num_elements = boat_tensor_nbytes(predictions) / sizeof(float);
    float delta = loss->delta;

    float sum_loss = 0.0f;
    for (size_t i = 0; i < num_elements; i++) {
        float diff = fabsf(pred_data[i] - target_data[i]);
        if (diff <= delta) {
            // Quadratic part
            sum_loss += 0.5f * diff * diff;
        } else {
            // Linear part
            sum_loss += delta * (diff - 0.5f * delta);
        }
    }

    float huber = sum_loss / num_elements;

    // Update accumulated stats
    loss->sum += sum_loss;
    loss->count += num_elements;

    return huber;
}

// Compute Huber backward gradient
boat_tensor_t* huber_loss_backward(boat_loss_t* loss_ptr, const void* predictions_ptr, const void* targets_ptr) {
    if (!loss_ptr || !predictions_ptr || !targets_ptr) {
        return NULL;
    }

    huber_loss_t* loss = (huber_loss_t*)loss_ptr;
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
    float delta = loss->delta;

    boat_tensor_t* grad = boat_tensor_create_like(predictions);
    if (!grad) {
        return NULL;
    }

    float* grad_data = (float*)boat_tensor_data(grad);
    float inv_n = 1.0f / (float)num_elements;

    for (size_t i = 0; i < num_elements; i++) {
        float diff = pred_data[i] - target_data[i];
        float abs_diff = fabsf(diff);
        if (abs_diff <= delta) {
            grad_data[i] = diff * inv_n;
        } else if (diff > 0.0f) {
            grad_data[i] = delta * inv_n;
        } else {
            grad_data[i] = -delta * inv_n;
        }
    }

    return grad;
}