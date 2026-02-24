// mse.c - Mean Squared Error loss function
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/loss.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Forward declaration for dispatch
float mse_loss_compute(boat_loss_t* loss_ptr, const void* predictions_ptr, const void* targets_ptr);

// MSE loss structure
typedef struct {
    boat_loss_type_t type;  // Always BOAT_LOSS_MSE
    float sum;              // Accumulated sum for batch averaging
    int count;              // Number of accumulated elements
} mse_loss_t;

// Create MSE loss function
boat_loss_t* boat_mse_loss_create() {
    mse_loss_t* loss = (mse_loss_t*)boat_malloc(sizeof(mse_loss_t), BOAT_DEVICE_CPU);
    if (!loss) {
        return NULL;
    }

    loss->type = BOAT_LOSS_MSE;
    loss->sum = 0.0f;
    loss->count = 0;

    return (boat_loss_t*)loss;
}

// Compute MSE loss between predictions and targets
float mse_loss_compute(boat_loss_t* loss_ptr, const void* predictions_ptr, const void* targets_ptr) {
    if (!loss_ptr || !predictions_ptr || !targets_ptr) {
        return 0.0f;
    }

    mse_loss_t* loss = (mse_loss_t*)loss_ptr;
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

    // Compute MSE: average of (pred - target)^2
    const float* pred_data = (const float*)boat_tensor_data(predictions);
    const float* target_data = (const float*)boat_tensor_data(targets);
    size_t num_elements = boat_tensor_nbytes(predictions) / sizeof(float);

    float sum_squared_error = 0.0f;
    for (size_t i = 0; i < num_elements; i++) {
        float diff = pred_data[i] - target_data[i];
        sum_squared_error += diff * diff;
    }

    float mse = sum_squared_error / num_elements;

    // Update accumulated stats
    loss->sum += sum_squared_error;
    loss->count += (int)num_elements;

    return mse;
}

