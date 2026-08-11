// loss.h - Loss functions
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_LOSS_H
#define BOAT_LOSS_H

#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Loss function types
typedef enum {
    BOAT_LOSS_MSE,
    BOAT_LOSS_CROSS_ENTROPY,
    BOAT_LOSS_HUBER
} boat_loss_type_t;

#include <boat/tensor.h>

// Loss structure (opaque)
typedef struct boat_loss_t boat_loss_t;

// Create loss functions
BOAT_API boat_loss_t* boat_mse_loss_create();
BOAT_API boat_loss_t* boat_cross_entropy_loss_create();
BOAT_API boat_loss_t* boat_huber_loss_create(float delta);

// Loss operations
BOAT_API float boat_loss_compute(boat_loss_t* loss, const void* predictions, const void* targets);
BOAT_API boat_tensor_t* boat_loss_backward(boat_loss_t* loss, const void* predictions, const void* targets);
BOAT_API void boat_loss_free(boat_loss_t* loss);

#ifdef __cplusplus
}
#endif

#endif // BOAT_LOSS_H