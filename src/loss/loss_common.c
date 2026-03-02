// loss_common.c - Common loss function infrastructure and dispatch
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/loss.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdlib.h>

// Forward declarations for loss-specific compute functions
float mse_loss_compute(boat_loss_t* loss, const void* predictions, const void* targets);
float cross_entropy_loss_compute(boat_loss_t* loss, const void* predictions, const void* targets);
float huber_loss_compute(boat_loss_t* loss, const void* predictions, const void* targets);

// Generic loss structure with type
typedef struct {
    boat_loss_type_t type;
    // Can add common fields here if needed
} boat_loss_common_t;

// Dispatch compute based on loss type
float boat_loss_compute(boat_loss_t* loss, const void* predictions, const void* targets) {
    if (!loss) {
        return 0.0f;
    }

    const boat_loss_common_t* common_loss = (const boat_loss_common_t*)loss;

    switch (common_loss->type) {
        case BOAT_LOSS_MSE:
            return mse_loss_compute(loss, predictions, targets);
        case BOAT_LOSS_CROSS_ENTROPY:
            return cross_entropy_loss_compute(loss, predictions, targets);
        case BOAT_LOSS_HUBER:
            return huber_loss_compute(loss, predictions, targets);
        default:
            return 0.0f;
    }
}

// Generic free function
void boat_loss_free(boat_loss_t* loss) {
    if (!loss) {
        return;
    }

    boat_free(loss);
}