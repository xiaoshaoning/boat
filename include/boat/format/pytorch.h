// pytorch.h - PyTorch model format support
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_PYTORCH_H
#define BOAT_PYTORCH_H

#include "../model.h"
#include "../export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Load PyTorch model from file
BOAT_API boat_model_t* boat_pytorch_load(const char* filename);

// Save model to PyTorch format
BOAT_API bool boat_pytorch_save(const boat_model_t* model, const char* filename);

// Load PyTorch model from memory buffer
BOAT_API boat_model_t* boat_pytorch_load_from_memory(const void* data, size_t size);

// Save model to memory buffer in PyTorch format
BOAT_API bool boat_pytorch_save_to_memory(const boat_model_t* model, void** data, size_t* size);

// Check if file is a valid PyTorch model
BOAT_API bool boat_pytorch_check(const char* filename);

// Convert PyTorch model to Boat model with specific device
BOAT_API boat_model_t* boat_pytorch_convert(const char* filename, boat_device_t device);

#ifdef __cplusplus
}
#endif

#endif // BOAT_PYTORCH_H