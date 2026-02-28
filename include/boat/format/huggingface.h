// huggingface.h - Hugging Face Transformers model format support
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#ifndef BOAT_HUGGINGFACE_H
#define BOAT_HUGGINGFACE_H

#include "../model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Load Hugging Face model from directory
// The directory should contain config.json and model weights (pytorch_model.bin or model.safetensors)
BOAT_API boat_model_t* boat_huggingface_load(const char* model_dir);

// Load Hugging Face model from memory buffers
// config_json: JSON configuration string
// weights_data: binary weights data (safetensors format)
// weights_size: size of weights data in bytes
BOAT_API boat_model_t* boat_huggingface_load_from_memory(const char* config_json, const void* weights_data, size_t weights_size);

// Check if directory contains a valid Hugging Face model
BOAT_API bool boat_huggingface_check(const char* model_dir);

// Get model configuration information
// Returns JSON string with model configuration (caller must free)
BOAT_API char* boat_huggingface_get_config(const char* model_dir);

// Save model to Hugging Face format directory
BOAT_API bool boat_huggingface_save(const boat_model_t* model, const char* model_dir);

#ifdef __cplusplus
}
#endif

#endif // BOAT_HUGGINGFACE_H