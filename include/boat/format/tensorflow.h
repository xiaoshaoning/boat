// tensorflow.h - TensorFlow model format support
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#ifndef BOAT_TENSORFLOW_H
#define BOAT_TENSORFLOW_H

#include "../model.h"

#ifdef __cplusplus
extern "C" {
#endif

// Load TensorFlow model from file
boat_model_t* boat_tensorflow_load(const char* filename);

// Save model to TensorFlow format
bool boat_tensorflow_save(const boat_model_t* model, const char* filename);

// Load TensorFlow model from memory buffer
boat_model_t* boat_tensorflow_load_from_memory(const void* data, size_t size);

// Save model to memory buffer in TensorFlow format
bool boat_tensorflow_save_to_memory(const boat_model_t* model, void** data, size_t* size);

// Check if file is a valid TensorFlow model
bool boat_tensorflow_check(const char* filename);

// Load TensorFlow SavedModel directory
boat_model_t* boat_tensorflow_load_savedmodel(const char* directory);

// Load TensorFlow frozen graph
boat_model_t* boat_tensorflow_load_frozen_graph(const char* filename);

#ifdef __cplusplus
}
#endif

#endif // BOAT_TENSORFLOW_H