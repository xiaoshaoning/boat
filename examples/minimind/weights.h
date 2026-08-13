// weights.h - MiniMind weight loading
#pragma once
#include "model.h"

// Load all weights from model_dir (contains model.bin + model_meta.json).
// Also precomputes RoPE tables, allocates KV caches and working buffers.
// Returns 0 on success, -1 on error.
int minimind_weights_load(minimind_model_t* m, const char* model_dir);

// Free all allocated memory.
void minimind_weights_free(minimind_model_t* m);
