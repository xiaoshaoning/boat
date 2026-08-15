// data.h - Data pipeline: Dataset, DataLoader, and Transforms
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_DATA_H
#define BOAT_DATA_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include "tensor.h"
#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Dataset
// ---------------------------------------------------------------------------

typedef struct boat_dataset_t boat_dataset_t;

// Create a tensor dataset from pre-loaded data and label tensors.
// data:   [N, ...] tensor (e.g. N images)
// labels: [N] or [N, 1] tensor of label indices
// The dataset INCREMENTS ref counts on both tensors.
BOAT_API boat_dataset_t* boat_tensor_dataset_create(boat_tensor_t* data, boat_tensor_t* labels);

// Free the dataset (decrements ref counts on data/labels).
BOAT_API void boat_dataset_free(boat_dataset_t* dataset);

// Number of samples in the dataset.
BOAT_API size_t boat_dataset_size(const boat_dataset_t* dataset);

// Get the data tensor for sample at index (caller must unref).
// Returns a slice/view if possible, otherwise a copy.
BOAT_API boat_tensor_t* boat_dataset_get_data(const boat_dataset_t* dataset, size_t index);

// Get the label tensor for sample at index (caller must unref).
// Returns a 0-d or 1-element tensor containing the label.
BOAT_API boat_tensor_t* boat_dataset_get_label(const boat_dataset_t* dataset, size_t index);

// ---------------------------------------------------------------------------
// Transforms
// ---------------------------------------------------------------------------

// Transform function: receives a sample tensor, returns a (possibly new)
// tensor.  The caller owns the returned tensor.  If the transform modifies
// the input in-place, it may return the same pointer.
typedef boat_tensor_t* (*boat_transform_func_t)(boat_tensor_t* sample, void* context);

// Sequential transform chain.
typedef struct boat_transform_chain_t boat_transform_chain_t;

BOAT_API boat_transform_chain_t* boat_transform_chain_create(void);
BOAT_API void boat_transform_chain_free(boat_transform_chain_t* chain);
BOAT_API void boat_transform_chain_add(boat_transform_chain_t* chain, boat_transform_func_t fn,
                                       void* context);
// Apply all transforms in the chain sequentially.
// The input sample may be freed during the chain — the returned tensor
// is the final result (caller must unref).
BOAT_API boat_tensor_t* boat_transform_chain_apply(boat_transform_chain_t* chain,
                                                   boat_tensor_t* sample);

// --- Built-in transforms ---

// Normalize: out = (in - mean) / std (in-place on float32 tensors).
// context must point to two consecutive floats: [mean, std].
// If context is NULL, uses mean=0.0f, std=1.0f.
BOAT_API boat_tensor_t* boat_transform_normalize(boat_tensor_t* sample, void* context);

// Random horizontal flip (50% chance, in-place on float32 [C,H,W] tensors).
// context is unused (pass NULL).
BOAT_API boat_tensor_t* boat_transform_random_hflip(boat_tensor_t* sample, void* context);

// Random crop on float32 [C,H,W] tensors.
// context must point to size_t[2] { crop_h, crop_w }.
// If context is NULL, crops to (H-2, W-2).
BOAT_API boat_tensor_t* boat_transform_random_crop(boat_tensor_t* sample, void* context);

// ---------------------------------------------------------------------------
// DataLoader
// ---------------------------------------------------------------------------

typedef struct boat_dataloader_t boat_dataloader_t;

// Create a dataloader that iterates over dataset in batches.
// When shuffle is true, indices are randomly permuted each epoch.
BOAT_API boat_dataloader_t* boat_dataloader_create(boat_dataset_t* dataset, size_t batch_size,
                                                   bool shuffle);

// Free the dataloader (does NOT free the dataset).
BOAT_API void boat_dataloader_free(boat_dataloader_t* loader);

// Reset for a new epoch (re-shuffles indices if shuffle was enabled).
BOAT_API void boat_dataloader_reset(boat_dataloader_t* loader);

// Fetch the next batch.  Returns true on success, false at end of epoch.
// *batch_data and *batch_labels are newly created tensors — caller must
// unref them when done.
BOAT_API bool boat_dataloader_next(boat_dataloader_t* loader, boat_tensor_t** batch_data,
                                   boat_tensor_t** batch_labels);

// Number of batches per epoch.
BOAT_API size_t boat_dataloader_num_batches(const boat_dataloader_t* loader);

// Batch size.
BOAT_API size_t boat_dataloader_batch_size(const boat_dataloader_t* loader);

// Set a transform chain applied to each sample before batching.
// The dataloader does NOT take ownership of the chain — the caller must
// keep it alive and free it after the dataloader is destroyed.
BOAT_API void boat_dataloader_set_transform(boat_dataloader_t* loader,
                                            boat_transform_chain_t* transform);

// Current index within the epoch (0-based, resets on each boat_dataloader_reset).
BOAT_API size_t boat_dataloader_current_batch_idx(const boat_dataloader_t* loader);

#ifdef __cplusplus
}
#endif

#endif // BOAT_DATA_H
