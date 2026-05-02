// prune.h - Weight pruning and compression API
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_PRUNE_H
#define BOAT_PRUNE_H

#include "tensor.h"
#include "model.h"
#include "quantize.h"
#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Pruning configuration
typedef struct {
    float sparsity;            // Target sparsity ratio [0.0, 1.0), e.g. 0.5 = prune 50%
    bool structured;           // false = element-wise magnitude, true = structured (channel/filter)
    float threshold;           // Alternative to sparsity: prune weights with |w| < threshold. Set <=0 to compute from sparsity.
    size_t iterative_steps;    // Number of prune+finetune cycles. 1 = one-shot.
    size_t prune_dim;          // For structured: dimension along which to prune (0 = out_channels for Conv2D, 1 = output_features for Dense)
    float min_keep_ratio;      // For structured: minimum fraction of units to keep [0.0, 1.0]
} boat_prune_config_t;

// Default config: 50% sparsity, magnitude-based, one-shot
BOAT_API boat_prune_config_t boat_prune_config_default(void);

// Prune context — holds mask tensors for a model's pruned layers
typedef struct boat_prune_context_t boat_prune_context_t;

// Create a prune context for a model. Scans layers and allocates mask storage.
// Masks are initialized to all ones (no pruning).
// Returns NULL on error.
BOAT_API boat_prune_context_t* boat_prune_context_create(boat_model_t* model);

// Free a prune context and all its mask tensors.
BOAT_API void boat_prune_context_free(boat_prune_context_t* ctx);

// Get the mask tensor for a specific layer. Returns NULL if layer not pruned or no mask.
BOAT_API boat_tensor_t* boat_prune_get_mask(const boat_prune_context_t* ctx, size_t layer_index);

// Compute the absolute-value threshold for a given sparsity ratio.
// Sorts all |weight| values, picks the value at the sparsity percentile.
BOAT_API float boat_compute_prune_threshold(const boat_tensor_t* weight, float sparsity);

// Create a magnitude-based pruning mask: |w| <= threshold → 0.0, else 1.0.
// Returns a new FP32 tensor with the same shape as weight. Caller owns it.
BOAT_API boat_tensor_t* boat_create_magnitude_mask(const boat_tensor_t* weight, float threshold);

// Create a structured pruning mask: zero out entire slices along dim where L2 norm <= threshold.
// min_keep_ratio prevents pruning every slice.
// Returns a new FP32 tensor. Caller owns it.
BOAT_API boat_tensor_t* boat_create_structured_mask(const boat_tensor_t* weight, size_t dim,
                                                     float threshold, float min_keep_ratio);

// Apply a mask to a weight tensor in-place: weight[i] *= mask[i].
// Both must be FP32 with identical shape.
// Returns true on success.
BOAT_API bool boat_apply_mask(boat_tensor_t* weight, const boat_tensor_t* mask);

// Prune a single layer's weight and store the mask in the context.
// If a mask already exists for this layer, it is replaced.
BOAT_API bool boat_prune_layer(boat_prune_context_t* ctx, size_t layer_index,
                                const boat_prune_config_t* config);

// Prune all prunable layers (Dense, Conv2D) in the model according to config.
BOAT_API bool boat_prune_model(boat_prune_context_t* ctx, const boat_prune_config_t* config);

// Re-apply all masks to their respective weight tensors.
// Call this after each optimizer step during fine-tuning to keep pruned weights at zero.
BOAT_API bool boat_prune_apply_masks(const boat_prune_context_t* ctx);

// Remove the mask for a layer (weight tensor is left as-is, zeros stay zero).
BOAT_API bool boat_prune_remove_mask(boat_prune_context_t* ctx, size_t layer_index);

// Remove all masks (weights left as-is).
BOAT_API void boat_prune_remove_all_masks(boat_prune_context_t* ctx);

// Apply fake quantization (QAT) to all pruned layers' weights.
// Uses boat_fake_quantize on each weight. The model must be FP32.
BOAT_API bool boat_prune_fake_quantize_model(const boat_prune_context_t* ctx,
                                              const boat_quant_config_t* quant_config);

// Compute sparsity of a weight tensor: fraction of elements exactly zero.
BOAT_API float boat_compute_sparsity(const boat_tensor_t* weight);

// Compute structured sparsity: fraction of slices entirely zero along dim.
BOAT_API float boat_compute_structured_sparsity(const boat_tensor_t* weight, size_t dim);

#ifdef __cplusplus
}
#endif

#endif // BOAT_PRUNE_H
