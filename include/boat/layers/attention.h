// attention.h - Attention mechanisms for transformer models
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_ATTENTION_H
#define BOAT_ATTENTION_H

#include "../tensor.h"
#include "../export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
typedef struct boat_attention_t boat_attention_t;

// Attention configuration
typedef struct {
    size_t hidden_size;       // Hidden size (d_model)
    size_t num_heads;         // Number of attention heads
    size_t head_size;         // Size of each head (hidden_size / num_heads)
    float dropout_prob;       // Dropout probability (0.0 for no dropout)
    bool causal_mask;         // Whether to apply causal masking (for autoregressive models)
    bool use_bias;            // Whether to use bias in linear projections
    bool use_rotary;          // Whether to use rotary position encoding (RoPE)
    float rotary_theta;       // Rotary encoding base theta (default 10000.0)
} boat_attention_config_t;

// Create attention layer
BOAT_API boat_attention_t* BOAT_CALL boat_attention_create(const boat_attention_config_t* config);

// Free attention layer
BOAT_API void BOAT_CALL boat_attention_free(const boat_attention_t* attention);

// Forward pass
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_forward(const boat_attention_t* attention,
                                       const boat_tensor_t* query,
                                       const boat_tensor_t* key,
                                       const boat_tensor_t* value,
                                       const boat_tensor_t* attention_mask);

// Backward pass (for training)
// Returns gradient tensors for query, key, value inputs via output parameters.
// Returns true on success, false on failure.
// Parameter gradients are stored in the attention layer and can be accessed via accessor functions.
BOAT_API bool BOAT_CALL boat_attention_backward(const boat_attention_t* attention,
                                        const boat_tensor_t* grad_output,
                                        boat_tensor_t** grad_query,
                                        boat_tensor_t** grad_key,
                                        boat_tensor_t** grad_value);

// Update parameters (for training)
BOAT_API void BOAT_CALL boat_attention_update(const boat_attention_t* attention, float learning_rate);

// Set dropout probability
BOAT_API void BOAT_CALL boat_attention_set_dropout(const boat_attention_t* attention, float dropout_prob);

// Enable/disable causal masking
BOAT_API void BOAT_CALL boat_attention_set_causal(const boat_attention_t* attention, bool causal);

// Multi-head attention (simplified API for self-attention)
BOAT_API boat_tensor_t* BOAT_CALL boat_multihead_attention(const boat_tensor_t* input,
                                         size_t num_heads,
                                         float dropout_prob,
                                         bool causal_mask,
                                         const boat_tensor_t* attention_mask);

// Scaled dot-product attention (low-level function)
BOAT_API boat_tensor_t* BOAT_CALL boat_scaled_dot_product_attention(const boat_tensor_t* query,
                                                  const boat_tensor_t* key,
                                                  const boat_tensor_t* value,
                                                  float scale_factor,
                                                  const boat_tensor_t* attention_mask,
                                                  bool causal_mask,
                                                  float dropout_prob);

// Rotary position encoding (RoPE)
BOAT_API boat_tensor_t* BOAT_CALL boat_rotary_position_encoding(const boat_tensor_t* tensor,
                                              size_t seq_len,
                                              size_t head_size,
                                              float theta);

// Apply rotary embedding to query and key
BOAT_API void BOAT_CALL boat_apply_rotary_embedding(boat_tensor_t* query,
                                  boat_tensor_t* key,
                                  size_t seq_len,
                                  size_t head_size,
                                  float theta);

// Accessor functions for testing (get weights and gradients)
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_get_weight_q(const boat_attention_t* attention);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_get_weight_k(const boat_attention_t* attention);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_get_weight_v(const boat_attention_t* attention);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_get_weight_o(const boat_attention_t* attention);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_get_bias_q(const boat_attention_t* attention);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_get_bias_k(const boat_attention_t* attention);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_get_bias_v(const boat_attention_t* attention);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_get_bias_o(const boat_attention_t* attention);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_get_grad_weight_q(const boat_attention_t* attention);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_get_grad_weight_k(const boat_attention_t* attention);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_get_grad_weight_v(const boat_attention_t* attention);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_get_grad_weight_o(const boat_attention_t* attention);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_get_grad_bias_q(const boat_attention_t* attention);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_get_grad_bias_k(const boat_attention_t* attention);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_get_grad_bias_v(const boat_attention_t* attention);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_get_grad_bias_o(const boat_attention_t* attention);

#ifdef __cplusplus
}
#endif

#endif // BOAT_ATTENTION_H