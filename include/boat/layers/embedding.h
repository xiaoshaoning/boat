// embedding.h - Embedding lookup layer for token/row embeddings
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_EMBEDDING_H
#define BOAT_EMBEDDING_H

#include "../tensor.h"
#include "../export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
typedef struct boat_embedding_t boat_embedding_t;

// Create an embedding layer
// num_embeddings: vocabulary size / number of rows in the lookup table
// embedding_dim:  dimensionality of each embedding vector
BOAT_API boat_embedding_t* BOAT_CALL boat_embedding_create(size_t num_embeddings,
                                                           size_t embedding_dim);

// Free embedding layer
BOAT_API void BOAT_CALL boat_embedding_free(boat_embedding_t* emb);

// Forward pass: lookup embeddings for given indices
// indices: int32 tensor of shape [N] (token IDs)
// returns: float32 tensor of shape [N, embedding_dim]
BOAT_API boat_tensor_t* BOAT_CALL boat_embedding_forward(boat_embedding_t* emb,
                                                         const boat_tensor_t* indices);

// Set weight tensor (for model loading)
// weight: float32 tensor of shape [num_embeddings, embedding_dim]
BOAT_API void BOAT_CALL boat_embedding_set_weight(boat_embedding_t* emb, boat_tensor_t* weight);

// Get weight tensor
BOAT_API boat_tensor_t* BOAT_CALL boat_embedding_get_weight(const boat_embedding_t* emb);

// Get number of embeddings (vocab size)
BOAT_API size_t BOAT_CALL boat_embedding_num_embeddings(const boat_embedding_t* emb);

// Get embedding dimension
BOAT_API size_t BOAT_CALL boat_embedding_dim(const boat_embedding_t* emb);

#ifdef __cplusplus
}
#endif

#endif // BOAT_EMBEDDING_H
