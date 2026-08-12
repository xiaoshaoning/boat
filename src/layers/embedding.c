// embedding.c - Embedding lookup layer implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers/embedding.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Embedding layer structure
struct boat_embedding_t {
    size_t num_embeddings;   // Vocab size / number of rows
    size_t embedding_dim;    // Dimensionality of each embedding
    boat_tensor_t* weight;   // Weight tensor [num_embeddings, embedding_dim]
};

BOAT_API BOAT_API boat_embedding_t* BOAT_CALL boat_embedding_create(size_t num_embeddings, size_t embedding_dim) {
    if (num_embeddings == 0 || embedding_dim == 0) {
        return NULL;
    }

    boat_embedding_t* emb = (boat_embedding_t*)boat_malloc(sizeof(boat_embedding_t), BOAT_DEVICE_CPU);
    if (!emb) {
        return NULL;
    }

    emb->num_embeddings = num_embeddings;
    emb->embedding_dim = embedding_dim;

    // Create weight tensor initialized to zeros
    const int64_t shape[] = { (int64_t)num_embeddings, (int64_t)embedding_dim };
    emb->weight = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!emb->weight) {
        boat_free(emb);
        return NULL;
    }

    return emb;
}

BOAT_API BOAT_API void BOAT_CALL boat_embedding_free(boat_embedding_t* emb) {
    if (!emb) {
        return;
    }
    if (emb->weight) {
        boat_tensor_free(emb->weight);
    }
    boat_free(emb);
}

BOAT_API BOAT_API boat_tensor_t* BOAT_CALL boat_embedding_forward(boat_embedding_t* emb, const boat_tensor_t* indices) {
    if (!emb || !emb->weight || !indices) {
        return NULL;
    }

    // Validate indices dtype is int32
    if (boat_tensor_dtype(indices) != BOAT_DTYPE_INT32) {
        return NULL;
    }

    // Get indices shape and data
    size_t ndim = boat_tensor_ndim(indices);
    size_t num_indices = (size_t)boat_tensor_nelements(indices);

    // Create output tensor [num_indices, embedding_dim]
    const int64_t out_shape[] = { (int64_t)num_indices, (int64_t)emb->embedding_dim };
    boat_tensor_t* output = boat_tensor_create(out_shape, 2, BOAT_DTYPE_FLOAT32,
                                                boat_tensor_device(indices));
    if (!output) {
        return NULL;
    }

    // Gather rows from weight
    const int32_t* idx_data = (const int32_t*)boat_tensor_data(indices);
    const float* weight_data = (const float*)boat_tensor_data(emb->weight);
    float* out_data = (float*)boat_tensor_data(output);

    size_t dim = emb->embedding_dim;
    for (size_t i = 0; i < num_indices; i++) {
        int32_t token = idx_data[i];
        if (token >= 0 && token < (int32_t)emb->num_embeddings) {
            memcpy(out_data + i * dim, weight_data + (size_t)token * dim, dim * sizeof(float));
        } else {
            // Out-of-range token: zero-fill
            memset(out_data + i * dim, 0, dim * sizeof(float));
        }
    }

    return output;
}

BOAT_API BOAT_API void BOAT_CALL boat_embedding_set_weight(boat_embedding_t* emb, boat_tensor_t* weight) {
    if (!emb || !weight) {
        return;
    }

    // Validate shape
    size_t ndim = boat_tensor_ndim(weight);
    const int64_t* shape = boat_tensor_shape(weight);

    if (ndim != 2 || (size_t)shape[0] != emb->num_embeddings || (size_t)shape[1] != emb->embedding_dim) {
        return;
    }

    // Replace weight (free old, ref new)
    if (emb->weight) {
        boat_tensor_free(emb->weight);
    }
    emb->weight = weight;
    boat_tensor_ref(weight);
}

BOAT_API BOAT_API boat_tensor_t* BOAT_CALL boat_embedding_get_weight(const boat_embedding_t* emb) {
    if (!emb) {
        return NULL;
    }
    return emb->weight;
}

BOAT_API BOAT_API size_t BOAT_CALL boat_embedding_num_embeddings(const boat_embedding_t* emb) {
    if (!emb) return 0;
    return emb->num_embeddings;
}

BOAT_API BOAT_API size_t BOAT_CALL boat_embedding_dim(const boat_embedding_t* emb) {
    if (!emb) return 0;
    return emb->embedding_dim;
}
