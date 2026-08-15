// attention.c - Attention mechanisms implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/layers/attention.h>
#include <boat/export.h>
#ifdef BOAT_BUILDING_DLL
#pragma message("BOAT_BUILDING_DLL is defined")
#else
#pragma message("BOAT_BUILDING_DLL is NOT defined")
#endif
#include <boat/ops.h>
#include <boat/memory.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

// Forward declaration for internal implementation
static boat_tensor_t*
scaled_dot_product_attention_impl(const boat_tensor_t* query, const boat_tensor_t* key,
                                  const boat_tensor_t* value, float scale_factor,
                                  const boat_tensor_t* attention_mask, bool causal_mask,
                                  float dropout_prob, boat_tensor_t** cache_weights);

// Gradient helper functions
static bool linear_projection_backward(const boat_tensor_t* input, const boat_tensor_t* weight,
                                       const boat_tensor_t* bias, const boat_tensor_t* grad_output,
                                       boat_tensor_t** grad_input, boat_tensor_t** grad_weight,
                                       boat_tensor_t** grad_bias);
static bool attention_backward(const boat_tensor_t* query, const boat_tensor_t* key,
                               const boat_tensor_t* value, const boat_tensor_t* attention_weights,
                               const boat_tensor_t* grad_output, boat_tensor_t** grad_query,
                               boat_tensor_t** grad_key, boat_tensor_t** grad_value);

// Attention layer structure
struct boat_attention_t {
    boat_attention_config_t config;

    // Weight matrices: W_q, W_k, W_v, W_o
    boat_tensor_t* weight_q;
    boat_tensor_t* weight_k;
    boat_tensor_t* weight_v;
    boat_tensor_t* weight_o;

    // Bias vectors (optional)
    boat_tensor_t* bias_q;
    boat_tensor_t* bias_k;
    boat_tensor_t* bias_v;
    boat_tensor_t* bias_o;

    // Gradient accumulators for training
    boat_tensor_t* grad_weight_q;
    boat_tensor_t* grad_weight_k;
    boat_tensor_t* grad_weight_v;
    boat_tensor_t* grad_weight_o;
    boat_tensor_t* grad_bias_q;
    boat_tensor_t* grad_bias_k;
    boat_tensor_t* grad_bias_v;
    boat_tensor_t* grad_bias_o;

    // Dropout mask (for training)
    boat_tensor_t* dropout_mask;

    // Cache for backward pass
    boat_tensor_t* cache_query;
    boat_tensor_t* cache_key;
    boat_tensor_t* cache_value;
    boat_tensor_t* cache_q_proj;
    boat_tensor_t* cache_k_proj;
    boat_tensor_t* cache_v_proj;
    boat_tensor_t* cache_attention_weights;
    boat_tensor_t* cache_attention_output; // Attention output before final projection
};

// Helper function to create linear projection weights
static boat_tensor_t* create_linear_weights(size_t in_features, size_t out_features,
                                            bool use_bias) {
    (void)use_bias;
    const int64_t weight_shape[] = {(int64_t)in_features, (int64_t)out_features};
    boat_tensor_t* weights =
        boat_tensor_create(weight_shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

    if (!weights) {
        return NULL;
    }

    // Initialize with Xavier/Glorot initialization
    float* data = (float*)boat_tensor_data(weights);
    size_t num_elements = boat_tensor_nelements(weights);
    float scale = sqrtf(2.0f / (in_features + out_features));

    for (size_t i = 0; i < num_elements; i++) {
        data[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }

    return weights;
}

// Helper function to create bias vector
static boat_tensor_t* create_bias_vector(size_t features) {
    const int64_t bias_shape[] = {(int64_t)features};
    boat_tensor_t* bias = boat_tensor_create(bias_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

    if (!bias) {
        return NULL;
    }

    // Initialize with zeros
    float* data = (float*)boat_tensor_data(bias);
    size_t num_elements = boat_tensor_nelements(bias);
    memset(data, 0, num_elements * sizeof(float));

    return bias;
}

BOAT_API boat_attention_t* BOAT_CALL boat_attention_create(const boat_attention_config_t* config) {
    if (!config || config->hidden_size == 0 || config->num_heads == 0) {
        return NULL;
    }

    if (config->hidden_size % config->num_heads != 0) {
        // Hidden size must be divisible by number of heads
        return NULL;
    }

    boat_attention_t* attention =
        (boat_attention_t*)boat_malloc(sizeof(boat_attention_t), BOAT_DEVICE_CPU);
    if (!attention) {
        return NULL;
    }

    // Copy configuration
    memcpy(&attention->config, config, sizeof(boat_attention_config_t));

    // Calculate head size if not provided
    if (config->head_size == 0) {
        attention->config.head_size = config->hidden_size / config->num_heads;
    }
    // Default: GQA disabled (num_kv_heads = num_heads)
    if (attention->config.num_kv_heads == 0) {
        attention->config.num_kv_heads = attention->config.num_heads;
    }

    // Create weight matrices
    size_t hidden_size = attention->config.hidden_size;
    size_t kv_hidden = attention->config.num_kv_heads * attention->config.head_size;

    attention->weight_q = create_linear_weights(hidden_size, hidden_size, config->use_bias);
    attention->weight_k = create_linear_weights(hidden_size, kv_hidden, config->use_bias);
    attention->weight_v = create_linear_weights(hidden_size, kv_hidden, config->use_bias);
    attention->weight_o = create_linear_weights(hidden_size, hidden_size, config->use_bias);

    if (!attention->weight_q || !attention->weight_k || !attention->weight_v ||
        !attention->weight_o) {
        boat_attention_free(attention);
        return NULL;
    }

    // Create bias vectors if requested
    if (config->use_bias) {
        attention->bias_q = create_bias_vector(hidden_size);
        attention->bias_k = create_bias_vector(hidden_size);
        attention->bias_v = create_bias_vector(hidden_size);
        attention->bias_o = create_bias_vector(hidden_size);

        if (!attention->bias_q || !attention->bias_k || !attention->bias_v || !attention->bias_o) {
            boat_attention_free(attention);
            return NULL;
        }
    } else {
        attention->bias_q = NULL;
        attention->bias_k = NULL;
        attention->bias_v = NULL;
        attention->bias_o = NULL;
    }

    // Initialize gradient accumulators to NULL
    attention->grad_weight_q = NULL;
    attention->grad_weight_k = NULL;
    attention->grad_weight_v = NULL;
    attention->grad_weight_o = NULL;
    attention->grad_bias_q = NULL;
    attention->grad_bias_k = NULL;
    attention->grad_bias_v = NULL;
    attention->grad_bias_o = NULL;

    // Initialize cache pointers to NULL
    attention->dropout_mask = NULL;
    attention->cache_query = NULL;
    attention->cache_key = NULL;
    attention->cache_value = NULL;
    attention->cache_q_proj = NULL;
    attention->cache_k_proj = NULL;
    attention->cache_v_proj = NULL;
    attention->cache_attention_weights = NULL;
    attention->cache_attention_output = NULL;

    return attention;
}

BOAT_API void BOAT_CALL boat_attention_free(boat_attention_t* attention) {
    if (!attention) {
        return;
    }

    // Free weight matrices
    if (attention->weight_q) boat_tensor_free(attention->weight_q);
    if (attention->weight_k) boat_tensor_free(attention->weight_k);
    if (attention->weight_v) boat_tensor_free(attention->weight_v);
    if (attention->weight_o) boat_tensor_free(attention->weight_o);

    // Free bias vectors
    if (attention->bias_q) boat_tensor_free(attention->bias_q);
    if (attention->bias_k) boat_tensor_free(attention->bias_k);
    if (attention->bias_v) boat_tensor_free(attention->bias_v);
    if (attention->bias_o) boat_tensor_free(attention->bias_o);

    // Free gradient accumulators
    if (attention->grad_weight_q) boat_tensor_free(attention->grad_weight_q);
    if (attention->grad_weight_k) boat_tensor_free(attention->grad_weight_k);
    if (attention->grad_weight_v) boat_tensor_free(attention->grad_weight_v);
    if (attention->grad_weight_o) boat_tensor_free(attention->grad_weight_o);
    if (attention->grad_bias_q) boat_tensor_free(attention->grad_bias_q);
    if (attention->grad_bias_k) boat_tensor_free(attention->grad_bias_k);
    if (attention->grad_bias_v) boat_tensor_free(attention->grad_bias_v);
    if (attention->grad_bias_o) boat_tensor_free(attention->grad_bias_o);

    // Free cache tensors
    if (attention->dropout_mask) boat_tensor_free(attention->dropout_mask);
    if (attention->cache_query) boat_tensor_free(attention->cache_query);
    if (attention->cache_key) boat_tensor_free(attention->cache_key);
    if (attention->cache_value) boat_tensor_free(attention->cache_value);
    if (attention->cache_q_proj) boat_tensor_free(attention->cache_q_proj);
    if (attention->cache_k_proj) boat_tensor_free(attention->cache_k_proj);
    if (attention->cache_v_proj) boat_tensor_free(attention->cache_v_proj);
    if (attention->cache_attention_weights) boat_tensor_free(attention->cache_attention_weights);
    if (attention->cache_attention_output) boat_tensor_free(attention->cache_attention_output);

    // Free attention structure
    boat_free(attention);
}

// Helper function for linear projection
static boat_tensor_t* linear_projection(const boat_tensor_t* input, const boat_tensor_t* weight,
                                        const boat_tensor_t* bias) {

    boat_tensor_t* projected = NULL;
    boat_tensor_t* input_reshaped = NULL;
    boat_tensor_t* projected_reshaped = NULL;

    size_t input_ndim = boat_tensor_ndim(input);
    size_t weight_ndim = boat_tensor_ndim(weight);

    // Handle 3D input with 2D weight (common case for attention layers)
    if (input_ndim == 3 && weight_ndim == 2) {
        // Reshape input from [batch, seq_len, hidden] to [batch*seq_len, hidden]
        const int64_t* input_shape = boat_tensor_shape(input);
        int64_t batch = input_shape[0];
        int64_t seq_len = input_shape[1];
        int64_t hidden = input_shape[2];

        const int64_t reshaped_shape[] = {batch * seq_len, hidden};
        input_reshaped = boat_tensor_reshape(input, reshaped_shape, 2);
        if (!input_reshaped) {
            return NULL;
        }

        // Perform matrix multiplication on 2D tensors
        projected = boat_matmul(input_reshaped, weight);
        boat_tensor_unref(input_reshaped);

        if (!projected) {
            return NULL;
        }

        // Reshape result back to 3D: [batch*seq_len, hidden] -> [batch, seq_len, hidden]
        const int64_t original_shape[] = {batch, seq_len, hidden};
        projected_reshaped = boat_tensor_reshape(projected, original_shape, 3);
        boat_tensor_unref(projected);
        if (!projected_reshaped) {
            return NULL;
        }
        projected = projected_reshaped;
    } else {
        // Use direct matrix multiplication for other cases
        projected = boat_matmul(input, weight);
        if (!projected) {
            return NULL;
        }
    }

    // Add bias if provided
    if (bias) {
        boat_tensor_t* biased = boat_add(projected, bias);
        if (!biased) {
            boat_tensor_free(projected);
            return NULL;
        }
        boat_tensor_free(projected);
        projected = biased;
    }

    return projected;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_attention_forward(boat_attention_t* attention,
                                                         const boat_tensor_t* query,
                                                         const boat_tensor_t* key,
                                                         const boat_tensor_t* value,
                                                         const boat_tensor_t* attention_mask) {
    if (!attention || !query || !key || !value) {
        return NULL;
    }

    // Store inputs for backward pass (if needed for training)
    // Clear old cache first
    if (attention->cache_query) {
        boat_tensor_free(attention->cache_query);
        attention->cache_query = NULL;
    }
    if (attention->cache_key) {
        boat_tensor_free(attention->cache_key);
        attention->cache_key = NULL;
    }
    if (attention->cache_value) {
        boat_tensor_free(attention->cache_value);
        attention->cache_value = NULL;
    }
    if (attention->cache_attention_weights) {
        boat_tensor_free(attention->cache_attention_weights);
        attention->cache_attention_weights = NULL;
    }
    if (attention->cache_q_proj) {
        boat_tensor_free(attention->cache_q_proj);
        attention->cache_q_proj = NULL;
    }
    if (attention->cache_k_proj) {
        boat_tensor_free(attention->cache_k_proj);
        attention->cache_k_proj = NULL;
    }
    if (attention->cache_v_proj) {
        boat_tensor_free(attention->cache_v_proj);
        attention->cache_v_proj = NULL;
    }
    if (attention->cache_attention_output) {
        boat_tensor_free(attention->cache_attention_output);
        attention->cache_attention_output = NULL;
    }

    // Store input tensors (increase ref count)
    attention->cache_query = (boat_tensor_t*)query;
    attention->cache_key = (boat_tensor_t*)key;
    attention->cache_value = (boat_tensor_t*)value;
    boat_tensor_ref(attention->cache_query);
    boat_tensor_ref(attention->cache_key);
    boat_tensor_ref(attention->cache_value);

    // Project inputs to query, key, value spaces
    boat_tensor_t* q_proj = linear_projection(query, attention->weight_q, attention->bias_q);
    boat_tensor_t* k_proj = linear_projection(key, attention->weight_k, attention->bias_k);
    boat_tensor_t* v_proj = linear_projection(value, attention->weight_v, attention->bias_v);

    if (!q_proj || !k_proj || !v_proj) {
        if (q_proj) boat_tensor_free(q_proj);
        if (k_proj) boat_tensor_free(k_proj);
        if (v_proj) boat_tensor_free(v_proj);
        return NULL;
    }

    // Store projected tensors for backward pass
    attention->cache_q_proj = q_proj;
    attention->cache_k_proj = k_proj;
    attention->cache_v_proj = v_proj;
    boat_tensor_ref(q_proj); // Increase ref count since we're keeping a reference
    boat_tensor_ref(k_proj);
    boat_tensor_ref(v_proj);

    // Reshape for multi-head attention: [batch, seq_len, hidden] -> [batch, num_heads, seq_len,
    // head_size]

    // Get shape of projected tensors
    const int64_t* q_shape = boat_tensor_shape(q_proj);
    int64_t batch = q_shape[0];
    int64_t seq_len = q_shape[1];
    int64_t hidden = q_shape[2];

    size_t num_heads = attention->config.num_heads;
    size_t num_kv_heads = attention->config.num_kv_heads;
    size_t head_size = attention->config.head_size;

    // Reshape Q to 4D: [batch, num_heads, seq_len, head_size]
    const int64_t q_reshaped_shape[] = {batch, (int64_t)num_heads, seq_len, (int64_t)head_size};
    boat_tensor_t* q_reshaped = boat_tensor_reshape(q_proj, q_reshaped_shape, 4);

    // Reshape K/V to 4D: [batch, num_kv_heads, seq_len, head_size] (supports GQA)
    const int64_t kv_reshaped_shape[] = {batch, (int64_t)num_kv_heads, seq_len, (int64_t)head_size};
    boat_tensor_t* k_reshaped = boat_tensor_reshape(k_proj, kv_reshaped_shape, 4);
    boat_tensor_t* v_reshaped = boat_tensor_reshape(v_proj, kv_reshaped_shape, 4);

    if (!q_reshaped || !k_reshaped || !v_reshaped) {
        if (q_reshaped) boat_tensor_free(q_reshaped);
        if (k_reshaped) boat_tensor_free(k_reshaped);
        if (v_reshaped) boat_tensor_free(v_reshaped);
        boat_tensor_free(q_proj);
        boat_tensor_free(k_proj);
        boat_tensor_free(v_proj);
        return NULL;
    }

    // Replace original tensors with reshaped views (drop local refs on the originals)
    boat_tensor_unref(q_proj);
    boat_tensor_unref(k_proj);
    boat_tensor_unref(v_proj);
    q_proj = q_reshaped;
    k_proj = k_reshaped;
    v_proj = v_reshaped;

    // GQA: repeat K/V heads to match Q head count
    if (num_kv_heads != 0 && num_kv_heads < num_heads) {
        int repeat = (int)(num_heads / num_kv_heads);
        // k_proj shape: [batch, num_kv_heads, seq_len, head_size]
        // output shape: [batch, num_heads, seq_len, head_size]
        const int64_t* kv_shape = boat_tensor_shape(k_proj);
        int64_t B = kv_shape[0], n_kv = kv_shape[1], T = kv_shape[2], hd = kv_shape[3];
        const int64_t out_shape[] = {B, (int64_t)num_heads, T, hd};
        boat_tensor_t* k_repeated =
            boat_tensor_create(out_shape, 4, boat_tensor_dtype(k_proj), boat_tensor_device(k_proj));
        boat_tensor_t* v_repeated =
            boat_tensor_create(out_shape, 4, boat_tensor_dtype(v_proj), boat_tensor_device(v_proj));
        if (!k_repeated || !v_repeated) {
            if (k_repeated) boat_tensor_free(k_repeated);
            if (v_repeated) boat_tensor_free(v_repeated);
            return NULL;
        }
        size_t row_bytes = (size_t)(T * hd) * sizeof(float);
        float* k_src = (float*)boat_tensor_data(k_proj);
        float* v_src = (float*)boat_tensor_data(v_proj);
        float* k_dst = (float*)boat_tensor_data(k_repeated);
        float* v_dst = (float*)boat_tensor_data(v_repeated);
        for (int64_t b = 0; b < B; b++) {
            for (int r = 0; r < repeat; r++) {
                for (int64_t h = 0; h < n_kv; h++) {
                    size_t src_idx = ((size_t)(b * n_kv + h) * (size_t)T * (size_t)hd);
                    size_t dst_idx =
                        ((size_t)(b * num_heads + r * n_kv + h) * (size_t)T * (size_t)hd);
                    memcpy(k_dst + dst_idx, k_src + src_idx, row_bytes);
                    memcpy(v_dst + dst_idx, v_src + src_idx, row_bytes);
                }
            }
        }
        boat_tensor_free(k_proj);
        boat_tensor_free(v_proj);
        k_proj = k_repeated;
        v_proj = v_repeated;
    }

    // Apply rotary position encoding if enabled
    if (attention->config.use_rotary) {
        boat_apply_rotary_embedding(q_proj, k_proj, (size_t)seq_len, head_size,
                                    attention->config.rotary_theta);
    }

    // Calculate scaled dot-product attention with cache for backward pass
    boat_tensor_t* output = scaled_dot_product_attention_impl(
        q_proj, k_proj, v_proj, 1.0f / sqrtf((float)attention->config.head_size), attention_mask,
        attention->config.causal_mask, attention->config.dropout_prob,
        &attention->cache_attention_weights);

    if (!output) {
        return NULL;
    }

    // Clean up intermediate tensors
#if 1
    boat_tensor_free(q_proj);
    boat_tensor_free(k_proj);
    boat_tensor_free(v_proj);
#endif

    // Reshape back: [batch, num_heads, seq_len, head_size] -> [batch, seq_len, hidden]
    const int64_t original_shape[] = {batch, seq_len, hidden};
    boat_tensor_t* output_reshaped = boat_tensor_reshape(output, original_shape, 3);
    if (!output_reshaped) {
        boat_tensor_free(output);
        return NULL;
    }
    // Drop our local ref on the original output; the reshape view and the
    // cache below keep it alive (cache_attention_output is set to the view).
    boat_tensor_unref(output);
    output = output_reshaped;

    // Cache attention output for backward pass (before final projection)
    attention->cache_attention_output = output;
    boat_tensor_ref(attention->cache_attention_output); // Increase ref count

    // Final linear projection
    boat_tensor_t* final_output = linear_projection(output, attention->weight_o, attention->bias_o);
    if (!final_output) {
        return output;
    }
    boat_tensor_free(output);
    return final_output;
}

BOAT_API bool BOAT_CALL boat_attention_backward(boat_attention_t* attention,
                                                const boat_tensor_t* grad_output,
                                                boat_tensor_t** grad_query,
                                                boat_tensor_t** grad_key,
                                                boat_tensor_t** grad_value) {
    if (!attention || !grad_output) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[Attention] boat_attention_backward: invalid arguments\n");
        return false;
    }

    // Check that all required cached tensors exist
    if (!attention->cache_query || !attention->cache_key || !attention->cache_value ||
        !attention->cache_q_proj || !attention->cache_k_proj || !attention->cache_v_proj) {
        boat_set_errorf(BOAT_ERROR_INVALID_OPERATION,
                        "[Attention] boat_attention_backward: missing cached tensors\n");
        return false;
    }

    // Step 1: Gradient for final linear projection (W_o, b_o)
    // grad_attention_output = grad_output @ weight_o^T
    // grad_weight_o = attention_output^T @ grad_output
    // grad_bias_o = sum(grad_output, axis=0)

    // Check that we have cached attention output
    if (!attention->cache_attention_output) {
        boat_set_errorf(BOAT_ERROR_INVALID_OPERATION,
                        "[Attention] boat_attention_backward: missing cache_attention_output\n");
        return false;
    }

    // Compute gradients for final projection
    boat_tensor_t* grad_attention_output = NULL;
    boat_tensor_t* grad_weight_o_local = NULL;
    boat_tensor_t* grad_bias_o_local = NULL;

    if (!linear_projection_backward(attention->cache_attention_output, attention->weight_o,
                                    attention->bias_o, grad_output, &grad_attention_output,
                                    &grad_weight_o_local, &grad_bias_o_local)) {
        boat_set_errorf(BOAT_ERROR_INVALID_OPERATION,
                        "[Attention] boat_attention_backward: linear_projection_backward failed "
                        "for final projection\n");
        return false;
    }

    // Store gradients for weight_o and bias_o (will be assigned after all steps succeed)
    // attention->grad_weight_o = grad_weight_o_local;
    // attention->grad_bias_o = grad_bias_o_local;

    // Step 2: Gradient for attention mechanism
    // Compute gradients for projected Q, K, V
    boat_tensor_t* grad_q_proj = NULL;
    boat_tensor_t* grad_k_proj = NULL;
    boat_tensor_t* grad_v_proj = NULL;

    // Reshape 3D tensors to 4D for attention backward
    // cache_q_proj shape: [batch, seq_len, hidden]
    // Need to reshape to: [batch, num_heads, seq_len, head_size]
    const int64_t* cache_shape = boat_tensor_shape(attention->cache_q_proj);
    int64_t batch = cache_shape[0];
    int64_t seq_len = cache_shape[1];
    int64_t hidden = cache_shape[2];
    size_t num_heads = attention->config.num_heads;
    size_t head_size = attention->config.head_size;

    // Verify hidden == num_heads * head_size
    if (hidden != (int64_t)(num_heads * head_size)) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[Attention] boat_attention_backward: hidden size mismatch\n");
        boat_tensor_unref(grad_attention_output);
        if (grad_weight_o_local) boat_tensor_unref(grad_weight_o_local);
        if (grad_bias_o_local) boat_tensor_unref(grad_bias_o_local);
        return false;
    }

    const int64_t reshaped_shape[] = {batch, (int64_t)num_heads, seq_len, (int64_t)head_size};
    boat_tensor_t* cache_q_proj_4d =
        boat_tensor_reshape(attention->cache_q_proj, reshaped_shape, 4);
    boat_tensor_t* cache_k_proj_4d =
        boat_tensor_reshape(attention->cache_k_proj, reshaped_shape, 4);
    boat_tensor_t* cache_v_proj_4d =
        boat_tensor_reshape(attention->cache_v_proj, reshaped_shape, 4);
    boat_tensor_t* grad_attention_output_4d =
        boat_tensor_reshape(grad_attention_output, reshaped_shape, 4);

    if (!cache_q_proj_4d || !cache_k_proj_4d || !cache_v_proj_4d || !grad_attention_output_4d) {
        boat_set_errorf(BOAT_ERROR_OUT_OF_MEMORY,
                        "[Attention] boat_attention_backward: failed to create 4D tensors\n");
        if (cache_k_proj_4d) boat_tensor_unref(cache_k_proj_4d);
        if (cache_v_proj_4d) boat_tensor_unref(cache_v_proj_4d);
        if (grad_attention_output_4d) boat_tensor_unref(grad_attention_output_4d);
        boat_tensor_unref(grad_attention_output);
        if (grad_weight_o_local) boat_tensor_unref(grad_weight_o_local);
        if (grad_bias_o_local) boat_tensor_unref(grad_bias_o_local);
        return false;
    }

    // Call attention backward with 4D tensors
    boat_tensor_t* grad_q_proj_4d = NULL;
    boat_tensor_t* grad_k_proj_4d = NULL;
    boat_tensor_t* grad_v_proj_4d = NULL;

    // Check that attention weights are cached
    if (!attention->cache_attention_weights) {
        boat_set_errorf(BOAT_ERROR_INVALID_OPERATION,
                        "[Attention] boat_attention_backward: missing cache_attention_weights\n");
        boat_tensor_unref(cache_q_proj_4d);
        boat_tensor_unref(cache_k_proj_4d);
        boat_tensor_unref(cache_v_proj_4d);
        boat_tensor_unref(grad_attention_output_4d);
        boat_tensor_unref(grad_attention_output);
        if (grad_weight_o_local) boat_tensor_unref(grad_weight_o_local);
        if (grad_bias_o_local) boat_tensor_unref(grad_bias_o_local);
        return false;
    }

    // Call attention backward to compute gradients for projected Q, K, V
    if (!attention_backward(cache_q_proj_4d, cache_k_proj_4d, cache_v_proj_4d,
                            attention->cache_attention_weights, grad_attention_output_4d,
                            &grad_q_proj_4d, &grad_k_proj_4d, &grad_v_proj_4d)) {
        boat_set_errorf(BOAT_ERROR_INVALID_OPERATION,
                        "[Attention] boat_attention_backward: attention_backward failed\n");
        boat_tensor_unref(cache_q_proj_4d);
        boat_tensor_unref(cache_k_proj_4d);
        boat_tensor_unref(cache_v_proj_4d);
        boat_tensor_unref(grad_attention_output_4d);
        boat_tensor_unref(grad_attention_output);
        if (grad_weight_o_local) boat_tensor_unref(grad_weight_o_local);
        if (grad_bias_o_local) boat_tensor_unref(grad_bias_o_local);
        return false;
    }

    // Reshape gradients back to 3D for linear projection backward
    const int64_t original_shape[] = {batch, seq_len, hidden};
    boat_tensor_t* grad_q_proj_3d = boat_tensor_reshape(grad_q_proj_4d, original_shape, 3);
    boat_tensor_t* grad_k_proj_3d = boat_tensor_reshape(grad_k_proj_4d, original_shape, 3);
    boat_tensor_t* grad_v_proj_3d = boat_tensor_reshape(grad_v_proj_4d, original_shape, 3);

    // Clean up 4D intermediates
    boat_tensor_unref(cache_q_proj_4d);
    boat_tensor_unref(cache_k_proj_4d);
    boat_tensor_unref(cache_v_proj_4d);
    boat_tensor_unref(grad_attention_output_4d);
    boat_tensor_unref(grad_q_proj_4d);
    boat_tensor_unref(grad_k_proj_4d);
    boat_tensor_unref(grad_v_proj_4d);

    if (!grad_q_proj_3d || !grad_k_proj_3d || !grad_v_proj_3d) {
        if (grad_q_proj_3d) boat_tensor_unref(grad_q_proj_3d);
        if (grad_k_proj_3d) boat_tensor_unref(grad_k_proj_3d);
        if (grad_v_proj_3d) boat_tensor_unref(grad_v_proj_3d);
        boat_tensor_unref(grad_attention_output);
        if (grad_weight_o_local) boat_tensor_unref(grad_weight_o_local);
        if (grad_bias_o_local) boat_tensor_unref(grad_bias_o_local);
        return false;
    }

    // Assign reshaped gradients to output pointers
    grad_q_proj = grad_q_proj_3d;
    grad_k_proj = grad_k_proj_3d;
    grad_v_proj = grad_v_proj_3d;

    // Step 3: Gradient for linear projections (Q, K, V)
    boat_tensor_t* grad_q = NULL;
    boat_tensor_t* grad_k = NULL;
    boat_tensor_t* grad_v = NULL;

    // Gradient for Q projection (replace previous iteration's accumulators)
    if (attention->grad_weight_q) boat_tensor_free(attention->grad_weight_q);
    if (attention->grad_bias_q) boat_tensor_free(attention->grad_bias_q);
    if (!linear_projection_backward(attention->cache_query, attention->weight_q, attention->bias_q,
                                    grad_q_proj, &grad_q, &attention->grad_weight_q,
                                    &attention->grad_bias_q)) {
        boat_tensor_unref(grad_attention_output);
        boat_tensor_unref(grad_q_proj);
        boat_tensor_unref(grad_k_proj);
        boat_tensor_unref(grad_v_proj);
        if (grad_weight_o_local) boat_tensor_unref(grad_weight_o_local);
        if (grad_bias_o_local) boat_tensor_unref(grad_bias_o_local);
        return false;
    }

    // Gradient for K projection (replace previous iteration's accumulators)
    if (attention->grad_weight_k) boat_tensor_free(attention->grad_weight_k);
    if (attention->grad_bias_k) boat_tensor_free(attention->grad_bias_k);
    if (!linear_projection_backward(attention->cache_key, attention->weight_k, attention->bias_k,
                                    grad_k_proj, &grad_k, &attention->grad_weight_k,
                                    &attention->grad_bias_k)) {
        boat_tensor_unref(grad_attention_output);
        boat_tensor_unref(grad_q_proj);
        boat_tensor_unref(grad_k_proj);
        boat_tensor_unref(grad_v_proj);
        if (grad_q) boat_tensor_unref(grad_q);
        if (grad_weight_o_local) boat_tensor_unref(grad_weight_o_local);
        if (grad_bias_o_local) boat_tensor_unref(grad_bias_o_local);
        return false;
    }

    // Gradient for V projection (replace previous iteration's accumulators)
    if (attention->grad_weight_v) boat_tensor_free(attention->grad_weight_v);
    if (attention->grad_bias_v) boat_tensor_free(attention->grad_bias_v);
    if (!linear_projection_backward(attention->cache_value, attention->weight_v, attention->bias_v,
                                    grad_v_proj, &grad_v, &attention->grad_weight_v,
                                    &attention->grad_bias_v)) {
        boat_tensor_unref(grad_attention_output);
        boat_tensor_unref(grad_q_proj);
        boat_tensor_unref(grad_k_proj);
        boat_tensor_unref(grad_v_proj);
        if (grad_q) boat_tensor_unref(grad_q);
        if (grad_k) boat_tensor_unref(grad_k);
        if (grad_weight_o_local) boat_tensor_unref(grad_weight_o_local);
        if (grad_bias_o_local) boat_tensor_unref(grad_bias_o_local);
        return false;
    }

    // Step 4: Store final projection gradients (already computed in Step 1)
    if (attention->grad_weight_o) boat_tensor_free(attention->grad_weight_o);
    if (attention->grad_bias_o) boat_tensor_free(attention->grad_bias_o);
    attention->grad_weight_o = grad_weight_o_local;
    attention->grad_bias_o = grad_bias_o_local;

    // Clean up intermediate tensors
    boat_tensor_unref(grad_attention_output);
    boat_tensor_unref(grad_q_proj);
    boat_tensor_unref(grad_k_proj);
    boat_tensor_unref(grad_v_proj);

    // Return gradients for all three inputs via output pointers
    // If an output pointer is NULL, discard the corresponding gradient tensor

    if (grad_query) {
        *grad_query = grad_q;
    } else if (grad_q) {
        boat_tensor_unref(grad_q);
    }
    if (grad_key) {
        *grad_key = grad_k;
    } else if (grad_k) {
        boat_tensor_unref(grad_k);
    }
    if (grad_value) {
        *grad_value = grad_v;
    } else if (grad_v) {
        boat_tensor_unref(grad_v);
    }

    return true;
}

BOAT_API void BOAT_CALL boat_attention_update(boat_attention_t* attention, float learning_rate) {
    if (!attention) {
        return;
    }

    // Simple SGD update: weight = weight - learning_rate * gradient

    // Update weight_q if gradient exists
    if (attention->grad_weight_q && attention->weight_q) {
        // weight_q = weight_q - learning_rate * grad_weight_q
        boat_tensor_t* scaled_grad = boat_mul_scalar(attention->grad_weight_q, learning_rate);
        if (scaled_grad) {
            boat_sub_(attention->weight_q,
                      scaled_grad); // weight_q -= learning_rate * grad_weight_q
            boat_tensor_unref(scaled_grad);
        }
    }

    // Update weight_k
    if (attention->grad_weight_k && attention->weight_k) {
        boat_tensor_t* scaled_grad = boat_mul_scalar(attention->grad_weight_k, learning_rate);
        if (scaled_grad) {
            boat_sub_(attention->weight_k, scaled_grad);
            boat_tensor_unref(scaled_grad);
        }
    }

    // Update weight_v
    if (attention->grad_weight_v && attention->weight_v) {
        boat_tensor_t* scaled_grad = boat_mul_scalar(attention->grad_weight_v, learning_rate);
        if (scaled_grad) {
            boat_sub_(attention->weight_v, scaled_grad);
            boat_tensor_unref(scaled_grad);
        }
    }

    // Update weight_o
    if (attention->grad_weight_o && attention->weight_o) {
        boat_tensor_t* scaled_grad = boat_mul_scalar(attention->grad_weight_o, learning_rate);
        if (scaled_grad) {
            boat_sub_(attention->weight_o, scaled_grad);
            boat_tensor_unref(scaled_grad);
        }
    }

    // Update bias_q if gradient exists
    if (attention->grad_bias_q && attention->bias_q) {
        boat_tensor_t* scaled_grad = boat_mul_scalar(attention->grad_bias_q, learning_rate);
        if (scaled_grad) {
            boat_sub_(attention->bias_q, scaled_grad);
            boat_tensor_unref(scaled_grad);
        }
    }

    // Update bias_k
    if (attention->grad_bias_k && attention->bias_k) {
        boat_tensor_t* scaled_grad = boat_mul_scalar(attention->grad_bias_k, learning_rate);
        if (scaled_grad) {
            boat_sub_(attention->bias_k, scaled_grad);
            boat_tensor_unref(scaled_grad);
        }
    }

    // Update bias_v
    if (attention->grad_bias_v && attention->bias_v) {
        boat_tensor_t* scaled_grad = boat_mul_scalar(attention->grad_bias_v, learning_rate);
        if (scaled_grad) {
            boat_sub_(attention->bias_v, scaled_grad);
            boat_tensor_unref(scaled_grad);
        }
    }

    // Update bias_o
    if (attention->grad_bias_o && attention->bias_o) {
        boat_tensor_t* scaled_grad = boat_mul_scalar(attention->grad_bias_o, learning_rate);
        if (scaled_grad) {
            boat_sub_(attention->bias_o, scaled_grad);
            boat_tensor_unref(scaled_grad);
        }
    }
}

BOAT_API BOAT_NOINLINE void BOAT_CALL boat_attention_set_dropout(boat_attention_t* attention,
                                                                 float dropout_prob) {
    if (attention) {
        attention->config.dropout_prob = dropout_prob;
    }
}

BOAT_API BOAT_NOINLINE void BOAT_CALL boat_attention_set_causal(boat_attention_t* attention,
                                                                bool causal) {
    if (attention) {
        attention->config.causal_mask = causal;
    }
}

// Simplified multi-head attention function
BOAT_API boat_tensor_t* BOAT_CALL boat_multihead_attention(const boat_tensor_t* input,
                                                           size_t num_heads, float dropout_prob,
                                                           bool causal_mask,
                                                           const boat_tensor_t* attention_mask) {
    if (!input) {
        return NULL;
    }

    // Get input shape
    size_t ndim = boat_tensor_ndim(input);
    const int64_t* shape = boat_tensor_shape(input);

    if (ndim != 3) {
        // Expected shape: [batch, seq_len, hidden]
        return NULL;
    }

    size_t hidden_size = shape[2];

    if (hidden_size % num_heads != 0) {
        return NULL;
    }

    // Create attention configuration
    boat_attention_config_t config = {.hidden_size = hidden_size,
                                      .num_heads = num_heads,
                                      .head_size = hidden_size / num_heads,
                                      .dropout_prob = dropout_prob,
                                      .causal_mask = causal_mask,
                                      .use_bias = true,
                                      .use_rotary = false,
                                      .rotary_theta = 10000.0f};

    // Create temporary attention layer (MHA, no GQA for multihead_attention)
    // This simplified API always uses MHA (num_kv_heads = num_heads)
    boat_attention_t* attention = boat_attention_create(&config);
    // Override weights after create since the simplified API creates MHA-sizes
    // (weights will be loaded externally if needed)
    if (!attention) {
        return NULL;
    }

    // Self-attention: query = key = value = input
    boat_tensor_t* output = boat_attention_forward(attention, input, input, input, attention_mask);

    // Free attention layer
    boat_attention_free(attention);

    return output;
}

// Internal implementation with cache support
static boat_tensor_t*
scaled_dot_product_attention_impl(const boat_tensor_t* query, const boat_tensor_t* key,
                                  const boat_tensor_t* value, float scale_factor,
                                  const boat_tensor_t* attention_mask, bool causal_mask,
                                  float dropout_prob, boat_tensor_t** cache_weights) {
    (void)dropout_prob;
    if (!query || !key || !value) {
        return NULL;
    }

    // Check shapes
    size_t q_ndim = boat_tensor_ndim(query);
    size_t k_ndim = boat_tensor_ndim(key);
    size_t v_ndim = boat_tensor_ndim(value);

    if (q_ndim < 3 || k_ndim < 3 || v_ndim < 3) {
        return NULL;
    }

    // Expect 4D: [batch, num_heads, seq_len, head_size]
    if (q_ndim != 4 || k_ndim != 4 || v_ndim != 4) {
        // Fallback for non-4D input
        const int64_t* value_shape = boat_tensor_shape(value);
        boat_tensor_t* output = boat_tensor_create(value_shape, v_ndim, boat_tensor_dtype(value),
                                                   boat_tensor_device(value));
        if (!output) return NULL;
        size_t nbytes = boat_tensor_nbytes(value);
        const void* src_data = boat_tensor_data(value);
        void* dst_data = boat_tensor_data(output);
        memcpy(dst_data, src_data, nbytes);
        return output;
    }

    const int64_t* q_shape = boat_tensor_shape(query);
    const int64_t* k_shape = boat_tensor_shape(key);
    const int64_t* v_shape = boat_tensor_shape(value);

    int64_t batch = q_shape[0];
    int64_t num_heads = q_shape[1];
    int64_t q_seq_len = q_shape[2];
    int64_t kv_seq_len = k_shape[2];
    int64_t head_size = q_shape[3];

    // Validate shapes match
    if (k_shape[0] != batch || k_shape[1] != num_heads || k_shape[3] != head_size ||
        v_shape[0] != batch || v_shape[1] != num_heads || v_shape[2] != kv_seq_len ||
        v_shape[3] != head_size) {
        return NULL;
    }

    boat_dtype_t dtype = boat_tensor_dtype(query);
    if (dtype != BOAT_DTYPE_FLOAT32) {
        return NULL;
    }

    // --- Op chain for scaled dot-product attention ---

    // 1. scores = Q @ K^T * scale
    // K^T: [batch, num_heads, kv_seq_len, head_size] -> [batch, num_heads, head_size, kv_seq_len]
    boat_tensor_t* key_t = boat_transpose(key, 2, 3);
    if (!key_t) return NULL;

    boat_tensor_t* scores = boat_matmul(query, key_t);
    boat_tensor_unref(key_t);
    if (!scores) return NULL;

    boat_tensor_t* scaled_scores = boat_mul_scalar(scores, (double)scale_factor);
    boat_tensor_unref(scores);
    if (!scaled_scores) return NULL;
    scores = scaled_scores;

    // 2. Apply causal mask (upper triangular set to -inf)
    if (causal_mask && q_seq_len == kv_seq_len) {
        float* scores_data = (float*)boat_tensor_data(scores);
        for (int64_t b = 0; b < batch; b++) {
            for (int64_t h = 0; h < num_heads; h++) {
                for (int64_t i = 0; i < q_seq_len; i++) {
                    for (int64_t j = i + 1; j < kv_seq_len; j++) {
                        int64_t idx = ((b * num_heads + h) * q_seq_len + i) * kv_seq_len + j;
                        scores_data[idx] = -INFINITY;
                    }
                }
            }
        }
    }

    // 3. Apply attention mask if provided
    if (attention_mask) {
        // Element-wise add mask to scores
        boat_tensor_t* masked = boat_add(scores, attention_mask);
        boat_tensor_unref(scores);
        if (!masked) return NULL;
        scores = masked;
    }

    // 4. Softmax over last dimension (kv_seq_len)
    boat_tensor_t* weights = boat_softmax(scores, 3);
    boat_tensor_unref(scores);
    if (!weights) return NULL;

    // Cache weights for backward pass if requested
    if (cache_weights) {
        *cache_weights = weights;
        boat_tensor_ref(weights);
    }

    // 5. Dropout (skip for now)

    // 6. output = weights @ V
    boat_tensor_t* output = boat_matmul(weights, value);

    // Release our reference to weights (still alive if cached via cache_weights)
    boat_tensor_unref(weights);

    return output;
}

// Public API wrapper (maintains backward compatibility)
BOAT_API boat_tensor_t* BOAT_CALL boat_scaled_dot_product_attention(
    const boat_tensor_t* query, const boat_tensor_t* key, const boat_tensor_t* value,
    float scale_factor, const boat_tensor_t* attention_mask, bool causal_mask, float dropout_prob) {
    return scaled_dot_product_attention_impl(query, key, value, scale_factor, attention_mask,
                                             causal_mask, dropout_prob, NULL);
}

// Rotary position encoding
BOAT_API boat_tensor_t* BOAT_CALL boat_rotary_position_encoding(const boat_tensor_t* tensor,
                                                                size_t seq_len, size_t head_size,
                                                                float theta) {
    if (!tensor) return NULL;
    // Create output with same shape
    boat_tensor_t* out = boat_tensor_create_like(tensor);
    if (!out) return NULL;

    size_t n_heads = boat_tensor_shape(tensor)[1];
    size_t n = boat_tensor_nelements(tensor);
    float* src = (float*)boat_tensor_data(tensor);
    float* dst = (float*)boat_tensor_data(out);
    memcpy(dst, src, n * sizeof(float));

    // Apply RoPE in-place on the output
    size_t half = head_size / 2;
    for (size_t b = 0; b < (size_t)boat_tensor_shape(tensor)[0]; b++) {
        for (size_t h = 0; h < n_heads; h++) {
            for (size_t p = 0; p < seq_len; p++) {
                for (size_t i = 0; i < half; i++) {
                    float freq = powf(theta, -2.0f * (float)i / (float)head_size);
                    float cos_v = cosf((float)p * freq);
                    float sin_v = sinf((float)p * freq);

                    size_t idx = ((b * n_heads + h) * seq_len + p) * head_size;
                    float x0 = dst[idx + 2 * i];
                    float x1 = dst[idx + 2 * i + 1];
                    dst[idx + 2 * i] = x0 * cos_v - x1 * sin_v;
                    dst[idx + 2 * i + 1] = x1 * cos_v + x0 * sin_v;
                }
            }
        }
    }

    return out;
}

BOAT_API void BOAT_CALL boat_apply_rotary_embedding(boat_tensor_t* query, boat_tensor_t* key,
                                                    size_t seq_len, size_t head_size, float theta) {
    if (!query || !key) return;

    size_t half = head_size / 2;
    size_t q_heads = (size_t)boat_tensor_shape(query)[1];
    size_t k_heads = (size_t)boat_tensor_shape(key)[1];

    // Apply to query
    float* q = (float*)boat_tensor_data(query);
    for (size_t b = 0; b < (size_t)boat_tensor_shape(query)[0]; b++) {
        for (size_t h = 0; h < q_heads; h++) {
            for (size_t p = 0; p < seq_len; p++) {
                for (size_t i = 0; i < half; i++) {
                    float freq = powf(theta, -2.0f * (float)i / (float)head_size);
                    float cos_v = cosf((float)p * freq);
                    float sin_v = sinf((float)p * freq);

                    size_t idx = ((b * q_heads + h) * seq_len + p) * head_size;
                    float x0 = q[idx + 2 * i];
                    float x1 = q[idx + 2 * i + 1];
                    q[idx + 2 * i] = x0 * cos_v - x1 * sin_v;
                    q[idx + 2 * i + 1] = x1 * cos_v + x0 * sin_v;
                }
            }
        }
    }

    // Apply to key
    float* k = (float*)boat_tensor_data(key);
    for (size_t b = 0; b < (size_t)boat_tensor_shape(key)[0]; b++) {
        for (size_t h = 0; h < k_heads; h++) {
            for (size_t p = 0; p < seq_len; p++) {
                for (size_t i = 0; i < half; i++) {
                    float freq = powf(theta, -2.0f * (float)i / (float)head_size);
                    float cos_v = cosf((float)p * freq);
                    float sin_v = sinf((float)p * freq);

                    size_t idx = ((b * k_heads + h) * seq_len + p) * head_size;
                    float x0 = k[idx + 2 * i];
                    float x1 = k[idx + 2 * i + 1];
                    k[idx + 2 * i] = x0 * cos_v - x1 * sin_v;
                    k[idx + 2 * i + 1] = x1 * cos_v + x0 * sin_v;
                }
            }
        }
    }
}

// Adapter for generic attention layer interface (layers.h)
typedef boat_attention_t boat_attention_layer_t;

BOAT_API boat_attention_t* BOAT_CALL boat_attention_create_gqa(size_t hidden_size, size_t num_heads,
                                                               size_t num_kv_heads,
                                                               size_t head_size, bool causal_mask,
                                                               float rotary_theta) {
    boat_attention_config_t config = {
        .hidden_size = hidden_size,
        .num_heads = num_heads,
        .num_kv_heads = num_kv_heads,
        .head_size = head_size,
        .dropout_prob = 0.0f,
        .causal_mask = causal_mask,
        .use_bias = false,
        .use_rotary = (rotary_theta > 0.0f),
        .rotary_theta = rotary_theta > 0.0f ? rotary_theta : 10000.0f,
    };
    return boat_attention_create(&config);
}

BOAT_API boat_attention_layer_t* BOAT_CALL boat_attention_layer_create(size_t hidden_size,
                                                                       size_t num_heads,
                                                                       size_t num_kv_heads,
                                                                       float dropout_prob,
                                                                       bool causal_mask) {
    boat_attention_config_t config = {.hidden_size = hidden_size,
                                      .num_heads = num_heads,
                                      .num_kv_heads = num_kv_heads,
                                      .head_size = hidden_size / num_heads,
                                      .dropout_prob = dropout_prob,
                                      .causal_mask = causal_mask,
                                      .use_bias = true,
                                      .use_rotary = false,
                                      .rotary_theta = 10000.0f};
    if (hidden_size % num_heads != 0) {
        config.head_size = (hidden_size + num_heads - 1) / num_heads;
    }
    return boat_attention_create(&config);
}

BOAT_API void BOAT_CALL boat_attention_layer_free(boat_attention_layer_t* layer) {
    boat_attention_free(layer);
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL boat_attention_layer_forward(
    boat_attention_layer_t* layer, const boat_tensor_t* query, const boat_tensor_t* key,
    const boat_tensor_t* value, const boat_tensor_t* attention_mask) {

    // Delegate to the actual implementation
    boat_tensor_t* result = boat_attention_forward(layer, query, key, value, attention_mask);

    return result;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_layer_backward(boat_attention_layer_t* layer, const boat_tensor_t* grad_output) {
    boat_tensor_t* grad_query = NULL;
    boat_tensor_t* grad_key = NULL;
    boat_tensor_t* grad_value = NULL;

    if (boat_attention_backward((boat_attention_t*)layer, grad_output, &grad_query, &grad_key,
                                &grad_value)) {
        // Free unused gradients (key and value) as layer interface only returns query gradient
        if (grad_key) boat_tensor_free(grad_key);
        if (grad_value) boat_tensor_free(grad_value);
        return grad_query;
    }
    return NULL;
}

BOAT_API BOAT_NOINLINE void BOAT_CALL boat_attention_layer_update(boat_attention_layer_t* layer,
                                                                  float learning_rate) {
    boat_attention_update(layer, learning_rate);
}

// ============================================================================
// Gradient computation helper functions for attention layer
// ============================================================================

// Gradient for linear projection: output = input @ weight + bias
// Computes:
//   grad_input = grad_output @ weight^T
//   grad_weight = input^T @ grad_output (summed over batch and sequence dimensions)
//   grad_bias = sum(grad_output, axis=(0,1)) if bias exists
static bool linear_projection_backward(const boat_tensor_t* input, const boat_tensor_t* weight,
                                       const boat_tensor_t* bias, const boat_tensor_t* grad_output,
                                       boat_tensor_t** grad_input, boat_tensor_t** grad_weight,
                                       boat_tensor_t** grad_bias) {
    if (!input || !weight || !grad_output) {
        return false;
    }

    // Get shapes
    size_t input_ndim = boat_tensor_ndim(input);
    size_t grad_ndim = boat_tensor_ndim(grad_output);
    const int64_t* input_shape = boat_tensor_shape(input);
    const int64_t* grad_shape = boat_tensor_shape(grad_output);
    const int64_t* weight_shape = boat_tensor_shape(weight);

    // Handle 3D case: [batch, seq_len, hidden] typical for attention layers
    if (input_ndim == 3 && grad_ndim == 3 && weight_shape[0] == input_shape[2] &&
        weight_shape[1] == grad_shape[2]) {
        int64_t batch = input_shape[0];
        int64_t seq_len = input_shape[1];
        int64_t hidden = input_shape[2];

        // Compute grad_input = grad_output @ weight^T (batch matrix multiplication)
        // grad_output is 3D [batch, seq_len, hidden], weight_transposed is 2D [hidden, hidden]
        // Need to reshape grad_output to 2D for matmul, then reshape back
        boat_tensor_t* weight_transposed = boat_transpose(weight, 0, 1);
        if (!weight_transposed) {
            return false;
        }
        // Reshape grad_output to 2D: [batch*seq_len, hidden]
        const int64_t grad_2d_shape[] = {batch * seq_len, hidden};
        boat_tensor_t* grad_output_2d_for_input =
            boat_tensor_reshape(grad_output, grad_2d_shape, 2);
        if (!grad_output_2d_for_input) {
            boat_tensor_unref(weight_transposed);
            return false;
        }
        // Compute grad_input_2d = grad_output_2d_for_input @ weight_transposed
        boat_tensor_t* grad_input_2d = boat_matmul(grad_output_2d_for_input, weight_transposed);
        boat_tensor_unref(grad_output_2d_for_input);
        boat_tensor_unref(weight_transposed);
        if (!grad_input_2d) {
            return false;
        }
        // Reshape back to 3D: [batch, seq_len, hidden]
        boat_tensor_t* grad_input_local = boat_tensor_reshape(grad_input_2d, input_shape, 3);
        boat_tensor_unref(grad_input_2d);
        if (!grad_input_local) {
            return false;
        }

        // Compute grad_weight = sum_over_batch_seq( input^T @ grad_output )
        // Reshape input and grad_output to 2D: [batch*seq_len, hidden]
        const int64_t reshaped_shape[] = {batch * seq_len, hidden};
        boat_tensor_t* input_2d = boat_tensor_reshape(input, reshaped_shape, 2);
        boat_tensor_t* grad_output_2d = boat_tensor_reshape(grad_output, reshaped_shape, 2);
        if (!input_2d || !grad_output_2d) {
            if (input_2d) boat_tensor_unref(input_2d);
            if (grad_output_2d) boat_tensor_unref(grad_output_2d);
            boat_tensor_unref(grad_input_local);
            return false;
        }
        // Compute input^T @ grad_output: [hidden, batch*seq_len] @ [batch*seq_len, hidden] =
        // [hidden, hidden]
        boat_tensor_t* input_transposed = boat_transpose(input_2d, 0, 1);
        if (!input_transposed) {
            boat_tensor_unref(input_2d);
            boat_tensor_unref(grad_output_2d);
            boat_tensor_unref(grad_input_local);
            return false;
        }
        boat_tensor_t* grad_weight_local = boat_matmul(input_transposed, grad_output_2d);
        boat_tensor_unref(input_transposed);
        boat_tensor_unref(input_2d);
        boat_tensor_unref(grad_output_2d);
        if (!grad_weight_local) {
            boat_tensor_unref(grad_input_local);
            return false;
        }

        // Compute grad_bias if bias exists
        boat_tensor_t* grad_bias_local = NULL;
        if (bias) {
            // Sum grad_output over batch and sequence dimensions (axes 0 and 1)
            // grad_output shape: [batch, seq_len, hidden]
            grad_bias_local = boat_tensor_create_like(bias);
            if (grad_bias_local) {
                const float* grad_out_data = (float*)boat_tensor_data(grad_output);
                float* grad_bias_data = (float*)boat_tensor_data(grad_bias_local);
                memset(grad_bias_data, 0, hidden * sizeof(float));
                // Sum over batch and sequence
                for (int64_t b = 0; b < batch; b++) {
                    for (int64_t s = 0; s < seq_len; s++) {
                        for (int64_t h = 0; h < hidden; h++) {
                            grad_bias_data[h] += grad_out_data[(b * seq_len + s) * hidden + h];
                        }
                    }
                }
            }
        }

        // Return results
        if (grad_input) *grad_input = grad_input_local;
        if (grad_weight) *grad_weight = grad_weight_local;
        if (grad_bias) *grad_bias = grad_bias_local;

        return true;
    }
    // Fallback to original 2D implementation
    else if (input_ndim == 2 && grad_ndim == 2) {
        // Compute grad_input = grad_output @ weight^T
        boat_tensor_t* weight_transposed = boat_transpose(weight, 0, 1);
        if (!weight_transposed) {
            return false;
        }
        boat_tensor_t* grad_input_local = boat_matmul(grad_output, weight_transposed);
        boat_tensor_unref(weight_transposed);
        if (!grad_input_local) {
            return false;
        }

        // Compute grad_weight = input^T @ grad_output
        boat_tensor_t* input_transposed = boat_transpose(input, 0, 1);
        if (!input_transposed) {
            boat_tensor_unref(grad_input_local);
            return false;
        }
        boat_tensor_t* grad_weight_local = boat_matmul(input_transposed, grad_output);
        boat_tensor_unref(input_transposed);
        if (!grad_weight_local) {
            boat_tensor_unref(grad_input_local);
            return false;
        }

        // Compute grad_bias if bias exists (2D case)
        boat_tensor_t* grad_bias_local = NULL;
        if (bias) {
            const int64_t* grad_shape_local = boat_tensor_shape(grad_output);
            int64_t batch = grad_shape_local[0];
            int64_t features = grad_shape_local[1];

            grad_bias_local = boat_tensor_create_like(bias);
            if (grad_bias_local) {
                const float* grad_out_data = (float*)boat_tensor_data(grad_output);
                float* grad_bias_data = (float*)boat_tensor_data(grad_bias_local);
                memset(grad_bias_data, 0, features * sizeof(float));
                for (int64_t b = 0; b < batch; b++) {
                    for (int64_t f = 0; f < features; f++) {
                        grad_bias_data[f] += grad_out_data[b * features + f];
                    }
                }
            }
        }

        if (grad_input) *grad_input = grad_input_local;
        if (grad_weight) *grad_weight = grad_weight_local;
        if (grad_bias) *grad_bias = grad_bias_local;

        return true;
    } else {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[Attention] linear_projection_backward: unsupported input/grad_output "
                        "dimensions: input_ndim=%zu, grad_ndim=%zu\n",
                        input_ndim, grad_ndim);
        return false;
    }
}

// Helper function: sum over last dimension of 4D tensor, keepdim=true

// Gradient for scaled dot-product attention
// Computes gradients for Q, K, V given attention weights and grad_output
// This is a simplified implementation that assumes cached attention weights
static bool
attention_backward(const boat_tensor_t* query, // [batch, num_heads, seq_len, head_size]
                   const boat_tensor_t* key, const boat_tensor_t* value,
                   const boat_tensor_t* attention_weights, // [batch, num_heads, seq_len, seq_len]
                   const boat_tensor_t* grad_output,       // [batch, num_heads, seq_len, head_size]
                   boat_tensor_t** grad_query, boat_tensor_t** grad_key,
                   boat_tensor_t** grad_value) {
    if (!query || !key || !value || !attention_weights || !grad_output) {
        return false;
    }
    const boat_tensor_t* grad_output_4d = grad_output;
    const boat_tensor_t* grad_output_reshaped = NULL;
    if (boat_tensor_ndim(grad_output) == 3) {
        // Reshape to 4D using query shape
        const int64_t* query_shape = boat_tensor_shape(query);
        const int64_t reshaped_shape[] = {query_shape[0], query_shape[1], query_shape[2],
                                          query_shape[3]};
        grad_output_reshaped = boat_tensor_reshape(grad_output, reshaped_shape, 4);
        if (!grad_output_reshaped) {
            return false;
        }
        grad_output_4d = grad_output_reshaped;
    }

    // Get shape information
    const int64_t* q_shape = boat_tensor_shape(query);
    int64_t batch = q_shape[0];
    int64_t num_heads = q_shape[1];
    int64_t seq_len = q_shape[2];
    int64_t head_size = q_shape[3];
    float scale = 1.0f / sqrtf((float)head_size);

    // Step 1: Compute gradient for value: dV = A^T @ dO
    // Transpose attention weights on last two dimensions: [batch, num_heads, seq_len, seq_len] ->
    // [batch, num_heads, seq_len, seq_len]
    boat_tensor_t* attn_t = boat_transpose(attention_weights, 2, 3);
    if (!attn_t) return false;
    boat_tensor_t* grad_value_local = boat_matmul(attn_t, grad_output_4d);
    boat_tensor_unref(attn_t);
    if (!grad_value_local) return false;

    // Step 2: Compute gradient for attention weights: dA = dO @ V^T
    // Transpose value on last two dimensions: [batch, num_heads, seq_len, head_size] -> [batch,
    // num_heads, head_size, seq_len]
    boat_tensor_t* value_t = boat_transpose(value, 2, 3);
    if (!value_t) {
        boat_tensor_unref(grad_value_local);
        return false;
    }
    boat_tensor_t* grad_attn = boat_matmul(grad_output_4d, value_t);
    boat_tensor_unref(value_t);
    if (!grad_attn) {
        boat_tensor_unref(grad_value_local);
        return false;
    }

    // Step 3: Compute gradient for scores (pre-softmax): dS = softmax_backward(dA, A)
    // softmax gradient: dS = A * (dA - sum(dA * A, dim=-1, keepdim=True))
    // We need to compute sum(dA * A, dim=-1, keepdim=True)
    // First compute dA * A element-wise
    boat_tensor_t* dA_mul_A = boat_mul(grad_attn, attention_weights);
    if (!dA_mul_A) {
        boat_tensor_unref(grad_value_local);
        boat_tensor_unref(grad_attn);
        return false;
    }
    // Compute sum over last dimension (seq_len dimension) of dA * A
    // We'll compute manually since boat_sum may not be available
    boat_tensor_unref(dA_mul_A); // Not needed for now, we'll compute directly
    // Instead, compute softmax gradient directly: dS = A * (dA - sum(dA * A, dim=-1, keepdim=True))
    // Get shapes and data pointers (use existing variables batch, num_heads, seq_len)
    // attention_weights shape: [batch, num_heads, seq_len, seq_len]
    // batch, num_heads, seq_len already defined above
    // attn_shape[3] should equal seq_len for self-attention
    const float* dA_data = (float*)boat_tensor_data(grad_attn);
    const float* A_data = (float*)boat_tensor_data(attention_weights);
    // Create output tensor for gradient scores
    boat_tensor_t* grad_scores = boat_tensor_create_like(attention_weights);
    if (!grad_scores) {
        boat_tensor_unref(grad_value_local);
        boat_tensor_unref(grad_attn);
        return false;
    }
    float* dS_data = (float*)boat_tensor_data(grad_scores);
    // Compute softmax gradient
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t h = 0; h < num_heads; h++) {
            for (int64_t i = 0; i < seq_len; i++) {
                // Compute sum = sum_j dA[b,h,i,j] * A[b,h,i,j] with double precision
                double sum = 0.0;
                for (int64_t j = 0; j < seq_len; j++) {
                    int64_t idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    sum += (double)dA_data[idx] * (double)A_data[idx];
                }
                // Compute dS = A * (dA - sum)
                for (int64_t j = 0; j < seq_len; j++) {
                    int64_t idx = ((b * num_heads + h) * seq_len + i) * seq_len + j;
                    dS_data[idx] = A_data[idx] * (dA_data[idx] - sum);
                }
            }
        }
    }
    // Now we have grad_scores, can free grad_attn
    boat_tensor_unref(grad_attn);
    // Skip the previous dA_minus_sum step, proceed to scaling
    // grad_scores already computed above, skip this step
    // grad_attn already freed above

    // Step 4: Apply scale factor to scores gradient (since scores = QK^T * scale)
    // dS_scaled = dS * scale
    boat_tensor_t* grad_scores_scaled = boat_mul_scalar(grad_scores, scale);
    boat_tensor_unref(grad_scores);
    if (!grad_scores_scaled) {
        boat_tensor_unref(grad_value_local);
        return false;
    }

    // Step 5: Compute gradient for query: dQ = dS_scaled @ K
    boat_tensor_t* grad_query_local = boat_matmul(grad_scores_scaled, key);
    if (!grad_query_local) {
        boat_tensor_unref(grad_value_local);
        boat_tensor_unref(grad_scores_scaled);
        return false;
    }

    // Step 6: Compute gradient for key: dK = dS_scaled^T @ Q
    boat_tensor_t* grad_scores_scaled_t = boat_transpose(grad_scores_scaled, 2, 3);
    boat_tensor_unref(grad_scores_scaled);
    if (!grad_scores_scaled_t) {
        boat_tensor_unref(grad_value_local);
        boat_tensor_unref(grad_query_local);
        return false;
    }
    boat_tensor_t* grad_key_local = boat_matmul(grad_scores_scaled_t, query);
    boat_tensor_unref(grad_scores_scaled_t);
    if (!grad_key_local) {
        boat_tensor_unref(grad_value_local);
        boat_tensor_unref(grad_query_local);
        return false;
    }

    // Return results
    if (grad_query) *grad_query = grad_query_local;
    if (grad_key) *grad_key = grad_key_local;
    if (grad_value) *grad_value = grad_value_local;

    return true;
}

// Accessor functions for testing
BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_get_weight_q(const boat_attention_t* attention) {
    const struct boat_attention_t* attn = (const struct boat_attention_t*)attention;
    if (!attn) return NULL;
    return attn->weight_q;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_get_weight_k(const boat_attention_t* attention) {
    const struct boat_attention_t* attn = (const struct boat_attention_t*)attention;
    if (!attn) return NULL;
    return attn->weight_k;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_get_weight_v(const boat_attention_t* attention) {
    const struct boat_attention_t* attn = (const struct boat_attention_t*)attention;
    if (!attn) return NULL;
    return attn->weight_v;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_get_weight_o(const boat_attention_t* attention) {
    const struct boat_attention_t* attn = (const struct boat_attention_t*)attention;
    if (!attn) return NULL;
    return attn->weight_o;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_get_bias_q(const boat_attention_t* attention) {
    const struct boat_attention_t* attn = (const struct boat_attention_t*)attention;
    if (!attn) return NULL;
    return attn->bias_q;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_get_bias_k(const boat_attention_t* attention) {
    const struct boat_attention_t* attn = (const struct boat_attention_t*)attention;
    if (!attn) return NULL;
    return attn->bias_k;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_get_bias_v(const boat_attention_t* attention) {
    const struct boat_attention_t* attn = (const struct boat_attention_t*)attention;
    if (!attn) return NULL;
    return attn->bias_v;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_get_bias_o(const boat_attention_t* attention) {
    const struct boat_attention_t* attn = (const struct boat_attention_t*)attention;
    if (!attn) return NULL;
    return attn->bias_o;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_get_grad_weight_q(const boat_attention_t* attention) {
    const struct boat_attention_t* attn = (const struct boat_attention_t*)attention;
    if (!attn) return NULL;
    return attn->grad_weight_q;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_get_grad_weight_k(const boat_attention_t* attention) {
    const struct boat_attention_t* attn = (const struct boat_attention_t*)attention;
    if (!attn) return NULL;
    return attn->grad_weight_k;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_get_grad_weight_v(const boat_attention_t* attention) {
    const struct boat_attention_t* attn = (const struct boat_attention_t*)attention;
    if (!attn) return NULL;
    return attn->grad_weight_v;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_get_grad_weight_o(const boat_attention_t* attention) {
    const struct boat_attention_t* attn = (const struct boat_attention_t*)attention;
    if (!attn) return NULL;
    return attn->grad_weight_o;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_get_grad_bias_q(const boat_attention_t* attention) {
    const struct boat_attention_t* attn = (const struct boat_attention_t*)attention;
    if (!attn) return NULL;
    return attn->grad_bias_q;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_get_grad_bias_k(const boat_attention_t* attention) {
    const struct boat_attention_t* attn = (const struct boat_attention_t*)attention;
    if (!attn) return NULL;
    return attn->grad_bias_k;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_get_grad_bias_v(const boat_attention_t* attention) {
    const struct boat_attention_t* attn = (const struct boat_attention_t*)attention;
    if (!attn) return NULL;
    return attn->grad_bias_v;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_attention_get_grad_bias_o(const boat_attention_t* attention) {
    const struct boat_attention_t* attn = (const struct boat_attention_t*)attention;
    if (!attn) return NULL;
    return attn->grad_bias_o;
}

// Weight setters for model loading
BOAT_API BOAT_NOINLINE void BOAT_CALL boat_attention_set_weight_q(boat_attention_t* attention,
                                                                  boat_tensor_t* w) {
    struct boat_attention_t* attn = (struct boat_attention_t*)attention;
    if (!attn || !w) return;
    if (attn->weight_q) boat_tensor_free(attn->weight_q);
    attn->weight_q = w;
    boat_tensor_ref(w);
}

BOAT_API BOAT_NOINLINE void BOAT_CALL boat_attention_set_weight_k(boat_attention_t* attention,
                                                                  boat_tensor_t* w) {
    struct boat_attention_t* attn = (struct boat_attention_t*)attention;
    if (!attn || !w) return;
    if (attn->weight_k) boat_tensor_free(attn->weight_k);
    attn->weight_k = w;
    boat_tensor_ref(w);
}

BOAT_API BOAT_NOINLINE void BOAT_CALL boat_attention_set_weight_v(boat_attention_t* attention,
                                                                  boat_tensor_t* w) {
    struct boat_attention_t* attn = (struct boat_attention_t*)attention;
    if (!attn || !w) return;
    if (attn->weight_v) boat_tensor_free(attn->weight_v);
    attn->weight_v = w;
    boat_tensor_ref(w);
}

BOAT_API BOAT_NOINLINE void BOAT_CALL boat_attention_set_weight_o(boat_attention_t* attention,
                                                                  boat_tensor_t* w) {
    struct boat_attention_t* attn = (struct boat_attention_t*)attention;
    if (!attn || !w) return;
    if (attn->weight_o) boat_tensor_free(attn->weight_o);
    attn->weight_o = w;
    boat_tensor_ref(w);
}