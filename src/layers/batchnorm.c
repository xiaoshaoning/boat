// batchnorm.c - Batch normalization layer implementation (BatchNorm2d)
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat.h>
#include <boat/cuda_runtime.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Batch normalization layer structure (BatchNorm2d)
struct boat_batchnorm2d_layer_t {
    size_t num_features;
    float eps;
    float momentum;
    bool affine;
    bool training;  // Whether in training mode

    // Learnable parameters (if affine is true)
    boat_tensor_t* weight;   // gamma
    boat_tensor_t* bias;     // beta

    // Running statistics
    boat_tensor_t* running_mean;
    boat_tensor_t* running_var;

    // Gradient accumulators
    boat_tensor_t* grad_weight;
    boat_tensor_t* grad_bias;

    // Cache for backward pass
    boat_tensor_t* cache_input;
    boat_tensor_t* cache_save_mean;
    boat_tensor_t* cache_save_inv_var;  // 1/sqrt(var + eps)
};

BOAT_API boat_batchnorm2d_layer_t* BOAT_CALL boat_batchnorm2d_layer_create(size_t num_features, float eps, float momentum, bool affine) {
    boat_batchnorm2d_layer_t* layer = (boat_batchnorm2d_layer_t*)boat_malloc(sizeof(boat_batchnorm2d_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        return NULL;
    }

    layer->num_features = num_features;
    layer->eps = eps;
    layer->momentum = momentum;
    layer->affine = affine;
    layer->training = false; // Default to inference mode

    layer->weight = NULL;
    layer->bias = NULL;
    layer->grad_weight = NULL;
    layer->grad_bias = NULL;
    layer->cache_input = NULL;
    layer->cache_save_mean = NULL;
    layer->cache_save_inv_var = NULL;

    // Create running mean tensor: [num_features]
    const int64_t running_shape[] = { (int64_t)num_features };
    layer->running_mean = boat_tensor_create(running_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!layer->running_mean) {
        boat_free(layer);
        return NULL;
    }

    // Initialize running mean to zeros
    float* running_mean_data = (float*)boat_tensor_data(layer->running_mean);
    size_t running_mean_elements = boat_tensor_nelements(layer->running_mean);
    memset(running_mean_data, 0, running_mean_elements * sizeof(float));

    // Create running variance tensor: [num_features]
    layer->running_var = boat_tensor_create(running_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!layer->running_var) {
        boat_tensor_free(layer->running_mean);
        boat_free(layer);
        return NULL;
    }

    // Initialize running variance to ones
    float* running_var_data = (float*)boat_tensor_data(layer->running_var);
    size_t running_var_elements = boat_tensor_nelements(layer->running_var);
    for (size_t i = 0; i < running_var_elements; i++) {
        running_var_data[i] = 1.0f;
    }

    // Create weight and bias if affine is true
    if (affine) {
        layer->weight = boat_tensor_create(running_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (!layer->weight) {
            boat_tensor_free(layer->running_mean);
            boat_tensor_free(layer->running_var);
            boat_free(layer);
            return NULL;
        }

        // Initialize weight to ones
        float* weight_data = (float*)boat_tensor_data(layer->weight);
        for (size_t i = 0; i < num_features; i++) {
            weight_data[i] = 1.0f;
        }

        layer->bias = boat_tensor_create(running_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (!layer->bias) {
            boat_tensor_free(layer->running_mean);
            boat_tensor_free(layer->running_var);
            boat_tensor_free(layer->weight);
            boat_free(layer);
            return NULL;
        }

        // Initialize bias to zeros
        float* bias_data = (float*)boat_tensor_data(layer->bias);
        memset(bias_data, 0, num_features * sizeof(float));

        // Create gradient accumulators
        layer->grad_weight = boat_tensor_create(running_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (!layer->grad_weight) {
            boat_tensor_free(layer->running_mean);
            boat_tensor_free(layer->running_var);
            boat_tensor_free(layer->weight);
            boat_tensor_free(layer->bias);
            boat_free(layer);
            return NULL;
        }
        memset(boat_tensor_data(layer->grad_weight), 0, num_features * sizeof(float));

        layer->grad_bias = boat_tensor_create(running_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (!layer->grad_bias) {
            boat_tensor_free(layer->running_mean);
            boat_tensor_free(layer->running_var);
            boat_tensor_free(layer->weight);
            boat_tensor_free(layer->bias);
            boat_tensor_free(layer->grad_weight);
            boat_free(layer);
            return NULL;
        }
        memset(boat_tensor_data(layer->grad_bias), 0, num_features * sizeof(float));
    }

    return layer;
}

BOAT_API void BOAT_CALL boat_batchnorm2d_layer_free(boat_batchnorm2d_layer_t* layer) {
    if (!layer) {
        return;
    }

    if (layer->weight) boat_tensor_free(layer->weight);
    if (layer->bias) boat_tensor_free(layer->bias);
    if (layer->grad_weight) boat_tensor_free(layer->grad_weight);
    if (layer->grad_bias) boat_tensor_free(layer->grad_bias);
    if (layer->running_mean) boat_tensor_free(layer->running_mean);
    if (layer->running_var) boat_tensor_free(layer->running_var);
    if (layer->cache_input) boat_tensor_free(layer->cache_input);
    if (layer->cache_save_mean) boat_tensor_free(layer->cache_save_mean);
    if (layer->cache_save_inv_var) boat_tensor_free(layer->cache_save_inv_var);
    boat_free(layer);
}

BOAT_API boat_tensor_t* BOAT_CALL boat_batchnorm2d_layer_forward(const boat_batchnorm2d_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !input) {
        return NULL;
    }

    // Input should be 4D: [batch, channels, height, width]
    const int64_t* input_shape = boat_tensor_shape(input);
    if (boat_tensor_ndim(input) != 4) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[BatchNormLayer] BatchNorm2d expects 4D input tensor\n");
        return NULL;
    }

    int64_t channels = input_shape[1];
    if ((size_t)channels != layer->num_features) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[BatchNormLayer] Input channels %lld don't match layer num_features %zu\n", channels, layer->num_features);
        return NULL;
    }

    if (boat_tensor_dtype(input) != BOAT_DTYPE_FLOAT32) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[BatchNormLayer] Only FLOAT32 input supported\n");
        return NULL;
    }

    boat_device_t device = boat_tensor_device(input);
    int64_t N = input_shape[0];
    int64_t C = input_shape[1];
    int64_t H = input_shape[2];
    int64_t W = input_shape[3];
    int64_t spatial_size = H * W;
    int64_t elements_per_channel = N * spatial_size;

    // Create output tensor on same device
    boat_tensor_t* output = boat_tensor_create(input_shape, 4, BOAT_DTYPE_FLOAT32, device);
    if (!output) return NULL;

    if (layer->training) {
        // --- Training mode: compute batch statistics ---
#ifdef BOAT_WITH_CUDNN
        if (device == BOAT_DEVICE_CUDA) {
            // cuDNN forward handles mean/var internally
            const float* d_input = (const float*)boat_tensor_data(input);
            float* d_output = (float*)boat_tensor_data(output);

            // Allocate device tensors for save_mean and save_inv_var
            const int64_t stats_shape[] = { C };
            // We need mutable layer access to cache — cast away const since the
            // cache is an implementation detail (not a user-visible mutation)
            boat_batchnorm2d_layer_t* mutable_layer = (boat_batchnorm2d_layer_t*)layer;

            // Free old cached tensors and create new ones on CUDA
            if (mutable_layer->cache_save_mean) boat_tensor_free(mutable_layer->cache_save_mean);
            if (mutable_layer->cache_save_inv_var) boat_tensor_free(mutable_layer->cache_save_inv_var);
            mutable_layer->cache_save_mean = boat_tensor_create(stats_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CUDA);
            mutable_layer->cache_save_inv_var = boat_tensor_create(stats_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CUDA);
            float* d_save_mean = (float*)boat_tensor_data(mutable_layer->cache_save_mean);
            float* d_save_inv_var = (float*)boat_tensor_data(mutable_layer->cache_save_inv_var);

            // Get gamma and beta (may be NULL if not affine)
            const float* d_gamma = layer->affine ? (const float*)boat_tensor_const_data(layer->weight) : NULL;
            const float* d_beta = layer->affine ? (const float*)boat_tensor_const_data(layer->bias) : NULL;

            // cuDNN needs its own saveInvVariance buffer — we use cache_save_inv_var directly
            float* d_var = NULL;  // We'll compute var if needed for running stats

            // Use the cuDNN forward wrapper
            boat_cuda_batchnorm_cudnn_forward_f32(d_input, d_output,
                d_gamma, d_beta,
                d_save_mean,    // cuDNN writes saveMean here
                d_save_inv_var, // cuDNN writes var here (after inv_var_to_var conversion)
                (size_t)N, (size_t)C, (size_t)H, (size_t)W, layer->eps);

            // Convert var back to inv_var in-place for backward compatibility
            boat_cuda_var_to_inv_var_f32(d_save_inv_var, (size_t)C, layer->eps);

            // Update running stats via EMA
            // Copy save_mean and converted inv_var to host for running stats update
            float* h_mean = (float*)malloc((size_t)C * sizeof(float));
            float* h_var = (float*)malloc((size_t)C * sizeof(float));
            if (h_mean && h_var) {
                // Copy save_mean from device
                boat_cuda_memcpy_d2h(h_mean, d_save_mean, (size_t)C * sizeof(float));
                // Copy inv_var from device, then compute var = 1/(inv_var^2) - eps
                boat_cuda_memcpy_d2h(h_var, d_save_inv_var, (size_t)C * sizeof(float));
                for (int64_t c = 0; c < C; c++) {
                    float iv = h_var[c];
                    h_var[c] = 1.0f / (iv * iv) - layer->eps;
                    if (h_var[c] < 0.0f) h_var[c] = 0.0f;
                }
                // Update running stats: running = momentum * running + (1-momentum) * batch
                float* rmean = (float*)boat_tensor_data(mutable_layer->running_mean);
                float* rvar = (float*)boat_tensor_data(mutable_layer->running_var);
                float m = layer->momentum;
                float inv_m = 1.0f - m;
                for (int64_t c = 0; c < C; c++) {
                    rmean[c] = m * rmean[c] + inv_m * h_mean[c];
                    rvar[c]  = m * rvar[c]  + inv_m * h_var[c];
                }
            }
            free(h_mean);
            free(h_var);

            // Cache input for backward
            if (mutable_layer->cache_input) boat_tensor_free(mutable_layer->cache_input);
            mutable_layer->cache_input = (boat_tensor_t*)input;
            boat_tensor_ref(mutable_layer->cache_input);

            return output;
        }
#endif
        // --- CPU training path ---
        const float* input_data = (const float*)boat_tensor_data(input);
        float* output_data = (float*)boat_tensor_data(output);
        const float* gamma_data = layer->affine ? (const float*)boat_tensor_const_data(layer->weight) : NULL;
        const float* beta_data = layer->affine ? (const float*)boat_tensor_const_data(layer->bias) : NULL;

        // Step 1: Compute mean per channel
        float* mean = (float*)calloc((size_t)C, sizeof(float));
        float* var = (float*)calloc((size_t)C, sizeof(float));
        if (!mean || !var) {
            free(mean); free(var);
            boat_tensor_free(output);
            return NULL;
        }

        for (int64_t c = 0; c < C; c++) {
            double sum = 0.0;
            for (int64_t n = 0; n < N; n++) {
                for (int64_t h = 0; h < H; h++) {
                    for (int64_t w = 0; w < W; w++) {
                        size_t idx = (size_t)((n * C + c) * H + h) * (size_t)W + (size_t)w;
                        sum += input_data[idx];
                    }
                }
            }
            mean[c] = (float)(sum / (double)elements_per_channel);
        }

        // Step 2: Compute variance per channel
        for (int64_t c = 0; c < C; c++) {
            double sum_sq = 0.0;
            for (int64_t n = 0; n < N; n++) {
                for (int64_t h = 0; h < H; h++) {
                    for (int64_t w = 0; w < W; w++) {
                        size_t idx = (size_t)((n * C + c) * H + h) * (size_t)W + (size_t)w;
                        double diff = input_data[idx] - mean[c];
                        sum_sq += diff * diff;
                    }
                }
            }
            var[c] = (float)(sum_sq / (double)elements_per_channel);
        }

        // Step 3: Normalize and cache save_mean + inv_std
        boat_batchnorm2d_layer_t* mutable_layer = (boat_batchnorm2d_layer_t*)layer;
        const int64_t stats_shape[] = { C };
        if (mutable_layer->cache_save_mean) boat_tensor_free(mutable_layer->cache_save_mean);
        mutable_layer->cache_save_mean = boat_tensor_create(stats_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (mutable_layer->cache_save_inv_var) boat_tensor_free(mutable_layer->cache_save_inv_var);
        mutable_layer->cache_save_inv_var = boat_tensor_create(stats_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        float* save_mean = (float*)boat_tensor_data(mutable_layer->cache_save_mean);
        float* save_inv_var = (float*)boat_tensor_data(mutable_layer->cache_save_inv_var);

        for (int64_t c = 0; c < C; c++) {
            float inv_std = 1.0f / sqrtf(var[c] + layer->eps);
            save_mean[c] = mean[c];
            save_inv_var[c] = inv_std;

            for (int64_t n = 0; n < N; n++) {
                for (int64_t h = 0; h < H; h++) {
                    for (int64_t w = 0; w < W; w++) {
                        size_t idx = (size_t)((n * C + c) * H + h) * (size_t)W + (size_t)w;
                        float x_norm = (input_data[idx] - mean[c]) * inv_std;
                        output_data[idx] = x_norm;
                        if (gamma_data) output_data[idx] *= gamma_data[c];
                        if (beta_data)  output_data[idx] += beta_data[c];
                    }
                }
            }

            // Update running stats
            float* rmean = (float*)boat_tensor_data(mutable_layer->running_mean);
            float* rvar = (float*)boat_tensor_data(mutable_layer->running_var);
            float m = layer->momentum;
            float inv_m = 1.0f - m;
            rmean[c] = m * rmean[c] + inv_m * mean[c];
            rvar[c]  = m * rvar[c]  + inv_m * var[c];
        }

        free(mean);
        free(var);

        // Cache input for backward
        if (mutable_layer->cache_input) boat_tensor_free(mutable_layer->cache_input);
        mutable_layer->cache_input = (boat_tensor_t*)input;
        boat_tensor_ref(mutable_layer->cache_input);

    } else {
        // --- Inference mode: use running statistics ---
        const float* input_data = (const float*)boat_tensor_data(input);
        float* output_data = (float*)boat_tensor_data(output);
        const float* gamma_data = layer->affine ? (const float*)boat_tensor_const_data(layer->weight) : NULL;
        const float* beta_data = layer->affine ? (const float*)boat_tensor_const_data(layer->bias) : NULL;
        const float* rmean = (const float*)boat_tensor_const_data(layer->running_mean);
        const float* rvar = (const float*)boat_tensor_const_data(layer->running_var);

        for (int64_t c = 0; c < C; c++) {
            float inv_std = 1.0f / sqrtf(rvar[c] + layer->eps);
            float g = gamma_data ? gamma_data[c] : 1.0f;
            float b = beta_data ? beta_data[c] : 0.0f;

            for (int64_t n = 0; n < N; n++) {
                for (int64_t h = 0; h < H; h++) {
                    for (int64_t w = 0; w < W; w++) {
                        size_t idx = (size_t)((n * C + c) * H + h) * (size_t)W + (size_t)w;
                        output_data[idx] = g * (input_data[idx] - rmean[c]) * inv_std + b;
                    }
                }
            }
        }
    }

    return output;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_batchnorm2d_layer_backward(boat_batchnorm2d_layer_t* layer, const boat_tensor_t* grad_output) {
    if (!layer || !grad_output) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[BatchNormLayer] bn backward: NULL input\n");
        return NULL;
    }
    if (!layer->cache_input || !layer->cache_save_mean || !layer->cache_save_inv_var) {
        boat_set_errorf(BOAT_ERROR_INVALID_OPERATION, "[BatchNormLayer] bn backward: forward not called or cache cleared\n");
        return NULL;
    }

    const int64_t* grad_shape = boat_tensor_shape(grad_output);
    if (boat_tensor_ndim(grad_output) != 4) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[BatchNormLayer] bn backward: expected 4D grad_output\n");
        return NULL;
    }

    int64_t N = grad_shape[0];
    int64_t C = grad_shape[1];
    int64_t H = grad_shape[2];
    int64_t W = grad_shape[3];
    int64_t spatial_size = H * W;
    int64_t elements_per_channel = N * spatial_size;
    boat_device_t device = boat_tensor_device(grad_output);

    // Create grad_input on same device
    boat_tensor_t* grad_input = boat_tensor_create(grad_shape, 4, BOAT_DTYPE_FLOAT32, device);
    if (!grad_input) return NULL;

    // --- cuDNN backward dispatch ---
#ifdef BOAT_WITH_CUDNN
    if (device == BOAT_DEVICE_CUDA) {
        const float* d_input = (const float*)boat_tensor_const_data(layer->cache_input);
        const float* d_grad_output = (const float*)boat_tensor_data(grad_output);
        float* d_grad_input = (float*)boat_tensor_data(grad_input);
        const float* d_gamma = layer->affine ? (const float*)boat_tensor_const_data(layer->weight) : NULL;
        const float* d_save_mean = (const float*)boat_tensor_const_data(layer->cache_save_mean);
        const float* d_save_inv_var = (const float*)boat_tensor_const_data(layer->cache_save_inv_var);

        // Create/replace CUDA grad_weight and grad_bias
        const int64_t stats_shape[] = { C };
        if (layer->grad_weight) {
            if (boat_tensor_device(layer->grad_weight) != BOAT_DEVICE_CUDA) {
                boat_tensor_free(layer->grad_weight);
                layer->grad_weight = boat_tensor_create(stats_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CUDA);
            }
        } else if (layer->affine) {
            layer->grad_weight = boat_tensor_create(stats_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CUDA);
        }
        if (layer->grad_bias) {
            if (boat_tensor_device(layer->grad_bias) != BOAT_DEVICE_CUDA) {
                boat_tensor_free(layer->grad_bias);
                layer->grad_bias = boat_tensor_create(stats_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CUDA);
            }
        } else if (layer->affine) {
            layer->grad_bias = boat_tensor_create(stats_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CUDA);
        }

        float* d_grad_gamma = layer->affine ? (float*)boat_tensor_data(layer->grad_weight) : NULL;
        float* d_grad_beta = layer->affine ? (float*)boat_tensor_data(layer->grad_bias) : NULL;

        boat_cuda_batchnorm_cudnn_backward_f32(d_input, d_grad_output, d_grad_input,
            d_gamma, d_grad_gamma, d_grad_beta,
            d_save_mean, d_save_inv_var,
            (size_t)N, (size_t)C, (size_t)H, (size_t)W, layer->eps);

        return grad_input;
    }
#endif

    // --- CPU backward ---
    const float* input_data = (const float*)boat_tensor_const_data(layer->cache_input);
    const float* grad_output_data = (const float*)boat_tensor_data(grad_output);
    float* grad_input_data = (float*)boat_tensor_data(grad_input);
    const float* save_mean = (const float*)boat_tensor_const_data(layer->cache_save_mean);
    const float* save_inv_var = (const float*)boat_tensor_const_data(layer->cache_save_inv_var);
    const float* gamma_data = layer->affine ? (const float*)boat_tensor_const_data(layer->weight) : NULL;

    // Compute dL/dbeta and dL/dgamma
    float* grad_beta_data = NULL;
    float* grad_gamma_data = NULL;
    if (layer->affine) {
        if (!layer->grad_weight) {
            const int64_t stats_shape[] = { C };
            layer->grad_weight = boat_tensor_create(stats_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        }
        if (!layer->grad_bias) {
            const int64_t stats_shape[] = { C };
            layer->grad_bias = boat_tensor_create(stats_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        }
        grad_beta_data = (float*)boat_tensor_data(layer->grad_bias);
        grad_gamma_data = (float*)boat_tensor_data(layer->grad_weight);
        memset(grad_beta_data, 0, (size_t)C * sizeof(float));
        memset(grad_gamma_data, 0, (size_t)C * sizeof(float));
    }

    float* dbeta = grad_beta_data;
    float* dgamma = grad_gamma_data;

    // Pre-compute dbeta and dgamma
    if (dbeta || dgamma) {
        for (int64_t c = 0; c < C; c++) {
            float sum_beta = 0.0f;
            float sum_gamma = 0.0f;
            float inv_std = save_inv_var[c];
            float mu = save_mean[c];

            for (int64_t n = 0; n < N; n++) {
                for (int64_t h = 0; h < H; h++) {
                    for (int64_t w = 0; w < W; w++) {
                        size_t idx = (size_t)((n * C + c) * H + h) * (size_t)W + (size_t)w;
                        float go = grad_output_data[idx];
                        sum_beta += go;
                        if (dgamma) {
                            float x_norm = (input_data[idx] - mu) * inv_std;
                            sum_gamma += go * x_norm;
                        }
                    }
                }
            }

            if (dbeta)  dbeta[c] = sum_beta;
            if (dgamma) dgamma[c] = sum_gamma;
        }
    }

    // Compute dL/dx
    float inv_nelem = 1.0f / (float)elements_per_channel;
    for (int64_t c = 0; c < C; c++) {
        float inv_std = save_inv_var[c];
        float gamma = gamma_data ? gamma_data[c] : 1.0f;
        float factor = gamma * inv_std;
        float mu = save_mean[c];

        // Pre-compute common terms
        float dbeta_c = dbeta ? dbeta[c] : 0.0f;
        float sum_gamma_norm = 0.0f;
        if (dgamma) {
            // Σ (dL/dy * x_norm)
            for (int64_t n = 0; n < N; n++) {
                for (int64_t h = 0; h < H; h++) {
                    for (int64_t w = 0; w < W; w++) {
                        size_t idx = (size_t)((n * C + c) * H + h) * (size_t)W + (size_t)w;
                        float x_norm = (input_data[idx] - mu) * inv_std;
                        sum_gamma_norm += grad_output_data[idx] * x_norm;
                    }
                }
            }
        }

        for (int64_t n = 0; n < N; n++) {
            for (int64_t h = 0; h < H; h++) {
                for (int64_t w = 0; w < W; w++) {
                    size_t idx = (size_t)((n * C + c) * H + h) * (size_t)W + (size_t)w;
                    float x_norm = (input_data[idx] - mu) * inv_std;
                    // dL/dx = (gamma * inv_std / N_elem) * (N_elem * dL/dy - dbeta - x_norm * dgamma)
                    float grad = factor * (grad_output_data[idx] - dbeta_c * inv_nelem - x_norm * sum_gamma_norm * inv_nelem);
                    grad_input_data[idx] = grad;
                }
            }
        }
    }

    return grad_input;
}

BOAT_API void BOAT_CALL boat_batchnorm2d_layer_update(boat_batchnorm2d_layer_t* layer, float learning_rate) {
    if (!layer || !layer->affine) return;

    if (layer->grad_weight && layer->weight) {
        boat_tensor_t* scaled_grad = boat_mul_scalar(layer->grad_weight, learning_rate);
        if (scaled_grad) {
            boat_sub_(layer->weight, scaled_grad);
            boat_tensor_unref(scaled_grad);
        }
    }

    if (layer->grad_bias && layer->bias) {
        boat_tensor_t* scaled_grad = boat_mul_scalar(layer->grad_bias, learning_rate);
        if (scaled_grad) {
            boat_sub_(layer->bias, scaled_grad);
            boat_tensor_unref(scaled_grad);
        }
    }
}

// Parameter access functions
BOAT_API void BOAT_CALL boat_batchnorm2d_layer_set_weight(boat_batchnorm2d_layer_t* layer, boat_tensor_t* weight) {
    if (!layer || !weight) {
        return;
    }
    if (!layer->affine) {
        BOAT_DEBUG_PRINT("[BatchNormLayer] Warning: Layer was created without affine transform, ignoring weight tensor\n");
        return;
    }
    const int64_t* weight_shape = boat_tensor_shape(weight);
    if (weight_shape[0] != (int64_t)layer->num_features) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[BatchNormLayer] Weight shape [%lld] does not match num_features %zu\n",
                weight_shape[0], layer->num_features);
        return;
    }
    if (layer->weight) {
        boat_tensor_free(layer->weight);
    }
    layer->weight = weight;
    boat_tensor_ref(weight);
}

BOAT_API void BOAT_CALL boat_batchnorm2d_layer_set_bias(boat_batchnorm2d_layer_t* layer, boat_tensor_t* bias) {
    if (!layer || !bias) {
        return;
    }
    if (!layer->affine) {
        BOAT_DEBUG_PRINT("[BatchNormLayer] Warning: Layer was created without affine transform, ignoring bias tensor\n");
        return;
    }
    const int64_t* bias_shape = boat_tensor_shape(bias);
    if (bias_shape[0] != (int64_t)layer->num_features) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[BatchNormLayer] Bias shape [%lld] does not match num_features %zu\n",
                bias_shape[0], layer->num_features);
        return;
    }
    if (layer->bias) {
        boat_tensor_free(layer->bias);
    }
    layer->bias = bias;
    boat_tensor_ref(bias);
}

BOAT_API void BOAT_CALL boat_batchnorm2d_layer_set_running_mean(boat_batchnorm2d_layer_t* layer, boat_tensor_t* running_mean) {
    if (!layer || !running_mean) {
        return;
    }
    const int64_t* running_mean_shape = boat_tensor_shape(running_mean);
    if (running_mean_shape[0] != (int64_t)layer->num_features) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[BatchNormLayer] Running mean shape [%lld] does not match num_features %zu\n",
                running_mean_shape[0], layer->num_features);
        return;
    }
    if (layer->running_mean) {
        boat_tensor_free(layer->running_mean);
    }
    layer->running_mean = running_mean;
    boat_tensor_ref(running_mean);
}

BOAT_API void BOAT_CALL boat_batchnorm2d_layer_set_running_var(boat_batchnorm2d_layer_t* layer, boat_tensor_t* running_var) {
    if (!layer || !running_var) {
        return;
    }
    const int64_t* running_var_shape = boat_tensor_shape(running_var);
    if (running_var_shape[0] != (int64_t)layer->num_features) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[BatchNormLayer] Running var shape [%lld] does not match num_features %zu\n",
                running_var_shape[0], layer->num_features);
        return;
    }
    if (layer->running_var) {
        boat_tensor_free(layer->running_var);
    }
    layer->running_var = running_var;
    boat_tensor_ref(running_var);
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL boat_batchnorm2d_layer_get_weight(const boat_batchnorm2d_layer_t* layer) {
    return layer ? layer->weight : NULL;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL boat_batchnorm2d_layer_get_bias(const boat_batchnorm2d_layer_t* layer) {
    return layer ? layer->bias : NULL;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL boat_batchnorm2d_layer_get_running_mean(const boat_batchnorm2d_layer_t* layer) {
    return layer ? layer->running_mean : NULL;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL boat_batchnorm2d_layer_get_running_var(const boat_batchnorm2d_layer_t* layer) {
    return layer ? layer->running_var : NULL;
}

BOAT_API BOAT_NOINLINE float BOAT_CALL boat_batchnorm2d_layer_get_eps(const boat_batchnorm2d_layer_t* layer) {
    return layer ? layer->eps : 0.0f;
}

BOAT_API BOAT_NOINLINE float BOAT_CALL boat_batchnorm2d_layer_get_momentum(const boat_batchnorm2d_layer_t* layer) {
    return layer ? layer->momentum : 0.0f;
}

BOAT_API BOAT_NOINLINE bool BOAT_CALL boat_batchnorm2d_layer_get_affine(const boat_batchnorm2d_layer_t* layer) {
    return layer ? layer->affine : false;
}