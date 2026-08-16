// layers.h - Neural network layer definitions
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_LAYERS_H
#define BOAT_LAYERS_H

#include "tensor.h"
#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Multi-input layer argument: one tensor per incoming edge (merge layers).
typedef struct {
    const boat_tensor_t* t;
} boat_layer_input_t;

// Layer type enumeration for serialization
typedef enum {
    BOAT_LAYER_TYPE_UNKNOWN = 0,
    BOAT_LAYER_TYPE_DENSE,
    BOAT_LAYER_TYPE_CONV2D,
    BOAT_LAYER_TYPE_BATCHNORM2D,
    BOAT_LAYER_TYPE_MAXPOOL2D,
    BOAT_LAYER_TYPE_RELU,
    BOAT_LAYER_TYPE_SOFTMAX,
    BOAT_LAYER_TYPE_FLATTEN,
    BOAT_LAYER_TYPE_LAYERNORM,
    BOAT_LAYER_TYPE_RMSNORM,
    BOAT_LAYER_TYPE_LSTM,
    BOAT_LAYER_TYPE_GRU,
    BOAT_LAYER_TYPE_ATTENTION,
    BOAT_LAYER_TYPE_PRELU,
    BOAT_LAYER_TYPE_EMBEDDING,
    BOAT_LAYER_TYPE_CONCAT,  // Multi-input merge: join along an axis
    BOAT_LAYER_TYPE_ADD,     // Multi-input merge: element-wise sum
    BOAT_LAYER_TYPE_COUNT
} boat_layer_type_t;

// Forward declarations for all layer types
typedef struct boat_dense_layer_t boat_dense_layer_t;
typedef struct boat_conv_layer_t boat_conv_layer_t;
typedef struct boat_pool_layer_t boat_pool_layer_t;
typedef struct boat_norm_layer_t boat_norm_layer_t;
typedef struct boat_attention_layer_t boat_attention_layer_t;
typedef struct boat_swin_t boat_swin_t;
typedef struct boat_decoder_layer_t boat_decoder_layer_t;

// Dense layer functions
BOAT_API boat_dense_layer_t* BOAT_CALL boat_dense_layer_create(size_t input_features,
                                                               size_t output_features,
                                                               bool use_bias);
BOAT_API void BOAT_CALL boat_dense_layer_free(boat_dense_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_dense_layer_forward(boat_dense_layer_t* layer,
                                                           const boat_tensor_t* input);
BOAT_API boat_tensor_t* BOAT_CALL boat_dense_layer_backward(boat_dense_layer_t* layer,
                                                            const boat_tensor_t* grad_output);
BOAT_API void BOAT_CALL boat_dense_layer_update(boat_dense_layer_t* layer, float learning_rate);

// Parameter access for model loading
BOAT_API void BOAT_CALL boat_dense_layer_set_weight(boat_dense_layer_t* layer,
                                                    boat_tensor_t* weight);
BOAT_API void BOAT_CALL boat_dense_layer_set_bias(boat_dense_layer_t* layer, boat_tensor_t* bias);
BOAT_API boat_tensor_t* BOAT_CALL boat_dense_layer_get_weight(const boat_dense_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_dense_layer_get_bias(const boat_dense_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_dense_layer_get_grad_weight(const boat_dense_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_dense_layer_get_grad_bias(const boat_dense_layer_t* layer);

// Convolutional layer functions
BOAT_API boat_conv_layer_t* BOAT_CALL boat_conv_layer_create(size_t in_channels,
                                                             size_t out_channels,
                                                             size_t kernel_size, size_t stride,
                                                             size_t padding, size_t groups);
BOAT_API void BOAT_CALL boat_conv_layer_free(boat_conv_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_forward(boat_conv_layer_t* layer,
                                                          const boat_tensor_t* input);
BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_backward(boat_conv_layer_t* layer,
                                                           const boat_tensor_t* grad_output);
BOAT_API void BOAT_CALL boat_conv_layer_update(boat_conv_layer_t* layer, float learning_rate);

// Parameter access for model loading
BOAT_API void BOAT_CALL boat_conv_layer_set_weight(boat_conv_layer_t* layer, boat_tensor_t* weight);
BOAT_API void BOAT_CALL boat_conv_layer_set_bias(boat_conv_layer_t* layer, boat_tensor_t* bias);
BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_get_weight(const boat_conv_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_get_bias(const boat_conv_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_get_grad_weight(const boat_conv_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_get_grad_bias(const boat_conv_layer_t* layer);
BOAT_API size_t BOAT_CALL boat_conv_layer_get_stride(const boat_conv_layer_t* layer);
BOAT_API size_t BOAT_CALL boat_conv_layer_get_padding(const boat_conv_layer_t* layer);
BOAT_API size_t BOAT_CALL boat_conv_layer_get_groups(const boat_conv_layer_t* layer);

// Batch normalization layer functions (BatchNorm2d)
typedef struct boat_batchnorm2d_layer_t boat_batchnorm2d_layer_t;
BOAT_API boat_batchnorm2d_layer_t* BOAT_CALL boat_batchnorm2d_layer_create(size_t num_features,
                                                                           float eps,
                                                                           float momentum,
                                                                           bool affine);
BOAT_API void BOAT_CALL boat_batchnorm2d_layer_free(boat_batchnorm2d_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL
boat_batchnorm2d_layer_forward(const boat_batchnorm2d_layer_t* layer, const boat_tensor_t* input);
BOAT_API boat_tensor_t* BOAT_CALL boat_batchnorm2d_layer_backward(boat_batchnorm2d_layer_t* layer,
                                                                  const boat_tensor_t* grad_output);
BOAT_API void BOAT_CALL boat_batchnorm2d_layer_update(boat_batchnorm2d_layer_t* layer,
                                                      float learning_rate);
// Training mode: batch statistics vs running statistics.
BOAT_API void BOAT_CALL boat_batchnorm2d_layer_set_training(boat_batchnorm2d_layer_t* layer,
                                                            bool training);
BOAT_API bool BOAT_CALL boat_batchnorm2d_layer_get_training(const boat_batchnorm2d_layer_t* layer);

// Parameter access for BatchNorm2d
BOAT_API void BOAT_CALL boat_batchnorm2d_layer_set_weight(boat_batchnorm2d_layer_t* layer,
                                                          boat_tensor_t* weight);
BOAT_API void BOAT_CALL boat_batchnorm2d_layer_set_bias(boat_batchnorm2d_layer_t* layer,
                                                        boat_tensor_t* bias);
BOAT_API void BOAT_CALL boat_batchnorm2d_layer_set_running_mean(boat_batchnorm2d_layer_t* layer,
                                                                boat_tensor_t* running_mean);
BOAT_API void BOAT_CALL boat_batchnorm2d_layer_set_running_var(boat_batchnorm2d_layer_t* layer,
                                                               boat_tensor_t* running_var);
BOAT_API boat_tensor_t* BOAT_CALL
boat_batchnorm2d_layer_get_weight(const boat_batchnorm2d_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL
boat_batchnorm2d_layer_get_bias(const boat_batchnorm2d_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL
boat_batchnorm2d_layer_get_running_mean(const boat_batchnorm2d_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL
boat_batchnorm2d_layer_get_running_var(const boat_batchnorm2d_layer_t* layer);
BOAT_API float BOAT_CALL boat_batchnorm2d_layer_get_eps(const boat_batchnorm2d_layer_t* layer);
BOAT_API float BOAT_CALL boat_batchnorm2d_layer_get_momentum(const boat_batchnorm2d_layer_t* layer);
BOAT_API bool BOAT_CALL boat_batchnorm2d_layer_get_affine(const boat_batchnorm2d_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL
boat_batchnorm2d_layer_get_grad_weight(const boat_batchnorm2d_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL
boat_batchnorm2d_layer_get_grad_bias(const boat_batchnorm2d_layer_t* layer);

// Pooling layer functions (MaxPool2d)
BOAT_API boat_pool_layer_t* BOAT_CALL boat_pool_layer_create(size_t pool_size, size_t stride,
                                                             size_t padding);
BOAT_API void BOAT_CALL boat_pool_layer_free(boat_pool_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_pool_layer_forward(boat_pool_layer_t* layer,
                                                          const boat_tensor_t* input);
BOAT_API boat_tensor_t* BOAT_CALL boat_pool_layer_backward(boat_pool_layer_t* layer,
                                                           const boat_tensor_t* grad_output);
BOAT_API void BOAT_CALL boat_pool_layer_update(boat_pool_layer_t* layer, float learning_rate);
BOAT_API size_t BOAT_CALL boat_pool_layer_get_pool_size(const boat_pool_layer_t* layer);
BOAT_API size_t BOAT_CALL boat_pool_layer_get_stride(const boat_pool_layer_t* layer);
BOAT_API size_t BOAT_CALL boat_pool_layer_get_padding(const boat_pool_layer_t* layer);

// Normalization layer functions (simplified interface)
BOAT_API boat_norm_layer_t* BOAT_CALL boat_norm_layer_create(size_t normalized_shape, float eps,
                                                             bool elementwise_affine);
BOAT_API void BOAT_CALL boat_norm_layer_free(boat_norm_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_norm_layer_forward(boat_norm_layer_t* layer,
                                                          const boat_tensor_t* input);
BOAT_API boat_tensor_t* BOAT_CALL boat_norm_layer_backward(boat_norm_layer_t* layer,
                                                           const boat_tensor_t* grad_output);
BOAT_API void BOAT_CALL boat_norm_layer_update(boat_norm_layer_t* layer, float learning_rate);

// Attention layer functions (simplified interface)
BOAT_API boat_attention_layer_t* BOAT_CALL boat_attention_layer_create(size_t hidden_size,
                                                                       size_t num_heads,
                                                                       size_t num_kv_heads,
                                                                       float dropout_prob,
                                                                       bool causal_mask);
BOAT_API void BOAT_CALL boat_attention_layer_free(boat_attention_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_layer_forward(boat_attention_layer_t* layer,
                                                               const boat_tensor_t* query,
                                                               const boat_tensor_t* key,
                                                               const boat_tensor_t* value,
                                                               const boat_tensor_t* attention_mask);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_layer_backward(boat_attention_layer_t* layer,
                                                                const boat_tensor_t* grad_output);
BOAT_API void BOAT_CALL boat_attention_layer_update(boat_attention_layer_t* layer,
                                                    float learning_rate);

// Activation layers
typedef struct boat_relu_layer_t boat_relu_layer_t;
BOAT_API boat_relu_layer_t* BOAT_CALL boat_relu_layer_create();
BOAT_API void BOAT_CALL boat_relu_layer_free(boat_relu_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_relu_layer_forward(boat_relu_layer_t* layer,
                                                          const boat_tensor_t* input);
BOAT_API boat_tensor_t* BOAT_CALL boat_relu_layer_backward(boat_relu_layer_t* layer,
                                                           const boat_tensor_t* grad_output);
BOAT_API void BOAT_CALL boat_relu_layer_update(boat_relu_layer_t* layer, float learning_rate);

typedef struct boat_prelu_layer_t boat_prelu_layer_t;
// PReLU layer (Parametric ReLU): f(x) = max(0,x) + slope * min(0,x)
BOAT_API boat_prelu_layer_t* BOAT_CALL boat_prelu_layer_create(size_t num_params);
BOAT_API void BOAT_CALL boat_prelu_layer_free(boat_prelu_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_prelu_layer_forward(const boat_prelu_layer_t* layer,
                                                           const boat_tensor_t* input);
BOAT_API boat_tensor_t* BOAT_CALL boat_prelu_layer_backward(boat_prelu_layer_t* layer,
                                                            const boat_tensor_t* grad_output);
BOAT_API void BOAT_CALL boat_prelu_layer_update(boat_prelu_layer_t* layer, float learning_rate);
BOAT_API void BOAT_CALL boat_prelu_layer_set_slope(boat_prelu_layer_t* layer,
                                                   const boat_tensor_t* slope);
BOAT_API boat_tensor_t* BOAT_CALL boat_prelu_layer_get_slope(const boat_prelu_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_prelu_layer_get_grad_slope(const boat_prelu_layer_t* layer);

typedef struct boat_softmax_layer_t boat_softmax_layer_t;
BOAT_API boat_softmax_layer_t* BOAT_CALL boat_softmax_layer_create(int axis);
BOAT_API void BOAT_CALL boat_softmax_layer_free(boat_softmax_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_softmax_layer_forward(const boat_softmax_layer_t* layer,
                                                             const boat_tensor_t* input);
BOAT_API boat_tensor_t* BOAT_CALL boat_softmax_layer_backward(boat_softmax_layer_t* layer,
                                                              const boat_tensor_t* grad_output);
BOAT_API void BOAT_CALL boat_softmax_layer_update(boat_softmax_layer_t* layer, float learning_rate);
BOAT_API int BOAT_CALL boat_softmax_layer_get_axis(const boat_softmax_layer_t* layer);

typedef struct boat_flatten_layer_t boat_flatten_layer_t;

// Concatenation layer: joins N inputs along `dim`. `dim` is 0-based and may
// be negative (counted from the last dimension, MATLAB-style).
typedef struct boat_concat_layer_t boat_concat_layer_t;
BOAT_API boat_concat_layer_t* BOAT_CALL boat_concat_layer_create(int64_t dim);
BOAT_API void BOAT_CALL boat_concat_layer_free(boat_concat_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL
boat_concat_layer_forward_many(boat_concat_layer_t* layer, const boat_layer_input_t* inputs,
                               size_t n_inputs);

// Addition layer: element-wise sum of N inputs (broadcasting; a single input
// is passed through unchanged).
typedef struct boat_add_layer_t boat_add_layer_t;
BOAT_API boat_add_layer_t* BOAT_CALL boat_add_layer_create(void);
BOAT_API void BOAT_CALL boat_add_layer_free(boat_add_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL
boat_add_layer_forward_many(boat_add_layer_t* layer, const boat_layer_input_t* inputs,
                            size_t n_inputs);

// Recurrent layers
typedef struct boat_lstm_layer_t boat_lstm_layer_t;
typedef struct boat_gru_layer_t boat_gru_layer_t;

// LSTM layer functions
BOAT_API boat_lstm_layer_t* BOAT_CALL boat_lstm_layer_create(size_t input_size, size_t hidden_size,
                                                             size_t num_layers, bool bidirectional,
                                                             float dropout);
BOAT_API void BOAT_CALL boat_lstm_layer_free(boat_lstm_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_lstm_layer_forward(boat_lstm_layer_t* layer,
                                                          const boat_tensor_t* input);
BOAT_API boat_tensor_t* BOAT_CALL boat_lstm_layer_backward(boat_lstm_layer_t* layer,
                                                           const boat_tensor_t* grad_output);
BOAT_API void BOAT_CALL boat_lstm_layer_update(boat_lstm_layer_t* layer, float learning_rate);
BOAT_API boat_tensor_t* BOAT_CALL
boat_lstm_layer_get_grad_weight_ih(const boat_lstm_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL
boat_lstm_layer_get_grad_weight_hh(const boat_lstm_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_lstm_layer_get_grad_bias_ih(const boat_lstm_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_lstm_layer_get_grad_bias_hh(const boat_lstm_layer_t* layer);
BOAT_API void BOAT_CALL boat_lstm_layer_set_weight_ih(boat_lstm_layer_t* layer,
                                                      boat_tensor_t* weight);
BOAT_API void BOAT_CALL boat_lstm_layer_set_weight_hh(boat_lstm_layer_t* layer,
                                                      boat_tensor_t* weight);
BOAT_API void BOAT_CALL boat_lstm_layer_set_bias_ih(boat_lstm_layer_t* layer, boat_tensor_t* bias);
BOAT_API void BOAT_CALL boat_lstm_layer_set_bias_hh(boat_lstm_layer_t* layer, boat_tensor_t* bias);

// GRU layer functions
BOAT_API boat_gru_layer_t* BOAT_CALL boat_gru_layer_create(size_t input_size, size_t hidden_size,
                                                           size_t num_layers, bool bidirectional,
                                                           float dropout);
BOAT_API void BOAT_CALL boat_gru_layer_free(boat_gru_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_forward(boat_gru_layer_t* layer,
                                                         const boat_tensor_t* input);
BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_backward(boat_gru_layer_t* layer,
                                                          const boat_tensor_t* grad_output);
BOAT_API void BOAT_CALL boat_gru_layer_update(boat_gru_layer_t* layer, float learning_rate);
BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_get_grad_weight_ih(const boat_gru_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_get_grad_weight_hh(const boat_gru_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_get_grad_bias_ih(const boat_gru_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_gru_layer_get_grad_bias_hh(const boat_gru_layer_t* layer);
BOAT_API void BOAT_CALL boat_gru_layer_set_weight_ih(boat_gru_layer_t* layer,
                                                     boat_tensor_t* weight);
BOAT_API void BOAT_CALL boat_gru_layer_set_weight_hh(boat_gru_layer_t* layer,
                                                     boat_tensor_t* weight);
BOAT_API void BOAT_CALL boat_gru_layer_set_bias_ih(boat_gru_layer_t* layer, boat_tensor_t* bias);
BOAT_API void BOAT_CALL boat_gru_layer_set_bias_hh(boat_gru_layer_t* layer, boat_tensor_t* bias);
BOAT_API boat_flatten_layer_t* BOAT_CALL boat_flatten_layer_create();
BOAT_API void BOAT_CALL boat_flatten_layer_free(boat_flatten_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_flatten_layer_forward(boat_flatten_layer_t* layer,
                                                             const boat_tensor_t* input);
BOAT_API boat_tensor_t* BOAT_CALL boat_flatten_layer_backward(const boat_flatten_layer_t* layer,
                                                              const boat_tensor_t* grad_output);
BOAT_API void BOAT_CALL boat_flatten_layer_update(boat_flatten_layer_t* layer, float learning_rate);

#ifdef __cplusplus
}
#endif

#endif // BOAT_LAYERS_H