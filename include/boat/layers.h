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

// Forward declarations for all layer types
typedef struct boat_dense_layer_t boat_dense_layer_t;
typedef struct boat_conv_layer_t boat_conv_layer_t;
typedef struct boat_pool_layer_t boat_pool_layer_t;
typedef struct boat_norm_layer_t boat_norm_layer_t;
typedef struct boat_attention_layer_t boat_attention_layer_t;

// Dense layer functions
BOAT_API boat_dense_layer_t* BOAT_CALL boat_dense_layer_create(size_t input_features, size_t output_features, bool use_bias);
BOAT_API void BOAT_CALL boat_dense_layer_free(boat_dense_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_dense_layer_forward(boat_dense_layer_t* layer, const boat_tensor_t* input);
BOAT_API boat_tensor_t* BOAT_CALL boat_dense_layer_backward(boat_dense_layer_t* layer, const boat_tensor_t* grad_output);
BOAT_API void BOAT_CALL boat_dense_layer_update(boat_dense_layer_t* layer, float learning_rate);

// Parameter access for model loading
BOAT_API void BOAT_CALL boat_dense_layer_set_weight(boat_dense_layer_t* layer, boat_tensor_t* weight);
BOAT_API void BOAT_CALL boat_dense_layer_set_bias(boat_dense_layer_t* layer, boat_tensor_t* bias);

// Convolutional layer functions
BOAT_API boat_conv_layer_t* BOAT_CALL boat_conv_layer_create(size_t in_channels, size_t out_channels,
                                           size_t kernel_size, size_t stride, size_t padding);
BOAT_API void BOAT_CALL boat_conv_layer_free(boat_conv_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_forward(boat_conv_layer_t* layer, const boat_tensor_t* input);
BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_backward(boat_conv_layer_t* layer, const boat_tensor_t* grad_output);
BOAT_API void BOAT_CALL boat_conv_layer_update(boat_conv_layer_t* layer, float learning_rate);

// Parameter access for model loading
BOAT_API void BOAT_CALL boat_conv_layer_set_weight(boat_conv_layer_t* layer, boat_tensor_t* weight);
BOAT_API void BOAT_CALL boat_conv_layer_set_bias(boat_conv_layer_t* layer, boat_tensor_t* bias);
BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_get_weight(boat_conv_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_get_bias(boat_conv_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_get_grad_weight(boat_conv_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_get_grad_bias(boat_conv_layer_t* layer);

// Batch normalization layer functions (BatchNorm2d)
typedef struct boat_batchnorm2d_layer_t boat_batchnorm2d_layer_t;
boat_batchnorm2d_layer_t* boat_batchnorm2d_layer_create(size_t num_features, float eps, float momentum, bool affine);
void boat_batchnorm2d_layer_free(boat_batchnorm2d_layer_t* layer);
boat_tensor_t* boat_batchnorm2d_layer_forward(boat_batchnorm2d_layer_t* layer, const boat_tensor_t* input);
boat_tensor_t* boat_batchnorm2d_layer_backward(boat_batchnorm2d_layer_t* layer, const boat_tensor_t* grad_output);
void boat_batchnorm2d_layer_update(boat_batchnorm2d_layer_t* layer, float learning_rate);

// Parameter access for BatchNorm2d
void boat_batchnorm2d_layer_set_weight(boat_batchnorm2d_layer_t* layer, boat_tensor_t* weight);
void boat_batchnorm2d_layer_set_bias(boat_batchnorm2d_layer_t* layer, boat_tensor_t* bias);
void boat_batchnorm2d_layer_set_running_mean(boat_batchnorm2d_layer_t* layer, boat_tensor_t* running_mean);
void boat_batchnorm2d_layer_set_running_var(boat_batchnorm2d_layer_t* layer, boat_tensor_t* running_var);

// Pooling layer functions (MaxPool2d)
boat_pool_layer_t* boat_pool_layer_create(size_t pool_size, size_t stride, size_t padding);
void boat_pool_layer_free(boat_pool_layer_t* layer);
boat_tensor_t* boat_pool_layer_forward(boat_pool_layer_t* layer, const boat_tensor_t* input);
boat_tensor_t* boat_pool_layer_backward(boat_pool_layer_t* layer, const boat_tensor_t* grad_output);
void boat_pool_layer_update(boat_pool_layer_t* layer, float learning_rate);

// Normalization layer functions (simplified interface)
BOAT_API boat_norm_layer_t* BOAT_CALL boat_norm_layer_create(size_t normalized_shape, float eps, bool elementwise_affine);
BOAT_API void BOAT_CALL boat_norm_layer_free(boat_norm_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_norm_layer_forward(boat_norm_layer_t* layer, const boat_tensor_t* input);
BOAT_API boat_tensor_t* BOAT_CALL boat_norm_layer_backward(boat_norm_layer_t* layer, const boat_tensor_t* grad_output);
BOAT_API void BOAT_CALL boat_norm_layer_update(boat_norm_layer_t* layer, float learning_rate);

// Attention layer functions (simplified interface)
BOAT_API boat_attention_layer_t* BOAT_CALL boat_attention_layer_create(size_t hidden_size, size_t num_heads,
                                                              float dropout_prob, bool causal_mask);
BOAT_API void BOAT_CALL boat_attention_layer_free(boat_attention_layer_t* layer);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_layer_forward(boat_attention_layer_t* layer,
                                                      const boat_tensor_t* query,
                                                      const boat_tensor_t* key,
                                                      const boat_tensor_t* value,
                                                      const boat_tensor_t* attention_mask);
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_layer_backward(boat_attention_layer_t* layer,
                                                       const boat_tensor_t* grad_output);
BOAT_API void BOAT_CALL boat_attention_layer_update(boat_attention_layer_t* layer, float learning_rate);

// Activation layers
typedef struct boat_relu_layer_t boat_relu_layer_t;
boat_relu_layer_t* boat_relu_layer_create();
void boat_relu_layer_free(boat_relu_layer_t* layer);
boat_tensor_t* boat_relu_layer_forward(boat_relu_layer_t* layer, const boat_tensor_t* input);
boat_tensor_t* boat_relu_layer_backward(boat_relu_layer_t* layer, const boat_tensor_t* grad_output);
void boat_relu_layer_update(boat_relu_layer_t* layer, float learning_rate);

typedef struct boat_softmax_layer_t boat_softmax_layer_t;
boat_softmax_layer_t* boat_softmax_layer_create(int axis);
void boat_softmax_layer_free(boat_softmax_layer_t* layer);
boat_tensor_t* boat_softmax_layer_forward(boat_softmax_layer_t* layer, const boat_tensor_t* input);
boat_tensor_t* boat_softmax_layer_backward(boat_softmax_layer_t* layer, const boat_tensor_t* grad_output);
void boat_softmax_layer_update(boat_softmax_layer_t* layer, float learning_rate);

typedef struct boat_flatten_layer_t boat_flatten_layer_t;

// Recurrent layers
typedef struct boat_lstm_layer_t boat_lstm_layer_t;
typedef struct boat_gru_layer_t boat_gru_layer_t;

// LSTM layer functions
boat_lstm_layer_t* boat_lstm_layer_create(size_t input_size, size_t hidden_size, size_t num_layers, bool bidirectional, float dropout);
void boat_lstm_layer_free(boat_lstm_layer_t* layer);
boat_tensor_t* boat_lstm_layer_forward(boat_lstm_layer_t* layer, const boat_tensor_t* input);
boat_tensor_t* boat_lstm_layer_backward(boat_lstm_layer_t* layer, const boat_tensor_t* grad_output);
void boat_lstm_layer_update(boat_lstm_layer_t* layer, float learning_rate);

// GRU layer functions
boat_gru_layer_t* boat_gru_layer_create(size_t input_size, size_t hidden_size, size_t num_layers, bool bidirectional, float dropout);
void boat_gru_layer_free(boat_gru_layer_t* layer);
boat_tensor_t* boat_gru_layer_forward(boat_gru_layer_t* layer, const boat_tensor_t* input);
boat_tensor_t* boat_gru_layer_backward(boat_gru_layer_t* layer, const boat_tensor_t* grad_output);
void boat_gru_layer_update(boat_gru_layer_t* layer, float learning_rate);
boat_flatten_layer_t* boat_flatten_layer_create();
void boat_flatten_layer_free(boat_flatten_layer_t* layer);
boat_tensor_t* boat_flatten_layer_forward(boat_flatten_layer_t* layer, const boat_tensor_t* input);
boat_tensor_t* boat_flatten_layer_backward(boat_flatten_layer_t* layer, const boat_tensor_t* grad_output);
void boat_flatten_layer_update(boat_flatten_layer_t* layer, float learning_rate);

#ifdef __cplusplus
}
#endif

#endif // BOAT_LAYERS_H