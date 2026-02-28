// autodiff.h - Automatic differentiation for deep learning framework
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#ifndef BOAT_AUTODIFF_H
#define BOAT_AUTODIFF_H

#include "tensor.h"
#include "graph.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
typedef struct boat_variable_t boat_variable_t;
typedef struct boat_autodiff_context_t boat_autodiff_context_t;
struct boat_conv_layer_t;  // Forward declaration for convolutional layer
struct boat_attention_t;   // Forward declaration for attention layer
struct boat_pool_layer_t;  // Forward declaration for pooling layer
struct boat_flatten_layer_t;  // Forward declaration for flatten layer
struct boat_dense_layer_t;  // Forward declaration for dense layer

// Variable creation and destruction
BOAT_API boat_variable_t* boat_variable_create(boat_tensor_t* tensor, bool requires_grad);
BOAT_API boat_variable_t* boat_variable_create_with_shape(const int64_t* shape, size_t ndim,
                                                 boat_dtype_t dtype, bool requires_grad);
BOAT_API void boat_variable_free(boat_variable_t* variable);

// Variable properties
BOAT_API boat_tensor_t* boat_variable_data(const boat_variable_t* variable);
BOAT_API boat_tensor_t* boat_variable_grad(const boat_variable_t* variable);
BOAT_API bool boat_variable_requires_grad(const boat_variable_t* variable);
BOAT_API void boat_variable_set_requires_grad(boat_variable_t* variable, bool requires_grad);

// Variable data reset/reuse (for performance optimization)
BOAT_API bool boat_variable_reset_data(boat_variable_t* variable, boat_tensor_t* new_tensor);

// Gradient operations
BOAT_API void boat_variable_zero_grad(boat_variable_t* variable);
BOAT_API void boat_variable_retain_grad(boat_variable_t* variable, bool retain);
BOAT_API void boat_variable_backward(boat_variable_t* variable, boat_tensor_t* grad_output);
BOAT_API void boat_variable_backward_full(boat_variable_t* variable);

// Arithmetic operations with gradient tracking
BOAT_API boat_variable_t* boat_var_add(boat_variable_t* a, boat_variable_t* b);
BOAT_API boat_variable_t* boat_var_sub(boat_variable_t* a, boat_variable_t* b);
BOAT_API boat_variable_t* boat_var_mul(boat_variable_t* a, boat_variable_t* b);
BOAT_API boat_variable_t* boat_var_div(boat_variable_t* a, boat_variable_t* b);
BOAT_API boat_variable_t* boat_var_matmul(boat_variable_t* a, boat_variable_t* b);
BOAT_API boat_variable_t* boat_var_dot(boat_variable_t* a, boat_variable_t* b);

// Convolution operation with gradient tracking
BOAT_API boat_variable_t* boat_var_conv(boat_variable_t* input, struct boat_conv_layer_t* layer);

// Pooling operation with gradient tracking
BOAT_API boat_variable_t* boat_var_pool(boat_variable_t* input, struct boat_pool_layer_t* layer);

// Flatten operation with gradient tracking
BOAT_API boat_variable_t* boat_var_flatten(boat_variable_t* input);

// Dense (fully connected) operation with gradient tracking
BOAT_API boat_variable_t* boat_var_dense(boat_variable_t* input, struct boat_dense_layer_t* layer);

// Attention operation with gradient tracking
BOAT_API boat_variable_t* boat_var_attention(boat_variable_t* query, boat_variable_t* key, boat_variable_t* value, struct boat_attention_t* attention, const boat_tensor_t* attention_mask);

// Activation functions with gradient tracking
BOAT_API boat_variable_t* boat_var_relu(boat_variable_t* a);
BOAT_API boat_variable_t* boat_var_sigmoid(boat_variable_t* a);
BOAT_API boat_variable_t* boat_var_tanh(boat_variable_t* a);
BOAT_API boat_variable_t* boat_var_softmax(boat_variable_t* a, int axis);
BOAT_API boat_variable_t* boat_var_log_softmax(boat_variable_t* a, int axis);

// Reduction operations with gradient tracking
BOAT_API boat_variable_t* boat_var_sum(boat_variable_t* a, int64_t* dims, size_t n_dims, bool keepdim);
BOAT_API boat_variable_t* boat_var_mean(boat_variable_t* a, int64_t* dims, size_t n_dims, bool keepdim);
BOAT_API boat_variable_t* boat_var_max(boat_variable_t* a, int64_t* dims, size_t n_dims, bool keepdim);
BOAT_API boat_variable_t* boat_var_min(boat_variable_t* a, int64_t* dims, size_t n_dims, bool keepdim);

// Context management (for controlling gradient computation)
BOAT_API boat_autodiff_context_t* boat_autodiff_context_create();
BOAT_API void boat_autodiff_context_free(boat_autodiff_context_t* context);
BOAT_API void boat_autodiff_context_enable_grad(boat_autodiff_context_t* context);
BOAT_API void boat_autodiff_context_disable_grad(boat_autodiff_context_t* context);
BOAT_API bool boat_autodiff_context_grad_enabled(boat_autodiff_context_t* context);
BOAT_API void boat_autodiff_context_set_graph(boat_autodiff_context_t* context, boat_graph_t* graph);
BOAT_API boat_graph_t* boat_autodiff_context_get_graph(const boat_autodiff_context_t* context);
BOAT_API void boat_autodiff_set_current_context(boat_autodiff_context_t* context);
BOAT_API boat_autodiff_context_t* boat_autodiff_get_current_context();

// Gradient checkpointing
BOAT_API void boat_autodiff_set_grad_checkpointing(bool enabled);
BOAT_API void boat_autodiff_clear_computation_graph();

// Utility functions
BOAT_API void boat_autodiff_print_graph(const boat_variable_t* variable);
BOAT_API char* boat_autodiff_graph_to_dot(const boat_variable_t* variable);

#ifdef __cplusplus
}
#endif

#endif // BOAT_AUTODIFF_H