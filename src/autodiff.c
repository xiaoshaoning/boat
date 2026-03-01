// autodiff.c - Automatic differentiation implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#define BOAT_BUILDING_DLL
#include <boat/autodiff.h>
#include <boat/graph.h>
#include <boat/memory.h>
#include <boat/ops.h>
#include <boat/layers.h>
#include <boat/layers/attention.h>
#include <boat/tensor.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#ifdef _WIN32
#include <windows.h>
#endif

// Internal variable structure
struct boat_variable_t {
    boat_tensor_t* data;           // Tensor data
    boat_tensor_t* grad;           // Gradient tensor (nullable)
    bool requires_grad;            // Whether gradient is required
    boat_node_t* node;             // Corresponding graph node
    boat_graph_t* graph;           // Computational graph containing variable
    boat_node_t* producer_node;    // Operation node that produced this variable (nullable)
};

// Operation types for automatic differentiation
typedef enum {
    BOAT_OP_ADD,
    BOAT_OP_SUB,
    BOAT_OP_MUL,
    BOAT_OP_DIV,
    BOAT_OP_RELU,
    BOAT_OP_SIGMOID,
    BOAT_OP_TANH,
    BOAT_OP_MATMUL,
    BOAT_OP_DOT,
    BOAT_OP_SUM,
    BOAT_OP_MEAN,
    BOAT_OP_SOFTMAX,
    BOAT_OP_LOG_SOFTMAX,
    BOAT_OP_CONV,
    BOAT_OP_POOL,
    BOAT_OP_FLATTEN,
    BOAT_OP_DENSE,
    BOAT_OP_ATTENTION
} boat_op_type_t;

// Operation node data (stored in graph node)
typedef struct {
    boat_op_type_t op_type;        // Operation type
    boat_variable_t** inputs;      // Input variables
    size_t num_inputs;             // Number of inputs
    boat_variable_t* output;       // Output variable
    void* extra_data;              // Extra data for specific operations (e.g., axis)
} boat_op_node_data_t;

// Internal context structure
struct boat_autodiff_context_t {
    bool grad_enabled;
    boat_graph_t* graph;  // Computational graph associated with this context
    // TODO: add more context fields
};



// Thread-local current autodiff context
#ifdef _WIN32
static __declspec(thread) boat_autodiff_context_t* current_context = NULL;
#else
static thread_local boat_autodiff_context_t* current_context = NULL;
#endif

// Debug counter for tracking function execution
static volatile int debug_counter = 0;


// Forward declarations for helper functions
static boat_tensor_t* compute_forward_add(const boat_tensor_t* a, const boat_tensor_t* b);
static boat_tensor_t* compute_forward_sub(const boat_tensor_t* a, const boat_tensor_t* b);
static boat_tensor_t* compute_forward_mul(const boat_tensor_t* a, const boat_tensor_t* b);
static boat_tensor_t* compute_forward_div(const boat_tensor_t* a, const boat_tensor_t* b);
static boat_tensor_t* compute_forward_dot(const boat_tensor_t* a, const boat_tensor_t* b);
static boat_tensor_t* compute_forward_relu(const boat_tensor_t* a);
static boat_tensor_t* compute_forward_sigmoid(const boat_tensor_t* a);
static boat_tensor_t* compute_forward_tanh(const boat_tensor_t* a);
static boat_tensor_t* compute_forward_matmul(const boat_tensor_t* a, const boat_tensor_t* b);
static boat_tensor_t* compute_forward_sum(boat_tensor_t* a, int64_t* dims, size_t n_dims, bool keepdim);
static boat_tensor_t* compute_forward_sum_single(boat_tensor_t* a);
static boat_tensor_t* compute_forward_mean(boat_tensor_t* a, int64_t* dims, size_t n_dims, bool keepdim);
static boat_tensor_t* compute_forward_softmax(const boat_tensor_t* a);
static boat_tensor_t* compute_forward_log_softmax(const boat_tensor_t* a);
static boat_tensor_t* compute_forward_conv(boat_tensor_t* input, void* layer_ptr);
static void compute_backward_conv(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static boat_tensor_t* compute_forward_pool(const boat_tensor_t* input, void* layer_ptr);
static void compute_backward_pool(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static boat_tensor_t* compute_forward_flatten(const boat_tensor_t* input);
static void compute_backward_flatten(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static boat_tensor_t* compute_forward_dense(const boat_tensor_t* input, const void* layer_ptr);
static void compute_backward_dense(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static boat_variable_t* create_attention_operation(const boat_variable_t* query, const boat_variable_t* key, const boat_variable_t* value, const struct boat_attention_t* attention, const boat_tensor_t* attention_mask);
static void compute_backward_attention(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_add(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_sub(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_mul(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_div(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_dot(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_relu(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_sigmoid(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_tanh(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_matmul(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_sum(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_mean(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_softmax(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_log_softmax(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static boat_op_node_data_t* create_op_node_data(boat_op_type_t op_type,
                                                boat_variable_t** inputs,
                                                size_t num_inputs,
                                                const boat_variable_t* output);
static void free_op_node_data(const void* data);
static void free_variable_data(const void* data);
static boat_variable_t* create_operation(boat_op_type_t op_type,
                                         boat_variable_t** inputs,
                                         size_t num_inputs,
                                         boat_tensor_t* (*forward_fn)(const boat_tensor_t*, const boat_tensor_t*),
                                         boat_tensor_t* (*forward_single_fn)(const boat_tensor_t*));
static boat_variable_t* create_conv_operation(const boat_variable_t* input, const struct boat_conv_layer_t* layer);
static boat_variable_t* create_pool_operation(const boat_variable_t* input, const struct boat_pool_layer_t* layer);
static boat_variable_t* create_dense_operation(const boat_variable_t* input, const struct boat_dense_layer_t* layer);

// Variable creation and destruction
BOAT_API boat_variable_t* boat_variable_create(boat_tensor_t* tensor, bool requires_grad) {

#ifdef _WIN32
    // Debug buffer removed as unused
#endif
    if (!tensor) {
        return NULL;
    }

    boat_variable_t* var = boat_malloc(sizeof(boat_variable_t), BOAT_DEVICE_CPU);
    if (!var) {
        return NULL;
    }

    var->data = tensor;
    boat_tensor_ref(tensor);  // Take ownership of the tensor reference
    var->grad = NULL;
    var->requires_grad = requires_grad;
    var->node = NULL;
    var->graph = NULL;
    var->producer_node = NULL;


    // Create graph node if gradient is required

#ifdef _WIN32
#endif

    // Use current context graph for explicit graph passing architecture
    const boat_autodiff_context_t* ctx = boat_autodiff_get_current_context();
    boat_graph_t* graph = NULL;

    if (ctx) {
        graph = boat_autodiff_context_get_graph(ctx);
    } else {
        ctx = boat_autodiff_context_create();
        if (!ctx) {
            boat_free(var);
            return NULL;
        }
        boat_autodiff_set_current_context(ctx);
    }

    // If context has no graph, create one
    if (!graph) {
        graph = boat_graph_create_with_device(boat_tensor_device(tensor));
        if (!graph) {
            if (ctx) boat_autodiff_context_free(ctx);
            boat_free(var);
            return NULL;
        }
        boat_autodiff_context_set_graph(ctx, graph);
    }

    // Always associate variable with graph (even if requires_grad is false)
    var->graph = graph;

    // Create a variable node in the graph if gradient is required
    if (requires_grad) {
        var->node = boat_graph_add_node(var->graph, var, BOAT_NODE_TYPE_VARIABLE, free_variable_data);
        if (!var->node) {
            // Don't free graph, it's owned by context
            boat_free(var);
            return NULL;
        }
    }

    return var;
}

boat_variable_t* boat_variable_create_with_shape(const int64_t* shape, size_t ndim,
                                                 boat_dtype_t dtype, bool requires_grad) {
    boat_tensor_t* tensor = boat_tensor_create(shape, ndim, dtype, BOAT_DEVICE_CPU);
    if (!tensor) {
        return NULL;
    }

    return boat_variable_create(tensor, requires_grad);
}

BOAT_API void boat_variable_free(const boat_variable_t* variable) {
    if (!variable) return;

    // Free gradient tensor if exists
    if (variable->grad) {
        boat_tensor_unref(variable->grad);
    }

    // Free data tensor (variable owns it)
    if (variable->data) {
        boat_tensor_unref(variable->data);
    }


    boat_free((void*)variable);
}

// Variable properties
BOAT_API boat_tensor_t* boat_variable_data(const boat_variable_t* variable) {
    return variable ? variable->data : NULL;
}

BOAT_API boat_tensor_t* boat_variable_grad(const boat_variable_t* variable) {
    return variable ? variable->grad : NULL;
}

bool boat_variable_requires_grad(const boat_variable_t* variable) {
    return variable ? variable->requires_grad : false;
}

void boat_variable_set_requires_grad(boat_variable_t* variable, bool requires_grad) {
    if (!variable) return;
    variable->requires_grad = requires_grad;
}

// Variable data reset/reuse
BOAT_API bool boat_variable_reset_data(boat_variable_t* variable, boat_tensor_t* new_tensor) {
    if (!variable || !new_tensor) {
        return false;
    }

    // Check if variable is part of a computation graph
    // If it has a node, we cannot safely reset data as it would break the graph
    if (variable->node != NULL) {
        fprintf(stderr, "Warning: Cannot reset data for variable with computation graph node\n");
        return false;
    }

    // Free old data tensor
    if (variable->data) {
        boat_tensor_unref(variable->data);
    }

    // Set new data tensor and increment its reference count
    variable->data = new_tensor;
    boat_tensor_ref(new_tensor);

    // Also reset gradient if it exists
    if (variable->grad) {
        boat_tensor_unref(variable->grad);
        variable->grad = NULL;
    }

    return true;
}

// Gradient operations
void boat_variable_zero_grad(boat_variable_t* variable) {
    if (!variable) return;

    if (variable->grad) {
        boat_tensor_unref(variable->grad);
        variable->grad = NULL;
    }
}

void boat_variable_retain_grad(const boat_variable_t* variable, bool retain) {
    if (!variable) return;
    // TODO: implement gradient retention
}

void boat_variable_backward(boat_variable_t* variable, boat_tensor_t* grad_output) {
    setbuf(stdout, NULL);
    setbuf(stderr, NULL);
    if (variable) {
    }

    if (!variable || !variable->requires_grad) {
        return;
    }

    // If grad_output is NULL, assume scalar loss with gradient 1
    // Create a tensor of ones with same shape as variable's data
    boat_tensor_t* local_grad = grad_output;
    bool local_grad_allocated = false;
    if (!local_grad) {
        local_grad = boat_tensor_create_like(variable->data);
        if (!local_grad) {
            return;
        }
        // Fill with 1.0
        size_t nelements = boat_tensor_nelements(local_grad);
        void* data = boat_tensor_data(local_grad);
        boat_dtype_t dtype = boat_tensor_dtype(local_grad);
        switch (dtype) {
            case BOAT_DTYPE_FLOAT32: {
                float* ptr = (float*)data;
                for (size_t i = 0; i < nelements; i++) ptr[i] = 1.0f;
                break;
            }
            case BOAT_DTYPE_FLOAT64: {
                double* ptr = (double*)data;
                for (size_t i = 0; i < nelements; i++) ptr[i] = 1.0;
                break;
            }
            default:
                // Unsupported type for gradient
                boat_tensor_unref(local_grad);
                return;
        }
        local_grad_allocated = true;
    } else {
    }

    // Get producer operation node
    const boat_node_t* producer = variable->producer_node;
    if (producer) {
    }
    if (!producer) {
        // Variable is a leaf (no producer), gradient is stored directly
        if (!variable->grad) {
            variable->grad = boat_tensor_create_like(variable->data);
        } else {
        }
        if (variable->grad) {
            // Accumulate gradient
            boat_add_(variable->grad, local_grad);
        } else {
        }
        if (local_grad_allocated) {
            boat_tensor_unref(local_grad);
        }
        return;
    }

    // Get operation data
    void* node_data = boat_node_data(producer);
    if (!node_data) {
        if (local_grad_allocated) boat_tensor_unref(local_grad);
        return;
    }

    boat_op_node_data_t* op_data = (boat_op_node_data_t*)node_data;

    // Dispatch to appropriate backward function
    switch (op_data->op_type) {
        case BOAT_OP_ADD:
            compute_backward_add(op_data, local_grad);
            break;
        case BOAT_OP_SUB:
            compute_backward_sub(op_data, local_grad);
            break;
        case BOAT_OP_MUL:
            compute_backward_mul(op_data, local_grad);
            break;
        case BOAT_OP_DIV:
            compute_backward_div(op_data, local_grad);
            break;
        case BOAT_OP_RELU:
            compute_backward_relu(op_data, local_grad);
            break;
        case BOAT_OP_SIGMOID:
            compute_backward_sigmoid(op_data, local_grad);
            break;
        case BOAT_OP_TANH:
            compute_backward_tanh(op_data, local_grad);
            break;
        case BOAT_OP_MATMUL:
            compute_backward_matmul(op_data, local_grad);
            break;
        case BOAT_OP_DOT:
            compute_backward_dot(op_data, local_grad);
            break;
        case BOAT_OP_SUM:
            compute_backward_sum(op_data, local_grad);
            break;
        case BOAT_OP_MEAN:
            compute_backward_mean(op_data, local_grad);
            break;
        case BOAT_OP_SOFTMAX:
            compute_backward_softmax(op_data, local_grad);
            break;
        case BOAT_OP_LOG_SOFTMAX:
            compute_backward_log_softmax(op_data, local_grad);
            break;
        case BOAT_OP_CONV:
            compute_backward_conv(op_data, local_grad);
            break;
        case BOAT_OP_DENSE:
            compute_backward_dense(op_data, local_grad);
            break;
        case BOAT_OP_POOL:
            compute_backward_pool(op_data, local_grad);
            break;
        case BOAT_OP_FLATTEN:
            compute_backward_flatten(op_data, local_grad);
            break;
        case BOAT_OP_ATTENTION:
            compute_backward_attention(op_data, local_grad);
            break;
    }

    // Recursively backward to input variables (chain rule)
    for (size_t i = 0; i < op_data->num_inputs; i++) {
        boat_variable_t* input_var = op_data->inputs[i];
        if (input_var && input_var->requires_grad) {
            // Get gradient for this input (should have been computed by compute_backward_*)
            boat_tensor_t* input_grad = input_var->grad;
            if (input_grad) {
                // Only propagate to non-leaf variables (those with producers)
                // Leaf variables already have gradients accumulated by compute_backward_*
                if (input_var->producer_node) {
                    // Increase refcount since we're passing it to backward
                    boat_tensor_ref(input_grad);
                    boat_variable_backward(input_var, input_grad);
                    // backward doesn't consume the tensor, so unref
                    boat_tensor_unref(input_grad);
                }
                // For leaf variables, gradient is already stored in input_var->grad
                // No need to call backward further
            }
        }
    }

    if (local_grad_allocated) {
        boat_tensor_unref(local_grad);
    }
}

BOAT_API void boat_variable_backward_full(const boat_variable_t* variable) {
    if (!variable || !variable->requires_grad) return;

    // For now, just call backward with NULL gradient (scalar loss)
    // In the future, implement full graph traversal
    boat_variable_backward(variable, NULL);
}

// Arithmetic operations with gradient tracking
BOAT_API boat_variable_t* boat_var_add(const boat_variable_t* a, const boat_variable_t* b) {
    if (!a || !b) return NULL;

    boat_variable_t* inputs[] = {a, b};
    return create_operation(BOAT_OP_ADD, inputs, 2, compute_forward_add, NULL);
}

BOAT_API boat_variable_t* boat_var_sub(const boat_variable_t* a, const boat_variable_t* b) {
    if (!a || !b) return NULL;

    boat_variable_t* inputs[] = {a, b};
    return create_operation(BOAT_OP_SUB, inputs, 2, compute_forward_sub, NULL);
}

BOAT_API boat_variable_t* boat_var_mul(const boat_variable_t* a, const boat_variable_t* b) {
    if (!a || !b) return NULL;

    boat_variable_t* inputs[] = {a, b};
    return create_operation(BOAT_OP_MUL, inputs, 2, compute_forward_mul, NULL);
}

BOAT_API boat_variable_t* boat_var_div(const boat_variable_t* a, const boat_variable_t* b) {
    if (!a || !b) return NULL;

    boat_variable_t* inputs[] = {a, b};
    return create_operation(BOAT_OP_DIV, inputs, 2, compute_forward_div, NULL);
}

BOAT_API boat_variable_t* boat_var_matmul(const boat_variable_t* a, const boat_variable_t* b) {
    if (!a || !b) return NULL;

    boat_variable_t* inputs[] = {a, b};
    return create_operation(BOAT_OP_MATMUL, inputs, 2, compute_forward_matmul, NULL);
}

BOAT_API boat_variable_t* boat_var_dot(const boat_variable_t* a, const boat_variable_t* b) {
    if (!a || !b) return NULL;

    boat_variable_t* inputs[] = {a, b};
    return create_operation(BOAT_OP_DOT, inputs, 2, compute_forward_dot, NULL);
}

// Activation functions with gradient tracking
BOAT_API boat_variable_t* boat_var_relu(const boat_variable_t* a) {
    if (!a) return NULL;

    boat_variable_t* inputs[] = {a};
    return create_operation(BOAT_OP_RELU, inputs, 1, NULL, compute_forward_relu);
}

boat_variable_t* boat_var_sigmoid(const boat_variable_t* a) {
    if (!a) return NULL;

    boat_variable_t* inputs[] = {a};
    return create_operation(BOAT_OP_SIGMOID, inputs, 1, NULL, compute_forward_sigmoid);
}

boat_variable_t* boat_var_tanh(const boat_variable_t* a) {
    if (!a) return NULL;

    boat_variable_t* inputs[] = {a};
    return create_operation(BOAT_OP_TANH, inputs, 1, NULL, compute_forward_tanh);
}

boat_variable_t* boat_var_softmax(const boat_variable_t* a, int axis) {
    if (!a) return NULL;
    (void)axis; // TODO: support axis parameter
    boat_variable_t* inputs[] = {a};
    return create_operation(BOAT_OP_SOFTMAX, inputs, 1, NULL, compute_forward_softmax);
}

BOAT_API boat_variable_t* boat_var_flatten(const boat_variable_t* a) {
    if (!a) return NULL;
    boat_variable_t* inputs[] = {a};
    return create_operation(BOAT_OP_FLATTEN, inputs, 1, NULL, compute_forward_flatten);
}

BOAT_API boat_variable_t* boat_var_log_softmax(const boat_variable_t* a, int axis) {
    if (!a) return NULL;
    (void)axis; // TODO: support axis parameter
    boat_variable_t* inputs[] = {a};
    return create_operation(BOAT_OP_LOG_SOFTMAX, inputs, 1, NULL, compute_forward_log_softmax);
}

// Convolution operation with gradient tracking
BOAT_API boat_variable_t* boat_var_conv(const boat_variable_t* input, const struct boat_conv_layer_t* layer) {
    if (!input || !layer) return NULL;
    return create_conv_operation(input, layer);
}

// Pooling operation with gradient tracking
BOAT_API boat_variable_t* boat_var_pool(const boat_variable_t* input, const struct boat_pool_layer_t* layer) {
    if (!input || !layer) return NULL;
    return create_pool_operation(input, layer);
}

// Dense operation with gradient tracking
BOAT_API boat_variable_t* boat_var_dense(const boat_variable_t* input, const struct boat_dense_layer_t* layer) {
    if (!input || !layer) return NULL;
    return create_dense_operation(input, layer);
}

// Attention operation with gradient tracking
boat_variable_t* boat_var_attention(const boat_variable_t* query, const boat_variable_t* key, const boat_variable_t* value, const struct boat_attention_t* attention, const boat_tensor_t* attention_mask) {
    if (!query || !key || !value || !attention) return NULL;
    return create_attention_operation(query, key, value, attention, attention_mask);
}

// Reduction operations with gradient tracking
BOAT_API boat_variable_t* boat_var_sum(const boat_variable_t* a, const int64_t* dims, size_t n_dims, bool keepdim) {
    if (!a) return NULL;

    // For now, only support full reduction (dims == NULL, n_dims == 0)
    // TODO: Support reduction along specific dimensions
    if (dims != NULL || n_dims != 0 || keepdim) {
        // Not yet implemented
        fprintf(stderr, "ERROR: boat_var_sum only supports full reduction (dims=NULL, n_dims=0, keepdim=false) for now\n");
        return NULL;
    }

    boat_variable_t* inputs[] = {a};
    return create_operation(BOAT_OP_SUM, inputs, 1, NULL, compute_forward_sum_single);
}

boat_variable_t* boat_var_mean(const boat_variable_t* a, int64_t* dims, size_t n_dims, bool keepdim) {
    (void)a; (void)dims; (void)n_dims; (void)keepdim;
    return NULL;
}

boat_variable_t* boat_var_max(const boat_variable_t* a, int64_t* dims, size_t n_dims, bool keepdim) {
    (void)a; (void)dims; (void)n_dims; (void)keepdim;
    return NULL;
}

boat_variable_t* boat_var_min(const boat_variable_t* a, int64_t* dims, size_t n_dims, bool keepdim) {
    (void)a; (void)dims; (void)n_dims; (void)keepdim;
    return NULL;
}

// Context management
boat_autodiff_context_t* boat_autodiff_context_create() {
    boat_autodiff_context_t* ctx = boat_malloc(sizeof(boat_autodiff_context_t), BOAT_DEVICE_CPU);
    if (!ctx) return NULL;

    ctx->grad_enabled = true;
    ctx->graph = NULL;
    return ctx;
}

void boat_autodiff_context_free(const boat_autodiff_context_t* context) {
    if (!context) return;
    boat_free(context);
}

void boat_autodiff_context_enable_grad(const boat_autodiff_context_t* context) {
    if (!context) return;
    context->grad_enabled = true;
}

void boat_autodiff_context_disable_grad(const boat_autodiff_context_t* context) {
    if (!context) return;
    context->grad_enabled = false;
}

bool boat_autodiff_context_grad_enabled(const boat_autodiff_context_t* context) {
    return context ? context->grad_enabled : false;
}

void boat_autodiff_context_set_graph(const boat_autodiff_context_t* context, const boat_graph_t* graph) {
    if (!context) return;
    context->graph = graph;
}

boat_graph_t* boat_autodiff_context_get_graph(const boat_autodiff_context_t* context) {
    return context ? context->graph : NULL;
}

void boat_autodiff_set_current_context(const boat_autodiff_context_t* context) {
    current_context = context;
}

boat_autodiff_context_t* boat_autodiff_get_current_context() {
    return current_context;
}

// Gradient checkpointing
void boat_autodiff_set_grad_checkpointing(bool enabled) {
    (void)enabled;
    // TODO: implement gradient checkpointing
}

void boat_autodiff_clear_computation_graph() {
    // Get current autodiff context
    const boat_autodiff_context_t* ctx = boat_autodiff_get_current_context();
    if (!ctx) return;

    // Get graph from context
    const boat_graph_t* graph = boat_autodiff_context_get_graph(ctx);
    if (!graph) return;

    // Get node count
    size_t node_count = boat_graph_node_count(graph);
    if (node_count == 0) return;

    // Collect nodes to remove
    boat_node_t** nodes_to_remove = boat_malloc(sizeof(boat_node_t*) * node_count, BOAT_DEVICE_CPU);
    if (!nodes_to_remove) return;

    size_t remove_count = 0;

    // Iterate through all nodes
    for (size_t i = 0; i < node_count; i++) {
        boat_node_t* node = boat_graph_get_node_at_index(graph, i);
        if (!node) continue;

        boat_node_type_t node_type = boat_node_type(node);

        // Remove all operation nodes
        if (node_type == BOAT_NODE_TYPE_OPERATION) {
            nodes_to_remove[remove_count++] = node;
        }
        // Also remove variable nodes that require gradient (temporary variables)
        else if (node_type == BOAT_NODE_TYPE_VARIABLE) {
            const void* node_data = boat_node_data(node);
            if (node_data) {
                const boat_variable_t* var = (const boat_variable_t*)node_data;
                // Remove variable nodes for temporary variables:
                // Variables with producer nodes (non-leaf, intermediate results)
                // Leaf variables (no producer) should be kept for next batch
                if (var->producer_node) {
                    nodes_to_remove[remove_count++] = node;
                }
            }
        }
    }

    // Remove nodes and update variable references
    for (size_t i = 0; i < remove_count; i++) {
        const boat_node_t* node = nodes_to_remove[i];
        boat_node_type_t node_type = boat_node_type(node);

        // For variable nodes, update the variable structure before removing the node
        if (node_type == BOAT_NODE_TYPE_VARIABLE) {
            void* node_data = boat_node_data(node);
            if (node_data) {
                boat_variable_t* var = (boat_variable_t*)node_data;
                // Clear the node reference in the variable structure
                var->node = NULL;
            }
        }

        boat_graph_remove_node(graph, node);
    }

    boat_free(nodes_to_remove);

    // Clear gradients as well
    boat_computation_graph_clear_gradients(graph);
}

// Utility functions
void boat_autodiff_print_graph(const boat_variable_t* variable) {
    if (!variable) return;
    // TODO: implement graph printing
}

char* boat_autodiff_graph_to_dot(const boat_variable_t* variable) {
    if (!variable) return NULL;
    // TODO: implement DOT generation
    return NULL;
}

// Helper function implementations
static boat_op_node_data_t* create_op_node_data(boat_op_type_t op_type,
                                                boat_variable_t** inputs,
                                                size_t num_inputs,
                                                const boat_variable_t* output) {
    boat_op_node_data_t* op_data = boat_malloc(sizeof(boat_op_node_data_t), BOAT_DEVICE_CPU);
    if (!op_data) return NULL;

    op_data->op_type = op_type;
    op_data->num_inputs = num_inputs;
    op_data->output = output;

    // Copy input pointers
    if (num_inputs > 0) {
        op_data->inputs = boat_malloc(sizeof(boat_variable_t*) * num_inputs, BOAT_DEVICE_CPU);
        if (!op_data->inputs) {
            boat_free(op_data);
            return NULL;
        }
        for (size_t i = 0; i < num_inputs; i++) {
            op_data->inputs[i] = inputs[i];
        }
    } else {
        op_data->inputs = NULL;
    }

    op_data->extra_data = NULL;
    return op_data;
}

static void free_op_node_data(const void* data) {
    if (!data) return;

    boat_op_node_data_t* op_data = (boat_op_node_data_t*)data;
    if (op_data->inputs) {
        boat_free(op_data->inputs);
    }
    if (op_data->extra_data) {
        boat_free(op_data->extra_data);
    }
    boat_free(op_data);
}

static void free_variable_data(const void* data) {
    if (!data) return;
    const boat_variable_t* var = (const boat_variable_t*)data;
    boat_variable_free(var);
}

// Forward computation functions
static boat_tensor_t* compute_forward_add(const boat_tensor_t* a, const boat_tensor_t* b) {
    return boat_add(a, b);
}

static boat_tensor_t* compute_forward_sub(const boat_tensor_t* a, const boat_tensor_t* b) {
    return boat_sub(a, b);
}

static boat_tensor_t* compute_forward_mul(const boat_tensor_t* a, const boat_tensor_t* b) {
    return boat_mul(a, b);
}

static boat_tensor_t* compute_forward_div(const boat_tensor_t* a, const boat_tensor_t* b) {
    return boat_div(a, b);
}

static boat_tensor_t* compute_forward_dot(const boat_tensor_t* a, const boat_tensor_t* b) {
    return boat_dot(a, b);
}

static boat_tensor_t* compute_forward_relu(const boat_tensor_t* a) {
    // TODO: Implement relu operation in ops
    // For now, create a simple implementation
    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) return NULL;

    size_t nelements = boat_tensor_nelements(a);
    const void* a_data = boat_tensor_const_data(a);
    void* out_data = boat_tensor_data(out);

    boat_dtype_t dtype = boat_tensor_dtype(a);
    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* a_ptr = (const float*)a_data;
            float* out_ptr = (float*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                out_ptr[i] = a_ptr[i] > 0 ? a_ptr[i] : 0;
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* a_ptr = (const double*)a_data;
            double* out_ptr = (double*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                out_ptr[i] = a_ptr[i] > 0 ? a_ptr[i] : 0;
            }
            break;
        }
        default:
            boat_tensor_free(out);
            return NULL;
    }

    return out;
}

static boat_tensor_t* compute_forward_sigmoid(const boat_tensor_t* a) {
    // Sigmoid: 1 / (1 + exp(-x))
    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) return NULL;

    size_t nelements = boat_tensor_nelements(a);
    const void* a_data = boat_tensor_const_data(a);
    void* out_data = boat_tensor_data(out);
    boat_dtype_t dtype = boat_tensor_dtype(a);

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* a_ptr = (const float*)a_data;
            float* out_ptr = (float*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                float x = a_ptr[i];
                out_ptr[i] = 1.0f / (1.0f + expf(-x));
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* a_ptr = (const double*)a_data;
            double* out_ptr = (double*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                double x = a_ptr[i];
                out_ptr[i] = 1.0 / (1.0 + exp(-x));
            }
            break;
        }
        default:
            boat_tensor_free(out);
            return NULL;
    }

    return out;
}

static boat_tensor_t* compute_forward_tanh(const boat_tensor_t* a) {
    // Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) return NULL;

    size_t nelements = boat_tensor_nelements(a);
    const void* a_data = boat_tensor_const_data(a);
    void* out_data = boat_tensor_data(out);
    boat_dtype_t dtype = boat_tensor_dtype(a);

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* a_ptr = (const float*)a_data;
            float* out_ptr = (float*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                float x = a_ptr[i];
                float ex = expf(x);
                float emx = expf(-x);
                out_ptr[i] = (ex - emx) / (ex + emx);
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* a_ptr = (const double*)a_data;
            double* out_ptr = (double*)out_data;
            for (size_t i = 0; i < nelements; i++) {
                double x = a_ptr[i];
                double ex = exp(x);
                double emx = exp(-x);
                out_ptr[i] = (ex - emx) / (ex + emx);
            }
            break;
        }
        default:
            boat_tensor_free(out);
            return NULL;
    }

    return out;
}

static boat_tensor_t* compute_forward_matmul(const boat_tensor_t* a, const boat_tensor_t* b) {
    // Use the boat_matmul operation from ops
    return boat_matmul(a, b);
}

static boat_tensor_t* compute_forward_sum(const boat_tensor_t* a, const int64_t* dims, size_t n_dims, bool keepdim) {
    // Sum reduction
    // TODO: Implement proper sum with dimension support
    // For now, implement simple total sum
    (void)dims; (void)n_dims; (void)keepdim;

    // Use boat_sum operation which already implements full reduction
    return boat_sum(a, NULL, 0, false);
}

static boat_tensor_t* compute_forward_sum_single(const boat_tensor_t* a) {
    // Wrapper for create_operation compatibility
    return compute_forward_sum(a, NULL, 0, false);
}

static boat_tensor_t* compute_forward_mean(const boat_tensor_t* a, const int64_t* dims, size_t n_dims, bool keepdim) {
    // Mean reduction
    // TODO: Implement proper mean with dimension support
    // For now, implement simple total mean
    (void)dims; (void)n_dims; (void)keepdim;

    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) return NULL;

    size_t nelements = boat_tensor_nelements(a);
    if (nelements == 0) {
        boat_tensor_free(out);
        return NULL;
    }

    const void* a_data = boat_tensor_const_data(a);
    void* out_data = boat_tensor_data(out);
    boat_dtype_t dtype = boat_tensor_dtype(a);

    // Simple total mean (all elements)
    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* a_ptr = (const float*)a_data;
            float sum = 0.0f;
            for (size_t i = 0; i < nelements; i++) {
                sum += a_ptr[i];
            }
            float* out_ptr = (float*)out_data;
            out_ptr[0] = sum / (float)nelements;
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* a_ptr = (const double*)a_data;
            double sum = 0.0;
            for (size_t i = 0; i < nelements; i++) {
                sum += a_ptr[i];
            }
            double* out_ptr = (double*)out_data;
            out_ptr[0] = sum / (double)nelements;
            break;
        }
        default:
            boat_tensor_free(out);
            return NULL;
    }

    return out;
}

// Convolution forward computation
static boat_tensor_t* compute_forward_conv(const boat_tensor_t* input, const void* layer_ptr) {
    if (!input || !layer_ptr) {
        return NULL;
    }
    const boat_conv_layer_t* layer = (const boat_conv_layer_t*)layer_ptr;
    boat_tensor_t* output = boat_conv_layer_forward(layer, input);
    return output;
}

// Backward computation functions
static void compute_backward_add(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 2 || !grad_output) return;


    // Gradient for addition: ∂L/∂a = ∂L/∂c, ∂L/∂b = ∂L/∂c
    // where c = a + b
    boat_variable_t* a = op_data->inputs[0];
    boat_variable_t* b = op_data->inputs[1];


    if (a->requires_grad) {
        if (!a->grad) {
            // Create gradient tensor and copy grad_output
            a->grad = boat_tensor_create_like(a->data);
            if (a->grad) {
                // Copy grad_output to gradient
                // Note: grad_output may need broadcasting, but for addition gradient shapes match
                // For now, assume same shape (should be true for element-wise ops)
                boat_tensor_t* grad_output_clone = boat_tensor_create_like(grad_output);
                if (grad_output_clone) {
                    memcpy(boat_tensor_data(grad_output_clone), boat_tensor_const_data(grad_output),
                           boat_tensor_nbytes(grad_output));
                    // Add to gradient (grad is zero, so just copy)
                    boat_add_(a->grad, grad_output_clone);
                    boat_tensor_unref(grad_output_clone);
                }
            } else {
            }
        } else {
            // Accumulate gradient: a->grad += grad_output
            boat_tensor_t* grad_output_clone = boat_tensor_create_like(grad_output);
            if (grad_output_clone) {
                memcpy(boat_tensor_data(grad_output_clone), boat_tensor_const_data(grad_output),
                       boat_tensor_nbytes(grad_output));
                boat_add_(a->grad, grad_output_clone);
                boat_tensor_unref(grad_output_clone);
            }
        }
    }

    if (b->requires_grad) {
        if (!b->grad) {
            b->grad = boat_tensor_create_like(b->data);
            if (b->grad) {
                boat_tensor_t* grad_output_clone = boat_tensor_create_like(grad_output);
                if (grad_output_clone) {
                    memcpy(boat_tensor_data(grad_output_clone), boat_tensor_const_data(grad_output),
                           boat_tensor_nbytes(grad_output));
                    boat_add_(b->grad, grad_output_clone);
                    boat_tensor_unref(grad_output_clone);
                }
            }
        } else {
            boat_tensor_t* grad_output_clone = boat_tensor_create_like(grad_output);
            if (grad_output_clone) {
                memcpy(boat_tensor_data(grad_output_clone), boat_tensor_const_data(grad_output),
                       boat_tensor_nbytes(grad_output));
                boat_add_(b->grad, grad_output_clone);
                boat_tensor_unref(grad_output_clone);
            }
        }
    }
}

static void compute_backward_sub(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 2 || !grad_output) return;

    // Gradient for subtraction: ∂L/∂a = ∂L/∂c, ∂L/∂b = -∂L/∂c
    // where c = a - b
    boat_variable_t* a = op_data->inputs[0];
    boat_variable_t* b = op_data->inputs[1];

    if (a->requires_grad) {
        if (!a->grad) {
            // Create gradient tensor and copy grad_output
            a->grad = boat_tensor_create_like(a->data);
            if (a->grad) {
                boat_tensor_t* grad_output_clone = boat_tensor_create_like(grad_output);
                if (grad_output_clone) {
                    memcpy(boat_tensor_data(grad_output_clone), boat_tensor_const_data(grad_output),
                           boat_tensor_nbytes(grad_output));
                    boat_add_(a->grad, grad_output_clone);
                    boat_tensor_unref(grad_output_clone);
                }
            }
        } else {
            boat_tensor_t* grad_output_clone = boat_tensor_create_like(grad_output);
            if (grad_output_clone) {
                memcpy(boat_tensor_data(grad_output_clone), boat_tensor_const_data(grad_output),
                       boat_tensor_nbytes(grad_output));
                boat_add_(a->grad, grad_output_clone);
                boat_tensor_unref(grad_output_clone);
            }
        }
    }

    if (b->requires_grad) {
        // Gradient for b is negative of grad_output
        boat_tensor_t* neg_grad_output = boat_mul_scalar(grad_output, -1.0);
        if (!neg_grad_output) return;

        if (!b->grad) {
            b->grad = boat_tensor_create_like(b->data);
            if (b->grad) {
                boat_add_(b->grad, neg_grad_output);
            }
        } else {
            boat_add_(b->grad, neg_grad_output);
        }
        boat_tensor_unref(neg_grad_output);
    }
}

static void compute_backward_mul(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 2 || !grad_output) return;

    // Gradient for multiplication: ∂L/∂a = ∂L/∂c * b, ∂L/∂b = ∂L/∂c * a
    // where c = a * b
    boat_variable_t* a = op_data->inputs[0];
    boat_variable_t* b = op_data->inputs[1];

    if (a->requires_grad) {
        // Compute gradient contribution: grad_output * b
        boat_tensor_t* grad_a = boat_mul(grad_output, b->data);
        if (grad_a) {
            if (!a->grad) {
                a->grad = grad_a;
            } else {
                // Accumulate gradient: a->grad += grad_a
                boat_add_(a->grad, grad_a);
                boat_tensor_unref(grad_a);
            }
        }
    }

    if (b->requires_grad) {
        // Compute gradient contribution: grad_output * a
        boat_tensor_t* grad_b = boat_mul(grad_output, a->data);
        if (grad_b) {
            if (!b->grad) {
                b->grad = grad_b;
            } else {
                boat_add_(b->grad, grad_b);
                boat_tensor_unref(grad_b);
            }
        }
    }
}

static void compute_backward_div(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 2 || !grad_output) return;

    // Gradient for division: ∂L/∂a = ∂L/∂c / b, ∂L/∂b = -∂L/∂c * a / b²
    // where c = a / b
    boat_variable_t* a = op_data->inputs[0];
    boat_variable_t* b = op_data->inputs[1];

    if (a->requires_grad) {
        // Compute gradient contribution: grad_output / b
        boat_tensor_t* grad_a = boat_div(grad_output, b->data);
        if (grad_a) {
            if (!a->grad) {
                a->grad = grad_a;
            } else {
                boat_add_(a->grad, grad_a);
                boat_tensor_unref(grad_a);
            }
        }
    }

    if (b->requires_grad) {
        // Compute gradient contribution: -grad_output * a / (b * b)
        // First compute b²
        boat_tensor_t* b_squared = boat_mul(b->data, b->data);
        if (!b_squared) return;

        // Compute a / b²
        boat_tensor_t* a_div_bsq = boat_div(a->data, b_squared);
        boat_tensor_unref(b_squared);
        if (!a_div_bsq) return;

        // Compute -grad_output * (a / b²)
        boat_tensor_t* neg_grad = boat_mul_scalar(grad_output, -1.0);
        if (!neg_grad) {
            boat_tensor_unref(a_div_bsq);
            return;
        }

        boat_tensor_t* grad_b = boat_mul(neg_grad, a_div_bsq);
        boat_tensor_unref(neg_grad);
        boat_tensor_unref(a_div_bsq);

        if (grad_b) {
            if (!b->grad) {
                b->grad = grad_b;
            } else {
                boat_add_(b->grad, grad_b);
                boat_tensor_unref(grad_b);
            }
        }
    }
}

static void compute_backward_dot(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 2 || !grad_output) return;

    // Gradient for dot product: ∂L/∂a = ∂L/∂c * b, ∂L/∂b = ∂L/∂c * a
    // where c = dot(a, b) (scalar)
    boat_variable_t* a = op_data->inputs[0];
    boat_variable_t* b = op_data->inputs[1];

    // grad_output is scalar (dot product output is scalar)
    // Need to broadcast grad_output to shape of a and b
    // For dot product of vectors, gradient w.r.t a is grad_output * b (element-wise)

    if (a->requires_grad) {
        // Compute gradient contribution: grad_output * b
        // Since grad_output is scalar, we can multiply scalar
        boat_tensor_t* grad_a = boat_mul_scalar(b->data, *((float*)boat_tensor_data(grad_output)));
        if (grad_a) {
            if (!a->grad) {
                a->grad = grad_a;
            } else {
                boat_add_(a->grad, grad_a);
                boat_tensor_unref(grad_a);
            }
        }
    }

    if (b->requires_grad) {
        boat_tensor_t* grad_b = boat_mul_scalar(a->data, *((float*)boat_tensor_data(grad_output)));
        if (grad_b) {
            if (!b->grad) {
                b->grad = grad_b;
            } else {
                boat_add_(b->grad, grad_b);
                boat_tensor_unref(grad_b);
            }
        }
    }
}

static void compute_backward_relu(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 1 || !grad_output) return;

    // Gradient for ReLU: ∂L/∂a = ∂L/∂c * (a > 0 ? 1 : 0)
    // where c = relu(a)
    boat_variable_t* a = op_data->inputs[0];

    if (a->requires_grad) {
        // Create mask tensor: 1 where a->data > 0, else 0
        boat_tensor_t* mask = boat_tensor_create_like(a->data);
        if (!mask) return;

        const void* a_data = boat_tensor_data(a->data);
        void* mask_data = boat_tensor_data(mask);
        size_t nelements = boat_tensor_nelements(a->data);
        boat_dtype_t dtype = boat_tensor_dtype(a->data);

        switch (dtype) {
            case BOAT_DTYPE_FLOAT32: {
                const float* a_ptr = (const float*)a_data;
                float* mask_ptr = (float*)mask_data;
                for (size_t i = 0; i < nelements; i++) {
                    mask_ptr[i] = a_ptr[i] > 0 ? 1.0f : 0.0f;
                }
                break;
            }
            case BOAT_DTYPE_FLOAT64: {
                const double* a_ptr = (const double*)a_data;
                double* mask_ptr = (double*)mask_data;
                for (size_t i = 0; i < nelements; i++) {
                    mask_ptr[i] = a_ptr[i] > 0 ? 1.0 : 0.0;
                }
                break;
            }
            default:
                // Unsupported type for ReLU gradient
                boat_tensor_unref(mask);
                return;
        }

        // Compute gradient: grad_output * mask
        boat_tensor_t* grad = boat_mul(grad_output, mask);
        boat_tensor_unref(mask);
        if (!grad) return;

        if (!a->grad) {
            a->grad = grad;
        } else {
            boat_add_(a->grad, grad);
            boat_tensor_unref(grad);
        }
    }
}

static void compute_backward_sigmoid(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 1 || !grad_output) return;

    // Gradient for sigmoid: ∂L/∂a = ∂L/∂c * sigmoid(a) * (1 - sigmoid(a))
    // where c = sigmoid(a), and sigmoid(a) is stored in output->data
    boat_variable_t* a = op_data->inputs[0];
    const boat_variable_t* c = op_data->output;  // c = sigmoid(a)

    if (a->requires_grad) {
        // Compute gradient contribution: grad_output * c * (1 - c)
        // where c = sigmoid(a)
        const boat_tensor_t* c_data = c->data;

        // Create tensor (1 - c)
        boat_tensor_t* one_minus_c = boat_tensor_create_like(c_data);
        if (!one_minus_c) return;

        const void* c_ptr = boat_tensor_data(c_data);
        void* omc_ptr = boat_tensor_data(one_minus_c);
        size_t nelements = boat_tensor_nelements(c_data);
        boat_dtype_t dtype = boat_tensor_dtype(c_data);

        switch (dtype) {
            case BOAT_DTYPE_FLOAT32: {
                const float* c_data_ptr = (const float*)c_ptr;
                float* omc_data_ptr = (float*)omc_ptr;
                for (size_t i = 0; i < nelements; i++) {
                    omc_data_ptr[i] = 1.0f - c_data_ptr[i];
                }
                break;
            }
            case BOAT_DTYPE_FLOAT64: {
                const double* c_data_ptr = (const double*)c_ptr;
                double* omc_data_ptr = (double*)omc_ptr;
                for (size_t i = 0; i < nelements; i++) {
                    omc_data_ptr[i] = 1.0 - c_data_ptr[i];
                }
                break;
            }
            default:
                boat_tensor_unref(one_minus_c);
                return;
        }

        // Compute c * (1 - c)
        boat_tensor_t* c_times_omc = boat_mul(c_data, one_minus_c);
        boat_tensor_unref(one_minus_c);
        if (!c_times_omc) return;

        // Compute grad_output * c * (1 - c)
        boat_tensor_t* grad = boat_mul(grad_output, c_times_omc);
        boat_tensor_unref(c_times_omc);
        if (!grad) return;

        if (!a->grad) {
            a->grad = grad;
        } else {
            boat_add_(a->grad, grad);
            boat_tensor_unref(grad);
        }
    }
}

static void compute_backward_tanh(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 1 || !grad_output) return;

    // Gradient for tanh: ∂L/∂a = ∂L/∂c * (1 - tanh²(a))
    // where c = tanh(a), and c is stored in output->data
    boat_variable_t* a = op_data->inputs[0];
    const boat_variable_t* c = op_data->output;  // c = tanh(a)

    if (a->requires_grad) {
        // Compute gradient contribution: grad_output * (1 - c²)
        const boat_tensor_t* c_data = c->data;

        // Create tensor c²
        boat_tensor_t* c_squared = boat_tensor_create_like(c_data);
        if (!c_squared) return;

        const void* c_ptr = boat_tensor_data(c_data);
        void* csq_ptr = boat_tensor_data(c_squared);
        size_t nelements = boat_tensor_nelements(c_data);
        boat_dtype_t dtype = boat_tensor_dtype(c_data);

        switch (dtype) {
            case BOAT_DTYPE_FLOAT32: {
                const float* c_data_ptr = (const float*)c_ptr;
                float* csq_data_ptr = (float*)csq_ptr;
                for (size_t i = 0; i < nelements; i++) {
                    float val = c_data_ptr[i];
                    csq_data_ptr[i] = val * val;
                }
                break;
            }
            case BOAT_DTYPE_FLOAT64: {
                const double* c_data_ptr = (const double*)c_ptr;
                double* csq_data_ptr = (double*)csq_ptr;
                for (size_t i = 0; i < nelements; i++) {
                    double val = c_data_ptr[i];
                    csq_data_ptr[i] = val * val;
                }
                break;
            }
            default:
                boat_tensor_unref(c_squared);
                return;
        }

        // Create tensor (1 - c²)
        boat_tensor_t* one_minus_csq = boat_tensor_create_like(c_data);
        if (!one_minus_csq) {
            boat_tensor_unref(c_squared);
            return;
        }

        void* omcsq_ptr = boat_tensor_data(one_minus_csq);

        switch (dtype) {
            case BOAT_DTYPE_FLOAT32: {
                const float* csq_data_ptr = (const float*)csq_ptr;
                float* omcsq_data_ptr = (float*)omcsq_ptr;
                for (size_t i = 0; i < nelements; i++) {
                    omcsq_data_ptr[i] = 1.0f - csq_data_ptr[i];
                }
                break;
            }
            case BOAT_DTYPE_FLOAT64: {
                const double* csq_data_ptr = (const double*)csq_ptr;
                double* omcsq_data_ptr = (double*)omcsq_ptr;
                for (size_t i = 0; i < nelements; i++) {
                    omcsq_data_ptr[i] = 1.0 - csq_data_ptr[i];
                }
                break;
            }
            default:
                boat_tensor_unref(c_squared);
                boat_tensor_unref(one_minus_csq);
                return;
        }

        boat_tensor_unref(c_squared);

        // Compute grad_output * (1 - c²)
        boat_tensor_t* grad = boat_mul(grad_output, one_minus_csq);
        boat_tensor_unref(one_minus_csq);
        if (!grad) return;

        if (!a->grad) {
            a->grad = grad;
        } else {
            boat_add_(a->grad, grad);
            boat_tensor_unref(grad);
        }
    }
}

static void compute_backward_matmul(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 2 || !grad_output) return;

    boat_variable_t* a = op_data->inputs[0];
    boat_variable_t* b = op_data->inputs[1];
    // boat_variable_t* c = op_data->output; // Not used in this implementation

    if (a->requires_grad) {
        // ∂L/∂A = ∂L/∂C @ Bᵀ
        boat_tensor_t* b_transposed = boat_transpose(b->data, 0, 1);
        if (b_transposed) {
            boat_tensor_t* grad_a = boat_matmul(grad_output, b_transposed);
            boat_tensor_unref(b_transposed);

            if (grad_a) {
                if (!a->grad) {
                    a->grad = grad_a;
                } else {
                    boat_add_(a->grad, grad_a);
                    boat_tensor_unref(grad_a);
                }
            }
        }
    }

    if (b->requires_grad) {
        // ∂L/∂B = Aᵀ @ ∂L/∂C
        boat_tensor_t* a_transposed = boat_transpose(a->data, 0, 1);
        if (a_transposed) {
            boat_tensor_t* grad_b = boat_matmul(a_transposed, grad_output);
            boat_tensor_unref(a_transposed);

            if (grad_b) {
                if (!b->grad) {
                    b->grad = grad_b;
                } else {
                    boat_add_(b->grad, grad_b);
                    boat_tensor_unref(grad_b);
                }
            }
        }
    }
}

static void compute_backward_sum(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 1 || !grad_output) return;

    // Gradient for sum: ∂L/∂a = ∂L/∂c * 1 (broadcasted to original shape)
    // where c = sum(a)
    boat_variable_t* a = op_data->inputs[0];

    if (a->requires_grad) {
        // For sum reduction, gradient is grad_output broadcasted to input shape
        // Since our current implementation only does total sum, grad_output is scalar
        // We need to create a tensor of ones with same shape as a and multiply by grad_output

        boat_tensor_t* ones = boat_tensor_create_like(a->data);
        if (!ones) return;

        // Fill with 1.0
        size_t nelements = boat_tensor_nelements(ones);
        void* data = boat_tensor_data(ones);
        boat_dtype_t dtype = boat_tensor_dtype(ones);

        switch (dtype) {
            case BOAT_DTYPE_FLOAT32: {
                float* ptr = (float*)data;
                for (size_t i = 0; i < nelements; i++) ptr[i] = 1.0f;
                break;
            }
            case BOAT_DTYPE_FLOAT64: {
                double* ptr = (double*)data;
                for (size_t i = 0; i < nelements; i++) ptr[i] = 1.0;
                break;
            }
            default:
                boat_tensor_unref(ones);
                return;
        }

        // Multiply ones by grad_output (scalar)
        // grad_output is scalar (1 element) from total sum
        boat_tensor_t* grad = boat_mul_scalar(ones, *((float*)boat_tensor_data(grad_output)));
        boat_tensor_unref(ones);
        if (!grad) return;

        if (!a->grad) {
            a->grad = grad;
        } else {
            boat_add_(a->grad, grad);
            boat_tensor_unref(grad);
        }
    }
}

static void compute_backward_mean(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 1 || !grad_output) return;

    // Gradient for mean: ∂L/∂a = ∂L/∂c * (1 / n) (broadcasted to original shape)
    // where c = mean(a), n = number of elements
    boat_variable_t* a = op_data->inputs[0];

    if (a->requires_grad) {
        size_t nelements = boat_tensor_nelements(a->data);
        if (nelements == 0) return;

        boat_tensor_t* ones = boat_tensor_create_like(a->data);
        if (!ones) return;

        // Fill with 1.0
        void* data = boat_tensor_data(ones);
        boat_dtype_t dtype = boat_tensor_dtype(ones);

        switch (dtype) {
            case BOAT_DTYPE_FLOAT32: {
                float* ptr = (float*)data;
                for (size_t i = 0; i < nelements; i++) ptr[i] = 1.0f;
                break;
            }
            case BOAT_DTYPE_FLOAT64: {
                double* ptr = (double*)data;
                for (size_t i = 0; i < nelements; i++) ptr[i] = 1.0;
                break;
            }
            default:
                boat_tensor_unref(ones);
                return;
        }

        // Multiply ones by grad_output / n
        if (dtype == BOAT_DTYPE_FLOAT64) {
            double scale_d = 1.0 / nelements;
            boat_tensor_t* scaled = boat_mul_scalar(ones, scale_d);
            boat_tensor_unref(ones);
            if (!scaled) return;

            boat_tensor_t* grad = boat_mul(scaled, grad_output);
            boat_tensor_unref(scaled);
            if (!grad) return;

            if (!a->grad) {
                a->grad = grad;
            } else {
                boat_add_(a->grad, grad);
                boat_tensor_unref(grad);
            }
        } else {
            float scale = 1.0f / nelements;
            boat_tensor_t* scaled = boat_mul_scalar(ones, scale);
            boat_tensor_unref(ones);
            if (!scaled) return;

            boat_tensor_t* grad = boat_mul(scaled, grad_output);
            boat_tensor_unref(scaled);
            if (!grad) return;

            if (!a->grad) {
                a->grad = grad;
            } else {
                boat_add_(a->grad, grad);
                boat_tensor_unref(grad);
            }
        }
    }
}

// Helper function to unify variable graphs
static bool unify_variable_graphs(boat_variable_t** inputs, size_t num_inputs, boat_graph_t** target_graph) {
    if (!inputs || num_inputs == 0 || !target_graph) {
        return false;
    }

    // Find the first variable with a graph to use as target
    boat_graph_t* target = NULL;
    size_t target_index = SIZE_MAX;
    for (size_t i = 0; i < num_inputs; i++) {
        if (inputs[i] && inputs[i]->graph) {
            target = inputs[i]->graph;
            target_index = i;
            break;
        }
    }

    // If no variable has a graph, create a new one
    if (!target) {
        // Use the device from the first variable's tensor
        boat_device_t device = BOAT_DEVICE_CPU;
        for (size_t i = 0; i < num_inputs; i++) {
            if (inputs[i] && inputs[i]->data) {
                device = boat_tensor_device(inputs[i]->data);
                break;
            }
        }
        target = boat_graph_create_with_device(device);
        if (!target) {
            return false;
        }
    }


    // Migrate all variable nodes to the target graph
    bool migration_failed = false;
    for (size_t i = 0; i < num_inputs; i++) {
        boat_variable_t* var = inputs[i];
        if (!var) continue;

        // Skip variables without graph (requires_grad may be false)
        if (!var->graph) {
            // If variable has no graph but has a node (should not happen), handle it
            if (var->node) {
            }
            // Set graph to target for consistency (even if no node)
            var->graph = target;
            continue;
        }

        // If variable already in target graph, skip
        if (var->graph == target) {
            continue;
        }

        // If variable has a node, migrate it
        if (var->node) {
            if (!boat_graph_migrate_node(target, var->graph, var->node)) {
                migration_failed = true;
                continue;
            }
            // Update variable's graph reference
            var->graph = target;
        } else {
            // Variable has graph but no node (unusual but possible)
            // Just update graph reference
            var->graph = target;
        }
    }

    if (migration_failed) {
        // Continue anyway, but operations may fail
    }

    *target_graph = target;
    return true;
}

// Generic operation creation function

// Softmax operations
static boat_tensor_t* compute_forward_softmax(const boat_tensor_t* a) {
    // Numerical stable softmax along last dimension
    // softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) return NULL;

    size_t nelements = boat_tensor_nelements(a);
    if (nelements == 0) return out;

    const void* a_data = boat_tensor_const_data(a);
    void* out_data = boat_tensor_data(out);
    boat_dtype_t dtype = boat_tensor_dtype(a);

    // Get shape information for last dimension
    const int64_t* shape = boat_tensor_shape(a);
    size_t ndim = boat_tensor_ndim(a);
    size_t last_dim = ndim > 0 ? shape[ndim - 1] : 1;
    size_t n_rows = nelements / last_dim;

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* a_ptr = (const float*)a_data;
            float* out_ptr = (float*)out_data;

            for (size_t row = 0; row < n_rows; row++) {
                size_t row_offset = row * last_dim;

                // Find max in this row for numerical stability
                float row_max = a_ptr[row_offset];
                for (size_t i = 1; i < last_dim; i++) {
                    float val = a_ptr[row_offset + i];
                    if (val > row_max) row_max = val;
                }

                // Compute exp(x - max) and sum
                float exp_sum = 0.0f;
                for (size_t i = 0; i < last_dim; i++) {
                    float val = a_ptr[row_offset + i] - row_max;
                    float exp_val = expf(val);
                    out_ptr[row_offset + i] = exp_val;
                    exp_sum += exp_val;
                }

                // Normalize
                if (exp_sum != 0.0f) {
                    float inv_exp_sum = 1.0f / exp_sum;
                    for (size_t i = 0; i < last_dim; i++) {
                        out_ptr[row_offset + i] *= inv_exp_sum;
                    }
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* a_ptr = (const double*)a_data;
            double* out_ptr = (double*)out_data;

            for (size_t row = 0; row < n_rows; row++) {
                size_t row_offset = row * last_dim;

                // Find max in this row for numerical stability
                double row_max = a_ptr[row_offset];
                for (size_t i = 1; i < last_dim; i++) {
                    double val = a_ptr[row_offset + i];
                    if (val > row_max) row_max = val;
                }

                // Compute exp(x - max) and sum
                double exp_sum = 0.0;
                for (size_t i = 0; i < last_dim; i++) {
                    double val = a_ptr[row_offset + i] - row_max;
                    double exp_val = exp(val);
                    out_ptr[row_offset + i] = exp_val;
                    exp_sum += exp_val;
                }

                // Normalize
                if (exp_sum != 0.0) {
                    double inv_exp_sum = 1.0 / exp_sum;
                    for (size_t i = 0; i < last_dim; i++) {
                        out_ptr[row_offset + i] *= inv_exp_sum;
                    }
                }
            }
            break;
        }
        default:
            // Only support float types for softmax
            boat_tensor_free(out);
            return NULL;
    }

    return out;
}

static boat_tensor_t* compute_forward_log_softmax(const boat_tensor_t* a) {
    // Numerical stable log_softmax along last dimension
    // log_softmax(x_i) = x_i - max(x) - log(sum(exp(x_j - max(x))))
    boat_tensor_t* out = boat_tensor_create_like(a);
    if (!out) return NULL;

    size_t nelements = boat_tensor_nelements(a);
    if (nelements == 0) return out;

    const void* a_data = boat_tensor_const_data(a);
    void* out_data = boat_tensor_data(out);
    boat_dtype_t dtype = boat_tensor_dtype(a);

    // Get shape information for last dimension
    const int64_t* shape = boat_tensor_shape(a);
    size_t ndim = boat_tensor_ndim(a);
    size_t last_dim = ndim > 0 ? shape[ndim - 1] : 1;
    size_t n_rows = nelements / last_dim;

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* a_ptr = (const float*)a_data;
            float* out_ptr = (float*)out_data;

            for (size_t row = 0; row < n_rows; row++) {
                size_t row_offset = row * last_dim;

                // Find max in this row for numerical stability
                float row_max = a_ptr[row_offset];
                for (size_t i = 1; i < last_dim; i++) {
                    float val = a_ptr[row_offset + i];
                    if (val > row_max) row_max = val;
                }

                // Compute exp(x - max) and sum
                float exp_sum = 0.0f;
                for (size_t i = 0; i < last_dim; i++) {
                    float val = a_ptr[row_offset + i] - row_max;
                    float exp_val = expf(val);
                    exp_sum += exp_val;
                }

                // Compute log_softmax
                float log_exp_sum = logf(fmaxf(exp_sum, FLT_MIN));
                for (size_t i = 0; i < last_dim; i++) {
                    out_ptr[row_offset + i] = a_ptr[row_offset + i] - row_max - log_exp_sum;
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* a_ptr = (const double*)a_data;
            double* out_ptr = (double*)out_data;

            for (size_t row = 0; row < n_rows; row++) {
                size_t row_offset = row * last_dim;

                // Find max in this row for numerical stability
                double row_max = a_ptr[row_offset];
                for (size_t i = 1; i < last_dim; i++) {
                    double val = a_ptr[row_offset + i];
                    if (val > row_max) row_max = val;
                }

                // Compute exp(x - max) and sum
                double exp_sum = 0.0;
                for (size_t i = 0; i < last_dim; i++) {
                    double val = a_ptr[row_offset + i] - row_max;
                    double exp_val = exp(val);
                    exp_sum += exp_val;
                }

                // Compute log_softmax
                double log_exp_sum = log(fmax(exp_sum, DBL_MIN));
                for (size_t i = 0; i < last_dim; i++) {
                    out_ptr[row_offset + i] = a_ptr[row_offset + i] - row_max - log_exp_sum;
                }
            }
            break;
        }
        default:
            // Only support float types for softmax
            boat_tensor_free(out);
            return NULL;
    }

    return out;
}

static void compute_backward_softmax(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 1 || !grad_output || !op_data->output) return;

    // Gradient for softmax: ∂L/∂x_i = y_i * (∂L/∂y_i - sum(y_j * ∂L/∂y_j))
    // where y = softmax(x)
    boat_variable_t* a = op_data->inputs[0];
    const boat_variable_t* y_var = op_data->output;

    if (!a->requires_grad) return;

    const boat_tensor_t* y = y_var->data;
    size_t nelements = boat_tensor_nelements(y);
    if (nelements == 0) return;

    const void* y_data = boat_tensor_data(y);
    const void* grad_output_data = boat_tensor_data(grad_output);
    boat_dtype_t dtype = boat_tensor_dtype(y);

    // Get shape information for last dimension
    const int64_t* shape = boat_tensor_shape(y);
    size_t ndim = boat_tensor_ndim(y);
    size_t last_dim = ndim > 0 ? shape[ndim - 1] : 1;
    size_t n_rows = nelements / last_dim;

    // Create gradient tensor for input
    boat_tensor_t* grad = boat_tensor_create_like(y);
    if (!grad) return;

    void* grad_data = boat_tensor_data(grad);

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* y_ptr = (const float*)y_data;
            const float* grad_out_ptr = (const float*)grad_output_data;
            float* grad_ptr = (float*)grad_data;

            for (size_t row = 0; row < n_rows; row++) {
                size_t row_offset = row * last_dim;

                // Compute sum(y_j * ∂L/∂y_j) for this row
                float sum_y_grad = 0.0f;
                for (size_t i = 0; i < last_dim; i++) {
                    sum_y_grad += y_ptr[row_offset + i] * grad_out_ptr[row_offset + i];
                }

                // Compute gradient: y_i * (∂L/∂y_i - sum_y_grad)
                for (size_t i = 0; i < last_dim; i++) {
                    float grad_out = grad_out_ptr[row_offset + i];
                    grad_ptr[row_offset + i] = y_ptr[row_offset + i] * (grad_out - sum_y_grad);
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* y_ptr = (const double*)y_data;
            const double* grad_out_ptr = (const double*)grad_output_data;
            double* grad_ptr = (double*)grad_data;

            for (size_t row = 0; row < n_rows; row++) {
                size_t row_offset = row * last_dim;

                // Compute sum(y_j * ∂L/∂y_j) for this row
                double sum_y_grad = 0.0;
                for (size_t i = 0; i < last_dim; i++) {
                    sum_y_grad += y_ptr[row_offset + i] * grad_out_ptr[row_offset + i];
                }

                // Compute gradient: y_i * (∂L/∂y_i - sum_y_grad)
                for (size_t i = 0; i < last_dim; i++) {
                    double grad_out = grad_out_ptr[row_offset + i];
                    grad_ptr[row_offset + i] = y_ptr[row_offset + i] * (grad_out - sum_y_grad);
                }
            }
            break;
        }
        default:
            // Only support float types for softmax gradient
            boat_tensor_unref(grad);
            return;
    }

    // Accumulate gradient
    if (!a->grad) {
        a->grad = grad;
    } else {
        boat_add_(a->grad, grad);
        boat_tensor_unref(grad);
    }
}

static void compute_backward_log_softmax(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 1 || !grad_output || !op_data->output) return;

    // Gradient for log_softmax: ∂L/∂x_i = ∂L/∂y_i - exp(y_i) * sum(∂L/∂y_j)
    // where y = log_softmax(x), and exp(y) = softmax(x)
    // Actually simpler: ∂L/∂x_i = ∂L/∂y_i - softmax(x_i) * sum(∂L/∂y_j)
    boat_variable_t* a = op_data->inputs[0];
    const boat_variable_t* y_var = op_data->output;

    if (!a->requires_grad) return;

    // Need softmax(x) = exp(y) for gradient computation
    // Since we don't store softmax separately, compute it from log_softmax output
    const boat_tensor_t* y = y_var->data;
    size_t nelements = boat_tensor_nelements(y);
    if (nelements == 0) return;

    const void* y_data = boat_tensor_data(y);
    const void* grad_output_data = boat_tensor_data(grad_output);
    boat_dtype_t dtype = boat_tensor_dtype(y);

    // Get shape information for last dimension
    const int64_t* shape = boat_tensor_shape(y);
    size_t ndim = boat_tensor_ndim(y);
    size_t last_dim = ndim > 0 ? shape[ndim - 1] : 1;
    size_t n_rows = nelements / last_dim;

    // Create gradient tensor for input
    boat_tensor_t* grad = boat_tensor_create_like(y);
    if (!grad) return;

    void* grad_data = boat_tensor_data(grad);

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* y_ptr = (const float*)y_data;
            const float* grad_out_ptr = (const float*)grad_output_data;
            float* grad_ptr = (float*)grad_data;

            for (size_t row = 0; row < n_rows; row++) {
                size_t row_offset = row * last_dim;

                // Compute sum(∂L/∂y_j) for this row
                float sum_grad = 0.0f;
                for (size_t i = 0; i < last_dim; i++) {
                    sum_grad += grad_out_ptr[row_offset + i];
                }

                // Compute softmax(x_i) = exp(y_i)
                // Then gradient: ∂L/∂x_i = ∂L/∂y_i - exp(y_i) * sum_grad
                for (size_t i = 0; i < last_dim; i++) {
                    float softmax_val = expf(y_ptr[row_offset + i]);
                    grad_ptr[row_offset + i] = grad_out_ptr[row_offset + i] - softmax_val * sum_grad;
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* y_ptr = (const double*)y_data;
            const double* grad_out_ptr = (const double*)grad_output_data;
            double* grad_ptr = (double*)grad_data;

            for (size_t row = 0; row < n_rows; row++) {
                size_t row_offset = row * last_dim;

                // Compute sum(∂L/∂y_j) for this row
                double sum_grad = 0.0;
                for (size_t i = 0; i < last_dim; i++) {
                    sum_grad += grad_out_ptr[row_offset + i];
                }

                // Compute softmax(x_i) = exp(y_i)
                // Then gradient: ∂L/∂x_i = ∂L/∂y_i - exp(y_i) * sum_grad
                for (size_t i = 0; i < last_dim; i++) {
                    double softmax_val = exp(y_ptr[row_offset + i]);
                    grad_ptr[row_offset + i] = grad_out_ptr[row_offset + i] - softmax_val * sum_grad;
                }
            }
            break;
        }
        default:
            // Only support float types for log_softmax gradient
            boat_tensor_unref(grad);
            return;
    }

    // Accumulate gradient
    if (!a->grad) {
        a->grad = grad;
    } else {
        boat_add_(a->grad, grad);
        boat_tensor_unref(grad);
    }
}

static boat_variable_t* create_operation(boat_op_type_t op_type,
                                         boat_variable_t** inputs,
                                         size_t num_inputs,
                                         boat_tensor_t* (*forward_fn)(const boat_tensor_t*, const boat_tensor_t*),
                                         boat_tensor_t* (*forward_single_fn)(const boat_tensor_t*)) {
    if (!inputs || num_inputs == 0) return NULL;

    // Check if any input requires gradient
    bool requires_grad = false;
    for (size_t i = 0; i < num_inputs; i++) {
        if (inputs[i] && inputs[i]->requires_grad) {
            requires_grad = true;
            break;
        }
    }

    // Perform forward computation
    boat_tensor_t* output_tensor = NULL;
    if (num_inputs == 1 && forward_single_fn) {
        output_tensor = forward_single_fn(inputs[0]->data);
    } else if (num_inputs == 2 && forward_fn) {
        output_tensor = forward_fn(inputs[0]->data, inputs[1]->data);
    } else {
        return NULL; // Unsupported number of inputs
    }

    if (!output_tensor) return NULL;

    // Create output variable
    boat_variable_t* output_var = boat_variable_create(output_tensor, requires_grad);
    if (!output_var) {
        boat_tensor_unref(output_tensor);
        return NULL;
    }

    // If gradient is required, create operation node and connect to graph
    if (requires_grad) {
        // Create operation node data
        boat_op_node_data_t* op_data = create_op_node_data(op_type, inputs, num_inputs, output_var);
        if (!op_data) {
            boat_variable_free(output_var);
            return NULL;
        }

        // Unify variable graphs to ensure all inputs are in the same graph
        boat_graph_t* graph = NULL;
        if (!unify_variable_graphs(inputs, num_inputs, &graph)) {
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }

        boat_node_t* op_node = boat_graph_add_node(graph, op_data, BOAT_NODE_TYPE_OPERATION, free_op_node_data);
        if (!op_node) {
            if (graph != inputs[0]->graph) {
                boat_graph_free(graph);
            }
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }
        output_var->producer_node = op_node;

        // Connect input nodes to operation node
        for (size_t i = 0; i < num_inputs; i++) {
            if (inputs[i]->node) {
                boat_graph_add_edge(graph, inputs[i]->node, op_node, BOAT_EDGE_DIRECTION_FORWARD);
            }
        }

        // Connect operation node to output node
        if (output_var->node) {
            boat_graph_add_edge(graph, op_node, output_var->node, BOAT_EDGE_DIRECTION_FORWARD);
        }

        // Set output variable's graph
        output_var->graph = graph;
    }

    return output_var;
}
static boat_variable_t* create_conv_operation(const boat_variable_t* input, const struct boat_conv_layer_t* layer) {
    if (!input || !layer) return NULL;

    // Check if input requires gradient
    bool requires_grad = input->requires_grad;

    // Perform forward computation using layer
    boat_tensor_t* output_tensor = boat_conv_layer_forward(layer, input->data);
    if (!output_tensor) {
        return NULL;
    }

    // Create output variable
    boat_variable_t* output_var = boat_variable_create(output_tensor, requires_grad);
    if (!output_var) {
        boat_tensor_unref(output_tensor);
        return NULL;
    }

    // If gradient is required, create operation node and connect to graph
    if (requires_grad) {
        // Create operation node data with layer pointer in extra_data
        boat_op_node_data_t* op_data = create_op_node_data(BOAT_OP_CONV, &input, 1, output_var);
        if (!op_data) {
            boat_variable_free(output_var);
            return NULL;
        }
        // Store layer pointer in extra_data
        op_data->extra_data = layer;

        // Unify variable graphs (only one input)
        boat_graph_t* graph = NULL;
        if (!unify_variable_graphs(&input, 1, &graph)) {
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }

        boat_node_t* op_node = boat_graph_add_node(graph, op_data, BOAT_NODE_TYPE_OPERATION, free_op_node_data);
        if (!op_node) {
            if (graph != input->graph) {
                boat_graph_free(graph);
            }
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }
        output_var->producer_node = op_node;

        // Connect input node to operation node
        if (input->node) {
            boat_graph_add_edge(graph, input->node, op_node, BOAT_EDGE_DIRECTION_FORWARD);
        }

        // Connect operation node to output node
        if (output_var->node) {
            boat_graph_add_edge(graph, op_node, output_var->node, BOAT_EDGE_DIRECTION_FORWARD);
        }

        // Set output variable's graph
        output_var->graph = graph;
    }

    return output_var;
}

static boat_variable_t* create_pool_operation(const boat_variable_t* input, const struct boat_pool_layer_t* layer) {
    if (!input || !layer) return NULL;

    // Check if input requires gradient
    bool requires_grad = input->requires_grad;

    // Perform forward computation using layer
    boat_tensor_t* output_tensor = boat_pool_layer_forward(layer, input->data);
    if (!output_tensor) {
        return NULL;
    }

    // Create output variable
    boat_variable_t* output_var = boat_variable_create(output_tensor, requires_grad);
    if (!output_var) {
        boat_tensor_unref(output_tensor);
        return NULL;
    }

    // If gradient is required, create operation node and connect to graph
    if (requires_grad) {
        // Create operation node data with layer pointer in extra_data
        boat_op_node_data_t* op_data = create_op_node_data(BOAT_OP_POOL, &input, 1, output_var);
        if (!op_data) {
            boat_variable_free(output_var);
            return NULL;
        }
        // Store layer pointer in extra_data
        op_data->extra_data = layer;

        // Unify variable graphs (only one input)
        boat_graph_t* graph = NULL;
        if (!unify_variable_graphs(&input, 1, &graph)) {
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }

        boat_node_t* op_node = boat_graph_add_node(graph, op_data, BOAT_NODE_TYPE_OPERATION, free_op_node_data);
        if (!op_node) {
            if (graph != input->graph) {
                boat_graph_free(graph);
            }
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }
        output_var->producer_node = op_node;

        // Connect input node to operation node
        if (input->node) {
            boat_graph_add_edge(graph, input->node, op_node, BOAT_EDGE_DIRECTION_FORWARD);
        }

        // Connect operation node to output node
        if (output_var->node) {
            boat_graph_add_edge(graph, op_node, output_var->node, BOAT_EDGE_DIRECTION_FORWARD);
        }

        // Set output variable's graph
        output_var->graph = graph;
    }

    return output_var;
}

static boat_variable_t* create_dense_operation(const boat_variable_t* input, const struct boat_dense_layer_t* layer) {
    if (!input || !layer) return NULL;

    // Check if input requires gradient
    bool requires_grad = input->requires_grad;

    // Perform forward computation using layer
    boat_tensor_t* output_tensor = boat_dense_layer_forward(layer, input->data);
    if (!output_tensor) {
        return NULL;
    }

    // Create output variable
    boat_variable_t* output_var = boat_variable_create(output_tensor, requires_grad);
    if (!output_var) {
        boat_tensor_unref(output_tensor);
        return NULL;
    }

    // If gradient is required, create operation node and connect to graph
    if (requires_grad) {
        // Create operation node data with layer pointer in extra_data
        boat_op_node_data_t* op_data = create_op_node_data(BOAT_OP_DENSE, &input, 1, output_var);
        if (!op_data) {
            boat_variable_free(output_var);
            return NULL;
        }
        // Store layer pointer in extra_data
        op_data->extra_data = layer;

        // Unify variable graphs (only one input)
        boat_graph_t* graph = NULL;
        if (!unify_variable_graphs(&input, 1, &graph)) {
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }

        boat_node_t* op_node = boat_graph_add_node(graph, op_data, BOAT_NODE_TYPE_OPERATION, free_op_node_data);
        if (!op_node) {
            if (graph != input->graph) {
                boat_graph_free(graph);
            }
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }
        output_var->producer_node = op_node;

        // Connect input node to operation node
        if (input->node) {
            boat_graph_add_edge(graph, input->node, op_node, BOAT_EDGE_DIRECTION_FORWARD);
        }

        // Connect operation node to output node
        if (output_var->node) {
            boat_graph_add_edge(graph, op_node, output_var->node, BOAT_EDGE_DIRECTION_FORWARD);
        }

        // Set output variable's graph
        output_var->graph = graph;
    }

    return output_var;
}

static void compute_backward_conv(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 1 || !grad_output) {
        return;
    }
    
    boat_variable_t* input = op_data->inputs[0];
    const boat_conv_layer_t* layer = (const boat_conv_layer_t*)op_data->extra_data;
    if (!input || !layer) {
        return;
    }
    
    
    // Call layer backward function to compute gradient with respect to input
    // This will also compute gradients for weight and bias and store them in layer
    boat_tensor_t* grad_input = boat_conv_layer_backward(layer, grad_output);
    if (!grad_input) {
        return;
    }
    
    // If input requires gradient, accumulate gradient
    if (input->requires_grad) {
        if (!input->grad) {
            input->grad = boat_tensor_create_like(input->data);
            if (!input->grad) {
                boat_tensor_unref(grad_input);
                return;
            }
            // Initialize with zeros
            size_t nbytes = boat_tensor_nbytes(input->grad);
            memset(boat_tensor_data(input->grad), 0, nbytes);
        }
        // Accumulate gradient: input->grad += grad_input
        boat_add_(input->grad, grad_input);
    }
    
    boat_tensor_unref(grad_input);
}

// Attention operation with gradient tracking
static boat_variable_t* create_attention_operation(const boat_variable_t* query, const boat_variable_t* key, const boat_variable_t* value, const struct boat_attention_t* attention, const boat_tensor_t* attention_mask) {
    if (!query || !key || !value || !attention) return NULL;

    // Check if any input requires gradient
    bool requires_grad = query->requires_grad || key->requires_grad || value->requires_grad;

    // Perform forward computation using layer
    boat_tensor_t* output_tensor = boat_attention_forward(attention, query->data, key->data, value->data, attention_mask);
    if (!output_tensor) {
        return NULL;
    }

    // Create output variable
    boat_variable_t* output_var = boat_variable_create(output_tensor, requires_grad);
    if (!output_var) {
        boat_tensor_unref(output_tensor);
        return NULL;
    }

    // If gradient is required, create operation node and connect to graph
    if (requires_grad) {
        // Prepare input array
        boat_variable_t* inputs[] = {query, key, value};
        // Create operation node data with layer pointer in extra_data
        boat_op_node_data_t* op_data = create_op_node_data(BOAT_OP_ATTENTION, inputs, 3, output_var);
        if (!op_data) {
            boat_variable_free(output_var);
            return NULL;
        }
        // Store layer pointer in extra_data (attention mask is not stored, as it's not needed for backward)
        op_data->extra_data = attention;

        // Unify variable graphs
        boat_graph_t* graph = NULL;
        if (!unify_variable_graphs(inputs, 3, &graph)) {
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }

        boat_node_t* op_node = boat_graph_add_node(graph, op_data, BOAT_NODE_TYPE_OPERATION, free_op_node_data);
        if (!op_node) {
            if (graph != query->graph) {
                boat_graph_free(graph);
            }
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }
        output_var->producer_node = op_node;

        // Connect input nodes to operation node
        for (size_t i = 0; i < 3; i++) {
            if (inputs[i]->node) {
                boat_graph_add_edge(graph, inputs[i]->node, op_node, BOAT_EDGE_DIRECTION_FORWARD);
            }
        }

        // Connect operation node to output node
        if (output_var->node) {
            boat_graph_add_edge(graph, op_node, output_var->node, BOAT_EDGE_DIRECTION_FORWARD);
        }

        // Set output variable's graph
        output_var->graph = graph;
    }

    return output_var;
}

static void compute_backward_attention(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 3 || !grad_output) {
        return;
    }

    boat_variable_t* query = op_data->inputs[0];
    boat_variable_t* key = op_data->inputs[1];
    boat_variable_t* value = op_data->inputs[2];
    boat_attention_t* attention = (boat_attention_t*)op_data->extra_data;
    if (!query || !key || !value || !attention) {
        return;
    }


    // Call layer backward function to compute gradients with respect to inputs
    boat_tensor_t* grad_query = NULL;
    boat_tensor_t* grad_key = NULL;
    boat_tensor_t* grad_value = NULL;
    bool success = boat_attention_backward(attention, grad_output, &grad_query, &grad_key, &grad_value);
    if (!success || !grad_query || !grad_key || !grad_value) {
        return;
    }

    // Accumulate gradients to input variables if they require gradient
    if (query->requires_grad) {
        if (!query->grad) {
            query->grad = boat_tensor_create_like(query->data);
            if (!query->grad) {
                boat_tensor_unref(grad_query);
                boat_tensor_unref(grad_key);
                boat_tensor_unref(grad_value);
                return;
            }
            // Initialize with zeros
            size_t nbytes = boat_tensor_nbytes(query->grad);
            memset(boat_tensor_data(query->grad), 0, nbytes);
        }
        boat_add_(query->grad, grad_query);
    }

    if (key->requires_grad) {
        if (!key->grad) {
            key->grad = boat_tensor_create_like(key->data);
            if (!key->grad) {
                boat_tensor_unref(grad_query);
                boat_tensor_unref(grad_key);
                boat_tensor_unref(grad_value);
                return;
            }
            size_t nbytes = boat_tensor_nbytes(key->grad);
            memset(boat_tensor_data(key->grad), 0, nbytes);
        }
        boat_add_(key->grad, grad_key);
    }

    if (value->requires_grad) {
        if (!value->grad) {
            value->grad = boat_tensor_create_like(value->data);
            if (!value->grad) {
                boat_tensor_unref(grad_query);
                boat_tensor_unref(grad_key);
                boat_tensor_unref(grad_value);
                return;
            }
            size_t nbytes = boat_tensor_nbytes(value->grad);
            memset(boat_tensor_data(value->grad), 0, nbytes);
        }
        boat_add_(value->grad, grad_value);
    }

    // Clean up temporary gradient tensors
    boat_tensor_unref(grad_query);
    boat_tensor_unref(grad_key);
    boat_tensor_unref(grad_value);
}

// Flatten operation forward pass
static boat_tensor_t* compute_forward_flatten(const boat_tensor_t* input) {
    if (!input) return NULL;

    // Get input shape
    const int64_t* shape = boat_tensor_shape(input);
    size_t ndim = boat_tensor_ndim(input);

    if (ndim < 2) {
        fprintf(stderr, "Error: Flatten expects at least 2D input tensor\n");
        return NULL;
    }

    // Calculate flattened shape: [batch, product of remaining dimensions]
    int64_t batch = shape[0];
    int64_t features = 1;
    for (size_t i = 1; i < ndim; i++) {
        features *= shape[i];
    }

    const int64_t new_shape[] = {batch, features};
    return boat_tensor_reshape(input, new_shape, 2);
}

// Flatten operation backward pass
static void compute_backward_flatten(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 1 || !grad_output) {
        return;
    }

    boat_variable_t* input = op_data->inputs[0];
    if (!input || !input->requires_grad) {
        return;
    }

    // Gradient w.r.t input is just reshaping grad_output back to input shape
    boat_tensor_t* grad_input = boat_tensor_reshape(grad_output,
                                                    boat_tensor_shape(input->data),
                                                    boat_tensor_ndim(input->data));
    if (!grad_input) {
        return;
    }

    // Accumulate gradient
    if (!input->grad) {
        input->grad = boat_tensor_create_like(input->data);
        if (!input->grad) {
            boat_tensor_unref(grad_input);
            return;
        }
    }

    // Add gradient
    boat_add_(input->grad, grad_input);
    boat_tensor_unref(grad_input);
}

// Pooling operation forward pass
static boat_tensor_t* compute_forward_pool(const boat_tensor_t* input, void* layer_ptr) {
    if (!input || !layer_ptr) return NULL;
    boat_pool_layer_t* layer = (boat_pool_layer_t*)layer_ptr;
    return boat_pool_layer_forward(layer, input);
}

// Pooling operation backward pass
static void compute_backward_pool(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 1 || !grad_output) {
        return;
    }

    boat_variable_t* input = op_data->inputs[0];
    boat_pool_layer_t* layer = (boat_pool_layer_t*)op_data->extra_data;
    if (!input || !layer) {
        return;
    }

    // Call layer backward function to compute gradient with respect to input
    boat_tensor_t* grad_input = boat_pool_layer_backward(layer, grad_output);
    if (!grad_input) {
        return;
    }

    // If input requires gradient, accumulate gradient
    if (input->requires_grad) {
        if (!input->grad) {
            input->grad = boat_tensor_create_like(input->data);
            if (!input->grad) {
                boat_tensor_unref(grad_input);
                return;
            }
        }
        // Add gradient
        boat_add_(input->grad, grad_input);
    }

    boat_tensor_unref(grad_input);
}

// Dense operation forward pass
static boat_tensor_t* compute_forward_dense(const boat_tensor_t* input, const void* layer_ptr) {
    if (!input || !layer_ptr) return NULL;
    const boat_dense_layer_t* layer = (const boat_dense_layer_t*)layer_ptr;
    return boat_dense_layer_forward(layer, input);
}

// Dense operation backward pass
static void compute_backward_dense(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 1 || !grad_output) {
        return;
    }

    boat_variable_t* input = op_data->inputs[0];
    const boat_dense_layer_t* layer = (const boat_dense_layer_t*)op_data->extra_data;
    if (!input || !layer) {
        return;
    }

    // Call layer backward function to compute gradient with respect to input
    // This will also compute gradients for weight and bias and store them in layer
    boat_tensor_t* grad_input = boat_dense_layer_backward(layer, grad_output);
    if (!grad_input) {
        return;
    }

    // If input requires gradient, accumulate gradient
    if (input->requires_grad) {
        if (!input->grad) {
            input->grad = boat_tensor_create_like(input->data);
            if (!input->grad) {
                boat_tensor_unref(grad_input);
                return;
            }
        }
        // Add gradient
        boat_add_(input->grad, grad_input);
    }

    boat_tensor_unref(grad_input);
}
