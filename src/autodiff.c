// autodiff.c - Automatic differentiation implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

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
#include <boat.h>

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
    BOAT_OP_MAX,
    BOAT_OP_MIN,
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
    void (*free_extra_data)(void*); // Frees extra_data (NULL = not owned by node)
} boat_op_node_data_t;

// Reduction parameters stored in extra_data for sum/mean/max/min ops.
typedef struct {
    int64_t* dims;      // heap copy of reduction dims (NULL for full reduction)
    size_t n_dims;      // number of dims (0 = full reduction)
    bool keepdim;       // keep reduced dims as size-1 dims
} boat_reduce_params_t;

static void free_reduce_params(void* data) {
    boat_reduce_params_t* p = (boat_reduce_params_t*)data;
    if (!p) return;
    if (p->dims) boat_free(p->dims);
    boat_free(p);
}

// Softmax/log_softmax axis stored in extra_data.
typedef struct {
    int axis;
} softmax_params_t;

static void free_softmax_params(void* data) {
    boat_free(data);
}

// Internal context structure
struct boat_autodiff_context_t {
    bool grad_enabled;
    boat_graph_t* graph;  // Computational graph associated with this context
    bool auto_created;    // true if created implicitly (freed at process exit)
};



// Thread-local current autodiff context
static _Thread_local boat_autodiff_context_t* current_context = NULL;

// Debug counter for tracking function execution
static volatile int debug_counter = 0;

// At-exit cleanup for implicitly-created contexts (see boat_variable_create).
static bool atexit_cleanup_registered = false;
static void boat_autodiff_atexit_cleanup(void);


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
static boat_variable_t* create_reduce_operation(boat_op_type_t op_type, const boat_variable_t* a,
                                                const int64_t* dims, size_t n_dims, bool keepdim);
static boat_variable_t* create_softmax_operation(boat_op_type_t op_type, const boat_variable_t* a, int axis);
static void compute_backward_conv(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_pool(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static boat_tensor_t* compute_forward_flatten(const boat_tensor_t* input);
static void compute_backward_flatten(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_dense(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static boat_variable_t* create_attention_operation(const boat_variable_t* query, const boat_variable_t* key, const boat_variable_t* value, const struct boat_attention_t* attention, const boat_tensor_t* attention_mask);
static void compute_backward_attention(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
// Accumulate grad_output into grad, reducing over leading broadcast
// dimensions when grad is a repeated suffix of grad_output (e.g. a bias of
// shape [out] added to a [batch, out] tensor). Same-shape gradients use a
// direct in-place add.
static void accumulate_grad_broadcast(boat_tensor_t* grad, const boat_tensor_t* grad_output) {
    if (!grad || !grad_output) return;
    if (boat_tensor_dtype(grad) != boat_tensor_dtype(grad_output)) return;

    size_t grad_n = boat_tensor_nelements(grad);
    size_t out_n = boat_tensor_nelements(grad_output);

    if (grad_n == out_n) {
        boat_add_(grad, grad_output);
        return;
    }
    if (grad_n == 0 || out_n % grad_n != 0) {
        return;
    }

    // Require grad's shape (after stripping leading size-1 dims) to be a
    // suffix of grad_output's shape so the element mapping below is a pure
    // broadcast over leading dimensions. This covers [out] and [1,out] grads
    // reduced from [batch,out] while rejecting interior-broadcast shapes like
    // [2,1,3] reduced from [2,4,3], where a flat block-sum would be wrong.
    {
        size_t g_ndim = boat_tensor_ndim(grad);
        size_t o_ndim = boat_tensor_ndim(grad_output);
        const int64_t* g_shape = boat_tensor_shape(grad);
        const int64_t* o_shape = boat_tensor_shape(grad_output);
        size_t g_start = 0;
        while (g_start < g_ndim && g_shape[g_start] == 1) g_start++;
        size_t g_eff = g_ndim - g_start;
        if (g_eff > o_ndim) return;
        for (size_t i = 0; i < g_eff; i++) {
            if (g_shape[g_ndim - 1 - i] != o_shape[o_ndim - 1 - i]) {
                return;
            }
        }
    }

    // grad is a broadcast (repeated suffix) of grad_output: sum the repeated
    // leading blocks into the gradient.
    size_t repeats = out_n / grad_n;
    const void* gd = boat_tensor_const_data(grad_output);
    void* ad = boat_tensor_data(grad);
    switch (boat_tensor_dtype(grad)) {
        case BOAT_DTYPE_FLOAT32: {
            const float* g = (const float*)gd;
            float* a = (float*)ad;
            for (size_t r = 0; r < repeats; r++) {
                for (size_t i = 0; i < grad_n; i++) {
                    a[i] += g[r * grad_n + i];
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* g = (const double*)gd;
            double* a = (double*)ad;
            for (size_t r = 0; r < repeats; r++) {
                for (size_t i = 0; i < grad_n; i++) {
                    a[i] += g[r * grad_n + i];
                }
            }
            break;
        }
        default:
            break;
    }
}

static void compute_backward_add(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 2 || !grad_output) return;

    // Gradient for addition: dL/da = dL/dc, dL/db = dL/dc where c = a + b
    boat_variable_t* a = op_data->inputs[0];
    boat_variable_t* b = op_data->inputs[1];

    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = boat_tensor_create_like(a->data);
        }
        if (a->grad) {
            accumulate_grad_broadcast(a->grad, grad_output);
        }
    }

    if (b->requires_grad) {
        if (!b->grad) {
            b->grad = boat_tensor_create_like(b->data);
        }
        if (b->grad) {
            accumulate_grad_broadcast(b->grad, grad_output);
        }
    }
}
static void compute_backward_sub(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_mul(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_div(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_dot(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_relu(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_sigmoid(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_tanh(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_matmul(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_reduce(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_softmax(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static void compute_backward_log_softmax(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output);
static boat_op_node_data_t* create_op_node_data(boat_op_type_t op_type,
                                                boat_variable_t** inputs,
                                                size_t num_inputs,
                                                const boat_variable_t* output);
static void free_op_node_data(void* data);
static void free_variable_data(void* data);
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
    bool context_auto_created = false;

    if (ctx) {
        graph = boat_autodiff_context_get_graph(ctx);
    } else {
        ctx = boat_autodiff_context_create();
        if (!ctx) {
            boat_tensor_unref(tensor);
            boat_free(var);
            return NULL;
        }
        ((boat_autodiff_context_t*)ctx)->auto_created = true;
        context_auto_created = true;
        boat_autodiff_set_current_context(ctx);
        // Implicitly-created contexts have no owner, so free them at process
        // exit to avoid a leak (explicitly-created contexts are caller-owned).
        if (!atexit_cleanup_registered) {
            atexit_cleanup_registered = true;
            atexit(boat_autodiff_atexit_cleanup);
        }
    }

    // If context has no graph, create one
    if (!graph) {
        graph = boat_graph_create_with_device(boat_tensor_device(tensor));
        if (!graph) {
            // Only tear down a context we created ourselves; never free a
            // caller-provided context here.
            if (context_auto_created) boat_autodiff_context_free(ctx);
            boat_tensor_unref(tensor);
            boat_free(var);
            return NULL;
        }
        boat_autodiff_context_set_graph((boat_autodiff_context_t*)ctx, graph);
    }

    // Always associate variable with graph (even if requires_grad is false)
    var->graph = graph;

    // Create a variable node in the graph if gradient is required
    if (requires_grad) {
        var->node = boat_graph_add_node(var->graph, var, BOAT_NODE_TYPE_VARIABLE, free_variable_data);
        if (!var->node) {
            // Don't free graph, it's owned by context
            boat_tensor_unref(tensor);
            boat_free(var);
            return NULL;
        }
    }

    return var;
}

BOAT_API boat_variable_t* boat_variable_create_with_shape(const int64_t* shape, size_t ndim,
                                                 boat_dtype_t dtype, bool requires_grad) {
    boat_tensor_t* tensor = boat_tensor_create(shape, ndim, dtype, BOAT_DEVICE_CPU);
    if (!tensor) {
        return NULL;
    }

    boat_variable_t* var = boat_variable_create(tensor, requires_grad);
    if (!var) {
        boat_tensor_unref(tensor);
        return NULL;
    }
    // boat_variable_create holds its own reference; release the caller's reference
    boat_tensor_unref(tensor);
    return var;
}

// Internal free: releases the variable's tensors and struct without touching the graph.
// Used by free_variable_data (graph teardown) and by boat_variable_free when the
// variable has no graph node.
static void boat_variable_free_internal(boat_variable_t* var) {
    if (!var) return;

    // Free gradient tensor if exists
    if (var->grad) {
        boat_tensor_unref(var->grad);
    }

    // Free data tensor (variable owns a reference)
    if (var->data) {
        boat_tensor_unref(var->data);
    }

    boat_free(var);
}

BOAT_API void boat_variable_free(const boat_variable_t* variable) {
    if (!variable) return;

    boat_variable_t* var = (boat_variable_t*)variable;
    boat_graph_t* graph = var->graph;
    boat_node_t* var_node = var->node;
    boat_node_t* producer = var->producer_node;

    // Detach from the graph first. Removing the variable node calls
    // free_variable_data, which frees the variable struct via the internal path.
    var->node = NULL;
    var->producer_node = NULL;
    var->graph = NULL;

    if (var_node && graph) {
        boat_graph_safe_remove_node(graph, var_node);
    } else {
        boat_variable_free_internal(var);
    }

    // The producer operation node is dead once its output variable is freed.
    if (producer && graph) {
        boat_graph_safe_remove_node(graph, producer);
    }
}

// Variable properties
BOAT_API boat_tensor_t* boat_variable_data(const boat_variable_t* variable) {
    return variable ? variable->data : NULL;
}

BOAT_API boat_tensor_t* boat_variable_grad(const boat_variable_t* variable) {
    return variable ? variable->grad : NULL;
}

BOAT_API bool boat_variable_requires_grad(const boat_variable_t* variable) {
    return variable ? variable->requires_grad : false;
}

BOAT_API void boat_variable_set_requires_grad(boat_variable_t* variable, bool requires_grad) {
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
        BOAT_DEBUG_PRINT("Warning: Cannot reset data for variable with computation graph node\n");
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
BOAT_API void boat_variable_zero_grad(boat_variable_t* variable) {
    if (!variable) return;

    if (variable->grad) {
        boat_tensor_unref(variable->grad);
        variable->grad = NULL;
    }
}

BOAT_API void boat_variable_retain_grad(const boat_variable_t* variable, bool retain) {
    if (!variable) return;
    (void)retain;
    boat_set_errorf(BOAT_ERROR_NOT_IMPLEMENTED, "[Autodiff] retain_grad not implemented\n");
}

BOAT_API void boat_variable_backward(boat_variable_t* variable, boat_tensor_t* grad_output) {
    setbuf(stderr, NULL);
    BOAT_DEBUG_PRINT("[autodiff] boat_variable_backward: variable=%p, requires_grad=%d, grad_output=%p\n",
            variable, variable ? variable->requires_grad : -1, grad_output);
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
    BOAT_DEBUG_PRINT("[autodiff] boat_variable_backward: op_data=%p, op_type=%d\n", op_data, op_data->op_type);
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
        case BOAT_OP_MEAN:
        case BOAT_OP_MAX:
        case BOAT_OP_MIN:
            compute_backward_reduce(op_data, local_grad);
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
    BOAT_DEBUG_PRINT("[autodiff] boat_variable_backward: recursive loop, num_inputs=%zu\n", op_data->num_inputs);
    for (size_t i = 0; i < op_data->num_inputs; i++) {
        boat_variable_t* input_var = op_data->inputs[i];
        BOAT_DEBUG_PRINT("[autodiff]   input %zu: var=%p, requires_grad=%d, grad=%p, producer_node=%p\n",
                i, input_var, input_var ? input_var->requires_grad : -1, input_var ? input_var->grad : NULL, input_var ? input_var->producer_node : NULL);
        if (input_var && input_var->requires_grad) {
            // Get gradient for this input (should have been computed by compute_backward_*)
            boat_tensor_t* input_grad = input_var->grad;
            if (input_grad) {
                // Only propagate to non-leaf variables (those with producers)
                // Leaf variables already have gradients accumulated by compute_backward_*
                if (input_var->producer_node) {
                    BOAT_DEBUG_PRINT("[autodiff]   calling backward on input %zu (non-leaf)\n", i);
                    // Increase refcount since we're passing it to backward
                    boat_tensor_ref(input_grad);
                    boat_variable_backward(input_var, input_grad);
                    // backward doesn't consume the tensor, so unref
                    boat_tensor_unref(input_grad);
                } else {
                    BOAT_DEBUG_PRINT("[autodiff]   input %zu is leaf, gradient accumulated\n", i);
                }
                // For leaf variables, gradient is already stored in input_var->grad
                // No need to call backward further
            } else {
                BOAT_DEBUG_PRINT("[autodiff]   input %zu has requires_grad but grad is NULL\n", i);
            }
        }
    }

    if (local_grad_allocated) {
        boat_tensor_unref(local_grad);
    }
}

BOAT_API void boat_variable_backward_full(const boat_variable_t* variable) {
    BOAT_DEBUG_PRINT("[autodiff] boat_variable_backward_full: variable=%p, requires_grad=%d\n", variable, variable ? variable->requires_grad : -1);
    if (!variable || !variable->requires_grad) return;

    // For now, just call backward with NULL gradient (scalar loss)
    // In the future, implement full graph traversal
    boat_variable_backward((boat_variable_t*)variable, NULL);
}

// Arithmetic operations with gradient tracking
BOAT_API boat_variable_t* boat_var_add(const boat_variable_t* a, const boat_variable_t* b) {
    if (!a || !b) return NULL;

    boat_variable_t* inputs[] = {(boat_variable_t*)a, (boat_variable_t*)b};
    return create_operation(BOAT_OP_ADD, inputs, 2, compute_forward_add, NULL);
}

BOAT_API boat_variable_t* boat_var_sub(const boat_variable_t* a, const boat_variable_t* b) {
    if (!a || !b) return NULL;

    boat_variable_t* inputs[] = {(boat_variable_t*)a, (boat_variable_t*)b};
    return create_operation(BOAT_OP_SUB, inputs, 2, compute_forward_sub, NULL);
}

BOAT_API boat_variable_t* boat_var_mul(const boat_variable_t* a, const boat_variable_t* b) {
    if (!a || !b) return NULL;

    boat_variable_t* inputs[] = {(boat_variable_t*)a, (boat_variable_t*)b};
    return create_operation(BOAT_OP_MUL, inputs, 2, compute_forward_mul, NULL);
}

BOAT_API boat_variable_t* boat_var_div(const boat_variable_t* a, const boat_variable_t* b) {
    if (!a || !b) return NULL;

    boat_variable_t* inputs[] = {(boat_variable_t*)a, (boat_variable_t*)b};
    return create_operation(BOAT_OP_DIV, inputs, 2, compute_forward_div, NULL);
}

BOAT_API boat_variable_t* boat_var_matmul(const boat_variable_t* a, const boat_variable_t* b) {
    if (!a || !b) return NULL;

    boat_variable_t* inputs[] = {(boat_variable_t*)a, (boat_variable_t*)b};
    return create_operation(BOAT_OP_MATMUL, inputs, 2, compute_forward_matmul, NULL);
}

BOAT_API boat_variable_t* boat_var_dot(const boat_variable_t* a, const boat_variable_t* b) {
    if (!a || !b) return NULL;

    boat_variable_t* inputs[] = {(boat_variable_t*)a, (boat_variable_t*)b};
    return create_operation(BOAT_OP_DOT, inputs, 2, compute_forward_dot, NULL);
}

// Activation functions with gradient tracking
BOAT_API boat_variable_t* boat_var_relu(const boat_variable_t* a) {
    if (!a) return NULL;

    boat_variable_t* inputs[] = {(boat_variable_t*)a};
    return create_operation(BOAT_OP_RELU, inputs, 1, NULL, compute_forward_relu);
}

BOAT_API boat_variable_t* boat_var_sigmoid(const boat_variable_t* a) {
    if (!a) return NULL;

    boat_variable_t* inputs[] = {(boat_variable_t*)a};
    return create_operation(BOAT_OP_SIGMOID, inputs, 1, NULL, compute_forward_sigmoid);
}

BOAT_API boat_variable_t* boat_var_tanh(const boat_variable_t* a) {
    if (!a) return NULL;

    boat_variable_t* inputs[] = {(boat_variable_t*)a};
    return create_operation(BOAT_OP_TANH, inputs, 1, NULL, compute_forward_tanh);
}

BOAT_API boat_variable_t* boat_var_softmax(const boat_variable_t* a, int axis) {
    if (!a) return NULL;
    return create_softmax_operation(BOAT_OP_SOFTMAX, a, axis);
}

BOAT_API boat_variable_t* boat_var_flatten(const boat_variable_t* a) {
    if (!a) return NULL;
    boat_variable_t* inputs[] = {(boat_variable_t*)a};
    return create_operation(BOAT_OP_FLATTEN, inputs, 1, NULL, compute_forward_flatten);
}

BOAT_API boat_variable_t* boat_var_log_softmax(const boat_variable_t* a, int axis) {
    if (!a) return NULL;
    return create_softmax_operation(BOAT_OP_LOG_SOFTMAX, a, axis);
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
BOAT_API boat_variable_t* boat_var_attention(const boat_variable_t* query, const boat_variable_t* key, const boat_variable_t* value, const struct boat_attention_t* attention, const boat_tensor_t* attention_mask) {
    if (!query || !key || !value || !attention) return NULL;
    return create_attention_operation(query, key, value, attention, attention_mask);
}

// Reduction operations with gradient tracking
BOAT_API boat_variable_t* boat_var_sum(const boat_variable_t* a, const int64_t* dims, size_t n_dims, bool keepdim) {
    if (!a) return NULL;
    return create_reduce_operation(BOAT_OP_SUM, a, dims, n_dims, keepdim);
}

BOAT_API boat_variable_t* boat_var_mean(const boat_variable_t* a, int64_t* dims, size_t n_dims, bool keepdim) {
    if (!a) return NULL;
    return create_reduce_operation(BOAT_OP_MEAN, a, dims, n_dims, keepdim);
}

BOAT_API boat_variable_t* boat_var_max(const boat_variable_t* a, int64_t* dims, size_t n_dims, bool keepdim) {
    if (!a) return NULL;
    return create_reduce_operation(BOAT_OP_MAX, a, dims, n_dims, keepdim);
}

BOAT_API boat_variable_t* boat_var_min(const boat_variable_t* a, int64_t* dims, size_t n_dims, bool keepdim) {
    if (!a) return NULL;
    return create_reduce_operation(BOAT_OP_MIN, a, dims, n_dims, keepdim);
}

// Context management
BOAT_API boat_autodiff_context_t* boat_autodiff_context_create() {
    boat_autodiff_context_t* ctx = boat_malloc(sizeof(boat_autodiff_context_t), BOAT_DEVICE_CPU);
    if (!ctx) return NULL;

    ctx->grad_enabled = true;
    ctx->graph = NULL;
    ctx->auto_created = false;
    return ctx;
}

BOAT_API void boat_autodiff_context_free(const boat_autodiff_context_t* context) {
    if (!context) return;

    // The context owns the graph it holds; tear it down (frees remaining nodes).
    if (context->graph) {
        boat_graph_free((boat_graph_t*)context->graph);
    }

    if (current_context == context) {
        current_context = NULL;
    }

    boat_free((void*)context);
}

BOAT_API void boat_autodiff_context_enable_grad(boat_autodiff_context_t* context) {
    if (!context) return;
    context->grad_enabled = true;
}

BOAT_API void boat_autodiff_context_disable_grad(boat_autodiff_context_t* context) {
    if (!context) return;
    context->grad_enabled = false;
}

BOAT_API bool boat_autodiff_context_grad_enabled(const boat_autodiff_context_t* context) {
    return context ? context->grad_enabled : false;
}

BOAT_API void boat_autodiff_context_set_graph(boat_autodiff_context_t* context, const boat_graph_t* graph) {
    if (!context) return;
    context->graph = (boat_graph_t*)graph;
}

BOAT_API boat_graph_t* boat_autodiff_context_get_graph(const boat_autodiff_context_t* context) {
    return context ? context->graph : NULL;
}

BOAT_API void boat_autodiff_set_current_context(const boat_autodiff_context_t* context) {
    current_context = (boat_autodiff_context_t*)context;
}

BOAT_API boat_autodiff_context_t* boat_autodiff_get_current_context() {
    return current_context;
}

// Frees an implicitly-created context (if any) at process exit so it is not
// reported as a leak. Explicitly-created contexts are owned by the caller.
static void boat_autodiff_atexit_cleanup(void) {
    boat_autodiff_context_t* ctx = current_context;
    if (ctx && ctx->auto_created) {
        boat_autodiff_context_free(ctx);
    }
}

// Gradient checkpointing
BOAT_API void boat_autodiff_set_grad_checkpointing(bool enabled) {
    (void)enabled;
    boat_set_errorf(BOAT_ERROR_NOT_IMPLEMENTED, "[Autodiff] gradient checkpointing not implemented\n");
}

BOAT_API void boat_autodiff_clear_computation_graph() {
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

        boat_graph_safe_remove_node((boat_graph_t*)graph, node);
    }

    boat_free(nodes_to_remove);

    // Clear gradients as well
    // (the computation-graph executor was removed; autodiff manages grads itself)
}

// Utility functions
BOAT_API void boat_autodiff_print_graph(const boat_variable_t* variable) {
    if (!variable) return;
    boat_set_errorf(BOAT_ERROR_NOT_IMPLEMENTED, "[Autodiff] graph printing not implemented\n");
}

BOAT_API char* boat_autodiff_graph_to_dot(const boat_variable_t* variable) {
    if (!variable) return NULL;
    boat_set_errorf(BOAT_ERROR_NOT_IMPLEMENTED, "[Autodiff] DOT generation not implemented\n");
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
    op_data->output = (boat_variable_t*)output;

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
    op_data->free_extra_data = NULL;
    return op_data;
}

static void free_op_node_data(void* data) {
    if (!data) return;

    boat_op_node_data_t* op_data = (boat_op_node_data_t*)data;
    if (op_data->inputs) {
        boat_free(op_data->inputs);
    }
    if (op_data->extra_data) {
        if (op_data->free_extra_data) {
            op_data->free_extra_data(op_data->extra_data);
        }
        // else: extra_data is a layer pointer owned by the model (not freed)
    }
    boat_free(op_data);
}

static void free_variable_data(void* data) {
    if (!data) return;
    boat_variable_t* var = (boat_variable_t*)data;
    // The node that owns this data is being destroyed; detach to keep the
    // variable consistent, then free without recursing into the graph.
    var->node = NULL;
    var->producer_node = NULL;
    var->graph = NULL;
    boat_variable_free_internal(var);
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


// Convolution forward computation

// Backward computation functions
static void compute_backward_sub(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 2 || !grad_output) return;

    // Gradient for subtraction: ∂L/∂a = ∂L/∂c, ∂L/∂b = -∂L/∂c
    // where c = a - b
    boat_variable_t* a = op_data->inputs[0];
    boat_variable_t* b = op_data->inputs[1];

    if (a->requires_grad) {
        if (!a->grad) {
            a->grad = boat_tensor_create_like(a->data);
        }
        if (a->grad) {
            accumulate_grad_broadcast(a->grad, grad_output);
        }
    }

    if (b->requires_grad) {
        // Gradient for b is negative of grad_output
        boat_tensor_t* neg_grad_output = boat_mul_scalar(grad_output, -1.0);
        if (!neg_grad_output) return;

        if (!b->grad) {
            b->grad = boat_tensor_create_like(b->data);
        }
        if (b->grad) {
            accumulate_grad_broadcast(b->grad, neg_grad_output);
        }
        boat_tensor_unref(neg_grad_output);
    }
}

static void compute_backward_mul(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 2 || !grad_output) return;

    // Gradient for multiplication: dL/da = dL/dc * b, dL/db = dL/dc * a
    boat_variable_t* a = op_data->inputs[0];
    boat_variable_t* b = op_data->inputs[1];

    if (a->requires_grad) {
        // grad_output * b has the broadcast shape; reduce it down to a's shape
        // when a is broadcast (e.g. a bias [out] multiplied into [batch, out]).
        boat_tensor_t* grad_a = boat_mul(grad_output, b->data);
        if (grad_a) {
            if (!a->grad) {
                a->grad = boat_tensor_create_like(a->data);
            }
            if (a->grad) {
                accumulate_grad_broadcast(a->grad, grad_a);
            }
            boat_tensor_unref(grad_a);
        }
    }

    if (b->requires_grad) {
        boat_tensor_t* grad_b = boat_mul(grad_output, a->data);
        if (grad_b) {
            if (!b->grad) {
                b->grad = boat_tensor_create_like(b->data);
            }
            if (b->grad) {
                accumulate_grad_broadcast(b->grad, grad_b);
            }
            boat_tensor_unref(grad_b);
        }
    }
}

static void compute_backward_div(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 2 || !grad_output) return;

    // Gradient for division: dL/da = dL/dc / b, dL/db = -dL/dc * a / b^2
    boat_variable_t* a = op_data->inputs[0];
    boat_variable_t* b = op_data->inputs[1];

    if (a->requires_grad) {
        boat_tensor_t* grad_a = boat_div(grad_output, b->data);
        if (grad_a) {
            if (!a->grad) {
                a->grad = boat_tensor_create_like(a->data);
            }
            if (a->grad) {
                accumulate_grad_broadcast(a->grad, grad_a);
            }
            boat_tensor_unref(grad_a);
        }
    }

    if (b->requires_grad) {
        // dL/db = -dL/dc * a / b^2
        boat_tensor_t* b_squared = boat_mul(b->data, b->data);
        if (!b_squared) return;

        boat_tensor_t* a_div_bsq = boat_div(a->data, b_squared);
        boat_tensor_unref(b_squared);
        if (!a_div_bsq) return;

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
                b->grad = boat_tensor_create_like(b->data);
            }
            if (b->grad) {
                accumulate_grad_broadcast(b->grad, grad_b);
            }
            boat_tensor_unref(grad_b);
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

// Reduce the leading batch dims of `t` (shape = full batch + matrix) that were
// broadcast from a size-1 dim in `target_shape` (right-aligned). Returns a new
// tensor with the target batch shape, or `t` itself if no reduction is needed.
static boat_tensor_t* reduce_broadcast_batch(boat_tensor_t* t, const int64_t* target_shape,
                                             const int64_t* full_shape, size_t full_bd,
                                             size_t target_bd) {
    int64_t reduce_dims[4];
    size_t n_reduce = 0;
    size_t off = full_bd - target_bd;  // target is right-aligned to full
    for (size_t i = 0; i < full_bd; i++) {
        int64_t tdim = (i >= off) ? target_shape[i - off] : 1;
        if (tdim == 1 && full_shape[i] > 1) {
            reduce_dims[n_reduce++] = (int64_t)i;
        }
    }
    if (n_reduce == 0) return t;
    boat_tensor_t* result = boat_sum(t, reduce_dims, n_reduce, false);
    boat_tensor_unref(t);
    return result;
}

static void compute_backward_matmul(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 2 || !grad_output) return;

    boat_variable_t* a = op_data->inputs[0];
    boat_variable_t* b = op_data->inputs[1];

    size_t a_ndim = boat_tensor_ndim(a->data);
    size_t b_ndim = boat_tensor_ndim(b->data);
    const int64_t* a_shape = boat_tensor_shape(a->data);
    const int64_t* b_shape = boat_tensor_shape(b->data);
    const int64_t* out_shape = boat_tensor_shape(grad_output);
    size_t out_bd = boat_tensor_ndim(grad_output) - 2;

    if (a->requires_grad) {
        // grad_a = (grad_output @ B^T), summed over dims where A was broadcast.
        boat_tensor_t* b_T = boat_transpose(b->data, (int)b_ndim - 2, (int)b_ndim - 1);
        if (b_T) {
            boat_tensor_t* grad_a_full = boat_matmul(grad_output, b_T);
            boat_tensor_unref(b_T);
            if (grad_a_full) {
                boat_tensor_t* grad_a = reduce_broadcast_batch(
                    grad_a_full, a_shape, out_shape, out_bd, a_ndim - 2);
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
    }

    if (b->requires_grad) {
        // grad_b = (A^T @ grad_output), summed over dims where B was broadcast.
        boat_tensor_t* a_T = boat_transpose(a->data, (int)a_ndim - 2, (int)a_ndim - 1);
        if (a_T) {
            boat_tensor_t* grad_b_full = boat_matmul(a_T, grad_output);
            boat_tensor_unref(a_T);
            if (grad_b_full) {
                boat_tensor_t* grad_b = reduce_broadcast_batch(
                    grad_b_full, b_shape, out_shape, out_bd, b_ndim - 2);
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
}
static void compute_backward_reduce(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 1 || !grad_output) return;
    boat_variable_t* a = op_data->inputs[0];
    if (!a->requires_grad) return;

    boat_reduce_params_t* params = (boat_reduce_params_t*)op_data->extra_data;
    if (!params) return;
    boat_tensor_t* input = a->data;
    boat_dtype_t dtype = boat_tensor_dtype(input);
    if (dtype != BOAT_DTYPE_FLOAT32 && dtype != BOAT_DTYPE_FLOAT64) return;
    bool is_f64 = (dtype == BOAT_DTYPE_FLOAT64);

    size_t ndim = boat_tensor_ndim(input);
    const int64_t* shape = boat_tensor_shape(input);

    if (ndim == 0) {
        // Reducing a scalar is identity; gradient passes straight through.
        if (!a->grad) a->grad = boat_tensor_create_like(input);
        if (a->grad) boat_add_(a->grad, grad_output);
        return;
    }

    // Derive reduction kind from the op type (0=sum, 1=mean, 2=max, 3=min).
    int kind;
    switch (op_data->op_type) {
        case BOAT_OP_SUM: kind = 0; break;
        case BOAT_OP_MEAN: kind = 1; break;
        case BOAT_OP_MAX: kind = 2; break;
        case BOAT_OP_MIN: kind = 3; break;
        default: return;
    }

    // Normalize reduction dims.
    bool reduced[BOAT_MAX_DIMS];
    for (size_t i = 0; i < ndim; i++) reduced[i] = false;
    if (params->dims == NULL || params->n_dims == 0) {
        for (size_t i = 0; i < ndim; i++) reduced[i] = true;
    } else {
        for (size_t d = 0; d < params->n_dims; d++) {
            int64_t dim = params->dims[d];
            if (dim < 0) dim += (int64_t)ndim;
            if (dim < 0 || dim >= (int64_t)ndim) return;
            reduced[(size_t)dim] = true;
        }
    }

    // Input row-major strides.
    size_t in_stride[BOAT_MAX_DIMS];
    in_stride[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; i--) in_stride[i] = in_stride[i + 1] * (size_t)shape[i + 1];

    // Reduced dims sizes/strides.
    size_t red_sizes[BOAT_MAX_DIMS];
    size_t red_strides[BOAT_MAX_DIMS];
    size_t n_red = 0, red_total = 1;
    for (size_t i = 0; i < ndim; i++) {
        if (reduced[i]) {
            red_sizes[n_red] = (size_t)shape[i];
            red_strides[n_red] = in_stride[i];
            red_total *= (size_t)shape[i];
            n_red++;
        }
    }

    // Output shape/mapping/strides.
    int64_t out_shape[BOAT_MAX_DIMS];
    size_t out_to_in[BOAT_MAX_DIMS];
    size_t out_ndim = 0;
    for (size_t i = 0; i < ndim; i++) {
        if (!reduced[i]) { out_to_in[out_ndim] = i; out_shape[out_ndim++] = shape[i]; }
        else if (params->keepdim) { out_to_in[out_ndim] = i; out_shape[out_ndim++] = 1; }
    }
    size_t out_stride[BOAT_MAX_DIMS];
    size_t out_nelements = 1;
    if (out_ndim > 0) {
        out_stride[out_ndim - 1] = 1;
        for (int i = (int)out_ndim - 2; i >= 0; i--) out_stride[i] = out_stride[i + 1] * (size_t)out_shape[i + 1];
        for (size_t i = 0; i < out_ndim; i++) out_nelements *= (size_t)out_shape[i];
    }

    if (!a->grad) {
        a->grad = boat_tensor_create_like(input);
        if (!a->grad) return;
    }
    float* g_f = is_f64 ? NULL : (float*)boat_tensor_data(a->grad);
    double* g_d = is_f64 ? (double*)boat_tensor_data(a->grad) : NULL;
    const float* go_f = is_f64 ? NULL : (const float*)boat_tensor_const_data(grad_output);
    const double* go_d = is_f64 ? (const double*)boat_tensor_const_data(grad_output) : NULL;
    const float* in_f = is_f64 ? NULL : (const float*)boat_tensor_const_data(input);
    const double* in_d = is_f64 ? (const double*)boat_tensor_const_data(input) : NULL;

    for (size_t oi = 0; oi < out_nelements; oi++) {
        size_t rem = oi, base_off = 0;
        for (size_t d = 0; d < out_ndim; d++) {
            size_t coord = rem / out_stride[d];
            rem %= out_stride[d];
            base_off += coord * in_stride[out_to_in[d]];
        }
        double go_val = is_f64 ? go_d[oi] : (double)go_f[oi];
        if (kind == 1 && red_total > 0) go_val /= (double)red_total;

        if (kind == 0 || kind == 1) {
            // sum/mean: route to every element in the reduced region.
            for (size_t r = 0; r < red_total; r++) {
                size_t rr = r, red_off = 0;
                for (size_t d = 0; d < n_red; d++) {
                    size_t coord = rr % red_sizes[d];
                    rr /= red_sizes[d];
                    red_off += coord * red_strides[d];
                }
                size_t off = base_off + red_off;
                if (is_f64) g_d[off] += go_val;
                else g_f[off] += (float)go_val;
            }
        } else {
            // max/min: route only to the argmax/argmin element.
            size_t best_off = base_off;
            double best = is_f64 ? in_d[base_off] : (double)in_f[base_off];
            for (size_t r = 1; r < red_total; r++) {
                size_t rr = r, red_off = 0;
                for (size_t d = 0; d < n_red; d++) {
                    size_t coord = rr % red_sizes[d];
                    rr /= red_sizes[d];
                    red_off += coord * red_strides[d];
                }
                double v = is_f64 ? in_d[base_off + red_off] : (double)in_f[base_off + red_off];
                if ((kind == 2 && v > best) || (kind == 3 && v < best)) {
                    best = v;
                    best_off = base_off + red_off;
                }
            }
            if (is_f64) g_d[best_off] += go_val;
            else g_f[best_off] += (float)go_val;
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
    for (size_t i = 0; i < num_inputs; i++) {
        if (inputs[i] && inputs[i]->graph) {
            target = inputs[i]->graph;
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
static void compute_backward_softmax(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 1 || !grad_output || !op_data->output) return;

    // Gradient for softmax: dL/dx_i = y_i * (dL/dy_i - sum_k(y_k * dL/dy_k)),
    // computed per softmax slice (along the stored axis).
    boat_variable_t* a = op_data->inputs[0];
    const boat_variable_t* y_var = op_data->output;
    if (!a->requires_grad) return;

    softmax_params_t* params = (softmax_params_t*)op_data->extra_data;
    const boat_tensor_t* y = y_var->data;
    size_t nelements = boat_tensor_nelements(y);
    if (nelements == 0) return;
    boat_dtype_t dtype = boat_tensor_dtype(y);

    const int64_t* shape = boat_tensor_shape(y);
    size_t ndim = boat_tensor_ndim(y);

    int axis = params ? params->axis : -1;
    if (axis < 0) axis += (int)ndim;
    if (axis < 0 || (size_t)axis >= ndim) return;

    size_t axis_size = (size_t)shape[axis];
    size_t outer_elements = 1;
    for (size_t i = 0; i < (size_t)axis; i++) outer_elements *= (size_t)shape[i];
    size_t inner_stride = 1;
    for (size_t i = (size_t)axis + 1; i < ndim; i++) inner_stride *= (size_t)shape[i];

    boat_tensor_t* grad = boat_tensor_create_like(y);
    if (!grad) return;
    void* grad_data = boat_tensor_data(grad);
    const void* y_data = boat_tensor_data(y);
    const void* grad_output_data = boat_tensor_data(grad_output);

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* y_ptr = (const float*)y_data;
            const float* go_ptr = (const float*)grad_output_data;
            float* g_ptr = (float*)grad_data;
            for (size_t outer = 0; outer < outer_elements; outer++) {
                for (size_t inner = 0; inner < inner_stride; inner++) {
                    size_t base = outer * axis_size * inner_stride + inner;
                    float sum_y_grad = 0.0f;
                    for (size_t k = 0; k < axis_size; k++) {
                        size_t idx = base + k * inner_stride;
                        sum_y_grad += y_ptr[idx] * go_ptr[idx];
                    }
                    for (size_t k = 0; k < axis_size; k++) {
                        size_t idx = base + k * inner_stride;
                        g_ptr[idx] = y_ptr[idx] * (go_ptr[idx] - sum_y_grad);
                    }
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* y_ptr = (const double*)y_data;
            const double* go_ptr = (const double*)grad_output_data;
            double* g_ptr = (double*)grad_data;
            for (size_t outer = 0; outer < outer_elements; outer++) {
                for (size_t inner = 0; inner < inner_stride; inner++) {
                    size_t base = outer * axis_size * inner_stride + inner;
                    double sum_y_grad = 0.0;
                    for (size_t k = 0; k < axis_size; k++) {
                        size_t idx = base + k * inner_stride;
                        sum_y_grad += y_ptr[idx] * go_ptr[idx];
                    }
                    for (size_t k = 0; k < axis_size; k++) {
                        size_t idx = base + k * inner_stride;
                        g_ptr[idx] = y_ptr[idx] * (go_ptr[idx] - sum_y_grad);
                    }
                }
            }
            break;
        }
        default:
            boat_tensor_unref(grad);
            return;
    }

    if (!a->grad) {
        a->grad = grad;
    } else {
        boat_add_(a->grad, grad);
        boat_tensor_unref(grad);
    }
}

static void compute_backward_log_softmax(boat_op_node_data_t* op_data, const boat_tensor_t* grad_output) {
    if (!op_data || op_data->num_inputs != 1 || !grad_output || !op_data->output) return;

    // Gradient for log_softmax: dL/dx_i = dL/dy_i - exp(y_i) * sum_k(dL/dy_k),
    // computed per slice (along the stored axis).
    boat_variable_t* a = op_data->inputs[0];
    const boat_variable_t* y_var = op_data->output;
    if (!a->requires_grad) return;

    softmax_params_t* params = (softmax_params_t*)op_data->extra_data;
    const boat_tensor_t* y = y_var->data;
    size_t nelements = boat_tensor_nelements(y);
    if (nelements == 0) return;
    boat_dtype_t dtype = boat_tensor_dtype(y);

    const int64_t* shape = boat_tensor_shape(y);
    size_t ndim = boat_tensor_ndim(y);

    int axis = params ? params->axis : -1;
    if (axis < 0) axis += (int)ndim;
    if (axis < 0 || (size_t)axis >= ndim) return;

    size_t axis_size = (size_t)shape[axis];
    size_t outer_elements = 1;
    for (size_t i = 0; i < (size_t)axis; i++) outer_elements *= (size_t)shape[i];
    size_t inner_stride = 1;
    for (size_t i = (size_t)axis + 1; i < ndim; i++) inner_stride *= (size_t)shape[i];

    boat_tensor_t* grad = boat_tensor_create_like(y);
    if (!grad) return;
    void* grad_data = boat_tensor_data(grad);
    const void* y_data = boat_tensor_data(y);
    const void* grad_output_data = boat_tensor_data(grad_output);

    switch (dtype) {
        case BOAT_DTYPE_FLOAT32: {
            const float* y_ptr = (const float*)y_data;
            const float* go_ptr = (const float*)grad_output_data;
            float* g_ptr = (float*)grad_data;
            for (size_t outer = 0; outer < outer_elements; outer++) {
                for (size_t inner = 0; inner < inner_stride; inner++) {
                    size_t base = outer * axis_size * inner_stride + inner;
                    float sum_grad = 0.0f;
                    for (size_t k = 0; k < axis_size; k++) {
                        sum_grad += go_ptr[base + k * inner_stride];
                    }
                    for (size_t k = 0; k < axis_size; k++) {
                        size_t idx = base + k * inner_stride;
                        g_ptr[idx] = go_ptr[idx] - expf(y_ptr[idx]) * sum_grad;
                    }
                }
            }
            break;
        }
        case BOAT_DTYPE_FLOAT64: {
            const double* y_ptr = (const double*)y_data;
            const double* go_ptr = (const double*)grad_output_data;
            double* g_ptr = (double*)grad_data;
            for (size_t outer = 0; outer < outer_elements; outer++) {
                for (size_t inner = 0; inner < inner_stride; inner++) {
                    size_t base = outer * axis_size * inner_stride + inner;
                    double sum_grad = 0.0;
                    for (size_t k = 0; k < axis_size; k++) {
                        sum_grad += go_ptr[base + k * inner_stride];
                    }
                    for (size_t k = 0; k < axis_size; k++) {
                        size_t idx = base + k * inner_stride;
                        g_ptr[idx] = go_ptr[idx] - exp(y_ptr[idx]) * sum_grad;
                    }
                }
            }
            break;
        }
        default:
            boat_tensor_unref(grad);
            return;
    }

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
    // boat_variable_create holds its own reference; release the caller's reference
    boat_tensor_unref(output_tensor);

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

// Create a reduction operation (sum/mean/max/min) with axis support. The
// reduction parameters are stored in op_data->extra_data for the backward pass.
static boat_variable_t* create_reduce_operation(boat_op_type_t op_type, const boat_variable_t* a,
                                                const int64_t* dims, size_t n_dims, bool keepdim) {
    if (!a) return NULL;

    boat_tensor_t* output_tensor = NULL;
    switch (op_type) {
        case BOAT_OP_SUM: output_tensor = boat_sum(a->data, dims, n_dims, keepdim); break;
        case BOAT_OP_MEAN: output_tensor = boat_mean(a->data, dims, n_dims, keepdim); break;
        case BOAT_OP_MAX: output_tensor = boat_max(a->data, dims, n_dims, keepdim); break;
        case BOAT_OP_MIN: output_tensor = boat_min(a->data, dims, n_dims, keepdim); break;
        default: return NULL;
    }
    if (!output_tensor) return NULL;

    boat_variable_t* output_var = boat_variable_create(output_tensor, a->requires_grad);
    boat_tensor_unref(output_tensor);
    if (!output_var) return NULL;

    if (a->requires_grad) {
        boat_variable_t* inputs[] = {(boat_variable_t*)a};
        boat_op_node_data_t* op_data = create_op_node_data(op_type, inputs, 1, output_var);
        if (!op_data) { boat_variable_free(output_var); return NULL; }

        // Copy the reduction parameters so the backward pass can use them.
        boat_reduce_params_t* params = boat_malloc(sizeof(boat_reduce_params_t), BOAT_DEVICE_CPU);
        if (!params) { free_op_node_data(op_data); boat_variable_free(output_var); return NULL; }
        params->n_dims = n_dims;
        params->keepdim = keepdim;
        params->dims = NULL;
        if (dims && n_dims > 0) {
            params->dims = boat_malloc(sizeof(int64_t) * n_dims, BOAT_DEVICE_CPU);
            if (!params->dims) {
                boat_free(params);
                free_op_node_data(op_data);
                boat_variable_free(output_var);
                return NULL;
            }
            memcpy(params->dims, dims, sizeof(int64_t) * n_dims);
        }
        op_data->extra_data = params;
        op_data->free_extra_data = free_reduce_params;

        boat_graph_t* graph = NULL;
        if (!unify_variable_graphs(inputs, 1, &graph)) {
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }

        boat_node_t* op_node = boat_graph_add_node(graph, op_data, BOAT_NODE_TYPE_OPERATION, free_op_node_data);
        if (!op_node) {
            if (graph != a->graph) boat_graph_free(graph);
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }
        output_var->producer_node = op_node;

        if (a->node) {
            boat_graph_add_edge(graph, a->node, op_node, BOAT_EDGE_DIRECTION_FORWARD);
        }
        if (output_var->node) {
            boat_graph_add_edge(graph, op_node, output_var->node, BOAT_EDGE_DIRECTION_FORWARD);
        }
        output_var->graph = graph;
    }

    return output_var;
}

// Create a softmax/log_softmax operation along the given axis. The axis is
// stored in op_data->extra_data for the backward pass.
static boat_variable_t* create_softmax_operation(boat_op_type_t op_type, const boat_variable_t* a, int axis) {
    if (!a) return NULL;

    boat_tensor_t* output_tensor = (op_type == BOAT_OP_SOFTMAX)
        ? boat_softmax(a->data, axis) : boat_log_softmax(a->data, axis);
    if (!output_tensor) return NULL;

    boat_variable_t* output_var = boat_variable_create(output_tensor, a->requires_grad);
    boat_tensor_unref(output_tensor);
    if (!output_var) return NULL;

    if (a->requires_grad) {
        boat_variable_t* inputs[] = {(boat_variable_t*)a};
        boat_op_node_data_t* op_data = create_op_node_data(op_type, inputs, 1, output_var);
        if (!op_data) { boat_variable_free(output_var); return NULL; }

        softmax_params_t* params = boat_malloc(sizeof(softmax_params_t), BOAT_DEVICE_CPU);
        if (!params) { free_op_node_data(op_data); boat_variable_free(output_var); return NULL; }
        params->axis = axis;
        op_data->extra_data = params;
        op_data->free_extra_data = free_softmax_params;

        boat_graph_t* graph = NULL;
        if (!unify_variable_graphs(inputs, 1, &graph)) {
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }

        boat_node_t* op_node = boat_graph_add_node(graph, op_data, BOAT_NODE_TYPE_OPERATION, free_op_node_data);
        if (!op_node) {
            if (graph != a->graph) boat_graph_free(graph);
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }
        output_var->producer_node = op_node;

        if (a->node) {
            boat_graph_add_edge(graph, a->node, op_node, BOAT_EDGE_DIRECTION_FORWARD);
        }
        if (output_var->node) {
            boat_graph_add_edge(graph, op_node, output_var->node, BOAT_EDGE_DIRECTION_FORWARD);
        }
        output_var->graph = graph;
    }

    return output_var;
}
static boat_variable_t* create_conv_operation(const boat_variable_t* input, const struct boat_conv_layer_t* layer) {
    if (!input || !layer) return NULL;

    // Check if input requires gradient
    bool requires_grad = input->requires_grad;
    // Convolution layers always have trainable parameters (weight and bias)
    bool layer_has_params = true;
    // Output requires gradient if either input requires gradient or layer has parameters
    bool output_requires_grad = requires_grad || layer_has_params;
    BOAT_DEBUG_PRINT("[autodiff] create_conv_operation: input=%p, layer=%p, requires_grad=%d, layer_has_params=%d, output_requires_grad=%d\n", input, layer, requires_grad, layer_has_params, output_requires_grad);

    // Perform forward computation using layer
    boat_tensor_t* output_tensor = boat_conv_layer_forward((boat_conv_layer_t*)layer, input->data);
    BOAT_DEBUG_PRINT("[autodiff] conv forward: output_tensor=%p\n", output_tensor);
    if (!output_tensor) {
        return NULL;
    }

    // Create output variable
    boat_variable_t* output_var = boat_variable_create(output_tensor, output_requires_grad);
    BOAT_DEBUG_PRINT("[autodiff] create_conv_operation: output_var=%p, output_requires_grad=%d\n", output_var, output_requires_grad);
    if (!output_var) {
        boat_tensor_unref(output_tensor);
        return NULL;
    }
    // boat_variable_create holds its own reference; release the caller's reference
    boat_tensor_unref(output_tensor);

    // If gradient is required, create operation node and connect to graph
    if (output_requires_grad) {
        // Create operation node data with layer pointer in extra_data
        boat_op_node_data_t* op_data = create_op_node_data(BOAT_OP_CONV, (boat_variable_t**)&input, 1, output_var);
        if (!op_data) {
            boat_variable_free(output_var);
            return NULL;
        }
        // Store layer pointer in extra_data
        op_data->extra_data = (void*)layer;

        // Unify variable graphs (only one input)
        boat_graph_t* graph = NULL;
        if (!unify_variable_graphs((boat_variable_t**)&input, 1, &graph)) {
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }
        BOAT_DEBUG_PRINT("[autodiff] unify succeeded, graph=%p\n", graph);

        boat_node_t* op_node = boat_graph_add_node(graph, op_data, BOAT_NODE_TYPE_OPERATION, free_op_node_data);
        BOAT_DEBUG_PRINT("[autodiff] op_node=%p, op_type=%d\n", op_node, op_data->op_type);
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
    boat_tensor_t* output_tensor = boat_pool_layer_forward((boat_pool_layer_t*)layer, input->data);
    if (!output_tensor) {
        return NULL;
    }

    // Create output variable
    boat_variable_t* output_var = boat_variable_create(output_tensor, requires_grad);
    if (!output_var) {
        boat_tensor_unref(output_tensor);
        return NULL;
    }
    // boat_variable_create holds its own reference; release the caller's reference
    boat_tensor_unref(output_tensor);

    // If gradient is required, create operation node and connect to graph
    if (requires_grad) {
        // Create operation node data with layer pointer in extra_data
        boat_op_node_data_t* op_data = create_op_node_data(BOAT_OP_POOL, (boat_variable_t**)&input, 1, output_var);
        if (!op_data) {
            boat_variable_free(output_var);
            return NULL;
        }
        // Store layer pointer in extra_data
        op_data->extra_data = (void*)layer;

        // Unify variable graphs (only one input)
        boat_graph_t* graph = NULL;
        if (!unify_variable_graphs((boat_variable_t**)&input, 1, &graph)) {
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }
        BOAT_DEBUG_PRINT("[autodiff] unify succeeded, graph=%p\n", graph);

        boat_node_t* op_node = boat_graph_add_node(graph, op_data, BOAT_NODE_TYPE_OPERATION, free_op_node_data);
        BOAT_DEBUG_PRINT("[autodiff] op_node=%p, op_type=%d\n", op_node, op_data->op_type);
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
    // Dense layers always have trainable parameters (weight and bias)
    bool layer_has_params = true;
    // Output requires gradient if either input requires gradient or layer has parameters
    bool output_requires_grad = requires_grad || layer_has_params;

    // Perform forward computation using layer
    boat_tensor_t* output_tensor = boat_dense_layer_forward((boat_dense_layer_t*)layer, input->data);
    if (!output_tensor) {
        return NULL;
    }

    // Create output variable
    boat_variable_t* output_var = boat_variable_create(output_tensor, output_requires_grad);
    if (!output_var) {
        boat_tensor_unref(output_tensor);
        return NULL;
    }
    // boat_variable_create holds its own reference; release the caller's reference
    boat_tensor_unref(output_tensor);

    // If gradient is required, create operation node and connect to graph
    if (output_requires_grad) {
        // Create operation node data with layer pointer in extra_data
        boat_op_node_data_t* op_data = create_op_node_data(BOAT_OP_DENSE, (boat_variable_t**)&input, 1, output_var);
        if (!op_data) {
            boat_variable_free(output_var);
            return NULL;
        }
        // Store layer pointer in extra_data
        op_data->extra_data = (void*)layer;

        // Unify variable graphs (only one input)
        boat_graph_t* graph = NULL;
        if (!unify_variable_graphs((boat_variable_t**)&input, 1, &graph)) {
            free_op_node_data(op_data);
            boat_variable_free(output_var);
            return NULL;
        }
        BOAT_DEBUG_PRINT("[autodiff] unify succeeded, graph=%p\n", graph);

        boat_node_t* op_node = boat_graph_add_node(graph, op_data, BOAT_NODE_TYPE_OPERATION, free_op_node_data);
        BOAT_DEBUG_PRINT("[autodiff] op_node=%p, op_type=%d\n", op_node, op_data->op_type);
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
    BOAT_DEBUG_PRINT("[autodiff] compute_backward_conv called, op_data=%p, grad_output=%p\n", op_data, grad_output);
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
    BOAT_DEBUG_PRINT("[autodiff compute_backward_conv] calling boat_conv_layer_backward, layer=%p, grad_output=%p\n", layer, grad_output);
    boat_tensor_t* grad_input = boat_conv_layer_backward((boat_conv_layer_t*)layer, grad_output);
    BOAT_DEBUG_PRINT("[autodiff compute_backward_conv] boat_conv_layer_backward returned grad_input=%p\n", grad_input);
    if (!grad_input) {
        BOAT_DEBUG_PRINT("[autodiff compute_backward_conv] grad_input is NULL, returning\n");
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
    boat_tensor_t* output_tensor = boat_attention_forward((boat_attention_t*)attention, query->data, key->data, value->data, attention_mask);
    if (!output_tensor) {
        return NULL;
    }

    // Create output variable
    boat_variable_t* output_var = boat_variable_create(output_tensor, requires_grad);
    if (!output_var) {
        boat_tensor_unref(output_tensor);
        return NULL;
    }
    // boat_variable_create holds its own reference; release the caller's reference
    boat_tensor_unref(output_tensor);

    // If gradient is required, create operation node and connect to graph
    if (requires_grad) {
        // Prepare input array
        boat_variable_t* inputs[] = {(boat_variable_t*)query, (boat_variable_t*)key, (boat_variable_t*)value};
        // Create operation node data with layer pointer in extra_data
        boat_op_node_data_t* op_data = create_op_node_data(BOAT_OP_ATTENTION, inputs, 3, output_var);
        if (!op_data) {
            boat_variable_free(output_var);
            return NULL;
        }
        // Store layer pointer in extra_data (attention mask is not stored, as it's not needed for backward)
        op_data->extra_data = (void*)attention;

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
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Autodiff] Flatten expects at least 2D input tensor\n");
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
    boat_tensor_t* grad_input = boat_dense_layer_backward((boat_dense_layer_t*)layer, grad_output);
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
