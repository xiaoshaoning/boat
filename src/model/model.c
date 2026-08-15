// model.c - Model definition and serialization
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat.h>
#include <boat/model.h>
#include <boat/memory.h>
#include <boat/tensor.h>
#include <boat/graph.h>
#include <boat/layers/norm.h>
#include <boat/layers/attention.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Model structure
struct boat_model_t {
    boat_graph_t* graph;           // Computational graph representing the model
    char* name;                    // Model name (optional)
    boat_device_t device;          // Device where model is stored
    bool trainable;                // Whether model is in training mode
    void* user_data;               // User-defined data (optional)
    void (*free_user_data)(void*); // Function to free user_data
    // Layers storage (temporary until integrated with graph)
    boat_layer_t** layers; // Array of layer pointers
    boat_node_t** nodes;   // Array of graph node pointers (parallel to layers)
    size_t layer_count;    // Number of layers
    size_t layer_capacity; // Capacity of layers array
};

// Layer structure is now defined in model.h

// Model creation and management
BOAT_API boat_model_t* boat_model_create() {
    boat_model_t* model = boat_malloc(sizeof(boat_model_t), BOAT_DEVICE_CPU);
    if (!model) {
        return NULL;
    }

    // Create a new computational graph for the model
    model->graph = boat_graph_create();
    if (!model->graph) {
        boat_free(model);
        return NULL;
    }

    model->name = NULL;
    model->device = BOAT_DEVICE_CPU;
    model->trainable = true;
    model->user_data = NULL;
    model->free_user_data = NULL;

    // Initialize layers storage
    model->layers = NULL;
    model->nodes = NULL;
    model->layer_count = 0;
    model->layer_capacity = 0;

    return model;
}

BOAT_API boat_model_t* boat_model_create_with_graph(const boat_graph_t* graph) {
    if (!graph) {
        return NULL;
    }

    boat_model_t* model = boat_malloc(sizeof(boat_model_t), BOAT_DEVICE_CPU);
    if (!model) {
        return NULL;
    }

    model->graph = (boat_graph_t*)graph;
    // Note: graph ownership is transferred to model
    model->name = NULL;
    model->device = boat_graph_device(graph);
    model->trainable = true;
    model->user_data = NULL;
    model->free_user_data = NULL;

    // Initialize layers storage
    model->layers = NULL;
    model->nodes = NULL;
    model->layer_count = 0;
    model->layer_capacity = 0;

    return model;
}

BOAT_API void boat_model_free(boat_model_t* model) {
    if (!model) {
        return;
    }

    // Free layers
    if (model->layers) {
        for (size_t i = 0; i < model->layer_count; i++) {
            boat_layer_t* layer = model->layers[i];
            if (layer) {
                if (layer->ops && layer->ops->free) {
                    layer->ops->free(layer);
                } else {
                    free(layer);
                }
            }
        }
        boat_free(model->layers);
    }

    // Free nodes array (nodes themselves are owned by the graph)
    if (model->nodes) {
        boat_free(model->nodes);
    }

    if (model->graph) {
        boat_graph_free(model->graph);
    }

    if (model->name) {
        boat_free(model->name);
    }

    if (model->user_data && model->free_user_data) {
        model->free_user_data(model->user_data);
    }

    boat_free(model);
}

// Graph access
BOAT_API boat_graph_t* boat_model_graph(const boat_model_t* model) {
    return model ? model->graph : NULL;
}

BOAT_API void boat_model_set_graph(boat_model_t* model, boat_graph_t* graph) {
    if (!model || !graph) {
        return;
    }

    if (model->graph) {
        boat_graph_free(model->graph);
    }

    model->graph = (boat_graph_t*)graph;
    // Note: graph ownership is transferred to model
    model->device = boat_graph_device(graph);
}

// Model operations
BOAT_API boat_tensor_t* boat_model_forward(const boat_model_t* model, const boat_tensor_t* input) {
    if (!model || !input) return NULL;
    if (model->layer_count == 0) return NULL;

    // If graph is empty (no nodes), fall back to sequential execution
    if (boat_graph_node_count(model->graph) == 0) {
        boat_tensor_t* current = NULL;
        boat_tensor_t* next = NULL;

        // Process first layer
        boat_layer_t* first_layer = model->layers[0];
        if (!first_layer || !first_layer->ops || !first_layer->ops->forward) {
            return NULL;
        }
        current = first_layer->ops->forward(first_layer, input);
        if (!current) return NULL;

        // Process remaining layers
        for (size_t i = 1; i < model->layer_count; i++) {
            boat_layer_t* layer = model->layers[i];
            if (!layer || !layer->ops || !layer->ops->forward) {
                boat_tensor_free(current);
                return NULL;
            }
            next = layer->ops->forward(layer, current);
            boat_tensor_free(current);
            if (!next) return NULL;
            current = next;
        }

        return current;
    }

    // Use computational graph for forward propagation
    const boat_graph_t* graph = model->graph;

    // Get topological sort of nodes
    size_t node_count = boat_graph_node_count(graph);
    boat_node_t** sorted_nodes = boat_malloc(node_count * sizeof(boat_node_t*), BOAT_DEVICE_CPU);
    if (!sorted_nodes) {
        return NULL;
    }

    size_t sorted_count = 0;
    boat_graph_topological_sort(graph, sorted_nodes, &sorted_count);

    // Map from node to output tensor
    boat_tensor_t** node_outputs =
        boat_calloc(node_count * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    if (!node_outputs) {
        boat_free(sorted_nodes);
        return NULL;
    }

    // Map from node to layer (create index mapping)
    // Build map from node pointer to layer index
    // Since model->nodes array parallels model->layers, we can search
    boat_tensor_t* final_output = NULL;

    // Process nodes in topological order
    for (size_t i = 0; i < sorted_count; i++) {
        const boat_node_t* node = sorted_nodes[i];

        // Find layer index for this node
        size_t layer_index = SIZE_MAX;
        for (size_t j = 0; j < model->layer_count; j++) {
            if (model->nodes[j] == node) {
                layer_index = j;
                break;
            }
        }

        if (layer_index == SIZE_MAX) {
            // Node not found in model nodes (could be input/output node)
            // Skip for now
            continue;
        }

        boat_layer_t* layer = model->layers[layer_index];
        if (!layer || !layer->ops || !layer->ops->forward) {
            // Cleanup and return error
            for (size_t k = 0; k < node_count; k++) {
                if (node_outputs[k]) {
                    boat_tensor_free(node_outputs[k]);
                }
            }
            boat_free(node_outputs);
            boat_free(sorted_nodes);
            return NULL;
        }

        // Collect input tensors from predecessor nodes
        // Count incoming edges with forward direction
        size_t num_inputs = 0;
        for (size_t j = 0; j < boat_graph_edge_count(graph); j++) {
            const boat_edge_t* edge = boat_graph_get_edge_at_index(graph, j);
            if (!edge) continue;
            if (boat_edge_target(edge) == node &&
                boat_edge_direction(edge) == BOAT_EDGE_DIRECTION_FORWARD) {
                num_inputs++;
            }
        }

        boat_tensor_t* layer_input = NULL;

        if (num_inputs == 0) {
            // First layer in graph - use external input
            layer_input = (boat_tensor_t*)input; // Cast away const for API compatibility
        } else if (num_inputs == 1) {
            // Single input - typical for sequential models
            for (size_t j = 0; j < boat_graph_edge_count(graph); j++) {
                const boat_edge_t* edge = boat_graph_get_edge_at_index(graph, j);
                if (!edge) continue;
                if (boat_edge_target(edge) == node &&
                    boat_edge_direction(edge) == BOAT_EDGE_DIRECTION_FORWARD) {
                    const boat_node_t* source_node = boat_edge_source(edge);
                    // Find source node index
                    size_t source_idx = SIZE_MAX;
                    for (size_t k = 0; k < model->layer_count; k++) {
                        if (model->nodes[k] == source_node) {
                            source_idx = k;
                            break;
                        }
                    }
                    if (source_idx != SIZE_MAX && source_idx < node_count) {
                        layer_input = node_outputs[source_idx];
                    }
                    break;
                }
            }
        } else {
            // Multiple inputs - not yet supported for simple sequential models
            // For now, use first input
            for (size_t j = 0; j < boat_graph_edge_count(graph); j++) {
                const boat_edge_t* edge = boat_graph_get_edge_at_index(graph, j);
                if (!edge) continue;
                if (boat_edge_target(edge) == node &&
                    boat_edge_direction(edge) == BOAT_EDGE_DIRECTION_FORWARD) {
                    const boat_node_t* source_node = boat_edge_source(edge);
                    size_t source_idx = SIZE_MAX;
                    for (size_t k = 0; k < model->layer_count; k++) {
                        if (model->nodes[k] == source_node) {
                            source_idx = k;
                            break;
                        }
                    }
                    if (source_idx != SIZE_MAX && source_idx < node_count) {
                        layer_input = node_outputs[source_idx];
                        break;
                    }
                }
            }
        }

        if (!layer_input && num_inputs > 0) {
            // Input not available
            for (size_t k = 0; k < node_count; k++) {
                if (node_outputs[k]) {
                    boat_tensor_free(node_outputs[k]);
                }
            }
            boat_free(node_outputs);
            boat_free(sorted_nodes);
            return NULL;
        }

        // Call layer forward
        boat_tensor_t* output = layer->ops->forward(layer, layer_input);
        if (!output) {
            // Cleanup
            for (size_t k = 0; k < node_count; k++) {
                if (node_outputs[k]) {
                    boat_tensor_free(node_outputs[k]);
                }
            }
            boat_free(node_outputs);
            boat_free(sorted_nodes);
            return NULL;
        }

        // Store output
        node_outputs[layer_index] = output;

        // If this is the last layer (no outgoing edges), it's the final output
        bool has_outgoing = false;
        for (size_t j = 0; j < boat_graph_edge_count(graph); j++) {
            const boat_edge_t* edge = boat_graph_get_edge_at_index(graph, j);
            if (!edge) continue;
            if (boat_edge_source(edge) == node &&
                boat_edge_direction(edge) == BOAT_EDGE_DIRECTION_FORWARD) {
                has_outgoing = true;
                break;
            }
        }

        if (!has_outgoing) {
            final_output = output;
        }
    }

    // Cleanup temporary arrays and intermediate outputs (keep the final output)
    for (size_t k = 0; k < node_count; k++) {
        if (node_outputs[k] && node_outputs[k] != final_output) {
            boat_tensor_free(node_outputs[k]);
        }
    }
    boat_free(sorted_nodes);
    boat_free(node_outputs);

    return final_output;
}

BOAT_API boat_tensor_t* boat_model_backward(const boat_model_t* model,
                                            const boat_tensor_t* grad_output) {
    (void)model;
    (void)grad_output;
    // TODO: Implement backward pass through computational graph
    return NULL;
}

BOAT_API void boat_model_update(const boat_model_t* model, float learning_rate) {
    (void)model;
    (void)learning_rate;
    // TODO: Implement parameter update
}

// Model serialization

// --- Static tensor I/O helpers ---

static bool save_tensor_to_file(FILE* f, const boat_tensor_t* tensor, uint32_t version) {
    uint32_t is_null = (tensor == NULL) ? 1 : 0;
    if (fwrite(&is_null, sizeof(uint32_t), 1, f) != 1) return false;
    if (is_null) return true;

    uint32_t ndim = (uint32_t)boat_tensor_ndim(tensor);
    if (fwrite(&ndim, sizeof(uint32_t), 1, f) != 1) return false;
    const int64_t* shape = boat_tensor_shape(tensor);
    for (uint32_t i = 0; i < ndim; i++) {
        uint32_t dim = (uint32_t)shape[i];
        if (fwrite(&dim, sizeof(uint32_t), 1, f) != 1) return false;
    }
    uint32_t dtype = (uint32_t)boat_tensor_dtype(tensor);
    if (fwrite(&dtype, sizeof(uint32_t), 1, f) != 1) return false;

    size_t nbytes = boat_tensor_nbytes(tensor);
    const void* data = boat_tensor_const_data(tensor);
    if (fwrite(data, 1, nbytes, f) != nbytes) return false;

    if (version >= 2) {
        boat_dtype_t dt = boat_tensor_dtype(tensor);
        bool quantized = ((dt == BOAT_DTYPE_UINT8 || dt == BOAT_DTYPE_INT8 ||
                           dt == BOAT_DTYPE_BITS2 || dt == BOAT_DTYPE_BITS1) &&
                          boat_tensor_get_scale(tensor) != 0.0f) ||
                         dt == BOAT_DTYPE_FLOAT4;
        uint32_t quant_flag = quantized ? 1 : 0;
        if (fwrite(&quant_flag, sizeof(uint32_t), 1, f) != 1) return false;
        if (quantized) {
            float scale = boat_tensor_get_scale(tensor);
            int32_t zero_point = boat_tensor_get_zero_point(tensor);
            if (fwrite(&scale, sizeof(float), 1, f) != 1) return false;
            if (fwrite(&zero_point, sizeof(int32_t), 1, f) != 1) return false;
        }
    }
    if (version >= 3) {
        bool per_channel = boat_tensor_is_per_channel(tensor);
        uint32_t pc_flag = per_channel ? 1 : 0;
        if (fwrite(&pc_flag, sizeof(uint32_t), 1, f) != 1) return false;
        if (per_channel) {
            uint32_t n_channels = (uint32_t)boat_tensor_num_channels(tensor);
            if (fwrite(&n_channels, sizeof(uint32_t), 1, f) != 1) return false;
            const float* scales = boat_tensor_get_scales(tensor);
            const int32_t* zero_points = boat_tensor_get_zero_points(tensor);
            for (uint32_t i = 0; i < n_channels; i++) {
                if (fwrite(&scales[i], sizeof(float), 1, f) != 1) return false;
                if (fwrite(&zero_points[i], sizeof(int32_t), 1, f) != 1) return false;
            }
        }
    }
    return true;
}

static boat_tensor_t* load_tensor_from_file(FILE* f, uint32_t version) {
    uint32_t is_null;
    if (fread(&is_null, sizeof(uint32_t), 1, f) != 1) return NULL;
    if (is_null) return NULL;

    uint32_t ndim;
    if (fread(&ndim, sizeof(uint32_t), 1, f) != 1) return NULL;

    int64_t* shape = malloc(sizeof(int64_t) * ndim);
    if (!shape) return NULL;
    for (uint32_t i = 0; i < ndim; i++) {
        uint32_t dim;
        if (fread(&dim, sizeof(uint32_t), 1, f) != 1) {
            free(shape);
            return NULL;
        }
        shape[i] = (int64_t)dim;
    }

    uint32_t dtype_u32;
    if (fread(&dtype_u32, sizeof(uint32_t), 1, f) != 1) {
        free(shape);
        return NULL;
    }
    boat_dtype_t dtype = (boat_dtype_t)dtype_u32;

    size_t total_elems = 1;
    for (uint32_t i = 0; i < ndim; i++)
        total_elems *= (size_t)shape[i];
    size_t nbytes;
    switch (dtype) {
    case BOAT_DTYPE_FLOAT4: nbytes = (total_elems + 1) / 2; break;
    case BOAT_DTYPE_BITS2: nbytes = (total_elems + 3) / 4; break;
    case BOAT_DTYPE_BITS1: nbytes = (total_elems + 7) / 8; break;
    default: nbytes = total_elems * boat_dtype_size(dtype); break;
    }

    void* data = malloc(nbytes);
    if (!data) {
        free(shape);
        return NULL;
    }
    if (fread(data, 1, nbytes, f) != nbytes) {
        free(data);
        free(shape);
        return NULL;
    }

    boat_tensor_t* tensor = boat_tensor_from_data(shape, (size_t)ndim, dtype, data);
    free(data);
    free(shape);
    if (!tensor) return NULL;

    if (version >= 2) {
        uint32_t quant_flag;
        if (fread(&quant_flag, sizeof(uint32_t), 1, f) != 1) {
            boat_tensor_unref(tensor);
            return NULL;
        }
        if (quant_flag) {
            float scale;
            int32_t zero_point;
            if (fread(&scale, sizeof(float), 1, f) != 1 ||
                fread(&zero_point, sizeof(int32_t), 1, f) != 1) {
                boat_tensor_unref(tensor);
                return NULL;
            }
            boat_tensor_set_quant_params(tensor, scale, zero_point);
        }
    }
    if (version >= 3) {
        uint32_t pc_flag;
        if (fread(&pc_flag, sizeof(uint32_t), 1, f) != 1) {
            boat_tensor_unref(tensor);
            return NULL;
        }
        if (pc_flag) {
            uint32_t n_channels;
            if (fread(&n_channels, sizeof(uint32_t), 1, f) != 1) {
                boat_tensor_unref(tensor);
                return NULL;
            }
            float* scales = (float*)malloc(sizeof(float) * n_channels);
            int32_t* zero_points = (int32_t*)malloc(sizeof(int32_t) * n_channels);
            if (!scales || !zero_points) {
                free(scales);
                free(zero_points);
                boat_tensor_unref(tensor);
                return NULL;
            }
            for (uint32_t i = 0; i < n_channels; i++) {
                if (fread(&scales[i], sizeof(float), 1, f) != 1 ||
                    fread(&zero_points[i], sizeof(int32_t), 1, f) != 1) {
                    free(scales);
                    free(zero_points);
                    boat_tensor_unref(tensor);
                    return NULL;
                }
            }
            boat_tensor_set_per_channel_quant_params(tensor, scales, zero_points,
                                                     (size_t)n_channels);
            free(scales);
            free(zero_points);
        }
    }
    return tensor;
}

// --- Ops tables for loaded layers ---

static boat_tensor_t* dense_forward_op(const boat_layer_t* layer, const boat_tensor_t* input) {
    return boat_dense_layer_forward((boat_dense_layer_t*)layer->data, input);
}
static boat_tensor_t* dense_backward_op(const boat_layer_t* layer, const boat_tensor_t* grad) {
    return boat_dense_layer_backward((boat_dense_layer_t*)layer->data, grad);
}
static void dense_update_op(const boat_layer_t* layer, float lr) {
    boat_dense_layer_update((boat_dense_layer_t*)layer->data, lr);
}
static void dense_free_op(const boat_layer_t* layer) {
    if (layer && layer->data) {
        boat_dense_layer_free((boat_dense_layer_t*)layer->data);
        free((void*)layer);
    }
}

static boat_tensor_t* conv_forward_op(const boat_layer_t* layer, const boat_tensor_t* input) {
    return boat_conv_layer_forward((boat_conv_layer_t*)layer->data, input);
}
static boat_tensor_t* conv_backward_op(const boat_layer_t* layer, const boat_tensor_t* grad) {
    return boat_conv_layer_backward((boat_conv_layer_t*)layer->data, grad);
}
static void conv_update_op(const boat_layer_t* layer, float lr) {
    boat_conv_layer_update((boat_conv_layer_t*)layer->data, lr);
}
static void conv_free_op(const boat_layer_t* layer) {
    if (layer && layer->data) {
        boat_conv_layer_free((boat_conv_layer_t*)layer->data);
        free((void*)layer);
    }
}

static boat_tensor_t* pool_forward_op(const boat_layer_t* layer, const boat_tensor_t* input) {
    return boat_pool_layer_forward((boat_pool_layer_t*)layer->data, input);
}
static boat_tensor_t* pool_backward_op(const boat_layer_t* layer, const boat_tensor_t* grad) {
    return boat_pool_layer_backward((boat_pool_layer_t*)layer->data, grad);
}
static void pool_update_op(const boat_layer_t* layer, float lr) {
    boat_pool_layer_update((boat_pool_layer_t*)layer->data, lr);
}
static void pool_free_op(const boat_layer_t* layer) {
    if (layer && layer->data) {
        boat_pool_layer_free((boat_pool_layer_t*)layer->data);
        free((void*)layer);
    }
}

static boat_tensor_t* relu_forward_op(const boat_layer_t* layer, const boat_tensor_t* input) {
    return boat_relu_layer_forward((boat_relu_layer_t*)layer->data, input);
}
static boat_tensor_t* relu_backward_op(const boat_layer_t* layer, const boat_tensor_t* grad) {
    return boat_relu_layer_backward((boat_relu_layer_t*)layer->data, grad);
}
static void relu_update_op(const boat_layer_t* layer, float lr) {
    boat_relu_layer_update((boat_relu_layer_t*)layer->data, lr);
}
static void relu_free_op(const boat_layer_t* layer) {
    if (layer && layer->data) {
        boat_relu_layer_free((boat_relu_layer_t*)layer->data);
        free((void*)layer);
    }
}

static boat_tensor_t* softmax_forward_op(const boat_layer_t* layer, const boat_tensor_t* input) {
    return boat_softmax_layer_forward((boat_softmax_layer_t*)layer->data, input);
}
static boat_tensor_t* softmax_backward_op(const boat_layer_t* layer, const boat_tensor_t* grad) {
    return boat_softmax_layer_backward((boat_softmax_layer_t*)layer->data, grad);
}
static void softmax_update_op(const boat_layer_t* layer, float lr) {
    boat_softmax_layer_update((boat_softmax_layer_t*)layer->data, lr);
}
static void softmax_free_op(const boat_layer_t* layer) {
    if (layer && layer->data) {
        boat_softmax_layer_free((boat_softmax_layer_t*)layer->data);
        free((void*)layer);
    }
}

static boat_tensor_t* flatten_forward_op(const boat_layer_t* layer, const boat_tensor_t* input) {
    return boat_flatten_layer_forward((boat_flatten_layer_t*)layer->data, input);
}
static boat_tensor_t* flatten_backward_op(const boat_layer_t* layer, const boat_tensor_t* grad) {
    return boat_flatten_layer_backward((boat_flatten_layer_t*)layer->data, grad);
}
static void flatten_update_op(const boat_layer_t* layer, float lr) {
    boat_flatten_layer_update((boat_flatten_layer_t*)layer->data, lr);
}
static void flatten_free_op(const boat_layer_t* layer) {
    if (layer && layer->data) {
        boat_flatten_layer_free((boat_flatten_layer_t*)layer->data);
        free((void*)layer);
    }
}

static boat_tensor_t* bn_forward_op(const boat_layer_t* layer, const boat_tensor_t* input) {
    return boat_batchnorm2d_layer_forward((boat_batchnorm2d_layer_t*)layer->data, input);
}
static boat_tensor_t* bn_backward_op(const boat_layer_t* layer, const boat_tensor_t* grad) {
    return boat_batchnorm2d_layer_backward((boat_batchnorm2d_layer_t*)layer->data, grad);
}
static void bn_update_op(const boat_layer_t* layer, float lr) {
    boat_batchnorm2d_layer_update((boat_batchnorm2d_layer_t*)layer->data, lr);
}
static void bn_free_op(const boat_layer_t* layer) {
    if (layer && layer->data) {
        boat_batchnorm2d_layer_free((boat_batchnorm2d_layer_t*)layer->data);
        free((void*)layer);
    }
}

static const boat_layer_ops_t dense_ops = {.forward = dense_forward_op,
                                           .backward = dense_backward_op,
                                           .update = dense_update_op,
                                           .free = dense_free_op};
static const boat_layer_ops_t conv_ops = {.forward = conv_forward_op,
                                          .backward = conv_backward_op,
                                          .update = conv_update_op,
                                          .free = conv_free_op};
static const boat_layer_ops_t pool_ops = {.forward = pool_forward_op,
                                          .backward = pool_backward_op,
                                          .update = pool_update_op,
                                          .free = pool_free_op};
static const boat_layer_ops_t relu_ops = {.forward = relu_forward_op,
                                          .backward = relu_backward_op,
                                          .update = relu_update_op,
                                          .free = relu_free_op};
static const boat_layer_ops_t softmax_ops = {.forward = softmax_forward_op,
                                             .backward = softmax_backward_op,
                                             .update = softmax_update_op,
                                             .free = softmax_free_op};
static const boat_layer_ops_t flatten_ops = {.forward = flatten_forward_op,
                                             .backward = flatten_backward_op,
                                             .update = flatten_update_op,
                                             .free = flatten_free_op};
static const boat_layer_ops_t bn_ops = {.forward = bn_forward_op,
                                        .backward = bn_backward_op,
                                        .update = bn_update_op,
                                        .free = bn_free_op};

// --- Attention ops ---
static boat_tensor_t* attn_forward_op(const boat_layer_t* layer, const boat_tensor_t* input) {
    return boat_attention_forward((boat_attention_t*)layer->data, input, input, input, NULL);
}
static boat_tensor_t* attn_backward_op(const boat_layer_t* layer, const boat_tensor_t* grad) {
    boat_tensor_t* gq = NULL;
    boat_attention_backward((boat_attention_t*)layer->data, grad, &gq, NULL, NULL);
    return gq;
}
static void attn_update_op(const boat_layer_t* layer, float lr) {
    boat_attention_update((boat_attention_t*)layer->data, lr);
}
static void attn_free_op(const boat_layer_t* layer) {
    if (layer && layer->data) {
        boat_attention_free((boat_attention_t*)layer->data);
        free((void*)layer);
    }
}
static const boat_layer_ops_t attn_ops = {.forward = attn_forward_op,
                                          .backward = attn_backward_op,
                                          .update = attn_update_op,
                                          .free = attn_free_op};

// --- RMSNorm ops ---
static boat_tensor_t* rms_forward_op(const boat_layer_t* layer, const boat_tensor_t* input) {
    return boat_rmsnorm_forward((boat_rmsnorm_t*)layer->data, input);
}
static boat_tensor_t* rms_backward_op(const boat_layer_t* layer, const boat_tensor_t* grad) {
    return boat_rmsnorm_backward((boat_rmsnorm_t*)layer->data, grad);
}
static void rms_update_op(const boat_layer_t* layer, float lr) {
    boat_rmsnorm_update((boat_rmsnorm_t*)layer->data, lr);
}
static void rms_free_op(const boat_layer_t* layer) {
    if (layer && layer->data) {
        boat_rmsnorm_free((boat_rmsnorm_t*)layer->data);
        free((void*)layer);
    }
}
static const boat_layer_ops_t rms_ops = {.forward = rms_forward_op,
                                         .backward = rms_backward_op,
                                         .update = rms_update_op,
                                         .free = rms_free_op};

static void set_layer_ops(boat_layer_t* wrapper) {
    switch (wrapper->type) {
    case BOAT_LAYER_TYPE_DENSE: wrapper->ops = &dense_ops; break;
    case BOAT_LAYER_TYPE_CONV2D: wrapper->ops = &conv_ops; break;
    case BOAT_LAYER_TYPE_MAXPOOL2D: wrapper->ops = &pool_ops; break;
    case BOAT_LAYER_TYPE_RELU: wrapper->ops = &relu_ops; break;
    case BOAT_LAYER_TYPE_SOFTMAX: wrapper->ops = &softmax_ops; break;
    case BOAT_LAYER_TYPE_FLATTEN: wrapper->ops = &flatten_ops; break;
    case BOAT_LAYER_TYPE_BATCHNORM2D: wrapper->ops = &bn_ops; break;
    case BOAT_LAYER_TYPE_ATTENTION: wrapper->ops = &attn_ops; break;
    case BOAT_LAYER_TYPE_RMSNORM: wrapper->ops = &rms_ops; break;
    default: wrapper->ops = NULL; break;
    }
}

// --- Save ---

BOAT_API bool boat_model_save(const boat_model_t* model, const char* filename) {
    if (!model || !filename || model->layer_count == 0) return false;

    FILE* f = fopen(filename, "wb");
    if (!f) return false;

    // Determine format version: version 2 if any weight is quantized
    uint32_t version = 1;
    for (size_t i = 0; i < model->layer_count; i++) {
        const boat_layer_t* wrapper = model->layers[i];
        if (!wrapper) continue;
        boat_tensor_t* w = NULL;
        switch (wrapper->type) {
        case BOAT_LAYER_TYPE_DENSE:
            w = boat_dense_layer_get_weight((const boat_dense_layer_t*)wrapper->data);
            break;
        case BOAT_LAYER_TYPE_CONV2D:
            w = boat_conv_layer_get_weight((const boat_conv_layer_t*)wrapper->data);
            break;
        default: break;
        }
        if (!w) continue;
        // Check per-channel first (version 3)
        if (boat_tensor_is_per_channel(w)) {
            version = 3;
            break;
        }
        // Check per-tensor quantized (version 2)
        boat_dtype_t wdt = boat_tensor_dtype(w);
        if ((wdt == BOAT_DTYPE_UINT8 || wdt == BOAT_DTYPE_INT8 || wdt == BOAT_DTYPE_BITS2 ||
             wdt == BOAT_DTYPE_BITS1) &&
            boat_tensor_get_scale(w) != 0.0f) {
            version = 2;
        } else if (wdt == BOAT_DTYPE_FLOAT4) {
            version = 2;
        }
    }

    uint32_t magic = 0x424F4154; // "BOAT"
    uint32_t layer_count = (uint32_t)model->layer_count;

    if (fwrite(&magic, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&version, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&layer_count, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return false;
    }

    for (size_t i = 0; i < model->layer_count; i++) {
        const boat_layer_t* wrapper = model->layers[i];
        if (!wrapper) {
            fclose(f);
            return false;
        }

        uint32_t type_u32 = (uint32_t)wrapper->type;
        if (fwrite(&type_u32, sizeof(uint32_t), 1, f) != 1) {
            fclose(f);
            return false;
        }

        switch (wrapper->type) {
        case BOAT_LAYER_TYPE_DENSE: {
            boat_dense_layer_t* d = (boat_dense_layer_t*)wrapper->data;
            boat_tensor_t* w = boat_dense_layer_get_weight(d);
            boat_tensor_t* b = boat_dense_layer_get_bias(d);

            uint32_t hp_size = sizeof(uint64_t) * 3;
            const int64_t* ws = boat_tensor_shape(w);
            uint64_t hp_in = (uint64_t)ws[0];
            uint64_t hp_out = (uint64_t)ws[1];
            uint64_t hp_bias = (b != NULL) ? 1 : 0;

            if (fwrite(&hp_size, sizeof(uint32_t), 1, f) != 1) {
                fclose(f);
                return false;
            }
            fwrite(&hp_in, sizeof(uint64_t), 1, f);
            fwrite(&hp_out, sizeof(uint64_t), 1, f);
            fwrite(&hp_bias, sizeof(uint64_t), 1, f);

            uint32_t tc = 2;
            fwrite(&tc, sizeof(uint32_t), 1, f);
            if (!save_tensor_to_file(f, w, version) || !save_tensor_to_file(f, b, version)) {
                fclose(f);
                return false;
            }
            break;
        }
        case BOAT_LAYER_TYPE_CONV2D: {
            boat_conv_layer_t* c = (boat_conv_layer_t*)wrapper->data;
            boat_tensor_t* w = boat_conv_layer_get_weight(c);
            boat_tensor_t* b = boat_conv_layer_get_bias(c);
            const int64_t* ws = boat_tensor_shape(w);

            uint32_t hp_size = sizeof(uint64_t) * 6;
            uint64_t hp_in = (uint64_t)ws[1], hp_out = (uint64_t)ws[0];
            uint64_t hp_k = (uint64_t)ws[2];
            uint64_t hp_s = (uint64_t)boat_conv_layer_get_stride(c);
            uint64_t hp_p = (uint64_t)boat_conv_layer_get_padding(c);
            uint64_t hp_g = (uint64_t)boat_conv_layer_get_groups(c);

            if (fwrite(&hp_size, sizeof(uint32_t), 1, f) != 1) {
                fclose(f);
                return false;
            }
            fwrite(&hp_in, sizeof(uint64_t), 1, f);
            fwrite(&hp_out, sizeof(uint64_t), 1, f);
            fwrite(&hp_k, sizeof(uint64_t), 1, f);
            fwrite(&hp_s, sizeof(uint64_t), 1, f);
            fwrite(&hp_p, sizeof(uint64_t), 1, f);
            fwrite(&hp_g, sizeof(uint64_t), 1, f);

            uint32_t tc = 2;
            fwrite(&tc, sizeof(uint32_t), 1, f);
            if (!save_tensor_to_file(f, w, version) || !save_tensor_to_file(f, b, version)) {
                fclose(f);
                return false;
            }
            break;
        }
        case BOAT_LAYER_TYPE_BATCHNORM2D: {
            boat_batchnorm2d_layer_t* bn = (boat_batchnorm2d_layer_t*)wrapper->data;
            boat_tensor_t* w = boat_batchnorm2d_layer_get_weight(bn);
            boat_tensor_t* b = boat_batchnorm2d_layer_get_bias(bn);
            boat_tensor_t* rm = boat_batchnorm2d_layer_get_running_mean(bn);
            boat_tensor_t* rv = boat_batchnorm2d_layer_get_running_var(bn);

            uint32_t hp_size = sizeof(uint64_t) + sizeof(float) * 2 + sizeof(uint64_t);
            uint64_t hp_nf = (uint64_t)boat_batchnorm2d_layer_get_affine(bn)
                                 ? (uint64_t)boat_tensor_shape(w)[0]
                                 : (uint64_t)boat_tensor_shape(rm)[0];
            float hp_eps = boat_batchnorm2d_layer_get_eps(bn);
            float hp_mom = boat_batchnorm2d_layer_get_momentum(bn);
            uint64_t hp_aff = boat_batchnorm2d_layer_get_affine(bn) ? 1 : 0;

            if (fwrite(&hp_size, sizeof(uint32_t), 1, f) != 1) {
                fclose(f);
                return false;
            }
            fwrite(&hp_nf, sizeof(uint64_t), 1, f);
            fwrite(&hp_eps, sizeof(float), 1, f);
            fwrite(&hp_mom, sizeof(float), 1, f);
            fwrite(&hp_aff, sizeof(uint64_t), 1, f);

            uint32_t tc = 4;
            fwrite(&tc, sizeof(uint32_t), 1, f);
            if (!save_tensor_to_file(f, w, version) || !save_tensor_to_file(f, b, version) ||
                !save_tensor_to_file(f, rm, version) || !save_tensor_to_file(f, rv, version)) {
                fclose(f);
                return false;
            }
            break;
        }
        case BOAT_LAYER_TYPE_MAXPOOL2D: {
            boat_pool_layer_t* p = (boat_pool_layer_t*)wrapper->data;
            uint32_t hp_size = sizeof(uint64_t) * 3;
            uint64_t hp_ps = (uint64_t)boat_pool_layer_get_pool_size(p);
            uint64_t hp_s = (uint64_t)boat_pool_layer_get_stride(p);
            uint64_t hp_pad = (uint64_t)boat_pool_layer_get_padding(p);

            if (fwrite(&hp_size, sizeof(uint32_t), 1, f) != 1) {
                fclose(f);
                return false;
            }
            fwrite(&hp_ps, sizeof(uint64_t), 1, f);
            fwrite(&hp_s, sizeof(uint64_t), 1, f);
            fwrite(&hp_pad, sizeof(uint64_t), 1, f);

            uint32_t tc = 0;
            fwrite(&tc, sizeof(uint32_t), 1, f);
            break;
        }
        case BOAT_LAYER_TYPE_RELU:
        case BOAT_LAYER_TYPE_FLATTEN: {
            uint32_t hp_size = 0;
            if (fwrite(&hp_size, sizeof(uint32_t), 1, f) != 1) {
                fclose(f);
                return false;
            }
            uint32_t tc = 0;
            fwrite(&tc, sizeof(uint32_t), 1, f);
            break;
        }
        case BOAT_LAYER_TYPE_SOFTMAX: {
            boat_softmax_layer_t* sm = (boat_softmax_layer_t*)wrapper->data;
            uint32_t hp_size = sizeof(int32_t);
            int32_t hp_axis = boat_softmax_layer_get_axis(sm);

            if (fwrite(&hp_size, sizeof(uint32_t), 1, f) != 1) {
                fclose(f);
                return false;
            }
            fwrite(&hp_axis, sizeof(int32_t), 1, f);

            uint32_t tc = 0;
            fwrite(&tc, sizeof(uint32_t), 1, f);
            break;
        }
        default:
            boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                            "[Model] Save: unsupported layer type %u\n", type_u32);
            fclose(f);
            return false;
        }
    }

    fclose(f);
    return true;
}

// --- Load ---

BOAT_API boat_model_t* boat_model_load(const char* filename) {
    if (!filename) return NULL;

    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    uint32_t magic, version;
    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 ||
        fread(&version, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return NULL;
    }
    if (magic != 0x424F4154) {
        fclose(f);
        return NULL;
    }
    if (version < 1 || version > 3) {
        fclose(f);
        return NULL;
    }

    uint32_t layer_count_u32;
    if (fread(&layer_count_u32, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        return NULL;
    }

    boat_model_t* model = boat_model_create();
    if (!model) {
        fclose(f);
        return NULL;
    }

    for (uint32_t i = 0; i < layer_count_u32; i++) {
        uint32_t type_u32;
        if (fread(&type_u32, sizeof(uint32_t), 1, f) != 1) {
            boat_model_free(model);
            fclose(f);
            return NULL;
        }
        boat_layer_type_t type = (boat_layer_type_t)type_u32;

        boat_layer_t* wrapper = malloc(sizeof(boat_layer_t));
        if (!wrapper) {
            boat_model_free(model);
            fclose(f);
            return NULL;
        }
        wrapper->type = type;
        wrapper->ops = NULL;
        wrapper->data = NULL;

        switch (type) {
        case BOAT_LAYER_TYPE_DENSE: {
            uint32_t hp_size;
            if (fread(&hp_size, sizeof(uint32_t), 1, f) != 1 || hp_size != sizeof(uint64_t) * 3) {
                free(wrapper);
                boat_model_free(model);
                fclose(f);
                return NULL;
            }
            uint64_t hp_in, hp_out, hp_bias;
            fread(&hp_in, sizeof(uint64_t), 1, f);
            fread(&hp_out, sizeof(uint64_t), 1, f);
            fread(&hp_bias, sizeof(uint64_t), 1, f);

            boat_dense_layer_t* dense =
                boat_dense_layer_create((size_t)hp_in, (size_t)hp_out, hp_bias != 0);
            if (!dense) {
                free(wrapper);
                boat_model_free(model);
                fclose(f);
                return NULL;
            }
            wrapper->data = dense;

            uint32_t tc;
            fread(&tc, sizeof(uint32_t), 1, f);
            boat_tensor_t* wt = load_tensor_from_file(f, version);
            boat_tensor_t* bt = load_tensor_from_file(f, version);
            if (wt) {
                boat_dense_layer_set_weight(dense, wt);
                boat_tensor_unref(wt);
            }
            if (bt) {
                boat_dense_layer_set_bias(dense, bt);
                boat_tensor_unref(bt);
            }
            break;
        }
        case BOAT_LAYER_TYPE_CONV2D: {
            uint32_t hp_size;
            if (fread(&hp_size, sizeof(uint32_t), 1, f) != 1 ||
                (hp_size != sizeof(uint64_t) * 5 && hp_size != sizeof(uint64_t) * 6)) {
                free(wrapper);
                boat_model_free(model);
                fclose(f);
                return NULL;
            }
            uint64_t hp_in, hp_out, hp_k, hp_s, hp_p, hp_g;
            fread(&hp_in, sizeof(uint64_t), 1, f);
            fread(&hp_out, sizeof(uint64_t), 1, f);
            fread(&hp_k, sizeof(uint64_t), 1, f);
            fread(&hp_s, sizeof(uint64_t), 1, f);
            fread(&hp_p, sizeof(uint64_t), 1, f);
            hp_g = 1;
            if (hp_size == sizeof(uint64_t) * 6) {
                fread(&hp_g, sizeof(uint64_t), 1, f);
            }

            boat_conv_layer_t* conv =
                boat_conv_layer_create((size_t)(hp_in * hp_g), (size_t)hp_out, (size_t)hp_k,
                                       (size_t)hp_s, (size_t)hp_p, (size_t)hp_g);
            if (!conv) {
                free(wrapper);
                boat_model_free(model);
                fclose(f);
                return NULL;
            }
            wrapper->data = conv;

            uint32_t tc;
            fread(&tc, sizeof(uint32_t), 1, f);
            boat_tensor_t* wt = load_tensor_from_file(f, version);
            boat_tensor_t* bt = load_tensor_from_file(f, version);
            if (wt) {
                boat_conv_layer_set_weight(conv, wt);
                boat_tensor_unref(wt);
            }
            if (bt) {
                boat_conv_layer_set_bias(conv, bt);
                boat_tensor_unref(bt);
            }
            break;
        }
        case BOAT_LAYER_TYPE_BATCHNORM2D: {
            uint32_t hp_size;
            if (fread(&hp_size, sizeof(uint32_t), 1, f) != 1 ||
                hp_size != sizeof(uint64_t) + sizeof(float) * 2 + sizeof(uint64_t)) {
                free(wrapper);
                boat_model_free(model);
                fclose(f);
                return NULL;
            }
            uint64_t hp_nf;
            float hp_eps, hp_mom;
            uint64_t hp_aff;
            fread(&hp_nf, sizeof(uint64_t), 1, f);
            fread(&hp_eps, sizeof(float), 1, f);
            fread(&hp_mom, sizeof(float), 1, f);
            fread(&hp_aff, sizeof(uint64_t), 1, f);

            boat_batchnorm2d_layer_t* bn =
                boat_batchnorm2d_layer_create((size_t)hp_nf, hp_eps, hp_mom, hp_aff != 0);
            if (!bn) {
                free(wrapper);
                boat_model_free(model);
                fclose(f);
                return NULL;
            }
            wrapper->data = bn;

            uint32_t tc;
            fread(&tc, sizeof(uint32_t), 1, f);
            boat_tensor_t* wt = load_tensor_from_file(f, version);
            boat_tensor_t* bt = load_tensor_from_file(f, version);
            boat_tensor_t* rmt = load_tensor_from_file(f, version);
            boat_tensor_t* rvt = load_tensor_from_file(f, version);
            if (wt) {
                boat_batchnorm2d_layer_set_weight(bn, wt);
                boat_tensor_unref(wt);
            }
            if (bt) {
                boat_batchnorm2d_layer_set_bias(bn, bt);
                boat_tensor_unref(bt);
            }
            if (rmt) {
                boat_batchnorm2d_layer_set_running_mean(bn, rmt);
                boat_tensor_unref(rmt);
            }
            if (rvt) {
                boat_batchnorm2d_layer_set_running_var(bn, rvt);
                boat_tensor_unref(rvt);
            }
            break;
        }
        case BOAT_LAYER_TYPE_MAXPOOL2D: {
            uint32_t hp_size;
            if (fread(&hp_size, sizeof(uint32_t), 1, f) != 1 || hp_size != sizeof(uint64_t) * 3) {
                free(wrapper);
                boat_model_free(model);
                fclose(f);
                return NULL;
            }
            uint64_t hp_ps, hp_s, hp_pad;
            fread(&hp_ps, sizeof(uint64_t), 1, f);
            fread(&hp_s, sizeof(uint64_t), 1, f);
            fread(&hp_pad, sizeof(uint64_t), 1, f);

            boat_pool_layer_t* pool =
                boat_pool_layer_create((size_t)hp_ps, (size_t)hp_s, (size_t)hp_pad);
            if (!pool) {
                free(wrapper);
                boat_model_free(model);
                fclose(f);
                return NULL;
            }
            wrapper->data = pool;

            uint32_t tc;
            fread(&tc, sizeof(uint32_t), 1, f);
            break;
        }
        case BOAT_LAYER_TYPE_RELU: {
            uint32_t hp_size;
            fread(&hp_size, sizeof(uint32_t), 1, f);
            boat_relu_layer_t* relu = boat_relu_layer_create();
            if (!relu) {
                free(wrapper);
                boat_model_free(model);
                fclose(f);
                return NULL;
            }
            wrapper->data = relu;
            uint32_t tc;
            fread(&tc, sizeof(uint32_t), 1, f);
            break;
        }
        case BOAT_LAYER_TYPE_SOFTMAX: {
            uint32_t hp_size;
            if (fread(&hp_size, sizeof(uint32_t), 1, f) != 1 || hp_size != sizeof(int32_t)) {
                free(wrapper);
                boat_model_free(model);
                fclose(f);
                return NULL;
            }
            int32_t hp_axis;
            fread(&hp_axis, sizeof(int32_t), 1, f);

            boat_softmax_layer_t* sm = boat_softmax_layer_create(hp_axis);
            if (!sm) {
                free(wrapper);
                boat_model_free(model);
                fclose(f);
                return NULL;
            }
            wrapper->data = sm;

            uint32_t tc;
            fread(&tc, sizeof(uint32_t), 1, f);
            break;
        }
        case BOAT_LAYER_TYPE_FLATTEN: {
            uint32_t hp_size;
            fread(&hp_size, sizeof(uint32_t), 1, f);
            boat_flatten_layer_t* flat = boat_flatten_layer_create();
            if (!flat) {
                free(wrapper);
                boat_model_free(model);
                fclose(f);
                return NULL;
            }
            wrapper->data = flat;
            uint32_t tc;
            fread(&tc, sizeof(uint32_t), 1, f);
            break;
        }
        default:
            boat_set_errorf(BOAT_ERROR_FORMAT, "[Model] Load: unsupported layer type %u\n",
                            type_u32);
            free(wrapper);
            boat_model_free(model);
            fclose(f);
            return NULL;
        }

        set_layer_ops(wrapper);
        boat_model_add_layer(model, wrapper);
    }

    fclose(f);
    return model;
}

// Get layer by index
BOAT_API boat_layer_t* boat_model_get_layer(const boat_model_t* model, size_t index) {
    if (!model || index >= model->layer_count) return NULL;
    return model->layers[index];
}

// User data management
BOAT_API void* boat_model_get_user_data(const boat_model_t* model) {
    return model ? model->user_data : NULL;
}

BOAT_API void boat_model_set_user_data(boat_model_t* model, void* user_data,
                                       void (*free_fn)(void*)) {
    if (!model) {
        return;
    }

    // Free existing user data if any
    if (model->user_data && model->free_user_data) {
        model->free_user_data(model->user_data);
    }

    model->user_data = user_data;
    model->free_user_data = free_fn;
}

// Get layer count
BOAT_API size_t boat_model_layer_count(const boat_model_t* model) {
    return model ? model->layer_count : 0;
}

// Add layer to model
BOAT_API void boat_model_add_layer(boat_model_t* model, boat_layer_t* layer) {
    if (!model || !layer) {
        return;
    }

    // Auto-set ops if not already assigned
    if (!layer->ops) {
        set_layer_ops(layer);
    }

    // Expand layers and nodes arrays if needed
    if (model->layer_count >= model->layer_capacity) {
        size_t new_capacity = model->layer_capacity == 0 ? 4 : model->layer_capacity * 2;

        // Reallocate layers array
        boat_layer_t** new_layers =
            boat_realloc(model->layers, new_capacity * sizeof(boat_layer_t*), BOAT_DEVICE_CPU);
        if (!new_layers) {
            boat_set_errorf(BOAT_ERROR_OUT_OF_MEMORY, "[Model] Failed to expand layers array\n");
            return;
        }
        model->layers = new_layers;

        // Reallocate nodes array
        boat_node_t** new_nodes =
            boat_realloc(model->nodes, new_capacity * sizeof(boat_node_t*), BOAT_DEVICE_CPU);
        if (!new_nodes) {
            boat_set_errorf(BOAT_ERROR_OUT_OF_MEMORY, "[Model] Failed to expand nodes array\n");
            // Note: layers array already reallocated, but this is an error state
            return;
        }
        model->nodes = new_nodes;
        model->layer_capacity = new_capacity;
    }

    // Create graph node for this layer
    boat_node_t* node = boat_graph_add_node(model->graph, layer, BOAT_NODE_TYPE_OPERATION, NULL);
    if (!node) {
        boat_set_errorf(BOAT_ERROR_INVALID_OPERATION,
                        "[Model] Failed to create graph node for layer\n");
        return;
    }

    // Connect to previous node if exists
    if (model->layer_count > 0) {
        const boat_node_t* prev_node = model->nodes[model->layer_count - 1];
        if (prev_node) {
            const boat_edge_t* edge =
                boat_graph_add_edge(model->graph, prev_node, node, BOAT_EDGE_DIRECTION_FORWARD);
            if (!edge) {
                BOAT_DEBUG_PRINT("[Model] Warning: Failed to add edge between layer nodes\n");
            }
        }
    }

    // Store layer and node
    model->layers[model->layer_count] = layer;
    model->nodes[model->layer_count] = node;
    model->layer_count++;
}
