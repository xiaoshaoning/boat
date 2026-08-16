// executor.c - Topological-order graph forward execution (multi-input)
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/graph.h>
#include <boat/memory.h>
#include <boat/tensor.h>
#include <boat/model.h>
#include <stdlib.h>
#include <string.h>
#include "graph_private.h"

// Default operation-node forward: node data is a boat_layer_t* wrapper (as
// created by boat_model_add_layer / the model layer wrappers). Single-input
// layers run through ops->forward; merge layers run through ops->forward_many.
static boat_tensor_t* default_op_forward(const boat_graph_t* graph, const boat_node_t* node,
                                         const boat_tensor_t* const* inputs, size_t n_inputs) {
    (void)graph;
    boat_layer_t* layer = (boat_layer_t*)boat_node_data(node);
    if (!layer) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[GraphForward] operation node data is not a layer\n");
        return NULL;
    }
    if (!layer->ops) boat_layer_resolve_ops(layer); // raw graph nodes may not have a table
    if (!layer->ops) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[GraphForward] layer has no ops table (unknown type %d)\n",
                        (int)layer->type);
        return NULL;
    }
    if (n_inputs == 1 && layer->ops->forward) {
        return layer->ops->forward(layer, inputs[0]);
    }
    if (layer->ops->forward_many) {
        boat_layer_input_t* in = NULL;
        if (n_inputs > 0) {
            in = (boat_layer_input_t*)boat_malloc(n_inputs * sizeof(boat_layer_input_t),
                                                  BOAT_DEVICE_CPU);
            if (!in) return NULL;
            for (size_t i = 0; i < n_inputs; i++)
                in[i].t = inputs[i];
        }
        boat_tensor_t* out = layer->ops->forward_many(layer, in, n_inputs);
        if (in) boat_free(in);
        return out;
    }
    boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                    "[GraphForward] node with %zu inputs has no forward_many\n", n_inputs);
    return NULL;
}

static size_t node_index(const boat_graph_t* graph, const boat_node_t* node) {
    for (size_t i = 0; i < graph->node_count; i++) {
        if (graph->nodes[i] == node) return i;
    }
    return SIZE_MAX;
}

// Set the operation-node forward evaluator.
BOAT_API void boat_graph_set_forward_fn(boat_graph_t* graph, boat_graph_forward_fn_t fn) {
    if (!graph) return;
    graph->forward_fn = fn;
}

BOAT_API int boat_graph_forward(const boat_graph_t* graph, const boat_graph_io_t* inputs,
                                size_t n_inputs, const boat_graph_io_t* outputs, size_t n_outputs) {
    if (!graph) return -1;
    const size_t nc = graph->node_count;
    if (nc == 0) return n_outputs == 0 ? 0 : -1;

    // Reject cyclic graphs up front (topological sort fails on them).
    boat_node_t** order = (boat_node_t**)boat_malloc(nc * sizeof(boat_node_t*), BOAT_DEVICE_CPU);
    if (!order) return -1;
    size_t n_ordered = 0;
    boat_graph_topological_sort(graph, order, &n_ordered);
    if (n_ordered != nc) {
        boat_free(order);
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[GraphForward] graph has a cycle\n");
        return -1;
    }

    // results[node_position] holds one reference to each computed tensor.
    boat_tensor_t** results =
        (boat_tensor_t**)boat_calloc(nc * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    if (!results) {
        boat_free(order);
        return -1;
    }

    // Bind the caller-provided inputs (placeholders or external feeds).
    for (size_t i = 0; i < n_inputs; i++) {
        if (!inputs[i].node || !inputs[i].tensor) {
            boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[GraphForward] NULL input binding\n");
            boat_free(order);
            boat_free(results);
            return -1;
        }
        size_t idx = node_index(graph, inputs[i].node);
        if (idx == SIZE_MAX) {
            boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                            "[GraphForward] input node not in graph\n");
            boat_free(order);
            boat_free(results);
            return -1;
        }
        results[idx] = inputs[i].tensor;
        boat_tensor_ref(results[idx]);
    }

    boat_graph_forward_fn_t fn = graph->forward_fn ? graph->forward_fn : default_op_forward;
    int rc = 0;

    for (size_t oi = 0; oi < n_ordered && rc == 0; oi++) {
        const boat_node_t* node = order[oi];
        size_t idx = node_index(graph, node);
        if (results[idx]) continue; // already bound by the inputs map

        switch (boat_node_type(node)) {
        case BOAT_NODE_TYPE_PLACEHOLDER:
            boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                            "[GraphForward] unbound placeholder node\n");
            rc = -1;
            break;

        case BOAT_NODE_TYPE_CONSTANT:
        case BOAT_NODE_TYPE_VARIABLE: {
            // Node data is the tensor itself.
            void* data = boat_node_data(node);
            if (!data) {
                boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                                "[GraphForward] constant/variable node has no tensor\n");
                rc = -1;
                break;
            }
            results[idx] = (boat_tensor_t*)data;
            boat_tensor_ref(results[idx]);
            break;
        }

        case BOAT_NODE_TYPE_OUTPUT: {
            // Pass through the single forward input.
            const boat_node_t* src = NULL;
            for (size_t e = 0; e < graph->edge_count; e++) {
                const boat_edge_t* edge = graph->edges[e];
                if (!edge || boat_edge_direction(edge) != BOAT_EDGE_DIRECTION_FORWARD) continue;
                if (boat_edge_target(edge) == node) {
                    src = boat_edge_source(edge);
                    break;
                }
            }
            size_t sidx = src ? node_index(graph, src) : SIZE_MAX;
            if (sidx == SIZE_MAX || !results[sidx]) {
                boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                                "[GraphForward] output node has no computed input\n");
                rc = -1;
                break;
            }
            results[idx] = results[sidx];
            boat_tensor_ref(results[idx]);
            break;
        }

        case BOAT_NODE_TYPE_OPERATION: {
            // Collect the tensors from incoming forward edges, in edge order.
            size_t n_in = 0;
            for (size_t e = 0; e < graph->edge_count; e++) {
                const boat_edge_t* edge = graph->edges[e];
                if (!edge || boat_edge_direction(edge) != BOAT_EDGE_DIRECTION_FORWARD) continue;
                if (boat_edge_target(edge) == node) n_in++;
            }
            const boat_tensor_t** ins = NULL;
            if (n_in > 0) {
                ins = (const boat_tensor_t**)boat_malloc(n_in * sizeof(boat_tensor_t*),
                                                         BOAT_DEVICE_CPU);
                if (!ins) {
                    rc = -1;
                    break;
                }
            }
            size_t k = 0;
            for (size_t e = 0; e < graph->edge_count && rc == 0; e++) {
                const boat_edge_t* edge = graph->edges[e];
                if (!edge || boat_edge_direction(edge) != BOAT_EDGE_DIRECTION_FORWARD) continue;
                if (boat_edge_target(edge) != node) continue;
                size_t sidx = node_index(graph, boat_edge_source(edge));
                if (sidx == SIZE_MAX || !results[sidx]) {
                    boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                                    "[GraphForward] operation input not ready\n");
                    rc = -1;
                    break;
                }
                ins[k++] = results[sidx];
            }
            if (rc != 0) {
                if (ins) boat_free(ins);
                break;
            }
            boat_tensor_t* out = fn(graph, node, ins, n_in);
            if (ins) boat_free(ins);
            if (!out) {
                rc = -1;
                break;
            }
            results[idx] = out;
            break;
        }

        default:
            boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[GraphForward] unknown node type\n");
            rc = -1;
            break;
        }
    }

    // Resolve requested outputs (each gets one reference for the caller).
    for (size_t i = 0; i < n_outputs && rc == 0; i++) {
        if (!outputs[i].node) {
            rc = -1;
            break;
        }
        size_t idx = node_index(graph, outputs[i].node);
        if (idx == SIZE_MAX || !results[idx]) {
            boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                            "[GraphForward] requested output node was not produced\n");
            rc = -1;
            break;
        }
        // `outputs` is const in the signature; the tensor field is the
        // out-parameter and is documented as written by this call.
        ((boat_graph_io_t*)outputs)[i].tensor = results[idx];
        boat_tensor_ref(results[idx]);
    }

    // Drop the executor's own references (caller holds the outputs' refs).
    for (size_t i = 0; i < nc; i++) {
        if (results[i]) boat_tensor_free(results[i]);
    }
    boat_free(order);
    boat_free(results);
    return rc;
}
