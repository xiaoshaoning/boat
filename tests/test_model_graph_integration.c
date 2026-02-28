// test_model_graph_integration.c - Test computational graph integration with model layers
#include <boat/model.h>
#include <boat/layers.h>
#include <boat/graph.h>
#include <stdio.h>
#include <stdlib.h>

// Layer operations for dense layer wrapper
static boat_tensor_t* dense_layer_wrapper_forward(boat_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !layer->data || !input) return NULL;
    boat_dense_layer_t* dense_layer = (boat_dense_layer_t*)layer->data;
    return boat_dense_layer_forward(dense_layer, input);
}

static boat_tensor_t* dense_layer_wrapper_backward(boat_layer_t* layer, const boat_tensor_t* grad_output) {
    if (!layer || !layer->data || !grad_output) return NULL;
    boat_dense_layer_t* dense_layer = (boat_dense_layer_t*)layer->data;
    return boat_dense_layer_backward(dense_layer, grad_output);
}

static void dense_layer_wrapper_update(boat_layer_t* layer, float learning_rate) {
    if (!layer || !layer->data) return;
    boat_dense_layer_t* dense_layer = (boat_dense_layer_t*)layer->data;
    boat_dense_layer_update(dense_layer, learning_rate);
}

static void dense_layer_wrapper_free(boat_layer_t* layer) {
    if (!layer || !layer->data) return;

    boat_dense_layer_t* dense_layer = (boat_dense_layer_t*)layer->data;
    boat_dense_layer_free(dense_layer);

    // Free the layer wrapper itself
    free(layer);
}

static const boat_layer_ops_t dense_layer_ops = {
    .forward = dense_layer_wrapper_forward,
    .backward = dense_layer_wrapper_backward,
    .update = dense_layer_wrapper_update,
    .free = dense_layer_wrapper_free
};

// Layer operations for normalization layer wrapper
static boat_tensor_t* norm_layer_wrapper_forward(boat_layer_t* layer, const boat_tensor_t* input) {
    if (!layer || !layer->data || !input) return NULL;
    boat_norm_layer_t* norm_layer = (boat_norm_layer_t*)layer->data;
    return boat_norm_layer_forward(norm_layer, input);
}

static boat_tensor_t* norm_layer_wrapper_backward(boat_layer_t* layer, const boat_tensor_t* grad_output) {
    if (!layer || !layer->data || !grad_output) return NULL;
    boat_norm_layer_t* norm_layer = (boat_norm_layer_t*)layer->data;
    return boat_norm_layer_backward(norm_layer, grad_output);
}

static void norm_layer_wrapper_update(boat_layer_t* layer, float learning_rate) {
    if (!layer || !layer->data) return;
    boat_norm_layer_t* norm_layer = (boat_norm_layer_t*)layer->data;
    boat_norm_layer_update(norm_layer, learning_rate);
}

static void norm_layer_wrapper_free(boat_layer_t* layer) {
    if (!layer || !layer->data) return;

    boat_norm_layer_t* norm_layer = (boat_norm_layer_t*)layer->data;
    boat_norm_layer_free(norm_layer);

    // Free the layer wrapper itself
    free(layer);
}

static const boat_layer_ops_t norm_layer_ops = {
    .forward = norm_layer_wrapper_forward,
    .backward = norm_layer_wrapper_backward,
    .update = norm_layer_wrapper_update,
    .free = norm_layer_wrapper_free
};

int main() {
    setbuf(stdout, NULL); // Disable buffering for stdout
    printf("Testing model and computational graph integration...\n");

    // 1. Create model
    boat_model_t* model = boat_model_create();
    if (!model) {
        fprintf(stderr, "Failed to create model\n");
        return 1;
    }

    // 2. Check initial graph state
    boat_graph_t* graph = boat_model_graph(model);
    if (!graph) {
        fprintf(stderr, "Model has no graph\n");
        boat_model_free(model);
        return 1;
    }

    // 3. Create layers
    // Dense layer
    boat_dense_layer_t* dense = boat_dense_layer_create(128, 64, true);
    if (!dense) {
        fprintf(stderr, "Failed to create dense layer\n");
        boat_model_free(model);
        return 1;
    }
    // Wrap in generic layer
    boat_layer_t* dense_layer = malloc(sizeof(boat_layer_t));
    if (!dense_layer) {
        fprintf(stderr, "Failed to allocate layer wrapper\n");
        boat_dense_layer_free(dense);
        boat_model_free(model);
        return 1;
    }
    dense_layer->data = dense;
    dense_layer->ops = &dense_layer_ops;

    // Layer normalization layer
    boat_norm_layer_t* norm = boat_norm_layer_create(128, 1e-5f, true);
    if (!norm) {
        fprintf(stderr, "Failed to create normalization layer\n");
        free(dense_layer);
        boat_dense_layer_free(dense);
        boat_model_free(model);
        return 1;
    }
    boat_layer_t* norm_layer = malloc(sizeof(boat_layer_t));
    if (!norm_layer) {
        fprintf(stderr, "Failed to allocate layer wrapper\n");
        boat_norm_layer_free(norm);
        free(dense_layer);
        boat_dense_layer_free(dense);
        boat_model_free(model);
        return 1;
    }
    norm_layer->data = norm;
    norm_layer->ops = &norm_layer_ops;

    // 4. Add layers to model
    printf("Adding dense layer to model...\n");
    boat_model_add_layer(model, dense_layer);
    printf("Adding normalization layer to model...\n");
    boat_model_add_layer(model, norm_layer);

    // 5. Verify layer count
    size_t layer_count = boat_model_layer_count(model);
    if (layer_count != 2) {
        fprintf(stderr, "Expected 2 layers, got %zu\n", layer_count);
        boat_norm_layer_free(norm);
        free(norm_layer);
        free(dense_layer);
        boat_dense_layer_free(dense);
        boat_model_free(model);
        return 1;
    }
    printf("Layer count correct: %zu\n", layer_count);

    // 6. Verify graph node count
    size_t node_count = boat_graph_node_count(graph);
    if (node_count != 2) {
        fprintf(stderr, "Expected 2 graph nodes, got %zu\n", node_count);
        boat_norm_layer_free(norm);
        free(norm_layer);
        free(dense_layer);
        boat_dense_layer_free(dense);
        boat_model_free(model);
        return 1;
    }
    printf("Graph node count correct: %zu\n", node_count);

    // 7. Verify edge count (should be 1 edge between the two nodes)
    size_t edge_count = boat_graph_edge_count(graph);
    if (edge_count != 1) {
        fprintf(stderr, "Expected 1 edge, got %zu\n", edge_count);
        boat_norm_layer_free(norm);
        free(norm_layer);
        free(dense_layer);
        boat_dense_layer_free(dense);
        boat_model_free(model);
        return 1;
    }
    printf("Graph edge count correct: %zu\n", edge_count);

    // 8. Verify node types
    // Get first node
    boat_node_t* node1 = boat_graph_get_node_at_index(graph, 0);
    if (!node1) {
        fprintf(stderr, "Failed to get node 0\n");
        boat_norm_layer_free(norm);
        free(norm_layer);
        free(dense_layer);
        boat_dense_layer_free(dense);
        boat_model_free(model);
        return 1;
    }
    boat_node_type_t type1 = boat_node_type(node1);
    printf("Node 0 type: %s\n", boat_node_type_name(type1));
    // 9. Verify node data points to layer
    void* node_data = boat_node_data(node1);
    printf("Debug: node_data = %p, dense_layer = %p\n", node_data, dense_layer);
    if (node_data != dense_layer) {
        fprintf(stderr, "Node 0 data does not point to dense layer\n");
        fprintf(stderr, "  node_data = %p, dense_layer = %p\n", node_data, dense_layer);
        boat_norm_layer_free(norm);
        free(norm_layer);
        free(dense_layer);
        boat_dense_layer_free(dense);
        boat_model_free(model);
        return 1;
    }

    // 10. Cleanup (model will free layers)
    // Note: model takes ownership of layers
    // layers are freed by model, so we don't free them here
    printf("Calling boat_model_free...\n");
    boat_model_free(model);
    printf("boat_model_free completed.\n");

    printf("All tests passed!\n");
    return 0;
}