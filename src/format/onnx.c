// onnx.c - ONNX model format loader
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/format/onnx.h>
#include <boat/model.h>
#include <boat/layers.h>
#include <boat/memory.h>
#include <boat/tensor.h>
#include <boat/graph.h>
#include "onnx_pb.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

static onnx_tensor_t* find_init_tensor(const onnx_graph_t* graph, const char* name) {
    for (int i = 0; i < graph->num_initializers; i++) {
        if (graph->initializers[i].name &&
            strcmp(graph->initializers[i].name, name) == 0) {
            return &graph->initializers[i];
        }
    }
    return NULL;
}

// Convert an ONNX tensor initializer to a boat tensor
static boat_tensor_t* onnx_tensor_to_boat(const onnx_tensor_t* src) {
    if (!src || src->data_type != ONNX_DTYPE_FLOAT) return NULL;

    // Create boat tensor with the same shape
    boat_tensor_t* t = boat_tensor_create(src->dims, (size_t)src->num_dims,
                                           BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!t) return NULL;

    // Copy data
    float* dst = (float*)boat_tensor_data(t);
    size_t count = boat_tensor_nelements(t);
    size_t data_floats = src->raw_data_size / sizeof(float);
    size_t copy = count < data_floats ? count : data_floats;
    memcpy(dst, src->raw_data, copy * sizeof(float));

    return t;
}

// Transpose a 2D boat tensor (swap dims 0 and 1)
static boat_tensor_t* transpose_2d(const boat_tensor_t* t) {
    if (!t || boat_tensor_ndim(t) != 2) return NULL;
    const int64_t* shape = boat_tensor_shape(t);
    int64_t new_shape[] = { shape[1], shape[0] };
    boat_tensor_t* out = boat_tensor_create(new_shape, 2, boat_tensor_dtype(t), BOAT_DEVICE_CPU);
    if (!out) return NULL;

    const float* src = (const float*)boat_tensor_const_data(t);
    float* dst = (float*)boat_tensor_data(out);
    for (int64_t i = 0; i < shape[0]; i++)
        for (int64_t j = 0; j < shape[1]; j++)
            dst[j * shape[0] + i] = src[i * shape[1] + j];
    return out;
}

// Parse Gemm attributes to get transB flag
static bool gemm_transB(const onnx_node_t* node) {
    for (int i = 0; i < node->num_attrs; i++) {
        if (strcmp(node->attr_names[i], "transB") == 0)
            return node->attr_ints[i] != 0;
    }
    return false; // ONNX default
}

// Attribute lookup helpers (return default if not found)
static int64_t get_attr_int(const onnx_node_t* node, const char* name, int64_t def) {
    for (int i = 0; i < node->num_attrs; i++) {
        if (strcmp(node->attr_names[i], name) == 0)
            return node->attr_ints[i];
    }
    return def;
}

static float get_attr_float(const onnx_node_t* node, const char* name, float def) {
    for (int i = 0; i < node->num_attrs; i++) {
        if (strcmp(node->attr_names[i], name) == 0)
            return node->attr_floats[i];
    }
    return def;
}

// Get first element of an INTS attribute (used for kernel_shape, strides, pads)
// Falls back to scalar INT if INTS not found
static int64_t get_attr_ints_first(const onnx_node_t* node, const char* name, int64_t def) {
    for (int i = 0; i < node->num_attrs; i++) {
        if (strcmp(node->attr_names[i], name) == 0) {
            // For INTS type, attr_ints stores the first element; for INT type, use directly
            return node->attr_ints[i];
        }
    }
    return def;
}

// Build a boat_model_t from a parsed onnx_model_t
static boat_model_t* build_model(const onnx_model_t* onnx) {
    boat_model_t* model = boat_model_create();
    if (!model) return NULL;

    for (int i = 0; i < onnx->graph.num_nodes; i++) {
        const onnx_node_t* node = &onnx->graph.nodes[i];

        if (strcmp(node->op_type, "Gemm") == 0) {
            // Gemm: Y = alpha * A * B + beta * C
            // With transB=1: Y = A * B^T + C (standard for Gemm->Dense mapping)
            const onnx_tensor_t* w_init = (node->num_inputs > 1) ?
                find_init_tensor(&onnx->graph, node->names[1]) : NULL;
            const onnx_tensor_t* b_init = (node->num_inputs > 2) ?
                find_init_tensor(&onnx->graph, node->names[2]) : NULL;

            if (!w_init || w_init->data_type != ONNX_DTYPE_FLOAT) {
                boat_model_free(model);
                return NULL;
            }

            bool trans_b = gemm_transB(node);
            // ONNX weight shape depends on transB:
            //   transB=0: [input_features, output_features] (matches Dense directly)
            //   transB=1: [output_features, input_features] (needs transpose)
            int64_t out_features = trans_b ? w_init->dims[0] : w_init->dims[1];
            int64_t in_features = trans_b ? w_init->dims[1] : w_init->dims[0];

            boat_dense_layer_t* dense = boat_dense_layer_create(
                (size_t)in_features, (size_t)out_features, b_init != NULL);
            if (!dense) { boat_model_free(model); return NULL; }

            boat_tensor_t* w = onnx_tensor_to_boat(w_init);
            if (!w) { boat_dense_layer_free(dense); boat_model_free(model); return NULL; }

            if (trans_b) {
                // Transpose from [out_features, in_features] to [in_features, out_features]
                boat_tensor_t* w_T = transpose_2d(w);
                boat_tensor_unref(w);
                if (!w_T) { boat_dense_layer_free(dense); boat_model_free(model); return NULL; }
                boat_dense_layer_set_weight(dense, w_T);
                boat_tensor_unref(w_T);
            } else {
                boat_dense_layer_set_weight(dense, w);
                boat_tensor_unref(w);
            }

            // Set bias
            if (b_init && b_init->raw_data_size > 0) {
                boat_tensor_t* b = onnx_tensor_to_boat(b_init);
                if (b) {
                    boat_dense_layer_set_bias(dense, b);
                    boat_tensor_unref(b);
                }
            }

            boat_layer_t* wrapper = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            if (!wrapper) { boat_dense_layer_free(dense); boat_model_free(model); return NULL; }
            wrapper->data = dense;
            wrapper->type = BOAT_LAYER_TYPE_DENSE;
            wrapper->ops = NULL;
            boat_model_add_layer(model, wrapper);

        } else if (strcmp(node->op_type, "Relu") == 0) {
            boat_relu_layer_t* relu = boat_relu_layer_create();
            if (!relu) { boat_model_free(model); return NULL; }

            boat_layer_t* wrapper = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            if (!wrapper) { boat_relu_layer_free(relu); boat_model_free(model); return NULL; }
            wrapper->data = relu;
            wrapper->type = BOAT_LAYER_TYPE_RELU;
            wrapper->ops = NULL;
            boat_model_add_layer(model, wrapper);

        } else if (strcmp(node->op_type, "Conv") == 0) {
            // Conv: Y = Conv(X, W, B)
            // ONNX weight shape: [M, C, kH, kW] (same as boat NCHW layout)
            const onnx_tensor_t* w_init = (node->num_inputs > 1) ?
                find_init_tensor(&onnx->graph, node->names[1]) : NULL;
            const onnx_tensor_t* b_init = (node->num_inputs > 2) ?
                find_init_tensor(&onnx->graph, node->names[2]) : NULL;

            if (!w_init || w_init->data_type != ONNX_DTYPE_FLOAT || w_init->num_dims < 4) {
                boat_model_free(model);
                return NULL;
            }

            size_t in_channels = (size_t)w_init->dims[1];
            size_t out_channels = (size_t)w_init->dims[0];
            size_t kernel_size = (size_t)get_attr_ints_first(node, "kernel_shape", 1);
            size_t stride = (size_t)get_attr_ints_first(node, "strides", 1);
            size_t padding = (size_t)get_attr_ints_first(node, "pads", 0);

            boat_conv_layer_t* conv = boat_conv_layer_create(
                in_channels, out_channels, kernel_size, stride, padding);
            if (!conv) { boat_model_free(model); return NULL; }

            boat_tensor_t* w = onnx_tensor_to_boat(w_init);
            if (!w) { boat_conv_layer_free(conv); boat_model_free(model); return NULL; }
            boat_conv_layer_set_weight(conv, w);
            boat_tensor_unref(w);

            if (b_init && b_init->raw_data_size > 0) {
                boat_tensor_t* b = onnx_tensor_to_boat(b_init);
                if (b) {
                    boat_conv_layer_set_bias(conv, b);
                    boat_tensor_unref(b);
                }
            }

            boat_layer_t* wrapper = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            if (!wrapper) { boat_conv_layer_free(conv); boat_model_free(model); return NULL; }
            wrapper->data = conv;
            wrapper->type = BOAT_LAYER_TYPE_CONV2D;
            wrapper->ops = NULL;
            boat_model_add_layer(model, wrapper);

        } else if (strcmp(node->op_type, "BatchNormalization") == 0) {
            // BatchNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
            // Inputs: X (0), scale/gamma (1), B/beta (2), mean (3), var (4)
            const onnx_tensor_t* scale_init = (node->num_inputs > 1) ?
                find_init_tensor(&onnx->graph, node->names[1]) : NULL;
            const onnx_tensor_t* b_init = (node->num_inputs > 2) ?
                find_init_tensor(&onnx->graph, node->names[2]) : NULL;
            const onnx_tensor_t* mean_init = (node->num_inputs > 3) ?
                find_init_tensor(&onnx->graph, node->names[3]) : NULL;
            const onnx_tensor_t* var_init = (node->num_inputs > 4) ?
                find_init_tensor(&onnx->graph, node->names[4]) : NULL;

            if (!scale_init || scale_init->num_dims < 1) {
                boat_model_free(model);
                return NULL;
            }

            size_t num_features = (size_t)scale_init->dims[0];
            float eps = get_attr_float(node, "epsilon", 1e-5f);
            float momentum = get_attr_float(node, "momentum", 0.9f);

            boat_batchnorm2d_layer_t* bn = boat_batchnorm2d_layer_create(
                num_features, eps, momentum, true);
            if (!bn) { boat_model_free(model); return NULL; }

            if (scale_init && scale_init->raw_data_size > 0) {
                boat_tensor_t* t = onnx_tensor_to_boat(scale_init);
                if (t) { boat_batchnorm2d_layer_set_weight(bn, t); boat_tensor_unref(t); }
            }
            if (b_init && b_init->raw_data_size > 0) {
                boat_tensor_t* t = onnx_tensor_to_boat(b_init);
                if (t) { boat_batchnorm2d_layer_set_bias(bn, t); boat_tensor_unref(t); }
            }
            if (mean_init && mean_init->raw_data_size > 0) {
                boat_tensor_t* t = onnx_tensor_to_boat(mean_init);
                if (t) { boat_batchnorm2d_layer_set_running_mean(bn, t); boat_tensor_unref(t); }
            }
            if (var_init && var_init->raw_data_size > 0) {
                boat_tensor_t* t = onnx_tensor_to_boat(var_init);
                if (t) { boat_batchnorm2d_layer_set_running_var(bn, t); boat_tensor_unref(t); }
            }

            boat_layer_t* wrapper = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            if (!wrapper) { boat_batchnorm2d_layer_free(bn); boat_model_free(model); return NULL; }
            wrapper->data = bn;
            wrapper->type = BOAT_LAYER_TYPE_BATCHNORM2D;
            wrapper->ops = NULL;
            boat_model_add_layer(model, wrapper);

        } else if (strcmp(node->op_type, "MaxPool") == 0) {
            // MaxPool: Y = MaxPool(X)
            size_t pool_size = (size_t)get_attr_ints_first(node, "kernel_shape", 1);
            size_t stride = (size_t)get_attr_ints_first(node, "strides", pool_size);
            size_t padding = (size_t)get_attr_ints_first(node, "pads", 0);

            boat_pool_layer_t* pool = boat_pool_layer_create(pool_size, stride, padding);
            if (!pool) { boat_model_free(model); return NULL; }

            boat_layer_t* wrapper = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            if (!wrapper) { boat_pool_layer_free(pool); boat_model_free(model); return NULL; }
            wrapper->data = pool;
            wrapper->type = BOAT_LAYER_TYPE_MAXPOOL2D;
            wrapper->ops = NULL;
            boat_model_add_layer(model, wrapper);

        } else if (strcmp(node->op_type, "Softmax") == 0) {
            // Softmax: hardcode axis=1 (standard for NCHW feature dim)
            boat_softmax_layer_t* softmax = boat_softmax_layer_create(1);
            if (!softmax) { boat_model_free(model); return NULL; }

            boat_layer_t* wrapper = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            if (!wrapper) { boat_softmax_layer_free(softmax); boat_model_free(model); return NULL; }
            wrapper->data = softmax;
            wrapper->type = BOAT_LAYER_TYPE_SOFTMAX;
            wrapper->ops = NULL;
            boat_model_add_layer(model, wrapper);

        } else if (strcmp(node->op_type, "Flatten") == 0) {
            // Flatten: flatten from axis=1 (ONNX default)
            boat_flatten_layer_t* flatten = boat_flatten_layer_create();
            if (!flatten) { boat_model_free(model); return NULL; }

            boat_layer_t* wrapper = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            if (!wrapper) { boat_flatten_layer_free(flatten); boat_model_free(model); return NULL; }
            wrapper->data = flatten;
            wrapper->type = BOAT_LAYER_TYPE_FLATTEN;
            wrapper->ops = NULL;
            boat_model_add_layer(model, wrapper);

        } else {
            // Unsupported op type
            boat_model_free(model);
            return NULL;
        }
    }

    return model;
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

boat_model_t* boat_onnx_load_from_memory(const void* data, size_t size) {
    if (!data || size == 0) {
        BOAT_DEBUG_PRINT("[ONNX] load_from_memory: NULL or size=0\n");
        return NULL;
    }

    pb_reader_t reader;
    pb_reader_init(&reader, data, size);
    BOAT_DEBUG_PRINT("[ONNX] load_from_memory: data=%p, size=%zu, first_bytes=%02x %02x %02x\n",
        data, size,
        size > 0 ? ((const uint8_t*)data)[0] : 0,
        size > 1 ? ((const uint8_t*)data)[1] : 0,
        size > 2 ? ((const uint8_t*)data)[2] : 0);

    onnx_model_t onnx;
    if (!onnx_parse(&reader, &onnx)) {
        BOAT_DEBUG_PRINT("[ONNX] onnx_parse failed\n");
        onnx_model_free(&onnx);
        return NULL;
    }

    BOAT_DEBUG_PRINT("[ONNX] parsed: ir_version=%lld, nodes=%d, inits=%d\n",
        onnx.ir_version, onnx.graph.num_nodes, onnx.graph.num_initializers);

    // Validate we have at least one node
    if (onnx.graph.num_nodes == 0) {
        BOAT_DEBUG_PRINT("[ONNX] no nodes found\n");
        onnx_model_free(&onnx);
        return NULL;
    }

    boat_model_t* model = build_model(&onnx);
    onnx_model_free(&onnx);
    if (!model) BOAT_DEBUG_PRINT("[ONNX] build_model failed\n");
    return model;
}

boat_model_t* boat_onnx_load(const char* filename) {
    if (!filename) return NULL;

    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    // Get file size
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    if (size <= 0) { fclose(f); return NULL; }
    fseek(f, 0, SEEK_SET);

    // Read entire file
    void* data = boat_malloc((size_t)size, BOAT_DEVICE_CPU);
    if (!data) { fclose(f); return NULL; }

    size_t bytes_read = fread(data, 1, (size_t)size, f);
    fclose(f);
    if (bytes_read != (size_t)size) {
        boat_free(data);
        return NULL;
    }

    boat_model_t* model = boat_onnx_load_from_memory(data, (size_t)size);
    boat_free(data);
    return model;
}

bool boat_onnx_check(const char* filename) {
    if (!filename) return false;
    FILE* f = fopen(filename, "rb");
    if (!f) return false;

    // Check protobuf magic: ONNX files start with varint field 1
    // Proto3: first field should be ir_version (field 1, wire type 0)
    // A valid ONNX file has 08 xx as first bytes
    uint8_t magic[2];
    if (fread(magic, 1, 2, f) != 2) { fclose(f); return false; }
    fclose(f);

    // Tag 1 << 3 | 0 = 0x08, followed by a varint for ir_version
    if (magic[0] != 0x08) return false;
    if (magic[1] > 0x10) return false; // ir_version shouldn't be huge

    // Try full parse
    FILE* f2 = fopen(filename, "rb");
    if (!f2) return false;
    fseek(f2, 0, SEEK_END);
    long size = ftell(f2);
    fseek(f2, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc((size_t)size);
    if (!buf) { fclose(f2); return false; }
    fread(buf, 1, (size_t)size, f2);
    fclose(f2);

    pb_reader_t reader;
    pb_reader_init(&reader, buf, (size_t)size);
    onnx_model_t onnx;
    bool ok = onnx_parse(&reader, &onnx);
    onnx_model_free(&onnx);
    free(buf);
    return ok;
}

bool boat_onnx_get_version(const char* filename, int* major, int* minor, int* patch) {
    if (!filename) return false;

    FILE* f = fopen(filename, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (size <= 0) { fclose(f); return false; }

    uint8_t* buf = (uint8_t*)malloc((size_t)size);
    if (!buf) { fclose(f); return false; }
    fread(buf, 1, (size_t)size, f);
    fclose(f);

    pb_reader_t reader;
    pb_reader_init(&reader, buf, (size_t)size);
    onnx_model_t onnx;
    bool ok = onnx_parse(&reader, &onnx);

    if (ok && major) *major = (int)(onnx.ir_version);
    if (minor) *minor = 0;
    if (patch) *patch = 0;

    onnx_model_free(&onnx);
    free(buf);
    return ok;
}

bool boat_onnx_save(const boat_model_t* model, const char* filename) {
    (void)model;
    (void)filename;
    // TODO: Implement ONNX model saving
    return false;
}

bool boat_onnx_save_to_memory(const boat_model_t* model, void** data, size_t* size) {
    (void)model;
    (void)data;
    (void)size;
    // TODO: Implement ONNX model saving to memory
    return false;
}
