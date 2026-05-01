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

// -----------------------------------------------------------------------
// ONNX save helpers
// -----------------------------------------------------------------------

// Write an INT attribute inside an AttributeProto submessage (field 5)
static void write_int_attr(pb_builder_t* b, const char* name, int64_t val) {
    size_t pos = pb_begin_submessage(b, 5);
    pb_write_string(b, 1, name);
    pb_write_tag(b, 5, 0); pb_write_varint(b, 2);
    pb_write_tag(b, 3, 0); pb_write_varint(b, val);
    pb_patch_length(b, pos);
}

// Write an INTS attribute inside an AttributeProto submessage
static void write_ints_attr(pb_builder_t* b, const char* name, const int64_t* vals, int count) {
    size_t pos = pb_begin_submessage(b, 5);
    pb_write_string(b, 1, name);
    pb_write_tag(b, 5, 0); pb_write_varint(b, 7);
    for (int i = 0; i < count; i++) {
        pb_write_tag(b, 7, 0); pb_write_varint(b, vals[i]);
    }
    pb_patch_length(b, pos);
}

// Write a FLOAT attribute inside an AttributeProto submessage
static void write_float_attr(pb_builder_t* b, const char* name, float val) {
    size_t pos = pb_begin_submessage(b, 5);
    pb_write_string(b, 1, name);
    pb_write_tag(b, 5, 0); pb_write_varint(b, 1);
    pb_write_tag(b, 2, 5);
    pb_write_raw(b, &val, sizeof(val));
    pb_patch_length(b, pos);
}

// Write a boat_tensor as an ONNX TensorProto initializer (field 5 of GraphProto)
static void write_tensor_initializer(pb_builder_t* b, const char* name, const boat_tensor_t* t) {
    if (!t) return;
    size_t pos = pb_begin_submessage(b, 5);

    const int64_t* shape = boat_tensor_shape(t);
    size_t ndim = boat_tensor_ndim(t);
    for (size_t d = 0; d < ndim; d++) {
        pb_write_tag(b, 1, 0);
        pb_write_varint(b, (uint64_t)shape[d]);
    }

    pb_write_tag(b, 5, 0);
    pb_write_varint(b, 1);  // data_type = FLOAT

    pb_write_string(b, 7, name);

    const void* data = boat_tensor_const_data(t);
    size_t nbytes = boat_tensor_nbytes(t);
    pb_write_bytes(b, 10, data, nbytes);

    pb_patch_length(b, pos);
}

// Write a NodeProto for an activation-type op (Relu, Softmax, Flatten)
static void write_simple_node(pb_builder_t* b, const char* input_name,
                              const char* output_name, const char* node_name,
                              const char* op_type) {
    size_t pos = pb_begin_submessage(b, 1);
    pb_write_string(b, 1, input_name);
    pb_write_string(b, 2, output_name);
    pb_write_string(b, 3, node_name);
    pb_write_string(b, 4, op_type);
    pb_patch_length(b, pos);
}

// -----------------------------------------------------------------------
// ONNX public API — save
// -----------------------------------------------------------------------

bool boat_onnx_save_to_memory(const boat_model_t* model, void** out_data, size_t* out_size) {
    if (!model || !out_data || !out_size) return false;

    size_t layer_count = boat_model_layer_count(model);
    if (layer_count == 0) return false;

    pb_builder_t b;
    pb_builder_init(&b);

    // ir_version = 8 (field 1, int64)
    pb_write_tag(&b, 1, 0);
    pb_write_varint(&b, 8);

    // producer_name = "boat" (field 2)
    pb_write_string(&b, 2, "boat");
    // producer_version = "0.1.0" (field 3)
    pb_write_string(&b, 3, "0.1.0");

    // opset_import { domain: "", version: 18 } (field 8)
    size_t opset_pos = pb_begin_submessage(&b, 8);
    pb_write_string(&b, 1, "");
    pb_write_tag(&b, 2, 0);
    pb_write_varint(&b, 18);
    pb_patch_length(&b, opset_pos);

    // GraphProto (field 7)
    size_t graph_pos = pb_begin_submessage(&b, 7);
    pb_write_string(&b, 2, "exported");

    char prev_output[64] = "input";

    for (size_t i = 0; i < layer_count; i++) {
        boat_layer_t* wrapper = boat_model_get_layer(model, i);
        char cur_output[64];
        snprintf(cur_output, sizeof(cur_output), "layer_%zu_output", i);

        switch (wrapper->type) {
            case BOAT_LAYER_TYPE_DENSE: {
                boat_dense_layer_t* dense = (boat_dense_layer_t*)wrapper->data;
                boat_tensor_t* w = boat_dense_layer_get_weight(dense);
                boat_tensor_t* bias = boat_dense_layer_get_bias(dense);

                char w_name[64], b_name[64];
                snprintf(w_name, sizeof(w_name), "layer_%zu_weight", i);
                snprintf(b_name, sizeof(b_name), "layer_%zu_bias", i);

                write_tensor_initializer(&b, w_name, w);
                bool has_bias = bias && boat_tensor_nbytes(bias) > 0;
                if (has_bias) write_tensor_initializer(&b, b_name, bias);

                char node_name[64];
                snprintf(node_name, sizeof(node_name), "layer_%zu", i);

                // Gemm node with transB=0 (weight shape matches boat layout directly)
                size_t np = pb_begin_submessage(&b, 1);
                pb_write_string(&b, 1, prev_output);
                pb_write_string(&b, 1, w_name);
                if (has_bias) pb_write_string(&b, 1, b_name);
                pb_write_string(&b, 2, cur_output);
                pb_write_string(&b, 3, node_name);
                pb_write_string(&b, 4, "Gemm");
                write_int_attr(&b, "transB", 0);
                pb_patch_length(&b, np);
                break;
            }

            case BOAT_LAYER_TYPE_CONV2D: {
                boat_conv_layer_t* conv = (boat_conv_layer_t*)wrapper->data;
                boat_tensor_t* w = boat_conv_layer_get_weight(conv);
                boat_tensor_t* bias = boat_conv_layer_get_bias(conv);

                char w_name[64], b_name[64];
                snprintf(w_name, sizeof(w_name), "layer_%zu_weight", i);
                snprintf(b_name, sizeof(b_name), "layer_%zu_bias", i);

                write_tensor_initializer(&b, w_name, w);
                bool has_bias = bias && boat_tensor_nbytes(bias) > 0;
                if (has_bias) write_tensor_initializer(&b, b_name, bias);

                const int64_t* ws = boat_tensor_shape(w);
                int64_t kernel_size = (boat_tensor_ndim(w) >= 3) ? ws[2] : 1;
                int64_t stride = (int64_t)boat_conv_layer_get_stride(conv);
                int64_t padding = (int64_t)boat_conv_layer_get_padding(conv);

                char node_name[64];
                snprintf(node_name, sizeof(node_name), "layer_%zu", i);

                size_t np = pb_begin_submessage(&b, 1);
                pb_write_string(&b, 1, prev_output);
                pb_write_string(&b, 1, w_name);
                if (has_bias) pb_write_string(&b, 1, b_name);
                pb_write_string(&b, 2, cur_output);
                pb_write_string(&b, 3, node_name);
                pb_write_string(&b, 4, "Conv");
                { int64_t ks[] = {kernel_size, kernel_size};
                  write_ints_attr(&b, "kernel_shape", ks, 2);
                  int64_t st[] = {stride, stride};
                  write_ints_attr(&b, "strides", st, 2);
                  int64_t pd[] = {padding, padding};
                  write_ints_attr(&b, "pads", pd, 2); }
                pb_patch_length(&b, np);
                break;
            }

            case BOAT_LAYER_TYPE_BATCHNORM2D: {
                boat_batchnorm2d_layer_t* bn = (boat_batchnorm2d_layer_t*)wrapper->data;
                boat_tensor_t* scale = boat_batchnorm2d_layer_get_weight(bn);
                boat_tensor_t* bn_bias = boat_batchnorm2d_layer_get_bias(bn);
                boat_tensor_t* mean = boat_batchnorm2d_layer_get_running_mean(bn);
                boat_tensor_t* var = boat_batchnorm2d_layer_get_running_var(bn);

                char s_name[64], b_name[64], m_name[64], v_name[64];
                snprintf(s_name, sizeof(s_name), "layer_%zu_scale", i);
                snprintf(b_name, sizeof(b_name), "layer_%zu_bias", i);
                snprintf(m_name, sizeof(m_name), "layer_%zu_mean", i);
                snprintf(v_name, sizeof(v_name), "layer_%zu_var", i);

                write_tensor_initializer(&b, s_name, scale);
                write_tensor_initializer(&b, b_name, bn_bias);
                write_tensor_initializer(&b, m_name, mean);
                write_tensor_initializer(&b, v_name, var);

                float eps = boat_batchnorm2d_layer_get_eps(bn);

                char node_name[64];
                snprintf(node_name, sizeof(node_name), "layer_%zu", i);

                // BatchNormalization node: 5 inputs (X, scale, B, mean, var)
                size_t np = pb_begin_submessage(&b, 1);
                pb_write_string(&b, 1, prev_output);
                pb_write_string(&b, 1, s_name);
                pb_write_string(&b, 1, b_name);
                pb_write_string(&b, 1, m_name);
                pb_write_string(&b, 1, v_name);
                pb_write_string(&b, 2, cur_output);
                pb_write_string(&b, 3, node_name);
                pb_write_string(&b, 4, "BatchNormalization");
                write_float_attr(&b, "epsilon", eps);
                pb_patch_length(&b, np);
                break;
            }

            case BOAT_LAYER_TYPE_MAXPOOL2D: {
                boat_pool_layer_t* pool = (boat_pool_layer_t*)wrapper->data;
                int64_t pool_size = (int64_t)boat_pool_layer_get_pool_size(pool);
                int64_t stride = (int64_t)boat_pool_layer_get_stride(pool);
                int64_t padding = (int64_t)boat_pool_layer_get_padding(pool);

                char node_name[64];
                snprintf(node_name, sizeof(node_name), "layer_%zu", i);

                size_t np = pb_begin_submessage(&b, 1);
                pb_write_string(&b, 1, prev_output);
                pb_write_string(&b, 2, cur_output);
                pb_write_string(&b, 3, node_name);
                pb_write_string(&b, 4, "MaxPool");
                { int64_t ks[] = {pool_size, pool_size};
                  write_ints_attr(&b, "kernel_shape", ks, 2);
                  int64_t st[] = {stride, stride};
                  write_ints_attr(&b, "strides", st, 2);
                  int64_t pd[] = {padding, padding};
                  write_ints_attr(&b, "pads", pd, 2); }
                pb_patch_length(&b, np);
                break;
            }

            case BOAT_LAYER_TYPE_RELU:
                { char nn[64]; snprintf(nn, sizeof(nn), "layer_%zu", i);
                  write_simple_node(&b, prev_output, cur_output, nn, "Relu"); }
                break;

            case BOAT_LAYER_TYPE_SOFTMAX:
                { char nn[64]; snprintf(nn, sizeof(nn), "layer_%zu", i);
                  write_simple_node(&b, prev_output, cur_output, nn, "Softmax"); }
                break;

            case BOAT_LAYER_TYPE_FLATTEN:
                { char nn[64]; snprintf(nn, sizeof(nn), "layer_%zu", i);
                  write_simple_node(&b, prev_output, cur_output, nn, "Flatten"); }
                break;

            default:
                pb_builder_free(&b);
                return false;
        }

        strcpy(prev_output, cur_output);
    }

    // Input value info (field 11)
    size_t inp_pos = pb_begin_submessage(&b, 11);
    pb_write_string(&b, 1, "input");
    pb_patch_length(&b, inp_pos);

    // Output value info (field 12)
    size_t out_pos = pb_begin_submessage(&b, 12);
    pb_write_string(&b, 1, prev_output);
    pb_patch_length(&b, out_pos);

    pb_patch_length(&b, graph_pos);

    *out_data = b.data;
    *out_size = b.size;
    return true;
}

bool boat_onnx_save(const boat_model_t* model, const char* filename) {
    if (!model || !filename) return false;

    void* data;
    size_t size;
    if (!boat_onnx_save_to_memory(model, &data, &size)) return false;

    FILE* f = fopen(filename, "wb");
    if (!f) { free(data); return false; }

    size_t written = fwrite(data, 1, size, f);
    fclose(f);

    if (written != size) {
        free(data);
        remove(filename);
        return false;
    }

    free(data);
    return true;
}
