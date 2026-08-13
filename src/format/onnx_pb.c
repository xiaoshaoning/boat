// onnx_pb.c - Minimal protobuf wire format decoder for ONNX
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include "onnx_pb.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// -----------------------------------------------------------------------
// Reader
// -----------------------------------------------------------------------

void pb_reader_init(pb_reader_t* r, const void* data, size_t size) {
    r->data = (const uint8_t*)data;
    r->size = size;
    r->pos = 0;
}

bool pb_ok(const pb_reader_t* r) {
    return r->pos <= r->size;
}

uint64_t pb_read_varint(pb_reader_t* r) {
    uint64_t val = 0;
    int shift = 0;
    while (r->pos < r->size) {
        uint8_t byte = r->data[r->pos++];
        val |= (uint64_t)(byte & 0x7F) << shift;
        shift += 7;
        if (!(byte & 0x80)) return val;
    }
    return 0;
}

bool pb_read_tag(pb_reader_t* r, uint32_t* field, uint32_t* wire_type) {
    if (r->pos >= r->size) return false;
    uint64_t tag = pb_read_varint(r);
    *field = (uint32_t)(tag >> 3);
    *wire_type = (uint32_t)(tag & 0x7);
    return true;
}

bool pb_skip_field(pb_reader_t* r, uint32_t wire_type) {
    switch (wire_type) {
        case 0: pb_read_varint(r); return true;
        case 1: r->pos += 8; return r->pos <= r->size;
        case 2: {
            uint64_t len = pb_read_varint(r);
            r->pos += (size_t)len;
            return r->pos <= r->size;
        }
        case 5: r->pos += 4; return r->pos <= r->size;
        default: return false;
    }
}

bool pb_read_submessage(pb_reader_t* r, pb_reader_t* sub, size_t len) {
    if (r->pos + len > r->size) return false;
    pb_reader_init(sub, r->data + r->pos, len);
    r->pos += len;
    return true;
}

// -----------------------------------------------------------------------
// Builder
// -----------------------------------------------------------------------

void pb_builder_init(pb_builder_t* b) {
    b->data = NULL;
    b->size = 0;
    b->cap = 0;
}

void pb_builder_free(pb_builder_t* b) {
    free(b->data);
    b->data = NULL;
    b->size = b->cap = 0;
}

void pb_builder_reset(pb_builder_t* b) {
    b->size = 0;
}

static bool pb_grow(pb_builder_t* b, size_t needed) {
    if (b->size + needed <= b->cap) return true;
    size_t new_cap = b->cap ? b->cap * 2 : 256;
    while (new_cap < b->size + needed) new_cap *= 2;
    uint8_t* new_data = (uint8_t*)realloc(b->data, new_cap);
    if (!new_data) return false;
    b->data = new_data;
    b->cap = new_cap;
    return true;
}

void pb_write_raw(pb_builder_t* b, const void* data, size_t len) {
    if (!pb_grow(b, len)) return;
    memcpy(b->data + b->size, data, len);
    b->size += len;
}

void pb_write_tag(pb_builder_t* b, uint32_t field, uint32_t wire_type) {
    uint64_t tag = ((uint64_t)field << 3) | wire_type;
    // Write varint
    do {
        uint8_t byte = tag & 0x7F;
        tag >>= 7;
        if (tag) byte |= 0x80;
        pb_write_raw(b, &byte, 1);
    } while (tag);
}

void pb_write_varint(pb_builder_t* b, uint64_t val) {
    do {
        uint8_t byte = val & 0x7F;
        val >>= 7;
        if (val) byte |= 0x80;
        pb_write_raw(b, &byte, 1);
    } while (val);
}

void pb_write_string(pb_builder_t* b, uint32_t field, const char* s) {
    size_t len = s ? strlen(s) : 0;
    pb_write_tag(b, field, 2);
    pb_write_varint(b, len);
    if (len) pb_write_raw(b, s, len);
}

void pb_write_bytes(pb_builder_t* b, uint32_t field, const void* data, size_t len) {
    pb_write_tag(b, field, 2);
    pb_write_varint(b, len);
    if (len) pb_write_raw(b, data, len);
}

void pb_write_float(pb_builder_t* b, uint32_t field, float val) {
    pb_write_tag(b, field, 5);
    pb_write_raw(b, &val, sizeof(float));
}

void pb_write_int32(pb_builder_t* b, uint32_t field, int32_t val) {
    (void)field;
    pb_write_varint(b, (uint64_t)(int64_t)val);
}

size_t pb_begin_submessage(pb_builder_t* b, uint32_t field) {
    pb_write_tag(b, field, 2);
    // Reserve space for length varint (max 10 bytes)
    // We'll patch it later — write a placeholder, record position
    size_t pos = b->size;
    // Write a placeholder of 10 bytes for the length varint
    uint8_t placeholder[10] = {0};
    pb_write_raw(b, placeholder, 10);
    return pos;
}

void pb_patch_length(pb_builder_t* b, size_t pos) {
    // Compute actual message size after the placeholder
    size_t data_start = pos + 10;
    size_t msg_size = b->size - data_start;

    // Encode msg_size as varint
    uint8_t encoded[10];
    size_t encoded_len = 0;
    uint64_t val = msg_size;
    do {
        encoded[encoded_len] = val & 0x7F;
        val >>= 7;
        if (val) encoded[encoded_len] |= 0x80;
        encoded_len++;
    } while (val);

    // Shift data to make room for the actual varint length
    size_t shift = 10 - encoded_len;
    if (shift > 0) {
        memmove(b->data + pos + encoded_len, b->data + data_start, msg_size);
    }
    memcpy(b->data + pos, encoded, encoded_len);
    b->size -= shift;
}

// -----------------------------------------------------------------------
// ONNX parser
// -----------------------------------------------------------------------

// Field numbers for ONNX protobuf messages
#define ONNX_IR_VERSION     1   // ModelProto.ir_version (int64)
#define ONNX_OPSET_IMPORT   8   // ModelProto.opset_import (repeated OperatorSetIdProto)
#define ONNX_GRAPH          7   // ModelProto.graph (GraphProto)

#define GRAPH_NODE          1   // GraphProto.node (repeated NodeProto)
#define GRAPH_NAME          2   // GraphProto.name (string)
#define GRAPH_INITIALIZER   5   // GraphProto.initializer (repeated TensorProto)
#define GRAPH_INPUT         11  // GraphProto.input (repeated ValueInfoProto)
#define GRAPH_OUTPUT        12  // GraphProto.output (repeated ValueInfoProto)

#define NODE_INPUT          1   // NodeProto.input (repeated string)
#define NODE_OUTPUT         2   // NodeProto.output (repeated string)
#define NODE_NAME           3   // NodeProto.name (string)
#define NODE_OP_TYPE        4   // NodeProto.op_type (string)

#define TENSOR_DIMS         1   // TensorProto.dims (repeated int64, packed)
#define TENSOR_DATA_TYPE    2   // TensorProto.data_type (int32)
#define TENSOR_NAME         8   // TensorProto.name (string)
#define TENSOR_RAW_DATA     9   // TensorProto.raw_data (bytes)

#define OPSET_DOMAIN        1   // OperatorSetIdProto.domain (string)
#define OPSET_VERSION       2   // OperatorSetIdProto.version (int64)

// Attribute proto fields we care about
#define ATTR_NAME           1   // AttributeProto.name (string)
#define ATTR_TYPE           20  // AttributeProto.type (int32)
#define ATTR_F              2   // AttributeProto.f (float)
#define ATTR_I              3   // AttributeProto.i (int64)
#define ATTR_INTS           7   // AttributeProto.ints (repeated int64)

// Read a string field from a reader (tag + len already consumed)
static char* read_string_field(pb_reader_t* r, size_t len) {
    char* s = (char*)malloc(len + 1);
    if (!s) return NULL;
    memcpy(s, r->data + r->pos, len);
    s[len] = '\0';
    r->pos += len;
    return s;
}

// Parse a NodeProto from a reader positioned at its contents
static bool parse_node(pb_reader_t* r, onnx_node_t* node) {
    memset(node, 0, sizeof(*node));
    int inputs_cap = 0, outputs_cap = 0;

    while (r->pos < r->size) {
        uint32_t field, wire;
        if (!pb_read_tag(r, &field, &wire)) break;

        switch (field) {
            case NODE_INPUT: {
                if (wire != 2) { pb_skip_field(r, wire); break; }
                uint64_t len = pb_read_varint(r);
                char* s = read_string_field(r, (size_t)len);
                if (!s) return false;
                if (node->num_inputs >= inputs_cap) {
                    inputs_cap = inputs_cap ? inputs_cap * 2 : 4;
                    char** tmp = (char**)realloc(node->names, inputs_cap * sizeof(char*));
                    if (!tmp) { free(s); return false; }
                    node->names = tmp;
                }
                node->names[node->num_inputs++] = s;
                break;
            }
            case NODE_OUTPUT: {
                if (wire != 2) { pb_skip_field(r, wire); break; }
                uint64_t len = pb_read_varint(r);
                char* s = read_string_field(r, (size_t)len);
                if (!s) return false;
                if (node->num_outputs >= outputs_cap) {
                    outputs_cap = outputs_cap ? outputs_cap * 2 : 4;
                    char** tmp = (char**)realloc(node->outputs, outputs_cap * sizeof(char*));
                    if (!tmp) { free(s); return false; }
                    node->outputs = tmp;
                }
                node->outputs[node->num_outputs++] = s;
                break;
            }
            case NODE_NAME: {
                if (wire != 2) { pb_skip_field(r, wire); break; }
                uint64_t len = pb_read_varint(r);
                node->name = read_string_field(r, (size_t)len);
                break;
            }
            case NODE_OP_TYPE: {
                if (wire != 2) { pb_skip_field(r, wire); break; }
                uint64_t len = pb_read_varint(r);
                node->op_type = read_string_field(r, (size_t)len);
                break;
            }
            case 5: { // AttributeProto (repeated)
                if (wire != 2) { pb_skip_field(r, wire); break; }
                uint64_t alen = pb_read_varint(r);
                pb_reader_t attr;
                if (!pb_read_submessage(r, &attr, (size_t)alen)) break;
                // Parse AttributeProto: extract name, type, and value
                char* attr_name = NULL;
                int32_t attr_type = 0;
                int64_t attr_int = 0;
                float attr_float = 0.0f;
                // Temp storage for INTS list (max 8 elements covers all our needs)
                int64_t ints_list[8];
                int num_ints = 0;
                while (attr.pos < attr.size) {
                    uint32_t af, aw;
                    if (!pb_read_tag(&attr, &af, &aw)) break;
                    switch (af) {
                        case ATTR_NAME:
                            if (aw != 2) { pb_skip_field(&attr, aw); break; }
                            { uint64_t sl = pb_read_varint(&attr);
                              attr_name = read_string_field(&attr, (size_t)sl); }
                            break;
                        case ATTR_TYPE:
                            if (aw != 0) { pb_skip_field(&attr, aw); break; }
                            attr_type = (int32_t)pb_read_varint(&attr);
                            break;
                        case ATTR_F:
                            if (aw != 5) { pb_skip_field(&attr, aw); break; }
                            memcpy(&attr_float, attr.data + attr.pos, 4);
                            attr.pos += 4;
                            break;
                        case ATTR_I:
                            if (aw != 0) { pb_skip_field(&attr, aw); break; }
                            attr_int = (int64_t)pb_read_varint(&attr);
                            break;
                        case 8: { // Some models use field 8 for integer array values (instead of field 7)
                            if (aw == 0) {
                                if (num_ints < 8) ints_list[num_ints++] = (int64_t)pb_read_varint(&attr);
                            } else {
                                pb_skip_field(&attr, aw);
                            }
                            break;
                        }
                        case ATTR_INTS: {
                            if (aw == 0) {
                                // Non-packed repeated varint
                                if (num_ints < 8) {
                                    ints_list[num_ints++] = (int64_t)pb_read_varint(&attr);
                                }
                            } else if (aw == 2) {
                                // Packed repeated varints
                                uint64_t list_len = pb_read_varint(&attr);
                                size_t end = attr.pos + (size_t)list_len;
                                while (attr.pos < end && num_ints < 8) {
                                    ints_list[num_ints++] = (int64_t)pb_read_varint(&attr);
                                }
                            } else {
                                pb_skip_field(&attr, aw);
                            }
                            break;
                        }
                        default:
                            pb_skip_field(&attr, aw);
                            break;
                    }
                }
                if (attr_name) {
                    if (node->num_attrs >= node->attrs_cap) {
                        node->attrs_cap = node->attrs_cap ? node->attrs_cap * 2 : 4;
                        char** tn = (char**)realloc(node->attr_names,
                            node->attrs_cap * sizeof(char*));
                        int64_t* ti = (int64_t*)realloc(node->attr_ints,
                            node->attrs_cap * sizeof(int64_t));
                        float* tf = (float*)realloc(node->attr_floats,
                            node->attrs_cap * sizeof(float));
                        int* ttyp = (int*)realloc(node->attr_types,
                            node->attrs_cap * sizeof(int));
                        if (!tn || !ti || !tf || !ttyp) { free(attr_name); return false; }
                        node->attr_names = tn;
                        node->attr_ints = ti;
                        node->attr_floats = tf;
                        node->attr_types = ttyp;
                    }
                    node->attr_names[node->num_attrs] = attr_name;
                    node->attr_types[node->num_attrs] = attr_type;
                    if (attr_type == ONNX_ATTR_FLOAT) {
                        node->attr_floats[node->num_attrs] = attr_float;
                        node->attr_ints[node->num_attrs] = 0;
                    } else if (attr_type == ONNX_ATTR_INTS) {
                        // Store first element; boat layers use scalar kernel/stride/padding
                        node->attr_ints[node->num_attrs] = num_ints > 0 ? ints_list[0] : 0;
                        node->attr_floats[node->num_attrs] = 0.0f;
                    } else {
                        // INT (default)
                        node->attr_ints[node->num_attrs] = attr_int;
                        node->attr_floats[node->num_attrs] = 0.0f;
                    }
                    node->num_attrs++;
                }
                break;
            }
            default:
                pb_skip_field(r, wire);
                break;
        }
    }
    return true;
}

// Parse a TensorProto from a reader
static bool parse_tensor(pb_reader_t* r, onnx_tensor_t* tensor) {
    memset(tensor, 0, sizeof(*tensor));

    while (r->pos < r->size) {
        uint32_t field, wire;
        if (!pb_read_tag(r, &field, &wire)) break;

        switch (field) {
            case TENSOR_DIMS: {
                // dims are int64, wire type 0 (varint) or packed (wire 2, length-delimited)
                if (wire == 0) {
                    // Single varint (non-packed repeated field)
                    int64_t* tmp = (int64_t*)realloc(tensor->dims, (tensor->num_dims + 1) * sizeof(int64_t));
                    if (!tmp) return false;
                    tensor->dims = tmp;
                    tensor->dims[tensor->num_dims++] = (int64_t)pb_read_varint(r);
                } else if (wire == 2) {
                    // Packed repeated varints
                    uint64_t len = pb_read_varint(r);
                    size_t end = r->pos + (size_t)len;
                    while (r->pos < end) {
                        int64_t* tmp = (int64_t*)realloc(tensor->dims, (tensor->num_dims + 1) * sizeof(int64_t));
                        if (!tmp) return false;
                        tensor->dims = tmp;
                        tensor->dims[tensor->num_dims++] = (int64_t)pb_read_varint(r);
                    }
                } else {
                    pb_skip_field(r, wire);
                }
                break;
            }
            case TENSOR_DATA_TYPE: {
                if (wire != 0) { pb_skip_field(r, wire); break; }
                tensor->data_type = (int32_t)pb_read_varint(r);
                break;
            }
            case TENSOR_NAME: {
                if (wire != 2) { pb_skip_field(r, wire); break; }
                uint64_t len = pb_read_varint(r);
                tensor->name = read_string_field(r, (size_t)len);
                break;
            }
            case TENSOR_RAW_DATA: {
                if (wire != 2) { pb_skip_field(r, wire); break; }
                uint64_t len = pb_read_varint(r);
                tensor->raw_data = (uint8_t*)malloc((size_t)len);
                if (!tensor->raw_data && len > 0) return false;
                memcpy(tensor->raw_data, r->data + r->pos, (size_t)len);
                tensor->raw_data_size = (size_t)len;
                r->pos += (size_t)len;
                break;
            }
            default:
                pb_skip_field(r, wire);
                break;
        }
    }
    return true;
}

static bool parse_graph(pb_reader_t* r, onnx_graph_t* graph);


// Parse the full ONNX model
bool onnx_parse(pb_reader_t* r, onnx_model_t* model) {
    memset(model, 0, sizeof(*model));

    while (r->pos < r->size) {
        uint32_t field, wire;
        if (!pb_read_tag(r, &field, &wire)) break;

        switch (field) {
            case ONNX_IR_VERSION: {
                if (wire != 0) { pb_skip_field(r, wire); break; }
                model->ir_version = (int64_t)pb_read_varint(r);
                break;
            }
            case ONNX_OPSET_IMPORT: {
                if (wire != 2) { pb_skip_field(r, wire); break; }
                uint64_t len = pb_read_varint(r);
                pb_reader_t sub;
                if (!pb_read_submessage(r, &sub, (size_t)len)) break;
                // Parse OperatorSetIdProto
                while (sub.pos < sub.size) {
                    uint32_t sf, sw;
                    if (!pb_read_tag(&sub, &sf, &sw)) break;
                    if (sf == OPSET_VERSION && sw == 0) {
                        model->opset_version = (int64_t)pb_read_varint(&sub);
                    } else {
                        pb_skip_field(&sub, sw);
                    }
                }
                break;
            }
            case ONNX_GRAPH: {
                if (wire != 2) { pb_skip_field(r, wire); break; }
                uint64_t len = pb_read_varint(r);
                pb_reader_t sub;
                if (!pb_read_submessage(r, &sub, (size_t)len)) break;
                parse_graph(&sub, &model->graph);
                break;
            }
            default:
                pb_skip_field(r, wire);
                break;
        }
    }
    return true;
}

static bool parse_graph(pb_reader_t* r, onnx_graph_t* graph) {
    memset(graph, 0, sizeof(*graph));

    while (r->pos < r->size) {
        uint32_t field, wire;
        if (!pb_read_tag(r, &field, &wire)) break;

        switch (field) {
            case GRAPH_NODE: {
                if (wire != 2) { pb_skip_field(r, wire); break; }
                uint64_t len = pb_read_varint(r);
                pb_reader_t sub;
                if (!pb_read_submessage(r, &sub, (size_t)len)) break;
                if (graph->num_nodes >= graph->nodes_cap) {
                    graph->nodes_cap = graph->nodes_cap ? graph->nodes_cap * 2 : 8;
                    onnx_node_t* tmp = (onnx_node_t*)realloc(graph->nodes,
                        graph->nodes_cap * sizeof(onnx_node_t));
                    if (!tmp) return false;
                    graph->nodes = tmp;
                }
                if (!parse_node(&sub, &graph->nodes[graph->num_nodes])) return false;
                graph->num_nodes++;
                break;
            }
            case GRAPH_NAME: {
                if (wire != 2) { pb_skip_field(r, wire); break; }
                uint64_t len = pb_read_varint(r);
                graph->graph_name = read_string_field(r, (size_t)len);
                break;
            }
            case GRAPH_INITIALIZER: {
                if (wire != 2) { pb_skip_field(r, wire); break; }
                uint64_t len = pb_read_varint(r);
                pb_reader_t sub;
                if (!pb_read_submessage(r, &sub, (size_t)len)) break;
                if (graph->num_initializers >= graph->init_cap) {
                    graph->init_cap = graph->init_cap ? graph->init_cap * 2 : 8;
                    onnx_tensor_t* tmp = (onnx_tensor_t*)realloc(graph->initializers,
                        graph->init_cap * sizeof(onnx_tensor_t));
                    if (!tmp) return false;
                    graph->initializers = tmp;
                }
                if (!parse_tensor(&sub, &graph->initializers[graph->num_initializers])) return false;
                graph->num_initializers++;
                break;
            }
            case GRAPH_INPUT:
            case GRAPH_OUTPUT: {
                if (wire != 2) { pb_skip_field(r, wire); break; }
                uint64_t len = pb_read_varint(r);
                r->pos += (size_t)len;
                break;
            }
            default:
                pb_skip_field(r, wire);
                break;
        }
    }
    return true;
}

void onnx_model_free(onnx_model_t* model) {
    if (!model) return;

    // Free nodes
    for (int i = 0; i < model->graph.num_nodes; i++) {
        onnx_node_t* node = &model->graph.nodes[i];
        for (int j = 0; j < node->num_inputs; j++) free(node->names[j]);
        for (int j = 0; j < node->num_outputs; j++) free(node->outputs[j]);
        free(node->names);
        free(node->outputs);
        free(node->name);
        free(node->op_type);
        for (int j = 0; j < node->num_attrs; j++) free(node->attr_names[j]);
        free(node->attr_names);
        free(node->attr_ints);
        free(node->attr_floats);
        free(node->attr_types);
    }
    free(model->graph.nodes);

    // Free initializers
    for (int i = 0; i < model->graph.num_initializers; i++) {
        onnx_tensor_t* t = &model->graph.initializers[i];
        free(t->dims);
        free(t->name);
        free(t->raw_data);
    }
    free(model->graph.initializers);
    free(model->graph.graph_name);
}
