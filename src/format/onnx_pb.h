// onnx_pb.h - Minimal protobuf wire format decoder for ONNX
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef ONNX_PB_H
#define ONNX_PB_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// -----------------------------------------------------------------------
// Protobuf wire format reader
// -----------------------------------------------------------------------
typedef struct {
    const uint8_t* data;
    size_t size;
    size_t pos;
} pb_reader_t;

void pb_reader_init(pb_reader_t* r, const void* data, size_t size);
bool pb_ok(const pb_reader_t* r);
uint64_t pb_read_varint(pb_reader_t* r);
bool pb_read_tag(pb_reader_t* r, uint32_t* field, uint32_t* wire_type);
bool pb_skip_field(pb_reader_t* r, uint32_t wire_type);

// Read a submessage: returns a reader over the submessage bytes
// Caller must have already read the length-delimited tag+len
bool pb_read_submessage(pb_reader_t* r, pb_reader_t* sub, size_t len);

// -----------------------------------------------------------------------
// Protobuf wire format builder (dynamic buffer, for testing)
// -----------------------------------------------------------------------
typedef struct {
    uint8_t* data;
    size_t size;
    size_t cap;
} pb_builder_t;

void pb_builder_init(pb_builder_t* b);
void pb_builder_free(pb_builder_t* b);
void pb_builder_reset(pb_builder_t* b);

void pb_write_tag(pb_builder_t* b, uint32_t field, uint32_t wire_type);
void pb_write_varint(pb_builder_t* b, uint64_t val);
void pb_write_string(pb_builder_t* b, uint32_t field, const char* s);
void pb_write_bytes(pb_builder_t* b, uint32_t field, const void* data, size_t len);
void pb_write_float(pb_builder_t* b, uint32_t field, float val);
void pb_write_int32(pb_builder_t* b, uint32_t field, int32_t val);

// Write a submessage: returns the position where length should be patched.
// Call pb_patch_length after writing submessage contents.
size_t pb_begin_submessage(pb_builder_t* b, uint32_t field);
void pb_patch_length(pb_builder_t* b, size_t pos);

// Append raw bytes
void pb_write_raw(pb_builder_t* b, const void* data, size_t len);

// -----------------------------------------------------------------------
// ONNX data structures (minimal, for parser use)
// -----------------------------------------------------------------------

// ONNX TensorProto data types we care about
#define ONNX_DTYPE_FLOAT 1

// ONNX AttributeProto types
#define ONNX_ATTR_FLOAT 1
#define ONNX_ATTR_INT 2
#define ONNX_ATTR_INTS 7

typedef struct {
    char** names; // input names (string field 1, repeated)
    int num_inputs;
    char** outputs; // output names (string field 2, repeated)
    int num_outputs;
    char* name;    // node name (field 3)
    char* op_type; // op_type (field 4)
    // Simple attribute dict (name->int64 mapping)
    char** attr_names;
    int64_t* attr_ints;
    float* attr_floats;
    int* attr_types; // ONNX_ATTR_* values, parallel to attr_names
    int num_attrs;
    int attrs_cap;
} onnx_node_t;

typedef struct {
    int64_t* dims; // dims (field 1, packed varint)
    int num_dims;
    int32_t data_type; // data_type (field 5)
    char* name;        // name (field 7)
    uint8_t* raw_data; // raw_data (field 10)
    size_t raw_data_size;
} onnx_tensor_t;

typedef struct {
    onnx_node_t* nodes;
    int num_nodes;
    int nodes_cap;
    onnx_tensor_t* initializers;
    int num_initializers;
    int init_cap;
    char* graph_name;
} onnx_graph_t;

typedef struct {
    int64_t ir_version;
    onnx_graph_t graph;
    int64_t opset_version; // from first opset_import
} onnx_model_t;

// Parse an ONNX model from a byte buffer
bool onnx_parse(pb_reader_t* r, onnx_model_t* model);

// Free all allocated memory in an ONNX model
void onnx_model_free(onnx_model_t* model);

#ifdef __cplusplus
}
#endif

#endif // ONNX_PB_H
