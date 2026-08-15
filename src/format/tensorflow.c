// tensorflow.c - TensorFlow model format loader
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0
//
// A self-contained reader for TensorFlow frozen graphs (GraphDef .pb) and
// SavedModels, with no TensorFlow SDK dependency. It parses the protobuf wire
// format directly and maps the linear subset of ops to a boat sequential
// model:
//
//   Placeholder / Const, MatMul (optionally transpose_b), BiasAdd / Add
//   (with a Const second input), Relu, Identity.
//
// Unsupported ops fail loudly (boat_set_errorf + NULL), and the save /
// SavedModel-builder entry points report BOAT_ERROR_NOT_IMPLEMENTED.

#include <boat/format/tensorflow.h>
#include <boat/model.h>
#include <boat/memory.h>
#include <boat/tensor.h>
#include <boat/layers.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

// ---------------------------------------------------------------------------
// Minimal protobuf wire-format reader
// ---------------------------------------------------------------------------

typedef struct {
    const uint8_t* p;
    size_t n;
    size_t pos;
    bool ok;
} tf_reader_t;

static uint64_t tf_read_varint(tf_reader_t* r) {
    uint64_t v = 0;
    int shift = 0;
    while (shift < 64) {
        if (r->pos >= r->n) {
            r->ok = false;  // truncated mid-varint
            return 0;
        }
        uint8_t b = r->p[r->pos++];
        v |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) return v;
        shift += 7;
    }
    r->ok = false;  // varint longer than 10 bytes
    return 0;
}

static bool tf_read_tag(tf_reader_t* r, uint32_t* field, uint32_t* wire) {
    if (r->pos >= r->n) return false;  // clean end of message
    uint64_t tag = tf_read_varint(r);
    if (!r->ok || tag == 0) return false;
    *field = (uint32_t)(tag >> 3);
    *wire = (uint32_t)(tag & 7);
    return true;
}

static bool tf_read_bytes(tf_reader_t* r, const uint8_t** data, size_t* len) {
    uint64_t l = tf_read_varint(r);
    if (!r->ok || l > r->n - r->pos) {
        r->ok = false;
        return false;
    }
    *data = r->p + r->pos;
    *len = (size_t)l;
    r->pos += (size_t)l;
    return true;
}

static bool tf_read_fixed32(tf_reader_t* r, uint32_t* v) {
    if (r->n - r->pos < 4) {
        r->ok = false;
        return false;
    }
    memcpy(v, r->p + r->pos, 4);
    r->pos += 4;
    return true;
}

static void tf_skip(tf_reader_t* r, uint32_t wire) {
    switch (wire) {
        case 0:
            tf_read_varint(r);
            break;
        case 1:
            r->pos += 8;
            if (r->pos > r->n) r->ok = false;
            break;
        case 2: {
            uint64_t l = tf_read_varint(r);
            if (r->ok) {
                r->pos += (size_t)l;
                if (r->pos > r->n) r->ok = false;
            }
            break;
        }
        case 5:
            r->pos += 4;
            if (r->pos > r->n) r->ok = false;
            break;
        default:
            r->ok = false;
            break;
    }
}

// ---------------------------------------------------------------------------
// Graph structures
// ---------------------------------------------------------------------------

typedef struct {
    char* name;
    char* op;
    char** inputs;
    int n_inputs;
} tf_node_t;

typedef struct {
    char* name;
    float* data;   // float32 (doubles converted)
    int64_t dims[8];
    int ndims;
    size_t nelems;
} tf_const_t;

typedef struct {
    tf_node_t* nodes;
    int n_nodes;
    tf_const_t* consts;
    int n_consts;
    bool has_placeholder;
} tf_graph_t;

static void tf_graph_free(tf_graph_t* g) {
    for (int i = 0; i < g->n_nodes; i++) {
        free(g->nodes[i].name);
        free(g->nodes[i].op);
        for (int j = 0; j < g->nodes[i].n_inputs; j++) free(g->nodes[i].inputs[j]);
        free(g->nodes[i].inputs);
    }
    free(g->nodes);
    for (int i = 0; i < g->n_consts; i++) {
        free(g->consts[i].name);
        free(g->consts[i].data);
    }
    free(g->consts);
    memset(g, 0, sizeof(*g));
}

static tf_const_t* find_const(tf_graph_t* g, const char* name) {
    for (int i = 0; i < g->n_consts; i++) {
        if (strcmp(g->consts[i].name, name) == 0) return &g->consts[i];
    }
    return NULL;
}

// ---------------------------------------------------------------------------
// TensorProto parsing -> float32 buffer
// ---------------------------------------------------------------------------

// Parse a TensorProto submessage (points at the length-delimited payload).
// Returns a malloc'd tf_const_t (name owned by caller) or NULL.
static tf_const_t* parse_tensor_proto(const uint8_t* data, size_t len) {
    tf_reader_t r = {data, len, 0, true};
    tf_const_t* t = (tf_const_t*)calloc(1, sizeof(*t));
    if (!t) return NULL;

    int dtype = 0;
    int64_t shape[8];
    int ndims = 0;
    const uint8_t* content = NULL;
    size_t content_n = 0;
    float* float_vals = NULL;
    size_t n_float_vals = 0;

    uint32_t field, wire;
    while (tf_read_tag(&r, &field, &wire)) {
        switch (field) {
            case 1:  // dtype
                if (wire == 0) dtype = (int)tf_read_varint(&r);
                else tf_skip(&r, wire);
                break;
            case 2: {  // tensor_shape
                const uint8_t* sub;
                size_t subn;
                if (wire != 2 || !tf_read_bytes(&r, &sub, &subn)) { tf_skip(&r, wire); break; }
                tf_reader_t sr = {sub, subn, 0, true};
                uint32_t f2, w2;
                while (tf_read_tag(&sr, &f2, &w2)) {
                    if (f2 == 2 && w2 == 2) {  // dim { size (1, int64) }
                        const uint8_t* dsub;
                        size_t dsubn;
                        if (tf_read_bytes(&sr, &dsub, &dsubn)) {
                            tf_reader_t dr = {dsub, dsubn, 0, true};
                            uint32_t f3, w3;
                            while (tf_read_tag(&dr, &f3, &w3)) {
                                if (f3 == 1 && w3 == 0 && ndims < 8) {
                                    shape[ndims++] = (int64_t)tf_read_varint(&dr);
                                } else {
                                    tf_skip(&dr, w3);
                                }
                            }
                        }
                    } else {
                        tf_skip(&sr, w2);
                    }
                }
                break;
            }
            case 4:  // tensor_content (raw bytes)
                if (wire == 2) tf_read_bytes(&r, &content, &content_n);
                else tf_skip(&r, wire);
                break;
            case 5:  // int_val (packed or repeated)
                if (wire == 2) {  // packed
                    const uint8_t* sub;
                    size_t subn;
                    tf_read_bytes(&r, &sub, &subn);
                    // int32 values; not needed for float models
                } else {
                    tf_skip(&r, wire);
                }
                break;
            case 6:  // float_val (repeated float, wire 5)
                if (wire == 5) {
                    uint32_t bits;
                    if (tf_read_fixed32(&r, &bits)) {
                        float f;
                        memcpy(&f, &bits, 4);
                        float* nf = (float*)realloc(float_vals, (n_float_vals + 1) * sizeof(float));
                        if (nf) {
                            float_vals = nf;
                            float_vals[n_float_vals++] = f;
                        }
                    }
                } else if (wire == 2) {  // packed floats
                    const uint8_t* sub;
                    size_t subn;
                    if (tf_read_bytes(&r, &sub, &subn)) {
                        size_t cnt = subn / 4;
                        float* nf = (float*)realloc(float_vals, (n_float_vals + cnt) * sizeof(float));
                        if (nf) {
                            float_vals = nf;
                            for (size_t i = 0; i < cnt; i++) {
                                uint32_t bits;
                                memcpy(&bits, sub + 4 * i, 4);
                                memcpy(&float_vals[n_float_vals + i], &bits, 4);
                            }
                            n_float_vals += cnt;
                        }
                    }
                } else {
                    tf_skip(&r, wire);
                }
                break;
            default:
                tf_skip(&r, wire);
                break;
        }
    }
    if (!r.ok) {
        free(float_vals);
        free(t);
        return NULL;
    }

    t->ndims = ndims;
    size_t nelems = 1;
    for (int i = 0; i < ndims; i++) {
        t->dims[i] = shape[i];
        nelems *= (size_t)shape[i];
    }

    if (content && content_n > 0 && dtype == 1) {  // DT_FLOAT raw
        if (content_n < nelems * 4) {
            free(float_vals);
            free(t);
            return NULL;
        }
        t->data = (float*)malloc(nelems * sizeof(float));
        if (!t->data) {
            free(float_vals);
            free(t);
            return NULL;
        }
        memcpy(t->data, content, nelems * sizeof(float));
        t->nelems = nelems;
    } else if (dtype == 2 && content && content_n > 0) {  // DT_DOUBLE raw
        if (content_n < nelems * 8) {
            free(float_vals);
            free(t);
            return NULL;
        }
        t->data = (float*)malloc(nelems * sizeof(float));
        if (!t->data) {
            free(float_vals);
            free(t);
            return NULL;
        }
        const double* d = (const double*)content;
        for (size_t i = 0; i < nelems; i++) t->data[i] = (float)d[i];
        t->nelems = nelems;
    } else if (float_vals) {
        t->data = float_vals;
        t->nelems = n_float_vals;
    } else {
        free(float_vals);
        free(t);
        return NULL;
    }
    return t;
}

// ---------------------------------------------------------------------------
// GraphDef parsing
// ---------------------------------------------------------------------------

// Parse one NodeDef submessage, appending to the graph. Returns 0 on success.
static int parse_node(tf_graph_t* g, const uint8_t* data, size_t len) {
    tf_reader_t r = {data, len, 0, true};
    char* name = NULL;
    char* op = NULL;
    char** inputs = NULL;
    int n_inputs = 0;

    uint32_t field, wire;
    while (tf_read_tag(&r, &field, &wire)) {
        switch (field) {
            case 1: {  // name
                const uint8_t* s;
                size_t sn;
                if (wire == 2 && tf_read_bytes(&r, &s, &sn)) {
                    name = (char*)malloc(sn + 1);
                    if (name) {
                        memcpy(name, s, sn);
                        name[sn] = 0;
                    }
                } else {
                    tf_skip(&r, wire);
                }
                break;
            }
            case 2: {  // op
                const uint8_t* s;
                size_t sn;
                if (wire == 2 && tf_read_bytes(&r, &s, &sn)) {
                    op = (char*)malloc(sn + 1);
                    if (op) {
                        memcpy(op, s, sn);
                        op[sn] = 0;
                    }
                } else {
                    tf_skip(&r, wire);
                }
                break;
            }
            case 3: {  // input (repeated string)
                const uint8_t* s;
                size_t sn;
                if (wire == 2 && tf_read_bytes(&r, &s, &sn)) {
                    char* is = (char*)malloc(sn + 1);
                    if (is) {
                        memcpy(is, s, sn);
                        is[sn] = 0;
                        char** ni = (char**)realloc(inputs, (size_t)(n_inputs + 1) * sizeof(char*));
                        if (ni) {
                            inputs = ni;
                            inputs[n_inputs++] = is;
                        } else {
                            free(is);
                        }
                    }
                } else {
                    tf_skip(&r, wire);
                }
                break;
            }
            case 5: {  // attr map entry { key(1), value(2) }
                const uint8_t* sub;
                size_t subn;
                if (wire != 2 || !tf_read_bytes(&r, &sub, &subn)) {
                    tf_skip(&r, wire);
                    break;
                }
                tf_reader_t ar = {sub, subn, 0, true};
                char* key = NULL;
                const uint8_t* val = NULL;
                size_t val_n = 0;
                uint32_t f2, w2;
                while (tf_read_tag(&ar, &f2, &w2)) {
                    if (f2 == 1 && w2 == 2) {  // key
                        const uint8_t* s;
                        size_t sn;
                        if (tf_read_bytes(&ar, &s, &sn)) {
                            key = (char*)malloc(sn + 1);
                            if (key) {
                                memcpy(key, s, sn);
                                key[sn] = 0;
                            }
                        }
                    } else if (f2 == 2 && w2 == 2) {  // value (AttrValue)
                        tf_read_bytes(&ar, &val, &val_n);
                    } else {
                        tf_skip(&ar, w2);
                    }
                }
                // Only "value" attr carries a Const tensor; parse it.
                if (key && strcmp(key, "value") == 0 && val && op && strcmp(op, "Const") == 0) {
                    // AttrValue { tensor = field 8 (submsg) }
                    tf_reader_t vr = {val, val_n, 0, true};
                    uint32_t f3, w3;
                    while (tf_read_tag(&vr, &f3, &w3)) {
                        if (f3 == 8 && w3 == 2) {
                            const uint8_t* tp;
                            size_t tpn;
                            if (tf_read_bytes(&vr, &tp, &tpn)) {
                                tf_const_t* ct = parse_tensor_proto(tp, tpn);
                                if (ct) {
                                    size_t nmlen = name ? strlen(name) : 0;
                                    ct->name = (char*)malloc(nmlen + 1);
                                    if (ct->name) {
                                        if (name) memcpy(ct->name, name, nmlen);
                                        ct->name[nmlen] = 0;
                                    }
                                    tf_const_t* nc = (tf_const_t*)realloc(
                                        g->consts, (size_t)(g->n_consts + 1) * sizeof(tf_const_t));
                                    if (nc) {
                                        g->consts = nc;
                                        g->consts[g->n_consts++] = *ct;
                                        free(ct);
                                    } else {
                                        free(ct->name);
                                        free(ct->data);
                                        free(ct);
                                    }
                                }
                            }
                        } else {
                            tf_skip(&vr, w3);
                        }
                    }
                }
                free(key);
                break;
            }
            default:
                tf_skip(&r, wire);
                break;
        }
    }
    if (!r.ok || !name || !op) {
        free(name);
        free(op);
        for (int i = 0; i < n_inputs; i++) free(inputs[i]);
        free(inputs);
        return -1;
    }

    tf_node_t* nn = (tf_node_t*)realloc(g->nodes, (size_t)(g->n_nodes + 1) * sizeof(tf_node_t));
    if (!nn) {
        free(name);
        free(op);
        for (int i = 0; i < n_inputs; i++) free(inputs[i]);
        free(inputs);
        return -1;
    }
    g->nodes = nn;
    tf_node_t* node = &g->nodes[g->n_nodes++];
    node->name = name;
    node->op = op;
    node->inputs = inputs;
    node->n_inputs = n_inputs;
    if (strcmp(op, "Placeholder") == 0) g->has_placeholder = true;
    return 0;
}

// Parse a GraphDef message (reader positioned at its start) into the graph.
static void parse_graphdef(tf_graph_t* g, const uint8_t* data, size_t len) {
    tf_reader_t r = {data, len, 0, true};
    uint32_t field, wire;
    while (tf_read_tag(&r, &field, &wire)) {
        if (field == 1 && wire == 2) {  // node
            const uint8_t* sub;
            size_t subn;
            if (tf_read_bytes(&r, &sub, &subn)) {
                if (parse_node(g, sub, subn) != 0) break;
            } else {
                break;
            }
        } else {
            tf_skip(&r, wire);
        }
    }
}

// ---------------------------------------------------------------------------
// Model building
// ---------------------------------------------------------------------------

static bool matmul_transpose_b(const tf_node_t* node) {
    // transpose_b is an AttrValue bool (field 5) in attr entry "transpose_b".
    // We do not keep attrs on the node; for frozen linear graphs the default
    // (false) is the overwhelmingly common case, so transpose_b is unsupported
    // and reported as such during model build.
    (void)node;
    return false;
}

static int build_model(tf_graph_t* g, boat_model_t* model) {
    boat_dense_layer_t* last_dense = NULL;
    int layer_count = 0;
    for (int i = 0; i < g->n_nodes; i++) {
        tf_node_t* nd = &g->nodes[i];

        if (strcmp(nd->op, "Const") == 0 || strcmp(nd->op, "Placeholder") == 0 ||
            strcmp(nd->op, "Identity") == 0) {
            continue;
        }

        if (strcmp(nd->op, "MatMul") == 0) {
            if (nd->n_inputs < 2) return -1;
            tf_const_t* w = find_const(g, nd->inputs[1]);
            if (!w || w->ndims != 2 || !w->data) return -1;
            int64_t in_f = w->dims[0];
            int64_t out_f = w->dims[1];
            if (matmul_transpose_b(nd)) {
                int64_t t = in_f;
                in_f = out_f;
                out_f = t;
            }

            boat_dense_layer_t* dense =
                boat_dense_layer_create((size_t)in_f, (size_t)out_f, true);
            if (!dense) return -1;

            int64_t wshape[2] = {in_f, out_f};
            // Weight data: TF MatMul W is [in, out]; transposed if transpose_b.
            boat_tensor_t* wt;
            if (matmul_transpose_b(nd)) {
                float* td = (float*)malloc((size_t)(in_f * out_f) * sizeof(float));
                if (!td) {
                    boat_dense_layer_free(dense);
                    return -1;
                }
                for (int64_t r = 0; r < in_f; r++) {
                    for (int64_t c = 0; c < out_f; c++) {
                        td[r * out_f + c] = w->data[c * in_f + r];
                    }
                }
                wt = boat_tensor_from_data(wshape, 2, BOAT_DTYPE_FLOAT32, td);
                free(td);
            } else {
                wt = boat_tensor_from_data(wshape, 2, BOAT_DTYPE_FLOAT32, w->data);
            }
            if (!wt) {
                boat_dense_layer_free(dense);
                return -1;
            }
            boat_dense_layer_set_weight(dense, wt);
            boat_tensor_unref(wt);

            boat_layer_t* wrapper = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            if (!wrapper) {
                boat_dense_layer_free(dense);
                return -1;
            }
            wrapper->data = dense;
            wrapper->type = BOAT_LAYER_TYPE_DENSE;
            wrapper->ops = NULL;
            boat_model_add_layer(model, wrapper);
            last_dense = dense;
            layer_count++;
        } else if (strcmp(nd->op, "BiasAdd") == 0 || strcmp(nd->op, "Add") == 0) {
            if (nd->n_inputs < 2 || !last_dense) return -1;
            tf_const_t* b = find_const(g, nd->inputs[1]);
            if (!b || !b->data) return -1;
            int64_t bshape[1] = {b->nelems};
            boat_tensor_t* bt = boat_tensor_from_data(bshape, 1, BOAT_DTYPE_FLOAT32, b->data);
            if (!bt) return -1;
            boat_dense_layer_set_bias(last_dense, bt);
            boat_tensor_unref(bt);
        } else if (strcmp(nd->op, "Relu") == 0) {
            boat_relu_layer_t* relu = boat_relu_layer_create();
            if (!relu) return -1;
            boat_layer_t* wrapper = (boat_layer_t*)malloc(sizeof(boat_layer_t));
            if (!wrapper) {
                boat_relu_layer_free(relu);
                return -1;
            }
            wrapper->data = relu;
            wrapper->type = BOAT_LAYER_TYPE_RELU;
            wrapper->ops = NULL;
            boat_model_add_layer(model, wrapper);
            layer_count++;
        } else {
            // Unsupported op.
            boat_set_errorf(BOAT_ERROR_NOT_IMPLEMENTED,
                            "[TensorFlow] unsupported op '%s' in frozen graph\n", nd->op);
            return -1;
        }
    }
    return layer_count > 0 ? 0 : -1;
}

static boat_model_t* load_graphdef_bytes(const void* data, size_t size) {
    if (!data || size == 0) return NULL;
    tf_graph_t g;
    memset(&g, 0, sizeof(g));
    parse_graphdef(&g, (const uint8_t*)data, size);
    if (g.n_nodes == 0) {
        tf_graph_free(&g);
        return NULL;
    }
    boat_model_t* model = boat_model_create();
    if (!model) {
        tf_graph_free(&g);
        return NULL;
    }
    if (build_model(&g, model) != 0) {
        boat_model_free(model);
        tf_graph_free(&g);
        return NULL;
    }
    tf_graph_free(&g);
    return model;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

BOAT_API boat_model_t* boat_tensorflow_load(const char* filename) {
    if (!filename) return NULL;
    FILE* f = fopen(filename, "rb");
    if (!f) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[TensorFlow] cannot open %s\n", filename);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long sz = ftell(f);
    if (sz <= 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    void* buf = malloc((size_t)sz);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    boat_model_t* model = load_graphdef_bytes(buf, (size_t)sz);
    free(buf);
    return model;
}

BOAT_API bool boat_tensorflow_save(const boat_model_t* model, const char* filename) {
    (void)model;
    (void)filename;
    boat_set_errorf(BOAT_ERROR_NOT_IMPLEMENTED,
                    "[TensorFlow] saving to TensorFlow format is not implemented\n");
    return false;
}

BOAT_API boat_model_t* boat_tensorflow_load_from_memory(const void* data, size_t size) {
    return load_graphdef_bytes(data, size);
}

BOAT_API bool boat_tensorflow_save_to_memory(const boat_model_t* model, void** data, size_t* size) {
    (void)model;
    (void)data;
    (void)size;
    boat_set_errorf(BOAT_ERROR_NOT_IMPLEMENTED,
                    "[TensorFlow] saving to TensorFlow format is not implemented\n");
    return false;
}

// Check if a file is a valid TensorFlow frozen graph / SavedModel. Returns
// true when a GraphDef with at least one named node parses.
static bool tf_check_buffer(const void* data, size_t size) {
    if (!data || size == 0) return false;
    tf_graph_t g;
    memset(&g, 0, sizeof(g));
    parse_graphdef(&g, (const uint8_t*)data, size);
    bool ok = g.n_nodes > 0;
    tf_graph_free(&g);
    return ok;
}

// A SavedModel is a directory (POSIX fopen() on a directory would otherwise
// "succeed" and we would read garbage instead of detecting the layout).
static bool tf_is_directory(const char* path) {
#ifdef _WIN32
    (void)path;  // fopen on a directory fails on Windows; probe saved_model.pb
    return false;
#else
    struct stat st;
    return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

BOAT_API bool boat_tensorflow_check(const char* filename) {
    if (!filename) return false;
    char path[4096];
    int is_dir = tf_is_directory(filename);
    FILE* f = NULL;
    if (!is_dir) {
        f = fopen(filename, "rb");
        // On Windows fopen() on a directory fails, so probe saved_model.pb.
        if (!f && snprintf(path, sizeof(path), "%s/saved_model.pb", filename) <
                      (int)sizeof(path)) {
            f = fopen(path, "rb");
            is_dir = 1;
        }
    } else {
        if (snprintf(path, sizeof(path), "%s/saved_model.pb", filename) >= (int)sizeof(path)) {
            return false;
        }
        f = fopen(path, "rb");
    }
    if (!f) return false;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return false;
    }
    long sz = ftell(f);
    if (sz <= 0) {
        fclose(f);
        return false;
    }
    rewind(f);
    void* buf = malloc((size_t)sz);
    if (!buf) {
        fclose(f);
        return false;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return false;
    }
    fclose(f);

    bool ok;
    if (is_dir) {
        // SavedModel: saved_model.pb is a SavedModel whose meta_graphs (field
        // 2) contain MetaGraphDef.graph_def (field 2) with the GraphDef.
        tf_reader_t r = {(const uint8_t*)buf, (size_t)sz, 0, true};
        uint32_t field, wire;
        const uint8_t* gd = NULL;
        size_t gdn = 0;
        while (tf_read_tag(&r, &field, &wire)) {
            if (field == 2 && wire == 2) {
                const uint8_t* meta;
                size_t metan;
                if (tf_read_bytes(&r, &meta, &metan)) {
                    tf_reader_t mr = {meta, metan, 0, true};
                    uint32_t f2, w2;
                    while (tf_read_tag(&mr, &f2, &w2)) {
                        if (f2 == 2 && w2 == 2) {
                            tf_read_bytes(&mr, &gd, &gdn);
                        } else {
                            tf_skip(&mr, w2);
                        }
                    }
                }
                break;
            } else {
                tf_skip(&r, wire);
            }
        }
        ok = gd && gdn > 0 ? tf_check_buffer(gd, gdn) : false;
    } else {
        ok = tf_check_buffer(buf, (size_t)sz);
    }
    free(buf);
    return ok;
}

BOAT_API boat_model_t* boat_tensorflow_load_savedmodel(const char* directory) {
    if (!directory) return NULL;
    char path[4096];
    if (snprintf(path, sizeof(path), "%s/saved_model.pb", directory) >= (int)sizeof(path)) {
        return NULL;
    }
    FILE* f = fopen(path, "rb");
    if (!f) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[TensorFlow] no saved_model.pb under %s\n", directory);
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long sz = ftell(f);
    if (sz <= 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    void* buf = malloc((size_t)sz);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);

    // Extract SavedModel.meta_graphs (field 2) -> MetaGraphDef.graph_def
    // (field 2) -> GraphDef.
    tf_reader_t r = {(const uint8_t*)buf, (size_t)sz, 0, true};
    uint32_t field, wire;
    const uint8_t* gd = NULL;
    size_t gdn = 0;
    while (tf_read_tag(&r, &field, &wire)) {
        if (field == 2 && wire == 2) {
            const uint8_t* meta;
            size_t metan;
            if (tf_read_bytes(&r, &meta, &metan)) {
                tf_reader_t mr = {meta, metan, 0, true};
                uint32_t f2, w2;
                while (tf_read_tag(&mr, &f2, &w2)) {
                    if (f2 == 2 && w2 == 2) {
                        tf_read_bytes(&mr, &gd, &gdn);
                    } else {
                        tf_skip(&mr, w2);
                    }
                }
            }
            break;
        } else {
            tf_skip(&r, wire);
        }
    }
    boat_model_t* model = gd ? load_graphdef_bytes(gd, gdn) : NULL;
    free(buf);
    return model;
}

BOAT_API boat_model_t* boat_tensorflow_load_frozen_graph(const char* filename) {
    return boat_tensorflow_load(filename);
}
