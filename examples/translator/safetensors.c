// safetensors.c - Minimal safetensors weight reader
#include "safetensors.h"
#include "json.h"
#include <boat/memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

int safetensors_open(safetensors_t* st, const char* filename) {
    memset(st, 0, sizeof(*st));

    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "[ERROR] Cannot open safetensors file: %s\n", filename);
        return 0;
    }

    // Get file size
    fseek(f, 0, SEEK_END);
    st->file_size = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);

    // Read entire file into memory
    st->file_data = (uint8_t*)malloc(st->file_size);
    if (!st->file_data) {
        fclose(f);
        fprintf(stderr, "[ERROR] Out of memory reading safetensors file\n");
        return 0;
    }
    if (fread(st->file_data, 1, st->file_size, f) != st->file_size) {
        fclose(f);
        fprintf(stderr, "[ERROR] Failed to read safetensors file\n");
        free(st->file_data);
        st->file_data = NULL;
        return 0;
    }
    fclose(f);

    // Read header length (8 bytes, little-endian uint64)
    if (st->file_size < 8) {
        fprintf(stderr, "[ERROR] Truncated safetensors file\n");
        free(st->file_data);
        st->file_data = NULL;
        return 0;
    }
    uint64_t header_len = 0;
    for (int i = 0; i < 8; i++) header_len |= ((uint64_t)st->file_data[i]) << (i * 8);
    st->header_size = (size_t)header_len + 8;

    if (st->file_size < st->header_size) {
        fprintf(stderr, "[ERROR] Truncated safetensors header\n");
        free(st->file_data);
        st->file_data = NULL;
        return 0;
    }

    // Parse JSON header
    char* json_str = (char*)(st->file_data + 8);
    size_t json_len = (size_t)header_len;

    json_ctx_t jctx;
    json_init(&jctx, json_str, json_len);

    if (json_next(&jctx) != '{') {
        fprintf(stderr, "[ERROR] Invalid safetensors header (not an object)\n");
        free(st->file_data);
        st->file_data = NULL;
        return 0;
    }

    // Count tensors by scanning
    int capacity = 64;
    st->tensors = (st_tensor_info_t*)malloc(sizeof(st_tensor_info_t) * capacity);
    st->count = 0;

    while (1) {
        json_skip_ws(&jctx);
        if (jctx.pos >= json_len || json_str[jctx.pos] == '}') break;

        // Parse tensor name
        char* name = json_parse_string(&jctx);
        if (!name) break;
        json_skip_ws(&jctx);
        if (!json_expect(&jctx, ':')) {
            free(name);
            break;
        }

        // Parse tensor info object
        if (json_next(&jctx) != '{') {
            free(name);
            break;
        }

        st_tensor_info_t info;
        memset(&info, 0, sizeof(info));
        info.name = name;

        // Read dtype, shape, data_offsets by scanning all keys in this object
        for (int k = 0; k < 10; k++) {
            json_skip_ws(&jctx);
            if (jctx.pos >= json_len || json_str[jctx.pos] == '}') break;

            char* kname = json_parse_string(&jctx);
            if (!kname) break;
            json_skip_ws(&jctx);
            json_expect(&jctx, ':');

            if (strcmp(kname, "dtype") == 0) {
                char* dtype_str = json_parse_string(&jctx);
                free(dtype_str);
            } else if (strcmp(kname, "shape") == 0) {
                if (json_next(&jctx) == '[') {
                    info.shape = NULL;
                    info.ndim = 0;
                    int shape_cap = 0;
                    while (1) {
                        json_skip_ws(&jctx);
                        if (jctx.pos < json_len && json_str[jctx.pos] == ']') {
                            jctx.pos++;
                            break;
                        }
                        if (info.ndim >= shape_cap) {
                            shape_cap = shape_cap == 0 ? 4 : shape_cap * 2;
                            int64_t* tmp = (int64_t*)realloc(info.shape, sizeof(int64_t) * shape_cap);
                            if (!tmp) { free(kname); break; }
                            info.shape = tmp;
                        }
                        info.shape[info.ndim++] = json_parse_int(&jctx);
                        json_skip_ws(&jctx);
                        if (jctx.pos < json_len && json_str[jctx.pos] == ',') jctx.pos++;
                    }
                }
            } else if (strcmp(kname, "data_offsets") == 0) {
                if (json_next(&jctx) == '[') {
                    uint64_t start = (uint64_t)json_parse_int(&jctx);
                    json_skip_ws(&jctx);
                    if (jctx.pos < json_len && json_str[jctx.pos] == ',') jctx.pos++;
                    uint64_t end = (uint64_t)json_parse_int(&jctx);
                    json_skip_ws(&jctx);
                    if (jctx.pos < json_len && json_str[jctx.pos] == ']') jctx.pos++;
                    info.data_offset = (size_t)start;
                    info.data_size = (size_t)(end - start);
                }
            } else {
                json_skip_value(&jctx);
            }

            free(kname);
            json_skip_ws(&jctx);
            if (jctx.pos < json_len && json_str[jctx.pos] == ',') jctx.pos++;
        }

        // Skip to end of this tensor's object
        json_skip_ws(&jctx);
        if (jctx.pos < json_len && json_str[jctx.pos] == '}') jctx.pos++;

        // Add to list
        if (st->count >= capacity) {
            capacity *= 2;
            st->tensors = (st_tensor_info_t*)realloc(st->tensors, sizeof(st_tensor_info_t) * capacity);
        }
        st->tensors[st->count++] = info;

        json_skip_ws(&jctx);
        if (jctx.pos < json_len && json_str[jctx.pos] == ',') jctx.pos++;
    }

    return 1;
}

int safetensors_find(const safetensors_t* st, const char* name) {
    for (int i = 0; i < st->count; i++) {
        if (strcmp(st->tensors[i].name, name) == 0) return i;
    }
    return -1;
}

boat_tensor_t* safetensors_load_tensor(const safetensors_t* st, int idx, int do_transpose) {
    if (idx < 0 || idx >= st->count) return NULL;
    st_tensor_info_t* info = &st->tensors[idx];

    // Calculate total elements
    size_t total = 1;
    for (int i = 0; i < info->ndim; i++) total *= (size_t)info->shape[i];

    // Source data in file
    float* src = (float*)(st->file_data + st->header_size + info->data_offset);

    if (do_transpose && info->ndim == 2) {
        // Transpose [out_features, in_features] -> [in_features, out_features]
        int64_t rows = info->shape[0];  // out_features
        int64_t cols = info->shape[1];  // in_features
        int64_t tshape[] = { cols, rows };
        boat_tensor_t* t = boat_tensor_create(tshape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (!t) return NULL;
        float* dst = (float*)boat_tensor_data(t);
        for (int64_t i = 0; i < rows; i++) {
            for (int64_t j = 0; j < cols; j++) {
                dst[j * rows + i] = src[i * cols + j];
            }
        }
        return t;
    } else {
        // Direct copy
        boat_tensor_t* t = boat_tensor_from_data(info->shape, (size_t)info->ndim, BOAT_DTYPE_FLOAT32, src);
        return t;
    }
}

void safetensors_close(safetensors_t* st) {
    if (!st) return;
    for (int i = 0; i < st->count; i++) {
        free(st->tensors[i].name);
        free(st->tensors[i].shape);
    }
    free(st->tensors);
    free(st->file_data);
    st->file_data = NULL;
    st->tensors = NULL;
    st->count = 0;
}
