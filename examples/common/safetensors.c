// safetensors.c - Minimal safetensors weight reader with BF16 to FP32 conversion
#include "safetensors.h"
#include "json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#ifdef _WIN32
    #define ftell64 _ftelli64
    #define fseek64 _fseeki64
#else
    #define ftell64 ftello
    #define fseek64 fseeko
#endif

int safetensors_open(safetensors_t* st, const char* filename) {
    memset(st, 0, sizeof(*st));
    FILE* f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot open safetensors: %s\n", filename); return 0; }
    fseek64(f, 0, SEEK_END);
    int64_t fs = ftell64(f);
    if (fs < 0) { fprintf(stderr, "[ERROR] ftell64 failed\n"); fclose(f); return 0; }
    st->file_size = (size_t)fs;
    fseek64(f, 0, SEEK_SET);
    st->file_data = (uint8_t*)malloc(st->file_size);
    if (!st->file_data) { fprintf(stderr, "[ERROR] Out of memory (%zu bytes)\n", st->file_size); fclose(f); return 0; }
    if (fread(st->file_data, 1, st->file_size, f) != st->file_size) { free(st->file_data); fclose(f); return 0; }
    fclose(f);

    if (st->file_size < 8) { free(st->file_data); return 0; }
    uint64_t header_len = 0;
    for (int i = 0; i < 8; i++) header_len |= ((uint64_t)st->file_data[i]) << (i * 8);
    st->header_size = (size_t)header_len + 8;
    if (st->file_size < st->header_size) { free(st->file_data); return 0; }

    char* json_str = (char*)(st->file_data + 8);
    size_t json_len = (size_t)header_len;
    json_ctx_t jctx;
    json_init(&jctx, json_str, json_len);
    if (json_next(&jctx) != '{') { free(st->file_data); return 0; }

    int capacity = 128;
    st->tensors = (st_tensor_info_t*)malloc(sizeof(st_tensor_info_t) * capacity);
    st->count = 0;

    while (1) {
        json_skip_ws(&jctx);
        if (jctx.pos >= json_len || json_str[jctx.pos] == '}') break;
        char* name = json_parse_string(&jctx);
        if (!name) break;
        json_skip_ws(&jctx);
        if (!json_expect(&jctx, ':')) { free(name); break; }
        if (json_next(&jctx) != '{') { free(name); break; }

        st_tensor_info_t info;
        memset(&info, 0, sizeof(info));
        info.name = name;

        for (int k = 0; k < 10; k++) {
            json_skip_ws(&jctx);
            if (jctx.pos >= json_len || json_str[jctx.pos] == '}') break;
            char* kname = json_parse_string(&jctx);
            if (!kname) break;
            json_skip_ws(&jctx);
            json_expect(&jctx, ':');

            if (strcmp(kname, "dtype") == 0) {
                char* dtype_str = json_parse_string(&jctx);
                if (dtype_str) { strncpy(info.dtype, dtype_str, sizeof(info.dtype) - 1); free(dtype_str); }
            } else if (strcmp(kname, "shape") == 0) {
                if (json_next(&jctx) == '[') {
                    info.shape = NULL; info.ndim = 0;
                    int shape_cap = 0;
                    while (1) {
                        json_skip_ws(&jctx);
                        if (jctx.pos < json_len && json_str[jctx.pos] == ']') { jctx.pos++; break; }
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
            } else { json_skip_value(&jctx); }

            free(kname);
            json_skip_ws(&jctx);
            if (jctx.pos < json_len && json_str[jctx.pos] == ',') jctx.pos++;
        }
        json_skip_ws(&jctx);
        if (jctx.pos < json_len && json_str[jctx.pos] == '}') jctx.pos++;

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
    for (int i = 0; i < st->count; i++)
        if (strcmp(st->tensors[i].name, name) == 0) return i;
    return -1;
}

static void bf16_to_fp32_batch(const uint16_t* src, float* dst, size_t n) {
    for (size_t i = 0; i < n; i++) dst[i] = boat_bf16_to_f32(src[i]);
}

static int dtype_is_bf16(const char* dtype) {
    return (strcmp(dtype, "BF16") == 0 || strcmp(dtype, "bfloat16") == 0 ||
            strcmp(dtype, "BFloat16") == 0 || strcmp(dtype, "bf16") == 0);
}

static int dtype_is_i64(const char* dtype) {
    return (strcmp(dtype, "I64") == 0 || strcmp(dtype, "INT64") == 0 ||
            strcmp(dtype, "int64") == 0 || strcmp(dtype, "Int64") == 0);
}

boat_tensor_t* safetensors_load_tensor(const safetensors_t* st, int idx, int do_transpose) {
    if (idx < 0 || idx >= st->count) return NULL;
    st_tensor_info_t* info = &st->tensors[idx];
    int need_bf16_convert = dtype_is_bf16(info->dtype);
    int need_i64 = dtype_is_i64(info->dtype);

    size_t total = 1;
    for (int i = 0; i < info->ndim; i++) total *= (size_t)info->shape[i];

    void* src_ptr = (void*)(st->file_data + st->header_size + info->data_offset);
    int elem_size = need_bf16_convert ? 2 : (need_i64 ? 8 : 4);

    if (do_transpose && info->ndim == 2) {
        int64_t rows = info->shape[0];
        int64_t cols = info->shape[1];
        int64_t tshape[] = { cols, rows };
        boat_tensor_t* t = boat_tensor_create(tshape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (!t) return NULL;
        float* dst = (float*)boat_tensor_data(t);
        if (need_bf16_convert) {
            float* bf16_row = (float*)malloc(cols * 4);
            for (int64_t i = 0; i < rows; i++) {
                bf16_to_fp32_batch((uint16_t*)src_ptr + i * cols, bf16_row, cols);
                for (int64_t j = 0; j < cols; j++)
                    dst[j * rows + i] = bf16_row[j];
            }
            free(bf16_row);
        } else {
            float* src = (float*)src_ptr;
            for (int64_t i = 0; i < rows; i++)
                for (int64_t j = 0; j < cols; j++)
                    dst[j * rows + i] = src[i * cols + j];
        }
        return t;
    }

    boat_dtype_t tensor_dtype = need_i64 ? BOAT_DTYPE_INT64 : BOAT_DTYPE_FLOAT32;
    boat_tensor_t* t = boat_tensor_create(info->shape, (size_t)info->ndim, tensor_dtype, BOAT_DEVICE_CPU);
    if (!t) return NULL;
    if (need_i64) {
        int64_t* dst = (int64_t*)boat_tensor_data(t);
        memcpy(dst, src_ptr, total * 8);
    } else if (need_bf16_convert) {
        float* dst = (float*)boat_tensor_data(t);
        bf16_to_fp32_batch((uint16_t*)src_ptr, dst, total);
    } else {
        float* dst = (float*)boat_tensor_data(t);
        memcpy(dst, src_ptr, total * 4);
    }
    return t;
}

void safetensors_close(safetensors_t* st) {
    if (!st) return;
    for (int i = 0; i < st->count; i++) { free(st->tensors[i].name); free(st->tensors[i].shape); }
    free(st->tensors);
    free(st->file_data);
    memset(st, 0, sizeof(*st));
}
