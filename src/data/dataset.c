// dataset.c - Tensor dataset implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#define BOAT_BUILDING_DLL
#include <boat/data.h>
#include <boat.h>
#include <stdlib.h>
#include <string.h>

struct boat_dataset_t {
    boat_tensor_t* data;    // [N, ...] samples
    boat_tensor_t* labels;  // [N] or [N, 1] labels
};

boat_dataset_t* boat_tensor_dataset_create(boat_tensor_t* data, boat_tensor_t* labels) {
    if (!data || !labels) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
            "[Dataset] NULL data or labels\n");
        return NULL;
    }

    size_t n_data = boat_tensor_nelements(data);
    if (boat_tensor_ndim(data) == 0) n_data = 0;
    else {
        const int64_t* sh = boat_tensor_shape(data);
        n_data = (size_t)sh[0];
    }

    size_t n_labels = boat_tensor_nelements(labels);
    if (n_labels != n_data) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
            "[Dataset] data size (%zu) != labels size (%zu)\n", n_data, n_labels);
        return NULL;
    }

    boat_dataset_t* ds = boat_malloc(sizeof(boat_dataset_t), BOAT_DEVICE_CPU);
    if (!ds) return NULL;

    ds->data = data;
    ds->labels = labels;
    boat_tensor_ref(data);
    boat_tensor_ref(labels);

    return ds;
}

void boat_dataset_free(boat_dataset_t* dataset) {
    if (!dataset) return;
    boat_tensor_unref(dataset->data);
    boat_tensor_unref(dataset->labels);
    boat_free(dataset);
}

size_t boat_dataset_size(const boat_dataset_t* dataset) {
    if (!dataset) return 0;
    if (boat_tensor_ndim(dataset->data) == 0) return 0;
    return (size_t)boat_tensor_shape(dataset->data)[0];
}

boat_tensor_t* boat_dataset_get_data(const boat_dataset_t* dataset, size_t index) {
    if (!dataset) return NULL;

    size_t ndim = boat_tensor_ndim(dataset->data);
    if (ndim == 0) return NULL;

    size_t n = (size_t)boat_tensor_shape(dataset->data)[0];
    if (index >= n) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
            "[Dataset] index %zu out of bounds (size %zu)\n", index, n);
        return NULL;
    }

    // Build a slice for this single sample: shape[1:]
    size_t sample_ndim = ndim - 1;
    if (sample_ndim == 0) {
        // 1-D dataset: each sample is a scalar
        int64_t scalar_shape[] = {1};
        boat_tensor_t* out = boat_tensor_create(scalar_shape, 1,
            boat_tensor_dtype(dataset->data), boat_tensor_device(dataset->data));
        if (!out) return NULL;

        size_t elem_size = boat_dtype_size(boat_tensor_dtype(dataset->data));
        const char* src = (const char*)boat_tensor_const_data(dataset->data) + index * elem_size;
        memcpy(boat_tensor_data(out), src, elem_size);
        return out;
    }

    // Use slice: data[index:index+1]
    size_t* start = boat_malloc(sizeof(size_t) * ndim, BOAT_DEVICE_CPU);
    size_t* end = boat_malloc(sizeof(size_t) * ndim, BOAT_DEVICE_CPU);
    if (!start || !end) {
        boat_free(start);
        boat_free(end);
        return NULL;
    }

    const int64_t* shape = boat_tensor_shape(dataset->data);
    start[0] = index;
    end[0] = index + 1;
    for (size_t i = 1; i < ndim; i++) {
        start[i] = 0;
        end[i] = (size_t)shape[i];
    }

    boat_tensor_t* slice = boat_tensor_slice(dataset->data, start, end, NULL);
    boat_free(start);
    boat_free(end);

    if (!slice) return NULL;

    // Reshape from [1, ...] to [...]
    int64_t* sample_shape = boat_malloc(sizeof(int64_t) * sample_ndim, BOAT_DEVICE_CPU);
    if (!sample_shape) {
        boat_tensor_unref(slice);
        return NULL;
    }
    for (size_t i = 0; i < sample_ndim; i++) {
        sample_shape[i] = shape[i + 1];
    }

    boat_tensor_t* result = boat_tensor_reshape(slice, sample_shape, sample_ndim);
    boat_free(sample_shape);
    boat_tensor_unref(slice);

    return result;
}

boat_tensor_t* boat_dataset_get_label(const boat_dataset_t* dataset, size_t index) {
    if (!dataset) return NULL;

    size_t n = boat_dataset_size(dataset);
    if (index >= n) return NULL;

    // Get label value as scalar int64
    boat_dtype_t label_dtype = boat_tensor_dtype(dataset->labels);
    const void* label_data = boat_tensor_const_data(dataset->labels);
    size_t elem_size = boat_dtype_size(label_dtype);

    // Support flat [N] labels and [N, 1] labels
    const int64_t* lshape = boat_tensor_shape(dataset->labels);
    size_t label_ndim = boat_tensor_ndim(dataset->labels);
    size_t label_offset = index;
    if (label_ndim > 1) {
        // For multi-dim labels, compute offset
        size_t stride = 1;
        for (size_t i = label_ndim - 1; i > 0; i--) stride *= (size_t)lshape[i];
        label_offset = index * stride;
    }

    int64_t scalar_val = 0;
    const char* src = (const char*)label_data + label_offset * elem_size;

    switch (label_dtype) {
        case BOAT_DTYPE_FLOAT32: scalar_val = (int64_t)(*(const float*)src); break;
        case BOAT_DTYPE_FLOAT64: scalar_val = (int64_t)(*(const double*)src); break;
        case BOAT_DTYPE_INT64:   scalar_val = *(const int64_t*)src; break;
        case BOAT_DTYPE_INT32:   scalar_val = *(const int32_t*)src; break;
        case BOAT_DTYPE_INT8:    scalar_val = *(const int8_t*)src; break;
        case BOAT_DTYPE_UINT8:   scalar_val = *(const uint8_t*)src; break;
        default:
            // Fallback: read as INT64
            memcpy(&scalar_val, src, elem_size < sizeof(int64_t) ? elem_size : sizeof(int64_t));
            break;
    }

    int64_t label_shape[] = {1};
    boat_tensor_t* label_tensor = boat_tensor_from_data(label_shape, 1,
        BOAT_DTYPE_INT64, &scalar_val);
    return label_tensor;
}
