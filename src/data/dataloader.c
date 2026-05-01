// dataloader.c - DataLoader: batching, shuffling, and transform pipeline
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#define BOAT_BUILDING_DLL
#include <boat/data.h>
#include <boat.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ---------------------------------------------------------------------------
// Simple RNG (xorshift32, no external dependency)
// ---------------------------------------------------------------------------
static uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

struct boat_dataloader_t {
    boat_dataset_t* dataset;
    size_t batch_size;
    bool shuffle;

    size_t* indices;          // shuffled index array
    size_t num_samples;       // total samples in dataset
    size_t num_batches;       // batches per epoch
    size_t current_batch;     // 0-based batch index in current epoch

    boat_transform_chain_t* transform;
};

boat_dataloader_t* boat_dataloader_create(boat_dataset_t* dataset,
                                          size_t batch_size, bool shuffle) {
    if (!dataset) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
            "[DataLoader] NULL dataset\n");
        return NULL;
    }
    if (batch_size == 0) batch_size = 1;

    boat_dataloader_t* loader = boat_malloc(sizeof(boat_dataloader_t), BOAT_DEVICE_CPU);
    if (!loader) return NULL;

    loader->dataset = dataset;
    loader->batch_size = batch_size;
    loader->shuffle = shuffle;
    loader->num_samples = boat_dataset_size(dataset);
    loader->current_batch = 0;
    loader->transform = NULL;

    loader->num_batches = loader->num_samples / batch_size;
    if (loader->num_samples % batch_size != 0) loader->num_batches++;

    // Allocate index array
    loader->indices = boat_malloc(sizeof(size_t) * loader->num_samples, BOAT_DEVICE_CPU);
    if (!loader->indices) {
        boat_free(loader);
        return NULL;
    }

    // Initialize sequential indices and shuffle if needed
    for (size_t i = 0; i < loader->num_samples; i++) {
        loader->indices[i] = i;
    }

    if (shuffle) {
        uint32_t seed = (uint32_t)time(NULL);
        if (seed == 0) seed = 42;
        // Fisher-Yates shuffle
        for (size_t i = loader->num_samples; i > 1; i--) {
            size_t j = (size_t)(xorshift32(&seed) % i);
            size_t tmp = loader->indices[i - 1];
            loader->indices[i - 1] = loader->indices[j];
            loader->indices[j] = tmp;
        }
    }

    return loader;
}

void boat_dataloader_free(boat_dataloader_t* loader) {
    if (!loader) return;
    boat_free(loader->indices);
    boat_free(loader);
}

void boat_dataloader_reset(boat_dataloader_t* loader) {
    if (!loader) return;

    loader->current_batch = 0;

    if (loader->shuffle) {
        uint32_t seed = (uint32_t)time(NULL);
        if (seed == 0) seed = 42;
        for (size_t i = loader->num_samples; i > 1; i--) {
            size_t j = (size_t)(xorshift32(&seed) % i);
            size_t tmp = loader->indices[i - 1];
            loader->indices[i - 1] = loader->indices[j];
            loader->indices[j] = tmp;
        }
    }
}

bool boat_dataloader_next(boat_dataloader_t* loader,
                          boat_tensor_t** batch_data,
                          boat_tensor_t** batch_labels) {
    if (!loader || !batch_data || !batch_labels) return false;

    if (loader->current_batch >= loader->num_batches) return false;

    size_t start_idx = loader->current_batch * loader->batch_size;
    size_t current_batch_size = loader->batch_size;
    if (start_idx + current_batch_size > loader->num_samples) {
        current_batch_size = loader->num_samples - start_idx;
    }

    // Get a sample to determine shapes
    size_t sample_idx = loader->indices[start_idx];
    boat_tensor_t* first_data = boat_dataset_get_data(loader->dataset, sample_idx);
    if (!first_data) return false;

    const int64_t* sample_shape = boat_tensor_shape(first_data);
    size_t sample_ndim = boat_tensor_ndim(first_data);
    boat_dtype_t data_dtype = boat_tensor_dtype(first_data);

    // Copy shape into locals before unreffing first_data
    size_t sample_elements = 1;
    for (size_t i = 0; i < sample_ndim; i++) {
        sample_elements *= (size_t)sample_shape[i];
    }

    // Build batch shape: [batch_size, ...sample_shape]
    int64_t* batch_shape = boat_malloc(sizeof(int64_t) * (sample_ndim + 1), BOAT_DEVICE_CPU);
    if (!batch_shape) {
        boat_tensor_unref(first_data);
        return false;
    }
    batch_shape[0] = (int64_t)current_batch_size;
    for (size_t i = 0; i < sample_ndim; i++) {
        batch_shape[i + 1] = sample_shape[i];
    }

    boat_tensor_unref(first_data);

    // Create batch tensors
    *batch_data = boat_tensor_create(batch_shape, sample_ndim + 1, data_dtype, BOAT_DEVICE_CPU);
    *batch_labels = boat_tensor_create(
        (int64_t[]){ (int64_t)current_batch_size }, 1, BOAT_DTYPE_INT64, BOAT_DEVICE_CPU);

    if (!*batch_data || !*batch_labels) {
        boat_free(batch_shape);
        boat_tensor_unref(*batch_data);
        boat_tensor_unref(*batch_labels);
        *batch_data = NULL;
        *batch_labels = NULL;
        return false;
    }

    // Fill batch
    size_t elem_size = boat_dtype_size(data_dtype);
    size_t sample_bytes = sample_elements * elem_size;

    char* data_ptr = (char*)boat_tensor_data(*batch_data);
    int64_t* label_ptr = (int64_t*)boat_tensor_data(*batch_labels);

    for (size_t i = 0; i < current_batch_size; i++) {
        size_t idx = loader->indices[start_idx + i];

        boat_tensor_t* item_data = boat_dataset_get_data(loader->dataset, idx);
        boat_tensor_t* item_label = boat_dataset_get_label(loader->dataset, idx);

        if (!item_data || !item_label) {
            // Skip this sample — fill with zeros
            memset(data_ptr + i * sample_bytes, 0, sample_bytes);
            label_ptr[i] = -1;
            boat_tensor_unref(item_data);
            boat_tensor_unref(item_label);
            continue;
        }

        // Apply transform if set
        if (loader->transform) {
            boat_tensor_t* transformed = boat_transform_chain_apply(
                loader->transform, item_data);
            if (transformed != item_data) {
                boat_tensor_unref(item_data);
                item_data = transformed;
            }
        }

        // Copy data
        if (item_data) {
            memcpy(data_ptr + i * sample_bytes,
                   boat_tensor_const_data(item_data), sample_bytes);
            boat_tensor_unref(item_data);
        }

        // Copy label
        if (item_label) {
            const int64_t* ld = (const int64_t*)boat_tensor_const_data(item_label);
            label_ptr[i] = ld[0];
            boat_tensor_unref(item_label);
        }
    }

    boat_free(batch_shape);
    loader->current_batch++;

    return true;
}

size_t boat_dataloader_num_batches(const boat_dataloader_t* loader) {
    return loader ? loader->num_batches : 0;
}

size_t boat_dataloader_batch_size(const boat_dataloader_t* loader) {
    return loader ? loader->batch_size : 0;
}

void boat_dataloader_set_transform(boat_dataloader_t* loader,
                                   boat_transform_chain_t* transform) {
    if (loader) {
        loader->transform = transform;
    }
}

size_t boat_dataloader_current_batch_idx(const boat_dataloader_t* loader) {
    return loader ? loader->current_batch : 0;
}
