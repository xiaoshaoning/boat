// tensor.c - Tensor implementation for deep learning framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#define BOAT_BUILDING_DLL
#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Internal tensor structure
struct boat_tensor_t {
    int64_t* shape;           // Array of dimensions
    size_t ndim;              // Number of dimensions
    boat_dtype_t dtype;       // Data type
    boat_device_t device;     // Device (CPU/GPU)
    void* data;               // Raw data pointer
    size_t nbytes;            // Total bytes allocated
    size_t nelements;         // Total number of elements
    bool is_contiguous;       // Whether memory is contiguous
    size_t ref_count;         // Reference count
    boat_tensor_t* parent;    // Parent tensor if this is a view (for reshape/slice)
    bool is_view;             // Whether this tensor is a view (shares data with parent)
};

// Helper functions
static size_t calculate_nelements(const int64_t* shape, size_t ndim) {
    size_t nelements = 1;
    for (size_t i = 0; i < ndim; i++) {
        nelements *= shape[i];
    }
    return nelements;
}

static size_t dtype_size(boat_dtype_t dtype) {
    switch (dtype) {
        case BOAT_DTYPE_FLOAT64: return sizeof(double);
        case BOAT_DTYPE_FLOAT32: return sizeof(float);
        case BOAT_DTYPE_FLOAT16: return 2;  // 16 bits = 2 bytes
        case BOAT_DTYPE_FLOAT8:  return 1;  // 8 bits = 1 byte
        case BOAT_DTYPE_FLOAT4:  return 1;  // 4 bits packed (2 per byte)
        case BOAT_DTYPE_INT64:   return sizeof(int64_t);
        case BOAT_DTYPE_INT32:   return sizeof(int32_t);
        case BOAT_DTYPE_UINT8:   return sizeof(uint8_t);
        case BOAT_DTYPE_BITS2:   return 1;  // 2 bits packed (4 per byte)
        case BOAT_DTYPE_BITS1:   return 1;  // 1 bit packed (8 per byte)
        case BOAT_DTYPE_BOOL:    return sizeof(bool);
        default: return 0;
    }
}

static void* allocate_memory(size_t nbytes, boat_device_t device) {
    return boat_malloc(nbytes, device);
}

static void free_memory(void* ptr, boat_device_t device) {
    boat_free(ptr);
    (void)device; // Unused parameter for now
}

// Public API implementation
BOAT_API boat_tensor_t* boat_tensor_create(const int64_t* shape, size_t ndim,
                                  boat_dtype_t dtype, boat_device_t device) {
    // Allow scalar tensors (ndim = 0)
    if (!shape && ndim > 0) {
        return NULL;
    }

    // Allocate tensor structure
    boat_tensor_t* tensor = boat_malloc(sizeof(boat_tensor_t), BOAT_DEVICE_CPU);
    if (!tensor) {
        return NULL;
    }

    // Copy shape (handle scalar tensors with ndim = 0)
    if (ndim > 0) {
        tensor->shape = boat_malloc(sizeof(int64_t) * ndim, BOAT_DEVICE_CPU);
        if (!tensor->shape) {
            boat_free(tensor);
            return NULL;
        }
        if (shape) {
            memcpy(tensor->shape, shape, sizeof(int64_t) * ndim);
        } else {
            // If shape is NULL but ndim > 0, initialize to 1
            for (size_t i = 0; i < ndim; i++) {
                tensor->shape[i] = 1;
            }
        }
    } else {
        // Scalar tensor: shape is NULL
        tensor->shape = NULL;
    }

    // Calculate size
    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->device = device;
    tensor->nelements = calculate_nelements(shape, ndim);
    tensor->nbytes = tensor->nelements * dtype_size(dtype);

    // Special handling for packed types
    if (dtype == BOAT_DTYPE_FLOAT4) {
        // 4 bits per element, packed 2 per byte
        tensor->nbytes = (tensor->nelements + 1) / 2;
    } else if (dtype == BOAT_DTYPE_BITS2) {
        // 2 bits per element, packed 4 per byte
        tensor->nbytes = (tensor->nelements + 3) / 4;
    } else if (dtype == BOAT_DTYPE_BITS1) {
        // 1 bit per element, packed 8 per byte
        tensor->nbytes = (tensor->nelements + 7) / 8;
    }

    // Allocate data
    tensor->data = allocate_memory(tensor->nbytes, device);
    if (!tensor->data) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    // Initialize
    tensor->is_contiguous = true;
    tensor->ref_count = 1;
    tensor->parent = NULL;
    tensor->is_view = false;

    // Zero out memory
    memset(tensor->data, 0, tensor->nbytes);

    return tensor;
}

BOAT_API boat_tensor_t* boat_tensor_from_data(const int64_t* shape, size_t ndim,
                                     boat_dtype_t dtype, const void* data) {
    boat_tensor_t* tensor = boat_tensor_create(shape, ndim, dtype, BOAT_DEVICE_CPU);
    if (!tensor) {
        return NULL;
    }

    if (data) {
        memcpy(tensor->data, data, tensor->nbytes);
    }

    return tensor;
}

BOAT_API boat_tensor_t* boat_tensor_create_like(const boat_tensor_t* other) {
    if (!other) {
        return NULL;
    }

    const int64_t* shape = boat_tensor_shape(other);
    size_t ndim = boat_tensor_ndim(other);
    boat_dtype_t dtype = boat_tensor_dtype(other);
    boat_device_t device = boat_tensor_device(other);

    return boat_tensor_create(shape, ndim, dtype, device);
}

BOAT_API void boat_tensor_free(boat_tensor_t* tensor) {
    if (!tensor) return;

    if (--tensor->ref_count == 0) {
        // If this tensor is a view, it shares data with parent
        // Don't free the data, just free shape and structure
        // Decrease parent reference count if exists
        if (tensor->parent) {
            boat_tensor_unref(tensor->parent);
        }
        // Only free data if this tensor owns it (not a view)
        if (!tensor->is_view && tensor->data) {
            free_memory(tensor->data, tensor->device);
        }
        boat_free(tensor->shape);
        boat_free(tensor);
    }
}

BOAT_API void boat_tensor_ref(boat_tensor_t* tensor) {
    if (tensor) {
        tensor->ref_count++;
    }
}

BOAT_API void boat_tensor_unref(boat_tensor_t* tensor) {
    boat_tensor_free(tensor);
}

BOAT_API const int64_t* boat_tensor_shape(const boat_tensor_t* tensor) {
    return tensor ? tensor->shape : NULL;
}

BOAT_API size_t boat_tensor_ndim(const boat_tensor_t* tensor) {
    return tensor ? tensor->ndim : 0;
}

BOAT_API boat_dtype_t boat_tensor_dtype(const boat_tensor_t* tensor) {
    return tensor ? tensor->dtype : BOAT_DTYPE_FLOAT32;
}

BOAT_API boat_device_t boat_tensor_device(const boat_tensor_t* tensor) {
    return tensor ? tensor->device : BOAT_DEVICE_CPU;
}

BOAT_API size_t boat_tensor_nbytes(const boat_tensor_t* tensor) {
    return tensor ? tensor->nbytes : 0;
}

BOAT_API size_t boat_tensor_nelements(const boat_tensor_t* tensor) {
    return tensor ? tensor->nelements : 0;
}

BOAT_API bool boat_tensor_is_contiguous(const boat_tensor_t* tensor) {
    return tensor ? tensor->is_contiguous : false;
}

BOAT_API void* boat_tensor_data(const boat_tensor_t* tensor) {
    return tensor ? tensor->data : NULL;
}

BOAT_API const void* boat_tensor_const_data(const boat_tensor_t* tensor) {
    return tensor ? tensor->data : NULL;
}

// More implementations would go here...
// Due to space constraints, we implement only basic functions.

BOAT_API size_t boat_dtype_size(boat_dtype_t dtype) {
    return dtype_size(dtype);
}

BOAT_API const char* boat_dtype_name(boat_dtype_t dtype) {
    switch (dtype) {
        case BOAT_DTYPE_FLOAT64: return "float64";
        case BOAT_DTYPE_FLOAT32: return "float32";
        case BOAT_DTYPE_FLOAT16: return "float16";
        case BOAT_DTYPE_FLOAT8:  return "float8";
        case BOAT_DTYPE_FLOAT4:  return "float4";
        case BOAT_DTYPE_INT64:   return "int64";
        case BOAT_DTYPE_INT32:   return "int32";
        case BOAT_DTYPE_UINT8:   return "uint8";
        case BOAT_DTYPE_BITS2:   return "bits2";
        case BOAT_DTYPE_BITS1:   return "bits1";
        case BOAT_DTYPE_BOOL:    return "bool";
        default: return "unknown";
    }
}

BOAT_API boat_tensor_t* boat_tensor_reshape(const boat_tensor_t* tensor, const int64_t* new_shape, size_t new_ndim) {
    if (!tensor || !new_shape) {
        return NULL;
    }

    // Calculate total elements in new shape
    size_t new_nelements = 1;
    for (size_t i = 0; i < new_ndim; i++) {
        new_nelements *= new_shape[i];
    }

    // Verify element count matches
    if (new_nelements != tensor->nelements) {
        fprintf(stderr, "Error: Reshape element count mismatch: %zu != %zu\n", new_nelements, tensor->nelements);
        return NULL;
    }

    // Create new tensor structure (shallow copy)
    boat_tensor_t* new_tensor = (boat_tensor_t*)boat_malloc(sizeof(boat_tensor_t), tensor->device);
    if (!new_tensor) {
        return NULL;
    }

    // Copy shape array
    new_tensor->shape = (int64_t*)boat_malloc(sizeof(int64_t) * new_ndim, tensor->device);
    if (!new_tensor->shape) {
        boat_free(new_tensor);
        return NULL;
    }
    memcpy(new_tensor->shape, new_shape, sizeof(int64_t) * new_ndim);

    // Share data pointer (increase ref count on original tensor)
    new_tensor->data = tensor->data;
    new_tensor->nbytes = tensor->nbytes;
    new_tensor->nelements = tensor->nelements;
    new_tensor->dtype = tensor->dtype;
    new_tensor->device = tensor->device;
    new_tensor->ndim = new_ndim;
    new_tensor->is_contiguous = false; // Reshaped tensor may not be contiguous
    new_tensor->ref_count = 1;
    new_tensor->is_view = true;
    new_tensor->parent = (boat_tensor_t*)tensor; // Cast away const for internal tracking

    // Increase reference count on original tensor to keep it alive
    boat_tensor_ref(new_tensor->parent);

    return new_tensor;
}

BOAT_API boat_tensor_t* boat_tensor_slice(const boat_tensor_t* tensor, const size_t* start, const size_t* end, const size_t* step) {
    if (!tensor || !start || !end) {
        return NULL;
    }

    size_t ndim = tensor->ndim;
#if BOAT_DEBUG
    fprintf(stderr, "DEBUG boat_tensor_slice: ndim=%zu\n", ndim);
    for (size_t i = 0; i < ndim; i++) {
        fprintf(stderr, "  dim[%zu]: start=%zu, end=%zu, shape=%lld\n", i, start[i], end[i], (long long)tensor->shape[i]);
    }
#endif

    // Validate dimensions
    if (ndim == 0) {
        fprintf(stderr, "Error: Cannot slice scalar tensor\n");
        return NULL;
    }

    // Calculate new shape and validate ranges
    int64_t* new_shape = (int64_t*)boat_malloc(sizeof(int64_t) * ndim, tensor->device);
    if (!new_shape) {
        return NULL;
    }

    size_t* effective_step = (size_t*)boat_malloc(sizeof(size_t) * ndim, tensor->device);
    if (!effective_step) {
        boat_free(new_shape);
        return NULL;
    }

    // Determine step values (default to 1 if step is NULL)
    for (size_t i = 0; i < ndim; i++) {
        effective_step[i] = (step != NULL) ? step[i] : 1;
        if (effective_step[i] == 0) {
            fprintf(stderr, "Error: Step cannot be zero\n");
            boat_free(new_shape);
            boat_free(effective_step);
            return NULL;
        }
    }

    // Calculate new shape and validate indices
    for (size_t i = 0; i < ndim; i++) {
        size_t dim_size = tensor->shape[i];
        if (start[i] >= dim_size || end[i] > dim_size || start[i] > end[i]) {
            fprintf(stderr, "Error: Invalid slice range for dimension %zu: [%zu, %zu) (dim size: %zu)\n",
                    i, start[i], end[i], dim_size);
            boat_free(new_shape);
            boat_free(effective_step);
            return NULL;
        }

        // Calculate size for this dimension
        size_t dim_range = end[i] - start[i];
        size_t dim_new_size = (dim_range + effective_step[i] - 1) / effective_step[i];
        new_shape[i] = (int64_t)dim_new_size;

        if (dim_new_size == 0) {
            fprintf(stderr, "Error: Slice results in zero-size dimension %zu\n", i);
            boat_free(new_shape);
            boat_free(effective_step);
            return NULL;
        }
    }

    // Calculate offset in bytes to the start of the slice
    size_t dtype_sz = dtype_size(tensor->dtype);
    size_t offset = 0;

    // For simple case where step is 1 for all dimensions and tensor is contiguous,
    // we can compute offset using strides.
    // For now, implement simple offset calculation for contiguous tensors only.
    // TODO: Support non-contiguous tensors and non-unit steps.

    // Check if tensor is contiguous
    if (!tensor->is_contiguous) {
        fprintf(stderr, "Error: Slicing non-contiguous tensors not supported\n");
        boat_free(new_shape);
        boat_free(effective_step);
        return NULL;
    }

    // Compute offset using row-major order
    size_t stride = 1;
    offset = 0;
    for (int i = (int)ndim - 1; i >= 0; i--) {
        offset += start[i] * stride;
        stride *= tensor->shape[i];
    }
    offset *= dtype_sz;

    // Check if any step != 1
    bool has_non_unit_step = false;
    for (size_t i = 0; i < ndim; i++) {
        if (effective_step[i] != 1) {
            has_non_unit_step = true;
            break;
        }
    }

    if (has_non_unit_step) {
        fprintf(stderr, "Error: Non-unit step slicing not yet implemented\n");
        boat_free(new_shape);
        boat_free(effective_step);
        return NULL;
    }

    // Create new tensor structure (view)
    boat_tensor_t* new_tensor = (boat_tensor_t*)boat_malloc(sizeof(boat_tensor_t), tensor->device);
    if (!new_tensor) {
        boat_free(new_shape);
        boat_free(effective_step);
        return NULL;
    }

    // Copy shape array
    new_tensor->shape = new_shape; // Already allocated
    new_tensor->ndim = ndim;

    // Calculate number of elements in sliced tensor
    size_t new_nelements = 1;
    for (size_t i = 0; i < ndim; i++) {
        new_nelements *= new_shape[i];
    }
    new_tensor->nelements = new_nelements;

    // Data pointer offset
    new_tensor->data = (char*)tensor->data + offset;
    new_tensor->nbytes = new_nelements * dtype_sz;

    // Check if the slice is still contiguous
    // A slice is contiguous if it takes whole rows in row-major order
    // For simplicity, we assume non-contiguous for now
    new_tensor->is_contiguous = false;

    new_tensor->dtype = tensor->dtype;
    new_tensor->device = tensor->device;
    new_tensor->ref_count = 1;
    new_tensor->is_view = true;
    new_tensor->parent = (boat_tensor_t*)tensor; // Cast away const

    // Increase reference count on original tensor
    boat_tensor_ref(new_tensor->parent);

    // Free step array
    boat_free(effective_step);

#if BOAT_DEBUG
    fprintf(stderr, "DEBUG boat_tensor_slice: returning new tensor at %p, offset=%zu, data=%p\n",
            new_tensor, offset, new_tensor->data);
#endif

    return new_tensor;
}