// tensor.c - Tensor implementation for deep learning framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#define BOAT_BUILDING_DLL
#include <boat/tensor.h>
#include <boat/memory.h>
#include <boat.h>
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
    int64_t* strides;         // Strides in elements per dim; NULL for scalar tensors

    // Quantization parameters
    float scale;              // 0.0f means not quantized
    int32_t zero_point;       // meaningful only when scale != 0.0f

    // Per-channel quantization (NULL/0 if per-tensor)
    float* per_channel_scales;
    int32_t* per_channel_zero_points;
    size_t n_channels;
};

// Helper functions
static size_t calculate_nelements(const int64_t* shape, size_t ndim) {
    if (ndim == 0) return 1;  // a scalar tensor holds exactly one element
    if (!shape) return 0;
    size_t nelements = 1;
    for (size_t i = 0; i < ndim; i++) {
        nelements *= shape[i];
    }
    return nelements;
}

// Row-major strides for a contiguous tensor with the given shape.
static int64_t* compute_row_major_strides(const int64_t* shape, size_t ndim) {
    if (ndim == 0) return NULL;
    int64_t* strides = (int64_t*)boat_malloc(sizeof(int64_t) * ndim, BOAT_DEVICE_CPU);
    if (!strides) return NULL;
    strides[ndim - 1] = 1;
    for (int i = (int)ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// True when the given strides are exactly the row-major strides of the shape
// (i.e. the buffer is contiguous in memory).
static bool strides_are_contiguous(const int64_t* shape, const int64_t* strides, size_t ndim) {
    if (ndim == 0) return true;
    int64_t expected = 1;
    for (int i = (int)ndim - 1; i >= 0; i--) {
        if (strides[i] != expected) return false;
        expected *= shape[i];
    }
    return true;
}

static size_t dtype_size(boat_dtype_t dtype) {
    switch (dtype) {
        case BOAT_DTYPE_FLOAT64: return sizeof(double);
        case BOAT_DTYPE_FLOAT32: return sizeof(float);
        case BOAT_DTYPE_FLOAT16:  return 2;  // 16 bits = 2 bytes
        case BOAT_DTYPE_BFLOAT16: return 2;  // 16 bits = 2 bytes
        case BOAT_DTYPE_FLOAT8:   return 1;  // 8 bits = 1 byte
        case BOAT_DTYPE_FLOAT4:  return 1;  // 4 bits packed (2 per byte)
        case BOAT_DTYPE_INT64:   return sizeof(int64_t);
        case BOAT_DTYPE_INT32:   return sizeof(int32_t);
        case BOAT_DTYPE_UINT8:   return sizeof(uint8_t);
        case BOAT_DTYPE_INT8:    return sizeof(int8_t);
        case BOAT_DTYPE_BITS2:   return 1;  // 2 bits packed (4 per byte)
        case BOAT_DTYPE_BITS1:   return 1;  // 1 bit packed (8 per byte)
        case BOAT_DTYPE_BOOL:    return sizeof(bool);
        default: return 0;
    }
}

static void* allocate_memory(size_t nbytes, boat_device_t device) {
#ifdef BOAT_WITH_CUDA
    if (device == BOAT_DEVICE_CUDA) {
        return boat_memory_allocate_device(nbytes, device, NULL, 0);
    }
#endif
    return boat_malloc(nbytes, device);
}

static void free_memory(void* ptr, boat_device_t device) {
#ifdef BOAT_WITH_CUDA
    if (device == BOAT_DEVICE_CUDA) {
        boat_memory_free_device(ptr, device);
        return;
    }
#endif
    boat_free(ptr);
    (void)device;
}

// Public API implementation
BOAT_API boat_tensor_t* boat_tensor_create(const int64_t* shape, size_t ndim,
                                  boat_dtype_t dtype, boat_device_t device) {
    // Allow scalar tensors (ndim = 0)
    if (!shape && ndim > 0) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Tensor] shape is NULL but ndim > 0\n");
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
        memcpy(tensor->shape, shape, sizeof(int64_t) * ndim);
    } else {
        // Scalar tensor: shape is NULL
        tensor->shape = NULL;
    }

    // Calculate size
    tensor->ndim = ndim;
    tensor->dtype = dtype;
    tensor->device = device;
    tensor->nelements = calculate_nelements(tensor->shape, ndim);
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
        boat_free(tensor->shape);
        boat_free(tensor);
        return NULL;
    }

    // Initialize
    tensor->is_contiguous = true;
    tensor->ref_count = 1;
    tensor->parent = NULL;
    tensor->is_view = false;
    tensor->strides = compute_row_major_strides(tensor->shape, ndim);
    if (ndim > 0 && !tensor->strides) {
        free_memory(tensor->data, device);
        boat_free(tensor->shape);
        boat_free(tensor);
        return NULL;
    }
    tensor->scale = 0.0f;
    tensor->zero_point = 0;
    tensor->per_channel_scales = NULL;
    tensor->per_channel_zero_points = NULL;
    tensor->n_channels = 0;

    // Zero out memory (device-aware)
    boat_memory_set(tensor->data, 0, tensor->nbytes, device);

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
        if (tensor->parent) {
            boat_tensor_unref(tensor->parent);
        }
        if (!tensor->is_view && tensor->data) {
            free_memory(tensor->data, tensor->device);
        }
        if (tensor->per_channel_scales) {
            boat_free(tensor->per_channel_scales);
        }
        if (tensor->per_channel_zero_points) {
            boat_free(tensor->per_channel_zero_points);
        }
        if (tensor->strides) {
            boat_free(tensor->strides);
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
        case BOAT_DTYPE_FLOAT16:  return "float16";
        case BOAT_DTYPE_BFLOAT16: return "bfloat16";
        case BOAT_DTYPE_FLOAT8:  return "float8";
        case BOAT_DTYPE_FLOAT4:  return "float4";
        case BOAT_DTYPE_INT64:   return "int64";
        case BOAT_DTYPE_INT32:   return "int32";
        case BOAT_DTYPE_UINT8:   return "uint8";
        case BOAT_DTYPE_INT8:    return "int8";
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

    // Reshaping a non-contiguous view is ambiguous; materialize a contiguous
    // copy first so the returned view shares a well-defined buffer.
    if (!tensor->is_contiguous) {
        boat_tensor_t* contig = boat_tensor_contiguous(tensor);
        if (!contig) return NULL;
        boat_tensor_t* reshaped = boat_tensor_reshape(contig, new_shape, new_ndim);
        boat_tensor_unref(contig);
        return reshaped;
    }

    // Calculate total elements in new shape
    size_t new_nelements = 1;
    for (size_t i = 0; i < new_ndim; i++) {
        new_nelements *= new_shape[i];
    }

    // Verify element count matches
    if (new_nelements != tensor->nelements) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Tensor] Reshape element count mismatch: %zu != %zu\n", new_nelements, tensor->nelements);
        return NULL;
    }

    // Create new tensor structure (shallow copy)
    boat_tensor_t* new_tensor = (boat_tensor_t*)boat_malloc(sizeof(boat_tensor_t), tensor->device);
    if (!new_tensor) {
        return NULL;
    }
    memset(new_tensor, 0, sizeof(boat_tensor_t));

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
    // Same buffer as the (contiguous) source: row-major strides of the new shape.
    new_tensor->strides = compute_row_major_strides(new_shape, new_ndim);
    if (new_ndim > 0 && !new_tensor->strides) {
        boat_free(new_tensor->shape);
        boat_free(new_tensor);
        return NULL;
    }
    new_tensor->is_contiguous = true;
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
        boat_set_errorf(BOAT_ERROR_INVALID_OPERATION, "[Tensor] Cannot slice scalar tensor\n");
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
            boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Tensor] Step cannot be zero\n");
            boat_free(new_shape);
            boat_free(effective_step);
            return NULL;
        }
    }

    // Calculate new shape and validate indices
    for (size_t i = 0; i < ndim; i++) {
        size_t dim_size = tensor->shape[i];
        if (start[i] >= dim_size || end[i] > dim_size || start[i] > end[i]) {
            boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Tensor] Invalid slice range for dimension %zu: [%zu, %zu) (dim size: %zu)\n",
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
            boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[Tensor] Slice results in zero-size dimension %zu\n", i);
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
        boat_set_errorf(BOAT_ERROR_INVALID_OPERATION, "[Tensor] Slicing non-contiguous tensors not supported\n");
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
        boat_set_errorf(BOAT_ERROR_NOT_IMPLEMENTED, "[Tensor] Non-unit step slicing not yet implemented\n");
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
    memset(new_tensor, 0, sizeof(boat_tensor_t));

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

    // A unit-step slice keeps the parent's element spacing (strides); it is
    // contiguous only when those strides are the row-major strides of the
    // sliced shape (i.e. all trailing dimensions are sliced in full).
    new_tensor->strides = (int64_t*)boat_malloc(sizeof(int64_t) * ndim, BOAT_DEVICE_CPU);
    if (!new_tensor->strides) {
        boat_free(new_tensor->shape);
        boat_free(new_tensor);
        boat_free(effective_step);
        return NULL;
    }
    for (size_t i = 0; i < ndim; i++) {
        new_tensor->strides[i] = tensor->strides[i];
    }
    new_tensor->is_contiguous = strides_are_contiguous(new_tensor->shape, new_tensor->strides, ndim);

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

BOAT_API float boat_tensor_get_scale(const boat_tensor_t* tensor) {
    return tensor ? tensor->scale : 0.0f;
}

BOAT_API int32_t boat_tensor_get_zero_point(const boat_tensor_t* tensor) {
    return tensor ? tensor->zero_point : 0;
}

BOAT_API void boat_tensor_set_quant_params(boat_tensor_t* tensor, float scale, int32_t zero_point) {
    if (tensor) {
        tensor->scale = scale;
        tensor->zero_point = zero_point;
    }
}

// Per-channel quantization accessors
BOAT_API bool boat_tensor_is_per_channel(const boat_tensor_t* tensor) {
    return tensor ? (tensor->per_channel_scales != NULL && tensor->n_channels > 0) : false;
}

BOAT_API size_t boat_tensor_num_channels(const boat_tensor_t* tensor) {
    return tensor ? tensor->n_channels : 0;
}

BOAT_API const float* boat_tensor_get_scales(const boat_tensor_t* tensor) {
    return tensor ? tensor->per_channel_scales : NULL;
}

BOAT_API const int32_t* boat_tensor_get_zero_points(const boat_tensor_t* tensor) {
    return tensor ? tensor->per_channel_zero_points : NULL;
}

BOAT_API void boat_tensor_set_per_channel_quant_params(boat_tensor_t* tensor, const float* scales, const int32_t* zero_points, size_t n_channels) {
    if (!tensor) return;

    // Free existing per-channel arrays
    if (tensor->per_channel_scales) boat_free(tensor->per_channel_scales);
    if (tensor->per_channel_zero_points) boat_free(tensor->per_channel_zero_points);

    if (scales && zero_points && n_channels > 0) {
        size_t arr_size = sizeof(float) * n_channels;
        tensor->per_channel_scales = boat_malloc(arr_size, BOAT_DEVICE_CPU);
        tensor->per_channel_zero_points = boat_malloc(sizeof(int32_t) * n_channels, BOAT_DEVICE_CPU);
        if (tensor->per_channel_scales && tensor->per_channel_zero_points) {
            memcpy(tensor->per_channel_scales, scales, arr_size);
            memcpy(tensor->per_channel_zero_points, zero_points, sizeof(int32_t) * n_channels);
            tensor->n_channels = n_channels;
        } else {
            if (tensor->per_channel_scales) { boat_free(tensor->per_channel_scales); tensor->per_channel_scales = NULL; }
            if (tensor->per_channel_zero_points) { boat_free(tensor->per_channel_zero_points); tensor->per_channel_zero_points = NULL; }
            tensor->n_channels = 0;
        }
    } else {
        tensor->per_channel_scales = NULL;
        tensor->per_channel_zero_points = NULL;
        tensor->n_channels = 0;
    }
}

BOAT_API boat_tensor_t* boat_tensor_transpose(const boat_tensor_t* tensor, const size_t* perm, size_t nperm) {
    if (!tensor || !perm) return NULL;

    size_t ndim = tensor->ndim;
    if (nperm != ndim) return NULL;

    // Allocate output shape
    int64_t* out_shape = boat_malloc(ndim * sizeof(int64_t), BOAT_DEVICE_CPU);
    if (!out_shape) return NULL;

    // Compute inverse permutation
    size_t* inv_perm = boat_malloc(ndim * sizeof(size_t), BOAT_DEVICE_CPU);
    if (!inv_perm) { boat_free(out_shape); return NULL; }

    for (size_t i = 0; i < ndim; i++) {
        if (perm[i] >= ndim) { boat_free(out_shape); boat_free(inv_perm); return NULL; }
        out_shape[i] = tensor->shape[perm[i]];
        inv_perm[perm[i]] = i;
    }

    boat_tensor_t* result = boat_tensor_create(out_shape, ndim, tensor->dtype, tensor->device);
    boat_free(out_shape);
    if (!result) { boat_free(inv_perm); return NULL; }

    // Total elements
    size_t total = 1;
    for (size_t i = 0; i < ndim; i++) total *= (size_t)tensor->shape[i];
    size_t elem_size = boat_dtype_size(tensor->dtype);

    // Iterate over all elements using a flat index
    size_t* out_idx = boat_malloc(ndim * sizeof(size_t), BOAT_DEVICE_CPU);
    if (!out_idx) { boat_free(inv_perm); boat_tensor_unref(result); return NULL; }

    const uint8_t* src = (const uint8_t*)tensor->data;
    uint8_t* dst = (uint8_t*)result->data;

    for (size_t flat = 0; flat < total; flat++) {
        // Convert flat index to output indices
        size_t rem = flat;
        for (size_t i = ndim; i > 0; i--) {
            size_t dim = i - 1;
            out_idx[dim] = rem % (size_t)result->shape[dim];
            rem /= (size_t)result->shape[dim];
        }

        // Map to input indices via inverse permutation
        size_t src_flat = 0;
        size_t stride = 1;
        for (size_t i = ndim; i > 0; i--) {
            size_t dim = i - 1;
            src_flat += out_idx[inv_perm[dim]] * stride;
            stride *= (size_t)tensor->shape[dim];
        }

        memcpy(dst + flat * elem_size, src + src_flat * elem_size, elem_size);
    }

    boat_free(out_idx);
    boat_free(inv_perm);
    return result;
}

BOAT_API boat_tensor_t* boat_tensor_clone(const boat_tensor_t* tensor) {
    if (!tensor) return NULL;

    boat_tensor_t* clone = boat_tensor_create(tensor->shape, tensor->ndim,
                                              tensor->dtype, tensor->device);
    if (!clone) return NULL;

    // Copy quantization parameters
    clone->scale = tensor->scale;
    clone->zero_point = tensor->zero_point;
    if (tensor->per_channel_scales && tensor->n_channels > 0) {
        size_t arr_size = sizeof(float) * tensor->n_channels;
        clone->per_channel_scales = boat_malloc(arr_size, BOAT_DEVICE_CPU);
        clone->per_channel_zero_points = boat_malloc(sizeof(int32_t) * tensor->n_channels, BOAT_DEVICE_CPU);
        if (clone->per_channel_scales && clone->per_channel_zero_points) {
            memcpy(clone->per_channel_scales, tensor->per_channel_scales, arr_size);
            memcpy(clone->per_channel_zero_points, tensor->per_channel_zero_points,
                   sizeof(int32_t) * tensor->n_channels);
            clone->n_channels = tensor->n_channels;
        }
    }

    // Copy data
    if (tensor->nbytes > 0 && tensor->data && clone->data) {
        boat_memory_copy(clone->data, tensor->data, tensor->nbytes,
                         clone->device, tensor->device);
    }

    return clone;
}

BOAT_API boat_tensor_t* boat_tensor_to_device(const boat_tensor_t* tensor, boat_device_t dev) {
    if (!tensor) return NULL;
    if (tensor->device == dev) {
        return boat_tensor_clone(tensor);
    }

    boat_tensor_t* result = boat_tensor_create(tensor->shape, tensor->ndim,
                                               tensor->dtype, dev);
    if (!result) return NULL;

    // Copy quantization parameters
    result->scale = tensor->scale;
    result->zero_point = tensor->zero_point;
    if (tensor->per_channel_scales && tensor->n_channels > 0) {
        size_t arr_size = sizeof(float) * tensor->n_channels;
        result->per_channel_scales = boat_malloc(arr_size, BOAT_DEVICE_CPU);
        result->per_channel_zero_points = boat_malloc(sizeof(int32_t) * tensor->n_channels, BOAT_DEVICE_CPU);
        if (result->per_channel_scales && result->per_channel_zero_points) {
            memcpy(result->per_channel_scales, tensor->per_channel_scales, arr_size);
            memcpy(result->per_channel_zero_points, tensor->per_channel_zero_points,
                   sizeof(int32_t) * tensor->n_channels);
            result->n_channels = tensor->n_channels;
        }
    }

    // Copy data across devices
    if (tensor->nbytes > 0 && tensor->data && result->data) {
        boat_memory_copy(result->data, tensor->data, tensor->nbytes,
                         dev, tensor->device);
    }

    return result;
}

BOAT_API boat_tensor_t* boat_tensor_contiguous(const boat_tensor_t* tensor) {
    if (!tensor) return NULL;
    if (tensor->is_contiguous) {
        return boat_tensor_clone(tensor);
    }
    // For non-contiguous tensors, create a new contiguous copy
    boat_tensor_t* result = boat_tensor_create(tensor->shape, tensor->ndim,
                                               tensor->dtype, tensor->device);
    if (!result) return NULL;

    size_t elem_size = boat_dtype_size(tensor->dtype);
    size_t total = tensor->nelements;
    size_t ndim = tensor->ndim;

    // Copy element by element using the view's strides so non-contiguous
    // views (e.g. a slice that drops interior columns) are materialized
    // correctly instead of being flat-copied.
    const uint8_t* src = (const uint8_t*)tensor->data;
    uint8_t* dst = (uint8_t*)result->data;

    if (ndim == 0) {
        if (total > 0) {
            boat_memory_copy(dst, src, elem_size, result->device, tensor->device);
        }
        return result;
    }

    for (size_t idx = 0; idx < total; idx++) {
        size_t rem = idx;
        size_t src_off = 0;
        for (size_t i = ndim; i > 0; i--) {
            size_t dim = i - 1;
            size_t coord = rem % (size_t)tensor->shape[dim];
            rem /= (size_t)tensor->shape[dim];
            src_off += coord * (size_t)tensor->strides[dim];
        }
        boat_memory_copy(dst + idx * elem_size, src + src_off * elem_size,
                         elem_size, result->device, tensor->device);
    }

    return result;
}

BOAT_API boat_tensor_t* boat_tensor_concatenate(const boat_tensor_t** tensors, size_t n_tensors, size_t axis) {
    if (!tensors || n_tensors == 0) return NULL;

    // Validate inputs and compute output shape
    size_t ndim = tensors[0]->ndim;
    boat_dtype_t dtype = tensors[0]->dtype;
    boat_device_t device = tensors[0]->device;

    // Compute output shape
    int64_t out_shape[BOAT_MAX_DIMS];
    for (size_t i = 0; i < ndim; i++) {
        out_shape[i] = tensors[0]->shape[i];
    }

    // Make sure axis is valid
    if (axis >= ndim) return NULL;

    for (size_t t = 1; t < n_tensors; t++) {
        if (tensors[t]->ndim != ndim) return NULL;
        if (tensors[t]->dtype != dtype) return NULL;
        if (tensors[t]->device != device) return NULL;
        for (size_t i = 0; i < ndim; i++) {
            if (i != axis && tensors[t]->shape[i] != out_shape[i]) return NULL;
        }
        out_shape[axis] += tensors[t]->shape[axis];
    }

    boat_tensor_t* result = boat_tensor_create(out_shape, ndim, dtype, device);
    if (!result) return NULL;

    size_t elem_size = boat_dtype_size(dtype);
    size_t outer = 1;
    for (size_t i = 0; i < axis; i++) outer *= (size_t)out_shape[i];
    size_t inner = 1;
    for (size_t i = axis + 1; i < ndim; i++) inner *= (size_t)out_shape[i];

    // Inputs are assumed contiguous (row-major); copy block by block so
    // concatenation along any axis interleaves correctly.
    uint8_t* dst = (uint8_t*)result->data;
    size_t axis_block_bytes = (size_t)out_shape[axis] * inner * elem_size;
    size_t axis_offset = 0;

    for (size_t t = 0; t < n_tensors; t++) {
        size_t t_axis = (size_t)tensors[t]->shape[axis];
        size_t t_block_bytes = t_axis * inner * elem_size;
        const uint8_t* src_base = (const uint8_t*)tensors[t]->data;
        for (size_t o = 0; o < outer; o++) {
            uint8_t* d = dst + o * axis_block_bytes + axis_offset * inner * elem_size;
            const uint8_t* s = src_base + o * t_block_bytes;
            if (tensors[t]->data && result->data) {
                boat_memory_copy(d, s, t_block_bytes, device, tensors[t]->device);
            }
        }
        axis_offset += t_axis;
    }

    return result;
}

BOAT_API boat_tensor_t* boat_tensor_stack(const boat_tensor_t** tensors, size_t n_tensors, size_t axis) {
    if (!tensors || n_tensors == 0) return NULL;

    size_t ndim = tensors[0]->ndim;
    // Stacking adds a new dimension
    if (ndim + 1 > BOAT_MAX_DIMS) return NULL;

    // Validate all tensors have the same shape, dtype, and device
    for (size_t t = 1; t < n_tensors; t++) {
        if (tensors[t]->ndim != ndim) return NULL;
        if (tensors[t]->dtype != tensors[0]->dtype) return NULL;
        if (tensors[t]->device != tensors[0]->device) return NULL;
        for (size_t i = 0; i < ndim; i++) {
            if (tensors[t]->shape[i] != tensors[0]->shape[i]) return NULL;
        }
    }

    // Build output shape with inserted dimension
    int64_t out_shape[BOAT_MAX_DIMS];
    for (size_t i = 0; i < axis; i++) out_shape[i] = tensors[0]->shape[i];
    out_shape[axis] = (int64_t)n_tensors;
    for (size_t i = axis; i < ndim; i++) out_shape[i + 1] = tensors[0]->shape[i];

    boat_tensor_t* result = boat_tensor_create(out_shape, ndim + 1,
                                               tensors[0]->dtype, tensors[0]->device);
    if (!result) return NULL;

    size_t elem_size = boat_dtype_size(tensors[0]->dtype);
    size_t outer = 1;
    for (size_t i = 0; i < axis; i++) outer *= (size_t)tensors[0]->shape[i];
    size_t inner = 1;
    for (size_t i = axis; i < ndim; i++) inner *= (size_t)tensors[0]->shape[i];

    // Inputs are assumed contiguous; interleave each tensor's block along the
    // new (inserted) axis so stacking works for any axis.
    size_t slice_bytes = inner * elem_size;
    size_t group_bytes = (size_t)n_tensors * slice_bytes;
    uint8_t* dst = (uint8_t*)result->data;

    for (size_t o = 0; o < outer; o++) {
        for (size_t t = 0; t < n_tensors; t++) {
            const uint8_t* src = (const uint8_t*)tensors[t]->data + o * slice_bytes;
            uint8_t* d = dst + o * group_bytes + t * slice_bytes;
            if (tensors[t]->data && result->data) {
                boat_memory_copy(d, src, slice_bytes, result->device, tensors[t]->device);
            }
        }
    }

    return result;
}

// Helper: get a host-readable pointer (copies from device if needed)
static void* get_host_readable(const boat_tensor_t* tensor, void** temp_buf) {
    *temp_buf = NULL;
    if (tensor->device == BOAT_DEVICE_CPU) {
        return tensor->data;
    }
#ifdef BOAT_WITH_CUDA
    if (tensor->device == BOAT_DEVICE_CUDA && tensor->data && tensor->nbytes > 0) {
        *temp_buf = boat_malloc(tensor->nbytes, BOAT_DEVICE_CPU);
        if (*temp_buf) {
            boat_memory_copy(*temp_buf, tensor->data, tensor->nbytes,
                             BOAT_DEVICE_CPU, BOAT_DEVICE_CUDA);
            return *temp_buf;
        }
    }
#endif
    return NULL;
}

BOAT_API void boat_tensor_print(const boat_tensor_t* tensor) {
    if (!tensor) { printf("NULL tensor\n"); return; }

    void* host_buf = NULL;
    void* data = get_host_readable(tensor, &host_buf);
    if (!data) { printf("<tensor on device %d, no data>\n", (int)tensor->device); return; }

    printf("Tensor(shape=[");
    for (size_t i = 0; i < tensor->ndim; i++) {
        if (i > 0) printf(", ");
        printf("%lld", (long long)tensor->shape[i]);
    }
    printf("], dtype=%s, device=%s, data=[",
           boat_dtype_name(tensor->dtype),
           tensor->device == BOAT_DEVICE_CUDA ? "cuda" : "cpu");

    size_t print_n = tensor->nelements < 8 ? tensor->nelements : 8;
    for (size_t i = 0; i < print_n; i++) {
        if (i > 0) printf(", ");
        switch (tensor->dtype) {
            case BOAT_DTYPE_FLOAT32: printf("%f", ((float*)data)[i]); break;
            case BOAT_DTYPE_INT32:   printf("%d", ((int32_t*)data)[i]); break;
            case BOAT_DTYPE_INT64:   printf("%lld", (long long)((int64_t*)data)[i]); break;
            default: printf("?"); break;
        }
    }
    if (tensor->nelements > 8) printf(", ...");
    printf("])\n");

    if (host_buf) boat_free(host_buf);
}

BOAT_API char* boat_tensor_to_string(const boat_tensor_t* tensor) {
    if (!tensor) return NULL;

    void* host_buf = NULL;
    void* data = get_host_readable(tensor, &host_buf);
    if (!data) return NULL;

    // Very simple string representation for now
    size_t buf_size = 1024;
    char* buf = boat_malloc(buf_size, BOAT_DEVICE_CPU);
    if (!buf) { if (host_buf) boat_free(host_buf); return NULL; }

    int written = snprintf(buf, buf_size, "Tensor(shape=[");
    for (size_t i = 0; i < tensor->ndim; i++) {
        // Clamp to the remaining buffer so a truncated write cannot make
        // `buf + written` point past the allocation or underflow the size.
        if (written >= (int)buf_size) break;
        written += snprintf(buf + written, buf_size - (size_t)written,
                            "%s%lld", i > 0 ? ", " : "", (long long)tensor->shape[i]);
    }
    if (written < (int)buf_size) {
        snprintf(buf + written, buf_size - (size_t)written, "])");
    }

    if (host_buf) boat_free(host_buf);
    return buf;
}

BOAT_API bool boat_tensor_equal(const boat_tensor_t* a, const boat_tensor_t* b) {
    if (!a || !b) return false;
    if (a->ndim != b->ndim) return false;
    if (a->dtype != b->dtype) return false;
    for (size_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return false;
    }
    if (a->nelements != b->nelements) return false;

    void* host_a = NULL;
    void* host_b = NULL;
    void* data_a = get_host_readable(a, &host_a);
    void* data_b = get_host_readable(b, &host_b);
    if (!data_a || !data_b) { boat_free(host_a); boat_free(host_b); return false; }

    // Bitwise equality (memcmp): -0.0f and +0.0f are NOT equal here.
    // Use boat_tensor_allclose for tolerance-based float comparison.
    bool equal = (memcmp(data_a, data_b, a->nbytes) == 0);

    boat_free(host_a);
    boat_free(host_b);
    return equal;
}

BOAT_API bool boat_tensor_allclose(const boat_tensor_t* a, const boat_tensor_t* b, float rtol, float atol) {
    if (!a || !b) return false;
    if (a->ndim != b->ndim || a->dtype != b->dtype) return false;
    for (size_t i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) return false;
    }
    if (a->nelements != b->nelements) return false;
    if (a->dtype != BOAT_DTYPE_FLOAT32) return false; // only float32 for now

    void* host_a = NULL;
    void* host_b = NULL;
    void* data_a = get_host_readable(a, &host_a);
    void* data_b = get_host_readable(b, &host_b);
    if (!data_a || !data_b) { boat_free(host_a); boat_free(host_b); return false; }

    float* fa = (float*)data_a;
    float* fb = (float*)data_b;
    bool ok = true;
    for (size_t i = 0; i < a->nelements; i++) {
        float diff = fa[i] - fb[i];
        if (diff < 0) diff = -diff;
        float max_val = fa[i] > fb[i] ? fa[i] : fb[i];
        if (max_val < 0) max_val = -max_val;
        if (diff > atol + rtol * max_val) { ok = false; break; }
    }

    boat_free(host_a);
    boat_free(host_b);
    return ok;
}