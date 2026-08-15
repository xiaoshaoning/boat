// conv.c - Convolutional layer implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat.h>
#include <boat/cuda_runtime.h>
#include <boat/sgemm.h>
#include <boat/simd.h>
#include "../core/openmp.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Convolutional layer structure
struct boat_conv_layer_t {
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    size_t groups;
    bool use_bias;
    boat_tensor_t* weight;
    boat_tensor_t* bias;

    // Gradient accumulators for training
    boat_tensor_t* grad_weight;
    boat_tensor_t* grad_bias;

    // Cache for backward pass
    boat_tensor_t* cache_input;    // Input tensor from forward pass
    int64_t cache_input_shape[4];  // [batch, in_channels, height, width]
    int64_t cache_output_shape[4]; // [batch, out_channels, height_out, width_out]
};

// Helper function: compute gradient with respect to input
static boat_tensor_t* compute_input_gradient(const boat_conv_layer_t* layer,
                                             const boat_tensor_t* cached_input,
                                             const int64_t* input_shape,
                                             const int64_t* output_shape,
                                             const boat_tensor_t* grad_output) {
    if (!layer || !cached_input || !grad_output) {
        return NULL;
    }

    // Extract dimensions
    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    int64_t out_channels = output_shape[1];
    int64_t height_out = output_shape[2];
    int64_t width_out = output_shape[3];

    // Create gradient input tensor with same shape as input
    boat_tensor_t* grad_input =
        boat_tensor_create(input_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!grad_input) {
        return NULL;
    }

    // Get data pointers
    float* grad_input_data = (float*)boat_tensor_data(grad_input);
    const float* weight_data = (float*)boat_tensor_data(layer->weight);
    const float* grad_output_data = (float*)boat_tensor_data(grad_output);

    // Initialize gradient input with zeros
    size_t grad_input_elements = boat_tensor_nelements(grad_input);
    memset(grad_input_data, 0, grad_input_elements * sizeof(float));

    // Group convolution parameters
    size_t in_channels_per_group = layer->in_channels / layer->groups;
    size_t out_channels_per_group = layer->out_channels / layer->groups;

    // For each batch (parallel: grad_input[b] slices are disjoint)
    BOAT_OMP_PARALLEL_FOR_SCHEDULE(static)
    for (int64_t b = 0; b < batch; b++) {
        // For each group
        for (size_t g = 0; g < layer->groups; g++) {
            size_t oc_start = g * out_channels_per_group;
            size_t oc_end = oc_start + out_channels_per_group;
            size_t ic_start = g * in_channels_per_group;
            size_t ic_end = ic_start + in_channels_per_group;

            // For each output channel in this group
            for (int64_t oc = (int64_t)oc_start; oc < (int64_t)oc_end; oc++) {
                // For each input channel in this group
                for (int64_t ic = (int64_t)ic_start; ic < (int64_t)ic_end; ic++) {
                    size_t ic_local = (size_t)(ic - (int64_t)ic_start);
                    // For each kernel row
                    for (size_t kh = 0; kh < layer->kernel_size; kh++) {
                        // For each kernel column
                        for (size_t kw = 0; kw < layer->kernel_size; kw++) {
                            // For each output height position
                            for (int64_t oh = 0; oh < height_out; oh++) {
                                int64_t ih = oh * layer->stride - layer->padding + kh;
                                if (ih < 0 || ih >= height) continue;

                                // Fast path (stride 1): iw = ow - pad + kw is
                                // contiguous over ow, and the valid range is a
                                // single interval [lo, hi). The prefix/suffix
                                // positions are all out of bounds.
                                if (layer->stride == 1) {
                                    int64_t lo = (int64_t)layer->padding > (int64_t)kw
                                                     ? (int64_t)layer->padding - (int64_t)kw
                                                     : 0;
                                    int64_t hi = width + (int64_t)layer->padding - (int64_t)kw;
                                    if (hi > width_out) hi = width_out;
                                    if (lo < hi) {
                                        size_t weight_idx =
                                            ((oc * in_channels_per_group + ic_local) *
                                                 layer->kernel_size +
                                             kh) *
                                                layer->kernel_size +
                                            kw;
                                        size_t gbase =
                                            ((size_t)(b * in_channels + ic) * (size_t)height +
                                             (size_t)ih) *
                                            (size_t)width;
                                        size_t obase =
                                            ((size_t)(b * out_channels + oc) * (size_t)height_out +
                                             (size_t)oh) *
                                            (size_t)width_out;
                                        // iw = ow - pad + kw: the input column for
                                        // ow = lo is (lo - pad + kw) >= 0 by the
                                        // range definition, and the input side is
                                        // shifted by (kw - pad) vs the output side.
                                        size_t col_start =
                                            (size_t)(lo - (int64_t)layer->padding + (int64_t)kw);
                                        boat_simd_axpy_f32(grad_input_data + gbase + col_start,
                                                           grad_output_data + obase + lo,
                                                           weight_data[weight_idx],
                                                           (size_t)(hi - lo));
                                    }
                                    continue;
                                }

                                for (int64_t ow = 0; ow < width_out; ow++) {
                                    int64_t iw = ow * layer->stride - layer->padding + kw;
                                    if (iw < 0 || iw >= width) continue;

                                    // grad_input uses same weights as forward (no 180-degree
                                    // rotation)
                                    size_t weight_idx = ((oc * in_channels_per_group + ic_local) *
                                                             layer->kernel_size +
                                                         kh) *
                                                            layer->kernel_size +
                                                        kw;
                                    size_t grad_output_idx =
                                        ((b * out_channels + oc) * height_out + oh) * width_out +
                                        ow;
                                    size_t grad_input_idx =
                                        ((b * in_channels + ic) * height + ih) * width + iw;

                                    grad_input_data[grad_input_idx] +=
                                        weight_data[weight_idx] * grad_output_data[grad_output_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

// Helper function: compute gradient with respect to weights

static bool compute_weight_gradient_into(const boat_conv_layer_t* layer,
                                         const boat_tensor_t* cached_input,
                                         const int64_t* input_shape, const int64_t* output_shape,
                                         const boat_tensor_t* grad_output,
                                         boat_tensor_t* grad_weight) {
    if (!layer || !cached_input || !grad_output || !grad_weight) {
        return false;
    }

    // Extract dimensions
    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];
    int64_t out_channels = output_shape[1];
    int64_t height_out = output_shape[2];
    int64_t width_out = output_shape[3];

    // Get data pointers
    float* grad_weight_data = (float*)boat_tensor_data(grad_weight);
    const float* input_data = (float*)boat_tensor_data(cached_input);
    const float* grad_output_data = (float*)boat_tensor_data(grad_output);

    // Initialize gradient weight with zeros
    size_t grad_weight_elements = boat_tensor_nelements(grad_weight);
    memset(grad_weight_data, 0, grad_weight_elements * sizeof(float));

    // Group convolution parameters
    size_t in_channels_per_group = layer->in_channels / layer->groups;
    size_t out_channels_per_group = layer->out_channels / layer->groups;

    // For each output channel (parallel: grad_weight[oc] slices are disjoint;
    // the group is derived from oc since groups partition the channel dims).
    BOAT_OMP_PARALLEL_FOR_SCHEDULE(static)
    for (int64_t oc = 0; oc < (int64_t)out_channels; oc++) {
        size_t g = (size_t)oc / out_channels_per_group;
        size_t ic_start = g * in_channels_per_group;
        size_t ic_end = ic_start + in_channels_per_group;

        // For each batch
        for (int64_t b = 0; b < batch; b++) {
            // For each input channel in this group
            for (int64_t ic = (int64_t)ic_start; ic < (int64_t)ic_end; ic++) {
                size_t ic_local = (size_t)(ic - (int64_t)ic_start);
                // For each kernel row
                for (size_t kh = 0; kh < layer->kernel_size; kh++) {
                    // For each kernel column
                    for (size_t kw = 0; kw < layer->kernel_size; kw++) {
                        // For each output height position
                        for (int64_t oh = 0; oh < height_out; oh++) {
                            int64_t ih = oh * layer->stride - layer->padding + kh;
                            if (ih < 0 || ih >= height) continue;

                            // Fast path (stride 1): the per-oh contribution
                            // is a dot over the contiguous valid interval
                            // [lo, hi) on both input and grad_output.
                            if (layer->stride == 1) {
                                int64_t lo = (int64_t)layer->padding > (int64_t)kw
                                                 ? (int64_t)layer->padding - (int64_t)kw
                                                 : 0;
                                int64_t hi = width + (int64_t)layer->padding - (int64_t)kw;
                                if (hi > width_out) hi = width_out;
                                if (lo < hi) {
                                    size_t weight_idx = ((oc * in_channels_per_group + ic_local) *
                                                             layer->kernel_size +
                                                         kh) *
                                                            layer->kernel_size +
                                                        kw;
                                    size_t input_base =
                                        ((size_t)(b * in_channels + ic) * (size_t)height +
                                         (size_t)ih) *
                                        (size_t)width;
                                    size_t grad_output_base =
                                        ((size_t)(b * out_channels + oc) * (size_t)height_out +
                                         (size_t)oh) *
                                        (size_t)width_out;
                                    // iw = ow - pad + kw: the input column for
                                    // ow = lo is (lo - pad + kw) >= 0 by the
                                    // range definition, and the input side is
                                    // shifted by (kw - pad) vs the output side.
                                    size_t col_start =
                                        (size_t)(lo - (int64_t)layer->padding + (int64_t)kw);
                                    grad_weight_data[weight_idx] +=
                                        boat_simd_dot_f32(input_data + input_base + col_start,
                                                          grad_output_data + grad_output_base + lo,
                                                          (size_t)(hi - lo));
                                }
                                continue;
                            }

                            for (int64_t ow = 0; ow < width_out; ow++) {
                                int64_t iw = ow * layer->stride - layer->padding + kw;
                                if (iw < 0 || iw >= width) continue;

                                size_t input_idx =
                                    ((b * in_channels + ic) * height + ih) * width + iw;
                                size_t grad_output_idx =
                                    ((b * out_channels + oc) * height_out + oh) * width_out + ow;
                                size_t weight_idx =
                                    ((oc * in_channels_per_group + ic_local) * layer->kernel_size +
                                     kh) *
                                        layer->kernel_size +
                                    kw;

                                grad_weight_data[weight_idx] +=
                                    input_data[input_idx] * grad_output_data[grad_output_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    return true;
}

// Helper function: compute gradient with respect to bias

static bool compute_bias_gradient_into(const boat_conv_layer_t* layer, const int64_t* output_shape,
                                       const boat_tensor_t* grad_output, boat_tensor_t* grad_bias) {
    if (!layer || !grad_output || !grad_bias) {
        return false;
    }
    // Only compute bias gradient if bias is used
    if (!layer->use_bias) {
        return false;
    }
    // Extract dimensions
    int64_t batch = output_shape[0];
    int64_t out_channels = output_shape[1];
    int64_t height_out = output_shape[2];
    int64_t width_out = output_shape[3];

    // Get data pointers
    float* grad_bias_data = (float*)boat_tensor_data(grad_bias);
    const float* grad_output_data = (float*)boat_tensor_data(grad_output);

    // Initialize gradient bias with zeros
    memset(grad_bias_data, 0, layer->out_channels * sizeof(float));

    // Sum grad_output over batch, height, width dimensions (parallel over oc:
    // grad_bias[oc] is disjoint per oc).
    BOAT_OMP_PARALLEL_FOR_SCHEDULE(static)
    for (int64_t oc = 0; oc < (int64_t)layer->out_channels; oc++) {
        float sum = 0.0f;
        for (int64_t b = 0; b < batch; b++) {
            const float* block = grad_output_data + (size_t)(b * out_channels + oc) *
                                                        (size_t)height_out * (size_t)width_out;
            sum += boat_simd_sum_reduce_f32(block, (size_t)height_out * (size_t)width_out);
        }
        grad_bias_data[oc] += sum;
    }
    return true;
}

BOAT_API boat_conv_layer_t* BOAT_CALL boat_conv_layer_create(size_t in_channels,
                                                             size_t out_channels,
                                                             size_t kernel_size, size_t stride,
                                                             size_t padding, size_t groups) {
    BOAT_DEBUG_PRINT("DEBUG conv_create called: in=%zu, out=%zu, k=%zu, groups=%zu\n", in_channels,
                     out_channels, kernel_size, groups);
    boat_conv_layer_t* layer =
        (boat_conv_layer_t*)boat_malloc(sizeof(boat_conv_layer_t), BOAT_DEVICE_CPU);
    if (!layer) {
        BOAT_DEBUG_PRINT("DEBUG conv_create: malloc failed\n");
        return NULL;
    }

    // Validate groups
    if (groups == 0 || in_channels % groups != 0 || out_channels % groups != 0) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[ConvLayer] groups=%zu must divide in_channels=%zu and out_channels=%zu\n",
                        groups, in_channels, out_channels);
        boat_free(layer);
        return NULL;
    }

    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->groups = groups;
    layer->use_bias = true; // Default to using bias

    // Create weight tensor: [out_channels, in_channels/groups, kernel_size, kernel_size]
    size_t in_channels_per_group = in_channels / groups;
    const int64_t weight_shape[] = {(int64_t)out_channels, (int64_t)in_channels_per_group,
                                    (int64_t)kernel_size, (int64_t)kernel_size};
    layer->weight = boat_tensor_create(weight_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!layer->weight) {
        boat_free(layer);
        return NULL;
    }

    // Initialize weights using Kaiming/He initialization for ReLU activations
    float* weight_data = (float*)boat_tensor_data(layer->weight);
    size_t weight_elements = boat_tensor_nelements(layer->weight);
    float scale = sqrtf(2.0f / (in_channels_per_group * kernel_size * kernel_size));
    for (size_t i = 0; i < weight_elements; i++) {
        weight_data[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
    }

    // Create bias tensor: [out_channels]
    const int64_t bias_shape[] = {(int64_t)out_channels};
    layer->bias = boat_tensor_create(bias_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!layer->bias) {
        boat_tensor_free(layer->weight);
        boat_free(layer);
        return NULL;
    }

    // Initialize bias to zeros
    float* bias_data = (float*)boat_tensor_data(layer->bias);
    size_t bias_elements = boat_tensor_nelements(layer->bias);
    memset(bias_data, 0, bias_elements * sizeof(float));

    // Create gradient accumulators with same shape as parameters
    layer->grad_weight = boat_tensor_create(weight_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    BOAT_DEBUG_PRINT("DEBUG conv create: grad_weight tensor created at %p\n", layer->grad_weight);
    if (!layer->grad_weight) {
        boat_tensor_free(layer->weight);
        boat_tensor_free(layer->bias);
        boat_free(layer);
        return NULL;
    }
    // Initialize gradient weight with zeros
    float* grad_weight_data = (float*)boat_tensor_data(layer->grad_weight);
    size_t grad_weight_elements = boat_tensor_nelements(layer->grad_weight);
    memset(grad_weight_data, 0, grad_weight_elements * sizeof(float));

    layer->grad_bias = boat_tensor_create(bias_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    BOAT_DEBUG_PRINT("DEBUG conv create: grad_bias tensor created at %p\n", layer->grad_bias);
    if (!layer->grad_bias) {
        boat_tensor_free(layer->weight);
        boat_tensor_free(layer->bias);
        boat_tensor_free(layer->grad_weight);
        boat_free(layer);
        return NULL;
    }
    // Initialize gradient bias with zeros
    float* grad_bias_data = (float*)boat_tensor_data(layer->grad_bias);
    size_t grad_bias_elements = boat_tensor_nelements(layer->grad_bias);
    memset(grad_bias_data, 0, grad_bias_elements * sizeof(float));

    layer->cache_input = NULL;
    memset(layer->cache_input_shape, 0, sizeof(layer->cache_input_shape));
    memset(layer->cache_output_shape, 0, sizeof(layer->cache_output_shape));

    return layer;
}

BOAT_API void BOAT_CALL boat_conv_layer_free(boat_conv_layer_t* layer) {
    if (!layer) {
        return;
    }

    if (layer->weight) boat_tensor_free(layer->weight);
    if (layer->bias) boat_tensor_free(layer->bias);
    if (layer->grad_weight) boat_tensor_free(layer->grad_weight);
    if (layer->grad_bias) boat_tensor_free(layer->grad_bias);
    if (layer->cache_input) boat_tensor_free(layer->cache_input);
    boat_free(layer);
}

// ---------------------------------------------------------------------------
// Stride-1 SIMD convolution over the interior output-width range.
//
// For a fixed (batch, output channel, output row, input channel, kh, kw) the
// valid input positions iw = ow + kw - pad are exactly those with ow in
// [max(0, pad-kw), min(wo, wi+pad-kw)), so the accumulation over ow is a
// contiguous, in-bounds vectorized FMA with no boundary branches.
// ---------------------------------------------------------------------------
static void conv2d_forward_stride1(const float* in, const float* w, const float* bias, float* out,
                                   int64_t batch, int64_t in_ch, int64_t out_ch, int64_t h,
                                   int64_t wi, int64_t ho, int64_t wo, size_t ks, size_t pad,
                                   size_t groups) {
    const size_t ocpg = out_ch / groups;
    const size_t icpg = in_ch / groups;
    // Parallelize over the batch: each image writes distinct output rows.
    BOAT_OMP_PARALLEL_FOR_SCHEDULE(static)
    for (int64_t b = 0; b < batch; b++) {
        for (size_t g = 0; g < groups; g++) {
            const size_t oc0 = g * ocpg;
            const size_t ic0 = g * icpg;
            for (size_t oc = oc0; oc < oc0 + ocpg; oc++) {
                for (int64_t oh = 0; oh < ho; oh++) {
                    const int64_t ih_base = oh - (int64_t)pad;
                    float* orow = out + ((b * out_ch + (int64_t)oc) * ho + oh) * wo;
                    for (size_t ic = ic0; ic < ic0 + icpg; ic++) {
                        const size_t icl = ic - ic0;
                        for (size_t kh = 0; kh < ks; kh++) {
                            const int64_t ih = ih_base + (int64_t)kh;
                            if (ih < 0 || ih >= h) continue;
                            const float* in_row = in + ((b * in_ch + (int64_t)ic) * h + ih) * wi;
                            const float* w_row = w + ((oc * icpg + icl) * ks + kh) * ks;
                            for (size_t kw = 0; kw < ks; kw++) {
                                const int64_t iw_off = (int64_t)kw - (int64_t)pad;
                                const int64_t lo = iw_off < 0 ? -iw_off : 0;
                                const int64_t hi = (wi - iw_off < wo) ? (wi - iw_off) : wo;
                                if (lo >= hi) continue;
                                const float wv = w_row[kw];
                                const float* in_off = in_row + iw_off;
#if BOAT_HAVE_AVX2
                                int64_t ow = lo;
                                const __m256 vw = _mm256_set1_ps(wv);
                                for (; ow + 8 <= hi; ow += 8) {
                                    __m256 va = _mm256_loadu_ps(orow + ow);
                                    __m256 vb = _mm256_loadu_ps(in_off + ow);
#if BOAT_HAVE_FMA
                                    va = _mm256_fmadd_ps(vb, vw, va);
#else
                                    va = _mm256_add_ps(va, _mm256_mul_ps(vb, vw));
#endif
                                    _mm256_storeu_ps(orow + ow, va);
                                }
                                for (; ow < hi; ow++)
                                    orow[ow] += in_off[ow] * wv;
#elif BOAT_HAVE_NEON
                                int64_t ow = lo;
                                const float32x4_t vw = vdupq_n_f32(wv);
                                for (; ow + 4 <= hi; ow += 4) {
                                    float32x4_t va = vld1q_f32(orow + ow);
                                    float32x4_t vb = vld1q_f32(in_off + ow);
#if BOAT_HAVE_FMA
                                    va = vfmaq_f32(va, vb, vw);
#else
                                    va = vmlaq_f32(va, vb, vw);
#endif
                                    vst1q_f32(orow + ow, va);
                                }
                                for (; ow < hi; ow++)
                                    orow[ow] += in_off[ow] * wv;
#else
                                for (int64_t ow = lo; ow < hi; ow++)
                                    orow[ow] += in_off[ow] * wv;
#endif
                            }
                        }
                    }
                }
                if (bias) {
                    const float bv = bias[oc];
                    float* op = out + (b * out_ch + (int64_t)oc) * ho * wo;
                    const size_t plane = (size_t)ho * wo;
                    for (size_t i = 0; i < plane; i++)
                        op[i] += bv;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Im2col + SGEMM convolution (any stride). Builds the column matrix
// col[ckk, N] per image (rows = flattened (ic, kh, kw), columns = output
// positions), then computes out[oc, N] = W[oc, ckk] @ col via boat_sgemm
// (SIMD-packed). Used when the GEMM shape is large enough to pay off.
// ---------------------------------------------------------------------------
static void conv2d_forward_im2col(const float* in, const float* w, const float* bias, float* out,
                                  int64_t batch, int64_t in_ch, int64_t out_ch, int64_t h,
                                  int64_t wi, int64_t ho, int64_t wo, size_t ks, size_t pad,
                                  size_t stride, size_t groups) {
    const size_t ocpg = out_ch / groups;
    const size_t icpg = in_ch / groups;
    const size_t ckk = icpg * ks * ks;
    const int64_t N = ho * wo; // output positions per image
    // Parallelize over the batch: each image builds its own column matrix and
    // GEMM (thread-local buffers). Without OpenMP this is a plain loop.
    BOAT_OMP_PARALLEL_FOR_SCHEDULE(static)
    for (int64_t b = 0; b < batch; b++) {
        float* col = (float*)malloc(ckk * (size_t)N * sizeof(float));
        float* cblk = (float*)malloc(ocpg * (size_t)N * sizeof(float));
        if (!col || !cblk) {
            free(col);
            free(cblk);
            continue;
        }
        for (size_t g = 0; g < groups; g++) {
            // Build col[(ic*ks+kh)*ks+kw][oh*wo+ow].
            for (size_t ic = 0; ic < icpg; ic++) {
                const float* ibase = in + ((b * in_ch + g * icpg + ic) * h) * wi;
                for (size_t kh = 0; kh < ks; kh++) {
                    for (size_t kw = 0; kw < ks; kw++) {
                        const size_t r = (ic * ks + kh) * ks + kw;
                        float* crow = col + r * (size_t)N;
                        for (int64_t oh = 0; oh < ho; oh++) {
                            const int64_t ih = oh * (int64_t)stride - (int64_t)pad + (int64_t)kh;
                            if (ih < 0 || ih >= h) {
                                memset(crow + (size_t)oh * wo, 0, (size_t)wo * sizeof(float));
                                continue;
                            }
                            const float* irow = ibase + ih * wi;
                            const int64_t iw0 = -(int64_t)pad + (int64_t)kw;
                            for (int64_t ow = 0; ow < wo; ow++) {
                                const int64_t iw = iw0 + ow * (int64_t)stride;
                                crow[(size_t)oh * wo + ow] = (iw >= 0 && iw < wi) ? irow[iw] : 0.0f;
                            }
                        }
                    }
                }
            }
            // GEMM: cblk[ocpg, N] = w[g*ocpg..][ocpg, ckk] @ col[ckk, N].
            const float* wg = w + g * ocpg * ckk;
            boat_sgemm((int64_t)ocpg, N, (int64_t)ckk, wg, col, cblk);
            // Scatter to out[b][oc][oh][ow] and add bias.
            float* obase = out + (b * out_ch + g * ocpg) * ho * wo;
            for (size_t oc = 0; oc < ocpg; oc++) {
                const float bv = bias ? bias[g * ocpg + oc] : 0.0f;
                float* orow = obase + (int64_t)oc * ho * wo;
                const float* crow = cblk + (int64_t)oc * N;
                for (int64_t i = 0; i < N; i++)
                    orow[i] = crow[i] + bv;
            }
        }
        free(col);
        free(cblk);
    }
}

BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_forward(boat_conv_layer_t* layer,
                                                          const boat_tensor_t* input) {
    if (!layer || !input) {
        return NULL;
    }

    // Output shape: [batch, out_channels, height_out, width_out]
    const int64_t* input_shape = boat_tensor_shape(input);
    if (boat_tensor_ndim(input) != 4) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[ConvLayer] Conv2d expects 4D input tensor\n");
        return NULL;
    }

    int64_t batch = input_shape[0];
    int64_t in_channels = input_shape[1];
    int64_t height = input_shape[2];
    int64_t width = input_shape[3];

    if ((size_t)in_channels != layer->in_channels) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[ConvLayer] Input channels %lld don't match layer in_channels %zu\n",
                        in_channels, layer->in_channels);
        return NULL;
    }

    // Check data types
    if (boat_tensor_dtype(input) != BOAT_DTYPE_FLOAT32) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[ConvLayer] Conv2d only supports FLOAT32 input tensors\n");
        return NULL;
    }
    // Allow quantized weights (UINT8/INT8/BITS2/FLOAT4 with scale != 0)
    boat_dtype_t wdt = boat_tensor_dtype(layer->weight);
    bool weight_quantized =
        (wdt == BOAT_DTYPE_UINT8 || wdt == BOAT_DTYPE_INT8 || wdt == BOAT_DTYPE_BITS2 ||
         wdt == BOAT_DTYPE_BITS1 || wdt == BOAT_DTYPE_FLOAT4) &&
        (wdt == BOAT_DTYPE_FLOAT4 || boat_tensor_get_scale(layer->weight) != 0.0f);
    if (wdt != BOAT_DTYPE_FLOAT32 && !weight_quantized) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[ConvLayer] Conv2d weight tensor must be FLOAT32 or quantized "
                        "UINT8/INT8/BITS2/BITS1/FLOAT4\n");
        return NULL;
    }
    if (layer->use_bias && layer->bias && boat_tensor_dtype(layer->bias) != BOAT_DTYPE_FLOAT32) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[ConvLayer] Conv2d bias tensor must be FLOAT32\n");
        return NULL;
    }

    // Handle quantized weights: dequantize to temporary FP32
    boat_tensor_t* dequantized_weight = NULL;
    const boat_tensor_t* effective_weight = layer->weight;
    if (weight_quantized) {
        if (boat_tensor_is_per_channel(layer->weight)) {
            dequantized_weight = boat_dequantize_tensor_per_channel(layer->weight);
        } else {
            dequantized_weight = boat_dequantize_tensor(layer->weight);
        }
        if (!dequantized_weight) return NULL;
        effective_weight = dequantized_weight;
    }

    // Calculate output dimensions
    int64_t height_out = (height + 2 * layer->padding - layer->kernel_size) / layer->stride + 1;
    int64_t width_out = (width + 2 * layer->padding - layer->kernel_size) / layer->stride + 1;

    // Validate output dimensions
    if (height_out <= 0 || width_out <= 0) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[ConvLayer] Invalid convolution parameters - output dimensions would be "
                        "non-positive: height_out=%lld, width_out=%lld (height=%lld, width=%lld, "
                        "padding=%zu, kernel_size=%zu, stride=%zu)\n",
                        height_out, width_out, height, width, layer->padding, layer->kernel_size,
                        layer->stride);
        return NULL;
    }

    // Clear old cache
    if (layer->cache_input) {
        boat_tensor_free(layer->cache_input);
        layer->cache_input = NULL;
    }

    // Cache input tensor for backward pass
    layer->cache_input = (boat_tensor_t*)input;
    boat_tensor_ref(layer->cache_input); // Increase ref count

    // Cache input shape
    layer->cache_input_shape[0] = batch;
    layer->cache_input_shape[1] = in_channels;
    layer->cache_input_shape[2] = height;
    layer->cache_input_shape[3] = width;

    // Cache output shape
    layer->cache_output_shape[0] = batch;
    layer->cache_output_shape[1] = layer->out_channels;
    layer->cache_output_shape[2] = height_out;
    layer->cache_output_shape[3] = width_out;

    // Create output tensor
    const int64_t output_shape[] = {batch, (int64_t)layer->out_channels, height_out, width_out};
    boat_tensor_t* output =
        boat_tensor_create(output_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    if (!output) {
        if (dequantized_weight) boat_tensor_free(dequantized_weight);
        return NULL;
    }

    // Get data pointers
    const float* input_data = (float*)boat_tensor_data(input);
    const float* weight_data = (const float*)boat_tensor_const_data(effective_weight);
    const float* bias_data = layer->use_bias ? (float*)boat_tensor_data(layer->bias) : NULL;
    float* output_data = (float*)boat_tensor_data(output);

    // Initialize output with zeros
    size_t output_elements = boat_tensor_nelements(output);
    memset(output_data, 0, output_elements * sizeof(float));

    // Perform convolution with group support.
    // Prefer the im2col + SGEMM path when the GEMM shape pays off; otherwise
    // use the SIMD interior kernel (stride 1) or the general scalar loop.
    const size_t ckk = (in_channels / layer->groups) * layer->kernel_size * layer->kernel_size;
    const int64_t npos = height_out * width_out;
    const bool use_im2col = (int64_t)(layer->out_channels * ckk) >= 1024 && npos >= 64;
    if (use_im2col) {
        conv2d_forward_im2col(input_data, weight_data, bias_data, output_data, batch,
                              (int64_t)in_channels, (int64_t)layer->out_channels, height, width,
                              height_out, width_out, layer->kernel_size, layer->padding,
                              layer->stride, layer->groups);
        if (dequantized_weight) boat_tensor_free(dequantized_weight);
        return output;
    }
    if (layer->stride == 1) {
        conv2d_forward_stride1(input_data, weight_data, bias_data, output_data, batch,
                               (int64_t)in_channels, (int64_t)layer->out_channels, height, width,
                               height_out, width_out, layer->kernel_size, layer->padding,
                               layer->groups);
        if (dequantized_weight) boat_tensor_free(dequantized_weight);
        return output;
    }

    // For each sample in batch
    size_t out_channels_per_group = layer->out_channels / layer->groups;
    size_t in_channels_per_group = layer->in_channels / layer->groups;
    for (int64_t b = 0; b < batch; b++) {
        // For each group
        for (size_t g = 0; g < layer->groups; g++) {
            size_t oc_start = g * out_channels_per_group;
            size_t oc_end = oc_start + out_channels_per_group;
            size_t ic_start = g * in_channels_per_group;
            size_t ic_end = ic_start + in_channels_per_group;

            // For each output channel in this group
            for (size_t oc = oc_start; oc < oc_end; oc++) {
                // For each input channel in this group
                for (size_t ic = ic_start; ic < ic_end; ic++) {
                    size_t ic_local = ic - ic_start;
                    // For each output height position
                    for (int64_t oh = 0; oh < height_out; oh++) {
                        int64_t ih_start = oh * layer->stride - layer->padding;
                        // For each output width position
                        for (int64_t ow = 0; ow < width_out; ow++) {
                            int64_t iw_start = ow * layer->stride - layer->padding;

                            // Convolve kernel with input patch
                            float sum = 0.0f;
                            for (size_t kh = 0; kh < layer->kernel_size; kh++) {
                                int64_t ih = ih_start + kh;
                                if (ih < 0 || ih >= height) continue;

                                for (size_t kw = 0; kw < layer->kernel_size; kw++) {
                                    int64_t iw = iw_start + kw;
                                    if (iw < 0 || iw >= width) continue;

                                    size_t input_idx =
                                        ((b * layer->in_channels + ic) * height + ih) * width + iw;
                                    size_t weight_idx = ((oc * in_channels_per_group + ic_local) *
                                                             layer->kernel_size +
                                                         kh) *
                                                            layer->kernel_size +
                                                        kw;

                                    sum += input_data[input_idx] * weight_data[weight_idx];
                                }
                            }

                            size_t output_idx =
                                ((b * layer->out_channels + oc) * height_out + oh) * width_out + ow;
                            output_data[output_idx] += sum;
                        }
                    }
                }

                // Add bias if present
                if (layer->use_bias && bias_data) {
                    float bias = bias_data[oc];
                    for (int64_t oh = 0; oh < height_out; oh++) {
                        for (int64_t ow = 0; ow < width_out; ow++) {
                            size_t output_idx =
                                ((b * layer->out_channels + oc) * height_out + oh) * width_out + ow;
                            output_data[output_idx] += bias;
                        }
                    }
                }
            }
        }
    }

    // Free temporary dequantized weight if any
    if (dequantized_weight) {
        boat_tensor_free(dequantized_weight);
    }

    return output;
}

BOAT_API boat_tensor_t* BOAT_CALL boat_conv_layer_backward(boat_conv_layer_t* layer,
                                                           const boat_tensor_t* grad_output) {
    if (!layer || !grad_output) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT, "[ConvLayer] conv backward: NULL input\n");
        return NULL;
    }
    BOAT_DEBUG_PRINT("[conv backward] layer=%p, grad_output=%p\n", (void*)layer,
                     (void*)grad_output);
    BOAT_DEBUG_PRINT("[conv backward] cache_input=%p\n", (void*)layer->cache_input);

    // Check that cached input exists
    if (!layer->cache_input) {
        boat_set_errorf(
            BOAT_ERROR_INVALID_OPERATION,
            "[ConvLayer] conv backward: no cached input (forward not called or cache cleared)\n");
        return NULL;
    }

    // Verify grad_output shape matches cached output shape
    const int64_t* grad_shape = boat_tensor_shape(grad_output);
    if (grad_shape[0] != layer->cache_output_shape[0] ||
        grad_shape[1] != layer->cache_output_shape[1] ||
        grad_shape[2] != layer->cache_output_shape[2] ||
        grad_shape[3] != layer->cache_output_shape[3]) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[ConvLayer] conv backward: grad_output shape [%lld, %lld, %lld, %lld] "
                        "doesn't match cached output shape [%lld, %lld, %lld, %lld]\n",
                        grad_shape[0], grad_shape[1], grad_shape[2], grad_shape[3],
                        layer->cache_output_shape[0], layer->cache_output_shape[1],
                        layer->cache_output_shape[2], layer->cache_output_shape[3]);
        return NULL;
    }

    // Dispatch to cuDNN backward for CUDA tensors
#ifdef BOAT_WITH_CUDNN
    if (boat_tensor_device(grad_output) == BOAT_DEVICE_CUDA &&
        boat_tensor_device(layer->cache_input) == BOAT_DEVICE_CUDA &&
        boat_tensor_device(layer->weight) == BOAT_DEVICE_CUDA) {

        // Allocate grad_input on CUDA device
        boat_tensor_t* grad_input =
            boat_tensor_create(layer->cache_input_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CUDA);
        if (!grad_input) {
            boat_set_errorf(BOAT_ERROR_OUT_OF_MEMORY,
                            "[ConvLayer] conv backward: failed to create CUDA grad_input\n");
            return NULL;
        }

        // Replace CPU grad_weight with CUDA version
        size_t in_cpg = layer->in_channels / layer->groups;
        const int64_t wshape[] = {(int64_t)layer->out_channels, (int64_t)in_cpg,
                                  (int64_t)layer->kernel_size, (int64_t)layer->kernel_size};
        if (layer->grad_weight) boat_tensor_free(layer->grad_weight);
        layer->grad_weight = boat_tensor_create(wshape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CUDA);
        if (!layer->grad_weight) {
            boat_tensor_free(grad_input);
            return NULL;
        }

        // Replace CPU grad_bias with CUDA version
        if (layer->use_bias) {
            if (layer->grad_bias) boat_tensor_free(layer->grad_bias);
            const int64_t bshape[] = {(int64_t)layer->out_channels};
            layer->grad_bias = boat_tensor_create(bshape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CUDA);
            if (!layer->grad_bias) {
                boat_tensor_free(grad_input);
                return NULL;
            }
        }

        // Get device pointers
        const float* d_grad_output = (const float*)boat_tensor_data(grad_output);
        const float* d_weight = (const float*)boat_tensor_const_data(layer->weight);
        const float* d_input = (const float*)boat_tensor_const_data(layer->cache_input);
        float* d_grad_input = (float*)boat_tensor_data(grad_input);
        float* d_grad_weight = (float*)boat_tensor_data(layer->grad_weight);
        float* d_grad_bias = layer->use_bias ? (float*)boat_tensor_data(layer->grad_bias) : NULL;

        // Compute input gradient via cuDNN
        boat_cuda_conv2d_cudnn_backward_input_f32(
            d_grad_output, d_weight, d_grad_input, (size_t)layer->cache_input_shape[0],
            (size_t)layer->cache_input_shape[1], (size_t)layer->cache_input_shape[2],
            (size_t)layer->cache_input_shape[3], layer->out_channels, layer->kernel_size,
            layer->kernel_size, layer->padding, layer->stride, layer->groups);

        // Compute weight + bias gradients via cuDNN
        boat_cuda_conv2d_cudnn_backward_filter_f32(
            d_input, d_grad_output, d_grad_weight, d_grad_bias, (size_t)layer->cache_input_shape[0],
            (size_t)layer->cache_input_shape[1], (size_t)layer->cache_input_shape[2],
            (size_t)layer->cache_input_shape[3], layer->out_channels, layer->kernel_size,
            layer->kernel_size, layer->padding, layer->stride, layer->groups);

        return grad_input;
    }
#endif

    // CPU backward path: compute gradients using helper functions
    boat_tensor_t* grad_input =
        compute_input_gradient(layer, layer->cache_input, layer->cache_input_shape,
                               layer->cache_output_shape, grad_output);
    if (!grad_input) {
        boat_set_errorf(BOAT_ERROR_INVALID_OPERATION,
                        "[ConvLayer] conv backward: failed to compute input gradient\n");
        return NULL;
    }

    // Compute gradients directly into layer's gradient tensors
    if (!layer->grad_weight) {
        // Create gradient weight tensor if it doesn't exist (should exist)
        size_t in_channels_per_group = layer->in_channels / layer->groups;
        const int64_t weight_shape[] = {(int64_t)layer->out_channels,
                                        (int64_t)in_channels_per_group, (int64_t)layer->kernel_size,
                                        (int64_t)layer->kernel_size};
        layer->grad_weight =
            boat_tensor_create(weight_shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        if (!layer->grad_weight) {
            boat_set_errorf(BOAT_ERROR_INVALID_OPERATION,
                            "[ConvLayer] conv backward: failed to create grad_weight tensor\n");
            boat_tensor_free(grad_input);
            return NULL;
        }
    }
    if (!compute_weight_gradient_into(layer, layer->cache_input, layer->cache_input_shape,
                                      layer->cache_output_shape, grad_output, layer->grad_weight)) {
        boat_set_errorf(BOAT_ERROR_INVALID_OPERATION,
                        "[ConvLayer] conv backward: failed to compute weight gradient\n");
        boat_tensor_free(grad_input);
        return NULL;
    }

    // Bias gradient (only if bias is used)
    if (layer->use_bias) {
        if (!layer->grad_bias) {
            const int64_t bias_shape[] = {(int64_t)layer->out_channels};
            layer->grad_bias =
                boat_tensor_create(bias_shape, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
            if (!layer->grad_bias) {
                boat_set_errorf(BOAT_ERROR_INVALID_OPERATION,
                                "[ConvLayer] conv backward: failed to create grad_bias tensor\n");
                boat_tensor_free(grad_input);
                return NULL;
            }
        }
        if (!compute_bias_gradient_into(layer, layer->cache_output_shape, grad_output,
                                        layer->grad_bias)) {
            boat_set_errorf(BOAT_ERROR_INVALID_OPERATION,
                            "[ConvLayer] conv backward: failed to compute bias gradient\n");
            boat_tensor_free(grad_input);
            return NULL;
        }
    } else {
        // Ensure grad_bias is NULL if bias not used
        if (layer->grad_bias) {
            boat_tensor_free(layer->grad_bias);
            layer->grad_bias = NULL;
        }
    }

    // Debug: print gradient info
    BOAT_DEBUG_PRINT("[conv backward] grad_weight stored, pointer=%p, nelem=%zu\n",
                     layer->grad_weight,
                     layer->grad_weight ? boat_tensor_nelements(layer->grad_weight) : 0);
    BOAT_DEBUG_PRINT("[conv backward] grad_bias stored, pointer=%p\n", layer->grad_bias);

    // Note: we don't free cache_input here, it will be freed in next forward pass or layer free
    // Return gradient with respect to input
    return grad_input;
}

BOAT_API void BOAT_CALL boat_conv_layer_update(boat_conv_layer_t* layer, float learning_rate) {
    if (!layer) {
        return;
    }

    // Simple SGD update: weight = weight - learning_rate * grad_weight
    if (layer->grad_weight && layer->weight) {
        boat_tensor_t* scaled_grad = boat_mul_scalar(layer->grad_weight, learning_rate);
        if (scaled_grad) {
            boat_sub_(layer->weight, scaled_grad); // weight -= learning_rate * grad_weight
            boat_tensor_unref(scaled_grad);
        }
    }

    if (layer->use_bias && layer->grad_bias && layer->bias) {
        boat_tensor_t* scaled_grad = boat_mul_scalar(layer->grad_bias, learning_rate);
        if (scaled_grad) {
            boat_sub_(layer->bias, scaled_grad); // bias -= learning_rate * grad_bias
            boat_tensor_unref(scaled_grad);
        }
    }

    // Note: we don't zero gradients after update; caller can decide to clear gradients
}

// Parameter access functions for model loading
BOAT_API void BOAT_CALL boat_conv_layer_set_weight(boat_conv_layer_t* layer,
                                                   boat_tensor_t* weight) {
    if (!layer || !weight) {
        return;
    }
    // Check weight shape matches layer dimensions (with groups)
    const int64_t* weight_shape = boat_tensor_shape(weight);
    size_t in_channels_per_group = layer->in_channels / layer->groups;
    if (weight_shape[0] != (int64_t)layer->out_channels ||
        weight_shape[1] != (int64_t)in_channels_per_group ||
        weight_shape[2] != (int64_t)layer->kernel_size ||
        weight_shape[3] != (int64_t)layer->kernel_size) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[ConvLayer] Weight shape [%lld, %lld, %lld, %lld] does not match layer "
                        "dimensions [%zu, %zu, %zu, %zu]\n",
                        weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3],
                        layer->out_channels, in_channels_per_group, layer->kernel_size,
                        layer->kernel_size);
        return;
    }
    // Replace weight tensor
    if (layer->weight) {
        boat_tensor_free(layer->weight);
    }
    layer->weight = weight;
    boat_tensor_ref(weight); // Increase ref count since layer now owns it
}

BOAT_API void BOAT_CALL boat_conv_layer_set_bias(boat_conv_layer_t* layer, boat_tensor_t* bias) {
    if (!layer || !bias) {
        return;
    }
    if (!layer->use_bias) {
        BOAT_DEBUG_PRINT(
            "[ConvLayer] Warning: Layer was created without bias, ignoring bias tensor\n");
        return;
    }
    // Check bias shape matches output channels
    const int64_t* bias_shape = boat_tensor_shape(bias);
    if (bias_shape[0] != (int64_t)layer->out_channels) {
        boat_set_errorf(BOAT_ERROR_INVALID_ARGUMENT,
                        "[ConvLayer] Bias shape [%lld] does not match output channels %zu\n",
                        bias_shape[0], layer->out_channels);
        return;
    }
    // Replace bias tensor
    if (layer->bias) {
        boat_tensor_free(layer->bias);
    }
    layer->bias = bias;
    boat_tensor_ref(bias);
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_conv_layer_get_weight(const boat_conv_layer_t* layer) {
    return layer ? layer->weight : NULL;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_conv_layer_get_bias(const boat_conv_layer_t* layer) {
    return layer ? layer->bias : NULL;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_conv_layer_get_grad_weight(const boat_conv_layer_t* layer) {
    BOAT_DEBUG_PRINT("DEBUG get_grad_weight: layer=%p, grad_weight=%p\n", layer,
                     layer ? layer->grad_weight : NULL);
    return layer ? layer->grad_weight : NULL;
}

BOAT_API BOAT_NOINLINE boat_tensor_t* BOAT_CALL
boat_conv_layer_get_grad_bias(const boat_conv_layer_t* layer) {
    return layer ? layer->grad_bias : NULL;
}

BOAT_API BOAT_NOINLINE size_t BOAT_CALL boat_conv_layer_get_stride(const boat_conv_layer_t* layer) {
    return layer ? layer->stride : 0;
}

BOAT_API BOAT_NOINLINE size_t BOAT_CALL
boat_conv_layer_get_padding(const boat_conv_layer_t* layer) {
    return layer ? layer->padding : 0;
}

BOAT_API BOAT_NOINLINE size_t BOAT_CALL boat_conv_layer_get_groups(const boat_conv_layer_t* layer) {
    return layer ? layer->groups : 1;
}