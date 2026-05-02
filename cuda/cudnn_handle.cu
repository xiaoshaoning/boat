// cudnn_handle.cu - cuDNN handle manager and wrapper functions
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cuda_runtime.h>
#include <cudnn.h>
#include <boat/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Error checking macro
// ---------------------------------------------------------------------------
#define CUDNN_CHECK(call) do {                                              \
    cudnnStatus_t stat = call;                                              \
    if (stat != CUDNN_STATUS_SUCCESS) {                                     \
        fprintf(stderr, "[cuDNN] %s:%d: error %d (%s)\n",                  \
                __FILE__, __LINE__, (int)stat, cudnnGetErrorString(stat));  \
        exit(1);                                                            \
    }                                                                       \
} while(0)

// ---------------------------------------------------------------------------
// Global cuDNN handle (lazy initialization)
// ---------------------------------------------------------------------------
static cudnnHandle_t g_cudnn_handle = NULL;
static int g_cudnn_initialized = 0;

extern "C" cudnnHandle_t boat_cuda_get_cudnn_handle(void) {
    if (!g_cudnn_initialized) {
        CUDNN_CHECK(cudnnCreate(&g_cudnn_handle));
        g_cudnn_initialized = 1;
    }
    return g_cudnn_handle;
}

extern "C" void boat_cuda_cudnn_destroy(void) {
    if (g_cudnn_initialized && g_cudnn_handle) {
        cudnnDestroy(g_cudnn_handle);
        g_cudnn_handle = NULL;
        g_cudnn_initialized = 0;
    }
}

// ---------------------------------------------------------------------------
// cuDNN Conv2D forward — supports groups via cudnnSetConvolutionGroupCount
// ---------------------------------------------------------------------------
void boat_cuda_conv2d_cudnn_forward_f32(const float* input, const float* weight,
                                          const float* bias, float* output,
                                          size_t N, size_t C, size_t H, size_t W,
                                          size_t OC, size_t KH, size_t KW,
                                          size_t pad, size_t stride, size_t groups) {
    if (groups == 0 || C % groups != 0 || OC % groups != 0) return;

    size_t OH = (H + 2 * pad - KH) / stride + 1;
    size_t OW = (W + 2 * pad - KW) / stride + 1;

    cudnnHandle_t handle = boat_cuda_get_cudnn_handle();

    // Create descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set tensor descriptors (cuDNN uses NCHW format)
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, (int)N, (int)C, (int)H, (int)W));
    // Filter: [OC, C, KH, KW] — cuDNN handles group decomposition internally
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW, (int)OC, (int)C, (int)KH, (int)KW));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, (int)N, (int)OC, (int)OH, (int)OW));

    // Set convolution descriptor
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
        (int)pad, (int)pad, (int)stride, (int)stride, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Set group count (cuDNN 7.6+)
    CUDNN_CHECK(cudnnSetConvolutionGroupCount(conv_desc, (int)groups));

    // Choose algorithm: IMPLICIT_GEMM for deterministic behavior
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;

    // Query workspace size
    size_t workspace_size = 0;
    cudnnStatus_t ws_status = cudnnGetConvolutionForwardWorkspaceSize(
        handle, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_size);
    if (ws_status != CUDNN_STATUS_SUCCESS) {
        // Fall back to IMPLICIT_PRECOMP_GEMM if IMPLICIT_GEMM not supported
        algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            handle, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_size));
    }

    // Allocate workspace if needed
    void* workspace = NULL;
    if (workspace_size > 0) {
        cudaError_t e = cudaMalloc(&workspace, workspace_size);
        if (e != cudaSuccess) { workspace_size = 0; workspace = NULL; }
    }

    // Execute convolution
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(handle,
        &alpha, input_desc, input, filter_desc, weight,
        conv_desc, algo, workspace, workspace_size,
        &beta, output_desc, output));

    // Add bias if provided
    if (bias) {
        cudnnTensorDescriptor_t bias_desc;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT, 1, (int)OC, 1, 1));
        float beta_bias = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(handle, &alpha, bias_desc, bias,
            &beta_bias, output_desc, output));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    }

    // Cleanup
    if (workspace) cudaFree(workspace);
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
}

// ---------------------------------------------------------------------------
// Tiny kernel: convert cuDNN's saveInvVariance to variance
// cuDNN stores saveInvVariance = 1/sqrt(var + eps)
// So: var = 1/(saveInvVariance^2) - eps
// ---------------------------------------------------------------------------
static __global__ void inv_var_to_var_kernel(float* __restrict__ var,
                                              const float* __restrict__ inv_var,
                                              size_t C, float eps) {
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float iv = inv_var[c];
    float v = 1.0f / (iv * iv) - eps;
    var[c] = v > 0.0f ? v : 0.0f;
}

// ---------------------------------------------------------------------------
// cuDNN BatchNorm forward — computes mean and var internally
// ---------------------------------------------------------------------------
void boat_cuda_batchnorm_cudnn_forward_f32(const float* input, float* output,
                                             const float* gamma, const float* beta,
                                             float* mean, float* var,
                                             size_t N, size_t C, size_t H, size_t W,
                                             float eps) {
    if (C == 0) return;
    cudnnHandle_t handle = boat_cuda_get_cudnn_handle();

    // Input/output descriptor: NCHW
    cudnnTensorDescriptor_t data_desc, norm_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&data_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&norm_desc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(data_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, (int)N, (int)C, (int)H, (int)W));
    // Scale/bias/mean/var descriptor: 1 x C x 1 x 1
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(norm_desc, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, 1, (int)C, 1, 1));

    // Allocate temporary device buffer for saveInvVariance
    float* d_save_inv_var = NULL;
    cudaError_t e = cudaMalloc(&d_save_inv_var, C * sizeof(float));
    if (e != cudaSuccess) {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(data_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(norm_desc));
        return;
    }

    // Run forward training: cuDNN computes mean and invVariance
    double alpha = 1.0, beta_d = 0.0;
    CUDNN_CHECK(cudnnBatchNormForwardTraining(handle,
        CUDNN_BATCHNORM_SPATIAL,
        &alpha, &beta_d,
        data_desc, input,
        data_desc, output,
        norm_desc,
        gamma, beta,
        1.0,              // exponentialAverageFactor = 1.0 → pure batch stats
        NULL,             // resultRunningMean (not tracking)
        NULL,             // resultRunningVariance (not tracking)
        (double)eps,
        mean,             // resultSaveMean
        d_save_inv_var    // resultSaveInvVariance = 1/sqrt(var+eps)
    ));

    // Convert invVariance to variance on device
    const int block = 256;
    unsigned int grid = (unsigned int)((C + block - 1) / block);
    inv_var_to_var_kernel<<<grid, block>>>(var, d_save_inv_var, C, eps);
    cudaFree(d_save_inv_var);

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(data_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(norm_desc));
}
