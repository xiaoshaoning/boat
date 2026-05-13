// cublas_handle.cu - cuBLAS handle management
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>

static cublasHandle_t g_cublas_handle = NULL;
static int g_initialized = 0;

extern "C" cublasHandle_t boat_cuda_cublas_get_handle(void) {
    if (!g_initialized) {
        cublasStatus_t stat = cublasCreate(&g_cublas_handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[CUDA] Failed to create cuBLAS handle\n");
            return NULL;
        }
        g_initialized = 1;
    }
    return g_cublas_handle;
}

extern "C" void boat_cuda_cublas_destroy(void) {
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = NULL;
    }
    g_initialized = 0;
}
