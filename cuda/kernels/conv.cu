// conv.cu - Conv2D CUDA kernels (implicit GEMM via im2col + cuBLAS)
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <boat/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "[CUDA] %s:%d: error %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(1);                                                      \
    }                                                                 \
} while(0)

#define CUBLAS_CHECK(call) do {                                       \
    cublasStatus_t stat = call;                                       \
    if (stat != CUBLAS_STATUS_SUCCESS) {                              \
        fprintf(stderr, "[cuBLAS] %s:%d: error %d\n",                \
                __FILE__, __LINE__, (int)stat);                       \
        exit(1);                                                      \
    }                                                                 \
} while(0)

// ---------------------------------------------------------------------------
// im2col: transform 4D input [N, C, H, W] to column matrix [C*KH*KW, N*OH*OW]
//
// Each thread computes one element of the column matrix.
// Block: 2D (16 x 16 = 256 threads)
// Grid:  (N*OH*OW / 16,  C*KH*KW / 16)
// ---------------------------------------------------------------------------
__global__ void im2col_f32_kernel(const float* __restrict__ input,
                                   float* __restrict__ col,
                                   size_t N, size_t C, size_t H, size_t W,
                                   size_t KH, size_t KW,
                                   int pad, int stride,
                                   size_t OH, size_t OW) {
    size_t col_col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col_row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t n_rows = C * KH * KW;
    size_t n_cols = N * OH * OW;
    if (col_row >= n_rows || col_col >= n_cols) return;

    // Decode col_row → (c, kh, kw)
    size_t k_lin = col_row % (KH * KW);
    size_t c = col_row / (KH * KW);
    size_t kh = k_lin / KW;
    size_t kw = k_lin % KW;

    // Decode col_col → (n, oh, ow)
    size_t p = col_col % (OH * OW);
    size_t n = col_col / (OH * OW);
    size_t oh = p / OW;
    size_t ow = p % OW;

    // Map to input coordinates (with padding)
    int ih = (int)(oh * stride - pad + kh);
    int iw = (int)(ow * stride - pad + kw);

    float val = 0.0f;
    if (ih >= 0 && ih < (int)H && iw >= 0 && iw < (int)W) {
        val = input[((n * C + c) * H + (size_t)ih) * W + (size_t)iw];
    }

    col[col_row * n_cols + col_col] = val;
}

// ---------------------------------------------------------------------------
// Bias add + reshape kernel: adds bias to conv output and reshapes
// Input:  d_workspace [OC, N*OH*OW]  (cuBLAS output, column-major in memory)
// Output: d_output    [N, OC, OH, OW] (row-major)
// For each element: output[n, c, oh, ow] = workspace[c, n*OH*OW + oh*OW + ow] + bias[c]
// ---------------------------------------------------------------------------
__global__ void conv_bias_reshape_f32_kernel(const float* __restrict__ workspace,
                                              const float* __restrict__ bias,
                                              float* __restrict__ output,
                                              size_t N, size_t OC,
                                              size_t OH, size_t OW) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * OC * OH * OW;
    if (idx >= total) return;

    // Decode output index (row-major: N, OC, OH, OW)
    size_t ow = idx % OW;
    size_t oh = (idx / OW) % OH;
    size_t c = (idx / (OH * OW)) % OC;
    size_t n = idx / (OC * OH * OW);

    // workspace is column-major from cuBLAS: (N*OH*OW) rows x OC cols
    // workspace[c, n*OH*OW + oh*OW + ow] = workspace_flat[c * (N*OH*OW) + n*OH*OW + oh*OW + ow]
    size_t ws_idx = c * (N * OH * OW) + n * OH * OW + oh * OW + ow;
    float val = workspace[ws_idx];
    if (bias) val += bias[c];
    output[idx] = val;
}

// ---------------------------------------------------------------------------
// Host wrapper
// ---------------------------------------------------------------------------
extern "C" {

void boat_cuda_conv2d_forward_f32(const float* input, const float* weight,
                                   const float* bias, float* output,
                                   size_t N, size_t C, size_t H, size_t W,
                                   size_t OC, size_t KH, size_t KW,
                                   size_t pad, size_t stride, size_t groups) {
    if (groups == 0 || C % groups != 0 || OC % groups != 0) return;

    size_t CG = C / groups;   // channels per group in input
    size_t OCG = OC / groups; // channels per group in output

    // Output spatial dimensions
    size_t OH = (H + 2 * pad - KH) / stride + 1;
    size_t OW = (W + 2 * pad - KW) / stride + 1;

    // cuBLAS handle
    extern cublasHandle_t boat_cuda_get_cublas_handle(void);
    cublasHandle_t handle = boat_cuda_get_cublas_handle();

    // Allocate column matrix on device: [CG*KH*KW, N*OH*OW] per group
    size_t col_rows = CG * KH * KW;
    size_t col_cols = N * OH * OW;
    float *d_col = NULL;
    CUDA_CHECK(cudaMalloc(&d_col, col_rows * col_cols * sizeof(float)));

    // im2col block/grid config
    dim3 block(16, 16);
    dim3 grid((col_cols + 15) / 16, (col_rows + 15) / 16);

    float alpha = 1.0f, beta = 0.0f;

    for (size_t g = 0; g < groups; g++) {
        // Input slice for this group: [N, CG, H, W]
        const float* group_input = input + g * CG * H * W;
        // Weight slice for this group: [OCG, CG, KH, KW]
        const float* group_weight = weight + g * OCG * CG * KH * KW;
        // Output slice for this group: [N, OCG, OH, OW] → col-major [OCG, N*OH*OW]
        float* group_output = output + g * OCG * OH * OW;

        // im2col: group_input → d_col
        im2col_f32_kernel<<<grid, block>>>(
            group_input, d_col, N, CG, H, W, KH, KW, (int)pad, (int)stride, OH, OW);
        CUDA_CHECK(cudaGetLastError());

        // cuBLAS Sgemm: weight [OCG, CG*KH*KW] @ col [CG*KH*KW, N*OH*OW] = out [OCG, N*OH*OW]
        // cuBLAS is column-major, so we call:
        //   C(N*OH*OW, OCG) = op(B)(N*OH*OW, CG*KH*KW) * op(A)(CG*KH*KW, OCG)
        // with op(B)=N, op(A)=N:
        //   cublasSgemm(handle, N, N, N*OH*OW, OCG, CG*KH*KW, &alpha, col, N*OH*OW, weight, CG*KH*KW, &beta, workspace, N*OH*OW)
        // Wait, this is tricky. Let me reason carefully:
        //
        // We want: output[OCG, N*OH*OW] = weight[OCG, CG*KH*KW] * col[CG*KH*KW, N*OH*OW]
        // In column-major: C(M=OCG, N=N*OH*OW) = A(M=OCG, K=CG*KH*KW) * B(K=CG*KH*KW, N=N*OH*OW)
        // cublasSgemm(handle, op(A), op(B), M, N, K, ...)
        // with op(A)=N, op(B)=N, M=OCG, N=N*OH*OW, K=CG*KH*KW:
        //   C = A * B  where A is OCG×CG*KH*KW (col-major = row-major weight), B is CG*KH*KW×N*OH*OW
        // But wait, weight is [OCG, CG, KH, KW] in row-major, which means in memory it's:
        //   weight[oc * (CG*KH*KW) + ...]
        // This is exactly the column-major layout for a matrix of size OCG × (CG*KH*KW).
        // Similarly, col in row-major is [(CG*KH*KW), N*OH*OW] which is column-major for (N*OH*OW) × (CG*KH*KW).
        //
        // In row-major: output[OCG, N*OH*OW] = weight[OCG, CG*KH*KW] * col[CG*KH*KW, N*OH*OW]
        // In col-major: output^T = col^T * weight^T
        //   output_colmajor(N*OH*OW, OCG) = col_colmajor(N*OH*OW, CG*KH*KW) * weight_colmajor(CG*KH*KW, OCG)
        // cublas: C = op(A) * op(B)
        //   op(A)=N, op(B)=N: C(N*OH*OW, OCG) = A(N*OH*OW, CG*KH*KW) * B(CG*KH*KW, OCG)
        //   A = col (col-major: N*OH*OW × CG*KH*KW = row-major in memory)
        //   B = weight (col-major: CG*KH*KW × OCG = row-major weight transposed... no)
        //
        // Hmm this is getting confusing. Let me use a simpler approach:
        //
        // Row-major view: we want output[oc, n*OH*OW] = sum(weight[oc, k] * col[k, n*OH*OW])
        // This is C(M=OCG, N=N*OH*OW) = A(M=OCG, K=CG*KH*KW) * B(K=CG*KH*KW, N=N*OH*OW)
        // in row-major. In cuBLAS column-major:
        //   C^T(N*OH*OW, OCG) = B^T(N*OH*OW, CG*KH*KW) * A^T(CG*KH*KW, OCG)
        // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N*OH*OW, OCG, CG*KH*KW, ...)
        //   where A = col (col-major: N*OH*OW × CG*KH*KW ✓)
        //         B = weight (col-major: CG*KH*KW × OCG ✓)
        //         C = workspace (col-major: N*OH*OW × OCG = row-major OCG × N*OH*OW ✓)
        //
        // Let me verify with the A matrix: in cuBLAS, A is M×K = N*OH*OW × CG*KH*KW in column-major.
        // That means in memory it's stored as [A[0,0], A[1,0], ..., A[N*OH*OW-1,0], A[0,1], ...]
        // which is exactly col[row][col] = col[k][n*OH*OW] stored as col[k * N*OH*OW + n*OH*OW]...
        // Actually no. col[k][n*OH*OW+oh*OW+ow] in row-major = col[k * N*OH*OW + n*OH*OW + oh*OW + ow].
        // As column-major matrix of size (N*OH*OW) × (CG*KH*KW), element (r,c) is at r + c*(N*OH*OW).
        // That's r = n*OH*OW+oh*OW+ow, c = k. So position = n*OH*OW+oh*OW+ow + k*N*OH*OW = exactly row-major.
        // So col in memory IS column-major N*OH*OW × CG*KH*KW. ✓
        //
        // For weight: row-major [oc][c*kh*kw] = weight[oc * CG*KH*KW + ...].
        // As column-major matrix of size (CG*KH*KW) × OCG, element (r,c) is at r + c*(CG*KH*KW).
        // That's r = c*kh*kw_linear, c = oc. Position = c*kh*kw_linear + oc*CG*KH*KW = exactly row-major.
        // So weight in memory IS column-major CG*KH*KW × OCG. ✓
        //
        // So: cublasSgemm(handle, N, N, N*OH*OW, OCG, CG*KH*KW, &alpha, col, N*OH*OW, weight, CG*KH*KW, &beta, output, N*OH*OW)
        // computes: output(N*OH*OW, OCG) = col(N*OH*OW, CG*KH*KW) * weight(CG*KH*KW, OCG)
        // = col^T * weight^T in row-major = (weight * col)^T in row-major
        // Since the output is in column-major N*OH*OW × OCG, in memory it's OCG × N*OH*OW row-major. ✓

        CUBLAS_CHECK(cublasSgemm(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            (int)(N * OH * OW), (int)OCG, (int)(CG * KH * KW),
            &alpha,
            d_col, (int)(N * OH * OW),
            group_weight, (int)(CG * KH * KW),
            &beta,
            group_output, (int)(N * OH * OW)));
    }

    // Apply bias + reshape from column-major workspace to row-major output
    // Since cuBLAS wrote group_output directly as col-major [N*OH*OW, OCG],
    // and we want row-major [N, OCG, OH, OW], and the data is already in the
    // correct memory layout (cuBLAS col-major output = row-major N×OCG×OH×OW),
    // we just need to add bias.
    if (bias) {
        size_t total = N * OC * OH * OW;
        const int bias_block = 256;
        unsigned int bias_grid = (unsigned int)((total + bias_block - 1) / bias_block);
        conv_bias_reshape_f32_kernel<<<bias_grid, bias_block>>>(
            output, bias, output, N, OC, OH, OW);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(d_col));
}

} // extern "C"
