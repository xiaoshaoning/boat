#include <stdio.h>
#include <stdint.h>
#include <string.h>

// Simulate the transpose algorithm
void transpose_algorithm(const int64_t* shape, size_t ndim, int dim0, int dim1,
                         const float* in_data, float* out_data) {
    size_t total_elements = 1;
    for (size_t i = 0; i < ndim; i++) total_elements *= shape[i];

    // Output shape (swapped)
    int64_t out_shape[ndim];
    for (size_t i = 0; i < ndim; i++) out_shape[i] = shape[i];
    out_shape[dim0] = shape[dim1];
    out_shape[dim1] = shape[dim0];

    // Compute strides
    size_t in_stride[ndim];
    size_t out_stride[ndim];

    in_stride[ndim-1] = 1;
    for (int i = ndim-2; i >= 0; i--) {
        in_stride[i] = in_stride[i+1] * shape[i+1];
    }

    out_stride[ndim-1] = 1;
    for (int i = ndim-2; i >= 0; i--) {
        out_stride[i] = out_stride[i+1] * out_shape[i+1];
    }

    printf("Input shape: [%lld, %lld]\n", shape[0], shape[1]);
    printf("Output shape: [%lld, %lld]\n", out_shape[0], out_shape[1]);
    printf("Input strides: [%zu, %zu]\n", in_stride[0], in_stride[1]);
    printf("Output strides: [%zu, %zu]\n", out_stride[0], out_stride[1]);

    for (size_t idx = 0; idx < total_elements; idx++) {
        size_t temp = idx;
        size_t coords[ndim];
        for (int i = ndim-1; i >= 0; i--) {
            coords[i] = temp % (size_t)shape[i];
            temp /= (size_t)shape[i];
        }

        size_t temp_coord = coords[dim0];
        coords[dim0] = coords[dim1];
        coords[dim1] = temp_coord;

        size_t out_idx = 0;
        for (size_t i = 0; i < ndim; i++) {
            out_idx += coords[i] * out_stride[i];
        }

        if (idx < 6) {
            printf("idx=%zu: coords=(%zu,%zu) swapped=(%zu,%zu) out_idx=%zu\n",
                   idx,
                   idx/3, idx%3,  // original coordinates (i,j)
                   coords[0], coords[1], out_idx);
        }

        out_data[out_idx] = in_data[idx];
    }
}

int main() {
    // 2x3 matrix
    int64_t shape[] = {2, 3};
    float in_data[6] = {1,2,3,4,5,6};
    float out_data[6] = {0};

    transpose_algorithm(shape, 2, 0, 1, in_data, out_data);

    printf("\nInput: ");
    for (int i = 0; i < 6; i++) printf("%.1f ", in_data[i]);
    printf("\nOutput: ");
    for (int i = 0; i < 6; i++) printf("%.1f ", out_data[i]);
    printf("\nExpected: 1.0 4.0 2.0 5.0 3.0 6.0\n");

    // Check correctness
    float expected[6] = {1,4,2,5,3,6};
    int correct = 1;
    for (int i = 0; i < 6; i++) {
        if (out_data[i] != expected[i]) {
            printf("Mismatch at %d: %.1f vs %.1f\n", i, out_data[i], expected[i]);
            correct = 0;
        }
    }
    printf(correct ? "Algorithm CORRECT\n" : "Algorithm WRONG\n");

    return 0;
}