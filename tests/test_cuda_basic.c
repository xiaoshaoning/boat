// test_cuda_basic.c - Verify CUDA backend integration
#include <boat/tensor.h>
#include <boat/memory.h>
#include <boat/cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main() {
    int errors = 0;
    printf("=== CUDA Backend Sanity Check ===\n\n");

    // 1. Device query
    printf("[Test] boat_cuda_device_count...\n");
    int ndev = boat_cuda_device_count();
    printf("  Devices found: %d\n", ndev);
    if (ndev <= 0) {
        fprintf(stderr, "  No CUDA devices available, skipping GPU tests.\n");
        // Not a failure — the machine might not have a GPU
        printf("  SKIP (no device)\n");
    } else {
        printf("  PASS\n");

        int dev = boat_cuda_get_device();
        printf("  Current device: %d\n", dev);

        // 2. CUDA malloc/free
        printf("\n[Test] boat_cuda_malloc/free...\n");
        size_t n = 1024;
        float* d_ptr = (float*)boat_cuda_malloc(n * sizeof(float));
        if (!d_ptr) {
            fprintf(stderr, "  FAIL: malloc returned NULL\n");
            errors++;
        } else {
            boat_cuda_free(d_ptr);
            printf("  PASS\n");
        }

        // 3. CUDA memcpy H2D + kernel launch (add)
        printf("\n[Test] boat_cuda_add_f32 kernel...\n");
        float h_a[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float h_b[] = {5.0f, 6.0f, 7.0f, 8.0f};
        float h_c[4] = {0};
        float* d_a = (float*)boat_cuda_malloc(4 * sizeof(float));
        float* d_b = (float*)boat_cuda_malloc(4 * sizeof(float));
        float* d_c = (float*)boat_cuda_malloc(4 * sizeof(float));

        boat_cuda_memcpy_h2d(d_a, h_a, 4 * sizeof(float));
        boat_cuda_memcpy_h2d(d_b, h_b, 4 * sizeof(float));
        boat_cuda_add_f32(d_a, d_b, d_c, 4);
        boat_cuda_memcpy_d2h(h_c, d_c, 4 * sizeof(float));

        int add_ok = 1;
        for (int i = 0; i < 4; i++) {
            if (fabs(h_c[i] - (h_a[i] + h_b[i])) > 1e-5f) {
                add_ok = 0;
                break;
            }
        }
        printf("  %s\n", add_ok ? "PASS" : "FAIL");
        if (!add_ok) errors++;

        boat_cuda_free(d_a);
        boat_cuda_free(d_b);
        boat_cuda_free(d_c);

        // 4. Tensor CPU→CUDA→CPU roundtrip via boat_tensor_to_device
        printf("\n[Test] boat_tensor_to_device CPU→CUDA→CPU...\n");
        int64_t shape[] = {2, 2};
        boat_tensor_t* t_cpu = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        float* tdata = (float*)boat_tensor_data(t_cpu);
        tdata[0] = 1.0f;
        tdata[1] = 2.0f;
        tdata[2] = 3.0f;
        tdata[3] = 4.0f;

        boat_tensor_t* t_cuda = boat_tensor_to_device(t_cpu, BOAT_DEVICE_CUDA);
        if (!t_cuda) {
            fprintf(stderr, "  FAIL: tensor_to_device(CUDA) returned NULL\n");
            errors++;
        } else if (boat_tensor_device(t_cuda) != BOAT_DEVICE_CUDA) {
            fprintf(stderr, "  FAIL: tensor device is not CUDA\n");
            errors++;
        } else {
            boat_tensor_t* t_back = boat_tensor_to_device(t_cuda, BOAT_DEVICE_CPU);
            if (!t_back) {
                fprintf(stderr, "  FAIL: tensor_to_device(CPU) returned NULL\n");
                errors++;
            } else {
                float* back_data = (float*)boat_tensor_data(t_back);
                int rt_ok = 1;
                for (int i = 0; i < 4; i++) {
                    if (fabs(back_data[i] - (float)(i + 1)) > 1e-5f) {
                        rt_ok = 0;
                        break;
                    }
                }
                printf("  %s\n", rt_ok ? "PASS" : "FAIL");
                if (!rt_ok) errors++;
            }
            boat_tensor_unref(t_back);
        }
        boat_tensor_unref(t_cpu);
        boat_tensor_unref(t_cuda);

        // 5. Tensor clone across devices
        printf("\n[Test] boat_tensor_clone on CUDA tensor...\n");
        int64_t shape2[] = {4};
        boat_tensor_t* src = boat_tensor_create(shape2, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
        float* sdata = (float*)boat_tensor_data(src);
        sdata[0] = 10.0f;
        sdata[1] = 20.0f;
        sdata[2] = 30.0f;
        sdata[3] = 40.0f;

        boat_tensor_t* src_cuda = boat_tensor_to_device(src, BOAT_DEVICE_CUDA);
        boat_tensor_t* cloned = boat_tensor_clone(src_cuda);
        if (!cloned) {
            fprintf(stderr, "  FAIL: clone returned NULL\n");
            errors++;
        } else {
            boat_tensor_t* cloned_cpu = boat_tensor_to_device(cloned, BOAT_DEVICE_CPU);
            float* cdata = (float*)boat_tensor_data(cloned_cpu);
            int clone_ok = 1;
            for (int i = 0; i < 4; i++) {
                if (fabs(cdata[i] - (float)((i + 1) * 10)) > 1e-5f) {
                    clone_ok = 0;
                    break;
                }
            }
            printf("  %s\n", clone_ok ? "PASS" : "FAIL");
            if (!clone_ok) errors++;
            boat_tensor_unref(cloned_cpu);
        }
        boat_tensor_unref(cloned);
        boat_tensor_unref(src_cuda);
        boat_tensor_unref(src);

        // 6. CUDA element-wise operations
        printf("\n[Test] CUDA element-wise ops (relu, sigmoid, mul)...\n");
        float h_in[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        float h_out[5] = {0};
        float* d_in = (float*)boat_cuda_malloc(5 * sizeof(float));
        float* d_out = (float*)boat_cuda_malloc(5 * sizeof(float));
        boat_cuda_memcpy_h2d(d_in, h_in, 5 * sizeof(float));

        // ReLU
        boat_cuda_relu_f32(d_in, d_out, 5);
        boat_cuda_memcpy_d2h(h_out, d_out, 5 * sizeof(float));
        int relu_ok = 1;
        float expected_relu[] = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};
        for (int i = 0; i < 5; i++)
            if (fabs(h_out[i] - expected_relu[i]) > 1e-5f) relu_ok = 0;
        printf("  ReLU: %s\n", relu_ok ? "PASS" : "FAIL");
        if (!relu_ok) errors++;

        // Sigmoid
        boat_cuda_sigmoid_f32(d_in, d_out, 5);
        boat_cuda_memcpy_d2h(h_out, d_out, 5 * sizeof(float));
        int sig_ok = 1;
        for (int i = 0; i < 5; i++) {
            float expected = 1.0f / (1.0f + expf(-h_in[i]));
            if (fabs(h_out[i] - expected) > 1e-5f) sig_ok = 0;
        }
        printf("  Sigmoid: %s\n", sig_ok ? "PASS" : "FAIL");
        if (!sig_ok) errors++;

        // Mul scalar
        boat_cuda_mul_scalar_f32(d_in, 3.0f, d_out, 5);
        boat_cuda_memcpy_d2h(h_out, d_out, 5 * sizeof(float));
        int mul_ok = 1;
        for (int i = 0; i < 5; i++)
            if (fabs(h_out[i] - h_in[i] * 3.0f) > 1e-5f) mul_ok = 0;
        printf("  MulScalar: %s\n", mul_ok ? "PASS" : "FAIL");
        if (!mul_ok) errors++;

        boat_cuda_free(d_in);
        boat_cuda_free(d_out);

        // 7. CUDA sum reduction
        printf("\n[Test] boat_cuda_sum_f32 reduction...\n");
        size_t big_n = 10000;
        float* h_big = (float*)malloc(big_n * sizeof(float));
        for (size_t i = 0; i < big_n; i++)
            h_big[i] = 1.0f;
        float* d_big = (float*)boat_cuda_malloc(big_n * sizeof(float));
        boat_cuda_memcpy_h2d(d_big, h_big, big_n * sizeof(float));
        float result = boat_cuda_sum_f32(d_big, big_n);
        int sum_ok = (fabs(result - (float)big_n) < 1e-3f);
        printf("  sum(10000 ones) = %f (expected 10000) — %s\n", result, sum_ok ? "PASS" : "FAIL");
        if (!sum_ok) errors++;
        boat_cuda_free(d_big);
        free(h_big);
    }

    printf("\n=== %s ===\n", errors == 0 ? "ALL TESTS PASSED" : "FAILED");
    return errors == 0 ? 0 : 1;
}
