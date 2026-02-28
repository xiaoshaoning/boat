#include <boat/boat.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Create tensor A: shape [2, 3]
    int64_t shape_a[] = {2, 3};
    boat_tensor_t* a = boat_tensor_create(shape_a, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* a_data = (float*)boat_tensor_data(a);
    for (int i = 0; i < 6; i++) a_data[i] = (float)i;
    
    // Create tensor B: shape [3] (will broadcast to [2,3])
    int64_t shape_b[] = {3};
    boat_tensor_t* b = boat_tensor_create(shape_b, 1, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* b_data = (float*)boat_tensor_data(b);
    for (int i = 0; i < 3; i++) b_data[i] = (float)(i + 10);
    
    printf("Tensor A shape: [%lld, %lld]\n", shape_a[0], shape_a[1]);
    printf("Tensor B shape: [%lld]\n", shape_b[0]);
    
    // Perform broadcast addition
    boat_tensor_t* c = boat_add(a, b);
    if (!c) {
        printf("ERROR: boat_add returned NULL\n");
        boat_tensor_free(a);
        boat_tensor_free(b);
        return 1;
    }
    
    // Check output shape
    const int64_t* shape_c = boat_tensor_shape(c);
    size_t ndim_c = boat_tensor_ndim(c);
    printf("Output shape: [");
    for (size_t i = 0; i < ndim_c; i++) {
        printf("%lld", shape_c[i]);
        if (i < ndim_c - 1) printf(", ");
    }
    printf("]\n");
    
    // Expected result:
    // A = [[0,1,2], [3,4,5]]
    // B = [10,11,12]
    // C = [[10,12,14], [13,15,17]]
    float* c_data = (float*)boat_tensor_data(c);
    float expected[] = {10.0f, 12.0f, 14.0f, 13.0f, 15.0f, 17.0f};
    int correct = 1;
    for (int i = 0; i < 6; i++) {
        if (c_data[i] != expected[i]) {
            printf("Mismatch at index %d: got %f, expected %f\n", i, c_data[i], expected[i]);
            correct = 0;
        }
    }
    
    if (correct) {
        printf("Broadcast addition test PASSED\n");
    } else {
        printf("Broadcast addition test FAILED\n");
    }
    
    boat_tensor_free(a);
    boat_tensor_free(b);
    boat_tensor_free(c);
    
    return correct ? 0 : 1;
}
