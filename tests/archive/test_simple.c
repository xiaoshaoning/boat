#include <stdio.h>
#include <boat/tensor.h>

int main() {
    printf("Testing Boat library...\n");
    
    // Create a simple tensor
    int64_t shape[] = {2, 3};
    boat_tensor_t* tensor = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    
    if (tensor) {
        printf("Tensor created successfully!\n");
        printf("Dimensions: %zu\n", boat_tensor_ndim(tensor));
        printf("Data type: %d\n", boat_tensor_dtype(tensor));
        
        boat_tensor_unref(tensor);
        printf("Tensor freed.\n");
    } else {
        printf("Failed to create tensor.\n");
    }
    
    return 0;
}
