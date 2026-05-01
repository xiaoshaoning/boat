// test_cuda_mini.c - Minimal CUDA device query
#include <boat/cuda_runtime.h>
#include <stdio.h>

int main() {
    printf("Querying CUDA devices...\n");
    int count = boat_cuda_device_count();
    printf("Devices: %d\n", count);
    if (count > 0) {
        int dev = boat_cuda_get_device();
        printf("Current device: %d\n", dev);
        printf("CUDA runtime works!\n");
    }
    return 0;
}
