// test_simple.c - Simple test to check linking
#define BOAT_BUILDING_DLL 0  // Explicitly not building DLL
#include <boat/tensor.h>
#include <stdio.h>

int main() {
    printf("Testing boat_tensor_create...\n");
    // Just declare a function pointer to check if symbol is available
    boat_tensor_t* (*func)(const int64_t*, size_t, boat_dtype_t, boat_device_t);

    // This should resolve at link time
    func = boat_tensor_create;

    printf("boat_tensor_create address: %p\n", (void*)func);
    return 0;
}