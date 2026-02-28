#define BOAT_STATIC_BUILD
#include <boat/layers.h>
#include <stdio.h>

int main() {
    printf("Testing boat_dense_layer_create...\n");
    // Just declare a function pointer
    boat_dense_layer_t* (*func)(size_t, size_t, bool);
    func = boat_dense_layer_create;
    printf("boat_dense_layer_create address: %p\n", (void*)func);
    return 0;
}