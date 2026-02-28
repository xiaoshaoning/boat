#include <boat/boat.h>
#include <stdio.h>

int main() {
    printf("Testing default graph mechanism...\n");

    // Create a variable with requires_grad=true
    int64_t shape[] = {2, 3};
    boat_variable_t* var = boat_variable_create_with_shape(shape, 2, BOAT_DTYPE_FLOAT32, true);
    if (!var) {
        fprintf(stderr, "Failed to create variable\n");
        return 1;
    }

    printf("Variable created successfully\n");
    boat_variable_free(var);
    return 0;
}