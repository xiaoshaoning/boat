// test_error.c - Test error handling implementation
#include <boat.h>
#include <stdio.h>

int main() {
    printf("Testing error handling...\n");

    // Clear any existing error
    boat_clear_error();

    // Test boat_error_string
    printf("BOAT_SUCCESS: %s\n", boat_error_string(BOAT_SUCCESS));
    printf("BOAT_ERROR_INVALID_ARGUMENT: %s\n", boat_error_string(BOAT_ERROR_INVALID_ARGUMENT));
    printf("BOAT_ERROR_OUT_OF_MEMORY: %s\n", boat_error_string(BOAT_ERROR_OUT_OF_MEMORY));
    printf("BOAT_ERROR_NOT_IMPLEMENTED: %s\n", boat_error_string(BOAT_ERROR_NOT_IMPLEMENTED));

    // Test boat_set_error and boat_get_last_error
    boat_set_error(BOAT_ERROR_FILE_IO, "Failed to open file");
    printf("Last error: %d\n", boat_get_last_error());
    printf("Last error message: %s\n", boat_get_last_error_message());
    printf("Has error: %d\n", boat_has_error());

    // Test boat_set_errorf
    boat_set_errorf(BOAT_ERROR_FORMAT, "Invalid format: %s", "PNG");
    printf("Formatted error: %s\n", boat_get_last_error_message());

    // Test boat_reset_error
    boat_reset_error();
    printf("After reset, has error: %d\n", boat_has_error());

    printf("Error handling test passed!\n");
    return 0;
}