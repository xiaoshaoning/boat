// error.c - Error handling implementation for Boat framework
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#include <boat.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

// Thread-local error state
#ifdef _WIN32
#define BOAT_THREAD_LOCAL __declspec(thread)
#else
#define BOAT_THREAD_LOCAL __thread
#endif

static BOAT_THREAD_LOCAL boat_error_t boat_last_error = BOAT_SUCCESS;
static BOAT_THREAD_LOCAL char boat_last_error_msg[256] = "";

// Convert error code to string
const char* boat_error_string(boat_error_t error) {
    switch (error) {
        case BOAT_SUCCESS: return "Success";
        case BOAT_ERROR_INVALID_ARGUMENT: return "Invalid argument";
        case BOAT_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case BOAT_ERROR_INVALID_OPERATION: return "Invalid operation";
        case BOAT_ERROR_DEVICE: return "Device error";
        case BOAT_ERROR_FILE_IO: return "File I/O error";
        case BOAT_ERROR_FORMAT: return "Format error";
        case BOAT_ERROR_NOT_IMPLEMENTED: return "Not implemented";
        case BOAT_ERROR_UNKNOWN: return "Unknown error";
        default: return "Invalid error code";
    }
}

// Get last error code
boat_error_t boat_get_last_error() {
    return boat_last_error;
}

// Clear last error
void boat_clear_error() {
    boat_last_error = BOAT_SUCCESS;
    boat_last_error_msg[0] = '\0';
}

// Set error with message
void boat_set_error(boat_error_t error, const char* message) {
    boat_last_error = error;
    if (message) {
        strncpy(boat_last_error_msg, message, sizeof(boat_last_error_msg) - 1);
        boat_last_error_msg[sizeof(boat_last_error_msg) - 1] = '\0';
    } else {
        boat_last_error_msg[0] = '\0';
    }
}

// Get last error message
const char* boat_get_last_error_message() {
    return boat_last_error_msg;
}

// Set error with formatted message
void boat_set_errorf(boat_error_t error, const char* format, ...) {
    va_list args;
    va_start(args, format);
    boat_last_error = error;
    vsnprintf(boat_last_error_msg, sizeof(boat_last_error_msg), format, args);
    va_end(args);
}

// Check if error occurred
bool boat_has_error() {
    return boat_last_error != BOAT_SUCCESS;
}

// Reset error state
void boat_reset_error() {
    boat_clear_error();
}