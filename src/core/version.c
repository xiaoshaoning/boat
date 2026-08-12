// version.c - Version information, library init/cleanup for Boat Deep Learning Framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/version.h>
#include <stdio.h>

// Library initialization
BOAT_API void boat_init(void) {
    // Currently no global state to initialize.
    // Reserved for future initialization of device backends,
    // global thread pool, logging system, etc.
}

// Library cleanup
BOAT_API void boat_cleanup(void) {
    // Currently no global state to clean up.
    // Reserved for future cleanup.
}

// Get version as string
BOAT_API const char* boat_get_version_string(void) {
    return BOAT_VERSION_STRING;
}

// Get git hash
BOAT_API const char* boat_get_git_hash(void) {
    return BOAT_GIT_HASH;
}

// Get git describe
BOAT_API const char* boat_get_git_describe(void) {
    return BOAT_GIT_DESCRIBE;
}

// Get full version string with git hash
BOAT_API const char* boat_get_version_full(void) {
    return BOAT_VERSION_FULL;
}

// Get version components
BOAT_API void boat_get_version(int* major, int* minor, int* patch) {
    if (major) *major = BOAT_VERSION_MAJOR;
    if (minor) *minor = BOAT_VERSION_MINOR;
    if (patch) *patch = BOAT_VERSION_PATCH;
}

// Get version as integer
BOAT_API unsigned int boat_get_version_int(void) {
    return BOAT_VERSION_INT;
}