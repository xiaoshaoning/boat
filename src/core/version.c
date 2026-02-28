// version.c - Version information implementation for Boat Deep Learning Framework
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#include <boat/version.h>

// Get version as string
const char* boat_get_version_string(void) {
    return BOAT_VERSION_STRING;
}

// Get git hash
const char* boat_get_git_hash(void) {
    return BOAT_GIT_HASH;
}

// Get git describe
const char* boat_get_git_describe(void) {
    return BOAT_GIT_DESCRIBE;
}

// Get full version string with git hash
const char* boat_get_version_full(void) {
    return BOAT_VERSION_FULL;
}

// Get version components
void boat_get_version(int* major, int* minor, int* patch) {
    if (major) *major = BOAT_VERSION_MAJOR;
    if (minor) *minor = BOAT_VERSION_MINOR;
    if (patch) *patch = BOAT_VERSION_PATCH;
}

// Get version as integer
unsigned int boat_get_version_int(void) {
    return BOAT_VERSION_INT;
}