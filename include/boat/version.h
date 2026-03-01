// version.h - Version information for Boat Deep Learning Framework
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#ifndef BOAT_VERSION_H
#define BOAT_VERSION_H

#include "export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Framework version
#define BOAT_VERSION_MAJOR 0
#define BOAT_VERSION_MINOR 1
#define BOAT_VERSION_PATCH 0
#define BOAT_VERSION_STRING "0.1.0"
#define BOAT_GIT_HASH "7039bc9"
#define BOAT_GIT_DESCRIBE "7039bc9"

// Combined version string with git info
#define BOAT_VERSION_FULL BOAT_VERSION_STRING "-" BOAT_GIT_HASH

// Version check macros
#define BOAT_VERSION_CHECK(major, minor, patch) \
    ((major) < BOAT_VERSION_MAJOR || \
     ((major) == BOAT_VERSION_MAJOR && (minor) < BOAT_VERSION_MINOR) || \
     ((major) == BOAT_VERSION_MAJOR && (minor) == BOAT_VERSION_MINOR && (patch) <= BOAT_VERSION_PATCH))

// Get version as integer (major << 16 | minor << 8 | patch)
#define BOAT_VERSION_INT ((BOAT_VERSION_MAJOR << 16) | (BOAT_VERSION_MINOR << 8) | BOAT_VERSION_PATCH)

// API to get version information
BOAT_API const char* boat_get_version_string(void);
BOAT_API const char* boat_get_git_hash(void);
BOAT_API const char* boat_get_git_describe(void);
BOAT_API const char* boat_get_version_full(void);
BOAT_API void boat_get_version(int* major, int* minor, int* patch);
BOAT_API unsigned int boat_get_version_int(void);

#ifdef __cplusplus
}
#endif

#endif // BOAT_VERSION_H
