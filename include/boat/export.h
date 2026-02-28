// export.h - Platform-specific export macros for Boat framework
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#ifndef BOAT_EXPORT_H
#define BOAT_EXPORT_H

// Platform detection
#if defined(_WIN32) || defined(_WIN64)
    #define BOAT_WINDOWS 1
#else
    #define BOAT_WINDOWS 0
#endif

// Calling convention macros
#if BOAT_WINDOWS && defined(_M_X64)
    // Windows x64: default calling convention (no explicit specifier needed)
    #define BOAT_CALL
#elif BOAT_WINDOWS
    // Windows x86: use __stdcall for compatibility
    #define BOAT_CALL __stdcall
#else
    // Non-Windows platforms
    #define BOAT_CALL
#endif

// Compiler-specific export/import macros
#ifdef BOAT_STATIC_BUILD
    // Static library build - no export/import needed
    #define BOAT_API
#elif BOAT_WINDOWS
    // Windows DLL export/import
    #ifdef BOAT_BUILDING_DLL
        #define BOAT_API __declspec(dllexport)
    #else
        #define BOAT_API __declspec(dllimport)
    #endif
#else
    // Non-Windows platforms (Linux, macOS, etc.)
    #if __GNUC__ >= 4
        #define BOAT_API __attribute__((visibility("default")))
    #else
        #define BOAT_API
    #endif
#endif

// For functions that are always inline or static
#define BOAT_INLINE static inline
#define BOAT_STATIC static

// Function inlining control
#if defined(_MSC_VER)
    #define BOAT_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
    #define BOAT_NOINLINE __attribute__((noinline))
#else
    #define BOAT_NOINLINE
#endif

// For deprecated functions
#if defined(__GNUC__) || defined(__clang__)
    #define BOAT_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER)
    #define BOAT_DEPRECATED __declspec(deprecated)
#else
    #define BOAT_DEPRECATED
#endif

// Debug output control
#ifndef BOAT_DEBUG
#define BOAT_DEBUG 0
#endif

#endif // BOAT_EXPORT_H