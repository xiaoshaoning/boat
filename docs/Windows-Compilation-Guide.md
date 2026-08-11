# Windows Compilation Guide

This document describes in detail the steps, considerations, and best practices for compiling the Boat deep learning framework on the Windows platform. It focuses on MSVC compiler-specific behavior, DLL build optimization control, and cross-platform compatibility guarantees.

## Table of Contents

1. [Environment Preparation](#environment-preparation)
2. [Build Configuration](#build-configuration)
3. [MSVC Compiler Features](#msvc-compiler-features)
4. [DLL Export Best Practices](#dll-export-best-practices)
5. [Compiler Optimization Control](#compiler-optimization-control)
6. [Debugging and Diagnostics](#debugging-and-diagnostics)
7. [Troubleshooting](#troubleshooting)
8. [Cross-Platform Considerations](#cross-platform-considerations)

## Environment Preparation

### Required Tools

1. **Visual Studio 2022 or later**
   - Includes the MSVC compiler toolchain
   - It is recommended to install the "Desktop development with C++" workload
   - Ensure the Windows SDK version is 10.0.19041.0 or later

2. **CMake 3.20 or later**
   - Download from [cmake.org](https://cmake.org/download/)
   - Add CMake to the system PATH

3. **Git**
   - Used to fetch the source code
   - Git for Windows is recommended

4. **Optional: Windows Terminal**
   - Provides a better command-line experience

### Environment Verification

```bash
# Verify MSVC compiler
cl.exe

# Verify CMake
cmake --version

# Verify Git
git --version
```

## Build Configuration

### Basic Build Steps

```bash
# 1. Clone repository
git clone https://github.com/your-org/boat.git
cd boat

# 2. Create build directory
mkdir build
cd build

# 3. Configure CMake (shared library build)
cmake .. -DBOAT_BUILD_SHARED=ON -DCMAKE_BUILD_TYPE=Release

# 4. Build project
cmake --build . --config Release
```

### Key CMake Options

| Option | Description | Default |
|------|------|--------|
| `BOAT_BUILD_SHARED` | Build a shared library (DLL); builds a static library by default | `OFF` |
| `BOAT_WITH_TESTS` | Build the tests | `OFF` |
| `BOAT_WITH_EXAMPLES` | Build the examples | `OFF` |
| `CMAKE_BUILD_TYPE` | Build type (Debug/Release/RelWithDebInfo) | unset |
| `CMAKE_INSTALL_PREFIX` | Installation directory | `C:/Program Files/boat` |

### Advanced Build Configuration

```bash
# Debug build with symbol information
cmake .. -DBOAT_BUILD_SHARED=ON -DCMAKE_BUILD_TYPE=Debug

# Release build with debug info
cmake .. -DBOAT_BUILD_SHARED=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Specify generator (Visual Studio 2022)
cmake .. -G "Visual Studio 17 2022" -A x64 -DBOAT_BUILD_SHARED=ON
```

### MinGW-w64 (GCC) Build

Boat can also be built on Windows using MinGW-w64 GCC (verified with the MSYS2
`mingw64` toolchain, gcc 13, ctest **29/29** all passing). The differences from an
MSVC build: a static library (`libboat.a`) is produced by default, using `-G "MinGW Makefiles"`.

```bash
# 1. Install MSYS2 and add mingw64 to PATH
#    https://www.msys2.org/  → pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake

# 2. Configure (explicitly specify the mingw compiler so MSVC is not picked)
cmake -S . -B build-mingw -G "MinGW Makefiles" ^
  -DCMAKE_C_COMPILER=C:/msys64/mingw64/bin/gcc.exe ^
  -DBOAT_WITH_TESTS=ON -DBOAT_WITH_EXAMPLES=ON ^
  -DBOAT_WITH_HUGGINGFACE=ON -DBOAT_WITH_ONNX=ON -DBOAT_WITH_GGUF=ON ^
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

# 3. Build and test
cmake --build build-mingw -j
ctest --test-dir build-mingw --output-on-failure -j
```

Notes:
- When running the generated `.exe`, add `C:\msys64\mingw64\bin` to `PATH`
  (it depends on runtime DLLs such as `libgomp-1.dll` and `libwinpthread-1.dll`).
- MinGW GCC reports a few pre-existing `-Wdiscarded-qualifiers` / `-Wattributes`
  warnings (e.g., discarded const in `src/graph/edge.c`, `error.c`'s `__thread`
  attribute); these do not affect the build or test results.
- The example programs (`examples/regression`, `serialization`, `transformer`, etc.)
  all run correctly under a MinGW build.

## MSVC Compiler Features

### Calling Convention

The Windows x64 platform uses a single unified calling convention, so no explicit specifier is required. The Windows x86 platform requires `__stdcall`.

```c
// include/boat/export.h
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
```

### DLL Export/Import

Use `__declspec(dllexport)` and `__declspec(dllimport)` to control symbol visibility.

```c
// include/boat/export.h
#if BOAT_WINDOWS
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
```

### Compiler Optimization Behavior

In Release mode, the MSVC compiler applies aggressive optimizations that can lead to the following issues:

1. **Function-Level Linking (/Gy)**
   - Packages functions into COMDAT sections
   - Allows the linker to eliminate unreferenced functions
   - Can cause simple wrapper functions to be eliminated

2. **Inlining Optimization**
   - Automatically inlines small functions
   - For simple wrapper functions, the function body may be optimized away entirely

3. **Global Optimization (/GL) and Link-Time Code Generation (LTCG)**
   - Cross-module optimization
   - Increases optimization capability, but may also introduce unexpected behavior

## DLL Export Best Practices

### 1. Protecting Simple Wrapper Functions

Simple wrapper functions (especially wrappers that only call another function) can easily be eliminated by compiler optimization. Use `BOAT_NOINLINE` to force the function body to be retained.

```c
// Dangerous: simple wrapper may be optimized away
BOAT_API boat_tensor_t* boat_norm_layer_backward(boat_norm_layer_t* layer,
                                                 const boat_tensor_t* grad_output) {
    return boat_layernorm_backward(layer, grad_output);
}

// Safe: use BOAT_NOINLINE to prevent optimization
BOAT_NOINLINE BOAT_API boat_tensor_t* boat_norm_layer_backward(boat_norm_layer_t* layer,
                                                               const boat_tensor_t* grad_output) {
    return boat_layernorm_backward(layer, grad_output);
}
```

### 2. Consistent Function Declarations

Ensure that header declarations and source definitions are consistent:

```c
// include/boat/layers.h - header declaration
BOAT_API boat_tensor_t* boat_dense_layer_backward(boat_dense_layer_t* layer,
                                                  const boat_tensor_t* grad_output);

// src/layers/dense.c - source definition (correct)
BOAT_API boat_tensor_t* boat_dense_layer_backward(boat_dense_layer_t* layer,
                                                  const boat_tensor_t* grad_output) {
    // implementation
}

// src/layers/dense.c - source definition (wrong, missing BOAT_API)
boat_tensor_t* boat_dense_layer_backward(boat_dense_layer_t* layer,
                                        const boat_tensor_t* grad_output) {
    // implementation - may not be exported correctly
}
```

### 3. Unified Calling Convention

Use the `BOAT_CALL` macro on all exported functions to ensure cross-platform consistency:

```c
// Recommended: use BOAT_CALL for correct calling convention
BOAT_API boat_tensor_t* BOAT_CALL boat_attention_layer_forward(boat_attention_layer_t* layer,
                                                               const boat_tensor_t* query,
                                                               const boat_tensor_t* key,
                                                               const boat_tensor_t* value,
                                                               const boat_tensor_t* attention_mask);
```

### 4. Function Complexity Thresholds

For the following types of functions, adding `BOAT_NOINLINE` is recommended:

| Function Type | Example | Risk Level |
|----------|------|----------|
| Simple wrapper | Only calls another function | High |
| Returns a constant | Returns `NULL`, `0`, etc. | High |
| Simple getter/setter | Returns a member variable or argument | Medium |
| Error-checking wrapper | Checks arguments, then calls the actual function | Medium |
| Complex function | Contains loops, allocations, system calls | Low |

## Compiler Optimization Control

### The BOAT_NOINLINE Macro

The framework provides a unified `BOAT_NOINLINE` macro that handles the noinline attribute across platforms:

```c
// include/boat/export.h
#if defined(_MSC_VER)
    #define BOAT_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
    #define BOAT_NOINLINE __attribute__((noinline))
#else
    #define BOAT_NOINLINE
#endif
```

### Use Cases

1. **All layer interface wrapper functions**
   ```c
   BOAT_NOINLINE BOAT_API boat_tensor_t* boat_norm_layer_backward(...);
   BOAT_NOINLINE BOAT_API boat_tensor_t* boat_attention_layer_forward(...);
   ```

2. **Simple utility functions**
   ```c
   BOAT_NOINLINE BOAT_API const char* boat_get_version_string(void);
   BOAT_NOINLINE BOAT_API size_t boat_get_alignment(void);
   ```

3. **Initialization/cleanup functions**
   ```c
   BOAT_NOINLINE BOAT_API void boat_initialize(void);
   BOAT_NOINLINE BOAT_API void boat_cleanup(void);
   ```

### Compiler Option Control

Control compiler optimization options in CMake:

```cmake
# CMakeLists.txt snippet
if(MSVC)
    # Disable function-level linking (prevent simple functions from being eliminated)
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /Gy-")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Gy-")

    # Prevent elimination of unreferenced functions
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /OPT:NOREF")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /OPT:NOREF")
endif()
```

## Debugging and Diagnostics

### Verifying DLL Export Symbols

Verify the DLL's exported symbols after the build:

```bash
# Use dumpbin to check exported functions
dumpbin /exports build/Release/boat.dll

# Find specific function
dumpbin /exports build/Release/boat.dll | findstr "boat_attention_layer_backward"
```

### Debug Build Configuration

Create a dedicated debug build configuration:

```bash
# Debug DLL build
cmake .. -DBOAT_BUILD_SHARED=ON -DCMAKE_BUILD_TYPE=Debug

# Build and generate PDB files
cmake --build . --config Debug
```

### Runtime Diagnostics

Add platform-specific diagnostics to your code:

```c
#include <boat/export.h>

BOAT_NOINLINE BOAT_API boat_tensor_t* BOAT_CALL boat_attention_layer_backward(...) {
    // platform-specific debug output
#ifdef _MSC_VER
    OutputDebugStringA("[DEBUG] boat_attention_layer_backward called\n");
#endif

    // function implementation
    // ...
}
```

## Troubleshooting

### Issue 1: Missing DLL Export Functions

**Symptoms**: When inspected with `dumpbin /exports`, an expected function is missing from the export table.

**Possible causes**:
1. The function does not use the `BOAT_API` modifier
2. The function was eliminated by compiler optimization
3. Linker optimization removed the unreferenced function

**Solutions**:
1. Ensure both the declaration and definition use `BOAT_API`
2. Add `BOAT_NOINLINE` to simple wrapper functions
3. Adjust the linker option: `/OPT:NOREF`

### Issue 2: Crash on Function Call

**Symptoms**: The program crashes when calling a DLL-exported function.

**Possible causes**:
1. Calling convention mismatch
2. Stack pointer misalignment
3. Incorrect argument passing

**Solutions**:
1. Ensure all exported functions use the `BOAT_CALL` macro
2. Verify function signature consistency
3. Use a debugger to inspect the call stack

### Issue 3: Performance Test Failure

**Symptoms**: A function returns `NULL` or an invalid value during performance testing.

**Possible causes**:
1. The function body was eliminated by compiler optimization
2. Cache management issues
3. Thread synchronization issues

**Solutions**:
1. Add `BOAT_NOINLINE` to prevent optimization
2. Add debug output to verify function execution
3. Check cache validity

### Issue 4: Inconsistent Cross-Platform Behavior

**Symptoms**: The code fails on Windows but works correctly on Linux/macOS.

**Possible causes**:
1. Platform-specific compiler optimization differences
2. DLL and shared library mechanisms differ
3. Calling convention differences

**Solutions**:
1. Use unified macros to handle platform differences
2. Run the full test suite on each platform
3. Implement platform compatibility tests

## Cross-Platform Considerations

### Macro Definition Compatibility

Ensure all platform-specific code is handled through macro definitions:

```c
// Wrong: directly using platform-specific syntax
#ifdef _MSC_VER
__declspec(noinline)
#endif
void my_function();

// Correct: use framework macro
BOAT_NOINLINE void my_function();
```

### Build System Compatibility

The CMake configuration should handle all platform differences:

```cmake
# Platform-specific compiler options
if(MSVC)
    set(PLATFORM_C_FLAGS "/Gy- /OPT:NOREF")
elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set(PLATFORM_C_FLAGS "-fno-inline-functions")
endif()

# Apply to all build targets
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${PLATFORM_C_FLAGS}")
```

### Testing Strategy

1. **Platform matrix testing**: Run the full test suite on Windows, Linux, and macOS
2. **Build type testing**: Test the Debug, Release, and RelWithDebInfo configurations
3. **Link type testing**: Test static and shared library builds
4. **Compiler testing**: Test different compiler versions (MSVC, GCC, Clang)

### Continuous Integration

Configure the CI pipeline to include:

```yaml
# GitHub Actions example
jobs:
  windows-build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      - name: Configure CMake
        run: cmake -DBOAT_BUILD_SHARED=ON -DBOAT_WITH_TESTS=ON -B build
      - name: Build
        run: cmake --build build --config Release
      - name: Test
        run: ctest --test-dir build --build-config Release

  linux-build:
    runs-on: ubuntu-latest
    # ...

  macos-build:
    runs-on: macos-latest
    # ...
```

## Summary

Compiling on Windows requires special attention to compiler optimization behavior and the DLL mechanism. Ensure compatibility through the following measures:

1. **Use framework macros consistently**: `BOAT_API`, `BOAT_CALL`, `BOAT_NOINLINE`
2. **Protect simple wrapper functions**: Prevent elimination by compiler optimization
3. **Verify exported symbols**: Check the DLL export table after the build
4. **Test comprehensively**: Multi-platform, multi-configuration test matrix
5. **Document**: Record platform-specific behavior and solutions

Following these guidelines ensures the Boat framework is stable and reliable on Windows, providing a solid foundation for production deployment.

## Appendix

### A. Useful Command Reference

```bash
# Generate Visual Studio solution
cmake -G "Visual Studio 17 2022" -A x64 ..

# Build specific configuration
cmake --build . --config Release --target boat

# Run tests
ctest -C Release -V

# Check DLL dependencies
dumpbin /dependents boat.dll

# Check DLL exported functions
dumpbin /exports boat.dll > exports.txt
```

### B. Recommended Development Tools

1. **Visual Studio 2022**: Integrated development environment
2. **CMake GUI**: Graphical configuration tool
3. **Dependencies** (formerly Dependency Walker): DLL analysis tool
4. **Process Monitor**: System monitoring tool
5. **DebugView**: System debug output viewer

### C. References

1. [Microsoft C/C++ documentation](https://docs.microsoft.com/cpp/)
2. [CMake documentation](https://cmake.org/documentation/)
3. [Windows DLL best practices](https://docs.microsoft.com/windows/win32/dlls/dynamic-link-library-best-practices)
4. [MSVC compiler options](https://docs.microsoft.com/cpp/build/reference/compiler-options-listed-alphabetically)

---

*Last updated: 2026-02-24*
*Document version: 1.0*
