# CMakeLists.txt - Build configuration for Boat Deep Learning Framework
cmake_minimum_required(VERSION 3.10)
project(boat VERSION 0.1.0 LANGUAGES C)

# Git version information
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_DESCRIBE
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
else()
    set(GIT_HASH "unknown")
    set(GIT_DESCRIBE "unknown")
endif()

# Set version variables
set(BOAT_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(BOAT_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(BOAT_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(BOAT_VERSION_STRING "${PROJECT_VERSION}")
set(BOAT_GIT_HASH "${GIT_HASH}")
set(BOAT_GIT_DESCRIBE "${GIT_DESCRIBE}")

# Configure version header
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/boat/version.h.in
    ${CMAKE_CURRENT_SOURCE_DIR}/include/boat/version.h
    @ONLY
)

# Options
option(BOAT_BUILD_SHARED "Build shared library (DLL on Windows)" OFF)
option(BOAT_WITH_CUDA "Enable CUDA support" OFF)
option(BOAT_WITH_TESTS "Build tests" OFF)
option(BOAT_WITH_EXAMPLES "Build examples" OFF)
option(BOAT_WITH_ONNX "Enable ONNX support" OFF)
option(BOAT_WITH_PYTORCH "Enable PyTorch support" OFF)
option(BOAT_WITH_TENSORFLOW "Enable TensorFlow support" OFF)
option(BOAT_WITH_HUGGINGFACE "Enable Hugging Face Transformers support" OFF)
option(BOAT_DEBUG "Enable debug output (0=disabled, 1=enabled)" 0)

# PyTorch path configuration
option(BOAT_PYTORCH_PATH "Path to LibTorch installation" "")
# cJSON path configuration
option(BOAT_CJSON_PATH "Path to cJSON installation" "")

# Set C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# AddressSanitizer for Debug builds (optional)
option(BOAT_WITH_ASAN "Enable AddressSanitizer for memory error detection" OFF)
if(BOAT_WITH_ASAN)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        message(STATUS "Enabling AddressSanitizer")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address -fno-omit-frame-pointer")
        set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address")
    else()
        message(WARNING "AddressSanitizer not supported for compiler ${CMAKE_C_COMPILER_ID}")
    endif()
endif()

# Platform-specific optimization control for DLL compatibility
option(BOAT_ENABLE_DLL_COMPAT_OPTIMIZATION "Enable platform-specific optimization control for DLL compatibility" ON)
if(BOAT_ENABLE_DLL_COMPAT_OPTIMIZATION)
    if(MSVC)
        # Disable function-level linking to prevent elimination of simple wrapper functions
        # /Gy- : disable function-level linking
        # /OPT:NOREF : keep functions that are never referenced
        # /OPT:NOICF : prevent identical COMDAT folding
        message(STATUS "Enabling MSVC DLL compatibility optimization controls")

        # Compiler flags for Release builds
        string(APPEND CMAKE_C_FLAGS_RELEASE " /Gy-")

        # Linker flags for Release builds
        string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE " /OPT:NOREF")
        string(APPEND CMAKE_SHARED_LINKER_FLAGS_RELEASE " /OPT:NOREF")

        # Optional: disable incremental linking in Release for more deterministic builds
        # string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE " /INCREMENTAL:NO")
        # string(APPEND CMAKE_SHARED_LINKER_FLAGS_RELEASE " /INCREMENTAL:NO")
    elseif(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        # GCC/Clang flags to prevent aggressive inlining
        message(STATUS "Enabling GCC/Clang DLL compatibility optimization controls")
        string(APPEND CMAKE_C_FLAGS_RELEASE " -fno-inline-functions -fno-inline-small-functions")
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Library source files
set(BOAT_SOURCES
    src/core/tensor.c
    src/core/memory.c
    src/core/error.c
    src/core/version.c
    src/core/packed.c
    src/ops/arithmetic.c
    src/ops/linear.c
    src/ops/activation.c
    src/autodiff.c
    src/graph/node.c
    src/graph/edge.c
    src/graph/graph.c
    src/graph/executor.c
    src/layers/dense.c
    src/layers/conv.c
    src/layers/pool.c
    src/layers/norm.c
    src/layers/attention.c
    src/layers/relu.c
    src/layers/batchnorm.c
    src/layers/softmax.c
    src/layers/flatten.c
    src/layers/lstm.c
    src/layers/gru.c
    src/model/model.c
    src/model/sequential.c
    src/model/graph_model.c
    src/optimizers/adam.c
    src/optimizers/rmsprop.c
    src/optimizers/adagrad.c
    src/optimizers/optimizer_common.c
    src/schedulers/scheduler_common.c
    src/schedulers/step_lr.c
    src/schedulers/cosine_annealing.c
    src/schedulers/lambda_lr.c
    src/loss/loss_common.c
    src/loss/mse.c
    src/loss/cross_entropy.c
    src/loss/huber.c
)

# Optional format loaders
if(BOAT_WITH_ONNX)
    list(APPEND BOAT_SOURCES src/format/onnx.c)
    # Find ONNX runtime
    find_package(ONNXRuntime REQUIRED)
endif()

if(BOAT_WITH_PYTORCH)
    # Change source file to C++
    list(APPEND BOAT_SOURCES src/format/pytorch.cpp)

    # Enable C++ compilation for this file
    set_source_files_properties(src/format/pytorch.cpp PROPERTIES LANGUAGE CXX)

    # Add C++ standard requirements
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # PyTorch/LibTorch configuration
    # Option 1: Use BOAT_PYTORCH_PATH if provided
    if(BOAT_PYTORCH_PATH)
        message(STATUS "Using PyTorch from: ${BOAT_PYTORCH_PATH}")

        # Check if this is a standard LibTorch installation with TorchConfig.cmake
        if(EXISTS "${BOAT_PYTORCH_PATH}/share/cmake/Torch/TorchConfig.cmake")
            # Standard LibTorch installation
            set(Torch_DIR "${BOAT_PYTORCH_PATH}/share/cmake/Torch")
            find_package(Torch REQUIRED)
            message(STATUS "Found standard LibTorch installation")
            message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
            message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
        else()
            # Manual configuration (e.g., from Python installation)
            message(STATUS "Manual PyTorch configuration from Python installation")

            # Set include directories
            set(TORCH_INCLUDE_DIRS
                "${BOAT_PYTORCH_PATH}/include"
                "${BOAT_PYTORCH_PATH}/include/torch/csrc/api/include"
            )

            # Set library directory
            set(TORCH_LIBRARY_DIR "${BOAT_PYTORCH_PATH}/lib")

            # Find required libraries
            set(TORCH_REQUIRED_LIBS torch torch_cpu c10 fmt cpuinfo sleef)
            set(TORCH_LIBRARIES "")

            foreach(lib ${TORCH_REQUIRED_LIBS})
                find_library(TORCH_${lib}_LIBRARY
                    NAMES ${lib}
                    PATHS "${TORCH_LIBRARY_DIR}"
                    NO_DEFAULT_PATH
                )
                if(TORCH_${lib}_LIBRARY)
                    list(APPEND TORCH_LIBRARIES ${TORCH_${lib}_LIBRARY})
                    message(STATUS "Found Torch library: ${lib} at ${TORCH_${lib}_LIBRARY}")
                else()
                    message(WARNING "Torch library not found: ${lib}")
                endif()
            endforeach()

            # Add include directories
            include_directories(${TORCH_INCLUDE_DIRS})
            link_directories(${TORCH_LIBRARY_DIR})
        endif()
    else()
        # Default to our downloaded LibTorch
        set(LIBTORCH_BASE "${CMAKE_SOURCE_DIR}/external/libtorch-shared-with-deps-latest")
        if(EXISTS "${LIBTORCH_BASE}/libtorch/share/cmake/Torch/TorchConfig.cmake")
            set(Torch_DIR "${LIBTORCH_BASE}/libtorch/share/cmake/Torch")
        else()
            set(Torch_DIR "${LIBTORCH_BASE}/share/cmake/Torch")
        endif()

        if(EXISTS "${Torch_DIR}/TorchConfig.cmake")
            find_package(Torch REQUIRED)
            message(STATUS "Found LibTorch installation at: ${Torch_DIR}")
        else()
            message(FATAL_ERROR "LibTorch not found. Please set BOAT_PYTORCH_PATH to the LibTorch installation directory.")
        endif()
    endif()
endif()

if(BOAT_WITH_TENSORFLOW)
    list(APPEND BOAT_SOURCES src/format/tensorflow.c)
    # Find TensorFlow C API
    find_package(TensorFlow REQUIRED)
endif()

if(BOAT_WITH_HUGGINGFACE)
    list(APPEND BOAT_SOURCES src/format/huggingface.c)

    # cJSON configuration
    if(BOAT_CJSON_PATH)
        message(STATUS "Using cJSON from: ${BOAT_CJSON_PATH}")

        # Add include directory
        include_directories(${BOAT_CJSON_PATH}/include)

        # Find library
        find_library(CJSON_LIBRARY
            NAMES cjson libcjson
            PATHS "${BOAT_CJSON_PATH}/lib"
            NO_DEFAULT_PATH
        )

        if(CJSON_LIBRARY)
            message(STATUS "Found cJSON library: ${CJSON_LIBRARY}")
            set(CJSON_LIBRARIES ${CJSON_LIBRARY})
            add_definitions(-DBOAT_USE_CJSON)
        else()
            message(FATAL_ERROR "cJSON library not found in ${BOAT_CJSON_PATH}/lib")
        endif()
    else()
        # Use system cJSON
        find_package(cJSON REQUIRED)
        add_definitions(-DBOAT_USE_CJSON)
    endif()
endif()

# CUDA support
if(BOAT_WITH_CUDA)
    enable_language(CUDA)

    # CUDA source files
    set(BOAT_CUDA_SOURCES
        cuda/tensor.cu
        cuda/ops/arithmetic.cu
        cuda/ops/linear.cu
        cuda/ops/activation.cu
        cuda/autodiff/grad.cu
        cuda/graph/executor.cu
        cuda/kernels/basic.cu
    )

    list(APPEND BOAT_SOURCES ${BOAT_CUDA_SOURCES})

    # CUDA configuration
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Find CUDA
    find_package(CUDAToolkit REQUIRED)
endif()

# Create library (shared or static based on option)
if(BOAT_BUILD_SHARED)
    add_library(boat SHARED ${BOAT_SOURCES})
    message(STATUS "Building shared library (DLL on Windows)")

    # Define export macro when building the DLL
    target_compile_definitions(boat PRIVATE BOAT_BUILDING_DLL)

    # Export all symbols on Windows
    if(WIN32)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS OFF)
    endif()
else()
    add_library(boat STATIC ${BOAT_SOURCES})
    message(STATUS "Building static library")
    # Define static build macro to control export macros
    target_compile_definitions(boat PUBLIC BOAT_STATIC_BUILD)
endif()

add_library(boat::boat ALIAS boat)

# Add debug definition if enabled
if(BOAT_DEBUG)
    target_compile_definitions(boat PRIVATE BOAT_DEBUG=1)
else()
    target_compile_definitions(boat PRIVATE BOAT_DEBUG=0)
endif()

# Set library properties
set_target_properties(boat PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
    POSITION_INDEPENDENT_CODE ON
)

# Disable optimizations for debugging autodiff issues
# Note: This was previously enabled with /Od to debug autodiff issues.
# Now that DLL boundary issues are resolved with BOAT_NOINLINE and /Gy-, /OPT:NOREF,
# we can enable optimizations for better performance.
# Uncomment the following lines if you need to disable optimizations for debugging:
#if(MSVC)
#    target_compile_options(boat PRIVATE "$<$<CONFIG:Release>:/Od>")
#endif()

# Set C++ standard when PyTorch is enabled
if(BOAT_WITH_PYTORCH)
    set_target_properties(boat PROPERTIES CXX_STANDARD 17)
    if(MSVC)
        target_compile_options(boat PRIVATE "/std:c++17" "/EHsc")
        # Additional linker flags for Windows
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:libcmt.lib")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:libcmt.lib")
    endif()
endif()

# Include directories for the library
target_include_directories(boat PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link libraries
if(BOAT_WITH_CUDA)
    target_link_libraries(boat PUBLIC CUDA::cudart CUDA::cublas)
endif()

if(BOAT_WITH_ONNX)
    target_link_libraries(boat PUBLIC ONNXRuntime::onnxruntime)
endif()

if(BOAT_WITH_TENSORFLOW)
    target_link_libraries(boat PUBLIC TensorFlow::TensorFlow)
endif()

if(BOAT_WITH_PYTORCH)
    if(DEFINED TORCH_LIBRARIES AND TORCH_LIBRARIES)
        # Use TORCH_LIBRARIES variable set by configuration
        target_link_libraries(boat PUBLIC ${TORCH_LIBRARIES})
        message(STATUS "Linking with Torch libraries: ${TORCH_LIBRARIES}")
    else()
        # Fallback: manually specify required libraries
        message(STATUS "Using fallback Torch library list")
        target_link_libraries(boat PUBLIC
            torch
            torch_cpu
            c10
            fmt
            cpuinfo
            sleef
            # Additional libraries that might be needed
        )
    endif()
endif()

if(BOAT_WITH_HUGGINGFACE)
    if(DEFINED CJSON_LIBRARIES AND CJSON_LIBRARIES)
        # Use CJSON_LIBRARIES variable set by custom path configuration
        target_link_libraries(boat PUBLIC ${CJSON_LIBRARIES})
        message(STATUS "Linking with cJSON library: ${CJSON_LIBRARIES}")
    else()
        # Use standard cJSON target
        target_link_libraries(boat PUBLIC cJSON::cJSON)
        message(STATUS "Linking with standard cJSON target")
    endif()
endif()

# Install targets
install(TARGETS boat
    EXPORT boat-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/boat
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

install(EXPORT boat-targets
    FILE boat-config.cmake
    NAMESPACE boat::
    DESTINATION lib/cmake/boat
)

# Examples
if(BOAT_WITH_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BOAT_WITH_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()