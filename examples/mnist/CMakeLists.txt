# MNIST example using Boat framework

cmake_minimum_required(VERSION 3.10)
project(mnist_example C)

# Find Boat library
# When included as subdirectory, CMAKE_SOURCE_DIR is the root source directory
set(BOAT_ROOT ${CMAKE_SOURCE_DIR})
# Convert to CMake path (normalize slashes)
file(TO_CMAKE_PATH ${BOAT_ROOT} BOAT_ROOT)
set(BOAT_INCLUDE_DIR ${BOAT_ROOT}/../../include)

# Allow user to manually specify Boat library directory
option(BOAT_LIB_DIR "Path to Boat library directory" "")

# Try to find Boat library in various possible locations
set(BOAT_LIB_DIR_AUTO "")

if(BOAT_LIB_DIR)
    # User specified library directory
    file(TO_CMAKE_PATH ${BOAT_LIB_DIR} BOAT_LIB_DIR_AUTO)
    message(STATUS "Using user-specified Boat library directory: ${BOAT_LIB_DIR_AUTO}")
else()
    message(STATUS "Searching for Boat library...")
    foreach(lib_path
        ${BOAT_ROOT}/build/Release
        ${BOAT_ROOT}/build/Debug
        ${BOAT_ROOT}/build_shared/Release
        ${BOAT_ROOT}/build_shared/Debug
        ${BOAT_ROOT}/build_test/Release
        ${BOAT_ROOT}/build_test/Debug
        ${BOAT_ROOT}/build/lib
        ${BOAT_ROOT}/build
    )
        file(TO_CMAKE_PATH ${lib_path} lib_path_norm)
        message(STATUS "Checking: ${lib_path_norm}")
        if(EXISTS ${lib_path_norm}/boat.lib OR EXISTS ${lib_path_norm}/libboat.so OR EXISTS ${lib_path_norm}/boat.dll)
            set(BOAT_LIB_DIR_AUTO ${lib_path_norm})
            message(STATUS "Found Boat library in: ${BOAT_LIB_DIR_AUTO}")
            break()
        endif()
    endforeach()
endif()

if(BOAT_LIB_DIR_AUTO)
    set(BOAT_LIB_DIR ${BOAT_LIB_DIR_AUTO})
else()
    message(WARNING "Could not find Boat library. Please build Boat first or specify BOAT_LIB_DIR.")
endif()

message(STATUS "Boat include directory: ${BOAT_INCLUDE_DIR}")
message(STATUS "Boat library directory: ${BOAT_LIB_DIR}")

# Add executables
add_executable(mnist mnist.c)
add_executable(mnist_autodiff mnist_autodiff.c)

# Include directories
if(BOAT_INCLUDE_DIR)
    # Convert to absolute path
    get_filename_component(BOAT_INCLUDE_DIR_ABS "${BOAT_INCLUDE_DIR}" ABSOLUTE)
    message(STATUS "Boat include absolute path: ${BOAT_INCLUDE_DIR_ABS}")
    target_include_directories(mnist SYSTEM PRIVATE ${BOAT_INCLUDE_DIR_ABS})
    target_include_directories(mnist_autodiff SYSTEM PRIVATE ${BOAT_INCLUDE_DIR_ABS})
else()
    message(WARNING "Boat include directory not set. Skipping include directories.")
endif()

# Link Boat library
if(BOAT_LIB_DIR)
    target_link_directories(mnist PRIVATE ${BOAT_LIB_DIR})
    target_link_libraries(mnist boat)
    # For static library build, define BOAT_STATIC_BUILD to avoid dllimport
    target_compile_definitions(mnist PUBLIC BOAT_STATIC_BUILD)
    target_link_directories(mnist_autodiff PRIVATE ${BOAT_LIB_DIR})
    target_link_libraries(mnist_autodiff boat)
    # For static library build, define BOAT_STATIC_BUILD to avoid dllimport
    target_compile_definitions(mnist_autodiff PUBLIC BOAT_STATIC_BUILD)

    # Link math library on Unix-like systems
    if(UNIX)
        target_link_libraries(mnist m)
        target_link_libraries(mnist_autodiff m)
    endif()
else()
    message(WARNING "Skipping Boat library linking - library not found")
endif()

# Copy DLL to executable directory on Windows
if(FALSE AND WIN32 AND BOAT_LIB_DIR)
    add_custom_command(TARGET mnist POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${BOAT_LIB_DIR}/boat.dll
            $<TARGET_FILE_DIR:mnist>
        COMMENT "Copying boat.dll to executable directory"
    )
    add_custom_command(TARGET mnist_autodiff POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${BOAT_LIB_DIR}/boat.dll
            $<TARGET_FILE_DIR:mnist_autodiff>
        COMMENT "Copying boat.dll to executable directory"
    )
endif()

# Set C standard
set_target_properties(mnist PROPERTIES C_STANDARD 11)
set_target_properties(mnist_autodiff PROPERTIES C_STANDARD 11)