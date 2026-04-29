# MNIST example using Boat framework

cmake_minimum_required(VERSION 3.10)
project(mnist_example C)

# When built as a subdirectory of the main Boat project, the target already exists
if(TARGET boat)
    add_executable(mnist mnist.c)
    add_executable(mnist_autodiff mnist_autodiff.c)
    target_link_libraries(mnist PRIVATE boat)
    target_link_libraries(mnist_autodiff PRIVATE boat)
    target_compile_definitions(mnist PRIVATE BOAT_STATIC_BUILD)
    target_compile_definitions(mnist_autodiff PRIVATE BOAT_STATIC_BUILD)
    set_target_properties(mnist PROPERTIES C_STANDARD 11)
    set_target_properties(mnist_autodiff PROPERTIES C_STANDARD 11)
    return()
endif()

# Standalone build — find pre-built Boat library
set(BOAT_ROOT ${CMAKE_SOURCE_DIR})
file(TO_CMAKE_PATH ${BOAT_ROOT} BOAT_ROOT)
set(BOAT_INCLUDE_DIR ${BOAT_ROOT}/../../include)
option(BOAT_LIB_DIR "Path to Boat library directory" "")

set(BOAT_LIB_DIR_AUTO "")
if(BOAT_LIB_DIR)
    file(TO_CMAKE_PATH ${BOAT_LIB_DIR} BOAT_LIB_DIR_AUTO)
else()
    foreach(lib_path
        ${BOAT_ROOT}/build/Release ${BOAT_ROOT}/build/Debug
        ${BOAT_ROOT}/build_shared/Release ${BOAT_ROOT}/build_shared/Debug
        ${BOAT_ROOT}/build_test/Release ${BOAT_ROOT}/build_test/Debug
        ${BOAT_ROOT}/build/lib ${BOAT_ROOT}/build
    )
        file(TO_CMAKE_PATH ${lib_path} p)
        if(EXISTS ${p}/boat.lib OR EXISTS ${p}/libboat.so OR EXISTS ${p}/boat.dll)
            set(BOAT_LIB_DIR_AUTO ${p})
            break()
        endif()
    endforeach()
endif()

if(BOAT_LIB_DIR_AUTO)
    set(BOAT_LIB_DIR ${BOAT_LIB_DIR_AUTO})
else()
    message(WARNING "Boat library not found. Set BOAT_LIB_DIR or build Boat first.")
endif()

add_executable(mnist mnist.c)
add_executable(mnist_autodiff mnist_autodiff.c)

if(BOAT_INCLUDE_DIR)
    get_filename_component(BOAT_INCLUDE_DIR_ABS "${BOAT_INCLUDE_DIR}" ABSOLUTE)
    target_include_directories(mnist PRIVATE ${BOAT_INCLUDE_DIR_ABS})
    target_include_directories(mnist_autodiff PRIVATE ${BOAT_INCLUDE_DIR_ABS})
endif()

if(BOAT_LIB_DIR)
    target_link_directories(mnist PRIVATE ${BOAT_LIB_DIR})
    target_link_libraries(mnist boat)
    target_compile_definitions(mnist PUBLIC BOAT_STATIC_BUILD)
    target_link_directories(mnist_autodiff PRIVATE ${BOAT_LIB_DIR})
    target_link_libraries(mnist_autodiff boat)
    target_compile_definitions(mnist_autodiff PUBLIC BOAT_STATIC_BUILD)
    if(UNIX)
        target_link_libraries(mnist m)
        target_link_libraries(mnist_autodiff m)
    endif()
endif()

set_target_properties(mnist PROPERTIES C_STANDARD 11)
set_target_properties(mnist_autodiff PROPERTIES C_STANDARD 11)
