# Regression/time series example using Boat framework

cmake_minimum_required(VERSION 3.10)
project(regression_example C)

if(TARGET boat)
    add_executable(regression regression.c)
    target_link_libraries(regression PRIVATE boat)
    target_compile_definitions(regression PRIVATE BOAT_STATIC_BUILD)
    set_target_properties(regression PROPERTIES C_STANDARD 11)
    return()
endif()

# Standalone build
set(BOAT_ROOT ${CMAKE_SOURCE_DIR})
file(TO_CMAKE_PATH ${BOAT_ROOT} BOAT_ROOT)
set(BOAT_INCLUDE_DIR ${BOAT_ROOT}/../../include)

set(BOAT_LIB_DIR_AUTO "")
foreach(lib_path
    ${BOAT_ROOT}/build/Release ${BOAT_ROOT}/build/Debug
    ${BOAT_ROOT}/build_shared/Release ${BOOT_ROOT}/build_shared/Debug
    ${BOAT_ROOT}/build_test/Release ${BOOT_ROOT}/build_test/Debug
    ${BOAT_ROOT}/build/lib ${BOOT_ROOT}/build
)
    file(TO_CMAKE_PATH ${lib_path} p)
    if(EXISTS ${p}/boat.lib OR EXISTS ${p}/libboat.so OR EXISTS ${p}/boat.dll)
        set(BOAT_LIB_DIR_AUTO ${p})
        break()
    endif()
endforeach()

if(BOAT_LIB_DIR_AUTO)
    set(BOAT_LIB_DIR ${BOAT_LIB_DIR_AUTO})
else()
    message(WARNING "Boat library not found. Set BOAT_LIB_DIR or build Boat first.")
endif()

add_executable(regression regression.c)

if(BOAT_INCLUDE_DIR)
    get_filename_component(BOAT_INCLUDE_DIR_ABS "${BOAT_INCLUDE_DIR}" ABSOLUTE)
    target_include_directories(regression PRIVATE ${BOAT_INCLUDE_DIR_ABS})
endif()

if(BOAT_LIB_DIR)
    target_link_directories(regression PRIVATE ${BOAT_LIB_DIR})
    target_link_libraries(regression boat)
    target_compile_definitions(regression PUBLIC BOAT_STATIC_BUILD)
    if(UNIX)
        target_link_libraries(regression m)
    endif()
endif()

set_target_properties(regression PROPERTIES C_STANDARD 11)
