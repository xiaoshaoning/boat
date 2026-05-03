# Translator example using Boat framework (MarianMT English-French)

cmake_minimum_required(VERSION 3.10)
project(translator_example C)

# When built as a subdirectory of the main Boat project
if(TARGET boat)
    add_executable(translator
        translator.c
        spm.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/json.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/safetensors.c
    )
    target_include_directories(translator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)
    target_link_libraries(translator PRIVATE boat)
    target_compile_definitions(translator PRIVATE BOAT_STATIC_BUILD _CRT_NONSTDC_NO_DEPRECATE)
    set_target_properties(translator PROPERTIES C_STANDARD 11)
    return()
endif()

# Standalone build
set(BOOT_ROOT ${CMAKE_SOURCE_DIR})
file(TO_CMAKE_PATH ${BOOT_ROOT} BOOT_ROOT)
set(BOAT_INCLUDE_DIR ${BOOT_ROOT}/../../include)
option(BOAT_LIB_DIR "Path to Boat library directory" "")

set(BOAT_LIB_DIR_AUTO "")
if(BOAT_LIB_DIR)
    file(TO_CMAKE_PATH ${BOAT_LIB_DIR} BOAT_LIB_DIR_AUTO)
else()
    foreach(lib_path
        ${BOOT_ROOT}/build/Release ${BOOT_ROOT}/build/Debug
        ${BOOT_ROOT}/build_shared/Release ${BOOT_ROOT}/build_shared/Debug
        ${BOOT_ROOT}/build_test/Release ${BOOT_ROOT}/build_test/Debug
        ${BOOT_ROOT}/build/lib ${BOOT_ROOT}/build
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

add_executable(translator
    translator.c
    spm.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/json.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/safetensors.c
)

if(BOAT_INCLUDE_DIR)
    get_filename_component(BOAT_INCLUDE_DIR_ABS "${BOAT_INCLUDE_DIR}" ABSOLUTE)
    target_include_directories(translator PRIVATE ${BOAT_INCLUDE_DIR_ABS})
endif()
target_include_directories(translator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

if(BOAT_LIB_DIR)
    target_link_directories(translator PRIVATE ${BOAT_LIB_DIR})
    target_link_libraries(translator boat)
    target_compile_definitions(translator PUBLIC BOAT_STATIC_BUILD _CRT_NONSTDC_NO_DEPRECATE)
    if(UNIX)
        target_link_libraries(translator m)
    endif()
endif()

set_target_properties(translator PROPERTIES C_STANDARD 11)
