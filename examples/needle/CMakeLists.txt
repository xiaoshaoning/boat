# Needle 2 inference example for the Boat framework.
#
# A from-scratch C reimplementation of the Simple Attention Network (SAN)
# decode path plus a parser for the self-contained .cact deployment blob
# produced by the cactus-needle repo. The model weights are NOT shipped in
# this repository: point the binary at a local .cact (e.g. the HF mirror
# D:\hugginface\needle2\needle2.cact) at runtime.

cmake_minimum_required(VERSION 3.10)
project(needle_example C)

# When built as a subdirectory of the main Boat project, the target exists.
if(TARGET boat)
    add_executable(needle2 main.c cact.c tokenizer.c san.c)
    target_link_libraries(needle2 PRIVATE boat)
    target_compile_definitions(needle2 PRIVATE BOAT_STATIC_BUILD)
    set_target_properties(needle2 PROPERTIES C_STANDARD 11)
    add_test(NAME needle2_selftest COMMAND needle2 --selftest)
    return()
endif()

# Standalone build -- find a pre-built Boat library.
set(BOAT_ROOT ${CMAKE_SOURCE_DIR})
set(BOAT_INCLUDE_DIR ${BOAT_ROOT}/include)
set(BOAT_LIB_DIR ${BOAT_ROOT}/build/lib)

add_executable(needle2 main.c cact.c tokenizer.c san.c)
target_include_directories(needle2 PRIVATE ${BOAT_INCLUDE_DIR})
target_link_directories(needle2 PRIVATE ${BOAT_LIB_DIR})
target_link_libraries(needle2 boat)
target_compile_definitions(needle2 PRIVATE BOAT_STATIC_BUILD)
set_target_properties(needle2 PROPERTIES C_STANDARD 11)
