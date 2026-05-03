# NanoChat CUDA inference example
# Requires boat library built with -DBOAT_WITH_CUDA=ON

cmake_minimum_required(VERSION 3.10)
project(nanochat_example)

if(TARGET boat AND BOAT_WITH_CUDA)
    # Common sources
    set(NANOCHAT_COMMON_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/json.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/safetensors.c
    )
    set(NANOCHAT_COMMON_INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
    )

    # ---- NanoChat CLI executable ----
    add_executable(nanochat_cli
        nanochat_cli.c
        tokenizer.c
        weights.c
        sampling.c
        engine.c
        model.cu
        nanochat_kernels.cu
        ${NANOCHAT_COMMON_SOURCES}
    )
    target_include_directories(nanochat_cli PRIVATE ${NANOCHAT_COMMON_INCLUDES})
    target_link_libraries(nanochat_cli PRIVATE boat)
    target_compile_definitions(nanochat_cli PRIVATE
        BOAT_STATIC_BUILD BOAT_WITH_CUDA
        _CRT_NONSTDC_NO_DEPRECATE _CRT_SECURE_NO_WARNINGS)
    if(WIN32)
        # No special libs needed beyond CUDA
    endif()
    if(UNIX)
        target_link_libraries(nanochat_cli m)
    endif()
    set_target_properties(nanochat_cli PROPERTIES C_STANDARD 11)
    set_target_properties(nanochat_cli PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    return()
endif()

message(WARNING "NanoChat CUDA example requires boat library built with BOAT_WITH_CUDA=ON. Skipping.")
