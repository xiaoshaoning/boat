# Qwen3-ASR CUDA inference example
# Requires boat library built with -DBOAT_WITH_CUDA=ON

cmake_minimum_required(VERSION 3.10)
project(qwen3asr_cuda)

if(TARGET boat AND BOAT_WITH_CUDA)
    # Common sources (json is a dependency of safetensors)
    set(QWEN3ASR_COMMON_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/json.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../common/safetensors.c
    )
    set(QWEN3ASR_COMMON_INCLUDES
        ${CMAKE_CURRENT_SOURCE_DIR}/../common
    )

    # ---- Qwen3-ASR CUDA executable ----
    add_executable(qwen3asr_cuda
        main.cu
        model.cu
        kernels.cu
        ../qwen3-asr/weights.c
        ../qwen3-asr/tokenizer.c
        ${QWEN3ASR_COMMON_SOURCES}
    )
    target_include_directories(qwen3asr_cuda PRIVATE
        ${QWEN3ASR_COMMON_INCLUDES}
        ${CMAKE_CURRENT_SOURCE_DIR}/../qwen3-asr
    )
    target_link_libraries(qwen3asr_cuda PRIVATE boat)
    target_compile_definitions(qwen3asr_cuda PRIVATE
        BOAT_STATIC_BUILD BOAT_WITH_CUDA
        _CRT_NONSTDC_NO_DEPRECATE _CRT_SECURE_NO_WARNINGS)
    if(WIN32)
        # No special libs needed beyond CUDA
    endif()
    if(UNIX)
        target_link_libraries(qwen3asr_cuda m)
    endif()
    set_target_properties(qwen3asr_cuda PROPERTIES C_STANDARD 11)
    set_target_properties(qwen3asr_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    return()
endif()

message(WARNING "Qwen3-ASR CUDA example requires boat library built with BOAT_WITH_CUDA=ON. Skipping.")
