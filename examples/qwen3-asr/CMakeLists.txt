# Qwen3-ASR inference example
# Requires boat library (CPU only)

cmake_minimum_required(VERSION 3.10)
project(qwen3_asr)

# Common sources (json is a dependency of safetensors)
set(QWEN3ASR_COMMON_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/json.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/safetensors.c
)
set(QWEN3ASR_COMMON_INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

# Framework sources needed by qwen3asr
set(BOAT_SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(BOAT_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

add_executable(qwen3asr
    main.c
    encoder.c
    decoder.c
    weights.c
    tokenizer.c
    ${BOAT_SRC_DIR}/sampling.c
    ${BOAT_SRC_DIR}/layers/embedding.c
    ${QWEN3ASR_COMMON_SOURCES}
)
target_include_directories(qwen3asr PRIVATE
    ${QWEN3ASR_COMMON_INCLUDES}
    ${BOAT_INCLUDE_DIR}
)
target_link_libraries(qwen3asr PRIVATE boat)
target_compile_definitions(qwen3asr PRIVATE
    BOAT_STATIC_BUILD
    _CRT_NONSTDC_NO_DEPRECATE _CRT_SECURE_NO_WARNINGS)
if(UNIX)
    target_link_libraries(qwen3asr m)
endif()
set_target_properties(qwen3asr PROPERTIES C_STANDARD 11)
