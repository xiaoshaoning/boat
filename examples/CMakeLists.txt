# Examples for Boat framework

add_subdirectory(mnist)
add_subdirectory(cifar10)
add_subdirectory(regression)
add_subdirectory(serialization)
add_subdirectory(transformer)
add_subdirectory(translator)
add_subdirectory(ocr)

# CUDA-accelerated OCR example (requires BOAT_WITH_CUDA=ON)
add_subdirectory(ocr_cuda)

# NanoChat CUDA inference example (requires BOAT_WITH_CUDA=ON)
add_subdirectory(nanochat)

# Qwen3-ASR inference example (CPU only)
add_subdirectory(qwen3-asr)

# InsightFace recognition test
add_executable(test_insightface test_insightface.c)
target_link_libraries(test_insightface PRIVATE boat)
target_compile_definitions(test_insightface PRIVATE BOAT_STATIC_BUILD _CRT_NONSTDC_NO_DEPRECATE)
set_target_properties(test_insightface PROPERTIES C_STANDARD 11)