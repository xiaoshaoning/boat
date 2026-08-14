# Examples for Boat framework

add_subdirectory(mnist)
add_subdirectory(cifar10)
add_subdirectory(regression)
add_subdirectory(serialization)
add_subdirectory(transformer)
add_subdirectory(translator)

# Needle 2 (Simple Attention Network) inference from the .cact blob (CPU).
add_subdirectory(needle)

# Learning rate scheduler usage example (CPU, self-contained)
add_executable(scheduler_usage scheduler_usage.c)
target_link_libraries(scheduler_usage PRIVATE boat)
target_compile_definitions(scheduler_usage PRIVATE BOAT_STATIC_BUILD)
set_target_properties(scheduler_usage PROPERTIES C_STANDARD 11)
add_test(NAME scheduler_usage COMMAND scheduler_usage)

# NanoChat CUDA inference example (requires BOAT_WITH_CUDA=ON)
add_subdirectory(nanochat)

# Nougat-LaTeX OCR inference example (requires BOAT_WITH_CUDA=ON)
add_subdirectory(latex)

# InsightFace recognition test (requires ONNX Runtime backend)
if(BOAT_WITH_ONNXRUNTIME)
    add_executable(test_insightface test_insightface.c)
    target_link_libraries(test_insightface PRIVATE boat)
    target_compile_definitions(test_insightface PRIVATE BOAT_STATIC_BUILD _CRT_NONSTDC_NO_DEPRECATE)
    set_target_properties(test_insightface PROPERTIES C_STANDARD 11)
endif()