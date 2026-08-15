# Examples for Boat framework

add_subdirectory(mnist)
add_subdirectory(cifar10)
add_subdirectory(regression)
add_subdirectory(serialization)
add_subdirectory(transformer)
add_subdirectory(translator)

# Data-backed examples (mnist/cifar10 datasets, a trained translator model).
# They exit with the SKIP_RETURN_CODE (77) when the data isn't present, so
# ctest marks them SKIPPED instead of failed in environments without it.
if(TARGET mnist)
    add_test(NAME mnist COMMAND mnist)
    set_tests_properties(mnist PROPERTIES ENVIRONMENT "MNIST_QUICK_TEST=1"
                                           SKIP_RETURN_CODE 77)
endif()
if(TARGET cifar10)
    add_test(NAME cifar10 COMMAND cifar10)
    set_tests_properties(cifar10 PROPERTIES ENVIRONMENT "CIFAR10_QUICK_TEST=1"
                                            SKIP_RETURN_CODE 77)
endif()
if(TARGET translator)
    add_test(NAME translator COMMAND translator ${CMAKE_SOURCE_DIR}/examples/translator/model)
    set_tests_properties(translator PROPERTIES SKIP_RETURN_CODE 77)
endif()

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