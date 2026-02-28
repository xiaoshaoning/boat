# Tests CMake configuration

# Include directories for all targets
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add test executable
add_executable(test_phase1 unit/test_phase1.c)

# Link with boat library
target_link_libraries(test_phase1 boat)

# Copy test model files to build directory (only if PyTorch support is enabled)
if(BOAT_WITH_PYTORCH)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_simple_model.pt
                   ${CMAKE_CURRENT_BINARY_DIR}/test_simple_model.pt COPYONLY)

    # Simple PyTorch loader test
    add_executable(test_pytorch_simple test_pytorch_simple.c)
    target_link_libraries(test_pytorch_simple boat)
endif()

# Safetensors parsing test (only if HuggingFace support is enabled)
if(BOAT_WITH_HUGGINGFACE)
    add_executable(test_safetensors test_safetensors.c)
    target_link_libraries(test_safetensors boat)
endif()

# Layer normalization integration test
add_executable(test_layernorm_simple test_layernorm_simple.c)
target_link_libraries(test_layernorm_simple boat)

# Model and computational graph integration test
add_executable(test_model_graph_integration test_model_graph_integration.c)
target_link_libraries(test_model_graph_integration boat)

# Dense layer forward pass test
add_executable(test_dense_forward test_dense_forward.c)
target_link_libraries(test_dense_forward boat)

# Convolutional layer forward pass test
add_executable(test_conv_forward test_conv_forward.c)
target_link_libraries(test_conv_forward boat)

# Attention layer gradient check test
add_executable(test_attention_gradient_check unit/test_attention_gradient_check.c)
target_link_libraries(test_attention_gradient_check boat)

# Attention layer end-to-end training test
add_executable(test_attention_training test_attention_training.c)
target_link_libraries(test_attention_training boat)

# Convolutional layer gradient check test
add_executable(test_conv_gradient_check unit/test_conv_gradient_check.c)
target_link_libraries(test_conv_gradient_check boat)

# Adam optimizer test
add_executable(test_optimizer_adam unit/test_optimizer_adam.c)
target_link_libraries(test_optimizer_adam boat)

# RMSprop optimizer test
add_executable(test_optimizer_rmsprop unit/test_optimizer_rmsprop.c)
target_link_libraries(test_optimizer_rmsprop boat)

# Adagrad optimizer test
add_executable(test_optimizer_adagrad unit/test_optimizer_adagrad.c)
target_link_libraries(test_optimizer_adagrad boat)

# Scheduler tests
add_executable(test_schedulers unit/test_schedulers.c)
target_link_libraries(test_schedulers boat)

# Optimizer benchmark
add_executable(test_optimizer_benchmark ../benchmarks/linear_regression.c)
target_link_libraries(test_optimizer_benchmark boat)

# Attention layer performance benchmark
add_executable(test_attention_performance ../benchmarks/attention_performance.c)
target_link_libraries(test_attention_performance boat)

# Gradient checking tests
add_executable(test_gradient_check unit/test_gradient_check.c)
target_link_libraries(test_gradient_check boat)

# Transpose operation tests
add_executable(test_transpose unit/test_transpose.c)
target_link_libraries(test_transpose boat)