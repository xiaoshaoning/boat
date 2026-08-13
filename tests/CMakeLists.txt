# Tests CMake configuration

# Include directories for all targets
include_directories(${CMAKE_SOURCE_DIR}/include)
# Register every test executable with CTest so `ctest` (and CI) actually
# runs the suite instead of silently passing.
macro(boat_add_test_executable name)
    add_executable(${name} ${ARGN})
    add_test(NAME ${name} COMMAND ${name})
endmacro()


# Add test executable
boat_add_test_executable(test_phase1 unit/test_phase1.c)

# Link with boat library
target_link_libraries(test_phase1 boat)

# Copy test model files to build directory (only if PyTorch support is enabled)
if(BOAT_WITH_PYTORCH)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_simple_model.pt
                   ${CMAKE_CURRENT_BINARY_DIR}/test_simple_model.pt COPYONLY)

    # Simple PyTorch loader test
    boat_add_test_executable(test_pytorch_simple test_pytorch_simple.c)
    target_link_libraries(test_pytorch_simple boat)
endif()

# Safetensors parsing test (only if HuggingFace support is enabled)
if(BOAT_WITH_HUGGINGFACE)
    boat_add_test_executable(test_safetensors test_safetensors.c)
    target_link_libraries(test_safetensors boat)
endif()

# Layer normalization integration test
boat_add_test_executable(test_layernorm_simple test_layernorm_simple.c)
target_link_libraries(test_layernorm_simple boat)

# Model and computational graph integration test
boat_add_test_executable(test_model_graph_integration test_model_graph_integration.c)
target_link_libraries(test_model_graph_integration boat)

# Dense layer forward pass test
boat_add_test_executable(test_dense_forward test_dense_forward.c)
target_link_libraries(test_dense_forward boat)

# Convolutional layer forward pass test
boat_add_test_executable(test_conv_forward test_conv_forward.c)
target_link_libraries(test_conv_forward boat)

# Attention layer gradient check test
boat_add_test_executable(test_attention_gradient_check unit/test_attention_gradient_check.c)
target_link_libraries(test_attention_gradient_check boat)

# Attention layer end-to-end training test
boat_add_test_executable(test_attention_training test_attention_training.c)
target_link_libraries(test_attention_training boat)

# Convolutional layer gradient check test
boat_add_test_executable(test_conv_gradient_check unit/test_conv_gradient_check.c)
target_link_libraries(test_conv_gradient_check boat)

# Adam optimizer test
boat_add_test_executable(test_optimizer_adam unit/test_optimizer_adam.c)
target_link_libraries(test_optimizer_adam boat)

# RMSprop optimizer test
boat_add_test_executable(test_optimizer_rmsprop unit/test_optimizer_rmsprop.c)
target_link_libraries(test_optimizer_rmsprop boat)

# Adagrad optimizer test
boat_add_test_executable(test_optimizer_adagrad unit/test_optimizer_adagrad.c)
target_link_libraries(test_optimizer_adagrad boat)

# SGD optimizer test
boat_add_test_executable(test_optimizer_sgd unit/test_optimizer_sgd.c)
target_link_libraries(test_optimizer_sgd boat)
boat_add_test_executable(test_optimizer_l2 unit/test_optimizer_l2.c)
target_link_libraries(test_optimizer_l2 boat)

# Scheduler tests
boat_add_test_executable(test_schedulers unit/test_schedulers.c)
target_link_libraries(test_schedulers boat)

# Optimizer benchmark
boat_add_test_executable(test_optimizer_benchmark ../benchmarks/linear_regression.c)
target_link_libraries(test_optimizer_benchmark boat)

# Attention layer performance benchmark (uses windows.h timing; Windows-only)
if(WIN32)
    boat_add_test_executable(test_attention_performance ../benchmarks/attention_performance.c)
    target_link_libraries(test_attention_performance boat)
endif()

# Gradient checking tests
boat_add_test_executable(test_gradient_check unit/test_gradient_check.c)
target_link_libraries(test_gradient_check boat)

# Transpose operation tests
boat_add_test_executable(test_transpose unit/test_transpose.c)
target_link_libraries(test_transpose boat)

# Tensor operation tests (creation, reshape, slice, concatenate, stack, views)
boat_add_test_executable(test_tensor unit/test_tensor.c)
target_link_libraries(test_tensor boat)

# Autodiff engine tests (variables, operations, backward pass)
boat_add_test_executable(test_autodiff unit/test_autodiff.c)
target_link_libraries(test_autodiff boat)

# Autodiff graph visualization (graph_to_dot / print_graph / retain_grad)
boat_add_test_executable(test_autodiff_graph unit/test_autodiff_graph.c)
target_link_libraries(test_autodiff_graph boat)

# Reduction ops with axis support (sum/mean/max/min + backward)
boat_add_test_executable(test_reduce_autodiff unit/test_reduce_autodiff.c)
target_link_libraries(test_reduce_autodiff boat)

# Softmax cross-entropy loss
boat_add_test_executable(test_softmax_ce unit/test_softmax_ce.c)
target_link_libraries(test_softmax_ce boat)

# Matmul batch broadcasting + softmax/log_softmax axis
boat_add_test_executable(test_matmul_softmax unit/test_matmul_softmax.c)
target_link_libraries(test_matmul_softmax boat)

# Dense + max-pool backward gradient checks
boat_add_test_executable(test_layer_gradcheck unit/test_layer_gradcheck.c)
target_link_libraries(test_layer_gradcheck boat)

# PReLU + BatchNorm2d backward gradient checks
boat_add_test_executable(test_prelu_batchnorm unit/test_prelu_batchnorm.c)
target_link_libraries(test_prelu_batchnorm boat)

# Swin Transformer encoder forward
boat_add_test_executable(test_swin unit/test_swin.c)
target_link_libraries(test_swin boat)

# Matrix inverse
boat_add_test_executable(test_inverse unit/test_inverse.c)
target_link_libraries(test_inverse boat)

# Graph robustness tests (merge, to_dot, node-removal adjacency, sequential lifecycle)
boat_add_test_executable(test_graph_phase4 unit/test_graph_phase4.c)
target_link_libraries(test_graph_phase4 boat)

# Graph feature tests (device-aware creation, copy/subgraph, safe edge/node ops)
boat_add_test_executable(test_graph_phase2 unit/test_graph_phase2.c)
target_link_libraries(test_graph_phase2 boat)

boat_add_test_executable(test_graph_phase3 unit/test_graph_phase3.c)
target_link_libraries(test_graph_phase3 boat)

# Layer link and memory management tests
boat_add_test_executable(test_layer_link unit/test_layer_link.c)
target_link_libraries(test_layer_link boat)

boat_add_test_executable(test_memory unit/test_memory.c)
target_link_libraries(test_memory boat)

boat_add_test_executable(test_simple unit/test_simple.c)
target_link_libraries(test_simple boat)

boat_add_test_executable(test_basic test_basic.c)
target_link_libraries(test_basic boat)

# Serialization integration test (train -> save -> load -> inference round-trip)
boat_add_test_executable(test_serialization_integration test_serialization_integration.c)
target_link_libraries(test_serialization_integration boat)

# Loss function integration tests (forward, backward, training, edge cases)
boat_add_test_executable(test_loss_integration test_loss_integration.c)
target_link_libraries(test_loss_integration boat)

# Quantization tests
boat_add_test_executable(test_quantize test_quantize.c)
target_link_libraries(test_quantize boat)

# Pruning tests
boat_add_test_executable(test_prune test_prune.c)
target_link_libraries(test_prune boat)

# ONNX loader tests (only when ONNX support is enabled: they link against
# src/format/onnx.c / onnx_pb.c, which are not compiled otherwise)
if(BOAT_WITH_ONNX)
    boat_add_test_executable(test_onnx test_onnx.c)
    target_include_directories(test_onnx PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(test_onnx boat)
endif()

# Data pipeline tests (Dataset, DataLoader, Transforms)
boat_add_test_executable(test_data_pipeline test_data_pipeline.c)
target_link_libraries(test_data_pipeline boat)

# GGUF format loader tests
if(BOAT_WITH_GGUF)
    boat_add_test_executable(test_gguf test_gguf.c)
    target_link_libraries(test_gguf boat)
endif()

# CUDA backend tests
if(BOAT_WITH_CUDA)
    boat_add_test_executable(test_cuda_mini test_cuda_mini.c)
    target_link_libraries(test_cuda_mini boat)
    boat_add_test_executable(test_cuda_basic test_cuda_basic.c)
    target_link_libraries(test_cuda_basic boat)
    boat_add_test_executable(test_cuda_layers test_cuda_layers.c)
    target_link_libraries(test_cuda_layers boat)
endif()

# Activation function tests (sigmoid / tanh / SELU)
boat_add_test_executable(test_activation unit/test_activation.c)
target_link_libraries(test_activation boat)

# Normalization backward tests (LayerNorm / RMSNorm)
boat_add_test_executable(test_norm_backward unit/test_norm_backward.c)
target_link_libraries(test_norm_backward boat)

# Recurrent layer tests (LSTM / GRU forward, backward and update)
boat_add_test_executable(test_lstm_gru unit/test_lstm_gru.c)
target_link_libraries(test_lstm_gru boat)

# ONNX Runtime backend tests (requires ONNX for protobuf builder used in test model construction)
if(BOAT_WITH_ONNXRUNTIME AND BOAT_WITH_ONNX)
    boat_add_test_executable(test_onnxruntime test_onnxruntime.c)
    target_include_directories(test_onnxruntime PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_compile_definitions(test_onnxruntime PRIVATE BOAT_STATIC_BUILD _CRT_NONSTDC_NO_DEPRECATE)
    target_link_libraries(test_onnxruntime boat)
endif()

