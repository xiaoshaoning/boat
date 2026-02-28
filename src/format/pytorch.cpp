// pytorch.cpp - PyTorch model format loader using LibTorch C++ API
// Copyright (c) 2026 Boat Framework Authors
// Distributed under the MIT License

#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <string>
#include <exception>
#include <unordered_map>
#include <iostream>

#include "boat/format/pytorch.h"
#include "boat/model.h"
#include "boat/tensor.h"
#include "boat/memory.h"
#include "boat/layers.h"
#include "boat.h"
#include "boat/export.h"

// Data type mapping helper
static boat_dtype_t torch_dtype_to_boat(torch::Dtype dtype) {
    switch (dtype) {
        case torch::kFloat32: return BOAT_DTYPE_FLOAT32;
        case torch::kFloat64: return BOAT_DTYPE_FLOAT64;
        case torch::kFloat16: return BOAT_DTYPE_FLOAT16;
        case torch::kBFloat16:
            // BFLOAT16 not yet supported in Boat, fall back to FLOAT32
            return BOAT_DTYPE_FLOAT32;
        case torch::kInt32:   return BOAT_DTYPE_INT32;
        case torch::kInt64:   return BOAT_DTYPE_INT64;
        case torch::kInt8:
            // INT8 not yet supported in Boat, fall back to INT32
            return BOAT_DTYPE_INT32;
        case torch::kUInt8:   return BOAT_DTYPE_UINT8;
        case torch::kBool:    return BOAT_DTYPE_BOOL;
        default:              return BOAT_DTYPE_FLOAT32;
    }
}

// Shape conversion helper
static void torch_shape_to_boat(const torch::Tensor& tensor,
                                int64_t** shape_ptr,
                                size_t* ndim_ptr) {
    auto sizes = tensor.sizes();
    *ndim_ptr = sizes.size();
    *shape_ptr = (int64_t*)boat_malloc(sizeof(int64_t) * (*ndim_ptr), BOAT_DEVICE_CPU);

    for (size_t i = 0; i < *ndim_ptr; i++) {
        (*shape_ptr)[i] = sizes[i];
    }
}

// Structure to hold model parameters loaded from PyTorch
typedef struct {
    std::unordered_map<std::string, boat_tensor_t*> parameters;
    std::unordered_map<std::string, boat_tensor_t*> buffers;
} pytorch_model_data_t;

// Free function for model user data
static void free_pytorch_model_data(void* data) {
    pytorch_model_data_t* model_data = static_cast<pytorch_model_data_t*>(data);
    if (!model_data) return;

    for (auto& pair : model_data->parameters) {
        boat_tensor_free(pair.second);
    }
    for (auto& pair : model_data->buffers) {
        boat_tensor_free(pair.second);
    }
    delete model_data;
}

// Convert PyTorch module to Boat model structure
static boat_model_t* convert_pytorch_module_to_boat_model(const torch::jit::Module& module,
                                                          pytorch_model_data_t* model_data) {
    // Create a sequential model
    boat_sequential_model_t* seq_model = boat_sequential_create();
    if (!seq_model) {
        return NULL;
    }

    // Get all named modules
    auto named_modules = module.named_modules();

    // Iterate through modules in order (PyTorch modules are already in forward order)
    for (const auto& named_module : named_modules) {
        const std::string& name = named_module.name;
        const torch::jit::Module& mod = named_module.value;

        // Get the module's Python type name (e.g., "Linear", "ReLU")
        std::string module_type = mod.type()->str();

        // Map PyTorch modules to Boat layers
        if (module_type.find("Linear") != std::string::npos) {
            // Extract weight and bias parameters
            std::string weight_key = name + ".weight";
            std::string bias_key = name + ".bias";

            auto weight_it = model_data->parameters.find(weight_key);
            if (weight_it == model_data->parameters.end()) {
                // Try without prefix (some models have different naming)
                weight_key = "weight";
                weight_it = model_data->parameters.find(weight_key);
            }

            if (weight_it == model_data->parameters.end()) {
                fprintf(stderr, "Warning: Could not find weight parameter for layer %s\n", name.c_str());
                continue;
            }

            boat_tensor_t* weight_tensor = weight_it->second;
            const int64_t* weight_shape = boat_tensor_shape(weight_tensor);
            size_t input_features = weight_shape[0];
            size_t output_features = weight_shape[1];

            // Check if bias exists
            bool use_bias = false;
            boat_tensor_t* bias_tensor = nullptr;
            auto bias_it = model_data->parameters.find(bias_key);
            if (bias_it != model_data->parameters.end()) {
                bias_tensor = bias_it->second;
                use_bias = true;
            }

            // Create dense layer
            boat_dense_layer_t* dense_layer = boat_dense_layer_create(input_features, output_features, use_bias);
            if (!dense_layer) {
                fprintf(stderr, "Warning: Failed to create dense layer for %s\n", name.c_str());
                continue;
            }

            // Set weight tensor (function takes ownership of reference)
            boat_dense_layer_set_weight(dense_layer, weight_tensor);

            // Set bias tensor if present
            if (use_bias && bias_tensor) {
                boat_dense_layer_set_bias(dense_layer, bias_tensor);
            }

            // Convert dense_layer to generic layer_t (layer_t is just a wrapper)
            boat_layer_t* generic_layer = (boat_layer_t*)dense_layer;
            boat_sequential_add(seq_model, generic_layer);

        } else if (module_type.find("ReLU") != std::string::npos) {
            // Create ReLU activation layer
            boat_relu_layer_t* relu_layer = boat_relu_layer_create();
            if (!relu_layer) {
                fprintf(stderr, "Warning: Failed to create ReLU layer for %s\n", name.c_str());
                continue;
            }
            // Convert to generic layer
            boat_layer_t* generic_layer = (boat_layer_t*)relu_layer;
            boat_sequential_add(seq_model, generic_layer);
        } else if (module_type.find("Conv2d") != std::string::npos) {
            // Extract Conv2d parameters
            std::string weight_key = name + ".weight";
            std::string bias_key = name + ".bias";

            auto weight_it = model_data->parameters.find(weight_key);
            if (weight_it == model_data->parameters.end()) {
                // Try without prefix
                weight_key = "weight";
                weight_it = model_data->parameters.find(weight_key);
            }

            if (weight_it == model_data->parameters.end()) {
                fprintf(stderr, "Warning: Could not find weight parameter for Conv2d layer %s\n", name.c_str());
                continue;
            }

            boat_tensor_t* weight_tensor = weight_it->second;
            const int64_t* weight_shape = boat_tensor_shape(weight_tensor);
            // weight shape: [out_channels, in_channels, kernel_height, kernel_width]
            size_t out_channels = weight_shape[0];
            size_t in_channels = weight_shape[1];
            size_t kernel_size = weight_shape[2]; // assuming square kernel

            // Default stride and padding (PyTorch defaults: stride=1, padding=0)
            size_t stride = 1;
            size_t padding = 0;

            // TODO: Extract actual stride and padding from PyTorch module attributes

            // Check if bias exists
            bool use_bias = false;
            boat_tensor_t* bias_tensor = nullptr;
            auto bias_it = model_data->parameters.find(bias_key);
            if (bias_it != model_data->parameters.end()) {
                bias_tensor = bias_it->second;
                use_bias = true;
            }

            // Create conv layer
            boat_conv_layer_t* conv_layer = boat_conv_layer_create(in_channels, out_channels, kernel_size, stride, padding);
            if (!conv_layer) {
                fprintf(stderr, "Warning: Failed to create conv layer for %s\n", name.c_str());
                continue;
            }

            // Set weight tensor
            boat_conv_layer_set_weight(conv_layer, weight_tensor);

            // Set bias tensor if present
            if (use_bias && bias_tensor) {
                boat_conv_layer_set_bias(conv_layer, bias_tensor);
            }

            // Convert to generic layer
            boat_layer_t* generic_layer = (boat_layer_t*)conv_layer;
            boat_sequential_add(seq_model, generic_layer);

        } else if (module_type.find("BatchNorm2d") != std::string::npos) {
            // Extract BatchNorm2d parameters
            std::string weight_key = name + ".weight";
            std::string bias_key = name + ".bias";
            std::string running_mean_key = name + ".running_mean";
            std::string running_var_key = name + ".running_var";

            // Get num_features from weight shape or running_mean shape
            size_t num_features = 0;
            bool affine = false;

            // First try to get weight tensor to determine affine and num_features
            auto weight_it = model_data->parameters.find(weight_key);
            if (weight_it != model_data->parameters.end()) {
                boat_tensor_t* weight_tensor = weight_it->second;
                const int64_t* weight_shape = boat_tensor_shape(weight_tensor);
                num_features = weight_shape[0];
                affine = true;
            } else {
                // Try running_mean to get num_features
                auto running_mean_it = model_data->buffers.find(running_mean_key);
                if (running_mean_it != model_data->buffers.end()) {
                    boat_tensor_t* running_mean_tensor = running_mean_it->second;
                    const int64_t* running_mean_shape = boat_tensor_shape(running_mean_tensor);
                    num_features = running_mean_shape[0];
                } else {
                    fprintf(stderr, "Warning: Could not determine num_features for BatchNorm2d layer %s\n", name.c_str());
                    continue;
                }
            }

            if (num_features == 0) {
                fprintf(stderr, "Warning: num_features is zero for BatchNorm2d layer %s\n", name.c_str());
                continue;
            }

            // Default eps and momentum (PyTorch defaults: eps=1e-5, momentum=0.1)
            float eps = 1e-5f;
            float momentum = 0.1f;

            // Create BatchNorm2d layer
            boat_batchnorm2d_layer_t* bn_layer = boat_batchnorm2d_layer_create(num_features, eps, momentum, affine);
            if (!bn_layer) {
                fprintf(stderr, "Warning: Failed to create BatchNorm2d layer for %s\n", name.c_str());
                continue;
            }

            // Set weight and bias if affine
            if (affine) {
                boat_tensor_t* weight_tensor = weight_it->second;
                boat_batchnorm2d_layer_set_weight(bn_layer, weight_tensor);

                auto bias_it = model_data->parameters.find(bias_key);
                if (bias_it != model_data->parameters.end()) {
                    boat_tensor_t* bias_tensor = bias_it->second;
                    boat_batchnorm2d_layer_set_bias(bn_layer, bias_tensor);
                }
            }

            // Set running statistics from buffers
            auto running_mean_it = model_data->buffers.find(running_mean_key);
            if (running_mean_it != model_data->buffers.end()) {
                boat_tensor_t* running_mean_tensor = running_mean_it->second;
                boat_batchnorm2d_layer_set_running_mean(bn_layer, running_mean_tensor);
            }

            auto running_var_it = model_data->buffers.find(running_var_key);
            if (running_var_it != model_data->buffers.end()) {
                boat_tensor_t* running_var_tensor = running_var_it->second;
                boat_batchnorm2d_layer_set_running_var(bn_layer, running_var_tensor);
            }

            // Convert to generic layer
            boat_layer_t* generic_layer = (boat_layer_t*)bn_layer;
            boat_sequential_add(seq_model, generic_layer);
        } else if (module_type.find("Softmax") != std::string::npos) {
            // Softmax layer
            // Extract dimension (default is -1 in PyTorch)
            int axis = -1; // TODO: Extract actual dim from PyTorch module
            boat_softmax_layer_t* softmax_layer = boat_softmax_layer_create(axis);
            if (!softmax_layer) {
                fprintf(stderr, "Warning: Failed to create Softmax layer for %s\n", name.c_str());
                continue;
            }
            // Convert to generic layer
            boat_layer_t* generic_layer = (boat_layer_t*)softmax_layer;
            boat_sequential_add(seq_model, generic_layer);
        } else if (module_type.find("MaxPool2d") != std::string::npos) {
            // MaxPool2d layer
            // TODO: Extract actual kernel_size, stride, padding from PyTorch module
            // Default values for nn.MaxPool2d(2, 2, 0)
            size_t kernel_size = 2;
            size_t stride = 2;
            size_t padding = 0;

            boat_pool_layer_t* pool_layer = boat_pool_layer_create(kernel_size, stride, padding);
            if (!pool_layer) {
                fprintf(stderr, "Warning: Failed to create MaxPool2d layer for %s\n", name.c_str());
                continue;
            }
            // Convert to generic layer
            boat_layer_t* generic_layer = (boat_layer_t*)pool_layer;
            boat_sequential_add(seq_model, generic_layer);
        } else if (module_type.find("Flatten") != std::string::npos) {
            // Flatten layer
            boat_flatten_layer_t* flatten_layer = boat_flatten_layer_create();
            if (!flatten_layer) {
                fprintf(stderr, "Warning: Failed to create Flatten layer for %s\n", name.c_str());
                continue;
            }
            // Convert to generic layer
            boat_layer_t* generic_layer = (boat_layer_t*)flatten_layer;
            boat_sequential_add(seq_model, generic_layer);
        } else if (module_type.find("LSTM") != std::string::npos) {
            // LSTM layer - extract parameters and create LSTM layer
            // TODO: Extract LSTM parameters (weight_ih, weight_hh, bias_ih, bias_hh)
            // For now, create a placeholder LSTM layer
            size_t input_size = 1;  // TODO: Extract from parameter shapes
            size_t hidden_size = 1;
            size_t num_layers = 1;
            bool bidirectional = false;
            float dropout = 0.0f;

            boat_lstm_layer_t* lstm_layer = boat_lstm_layer_create(input_size, hidden_size, num_layers, bidirectional, dropout);
            if (!lstm_layer) {
                fprintf(stderr, "Warning: Failed to create LSTM layer for %s\n", name.c_str());
                continue;
            }

            // TODO: Set parameters from model_data

            boat_layer_t* generic_layer = (boat_layer_t*)lstm_layer;
            boat_sequential_add(seq_model, generic_layer);
        } else if (module_type.find("GRU") != std::string::npos) {
            // GRU layer - similar to LSTM
            size_t input_size = 1;
            size_t hidden_size = 1;
            size_t num_layers = 1;
            bool bidirectional = false;
            float dropout = 0.0f;

            boat_gru_layer_t* gru_layer = boat_gru_layer_create(input_size, hidden_size, num_layers, bidirectional, dropout);
            if (!gru_layer) {
                fprintf(stderr, "Warning: Failed to create GRU layer for %s\n", name.c_str());
                continue;
            }

            // TODO: Set parameters

            boat_layer_t* generic_layer = (boat_layer_t*)gru_layer;
            boat_sequential_add(seq_model, generic_layer);
        } else if (module_type.find("Dropout") != std::string::npos) {
            // Dropout is usually only during training, can skip for inference
            continue;
        } else if (module_type.find("Sequential") != std::string::npos ||
                   module_type.find("Module") != std::string::npos ||
                   module_type.find("Container") != std::string::npos) {
            // Skip container modules, we're already iterating through their children
            continue;
        } else {
            fprintf(stderr, "Warning: Unsupported module type: %s (layer: %s)\n", module_type.c_str(), name.c_str());
        }
    }

    // If no layers were added, return empty sequential model
    return seq_model;
}

extern "C" {
    boat_model_t* boat_pytorch_load(const char* filename) {
        if (!filename) {
            return NULL;
        }

        try {
            // Load PyTorch model using LibTorch
            torch::jit::script::Module module;
            try {
                module = torch::jit::load(filename);
            } catch (const c10::Error& e) {
                fprintf(stderr, "Failed to load PyTorch model %s: %s\n",
                        filename, e.what());
                return NULL;
            }

            // Create Boat model
            boat_model_t* model = boat_model_create();
            if (!model) {
                fprintf(stderr, "Failed to create Boat model\n");
                return NULL;
            }

            // Create model data structure to store parameters and buffers
            pytorch_model_data_t* model_data = new pytorch_model_data_t();
            if (!model_data) {
                fprintf(stderr, "Failed to allocate model data\n");
                boat_model_free(model);
                return NULL;
            }

            // Extract model parameters
            auto params = module.named_parameters();
            for (const auto& param : params) {
                torch::Tensor torch_tensor = param.value;

                // Skip non-contiguous tensors (need to make contiguous first)
                if (!torch_tensor.is_contiguous()) {
                    torch_tensor = torch_tensor.contiguous();
                }

                // Convert shape
                int64_t* shape = NULL;
                size_t ndim = 0;
                torch_shape_to_boat(torch_tensor, &shape, &ndim);

                // Convert data type
                boat_dtype_t dtype = torch_dtype_to_boat(torch_tensor.scalar_type());

                // Get tensor data
                void* data = torch_tensor.data_ptr();
                size_t nbytes = torch_tensor.nbytes();

                // Create Boat tensor
                boat_tensor_t* boat_tensor = boat_tensor_create(shape, ndim, dtype, BOAT_DEVICE_CPU);
                if (!boat_tensor) {
                    boat_free(shape);
                    delete model_data;
                    boat_model_free(model);
                    return NULL;
                }

                // Copy data
                void* boat_data = boat_tensor_data(boat_tensor);
                memcpy(boat_data, data, nbytes);

                // Store tensor with its name
                model_data->parameters[param.name] = boat_tensor;

                // Clean up shape array
                boat_free(shape);
            }

            // Extract model buffers (non-parameter tensors)
            auto buffers = module.named_buffers();
            for (const auto& buffer : buffers) {
                torch::Tensor torch_tensor = buffer.value;

                if (!torch_tensor.is_contiguous()) {
                    torch_tensor = torch_tensor.contiguous();
                }

                int64_t* shape = NULL;
                size_t ndim = 0;
                torch_shape_to_boat(torch_tensor, &shape, &ndim);

                boat_dtype_t dtype = torch_dtype_to_boat(torch_tensor.scalar_type());
                void* data = torch_tensor.data_ptr();
                size_t nbytes = torch_tensor.nbytes();

                boat_tensor_t* boat_tensor = boat_tensor_create(shape, ndim, dtype, BOAT_DEVICE_CPU);
                if (!boat_tensor) {
                    boat_free(shape);
                    delete model_data;
                    boat_model_free(model);
                    return NULL;
                }

                void* boat_data = boat_tensor_data(boat_tensor);
                memcpy(boat_data, data, nbytes);

                model_data->buffers[buffer.name] = boat_tensor;

                boat_free(shape);
            }

            // Set model user data
            boat_model_set_user_data(model, model_data, free_pytorch_model_data);

            // Try to convert PyTorch module to Boat model structure
            boat_model_t* converted_model = convert_pytorch_module_to_boat_model(module, model_data);
            if (converted_model) {
                // Transfer user data to converted model
                boat_model_set_user_data(converted_model, model_data, free_pytorch_model_data);
                // Free the original model (it doesn't have any layers yet)
                boat_model_free(model);
                return converted_model;
            } else {
                // Conversion failed, return basic model with parameters
                fprintf(stderr, "Warning: Model architecture conversion failed, returning basic model\n");
                return model;
            }

        } catch (const std::exception& e) {
            fprintf(stderr, "PyTorch loading error: %s\n", e.what());
            return NULL;
        }
    }

    // Save model to PyTorch format
    bool boat_pytorch_save(const boat_model_t* model, const char* filename) {
        (void)model;
        (void)filename;
        fprintf(stderr, "PyTorch model saving not implemented yet\n");
        return false;
    }

    // Load PyTorch model from memory buffer
    boat_model_t* boat_pytorch_load_from_memory(const void* data, size_t size) {
        (void)data;
        (void)size;
        fprintf(stderr, "PyTorch model loading from memory not implemented yet\n");
        return NULL;
    }

    // Save model to memory buffer in PyTorch format
    bool boat_pytorch_save_to_memory(const boat_model_t* model, void** data, size_t* size) {
        (void)model;
        (void)data;
        (void)size;
        fprintf(stderr, "PyTorch model saving to memory not implemented yet\n");
        return false;
    }

    // Check if file is a valid PyTorch model
    bool boat_pytorch_check(const char* filename) {
        if (!filename) {
            return false;
        }

        try {
            // Try to load the model to check if it's valid
            torch::jit::script::Module module = torch::jit::load(filename);
            return true;
        } catch (const std::exception& e) {
            // File is not a valid PyTorch model
            (void)e;
            return false;
        }
    }

    // Convert PyTorch model to Boat model with specific device
    boat_model_t* boat_pytorch_convert(const char* filename, boat_device_t device) {
        // For now, just load and ignore device conversion
        (void)device;
        return boat_pytorch_load(filename);
    }
}