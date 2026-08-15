// onnxruntime.c - ONNX Runtime C inference backend
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat.h>
#include <boat/format/onnxruntime.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <onnxruntime_c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

// -----------------------------------------------------------------------
// Internal session structure
// -----------------------------------------------------------------------
struct boat_onnxruntime_session_t {
    const OrtApi* api;      // ORT C API function table
    OrtEnv* env;            // ORT environment
    OrtSession* session;    // Loaded ONNX session
    OrtMemoryInfo* cpu_mem; // CPU memory info (reused)
    char** input_names;     // Cached input names (OrtAllocator-owned copies)
    char** output_names;    // Cached output names
    size_t num_inputs;
    size_t num_outputs;
    boat_onnxruntime_provider_t provider;
    int device_id; // GPU device ID (for CUDA provider)
};

// -----------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------

// Check ORT status and print error on failure. Returns non-zero on failure.
static int check_ort_status(const OrtApi* api, OrtStatus* status, const char* msg) {
    if (!status) return 0;
    const char* err = api->GetErrorMessage(status);
    fprintf(stderr, "[ONNXRuntime] %s: %s\n", msg, err ? err : "unknown");
    boat_set_error(BOAT_ERROR_DEVICE, err ? err : msg);
    api->ReleaseStatus(status);
    return 1;
}

// Free an array of C strings allocated by ORT's allocator.
static void free_ort_strings(const OrtApi* api, char** names, size_t count,
                             OrtAllocator* allocator) {
    if (!names) return;
    for (size_t i = 0; i < count; i++) {
        if (names[i]) allocator->Free(allocator, names[i]);
    }
}

// -----------------------------------------------------------------------
// Session creation (internal)
// -----------------------------------------------------------------------

static boat_onnxruntime_session_t* create_session_impl(const void* model_data, size_t model_size,
                                                       int is_buffer,
                                                       boat_onnxruntime_provider_t provider) {

    // Get ORT API
    const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!api) {
        boat_set_error(BOAT_ERROR_INVALID_OPERATION,
                       "ONNX Runtime API not available. Check ORT version.");
        return NULL;
    }

    boat_onnxruntime_session_t* s = (boat_onnxruntime_session_t*)boat_malloc(
        sizeof(boat_onnxruntime_session_t), BOAT_DEVICE_CPU);
    if (!s) return NULL;
    memset(s, 0, sizeof(*s));
    s->api = api;
    s->provider = provider;
    s->device_id = 0;

    // Create environment
    OrtEnv* env = NULL;
    OrtStatus* status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "boat", &env);
    if (check_ort_status(api, status, "CreateEnv failed")) goto fail;
    s->env = env;

    // Create session options
    OrtSessionOptions* session_options = NULL;
    status = api->CreateSessionOptions(&session_options);
    if (check_ort_status(api, status, "CreateSessionOptions failed")) goto fail;

    // Enable graph optimization
    api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_ALL);

    // Set intra-op thread count
    api->SetIntraOpNumThreads(session_options, 4);
    api->SetInterOpNumThreads(session_options, 2);

    // Set session log id
    api->SetSessionLogId(session_options, "boat");

    // Try CUDA execution provider
    int cuda_available = 0;
    if (provider == BOAT_ORT_CUDA || provider == BOAT_ORT_AUTO) {
        // CUDA provider options (OrtCUDAProviderOptionsV2)
        OrtCUDAProviderOptionsV2* cuda_options = NULL;
        status = api->CreateCUDAProviderOptions(&cuda_options);
        if (!status && cuda_options) {
            // Set default CUDA options
            const char* keys[] = {"device_id", "arena_extend_strategy", "cudnn_conv_algo_search"};
            const char* vals[] = {"0", "kSameAsRequested", "DEFAULT"};
            status = api->UpdateCUDAProviderOptions(cuda_options, keys, vals, 3);
            if (!status) {
                status =
                    api->SessionOptionsAppendExecutionProvider_CUDA(session_options, cuda_options);
                if (!status) {
                    cuda_available = 1;
                } else {
                    // CUDA not available, release status
                    api->ReleaseStatus(status);
                    status = NULL;
                }
            }
            api->ReleaseCUDAProviderOptions(cuda_options);
        } else if (status) {
            api->ReleaseStatus(status);
            status = NULL;
        }

        if (provider == BOAT_ORT_CUDA && !cuda_available) {
            fprintf(stderr, "[ONNXRuntime] CUDA execution provider not available.\n");
            api->ReleaseSessionOptions(session_options);
            goto fail;
        }
    }

    // Create session
    OrtSession* session = NULL;
    if (is_buffer) {
        status =
            api->CreateSessionFromArray(env, model_data, model_size, session_options, &session);
    } else {
        status = api->CreateSession(env, (const char*)model_data, session_options, &session);
    }
    api->ReleaseSessionOptions(session_options);

    if (check_ort_status(api, status, "CreateSession failed")) goto fail;
    s->session = session;

    // Query input/output count
    size_t num_inputs = 0, num_outputs = 0;
    status = api->SessionGetInputCount(session, &num_inputs);
    if (check_ort_status(api, status, "SessionGetInputCount failed")) goto fail;
    status = api->SessionGetOutputCount(session, &num_outputs);
    if (check_ort_status(api, status, "SessionGetOutputCount failed")) goto fail;
    s->num_inputs = num_inputs;
    s->num_outputs = num_outputs;

    // Get allocator for name strings
    OrtAllocator* allocator = NULL;
    status = api->GetAllocatorWithDefaultOptions(&allocator);
    if (check_ort_status(api, status, "GetAllocatorWithDefaultOptions failed")) goto fail;

    // Cache input names
    s->input_names = (char**)boat_malloc(num_inputs * sizeof(char*), BOAT_DEVICE_CPU);
    if (!s->input_names) goto fail;
    memset(s->input_names, 0, num_inputs * sizeof(char*));
    for (size_t i = 0; i < num_inputs; i++) {
        char* name = NULL;
        status = api->SessionGetInputName(session, i, allocator, &name);
        if (check_ort_status(api, status, "SessionGetInputName failed")) goto fail;
        s->input_names[i] = name;
    }

    // Cache output names
    s->output_names = (char**)boat_malloc(num_outputs * sizeof(char*), BOAT_DEVICE_CPU);
    if (!s->output_names) goto fail;
    memset(s->output_names, 0, num_outputs * sizeof(char*));
    for (size_t i = 0; i < num_outputs; i++) {
        char* name = NULL;
        status = api->SessionGetOutputName(session, i, allocator, &name);
        if (check_ort_status(api, status, "SessionGetOutputName failed")) goto fail;
        s->output_names[i] = name;
    }

    // Create CPU memory info
    OrtMemoryInfo* cpu_mem = NULL;
    status = api->CreateMemoryInfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault, &cpu_mem);
    if (check_ort_status(api, status, "CreateMemoryInfo failed")) goto fail;
    s->cpu_mem = cpu_mem;

    return s;

fail:
    // Cleanup on failure
    if (s->session) api->ReleaseSession(s->session);
    if (s->env) api->ReleaseEnv(s->env);
    if (s->cpu_mem) api->ReleaseMemoryInfo(s->cpu_mem);
    if (s->input_names) {
        if (allocator) free_ort_strings(api, s->input_names, num_inputs, allocator);
        boat_free(s->input_names);
    }
    if (s->output_names) {
        if (allocator) free_ort_strings(api, s->output_names, num_outputs, allocator);
        boat_free(s->output_names);
    }
    boat_free(s);
    return NULL;
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

BOAT_API boat_onnxruntime_session_t* boat_onnxruntime_create(const char* model_path,
                                                             boat_onnxruntime_provider_t provider) {
    if (!model_path) {
        boat_set_error(BOAT_ERROR_INVALID_ARGUMENT, "boat_onnxruntime_create: model_path is NULL");
        return NULL;
    }
    return create_session_impl(model_path, 0, 0, provider);
}

BOAT_API boat_onnxruntime_session_t*
boat_onnxruntime_create_from_buffer(const void* data, size_t size,
                                    boat_onnxruntime_provider_t provider) {
    if (!data || size == 0) {
        boat_set_error(BOAT_ERROR_INVALID_ARGUMENT,
                       "boat_onnxruntime_create_from_buffer: data is NULL or empty");
        return NULL;
    }
    return create_session_impl(data, size, 1, provider);
}

BOAT_API void boat_onnxruntime_free(boat_onnxruntime_session_t* session) {
    if (!session) return;
    const OrtApi* api = session->api;

    // Get allocator for freeing name strings
    OrtAllocator* allocator = NULL;
    OrtStatus* status = api->GetAllocatorWithDefaultOptions(&allocator);
    if (status) {
        api->ReleaseStatus(status);
        // Can't free names gracefully, release them directly
        if (session->input_names) boat_free(session->input_names);
        if (session->output_names) boat_free(session->output_names);
    } else {
        free_ort_strings(api, session->input_names, session->num_inputs, allocator);
        free_ort_strings(api, session->output_names, session->num_outputs, allocator);
        boat_free(session->input_names);
        boat_free(session->output_names);
    }

    if (session->cpu_mem) api->ReleaseMemoryInfo(session->cpu_mem);
    if (session->session) api->ReleaseSession(session->session);
    if (session->env) api->ReleaseEnv(session->env);
    boat_free(session);
}

// -----------------------------------------------------------------------
// Introspection
// -----------------------------------------------------------------------

BOAT_API size_t boat_onnxruntime_input_count(const boat_onnxruntime_session_t* session) {
    return session ? session->num_inputs : 0;
}

BOAT_API size_t boat_onnxruntime_output_count(const boat_onnxruntime_session_t* session) {
    return session ? session->num_outputs : 0;
}

BOAT_API const char* boat_onnxruntime_input_name(const boat_onnxruntime_session_t* session,
                                                 size_t index) {
    if (!session || index >= session->num_inputs) return NULL;
    return session->input_names[index];
}

BOAT_API const char* boat_onnxruntime_output_name(const boat_onnxruntime_session_t* session,
                                                  size_t index) {
    if (!session || index >= session->num_outputs) return NULL;
    return session->output_names[index];
}

// -----------------------------------------------------------------------
// Inference
// -----------------------------------------------------------------------

// Convert a boat_tensor_t to an OrtValue for use as ORT input.
// Uses the session's CPU memory info.
static OrtValue* boat_tensor_to_ort_value(const boat_onnxruntime_session_t* session,
                                          const boat_tensor_t* tensor, const OrtApi* api) {

    if (!tensor) return NULL;

    // Get tensor properties
    const int64_t* shape = boat_tensor_shape(tensor);
    size_t ndim = boat_tensor_ndim(tensor);
    size_t nbytes = boat_tensor_nbytes(tensor);
    void* data = boat_tensor_data(tensor);

    if (!shape || ndim == 0 || !data) return NULL;

    // Create OrtValue wrapping the tensor data (no copy for CPU)
    OrtValue* ort_val = NULL;
    OrtStatus* status =
        api->CreateTensorWithDataAsOrtValue(session->cpu_mem, data, nbytes, shape, (int)ndim,
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ort_val);
    if (status) {
        check_ort_status(api, status, "CreateTensorWithDataAsOrtValue failed");
        return NULL;
    }
    return ort_val;
}

// Convert an OrtValue output to a boat_tensor_t (copies data).
static boat_tensor_t* ort_value_to_boat_tensor(const OrtValue* ort_val, const OrtApi* api) {

    // Get output type and shape
    OrtTensorTypeAndShapeInfo* info = NULL;
    OrtStatus* status = api->GetTensorTypeAndShape(ort_val, &info);
    if (check_ort_status(api, status, "GetTensorTypeAndShape failed")) return NULL;

    ONNXTensorElementDataType elem_type;
    status = api->GetTensorElementType(info, &elem_type);
    if (check_ort_status(api, status, "GetTensorElementType failed")) {
        api->ReleaseTensorTypeAndShapeInfo(info);
        return NULL;
    }

    size_t num_dims = 0;
    status = api->GetDimensionsCount(info, &num_dims);
    if (check_ort_status(api, status, "GetDimensionsCount failed")) {
        api->ReleaseTensorTypeAndShapeInfo(info);
        return NULL;
    }

    // Read shape
    int64_t* shape = (int64_t*)alloca(num_dims * sizeof(int64_t));
    status = api->GetDimensions(info, shape, num_dims);
    if (check_ort_status(api, status, "GetDimensions failed")) {
        api->ReleaseTensorTypeAndShapeInfo(info);
        return NULL;
    }

    // Get raw data pointer
    void* data = NULL;
    status = api->GetTensorMutableData((OrtValue*)ort_val, &data);
    if (check_ort_status(api, status, "GetTensorMutableData failed")) {
        api->ReleaseTensorTypeAndShapeInfo(info);
        return NULL;
    }

    // Compute total element count
    size_t total_count = 1;
    for (size_t i = 0; i < num_dims; i++)
        total_count *= (size_t)shape[i];

    // Determine boat dtype
    boat_dtype_t boat_dtype = BOAT_DTYPE_FLOAT32;
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        boat_dtype = BOAT_DTYPE_FLOAT32;
    } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        boat_dtype = BOAT_DTYPE_INT64;
    } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        boat_dtype = BOAT_DTYPE_INT32;
    } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        boat_dtype = BOAT_DTYPE_FLOAT64;
    } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
        boat_dtype = BOAT_DTYPE_UINT8;
    } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL) {
        boat_dtype = BOAT_DTYPE_BOOL;
    } else {
        // Fallback: copy as raw bytes using FLOAT32 size, then reinterpret
        // For unsupported types, default to float and copy byte-wise
        boat_dtype = BOAT_DTYPE_FLOAT32;
    }

    // Create boat tensor and copy data
    boat_tensor_t* result = boat_tensor_create(shape, num_dims, boat_dtype, BOAT_DEVICE_CPU);
    if (result) {
        size_t boat_nbytes = boat_tensor_nbytes(result);
        size_t copy_size =
            boat_nbytes < total_count * sizeof(float) ? boat_nbytes : total_count * sizeof(float);
        memcpy(boat_tensor_data(result), data, copy_size);
    }

    api->ReleaseTensorTypeAndShapeInfo(info);
    return result;
}

BOAT_API boat_tensor_t* boat_onnxruntime_run(boat_onnxruntime_session_t* session,
                                             const boat_tensor_t* input) {

    if (!session || !input) {
        boat_set_error(BOAT_ERROR_INVALID_ARGUMENT,
                       "boat_onnxruntime_run: session or input is NULL");
        return NULL;
    }

    const OrtApi* api = session->api;

    // Convert input tensor to OrtValue
    OrtValue* ort_input = boat_tensor_to_ort_value(session, input, api);
    if (!ort_input) return NULL;

    // Prepare input names
    const char* input_name = session->num_inputs > 0 ? session->input_names[0] : NULL;
    if (!input_name) {
        boat_set_error(BOAT_ERROR_INVALID_OPERATION,
                       "boat_onnxruntime_run: no input names available");
        api->ReleaseValue(ort_input);
        return NULL;
    }

    // Prepare output names
    size_t num_outputs = session->num_outputs;
    const char** output_names = (const char**)alloca(num_outputs * sizeof(char*));
    for (size_t i = 0; i < num_outputs; i++) {
        output_names[i] = session->output_names[i];
    }

    // Run inference
    OrtValue** ort_outputs = (OrtValue**)alloca(num_outputs * sizeof(OrtValue*));
    memset(ort_outputs, 0, num_outputs * sizeof(OrtValue*));

    OrtStatus* status = api->Run(session->session, NULL, &input_name, &ort_input, 1, output_names,
                                 num_outputs, ort_outputs);
    api->ReleaseValue(ort_input);

    if (check_ort_status(api, status, "Run failed")) {
        // Release any partial outputs
        for (size_t i = 0; i < num_outputs; i++) {
            if (ort_outputs[i]) api->ReleaseValue(ort_outputs[i]);
        }
        return NULL;
    }

    // Convert first output to boat tensor
    boat_tensor_t* result = ort_value_to_boat_tensor(ort_outputs[0], api);

    // Release all output OrtValues
    for (size_t i = 0; i < num_outputs; i++) {
        if (ort_outputs[i]) api->ReleaseValue(ort_outputs[i]);
    }

    return result;
}

BOAT_API boat_tensor_t** boat_onnxruntime_run_multi(boat_onnxruntime_session_t* session,
                                                    boat_tensor_t* const* inputs,
                                                    const char** input_names, size_t num_inputs,
                                                    size_t* num_outputs) {

    if (!session || !inputs || !input_names || num_inputs == 0) {
        boat_set_error(BOAT_ERROR_INVALID_ARGUMENT,
                       "boat_onnxruntime_run_multi: invalid arguments");
        if (num_outputs) *num_outputs = 0;
        return NULL;
    }

    const OrtApi* api = session->api;
    size_t num_outs = session->num_outputs;
    if (num_outputs) *num_outputs = num_outs;

    // Convert all input tensors to OrtValues
    OrtValue** ort_inputs = (OrtValue**)alloca(num_inputs * sizeof(OrtValue*));
    memset(ort_inputs, 0, num_inputs * sizeof(OrtValue*));
    for (size_t i = 0; i < num_inputs; i++) {
        ort_inputs[i] = boat_tensor_to_ort_value(session, inputs[i], api);
        if (!ort_inputs[i]) {
            for (size_t j = 0; j < i; j++)
                if (ort_inputs[j]) api->ReleaseValue(ort_inputs[j]);
            return NULL;
        }
    }

    // Prepare output names from session
    const char** ort_output_names = (const char**)alloca(num_outs * sizeof(char*));
    for (size_t i = 0; i < num_outs; i++) {
        ort_output_names[i] = session->output_names[i];
    }

    // Prepare output OrtValues
    OrtValue** ort_outputs = (OrtValue**)alloca(num_outs * sizeof(OrtValue*));
    memset(ort_outputs, 0, num_outs * sizeof(OrtValue*));

    // Run inference
    OrtStatus* status = api->Run(session->session, NULL, input_names, ort_inputs, num_inputs,
                                 ort_output_names, num_outs, ort_outputs);

    // Release input OrtValues
    for (size_t i = 0; i < num_inputs; i++) {
        if (ort_inputs[i]) api->ReleaseValue(ort_inputs[i]);
    }

    if (check_ort_status(api, status, "Run_multi failed")) {
        for (size_t i = 0; i < num_outs; i++)
            if (ort_outputs[i]) api->ReleaseValue(ort_outputs[i]);
        return NULL;
    }

    // Convert outputs to boat tensors
    boat_tensor_t** results =
        (boat_tensor_t**)boat_malloc(num_outs * sizeof(boat_tensor_t*), BOAT_DEVICE_CPU);
    if (!results) {
        for (size_t i = 0; i < num_outs; i++)
            if (ort_outputs[i]) api->ReleaseValue(ort_outputs[i]);
        return NULL;
    }

    size_t success_count = 0;
    for (size_t i = 0; i < num_outs; i++) {
        results[i] = ort_value_to_boat_tensor(ort_outputs[i], api);
        if (results[i]) success_count++;
        api->ReleaseValue(ort_outputs[i]);
    }

    if (success_count == 0) {
        boat_free(results);
        return NULL;
    }

    return results;
}
