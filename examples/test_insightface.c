// test_insightface.c - Test InsightFace ONNX recognition model
#include <boat.h>
#include <boat/format/onnx.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: test_insightface <model.onnx>\n");
        return 1;
    }

    boat_init();

    printf("Loading ONNX model: %s\n", argv[1]);
    boat_onnx_runtime_t* rt = boat_onnx_runtime_load(argv[1]);
    if (!rt) {
        fprintf(stderr, "Failed to load model\n");
        boat_cleanup();
        return 1;
    }
    printf("Model loaded successfully\n");

    // Create input: batch=1, 3x112x112, random pixels in [0,1]
    int64_t shape[] = {1, 3, 112, 112};
    boat_tensor_t* input = boat_tensor_create(shape, 4, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* d = (float*)boat_tensor_data(input);
    size_t n = boat_tensor_nelements(input);
    for (size_t i = 0; i < n; i++) d[i] = (float)(i % 256) / 255.0f;

    printf("Running inference...\n");
    boat_tensor_t* output = boat_onnx_runtime_run(rt, input);
    if (!output) {
        fprintf(stderr, "Inference failed\n");
        boat_tensor_unref(input);
        boat_onnx_runtime_free(rt);
        boat_cleanup();
        return 1;
    }

    printf("Output shape: [");
    const int64_t* oshape = boat_tensor_shape(output);
    for (size_t i = 0; i < boat_tensor_ndim(output); i++)
        printf("%lld%c", (long long)oshape[i], i+1 < boat_tensor_ndim(output) ? ',' : ']');
    printf("\n");

    const float* od = (const float*)boat_tensor_const_data(output);
    size_t on = boat_tensor_nelements(output);
    float sum = 0.0f, sum2 = 0.0f;
    for (size_t i = 0; i < on; i++) {
        sum += od[i];
        sum2 += od[i] * od[i];
    }
    float mean = sum / on;
    float rms = sqrtf(sum2 / on);
    printf("Output stats: min=%.4f max=%.4f mean=%.4f rms=%.4f\n",
           od[0], od[on-1], mean, rms);
    printf("First 8 values: ");
    for (int i = 0; i < 8 && i < (int)on; i++) printf("%.4f ", od[i]);
    printf("\n");

    boat_tensor_unref(output);
    boat_tensor_unref(input);
    boat_onnx_runtime_free(rt);
    boat_cleanup();
    printf("All done.\n");
    return 0;
}
