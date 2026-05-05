// Standalone sinusoidal embedding implementation for linking with pre-built boat.lib
// boat.lib was built before boat_sinusoidal_embedding was added to activation.c
#include <boat/tensor.h>
#include <math.h>
#include <stdlib.h>

boat_tensor_t* boat_sinusoidal_embedding(size_t seq_len, size_t embedding_dim, float theta) {
    size_t half = embedding_dim / 2;
    float *data = (float*)calloc(seq_len * embedding_dim, sizeof(float));
    if (!data) return NULL;

    for (size_t pos = 0; pos < seq_len; pos++) {
        for (size_t i = 0; i < half; i++) {
            float inv_timescale = powf(theta, -2.0f * i / embedding_dim);
            float angle = (float)pos * inv_timescale;
            data[pos * embedding_dim + i] = sinf(angle);            // first half: sin
            data[pos * embedding_dim + i + half] = cosf(angle);     // second half: cos
        }
    }

    const int64_t shape[] = {(int64_t)seq_len, (int64_t)embedding_dim};
    boat_tensor_t *result = boat_tensor_from_data(shape, 2, BOAT_DTYPE_FLOAT32, data);
    free(data);
    return result;
}
