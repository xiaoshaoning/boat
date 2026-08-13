// sampling.c - Top-k temperature sampling
#include "sampling.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

int minimind_sample_token(const float* logits, int vocab_size,
                           int top_k, float temperature) {
    // Copy logits
    float* work = (float*)malloc((size_t)vocab_size * sizeof(float));
    memcpy(work, logits, (size_t)vocab_size * sizeof(float));

    // Temperature scaling
    if (temperature > 0.0f) {
        float inv_temp = 1.0f / temperature;
        for (int i = 0; i < vocab_size; i++) work[i] *= inv_temp;
    }

    // Top-k filtering
    if (top_k > 0 && top_k < vocab_size) {
        // Find k-th largest value using partial sort
        float* sorted = (float*)malloc((size_t)vocab_size * sizeof(float));
        memcpy(sorted, work, (size_t)vocab_size * sizeof(float));
        // Simple: nth_element via full sort (small vocab, fine)
        for (int i = 0; i < vocab_size - 1; i++) {
            for (int j = i + 1; j < vocab_size; j++) {
                if (sorted[j] > sorted[i]) {
                    float t = sorted[i]; sorted[i] = sorted[j]; sorted[j] = t;
                }
            }
        }
        float threshold = sorted[top_k - 1];
        free(sorted);
        for (int i = 0; i < vocab_size; i++) {
            if (work[i] < threshold) work[i] = -INFINITY;
        }
    }

    // Softmax
    float max_val = -INFINITY;
    for (int i = 0; i < vocab_size; i++) {
        if (work[i] > max_val) max_val = work[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        if (work[i] <= -1e30f) { work[i] = 0.0f; continue; }
        work[i] = expf(work[i] - max_val);
        sum += work[i];
    }

    // Greedy if temp=0 or sum=0
    if (temperature <= 0.0f || sum <= 0.0f) {
        int best = 0;
        float best_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > best_val) { best_val = logits[i]; best = i; }
        }
        free(work);
        return best;
    }

    // Sample from distribution
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0.0f;
    int chosen = 0;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += work[i] / sum;
        if (r <= cumsum) { chosen = i; break; }
    }
    free(work);
    return chosen;
}
