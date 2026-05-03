// sampling.c - Text generation sampling implementations
#include "sampling.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int sample_greedy(const float* logits, int n_vocab) {
    int best = 0;
    for (int i = 1; i < n_vocab; i++)
        if (logits[i] > logits[best]) best = i;
    return best;
}

// Simple comparison for sorting (descending)
typedef struct { float val; int idx; } val_idx_t;
static int cmp_desc(const void* a, const void* b) {
    float diff = ((const val_idx_t*)b)->val - ((const val_idx_t*)a)->val;
    return (diff > 0) ? 1 : (diff < 0) ? -1 : 0;
}

int sample_topk(const float* logits, int n_vocab, int k, float temp) {
    if (k <= 0) k = 1;
    if (k > n_vocab) k = n_vocab;
    if (temp <= 0.0f) return sample_greedy(logits, n_vocab);

    // Build array of (value, index) pairs
    val_idx_t* arr = (val_idx_t*)malloc(n_vocab * sizeof(val_idx_t));
    for (int i = 0; i < n_vocab; i++) { arr[i].val = logits[i] / temp; arr[i].idx = i; }
    qsort(arr, n_vocab, sizeof(val_idx_t), cmp_desc);

    // Softmax over top-k
    float max_val = arr[0].val;
    float sum = 0;
    for (int i = 0; i < k; i++) { arr[i].val = expf(arr[i].val - max_val); sum += arr[i].val; }

    // Sample from distribution
    float r = (float)rand() / (float)RAND_MAX * sum;
    float cum = 0;
    for (int i = 0; i < k; i++) {
        cum += arr[i].val;
        if (r <= cum) { int idx = arr[i].idx; free(arr); return idx; }
    }
    int idx = arr[k - 1].idx;
    free(arr);
    return idx;
}

int sample_topp(const float* logits, int n_vocab, float p, float temp) {
    if (p <= 0.0f || p >= 1.0f) return sample_topk(logits, n_vocab, 40, temp);
    if (temp <= 0.0f) return sample_greedy(logits, n_vocab);

    val_idx_t* arr = (val_idx_t*)malloc(n_vocab * sizeof(val_idx_t));
    for (int i = 0; i < n_vocab; i++) { arr[i].val = logits[i] / temp; arr[i].idx = i; }
    qsort(arr, n_vocab, sizeof(val_idx_t), cmp_desc);

    // Softmax (full) to get probabilities
    float max_val = arr[0].val;
    for (int i = 0; i < n_vocab; i++) arr[i].val = expf(arr[i].val - max_val);

    float sum = 0;
    for (int i = 0; i < n_vocab; i++) sum += arr[i].val;

    // Find cutoff index where cumulative prob exceeds p
    float cum = 0;
    int cutoff = n_vocab;
    for (int i = 0; i < n_vocab; i++) {
        cum += arr[i].val / sum;
        if (cum > p) { cutoff = i + 1; break; }
    }

    // Normalize and sample from top-p set
    float sub_sum = 0;
    for (int i = 0; i < cutoff; i++) sub_sum += arr[i].val;

    float r = (float)rand() / (float)RAND_MAX * sub_sum;
    float sub_cum = 0;
    for (int i = 0; i < cutoff; i++) {
        sub_cum += arr[i].val;
        if (r <= sub_cum) { int idx = arr[i].idx; free(arr); return idx; }
    }
    int idx = arr[cutoff - 1].idx;
    free(arr);
    return idx;
}
