// sampling.c - Token sampling implementation
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/sampling.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

typedef struct { float val; int idx; } val_idx_t;

static int cmp_desc(const void* a, const void* b) {
    const val_idx_t* pa = (const val_idx_t*)a;
    const val_idx_t* pb = (const val_idx_t*)b;
    return (pa->val > pb->val) ? -1 : (pa->val < pb->val) ? 1 : 0;
}

BOAT_API int boat_sample_token(const float* logits, int vocab_size,
                       int top_k, float temperature) {
    if (!logits || vocab_size <= 0) {
        return 0;
    }

    // Greedy: return argmax
    if (temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > logits[best]) best = i;
        }
        return best;
    }

    if (top_k <= 0 || top_k > vocab_size) {
        top_k = vocab_size;
    }

    val_idx_t* arr = (val_idx_t*)malloc((size_t)vocab_size * sizeof(val_idx_t));
    if (!arr) {
        return 0;
    }

    // Find max logit for numerical stability
    float max_l = logits[0];
    for (int i = 1; i < vocab_size; i++) {
        if (logits[i] > max_l) max_l = logits[i];
    }

    // Scale by temperature and shift by max
    for (int i = 0; i < vocab_size; i++) {
        arr[i].val = (logits[i] - max_l) / temperature;
        arr[i].idx = i;
    }

    // Sort descending by value
    qsort(arr, (size_t)vocab_size, sizeof(val_idx_t), cmp_desc);

    // Softmax over top-k: exponentiate and sum
    float sum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        arr[i].val = expf(arr[i].val);
        sum += arr[i].val;
    }

    // Sample from the distribution
    float r = (float)rand() / (float)RAND_MAX * sum;
    float cum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        cum += arr[i].val;
        if (r <= cum) {
            int idx = arr[i].idx;
            free(arr);
            return idx;
        }
    }

    // Fallback (should not reach here if top_k > 0)
    int idx = arr[top_k - 1].idx;
    free(arr);
    return idx;
}
