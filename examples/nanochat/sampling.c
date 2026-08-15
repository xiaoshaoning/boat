// sampling.c - Top-k sampling
#include "sampling.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>

typedef struct {
    float val;
    int idx;
} val_idx_t;

static int cmp_desc(const void* a, const void* b) {
    const val_idx_t* pa = (const val_idx_t*)a;
    const val_idx_t* pb = (const val_idx_t*)b;
    return (pa->val > pb->val) ? -1 : (pa->val < pb->val) ? 1 : 0;
}

int nanochat_sample_token(const float* logits, int vocab_size, int top_k, float temp) {
    if (temp <= 0.0f) {
        int best = 0;
        for (int i = 1; i < vocab_size; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    val_idx_t* arr = (val_idx_t*)malloc((size_t)vocab_size * sizeof(val_idx_t));
    float max_l = logits[0];
    for (int i = 1; i < vocab_size; i++)
        if (logits[i] > max_l) max_l = logits[i];

    for (int i = 0; i < vocab_size; i++) {
        arr[i].val = (logits[i] - max_l) / temp;
        arr[i].idx = i;
    }
    qsort(arr, vocab_size, sizeof(val_idx_t), cmp_desc);

    float sum = 0.0f;
    for (int i = 0; i < top_k; i++) {
        arr[i].val = expf(arr[i].val);
        sum += arr[i].val;
    }

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
    int idx = arr[top_k - 1].idx;
    free(arr);
    return idx;
}
