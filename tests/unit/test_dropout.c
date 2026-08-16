// test_dropout.c - Dropout layer behavior
#include <boat.h>
#include <stdio.h>
#include <stdlib.h>

static int g_failures = 0;
#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            printf("  FAIL: %s\n", msg);                                         \
            g_failures++;                                                        \
        } else {                                                                 \
            printf("  OK: %s\n", msg);                                           \
        }                                                                        \
    } while (0)

int test_dropout_basic(void) {
    boat_dropout_layer_t* d = boat_dropout_layer_create(0.5f);
    boat_tensor_t* x = boat_tensor_create((const int64_t[]){1, 200}, 2,
                                          BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* xd = (float*)boat_tensor_data(x);
    for (int i = 0; i < 200; i++) xd[i] = 1.0f;

    /* inference: identity */
    boat_tensor_t* y = boat_dropout_layer_forward(d, x);
    CHECK(y != NULL, "inference forward returns output");
    const float* yd = (const float*)boat_tensor_const_data(y);
    int ident = 1;
    for (int i = 0; i < 200; i++)
        if (yd[i] != 1.0f) ident = 0;
    CHECK(ident, "inference is identity");
    boat_tensor_free(y);

    /* training: mask + scale, expect roughly p=0.5 dropped and ~1/(1-p) kept */
    boat_dropout_layer_set_training(d, true);
    srand(1234);
    y = boat_dropout_layer_forward(d, x);
    yd = (const float*)boat_tensor_const_data(y);
    int zero = 0;
    for (int i = 0; i < 200; i++) {
        if (yd[i] == 0.0f) zero++;
        else if (yd[i] != 2.0f) { CHECK(0, "training values are 0 or 1/(1-p)"); break; }
    }
    printf("  dropped %d/200 (expect ~100)\n", zero);
    CHECK(zero > 60 && zero < 140, "dropout rate roughly p");

    /* backward applies the same cached mask */
    boat_tensor_t* g = boat_tensor_create((const int64_t[]){1, 200}, 2,
                                          BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* gd = (float*)boat_tensor_data(g);
    for (int i = 0; i < 200; i++) gd[i] = 1.0f;
    boat_tensor_t* gi = boat_dropout_layer_backward(d, g);
    CHECK(gi != NULL, "backward returns grad_input");
    const float* gid = (const float*)boat_tensor_const_data(gi);
    int bzero = 0, btwo = 0;
    for (int i = 0; i < 200; i++) {
        if (gid[i] == 0.0f) bzero++;
        else if (gid[i] == 2.0f) btwo++;
    }
    CHECK(bzero == zero && btwo == 200 - zero, "backward mask matches forward");

    /* p = 0 disables masking even in training mode */
    boat_dropout_layer_t* d0 = boat_dropout_layer_create(0.0f);
    boat_dropout_layer_set_training(d0, true);
    boat_tensor_t* y0 = boat_dropout_layer_forward(d0, x);
    const float* y0d = (const float*)boat_tensor_const_data(y0);
    int ok0 = 1;
    for (int i = 0; i < 200; i++)
        if (y0d[i] != 1.0f) ok0 = 0;
    CHECK(ok0, "p=0 is identity in training mode");

    boat_tensor_free(y); boat_tensor_free(y0);
    boat_tensor_free(x); boat_tensor_free(g); boat_tensor_free(gi);
    boat_dropout_layer_free(d); boat_dropout_layer_free(d0);
    return g_failures;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Dropout Layer Tests\n");
    printf("===================\n\n");
    int fail = 0;
    fail |= test_dropout_basic();
    printf("\n%s\n", fail ? "FAILED" : "ALL PASSED");
    return fail != 0;
}
