// test_float16.c - FP16 conversion and element-wise arithmetic tests
// Copyright (c) 2026 Shaoning, Xiao 萧少宁
// Licensed under the Apache License, Version 2.0

#include <boat/ops.h>
#include <boat/tensor.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

static float fd(float a, float b) {
    float d = fabsf(a - b);
    return d;
}

static int close_f(float a, float b) {
    // half precision has ~3 decimal digits
    float d = fabsf(a - b);
    float scale = fabsf(a) + fabsf(b);
    return d < 1e-3f * scale + 1e-4f;
}

static void test_conversion(void) {
    printf("Testing FP16 <-> FP32 conversion...\n");
    float vals[] = {0.0f, 1.0f, -1.0f, 2.0f, 0.5f, 0.25f, 1.5f, -3.25f, 100.0f, 65504.0f};
    for (size_t i = 0; i < sizeof(vals) / sizeof(vals[0]); i++) {
        uint16_t h = boat_f32_to_f16(vals[i]);
        float back = boat_f16_to_f32(h);
        assert(close_f(back, vals[i]));
    }
    // Special values.
    assert(boat_f32_to_f16(INFINITY) == 0x7C00u);
    assert(boat_f32_to_f16(-INFINITY) == 0xFC00u);
    assert(boat_f16_to_f32(0x7C00u) == INFINITY);
    printf("  OK\n");
}

static void test_arith(void) {
    printf("Testing FP16 element-wise arithmetic...\n");
    int64_t sh[] = {4};
    uint16_t ad[] = {boat_f32_to_f16(1.0f), boat_f32_to_f16(2.0f), boat_f32_to_f16(3.5f),
                     boat_f32_to_f16(-2.0f)};
    uint16_t bd[] = {boat_f32_to_f16(2.0f), boat_f32_to_f16(3.0f), boat_f32_to_f16(0.5f),
                     boat_f32_to_f16(4.0f)};
    boat_tensor_t* a = boat_tensor_from_data(sh, 1, BOAT_DTYPE_FLOAT16, ad);
    boat_tensor_t* b = boat_tensor_from_data(sh, 1, BOAT_DTYPE_FLOAT16, bd);

    boat_tensor_t* s = boat_add(a, b);
    boat_tensor_t* d = boat_sub(a, b);
    boat_tensor_t* m = boat_mul(a, b);
    boat_tensor_t* v = boat_div(a, b);
    assert(s && d && m && v);
    uint16_t *sd = (uint16_t*)boat_tensor_data(s), *dd = (uint16_t*)boat_tensor_data(d);
    uint16_t *md = (uint16_t*)boat_tensor_data(m), *vd = (uint16_t*)boat_tensor_data(v);

    assert(close_f(boat_f16_to_f32(sd[0]), 3.0f) && close_f(boat_f16_to_f32(sd[1]), 5.0f));
    assert(close_f(boat_f16_to_f32(sd[2]), 4.0f) && close_f(boat_f16_to_f32(sd[3]), 2.0f));
    assert(close_f(boat_f16_to_f32(dd[0]), -1.0f) && close_f(boat_f16_to_f32(dd[3]), -6.0f));
    assert(close_f(boat_f16_to_f32(md[0]), 2.0f) && close_f(boat_f16_to_f32(md[2]), 1.75f));
    assert(close_f(boat_f16_to_f32(vd[0]), 0.5f) && close_f(boat_f16_to_f32(vd[2]), 7.0f));

    // FP16 should produce FP16 output dtype.
    assert(boat_tensor_dtype(s) == BOAT_DTYPE_FLOAT16);

    boat_tensor_free(a);
    boat_tensor_free(b);
    boat_tensor_free(s);
    boat_tensor_free(d);
    boat_tensor_free(m);
    boat_tensor_free(v);
    printf("  OK\n");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== FP16 Arithmetic Tests ===\n\n");
    test_conversion();
    test_arith();
    printf("\n=== FP16 tests passed ===\n");
    return 0;
}
