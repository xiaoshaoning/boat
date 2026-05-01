// test_data_pipeline.c - Dataset, DataLoader, and Transform tests
#include <boat.h>
#include <boat/data.h>
#include <boat/tensor.h>
#include <boat/memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static int tests_passed = 0;
static int tests_total = 0;

#define TEST(name) do { printf("  %s ... ", name); fflush(stdout); tests_total++; } while(0)
#define PASS() do { printf("PASS\n"); fflush(stdout); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); fflush(stdout); return 1; } while(0)
#define ASSERT(cond) do { if (!(cond)) { printf("FAIL at %d\n", __LINE__); fflush(stdout); return 1; } } while(0)

// --- Test 1: Tensor dataset creation ---
static int test_tensor_dataset_create(void) {
    TEST("Tensor dataset create");
    int64_t dshape[] = {10, 3, 4};
    int64_t lshape[] = {10};
    boat_tensor_t* data = boat_tensor_create(dshape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* labels = boat_tensor_create(lshape, 1, BOAT_DTYPE_INT64, BOAT_DEVICE_CPU);

    float* dp = (float*)boat_tensor_data(data);
    for (size_t i = 0; i < 10 * 3 * 4; i++) dp[i] = (float)i;
    int64_t* lp = (int64_t*)boat_tensor_data(labels);
    for (int64_t i = 0; i < 10; i++) lp[i] = i % 3;

    boat_dataset_t* ds = boat_tensor_dataset_create(data, labels);
    ASSERT(ds != NULL);
    ASSERT(boat_dataset_size(ds) == 10);

    boat_tensor_unref(data);
    boat_tensor_unref(labels);
    ASSERT(boat_dataset_size(ds) == 10);

    boat_dataset_free(ds);
    PASS();
    return 0;
}

// --- Test 2: Dataset get_item ---
static int test_tensor_dataset_get_item(void) {
    TEST("Tensor dataset get_item");
    int64_t dshape[] = {5, 2, 3};
    boat_tensor_t* data = boat_tensor_create(dshape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    int64_t lshape[] = {5};
    boat_tensor_t* labels = boat_tensor_create(lshape, 1, BOAT_DTYPE_INT64, BOAT_DEVICE_CPU);

    float* dp = (float*)boat_tensor_data(data);
    for (size_t i = 0; i < 5; i++)
        for (size_t j = 0; j < 2 * 3; j++)
            dp[i * 6 + j] = (float)(i * 100 + j);
    int64_t* lp = (int64_t*)boat_tensor_data(labels);
    for (int64_t i = 0; i < 5; i++) lp[i] = i * 10;

    boat_dataset_t* ds = boat_tensor_dataset_create(data, labels);
    boat_tensor_unref(data);
    boat_tensor_unref(labels);
    ASSERT(ds != NULL);

    boat_tensor_t* s2 = boat_dataset_get_data(ds, 2);
    ASSERT(s2 != NULL);
    ASSERT(boat_tensor_ndim(s2) == 2);
    ASSERT(boat_tensor_shape(s2)[0] == 2);
    ASSERT(boat_tensor_shape(s2)[1] == 3);
    const float* sd = (const float*)boat_tensor_const_data(s2);
    for (size_t j = 0; j < 6; j++) ASSERT(sd[j] == (float)(200 + j));
    boat_tensor_unref(s2);

    boat_tensor_t* l2 = boat_dataset_get_label(ds, 2);
    ASSERT(l2 != NULL);
    ASSERT(boat_tensor_nelements(l2) == 1);
    const int64_t* ll = (const int64_t*)boat_tensor_const_data(l2);
    ASSERT(ll[0] == 20);
    boat_tensor_unref(l2);

    boat_tensor_t* s4 = boat_dataset_get_data(ds, 4);
    ASSERT(s4 != NULL);
    sd = (const float*)boat_tensor_const_data(s4);
    for (size_t j = 0; j < 6; j++) ASSERT(sd[j] == (float)(400 + j));
    boat_tensor_unref(s4);

    boat_dataset_free(ds);
    PASS();
    return 0;
}

// --- Test 3: Dataset NULL/invalid handling ---
static int test_dataset_errors(void) {
    TEST("Dataset NULL handling");
    ASSERT(boat_tensor_dataset_create(NULL, NULL) == NULL);
    ASSERT(boat_dataset_size(NULL) == 0);
    ASSERT(boat_dataset_get_data(NULL, 0) == NULL);
    ASSERT(boat_dataset_get_label(NULL, 0) == NULL);
    boat_dataset_free(NULL);

    int64_t dshape[] = {10, 3};
    int64_t lshape[] = {5};
    boat_tensor_t* d = boat_tensor_create(dshape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* l = boat_tensor_create(lshape, 1, BOAT_DTYPE_INT64, BOAT_DEVICE_CPU);
    ASSERT(boat_tensor_dataset_create(d, l) == NULL);
    boat_tensor_unref(d);
    boat_tensor_unref(l);

    PASS();
    return 0;
}

// --- Test 4: DataLoader basic iteration ---
static int test_dataloader_basic(void) {
    TEST("DataLoader basic iteration");
    int64_t dshape[] = {20, 3, 4};
    boat_tensor_t* data = boat_tensor_create(dshape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    int64_t lshape[] = {20};
    boat_tensor_t* labels = boat_tensor_create(lshape, 1, BOAT_DTYPE_INT64, BOAT_DEVICE_CPU);

    float* dp = (float*)boat_tensor_data(data);
    for (int i = 0; i < 20 * 12; i++) dp[i] = (float)i;
    int64_t* lp = (int64_t*)boat_tensor_data(labels);
    for (int i = 0; i < 20; i++) lp[i] = i % 5;

    boat_dataset_t* ds = boat_tensor_dataset_create(data, labels);
    boat_tensor_unref(data);
    boat_tensor_unref(labels);

    boat_dataloader_t* loader = boat_dataloader_create(ds, 6, false);
    ASSERT(loader != NULL);
    ASSERT(boat_dataloader_num_batches(loader) == 4);
    ASSERT(boat_dataloader_batch_size(loader) == 6);
    ASSERT(boat_dataloader_current_batch_idx(loader) == 0);

    int batch_count = 0;
    boat_tensor_t* batch_data = NULL;
    boat_tensor_t* batch_labels = NULL;
    while (boat_dataloader_next(loader, &batch_data, &batch_labels)) {
        ASSERT(batch_data != NULL);
        ASSERT(batch_labels != NULL);
        size_t bs = (size_t)boat_tensor_shape(batch_data)[0];
        if (batch_count < 3) ASSERT(bs == 6);
        else ASSERT(bs == 2);
        ASSERT(boat_tensor_shape(batch_labels)[0] == (int64_t)bs);
        ASSERT(boat_tensor_dtype(batch_labels) == BOAT_DTYPE_INT64);
        boat_tensor_unref(batch_data);
        boat_tensor_unref(batch_labels);
        batch_count++;
    }
    ASSERT(batch_count == 4);
    ASSERT(boat_dataloader_next(loader, &batch_data, &batch_labels) == false);

    boat_dataloader_reset(loader);
    ASSERT(boat_dataloader_current_batch_idx(loader) == 0);
    batch_count = 0;
    while (boat_dataloader_next(loader, &batch_data, &batch_labels)) {
        boat_tensor_unref(batch_data);
        boat_tensor_unref(batch_labels);
        batch_count++;
    }
    ASSERT(batch_count == 4);

    boat_dataloader_free(loader);
    boat_dataset_free(ds);
    PASS();
    return 0;
}

// --- Test 5: DataLoader shuffle ---
static int test_dataloader_shuffle(void) {
    TEST("DataLoader shuffle");
    int64_t dshape[] = {10, 1};
    boat_tensor_t* data = boat_tensor_create(dshape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    int64_t lshape[] = {10};
    boat_tensor_t* labels = boat_tensor_create(lshape, 1, BOAT_DTYPE_INT64, BOAT_DEVICE_CPU);

    float* dp = (float*)boat_tensor_data(data);
    for (int i = 0; i < 10; i++) dp[i] = (float)i;
    int64_t* lp = (int64_t*)boat_tensor_data(labels);
    for (int i = 0; i < 10; i++) lp[i] = i;

    boat_dataset_t* ds = boat_tensor_dataset_create(data, labels);
    boat_tensor_unref(data);
    boat_tensor_unref(labels);

    boat_dataloader_t* loader = boat_dataloader_create(ds, 10, true);
    ASSERT(loader != NULL);

    boat_tensor_t* batch_data;
    boat_tensor_t* batch_labels;
    bool ok = boat_dataloader_next(loader, &batch_data, &batch_labels);
    ASSERT(ok);

    const int64_t* ll = (const int64_t*)boat_tensor_const_data(batch_labels);
    int in_order = 1;
    for (int i = 0; i < 10; i++) {
        if (ll[i] != i) { in_order = 0; break; }
    }
    if (in_order) {
        boat_tensor_unref(batch_data);
        boat_tensor_unref(batch_labels);
        boat_dataloader_reset(loader);
        boat_dataloader_next(loader, &batch_data, &batch_labels);
        ll = (const int64_t*)boat_tensor_const_data(batch_labels);
        in_order = 1;
        for (int i = 0; i < 10; i++) {
            if (ll[i] != i) { in_order = 0; break; }
        }
    }
    ASSERT(in_order == 0);

    boat_tensor_unref(batch_data);
    boat_tensor_unref(batch_labels);
    boat_dataloader_free(loader);
    boat_dataset_free(ds);
    PASS();
    return 0;
}

// --- Test 6: Normalize transform ---
static int test_transform_normalize(void) {
    TEST("Transform normalize");
    int64_t shape[] = {2, 3};
    boat_tensor_t* t = boat_tensor_create(shape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* d = (float*)boat_tensor_data(t);
    d[0] = 1.0f; d[1] = 2.0f; d[2] = 3.0f;
    d[3] = 4.0f; d[4] = 5.0f; d[5] = 6.0f;

    float params[] = {2.0f, 2.0f};
    boat_tensor_t* result = boat_transform_normalize(t, params);
    ASSERT(result == t);
    ASSERT(fabsf(d[0] - (-0.5f)) < 1e-6f);
    ASSERT(fabsf(d[1] - 0.0f) < 1e-6f);
    ASSERT(fabsf(d[2] - 0.5f) < 1e-6f);
    ASSERT(fabsf(d[3] - 1.0f) < 1e-6f);
    ASSERT(fabsf(d[4] - 1.5f) < 1e-6f);
    ASSERT(fabsf(d[5] - 2.0f) < 1e-6f);

    boat_tensor_unref(t);
    PASS();
    return 0;
}

// --- Test 7: Random horizontal flip ---
static int test_transform_hflip(void) {
    TEST("Transform random hflip");
    int64_t shape[] = {1, 1, 5};
    boat_tensor_t* t = boat_tensor_create(shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* d = (float*)boat_tensor_data(t);
    d[0] = 1.0f; d[1] = 2.0f; d[2] = 3.0f; d[3] = 4.0f; d[4] = 5.0f;

    int saw_flip = 0;
    for (int trial = 0; trial < 16; trial++) {
        d[0] = 1.0f; d[1] = 2.0f; d[2] = 3.0f; d[3] = 4.0f; d[4] = 5.0f;
        boat_tensor_t* r = boat_transform_random_hflip(t, NULL);
        ASSERT(r == t);
        if (d[0] == 5.0f && d[4] == 1.0f) { saw_flip = 1; break; }
    }
    ASSERT(saw_flip);

    int saw_noflip = 0;
    for (int trial = 0; trial < 16; trial++) {
        d[0] = 1.0f; d[1] = 2.0f; d[2] = 3.0f; d[3] = 4.0f; d[4] = 5.0f;
        boat_tensor_t* r = boat_transform_random_hflip(t, NULL);
        ASSERT(r == t);
        if (d[0] == 1.0f && d[4] == 5.0f) { saw_noflip = 1; break; }
    }
    ASSERT(saw_noflip);

    boat_tensor_unref(t);
    PASS();
    return 0;
}

// --- Test 8: Random crop ---
static int test_transform_crop(void) {
    TEST("Transform random crop");
    int64_t shape[] = {1, 4, 4};
    boat_tensor_t* t = boat_tensor_create(shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    float* d = (float*)boat_tensor_data(t);
    for (int i = 0; i < 16; i++) d[i] = (float)i;

    size_t crop_params[] = {2, 2};
    boat_tensor_t* result = boat_transform_random_crop(t, crop_params);
    ASSERT(result != NULL);
    ASSERT(boat_tensor_shape(result)[1] == 2);
    ASSERT(boat_tensor_shape(result)[2] == 2);
    boat_tensor_unref(t);
    if (result != t) boat_tensor_unref(result);
    PASS();
    return 0;
}

// --- Test 9: DataLoader with transform ---
static int test_dataloader_transform(void) {
    TEST("DataLoader with transform");
    int64_t dshape[] = {6, 1, 4};
    boat_tensor_t* data = boat_tensor_create(dshape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    int64_t lshape[] = {6};
    boat_tensor_t* labels = boat_tensor_create(lshape, 1, BOAT_DTYPE_INT64, BOAT_DEVICE_CPU);

    float* dp = (float*)boat_tensor_data(data);
    for (int i = 0; i < 6 * 4; i++) dp[i] = (float)(i % 10);
    int64_t* lp = (int64_t*)boat_tensor_data(labels);
    for (int i = 0; i < 6; i++) lp[i] = i;

    boat_dataset_t* ds = boat_tensor_dataset_create(data, labels);
    boat_tensor_unref(data);
    boat_tensor_unref(labels);

    boat_dataloader_t* loader = boat_dataloader_create(ds, 3, false);

    float params[] = {0.0f, 1.0f};
    boat_transform_chain_t* chain = boat_transform_chain_create();
    boat_transform_chain_add(chain, boat_transform_normalize, params);
    boat_dataloader_set_transform(loader, chain);

    boat_tensor_t* batch_data;
    boat_tensor_t* batch_labels;
    bool ok = boat_dataloader_next(loader, &batch_data, &batch_labels);
    ASSERT(ok);
    ASSERT(boat_tensor_shape(batch_data)[0] == 3);
    const float* bd = (const float*)boat_tensor_const_data(batch_data);
    const float* orig = (const float*)boat_tensor_const_data(data);
    for (int i = 0; i < 3 * 4; i++) ASSERT(bd[i] == orig[i]);

    boat_tensor_unref(batch_data);
    boat_tensor_unref(batch_labels);
    boat_dataloader_free(loader);
    boat_transform_chain_free(chain);
    boat_dataset_free(ds);
    PASS();
    return 0;
}

// --- Test 10: Transform chain ---
static int test_transform_chain(void) {
    TEST("Transform chain");
    int64_t shape[] = {1, 6, 8};
    boat_tensor_t* t = boat_tensor_create(shape, 3, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);

    boat_transform_chain_t* chain = boat_transform_chain_create();
    ASSERT(chain != NULL);
    float norm_params[] = {0.0f, 2.0f};
    boat_transform_chain_add(chain, boat_transform_normalize, norm_params);

    boat_tensor_t* result = boat_transform_chain_apply(chain, t);
    ASSERT(result != NULL);
    const float* rd = (const float*)boat_tensor_const_data(result);
    ASSERT(rd[0] == 0.0f);

    boat_tensor_unref(t);
    if (result != t) boat_tensor_unref(result);
    boat_transform_chain_free(chain);
    PASS();
    return 0;
}

// --- Test 11: Dataset get_label dtype handling ---
static int test_dataset_label_dtypes(void) {
    TEST("Dataset label dtypes");
    int64_t dshape[] = {3, 2};

    // UINT8 labels
    boat_tensor_t* data = boat_tensor_create(dshape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* labels = boat_tensor_create((int64_t[]){3}, 1, BOAT_DTYPE_UINT8, BOAT_DEVICE_CPU);
    ((uint8_t*)boat_tensor_data(labels))[0] = 7;
    ((uint8_t*)boat_tensor_data(labels))[1] = 8;
    ((uint8_t*)boat_tensor_data(labels))[2] = 9;

    boat_dataset_t* ds = boat_tensor_dataset_create(data, labels);
    boat_tensor_unref(data);
    boat_tensor_unref(labels);
    boat_tensor_t* l0 = boat_dataset_get_label(ds, 1);
    ASSERT(l0 != NULL);
    ASSERT(((const int64_t*)boat_tensor_const_data(l0))[0] == 8);
    boat_tensor_unref(l0);
    boat_dataset_free(ds);

    // INT32 labels
    data = boat_tensor_create(dshape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    labels = boat_tensor_create((int64_t[]){3}, 1, BOAT_DTYPE_INT32, BOAT_DEVICE_CPU);
    ((int32_t*)boat_tensor_data(labels))[0] = 42;
    ((int32_t*)boat_tensor_data(labels))[1] = 99;
    ((int32_t*)boat_tensor_data(labels))[2] = -1;

    ds = boat_tensor_dataset_create(data, labels);
    boat_tensor_unref(data);
    boat_tensor_unref(labels);
    l0 = boat_dataset_get_label(ds, 2);
    ASSERT(l0 != NULL);
    ASSERT(((const int64_t*)boat_tensor_const_data(l0))[0] == -1);
    boat_tensor_unref(l0);
    boat_dataset_free(ds);

    PASS();
    return 0;
}

// --- Test 12: Dataloader NULL / edge cases ---
static int test_dataloader_edges(void) {
    TEST("DataLoader edge cases");
    ASSERT(boat_dataloader_create(NULL, 4, false) == NULL);
    ASSERT(boat_dataloader_num_batches(NULL) == 0);
    ASSERT(boat_dataloader_batch_size(NULL) == 0);
    ASSERT(boat_dataloader_current_batch_idx(NULL) == 0);
    boat_dataloader_free(NULL);
    boat_dataloader_reset(NULL);

    boat_tensor_t* bd = (void*)0x1;  // non-NULL sentinel
    boat_tensor_t* bl = (void*)0x1;
    ASSERT(boat_dataloader_next(NULL, &bd, &bl) == false);
    ASSERT(bd == (void*)0x1);  // unchanged
    ASSERT(bl == (void*)0x1);

    int64_t dshape[] = {1, 2};
    boat_tensor_t* data = boat_tensor_create(dshape, 2, BOAT_DTYPE_FLOAT32, BOAT_DEVICE_CPU);
    boat_tensor_t* labels = boat_tensor_create((int64_t[]){1}, 1, BOAT_DTYPE_INT64, BOAT_DEVICE_CPU);
    boat_dataset_t* ds = boat_tensor_dataset_create(data, labels);
    boat_tensor_unref(data);
    boat_tensor_unref(labels);

    // batch_size=0 should be treated as 1
    boat_dataloader_t* loader = boat_dataloader_create(ds, 0, false);
    ASSERT(loader != NULL);
    ASSERT(boat_dataloader_batch_size(loader) == 1);
    ASSERT(boat_dataloader_num_batches(loader) == 1);
    boat_dataloader_free(loader);
    boat_dataset_free(ds);

    PASS();
    return 0;
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("Data pipeline tests:\n");

    if (test_tensor_dataset_create()) return 1;
    if (test_tensor_dataset_get_item()) return 1;
    if (test_dataset_errors()) return 1;
    if (test_dataloader_basic()) return 1;
    if (test_dataloader_shuffle()) return 1;
    if (test_transform_normalize()) return 1;
    if (test_transform_hflip()) return 1;
    if (test_transform_crop()) return 1;
    if (test_dataloader_transform()) return 1;
    if (test_transform_chain()) return 1;
    if (test_dataset_label_dtypes()) return 1;
    if (test_dataloader_edges()) return 1;

    printf("\n%d/%d tests passed!\n", tests_passed, tests_total);
    return tests_passed == tests_total ? 0 : 1;
}
