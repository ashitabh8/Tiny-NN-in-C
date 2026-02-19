/*
 * Standalone unit tests for nn_ops_float.h.
 * Compile from project root: gcc -O0 -g -I src/c_ops test/test_c_ops.c -lm -o test/test_c_ops && ./test/test_c_ops
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "nn_ops_float.h"

#define TOL 1e-5f

static void test_relu(void) {
    float x[] = {1.0f, -1.0f, 0.0f, 0.5f};
    float y[4];
    memcpy(y, x, sizeof(x));
    relu(y, 4);
    assert(y[0] == 1.0f);
    assert(y[1] == 0.0f);
    assert(y[2] == 0.0f);
    assert(y[3] == 0.5f);
    printf("  test_relu PASS\n");
}

static void test_global_average_pool_2d(void) {
    /* NHWC: 2x2 spatial, 3 channels. Values so means are 2.5, 6, 1. */
    float in[] = {
        1.0f, 0.0f, 1.0f,   /* (0,0) */
        2.0f, 4.0f, 1.0f,   /* (0,1) */
        3.0f, 8.0f, 1.0f,   /* (1,0) */
        4.0f, 12.0f, 1.0f   /* (1,1) */
    };
    float out[3];
    global_average_pool_2d(in, 2, 2, 3, out);
    assert(fabsf(out[0] - 2.5f) < TOL);
    assert(fabsf(out[1] - 6.0f) < TOL);
    assert(fabsf(out[2] - 1.0f) < TOL);
    printf("  test_global_average_pool_2d PASS\n");
}

static void test_adaptive_avg_pool_1x1(void) {
    float in[] = {
        1.0f, 0.0f, 1.0f,
        2.0f, 4.0f, 1.0f,
        3.0f, 8.0f, 1.0f,
        4.0f, 12.0f, 1.0f
    };
    float out_global[3], out_adaptive[3];
    global_average_pool_2d(in, 2, 2, 3, out_global);
    adaptive_avg_pool_2d_1x1(in, 2, 2, 3, out_adaptive);
    for (int i = 0; i < 3; ++i)
        assert(fabsf(out_adaptive[i] - out_global[i]) < TOL);
    printf("  test_adaptive_avg_pool_1x1 PASS\n");
}

static void test_flatten(void) {
    float src[12];
    for (int i = 0; i < 12; ++i) src[i] = (float)(i + 1);
    float dst[12];
    flatten(src, 12, dst);
    for (int i = 0; i < 12; ++i)
        assert(dst[i] == src[i]);
    printf("  test_flatten PASS\n");
}

static void test_dense(void) {
    /* 2 in_features, 3 out_features. W row-major [2,3]: rows are input dim. */
    float x[] = {1.0f, 2.0f};
    float W[] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    float b[] = {0.0f, 0.0f, 1.0f};
    float y[3];
    dense(x, 2, W, b, 3, y);
    assert(fabsf(y[0] - 1.0f) < TOL);
    assert(fabsf(y[1] - 2.0f) < TOL);
    assert(fabsf(y[2] - 1.0f) < TOL);
    printf("  test_dense PASS\n");
}

static void test_batchnorm2d_nhwc(void) {
    /* 1x1 spatial, 2 channels. in=[10, 20], mean=0, var=1, gamma=1, beta=0 */
    float in[] = {10.0f, 20.0f};
    float gamma[] = {1.0f, 1.0f};
    float beta[] = {0.0f, 0.0f};
    float mean[] = {0.0f, 0.0f};
    float var[] = {1.0f, 1.0f};
    float out[2];
    batchnorm2d_nhwc(in, 1, 1, 2, gamma, beta, mean, var, 1e-5f, out);
    assert(fabsf(out[0] - 10.0f) < 1e-4f);
    assert(fabsf(out[1] - 20.0f) < 1e-4f);
    printf("  test_batchnorm2d_nhwc PASS\n");
}

int main(void) {
    printf("Running C ops tests...\n");
    test_relu();
    test_global_average_pool_2d();
    test_adaptive_avg_pool_1x1();
    test_flatten();
    test_dense();
    test_batchnorm2d_nhwc();
    printf("All tests PASS.\n");
    return 0;
}
