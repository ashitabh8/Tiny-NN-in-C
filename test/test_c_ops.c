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

static void test_conv2d_nhwc_1x1(void) {
    /* in_h=2, in_w=2, in_c=2, out_c=1, kernel=1x1 */
    float in[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
    };
    float filt[] = {1.0f, -1.0f}; /* output = c0 - c1 */
    float bias[] = {0.5f};
    float out[4];
    conv2d_nhwc(in, 2, 2, 2, filt, 1, 1, 1, bias, 1, 1, 0, 0, out);
    assert(fabsf(out[0] - (-0.5f)) < TOL); /* 1-2+0.5 */
    assert(fabsf(out[1] - (-0.5f)) < TOL); /* 3-4+0.5 */
    assert(fabsf(out[2] - (-0.5f)) < TOL); /* 5-6+0.5 */
    assert(fabsf(out[3] - (-0.5f)) < TOL); /* 7-8+0.5 */
    printf("  test_conv2d_nhwc_1x1 PASS\n");
}

static void test_depthwise_conv2d_nhwc_identity(void) {
    /* 1x1 depthwise with filt=[1,1] should copy input (+bias) per channel */
    float in[] = {
        1.0f, 2.0f,
        3.0f, 4.0f
    }; /* h=1,w=2,c=2 */
    float filt[] = {1.0f, 1.0f}; /* k_h=1,k_w=1,c=2 */
    float bias[] = {0.0f, 0.5f};
    float out[4];
    depthwise_conv2d_nhwc(in, 1, 2, 2, filt, 1, 1, bias, 1, 1, 0, 0, out);
    assert(fabsf(out[0] - 1.0f) < TOL);
    assert(fabsf(out[1] - 2.5f) < TOL);
    assert(fabsf(out[2] - 3.0f) < TOL);
    assert(fabsf(out[3] - 4.5f) < TOL);
    printf("  test_depthwise_conv2d_nhwc_identity PASS\n");
}

static void test_permute_4d_nchw_to_nhwc(void) {
    /* NCHW [1,2,2,2] -> NHWC [1,2,2,2] with perm [0,2,3,1] */
    float in[8] = {
        /* c0 */
        1.0f, 2.0f, 3.0f, 4.0f,
        /* c1 */
        10.0f, 20.0f, 30.0f, 40.0f
    };
    float out[8];
    permute_4d(in, 1, 2, 2, 2, 0, 2, 3, 1, out);
    /* Expected NHWC: (h,w,c): [1,10], [2,20], [3,30], [4,40] */
    float exp[8] = {1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f, 4.0f, 40.0f};
    for (int i = 0; i < 8; ++i) {
        assert(fabsf(out[i] - exp[i]) < TOL);
    }
    printf("  test_permute_4d_nchw_to_nhwc PASS\n");
}

static void test_permute_3d_swap_last_two(void) {
    /* [d0=2, d1=2, d2=3] permuted with [0, 2, 1] -> [d0=2, d2=3, d1=2] */
    float in[12] = {
        /* d0=0 */
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        /* d0=1 */
        10.0f, 20.0f, 30.0f,
        40.0f, 50.0f, 60.0f
    };
    float out[12];
    permute_3d(in, 2, 2, 3, 0, 2, 1, out);
    /* For d0=0: in[d1, d2] -> out[d2, d1]
       out[0,0,0]=in[0,0,0]=1, out[0,0,1]=in[0,1,0]=4
       out[0,1,0]=in[0,0,1]=2, out[0,1,1]=in[0,1,1]=5
       out[0,2,0]=in[0,0,2]=3, out[0,2,1]=in[0,1,2]=6 */
    float exp[12] = {
        1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f,
        10.0f, 40.0f, 20.0f, 50.0f, 30.0f, 60.0f
    };
    for (int i = 0; i < 12; ++i) {
        assert(fabsf(out[i] - exp[i]) < TOL);
    }
    printf("  test_permute_3d_swap_last_two PASS\n");
}

static void test_permute_3d_identity(void) {
    /* perm [0,1,2] is identity */
    float in[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float out[6];
    permute_3d(in, 1, 2, 3, 0, 1, 2, out);
    for (int i = 0; i < 6; ++i) {
        assert(fabsf(out[i] - in[i]) < TOL);
    }
    printf("  test_permute_3d_identity PASS\n");
}

static void test_softmax(void) {
    float x[] = {1.0f, 2.0f, 3.0f};
    softmax(x, 3);
    float sum = x[0] + x[1] + x[2];
    assert(fabsf(sum - 1.0f) < 1e-6f);
    assert(x[2] > x[1] && x[1] > x[0]); /* monotonic with logits */
    printf("  test_softmax PASS\n");
}

static void test_mean_helpers(void) {
    float in_hwc[] = {
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
    }; /* h=2,w=2,c=2 */
    float out_a[2], out_b[2];
    global_average_pool_2d(in_hwc, 2, 2, 2, out_a);
    mean_hwc(in_hwc, 2, 2, 2, out_b);
    assert(fabsf(out_a[0] - out_b[0]) < TOL);
    assert(fabsf(out_a[1] - out_b[1]) < TOL);

    float mat[] = {2.0f, 4.0f, 6.0f, -3.0f, 0.0f, 3.0f}; /* rows=2, cols=3 */
    float row_mean[2];
    mean_last_dim(mat, 2, 3, row_mean);
    assert(fabsf(row_mean[0] - 4.0f) < TOL);
    assert(fabsf(row_mean[1] - 0.0f) < TOL);
    printf("  test_mean_helpers PASS\n");
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
    test_conv2d_nhwc_1x1();
    test_depthwise_conv2d_nhwc_identity();
    test_permute_4d_nchw_to_nhwc();
    test_permute_3d_swap_last_two();
    test_permute_3d_identity();
    test_softmax();
    test_mean_helpers();
    test_global_average_pool_2d();
    test_adaptive_avg_pool_1x1();
    test_flatten();
    test_dense();
    test_batchnorm2d_nhwc();
    printf("All tests PASS.\n");
    return 0;
}
