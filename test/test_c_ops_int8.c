/*
 * Unit tests for nn_ops_int8.h (W8A8 linear + conv).
 * Compile from project root:
 *   gcc -O0 -g -I src/c_ops test/test_c_ops_int8.c -lm -o test/test_c_ops_int8 && ./test/test_c_ops_int8
 */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "nn_ops_float.h"
#include "nn_ops_int8.h"

#define TOL 1e-5f

/* -------------------------------------------------------------------------- */
/* Reference implementations (same math as kernels) for exact comparisons   */
/* -------------------------------------------------------------------------- */

static int8_t ref_dense_one_out(
    const int8_t* x,
    int in_features,
    const int8_t* W,
    const float* b,
    int out_features,
    int o,
    float input_scale,
    float weight_scale,
    float output_scale,
    int offset)
{
    int32_t acc = 0;
    for (int i = 0; i < in_features; ++i) {
        acc += (int32_t)x[i] * (int32_t)W[i * out_features + o];
    }
    float result = (float)acc * input_scale * weight_scale;
    if (b != NULL) {
        result += b[o];
    }
    return quantize_scalar_int8(result, output_scale, offset);
}

static void ref_conv2d_nhwc_int8(
    const int8_t* in, int in_h, int in_w, int in_c,
    const int8_t* filt, int k_h, int k_w, int out_c,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float input_scale,
    float weight_scale,
    float output_scale,
    int offset,
    int8_t* out)
{
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
    float combined_scale = input_scale * weight_scale;

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int oc = 0; oc < out_c; ++oc) {
                int32_t acc = 0;
                for (int kh = 0; kh < k_h; ++kh) {
                    int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) {
                        continue;
                    }
                    for (int kw = 0; kw < k_w; ++kw) {
                        int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) {
                            continue;
                        }
                        const int8_t* in_px = in + ((ih * in_w + iw) * in_c);
                        const int8_t* f_base =
                            filt + (((kh * k_w + kw) * in_c) * out_c + oc);
                        for (int ic = 0; ic < in_c; ++ic) {
                            acc += (int32_t)in_px[ic] * (int32_t)f_base[ic * out_c];
                        }
                    }
                }
                float result = (float)acc * combined_scale;
                if (bias != NULL) {
                    result += bias[oc];
                }
                out[((oh * out_w + ow) * out_c) + oc] =
                    quantize_scalar_int8(result, output_scale, offset);
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/* Quantization helpers                                                       */
/* -------------------------------------------------------------------------- */

static void test_quantize_scalar_int8(void) {
    assert(quantize_scalar_int8(0.0f, 0.1f, 0) == 0);
    assert(quantize_scalar_int8(0.15f, 0.1f, 0) == 2); /* round */
    assert(quantize_scalar_int8(-0.15f, 0.1f, 0) == -2);
    /* clamp */
    assert(quantize_scalar_int8(1000.0f, 0.1f, 0) == 127);
    assert(quantize_scalar_int8(-1000.0f, 0.1f, 0) == -128);
    /* nonzero offset: Q = round(x/s) + offset */
    assert(quantize_scalar_int8(0.0f, 1.0f, 5) == 5);
    printf("  test_quantize_scalar_int8 PASS\n");
}

static void test_dequantize_scalar_int8(void) {
    assert(fabsf(dequantize_scalar_int8(10, 0.1f, 0) - 1.0f) < TOL);
    assert(fabsf(dequantize_scalar_int8(10, 0.1f, 5) - 0.5f) < TOL); /* 0.1*(10-5) */
    printf("  test_dequantize_scalar_int8 PASS\n");
}

static void test_roundtrip_scalar(void) {
    float scale = 0.05f;
    int offset = 0;
    float x = 0.37f;
    int8_t q = quantize_scalar_int8(x, scale, offset);
    float y = dequantize_scalar_int8(q, scale, offset);
    assert(fabsf(y - x) <= 0.5f * scale + 1e-6f);
    printf("  test_roundtrip_scalar PASS\n");
}

static void test_compute_dynamic_scale_int8(void) {
    float z[] = {0.0f, 0.0f, 0.0f};
    float s0 = compute_dynamic_scale_int8(z, 3);
    assert(fabsf(s0 - (1.0f / 127.0f)) < TOL);

    float a[] = {-2.54f, 1.0f, 0.0f};
    float s1 = compute_dynamic_scale_int8(a, 3);
    assert(fabsf(s1 - (2.54f / 127.0f)) < TOL);

    float one[] = {3.81f};
    float s2 = compute_dynamic_scale_int8(one, 1);
    assert(fabsf(s2 - (3.81f / 127.0f)) < TOL);
    printf("  test_compute_dynamic_scale_int8 PASS\n");
}

static void test_quantize_dequantize_array(void) {
    float in_f[] = {-1.0f, 0.0f, 0.25f, 0.9f};
    int8_t q[4];
    float out_f[4];
    float s = 0.1f;
    int off = 0;
    quantize_float_to_int8(in_f, 4, s, off, q);
    dequantize_int8_to_float(q, 4, s, off, out_f);
    for (int i = 0; i < 4; ++i) {
        assert(fabsf(out_f[i] - in_f[i]) <= 0.5f * s + 1e-5f);
    }
    printf("  test_quantize_dequantize_array PASS\n");
}

/* -------------------------------------------------------------------------- */
/* dense_int8                                                                 */
/* -------------------------------------------------------------------------- */

static void test_dense_int8_identity(void) {
    /* 2 -> 2: W = I scaled into int8 with sw=0.5, x = [2,3] quantized with sx=0.5 */
    float sx = 0.5f;
    float sw = 0.5f;
    float so = 0.25f;
    int off = 0;
    float xf[] = {1.0f, 1.5f};
    float Wf[] = {1.0f, 0.0f, 0.0f, 1.0f};
    int8_t x[2];
    int8_t W[4];
    for (int i = 0; i < 2; ++i) {
        x[i] = quantize_scalar_int8(xf[i], sx, off);
    }
    for (int i = 0; i < 4; ++i) {
        W[i] = quantize_scalar_int8(Wf[i], sw, off);
    }
    float b[] = {0.0f, 0.0f};
    int8_t y[2];
    dense_int8(x, 2, W, b, 2, sx, sw, so, off, y);
    for (int o = 0; o < 2; ++o) {
        int8_t exp = ref_dense_one_out(x, 2, W, b, 2, o, sx, sw, so, off);
        assert(y[o] == exp);
    }
    printf("  test_dense_int8_identity PASS\n");
}

static void test_dense_int8_vs_float(void) {
    const int in_f = 4;
    const int out_f = 3;
    float sx = 0.08f;
    float sw = 0.06f;
    float so = 0.07f;
    int off = 0;

    float xf[] = {0.1f, -0.2f, 0.3f, 0.4f};
    float Wf[] = {
        0.5f, -0.1f, 0.0f,
        0.2f, 0.3f, -0.4f,
        -0.1f, 0.2f, 0.1f,
        0.0f, 0.1f, -0.2f
    };
    float b[] = {0.01f, -0.02f, 0.03f};

    int8_t x[in_f];
    int8_t W[in_f * out_f];
    for (int i = 0; i < in_f; ++i) {
        x[i] = quantize_scalar_int8(xf[i], sx, off);
    }
    for (int i = 0; i < in_f * out_f; ++i) {
        W[i] = quantize_scalar_int8(Wf[i], sw, off);
    }

    int8_t y_q[out_f];
    dense_int8(x, in_f, W, b, out_f, sx, sw, so, off, y_q);

    float y_float[out_f];
    dense(xf, in_f, Wf, b, out_f, y_float);

    float y_from_q[out_f];
    dequantize_int8_to_float(y_q, out_f, so, off, y_from_q);

    /* Loose bound: input/weight quant + output quant */
    float err_budget = 2.0f * so + (float)in_f * sx * sw * 4.0f;
    for (int o = 0; o < out_f; ++o) {
        assert(fabsf(y_from_q[o] - y_float[o]) < err_budget);
    }
    printf("  test_dense_int8_vs_float PASS\n");
}

static void test_dense_int8_no_bias(void) {
    int8_t x[] = {10, -5};
    int8_t W[] = {1, 2, 3, 4};
    int8_t y[2];
    float s = 0.1f;
    dense_int8(x, 2, W, NULL, 2, s, s, s, 0, y);
    assert(y[0] == ref_dense_one_out(x, 2, W, NULL, 2, 0, s, s, s, 0));
    assert(y[1] == ref_dense_one_out(x, 2, W, NULL, 2, 1, s, s, s, 0));
    printf("  test_dense_int8_no_bias PASS\n");
}

static void test_dense_int8_saturation(void) {
    int8_t x[8];
    int8_t W[8 * 2];
    for (int i = 0; i < 8; ++i) {
        x[i] = 127;
        W[i * 2 + 0] = 127;
        W[i * 2 + 1] = -127;
    }
    float sx = 0.01f;
    float sw = 0.01f;
    float so = 1.0f; /* large output scale -> small q values */
    int8_t y[2];
    dense_int8(x, 8, W, NULL, 2, sx, sw, so, 0, y);
    /* Large MAC; verify kernel matches reference (may or may not hit int8 clamp) */
    assert(y[0] == ref_dense_one_out(x, 8, W, NULL, 2, 0, sx, sw, so, 0));
    assert(y[1] == ref_dense_one_out(x, 8, W, NULL, 2, 1, sx, sw, so, 0));
    printf("  test_dense_int8_saturation PASS\n");
}

static void test_dense_int8_output_scale(void) {
    /* Pre-quant real result exactly 1.0 so two output scales both represent it exactly */
    int8_t x[] = {50};
    int8_t W[] = {1};
    float is = 0.02f;
    float ws = 1.0f;
    float os_a = 0.1f;
    float os_b = 0.25f;
    int8_t y1[1], y2[1];
    dense_int8(x, 1, W, NULL, 1, is, ws, os_a, 0, y1);
    dense_int8(x, 1, W, NULL, 1, is, ws, os_b, 0, y2);
    float f_a = dequantize_scalar_int8(y1[0], os_a, 0);
    float f_b = dequantize_scalar_int8(y2[0], os_b, 0);
    assert(fabsf(f_a - 1.0f) < 1e-5f && fabsf(f_b - 1.0f) < 1e-5f);
    printf("  test_dense_int8_output_scale PASS\n");
}

/* -------------------------------------------------------------------------- */
/* conv2d_nhwc_int8                                                           */
/* -------------------------------------------------------------------------- */

static void test_conv2d_int8_1x1(void) {
    int in_h = 2, in_w = 2, in_c = 2, out_c = 1;
    int k_h = 1, k_w = 1;
    /* NHWC row-major: 4 pixels * 2 ch */
    int8_t in[] = {
        1, 2,   /* (0,0) */
        3, 4,   /* (0,1) */
        5, 6,   /* (1,0) */
        7, 8    /* (1,1) */
    };
    /* filt [1,1,2,1]: weights for oc=0 */
    int8_t filt[] = {1, 1};
    float bias[] = {0.0f};
    float is = 1.0f;
    float ws = 1.0f;
    float os = 1.0f;
    int out_h = 2, out_w = 2;
    int8_t out[4];
    int8_t ref[4];
    conv2d_nhwc_int8(in, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
                     1, 1, 0, 0, is, ws, os, 0, out);
    ref_conv2d_nhwc_int8(in, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
                         1, 1, 0, 0, is, ws, os, 0, ref);
    assert(memcmp(out, ref, sizeof(out)) == 0);
    /* (0,0): 1+2=3 */
    assert(out[0] == quantize_scalar_int8(3.0f, os, 0));
    printf("  test_conv2d_int8_1x1 PASS\n");
}

static void test_conv2d_int8_3x3_pad1(void) {
    int in_h = 3, in_w = 3, in_c = 1, out_c = 1;
    int k_h = 3, k_w = 3;
    int8_t in[9];
    for (int i = 0; i < 9; ++i) {
        in[i] = (int8_t)(i + 1);
    }
    int8_t filt[9];
    for (int i = 0; i < 9; ++i) {
        filt[i] = (i == 4) ? 1 : 0; /* center tap only */
    }
    float bias[] = {0.0f};
    int8_t out[9];
    int8_t ref[9];
    conv2d_nhwc_int8(in, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
                     1, 1, 1, 1, 1.0f, 1.0f, 1.0f, 0, out);
    ref_conv2d_nhwc_int8(in, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
                         1, 1, 1, 1, 1.0f, 1.0f, 1.0f, 0, ref);
    assert(memcmp(out, ref, sizeof(out)) == 0);
    printf("  test_conv2d_int8_3x3_pad1 PASS\n");
}

static void test_conv2d_int8_vs_float(void) {
    int in_h = 2, in_w = 3, in_c = 2, out_c = 2;
    int k_h = 2, k_w = 2;
    float sx = 0.05f;
    float sw = 0.04f;
    float so = 0.06f;

    float in_f[2 * 3 * 2];
    for (int i = 0; i < 12; ++i) {
        in_f[i] = 0.1f * (float)(i - 5);
    }
    float filt_f[2 * 2 * 2 * 2];
    for (int i = 0; i < 16; ++i) {
        filt_f[i] = 0.08f * (float)(i % 5 - 2);
    }
    float bias[] = {0.02f, -0.01f};

    int8_t in_q[12];
    int8_t filt_q[16];
    quantize_float_to_int8(in_f, 12, sx, 0, in_q);
    quantize_float_to_int8(filt_f, 16, sw, 0, filt_q);

    int out_h = (in_h - k_h) / 1 + 1;
    int out_w = (in_w - k_w) / 1 + 1;
    int out_elems = out_h * out_w * out_c;
    int8_t out_q[16];
    float out_f[16];

    conv2d_nhwc_int8(in_q, in_h, in_w, in_c, filt_q, k_h, k_w, out_c, bias,
                     1, 1, 0, 0, sx, sw, so, 0, out_q);
    conv2d_nhwc(in_f, in_h, in_w, in_c, filt_f, k_h, k_w, out_c, bias,
                1, 1, 0, 0, out_f);

    float out_from_q[16];
    dequantize_int8_to_float(out_q, out_elems, so, 0, out_from_q);

    float err_budget = 3.0f * so + (float)(k_h * k_w * in_c) * sx * sw * 6.0f;
    for (int i = 0; i < out_elems; ++i) {
        assert(fabsf(out_from_q[i] - out_f[i]) < err_budget);
    }
    printf("  test_conv2d_int8_vs_float PASS\n");
}

static void test_conv2d_int8_no_bias(void) {
    int8_t in[] = {2, 0, 0, 2};
    int8_t filt[] = {1, 1};
    int8_t out[2];
    int8_t ref[2];
    conv2d_nhwc_int8(in, 1, 2, 2, filt, 1, 1, 1, NULL, 1, 1, 0, 0,
                     1.0f, 1.0f, 1.0f, 0, out);
    ref_conv2d_nhwc_int8(in, 1, 2, 2, filt, 1, 1, 1, NULL, 1, 1, 0, 0,
                         1.0f, 1.0f, 1.0f, 0, ref);
    assert(memcmp(out, ref, sizeof(out)) == 0);
    printf("  test_conv2d_int8_no_bias PASS\n");
}

static void test_conv2d_int8_stride2(void) {
    int in_h = 4, in_w = 4, in_c = 1, out_c = 1;
    int k_h = 2, k_w = 2;
    int8_t in[16];
    for (int i = 0; i < 16; ++i) {
        in[i] = 1;
    }
    int8_t filt[4] = {1, 1, 1, 1};
    int out_h = (in_h - k_h) / 2 + 1;
    int out_w = (in_w - k_w) / 2 + 1;
    int n = out_h * out_w;
    int8_t out[16];
    int8_t ref[16];
    conv2d_nhwc_int8(in, in_h, in_w, in_c, filt, k_h, k_w, out_c, NULL,
                     2, 2, 0, 0, 1.0f, 1.0f, 1.0f, 0, out);
    ref_conv2d_nhwc_int8(in, in_h, in_w, in_c, filt, k_h, k_w, out_c, NULL,
                         2, 2, 0, 0, 1.0f, 1.0f, 1.0f, 0, ref);
    assert(memcmp(out, ref, (size_t)n * sizeof(int8_t)) == 0);
    /* each 2x2 sum of ones = 4 */
    for (int i = 0; i < n; ++i) {
        assert(out[i] == 4);
    }
    printf("  test_conv2d_int8_stride2 PASS\n");
}

static void test_conv2d_int8_output_scale(void) {
    int8_t in[] = {10, 0, 0, 10};
    int8_t filt[] = {1, 0};
    float is = 0.1f;
    float ws = 0.2f;
    int8_t o1[1], o2[1];
    conv2d_nhwc_int8(in, 1, 2, 2, filt, 1, 1, 1, NULL, 1, 1, 0, 0,
                     is, ws, 0.05f, 0, o1);
    conv2d_nhwc_int8(in, 1, 2, 2, filt, 1, 1, 1, NULL, 1, 1, 0, 0,
                     is, ws, 0.5f, 0, o2);
    float r1 = dequantize_scalar_int8(o1[0], 0.05f, 0);
    float r2 = dequantize_scalar_int8(o2[0], 0.5f, 0);
    assert(fabsf(r1 - r2) < 0.3f);
    printf("  test_conv2d_int8_output_scale PASS\n");
}

static void test_relu_int8(void) {
    int8_t x[] = {-5, 0, 3, -128};
    relu_int8(x, 4);
    assert(x[0] == 0 && x[1] == 0 && x[2] == 3 && x[3] == 0);
    printf("  test_relu_int8 PASS\n");
}

int main(void) {
    printf("Running nn_ops_int8 tests...\n");
    test_quantize_scalar_int8();
    test_dequantize_scalar_int8();
    test_roundtrip_scalar();
    test_compute_dynamic_scale_int8();
    test_quantize_dequantize_array();
    test_dense_int8_identity();
    test_dense_int8_vs_float();
    test_dense_int8_no_bias();
    test_dense_int8_saturation();
    test_dense_int8_output_scale();
    test_conv2d_int8_1x1();
    test_conv2d_int8_3x3_pad1();
    test_conv2d_int8_vs_float();
    test_conv2d_int8_no_bias();
    test_conv2d_int8_stride2();
    test_conv2d_int8_output_scale();
    test_relu_int8();
    printf("All nn_ops_int8 tests PASS.\n");
    return 0;
}
