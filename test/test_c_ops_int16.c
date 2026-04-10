/*
 * Unit tests for nn_ops_int16.h.
 * Compile from project root:
 *   gcc -O0 -g -I src/c_ops test/test_c_ops_int16.c -lm -o test/test_c_ops_int16 && ./test/test_c_ops_int16
 */
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "nn_ops_float.h"
#include "nn_ops_int16.h"

#define TOL 1e-5f

static int16_t ref_dense_one_out_int16(
    const int16_t* x,
    int in_features,
    const int16_t* W,
    const float* b,
    int out_features,
    int o,
    float input_scale,
    float weight_scale,
    float output_scale,
    int input_zp,
    int weight_zp,
    int output_zp)
{
    int64_t sum_qx = 0;
    for (int i = 0; i < in_features; ++i) {
        sum_qx += (int64_t)x[i];
    }
    int64_t zx = (int64_t)input_zp;
    int64_t zw = (int64_t)weight_zp;
    int64_t zp_term = zx * zw * (int64_t)in_features;

    int64_t acc = 0;
    int64_t sum_qw = 0;
    for (int i = 0; i < in_features; ++i) {
        int64_t wv = (int64_t)W[i * out_features + o];
        acc += (int64_t)x[i] * wv;
        sum_qw += wv;
    }
    int64_t dot_affine = acc - zw * sum_qx - zx * sum_qw + zp_term;
    float result = (float)dot_affine * input_scale * weight_scale;
    if (b != NULL) {
        result += b[o];
    }
    return quantize_float_to_int16_scalar(result, output_scale, output_zp);
}

static void ref_conv2d_nhwc_int16(
    const int16_t* in, int in_h, int in_w, int in_c,
    const int16_t* filt, int k_h, int k_w, int out_c,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float input_scale,
    float weight_scale,
    float output_scale,
    int input_zp,
    int weight_zp,
    int output_zp,
    int16_t* out)
{
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
    float combined_scale = input_scale * weight_scale;
    int64_t zx = (int64_t)input_zp;
    int64_t zw = (int64_t)weight_zp;

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int oc = 0; oc < out_c; ++oc) {
                int64_t acc = 0;
                int64_t sum_qx = 0;
                int64_t sum_qf = 0;
                int64_t p = 0;
                for (int kh = 0; kh < k_h; ++kh) {
                    int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kw = 0; kw < k_w; ++kw) {
                        int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) continue;
                        const int16_t* in_px = in + ((ih * in_w + iw) * in_c);
                        const int16_t* f_base =
                            filt + (((kh * k_w + kw) * in_c) * out_c + oc);
                        for (int ic = 0; ic < in_c; ++ic) {
                            int64_t qx = (int64_t)in_px[ic];
                            int64_t qf = (int64_t)f_base[ic * out_c];
                            acc += qx * qf;
                            sum_qx += qx;
                            sum_qf += qf;
                            p += 1;
                        }
                    }
                }
                int64_t dot_affine = acc - zw * sum_qx - zx * sum_qf + zx * zw * p;
                float result = (float)dot_affine * combined_scale;
                if (bias != NULL) result += bias[oc];
                out[((oh * out_w + ow) * out_c) + oc] =
                    quantize_float_to_int16_scalar(result, output_scale, output_zp);
            }
        }
    }
}

static void test_quantize_dequantize_scalar_int16(void) {
    assert(quantize_float_to_int16_scalar(0.0f, 0.1f, 0) == 0);
    assert(quantize_float_to_int16_scalar(0.15f, 0.1f, 0) == 2);
    assert(quantize_float_to_int16_scalar(-0.15f, 0.1f, 0) == -2);
    assert(quantize_float_to_int16_scalar(1e9f, 0.1f, 0) == 32767);
    assert(quantize_float_to_int16_scalar(-1e9f, 0.1f, 0) == -32768);
    assert(fabsf(dequantize_int16_to_float_scalar(10, 0.1f, 5) - 0.5f) < TOL);
    printf("  test_quantize_dequantize_scalar_int16 PASS\n");
}

static void test_compute_dynamic_scale_int16(void) {
    float z[] = {0.0f, 0.0f, 0.0f};
    float s0 = compute_dynamic_scale_int16(z, 3);
    assert(fabsf(s0 - (1.0f / 32767.0f)) < TOL);

    float a[] = {-3.2f, 1.0f};
    float s1 = compute_dynamic_scale_int16(a, 2);
    assert(fabsf(s1 - (3.2f / 32767.0f)) < TOL);
    printf("  test_compute_dynamic_scale_int16 PASS\n");
}

static void test_quantize_dequantize_vec_int16(void) {
    float in_f[] = {-1.0f, 0.0f, 0.25f, 0.9f};
    int16_t q[4];
    float out_f[4];
    float s = 0.01f;
    quantize_float_to_int16(in_f, 4, s, 0, q);
    dequantize_int16_to_float(q, 4, s, 0, out_f);
    for (int i = 0; i < 4; ++i) {
        assert(fabsf(out_f[i] - in_f[i]) <= 0.5f * s + 1e-6f);
    }
    printf("  test_quantize_dequantize_vec_int16 PASS\n");
}

static void test_dense_int16_identity(void) {
    int16_t x[] = {10, -5};
    int16_t W[] = {1, 0, 0, 1};
    float b[] = {0.0f, 0.0f};
    int16_t y[2];
    dense_int16(x, 2, W, b, 2, 1.0f, 1.0f, 1.0f, 0, 0, 0, y);
    assert(y[0] == 10);
    assert(y[1] == -5);
    printf("  test_dense_int16_identity PASS\n");
}

static void test_dense_int16_vs_reference(void) {
    const int in_f = 3;
    const int out_f = 2;
    int16_t x[] = {3, -2, 5};
    int16_t W[] = {
        2, 1,
        -1, 4,
        3, -2
    };
    float b[] = {0.5f, -1.5f};
    int16_t y[2];
    dense_int16(x, in_f, W, b, out_f, 0.1f, 0.2f, 0.05f, 0, 0, 0, y);
    for (int o = 0; o < out_f; ++o) {
        int16_t exp = ref_dense_one_out_int16(x, in_f, W, b, out_f, o, 0.1f, 0.2f, 0.05f, 0, 0, 0);
        assert(y[o] == exp);
    }
    printf("  test_dense_int16_vs_reference PASS\n");
}

static void test_relu_int16(void) {
    int16_t in[] = {-5, 0, 3, -32768};
    int16_t out[4];
    relu_int16(in, 4, 0, out);
    assert(out[0] == 0 && out[1] == 0 && out[2] == 3 && out[3] == 0);
    printf("  test_relu_int16 PASS\n");
}

static void test_conv2d_int16_1x1_reference(void) {
    int in_h = 2, in_w = 2, in_c = 2, out_c = 1;
    int16_t in[] = {
        1, 2,
        3, 4,
        5, 6,
        7, 8
    };
    int16_t filt[] = {1, 1};
    float bias[] = {0.0f};
    int16_t out[4], ref[4];
    conv2d_nhwc_int16(in, in_h, in_w, in_c, filt, 1, 1, out_c, bias,
                      1, 1, 0, 0, 1.0f, 1.0f, 1.0f, 0, 0, 0, out);
    ref_conv2d_nhwc_int16(in, in_h, in_w, in_c, filt, 1, 1, out_c, bias,
                          1, 1, 0, 0, 1.0f, 1.0f, 1.0f, 0, 0, 0, ref);
    assert(memcmp(out, ref, sizeof(out)) == 0);
    printf("  test_conv2d_int16_1x1_reference PASS\n");
}

static void test_conv2d_int16_vs_float(void) {
    int in_h = 2, in_w = 2, in_c = 1, out_c = 1;
    int k_h = 2, k_w = 2;
    float sx = 0.1f, sw = 0.1f, so = 0.1f;
    float in_f[] = {0.1f, 0.2f, 0.3f, 0.4f};
    float filt_f[] = {0.5f, -0.5f, 0.25f, 0.75f};
    float bias[] = {0.1f};
    int16_t in_q[4], filt_q[4], out_q[1];
    float out_float[1], out_from_q[1];

    quantize_float_to_int16(in_f, 4, sx, 0, in_q);
    quantize_float_to_int16(filt_f, 4, sw, 0, filt_q);

    conv2d_nhwc_int16(in_q, in_h, in_w, in_c, filt_q, k_h, k_w, out_c, bias,
                      1, 1, 0, 0, sx, sw, so, 0, 0, 0, out_q);
    conv2d_nhwc(in_f, in_h, in_w, in_c, filt_f, k_h, k_w, out_c, bias,
                1, 1, 0, 0, out_float);
    dequantize_int16_to_float(out_q, 1, so, 0, out_from_q);

    assert(fabsf(out_from_q[0] - out_float[0]) < 0.2f);
    printf("  test_conv2d_int16_vs_float PASS\n");
}

int main(void) {
    printf("Running nn_ops_int16 tests...\n");
    test_quantize_dequantize_scalar_int16();
    test_compute_dynamic_scale_int16();
    test_quantize_dequantize_vec_int16();
    test_dense_int16_identity();
    test_dense_int16_vs_reference();
    test_relu_int16();
    test_conv2d_int16_1x1_reference();
    test_conv2d_int16_vs_float();
    printf("All nn_ops_int16 tests PASS.\n");
    return 0;
}
