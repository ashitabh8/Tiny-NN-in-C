/*
 * Quantized Neural Network Operations - int8
 * 
 * Header-only implementation for int8 quantized operations.
 * Bias stays in float32 (design decision).
 */

#ifndef NN_OPS_INT8_H_
#define NN_OPS_INT8_H_

#include <stdint.h>
#include <math.h>

/* ============================================================================
 * Quantization Helpers
 * ============================================================================ */

/**
 * Quantize a single float value to int8
 * 
 * Formula: Q = clamp(round(x / scale) + offset, -128, 127)
 */
static inline int8_t quantize_scalar_int8(float x, float scale, int offset) {
    int32_t val = (int32_t)roundf(x / scale) + offset;
    if (val < -128) val = -128;
    if (val > 127) val = 127;
    return (int8_t)val;
}

/**
 * Dequantize a single int8 value to float
 * 
 * Formula: x = scale * (Q - offset)
 */
static inline float dequantize_scalar_int8(int8_t q, float scale, int offset) {
    return scale * (float)(q - offset);
}

/**
 * Compute dynamic quantization scale from input tensor
 * 
 * Uses symmetric quantization: scale = max(|min|, |max|) / 127
 * This computes the optimal scale at runtime for dynamic quantization.
 * 
 * @param input Input float array
 * @param size  Number of elements
 * @return      Computed scale for symmetric int8 quantization
 */
static inline float compute_dynamic_scale_int8(const float* input, int size) {
    float max_abs = 0.0f;
    for (int i = 0; i < size; i++) {
        float abs_val = fabsf(input[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }
    // Avoid division by zero
    if (max_abs == 0.0f) {
        return 1.0f / 127.0f;
    }
    return max_abs / 127.0f;
}

/**
 * Quantize float array to int8 array
 * 
 * @param input  Input float array
 * @param size   Number of elements
 * @param scale  Quantization scale
 * @param offset Zero point offset
 * @param output Output int8 array
 */
static inline void quantize_float_to_int8(
    const float* input,
    int size,
    float scale,
    int offset,
    int8_t* output)
{
    for (int i = 0; i < size; i++) {
        output[i] = quantize_scalar_int8(input[i], scale, offset);
    }
}

/**
 * Dequantize int8 array to float array
 * 
 * @param input  Input int8 array
 * @param size   Number of elements
 * @param scale  Quantization scale
 * @param offset Zero point offset
 * @param output Output float array
 */
static inline void dequantize_int8_to_float(
    const int8_t* input,
    int size,
    float scale,
    int offset,
    float* output)
{
    for (int i = 0; i < size; i++) {
        output[i] = dequantize_scalar_int8(input[i], scale, offset);
    }
}

/* ============================================================================
 * Quantized Dense (Linear) Layer
 * ============================================================================ */

/**
 * Quantized dense (linear) layer - int8
 *
 * Affine quantized matmul: sum_i (qx_i - zx)(qw_{i,o} - zw) dequantized with
 * sx*sw, plus bias, then requantized with output_scale / output_zp.
 * When zx=zw=0 this matches sum_i qx_i*qw_{i,o} * sx * sw.
 *
 * @param input_zp      Activation zero point (QuantizeNode)
 * @param weight_zp     Weight zero point (compile-time quant)
 * @param output_zp     Layer output zero point (DequantizeNode / requant)
 */
static inline void dense_int8(
    const int8_t* x,
    int in_features,
    const int8_t* W,
    const float* b,
    int out_features,
    float input_scale,
    float weight_scale,
    float output_scale,
    int input_zp,
    int weight_zp,
    int output_zp,
    int8_t* y)
{
    int64_t sum_qx = 0;
    for (int i = 0; i < in_features; ++i) {
        sum_qx += (int64_t)x[i];
    }
    int64_t zx = (int64_t)input_zp;
    int64_t zw = (int64_t)weight_zp;
    int64_t zp_term = zx * zw * (int64_t)in_features;

    for (int o = 0; o < out_features; ++o) {
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

        y[o] = quantize_scalar_int8(result, output_scale, output_zp);
    }
}

/**
 * Dense int8 — per-output-feature weight scales (columns of W).
 *
 * Same affine dot as dense_int8, but multiplies by weight_scales[o] instead of a
 * single weight_scale.
 */
static inline void dense_int8_per_channel(
    const int8_t* x,
    int in_features,
    const int8_t* W,
    const float* b,
    int out_features,
    float input_scale,
    const float* weight_scales,
    float output_scale,
    int input_zp,
    int weight_zp,
    int output_zp,
    int8_t* y)
{
    int64_t sum_qx = 0;
    for (int i = 0; i < in_features; ++i) {
        sum_qx += (int64_t)x[i];
    }
    int64_t zx = (int64_t)input_zp;
    int64_t zw = (int64_t)weight_zp;
    int64_t zp_term = zx * zw * (int64_t)in_features;

    for (int o = 0; o < out_features; ++o) {
        int64_t acc = 0;
        int64_t sum_qw = 0;
        for (int i = 0; i < in_features; ++i) {
            int64_t wv = (int64_t)W[i * out_features + o];
            acc += (int64_t)x[i] * wv;
            sum_qw += wv;
        }
        int64_t dot_affine = acc - zw * sum_qx - zx * sum_qw + zp_term;
        float result = (float)dot_affine * input_scale * weight_scales[o];

        if (b != NULL) {
            result += b[o];
        }

        y[o] = quantize_scalar_int8(result, output_scale, output_zp);
    }
}

/**
 * Dense int8 — per-group weight scales along input axis.
 *
 * weight_scales layout: [num_groups * out_features], index g*out_features + o.
 */
static inline void dense_int8_per_group(
    const int8_t* x,
    int in_features,
    const int8_t* W,
    const float* b,
    int out_features,
    int group_size,
    float input_scale,
    const float* weight_scales,
    float output_scale,
    int input_zp,
    int weight_zp,
    int output_zp,
    int8_t* y)
{
    int num_groups = in_features / group_size;
    for (int o = 0; o < out_features; ++o) {
        float result = 0.0f;
        for (int g = 0; g < num_groups; ++g) {
            int64_t acc = 0;
            int64_t sum_qx = 0;
            int64_t sum_qw = 0;
            int base = g * group_size;
            for (int i = 0; i < group_size; ++i) {
                int idx = base + i;
                int64_t wv = (int64_t)W[idx * out_features + o];
                acc += (int64_t)x[idx] * wv;
                sum_qx += (int64_t)x[idx];
                sum_qw += wv;
            }
            int64_t zp_term = (int64_t)input_zp * (int64_t)weight_zp * (int64_t)group_size;
            int64_t dot_affine = acc - (int64_t)weight_zp * sum_qx
                                 - (int64_t)input_zp * sum_qw + zp_term;
            result += (float)dot_affine * input_scale
                      * weight_scales[g * out_features + o];
        }
        if (b != NULL) {
            result += b[o];
        }
        y[o] = quantize_scalar_int8(result, output_scale, output_zp);
    }
}

/* ============================================================================
 * Quantized Activation Functions
 * ============================================================================ */

/**
 * Quantized ReLU - int8 in-place
 * 
 * For signed int8 with zero_point=0, ReLU is simply max(0, x)
 * 
 * @param x    Input/output int8 array
 * @param size Number of elements
 */
static inline void relu_int8(int8_t* x, int size, int offset) {
    for (int i = 0; i < size; i++) {
        if (x[i] < (int8_t)offset) x[i] = (int8_t)offset;
    }
}

/* ============================================================================
 * Quantized Conv2D
 * ============================================================================ */

/**
 * Quantized Conv2D NHWC - int8 weights, float32 bias
 * 
 * NHWC layout: input [H, W, C_in], filter [K_h, K_w, C_in, C_out]
 * 
 * Uses int64 accumulators for the affine dot, then dequantizes using input_scale * weight_scale,
 * adds float bias, and requantizes output.
 * 
 * @param in             Input int8 array [H, W, C_in]
 * @param in_h           Input height
 * @param in_w           Input width
 * @param in_c           Input channels
 * @param filt           Filter int8 array [K_h, K_w, C_in, C_out]
 * @param k_h            Kernel height
 * @param k_w            Kernel width
 * @param out_c          Output channels
 * @param bias           Bias float array [C_out] (or NULL)
 * @param stride_h       Stride height
 * @param stride_w       Stride width
 * @param pad_h          PyTorch-style padding on each row side
 * @param pad_w          PyTorch-style padding on each column side
 * @param input_scale    Scale used to quantize input
 * @param weight_scale   Scale used to quantize weights
 * @param output_scale   Scale for output int8 (requantization)
 * @param input_zp       Activation zero point
 * @param weight_zp      Weight zero point
 * @param output_zp      Layer output zero point
 * @param out            Output int8 array [H_out, W_out, C_out]
 */
static inline void conv2d_nhwc_int8(
    const int8_t* in, int in_h, int in_w, int in_c,
    const int8_t* filt, int k_h, int k_w, int out_c,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float input_scale,
    float weight_scale,
    float output_scale,
    int input_zp,
    int weight_zp,
    int output_zp,
    int8_t* out)
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

                        const int8_t* in_px = in + ((ih * in_w + iw) * in_c);
                        const int8_t* f_base =
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

                if (bias != NULL) {
                    result += bias[oc];
                }

                out[((oh * out_w + ow) * out_c) + oc] =
                    quantize_scalar_int8(result, output_scale, output_zp);
            }
        }
    }
}

/**
 * Conv2D NHWC int8 — per-output-channel weight scales + affine zero points.
 *
 * Same integer affine dot as conv2d_nhwc_int8, then multiply by
 * input_scale * weight_scales[oc] (per-output-channel weight scale).
 */
static inline void conv2d_nhwc_int8_per_channel(
    const int8_t* in, int in_h, int in_w, int in_c,
    const int8_t* filt, int k_h, int k_w, int out_c,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float input_scale,
    const float* weight_scales,
    float output_scale,
    int input_zp,
    int weight_zp,
    int output_zp,
    int8_t* out)
{
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

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
                        const int8_t* in_px = in + ((ih * in_w + iw) * in_c);
                        const int8_t* f_base =
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
                float result =
                    (float)dot_affine * input_scale * weight_scales[oc];
                if (bias != NULL) {
                    result += bias[oc];
                }
                out[((oh * out_w + ow) * out_c) + oc] =
                    quantize_scalar_int8(result, output_scale, output_zp);
            }
        }
    }
}

/**
 * Conv2D NHWC: float32 activations, int8 weights, per-output-channel weight scales -> float32
 */
static inline void conv2d_nhwc_float_input_int8_weight_per_channel(
    const float* in, int in_h, int in_w, int in_c,
    const int8_t* filt, int k_h, int k_w, int out_c,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    const float* weight_scales,
    float* out)
{
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int oc = 0; oc < out_c; ++oc) {
                float acc = 0.0f;
                for (int kh = 0; kh < k_h; ++kh) {
                    int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kw = 0; kw < k_w; ++kw) {
                        int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) continue;
                        const float* in_px = in + ((ih * in_w + iw) * in_c);
                        const int8_t* f_base =
                            filt + (((kh * k_w + kw) * in_c) * out_c + oc);
                        for (int ic = 0; ic < in_c; ++ic) {
                            acc += in_px[ic] * (float)f_base[ic * out_c] * weight_scales[oc];
                        }
                    }
                }
                if (bias != NULL) {
                    acc += bias[oc];
                }
                out[((oh * out_w + ow) * out_c) + oc] = acc;
            }
        }
    }
}

/* ============================================================================
 * Float-output kernels for dynamic quantization
 *
 * These compute in the integer domain but output float32 directly,
 * avoiding the unnecessary requantize->dequantize round-trip.
 * ============================================================================ */

/**
 * Dense layer: int8 weights + int8 activations -> float32 output
 *
 * Symmetric dot in int64; dequantize with input_scale * weight_scale (typical dynamic path).
 */
static inline void dense_int8_to_float(
    const int8_t* x,
    int in_features,
    const int8_t* W,
    const float* b,
    int out_features,
    float input_scale,
    float weight_scale,
    float* y)
{
    float combined_scale = input_scale * weight_scale;
    for (int o = 0; o < out_features; ++o) {
        int64_t acc = 0;
        for (int i = 0; i < in_features; ++i) {
            acc += (int64_t)x[i] * (int64_t)W[i * out_features + o];
        }
        float result = (float)acc * combined_scale;
        if (b != NULL) {
            result += b[o];
        }
        y[o] = result;
    }
}

/**
 * Conv2D NHWC: int8 activations + int8 weights -> float32 output (dynamic quantization).
 *
 * Padding matches PyTorch / conv2d_nhwc_int8: explicit pad_h, pad_w per side.
 */
static inline void conv2d_nhwc_int8_to_float(
    const int8_t* in, int in_h, int in_w, int in_c,
    const int8_t* filt, int k_h, int k_w, int out_c,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float input_scale,
    float weight_scale,
    float* out)
{
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
    float combined_scale = input_scale * weight_scale;

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int oc = 0; oc < out_c; ++oc) {
                int64_t acc = 0;
                for (int kh = 0; kh < k_h; ++kh) {
                    int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kw = 0; kw < k_w; ++kw) {
                        int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) continue;
                        const int8_t* in_px = in + ((ih * in_w + iw) * in_c);
                        const int8_t* f_base =
                            filt + (((kh * k_w + kw) * in_c) * out_c + oc);
                        for (int ic = 0; ic < in_c; ++ic) {
                            acc += (int64_t)in_px[ic] * (int64_t)f_base[ic * out_c];
                        }
                    }
                }
                float result = (float)acc * combined_scale;
                if (bias != NULL) {
                    result += bias[oc];
                }
                out[((oh * out_w + ow) * out_c) + oc] = result;
            }
        }
    }
}

/* ============================================================================
 * Depthwise Conv2D — per-tensor weight scale
 *
 * Each channel has its own spatial filter but all channels share a single
 * weight_scale.  Weight layout: HWC [k_h, k_w, channels].
 * ============================================================================ */

/**
 * Depthwise Conv2D NHWC int8 — per-tensor, requantized int8 output.
 *
 * Applies the full affine zero-point correction (matches conv2d_nhwc_int8).
 * With symmetric quantization (input_zp=weight_zp=0) the correction terms
 * vanish and the result is identical to the naive sum.
 */
static inline void depthwise_conv2d_nhwc_int8(
    const int8_t* in, int in_h, int in_w, int channels,
    const int8_t* filt, int k_h, int k_w,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float input_scale,
    float weight_scale,
    float output_scale,
    int input_zp,
    int weight_zp,
    int output_zp,
    int8_t* out)
{
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
    float combined_scale = input_scale * weight_scale;
    int64_t zx = (int64_t)input_zp;
    int64_t zw = (int64_t)weight_zp;

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int c = 0; c < channels; ++c) {
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
                        int64_t qx = (int64_t)in[((ih * in_w + iw) * channels) + c];
                        int64_t qf = (int64_t)filt[((kh * k_w + kw) * channels) + c];
                        acc += qx * qf;
                        sum_qx += qx;
                        sum_qf += qf;
                        p += 1;
                    }
                }
                int64_t dot_affine = acc - zw * sum_qx - zx * sum_qf + zx * zw * p;
                float result = (float)dot_affine * combined_scale;
                if (bias != NULL) result += bias[c];
                out[((oh * out_w + ow) * channels) + c] =
                    quantize_scalar_int8(result, output_scale, output_zp);
            }
        }
    }
}

/**
 * Depthwise Conv2D NHWC int8 -> float — per-tensor, float output
 */
static inline void depthwise_conv2d_nhwc_int8_to_float(
    const int8_t* in, int in_h, int in_w, int channels,
    const int8_t* filt, int k_h, int k_w,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float input_scale,
    float weight_scale,
    float* out)
{
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
    float combined_scale = input_scale * weight_scale;

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int c = 0; c < channels; ++c) {
                int64_t acc = 0;
                for (int kh = 0; kh < k_h; ++kh) {
                    int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kw = 0; kw < k_w; ++kw) {
                        int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) continue;
                        acc += (int64_t)in[((ih * in_w + iw) * channels) + c]
                             * (int64_t)filt[((kh * k_w + kw) * channels) + c];
                    }
                }
                float result = (float)acc * combined_scale;
                if (bias != NULL) result += bias[c];
                out[((oh * out_w + ow) * channels) + c] = result;
            }
        }
    }
}

/* ============================================================================
 * Depthwise Conv2D — per-channel weight scales
 *
 * Each channel uses its own weight_scales[c].  Useful for higher accuracy
 * when channel weight ranges vary.
 * ============================================================================ */

/**
 * Depthwise Conv2D NHWC int8 — per-channel weight scales + affine zero points
 */
static inline void depthwise_conv2d_nhwc_int8_per_channel(
    const int8_t* in, int in_h, int in_w, int channels,
    const int8_t* filt, int k_h, int k_w,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float input_scale,
    const float* weight_scales,
    float output_scale,
    int input_zp,
    int weight_zp,
    int output_zp,
    int8_t* out)
{
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

    int64_t zx = (int64_t)input_zp;
    int64_t zw = (int64_t)weight_zp;

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int c = 0; c < channels; ++c) {
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
                        int64_t qx = (int64_t)in[((ih * in_w + iw) * channels) + c];
                        int64_t qf = (int64_t)filt[((kh * k_w + kw) * channels) + c];
                        acc += qx * qf;
                        sum_qx += qx;
                        sum_qf += qf;
                        p += 1;
                    }
                }
                int64_t dot_affine = acc - zw * sum_qx - zx * sum_qf + zx * zw * p;
                float result =
                    (float)dot_affine * input_scale * weight_scales[c];
                if (bias != NULL) result += bias[c];
                out[((oh * out_w + ow) * channels) + c] =
                    quantize_scalar_int8(result, output_scale, output_zp);
            }
        }
    }
}

/**
 * Depthwise Conv2D NHWC int8 -> float — per-channel, float output
 */
static inline void depthwise_conv2d_nhwc_int8_to_float_per_channel(
    const int8_t* in, int in_h, int in_w, int channels,
    const int8_t* filt, int k_h, int k_w,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float input_scale,
    const float* weight_scales,
    float* out)
{
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int c = 0; c < channels; ++c) {
                int64_t acc = 0;
                for (int kh = 0; kh < k_h; ++kh) {
                    int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kw = 0; kw < k_w; ++kw) {
                        int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) continue;
                        acc += (int64_t)in[((ih * in_w + iw) * channels) + c]
                             * (int64_t)filt[((kh * k_w + kw) * channels) + c];
                    }
                }
                float result = (float)acc * input_scale * weight_scales[c];
                if (bias != NULL) result += bias[c];
                out[((oh * out_w + ow) * channels) + c] = result;
            }
        }
    }
}

/* ============================================================================
 * Reduction / pooling / layout (int8)
 * ============================================================================ */

/**
 * Global average over H and W per channel: NHWC [H,W,C] -> [C].
 * Dequantize with input_scale, average in float, requantize with output_scale.
 */
static inline void mean_hwc_int8(
    const int8_t* in, int h, int w, int c,
    float input_scale, float output_scale, int offset,
    int8_t* out)
{
    int n = h * w;
    for (int ch = 0; ch < c; ++ch) {
        float sum = 0.0f;
        for (int ih = 0; ih < h; ++ih) {
            for (int iw = 0; iw < w; ++iw) {
                int8_t q = in[((ih * w + iw) * c) + ch];
                sum += dequantize_scalar_int8(q, input_scale, 0);
            }
        }
        float mean = sum / (float)n;
        out[ch] = quantize_scalar_int8(mean, output_scale, offset);
    }
}

/**
 * Mean along last dimension: row-major [rows, cols] -> [rows].
 */
static inline void mean_last_dim_int8(
    const int8_t* in, int rows, int cols,
    float input_scale, float output_scale, int offset,
    int8_t* out)
{
    for (int r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (int j = 0; j < cols; ++j) {
            int8_t q = in[r * cols + j];
            sum += dequantize_scalar_int8(q, input_scale, 0);
        }
        float mean = sum / (float)cols;
        out[r] = quantize_scalar_int8(mean, output_scale, offset);
    }
}

static inline void global_average_pool_2d_int8(
    const int8_t* in, int h, int w, int c,
    float input_scale, float output_scale, int offset,
    int8_t* out)
{
    mean_hwc_int8(in, h, w, c, input_scale, output_scale, offset, out);
}

static inline void adaptive_avg_pool_2d_1x1_int8(
    const int8_t* in, int h, int w, int c,
    float input_scale, float output_scale, int offset,
    int8_t* out)
{
    global_average_pool_2d_int8(in, h, w, c, input_scale, output_scale, offset, out);
}

static inline void flatten_int8(const int8_t* src, int n, int8_t* dst)
{
    for (int i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}

/**
 * Dense: float activations, int8 weights, per-output-feature symmetric scales
 * -> float output.
 */
static inline void dense_float_input_int8_weight_per_channel(
    const float* x, int in_features,
    const int8_t* W, const float* b,
    int out_features,
    const float* weight_scales,
    float* y)
{
    for (int o = 0; o < out_features; ++o) {
        float acc = (b != NULL) ? b[o] : 0.0f;
        float scale_o = weight_scales[o];
        const int8_t* w_col = W + o;
        for (int i = 0; i < in_features; ++i) {
            acc += x[i] * ((float)w_col[i * out_features] * scale_o);
        }
        y[o] = acc;
    }
}

#endif /* NN_OPS_INT8_H_ */
