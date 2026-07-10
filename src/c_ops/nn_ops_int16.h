/**
 * nn_ops_int16.h - INT16 Neural Network Operations
 * 
 * Provides quantized int16 implementations for neural network layers.
 * Higher precision than int8, useful for output layers.
 */

#ifndef NN_OPS_INT16_H_
#define NN_OPS_INT16_H_

#include <stdint.h>
#include <math.h>

/* ==========================================================================
 * Quantization/Dequantization Utilities
 * ========================================================================== */

/**
 * Quantize a single float value to int16
 * formula: q = round(x / scale) + offset
 */
static inline int16_t quantize_float_to_int16_scalar(float x, float scale, int offset) {
    float val = roundf(x / scale) + offset;
    if (val < -32768.0f) return -32768;
    if (val > 32767.0f) return 32767;
    return (int16_t)val;
}

/**
 * Dequantize a single int16 value to float
 * formula: x = (q - offset) * scale
 */
static inline float dequantize_int16_to_float_scalar(int16_t q, float scale, int offset) {
    return ((float)(q - offset)) * scale;
}

/**
 * Compute dynamic quantization scale from input tensor
 * 
 * Uses symmetric quantization: scale = max(|min|, |max|) / 32767
 * This computes the optimal scale at runtime for dynamic quantization.
 * 
 * @param input Input float array
 * @param size  Number of elements
 * @return      Computed scale for symmetric int16 quantization
 */
static inline float compute_dynamic_scale_int16(const float* input, int size) {
    float max_abs = 0.0f;
    for (int i = 0; i < size; i++) {
        float abs_val = fabsf(input[i]);
        if (abs_val > max_abs) {
            max_abs = abs_val;
        }
    }
    // Avoid division by zero
    if (max_abs == 0.0f) {
        return 1.0f / 32767.0f;
    }
    return max_abs / 32767.0f;
}

/**
 * Quantize a float vector to int16 vector
 * (Note: named without _vec suffix to match int8 API)
 */
static inline void quantize_float_to_int16(
    const float* x,
    int size,
    float scale,
    int offset,
    int16_t* y)
{
    for (int i = 0; i < size; ++i) {
        y[i] = quantize_float_to_int16_scalar(x[i], scale, offset);
    }
}

/**
 * Dequantize an int16 vector to float vector
 * (Note: named without _vec suffix to match int8 API)
 */
static inline void dequantize_int16_to_float(
    const int16_t* x,
    int size,
    float scale,
    int offset,
    float* y)
{
    for (int i = 0; i < size; ++i) {
        y[i] = dequantize_int16_to_float_scalar(x[i], scale, offset);
    }
}

/* ==========================================================================
 * Dense/Linear Layer (int16)
 * ========================================================================== */

/**
 * Quantized dense (linear) layer - int16 (affine zero-points, same math as int8).
 *
 * @param input_zp      Activation zero point
 * @param weight_zp     Weight zero point
 * @param output_zp     Layer output zero point
 */
static inline void dense_int16(
    const int16_t* x,
    int in_features,
    const int16_t* W,
    const float* bias,
    int out_features,
    float input_scale,
    float weight_scale,
    float output_scale,
    int input_zp,
    int weight_zp,
    int output_zp,
    int16_t* y)
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

        if (bias) {
            result += bias[o];
        }

        y[o] = quantize_float_to_int16_scalar(result, output_scale, output_zp);
    }
}

/**
 * Dense int16 — per-output-feature weight scales (columns of W).
 */
static inline void dense_int16_per_channel(
    const int16_t* x,
    int in_features,
    const int16_t* W,
    const float* bias,
    int out_features,
    float input_scale,
    const float* weight_scales,
    float output_scale,
    int input_zp,
    int weight_zp,
    int output_zp,
    int16_t* y)
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

        if (bias) {
            result += bias[o];
        }

        y[o] = quantize_float_to_int16_scalar(result, output_scale, output_zp);
    }
}

static inline void dense_int16_per_group(
    const int16_t* x,
    int in_features,
    const int16_t* W,
    const float* bias,
    int out_features,
    int group_size,
    float input_scale,
    const float* weight_scales,
    float output_scale,
    int input_zp,
    int weight_zp,
    int output_zp,
    int16_t* y)
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
        if (bias) {
            result += bias[o];
        }
        y[o] = quantize_float_to_int16_scalar(result, output_scale, output_zp);
    }
}

/* ==========================================================================
 * ReLU Activation (int16)
 * ========================================================================== */

/**
 * ReLU activation for int16 data — in-place.
 *
 * For quantized ReLU, values below the zero point are clamped to the zero point.
 * If offset=0 this is simply max(0, x).  Matches the in-place API of relu_int8.
 *
 * @param x      Input/output int16 array (modified in place)
 * @param size   Number of elements
 * @param offset Zero point (values < offset become offset)
 */
static inline void relu_int16(int16_t* x, int size, int offset) {
    for (int i = 0; i < size; ++i) {
        if (x[i] < (int16_t)offset) x[i] = (int16_t)offset;
    }
}

/* ==========================================================================
 * Conv2D Layer (int16) - NHWC format
 * ========================================================================== */

/**
 * Quantized Conv2D NHWC - int16 weights, float32 bias
 * 
 * NHWC layout: input [H, W, C_in], filter [K_h, K_w, C_in, C_out]
 * 
 * Uses int32 accumulator, then dequantizes using input_scale * weight_scale,
 * adds float bias, and requantizes output.
 * 
 * @param in             Input int16 array [H, W, C_in]
 * @param in_h           Input height
 * @param in_w           Input width
 * @param in_c           Input channels
 * @param filt           Filter int16 array [K_h, K_w, C_in, C_out]
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
 * @param output_scale   Scale for output int16 (requantization)
 * @param input_zp       Activation zero point
 * @param weight_zp      Weight zero point
 * @param output_zp      Layer output zero point
 * @param out            Output int16 array [H_out, W_out, C_out]
 */
static inline void conv2d_nhwc_int16(
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

                if (bias != NULL) {
                    result += bias[oc];
                }

                out[((oh * out_w + ow) * out_c) + oc] =
                    quantize_float_to_int16_scalar(result, output_scale, output_zp);
            }
        }
    }
}

/**
 * Conv2D NHWC int16 — per-output-channel weight scales + affine zero points
 */
static inline void conv2d_nhwc_int16_per_channel(
    const int16_t* in, int in_h, int in_w, int in_c,
    const int16_t* filt, int k_h, int k_w, int out_c,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float input_scale,
    const float* weight_scales,
    float output_scale,
    int input_zp,
    int weight_zp,
    int output_zp,
    int16_t* out)
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
                float result =
                    (float)dot_affine * input_scale * weight_scales[oc];
                if (bias != NULL) {
                    result += bias[oc];
                }

                out[((oh * out_w + ow) * out_c) + oc] =
                    quantize_float_to_int16_scalar(result, output_scale, output_zp);
            }
        }
    }
}

/* ==========================================================================
 * Float-output kernels for dynamic quantization
 *
 * Compute in the integer domain but output float32 directly,
 * avoiding the unnecessary requantize->dequantize round-trip.
 * ========================================================================== */

/**
 * Dense layer: int16 weights + int16 activations -> float32 output
 */
static inline void dense_int16_to_float(
    const int16_t* x,
    int in_features,
    const int16_t* W,
    const float* bias,
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
        if (bias) {
            result += bias[o];
        }
        y[o] = result;
    }
}

/**
 * Conv2D NHWC: int16 activations + int16 weights -> float32 (dynamic path).
 * Padding: explicit pad_h, pad_w per side (same as conv2d_nhwc_int16).
 */
static inline void conv2d_nhwc_int16_to_float(
    const int16_t* in, int in_h, int in_w, int in_c,
    const int16_t* filt, int k_h, int k_w, int out_c,
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
                        const int16_t* in_px = in + ((ih * in_w + iw) * in_c);
                        const int16_t* f_base =
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

/* ==========================================================================
 * Depthwise Conv2D — per-tensor weight scale (int16)
 *
 * Weight layout: HWC [k_h, k_w, channels].  Uses int64 accumulators.
 * ========================================================================== */

/**
 * Depthwise Conv2D NHWC int16 — per-tensor, requantized int16 output.
 *
 * Applies the full affine zero-point correction (matches conv2d_nhwc_int16).
 * With symmetric quantization (input_zp=weight_zp=0) the correction terms
 * vanish and the result is identical to the naive sum.
 */
static inline void depthwise_conv2d_nhwc_int16(
    const int16_t* in, int in_h, int in_w, int channels,
    const int16_t* filt, int k_h, int k_w,
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
                    quantize_float_to_int16_scalar(result, output_scale, output_zp);
            }
        }
    }
}

/**
 * Depthwise Conv2D NHWC int16 -> float — per-tensor, float output
 */
static inline void depthwise_conv2d_nhwc_int16_to_float(
    const int16_t* in, int in_h, int in_w, int channels,
    const int16_t* filt, int k_h, int k_w,
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

/* ==========================================================================
 * Depthwise Conv2D — per-channel weight scales (int16)
 * ========================================================================== */

/**
 * Depthwise Conv2D NHWC int16 — per-channel weight scales + affine zero points
 */
static inline void depthwise_conv2d_nhwc_int16_per_channel(
    const int16_t* in, int in_h, int in_w, int channels,
    const int16_t* filt, int k_h, int k_w,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float input_scale,
    const float* weight_scales,
    float output_scale,
    int input_zp,
    int weight_zp,
    int output_zp,
    int16_t* out)
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
                    quantize_float_to_int16_scalar(result, output_scale, output_zp);
            }
        }
    }
}

/**
 * Depthwise Conv2D NHWC int16 -> float — per-channel, float output
 */
static inline void depthwise_conv2d_nhwc_int16_to_float_per_channel(
    const int16_t* in, int in_h, int in_w, int channels,
    const int16_t* filt, int k_h, int k_w,
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

#endif /* NN_OPS_INT16_H */
