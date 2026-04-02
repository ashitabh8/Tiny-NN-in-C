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
 * Computes: y = W * x + b
 *
 * Uses int32 accumulator, then dequantizes using input_scale * weight_scale,
 * adds float bias, and requantizes output with output_scale (must match the
 * scale used by a following dequantize step, e.g. StaticQuantRule output_scale).
 *
 * @param x             Input int8 array [in_features]
 * @param in_features   Number of input features
 * @param W             Weight int8 array [in_features * out_features] (row-major)
 * @param b             Bias float array [out_features] (or NULL)
 * @param out_features  Number of output features
 * @param input_scale   Scale used to quantize input
 * @param weight_scale  Scale used to quantize weights
 * @param output_scale  Scale for output int8 (requantization)
 * @param offset        Zero point offset for output
 * @param y             Output int8 array [out_features]
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
    int offset,
    int8_t* y)
{
    for (int o = 0; o < out_features; ++o) {
        // Integer accumulation
        int32_t acc = 0;

        for (int i = 0; i < in_features; ++i) {
            // W is stored as [in_features, out_features] row-major
            acc += (int32_t)x[i] * (int32_t)W[i * out_features + o];
        }

        // Dequantize: result = acc * input_scale * weight_scale
        float result = (float)acc * input_scale * weight_scale;

        // Add bias (float32)
        if (b != NULL) {
            result += b[o];
        }

        y[o] = quantize_scalar_int8(result, output_scale, offset);
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
static inline void relu_int8(int8_t* x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0) x[i] = 0;
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
 * Uses int32 accumulator, then dequantizes using input_scale * weight_scale,
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
 * @param offset         Zero point offset for output
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
    int offset,
    int8_t* out)
{
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;
    
    // Combined scale for dequantization
    float combined_scale = input_scale * weight_scale;
    
    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int oc = 0; oc < out_c; ++oc) {
                // Integer accumulation
                int32_t acc = 0;
                
                for (int kh = 0; kh < k_h; ++kh) {
                    int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) continue;
                    
                    for (int kw = 0; kw < k_w; ++kw) {
                        int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) continue;
                        
                        // Input pixel: in[ih, iw, :]
                        const int8_t* in_px = in + ((ih * in_w + iw) * in_c);
                        
                        // Filter: filt[kh, kw, :, oc]
                        // Layout: [K_h, K_w, C_in, C_out]
                        const int8_t* f_base = filt + (((kh * k_w + kw) * in_c) * out_c + oc);
                        
                        for (int ic = 0; ic < in_c; ++ic) {
                            acc += (int32_t)in_px[ic] * (int32_t)f_base[ic * out_c];
                        }
                    }
                }
                
                // Dequantize accumulated result
                float result = (float)acc * combined_scale;
                
                // Add bias (float32)
                if (bias != NULL) {
                    result += bias[oc];
                }
                
                out[((oh * out_w + ow) * out_c) + oc] =
                    quantize_scalar_int8(result, output_scale, offset);
            }
        }
    }
}

/* ============================================================================
 * Quantized Reductions / Pooling / View Helpers
 * ============================================================================ */

/**
 * Mean over spatial dimensions (H, W) for NHWC int8 input.
 *
 * Dequantize domain:
 *   mean_float = (sum(q) * input_scale) / (H*W)
 * Requantize:
 *   out_q = quantize(mean_float, output_scale, offset)
 */
static inline void mean_hwc_int8(
    const int8_t* in,
    int h,
    int w,
    int c,
    float input_scale,
    float output_scale,
    int offset,
    int8_t* out)
{
    int n = h * w;
    for (int ch = 0; ch < c; ++ch) {
        int32_t acc = 0;
        for (int ih = 0; ih < h; ++ih) {
            for (int iw = 0; iw < w; ++iw) {
                acc += (int32_t)in[((ih * w + iw) * c) + ch];
            }
        }
        float mean_val = (n > 0) ? ((float)acc * input_scale / (float)n) : 0.0f;
        out[ch] = quantize_scalar_int8(mean_val, output_scale, offset);
    }
}

/**
 * Mean over the last dimension of a 2D int8 tensor [rows, cols] -> [rows].
 */
static inline void mean_last_dim_int8(
    const int8_t* in,
    int rows,
    int cols,
    float input_scale,
    float output_scale,
    int offset,
    int8_t* out)
{
    for (int r = 0; r < rows; ++r) {
        int32_t acc = 0;
        const int8_t* row = in + r * cols;
        for (int c = 0; c < cols; ++c) {
            acc += (int32_t)row[c];
        }
        float mean_val = (cols > 0) ? ((float)acc * input_scale / (float)cols) : 0.0f;
        out[r] = quantize_scalar_int8(mean_val, output_scale, offset);
    }
}

/**
 * GlobalAveragePool2D over H and W for NHWC int8 input.
 * Thin wrapper around mean_hwc_int8.
 */
static inline void global_average_pool_2d_int8(
    const int8_t* in,
    int h,
    int w,
    int c,
    float input_scale,
    float output_scale,
    int offset,
    int8_t* out)
{
    mean_hwc_int8(in, h, w, c, input_scale, output_scale, offset, out);
}

/**
 * AdaptiveAvgPool2d((1,1)) equivalent for int8 NHWC input.
 */
static inline void adaptive_avg_pool_2d_1x1_int8(
    const int8_t* in,
    int in_h,
    int in_w,
    int in_c,
    float input_scale,
    float output_scale,
    int offset,
    int8_t* out)
{
    global_average_pool_2d_int8(
        in, in_h, in_w, in_c, input_scale, output_scale, offset, out
    );
}

/**
 * Flatten helper for int8 buffers (copy n elements).
 */
static inline void flatten_int8(const int8_t* src, int n, int8_t* dst) {
    for (int i = 0; i < n; ++i) {
        dst[i] = src[i];
    }
}

#endif /* NN_OPS_INT8_H_ */

