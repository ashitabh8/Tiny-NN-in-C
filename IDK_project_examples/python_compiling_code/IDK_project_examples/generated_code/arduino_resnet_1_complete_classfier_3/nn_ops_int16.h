/**
 * nn_ops_int16.h - INT16 Neural Network Operations
 * 
 * Provides quantized int16 implementations for neural network layers.
 * Higher precision than int8, useful for output layers.
 */

#ifndef NN_OPS_INT16_H
#define NN_OPS_INT16_H

#include <stdint.h>
#include <math.h>
#include <string.h>

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
 * Quantized dense (linear) layer - int16
 * 
 * Uses int32 accumulator to avoid overflow, then dequantizes using
 * input_scale * weight_scale, adds float bias, and requantizes.
 * 
 * @param x             Input vector (int16), shape: [in_features]
 * @param in_features   Number of input features
 * @param W             Weight matrix (int16), shape: [in_features, out_features] (row-major)
 * @param bias          Bias vector (float), shape: [out_features], or NULL
 * @param out_features  Number of output features
 * @param input_scale   Scale used to quantize input
 * @param weight_scale  Scale used to quantize weights
 * @param offset        Quantization offset (zero point)
 * @param y             Output vector (int16), shape: [out_features]
 */
static inline void dense_int16(
    const int16_t* x,
    int in_features,
    const int16_t* W,
    const float* bias,
    int out_features,
    float input_scale,
    float weight_scale,
    int offset,
    int16_t* y)
{
    for (int o = 0; o < out_features; ++o) {
        // Use int32 accumulator to prevent overflow
        int32_t acc = 0;
        for (int i = 0; i < in_features; ++i) {
            acc += (int32_t)x[i] * (int32_t)W[i * out_features + o];
        }
        
        // Dequantize: result = acc * input_scale * weight_scale
        float result = (float)acc * input_scale * weight_scale;
        
        // Add bias (in float domain)
        if (bias) {
            result += bias[o];
        }
        
        // Requantize to int16 using weight_scale
        y[o] = quantize_float_to_int16_scalar(result, weight_scale, offset);
    }
}

/* ==========================================================================
 * ReLU Activation (int16)
 * ========================================================================== */

/**
 * ReLU activation for int16 data.
 * 
 * For quantized ReLU, values below zero point are clamped.
 * If offset=0, this is simply max(0, x).
 * 
 * @param x Input vector (int16)
 * @param size Number of elements
 * @param offset Zero point (values < offset become offset)
 * @param y Output vector (int16)
 */
static inline void relu_int16(
    const int16_t* x,
    int size,
    int offset,
    int16_t* y)
{
    for (int i = 0; i < size; ++i) {
        y[i] = (x[i] > offset) ? x[i] : offset;
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
 * @param pad_same       1 for SAME padding, 0 for VALID
 * @param input_scale    Scale used to quantize input
 * @param weight_scale   Scale used to quantize weights
 * @param offset         Zero point offset for output
 * @param out            Output int16 array [H_out, W_out, C_out]
 */
static inline void conv2d_nhwc_int16(
    const int16_t* in, int in_h, int in_w, int in_c,
    const int16_t* filt, int k_h, int k_w, int out_c,
    const float* bias,
    int stride_h, int stride_w,
    int pad_same,
    float input_scale,
    float weight_scale,
    int offset,
    int16_t* out)
{
    // Calculate output dimensions
    int out_h, out_w;
    if (pad_same) {
        out_h = (in_h + stride_h - 1) / stride_h;
        out_w = (in_w + stride_w - 1) / stride_w;
    } else {
        out_h = (in_h - k_h) / stride_h + 1;
        out_w = (in_w - k_w) / stride_w + 1;
    }
    
    // Calculate padding
    int pad_h_total = pad_same ? ((out_h - 1) * stride_h + k_h - in_h) : 0;
    int pad_w_total = pad_same ? ((out_w - 1) * stride_w + k_w - in_w) : 0;
    int pad_top = pad_h_total / 2;
    int pad_left = pad_w_total / 2;
    
    // Combined scale for dequantization
    float combined_scale = input_scale * weight_scale;
    
    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int oc = 0; oc < out_c; ++oc) {
                // Integer accumulation
                int32_t acc = 0;
                
                for (int kh = 0; kh < k_h; ++kh) {
                    int ih = oh * stride_h + kh - pad_top;
                    if (ih < 0 || ih >= in_h) continue;
                    
                    for (int kw = 0; kw < k_w; ++kw) {
                        int iw = ow * stride_w + kw - pad_left;
                        if (iw < 0 || iw >= in_w) continue;
                        
                        // Input pixel: in[ih, iw, :]
                        const int16_t* in_px = in + ((ih * in_w + iw) * in_c);
                        
                        // Filter: filt[kh, kw, :, oc]
                        // Layout: [K_h, K_w, C_in, C_out]
                        const int16_t* f_base = filt + (((kh * k_w + kw) * in_c) * out_c + oc);
                        
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
                
                // Quantize output
                out[((oh * out_w + ow) * out_c) + oc] = quantize_float_to_int16_scalar(result, weight_scale, offset);
            }
        }
    }
}

#endif /* NN_OPS_INT16_H */

