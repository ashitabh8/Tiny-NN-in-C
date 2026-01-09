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
 * Dense (fully connected) layer with int16 weights and activations.
 * 
 * Uses int32 accumulator to avoid overflow, then requantizes to int16.
 * Bias remains in float32 for accuracy.
 * 
 * @param x Input vector (int16), shape: [in_features]
 * @param in_features Number of input features
 * @param W Weight matrix (int16), shape: [in_features, out_features] (row-major)
 * @param bias Bias vector (float), shape: [out_features], or NULL
 * @param out_features Number of output features
 * @param scale Quantization scale
 * @param offset Quantization offset (zero point)
 * @param y Output vector (int16), shape: [out_features]
 */
static inline void dense_int16(
    const int16_t* x,
    int in_features,
    const int16_t* W,
    const float* bias,
    int out_features,
    float scale,
    int offset,
    int16_t* y)
{
    for (int o = 0; o < out_features; ++o) {
        // Use int32 accumulator to prevent overflow
        int32_t acc = 0;
        for (int i = 0; i < in_features; ++i) {
            acc += (int32_t)x[i] * (int32_t)W[i * out_features + o];
        }
        
        // Convert accumulator to float, apply scale^2 (input_scale * weight_scale)
        float result = (float)acc * (scale * scale);
        
        // Add bias (in float domain)
        if (bias) {
            result += bias[o];
        }
        
        // Requantize to int16
        y[o] = quantize_float_to_int16_scalar(result, scale, offset);
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
 * Placeholder for future implementation
 * ========================================================================== */

/**
 * 2D Convolution with int16 weights (NHWC format).
 * 
 * Note: This is a placeholder. Full implementation pending.
 * Uses int32 accumulator for intermediate computations.
 * 
 * @param input Input tensor (int16), shape: [N, H, W, C_in]
 * @param batch Batch size (N)
 * @param in_h Input height (H)
 * @param in_w Input width (W)
 * @param in_c Input channels (C_in)
 * @param kernel Kernel tensor (int16), shape: [kH, kW, C_in, C_out]
 * @param kh Kernel height
 * @param kw Kernel width
 * @param bias Bias vector (float), shape: [C_out], or NULL
 * @param out_c Output channels (C_out)
 * @param stride_h Stride in height dimension
 * @param stride_w Stride in width dimension
 * @param pad_h Padding in height dimension
 * @param pad_w Padding in width dimension
 * @param scale Quantization scale
 * @param offset Quantization offset
 * @param output Output tensor (int16), shape: [N, H_out, W_out, C_out]
 */
static inline void conv2d_nhwc_int16(
    const int16_t* input,
    int batch, int in_h, int in_w, int in_c,
    const int16_t* kernel,
    int kh, int kw,
    const float* bias,
    int out_c,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float scale, int offset,
    int16_t* output)
{
    int out_h = (in_h + 2 * pad_h - kh) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - kw) / stride_w + 1;
    
    for (int n = 0; n < batch; ++n) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {
                for (int oc = 0; oc < out_c; ++oc) {
                    int32_t acc = 0;
                    
                    // Convolution over kernel window
                    for (int fh = 0; fh < kh; ++fh) {
                        for (int fw = 0; fw < kw; ++fw) {
                            int ih = oh * stride_h - pad_h + fh;
                            int iw = ow * stride_w - pad_w + fw;
                            
                            // Skip out-of-bounds (zero-padding)
                            if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) {
                                continue;
                            }
                            
                            for (int ic = 0; ic < in_c; ++ic) {
                                int in_idx = ((n * in_h + ih) * in_w + iw) * in_c + ic;
                                int k_idx = ((fh * kw + fw) * in_c + ic) * out_c + oc;
                                acc += (int32_t)input[in_idx] * (int32_t)kernel[k_idx];
                            }
                        }
                    }
                    
                    // Convert to float, apply bias, requantize
                    float result = (float)acc * (scale * scale);
                    if (bias) {
                        result += bias[oc];
                    }
                    
                    int out_idx = ((n * out_h + oh) * out_w + ow) * out_c + oc;
                    output[out_idx] = quantize_float_to_int16_scalar(result, scale, offset);
                }
            }
        }
    }
}

#endif /* NN_OPS_INT16_H */

