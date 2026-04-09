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
 * Quantized dense (linear) layer - int8 weights, float32 bias
 * 
 * Formula: y = W * x + b
 * 
 * The computation is done in integer accumulation, then converted to float
 * for bias addition, then quantized back to int8.
 * 
 * @param x           Input int8 array [in_features]
 * @param in_features Number of input features
 * @param W           Weight int8 array [in_features * out_features] (row-major)
 * @param b           Bias float array [out_features] (or NULL)
 * @param out_features Number of output features
 * @param scale       Quantization scale for output
 * @param offset      Zero point offset for output
 * @param y           Output int8 array [out_features]
 */
/**
 * Quantized dense (linear) layer - int8
 * 
 * Computes: y = W * x + b
 * 
 * Uses int32 accumulator, then dequantizes using input_scale * weight_scale,
 * adds float bias, and requantizes output.
 * 
 * @param x             Input int8 array [in_features]
 * @param in_features   Number of input features
 * @param W             Weight int8 array [in_features * out_features] (row-major)
 * @param b             Bias float array [out_features] (or NULL)
 * @param out_features  Number of output features
 * @param input_scale   Scale used to quantize input
 * @param weight_scale  Scale used to quantize weights
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
        
        // Quantize output using weight_scale
        y[o] = quantize_scalar_int8(result, weight_scale, offset);
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
 * @param pad_same       1 for SAME padding, 0 for VALID
 * @param input_scale    Scale used to quantize input
 * @param weight_scale   Scale used to quantize weights
 * @param offset         Zero point offset for output
 * @param out            Output int8 array [H_out, W_out, C_out]
 */
static inline void conv2d_nhwc_int8(
    const int8_t* in, int in_h, int in_w, int in_c,
    const int8_t* filt, int k_h, int k_w, int out_c,
    const float* bias,
    int stride_h, int stride_w,
    int pad_same,
    float input_scale,
    float weight_scale,
    int offset,
    int8_t* out)
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
                
                // Quantize output
                out[((oh * out_w + ow) * out_c) + oc] = quantize_scalar_int8(result, weight_scale, offset);
            }
        }
    }
}

#endif /* NN_OPS_INT8_H_ */

