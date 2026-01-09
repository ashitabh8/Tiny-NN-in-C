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
static inline void dense_int8(
    const int8_t* x,
    int in_features,
    const int8_t* W,
    const float* b,
    int out_features,
    float scale,
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
        
        // Scale accumulator - this is an approximation
        // In a proper implementation, we'd use fixed-point arithmetic
        float result = (float)acc * scale * scale;  // scale for both input and weights
        
        // Add bias (float32)
        if (b != NULL) {
            result += b[o];
        }
        
        // Quantize output
        y[o] = quantize_scalar_int8(result, scale, offset);
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
 * Quantized Conv2D (placeholder)
 * ============================================================================ */

/**
 * Quantized Conv2D NHWC - int8 weights, float32 bias
 * 
 * TODO: Implement full conv2d_nhwc_int8
 */

#endif /* NN_OPS_INT8_H_ */

