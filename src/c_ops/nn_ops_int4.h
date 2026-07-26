/*
 * Int4 packed-weight operations (W4A8) and palettized weight ops.
 *
 * Activations for int4 dense kernels are int8 (static/dynamic quantization).
 * Palettization remains float activations / float codebook lookup.
 */

#ifndef NN_OPS_INT4_H_
#define NN_OPS_INT4_H_

#include <stdint.h>
#include <math.h>
#include "nn_ops_int8.h"

static inline int8_t unpack_int4_low(int8_t packed) {
    int8_t v = (int8_t)(packed & 0x0F);
    return (v >= 8) ? (int8_t)(v - 16) : v;
}

static inline int8_t unpack_int4_high(int8_t packed) {
    int8_t v = (int8_t)((packed >> 4) & 0x0F);
    return (v >= 8) ? (int8_t)(v - 16) : v;
}

static inline int8_t unpack_int4_at(const int8_t* packed_w, int flat) {
    int byte_idx = flat / 2;
    return (flat & 1) ? unpack_int4_high(packed_w[byte_idx])
                      : unpack_int4_low(packed_w[byte_idx]);
}

/* Phase 3 unified affine dense — int8 act + packed int4 weights. */
NN_DENSE_AFFINE_DEFINE_ZP(
    dense_affine_int8_w4_core,
    int8_t, int8_t, NN_LOAD_W_INT4, int8_t, NN_STORE_INT8)

NN_DENSE_AFFINE_DEFINE_TO_FLOAT(
    dense_affine_int8_w4_to_float_core,
    int8_t, int8_t, NN_LOAD_W_INT4)

/* Public API keeps packed weight_count (unused; matches prior codegen). */
static inline void dense_affine_int8_w4(
    const int8_t* x,
    int in_features,
    const int8_t* packed_w,
    int weight_count,
    const float* b,
    int out_features,
    int group_size,
    int per_out_column,
    float input_scale,
    const float* weight_scales,
    float output_scale,
    int input_zp,
    int weight_zp,
    int output_zp,
    int a_symmetric,
    int w_symmetric,
    int8_t* y)
{
    (void)weight_count;
    dense_affine_int8_w4_core(
        x, in_features, packed_w, b, out_features, group_size, per_out_column,
        input_scale, weight_scales, output_scale, input_zp, weight_zp, output_zp,
        a_symmetric, w_symmetric, y);
}

static inline void dense_affine_int8_w4_to_float(
    const int8_t* x,
    int in_features,
    const int8_t* packed_w,
    int weight_count,
    const float* b,
    int out_features,
    int group_size,
    int per_out_column,
    float input_scale,
    const float* weight_scales,
    float* y)
{
    (void)weight_count;
    dense_affine_int8_w4_to_float_core(
        x, in_features, packed_w, b, out_features, group_size, per_out_column,
        input_scale, weight_scales, y);
}

static inline void dense_float_palettized(
    const float* x,
    int in_features,
    const uint8_t* indices,
    int weight_count,
    const float* codebook,
    int num_centroids,
    const float* b,
    int out_features,
    float* y)
{
    (void)weight_count;
    for (int o = 0; o < out_features; ++o) {
        float result = 0.0f;
        if (b != NULL) {
            result = b[o];
        }
        for (int i = 0; i < in_features; ++i) {
            int flat = i * out_features + o;
            float w;
            if (num_centroids <= 16) {
                int byte_idx = flat / 2;
                int nibble = (flat & 1)
                    ? ((indices[byte_idx] >> 4) & 0x0F)
                    : (indices[byte_idx] & 0x0F);
                w = codebook[nibble];
            } else {
                w = codebook[indices[flat]];
            }
            result += x[i] * w;
        }
        y[o] = result;
    }
}

#endif /* NN_OPS_INT4_H_ */
