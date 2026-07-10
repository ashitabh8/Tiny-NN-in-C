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

/**
 * Dense: int8 activations + packed int4 weights (per-group scales) -> int8.
 * packed_w: int8 array with 2 int4 weights per byte (layout matches float int4 packing).
 * weight_scales layout: [num_groups * out_features], index g*out_features + o.
 */
static inline void dense_int8_int4w_per_group(
    const int8_t* x,
    int in_features,
    const int8_t* packed_w,
    int weight_count,
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
    (void)weight_count;
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
                int flat = idx * out_features + o;
                int64_t wv = (int64_t)unpack_int4_at(packed_w, flat);
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

/**
 * Dense: int8 activations + packed int4 weights (per-group scales) -> float32.
 * Used by dynamic quantization (symmetric, weight_zp assumed 0).
 */
static inline void dense_int8_int4w_per_group_to_float(
    const int8_t* x,
    int in_features,
    const int8_t* packed_w,
    int weight_count,
    const float* b,
    int out_features,
    int group_size,
    float input_scale,
    const float* weight_scales,
    float* y)
{
    (void)weight_count;
    int num_groups = in_features / group_size;
    for (int o = 0; o < out_features; ++o) {
        float result = 0.0f;
        for (int g = 0; g < num_groups; ++g) {
            int64_t acc = 0;
            int base = g * group_size;
            for (int i = 0; i < group_size; ++i) {
                int idx = base + i;
                int flat = idx * out_features + o;
                int64_t wv = (int64_t)unpack_int4_at(packed_w, flat);
                acc += (int64_t)x[idx] * wv;
            }
            result += (float)acc * input_scale
                      * weight_scales[g * out_features + o];
        }
        if (b != NULL) {
            result += b[o];
        }
        y[o] = result;
    }
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
