/*
 * Unified affine dense matmul template (Phase 3).
 *
 * One code path covers PER_TENSOR / PER_CHANNEL / PER_GROUP via scale indexing:
 *   scale_cols = per_out_column ? out_features : 1
 *   col        = per_out_column ? o : 0
 *   ws         = weight_scales[g * scale_cols + col]
 *
 * Reductions:
 *   PER_TENSOR:  group_size == in_features, per_out_column == 0  => always scales[0]
 *   PER_CHANNEL: group_size == in_features, per_out_column == 1  => scales[o]
 *   PER_GROUP:   group_size == g,           per_out_column == 1  => scales[g*out+o]
 *
 * Instantiations live in nn_ops_int{8,16,4}.h after their quantize/unpack helpers.
 */

#ifndef NN_OPS_AFFINE_DENSE_H_
#define NN_OPS_AFFINE_DENSE_H_

#include <stdint.h>
#include <stddef.h>

#define NN_LOAD_W_DIRECT(W, idx, o, out_features) \
    ((int64_t)(W)[(idx) * (out_features) + (o)])

#define NN_LOAD_W_INT4(W, idx, o, out_features) \
    ((int64_t)unpack_int4_at((W), (idx) * (out_features) + (o)))

#define NN_STORE_INT8(y, o, result, output_scale, output_zp) \
    ((y)[(o)] = quantize_scalar_int8((result), (output_scale), (output_zp)))

#define NN_STORE_INT16(y, o, result, output_scale, output_zp) \
    ((y)[(o)] = quantize_float_to_int16_scalar((result), (output_scale), (output_zp)))

#define NN_STORE_FLOAT(y, o, result, output_scale, output_zp) \
    ((y)[(o)] = (result))

/*
 * Static / ZP-aware instantiation.
 * Symmetric fast path when a_symmetric && w_symmetric; else four-term ZP.
 */
#define NN_DENSE_AFFINE_DEFINE_ZP(NAME, ACT_T, W_T, LOAD_WV, OUT_T, STORE_Y)  \
static inline void NAME(                                                      \
    const ACT_T* x,                                                           \
    int in_features,                                                          \
    const W_T* W,                                                             \
    const float* b,                                                           \
    int out_features,                                                         \
    int group_size,                                                           \
    int per_out_column,                                                       \
    float input_scale,                                                        \
    const float* weight_scales,                                               \
    float output_scale,                                                       \
    int input_zp,                                                             \
    int weight_zp,                                                            \
    int output_zp,                                                            \
    int a_symmetric,                                                          \
    int w_symmetric,                                                          \
    OUT_T* y)                                                                 \
{                                                                             \
    int num_groups = in_features / group_size;                                \
    int scale_cols = per_out_column ? out_features : 1;                       \
    int use_sym = a_symmetric && w_symmetric;                                 \
    for (int o = 0; o < out_features; ++o) {                                  \
        float result = 0.0f;                                                  \
        int col = per_out_column ? o : 0;                                     \
        for (int g = 0; g < num_groups; ++g) {                                \
            int64_t acc = 0;                                                  \
            int base = g * group_size;                                        \
            if (use_sym) {                                                    \
                for (int i = 0; i < group_size; ++i) {                        \
                    int idx = base + i;                                       \
                    int64_t wv = LOAD_WV(W, idx, o, out_features);            \
                    acc += (int64_t)x[idx] * wv;                              \
                }                                                             \
                result += (float)acc * input_scale                            \
                          * weight_scales[g * scale_cols + col];              \
            } else {                                                          \
                int64_t sum_qx = 0;                                           \
                int64_t sum_qw = 0;                                           \
                for (int i = 0; i < group_size; ++i) {                        \
                    int idx = base + i;                                       \
                    int64_t wv = LOAD_WV(W, idx, o, out_features);            \
                    acc += (int64_t)x[idx] * wv;                              \
                    sum_qx += (int64_t)x[idx];                                \
                    sum_qw += wv;                                             \
                }                                                             \
                int64_t zp_term = (int64_t)input_zp * (int64_t)weight_zp       \
                                  * (int64_t)group_size;                      \
                int64_t dot_affine = acc - (int64_t)weight_zp * sum_qx        \
                                     - (int64_t)input_zp * sum_qw + zp_term;  \
                result += (float)dot_affine * input_scale                     \
                          * weight_scales[g * scale_cols + col];              \
            }                                                                 \
        }                                                                     \
        if (b != NULL) {                                                      \
            result += b[o];                                                   \
        }                                                                     \
        STORE_Y(y, o, result, output_scale, output_zp);                       \
    }                                                                         \
}

/* Dynamic / to_float instantiation — always symmetric (no ZP args). */
#define NN_DENSE_AFFINE_DEFINE_TO_FLOAT(NAME, ACT_T, W_T, LOAD_WV)            \
static inline void NAME(                                                      \
    const ACT_T* x,                                                           \
    int in_features,                                                          \
    const W_T* W,                                                             \
    const float* b,                                                           \
    int out_features,                                                         \
    int group_size,                                                           \
    int per_out_column,                                                       \
    float input_scale,                                                        \
    const float* weight_scales,                                               \
    float* y)                                                                 \
{                                                                             \
    int num_groups = in_features / group_size;                                \
    int scale_cols = per_out_column ? out_features : 1;                       \
    for (int o = 0; o < out_features; ++o) {                                  \
        float result = 0.0f;                                                  \
        int col = per_out_column ? o : 0;                                     \
        for (int g = 0; g < num_groups; ++g) {                                \
            int64_t acc = 0;                                                  \
            int base = g * group_size;                                        \
            for (int i = 0; i < group_size; ++i) {                            \
                int idx = base + i;                                           \
                int64_t wv = LOAD_WV(W, idx, o, out_features);                \
                acc += (int64_t)x[idx] * wv;                                  \
            }                                                                 \
            result += (float)acc * input_scale                                \
                      * weight_scales[g * scale_cols + col];                  \
        }                                                                     \
        if (b != NULL) {                                                      \
            result += b[o];                                                   \
        }                                                                     \
        y[o] = result;                                                        \
    }                                                                         \
}

#endif /* NN_OPS_AFFINE_DENSE_H_ */
