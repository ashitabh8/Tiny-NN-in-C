// Minimal TensorFlow-like ops implemented in portable C for embedded targets.
// Layout: NHWC for tensors. Convolutions are standard cross-correlation (TF style).
// No dynamic allocation; callers provide output buffers.
// All functions use float for simplicity; adapt to fixed-point as needed.

#ifndef TF_OPS_H_
#define TF_OPS_H_

#include <math.h>
#include <stddef.h>

// Clamp helper
static inline int clamp_int(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Conv2D (NHWC input, HWIO filters)
// in:  [H, W, C_in]
// filt:[K_h, K_w, C_in, C_out]
// bias:[C_out]
// out: [H_out, W_out, C_out]
// Stride: (stride_h, stride_w), Padding: SAME or VALID (string)
static inline void conv2d_nhwc(
    const float* in, int in_h, int in_w, int in_c,
    const float* filt, int k_h, int k_w, int out_c,
    const float* bias,
    int stride_h, int stride_w,
    int pad_same, // 1 for SAME, 0 for VALID
    float* out)
{
    int out_h, out_w;
    if (pad_same) {
        out_h = (in_h + stride_h - 1) / stride_h;
        out_w = (in_w + stride_w - 1) / stride_w;
    } else {
        out_h = (in_h - k_h) / stride_h + 1;
        out_w = (in_w - k_w) / stride_w + 1;
    }

    int pad_h_total = pad_same ? ((out_h - 1) * stride_h + k_h - in_h) : 0;
    int pad_w_total = pad_same ? ((out_w - 1) * stride_w + k_w - in_w) : 0;
    int pad_top = pad_h_total / 2;
    int pad_left = pad_w_total / 2;

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int oc = 0; oc < out_c; ++oc) {
                float acc = bias ? bias[oc] : 0.0f;
                for (int kh = 0; kh < k_h; ++kh) {
                    int ih = oh * stride_h + kh - pad_top;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kw = 0; kw < k_w; ++kw) {
                        int iw = ow * stride_w + kw - pad_left;
                        if (iw < 0 || iw >= in_w) continue;
                        const float* in_px = in + ((ih * in_w + iw) * in_c);
                        const float* f_base = filt + (((kh * k_w + kw) * in_c) * out_c + oc);
                        for (int ic = 0; ic < in_c; ++ic) {
                            acc += in_px[ic] * f_base[ic * out_c];
                        }
                    }
                }
                out[((oh * out_w + ow) * out_c) + oc] = acc;
            }
        }
    }
}

// ReLU
static inline void relu(float* x, int n) {
    for (int i = 0; i < n; ++i) x[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

// Dense (fully connected): y = xW + b
// x: [in_features]
// W: [in_features, out_features] row-major (in_features major)
// b: [out_features]
static inline void dense(const float* x, int in_features,
                         const float* W, const float* b,
                         int out_features,
                         float* y) {
    for (int o = 0; o < out_features; ++o) {
        float acc = b ? b[o] : 0.0f;
        const float* w_col = W + o; // access W[i * out_features + o]
        for (int i = 0; i < in_features; ++i) {
            acc += x[i] * w_col[i * out_features];
        }
        y[o] = acc;
    }
}

// GlobalAveragePool2D over H and W for NHWC
// in: [H, W, C]
// out: [C]
static inline void global_average_pool_2d(const float* in, int h, int w, int c, float* out) {
    for (int ch = 0; ch < c; ++ch) out[ch] = 0.0f;
    const int n = h * w;
    for (int ih = 0; ih < h; ++ih) {
        for (int iw = 0; iw < w; ++iw) {
            const float* px = in + ((ih * w + iw) * c);
            for (int ch = 0; ch < c; ++ch) out[ch] += px[ch];
        }
    }
    const float inv = n > 0 ? 1.0f / (float)n : 0.0f;
    for (int ch = 0; ch < c; ++ch) out[ch] *= inv;
}

// Softmax over last dimension
static inline void softmax(float* x, int n) {
    if (n <= 0) return;
    float maxv = x[0];
    for (int i = 1; i < n; ++i) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }
    const float inv = sum > 0 ? 1.0f / sum : 0.0f;
    for (int i = 0; i < n; ++i) x[i] *= inv;
}

#endif // TF_OPS_H_


