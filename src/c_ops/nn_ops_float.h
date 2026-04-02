// Minimal TensorFlow-like ops implemented in portable C for embedded targets.
// Layout: NHWC for tensors. Convolutions are standard cross-correlation (TF style).
// No dynamic allocation; callers provide output buffers.
// All functions use float for simplicity; adapt to fixed-point as needed.

#ifndef NN_OPS_FLOAT_H_
#define NN_OPS_FLOAT_H_

#include <math.h>
#include <stddef.h>

// Clamp helper
static inline int clamp_int(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Conv2D (NHWC input, HWIO filters) — matches PyTorch Conv2d padding
// in:  [H, W, C_in]
// filt:[K_h, K_w, C_in, C_out]
// bias:[C_out]
// out: [H_out, W_out, C_out]
// pad_h, pad_w: zero-padding added to each side (height/width), same as nn.Conv2d padding=
static inline void conv2d_nhwc(
    const float* in, int in_h, int in_w, int in_c,
    const float* filt, int k_h, int k_w, int out_c,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float* out)
{
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int oc = 0; oc < out_c; ++oc) {
                float acc = bias ? bias[oc] : 0.0f;
                for (int kh = 0; kh < k_h; ++kh) {
                    int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kw = 0; kw < k_w; ++kw) {
                        int iw = ow * stride_w + kw - pad_w;
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

// Depthwise Conv2D (NHWC input, HWC filters)
// in:   [H, W, C]
// filt: [K_h, K_w, C] (one filter per channel)
// bias: [C] (optional)
// out:  [H_out, W_out, C]
static inline void depthwise_conv2d_nhwc(
    const float* in, int in_h, int in_w, int channels,
    const float* filt, int k_h, int k_w,
    const float* bias,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float* out)
{
    int out_h = (in_h + 2 * pad_h - k_h) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - k_w) / stride_w + 1;

    for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
            for (int c = 0; c < channels; ++c) {
                float acc = bias ? bias[c] : 0.0f;
                for (int kh = 0; kh < k_h; ++kh) {
                    int ih = oh * stride_h + kh - pad_h;
                    if (ih < 0 || ih >= in_h) continue;
                    for (int kw = 0; kw < k_w; ++kw) {
                        int iw = ow * stride_w + kw - pad_w;
                        if (iw < 0 || iw >= in_w) continue;
                        const float x = in[((ih * in_w + iw) * channels) + c];
                        const float w = filt[((kh * k_w + kw) * channels) + c];
                        acc += x * w;
                    }
                }
                out[((oh * out_w + ow) * channels) + c] = acc;
            }
        }
    }
}

// Generic 4D permute for contiguous float buffers.
// Input shape [D0, D1, D2, D3], output shape [Dperm0, Dperm1, Dperm2, Dperm3].
// perm entries must be a permutation of {0,1,2,3}.
static inline void permute_4d(
    const float* in,
    int d0, int d1, int d2, int d3,
    int p0, int p1, int p2, int p3,
    float* out)
{
    int dims[4] = {d0, d1, d2, d3};
    int out_d0 = dims[p0];
    int out_d1 = dims[p1];
    int out_d2 = dims[p2];
    int out_d3 = dims[p3];

    int in_s0 = d1 * d2 * d3;
    int in_s1 = d2 * d3;
    int in_s2 = d3;

    int out_s0 = out_d1 * out_d2 * out_d3;
    int out_s1 = out_d2 * out_d3;
    int out_s2 = out_d3;

    for (int i0 = 0; i0 < out_d0; ++i0) {
        for (int i1 = 0; i1 < out_d1; ++i1) {
            for (int i2 = 0; i2 < out_d2; ++i2) {
                for (int i3 = 0; i3 < out_d3; ++i3) {
                    int idx_out = i0 * out_s0 + i1 * out_s1 + i2 * out_s2 + i3;

                    int src[4];
                    src[p0] = i0;
                    src[p1] = i1;
                    src[p2] = i2;
                    src[p3] = i3;

                    int idx_in = src[0] * in_s0 + src[1] * in_s1 + src[2] * in_s2 + src[3];
                    out[idx_out] = in[idx_in];
                }
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

// AdaptiveAvgPool2d((1,1)): same as global average pool over H,W for NHWC
// in:  [in_h, in_w, in_c]
// out: [in_c]
static inline void adaptive_avg_pool_2d_1x1(const float* in, int in_h, int in_w, int in_c, float* out) {
    global_average_pool_2d(in, in_h, in_w, in_c, out);
}

// Flatten: copy n floats from src to dst (for view/flatten after pooling)
static inline void flatten(const float* src, int n, float* dst) {
    for (int i = 0; i < n; ++i) dst[i] = src[i];
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

// Mean reduction over spatial dimensions (H, W) for NHWC
// Used for global average pooling or mean(dim=[2,3]) in PyTorch
// in:  [H, W, C] - 3D tensor in NHWC format (batch dimension removed)
// out: [C] - 1D tensor with channel means
// This is equivalent to global_average_pool_2d but named for clarity
static inline void mean_hwc(const float* in, int h, int w, int c, float* out) {
    const int n = h * w;
    // Initialize output to zero
    for (int ch = 0; ch < c; ++ch) {
        out[ch] = 0.0f;
    }
    // Sum over all spatial positions
    for (int ih = 0; ih < h; ++ih) {
        for (int iw = 0; iw < w; ++iw) {
            const float* px = in + ((ih * w + iw) * c);
            for (int ch = 0; ch < c; ++ch) {
                out[ch] += px[ch];
            }
        }
    }
    // Divide by number of elements
    const float inv = n > 0 ? 1.0f / (float)n : 0.0f;
    for (int ch = 0; ch < c; ++ch) {
        out[ch] *= inv;
    }
}

// Mean over the last dimension of a 2D tensor [rows, cols] -> [rows].
static inline void mean_last_dim(const float* in, int rows, int cols, float* out) {
    for (int r = 0; r < rows; ++r) {
        float acc = 0.0f;
        const float* row = in + r * cols;
        for (int c = 0; c < cols; ++c) {
            acc += row[c];
        }
        out[r] = (cols > 0) ? (acc / (float)cols) : 0.0f;
    }
}

// BatchNorm2D for NHWC
// in:  [H, W, C]
// gamma: [C] (scale)
// beta:  [C] (bias)
// mean:  [C] (running mean)
// var:   [C] (running variance)
// out:  [H, W, C]
static inline void batchnorm2d_nhwc(
    const float* in, int h, int w, int c,
    const float* gamma, const float* beta,
    const float* mean, const float* var,
    float eps,
    float* out)
{
    const int n = h * w;
    for (int ih = 0; ih < h; ++ih) {
        for (int iw = 0; iw < w; ++iw) {
            const float* in_px = in + ((ih * w + iw) * c);
            float* out_px = out + ((ih * w + iw) * c);
            for (int ch = 0; ch < c; ++ch) {
                float normalized = (in_px[ch] - mean[ch]) / sqrtf(var[ch] + eps);
                out_px[ch] = gamma[ch] * normalized + beta[ch];
            }
        }
    }
}

#endif // NN_OPS_FLOAT_H_


