"""
Float32 neural network operations implemented in Triton.

Layout: NHWC for tensors, HWIO for conv filters, [in, out] row-major for linear weights.
Matches semantics of src/c_ops/nn_ops_float.h.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Conv2D NHWC (HWIO filters)
# ---------------------------------------------------------------------------

@triton.jit
def _conv2d_nhwc_kernel(
    inp_ptr, filt_ptr, bias_ptr, out_ptr,
    in_h, in_w, in_c, out_h, out_w, out_c,
    k_h, k_w, stride_h, stride_w, pad_h, pad_w,
    has_bias: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid = tl.program_id(0)
    total_out = out_h * out_w
    oc_start = pid * BLOCK_OC
    oc_offs = oc_start + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < out_c

    for oh in range(out_h):
        for ow in range(out_w):
            for oc_base in range(0, out_c, BLOCK_OC):
                acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
                if has_bias:
                    acc = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)
                for kh in range(k_h):
                    ih = oh * stride_h + kh - pad_h
                    if ih < 0 or ih >= in_h:
                        continue
                    for kw in range(k_w):
                        iw = ow * stride_w + kw - pad_w
                        if iw < 0 or iw >= in_w:
                            continue
                        in_base = (ih * in_w + iw) * in_c
                        filt_khkw = (kh * k_w + kw) * in_c * out_c
                        for ic in range(in_c):
                            x = tl.load(inp_ptr + in_base + ic).to(tl.float32)
                            w = tl.load(
                                filt_ptr + filt_khkw + ic * out_c + oc_offs,
                                mask=oc_mask,
                                other=0.0,
                            ).to(tl.float32)
                            acc += x * w
                out_idx = (oh * out_w + ow) * out_c + oc_offs
                tl.store(out_ptr + out_idx, acc, mask=oc_mask)


def conv2d_nhwc(
    inp: torch.Tensor,
    in_h: int,
    in_w: int,
    in_c: int,
    filt: torch.Tensor,
    k_h: int,
    k_w: int,
    out_c: int,
    bias: Optional[torch.Tensor],
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    out: torch.Tensor,
) -> None:
    inp = inp.contiguous()
    filt = filt.contiguous()
    out_h = (in_h + 2 * pad_h - k_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - k_w) // stride_w + 1
    if bias is None:
        dummy = torch.zeros(1, device=inp.device, dtype=inp.dtype)
        _conv2d_nhwc_simple(inp, filt, dummy, out, in_h, in_w, in_c, out_h, out_w, out_c,
                            k_h, k_w, stride_h, stride_w, pad_h, pad_w, False)
    else:
        _conv2d_nhwc_simple(inp, filt, bias, out, in_h, in_w, in_c, out_h, out_w, out_c,
                            k_h, k_w, stride_h, stride_w, pad_h, pad_w, True)


@triton.jit
def _conv2d_nhwc_simple_kernel(
    inp_ptr, filt_ptr, bias_ptr, out_ptr,
    in_h, in_w, in_c, out_h, out_w, out_c,
    k_h, k_w, stride_h, stride_w, pad_h, pad_w,
    has_bias: tl.constexpr,
):
    """One program per output spatial location; loop over out channels."""
    pid = tl.program_id(0)
    if pid >= out_h * out_w:
        return
    oh = pid // out_w
    ow = pid % out_w
    for oc in range(out_c):
        acc = 0.0
        if has_bias:
            acc = tl.load(bias_ptr + oc).to(tl.float32)
        for kh in range(k_h):
            ih = oh * stride_h + kh - pad_h
            if ih >= 0:
                if ih < in_h:
                    for kw in range(k_w):
                        iw = ow * stride_w + kw - pad_w
                        if iw >= 0:
                            if iw < in_w:
                                in_base = (ih * in_w + iw) * in_c
                                filt_base = ((kh * k_w + kw) * in_c) * out_c + oc
                                for ic in range(in_c):
                                    x = tl.load(inp_ptr + in_base + ic).to(tl.float32)
                                    w = tl.load(filt_ptr + filt_base + ic * out_c).to(tl.float32)
                                    acc += x * w
        tl.store(out_ptr + (oh * out_w + ow) * out_c + oc, acc)


def _conv2d_nhwc_simple(inp, filt, bias, out, in_h, in_w, in_c, out_h, out_w, out_c,
                        k_h, k_w, sh, sw, ph, pw, has_bias):
    grid = (out_h * out_w,)
    _conv2d_nhwc_simple_kernel[grid](
        inp, filt, bias, out,
        in_h, in_w, in_c, out_h, out_w, out_c,
        k_h, k_w, sh, sw, ph, pw,
        has_bias=has_bias,
    )


# ---------------------------------------------------------------------------
# Depthwise Conv2D
# ---------------------------------------------------------------------------

@triton.jit
def _depthwise_conv2d_nhwc_kernel(
    inp_ptr, filt_ptr, bias_ptr, out_ptr,
    in_h, in_w, channels, out_h, out_w,
    k_h, k_w, stride_h, stride_w, pad_h, pad_w,
    has_bias: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= out_h * out_w * channels:
        return
    c = pid % channels
    tmp = pid // channels
    ow = tmp % out_w
    oh = tmp // out_w
    acc = 0.0
    if has_bias:
        acc = tl.load(bias_ptr + c).to(tl.float32)
    for kh in range(k_h):
        ih = oh * stride_h + kh - pad_h
        if ih >= 0:
            if ih < in_h:
                for kw in range(k_w):
                    iw = ow * stride_w + kw - pad_w
                    if iw >= 0:
                        if iw < in_w:
                            x = tl.load(inp_ptr + ((ih * in_w + iw) * channels) + c).to(tl.float32)
                            w = tl.load(filt_ptr + ((kh * k_w + kw) * channels) + c).to(tl.float32)
                            acc += x * w
    tl.store(out_ptr + ((oh * out_w + ow) * channels) + c, acc)


def depthwise_conv2d_nhwc(
    inp: torch.Tensor,
    in_h: int,
    in_w: int,
    channels: int,
    filt: torch.Tensor,
    k_h: int,
    k_w: int,
    bias: Optional[torch.Tensor],
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    out: torch.Tensor,
) -> None:
    inp = inp.contiguous()
    filt = filt.contiguous()
    out_h = (in_h + 2 * pad_h - k_h) // stride_h + 1
    out_w = (in_w + 2 * pad_w - k_w) // stride_w + 1
    has_bias = bias is not None
    if not has_bias:
        bias = torch.zeros(1, device=inp.device, dtype=inp.dtype)
    grid = (out_h * out_w * channels,)
    _depthwise_conv2d_nhwc_kernel[grid](
        inp, filt, bias, out,
        in_h, in_w, channels, out_h, out_w,
        k_h, k_w, stride_h, stride_w, pad_h, pad_w,
        has_bias=has_bias,
    )


# ---------------------------------------------------------------------------
# Dense (linear)
# ---------------------------------------------------------------------------

@triton.jit
def _dense_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    in_features, out_features,
    has_bias: tl.constexpr,
):
    o = tl.program_id(0)
    if o >= out_features:
        return
    acc = 0.0
    if has_bias:
        acc = tl.load(b_ptr + o).to(tl.float32)
    for i in range(in_features):
        x = tl.load(x_ptr + i).to(tl.float32)
        w = tl.load(w_ptr + i * out_features + o).to(tl.float32)
        acc += x * w
    tl.store(y_ptr + o, acc)


def dense(
    x: torch.Tensor,
    in_features: int,
    W: torch.Tensor,
    b: Optional[torch.Tensor],
    out_features: int,
    y: torch.Tensor,
) -> None:
    x = x.contiguous()
    W = W.contiguous()
    has_bias = b is not None
    if not has_bias:
        b = torch.zeros(1, device=x.device, dtype=x.dtype)
    grid = (out_features,)
    _dense_kernel[grid](x, W, b, y, in_features, out_features, has_bias=has_bias)


# ---------------------------------------------------------------------------
# Elementwise / activation
# ---------------------------------------------------------------------------

@triton.jit
def _relu_kernel(x_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(x_ptr + offs, tl.maximum(x, 0.0), mask=mask)


def relu(x: torch.Tensor, n: int) -> None:
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    _relu_kernel[grid](x, n, BLOCK=BLOCK)


@triton.jit
def _add_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=mask)


def add_tensors(a: torch.Tensor, b: torch.Tensor, n: int, out: torch.Tensor) -> None:
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    _add_kernel[grid](a, b, out, n, BLOCK=BLOCK)


@triton.jit
def _mul_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    va = tl.load(a_ptr + offs, mask=mask, other=0.0)
    vb = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, va * vb, mask=mask)


def mul_tensors(a: torch.Tensor, b: torch.Tensor, n: int, out: torch.Tensor) -> None:
    BLOCK = 256
    grid = (triton.cdiv(n, BLOCK),)
    _mul_kernel[grid](a, b, out, n, BLOCK=BLOCK)


# ---------------------------------------------------------------------------
# BatchNorm2D NHWC
# ---------------------------------------------------------------------------

@triton.jit
def _batchnorm2d_nhwc_kernel(
    inp_ptr, out_ptr, gamma_ptr, beta_ptr, mean_ptr, var_ptr,
    h, w, c, eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    hw = h * w
    if pid >= hw:
        return
    ih = pid // w
    iw = pid % w
    in_base = (ih * w + iw) * c
    out_base = in_base
    for ch_start in range(0, c, BLOCK):
        ch_offs = ch_start + tl.arange(0, BLOCK)
        mask = ch_offs < c
        x = tl.load(inp_ptr + in_base + ch_offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.load(mean_ptr + ch_offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.load(var_ptr + ch_offs, mask=mask, other=0.0).to(tl.float32)
        gamma = tl.load(gamma_ptr + ch_offs, mask=mask, other=1.0).to(tl.float32)
        beta = tl.load(beta_ptr + ch_offs, mask=mask, other=0.0).to(tl.float32)
        norm = (x - mean) / tl.sqrt(var + eps)
        tl.store(out_ptr + out_base + ch_offs, gamma * norm + beta, mask=mask)


def batchnorm2d_nhwc(
    inp: torch.Tensor,
    h: int,
    w: int,
    c: int,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    mean: torch.Tensor,
    var: torch.Tensor,
    eps: float,
    out: torch.Tensor,
) -> None:
    BLOCK = 64
    grid = (h * w,)
    _batchnorm2d_nhwc_kernel[grid](
        inp, out, gamma, beta, mean, var, h, w, c, eps, BLOCK=BLOCK
    )


# ---------------------------------------------------------------------------
# Softmax (flat buffer)
# ---------------------------------------------------------------------------

@triton.jit
def _softmax_kernel(x_ptr, n, BLOCK: tl.constexpr):
    """Single-block softmax over n elements (matches C reference for small n)."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=-float("inf"))
    maxv = tl.max(x, axis=0)
    x = tl.exp(x - maxv)
    sumv = tl.sum(x, axis=0)
    x = x / sumv
    tl.store(x_ptr + offs, x, mask=mask)


def softmax(x: torch.Tensor, n: int) -> None:
    BLOCK = triton.next_power_of_2(max(n, 1))
    BLOCK = min(BLOCK, 4096)
    _softmax_kernel[(1,)](x, n, BLOCK=BLOCK)


# ---------------------------------------------------------------------------
# Pooling / mean
# ---------------------------------------------------------------------------

@triton.jit
def _mean_hwc_kernel(inp_ptr, out_ptr, h, w, c, inv_n):
    ch = tl.program_id(0)
    if ch >= c:
        return
    acc = 0.0
    for ih in range(h):
        for iw in range(w):
            acc += tl.load(inp_ptr + ((ih * w + iw) * c) + ch).to(tl.float32)
    tl.store(out_ptr + ch, acc * inv_n)


def mean_hwc(inp: torch.Tensor, h: int, w: int, c: int, out: torch.Tensor) -> None:
    inv_n = 1.0 / float(h * w) if h * w > 0 else 0.0
    _mean_hwc_kernel[(c,)](inp, out, h, w, c, inv_n)


def global_average_pool_2d(inp: torch.Tensor, h: int, w: int, c: int, out: torch.Tensor) -> None:
    mean_hwc(inp, h, w, c, out)


def adaptive_avg_pool_2d_1x1(inp: torch.Tensor, in_h: int, in_w: int, in_c: int, out: torch.Tensor) -> None:
    global_average_pool_2d(inp, in_h, in_w, in_c, out)


@triton.jit
def _mean_last_dim_kernel(inp_ptr, out_ptr, rows, cols):
    r = tl.program_id(0)
    if r >= rows:
        return
    acc = 0.0
    base = r * cols
    for c in range(cols):
        acc += tl.load(inp_ptr + base + c).to(tl.float32)
    inv = 1.0 / cols if cols > 0 else 0.0
    tl.store(out_ptr + r, acc * inv)


def mean_last_dim(inp: torch.Tensor, rows: int, cols: int, out: torch.Tensor) -> None:
    _mean_last_dim_kernel[(rows,)](inp, out, rows, cols)


# ---------------------------------------------------------------------------
# Permute
# ---------------------------------------------------------------------------

@triton.jit
def _permute_3d_kernel(inp_ptr, out_ptr, d0, d1, d2, p0, p1, p2):
    pid = tl.program_id(0)
    total = d0 * d1 * d2
    if pid >= total:
        return
    i2 = pid % d2
    tmp = pid // d2
    i1 = tmp % d1
    i0 = tmp // d1
    out_i0 = tl.where(p0 == 0, i0, tl.where(p0 == 1, i1, i2))
    out_i1 = tl.where(p1 == 0, i0, tl.where(p1 == 1, i1, i2))
    out_i2 = tl.where(p2 == 0, i0, tl.where(p2 == 1, i1, i2))
    dims0 = tl.where(p0 == 0, d0, tl.where(p0 == 1, d1, d2))
    dims1 = tl.where(p1 == 0, d0, tl.where(p1 == 1, d1, d2))
    dims2 = tl.where(p2 == 0, d0, tl.where(p2 == 1, d1, d2))
    in_s0 = d1 * d2
    in_s1 = d2
    out_s0 = dims1 * dims2
    out_s1 = dims2
    idx_in = i0 * in_s0 + i1 * in_s1 + i2
    idx_out = out_i0 * out_s0 + out_i1 * out_s1 + out_i2
    tl.store(out_ptr + idx_out, tl.load(inp_ptr + idx_in))


def permute_3d(
    inp: torch.Tensor, d0: int, d1: int, d2: int,
    p0: int, p1: int, p2: int, out: torch.Tensor,
) -> None:
    total = d0 * d1 * d2
    _permute_3d_kernel[(total,)](inp, out, d0, d1, d2, p0, p1, p2)


@triton.jit
def _permute_4d_kernel(
    inp_ptr, out_ptr,
    d0, d1, d2, d3,
    p0, p1, p2, p3,
):
    pid = tl.program_id(0)
    total = d0 * d1 * d2 * d3
    if pid >= total:
        return
    i3 = pid % d3
    t = pid // d3
    i2 = t % d2
    t = t // d2
    i1 = t % d1
    i0 = t // d1

    def map_axis(p, i0, i1, i2, i3):
        return tl.where(p == 0, i0, tl.where(p == 1, i1, tl.where(p == 2, i2, i3)))

    o0 = map_axis(p0, i0, i1, i2, i3)
    o1 = map_axis(p1, i0, i1, i2, i3)
    o2 = map_axis(p2, i0, i1, i2, i3)
    o3 = map_axis(p3, i0, i1, i2, i3)

    def dim_at(p):
        return tl.where(p == 0, d0, tl.where(p == 1, d1, tl.where(p == 2, d2, d3)))

    od0, od1, od2, od3 = dim_at(p0), dim_at(p1), dim_at(p2), dim_at(p3)
    in_s0, in_s1, in_s2 = d1 * d2 * d3, d2 * d3, d3
    out_s0, out_s1, out_s2 = od1 * od2 * od3, od2 * od3, od3
    idx_in = i0 * in_s0 + i1 * in_s1 + i2 * in_s2 + i3
    idx_out = o0 * out_s0 + o1 * out_s1 + o2 * out_s2 + o3
    tl.store(out_ptr + idx_out, tl.load(inp_ptr + idx_in))


def permute_4d(
    inp: torch.Tensor,
    d0: int, d1: int, d2: int, d3: int,
    p0: int, p1: int, p2: int, p3: int,
    out: torch.Tensor,
) -> None:
    total = d0 * d1 * d2 * d3
    _permute_4d_kernel[(total,)](inp, out, d0, d1, d2, d3, p0, p1, p2, p3)


def flatten(src: torch.Tensor, n: int, dst: torch.Tensor) -> None:
    dst.copy_(src.reshape(-1)[:n])
