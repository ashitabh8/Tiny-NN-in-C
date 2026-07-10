"""
Quantized neural network operations in Triton (int8 / int16).

Arithmetic matches src/c_ops/nn_ops_int8.h and nn_ops_int16.h.
Bias stays float32.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

_QMAX_INT8 = 127
_QMAX_INT16 = 32767

# Tile alignment for per-group quantization (shared with compiler)
BLOCK_K = 32


def _qmax(dtype: str) -> int:
    return _QMAX_INT8 if dtype == "int8" else _QMAX_INT16


def _torch_dtype(dtype: str):
    return torch.int8 if dtype == "int8" else torch.int16


# ---------------------------------------------------------------------------
# Quantize / dequantize helpers
# ---------------------------------------------------------------------------

@triton.jit
def _quantize_float_kernel(inp_ptr, out_ptr, size, scale, offset, qmin, qmax, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    q = tl.floor(x / scale + offset + 0.5)
    q = tl.minimum(tl.maximum(q, qmin), qmax)
    tl.store(out_ptr + offs, q, mask=mask)


@triton.jit
def _dequantize_kernel(inp_ptr, out_ptr, size, scale, offset, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    q = tl.load(inp_ptr + offs, mask=mask, other=0).to(tl.float32)
    tl.store(out_ptr + offs, scale * (q - offset), mask=mask)


def quantize_float_to_int8(
    inp: torch.Tensor, size: int, scale: float, offset: int, out: torch.Tensor
) -> None:
    BLOCK = 256
    grid = (triton.cdiv(size, BLOCK),)
    _quantize_float_kernel[grid](
        inp, out, size, scale, offset, -128, 127, BLOCK=BLOCK
    )


def quantize_float_to_int16(
    inp: torch.Tensor, size: int, scale: float, offset: int, out: torch.Tensor
) -> None:
    BLOCK = 256
    grid = (triton.cdiv(size, BLOCK),)
    _quantize_float_kernel[grid](
        inp, out, size, scale, offset, -32768, 32767, BLOCK=BLOCK
    )


def dequantize_int8_to_float(
    inp: torch.Tensor, size: int, scale: float, offset: int, out: torch.Tensor
) -> None:
    BLOCK = 256
    grid = (triton.cdiv(size, BLOCK),)
    _dequantize_kernel[grid](inp, out, size, scale, offset, BLOCK=BLOCK)


def dequantize_int16_to_float(
    inp: torch.Tensor, size: int, scale: float, offset: int, out: torch.Tensor
) -> None:
    BLOCK = 256
    grid = (triton.cdiv(size, BLOCK),)
    _dequantize_kernel[grid](inp, out, size, scale, offset, BLOCK=BLOCK)


@triton.jit
def _dynamic_scale_kernel(inp_ptr, scale_ptr, size, qmax, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < size
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    block_max = tl.max(tl.abs(x), axis=0)
    tl.atomic_max(scale_ptr, block_max / qmax)


def compute_dynamic_scale_int8(inp: torch.Tensor, size: int) -> torch.Tensor:
    scale_buf = torch.zeros(1, device=inp.device, dtype=torch.float32)
    scale_buf[0] = 0.0
    BLOCK = 256
    grid = (triton.cdiv(size, BLOCK),)
    _dynamic_scale_kernel[grid](inp, scale_buf, size, 127.0, BLOCK=BLOCK)
    s = scale_buf[0].item()
    return 1.0 / 127.0 if s == 0.0 else s


def compute_dynamic_scale_int16(inp: torch.Tensor, size: int) -> torch.Tensor:
    scale_buf = torch.zeros(1, device=inp.device, dtype=torch.float32)
    scale_buf[0] = 0.0
    BLOCK = 256
    grid = (triton.cdiv(size, BLOCK),)
    _dynamic_scale_kernel[grid](inp, scale_buf, size, 32767.0, BLOCK=BLOCK)
    s = scale_buf[0].item()
    return 1.0 / 32767.0 if s == 0.0 else s


# ---------------------------------------------------------------------------
# Dense int8 / int16 (static quant, affine ZP)
# ---------------------------------------------------------------------------

@triton.jit
def _dense_quant_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    in_features, out_features,
    input_scale, weight_scale, output_scale,
    input_zp, weight_zp, output_zp,
    has_bias: tl.constexpr,
    use_int16: tl.constexpr,
    qmin, qmax,
):
    o = tl.program_id(0)
    if o >= out_features:
        return
    acc = 0
    sum_qx = 0
    sum_qw = 0
    for i in range(in_features):
        qx = tl.load(x_ptr + i).to(tl.int32)
        qw = tl.load(w_ptr + i * out_features + o).to(tl.int32)
        acc += qx * qw
        sum_qx += qx
        sum_qw += qw
    zp_term = input_zp * weight_zp * in_features
    dot_affine = acc - weight_zp * sum_qx - input_zp * sum_qw + zp_term
    result = dot_affine.to(tl.float32) * input_scale * weight_scale
    if has_bias:
        result += tl.load(b_ptr + o).to(tl.float32)
    q = tl.floor(result / output_scale + output_zp + 0.5)
    q = tl.minimum(tl.maximum(q, qmin), qmax)
    tl.store(y_ptr + o, q)


@triton.jit
def _dense_quant_per_channel_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, ws_ptr,
    in_features, out_features,
    input_scale, output_scale,
    input_zp, weight_zp, output_zp,
    has_bias: tl.constexpr,
    qmin, qmax,
):
    o = tl.program_id(0)
    if o >= out_features:
        return
    w_scale = tl.load(ws_ptr + o).to(tl.float32)
    acc = 0
    sum_qx = 0
    sum_qw = 0
    for i in range(in_features):
        qx = tl.load(x_ptr + i).to(tl.int32)
        qw = tl.load(w_ptr + i * out_features + o).to(tl.int32)
        acc += qx * qw
        sum_qx += qx
        sum_qw += qw
    zp_term = input_zp * weight_zp * in_features
    dot_affine = acc - weight_zp * sum_qx - input_zp * sum_qw + zp_term
    result = dot_affine.to(tl.float32) * input_scale * w_scale
    if has_bias:
        result += tl.load(b_ptr + o).to(tl.float32)
    q = tl.floor(result / output_scale + output_zp + 0.5)
    q = tl.minimum(tl.maximum(q, qmin), qmax)
    tl.store(y_ptr + o, q)


@triton.jit
def _dense_quant_to_float_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    in_features, out_features,
    input_scale, weight_scale,
    has_bias: tl.constexpr,
):
    o = tl.program_id(0)
    if o >= out_features:
        return
    acc = 0
    for i in range(in_features):
        qx = tl.load(x_ptr + i).to(tl.int64)
        qw = tl.load(w_ptr + i * out_features + o).to(tl.int64)
        acc += qx * qw
    result = acc.to(tl.float32) * input_scale * weight_scale
    if has_bias:
        result += tl.load(b_ptr + o).to(tl.float32)
    tl.store(y_ptr + o, result)


def _dense_quant(
    x, W, b, y, in_features, out_features,
    input_scale, weight_scale, output_scale,
    input_zp, weight_zp, output_zp,
    int_dtype: str,
):
    has_bias = b is not None
    if not has_bias:
        b = torch.zeros(1, device=x.device, dtype=torch.float32)
    qmin = -128 if int_dtype == "int8" else -32768
    qmax = 127 if int_dtype == "int8" else 32767
    use_int16 = int_dtype == "int16"
    _dense_quant_kernel[(out_features,)](
        x, W, b, y, in_features, out_features,
        input_scale, weight_scale, output_scale,
        input_zp, weight_zp, output_zp,
        has_bias=has_bias, use_int16=use_int16, qmin=qmin, qmax=qmax,
    )


def dense_int8(x, in_features, W, b, out_features, input_scale, weight_scale,
                 output_scale, input_zp, weight_zp, output_zp, y):
    _dense_quant(x, W, b, y, in_features, out_features, input_scale, weight_scale,
                 output_scale, input_zp, weight_zp, output_zp, "int8")


def dense_int16(x, in_features, W, b, out_features, input_scale, weight_scale,
                  output_scale, input_zp, weight_zp, output_zp, y):
    _dense_quant(x, W, b, y, in_features, out_features, input_scale, weight_scale,
                 output_scale, input_zp, weight_zp, output_zp, "int16")


def dense_int8_per_channel(x, in_features, W, b, out_features, input_scale, weight_scales,
                           output_scale, input_zp, weight_zp, output_zp, y):
    has_bias = b is not None
    if not has_bias:
        b = torch.zeros(1, device=x.device, dtype=torch.float32)
    _dense_quant_per_channel_kernel[(out_features,)](
        x, W, b, y, weight_scales, in_features, out_features,
        input_scale, output_scale, input_zp, weight_zp, output_zp,
        has_bias=has_bias, qmin=-128, qmax=127,
    )


def dense_int16_per_channel(x, in_features, W, b, out_features, input_scale, weight_scales,
                            output_scale, input_zp, weight_zp, output_zp, y):
    has_bias = b is not None
    if not has_bias:
        b = torch.zeros(1, device=x.device, dtype=torch.float32)
    _dense_quant_per_channel_kernel[(out_features,)](
        x, W, b, y, weight_scales, in_features, out_features,
        input_scale, output_scale, input_zp, weight_zp, output_zp,
        has_bias=has_bias, qmin=-32768, qmax=32767,
    )


@triton.jit
def _dense_quant_per_group_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, ws_ptr,
    in_features, out_features, group_size, num_groups,
    input_scale, output_scale,
    input_zp, weight_zp, output_zp,
    has_bias: tl.constexpr,
    qmin, qmax,
):
    o = tl.program_id(0)
    if o >= out_features:
        return
    result = 0.0
    for g in range(num_groups):
        acc = 0
        sum_qx = 0
        sum_qw = 0
        base = g * group_size
        for i in range(group_size):
            idx = base + i
            qx = tl.load(x_ptr + idx).to(tl.int32)
            qw = tl.load(w_ptr + idx * out_features + o).to(tl.int32)
            acc += qx * qw
            sum_qx += qx
            sum_qw += qw
        zp_term = input_zp * weight_zp * group_size
        dot_affine = acc - weight_zp * sum_qx - input_zp * sum_qw + zp_term
        w_scale = tl.load(ws_ptr + g * out_features + o).to(tl.float32)
        result += dot_affine.to(tl.float32) * input_scale * w_scale
    if has_bias:
        result += tl.load(b_ptr + o).to(tl.float32)
    q = tl.floor(result / output_scale + output_zp + 0.5)
    q = tl.minimum(tl.maximum(q, qmin), qmax)
    tl.store(y_ptr + o, q)


def dense_int8_per_group(x, in_features, W, b, out_features, group_size, input_scale,
                         weight_scales, output_scale, input_zp, weight_zp, output_zp, y):
    has_bias = b is not None
    if not has_bias:
        b = torch.zeros(1, device=x.device, dtype=torch.float32)
    num_groups = in_features // group_size
    _dense_quant_per_group_kernel[(out_features,)](
        x, W, b, y, weight_scales, in_features, out_features, group_size, num_groups,
        input_scale, output_scale, input_zp, weight_zp, output_zp,
        has_bias=has_bias, qmin=-128, qmax=127,
    )


def dense_int16_per_group(x, in_features, W, b, out_features, group_size, input_scale,
                          weight_scales, output_scale, input_zp, weight_zp, output_zp, y):
    has_bias = b is not None
    if not has_bias:
        b = torch.zeros(1, device=x.device, dtype=torch.float32)
    num_groups = in_features // group_size
    _dense_quant_per_group_kernel[(out_features,)](
        x, W, b, y, weight_scales, in_features, out_features, group_size, num_groups,
        input_scale, output_scale, input_zp, weight_zp, output_zp,
        has_bias=has_bias, qmin=-32768, qmax=32767,
    )


def _unpack_int4_at(packed_w, flat: int) -> int:
    byte_idx = flat // 2
    packed = int(packed_w[byte_idx].item())
    qw = (packed >> 4) if (flat & 1) else (packed & 0x0F)
    if qw >= 8:
        qw -= 16
    return qw


def dense_int8_int4w_per_group(
    x, in_features, packed_w, weight_count, b, out_features, group_size,
    input_scale, weight_scales, output_scale, input_zp, weight_zp, output_zp, y,
):
    """Reference W4A8 static dense (matches C dense_int8_int4w_per_group)."""
    del weight_count
    num_groups = in_features // group_size
    out_features = int(out_features)
    pw = packed_w.reshape(-1)
    for o in range(out_features):
        result = 0.0
        for g in range(num_groups):
            acc = 0
            sum_qx = 0
            sum_qw = 0
            base = g * group_size
            for i in range(group_size):
                idx = base + i
                flat = idx * out_features + o
                wv = _unpack_int4_at(pw, flat)
                xi = int(x[idx].item())
                acc += xi * wv
                sum_qx += xi
                sum_qw += wv
            zp_term = int(input_zp) * int(weight_zp) * int(group_size)
            dot_affine = (
                acc - int(weight_zp) * sum_qx - int(input_zp) * sum_qw + zp_term
            )
            result += float(dot_affine) * float(input_scale) * float(
                weight_scales[g * out_features + o].item()
            )
        if b is not None:
            result += float(b[o].item())
        q = int(round(result / float(output_scale))) + int(output_zp)
        q = max(-128, min(127, q))
        y[o] = q


def dense_int8_int4w_per_group_to_float(
    x, in_features, packed_w, weight_count, b, out_features, group_size,
    input_scale, weight_scales, y,
):
    """Reference W4A8 dynamic dense (matches C dense_int8_int4w_per_group_to_float)."""
    del weight_count
    num_groups = in_features // group_size
    out_features = int(out_features)
    pw = packed_w.reshape(-1)
    for o in range(out_features):
        result = 0.0
        for g in range(num_groups):
            acc = 0
            base = g * group_size
            for i in range(group_size):
                idx = base + i
                flat = idx * out_features + o
                wv = _unpack_int4_at(pw, flat)
                acc += int(x[idx].item()) * wv
            result += float(acc) * float(input_scale) * float(
                weight_scales[g * out_features + o].item()
            )
        if b is not None:
            result += float(b[o].item())
        y[o] = result


def dense_float_palettized(x, in_features, indices, weight_count, codebook, num_centroids,
                           b, out_features, y):
    """Reference palettized weight-only dense."""
    del weight_count
    out_features = int(out_features)
    k = int(num_centroids)
    cb = codebook.reshape(-1)
    idx = indices.reshape(-1)
    result = torch.zeros(out_features, device=x.device, dtype=torch.float32)
    if b is not None:
        result = result + b[:out_features]
    for o in range(out_features):
        acc = result[o]
        for i in range(in_features):
            flat = i * out_features + o
            if k <= 16:
                byte_idx = flat // 2
                nibble = (int(idx[byte_idx].item()) >> 4) if (flat & 1) else (int(idx[byte_idx].item()) & 0x0F)
                w = float(cb[nibble].item())
            else:
                w = float(cb[int(idx[flat].item())].item())
            acc += float(x[i].item()) * w
        y[o] = acc


def dense_int8_to_float(x, in_features, W, b, out_features, input_scale, weight_scale, y):
    has_bias = b is not None
    if not has_bias:
        b = torch.zeros(1, device=x.device, dtype=torch.float32)
    _dense_quant_to_float_kernel[(out_features,)](
        x, W, b, y, in_features, out_features, input_scale, weight_scale, has_bias=has_bias
    )


def dense_int16_to_float(x, in_features, W, b, out_features, input_scale, weight_scale, y):
    dense_int8_to_float(x, in_features, W, b, out_features, input_scale, weight_scale, y)


# ---------------------------------------------------------------------------
# Conv2D quant (delegates to reference torch for affine ZP correctness in v1)
# Uses same formulas as C; can be replaced with fused Triton tiles later.
# ---------------------------------------------------------------------------

def _conv2d_quant_ref(
    inp, filt, bias, out,
    in_h, in_w, in_c, k_h, k_w, out_c,
    sh, sw, ph, pw,
    input_scale, weight_scale, output_scale,
    input_zp, weight_zp, output_zp,
    per_channel_scales=None,
    output_float=False,
):
    """Reference conv matching C affine quantization (used from Triton launch wrappers)."""
    out_h = (in_h + 2 * ph - k_h) // sh + 1
    out_w = (in_w + 2 * pw - k_w) // sw + 1
    inp_i = inp.reshape(in_h, in_w, in_c).to(torch.int32)
    filt_i = filt.reshape(k_h, k_w, in_c, out_c).to(torch.int32)
    out_t = torch.zeros(out_h, out_w, out_c, device=inp.device, dtype=torch.float32)
    zx, zw = int(input_zp), int(weight_zp)
    for oh in range(out_h):
        for ow in range(out_w):
            for oc in range(out_c):
                acc = 0
                sum_qx = 0
                sum_qf = 0
                p = 0
                for kh in range(k_h):
                    ih = oh * sh + kh - ph
                    if ih < 0 or ih >= in_h:
                        continue
                    for kw in range(k_w):
                        iw = ow * sw + kw - pw
                        if iw < 0 or iw >= in_w:
                            continue
                        in_px = inp_i[ih, iw, :]
                        f_px = filt_i[kh, kw, :, oc]
                        acc += int((in_px * f_px).sum().item())
                        sum_qx += int(in_px.sum().item())
                        sum_qf += int(f_px.sum().item())
                        p += in_c
                dot = acc - zw * sum_qx - zx * sum_qf + zx * zw * p
                ws = weight_scale if per_channel_scales is None else float(per_channel_scales[oc])
                result = float(dot) * input_scale * ws
                if bias is not None:
                    result += float(bias[oc])
                out_t[oh, ow, oc] = result
    if output_float:
        out.copy_(out_t.reshape(-1))
    else:
        q = torch.round(out_t / output_scale) + output_zp
        q = torch.clamp(q, -128 if out.dtype == torch.int8 else -32768,
                        127 if out.dtype == torch.int8 else 32767)
        out.copy_(q.to(out.dtype).reshape(-1))


def conv2d_nhwc_int8(inp, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
                     sh, sw, ph, pw, input_scale, weight_scale, output_scale,
                     input_zp, weight_zp, output_zp, out):
    _conv2d_quant_ref(inp, filt, bias, out, in_h, in_w, in_c, k_h, k_w, out_c,
                      sh, sw, ph, pw, input_scale, weight_scale, output_scale,
                      input_zp, weight_zp, output_zp)


def conv2d_nhwc_int16(inp, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
                      sh, sw, ph, pw, input_scale, weight_scale, output_scale,
                      input_zp, weight_zp, output_zp, out):
    conv2d_nhwc_int8(inp, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
                     sh, sw, ph, pw, input_scale, weight_scale, output_scale,
                     input_zp, weight_zp, output_zp, out)


def conv2d_nhwc_int8_per_channel(inp, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
                                 sh, sw, ph, pw, input_scale, weight_scales, output_scale,
                                 input_zp, weight_zp, output_zp, out):
    _conv2d_quant_ref(inp, filt, bias, out, in_h, in_w, in_c, k_h, k_w, out_c,
                      sh, sw, ph, pw, input_scale, 1.0, output_scale,
                      input_zp, weight_zp, output_zp, per_channel_scales=weight_scales)


def conv2d_nhwc_int16_per_channel(inp, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
                                  sh, sw, ph, pw, input_scale, weight_scales, output_scale,
                                  input_zp, weight_zp, output_zp, out):
    conv2d_nhwc_int8_per_channel(inp, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
                                 sh, sw, ph, pw, input_scale, weight_scales, output_scale,
                                 input_zp, weight_zp, output_zp, out)


def conv2d_nhwc_int8_to_float(inp, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
                              sh, sw, ph, pw, input_scale, weight_scale, out):
    _conv2d_quant_ref(inp, filt, bias, out, in_h, in_w, in_c, k_h, k_w, out_c,
                      sh, sw, ph, pw, input_scale, weight_scale, 1.0, 0, 0, 0,
                      output_float=True)


def conv2d_nhwc_int16_to_float(inp, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
                               sh, sw, ph, pw, input_scale, weight_scale, out):
    conv2d_nhwc_int8_to_float(inp, in_h, in_w, in_c, filt, k_h, k_w, out_c, bias,
                              sh, sw, ph, pw, input_scale, weight_scale, out)


def _depthwise_quant_ref(inp, filt, bias, out, in_h, in_w, channels, k_h, k_w,
                         sh, sw, ph, pw, input_scale, weight_scale, output_scale,
                         input_zp, weight_zp, output_zp, per_channel_scales=None,
                         output_float=False):
    out_h = (in_h + 2 * ph - k_h) // sh + 1
    out_w = (in_w + 2 * pw - k_w) // sw + 1
    inp_i = inp.reshape(in_h, in_w, channels).to(torch.int32)
    filt_i = filt.reshape(k_h, k_w, channels).to(torch.int32)
    out_t = torch.zeros(out_h, out_w, channels, device=inp.device, dtype=torch.float32)
    zx, zw = int(input_zp), int(weight_zp)
    for oh in range(out_h):
        for ow in range(out_w):
            for c in range(channels):
                acc = sum_qx = sum_qf = p = 0
                for kh in range(k_h):
                    ih = oh * sh + kh - ph
                    if ih < 0 or ih >= in_h:
                        continue
                    for kw in range(k_w):
                        iw = ow * sw + kw - pw
                        if iw < 0 or iw >= in_w:
                            continue
                        qx = int(inp_i[ih, iw, c].item())
                        qf = int(filt_i[kh, kw, c].item())
                        acc += qx * qf
                        sum_qx += qx
                        sum_qf += qf
                        p += 1
                dot = acc - zw * sum_qx - zx * sum_qf + zx * zw * p
                ws = weight_scale if per_channel_scales is None else float(per_channel_scales[c])
                result = float(dot) * input_scale * ws
                if bias is not None:
                    result += float(bias[c])
                out_t[oh, ow, c] = result
    if output_float:
        out.copy_(out_t.reshape(-1))
    else:
        q = torch.round(out_t / output_scale) + output_zp
        q = torch.clamp(q, -128 if out.dtype == torch.int8 else -32768,
                        127 if out.dtype == torch.int8 else 32767)
        out.copy_(q.to(out.dtype).reshape(-1))


def depthwise_conv2d_nhwc_int8(inp, in_h, in_w, channels, filt, k_h, k_w, bias,
                               sh, sw, ph, pw, input_scale, weight_scale, output_scale,
                               input_zp, weight_zp, output_zp, out):
    _depthwise_quant_ref(inp, filt, bias, out, in_h, in_w, channels, k_h, k_w,
                         sh, sw, ph, pw, input_scale, weight_scale, output_scale,
                         input_zp, weight_zp, output_zp)


def depthwise_conv2d_nhwc_int16(inp, in_h, in_w, channels, filt, k_h, k_w, bias,
                                sh, sw, ph, pw, input_scale, weight_scale, output_scale,
                                input_zp, weight_zp, output_zp, out):
    depthwise_conv2d_nhwc_int8(inp, in_h, in_w, channels, filt, k_h, k_w, bias,
                               sh, sw, ph, pw, input_scale, weight_scale, output_scale,
                               input_zp, weight_zp, output_zp, out)


def depthwise_conv2d_nhwc_int8_per_channel(inp, in_h, in_w, channels, filt, k_h, k_w, bias,
                                           sh, sw, ph, pw, input_scale, weight_scales,
                                           output_scale, input_zp, weight_zp, output_zp, out):
    _depthwise_quant_ref(inp, filt, bias, out, in_h, in_w, channels, k_h, k_w,
                         sh, sw, ph, pw, input_scale, 1.0, output_scale,
                         input_zp, weight_zp, output_zp, per_channel_scales=weight_scales)


def depthwise_conv2d_nhwc_int16_per_channel(inp, in_h, in_w, channels, filt, k_h, k_w, bias,
                                            sh, sw, ph, pw, input_scale, weight_scales,
                                            output_scale, input_zp, weight_zp, output_zp, out):
    depthwise_conv2d_nhwc_int8_per_channel(inp, in_h, in_w, channels, filt, k_h, k_w, bias,
                                            sh, sw, ph, pw, input_scale, weight_scales,
                                            output_scale, input_zp, weight_zp, output_zp, out)


def depthwise_conv2d_nhwc_int8_to_float(inp, in_h, in_w, channels, filt, k_h, k_w, bias,
                                        sh, sw, ph, pw, input_scale, weight_scale, out):
    _depthwise_quant_ref(inp, filt, bias, out, in_h, in_w, channels, k_h, k_w,
                         sh, sw, ph, pw, input_scale, weight_scale, 1.0, 0, 0, 0,
                         output_float=True)


def depthwise_conv2d_nhwc_int16_to_float(inp, in_h, in_w, channels, filt, k_h, k_w, bias,
                                         sh, sw, ph, pw, input_scale, weight_scale, out):
    depthwise_conv2d_nhwc_int8_to_float(inp, in_h, in_w, channels, filt, k_h, k_w, bias,
                                       sh, sw, ph, pw, input_scale, weight_scale, out)


# ---------------------------------------------------------------------------
# Quantized pooling / mean (symmetric input ZP=0 as in C headers)
# ---------------------------------------------------------------------------

def mean_hwc_int8(inp, h, w, c, input_scale, output_scale, output_zp, out):
    from .nn_ops_float import mean_hwc
    tmp = torch.empty(c, device=inp.device, dtype=torch.float32)
    dequant = inp.to(torch.float32) * input_scale
    mean_hwc(dequant, h, w, c, tmp)
    q = torch.round(tmp / output_scale) + output_zp
    out.copy_(torch.clamp(q, -128, 127).to(torch.int8))


def mean_last_dim_int8(inp, rows, cols, input_scale, output_scale, output_zp, out):
    from .nn_ops_float import mean_last_dim
    tmp = torch.empty(rows, device=inp.device, dtype=torch.float32)
    dequant = inp.to(torch.float32) * input_scale
    mean_last_dim(dequant, rows, cols, tmp)
    q = torch.round(tmp / output_scale) + output_zp
    out.copy_(torch.clamp(q, -128, 127).to(torch.int8))


def global_average_pool_2d_int8(inp, h, w, c, input_scale, output_scale, output_zp, out):
    mean_hwc_int8(inp, h, w, c, input_scale, output_scale, output_zp, out)
