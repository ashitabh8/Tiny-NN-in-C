"""Kernel-level tests for Triton ops (GPU-gated)."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Triton kernel tests",
)


@pytest.fixture
def device():
    return torch.device("cuda")


def test_dense_matches_torch(device):
    from src.triton_ops.nn_ops_float import dense

    in_f, out_f = 16, 8
    x = torch.randn(in_f, device=device)
    W = torch.randn(in_f, out_f, device=device)
    b = torch.randn(out_f, device=device)
    y = torch.empty(out_f, device=device)
    dense(x, in_f, W, b, out_f, y)
    expected = x @ W + b
    torch.testing.assert_close(y, expected, rtol=1e-4, atol=1e-4)


def test_relu_inplace(device):
    from src.triton_ops.nn_ops_float import relu

    x = torch.tensor([-1.0, 0.0, 2.0], device=device)
    relu(x, 3)
    torch.testing.assert_close(x, torch.tensor([0.0, 0.0, 2.0], device=device))


def test_conv2d_nhwc_small(device):
    from src.triton_ops.nn_ops_float import conv2d_nhwc

    in_h, in_w, in_c, out_c = 4, 4, 3, 2
    k = 3
    inp = torch.randn(in_h * in_w * in_c, device=device)
    filt = torch.randn(k, k, in_c, out_c, device=device)
    bias = torch.randn(out_c, device=device)
    out_h = in_h - k + 1
    out_w = in_w - k + 1
    out = torch.empty(out_h * out_w * out_c, device=device)
    conv2d_nhwc(inp, in_h, in_w, in_c, filt, k, k, out_c, bias, 1, 1, 0, 0, out)
    assert out.shape[0] == out_h * out_w * out_c
    assert torch.isfinite(out).all()


def test_quantize_dequantize_roundtrip(device):
    from src.triton_ops.nn_ops_quant import (
        dequantize_int8_to_float,
        quantize_float_to_int8,
    )

    x = torch.linspace(-1, 1, 32, device=device)
    q = torch.empty(32, dtype=torch.int8, device=device)
    out = torch.empty(32, device=device)
    scale, offset = 0.05, 0
    quantize_float_to_int8(x, 32, scale, offset, q)
    dequantize_int8_to_float(q, 32, scale, offset, out)
    torch.testing.assert_close(out, x, rtol=0.1, atol=0.05)


def test_dense_int8_symmetric(device):
    from src.triton_ops.nn_ops_quant import dense_int8

    in_f, out_f = 8, 4
    x = torch.randint(-10, 11, (in_f,), dtype=torch.int8, device=device)
    W = torch.randint(-5, 6, (in_f, out_f), dtype=torch.int8, device=device)
    b = torch.randn(out_f, device=device)
    y = torch.empty(out_f, dtype=torch.int8, device=device)
    dense_int8(x, in_f, W, b, out_f, 0.1, 0.05, 0.1, 0, 0, 0, y)
    assert y.dtype == torch.int8
