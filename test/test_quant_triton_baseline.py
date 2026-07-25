"""Phase 1 Part A2 — Triton quant baseline for real @triton.jit dense kernels.

Covers W8A8/W16A16 static per-tensor / per-channel / per-group only.
Excludes W4A8, conv, palette (Python ref loops — built in Phase 4).

Phase 4 gate per cell: template_error <= SNAPSHOT + MARGIN.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import TinyMLP
from src.pytorch_to_c.quantization import (
    DynamicQuantRuleMinMaxPerTensor,
    QuantizationTransform,
    StaticPerChannelLinearQuantRule,
    StaticPerGroupLinearQuantRule,
    StaticQuantRule,
)
from tools.verify_model_triton import verify_model_triton

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for Triton baseline"
)

# Captured on CUDA box, torch.manual_seed(42), num_samples=20, TinyMLP(32,16,4).
TRITON_BASELINE = {
    "w8a8_static_per_tensor": 0.10852346,
    "w16a16_static_per_tensor": 0.01339865,
    "w8a8_static_per_channel": 0.03282639,
    "w16a16_static_per_channel": 0.00341088,
    "w8a8_static_per_group": 0.03282639,
    "w16a16_static_per_group": 0.00341088,
}
MARGIN = 0.1
SUITE_TOLERANCE = 5.0


def _rules_for(cell: str):
    if cell == "w8a8_static_per_tensor":
        return [StaticQuantRule(r".*fc.*", "int8", 0.05, 0, 0.05, 0, 0.05, 0)]
    if cell == "w16a16_static_per_tensor":
        return [StaticQuantRule(r".*fc.*", "int16", 0.005, 0, 0.005, 0, 0.005, 0)]
    if cell == "w8a8_static_per_channel":
        return [StaticPerChannelLinearQuantRule(r".*fc.*", "int8", 0.05, 0, 0.05, 0)]
    if cell == "w16a16_static_per_channel":
        return [StaticPerChannelLinearQuantRule(r".*fc.*", "int16", 0.005, 0, 0.005, 0)]
    if cell == "w8a8_static_per_group":
        return [
            StaticPerGroupLinearQuantRule(
                r".*fc.*", "int8", 0.05, 0, 0.05, 0, group_size=32
            )
        ]
    if cell == "w16a16_static_per_group":
        return [
            StaticPerGroupLinearQuantRule(
                r".*fc.*", "int16", 0.005, 0, 0.005, 0, group_size=32
            )
        ]
    raise KeyError(cell)


@pytest.mark.parametrize("cell", list(TRITON_BASELINE.keys()))
def test_triton_quant_static_baseline(cell):
    torch.manual_seed(42)
    model = TinyMLP(input_size=32, hidden_size=16, output_size=4).eval()
    x = torch.randn(1, 32)
    rules = _rules_for(cell)

    res = verify_model_triton(
        model,
        x,
        num_samples=20,
        tolerance=SUITE_TOLERANCE,
        quantize_fn=lambda ir: QuantizationTransform(rules).apply(ir),
    )
    snapshot = TRITON_BASELINE[cell]

    assert res.passed == res.num_samples
    assert res.overall_max_error <= snapshot + MARGIN, (
        f"{cell}: max={res.overall_max_error:.6f} > snapshot({snapshot}) + margin({MARGIN})"
    )


@pytest.mark.parametrize(
    "dtype",
    ["int8", "int16"],
)
def test_triton_dynamic_preexisting_kernel_bug(dtype):
    """Dynamic *_to_float Triton kernels fail to compile today (int32/int64 acc bug).

    Phase 4 must fix the kernel first, then this test becomes the fidelity baseline.
    """
    torch.manual_seed(42)
    model = TinyMLP(input_size=32, hidden_size=16, output_size=4).eval()
    x = torch.randn(1, 32)
    rules = [DynamicQuantRuleMinMaxPerTensor(r".*fc.*", dtype)]

    with pytest.raises(Exception) as excinfo:
        verify_model_triton(
            model,
            x,
            num_samples=5,
            tolerance=SUITE_TOLERANCE,
            quantize_fn=lambda ir: QuantizationTransform(rules).apply(ir),
        )
    msg = str(excinfo.value).lower()
    assert "int32" in msg or "int64" in msg or "source code" in msg
