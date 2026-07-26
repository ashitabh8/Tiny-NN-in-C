"""Phase 3 near-equality parity for unified C affine dense template.

Baselines captured 2026-07-25 against committed hand-written kernels on
migration/v2 @ 87e1ce0, TinyMLP(32,16,4), torch.manual_seed(42), num_samples=20.

Run-to-run jitter across 3 repeats was exactly 0.0 for every cell (deterministic
integer kernels). EPS is therefore 0.0 — any numeric drift fails the gate.
Do not widen EPS; diff against the baseline kernel instead.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import TinyMLP
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.quantization import (
    DynamicInt4PerGroupLinearQuantRule,
    DynamicQuantRuleMinMaxPerTensor,
    QuantizationTransform,
    StaticInt4PerGroupLinearQuantRule,
    StaticPerChannelLinearQuantRule,
    StaticPerGroupLinearQuantRule,
    StaticQuantRule,
)
from tools.verify_model import _gcc_available, verify_model

# Captured pre-template (hand-written kernels). jitter == 0 across 3 repeats.
PHASE3_C_BASELINE = {
    "w8a8_static_per_tensor": 0.03380173444747925,
    "w16a16_static_per_tensor": 0.0036468859761953354,
    "w8a8_static_per_channel": 0.02576872706413269,
    "w16a16_static_per_channel": 0.002278625965118408,
    "w8a8_static_per_group": 0.02576872706413269,
    "w16a16_static_per_group": 0.002278625965118408,
    "w8a8_dynamic": 0.005119919776916504,
    "w16a16_dynamic": 1.9431114196777344e-05,
    "w4a8_static": 0.04866981506347656,
    "w4a8_dynamic": 0.03863301873207092,
}

# Deterministic integer kernels: measured run-to-run jitter was 0.0.
EPS = 0.0
SUITE_TOLERANCE = 5.0  # coarse crash detector only


def _skip_no_gcc():
    if not _gcc_available():
        pytest.skip("gcc not available")


def _make_rules(cell: str):
    if cell == "w8a8_static_per_tensor":
        return [StaticQuantRule(r"fc1", "int8", 0.05, 0, 0.02, 0, 0.05, 0)]
    if cell == "w16a16_static_per_tensor":
        return [StaticQuantRule(r"fc1", "int16", 0.005, 0, 0.002, 0, 0.005, 0)]
    if cell == "w8a8_static_per_channel":
        return [StaticPerChannelLinearQuantRule(r"fc1", "int8", 0.05, 0, 0.05, 0)]
    if cell == "w16a16_static_per_channel":
        return [StaticPerChannelLinearQuantRule(r"fc1", "int16", 0.005, 0, 0.005, 0)]
    if cell == "w8a8_static_per_group":
        return [
            StaticPerGroupLinearQuantRule(
                r"fc1", "int8", 0.05, 0, 0.05, 0, group_size=32
            )
        ]
    if cell == "w16a16_static_per_group":
        return [
            StaticPerGroupLinearQuantRule(
                r"fc1", "int16", 0.005, 0, 0.005, 0, group_size=32
            )
        ]
    if cell == "w8a8_dynamic":
        return [DynamicQuantRuleMinMaxPerTensor(r"fc1", "int8")]
    if cell == "w16a16_dynamic":
        return [DynamicQuantRuleMinMaxPerTensor(r"fc1", "int16")]
    if cell == "w4a8_static":
        return [
            StaticInt4PerGroupLinearQuantRule(
                r"fc1", 0.05, 0, 0.05, 0, group_size=32
            )
        ]
    if cell == "w4a8_dynamic":
        return [DynamicInt4PerGroupLinearQuantRule(r"fc1", group_size=32)]
    raise KeyError(cell)


def _affine_symbol(cell: str) -> str:
    if cell.startswith("w4a8"):
        return (
            "dense_affine_int8_w4_to_float"
            if cell.endswith("dynamic")
            else "dense_affine_int8_w4"
        )
    if "dynamic" in cell:
        bits = "16" if "w16" in cell else "8"
        return f"dense_affine_int{bits}_to_float"
    bits = "16" if "w16" in cell else "8"
    return f"dense_affine_int{bits}"


@pytest.mark.parametrize("cell", list(PHASE3_C_BASELINE.keys()))
def test_phase3_c_near_equality(cell):
    """abs(measured - baseline) <= EPS for every in-scope C cell."""
    _skip_no_gcc()
    torch.manual_seed(42)
    model = TinyMLP(input_size=32, hidden_size=16, output_size=4).eval()
    x = torch.randn(1, 32)
    rules = _make_rules(cell)
    res = verify_model(
        model, x, num_samples=20, quantization_rules=rules, tolerance=SUITE_TOLERANCE
    )
    baseline = PHASE3_C_BASELINE[cell]
    assert res.passed == res.num_samples, (
        f"{cell}: {res.passed}/{res.num_samples} passed at suite tol"
    )
    delta = abs(res.overall_max_error - baseline)
    assert delta <= EPS, (
        f"{cell}: measured={res.overall_max_error!r} baseline={baseline!r} "
        f"delta={delta!r} > EPS={EPS!r} — template diverged; do not widen EPS"
    )

    ir = compile_model(model, x, return_ir=True, verbose=False)
    qir = QuantizationTransform(rules).apply(ir)
    code = CPrinter(qir).generate_model_c()
    sym = _affine_symbol(cell)
    assert sym in code, f"{cell}: expected {sym}(...) in generated C"


def test_three_layout_scale_index_collapse():
    """Unified indexing reduces to each named granularity."""

    def scale_index(g, o, *, in_features, group_size, per_out_column, out_features):
        # Mirror nn_ops_affine_dense.h
        assert in_features % group_size == 0
        scale_cols = out_features if per_out_column else 1
        col = o if per_out_column else 0
        return g * scale_cols + col

    in_f, out_f = 32, 16

    # PER_TENSOR: one group, column forced to 0
    assert scale_index(
        0, 7, in_features=in_f, group_size=in_f, per_out_column=False, out_features=out_f
    ) == 0

    # PER_CHANNEL: one group, index by output column
    assert scale_index(
        0, 7, in_features=in_f, group_size=in_f, per_out_column=True, out_features=out_f
    ) == 7

    # PER_GROUP(g=8): index by (g, o)
    assert scale_index(
        2, 7, in_features=in_f, group_size=8, per_out_column=True, out_features=out_f
    ) == 2 * out_f + 7


def test_three_layout_same_c_symbol():
    """Per-tensor / per-channel / per-group all call dense_affine_int8."""
    _skip_no_gcc()
    cells = [
        "w8a8_static_per_tensor",
        "w8a8_static_per_channel",
        "w8a8_static_per_group",
    ]
    torch.manual_seed(42)
    model = TinyMLP(input_size=32, hidden_size=16, output_size=4).eval()
    x = torch.randn(1, 32)
    for cell in cells:
        ir = compile_model(model, x, return_ir=True, verbose=False)
        qir = QuantizationTransform(_make_rules(cell)).apply(ir)
        code = CPrinter(qir).generate_model_c()
        assert "dense_affine_int8(" in code, f"{cell} must call dense_affine_int8"
        assert "dense_int8_per_channel" not in code
        assert "dense_int8_per_group" not in code
        # Old scalar name must not appear as a call (affine name contains the substring).
        assert "dense_int8(" not in code.replace("dense_affine_int8(", "AFFINE(")
