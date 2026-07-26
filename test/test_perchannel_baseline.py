"""Phase 1/3 per-channel linear C characterization baseline.

Phase 3 near-equality gate: abs(measured - baseline) <= EPS (EPS=0, jitter was 0).
Baseline re-captured 2026-07-25 on committed hand-written kernels (migration/v2).
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import TinyMLP
from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.quantization import QuantizationTransform, StaticPerChannelLinearQuantRule
from tools.verify_model import _gcc_available, verify_model

# Captured 2026-07-25, torch.manual_seed(42), num_samples=20, TinyMLP(32,16,4).
PERCHANNEL_BASELINE = {
    "int8": 0.02576872706413269,
    "int16": 0.002278625965118408,
}
EPS = 0.0  # deterministic; jitter measured at 0.0 across 3 repeats
SUITE_TOLERANCE = 5.0


def _skip_no_gcc():
    if not _gcc_available():
        pytest.skip("gcc not available")


@pytest.mark.parametrize(
    "dtype,kernel",
    [
        ("int8", "dense_affine_int8"),
        ("int16", "dense_affine_int16"),
    ],
)
def test_perchannel_linear_c_baseline(dtype, kernel):
    """Near-equality vs committed baseline; codegen uses unified affine symbol."""
    _skip_no_gcc()
    torch.manual_seed(42)

    model = TinyMLP(input_size=32, hidden_size=16, output_size=4).eval()
    x = torch.randn(1, 32)
    scale = 0.05 if dtype == "int8" else 0.005
    rules = [
        StaticPerChannelLinearQuantRule(
            pattern=r"fc1",
            dtype=dtype,
            input_scale=scale,
            input_offset=0,
            output_scale=scale,
            output_offset=0,
        )
    ]

    res = verify_model(
        model, x, num_samples=20, quantization_rules=rules, tolerance=SUITE_TOLERANCE
    )
    baseline = PERCHANNEL_BASELINE[dtype]

    assert res.passed == res.num_samples, (
        f"per-channel {dtype}: {res.passed}/{res.num_samples} passed at tol={SUITE_TOLERANCE}"
    )
    assert abs(res.overall_max_error - baseline) <= EPS, (
        f"per-channel {dtype}: max={res.overall_max_error!r} baseline={baseline!r} "
        f"delta={abs(res.overall_max_error - baseline)!r} > EPS={EPS}"
    )

    ir = compile_model(model, x, return_ir=True, verbose=False)
    qir = QuantizationTransform(rules).apply(ir)
    code = CPrinter(qir).generate_model_c()
    assert kernel in code, f"expected {kernel}(...) in generated C"
