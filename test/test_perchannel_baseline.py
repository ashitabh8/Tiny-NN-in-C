"""Phase 1 Part A1 — per-channel linear C characterization baseline.

Captures fidelity against current StaticPerChannelLinearQuantRule + old nodes.
Phase 3 gate: max_abs_error <= BASELINE + MARGIN (not just suite tolerance 5.0).
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

# Captured on CUDA box, torch.manual_seed(42), num_samples=20, TinyMLP(32,16,4).
PERCHANNEL_BASELINE = {
    "int8": 0.033286,
    "int16": 0.004679,
}
MARGIN = 0.1
SUITE_TOLERANCE = 5.0


def _skip_no_gcc():
    if not _gcc_available():
        pytest.skip("gcc not available")


@pytest.mark.parametrize(
    "dtype,kernel",
    [
        ("int8", "dense_int8_per_channel"),
        ("int16", "dense_int16_per_channel"),
    ],
)
def test_perchannel_linear_c_baseline(dtype, kernel):
    """Characterize current per-channel C numerics and codegen delegation target."""
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
    assert res.overall_max_error <= baseline + MARGIN, (
        f"per-channel {dtype}: max={res.overall_max_error:.6f} > "
        f"baseline({baseline}) + margin({MARGIN})"
    )

    ir = compile_model(model, x, return_ir=True, verbose=False)
    qir = QuantizationTransform(rules).apply(ir)
    code = CPrinter(qir).generate_model_c()
    assert kernel in code, f"expected {kernel}(...) in generated C"
