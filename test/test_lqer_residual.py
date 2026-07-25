"""Phase 2 LQER gate: weight-quant residual bit-identical after rebase.

LQER's correction is built against a specific dequant(quant(W)). If the
quantized-leg node changed that residual, the diamond would silently
approximate the wrong error.
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.quantization import (
    DynamicQuantRuleMinMaxPerTensor,
    LQERDynamicQuantRule,
    LQERStaticQuantRule,
    QuantizationTransform,
    StaticQuantRule,
)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 12)
        self.fc2 = nn.Linear(12, 4)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


# Captured pre-Phase-2 with torch.manual_seed(0), TinyMLP above.
RESIDUAL_BASELINE = {
    "static_int8": 0.000980684372383786,
    "dynamic_int8": 0.000980684372383786,
}


def _max_residual(W_float: np.ndarray, W_quant: np.ndarray, scale: float) -> float:
    recon = W_quant.astype(np.float64) * scale
    return float(np.max(np.abs(W_float.astype(np.float64) - recon)))


def test_lqer_static_weight_residual_unchanged():
    torch.manual_seed(0)
    model = TinyMLP().eval()
    x = torch.randn(1, 16)
    ir = compile_model(model, x, return_ir=True, verbose=False)
    W0 = ir.parameters["fc1_weight"].copy()
    absmax = float(np.max(np.abs(W0)))
    scale = absmax / 127.0
    E = W0 - np.clip(np.round(W0 / scale), -128, 127) * scale

    rules = [
        LQERStaticQuantRule(
            "fc1",
            "int8",
            input_scale=0.05,
            input_offset=0,
            weight_scale=scale,
            weight_offset=0,
            output_scale=0.05,
            output_offset=0,
            error_matrix=E,
            rank=4,
        )
    ]
    qir = QuantizationTransform(rules).apply(ir)
    Wq = qir.parameters["fc1_weight"]
    residual = _max_residual(W0, Wq, scale)
    assert residual == pytest.approx(RESIDUAL_BASELINE["static_int8"], abs=1e-12)


def test_lqer_dynamic_weight_residual_unchanged():
    torch.manual_seed(0)
    model = TinyMLP().eval()
    x = torch.randn(1, 16)
    ir = compile_model(model, x, return_ir=True, verbose=False)
    W0 = ir.parameters["fc1_weight"].copy()
    absmax = float(np.max(np.abs(W0)))
    scale = absmax / 127.0
    E = W0 - np.clip(np.round(W0 / scale), -128, 127) * scale

    rules = [
        LQERDynamicQuantRule("fc1", "int8", error_matrix=E, rank=4)
    ]
    qir = QuantizationTransform(rules).apply(ir)
    Wq = qir.parameters["fc1_weight"]
    residual = _max_residual(W0, Wq, scale)
    assert residual == pytest.approx(RESIDUAL_BASELINE["dynamic_int8"], abs=1e-12)


def test_plain_static_matches_lqer_static_residual():
    """Plain StaticQuantRule and LQERStaticQuantRule share the same weight quant."""
    torch.manual_seed(0)
    model = TinyMLP().eval()
    x = torch.randn(1, 16)
    ir = compile_model(model, x, return_ir=True, verbose=False)
    W0 = ir.parameters["fc1_weight"].copy()
    absmax = float(np.max(np.abs(W0)))
    scale = absmax / 127.0
    E = W0 - np.clip(np.round(W0 / scale), -128, 127) * scale

    ir_a = compile_model(model, x, return_ir=True, verbose=False)
    plain = QuantizationTransform(
        [
            StaticQuantRule(
                "fc1", "int8", 0.05, 0, scale, 0, 0.05, 0
            )
        ]
    ).apply(ir_a)

    ir_b = compile_model(model, x, return_ir=True, verbose=False)
    lqer = QuantizationTransform(
        [
            LQERStaticQuantRule(
                "fc1", "int8", 0.05, 0, scale, 0, 0.05, 0,
                error_matrix=E, rank=4,
            )
        ]
    ).apply(ir_b)

    np.testing.assert_array_equal(
        plain.parameters["fc1_weight"], lqer.parameters["fc1_weight"]
    )


def test_plain_dynamic_matches_lqer_dynamic_residual():
    torch.manual_seed(0)
    model = TinyMLP().eval()
    x = torch.randn(1, 16)
    ir = compile_model(model, x, return_ir=True, verbose=False)
    W0 = ir.parameters["fc1_weight"].copy()
    absmax = float(np.max(np.abs(W0)))
    scale = absmax / 127.0
    E = W0 - np.clip(np.round(W0 / scale), -128, 127) * scale

    ir_a = compile_model(model, x, return_ir=True, verbose=False)
    plain = QuantizationTransform(
        [DynamicQuantRuleMinMaxPerTensor("fc1", "int8")]
    ).apply(ir_a)

    ir_b = compile_model(model, x, return_ir=True, verbose=False)
    lqer = QuantizationTransform(
        [LQERDynamicQuantRule("fc1", "int8", error_matrix=E, rank=4)]
    ).apply(ir_b)

    np.testing.assert_array_equal(
        plain.parameters["fc1_weight"], lqer.parameters["fc1_weight"]
    )
