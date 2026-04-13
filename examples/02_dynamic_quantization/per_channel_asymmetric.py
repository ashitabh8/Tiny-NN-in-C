#!/usr/bin/env python3
"""
Static per-channel quantization with non-zero activation/weight offsets — compile and verify.

Uses :class:`StaticPerChannelConvQuantRule` and :class:`StaticPerChannelLinearQuantRule`
on a small conv + linear model. Per-channel weight scales are computed from weights at
compile time; ``input_offset``, ``output_offset``, and ``weight_offset`` are set to
non-zero values to exercise the asymmetric path end-to-end.

Usage:
    python examples/02_dynamic_quantization/per_channel_asymmetric.py

Prerequisites:
    gcc on PATH (same as ``tools.verify_model``).
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch

from models import MixedNet
from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.quantization import (
    QuantizationTransform,
    StaticPerChannelConvQuantRule,
    StaticPerChannelLinearQuantRule,
)
from tools.verify_model import verify_model

GENERATED_DIR = os.path.join(SCRIPT_DIR, "generated_per_channel_asymmetric")

# Small non-zero zero-points (shared with C per-channel kernels).
_IN_OFF = 3
_OUT_OFF = 2
_W_OFF = 2


def main() -> None:
    print("=" * 60)
    print("Example 02b -- Static Per-Channel Int8 (asymmetric offsets)")
    print("=" * 60)

    model = MixedNet(input_channels=3, num_classes=4)
    model.eval()
    example_input = torch.randn(1, 3, 32, 32)

    # Conv first, then linear (rule order: first match wins per node).
    rules = [
        StaticPerChannelConvQuantRule(
            pattern=r".*conv.*",
            dtype="int8",
            input_scale=0.05,
            input_offset=_IN_OFF,
            output_scale=0.05,
            output_offset=_OUT_OFF,
            weight_offset=_W_OFF,
        ),
        StaticPerChannelLinearQuantRule(
            pattern=r".*fc.*",
            dtype="int8",
            input_scale=0.05,
            input_offset=_IN_OFF,
            output_scale=0.05,
            output_offset=_OUT_OFF,
            weight_offset=_W_OFF,
        ),
    ]

    print("\n[1/3] Compiling to quantized C -> {}/".format(GENERATED_DIR))
    ir_graph = compile_model(model, example_input, return_ir=True, verbose=False)
    ir_graph = QuantizationTransform(rules).apply(ir_graph)
    os.makedirs(GENERATED_DIR, exist_ok=True)
    CPrinter(ir_graph).generate_all(GENERATED_DIR)
    print("  Done.")

    print("\n[2/3] Verifying PyTorch vs quantized C (50 samples)...")
    results = verify_model(
        model,
        example_input,
        num_samples=50,
        quantization_rules=rules,
        tolerance=10.0,
        verbose=True,
    )
    print()
    print(results.summary())
    sys.exit(0 if results.failed == 0 else 1)


if __name__ == "__main__":
    main()
