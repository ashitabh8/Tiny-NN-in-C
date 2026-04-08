#!/usr/bin/env python3
"""
Dynamic Per-Channel Quantization -- Compile and verify.

Applies DynamicQuantRuleMinMaxPerChannel to every conv/linear layer:
  - One scale per *output channel*, computed from per-channel weight statistics.
  - No calibration data needed.
  - Typically more accurate than per-tensor because each channel's range is
    used independently.

Usage:
    python examples/02_dynamic_quantization/per_channel.py

Prerequisites:
    Run examples/01_float_mnist/run.py first (or supply your own checkpoint).
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

import torch

from models import MNISTConvNet
from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.quantization import (
    DynamicQuantRuleMinMaxPerChannel,
    QuantizationTransform,
)
from tools.verify_model import verify_model

CHECKPOINT = os.path.join(SCRIPT_DIR, "..", "01_float_mnist", "mnist_cnn.pth")
GENERATED_DIR = os.path.join(SCRIPT_DIR, "generated_per_channel")


def main() -> None:
    print("=" * 60)
    print("Example 02b -- Dynamic Per-Channel Int8 Quantization")
    print("=" * 60)

    model = MNISTConvNet()
    example_input = torch.randn(1, 1, 28, 28)

    # --- Load trained weights ---
    if os.path.exists(CHECKPOINT):
        print(f"\n[1/4] Loading checkpoint: {CHECKPOINT}")
        model.load_state_dict(torch.load(CHECKPOINT, weights_only=True))
    else:
        print(f"\n[1/4] No checkpoint found at {CHECKPOINT}")
        print("       Using random weights (run 01_float_mnist first for trained weights).")
    model.eval()

    # --- Define quantization rules ---
    print("\n[2/4] Applying DynamicQuantRuleMinMaxPerChannel to all layers...")
    rules = [
        DynamicQuantRuleMinMaxPerChannel(pattern=r".*(conv|fc).*", dtype="int8"),
    ]

    # --- Compile with quantization ---
    print(f"\n[3/4] Compiling to per-channel quantized C -> {GENERATED_DIR}/")
    ir_graph = compile_model(model, example_input, return_ir=True, verbose=False)
    ir_graph = QuantizationTransform(rules).apply(ir_graph)
    os.makedirs(GENERATED_DIR, exist_ok=True)
    CPrinter(ir_graph).generate_all(GENERATED_DIR)
    print("  Done.")

    # --- Verify ---
    print("\n[4/4] Verifying PyTorch vs per-channel quantized C (50 samples)...")
    results = verify_model(
        model, example_input,
        num_samples=50,
        quantization_rules=rules,
        tolerance=0.5,
        verbose=True,
    )
    print()
    print(results.summary())
    sys.exit(0 if results.failed == 0 else 1)


if __name__ == "__main__":
    main()
