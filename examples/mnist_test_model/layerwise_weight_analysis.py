#!/usr/bin/env python3
"""
Layerwise parameter / weight analysis for MNISTConvNet.

Usage:
    python examples/mnist_test_model/layerwise_weight_analysis.py
"""

from __future__ import annotations

import os
import sys

import torch

_THIS = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.join(_THIS, "..")
sys.path.insert(0, _EXAMPLES)
from mnist_cnn import MNISTConvNet

WEIGHTS = os.path.join(_EXAMPLES, "mnist_cnn_weights.pth")


def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.2f} KiB"
    return f"{n / (1024 * 1024):.2f} MiB"


def main() -> None:
    if not os.path.isfile(WEIGHTS):
        print(f"Missing {WEIGHTS} — train first: python examples/mnist_cnn.py")
        sys.exit(1)

    m = MNISTConvNet()
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    m.eval()

    print("=" * 88)
    print("MNISTConvNet — layerwise parameter analysis (float32 storage)")
    print("=" * 88)
    print(f"{'Module / tensor':<40} {'shape':<26} {'#params':>10} {'size':>12}")
    print("-" * 88)

    total = 0
    by_module: dict[str, int] = {}

    for name, p in m.named_parameters():
        n = p.numel()
        total += n
        mod = name.split(".")[0]
        by_module[mod] = by_module.get(mod, 0) + n
        shape_str = str(tuple(p.shape))
        print(f"{name:<40} {shape_str:<26} {n:>10,} {human_bytes(n * 4):>12}")

    print("-" * 88)
    print(f"{'Trainable parameters (total)':<40} {'':<26} {total:>10,} {human_bytes(total * 4):>12}")

    # Buffers (BatchNorm running stats — stored in .pth and in generated weights.h)
    buf_total = 0
    print()
    print("Buffers (non-trainable, still serialized in state_dict / C weights):")
    print("-" * 88)
    for name, b in m.named_buffers():
        n = b.numel()
        buf_total += n
        shape_str = str(tuple(b.shape))
        print(f"{name:<40} {shape_str:<26} {n:>10,} {human_bytes(n * 4):>12}")
    print("-" * 88)
    print(f"{'Buffer total':<40} {'':<26} {buf_total:>10,} {human_bytes(buf_total * 4):>12}")

    grand = total + buf_total
    print()
    print(f"Grand total (params + buffers): {grand:,} scalars  {human_bytes(grand * 4)}  (float32)")

    print()
    print("Share of trainable params by module (Conv / BN / FC):")
    print("-" * 40)
    for mod in sorted(by_module.keys(), key=lambda k: -by_module[k]):
        pct = 100.0 * by_module[mod] / total
        bar = int(pct / 2)
        print(f"  {mod:<8}  {by_module[mod]:>10,}  ({pct:5.1f}%)  {'#' * bar}")

    print("=" * 88)


if __name__ == "__main__":
    main()
