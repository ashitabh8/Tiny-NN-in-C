#!/usr/bin/env python3
"""
Evaluate compression benefits: size reduction vs accuracy trade-offs.

Usage:
    python tools/evaluate_compression.py
    python tools/evaluate_compression.py --model mnist --samples 20
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import MNISTConvNet, TinyMLP
from src.passes import PruneRule, StructuredPruningPass
from src.pytorch_to_c.calibration import calibrate, make_static_rules
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.quantization import (
    StaticInt4PerGroupLinearQuantRule,
    PaletteWeightRule,
    QuantizationTransform,
    StaticPerChannelLinearQuantRule,
    StaticPerChannelConvQuantRule,
    StaticPerGroupLinearQuantRule,
    StaticQuantRule,
)
from tools.verify_model import VerificationResults, _gcc_available, verify_model


@dataclass
class BenefitRow:
    strategy: str
    weight_bytes: int
    compression_ratio: float
    num_params: int
    max_error: float
    mean_error: float
    top1_rate: float
    match_rate: float
    notes: str = ""


def _weight_bytes(ir) -> int:
    total = 0
    for name, arr in ir.parameters.items():
        if "bias" in name or "scale" in name or "codebook" in name:
            continue
        if "indices" in name:
            total += arr.nbytes
            continue
        total += arr.nbytes
    return total


def _param_count(ir) -> int:
    return sum(int(np.prod(arr.shape)) for arr in ir.parameters.values())


def _compile_ir(model, x, passes=None, quant_rules=None):
    ir = compile_model(model, x, return_ir=True, verbose=False)
    if passes:
        for p in passes:
            ir = p.apply(ir)
    if quant_rules:
        ir = QuantizationTransform(quant_rules).apply(ir)
    return ir


def _verify(
    model,
    x,
    passes=None,
    quant_rules=None,
    num_samples: int = 10,
    tolerance: float = 5.0,
) -> VerificationResults:
    return verify_model(
        model,
        x,
        num_samples=num_samples,
        passes=passes,
        quantization_rules=quant_rules,
        tolerance=tolerance,
    )


def _row(
    strategy: str,
    ir_baseline_bytes: int,
    ir,
    res: VerificationResults,
    notes: str = "",
) -> BenefitRow:
    wb = _weight_bytes(ir)
    ratio = ir_baseline_bytes / wb if wb > 0 else 0.0
    top1 = res.top1_matches / res.top1_total if res.top1_total else 1.0
    return BenefitRow(
        strategy=strategy,
        weight_bytes=wb,
        compression_ratio=ratio,
        num_params=_param_count(ir),
        max_error=res.overall_max_error,
        mean_error=res.overall_mean_error,
        top1_rate=top1,
        match_rate=res.match_rate,
        notes=notes,
    )


def evaluate_tiny_mlp(
    num_samples: int = 10,
    calib_batches: int = 16,
) -> Tuple[int, List[BenefitRow]]:
    torch.manual_seed(42)
    model = TinyMLP(input_size=784, hidden_size=256, output_size=10).eval()
    x = torch.randn(1, 784)
    calib_data = [torch.randn_like(x) for _ in range(calib_batches)]
    stats = calibrate(model, calib_data, collect_hessian=True, max_batches=calib_batches)

    baseline_ir = compile_model(model, x, return_ir=True, verbose=False)
    baseline_bytes = _weight_bytes(baseline_ir)
    rows: List[BenefitRow] = []

    rules_pt = [
        StaticQuantRule(
            pattern=r".*fc.*", dtype="int8",
            input_scale=stats.get_input_scale("fc1"), input_offset=0,
            weight_scale=0.02, weight_offset=0,
            output_scale=stats.get_input_scale("fc1"), output_offset=0,
        )
    ]
    rules_pc = [
        StaticPerChannelLinearQuantRule(
            pattern=r".*fc.*", dtype="int8",
            input_scale=stats.get_input_scale("fc1"), input_offset=0,
            output_scale=stats.get_input_scale("fc1"), output_offset=0,
        )
    ]
    rules_pg = [
        StaticPerGroupLinearQuantRule(
            pattern=r".*fc.*", dtype="int8",
            input_scale=stats.get_input_scale("fc1"), input_offset=0,
            output_scale=stats.get_input_scale("fc1"), output_offset=0,
            group_size="auto", error_budget=0.001,
        )
    ]
    rules_gptq = [
        StaticPerGroupLinearQuantRule(
            pattern=r".*fc.*", dtype="int8",
            input_scale=stats.get_input_scale("fc1"), input_offset=0,
            output_scale=stats.get_input_scale("fc1"), output_offset=0,
            group_size=128, rounding="gptq", calibration=stats,
        )
    ]
    rules_i4 = [
        StaticInt4PerGroupLinearQuantRule(
            pattern=r".*fc.*",
            input_scale=stats.get_input_scale("fc1"),
            input_offset=0,
            output_scale=stats.get_input_scale("fc1"),
            output_offset=0,
            group_size=128,
        )
    ]
    rules_pal = [PaletteWeightRule(pattern=r".*fc.*", num_centroids=16)]
    prune_pass = [StructuredPruningPass([PruneRule(pattern=r"fc1", amount=0.2)])]
    auto_rules = make_static_rules(stats, granularity="per_group")

    strategies = [
        ("float32 (baseline)", None, None),
        ("int8 per-tensor", rules_pt, None),
        ("int8 per-channel", rules_pc, None),
        ("int8 per-group (auto)", rules_pg, None),
        ("int8 per-group + GPTQ", rules_gptq, None),
        ("int4 W4A8 per-group", rules_i4, None),
        ("palette K=16", rules_pal, None),
        ("prune fc1 20%", None, prune_pass),
        ("calibrated per-group", auto_rules, None),
    ]
    notes: Dict[str, str] = {"prune fc1 20%": "structural"}

    for name, qr, passes in strategies:
        try:
            tol = 1e-3 if name.startswith("float") else (2.0 if passes and not qr else 5.0)
            res = _verify(model, x, passes=passes, quant_rules=qr,
                          num_samples=num_samples, tolerance=tol)
            ir = _compile_ir(model, x, passes=passes, quant_rules=qr)
            note = notes.get(name, "")
            if "per-group (auto)" in name:
                gs = [n.metadata.get("group_size") for n in ir.nodes if n.metadata.get("group_size")]
                note = f"groups={gs}"
            rows.append(_row(name, baseline_bytes, ir, res, notes=note))
        except Exception as exc:
            rows.append(BenefitRow(
                name, 0, 0.0, 0, float("nan"), float("nan"), 0.0, 0.0,
                notes=f"FAILED: {str(exc)[:60]}",
            ))

    return baseline_bytes, rows


def evaluate_mnist(num_samples: int = 5) -> Tuple[int, List[BenefitRow]]:
    torch.manual_seed(42)
    model = MNISTConvNet().eval()
    x = torch.randn(1, 1, 28, 28)
    stats = calibrate(model, [x] * 8, max_batches=8)

    baseline_ir = compile_model(model, x, return_ir=True, verbose=False)
    baseline_bytes = _weight_bytes(baseline_ir)
    rows: List[BenefitRow] = []

    res = _verify(model, x, num_samples=num_samples, tolerance=1e-2)
    rows.append(
        BenefitRow(
            "float32 (baseline)", baseline_bytes, 1.0,
            _param_count(baseline_ir), res.overall_max_error,
            res.overall_mean_error, 1.0, res.match_rate,
        )
    )

    rules_pc = [
        StaticPerChannelConvQuantRule(
            pattern=r".*conv.*",
            dtype="int8",
            input_scale=stats.get_input_scale("conv1"),
            input_offset=0,
            output_scale=stats.get_input_scale("conv1"),
            output_offset=0,
        ),
        StaticPerChannelLinearQuantRule(
            pattern=r".*fc.*",
            dtype="int8",
            input_scale=stats.get_input_scale("fc"),
            input_offset=0,
            output_scale=stats.get_input_scale("fc"),
            output_offset=0,
        ),
    ]
    ir = _compile_ir(model, x, quant_rules=rules_pc)
    res = _verify(model, x, quant_rules=rules_pc, num_samples=num_samples, tolerance=10.0)
    rows.append(_row("int8 per-channel conv+fc", baseline_bytes, ir, res))

    return baseline_bytes, rows


def print_table(baseline_bytes: int, rows: List[BenefitRow], title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print(f"Baseline weight storage (excl. biases/scales): {baseline_bytes:,} bytes ({baseline_bytes/1024:.1f} KB)")
    print("=" * 100)
    print(
        f"{'Strategy':<28} {'Weight B':>10} {'Compress':>9} "
        f"{'Max err':>10} {'Top-1':>8} {'Match':>8}  Notes"
    )
    print("-" * 100)
    for r in rows:
        print(
            f"{r.strategy:<28} {r.weight_bytes:>10,} {r.compression_ratio:>8.2f}x "
            f"{r.max_error:>10.4f} {r.top1_rate*100:>7.1f}% "
            f"{r.match_rate*100:>7.1f}%  {r.notes}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ML compression benefits")
    parser.add_argument("--model", choices=["tiny_mlp", "mnist", "both"], default="both")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    if not _gcc_available():
        print("ERROR: gcc required for C verification")
        sys.exit(1)

    if args.model in ("tiny_mlp", "both"):
        _, rows = evaluate_tiny_mlp(num_samples=args.samples)
        print_table(_, rows, "TinyMLP (784→256→10) — compression vs accuracy")

    if args.model in ("mnist", "both"):
        _, rows = evaluate_mnist(num_samples=args.samples)
        print_table(_, rows, "MNISTConvNet — compression vs accuracy")

    print("Interpretation:")
    print("  - Compress: weight storage ratio vs float32 (higher = smaller on disk/flash)")
    print("  - Top-1 / Match: prediction agreement with float PyTorch (higher = better)")
    print("  - int4 W4A8 / palette maximize compression; per-group/GPTQ improve accuracy at same bit-width")
    print("  - pruning reduces compute AND weights (structural)")


if __name__ == "__main__":
    main()
