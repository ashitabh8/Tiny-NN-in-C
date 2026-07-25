#!/usr/bin/env python3
"""
Benchmark PyTorch (CUDA) vs Triton-generated models on GPU.

The compiler currently emits batch-1 graphs, so the primary comparison is
single-sample latency. PyTorch batched numbers are included as a reference for
what highly optimized library kernels achieve at larger batch sizes.

Usage:
    python tools/benchmark_triton.py
    python tools/benchmark_triton.py --device 2 --models mnist mixed
    python tools/benchmark_triton.py --device 0 --iters 500 --warmup 100
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pytorch_to_c.codegen.triton_printer import generate_triton_code
from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.quantization import QuantizationTransform, StaticQuantRule


def _nchw_to_nhwc_flat(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x.permute(0, 2, 3, 1).contiguous()
    return x.reshape(-1)


@dataclass
class ModelSpec:
    name: str
    model: nn.Module
    example_input: torch.Tensor


def _model_specs(selected: Optional[List[str]] = None) -> List[ModelSpec]:
    from models import MNISTConvNet, MixedNet, TinyMLP

    all_specs = [
        ModelSpec(
            "tiny_mlp",
            TinyMLP(input_size=784, hidden_size=512, output_size=10),
            torch.randn(1, 784),
        ),
        ModelSpec(
            "mixed",
            MixedNet(input_channels=3, num_classes=10),
            torch.randn(1, 3, 32, 32),
        ),
        ModelSpec(
            "mnist",
            MNISTConvNet(),
            torch.randn(1, 1, 28, 28),
        ),
    ]
    if not selected:
        return all_specs
    wanted = {s.lower() for s in selected}
    return [s for s in all_specs if s.name in wanted]


def _quant_rules_for_ir(ir) -> List[StaticQuantRule]:
    has_conv = any(n.op_type == "conv2d" for n in ir.nodes)
    pattern = r".*(conv|fc).*" if has_conv else r".*fc.*"
    return [
        StaticQuantRule(
            pattern=pattern,
            dtype="int8",
            input_scale=0.05,
            input_offset=0,
            weight_scale=0.02,
            weight_offset=0,
            output_scale=0.05,
            output_offset=0,
        )
    ]


def _load_triton_module(output_dir: str):
    spec = importlib.util.spec_from_file_location(
        "triton_bench_model", os.path.join(output_dir, "model.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _output_size(ir) -> int:
    shape = ir.outputs[0].output_shape
    if shape and shape[0] == 1:
        return int(shape[1])
    return int(torch.tensor(shape).prod().item())


def compile_triton_runner(
    model: nn.Module,
    example_input: torch.Tensor,
    device: torch.device,
    quantize: bool,
    cache_dir: Path,
) -> Tuple[object, int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "quant" if quantize else "float"
    out_dir = cache_dir / tag
    dev_str = str(device)

    if quantize:
        ir = compile_model(
            model,
            example_input,
            output_dir=None,
            verbose=False,
            return_ir=True,
            backend="triton",
        )
        ir = QuantizationTransform(_quant_rules_for_ir(ir)).apply(ir)
        generate_triton_code(ir, str(out_dir), device=dev_str)
    else:
        ir = compile_model(
            model,
            example_input,
            output_dir=str(out_dir),
            verbose=False,
            return_ir=False,
            backend="triton",
        )

    mod = _load_triton_module(str(out_dir))
    return mod, _output_size(ir)


@dataclass
class BenchResult:
    label: str
    median_ms: float
    mean_ms: float
    p95_ms: float
    samples_per_sec: float


def cuda_bench(fn: Callable[[], None], warmup: int, iters: int) -> BenchResult:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    med = statistics.median(times)
    mean = statistics.mean(times)
    p95 = sorted(times)[int(0.95 * len(times)) - 1]
    return BenchResult("", med, mean, p95, 1000.0 / med if med > 0 else 0.0)


def benchmark_model(
    spec: ModelSpec,
    device_id: int,
    warmup: int,
    iters: int,
    cache_root: Path,
    pytorch_batches: List[int],
) -> Dict[str, BenchResult]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    model = spec.model.eval().to(device)
    x1 = spec.example_input.to(device)

    results: Dict[str, BenchResult] = {}

    # --- PyTorch batch=1 (fair vs Triton) ---
    out_pt = torch.empty(int(model(x1).numel()), device=device)

    def pt_forward_b1():
        with torch.no_grad():
            out_pt.copy_(model(x1).reshape(-1))

    r = cuda_bench(pt_forward_b1, warmup, iters)
    r.label = "pytorch_cuda (batch=1)"
    results[r.label] = r

    # --- PyTorch batched (reference only) ---
    for batch in pytorch_batches:
        if batch == 1:
            continue
        xb = spec.example_input.expand(batch, *spec.example_input.shape[1:]).contiguous().to(device)

        def pt_forward_batch(xb=xb, batch=batch):
            with torch.no_grad():
                model(xb)

        r = cuda_bench(pt_forward_batch, warmup, iters)
        r.label = f"pytorch_cuda (batch={batch})"
        results[r.label] = r

    # --- Triton float ---
    cache_dir = cache_root / spec.name / f"gpu{device_id}"
    triton_mod, out_size = compile_triton_runner(model.cpu(), spec.example_input, device, False, cache_dir)
    model.to(device)

    inp = _nchw_to_nhwc_flat(x1).float().contiguous()
    out_tr = torch.empty(out_size, device=device, dtype=torch.float32)

    def triton_float():
        triton_mod.model_forward(inp, out_tr)

    r = cuda_bench(triton_float, warmup, iters)
    r.label = "triton_float (batch=1)"
    results[r.label] = r

    # --- Triton quant ---
    try:
        triton_q, out_size_q = compile_triton_runner(
            model.cpu(), spec.example_input, device, True, cache_dir
        )
        model.to(device)
        assert out_size_q == out_size

        def triton_quant():
            triton_q.model_forward(inp, out_tr)

        r = cuda_bench(triton_quant, warmup, iters)
        r.label = "triton_int8 (batch=1)"
        results[r.label] = r
    except Exception as exc:
        print(f"  [skip] triton_int8 for {spec.name}: {exc}")

    return results


def _speedup(base_ms: float, other_ms: float) -> str:
    if other_ms <= 0:
        return "n/a"
    return f"{base_ms / other_ms:.2f}x"


def print_results(
    spec: ModelSpec,
    device_id: int,
    results: Dict[str, BenchResult],
) -> None:
    gpu_name = torch.cuda.get_device_name(device_id)
    print()
    print("=" * 72)
    print(f"Model: {spec.name}  |  GPU {device_id}: {gpu_name}")
    print(f"Input shape: {tuple(spec.example_input.shape)}  (Triton codegen is batch=1 only)")
    print("=" * 72)
    print(f"{'Backend':<28} {'median ms':>10} {'p95 ms':>10} {'samples/s':>12} {'vs PT b=1':>10}")
    print("-" * 72)

    base = results.get("pytorch_cuda (batch=1)")
    base_ms = base.median_ms if base else 1.0
    order = ["pytorch_cuda (batch=1)", "triton_float (batch=1)", "triton_int8 (batch=1)"]
    order += [k for k in results if k.startswith("pytorch_cuda (batch=") and "batch=1" not in k]

    for key in order:
        if key not in results:
            continue
        r = results[key]
        vs = "1.00x" if key == "pytorch_cuda (batch=1)" else _speedup(base_ms, r.median_ms)
        if key.startswith("triton") and r.median_ms > base_ms:
            vs = f"{vs} slower"
        elif key.startswith("triton"):
            vs = f"{vs} faster"
        print(
            f"{r.label:<28} {r.median_ms:10.4f} {r.p95_ms:10.4f} {r.samples_per_sec:12.1f} {vs:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs Triton on GPU")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (0-3)")
    parser.add_argument(
        "--models",
        nargs="*",
        choices=["tiny_mlp", "mixed", "mnist"],
        help="Models to benchmark (default: all)",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument(
        "--pytorch-batches",
        type=int,
        nargs="*",
        default=[1, 32, 128],
        help="Extra PyTorch batch sizes for reference throughput",
    )
    parser.add_argument(
        "--all-gpus",
        action="store_true",
        help="Run on every visible CUDA device",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    specs = _model_specs(args.models)
    cache_root = ROOT / ".benchmark_cache"

    device_ids = list(range(torch.cuda.device_count())) if args.all_gpus else [args.device]

    print(f"PyTorch {torch.__version__}  |  CUDA devices: {torch.cuda.device_count()}")
    for did in device_ids:
        print(f"  [{did}] {torch.cuda.get_device_name(did)}")

    for device_id in device_ids:
        for spec in specs:
            results = benchmark_model(
                spec,
                device_id,
                args.warmup,
                args.iters,
                cache_root,
                args.pytorch_batches,
            )
            print_results(spec, device_id, results)

    print()
    print("Notes:")
    print("  - Triton vs PyTorch batch=1 is the apples-to-apples comparison.")
    print("  - PyTorch batch>1 uses fused cuBLAS/cuDNN and is not comparable to batch=1 Triton.")
    print("  - Small models are often launch-bound; Triton may lose on tiny graphs.")
    print("  - Re-run with --device N or --all-gpus to test other GPUs.")


if __name__ == "__main__":
    main()
