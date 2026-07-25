"""
Verify that a PyTorch model was correctly compiled to C.

Compiles the model, generates a C test harness with embedded test vectors,
builds with gcc, and compares PyTorch vs C outputs across multiple samples.

Usage as module:
    from tools.verify_model import verify_model
    results = verify_model(model, example_input, num_samples=50)
    print(results.summary())

Usage as CLI:
    python -m tools.verify_model --model models/tiny_mlp.py:TinyMLP \
        --input-shape 1,784 --num-samples 50
"""

import os
import sys
import math
import shutil
import tempfile
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class VerificationResults:
    """Holds per-sample and aggregate statistics from a verification run."""

    num_samples: int = 0
    passed: int = 0
    failed: int = 0
    tolerance: float = 1e-3
    max_errors: List[float] = field(default_factory=list)
    mean_errors: List[float] = field(default_factory=list)
    top1_matches: int = 0
    top1_total: int = 0
    quantized: bool = False

    @property
    def match_rate(self) -> float:
        return self.passed / self.num_samples if self.num_samples else 0.0

    @property
    def overall_max_error(self) -> float:
        return max(self.max_errors) if self.max_errors else 0.0

    @property
    def overall_mean_error(self) -> float:
        return float(np.mean(self.mean_errors)) if self.mean_errors else 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Verification Results",
            "=" * 60,
            f"  Mode           : {'Quantized' if self.quantized else 'Float32'}",
            f"  Samples         : {self.num_samples}",
            f"  Tolerance       : {self.tolerance:.1e}",
            f"  Match rate      : {self.passed}/{self.num_samples} "
            f"({self.match_rate * 100:.1f}%)",
            f"  Max error       : {self.overall_max_error:.2e}",
            f"  Mean error      : {self.overall_mean_error:.2e}",
        ]
        if self.top1_total > 0:
            pct = self.top1_matches / self.top1_total * 100
            lines.append(
                f"  Top-1 class match: {self.top1_matches}/{self.top1_total} "
                f"({pct:.1f}%)"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# C harness generation
# ---------------------------------------------------------------------------

_HARNESS_TEMPLATE = r"""
#include <stdio.h>
#include <stdlib.h>
#include "model.h"

int main(int argc, char* argv[]) {{
    if (argc != 4) {{
        fprintf(stderr, "Usage: %s <num_samples> <input.bin> <output.bin>\n",
                argv[0]);
        return 1;
    }}

    int num_samples = atoi(argv[1]);
    const int input_size  = {input_size};
    const int output_size = {output_size};

    float* input  = (float*)malloc(input_size  * sizeof(float));
    float* output = (float*)malloc(output_size * sizeof(float));
    if (!input || !output) {{ fprintf(stderr, "OOM\n"); return 1; }}

    FILE* f_in  = fopen(argv[2], "rb");
    FILE* f_out = fopen(argv[3], "wb");
    if (!f_in || !f_out) {{ fprintf(stderr, "File open failed\n"); return 1; }}

    for (int s = 0; s < num_samples; ++s) {{
        if (fread(input, sizeof(float), input_size, f_in) != (size_t)input_size) {{
            fprintf(stderr, "Read error at sample %d\n", s);
            return 1;
        }}
        model_forward(input, output);
        fwrite(output, sizeof(float), output_size, f_out);
    }}

    fclose(f_in);
    fclose(f_out);
    free(input);
    free(output);
    return 0;
}}
"""


def _write_harness(tmpdir: str, input_size: int, output_size: int) -> str:
    """Write a C test harness into *tmpdir* and return the file path."""
    code = _HARNESS_TEMPLATE.format(input_size=input_size, output_size=output_size)
    path = os.path.join(tmpdir, "verify_harness.c")
    with open(path, "w") as f:
        f.write(code)
    return path


# ---------------------------------------------------------------------------
# gcc helpers
# ---------------------------------------------------------------------------

def _gcc_available() -> bool:
    try:
        subprocess.run(["gcc", "--version"], capture_output=True, timeout=5, check=True)
        return True
    except Exception:
        return False


def _compile_c(tmpdir: str, harness_path: str, openmp: bool = False) -> str:
    exe = os.path.join(tmpdir, "verify_model")
    model_c = os.path.join(tmpdir, "model.c")
    cmd = [
        "gcc", "-o", exe, harness_path, model_c,
        f"-I{tmpdir}", "-lm", "-std=c99", "-O2",
    ]
    if openmp:
        cmd.append("-fopenmp")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        raise RuntimeError(f"gcc failed:\n{result.stderr}")
    return exe


def _run_c(exe: str, num_samples: int, input_bin: str, output_bin: str) -> None:
    result = subprocess.run(
        [exe, str(num_samples), input_bin, output_bin],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"C executable failed:\n{result.stderr}")


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _compute_flat_input_size(example_input: torch.Tensor) -> int:
    shape = list(example_input.shape)
    if shape and shape[0] == 1:
        shape = shape[1:]
    return int(np.prod(shape)) if shape else 1


def _nchw_to_nhwc_flat(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to the C-side channels-last layout, flat.

    Used for both inputs (before sending to C) and outputs (PyTorch reference,
    so it lines up with the C output's flat layout).

    4D NCHW [B, C, H, W] -> NHWC [B, H, W, C].
    3D NCL  [B, C, L]    -> NLC  [B, L, C].
    Other ranks pass through unchanged.
    """
    if tensor.dim() == 4:
        tensor = tensor.permute(0, 2, 3, 1)
    elif tensor.dim() == 3:
        tensor = tensor.permute(0, 2, 1)
    return tensor.detach().numpy().flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_model(
    model: nn.Module,
    example_input: torch.Tensor,
    num_samples: int = 20,
    quantization_rules=None,
    passes=None,
    tolerance: float = 1e-3,
    verbose: bool = False,
    openmp: bool = False,
) -> VerificationResults:
    """Verify a PyTorch model against its compiled C equivalent.

    Parameters
    ----------
    model : nn.Module
        The model to verify (will be set to eval mode).
    example_input : torch.Tensor
        A single example input (batch-size 1).
    num_samples : int
        Number of random test vectors to generate.
    quantization_rules : list[QuantRule] | None
        If supplied, quantization is applied to the IR before codegen.
    passes : list[IRPass] | None
        Optional IR passes to apply before quantization (e.g. pruning).
    tolerance : float
        Maximum absolute error per element for a sample to be considered passing.
    verbose : bool
        Print per-sample details.
    openmp : bool
        Build with -fopenmp so parallel regions (e.g. LQER diamonds) execute
        concurrently. Without it, the OpenMP pragmas are ignored and the same
        code runs sequentially.

    Returns
    -------
    VerificationResults
    """
    if not _gcc_available():
        raise RuntimeError("gcc is required but not found on PATH")

    model.eval()
    is_quantized = quantization_rules is not None and len(quantization_rules) > 0
    passes = passes or []

    input_flat_size = _compute_flat_input_size(example_input)
    with torch.no_grad():
        sample_output = model(example_input)
    output_flat_size = int(np.prod(list(sample_output.shape)[1:])) if sample_output.dim() > 1 else int(np.prod(sample_output.shape))

    results = VerificationResults(
        num_samples=num_samples,
        tolerance=tolerance,
        quantized=is_quantized,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Compile model to IR
        ir_graph = compile_model(model, example_input, return_ir=True, verbose=False)

        # 2. Apply structural passes first (pruning, etc.)
        for p in passes:
            ir_graph = p.apply(ir_graph)

        # 3. Apply quantization if requested
        if is_quantized:
            from src.pytorch_to_c.quantization import QuantizationTransform
            ir_graph = QuantizationTransform(quantization_rules).apply(ir_graph)

        # 4. Generate C code
        CPrinter(ir_graph).generate_all(tmpdir)

        # 5. Write C harness and compile
        harness_path = _write_harness(tmpdir, input_flat_size, output_flat_size)
        exe = _compile_c(tmpdir, harness_path, openmp=openmp)

        # 6. Generate random inputs and collect PyTorch outputs
        torch.manual_seed(42)
        all_inputs_flat = []
        pytorch_outputs = []
        for _ in range(num_samples):
            inp = torch.randn_like(example_input)
            with torch.no_grad():
                out = model(inp)
            all_inputs_flat.append(_nchw_to_nhwc_flat(inp))
            pytorch_outputs.append(_nchw_to_nhwc_flat(out))

        # 7. Write all inputs as a single binary blob
        input_bin = os.path.join(tmpdir, "inputs.bin")
        output_bin = os.path.join(tmpdir, "outputs.bin")
        np.concatenate(all_inputs_flat).tofile(input_bin)

        # 8. Run C model on all samples
        _run_c(exe, num_samples, input_bin, output_bin)

        # 9. Read C outputs
        c_raw = np.fromfile(output_bin, dtype=np.float32)
        c_outputs = c_raw.reshape(num_samples, output_flat_size)

        # 10. Compare
        for i in range(num_samples):
            pt = pytorch_outputs[i]
            c = c_outputs[i]
            max_err = float(np.max(np.abs(pt - c)))
            mean_err = float(np.mean(np.abs(pt - c)))
            results.max_errors.append(max_err)
            results.mean_errors.append(mean_err)

            passed = max_err <= tolerance
            if passed:
                results.passed += 1
            else:
                results.failed += 1

            # Top-1 class match (useful for classification models)
            if output_flat_size > 1:
                results.top1_total += 1
                if np.argmax(pt) == np.argmax(c):
                    results.top1_matches += 1

            if verbose:
                status = "PASS" if passed else "FAIL"
                print(f"  [{status}] sample {i}: max_err={max_err:.2e}, mean_err={mean_err:.2e}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    import importlib

    parser = argparse.ArgumentParser(description="Verify a PyTorch model compiled to C")
    parser.add_argument("--model", required=True,
                        help="module_path:ClassName (e.g. models/tiny_mlp.py:TinyMLP)")
    parser.add_argument("--input-shape", required=True,
                        help="Comma-separated input shape, e.g. 1,784")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # Parse --model
    mod_path, cls_name = args.model.rsplit(":", 1)
    if mod_path.endswith(".py"):
        mod_path = mod_path[:-3]
    mod_path = mod_path.replace("/", ".").replace("\\", ".")
    mod = importlib.import_module(mod_path)
    model_cls = getattr(mod, cls_name)
    model = model_cls()

    shape = [int(s) for s in args.input_shape.split(",")]
    example_input = torch.randn(*shape)

    results = verify_model(
        model, example_input,
        num_samples=args.num_samples,
        tolerance=args.tolerance,
        verbose=args.verbose,
    )
    print(results.summary())

    sys.exit(0 if results.failed == 0 else 1)


if __name__ == "__main__":
    main()
