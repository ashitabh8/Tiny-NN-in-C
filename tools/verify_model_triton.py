"""
Verify PyTorch model against Triton-generated model.py on GPU.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pytorch_to_c.compiler import compile_model
from tools.verify_model import VerificationResults


def _nchw_to_nhwc_flat(x: torch.Tensor) -> torch.Tensor:
    """Convert NCHW batch-1 tensor to flat NHWC on same device."""
    if x.dim() == 4:
        x = x.permute(0, 2, 3, 1).contiguous()
    return x.reshape(-1)


def verify_model_triton(
    model: nn.Module,
    example_input: torch.Tensor,
    num_samples: int = 10,
    tolerance: float = 1e-3,
    device: str = "cuda",
    quantize_fn=None,
) -> VerificationResults:
    """
    Compile with Triton backend and compare outputs to PyTorch reference.

    Args:
        model: PyTorch model (eval mode recommended)
        example_input: Example input for tracing/shape inference
        num_samples: Number of random inputs to compare
        tolerance: Max absolute error threshold per sample
        device: CUDA device string
        quantize_fn: Optional callable(ir_graph) -> ir_graph for quantization transforms
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for Triton verification")

    model = model.eval()
    results = VerificationResults(tolerance=tolerance, quantized=quantize_fn is not None)

    with tempfile.TemporaryDirectory() as tmp:
        ir = compile_model(
            model, example_input, output_dir=tmp, verbose=False, return_ir=False, backend="triton"
        )
        if quantize_fn is not None:
            from src.pytorch_to_c.codegen.triton_printer import generate_triton_code
            ir = quantize_fn(ir)
            generate_triton_code(ir, tmp)

        spec = importlib.util.spec_from_file_location("triton_gen_model", os.path.join(tmp, "model.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        out_node = ir.outputs[0]
        out_size = int(np.prod(out_node.output_shape[1:]) if out_node.output_shape[0] == 1
                         else np.prod(out_node.output_shape))

        dev = torch.device(device)
        for _ in range(num_samples):
            with torch.no_grad():
                x = torch.randn_like(example_input)
                pt_out = model(x).detach().flatten()
                inp_flat = _nchw_to_nhwc_flat(x.to(dev)).float()
                triton_out = torch.empty(out_size, device=dev, dtype=torch.float32)
                mod.model_forward(inp_flat, triton_out)
                err = (pt_out.to(dev) - triton_out).abs()
                max_err = float(err.max().item())
                mean_err = float(err.mean().item())
                results.num_samples += 1
                results.max_errors.append(max_err)
                results.mean_errors.append(mean_err)
                if max_err <= tolerance:
                    results.passed += 1
                else:
                    results.failed += 1
                if pt_out.numel() > 1:
                    results.top1_total += 1
                    if int(pt_out.argmax()) == int(triton_out.argmax()):
                        results.top1_matches += 1

    return results


if __name__ == "__main__":
    from test.test_models import TinyMLP

    m = TinyMLP(input_size=16, hidden_size=8, output_size=4)
    ex = torch.randn(1, 16)
    r = verify_model_triton(m, ex, num_samples=5)
    print(r.summary())
