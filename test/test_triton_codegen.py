"""Tests for Triton code generation."""

import os
import tempfile

import pytest
import torch

from src.pytorch_to_c.codegen.triton_printer import TritonPrinter, generate_triton_code
from src.pytorch_to_c.frontend.fx_tracer import trace_model
from src.pytorch_to_c.lowering.lower import lower_fx_graph
from test.test_models import TinyMLP


def test_generate_triton_model_py():
    model = TinyMLP(input_size=8, hidden_size=4, output_size=2)
    example = torch.randn(1, 8)
    ir = lower_fx_graph(trace_model(model, example))
    printer = TritonPrinter(ir)
    code = printer.generate_model_py()
    assert "def model_forward" in code
    assert "ops_f.dense" in code
    assert "slot_0" in code


def test_generate_all_writes_files():
    model = TinyMLP(input_size=4, hidden_size=3, output_size=2)
    example = torch.randn(1, 4)
    ir = lower_fx_graph(trace_model(model, example))
    with tempfile.TemporaryDirectory() as tmp:
        generate_triton_code(ir, tmp)
        assert os.path.isfile(os.path.join(tmp, "model.py"))
        assert os.path.isfile(os.path.join(tmp, "weights.npz"))
        assert os.path.isdir(os.path.join(tmp, "triton_ops"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_generated_model_forward_runs():
    model = TinyMLP(input_size=8, hidden_size=4, output_size=2)
    example = torch.randn(1, 8)
    ir = lower_fx_graph(trace_model(model, example))
    with tempfile.TemporaryDirectory() as tmp:
        generate_triton_code(ir, tmp)
        import importlib.util
        spec = importlib.util.spec_from_file_location("gen_model", os.path.join(tmp, "model.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        inp = torch.randn(8, device="cuda")
        out = torch.empty(2, device="cuda")
        mod.model_forward(inp, out)
        assert out.shape == (2,)
        assert torch.isfinite(out).all()
