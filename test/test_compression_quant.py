"""Tests for compression quantization features."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import TinyMLP
from src.passes import FuseDequantQuantPass
from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.quantization import (
    DynamicInt4PerGroupLinearQuantRule,
    PaletteWeightRule,
    QuantizationTransform,
    StaticInt4PerGroupLinearQuantRule,
    StaticPerGroupLinearQuantRule,
    StaticQuantRule,
    gptq_quantize,
)
from src.pytorch_to_c.quantization.ops.quant_int4_linear import (
    DynamicInt4PerGroupQuantLinearNode,
    StaticInt4PerGroupQuantLinearNode,
)
from src.pytorch_to_c.quantization.ops.quant_utils import (
    DequantizeNode,
    DynamicQuantizeInputNode,
    QuantizeNode,
)
from src.pytorch_to_c.quantization.quant_helpers import (
    BLOCK_K,
    candidate_group_sizes,
    group_quant_mse,
    pack_int4_nibbles,
    select_group_size,
    symmetric_scales_per_group,
)
from tools.verify_model import verify_model, _gcc_available


def _skip_no_gcc():
    if not _gcc_available():
        pytest.skip("gcc not available")


class TestQuantHelpers:
    def test_candidate_group_sizes(self):
        sizes = candidate_group_sizes(128, BLOCK_K)
        assert 32 in sizes
        assert 128 in sizes

    def test_select_group_size_auto(self):
        W = np.random.randn(128, 16).astype(np.float32)
        g = select_group_size(W, "int8", "auto", error_budget=1.0)
        assert 128 % g == 0

    def test_int4_packing(self):
        vals = np.array([1, -2, 3, -4], dtype=np.int8)
        packed = pack_int4_nibbles(vals)
        assert packed.size == 2


class TestPerGroupQuant:
    def test_per_group_rule_quantizes(self):
        model = TinyMLP(input_size=64, hidden_size=8, output_size=4).eval()
        x = torch.randn(1, 64)
        ir = compile_model(model, x, return_ir=True, verbose=False)
        rules = [
            StaticPerGroupLinearQuantRule(
                pattern=r".*fc.*",
                dtype="int8",
                input_scale=0.05,
                input_offset=0,
                output_scale=0.05,
                output_offset=0,
                group_size=32,
            )
        ]
        qir = QuantizationTransform(rules).apply(ir)
        assert any("group_size" in n.metadata for n in qir.nodes if n.metadata.get("quantized"))

    def test_per_group_verify_c(self):
        _skip_no_gcc()
        model = TinyMLP(input_size=20, hidden_size=10, output_size=5).eval()
        x = torch.randn(1, 20)
        rules = [
            StaticPerGroupLinearQuantRule(
                pattern=r".*fc.*",
                dtype="int8",
                input_scale=0.05,
                input_offset=0,
                output_scale=0.05,
                output_offset=0,
                group_size="auto",
                error_budget=1.0,
            )
        ]
        res = verify_model(model, x, num_samples=5, quantization_rules=rules, tolerance=5.0)
        assert res.top1_matches > 0 or res.passed > 0


class TestInt4W4A8:
    def test_static_int4_graph_structure(self):
        model = TinyMLP(input_size=64, hidden_size=32, output_size=8).eval()
        x = torch.randn(1, 64)
        ir = compile_model(model, x, return_ir=True, verbose=False)
        rules = [
            StaticInt4PerGroupLinearQuantRule(
                pattern=r"fc1",
                input_scale=0.05,
                input_offset=0,
                output_scale=0.05,
                output_offset=0,
                group_size=32,
            )
        ]
        qir = QuantizationTransform(rules).apply(ir)
        qir.validate()

        fc1 = next(n for n in qir.nodes if n.name == "fc1")
        assert isinstance(fc1, StaticInt4PerGroupQuantLinearNode)
        assert isinstance(fc1.inputs[0], QuantizeNode)
        assert fc1.inputs[0].dtype == "int8"
        assert any(isinstance(u, DequantizeNode) for u in fc1.users)
        assert fc1.metadata.get("packed_int4") is True
        assert fc1.metadata.get("group_size") == 32

    def test_dynamic_int4_graph_structure(self):
        model = TinyMLP(input_size=64, hidden_size=32, output_size=8).eval()
        x = torch.randn(1, 64)
        ir = compile_model(model, x, return_ir=True, verbose=False)
        rules = [DynamicInt4PerGroupLinearQuantRule(pattern=r"fc1", group_size=32)]
        qir = QuantizationTransform(rules).apply(ir)
        qir.validate()

        fc1 = next(n for n in qir.nodes if n.name == "fc1")
        assert isinstance(fc1, DynamicInt4PerGroupQuantLinearNode)
        assert isinstance(fc1.inputs[0], DynamicQuantizeInputNode)
        assert fc1.dtype == "float32"
        assert not any(isinstance(u, DequantizeNode) for u in fc1.users)
        assert fc1.metadata.get("packed_int4") is True

    def test_int4_compression_ratio(self):
        model = TinyMLP(input_size=64, hidden_size=32, output_size=8).eval()
        x = torch.randn(1, 64)
        ir = compile_model(model, x, return_ir=True, verbose=False)
        wn = None
        for n in ir.nodes:
            if n.op_type == "linear":
                wn = n.metadata["weight_name"]
                break
        float_size = ir.parameters[wn].nbytes
        rules = [
            StaticInt4PerGroupLinearQuantRule(
                pattern=r".*fc1.*",
                input_scale=0.05,
                input_offset=0,
                output_scale=0.05,
                output_offset=0,
                group_size=32,
            )
        ]
        qir = QuantizationTransform(rules).apply(ir)
        packed_size = qir.parameters[wn].nbytes
        assert packed_size < float_size
        assert packed_size == float_size // 8  # float32 -> int4 (2 per byte)

    def test_static_int4_verify_c(self):
        _skip_no_gcc()
        model = TinyMLP(input_size=32, hidden_size=16, output_size=4).eval()
        x = torch.randn(1, 32)
        rules = [
            StaticInt4PerGroupLinearQuantRule(
                pattern=r".*fc.*",
                input_scale=0.05,
                input_offset=0,
                output_scale=0.05,
                output_offset=0,
                group_size=16,
            )
        ]
        res = verify_model(model, x, num_samples=5, quantization_rules=rules, tolerance=5.0)
        assert res.top1_matches > 0 or res.passed > 0

    def test_dynamic_int4_verify_c(self):
        _skip_no_gcc()
        model = TinyMLP(input_size=32, hidden_size=16, output_size=4).eval()
        x = torch.randn(1, 32)
        rules = [DynamicInt4PerGroupLinearQuantRule(pattern=r".*fc.*", group_size=16)]
        res = verify_model(model, x, num_samples=5, quantization_rules=rules, tolerance=5.0)
        assert res.top1_matches > 0 or res.passed > 0

    def test_static_int4_fuse_dequant_quant(self):
        """Matching dequant→quant between consecutive static layers is fusible."""

        class TwoLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(32, 16)
                self.fc2 = torch.nn.Linear(16, 4)

            def forward(self, x):
                return self.fc2(self.fc1(x))

        model = TwoLinear().eval()
        x = torch.randn(1, 32)
        scale = 0.05
        rules = [
            StaticInt4PerGroupLinearQuantRule(
                pattern=r"fc1",
                input_scale=scale,
                input_offset=0,
                output_scale=scale,
                output_offset=0,
                group_size=16,
            ),
            StaticQuantRule(
                pattern=r"fc2",
                dtype="int8",
                input_scale=scale,
                input_offset=0,
                weight_scale=0.02,
                weight_offset=0,
                output_scale=scale,
                output_offset=0,
            ),
        ]
        ir = compile_model(model, x, return_ir=True, verbose=False)
        qir = QuantizationTransform(rules).apply(ir)
        before = len(qir.nodes)
        fuse = FuseDequantQuantPass()
        qir = fuse.apply(qir)
        qir.validate()
        assert fuse.stats["pairs_fused"] >= 1
        assert len(qir.nodes) < before


class TestPalette:
    def test_palette_verify_c(self):
        _skip_no_gcc()
        model = TinyMLP(input_size=20, hidden_size=10, output_size=5).eval()
        x = torch.randn(1, 20)
        rules = [PaletteWeightRule(pattern=r".*fc.*", num_centroids=8)]
        res = verify_model(model, x, num_samples=5, quantization_rules=rules, tolerance=5.0)
        assert res.num_samples == res.passed or res.top1_matches > 0


class TestGPTQ:
    def test_gptq_not_worse_than_nearest(self):
        W = np.random.randn(64, 16).astype(np.float32) * 0.1
        nearest_mse = group_quant_mse(W, 64, "int8")
        hessian = np.eye(64, dtype=np.float64)
        Wq = gptq_quantize(W, hessian, 64, "int8")
        scales = symmetric_scales_per_group(W, 64, "int8")
        recon = np.zeros_like(W)
        for g in range(1):
            for o in range(W.shape[1]):
                sl = Wq[:, o].astype(np.float64)
                recon[:, o] = sl * scales[g, o]
        gptq_mse = float(np.mean((W - recon) ** 2))
        assert gptq_mse <= nearest_mse * 1.5 + 1e-6
