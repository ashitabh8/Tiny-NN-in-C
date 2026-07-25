"""
Tests for LQER (Low-rank Quantization Error Reconstruction) IR nodes.

Covers:
- svd_factorizer shapes and rank-monotone reconstruction error
- Diamond graph wiring (dynamic + static, linear + conv2d)
- Memory planner: no buffer aliasing between the two concurrent chains
- OpenMP section emission in generated C
- End-to-end C numerics: LQER-corrected output beats uncorrected quantization
- Compatibility with the existing pass design (FuseDequantQuantPass)
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.pytorch_to_c.compiler import compile_model
from src.pytorch_to_c.codegen.c_printer import CPrinter
from src.pytorch_to_c.codegen.memory_planner import (
    assign_buffer_slots,
    calculate_buffer_sizes,
    compute_buffer_last_use,
)
from src.pytorch_to_c.codegen.parallel_regions import find_parallel_regions
from src.pytorch_to_c.quantization import (
    QuantizationTransform,
    DynamicQuantRuleMinMaxPerTensor,
    LQERDynamicQuantRule,
    LQERStaticQuantRule,
    StaticQuantRule,
    svd_factorizer,
    LQERMatmulNode,
    LQERConvMatmulNode,
    LQERAddNode,
)
from src.pytorch_to_c.quantization.ops.quant_utils import DequantizeNode
from src.passes import FuseDequantQuantPass
from tools.verify_model import verify_model


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TinyMLP(nn.Module):
    def __init__(self, in_f=16, hidden=12, out_f=4):
        super().__init__()
        self.fc1 = nn.Linear(in_f, hidden)
        self.fc2 = nn.Linear(hidden, out_f)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TinyConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.mean(dim=[2, 3])  # global average pool (NHWC-safe in the C backend)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _symmetric_quant_error(weights: np.ndarray, dtype: str = "int8") -> np.ndarray:
    """E = W - dequant(quant(W)) using the same absmax-symmetric scheme as
    DynamicQuantRuleMinMaxPerTensor / StaticQuantRule with matching scale."""
    q_max = 127.0 if dtype == "int8" else 32767.0
    absmax = float(np.max(np.abs(weights)))
    scale = (absmax / q_max) if absmax > 0 else (1.0 / q_max)
    wq = np.clip(np.round(weights / scale), -q_max - 1, q_max)
    return weights - wq * scale, scale


def _error_matrices_for(model, example_input, layer_names, dtype="int8"):
    """Compute per-layer error matrices in the compiler's stored weight layout."""
    ir = compile_model(model, example_input, return_ir=True, verbose=False)
    errors = {}
    scales = {}
    for name in layer_names:
        w = ir.parameters[f"{name}_weight"]
        errors[name], scales[name] = _symmetric_quant_error(w, dtype)
    return errors, scales


# ---------------------------------------------------------------------------
# Factorizer
# ---------------------------------------------------------------------------

class TestSVDFactorizer:
    def test_shapes(self):
        E = np.random.RandomState(0).randn(20, 10)
        A, B = svd_factorizer(E, rank=4)
        assert A.shape == (20, 4)
        assert B.shape == (4, 10)

    def test_rank_clamped(self):
        E = np.random.RandomState(0).randn(6, 3)
        A, B = svd_factorizer(E, rank=100)
        assert A.shape[1] == 3  # min(6, 3)

    def test_error_decreases_with_rank(self):
        E = np.random.RandomState(0).randn(20, 10)
        errs = []
        for r in (1, 4, 10):
            A, B = svd_factorizer(E, r)
            errs.append(np.linalg.norm(E - A @ B))
        assert errs[0] > errs[1] > errs[2]
        assert errs[2] < 1e-5  # full rank reconstructs exactly

    def test_custom_factorizer_plugs_in(self):
        """A user factorizer (e.g. PCA) only needs the (E, rank)->(A, B) signature."""
        calls = []

        def my_factorizer(E, rank):
            calls.append(rank)
            return np.zeros((E.shape[0], rank), np.float32), \
                   np.zeros((rank, E.shape[1]), np.float32)

        model = TinyMLP()
        x = torch.randn(1, 16)
        errors, _ = _error_matrices_for(model, x, ["fc1"])
        rule = LQERDynamicQuantRule("fc1", "int8", error_matrix=errors,
                                    rank=3, factorizer=my_factorizer)
        ir = compile_model(model, x, return_ir=True, verbose=False)
        QuantizationTransform([rule]).apply(ir)
        assert calls == [3]


# ---------------------------------------------------------------------------
# Graph structure
# ---------------------------------------------------------------------------

class TestDiamondWiring:
    def _transform(self, model, x, rules):
        ir = compile_model(model, x, return_ir=True, verbose=False)
        return QuantizationTransform(rules).apply(ir)

    def test_dynamic_linear_diamond(self):
        model = TinyMLP()
        x = torch.randn(1, 16)
        errors, _ = _error_matrices_for(model, x, ["fc1"])
        rule = LQERDynamicQuantRule("fc1", "int8", error_matrix=errors["fc1"], rank=4)
        ir = self._transform(model, x, [rule])
        ir.validate()

        down = ir.get_node_by_name("fc1_lqer_down")
        up = ir.get_node_by_name("fc1_lqer_up")
        join = ir.get_node_by_name("fc1_lqer_add")
        fc1 = ir.get_node_by_name("fc1")
        assert isinstance(down, LQERMatmulNode)
        assert isinstance(up, LQERMatmulNode)
        assert isinstance(join, LQERAddNode)

        # Branch taps the ORIGINAL float input, not the quantize node
        assert down.inputs[0].dtype == "float32"
        assert down.inputs[0].op_type == "input"
        # Quantized path input goes through the dynamic quantize node
        assert fc1.inputs[0].op_type == "dynamic_quantize"
        # Both quantize node and branch head share the same source
        assert fc1.inputs[0].inputs[0] is down.inputs[0]

        # Join merges [quant path (float output), branch]
        assert join.inputs == [fc1, up]
        assert fc1.users == [join]
        # Original user (relu) now consumes the join
        assert any(u.op_type == "relu" for u in join.users)

        # Low-rank factors registered in the compiler's stored layout
        assert ir.parameters["fc1_lqer_A"].shape == (16, 4)
        assert ir.parameters["fc1_lqer_B"].shape == (4, 12)

    def test_static_linear_join_after_dequant(self):
        model = TinyMLP()
        x = torch.randn(1, 16)
        errors, scales = _error_matrices_for(model, x, ["fc1"])
        rule = LQERStaticQuantRule(
            "fc1", "int8",
            input_scale=0.05, input_offset=0,
            weight_scale=scales["fc1"], weight_offset=0,
            output_scale=0.05, output_offset=0,
            error_matrix=errors["fc1"], rank=4,
        )
        ir = self._transform(model, x, [rule])
        ir.validate()

        join = ir.get_node_by_name("fc1_lqer_add")
        # The dequantize post-node was spliced between the int8 layer and the join
        assert isinstance(join.inputs[0], DequantizeNode)
        assert join.inputs[0].dtype == "float32"
        assert join.inputs[1] is ir.get_node_by_name("fc1_lqer_up")

    def test_dynamic_conv_diamond(self):
        model = TinyConvNet()
        x = torch.randn(1, 3, 8, 8)
        errors, _ = _error_matrices_for(model, x, ["conv1"])
        rule = LQERDynamicQuantRule("conv1", "int8",
                                    error_matrix=errors["conv1"], rank=4)
        ir = self._transform(model, x, [rule])
        ir.validate()

        down = ir.get_node_by_name("conv1_lqer_down")
        up = ir.get_node_by_name("conv1_lqer_up")
        assert isinstance(down, LQERConvMatmulNode)
        assert isinstance(up, LQERConvMatmulNode)
        # down: same receptive field, r output channels
        assert down.metadata["kernel_size"] == (3, 3)
        assert down.metadata["out_channels"] == 4
        # up: 1x1 conv back to out_channels
        assert up.metadata["kernel_size"] == (1, 1)
        assert up.metadata["in_channels"] == 4
        assert up.metadata["out_channels"] == 8
        # HWIO factor layouts
        assert ir.parameters["conv1_lqer_A"].shape == (3, 3, 3, 4)
        assert ir.parameters["conv1_lqer_B"].shape == (1, 1, 4, 8)

    def test_consecutive_lqer_layers(self):
        """Two LQER layers back to back: each branch taps a float32 tensor."""
        model = TinyMLP()
        x = torch.randn(1, 16)
        errors, _ = _error_matrices_for(model, x, ["fc1", "fc2"])
        rule = LQERDynamicQuantRule("fc[12]", "int8", error_matrix=errors, rank=4)
        ir = self._transform(model, x, [rule])
        ir.validate()

        for name in ("fc1", "fc2"):
            down = ir.get_node_by_name(f"{name}_lqer_down")
            assert down.inputs[0].dtype == "float32"
        # fc2's diamond consumes fc1's relu output
        fc2_down = ir.get_node_by_name("fc2_lqer_down")
        assert fc2_down.inputs[0].op_type == "relu"

    def test_error_matrix_shape_mismatch_raises(self):
        model = TinyMLP()
        x = torch.randn(1, 16)
        bad = np.zeros((3, 3))
        rule = LQERDynamicQuantRule("fc1", "int8", error_matrix=bad, rank=2)
        ir = compile_model(model, x, return_ir=True, verbose=False)
        with pytest.raises(ValueError, match="does not match weight layout"):
            QuantizationTransform([rule]).apply(ir)

    def test_missing_dict_entry_raises(self):
        model = TinyMLP()
        x = torch.randn(1, 16)
        rule = LQERDynamicQuantRule("fc1", "int8",
                                    error_matrix={"other": np.zeros((16, 12))},
                                    rank=2)
        ir = compile_model(model, x, return_ir=True, verbose=False)
        with pytest.raises(ValueError, match="no error matrix provided"):
            QuantizationTransform([rule]).apply(ir)


# ---------------------------------------------------------------------------
# Codegen: regions, OpenMP, memory planning
# ---------------------------------------------------------------------------

class TestCodegen:
    def _lqer_ir(self):
        model = TinyMLP()
        x = torch.randn(1, 16)
        errors, _ = _error_matrices_for(model, x, ["fc1"])
        rule = LQERDynamicQuantRule("fc1", "int8", error_matrix=errors["fc1"], rank=4)
        ir = compile_model(model, x, return_ir=True, verbose=False)
        return QuantizationTransform([rule]).apply(ir)

    def test_region_detected(self):
        ir = self._lqer_ir()
        order = ir.topological_sort()
        regions = find_parallel_regions(order)
        assert len(regions) == 1
        region = regions[0]
        assert region.join.name == "fc1_lqer_add"
        assert [n.name for n in region.branch_chain] == ["fc1_lqer_down", "fc1_lqer_up"]
        assert [n.op_type for n in region.main_chain] == ["dynamic_quantize", "linear"]
        assert region.source.op_type == "input"

    def test_openmp_sections_emitted(self):
        ir = self._lqer_ir()
        code = CPrinter(ir).generate_model_c()
        assert code.count("#pragma omp parallel sections") == 1
        assert code.count("#pragma omp section") == 2
        # The join add is emitted AFTER the parallel region closes
        region_end = code.rindex("#pragma omp section")
        assert "fc1_lqer_add" in code[region_end:]

    def test_no_slot_aliasing_between_chains(self):
        """Buffers of the two concurrent chains must not share memory slots."""
        ir = self._lqer_ir()
        order = ir.topological_sort()
        sizes = calculate_buffer_sizes(ir)
        last_use = compute_buffer_last_use(order)
        slots, _, _, _ = assign_buffer_slots(order, sizes, last_use)

        # fc1 output (float, quant path) vs branch buffers (float): all distinct
        region_bufs = ["fc1", "fc1_lqer_down", "fc1_lqer_up"]
        assigned = [slots[name] for name in region_bufs]
        assert len(set(assigned)) == len(assigned), (
            f"parallel-region buffers alias slots: "
            f"{dict(zip(region_bufs, assigned))}"
        )

    def test_plain_graphs_have_no_regions(self):
        """No LQER -> no regions -> emission path identical to before."""
        model = TinyMLP()
        x = torch.randn(1, 16)
        ir = compile_model(model, x, return_ir=True, verbose=False)
        assert find_parallel_regions(ir.topological_sort()) == []
        code = CPrinter(ir).generate_model_c()
        assert "#pragma omp" not in code


# ---------------------------------------------------------------------------
# End-to-end C numerics
# ---------------------------------------------------------------------------

class TestEndToEndC:
    def test_dynamic_linear_lqer_improves_over_plain_quant(self):
        torch.manual_seed(7)
        model = TinyMLP()
        x = torch.randn(1, 16)
        errors, _ = _error_matrices_for(model, x, ["fc1", "fc2"])

        plain = verify_model(
            model, x, num_samples=20, tolerance=1e9,
            quantization_rules=[DynamicQuantRuleMinMaxPerTensor("fc[12]", "int8")],
        )
        # Full-rank correction cancels the weight-quantization error entirely
        lqer = verify_model(
            model, x, num_samples=20, tolerance=1e9,
            quantization_rules=[LQERDynamicQuantRule(
                "fc[12]", "int8", error_matrix=errors, rank=16)],
        )
        assert lqer.overall_mean_error < plain.overall_mean_error, (
            f"LQER mean err {lqer.overall_mean_error:.2e} should beat plain "
            f"quant {plain.overall_mean_error:.2e}"
        )

    def test_dynamic_linear_lqer_with_openmp(self):
        """Same model built with -fopenmp: threads must not change numerics."""
        torch.manual_seed(7)
        model = TinyMLP()
        x = torch.randn(1, 16)
        errors, _ = _error_matrices_for(model, x, ["fc1", "fc2"])
        rules = [LQERDynamicQuantRule("fc[12]", "int8", error_matrix=errors, rank=16)]

        seq = verify_model(model, x, num_samples=20, tolerance=1e9,
                           quantization_rules=rules, openmp=False)
        par = verify_model(model, x, num_samples=20, tolerance=1e9,
                           quantization_rules=rules, openmp=True)
        assert par.overall_max_error == pytest.approx(seq.overall_max_error, rel=1e-6)

    def test_static_linear_lqer_runs(self):
        torch.manual_seed(7)
        model = TinyMLP()
        x = torch.randn(1, 16)
        errors, scales = _error_matrices_for(model, x, ["fc1"])
        rules = [LQERStaticQuantRule(
            "fc1", "int8",
            input_scale=0.05, input_offset=0,
            weight_scale=scales["fc1"], weight_offset=0,
            output_scale=0.05, output_offset=0,
            error_matrix=errors["fc1"], rank=12,
        )]
        results = verify_model(model, x, num_samples=10, tolerance=1e9,
                               quantization_rules=rules, openmp=True)
        assert results.num_samples == 10
        assert np.isfinite(results.overall_max_error)
        # Static quant of one layer on a tiny model: output should stay sane
        assert results.overall_mean_error < 1.0

    def test_dynamic_conv_lqer_improves_over_plain_quant(self):
        torch.manual_seed(3)
        model = TinyConvNet()
        x = torch.randn(1, 3, 8, 8)
        errors, _ = _error_matrices_for(model, x, ["conv1"])

        plain = verify_model(
            model, x, num_samples=10, tolerance=1e9,
            quantization_rules=[DynamicQuantRuleMinMaxPerTensor("conv1", "int8")],
        )
        lqer = verify_model(
            model, x, num_samples=10, tolerance=1e9,
            quantization_rules=[LQERDynamicQuantRule(
                "conv1", "int8", error_matrix=errors["conv1"], rank=8)],
            openmp=True,
        )
        assert lqer.overall_mean_error < plain.overall_mean_error


# ---------------------------------------------------------------------------
# Compatibility with the existing pass design
# ---------------------------------------------------------------------------

class TestPassCompatibility:
    def test_fuse_dequant_quant_leaves_lqer_intact(self):
        """Static LQER layer followed by a plain static layer: the fusion pass
        must not touch the dequant feeding the LQER join (its user is an add,
        not a quantize)."""
        model = TinyMLP()
        x = torch.randn(1, 16)
        errors, scales = _error_matrices_for(model, x, ["fc1"])
        rules = [
            LQERStaticQuantRule(
                "fc1", "int8",
                input_scale=0.05, input_offset=0,
                weight_scale=scales["fc1"], weight_offset=0,
                output_scale=0.05, output_offset=0,
                error_matrix=errors["fc1"], rank=4,
            ),
            StaticQuantRule(
                "fc2", "int8",
                input_scale=0.05, input_offset=0,
                weight_scale=0.05, weight_offset=0,
                output_scale=0.05, output_offset=0,
            ),
        ]
        ir = compile_model(model, x, return_ir=True, verbose=False)
        ir = QuantizationTransform(rules).apply(ir)

        fuse = FuseDequantQuantPass()
        ir = fuse.apply(ir)
        ir.validate()

        # The LQER dequant survives and still feeds the join
        join = ir.get_node_by_name("fc1_lqer_add")
        assert join is not None
        assert isinstance(join.inputs[0], DequantizeNode)
        # Region is still detectable after the pass
        assert len(find_parallel_regions(ir.topological_sort())) == 1

    def test_non_lqer_rules_unaffected(self):
        """Plain dynamic quantization still produces the same graph shape
        (no branches, no joins) - the new transform step is a no-op."""
        model = TinyMLP()
        x = torch.randn(1, 16)
        ir = compile_model(model, x, return_ir=True, verbose=False)
        ir = QuantizationTransform(
            [DynamicQuantRuleMinMaxPerTensor("fc[12]", "int8")]
        ).apply(ir)
        ir.validate()
        assert not any(n.metadata.get("parallel_join") for n in ir.nodes)
        assert not any(n.metadata.get("parallel_branch") for n in ir.nodes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
