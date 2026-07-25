"""Unit tests for QuantLinearConfig validator and QuantLinearNode shell."""

import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pytorch_to_c.ir.node import IRNode
from src.pytorch_to_c.quantization.ops.quant_linear_general import QuantLinearNode
from src.pytorch_to_c.quantization.ops.quant_utils import (
    DequantizeNode,
    DynamicQuantizeInputNode,
    QuantizeNode,
)
from src.pytorch_to_c.quantization.quant_config import (
    AGran,
    IllegalConfig,
    QuantLinearConfig,
    UnimplementedConfig,
    validate,
)
from tools.gen_legal_config_doc import render_doc as render_doc_tool


def _linear_node(in_features: int = 128, out_features: int = 16) -> IRNode:
    n = IRNode(
        name="fc1",
        op_type="linear",
        output_shape=(1, out_features),
        dtype="float32",
        metadata={
            "in_features": in_features,
            "out_features": out_features,
            "input_shape": (1, in_features),
            "weight_name": "fc1_weight",
        },
    )
    return n


class TestValidatorCells:
    def test_w8a8_static_per_tensor_legal_implemented(self):
        cfg = QuantLinearConfig(8, 8, 128, False, AGran.PER_TENSOR)
        validate(cfg, "c")
        validate(cfg, "triton")

    def test_w16a16_per_group_legal_implemented(self):
        cfg = QuantLinearConfig(16, 16, 32, True, AGran.PER_TENSOR)
        validate(cfg, "c")
        validate(cfg, "triton")

    def test_w4a8_legal_implemented_c_and_triton(self):
        cfg = QuantLinearConfig(4, 8, 32, True, AGran.PER_TENSOR)
        validate(cfg, "c")
        validate(cfg, "triton")

    def test_w4a16_raises_unimplemented(self):
        cfg = QuantLinearConfig(4, 16, 32, True, AGran.PER_TENSOR)
        with pytest.raises(UnimplementedConfig):
            validate(cfg, "c")

    def test_w8a16_raises_unimplemented(self):
        cfg = QuantLinearConfig(8, 16, 128, False, AGran.PER_TENSOR)
        with pytest.raises(UnimplementedConfig):
            validate(cfg, "c")

    def test_w4a4_raises_unimplemented(self):
        cfg = QuantLinearConfig(4, 4, 32, True, AGran.PER_TENSOR)
        with pytest.raises(UnimplementedConfig):
            validate(cfg, "c")

    def test_w16a8_raises_illegal(self):
        cfg = QuantLinearConfig(16, 8, 128, False, AGran.PER_TENSOR)
        with pytest.raises(IllegalConfig):
            validate(cfg, "c")

    def test_per_token_raises_unimplemented_not_illegal(self):
        """Inverts in Phase 4: should validate + codegen once PER_TOKEN lands."""
        cfg = QuantLinearConfig(8, 8, 128, False, AGran.PER_TOKEN)
        with pytest.raises(UnimplementedConfig) as exc:
            validate(cfg, "triton")
        assert "PER_TOKEN" in str(exc.value)
        assert not isinstance(exc.value, IllegalConfig)

    def test_per_token_not_illegal_config_type(self):
        cfg = QuantLinearConfig(8, 8, 128, False, AGran.PER_TOKEN)
        with pytest.raises(UnimplementedConfig):
            validate(cfg, "c")

    def test_int4_non_32_group_legal(self):
        cfg = QuantLinearConfig(4, 8, 48, True, AGran.PER_TENSOR)
        validate(cfg, "c", in_features=96)

    def test_w8_per_group_non_32_raises_illegal(self):
        cfg = QuantLinearConfig(8, 8, 16, True, AGran.PER_TENSOR)
        with pytest.raises(IllegalConfig) as exc:
            validate(cfg, "c", in_features=128)
        assert "BLOCK_K" in str(exc.value) or "32" in str(exc.value)


class TestNoFallback:
    def test_validate_has_no_per_tensor_default_branch(self):
        src = inspect.getsource(validate)
        # Must not silently rewrite a_gran to PER_TENSOR.
        assert "a_gran = AGran.PER_TENSOR" not in src
        assert "a_gran=AGran.PER_TENSOR" not in src

    def test_quant_linear_node_no_granularity_fallback(self):
        src = inspect.getsource(QuantLinearNode.get_pre_nodes)
        assert "PER_TENSOR" not in src or "a_gran" in src
        assert "default" not in src.lower()


class TestPrePostDerivation:
    def test_static_w8_pre_post(self):
        cfg = QuantLinearConfig(8, 8, 128, False, AGran.PER_TENSOR)
        node = QuantLinearNode(
            _linear_node(), cfg, input_scale=0.05, output_scale=0.05
        )
        pre = node.get_pre_nodes()
        post = node.get_post_nodes()
        assert len(pre) == 1 and isinstance(pre[0], QuantizeNode)
        assert pre[0].target_dtype == "int8"
        assert len(post) == 1 and isinstance(post[0], DequantizeNode)
        assert node.matmul_output_dtype == "int8"

    def test_static_w16_pre_post(self):
        cfg = QuantLinearConfig(16, 16, 128, False, AGran.PER_TENSOR)
        node = QuantLinearNode(_linear_node(), cfg)
        pre = node.get_pre_nodes()
        post = node.get_post_nodes()
        assert isinstance(pre[0], QuantizeNode)
        assert pre[0].target_dtype == "int16"
        assert isinstance(post[0], DequantizeNode)

    def test_dynamic_pre_only_float_output(self):
        cfg = QuantLinearConfig(8, 8, 128, False, AGran.PER_TENSOR, dynamic_act=True)
        node = QuantLinearNode(_linear_node(), cfg)
        pre = node.get_pre_nodes()
        post = node.get_post_nodes()
        assert len(pre) == 1 and isinstance(pre[0], DynamicQuantizeInputNode)
        assert post == []
        assert node.dtype == "float32"
        assert node.matmul_output_dtype == "float32"

    def test_codegen_delegates_not_raises(self):
        """Phase 2: codegen emits kernel calls (no longer NotImplementedError)."""
        from src.pytorch_to_c.ir.node import IRNode as _IRNode

        class _P:
            def _get_input_buffer(self, node, idx=0):
                return "x"
            def _get_buffer_name(self, node):
                return "y"
            def _sanitize_name(self, name):
                return name
            def _w(self, name):
                return name

        cfg = QuantLinearConfig(8, 8, 128, False, AGran.PER_TENSOR)
        orig = _linear_node()
        orig.metadata["weight_name"] = "fc1_weight"
        orig.metadata["bias_name"] = "fc1_bias"
        orig.metadata["in_features"] = 128
        orig.metadata["out_features"] = 16
        node = QuantLinearNode(orig, cfg, input_scale=0.05, weight_scale=0.02, output_scale=0.05)
        lines = node.generate_c_code(_P())
        assert lines[0].startswith("dense_int8(")
        tlines = node.generate_triton_code(_P())
        assert tlines[0].startswith("ops_q.dense_int8(")


class TestDocIdempotency:
    def test_gen_legal_config_doc_idempotent(self):
        a = render_doc_tool()
        b = render_doc_tool()
        assert a == b
