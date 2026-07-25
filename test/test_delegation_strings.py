"""Phase 2 primary gate: byte-identical kernel call strings.

Frozen against pre-Phase-2 specialized nodes (captured 2026-07-10).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pytorch_to_c.ir.node import IRNode
from src.pytorch_to_c.quantization.ops.quant_int4_linear import (
    DynamicInt4PerGroupQuantLinearNode,
    StaticInt4PerGroupQuantLinearNode,
)
from src.pytorch_to_c.quantization.ops.quant_linear import (
    DynamicQuantLinearNode,
    StaticPerChannelQuantLinearNode,
    StaticPerGroupQuantLinearNode,
    StaticQuantLinearNode,
)


class MockPrinter:
    def _get_input_buffer(self, node, idx=0):
        return "buf_x"

    def _get_buffer_name(self, node):
        return "buf_y"

    def _sanitize_name(self, name):
        return name.replace(".", "_")

    def _w(self, name):
        return f"W['{name}']"


GOLDEN_C = {
    "w8a8_static_per_tensor": [
        "dense_int8(buf_x, 32, fc1_weight, fc1_bias, 16, 0.05f, 0.02f, 0.05f, 0, 0, 0, buf_y);"
    ],
    "w16a16_static_per_tensor": [
        "dense_int16(buf_x, 32, fc1_weight, fc1_bias, 16, 0.005f, 0.002f, 0.005f, 0, 0, 0, buf_y);"
    ],
    "w8a8_static_per_channel": [
        "dense_int8_per_channel(buf_x, 32, fc1_weight, fc1_bias, 16, 0.05f, "
        "fc1_weight_per_channel_scales, 0.05f, 0, 0, 0, buf_y);"
    ],
    "w16a16_static_per_channel": [
        "dense_int16_per_channel(buf_x, 32, fc1_weight, fc1_bias, 16, 0.005f, "
        "fc1_weight_per_channel_scales, 0.005f, 0, 0, 0, buf_y);"
    ],
    "w8a8_static_per_group": [
        "dense_int8_per_group(buf_x, 32, fc1_weight, fc1_bias, 16, 32, 0.05f, "
        "fc1_weight_per_group_scales, 0.05f, 0, 0, 0, buf_y);"
    ],
    "w16a16_static_per_group": [
        "dense_int16_per_group(buf_x, 32, fc1_weight, fc1_bias, 16, 32, 0.005f, "
        "fc1_weight_per_group_scales, 0.005f, 0, 0, 0, buf_y);"
    ],
    "w8a8_dynamic": [
        "dense_int8_to_float(buf_x, 32, fc1_weight, fc1_bias, 16, "
        "scale_fc1_input_dynq, 0.02f, buf_y);"
    ],
    "w16a16_dynamic": [
        "dense_int16_to_float(buf_x, 32, fc1_weight, fc1_bias, 16, "
        "scale_fc1_input_dynq, 0.002f, buf_y);"
    ],
    "w4a8_static": [
        "dense_int8_int4w_per_group(buf_x, 32, fc1_weight, 512, fc1_bias, 16, 32, "
        "0.05f, fc1_weight_per_group_scales, 0.05f, 0, 0, 0, buf_y);"
    ],
    "w4a8_dynamic": [
        "dense_int8_int4w_per_group_to_float(buf_x, 32, fc1_weight, 512, fc1_bias, "
        "16, 32, scale_fc1_input_dynq, fc1_weight_per_group_scales, buf_y);"
    ],
}

GOLDEN_TRITON = {
    "w8a8_static_per_tensor": [
        "ops_q.dense_int8(buf_x, 32, W['fc1_weight'], W['fc1_bias'], 16, "
        "0.05, 0.02, 0.05, 0, 0, 0, buf_y)"
    ],
    "w16a16_static_per_tensor": [
        "ops_q.dense_int16(buf_x, 32, W['fc1_weight'], W['fc1_bias'], 16, "
        "0.005, 0.002, 0.005, 0, 0, 0, buf_y)"
    ],
    "w8a8_static_per_channel": [
        "ops_q.dense_int8_per_channel(buf_x, 32, W['fc1_weight'], W['fc1_bias'], 16, "
        "0.05, W['fc1_weight_per_channel_scales'], 0.05, 0, 0, 0, buf_y)"
    ],
    "w16a16_static_per_channel": [
        "ops_q.dense_int16_per_channel(buf_x, 32, W['fc1_weight'], W['fc1_bias'], 16, "
        "0.005, W['fc1_weight_per_channel_scales'], 0.005, 0, 0, 0, buf_y)"
    ],
    "w8a8_static_per_group": [
        "ops_q.dense_int8_per_group(buf_x, 32, W['fc1_weight'], W['fc1_bias'], 16, 32, "
        "0.05, W['fc1_weight_per_group_scales'], 0.05, 0, 0, 0, buf_y)"
    ],
    "w16a16_static_per_group": [
        "ops_q.dense_int16_per_group(buf_x, 32, W['fc1_weight'], W['fc1_bias'], 16, 32, "
        "0.005, W['fc1_weight_per_group_scales'], 0.005, 0, 0, 0, buf_y)"
    ],
    "w8a8_dynamic": [
        "ops_q.dense_int8_to_float(buf_x, 32, W['fc1_weight'], W['fc1_bias'], 16, "
        "scale_fc1_input_dynq, 0.02, buf_y)"
    ],
    "w16a16_dynamic": [
        "ops_q.dense_int16_to_float(buf_x, 32, W['fc1_weight'], W['fc1_bias'], 16, "
        "scale_fc1_input_dynq, 0.002, buf_y)"
    ],
    "w4a8_static": [
        "ops_q.dense_int8_int4w_per_group(buf_x, 32, W['fc1_weight'], 512, W['fc1_bias'], "
        "16, 32, 0.05, W['fc1_weight_per_group_scales'], 0.05, 0, 0, 0, buf_y)"
    ],
    "w4a8_dynamic": [
        "ops_q.dense_int8_int4w_per_group_to_float(buf_x, 32, W['fc1_weight'], 512, "
        "W['fc1_bias'], 16, 32, scale_fc1_input_dynq, W['fc1_weight_per_group_scales'], buf_y)"
    ],
}


def _make_linear(name="fc1", in_f=32, out_f=16):
    return IRNode(
        name=name,
        op_type="linear",
        output_shape=(1, out_f),
        dtype="float32",
        metadata={
            "in_features": in_f,
            "out_features": out_f,
            "input_shape": (1, in_f),
            "weight_name": f"{name}_weight",
            "bias_name": f"{name}_bias",
        },
    )


def _make_dynq(name="fc1_input_dynq"):
    return IRNode(
        name=name,
        op_type="dynamic_quantize",
        output_shape=(1, 32),
        dtype="int8",
        metadata={},
    )


def _build_node(cell: str):
    n = _make_linear()
    if cell == "w8a8_static_per_tensor":
        return StaticQuantLinearNode(n, "int8", 0.05, 0.02, 0.05, 0, 0, 0)
    if cell == "w16a16_static_per_tensor":
        return StaticQuantLinearNode(n, "int16", 0.005, 0.002, 0.005, 0, 0, 0)
    if cell == "w8a8_static_per_channel":
        node = StaticPerChannelQuantLinearNode(n, "int8", 0.05, 0.05, 0, 0, 0)
        node.metadata["per_channel_weight_scales_param"] = "fc1_weight_per_channel_scales"
        return node
    if cell == "w16a16_static_per_channel":
        node = StaticPerChannelQuantLinearNode(n, "int16", 0.005, 0.005, 0, 0, 0)
        node.metadata["per_channel_weight_scales_param"] = "fc1_weight_per_channel_scales"
        return node
    if cell == "w8a8_static_per_group":
        node = StaticPerGroupQuantLinearNode(n, "int8", 0.05, 0.05, 0, 0, 0)
        node.metadata["per_group_weight_scales_param"] = "fc1_weight_per_group_scales"
        node.metadata["group_size"] = 32
        return node
    if cell == "w16a16_static_per_group":
        node = StaticPerGroupQuantLinearNode(n, "int16", 0.005, 0.005, 0, 0, 0)
        node.metadata["per_group_weight_scales_param"] = "fc1_weight_per_group_scales"
        node.metadata["group_size"] = 32
        return node
    if cell == "w8a8_dynamic":
        node = DynamicQuantLinearNode(n, "int8", 0.02, 0)
        node.inputs = [_make_dynq()]
        return node
    if cell == "w16a16_dynamic":
        node = DynamicQuantLinearNode(n, "int16", 0.002, 0)
        node.inputs = [_make_dynq()]
        return node
    if cell == "w4a8_static":
        node = StaticInt4PerGroupQuantLinearNode(n, 0.05, 0.05, 0, 0, 0, group_size=32)
        node.metadata["per_group_weight_scales_param"] = "fc1_weight_per_group_scales"
        node.metadata["group_size"] = 32
        node.metadata["packed_weight_count"] = 512
        return node
    if cell == "w4a8_dynamic":
        node = DynamicInt4PerGroupQuantLinearNode(n, group_size=32, weight_offset=0)
        node.metadata["per_group_weight_scales_param"] = "fc1_weight_per_group_scales"
        node.metadata["group_size"] = 32
        node.metadata["packed_weight_count"] = 512
        node.inputs = [_make_dynq()]
        return node
    raise KeyError(cell)


@pytest.mark.parametrize("cell", list(GOLDEN_C.keys()))
def test_delegation_c_string(cell):
    printer = MockPrinter()
    node = _build_node(cell)
    assert node.generate_c_code(printer) == GOLDEN_C[cell]


@pytest.mark.parametrize("cell", list(GOLDEN_TRITON.keys()))
def test_delegation_triton_string(cell):
    printer = MockPrinter()
    node = _build_node(cell)
    assert node.generate_triton_code(printer) == GOLDEN_TRITON[cell]
