"""
Int4 per-group quantized Linear nodes — thin QuantLinearNode subclasses (Phase 2).

Activations are int8 (via QuantizeNode / DynamicQuantizeInputNode).
Weights are packed int4 with per-group scales along the input axis.
"""

from ...ir.node import IRNode
from ..quant_config import dynamic_w4a8_config, static_w4a8_config
from .quant_linear_general import QuantLinearNode


class StaticInt4PerGroupQuantLinearNode(QuantLinearNode):
    """Static W4A8 linear: QuantizeNode -> int8 x int4w matmul -> DequantizeNode."""

    def __init__(
        self,
        original_node: IRNode,
        input_scale: float,
        output_scale: float,
        input_offset: int = 0,
        weight_offset: int = 0,
        output_offset: int = 0,
        group_size: int = 64,
    ):
        config = static_w4a8_config(
            input_offset=input_offset,
            weight_offset=weight_offset,
            group_size=group_size,
        )
        super().__init__(
            original_node=original_node,
            config=config,
            input_scale=input_scale,
            output_scale=output_scale,
            weight_scale=1.0,
            input_offset=input_offset,
            weight_offset=weight_offset,
            output_offset=output_offset,
            resolved_group_size=group_size,
        )
        self.group_size = group_size

    def __repr__(self) -> str:
        return (
            f"StaticInt4PerGroupQuantLinearNode(name='{self.name}', "
            f"in={self.metadata.get('in_features')}, "
            f"out={self.metadata.get('out_features')}, "
            f"group_size={self.metadata.get('group_size')})"
        )


class DynamicInt4PerGroupQuantLinearNode(QuantLinearNode):
    """Dynamic W4A8 linear: DynamicQuantizeInputNode -> int8 x int4w -> float32."""

    def __init__(
        self,
        original_node: IRNode,
        group_size: int = 64,
        weight_offset: int = 0,
    ):
        config = dynamic_w4a8_config(
            weight_offset=weight_offset,
            group_size=group_size,
        )
        super().__init__(
            original_node=original_node,
            config=config,
            input_scale=1.0,
            output_scale=1.0,
            weight_scale=1.0,
            input_offset=0,
            weight_offset=weight_offset,
            output_offset=0,
            resolved_group_size=group_size,
        )
        self.group_size = group_size
        self.computation_dtype = "int8"

    def __repr__(self) -> str:
        return (
            f"DynamicInt4PerGroupQuantLinearNode(name='{self.name}', "
            f"in={self.metadata.get('in_features')}, "
            f"out={self.metadata.get('out_features')}, "
            f"group_size={self.metadata.get('group_size')})"
        )
