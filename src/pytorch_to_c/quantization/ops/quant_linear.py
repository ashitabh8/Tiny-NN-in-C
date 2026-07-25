"""
Quantized Linear Nodes - thin QuantLinearNode subclasses (Phase 2).

Affine-linear variants pin a fixed QuantLinearConfig and inherit pre/post +
kernel delegation from QuantLinearNode. Weight-only / palette remain carve-outs.
"""

from typing import List

from ...ir.node import IRNode
from ...ir.quant_node import QuantIRNode
from ..quant_config import (
    dynamic_per_tensor_config,
    static_per_channel_config,
    static_per_group_config,
    static_per_tensor_config,
)
from .quant_linear_general import QuantLinearNode


class StaticQuantLinearNode(QuantLinearNode):
    """Static per-tensor quantized linear (W8A8 / W16A16)."""

    def __init__(
        self,
        original_node: IRNode,
        dtype: str,
        input_scale: float,
        weight_scale: float,
        output_scale: float,
        input_offset: int = 0,
        weight_offset: int = 0,
        output_offset: int = 0,
    ):
        in_features = original_node.metadata.get("in_features", 1)
        config = static_per_tensor_config(
            dtype,
            input_offset=input_offset,
            weight_offset=weight_offset,
            in_features=in_features,
        )
        super().__init__(
            original_node=original_node,
            config=config,
            input_scale=input_scale,
            output_scale=output_scale,
            weight_scale=weight_scale,
            input_offset=input_offset,
            weight_offset=weight_offset,
            output_offset=output_offset,
            resolved_group_size=in_features,
        )

    def __repr__(self) -> str:
        return (
            f"StaticQuantLinearNode(name='{self.name}', "
            f"in={self.metadata.get('in_features')}, "
            f"out={self.metadata.get('out_features')}, "
            f"dtype='{self.dtype}', "
            f"input_scale={self.input_scale}, "
            f"weight_scale={self.weight_scale}, "
            f"output_scale={self.output_scale}, "
            f"zp_in={self.input_offset}, zp_w={self.weight_offset}, zp_out={self.output_offset})"
        )


class StaticPerChannelQuantLinearNode(QuantLinearNode):
    """Static quantized linear with per-output-column weight scales."""

    def __init__(
        self,
        original_node: IRNode,
        dtype: str,
        input_scale: float,
        output_scale: float,
        input_offset: int = 0,
        weight_offset: int = 0,
        output_offset: int = 0,
    ):
        in_features = original_node.metadata.get("in_features", 1)
        config = static_per_channel_config(
            dtype,
            input_offset=input_offset,
            weight_offset=weight_offset,
            in_features=in_features,
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
            resolved_group_size=in_features,
        )

    def __repr__(self) -> str:
        return (
            f"StaticPerChannelQuantLinearNode(name='{self.name}', "
            f"in={self.metadata.get('in_features')}, "
            f"out={self.metadata.get('out_features')}, "
            f"dtype='{self.dtype}', "
            f"input_scale={self.input_scale}, "
            f"output_scale={self.output_scale}, "
            f"zp_in={self.input_offset}, zp_w={self.weight_offset}, zp_out={self.output_offset})"
        )


class StaticPerGroupQuantLinearNode(QuantLinearNode):
    """Static quantized linear with per-group weight scales along input axis."""

    def __init__(
        self,
        original_node: IRNode,
        dtype: str,
        input_scale: float,
        output_scale: float,
        input_offset: int = 0,
        weight_offset: int = 0,
        output_offset: int = 0,
    ):
        in_features = original_node.metadata.get("in_features", 1)
        # Placeholder group_size; real value set in metadata during quantize_weights.
        config = static_per_group_config(
            dtype,
            input_offset=input_offset,
            weight_offset=weight_offset,
            in_features=in_features,
            group_size=32,
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
            resolved_group_size=32,
        )


class DynamicQuantLinearNode(QuantLinearNode):
    """
    Dynamic quantized linear: runtime input scale, float32 output.
    """

    def __init__(
        self,
        original_node: IRNode,
        dtype: str,
        weight_scale: float,
        offset: int = 0,
    ):
        in_features = original_node.metadata.get("in_features", 1)
        config = dynamic_per_tensor_config(
            dtype,
            weight_offset=offset,
            in_features=in_features,
        )
        super().__init__(
            original_node=original_node,
            config=config,
            input_scale=1.0,
            output_scale=1.0,
            weight_scale=weight_scale,
            input_offset=0,
            weight_offset=offset,
            output_offset=0,
            resolved_group_size=in_features,
        )
        self.computation_dtype = dtype

    def __repr__(self) -> str:
        return (
            f"DynamicQuantLinearNode(name='{self.name}', "
            f"in={self.metadata.get('in_features')}, "
            f"out={self.metadata.get('out_features')}, "
            f"computation_dtype='{self.computation_dtype}', "
            f"weight_scale={self.weight_scale})"
        )


# ---------------------------------------------------------------------------
# Carve-outs (not QuantLinearConfig) — unchanged
# ---------------------------------------------------------------------------

class _WeightOnlyLinearNode(QuantIRNode):
    """Base for weight-only compressed linear (float activations in/out)."""

    def get_pre_nodes(self) -> List[IRNode]:
        return []

    def get_post_nodes(self) -> List[IRNode]:
        return []

    def get_c_dtype(self) -> str:
        return "float"

    def validate_input_dtypes(self) -> bool:
        for inp in self.inputs:
            if inp.dtype != "float32":
                raise TypeError(
                    f"Weight-only node '{self.name}' expects float32 input"
                )
        return True


class Int8WeightOnlyLinearNode(_WeightOnlyLinearNode):
    """Int8 weight-only linear: float in, float out, per-output-column scales."""

    def __init__(self, original_node: IRNode):
        super().__init__(
            original_node=original_node,
            dtype="int8",
            scale=1.0,
            offset=0,
            quant_strategy="int8_weight_only",
        )
        self.dtype = "float32"

    def generate_c_code(self, c_printer) -> List[str]:
        scales_param = self.metadata.get("per_channel_weight_scales_param")
        if not scales_param:
            raise ValueError(
                f"Int8WeightOnlyLinearNode '{self.name}': missing "
                f"metadata['per_channel_weight_scales_param']"
            )
        scales_c = c_printer._sanitize_name(scales_param)
        input_buffer = c_printer._get_input_buffer(self, 0)
        output_buffer = c_printer._get_buffer_name(self)
        weight_name = c_printer._sanitize_name(self.metadata["weight_name"])
        bias_name = (
            c_printer._sanitize_name(self.metadata["bias_name"])
            if self.metadata.get("bias_name")
            else "NULL"
        )
        in_features = self.metadata["in_features"]
        out_features = self.metadata["out_features"]
        rows = 1
        if self.inputs and self.inputs[0].output_shape is not None:
            shape = list(self.inputs[0].output_shape)
            if len(shape) > 0 and shape[0] == 1:
                shape = shape[1:]
            total = 1
            for dim in shape:
                total *= dim
            if in_features > 0 and total % in_features == 0:
                rows = total // in_features

        call = (
            f"dense_float_input_int8_weight_per_channel("
            f"{input_buffer}, {in_features}, "
            f"{weight_name}, {bias_name}, {out_features}, "
            f"{scales_c}, {output_buffer});"
        )
        if rows == 1:
            return [call]
        lines = [f"for (int r = 0; r < {rows}; ++r) {{"]
        lines.append(
            f"    dense_float_input_int8_weight_per_channel("
            f"{input_buffer} + r * {in_features}, {in_features}, "
            f"{weight_name}, {bias_name}, {out_features}, "
            f"{scales_c}, {output_buffer} + r * {out_features});"
        )
        lines.append("}")
        return lines

    def generate_triton_code(self, printer) -> List[str]:
        scales_param = self.metadata.get("per_channel_weight_scales_param")
        scales_c = printer._w(scales_param)
        input_buffer = printer._get_input_buffer(self, 0)
        output_buffer = printer._get_buffer_name(self)
        weight_name = printer._w(self.metadata["weight_name"])
        bias_name = (
            printer._w(self.metadata["bias_name"])
            if self.metadata.get("bias_name")
            else "None"
        )
        in_features = self.metadata["in_features"]
        out_features = self.metadata["out_features"]
        return [
            f"ops_q.dense_float_input_int8_weight_per_channel({input_buffer}, "
            f"{in_features}, {weight_name}, {bias_name}, {out_features}, "
            f"{scales_c}, {output_buffer})"
        ]


class PaletteWeightLinearNode(_WeightOnlyLinearNode):
    """Palettized weight-only linear: float in, float out."""

    def __init__(self, original_node: IRNode, num_centroids: int):
        super().__init__(
            original_node=original_node,
            dtype="palette",
            scale=1.0,
            offset=0,
            quant_strategy="palettization",
        )
        self.num_centroids = num_centroids
        self.dtype = "float32"

    def generate_c_code(self, c_printer) -> List[str]:
        cb = c_printer._sanitize_name(self.metadata["codebook_param"])
        idx = c_printer._sanitize_name(self.metadata["indices_param"])
        count = self.metadata.get("weight_count")
        k = self.metadata.get("num_centroids", self.num_centroids)
        input_buffer = c_printer._get_input_buffer(self, 0)
        output_buffer = c_printer._get_buffer_name(self)
        bias_name = (
            c_printer._sanitize_name(self.metadata["bias_name"])
            if self.metadata.get("bias_name")
            else "NULL"
        )
        in_features = self.metadata["in_features"]
        out_features = self.metadata["out_features"]
        return [
            f"dense_float_palettized("
            f"{input_buffer}, {in_features}, {idx}, {count}, {cb}, {k}, "
            f"{bias_name}, {out_features}, {output_buffer});"
        ]

    def generate_triton_code(self, printer) -> List[str]:
        cb = printer._w(self.metadata["codebook_param"])
        idx = printer._w(self.metadata["indices_param"])
        count = self.metadata.get("weight_count")
        k = self.metadata.get("num_centroids", self.num_centroids)
        input_buffer = printer._get_input_buffer(self, 0)
        output_buffer = printer._get_buffer_name(self)
        bias_name = (
            printer._w(self.metadata["bias_name"])
            if self.metadata.get("bias_name")
            else "None"
        )
        in_features = self.metadata["in_features"]
        out_features = self.metadata["out_features"]
        return [
            f"ops_q.dense_float_palettized({input_buffer}, {in_features}, "
            f"{idx}, {count}, {cb}, {k}, {bias_name}, {out_features}, {output_buffer})"
        ]
