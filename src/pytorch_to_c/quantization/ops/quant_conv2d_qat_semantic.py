"""
QAT-semantics quantized Conv2D nodes.

These nodes are additive alternatives to StaticQuantConv2dNode and are designed
to mirror QAT training semantics for DS-DW blocks:

- Depthwise: quantized activation input + int8 weights -> float output
- Pointwise: float input + int8 weights -> float output

Both nodes intentionally emit float outputs (no output requant/dequant node).
"""

from typing import List
from ...ir.quant_node import QuantIRNode
from ...ir.node import IRNode


class StaticQuantDepthwiseConv2dFloatOutNode(QuantIRNode):
    """Depthwise conv: int8 input + per-channel int8 weights -> float output."""

    def __init__(
        self,
        original_node: IRNode,
        quant_dtype: str,
        input_scale: float,
        weight_scale,
        offset: int = 0,
    ):
        # IR dtype is float32 because this op outputs float activations.
        super().__init__(
            original_node=original_node,
            dtype="float32",
            scale=weight_scale,
            offset=offset,
            quant_strategy="static_qat_semantic",
        )
        self.quant_dtype = quant_dtype
        self.input_scale = input_scale
        self.weight_scale = weight_scale
        self.metadata["quant_params"]["quant_dtype"] = quant_dtype

    def get_pre_nodes(self) -> List[IRNode]:
        from .quant_utils import QuantizeNode

        if self.quant_dtype != "int8":
            raise ValueError(
                f"{self.name}: only int8 is supported for QAT semantic depthwise conv"
            )
        quant_output_shape = None
        if "input_shape" in self.metadata:
            quant_output_shape = self.metadata["input_shape"]
        pre_node = QuantizeNode(
            name=f"{self.name}_input_q",
            target_dtype=self.quant_dtype,
            scale=self.input_scale,
            offset=self.offset,
            output_shape=quant_output_shape,
        )
        return [pre_node]

    def get_post_nodes(self) -> List[IRNode]:
        return []

    def get_c_dtype(self) -> str:
        return "float"

    def validate_input_dtypes(self) -> bool:
        for inp in self.inputs:
            if inp.dtype != "int8":
                raise TypeError(
                    f"{self.name}: expected int8 input, got {inp.dtype} from {inp.name}"
                )
        return True

    def generate_c_code(self, c_printer) -> List[str]:
        if self.quant_dtype != "int8":
            raise ValueError(
                f"{self.name}: only int8 is supported for QAT semantic depthwise conv"
            )
        input_buffer = c_printer._get_input_buffer(self, 0)
        output_buffer = c_printer._get_buffer_name(self)
        weight_name = c_printer._sanitize_name(self.metadata["weight_name"])
        weight_scale_name = c_printer._sanitize_name(f"{self.metadata['weight_name']}_scale")

        bias_name = (
            c_printer._sanitize_name(self.metadata["bias_name"])
            if self.metadata["bias_name"]
            else "NULL"
        )
        kernel_size = self.metadata["kernel_size"]
        stride = self.metadata["stride"]
        padding = self.metadata["padding"]
        in_channels = self.metadata["in_channels"]

        if isinstance(kernel_size, (tuple, list)):
            k_h = kernel_size[0]
            k_w = kernel_size[1]
        else:
            k_h = kernel_size
            k_w = kernel_size
        if isinstance(stride, (tuple, list)):
            s_h = stride[0]
            s_w = stride[1]
        else:
            s_h = stride
            s_w = stride
        if isinstance(padding, (tuple, list)):
            p_h = padding[0]
            p_w = padding[1]
        else:
            p_h = padding
            p_w = padding

        if self.inputs and self.inputs[0].output_shape:
            input_shape = self.inputs[0].output_shape
            in_h = input_shape[2]
            in_w = input_shape[3]
        else:
            in_h = 32
            in_w = 32

        return [
            f"depthwise_conv2d_nhwc_int8_to_float("
            f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
            f"{weight_name}, {k_h}, {k_w}, {bias_name}, "
            f"{s_h}, {s_w}, {p_h}, {p_w}, "
            f"{self.input_scale}f, {weight_scale_name}, {output_buffer});"
        ]


class StaticQuantPointwiseConv2dFloatInFloatOutNode(QuantIRNode):
    """Pointwise conv: float input + per-channel int8 weights -> float output."""

    def __init__(
        self,
        original_node: IRNode,
        quant_dtype: str,
        weight_scale,
        offset: int = 0,
    ):
        super().__init__(
            original_node=original_node,
            dtype="float32",
            scale=weight_scale,
            offset=offset,
            quant_strategy="static_qat_semantic",
        )
        self.quant_dtype = quant_dtype
        self.weight_scale = weight_scale
        self.metadata["quant_params"]["quant_dtype"] = quant_dtype

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
                    f"{self.name}: expected float32 input, got {inp.dtype} from {inp.name}"
                )
        return True

    def generate_c_code(self, c_printer) -> List[str]:
        if self.quant_dtype != "int8":
            raise ValueError(
                f"{self.name}: only int8 is supported for QAT semantic pointwise conv"
            )
        input_buffer = c_printer._get_input_buffer(self, 0)
        output_buffer = c_printer._get_buffer_name(self)
        weight_name = c_printer._sanitize_name(self.metadata["weight_name"])
        weight_scale_name = c_printer._sanitize_name(f"{self.metadata['weight_name']}_scale")

        bias_name = (
            c_printer._sanitize_name(self.metadata["bias_name"])
            if self.metadata["bias_name"]
            else "NULL"
        )
        kernel_size = self.metadata["kernel_size"]
        stride = self.metadata["stride"]
        padding = self.metadata["padding"]
        in_channels = self.metadata["in_channels"]
        out_channels = self.metadata["out_channels"]

        if isinstance(kernel_size, (tuple, list)):
            k_h = kernel_size[0]
            k_w = kernel_size[1]
        else:
            k_h = kernel_size
            k_w = kernel_size
        if isinstance(stride, (tuple, list)):
            s_h = stride[0]
            s_w = stride[1]
        else:
            s_h = stride
            s_w = stride
        if isinstance(padding, (tuple, list)):
            p_h = padding[0]
            p_w = padding[1]
        else:
            p_h = padding
            p_w = padding

        if self.inputs and self.inputs[0].output_shape:
            input_shape = self.inputs[0].output_shape
            in_h = input_shape[2]
            in_w = input_shape[3]
        else:
            in_h = 32
            in_w = 32

        return [
            f"conv2d_nhwc_float_input_int8_weight_per_channel("
            f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
            f"{weight_name}, {k_h}, {k_w}, {out_channels}, {bias_name}, "
            f"{s_h}, {s_w}, {p_h}, {p_w}, "
            f"{weight_scale_name}, {output_buffer});"
        ]
