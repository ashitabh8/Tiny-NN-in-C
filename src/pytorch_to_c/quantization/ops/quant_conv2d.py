"""
Quantized Conv2D Nodes - Static and Dynamic variants

Two separate classes for clarity:
- StaticQuantConv2dNode: User provides all scales (input, weight, output)
- DynamicQuantConv2dNode: Input scale computed at runtime, weight scale from weights
"""

from typing import List
from ...ir.quant_node import QuantIRNode
from ...ir.node import IRNode


class StaticQuantConv2dNode(QuantIRNode):
    """
    Static Quantized Conv2D operation.
    
    User provides pre-calibrated scales for:
    - Input activation quantization
    - Weight quantization  
    - Output dequantization
    
    Uses QuantizeNode and DequantizeNode for conversions.
    """
    
    def __init__(
        self,
        original_node: IRNode,
        dtype: str,
        input_scale: float,
        weight_scale: float,
        output_scale: float,
        input_offset: int = 0,
        weight_offset: int = 0,
        output_offset: int = 0
    ):
        """
        Initialize static quantized conv2d node.
        
        Args:
            original_node: The float conv2d node being quantized
            dtype: Target data type ('int8' or 'int16')
            input_scale: Scale for input activation quantization
            weight_scale: Scale for weight quantization
            output_scale: Scale for output activation (requant + dequantize)
            input_offset: Zero point for input (QuantizeNode)
            weight_offset: Zero point for weights (compile-time quant)
            output_offset: Zero point for layer output (conv requant + DequantizeNode)
        """
        super().__init__(
            original_node=original_node,
            dtype=dtype,
            scale=weight_scale,
            offset=weight_offset,
            quant_strategy='static'
        )
        
        self.input_scale = input_scale
        self.weight_scale = weight_scale
        self.output_scale = output_scale
        self.input_offset = input_offset
        self.weight_offset = weight_offset
        self.output_offset = output_offset
    
    def get_pre_nodes(self) -> List[IRNode]:
        """
        Insert QuantizeNode before this layer to quantize float input.
        """
        from .quant_utils import QuantizeNode
        
        pre_node = QuantizeNode(
            name=f"{self.name}_input_q",
            target_dtype=self.dtype,
            scale=self.input_scale,
            offset=self.input_offset,
            output_shape=self.metadata.get('input_shape')
        )
        
        return [pre_node]
    
    def get_post_nodes(self) -> List[IRNode]:
        """
        Insert DequantizeNode after this layer to convert output to float32.
        """
        from .quant_utils import DequantizeNode
        
        post_node = DequantizeNode(
            name=f"{self.name}_output_dq",
            source_dtype=self.dtype,
            scale=self.output_scale,
            offset=self.output_offset,
            output_shape=self.output_shape
        )
        
        return [post_node]
    
    def generate_c_code(self, c_printer) -> List[str]:
        """
        Generate C code for static quantized conv2d.

        Dispatches to depthwise_conv2d_nhwc_int8/int16 for depthwise layers
        (groups == in_channels == out_channels) and conv2d_nhwc_int8/int16 otherwise.
        """
        lines = []

        input_buffer = c_printer._get_input_buffer(self, 0)
        output_buffer = c_printer._get_buffer_name(self)
        weight_name = c_printer._sanitize_name(self.metadata['weight_name'])

        bias_name = c_printer._sanitize_name(self.metadata['bias_name']) \
                    if self.metadata.get('bias_name') else 'NULL'

        in_channels = self.metadata['in_channels']
        out_channels = self.metadata['out_channels']
        groups = self.metadata.get('groups', 1)

        k_h, k_w, s_h, s_w, p_h, p_w, in_h, in_w = self._unpack_conv_dims_h1_wrap()

        is_depthwise = (
            groups > 1
            and groups == in_channels
            and out_channels == in_channels
        )

        if is_depthwise:
            if self.dtype == 'int8':
                lines.append(
                    f"depthwise_conv2d_nhwc_int8("
                    f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                    f"{weight_name}, {k_h}, {k_w}, "
                    f"{bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
                    f"{self.input_scale}f, {self.weight_scale}f, {self.output_scale}f, "
                    f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
                    f"{output_buffer});"
                )
            elif self.dtype == 'int16':
                lines.append(
                    f"depthwise_conv2d_nhwc_int16("
                    f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                    f"{weight_name}, {k_h}, {k_w}, "
                    f"{bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
                    f"{self.input_scale}f, {self.weight_scale}f, {self.output_scale}f, "
                    f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
                    f"{output_buffer});"
                )
            else:
                raise ValueError(f"Unsupported dtype: {self.dtype}")
        else:
            if self.dtype == 'int8':
                lines.append(
                    f"conv2d_nhwc_int8("
                    f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                    f"{weight_name}, {k_h}, {k_w}, {out_channels}, "
                    f"{bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
                    f"{self.input_scale}f, {self.weight_scale}f, {self.output_scale}f, "
                    f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
                    f"{output_buffer});"
                )
            elif self.dtype == 'int16':
                lines.append(
                    f"conv2d_nhwc_int16("
                    f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                    f"{weight_name}, {k_h}, {k_w}, {out_channels}, "
                    f"{bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
                    f"{self.input_scale}f, {self.weight_scale}f, {self.output_scale}f, "
                    f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
                    f"{output_buffer});"
                )
            else:
                raise ValueError(f"Unsupported dtype: {self.dtype}")

        return lines

    def generate_triton_code(self, printer) -> List[str]:
        input_buffer = printer._get_input_buffer(self, 0)
        output_buffer = printer._get_buffer_name(self)
        weight_name = printer._w(self.metadata["weight_name"])
        bias_name = printer._w(self.metadata["bias_name"]) if self.metadata.get("bias_name") else "None"
        in_channels = self.metadata["in_channels"]
        out_channels = self.metadata["out_channels"]
        groups = self.metadata.get("groups", 1)
        k_h, k_w, s_h, s_w, p_h, p_w, in_h, in_w = self._unpack_conv_dims_h1_wrap()
        is_depthwise = groups > 1 and groups == in_channels and out_channels == in_channels
        fn_base = "depthwise_conv2d_nhwc" if is_depthwise else "conv2d_nhwc"
        fn = f"{fn_base}_int8" if self.dtype == "int8" else f"{fn_base}_int16"
        if is_depthwise:
            return [
                f"ops_q.{fn}({input_buffer}, {in_h}, {in_w}, {in_channels}, {weight_name}, "
                f"{k_h}, {k_w}, {bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
                f"{self.input_scale}, {self.weight_scale}, {self.output_scale}, "
                f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, {output_buffer})"
            ]
        return [
            f"ops_q.{fn}({input_buffer}, {in_h}, {in_w}, {in_channels}, {weight_name}, "
            f"{k_h}, {k_w}, {out_channels}, {bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
            f"{self.input_scale}, {self.weight_scale}, {self.output_scale}, "
            f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, {output_buffer})"
        ]
    
    def _get_input_spatial_dims(self) -> tuple:
        """NCHW (H, W) for Conv2d, or (1, L) for Conv1d via H=1 wrap."""
        if not self.inputs:
            raise ValueError(
                f"StaticQuantConv2dNode '{self.name}': no input node; graph is invalid."
            )
        input_node = self.inputs[0]
        if not input_node.output_shape:
            raise ValueError(
                f"StaticQuantConv2dNode '{self.name}': input '{input_node.name}' has no "
                f"output_shape. Compile with example_input so dimensions are known."
            )
        input_shape = input_node.output_shape
        if len(input_shape) == 4:
            return int(input_shape[2]), int(input_shape[3])
        if len(input_shape) == 3:
            # Conv1d input [B, C, L] -> H=1, W=L wrap
            return 1, int(input_shape[2])
        raise ValueError(
            f"StaticQuantConv2dNode '{self.name}': expected 3D or 4D input shape "
            f"from '{input_node.name}', got {input_shape!r}."
        )

    def _unpack_conv_dims_h1_wrap(self) -> tuple:
        """Return (k_h, k_w, s_h, s_w, p_h, p_w, in_h, in_w) handling Conv1d via H=1.

        Conv1d metadata stores kernel_size/stride/padding as scalars; Conv2d as tuples.
        """
        kernel_size = self.metadata['kernel_size']
        stride = self.metadata['stride']
        padding = self.metadata['padding']
        is_1d = not isinstance(kernel_size, (tuple, list))
        if is_1d:
            k_h, k_w = 1, int(kernel_size)
            s_h, s_w = 1, int(stride)
            p_h, p_w = 0, int(padding)
        else:
            k_h, k_w = int(kernel_size[0]), int(kernel_size[1])
            s_h, s_w = int(stride[0]), int(stride[1])
            p_h, p_w = int(padding[0]), int(padding[1])
        in_h, in_w = self._get_input_spatial_dims()
        return k_h, k_w, s_h, s_w, p_h, p_w, in_h, in_w
    
    def __repr__(self) -> str:
        return (f"StaticQuantConv2dNode(name='{self.name}', "
                f"in_ch={self.metadata.get('in_channels')}, "
                f"out_ch={self.metadata.get('out_channels')}, "
                f"dtype='{self.dtype}', "
                f"input_scale={self.input_scale}, "
                f"weight_scale={self.weight_scale}, "
                f"output_scale={self.output_scale}, "
                f"zp_in={self.input_offset}, zp_w={self.weight_offset}, zp_out={self.output_offset})")


class StaticPerChannelQuantConv2dNode(StaticQuantConv2dNode):
    """
    Static quantized conv2d with per-output-channel weight scales (C ``*_nhwc_*_per_channel``).

    For depthwise (groups == in_channels == out_channels), uses
    ``depthwise_conv2d_nhwc_*_per_channel``; otherwise ``conv2d_nhwc_*_per_channel``.
    """

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
        super().__init__(
            original_node=original_node,
            dtype=dtype,
            input_scale=input_scale,
            weight_scale=1.0,
            output_scale=output_scale,
            input_offset=input_offset,
            weight_offset=weight_offset,
            output_offset=output_offset,
        )

    def generate_c_code(self, c_printer) -> List[str]:
        scales_param = self.metadata.get('per_channel_weight_scales_param')
        if not scales_param:
            raise ValueError(
                f"StaticPerChannelQuantConv2dNode '{self.name}': missing "
                f"metadata['per_channel_weight_scales_param']. Run QuantizationTransform."
            )
        scales_c = c_printer._sanitize_name(scales_param)

        lines = []
        input_buffer = c_printer._get_input_buffer(self, 0)
        output_buffer = c_printer._get_buffer_name(self)
        weight_name = c_printer._sanitize_name(self.metadata['weight_name'])
        bias_name = c_printer._sanitize_name(self.metadata['bias_name']) \
                    if self.metadata.get('bias_name') else 'NULL'
        in_channels = self.metadata['in_channels']
        out_channels = self.metadata['out_channels']
        groups = self.metadata.get('groups', 1)

        k_h, k_w, s_h, s_w, p_h, p_w, in_h, in_w = self._unpack_conv_dims_h1_wrap()

        is_depthwise = (
            groups > 1
            and groups == in_channels
            and out_channels == in_channels
        )

        if is_depthwise:
            if self.dtype == 'int8':
                lines.append(
                    f"depthwise_conv2d_nhwc_int8_per_channel("
                    f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                    f"{weight_name}, {k_h}, {k_w}, "
                    f"{bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
                    f"{self.input_scale}f, {scales_c}, {self.output_scale}f, "
                    f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
                    f"{output_buffer});"
                )
            elif self.dtype == 'int16':
                lines.append(
                    f"depthwise_conv2d_nhwc_int16_per_channel("
                    f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                    f"{weight_name}, {k_h}, {k_w}, "
                    f"{bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
                    f"{self.input_scale}f, {scales_c}, {self.output_scale}f, "
                    f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
                    f"{output_buffer});"
                )
            else:
                raise ValueError(f"Unsupported dtype: {self.dtype}")
        else:
            if self.dtype == 'int8':
                lines.append(
                    f"conv2d_nhwc_int8_per_channel("
                    f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                    f"{weight_name}, {k_h}, {k_w}, {out_channels}, "
                    f"{bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
                    f"{self.input_scale}f, {scales_c}, {self.output_scale}f, "
                    f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
                    f"{output_buffer});"
                )
            elif self.dtype == 'int16':
                lines.append(
                    f"conv2d_nhwc_int16_per_channel("
                    f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                    f"{weight_name}, {k_h}, {k_w}, {out_channels}, "
                    f"{bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
                    f"{self.input_scale}f, {scales_c}, {self.output_scale}f, "
                    f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
                    f"{output_buffer});"
                )
            else:
                raise ValueError(f"Unsupported dtype: {self.dtype}")
        return lines

    def generate_triton_code(self, printer) -> List[str]:
        scales_param = self.metadata.get("per_channel_weight_scales_param")
        if not scales_param:
            raise ValueError(
                f"StaticPerChannelQuantConv2dNode '{self.name}': missing per_channel scales"
            )
        scales_c = printer._w(scales_param)
        input_buffer = printer._get_input_buffer(self, 0)
        output_buffer = printer._get_buffer_name(self)
        weight_name = printer._w(self.metadata["weight_name"])
        bias_name = printer._w(self.metadata["bias_name"]) if self.metadata.get("bias_name") else "None"
        in_channels = self.metadata["in_channels"]
        out_channels = self.metadata["out_channels"]
        groups = self.metadata.get("groups", 1)
        k_h, k_w, s_h, s_w, p_h, p_w, in_h, in_w = self._unpack_conv_dims_h1_wrap()
        is_depthwise = groups > 1 and groups == in_channels and out_channels == in_channels
        fn_base = "depthwise_conv2d_nhwc" if is_depthwise else "conv2d_nhwc"
        fn = f"{fn_base}_int8_per_channel" if self.dtype == "int8" else f"{fn_base}_int16_per_channel"
        if is_depthwise:
            return [
                f"ops_q.{fn}({input_buffer}, {in_h}, {in_w}, {in_channels}, {weight_name}, "
                f"{k_h}, {k_w}, {bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
                f"{self.input_scale}, {scales_c}, {self.output_scale}, "
                f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, {output_buffer})"
            ]
        return [
            f"ops_q.{fn}({input_buffer}, {in_h}, {in_w}, {in_channels}, {weight_name}, "
            f"{k_h}, {k_w}, {out_channels}, {bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
            f"{self.input_scale}, {scales_c}, {self.output_scale}, "
            f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, {output_buffer})"
        ]

    def __repr__(self) -> str:
        return (f"StaticPerChannelQuantConv2dNode(name='{self.name}', "
                f"in_ch={self.metadata.get('in_channels')}, "
                f"out_ch={self.metadata.get('out_channels')}, "
                f"dtype='{self.dtype}', "
                f"input_scale={self.input_scale}, "
                f"output_scale={self.output_scale}, "
                f"zp_in={self.input_offset}, zp_w={self.weight_offset}, zp_out={self.output_offset})")


class DynamicQuantConv2dNode(QuantIRNode):
    """
    Dynamic Quantized Conv2D operation.
    
    - Input scale: Computed at runtime from input values
    - Weight scale: Computed from weights at compile time
    - Output: float32 directly (uses float-output C kernel, no requantize step)
    
    Uses DynamicQuantizeInputNode for input (computes scale at runtime).
    No DequantizeNode needed — the float-output kernel avoids the
    unnecessary requantize->dequantize round-trip.
    """
    
    def __init__(
        self,
        original_node: IRNode,
        dtype: str,
        weight_scale: float,
        offset: int = 0
    ):
        super().__init__(
            original_node=original_node,
            dtype=dtype,
            scale=weight_scale,
            offset=offset,
            quant_strategy='dynamic'
        )
        
        self.weight_scale = weight_scale
        self.computation_dtype = dtype
        self.dtype = 'float32'
    
    def get_pre_nodes(self) -> List[IRNode]:
        """Insert DynamicQuantizeInputNode before this layer."""
        from .quant_utils import DynamicQuantizeInputNode
        
        pre_node = DynamicQuantizeInputNode(
            name=f"{self.name}_input_dynq",
            target_dtype=self.computation_dtype,
            output_shape=self.metadata.get('input_shape')
        )
        
        return [pre_node]
    
    def get_post_nodes(self) -> List[IRNode]:
        """No post-processing: float-output kernel writes float32 directly."""
        return []
    
    def get_c_dtype(self) -> str:
        return 'float'
    
    def validate_input_dtypes(self) -> bool:
        for inp in self.inputs:
            if inp.dtype not in ['int8', 'int16']:
                raise TypeError(
                    f"DynamicQuantConv2dNode '{self.name}' expects quantized input, "
                    f"got '{inp.dtype}' from '{inp.name}'"
                )
        return True
    
    def generate_c_code(self, c_printer) -> List[str]:
        """Generate C code using float-output kernel (no requantization)."""
        lines = []
        
        input_buffer = c_printer._get_input_buffer(self, 0)
        output_buffer = c_printer._get_buffer_name(self)
        weight_name = c_printer._sanitize_name(self.metadata['weight_name'])
        
        bias_name = c_printer._sanitize_name(self.metadata['bias_name']) \
                    if self.metadata.get('bias_name') else 'NULL'
        
        in_channels = self.metadata['in_channels']
        out_channels = self.metadata['out_channels']
        groups = self.metadata.get('groups', 1)

        input_scale_var = self._get_input_scale_variable(c_printer)

        k_h, k_w, s_h, s_w, p_h, p_w, in_h, in_w = self._unpack_conv_dims_h1_wrap()
        
        is_depthwise = (groups > 1
                        and groups == in_channels
                        and out_channels == in_channels)
        
        if is_depthwise:
            if self.computation_dtype == 'int8':
                lines.append(
                    f"depthwise_conv2d_nhwc_int8_to_float("
                    f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                    f"{weight_name}, {k_h}, {k_w}, "
                    f"{bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
                    f"{input_scale_var}, {self.weight_scale}f, "
                    f"{output_buffer});"
                )
            elif self.computation_dtype == 'int16':
                lines.append(
                    f"depthwise_conv2d_nhwc_int16_to_float("
                    f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                    f"{weight_name}, {k_h}, {k_w}, "
                    f"{bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
                    f"{input_scale_var}, {self.weight_scale}f, "
                    f"{output_buffer});"
                )
            else:
                raise ValueError(f"Unsupported computation dtype: {self.computation_dtype}")
        else:
            if self.computation_dtype == 'int8':
                lines.append(
                    f"conv2d_nhwc_int8_to_float("
                    f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                    f"{weight_name}, {k_h}, {k_w}, {out_channels}, "
                    f"{bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
                    f"{input_scale_var}, {self.weight_scale}f, "
                    f"{output_buffer});"
                )
            elif self.computation_dtype == 'int16':
                lines.append(
                    f"conv2d_nhwc_int16_to_float("
                    f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                    f"{weight_name}, {k_h}, {k_w}, {out_channels}, "
                    f"{bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
                    f"{input_scale_var}, {self.weight_scale}f, "
                    f"{output_buffer});"
                )
            else:
                raise ValueError(f"Unsupported computation dtype: {self.computation_dtype}")
        
        return lines

    def generate_triton_code(self, printer) -> List[str]:
        input_buffer = printer._get_input_buffer(self, 0)
        output_buffer = printer._get_buffer_name(self)
        weight_name = printer._w(self.metadata["weight_name"])
        bias_name = printer._w(self.metadata["bias_name"]) if self.metadata.get("bias_name") else "None"
        in_channels = self.metadata["in_channels"]
        out_channels = self.metadata["out_channels"]
        groups = self.metadata.get("groups", 1)
        input_scale_var = self._get_input_scale_variable(printer)
        k_h, k_w, s_h, s_w, p_h, p_w, in_h, in_w = self._unpack_conv_dims_h1_wrap()
        is_depthwise = groups > 1 and groups == in_channels and out_channels == in_channels
        fn_base = "depthwise_conv2d_nhwc" if is_depthwise else "conv2d_nhwc"
        fn = f"{fn_base}_int8_to_float" if self.computation_dtype == "int8" else f"{fn_base}_int16_to_float"
        if is_depthwise:
            return [
                f"ops_q.{fn}({input_buffer}, {in_h}, {in_w}, {in_channels}, {weight_name}, "
                f"{k_h}, {k_w}, {bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
                f"{input_scale_var}, {self.weight_scale}, {output_buffer})"
            ]
        return [
            f"ops_q.{fn}({input_buffer}, {in_h}, {in_w}, {in_channels}, {weight_name}, "
            f"{k_h}, {k_w}, {out_channels}, {bias_name}, {s_h}, {s_w}, {p_h}, {p_w}, "
            f"{input_scale_var}, {self.weight_scale}, {output_buffer})"
        ]
    
    def _get_input_spatial_dims(self) -> tuple:
        """(H, W) for Conv2d, or (1, L) for Conv1d via H=1 wrap."""
        if self.inputs and self.inputs[0].output_shape:
            input_shape = self.inputs[0].output_shape
            if len(input_shape) == 4:
                return int(input_shape[2]), int(input_shape[3])
            if len(input_shape) == 3:
                return 1, int(input_shape[2])
        raise ValueError(
            f"DynamicQuantConv2dNode '{self.name}': cannot determine input "
            f"spatial dimensions. Ensure input shape is available."
        )

    def _unpack_conv_dims_h1_wrap(self) -> tuple:
        """Same as StaticQuantConv2dNode helper; duplicated to avoid base-class import order."""
        kernel_size = self.metadata['kernel_size']
        stride = self.metadata['stride']
        padding = self.metadata['padding']
        is_1d = not isinstance(kernel_size, (tuple, list))
        if is_1d:
            k_h, k_w = 1, int(kernel_size)
            s_h, s_w = 1, int(stride)
            p_h, p_w = 0, int(padding)
        else:
            k_h, k_w = int(kernel_size[0]), int(kernel_size[1])
            s_h, s_w = int(stride[0]), int(stride[1])
            p_h, p_w = int(padding[0]), int(padding[1])
        in_h, in_w = self._get_input_spatial_dims()
        return k_h, k_w, s_h, s_w, p_h, p_w, in_h, in_w
    
    def _get_input_scale_variable(self, c_printer) -> str:
        """
        Get the scale variable name from the preceding DynamicQuantizeInputNode.
        
        Raises ValueError if the input is not a DynamicQuantizeInputNode.
        """
        if self.inputs:
            input_node = self.inputs[0]
            if input_node.op_type == 'dynamic_quantize':
                return f"scale_{c_printer._sanitize_name(input_node.name)}"
        
        raise ValueError(
            f"DynamicQuantConv2dNode '{self.name}': expected input from "
            f"DynamicQuantizeInputNode (op_type='dynamic_quantize'), but "
            f"got '{self.inputs[0].op_type if self.inputs else 'none'}' "
            f"from '{self.inputs[0].name if self.inputs else 'N/A'}'. "
            f"Graph transform must insert DynamicQuantizeInputNode before this node."
        )
    
    def __repr__(self) -> str:
        return (f"DynamicQuantConv2dNode(name='{self.name}', "
                f"in_ch={self.metadata.get('in_channels')}, "
                f"out_ch={self.metadata.get('out_channels')}, "
                f"computation_dtype='{self.computation_dtype}', "
                f"weight_scale={self.weight_scale})")
