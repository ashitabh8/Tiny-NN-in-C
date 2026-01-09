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
        offset: int = 0
    ):
        """
        Initialize static quantized conv2d node.
        
        Args:
            original_node: The float conv2d node being quantized
            dtype: Target data type ('int8' or 'int16')
            input_scale: Scale for input activation quantization
            weight_scale: Scale for weight quantization
            output_scale: Scale for output dequantization
            offset: Zero point offset (default 0 for symmetric)
        """
        super().__init__(
            original_node=original_node,
            dtype=dtype,
            scale=weight_scale,
            offset=offset,
            quant_strategy='static'
        )
        
        self.input_scale = input_scale
        self.weight_scale = weight_scale
        self.output_scale = output_scale
    
    def get_pre_nodes(self) -> List[IRNode]:
        """
        Insert QuantizeNode before this layer to quantize float input.
        """
        from .quant_utils import QuantizeNode
        
        pre_node = QuantizeNode(
            name=f"{self.name}_input_q",
            target_dtype=self.dtype,
            scale=self.input_scale,
            offset=self.offset,
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
            offset=self.offset,
            output_shape=self.output_shape
        )
        
        return [post_node]
    
    def generate_c_code(self, c_printer) -> List[str]:
        """
        Generate C code for static quantized conv2d.
        
        Uses conv2d_nhwc_int8/conv2d_nhwc_int16 with explicit input_scale and weight_scale.
        """
        lines = []
        
        input_buffer = c_printer._get_input_buffer(self, 0)
        output_buffer = c_printer._get_buffer_name(self)
        weight_name = c_printer._sanitize_name(self.metadata['weight_name'])
        
        bias_name = c_printer._sanitize_name(self.metadata['bias_name']) \
                    if self.metadata.get('bias_name') else 'NULL'
        
        # Extract conv parameters
        kernel_size = self.metadata['kernel_size']
        stride = self.metadata['stride']
        padding = self.metadata['padding']
        in_channels = self.metadata['in_channels']
        out_channels = self.metadata['out_channels']
        
        # Convert to scalars if tuples
        k_h, k_w = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        s_h, s_w = stride if isinstance(stride, (tuple, list)) else (stride, stride)
        p_h, p_w = padding if isinstance(padding, (tuple, list)) else (padding, padding)
        
        # Determine padding mode
        pad_same = 1 if p_h > 0 or p_w > 0 else 0
        
        # Get input spatial dimensions from input shape
        in_h, in_w = 32, 32  # Default
        if self.inputs and self.inputs[0].output_shape:
            input_shape = self.inputs[0].output_shape
            if len(input_shape) == 4:
                in_h, in_w = input_shape[2], input_shape[3]
        
        if self.dtype == 'int8':
            lines.append(
                f"conv2d_nhwc_int8("
                f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                f"{weight_name}, {k_h}, {k_w}, {out_channels}, "
                f"{bias_name}, {s_h}, {s_w}, {pad_same}, "
                f"{self.input_scale}f, {self.weight_scale}f, {self.offset}, "
                f"{output_buffer});"
            )
        elif self.dtype == 'int16':
            lines.append(
                f"conv2d_nhwc_int16("
                f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                f"{weight_name}, {k_h}, {k_w}, {out_channels}, "
                f"{bias_name}, {s_h}, {s_w}, {pad_same}, "
                f"{self.input_scale}f, {self.weight_scale}f, {self.offset}, "
                f"{output_buffer});"
            )
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        
        return lines
    
    def __repr__(self) -> str:
        return (f"StaticQuantConv2dNode(name='{self.name}', "
                f"in_ch={self.metadata.get('in_channels')}, "
                f"out_ch={self.metadata.get('out_channels')}, "
                f"dtype='{self.dtype}', "
                f"input_scale={self.input_scale}, "
                f"weight_scale={self.weight_scale}, "
                f"output_scale={self.output_scale})")


class DynamicQuantConv2dNode(QuantIRNode):
    """
    Dynamic Quantized Conv2D operation.
    
    - Input scale: Computed at runtime from input values
    - Weight scale: Computed from weights at compile time
    - Output scale: Uses weight_scale for dequantization
    
    Uses DynamicQuantizeInputNode for input (computes scale at runtime)
    and DequantizeNode for output.
    """
    
    def __init__(
        self,
        original_node: IRNode,
        dtype: str,
        weight_scale: float,
        offset: int = 0
    ):
        """
        Initialize dynamic quantized conv2d node.
        
        Args:
            original_node: The float conv2d node being quantized
            dtype: Target data type ('int8' or 'int16')
            weight_scale: Scale for weight quantization (computed from weights)
            offset: Zero point offset (default 0 for symmetric)
        """
        super().__init__(
            original_node=original_node,
            dtype=dtype,
            scale=weight_scale,
            offset=offset,
            quant_strategy='dynamic'
        )
        
        self.weight_scale = weight_scale
    
    def get_pre_nodes(self) -> List[IRNode]:
        """
        Insert DynamicQuantizeInputNode before this layer.
        """
        from .quant_utils import DynamicQuantizeInputNode
        
        pre_node = DynamicQuantizeInputNode(
            name=f"{self.name}_input_dq",
            target_dtype=self.dtype,
            output_shape=self.metadata.get('input_shape')
        )
        
        return [pre_node]
    
    def get_post_nodes(self) -> List[IRNode]:
        """
        Insert DequantizeNode after this layer.
        """
        from .quant_utils import DequantizeNode
        
        post_node = DequantizeNode(
            name=f"{self.name}_output_dq",
            source_dtype=self.dtype,
            scale=self.weight_scale,
            offset=self.offset,
            output_shape=self.output_shape
        )
        
        return [post_node]
    
    def generate_c_code(self, c_printer) -> List[str]:
        """
        Generate C code for dynamic quantized conv2d.
        
        Input scale comes from DynamicQuantizeInputNode variable.
        """
        lines = []
        
        input_buffer = c_printer._get_input_buffer(self, 0)
        output_buffer = c_printer._get_buffer_name(self)
        weight_name = c_printer._sanitize_name(self.metadata['weight_name'])
        
        bias_name = c_printer._sanitize_name(self.metadata['bias_name']) \
                    if self.metadata.get('bias_name') else 'NULL'
        
        # Extract conv parameters
        kernel_size = self.metadata['kernel_size']
        stride = self.metadata['stride']
        padding = self.metadata['padding']
        in_channels = self.metadata['in_channels']
        out_channels = self.metadata['out_channels']
        
        # Convert to scalars if tuples
        k_h, k_w = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        s_h, s_w = stride if isinstance(stride, (tuple, list)) else (stride, stride)
        p_h, p_w = padding if isinstance(padding, (tuple, list)) else (padding, padding)
        
        pad_same = 1 if p_h > 0 or p_w > 0 else 0
        
        # Get input scale variable from DynamicQuantizeInputNode
        input_scale_var = self._get_input_scale_variable(c_printer)
        
        # Get input spatial dimensions from input shape
        in_h, in_w = 32, 32  # Default
        if self.inputs and self.inputs[0].output_shape:
            input_shape = self.inputs[0].output_shape
            if len(input_shape) == 4:
                in_h, in_w = input_shape[2], input_shape[3]
        
        if self.dtype == 'int8':
            lines.append(
                f"conv2d_nhwc_int8("
                f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                f"{weight_name}, {k_h}, {k_w}, {out_channels}, "
                f"{bias_name}, {s_h}, {s_w}, {pad_same}, "
                f"{input_scale_var}, {self.weight_scale}f, {self.offset}, "
                f"{output_buffer});"
            )
        elif self.dtype == 'int16':
            lines.append(
                f"conv2d_nhwc_int16("
                f"{input_buffer}, {in_h}, {in_w}, {in_channels}, "
                f"{weight_name}, {k_h}, {k_w}, {out_channels}, "
                f"{bias_name}, {s_h}, {s_w}, {pad_same}, "
                f"{input_scale_var}, {self.weight_scale}f, {self.offset}, "
                f"{output_buffer});"
            )
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        
        return lines
    
    def _get_input_scale_variable(self, c_printer) -> str:
        """
        Get the scale variable name from the input DynamicQuantizeInputNode.
        """
        if self.inputs:
            input_node = self.inputs[0]
            if input_node.op_type == 'dynamic_quantize':
                return f"scale_{c_printer._sanitize_name(input_node.name)}"
        
        return f"{self.weight_scale}f"
    
    def __repr__(self) -> str:
        return (f"DynamicQuantConv2dNode(name='{self.name}', "
                f"in_ch={self.metadata.get('in_channels')}, "
                f"out_ch={self.metadata.get('out_channels')}, "
                f"dtype='{self.dtype}', "
                f"weight_scale={self.weight_scale})")

