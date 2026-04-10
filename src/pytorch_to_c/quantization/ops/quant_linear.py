"""
Quantized Linear Nodes - Static and Dynamic variants

Two separate classes for clarity:
- StaticQuantLinearNode: User provides all scales (input, weight, output)
- DynamicQuantLinearNode: Input scale computed at runtime, weight scale from weights
"""

from typing import List
from ...ir.quant_node import QuantIRNode
from ...ir.node import IRNode


class StaticQuantLinearNode(QuantIRNode):
    """
    Static Quantized Linear/Dense operation.
    
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
        Initialize static quantized linear node.
        
        Args:
            original_node: The float linear node being quantized
            dtype: Target data type ('int8' or 'int16')
            input_scale: Scale for input activation quantization
            weight_scale: Scale for weight quantization
            output_scale: Scale for output activation (requant + dequantize)
            input_offset: Zero point for input (QuantizeNode)
            weight_offset: Zero point for weights (compile-time quant)
            output_offset: Zero point for layer output (dense requant + DequantizeNode)
        """
        # Use weight_scale as the "main" scale for QuantIRNode
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
        
        Uses user-provided input_scale.
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
        
        Uses user-provided output_scale.
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
        Generate C code for static quantized linear.
        
        Uses dense_int8/dense_int16 with explicit input_scale and weight_scale.
        """
        lines = []
        
        input_buffer = c_printer._get_input_buffer(self, 0)
        output_buffer = c_printer._get_buffer_name(self)
        weight_name = c_printer._sanitize_name(self.metadata['weight_name'])
        
        bias_name = c_printer._sanitize_name(self.metadata['bias_name']) \
                    if self.metadata.get('bias_name') else 'NULL'
        
        in_features = self.metadata['in_features']
        out_features = self.metadata['out_features']
        
        if self.dtype == 'int8':
            lines.append(
                f"dense_int8("
                f"{input_buffer}, {in_features}, "
                f"{weight_name}, {bias_name}, {out_features}, "
                f"{self.input_scale}f, {self.weight_scale}f, {self.output_scale}f, "
                f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
                f"{output_buffer});"
            )
        elif self.dtype == 'int16':
            lines.append(
                f"dense_int16("
                f"{input_buffer}, {in_features}, "
                f"{weight_name}, {bias_name}, {out_features}, "
                f"{self.input_scale}f, {self.weight_scale}f, {self.output_scale}f, "
                f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
                f"{output_buffer});"
            )
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        
        return lines
    
    def __repr__(self) -> str:
        return (f"StaticQuantLinearNode(name='{self.name}', "
                f"in={self.metadata.get('in_features')}, "
                f"out={self.metadata.get('out_features')}, "
                f"dtype='{self.dtype}', "
                f"input_scale={self.input_scale}, "
                f"weight_scale={self.weight_scale}, "
                f"output_scale={self.output_scale}, "
                f"zp_in={self.input_offset}, zp_w={self.weight_offset}, zp_out={self.output_offset})")


class StaticPerChannelQuantLinearNode(StaticQuantLinearNode):
    """
    Static quantized linear with per-output-column weight scales (see C ``dense_*_per_channel``).

    Weight scales are stored as a float parameter; ``metadata['per_channel_weight_scales_param']``
    is set during ``quantize_weights`` on the transform.
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
                f"StaticPerChannelQuantLinearNode '{self.name}': missing "
                f"metadata['per_channel_weight_scales_param']. Run QuantizationTransform."
            )
        scales_c = c_printer._sanitize_name(scales_param)

        lines = []
        input_buffer = c_printer._get_input_buffer(self, 0)
        output_buffer = c_printer._get_buffer_name(self)
        weight_name = c_printer._sanitize_name(self.metadata['weight_name'])
        bias_name = c_printer._sanitize_name(self.metadata['bias_name']) \
                    if self.metadata.get('bias_name') else 'NULL'
        in_features = self.metadata['in_features']
        out_features = self.metadata['out_features']

        if self.dtype == 'int8':
            lines.append(
                f"dense_int8_per_channel("
                f"{input_buffer}, {in_features}, "
                f"{weight_name}, {bias_name}, {out_features}, "
                f"{self.input_scale}f, {scales_c}, {self.output_scale}f, "
                f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
                f"{output_buffer});"
            )
        elif self.dtype == 'int16':
            lines.append(
                f"dense_int16_per_channel("
                f"{input_buffer}, {in_features}, "
                f"{weight_name}, {bias_name}, {out_features}, "
                f"{self.input_scale}f, {scales_c}, {self.output_scale}f, "
                f"{self.input_offset}, {self.weight_offset}, {self.output_offset}, "
                f"{output_buffer});"
            )
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        return lines

    def __repr__(self) -> str:
        return (f"StaticPerChannelQuantLinearNode(name='{self.name}', "
                f"in={self.metadata.get('in_features')}, "
                f"out={self.metadata.get('out_features')}, "
                f"dtype='{self.dtype}', "
                f"input_scale={self.input_scale}, "
                f"output_scale={self.output_scale}, "
                f"zp_in={self.input_offset}, zp_w={self.weight_offset}, zp_out={self.output_offset})")


class DynamicQuantLinearNode(QuantIRNode):
    """
    Dynamic Quantized Linear/Dense operation.
    
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
                    f"DynamicQuantLinearNode '{self.name}' expects quantized input, "
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
        
        in_features = self.metadata['in_features']
        out_features = self.metadata['out_features']
        
        input_scale_var = self._get_input_scale_variable(c_printer)
        
        if self.computation_dtype == 'int8':
            lines.append(
                f"dense_int8_to_float("
                f"{input_buffer}, {in_features}, "
                f"{weight_name}, {bias_name}, {out_features}, "
                f"{input_scale_var}, {self.weight_scale}f, "
                f"{output_buffer});"
            )
        elif self.computation_dtype == 'int16':
            lines.append(
                f"dense_int16_to_float("
                f"{input_buffer}, {in_features}, "
                f"{weight_name}, {bias_name}, {out_features}, "
                f"{input_scale_var}, {self.weight_scale}f, "
                f"{output_buffer});"
            )
        else:
            raise ValueError(f"Unsupported computation dtype: {self.computation_dtype}")
        
        return lines
    
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
            f"DynamicQuantLinearNode '{self.name}': expected input from "
            f"DynamicQuantizeInputNode (op_type='dynamic_quantize'), but "
            f"got '{self.inputs[0].op_type if self.inputs else 'none'}' "
            f"from '{self.inputs[0].name if self.inputs else 'N/A'}'. "
            f"Graph transform must insert DynamicQuantizeInputNode before this node."
        )
    
    def __repr__(self) -> str:
        return (f"DynamicQuantLinearNode(name='{self.name}', "
                f"in={self.metadata.get('in_features')}, "
                f"out={self.metadata.get('out_features')}, "
                f"computation_dtype='{self.computation_dtype}', "
                f"weight_scale={self.weight_scale})")
