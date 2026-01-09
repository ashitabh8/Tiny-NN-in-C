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
        offset: int = 0
    ):
        """
        Initialize static quantized linear node.
        
        Args:
            original_node: The float linear node being quantized
            dtype: Target data type ('int8' or 'int16')
            input_scale: Scale for input activation quantization
            weight_scale: Scale for weight quantization
            output_scale: Scale for output dequantization
            offset: Zero point offset (default 0 for symmetric)
        """
        # Use weight_scale as the "main" scale for QuantIRNode
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
        
        Uses user-provided input_scale.
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
        
        Uses user-provided output_scale.
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
                f"{self.input_scale}f, {self.weight_scale}f, {self.offset}, "
                f"{output_buffer});"
            )
        elif self.dtype == 'int16':
            lines.append(
                f"dense_int16("
                f"{input_buffer}, {in_features}, "
                f"{weight_name}, {bias_name}, {out_features}, "
                f"{self.input_scale}f, {self.weight_scale}f, {self.offset}, "
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
                f"output_scale={self.output_scale})")


class DynamicQuantLinearNode(QuantIRNode):
    """
    Dynamic Quantized Linear/Dense operation.
    
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
        Initialize dynamic quantized linear node.
        
        Args:
            original_node: The float linear node being quantized
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
        
        This node computes scale from input at runtime.
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
        
        Uses weight_scale for output dequantization.
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
        Generate C code for dynamic quantized linear.
        
        Input scale comes from DynamicQuantizeInputNode variable.
        """
        lines = []
        
        input_buffer = c_printer._get_input_buffer(self, 0)
        output_buffer = c_printer._get_buffer_name(self)
        weight_name = c_printer._sanitize_name(self.metadata['weight_name'])
        
        bias_name = c_printer._sanitize_name(self.metadata['bias_name']) \
                    if self.metadata.get('bias_name') else 'NULL'
        
        in_features = self.metadata['in_features']
        out_features = self.metadata['out_features']
        
        # Get input scale variable from DynamicQuantizeInputNode
        input_scale_var = self._get_input_scale_variable(c_printer)
        
        if self.dtype == 'int8':
            lines.append(
                f"dense_int8("
                f"{input_buffer}, {in_features}, "
                f"{weight_name}, {bias_name}, {out_features}, "
                f"{input_scale_var}, {self.weight_scale}f, {self.offset}, "
                f"{output_buffer});"
            )
        elif self.dtype == 'int16':
            lines.append(
                f"dense_int16("
                f"{input_buffer}, {in_features}, "
                f"{weight_name}, {bias_name}, {out_features}, "
                f"{input_scale_var}, {self.weight_scale}f, {self.offset}, "
                f"{output_buffer});"
            )
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        
        return lines
    
    def _get_input_scale_variable(self, c_printer) -> str:
        """
        Get the scale variable name from the input DynamicQuantizeInputNode.
        
        The input node generates: float scale_xxx = compute_dynamic_scale_int8(...);
        """
        if self.inputs:
            input_node = self.inputs[0]
            if input_node.op_type == 'dynamic_quantize':
                return f"scale_{c_printer._sanitize_name(input_node.name)}"
        
        # Fallback (shouldn't happen in correct usage)
        return f"{self.weight_scale}f"
    
    def __repr__(self) -> str:
        return (f"DynamicQuantLinearNode(name='{self.name}', "
                f"in={self.metadata.get('in_features')}, "
                f"out={self.metadata.get('out_features')}, "
                f"dtype='{self.dtype}', "
                f"weight_scale={self.weight_scale})")
