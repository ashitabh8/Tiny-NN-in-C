"""
Quantization Utility Nodes - Quantize, DynamicQuantize, and Dequantize operations

These are separate IRNodes (not fused into quantized layer nodes)
to allow optimization passes to remove redundant conversions.
"""

from typing import List, Optional, Tuple
from ...ir.node import IRNode


class QuantizeNode(IRNode):
    """
    Conversion node: float32 → int8/int16
    
    This is a separate IRNode (not fused into quantized ops).
    Future optimization passes can remove redundant conversions.
    """
    
    def __init__(
        self,
        name: str,
        target_dtype: str,
        scale: float,
        offset: int,
        output_shape: Optional[Tuple[int, ...]] = None
    ):
        """
        Initialize a quantize node.
        
        Args:
            name: Node name
            target_dtype: Output dtype ('int8' or 'int16')
            scale: Quantization scale
            offset: Zero point offset
            output_shape: Shape of the output tensor
        """
        super().__init__(
            name=name,
            op_type='quantize',
            output_shape=output_shape,
            dtype=target_dtype,
            metadata={
                'scale': scale,
                'offset': offset,
                'target_dtype': target_dtype,
                'source_dtype': 'float32'
            }
        )
        self.scale = scale
        self.offset = offset
        self.target_dtype = target_dtype
    
    def generate_c_code(self, c_printer) -> List[str]:
        """
        Generate C code for quantization.
        
        Args:
            c_printer: CPrinter instance
            
        Returns:
            List of C code lines
        """
        input_buf = c_printer._get_input_buffer(self, 0)
        output_buf = c_printer._get_buffer_name(self)
        
        # Get size from buffer size calculation
        buffer_sizes = c_printer._calculate_buffer_sizes()
        size = buffer_sizes.get(self.name, 1024)
        
        if self.target_dtype == 'int8':
            return [
                f"quantize_float_to_int8({input_buf}, {size}, "
                f"{self.scale}f, {self.offset}, {output_buf});"
            ]
        elif self.target_dtype == 'int16':
            return [
                f"quantize_float_to_int16({input_buf}, {size}, "
                f"{self.scale}f, {self.offset}, {output_buf});"
            ]
        else:
            raise ValueError(f"Unsupported target dtype: {self.target_dtype}")
    
    def validate_input_dtypes(self) -> bool:
        """
        Validate that input is float32.
        
        Returns:
            True if valid
            
        Raises:
            TypeError: If input is not float32
        """
        for inp in self.inputs:
            if inp.dtype != 'float32':
                raise TypeError(
                    f"QuantizeNode '{self.name}' expects float32 input, "
                    f"got '{inp.dtype}' from '{inp.name}'"
                )
        return True
    
    def get_c_dtype(self) -> str:
        """Return C data type for output buffer."""
        return 'int8_t' if self.target_dtype == 'int8' else 'int16_t'
    
    def __repr__(self) -> str:
        return (f"QuantizeNode(name='{self.name}', target='{self.target_dtype}', "
                f"scale={self.scale}, offset={self.offset})")


class DynamicQuantizeInputNode(IRNode):
    """
    Dynamic quantization of input: float32 → int8/int16
    
    Unlike QuantizeNode (static scale), this node computes scale and offset
    from the input tensor at RUNTIME using min-max per-tensor.
    
    This is essential for dynamic quantization where input range is unknown.
    """
    
    def __init__(
        self,
        name: str,
        target_dtype: str,
        output_shape: Optional[Tuple[int, ...]] = None
    ):
        """
        Initialize a dynamic quantize input node.
        
        Args:
            name: Node name
            target_dtype: Output dtype ('int8' or 'int16')
            output_shape: Shape of the output tensor
        """
        super().__init__(
            name=name,
            op_type='dynamic_quantize',
            output_shape=output_shape,
            dtype=target_dtype,
            metadata={
                'target_dtype': target_dtype,
                'source_dtype': 'float32',
                'strategy': 'dynamic_minmax'
            }
        )
        self.target_dtype = target_dtype
        # Scale and offset are computed at runtime, but we need default values for metadata
        self.scale = None  # Computed at runtime
        self.offset = 0  # Symmetric quantization
    
    def generate_c_code(self, c_printer) -> List[str]:
        """
        Generate C code for dynamic quantization.
        
        Computes scale from input min/max at runtime, then quantizes.
        
        Args:
            c_printer: CPrinter instance
            
        Returns:
            List of C code lines
        """
        lines = []
        
        input_buf = c_printer._get_input_buffer(self, 0)
        output_buf = c_printer._get_buffer_name(self)
        
        # Get size from input node's shape (more reliable than our own shape)
        size = self._get_input_size(c_printer)
        
        # Generate code to compute scale dynamically
        scale_var = f"scale_{c_printer._sanitize_name(self.name)}"
        
        if self.target_dtype == 'int8':
            lines.extend([
                f"// Dynamic quantization: compute scale from input",
                f"float {scale_var} = compute_dynamic_scale_int8({input_buf}, {size});",
                f"quantize_float_to_int8({input_buf}, {size}, {scale_var}, 0, {output_buf});"
            ])
        elif self.target_dtype == 'int16':
            lines.extend([
                f"// Dynamic quantization: compute scale from input", 
                f"float {scale_var} = compute_dynamic_scale_int16({input_buf}, {size});",
                f"quantize_float_to_int16({input_buf}, {size}, {scale_var}, 0, {output_buf});"
            ])
        else:
            raise ValueError(f"Unsupported target dtype: {self.target_dtype}")
        
        return lines
    
    def _get_input_size(self, c_printer) -> int:
        """Get the size from the input node's shape."""
        if self.inputs:
            input_node = self.inputs[0]
            if input_node.output_shape:
                import math
                shape = input_node.output_shape
                # Remove batch dimension if present
                if len(shape) > 0 and shape[0] == 1:
                    shape = shape[1:]
                if len(shape) > 0:
                    return math.prod(shape)
        
        # Fallback: try our own shape
        if self.output_shape:
            import math
            shape = self.output_shape
            if len(shape) > 0 and shape[0] == 1:
                shape = shape[1:]
            if len(shape) > 0:
                return math.prod(shape)
        
        # Last resort fallback
        return 1024
    
    def validate_input_dtypes(self) -> bool:
        """Validate that input is float32."""
        for inp in self.inputs:
            if inp.dtype != 'float32':
                raise TypeError(
                    f"DynamicQuantizeInputNode '{self.name}' expects float32 input, "
                    f"got '{inp.dtype}' from '{inp.name}'"
                )
        return True
    
    def get_c_dtype(self) -> str:
        """Return C data type for output buffer."""
        return 'int8_t' if self.target_dtype == 'int8' else 'int16_t'
    
    def __repr__(self) -> str:
        return f"DynamicQuantizeInputNode(name='{self.name}', target='{self.target_dtype}')"


class DequantizeNode(IRNode):
    """
    Conversion node: int8/int16 → float32
    
    This is a separate IRNode for optimization flexibility.
    """
    
    def __init__(
        self,
        name: str,
        source_dtype: str,
        scale: float,
        offset: int,
        output_shape: Optional[Tuple[int, ...]] = None
    ):
        """
        Initialize a dequantize node.
        
        Args:
            name: Node name
            source_dtype: Input dtype ('int8' or 'int16')
            scale: Quantization scale (used for dequantization)
            offset: Zero point offset
            output_shape: Shape of the output tensor
        """
        super().__init__(
            name=name,
            op_type='dequantize',
            output_shape=output_shape,
            dtype='float32',  # Output is always float32
            metadata={
                'scale': scale,
                'offset': offset,
                'source_dtype': source_dtype,
                'target_dtype': 'float32'
            }
        )
        self.scale = scale
        self.offset = offset
        self.source_dtype = source_dtype
    
    def generate_c_code(self, c_printer) -> List[str]:
        """
        Generate C code for dequantization.
        
        Args:
            c_printer: CPrinter instance
            
        Returns:
            List of C code lines
        """
        input_buf = c_printer._get_input_buffer(self, 0)
        output_buf = c_printer._get_buffer_name(self)
        
        # Get size from buffer size calculation
        buffer_sizes = c_printer._calculate_buffer_sizes()
        size = buffer_sizes.get(self.name, 1024)
        
        if self.source_dtype == 'int8':
            return [
                f"dequantize_int8_to_float({input_buf}, {size}, "
                f"{self.scale}f, {self.offset}, {output_buf});"
            ]
        elif self.source_dtype == 'int16':
            return [
                f"dequantize_int16_to_float({input_buf}, {size}, "
                f"{self.scale}f, {self.offset}, {output_buf});"
            ]
        else:
            raise ValueError(f"Unsupported source dtype: {self.source_dtype}")
    
    def validate_input_dtypes(self) -> bool:
        """
        Validate that input is quantized (int8 or int16).
        
        Returns:
            True if valid
            
        Raises:
            TypeError: If input is not quantized
        """
        for inp in self.inputs:
            if inp.dtype not in ['int8', 'int16']:
                raise TypeError(
                    f"DequantizeNode '{self.name}' expects quantized input, "
                    f"got '{inp.dtype}' from '{inp.name}'"
                )
        return True
    
    def get_c_dtype(self) -> str:
        """Return C data type for output buffer."""
        return 'float'
    
    def __repr__(self) -> str:
        return (f"DequantizeNode(name='{self.name}', source='{self.source_dtype}', "
                f"scale={self.scale}, offset={self.offset})")

