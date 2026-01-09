"""
QuantIRNode - Base class for quantized IR nodes
"""

from abc import abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from .node import IRNode


class QuantIRNode(IRNode):
    """
    Base class for quantized operations.
    
    Extends IRNode with:
    - Quantization parameters (scale, offset)
    - C code generation method
    - Dtype validation for quantized inputs
    - Pre/post nodes for dtype conversion (flexible, node-controlled)
    """
    
    def __init__(
        self,
        original_node: IRNode,
        dtype: str,
        scale: float,
        offset: int,
        quant_strategy: str = 'static',
        **kwargs
    ):
        """
        Create a quantized node from an existing float node.
        
        Args:
            original_node: The float node being quantized
            dtype: Target data type ('int8' or 'int16')
            scale: Quantization scale factor (for weights)
            offset: Zero point offset
            quant_strategy: 'static' or 'dynamic' - affects how inputs are quantized
        """
        # Copy properties from original node
        super().__init__(
            name=original_node.name,
            op_type=original_node.op_type,
            output_shape=original_node.output_shape,
            dtype=dtype,
            metadata=original_node.metadata.copy() if original_node.metadata else {}
        )
        
        # Copy connections (will be updated by transform)
        self.inputs = original_node.inputs.copy() if original_node.inputs else []
        self.users = original_node.users.copy() if original_node.users else []
        
        # Quantization parameters
        self.scale = scale
        self.offset = offset
        self.quant_strategy = quant_strategy
        
        # Store quantization info in metadata
        self.metadata['quantized'] = True
        self.metadata['quant_params'] = {
            'scale': scale,
            'offset': offset,
            'dtype': dtype,
            'strategy': quant_strategy
        }
        
        # Store reference to original node (for debugging)
        self._original_node = original_node
    
    @abstractmethod
    def generate_c_code(self, c_printer) -> List[str]:
        """
        Generate C code for this quantized operation.
        
        Args:
            c_printer: CPrinter instance (for accessing helper methods)
            
        Returns:
            List of C code lines
        """
        pass
    
    def get_pre_nodes(self) -> List['IRNode']:
        """
        Get list of nodes to insert BEFORE this node.
        
        This allows the node to specify what dtype conversions or other
        preprocessing it needs. Default: empty list (no pre-processing).
        
        Override in subclasses to add Quantize/DynamicQuantize nodes.
        
        Returns:
            List of IRNode instances to insert before this node
        """
        return []
    
    def get_post_nodes(self) -> List['IRNode']:
        """
        Get list of nodes to insert AFTER this node.
        
        This allows the node to specify what dtype conversions or other
        postprocessing it needs. Default: empty list (no post-processing).
        
        Override in subclasses to add Dequantize nodes.
        
        Returns:
            List of IRNode instances to insert after this node
        """
        return []
    
    def get_c_dtype(self) -> str:
        """Return C data type string for this quantized dtype."""
        if self.dtype == 'int8':
            return 'int8_t'
        elif self.dtype == 'int16':
            return 'int16_t'
        else:
            raise ValueError(f"Unsupported quantized dtype: {self.dtype}")
    
    def validate_input_dtypes(self) -> bool:
        """
        Validate that inputs have compatible quantized dtypes.
        
        Quantized ops expect quantized inputs (int8 or int16).
        
        Returns:
            True if valid
            
        Raises:
            TypeError: If input dtypes are incompatible
        """
        for inp in self.inputs:
            if inp.dtype not in ['int8', 'int16']:
                raise TypeError(
                    f"QuantNode '{self.name}' expects quantized input, "
                    f"got '{inp.dtype}' from '{inp.name}'"
                )
        return True
    
    def __repr__(self) -> str:
        return (f"QuantIRNode(name='{self.name}', op_type='{self.op_type}', "
                f"dtype='{self.dtype}', scale={self.scale}, offset={self.offset}, "
                f"strategy='{self.quant_strategy}')")

